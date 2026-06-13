"""DMD2 (Yin et al., 2024 NeurIPS) primitives for the 2D toy, Stage B only.

DMD2 = DMD distribution matching + TTUR (more fake-score updates) + a GAN loss
on noised samples, with the regression loss removed. This module adds the GAN
pieces; the distribution-matching surrogate and one-step generator are reused
from M3 (src/dmd.py). No few-step generation, no backward simulation here.

Discriminator design (DMD2, adapted to 2D): the critic is a linear classification
head on mu_fake's *shared* last hidden layer, not a standalone network. The head
consumes mu_fake.forward_features(x_t, t); t flows in only through that shared
encoder (the head adds no separate time embedding). The mu_fake backbone is
updated jointly by the GAN loss (see src/train_dmd2.py).
"""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from .diffusion import load_teacher
from .models import MLPDenoiser


class DiscHead(nn.Module):
    """Linear real/fake head on mu_fake's last hidden features -> logit."""

    def __init__(self, hidden_dim=128):
        super().__init__()
        self.config = {"hidden_dim": hidden_dim}
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, features):
        return self.fc(features).squeeze(-1)

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def build_m4_models(teacher_ckpt, m3_student_ckpt, m3_fake_ckpt, device):
    """Frozen EMA teacher + trainable student/mu_fake (from best M3) + fresh head.

    Student and mu_fake start from the best M3 checkpoints (DMD2 continues
    training the existing student). mu_fake falls back to the EMA teacher weights
    if no M3 fake checkpoint is given. The discriminator head is fresh and shares
    mu_fake's backbone (head width = mu_fake hidden_dim).
    """
    teacher, diffusion = load_teacher(teacher_ckpt, device=device, which="ema")
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    sck = torch.load(m3_student_ckpt, map_location=device)
    student = MLPDenoiser.from_config(sck["model_config"]).to(device)
    student.load_state_dict(sck["model_state_dict"])

    mu_fake = MLPDenoiser.from_config(sck["model_config"]).to(device)
    if m3_fake_ckpt:
        fck = torch.load(m3_fake_ckpt, map_location=device)
        mu_fake.load_state_dict(fck["model_state_dict"])
    else:
        mu_fake.load_state_dict(copy.deepcopy(teacher.state_dict()))

    head = DiscHead(hidden_dim=sck["model_config"]["hidden_dim"]).to(device)
    return student, mu_fake, head, teacher, diffusion


def disc_logits(mu_fake, head, x_t, t):
    """Critic logit D(x_t) = head(mu_fake bottleneck). No separate time embed."""
    return head(mu_fake.forward_features(x_t, t))


def sample_gan_t(n, num_timesteps, device, t_min=None, t_max=None):
    """Random diffusion timestep for the GAN noising F(x, t).

    Defaults to the full schedule; pass t_min/t_max to restrict to a noise band
    (real and fake become indistinguishable at high noise, so a low/mid band
    gives the discriminator a usable signal).
    """
    lo = 0 if t_min is None else t_min
    hi = num_timesteps if t_max is None else t_max + 1
    return torch.randint(lo, hi, (n,), device=device)


def disc_loss(mu_fake, head, diffusion, x_real, x_fake_detached, t):
    """Non-saturating GAN discriminator loss on noised real/fake samples.

    L_D = softplus(-D(F(x_real, t))) + softplus(D(F(x_fake, t))), where F is the
    forward diffusion q_sample. Fake samples must already be detached. Gradients
    flow into the head and the shared mu_fake backbone.
    """
    real_t = diffusion.q_sample(x_real, t, torch.randn_like(x_real))
    fake_t = diffusion.q_sample(x_fake_detached, t, torch.randn_like(x_fake_detached))
    real_logit = disc_logits(mu_fake, head, real_t, t)
    fake_logit = disc_logits(mu_fake, head, fake_t, t)
    return F.softplus(-real_logit).mean() + F.softplus(fake_logit).mean()


def gen_gan_loss(mu_fake, head, diffusion, x_fake, t):
    """Generator GAN term: softplus(-D(F(x_fake, t))).

    Gradient flows into the generator through x_fake; the head and mu_fake are not
    stepped here (their optimizers are not stepped during the generator update).
    """
    fake_t = diffusion.q_sample(x_fake, t, torch.randn_like(x_fake))
    return F.softplus(-disc_logits(mu_fake, head, fake_t, t)).mean()


@torch.no_grad()
def disc_diagnostics(mu_fake, head, diffusion, x_real, x_fake_detached, t):
    """Whether the critic separates real from fake: logits and accuracies."""
    real_t = diffusion.q_sample(x_real, t, torch.randn_like(x_real))
    fake_t = diffusion.q_sample(x_fake_detached, t, torch.randn_like(x_fake_detached))
    real_logit = disc_logits(mu_fake, head, real_t, t)
    fake_logit = disc_logits(mu_fake, head, fake_t, t)
    real_acc = (real_logit > 0).float().mean().item()
    fake_acc = (fake_logit < 0).float().mean().item()
    return {
        "mean_real_logit": real_logit.mean().item(),
        "mean_fake_logit": fake_logit.mean().item(),
        "real_acc": real_acc,
        "fake_acc": fake_acc,
        "total_acc": 0.5 * (real_acc + fake_acc),
    }
