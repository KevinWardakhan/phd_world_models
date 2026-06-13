"""DMD (Yin et al., 2024) primitives for the 2D toy, Stage A only.

Distribution Matching Distillation turns the frozen multi-step teacher into a
one-step generator G_theta by matching distributions: the generator gradient is
the difference between the frozen teacher (real) score and an online fake score
trained on the student's own samples. No GAN here (that is M4/DMD2).
"""

import copy
import os

import torch

from .diffusion import load_teacher
from .models import MLPDenoiser
from .utils import ensure_dir


def build_student_and_fake(teacher_ckpt, device):
    """Frozen EMA teacher + trainable student and mu_fake, all same architecture.

    Student and mu_fake are initialized from the EMA teacher weights (DMD Alg. 1).
    """
    teacher, diffusion = load_teacher(teacher_ckpt, device=device, which="ema")
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    ckpt = torch.load(teacher_ckpt, map_location=device)
    cfg = ckpt["model_config"]

    student = MLPDenoiser.from_config(cfg).to(device)
    student.load_state_dict(copy.deepcopy(teacher.state_dict()))
    mu_fake = MLPDenoiser.from_config(cfg).to(device)
    mu_fake.load_state_dict(copy.deepcopy(teacher.state_dict()))
    return student, mu_fake, teacher, diffusion


def generate(diffusion, student, z, t_gen):
    """One-step generator G_theta(z): a single x0 prediction (differentiable)."""
    n = z.shape[0]
    t = torch.full((n,), t_gen, device=z.device, dtype=torch.long)
    eps = student(z, t)
    return diffusion.predict_x0(z, t, eps)


def dmd_generator_loss(diffusion, teacher, mu_fake, x, t_min, t_max):
    """Surrogate whose autograd grad wrt the generator equals the DMD KL gradient.

    grad wrt x = a_t (s_fake - s_real)  (DMD Eq. 7, w_t = 1 for the toy), with
    scores from eps via s = -eps/sigma_t. The score difference is detached so the
    frozen teacher and mu_fake are not updated by the generator step.
    """
    n = x.shape[0]
    t = torch.randint(t_min, t_max + 1, (n,), device=x.device)
    eps = torch.randn_like(x)
    x_t = diffusion.q_sample(x, t, eps)

    with torch.no_grad():
        eps_real = teacher(x_t, t)
        eps_fake = mu_fake(x_t, t)
        s_real = diffusion.eps_to_score(eps_real, t)
        s_fake = diffusion.eps_to_score(eps_fake, t)
        a_t = diffusion._gather(diffusion.sqrt_alphas_cumprod, t)
        grad_x = a_t * (s_fake - s_real)

    return (x * grad_x).sum() / n


def build_regression_pairs(diffusion, teacher, n, seed, device, path, teacher_ckpt):
    """Paired (z_ref, y_ref) dataset for the DMD regression loss.

    y_ref is the deterministic (DDIM eta=0) teacher sample from z_ref. Saved once
    and reused so the regression target is fixed across variants and runs.
    """
    if os.path.exists(path):
        data = torch.load(path, map_location=device)
        return data["z_ref"].to(device), data["y_ref"].to(device)

    gen = torch.Generator().manual_seed(seed)
    z_ref = torch.randn(n, 2, generator=gen).to(device)
    y_ref = diffusion.ddim_sample_loop(teacher, x_init=z_ref)
    ensure_dir(os.path.dirname(path))
    torch.save(
        {
            "z_ref": z_ref.cpu(),
            "y_ref": y_ref.cpu(),
            "sampler": "ddim_eta0",
            "num_steps": diffusion.num_timesteps,
            "teacher_ckpt": teacher_ckpt,
        },
        path,
    )
    return z_ref, y_ref


def regression_loss(pred, target, kind="mse"):
    if kind == "mse":
        return torch.nn.functional.mse_loss(pred, target)
    if kind == "smooth_l1":
        return torch.nn.functional.smooth_l1_loss(pred, target)
    if kind == "l1":
        return torch.nn.functional.l1_loss(pred, target)
    raise ValueError(f"unknown reg_loss_type: {kind}")
