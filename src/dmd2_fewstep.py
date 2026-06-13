"""Few-step (multi-step) generator primitives for DMD2 Stage B (M5).

Implements DMD2's multi-step student (Yin et al., 2024, sec 4.4 and 4.5) for the
2D toy:

- sec 4.4: a fixed N-step schedule, alternating one denoise (clean x0 estimate)
  and one renoise (forward diffusion to the next, lower timestep), shared by
  training and inference.
- sec 4.5: backward simulation. The input the student denoises during training is
  produced by running the student's own sampling loop for the earlier steps,
  instead of noising a real sample. This removes the train/inference mismatch.

Nothing in the network changes: the student already conditions on t, so the
single one-step denoise from src/dmd.generate is reused at each scheduled step.
"""

import copy

import torch

from . import dmd
from . import dmd2
from .diffusion import load_teacher
from .models import MLPDenoiser


def default_schedule(num_timesteps, n_steps):
    """Descending generator timesteps, mirroring DMD2's 999,749,499,249 (T=1000).

    t_k = (T-1) - k*(T // n_steps); for T=200, n_steps=4 -> [199, 149, 99, 49].
    """
    step = num_timesteps // n_steps
    return [num_timesteps - 1 - k * step for k in range(n_steps)]


def schedule_from_fracs(num_timesteps, fracs):
    """Map config fractions in (0, 1] to descending timestep indices in [0, T-1]."""
    return [min(num_timesteps - 1, max(0, int(round(f * num_timesteps)))) for f in fracs]


def generate_multistep(diffusion, student, z, schedule):
    """N-step student sampling (sec 4.4).

    From z ~ N(0, I): at each scheduled timestep, denoise to a clean estimate x0,
    then renoise to the next (lower) timestep; return the final x0. With a single
    entry schedule [T-1] this reduces exactly to the one-step generator.
    """
    n = z.shape[0]
    x = z
    x0 = z
    for i, t_i in enumerate(schedule):
        t = torch.full((n,), t_i, device=z.device, dtype=torch.long)
        x0 = diffusion.predict_x0(x, t, student(x, t))
        if i < len(schedule) - 1:
            t_next = torch.full((n,), schedule[i + 1], device=z.device, dtype=torch.long)
            x = diffusion.q_sample(x0, t_next, torch.randn_like(x0))
    return x0


@torch.no_grad()
def backward_simulate(diffusion, student, z, schedule, step_idx):
    """Build the input x_{t_i} to step `step_idx` from the current student (sec 4.5).

    Runs the sec 4.4 inference loop for the first `step_idx` steps (no grad), so the
    input the generator denoises during training matches what it sees at inference.
    step_idx=0 returns z unchanged (the first step always starts from pure noise).
    """
    n = z.shape[0]
    x = z
    for i in range(step_idx):
        t = torch.full((n,), schedule[i], device=z.device, dtype=torch.long)
        x0 = diffusion.predict_x0(x, t, student(x, t))
        t_next = torch.full((n,), schedule[i + 1], device=z.device, dtype=torch.long)
        x = diffusion.q_sample(x0, t_next, torch.randn_like(x0))
    return x


def simulated_fake(diffusion, student, schedule, bs, device, rng_steps):
    """Differentiable clean estimate G(x_{t_i}, t_i) at a randomly chosen step.

    Picks a scheduled step uniformly, backward-simulates its input from the current
    student (no grad, sec 4.5), then returns the differentiable one-step denoise at
    that step. Detach the result for the fake-score and discriminator updates.
    """
    step_idx = int(rng_steps.integers(0, len(schedule)))
    z = torch.randn(bs, 2, device=device)
    x_in = backward_simulate(diffusion, student, z, schedule, step_idx)
    return dmd.generate(diffusion, student, x_in, schedule[step_idx])


def build_m5_models(teacher_ckpt, device, init_student_ckpt=None, init_fake_ckpt=None):
    """Frozen EMA teacher + trainable student/mu_fake + fresh discriminator head.

    Student and mu_fake default to the EMA teacher weights (DMD2 initializes the
    generator from the pretrained diffusion model; this also keeps the student
    competent at every scheduled timestep, not just t=T-1). Optionally load them
    from a one-step M3/M4 checkpoint instead. The head shares mu_fake's backbone.
    """
    teacher, diffusion = load_teacher(teacher_ckpt, device=device, which="ema")
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    cfg = torch.load(teacher_ckpt, map_location=device)["model_config"]
    student = MLPDenoiser.from_config(cfg).to(device)
    mu_fake = MLPDenoiser.from_config(cfg).to(device)

    if init_student_ckpt:
        student.load_state_dict(torch.load(init_student_ckpt, map_location=device)["model_state_dict"])
    else:
        student.load_state_dict(copy.deepcopy(teacher.state_dict()))

    if init_fake_ckpt:
        mu_fake.load_state_dict(torch.load(init_fake_ckpt, map_location=device)["model_state_dict"])
    else:
        mu_fake.load_state_dict(copy.deepcopy(teacher.state_dict()))

    head = dmd2.DiscHead(hidden_dim=cfg["hidden_dim"]).to(device)
    return student, mu_fake, head, teacher, diffusion
