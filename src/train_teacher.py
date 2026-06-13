"""M1 entry point: train (or load) the DDPM teacher on pi_1 and evaluate it.

Run via the repo's single command:  ./run.sh m1
"""

import argparse
import os

import numpy as np
import torch
from tqdm import tqdm

from . import distributions as dist
from . import metrics
from . import plotting
from .diffusion import GaussianDiffusion, load_teacher
from .ema import EMA
from .models import MLPDenoiser
from .utils import ensure_dir, load_config, save_json, set_reproducibility


def resolve_device(name):
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name


def train(cfg, model, diffusion, device, rng):
    side, gap = cfg["data"]["side"], cfg["data"]["gap"]
    bs = cfg["train"]["batch_size"]
    opt = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"])

    ema_cfg = cfg["ema"]
    ema = EMA(model, ema_cfg["decay"]) if ema_cfg["enabled"] else None

    losses = []
    running = 0.0
    model.train()
    for step in tqdm(range(1, cfg["train"]["steps"] + 1), desc="train"):
        x0 = dist.sample_pi1(bs, rng, side=side, gap=gap)
        x0 = torch.as_tensor(x0, dtype=torch.float32, device=device)

        loss = diffusion.loss(model, x0)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if ema is not None and step >= ema_cfg["start_step"]:
            ema.update(model)

        running += loss.item()
        if step % cfg["train"]["log_every"] == 0:
            losses.append((step, running / cfg["train"]["log_every"]))
            running = 0.0
    return losses, ema


def sample_metrics(diffusion, model, n_eval, reference, centers, side):
    samples = diffusion.p_sample_loop(model, n_eval).cpu().numpy()
    return samples, metrics.evaluate_samples(samples, reference, centers, side=side)


def main():
    parser = argparse.ArgumentParser(description="M1: DDPM teacher.")
    parser.add_argument("--config", default="configs/teacher.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = resolve_device(cfg["device"])
    print(f"[M1] device: {device}")

    set_reproducibility(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    out_dir = ensure_dir(cfg["output_dir"])
    side, gap = cfg["data"]["side"], cfg["data"]["gap"]
    ckpt_path = cfg["checkpoint"]

    losses = []
    if cfg["load_if_exists"] and os.path.exists(ckpt_path):
        print(f"[M1] loading existing checkpoint {ckpt_path}")
        model_raw, diffusion = load_teacher(ckpt_path, device=device, which="raw")
        model_ema, _ = load_teacher(ckpt_path, device=device, which="ema")
    else:
        diffusion = GaussianDiffusion(device=device, **cfg["diffusion"])
        model_raw = MLPDenoiser(**cfg["model"]).to(device)
        losses, ema = train(cfg, model_raw, diffusion, device, rng)
        model_ema = ema.ema_model if ema is not None else None

    model_raw.eval()
    main_model = model_ema if model_ema is not None else model_raw
    ema_used = model_ema is not None

    # Evaluation samples.
    n_eval = cfg["eval"]["n_samples"]
    pi1 = dist.sample_pi1(n_eval, rng, side=side, gap=gap)
    pi1_b = dist.sample_pi1(n_eval, rng, side=side, gap=gap)
    pi0 = dist.sample_pi0(n_eval, rng, std=1.0)
    centers = dist.mode_centers(side=side, gap=gap)

    _, raw_m = sample_metrics(diffusion, model_raw, n_eval, pi1, centers, side)
    comparisons = {
        "pi1_vs_pi1_sanity": metrics.evaluate_samples(pi1_b, pi1, centers, side=side),
        "pi0_vs_pi1_baseline": metrics.evaluate_samples(pi0, pi1, centers, side=side),
        "teacher_raw_vs_pi1": raw_m,
    }
    if ema_used:
        _, ema_m = sample_metrics(diffusion, model_ema, n_eval, pi1, centers, side)
        comparisons["teacher_ema_vs_pi1"] = ema_m

    # Headline teacher = EMA when available.
    teacher_m = comparisons["teacher_ema_vs_pi1"] if ema_used else raw_m
    comparisons["teacher_vs_pi1"] = teacher_m
    base_m = comparisons["pi0_vs_pi1_baseline"]

    # Off-support is a sharpness signal: a continuous DDPM smooths the hard
    # square edges, so some bleed is expected. We require it to clearly beat the
    # pi_0 baseline rather than hit an absolute near-zero floor.
    acceptance = {
        "recovers_8_modes": teacher_m["recovered_mode_count"] >= 8,
        "high_mode_entropy": teacher_m["normalized_mode_entropy"] >= 0.95,
        "beats_pi0_off_support": teacher_m["off_support_fraction"] < base_m["off_support_fraction"],
        "beats_pi0_energy": teacher_m["energy_distance"] < base_m["energy_distance"],
        "beats_pi0_mmd": teacher_m["mmd_rbf"] < base_m["mmd_rbf"],
    }

    # Save the frozen teacher (EMA used for evaluation when enabled).
    torch.save(
        {
            "model_state_dict": model_raw.state_dict(),
            "ema_state_dict": model_ema.state_dict() if ema_used else None,
            "model_config": model_raw.config,
            "diffusion_config": diffusion.config,
            "data": {"side": side, "gap": gap},
            "config": cfg,
            "metrics": comparisons,
            "ema_used_for_eval": ema_used,
        },
        ckpt_path,
    )
    print(f"[M1] wrote {ckpt_path}")

    # Figures use the headline (EMA) model.
    label = "EMA" if ema_used else "raw"
    teacher_samples = diffusion.p_sample_loop(main_model, n_eval).cpu().numpy()
    if losses:
        plotting.plot_loss_curve(losses, os.path.join(out_dir, "loss_curve.png"))
    plotting.plot_compare(pi1, teacher_samples, os.path.join(out_dir, "compare_true_vs_teacher.png"))
    plotting.plot_samples(
        teacher_samples,
        os.path.join(out_dir, "teacher_samples.png"),
        title=f"DDPM teacher samples ({label}, multi-step)",
    )

    # Reverse trajectory -> GIF and static snapshots
    T = diffusion.num_timesteps
    _, traj = diffusion.p_sample_loop(main_model, cfg["gif"]["n_points"], return_trajectory=True)
    plotting.save_reverse_gif(
        traj,
        os.path.join(out_dir, "reverse_process.gif"),
        num_timesteps=T,
        n_frames=cfg["gif"]["n_frames"],
        fps=cfg["gif"]["fps"],
    )
    snap_ts = [T, int(0.75 * T), int(0.5 * T), int(0.25 * T), int(0.1 * T), 0]
    snapshots = [np.asarray(traj[T - t]) for t in snap_ts]
    plotting.plot_reverse_snapshots(
        snapshots, snap_ts, os.path.join(out_dir, "reverse_snapshots.png")
    )

    results = {
        "config": cfg,
        "device": device,
        "n_eval": n_eval,
        "ema": {**cfg["ema"], "used_for_eval": ema_used},
        "comparisons": comparisons,
        "acceptance": acceptance,
        "acceptance_passed": all(acceptance.values()),
        "final_loss": losses[-1][1] if losses else None,
    }
    save_json(results, os.path.join(out_dir, "metrics.json"))

    print(f"[M1] teacher raw vs pi_1: {raw_m}")
    if ema_used:
        print(f"[M1] teacher EMA vs pi_1: {comparisons['teacher_ema_vs_pi1']}")
    print(
        "[M1] off-support  raw: {:.3f}  {}: {:.3f}".format(
            raw_m["off_support_fraction"], label, teacher_m["off_support_fraction"]
        )
    )
    print(f"[M1] headline teacher = {label}; acceptance:", acceptance,
          "passed:", results["acceptance_passed"])


if __name__ == "__main__":
    main()
