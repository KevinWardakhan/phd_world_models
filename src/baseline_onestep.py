"""M2 entry point: the naive one-step teacher baseline.

Loads the frozen EMA DDPM teacher from M1 and, from a fixed Gaussian noise x_T,
produces a single x0 estimate (Eq. 15). Compares this one-step baseline against
true pi_1, the M1 multi-step EMA teacher, and the pi_0 baseline.

Run via:  ./run.sh m2
"""

import argparse
import os

import numpy as np
import torch

from . import distributions as dist
from . import metrics
from . import plotting
from .diffusion import load_teacher
from .utils import ensure_dir, get_eval_noise, load_config, save_json


def resolve_device(name):
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name


def main():
    parser = argparse.ArgumentParser(description="M2: naive one-step teacher baseline.")
    parser.add_argument("--config", default="configs/baseline.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = resolve_device(cfg["device"])
    print(f"[M2] device: {device}")

    ckpt_path = cfg["teacher_checkpoint"]
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Teacher checkpoint not found at {ckpt_path}. Run ./run.sh m1 first."
        )

    torch.manual_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    out_dir = ensure_dir(cfg["output_dir"])
    side, gap = cfg["data"]["side"], cfg["data"]["gap"]
    n_eval = cfg["eval"]["n_samples"]

    # Frozen EMA teacher only.
    model, diffusion = load_teacher(ckpt_path, device=device, which="ema")
    model.eval()

    centers = dist.mode_centers(side=side, gap=gap)
    pi1 = dist.sample_pi1(n_eval, rng, side=side, gap=gap)
    pi1_b = dist.sample_pi1(n_eval, rng, side=side, gap=gap)
    pi0 = dist.sample_pi0(n_eval, rng, std=1.0)

    with torch.no_grad():
        x_T = get_eval_noise(cfg["eval_noise"], n_eval, 2, cfg["seed"], device)
        onestep = diffusion.onestep_x0(model, x_T).cpu().numpy()
        # Same fixed x_T; seed the stochastic reverse sampler for reproducibility.
        torch.manual_seed(cfg["seed"])
        multistep = diffusion.p_sample_loop(model, x_init=x_T).cpu().numpy()

    comparisons = {
        "pi1_vs_pi1_sanity": metrics.evaluate_samples(pi1_b, pi1, centers, side=side),
        "pi0_vs_pi1_baseline": metrics.evaluate_samples(pi0, pi1, centers, side=side),
        "teacher_multistep_ema_vs_pi1": metrics.evaluate_samples(multistep, pi1, centers, side=side),
        "teacher_onestep_ema_vs_pi1": metrics.evaluate_samples(onestep, pi1, centers, side=side),
    }
    multi_m = comparisons["teacher_multistep_ema_vs_pi1"]
    one_m = comparisons["teacher_onestep_ema_vs_pi1"]
    degradation = {
        "off_support_fraction": one_m["off_support_fraction"] - multi_m["off_support_fraction"],
        "energy_distance": one_m["energy_distance"] - multi_m["energy_distance"],
        "mmd_rbf": one_m["mmd_rbf"] - multi_m["mmd_rbf"],
    }
    one_step_worse = (
        degradation["off_support_fraction"] > 0
        and degradation["energy_distance"] > 0
        and degradation["mmd_rbf"] > 0
    )

    # Figures.
    plotting.plot_samples(
        onestep,
        os.path.join(out_dir, "one_step_teacher_samples.png"),
        title="M2 one-step teacher (EMA)",
        color="tab:red",
    )
    plotting.plot_compare_triptych(
        pi1, multistep, onestep, os.path.join(out_dir, "compare_true_multistep_onestep.png")
    )
    plotting.plot_mode_counts(
        {
            "true": metrics.mode_fractions(pi1, centers),
            "multi-step": metrics.mode_fractions(multistep, centers),
            "one-step": metrics.mode_fractions(onestep, centers),
        },
        os.path.join(out_dir, "mode_counts_comparison.png"),
    )

    results = {
        "config": cfg,
        "device": device,
        "n_eval": n_eval,
        "comparisons": comparisons,
        "degradation_one_minus_multi": degradation,
        "one_step_clearly_worse": one_step_worse,
    }
    save_json(results, os.path.join(out_dir, "metrics.json"))

    print(f"[M2] teacher multi-step EMA vs pi_1: {multi_m}")
    print(f"[M2] teacher one-step  EMA vs pi_1: {one_m}")
    print(f"[M2] degradation (one-step - multi-step): {degradation}")
    print(f"[M2] one-step clearly worse than multi-step: {one_step_worse}")


if __name__ == "__main__":
    main()
