"""M0 entry point: build pi_0 and pi_1, plot them, compute sanity metrics.

Run via the repo's single command:  ./run.sh m0
"""

import argparse
import os

from . import distributions as dist
from . import metrics
from . import plotting
from .utils import ensure_dir, load_config, make_rng, save_json


def main():
    parser = argparse.ArgumentParser(description="M0: 2D toy data milestone.")
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    rng = make_rng(cfg["seed"])
    n = cfg["n_samples"]
    side = cfg["pi1"]["side"]
    gap = cfg["pi1"]["gap"]
    out_dir = ensure_dir(cfg["output_dir"])

    pi0 = dist.sample_pi0(n, rng, std=cfg["pi0"]["std"])
    pi1 = dist.sample_pi1(n, rng, side=side, gap=gap)
    centers = dist.mode_centers(side=side, gap=gap)

    overlay_path = os.path.join(out_dir, "pi0_pi1_overlay.png")
    hist_path = os.path.join(out_dir, "pi1_hist2d.png")
    plotting.plot_overlay(pi0, pi1, overlay_path)
    plotting.plot_hist2d(pi1, hist_path)

    # We draw n samples from pi_1 independent from the previous ones,
    # to compute the two-sample distances.
    # MMD and energy distance should be near 0
    pi1_b = dist.sample_pi1(n, rng, side=side, gap=gap)

    results = {
        "config": cfg,
        "n_samples": n,
        "n_modes_expected": len(centers),
        "pi1_mode_coverage": {
            "recovered_mode_count": metrics.recovered_mode_count(pi1, centers),
            "normalized_mode_entropy": metrics.normalized_mode_entropy(pi1, centers),
            "off_support_fraction": metrics.off_support_fraction(pi1, centers, side=side),
        },
        "pi0_moments": metrics.gaussian_moments(pi0),
        "two_sample_selftest_pi1_vs_pi1": {
            "note": "Both batches are drawn following pi_1, so metrics should be near 0",
            "n": n,
            "energy_distance": metrics.energy_distance(pi1, pi1_b),
            "mmd_rbf": metrics.mmd_rbf(pi1, pi1_b),
        },
    }

    metrics_path = os.path.join(out_dir, "metrics.json")
    save_json(results, metrics_path)

    print(f"[M0] wrote {overlay_path}")
    print(f"[M0] wrote {hist_path}")
    print(f"[M0] wrote {metrics_path}")
    print("[M0] mode coverage:", results["pi1_mode_coverage"])


if __name__ == "__main__":
    main()
