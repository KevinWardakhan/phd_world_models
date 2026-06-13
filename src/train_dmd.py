"""M3 entry point: DMD Stage A distillation with a small reg/TTUR ablation.

Trains one-step DMD students from the frozen EMA teacher and compares them with
the M2 naive one-step teacher and the M1 multi-step teacher.

Run via:  ./run.sh m3
"""

import argparse
import os

import numpy as np
import torch
from tqdm import tqdm

from . import distributions as dist
from . import dmd
from . import metrics
from . import plotting
from .diffusion import load_teacher
from .utils import ensure_dir, get_eval_noise, load_config, save_json, set_reproducibility


def resolve_device(name):
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name


def train_variant(cfg, variant, teacher_ckpt, z_ref, y_ref, t_gen, device):
    student, mu_fake, teacher, diffusion = dmd.build_student_and_fake(teacher_ckpt, device)
    opt_gen = torch.optim.Adam(student.parameters(), lr=cfg["gen_lr"])
    opt_fake = torch.optim.Adam(mu_fake.parameters(), lr=cfg["fake_lr"])

    T = diffusion.num_timesteps
    t_min = int(cfg["t_min_frac"] * T)
    t_max = int(cfg["t_max_frac"] * T) - 1
    bs = cfg["batch_size"]
    lambda_reg = variant["lambda_reg"]
    fake_updates = variant["fake_updates_per_gen"]
    reg_kind = cfg["reg_loss_type"]
    n_ref = z_ref.shape[0]

    curves = {"generator (KL)": [], "fake score": [], "regression": []}
    for step in tqdm(range(1, cfg["train_steps"] + 1), desc=variant["name"]):
        # Update the fake score model on current (detached) student samples.
        fake_loss_val = 0.0
        for _ in range(fake_updates):
            z = torch.randn(bs, 2, device=device)
            x_fake = dmd.generate(diffusion, student, z, t_gen).detach()
            loss_fake = diffusion.loss(mu_fake, x_fake)
            opt_fake.zero_grad()
            loss_fake.backward()
            opt_fake.step()
            fake_loss_val = loss_fake.item()

        # Generator step: distribution matching (+ optional regression).
        z = torch.randn(bs, 2, device=device)
        x = dmd.generate(diffusion, student, z, t_gen)
        loss_kl = dmd.dmd_generator_loss(diffusion, teacher, mu_fake, x, t_min, t_max)
        reg_val = 0.0
        loss = loss_kl
        if lambda_reg > 0:
            idx = torch.randint(0, n_ref, (bs,), device=device)
            pred = dmd.generate(diffusion, student, z_ref[idx], t_gen)
            loss_reg = dmd.regression_loss(pred, y_ref[idx], kind=reg_kind)
            loss = loss_kl + lambda_reg * loss_reg
            reg_val = loss_reg.item()
        opt_gen.zero_grad()
        loss.backward()
        opt_gen.step()

        if step % cfg["log_every"] == 0:
            curves["generator (KL)"].append((step, loss_kl.item()))
            curves["fake score"].append((step, fake_loss_val))
            if lambda_reg > 0:
                curves["regression"].append((step, reg_val))
    return student, mu_fake, diffusion, curves


def main():
    parser = argparse.ArgumentParser(description="M3: DMD Stage A.")
    parser.add_argument("--config", default="configs/dmd.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = resolve_device(cfg["device"])
    print(f"[M3] device: {device}")

    set_reproducibility(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    out_dir = ensure_dir(cfg["output_dir"])
    teacher_ckpt = cfg["teacher_checkpoint"]
    if not os.path.exists(teacher_ckpt):
        raise FileNotFoundError(
            f"Teacher checkpoint not found at {teacher_ckpt}. Run ./run.sh m1 first."
        )

    # Geometry from the teacher checkpoint, so M3 matches M1/M2.
    ck = torch.load(teacher_ckpt, map_location=device)
    side, gap = ck["data"]["side"], ck["data"]["gap"]
    n_eval = cfg["eval"]["n_samples"]

    teacher, diffusion = load_teacher(teacher_ckpt, device=device, which="ema")
    T = diffusion.num_timesteps
    t_gen = cfg["t_gen"] if cfg["t_gen"] is not None else T - 1

    centers = dist.mode_centers(side=side, gap=gap)
    pi1 = dist.sample_pi1(n_eval, rng, side=side, gap=gap)
    pi1_b = dist.sample_pi1(n_eval, rng, side=side, gap=gap)
    pi0 = dist.sample_pi0(n_eval, rng, std=1.0)
    x_T = get_eval_noise(cfg["eval_noise"], n_eval, 2, cfg["seed"], device)

    # Teacher references on the shared fixed noise.
    with torch.no_grad():
        onestep = diffusion.onestep_x0(teacher, x_T).cpu().numpy()
        torch.manual_seed(cfg["seed"])
        multistep = diffusion.p_sample_loop(teacher, x_init=x_T).cpu().numpy()

    references = {
        "pi1_vs_pi1_sanity": metrics.evaluate_samples(pi1_b, pi1, centers, side=side),
        "pi0_vs_pi1_baseline": metrics.evaluate_samples(pi0, pi1, centers, side=side),
        "teacher_multistep_ema_vs_pi1": metrics.evaluate_samples(multistep, pi1, centers, side=side),
        "teacher_onestep_ema_vs_pi1": metrics.evaluate_samples(onestep, pi1, centers, side=side),
    }
    m2_one = references["teacher_onestep_ema_vs_pi1"]

    # Deterministic regression pairs (shared across variants).
    z_ref, y_ref = dmd.build_regression_pairs(
        diffusion, teacher, cfg["reg"]["n_pairs"], cfg["seed"], device,
        cfg["reg"]["path"], teacher_ckpt,
    )

    variant_metrics = {}
    for variant in cfg["variants"]:
        name = variant["name"]
        vdir = ensure_dir(os.path.join(out_dir, name))
        student, mu_fake, vdiff, curves = train_variant(
            cfg, variant, teacher_ckpt, z_ref, y_ref, t_gen, device
        )

        with torch.no_grad():
            samples = dmd.generate(vdiff, student, x_T, t_gen).cpu().numpy()
        m = metrics.evaluate_samples(samples, pi1, centers, side=side)
        variant_metrics[name] = m

        torch.save({"model_state_dict": student.state_dict(),
                    "model_config": student.config, "variant": variant}, os.path.join(vdir, "student.pt"))
        torch.save({"model_state_dict": mu_fake.state_dict(),
                    "model_config": mu_fake.config, "variant": variant}, os.path.join(vdir, "mu_fake.pt"))
        save_json({"variant": variant}, os.path.join(vdir, "config.json"))
        save_json({"variant": variant, "vs_pi1": m}, os.path.join(vdir, "metrics.json"))

        plotting.plot_training_curves(curves, os.path.join(vdir, "training_curves.png"))
        plotting.plot_samples(samples, os.path.join(vdir, "student_samples.png"),
                              title=f"M3 DMD student ({name})", color="tab:blue")
        plotting.plot_scatter_row(
            [(pi1, r"True $\pi_1$", "tab:orange"),
             (onestep, "M2 one-step teacher", "tab:red"),
             (samples, f"M3 student ({name})", "tab:blue")],
            os.path.join(vdir, "compare_true_onestep_student.png"),
        )
        plotting.plot_scatter_row(
            [(pi1, r"True $\pi_1$", "tab:orange"),
             (multistep, "M1 multi-step teacher", "tab:green"),
             (samples, f"M3 student ({name})", "tab:blue")],
            os.path.join(vdir, "compare_true_multistep_student.png"),
        )
        plotting.plot_mode_counts(
            {"true": metrics.mode_fractions(pi1, centers),
             "one-step": metrics.mode_fractions(onestep, centers),
             "student": metrics.mode_fractions(samples, centers)},
            os.path.join(vdir, "mode_counts.png"),
        )
        print(f"[M3] {name}: {m}")

    # Summary: references + variants, plus best variant by energy distance.
    best = min(variant_metrics, key=lambda k: variant_metrics[k]["energy_distance"])
    summary = {
        "references": references,
        "variants": variant_metrics,
        "best_variant": best,
        "best_beats_m2_onestep": variant_metrics[best]["energy_distance"] < m2_one["energy_distance"],
    }
    save_json(summary, os.path.join(out_dir, "summary_metrics.json"))
    write_summary_md(summary, os.path.join(out_dir, "summary_metrics.md"))

    print(f"[M3] best variant (lowest energy distance): {best}")
    print(f"[M3] best beats M2 one-step: {summary['best_beats_m2_onestep']}")


def write_summary_md(summary, path):
    header = "| comparison | modes | norm. entropy | off-support | energy dist | MMD |"
    sep = "|---|:---:|:---:|:---:|:---:|:---:|"

    def row(label, m):
        return (f"| {label} | {m['recovered_mode_count']} | "
                f"{m['normalized_mode_entropy']:.3f} | {m['off_support_fraction']:.3f} | "
                f"{m['energy_distance']:.4f} | {m['mmd_rbf']:.4f} |")

    label_map = {
        "pi1_vs_pi1_sanity": "pi_1 vs pi_1 (sanity)",
        "pi0_vs_pi1_baseline": "pi_0 vs pi_1 (baseline)",
        "teacher_multistep_ema_vs_pi1": "M1 multi-step teacher",
        "teacher_onestep_ema_vs_pi1": "M2 one-step teacher",
    }
    lines = ["# M3 summary metrics (vs true pi_1)", "", header, sep]
    for key in ["pi1_vs_pi1_sanity", "pi0_vs_pi1_baseline",
                "teacher_multistep_ema_vs_pi1", "teacher_onestep_ema_vs_pi1"]:
        lines.append(row(label_map[key], summary["references"][key]))
    for name, m in summary["variants"].items():
        marker = " (best)" if name == summary["best_variant"] else ""
        lines.append(row(f"M3 {name}{marker}", m))
    lines += ["", f"Best variant: **{summary['best_variant']}**; "
              f"beats M2 one-step: **{summary['best_beats_m2_onestep']}**.", ""]
    with open(path, "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
