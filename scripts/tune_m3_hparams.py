"""TEMPORARY M3.1 hyperparameter tuning for the DMD student (removable).

Reuses the clean M3 code (src/train_dmd.train_variant, src/dmd, src/metrics, ...)
without modifying it: the dependency direction is tuning -> M3 only. Trains a
small curated set of DMD students, ranks them with a mode-coverage-first
composite metric, and writes a copy-paste-ready best_config.yaml plus figures.

Run:  python -m scripts.tune_m3_hparams --config configs/dmd_tuning.yaml

Delete to remove tuning entirely:
  configs/dmd_tuning.yaml, scripts/, outputs/m3_tuning/
(plus the README "M3.1" subsection and the ai_logs tuning entry).
"""

import argparse
import os

import numpy as np
import torch
import yaml

from src import distributions as dist
from src import dmd
from src import metrics
from src import plotting
from src.diffusion import load_teacher
from src.train_dmd import resolve_device, train_variant
from src.utils import ensure_dir, get_eval_noise, load_config, save_json


def composite_score(m, w_off):
    """Distributional quality + a light off-support tie-breaker."""
    return m["energy_distance"] + m["mmd_rbf"] + w_off * m["off_support_fraction"]


def is_valid(m, ranking):
    return (m["recovered_mode_count"] >= ranking["min_modes"]
            and m["normalized_mode_entropy"] >= ranking["min_entropy"])


def best_by(records, predicate, key="score"):
    """Lowest-`key` record among those satisfying predicate, or None."""
    pool = [r for r in records if predicate(r)]
    if not pool:
        return None
    return min(pool, key=lambda r: r[key])


def render_trial_figures(rec, vdir, pi1_plot, onestep_plot, multistep_plot, centers):
    samples = rec["plot_samples"]
    name = rec["name"]
    plotting.plot_samples(samples, os.path.join(vdir, "student_samples.png"),
                          title=f"M3.1 student ({name})", color="tab:blue")
    plotting.plot_scatter_row(
        [(pi1_plot, r"True $\pi_1$", "tab:orange"),
         (onestep_plot, "M2 one-step teacher", "tab:red"),
         (samples, f"M3.1 student ({name})", "tab:blue")],
        os.path.join(vdir, "compare_true_onestep_student.png"),
    )
    plotting.plot_scatter_row(
        [(pi1_plot, r"True $\pi_1$", "tab:orange"),
         (multistep_plot, "M1 multi-step teacher", "tab:green"),
         (samples, f"M3.1 student ({name})", "tab:blue")],
        os.path.join(vdir, "compare_true_multistep_student.png"),
    )
    plotting.plot_mode_counts(
        {"true": metrics.mode_fractions(pi1_plot, centers),
         "one-step": metrics.mode_fractions(onestep_plot, centers),
         "student": metrics.mode_fractions(samples, centers)},
        os.path.join(vdir, "mode_counts.png"),
    )
    plotting.plot_training_curves(rec["curves"], os.path.join(vdir, "training_curves.png"))


def best_config_dict(cfg, rec):
    """Clean M3-style config (mirrors configs/dmd.yaml) with the best hparams."""
    base, d = cfg["base"], cfg["diffusion"]
    h = rec["hparams"]
    return {
        "seed": cfg["seed"],
        "device": cfg["device"],
        "teacher_checkpoint": cfg["teacher_checkpoint"],
        "output_dir": "outputs/m3",
        "eval_noise": "outputs/eval_noise.pt",
        "diffusion": {"num_timesteps": d["num_timesteps"],
                      "beta_start": d["beta_start"], "beta_end": d["beta_end"]},
        "t_gen": base["t_gen"],
        "t_min_frac": h["t_min_frac"],
        "t_max_frac": h["t_max_frac"],
        "train_steps": base["train_steps"],
        "batch_size": base["batch_size"],
        "gen_lr": h["gen_lr"],
        "fake_lr": h["fake_lr"],
        "log_every": base["log_every"],
        "lambda_reg": h["lambda_reg"],
        "reg_loss_type": h["reg_loss_type"],
        "reg": {"n_pairs": base["reg"]["n_pairs"], "path": "outputs/m3/reg_pairs.pt",
                "det_sampler": base["reg"]["det_sampler"]},
        "eval": {"n_samples": cfg["eval"]["n_samples_plot"]},
        "variants": [{"name": rec["name"], "lambda_reg": h["lambda_reg"],
                      "fake_updates_per_gen": h["fake_updates_per_gen"]}],
    }


def write_summary_md(path, records, ranking, baseline_score, m2_one, m1_multi):
    w, w_alt = ranking["off_support_weight"], ranking["off_support_weight_alt"]
    header = ("| rank | trial | lambda | reg | fu | gen_lr | fake_lr | modes | "
              "entropy | off-sup | ED | MMD | score | score(0.25) | valid |")
    sep = "|---:|---|---:|---|---:|---:|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|"
    lines = ["# M3.1 hyperparameter tuning summary (ranked, vs true pi_1)", "",
             f"Composite score = ED + MMD + {w} * off_support (lower is better); "
             f"trials with <8 modes or entropy <{ranking['min_entropy']} are gated "
             f"out (penalty {ranking['invalid_penalty']}).", "",
             header, sep]
    for i, r in enumerate(records):
        h, m = r["hparams"], r["metrics"]
        lines.append(
            f"| {i + 1} | {r['name']} | {h['lambda_reg']:g} | "
            f"{h['reg_loss_type'] if h['lambda_reg'] > 0 else '-'} | "
            f"{h['fake_updates_per_gen']} | {h['gen_lr']:g} | {h['fake_lr']:g} | "
            f"{m['recovered_mode_count']} | {m['normalized_mode_entropy']:.3f} | "
            f"{m['off_support_fraction']:.3f} | {m['energy_distance']:.4f} | "
            f"{m['mmd_rbf']:.4f} | {r['score']:.4f} | {r['score_alt']:.4f} | "
            f"{'yes' if r['valid'] else 'NO'} |")
    lines += [
        "",
        "Reference rows (not ranked):",
        "",
        "| reference | modes | entropy | off-sup | ED | MMD |",
        "|---|:---:|:---:|:---:|:---:|:---:|",
        (f"| M1 multi-step teacher | {m1_multi['recovered_mode_count']} | "
         f"{m1_multi['normalized_mode_entropy']:.3f} | {m1_multi['off_support_fraction']:.3f} | "
         f"{m1_multi['energy_distance']:.4f} | {m1_multi['mmd_rbf']:.4f} |"),
        (f"| M2 one-step teacher | {m2_one['recovered_mode_count']} | "
         f"{m2_one['normalized_mode_entropy']:.3f} | {m2_one['off_support_fraction']:.3f} | "
         f"{m2_one['energy_distance']:.4f} | {m2_one['mmd_rbf']:.4f} |"),
        "",
        f"Original M3 best (baseline_no_reg) composite score: {baseline_score:.4f}.",
        "",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="M3.1 DMD hyperparameter tuning (temporary).")
    parser.add_argument("--config", default="configs/dmd_tuning.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = cfg["seed"]
    device = resolve_device(cfg["device"])
    ranking = cfg["ranking"]
    w_off, w_alt = ranking["off_support_weight"], ranking["off_support_weight_alt"]
    out_dir = ensure_dir(cfg["output_dir"])
    print(f"[M3.1] device: {device}; {len(cfg['trials'])} trials")

    teacher_ckpt = cfg["teacher_checkpoint"]
    if not os.path.exists(teacher_ckpt):
        raise FileNotFoundError(f"Teacher checkpoint not found at {teacher_ckpt}. Run ./run.sh m1 first.")

    ck = torch.load(teacher_ckpt, map_location=device)
    side, gap = ck["data"]["side"], ck["data"]["gap"]
    centers = dist.mode_centers(side=side, gap=gap)

    teacher, diffusion = load_teacher(teacher_ckpt, device=device, which="ema")
    T = diffusion.num_timesteps
    base = cfg["base"]
    t_gen = base["t_gen"] if base["t_gen"] is not None else T - 1

    n_plot = cfg["eval"]["n_samples_plot"]
    n_metrics = cfg["eval"]["n_samples_metrics"]
    rng = np.random.default_rng(seed)
    pi1_plot = dist.sample_pi1(n_plot, rng, side=side, gap=gap)
    pi1_metrics = dist.sample_pi1(n_metrics, rng, side=side, gap=gap)
    x_T_plot = get_eval_noise(cfg["eval_noise_plot"], n_plot, 2, seed, device)
    x_T_metrics = get_eval_noise(cfg["eval_noise_metrics"], n_metrics, 2, seed, device)

    # Teacher references: plots from the 2k noise, M2 one-step metrics from the 10k.
    with torch.no_grad():
        onestep_plot = diffusion.onestep_x0(teacher, x_T_plot).cpu().numpy()
        torch.manual_seed(seed)
        multistep_plot = diffusion.p_sample_loop(teacher, x_init=x_T_plot).cpu().numpy()
        onestep_metrics = diffusion.onestep_x0(teacher, x_T_metrics).cpu().numpy()
        torch.manual_seed(seed)
        multistep_metrics = diffusion.p_sample_loop(teacher, x_init=x_T_metrics).cpu().numpy()
    m2_one = metrics.evaluate_samples(onestep_metrics, pi1_metrics, centers, side=side)
    m1_multi = metrics.evaluate_samples(multistep_metrics, pi1_metrics, centers, side=side)

    z_ref, y_ref = dmd.build_regression_pairs(
        diffusion, teacher, base["reg"]["n_pairs"], seed, device,
        base["reg"]["path"], teacher_ckpt,
    )

    records = []
    for i, trial in enumerate(cfg["trials"]):
        eff = {**base, **trial}
        name = trial["name"]
        variant = {"name": name, "lambda_reg": eff["lambda_reg"],
                   "fake_updates_per_gen": eff["fake_updates_per_gen"]}
        hparams = {k: eff[k] for k in ("lambda_reg", "reg_loss_type", "fake_updates_per_gen",
                                       "gen_lr", "fake_lr", "t_min_frac", "t_max_frac")}
        hparams["t_gen"] = t_gen

        torch.manual_seed(seed)   # identical noise stream across trials -> fair comparison
        student, _mu_fake, vdiff, curves = train_variant(
            eff, variant, teacher_ckpt, z_ref, y_ref, t_gen, device
        )
        with torch.no_grad():
            metric_samples = dmd.generate(vdiff, student, x_T_metrics, t_gen).cpu().numpy()
            plot_samples = dmd.generate(vdiff, student, x_T_plot, t_gen).cpu().numpy()
        m = metrics.evaluate_samples(metric_samples, pi1_metrics, centers, side=side)
        valid = is_valid(m, ranking)
        score = composite_score(m, w_off) + (0.0 if valid else ranking["invalid_penalty"])
        score_alt = composite_score(m, w_alt) + (0.0 if valid else ranking["invalid_penalty"])

        rec = {
            "name": name, "hparams": hparams, "metrics": m,
            "valid": valid, "score": score, "score_alt": score_alt,
            "plot_samples": plot_samples, "curves": curves,
            "student_state": {k: v.cpu() for k, v in student.state_dict().items()},
            "model_config": student.config,
        }
        records.append(rec)

        vdir = ensure_dir(os.path.join(out_dir, f"trial_{i:02d}_{name}"))
        save_json({"name": name, "hparams": hparams}, os.path.join(vdir, "config.json"))
        save_json({"name": name, "metrics": m, "valid": valid,
                   "score": score, "score_alt": score_alt}, os.path.join(vdir, "metrics.json"))
        print(f"[M3.1] trial {i:02d} {name}: score={score:.4f} valid={valid} {m}")

    records.sort(key=lambda r: r["score"])
    by_name = {r["name"]: r for r in records}
    baseline = by_name["baseline_no_reg"]

    best = records[0]
    best_reg = best_by(records, lambda r: r["hparams"]["lambda_reg"] > 0)
    best_no_reg = best_by(records, lambda r: r["hparams"]["lambda_reg"] == 0)
    best_ttur = best_by(records, lambda r: r["hparams"]["fake_updates_per_gen"] > 1)

    selected, seen = [], set()
    for r in (best, baseline, best_reg, best_no_reg, best_ttur):
        if r is not None and r["name"] not in seen:
            seen.add(r["name"])
            selected.append(r)
    for r in selected:
        idx = next(i for i, t in enumerate(cfg["trials"]) if t["name"] == r["name"])
        vdir = os.path.join(out_dir, f"trial_{idx:02d}_{r['name']}")
        render_trial_figures(r, vdir, pi1_plot, onestep_plot, multistep_plot, centers)

    # Best-variant top-level artifacts.
    plotting.plot_samples(best["plot_samples"], os.path.join(out_dir, "best_variant_samples.png"),
                          title=f"M3.1 best student ({best['name']})", color="tab:blue")
    plotting.plot_scatter_row(
        [(pi1_plot, r"True $\pi_1$", "tab:orange"),
         (onestep_plot, "M2 one-step teacher", "tab:red"),
         (best["plot_samples"], f"M3.1 best ({best['name']})", "tab:blue")],
        os.path.join(out_dir, "best_variant_comparison.png"),
    )
    plotting.plot_mode_counts(
        {"true": metrics.mode_fractions(pi1_plot, centers),
         "one-step": metrics.mode_fractions(onestep_plot, centers),
         "best": metrics.mode_fractions(best["plot_samples"], centers)},
        os.path.join(out_dir, "best_variant_mode_counts.png"),
    )
    save_json({"name": best["name"], "hparams": best["hparams"], "metrics": best["metrics"],
               "score": best["score"], "score_alt": best["score_alt"]},
              os.path.join(out_dir, "best_metrics.json"))
    torch.save({"model_state_dict": best["student_state"], "model_config": best["model_config"],
                "hparams": best["hparams"]}, os.path.join(out_dir, "best_student.pt"))

    best_cfg = best_config_dict(cfg, best)
    with open(os.path.join(out_dir, "best_config.yaml"), "w") as f:
        yaml.safe_dump(best_cfg, f, sort_keys=False)

    summary = {
        "references": {"teacher_multistep_ema_vs_pi1": m1_multi,
                       "teacher_onestep_ema_vs_pi1": m2_one},
        "ranking": ranking,
        "best": best["name"],
        "original_m3_best": "baseline_no_reg",
        "trials": [{"name": r["name"], "hparams": r["hparams"], "metrics": r["metrics"],
                    "valid": r["valid"], "score": r["score"], "score_alt": r["score_alt"]}
                   for r in records],
    }
    save_json(summary, os.path.join(out_dir, "trials_summary.json"))
    write_summary_md(os.path.join(out_dir, "trials_summary.md"),
                     records, ranking, baseline["score"], m2_one, m1_multi)

    _print_report(best, baseline, best_reg, best_no_reg, best_ttur, m2_one, best_cfg, out_dir)


def _print_report(best, baseline, best_reg, best_no_reg, best_ttur, m2_one, best_cfg, out_dir):
    print("\n" + "=" * 70)
    print(f"[M3.1] best trial: {best['name']}  (score {best['score']:.4f})")
    print(f"[M3.1] best hparams: {best['hparams']}")
    print(f"[M3.1] best metrics: {best['metrics']}")
    print(f"[M3.1] original M3 best (baseline_no_reg) score: {baseline['score']:.4f}")
    improves = best["score"] < baseline["score"] and best["name"] != baseline["name"]
    print(f"[M3.1] improves over original M3: {improves}")
    print(f"[M3.1] best beats M2 one-step (energy distance): "
          f"{best['metrics']['energy_distance'] < m2_one['energy_distance']}")
    if best_reg is not None:
        print(f"[M3.1] regression helped (best reg < baseline): "
              f"{best_reg['score'] < baseline['score']}  "
              f"(best reg: {best_reg['name']} {best_reg['score']:.4f})")
    if best_ttur is not None:
        print(f"[M3.1] TTUR helped (best ttur < baseline): "
              f"{best_ttur['score'] < baseline['score']}  "
              f"(best ttur: {best_ttur['name']} {best_ttur['score']:.4f})")

    print("\n[M3.1] suggested patch for configs/dmd.yaml (copy if clearly better):")
    h = best["hparams"]
    print(f"  gen_lr: {h['gen_lr']:g}")
    print(f"  fake_lr: {h['fake_lr']:g}")
    print(f"  t_min_frac: {h['t_min_frac']:g}")
    print(f"  t_max_frac: {h['t_max_frac']:g}")
    print(f"  lambda_reg: {h['lambda_reg']:g}")
    print(f"  reg_loss_type: {h['reg_loss_type']}")
    print(f"  # winning variant: name={best['name']}, lambda_reg={h['lambda_reg']:g}, "
          f"fake_updates_per_gen={h['fake_updates_per_gen']}")
    print(f"  (full clean config written to {os.path.join(out_dir, 'best_config.yaml')})")

    print("\n[M3.1] temporary files to delete before submission (if dropping tuning):")
    print("  configs/dmd_tuning.yaml, scripts/, outputs/m3_tuning/")
    print("  (plus the README 'M3.1' subsection and the ai_logs tuning entry)")
    print("=" * 70)


if __name__ == "__main__":
    main()
