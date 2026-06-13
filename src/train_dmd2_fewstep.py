"""M5 entry point: few-step DMD2 student (DMD2 sec 4.4 + 4.5).

Continues the DMD2 Stage B recipe (distribution matching + TTUR + GAN, no
regression) but with an N-step generator and backward simulation instead of a
one-step generator. Goal: a 4-step student that sharpens the checkerboard
boundaries, i.e. clearly lower off-support fraction than the M3 one-step student,
while keeping all 8 modes.

`train_m5` is reused by the +/-GAN / +/-TTUR / 1-step-vs-4-step ablation.

Run via:  ./run.sh m5
"""

import argparse
import copy
import os

import numpy as np
import torch
from tqdm import tqdm

from . import distributions as dist
from . import dmd
from . import dmd2
from . import dmd2_fewstep as fs
from . import metrics
from . import plotting
from .diffusion import load_teacher
from .models import MLPDenoiser
from .train_dmd import resolve_device
from .train_dmd2 import _grad_global_norm
from .utils import ensure_dir, get_eval_noise, load_config, save_json, set_reproducibility


def build_schedule(cfg, T):
    if cfg.get("schedule_fracs"):
        return fs.schedule_from_fracs(T, cfg["schedule_fracs"])
    return fs.default_schedule(T, cfg["num_gen_steps"])


def train_m5(cfg, device, side, gap):
    """Train one few-step DMD2 student; returns models, diffusion, schedule, curves.

    lambda_gan=0 disables the GAN (the -GAN ablation arm); a single-entry schedule
    (num_gen_steps=1) reduces to one-step DMD2 (the 1-step ablation arm).
    """
    student, mu_fake, head, teacher, diffusion = fs.build_m5_models(
        cfg["teacher_checkpoint"], device,
        cfg.get("init_student_checkpoint"), cfg.get("init_fake_checkpoint"),
    )
    T = diffusion.num_timesteps
    schedule = build_schedule(cfg, T)

    t_min = int(cfg["t_min_frac"] * T)
    t_max = int(cfg["t_max_frac"] * T) - 1
    gan_t_min = int(cfg["gan_t_min_frac"] * T)
    gan_t_max = int(cfg["gan_t_max_frac"] * T) - 1

    opt_gen = torch.optim.Adam(student.parameters(), lr=cfg["gen_lr"])
    opt_fake = torch.optim.Adam(mu_fake.parameters(), lr=cfg["fake_lr"])
    disc_params = list(head.parameters()) + list(mu_fake.parameters())
    opt_disc = torch.optim.Adam(disc_params, lr=cfg["disc_lr"])

    bs = cfg["batch_size"]
    fake_updates = cfg["fake_updates_per_gen"]
    disc_updates = cfg["disc_updates_per_gen"]
    lambda_gan = cfg["lambda_gan"]
    use_gan = lambda_gan > 0
    rng = np.random.default_rng(cfg["seed"])
    rng_steps = np.random.default_rng(cfg["seed"] + 1)

    curves = {"DMD surrogate": [], "fake score": [], "discriminator": [],
              "generator GAN": [], "generator total": []}
    diag_curves = {"mean_real_logit": [], "mean_fake_logit": [], "disc_total_acc": [],
                   "disc_grad_norm": [], "gen_gan_grad_norm": [], "dmd_grad_norm": []}

    desc = f"dmd2-m5 (n_steps={len(schedule)}, gan={use_gan}, fu={fake_updates})"
    for step in tqdm(range(1, cfg["train_steps"] + 1), desc=desc):
        # 1) Fake score updates (TTUR) on detached student samples (backward sim).
        fake_loss_val = 0.0
        for _ in range(fake_updates):
            x_fake = fs.simulated_fake(diffusion, student, schedule, bs, device, rng_steps).detach()
            loss_fake = diffusion.loss(mu_fake, x_fake)
            opt_fake.zero_grad()
            loss_fake.backward()
            opt_fake.step()
            fake_loss_val = loss_fake.item()

        # 2) Discriminator updates (head + shared mu_fake) on noised real/fake.
        disc_loss_val = 0.0
        disc_gn = 0.0
        if use_gan:
            for _ in range(disc_updates):
                x_real = torch.from_numpy(
                    dist.sample_pi1(bs, rng, side=side, gap=gap)
                ).float().to(device)
                x_fake = fs.simulated_fake(diffusion, student, schedule, bs, device, rng_steps).detach()
                t = dmd2.sample_gan_t(bs, T, device, gan_t_min, gan_t_max)
                loss_d = dmd2.disc_loss(mu_fake, head, diffusion, x_real, x_fake, t)
                opt_disc.zero_grad()
                loss_d.backward()
                disc_gn = _grad_global_norm([p.grad for p in disc_params])
                opt_disc.step()
                disc_loss_val = loss_d.item()

        # 3) Generator update: DMD distribution matching (+ optional GAN).
        x_fake = fs.simulated_fake(diffusion, student, schedule, bs, device, rng_steps)
        loss_dmd = dmd.dmd_generator_loss(diffusion, teacher, mu_fake, x_fake, t_min, t_max)
        loss_g = loss_dmd
        loss_gan_val = 0.0
        if use_gan:
            t = dmd2.sample_gan_t(bs, T, device, gan_t_min, gan_t_max)
            loss_gan = dmd2.gen_gan_loss(mu_fake, head, diffusion, x_fake, t)
            loss_g = loss_dmd + lambda_gan * loss_gan
            loss_gan_val = loss_gan.item()

        is_log = step % cfg["log_every"] == 0
        dmd_gn = gan_gn = 0.0
        if is_log:
            sp = list(student.parameters())
            dmd_gn = _grad_global_norm(
                torch.autograd.grad(loss_dmd, sp, retain_graph=True, allow_unused=True))
            if use_gan:
                gan_gn = _grad_global_norm(
                    torch.autograd.grad(loss_gan, sp, retain_graph=True, allow_unused=True))
        opt_gen.zero_grad()
        loss_g.backward()
        opt_gen.step()

        if is_log:
            curves["DMD surrogate"].append((step, loss_dmd.item()))
            curves["fake score"].append((step, fake_loss_val))
            curves["discriminator"].append((step, disc_loss_val))
            curves["generator GAN"].append((step, loss_gan_val))
            curves["generator total"].append((step, loss_g.item()))
            if use_gan:
                x_real = torch.from_numpy(
                    dist.sample_pi1(bs, rng, side=side, gap=gap)
                ).float().to(device)
                x_fake_d = fs.simulated_fake(diffusion, student, schedule, bs, device, rng_steps).detach()
                t = dmd2.sample_gan_t(bs, T, device, gan_t_min, gan_t_max)
                d = dmd2.disc_diagnostics(mu_fake, head, diffusion, x_real, x_fake_d, t)
                diag_curves["mean_real_logit"].append((step, d["mean_real_logit"]))
                diag_curves["mean_fake_logit"].append((step, d["mean_fake_logit"]))
                diag_curves["disc_total_acc"].append((step, d["total_acc"]))
                diag_curves["disc_grad_norm"].append((step, disc_gn))
                diag_curves["gen_gan_grad_norm"].append((step, gan_gn))
                diag_curves["dmd_grad_norm"].append((step, dmd_gn))

    return student, mu_fake, head, diffusion, schedule, curves, diag_curves


def _eval_onestep_ckpt(ckpt_path, diffusion, device, x_T, pi1, centers, side):
    """Load a one-step student checkpoint and evaluate it on the fixed noise."""
    ck = torch.load(ckpt_path, map_location=device)
    model = MLPDenoiser.from_config(ck["model_config"]).to(device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    t_gen = diffusion.num_timesteps - 1
    with torch.no_grad():
        samples = dmd.generate(diffusion, model, x_T, t_gen).cpu().numpy()
    return samples, metrics.evaluate_samples(samples, pi1, centers, side=side)


def run_ablation(cfg, device, side, gap, x_T, pi1, centers, main_metrics, out_dir):
    """+/-GAN, +/-TTUR, 1-step vs 4-step, all via train_m5 with matched setup.

    The main M5 student (4-step, +GAN, fu=3) is already trained; its metrics are
    reused for that row. The other arms are trained here with the same teacher-EMA
    init, steps and hyperparameters, so only the ablated axis differs.
    """
    steps = cfg.get("ablation_train_steps", cfg["train_steps"])
    arms = [
        ("1-step, no GAN (fu=3)", {"num_gen_steps": 1, "lambda_gan": 0.0, "fake_updates_per_gen": 3}),
        ("1-step, +GAN (fu=3)", {"num_gen_steps": 1, "lambda_gan": cfg["lambda_gan"], "fake_updates_per_gen": 3}),
        ("4-step, no GAN, fu=1 (-TTUR)", {"num_gen_steps": 4, "lambda_gan": 0.0, "fake_updates_per_gen": 1}),
        ("4-step, no GAN (fu=3)", {"num_gen_steps": 4, "lambda_gan": 0.0, "fake_updates_per_gen": 3}),
    ]
    table = {}
    for label, override in arms:
        acfg = copy.deepcopy(cfg)
        acfg.update(override)
        acfg["train_steps"] = steps
        acfg["schedule_fracs"] = None if override["num_gen_steps"] != 4 else cfg.get("schedule_fracs")
        student, _, _, diffusion, schedule, _, _ = train_m5(acfg, device, side, gap)
        student.eval()
        with torch.no_grad():
            s = fs.generate_multistep(diffusion, student, x_T, schedule).cpu().numpy()
        table[label] = metrics.evaluate_samples(s, pi1, centers, side=side)
    table["4-step, +GAN (fu=3) [main M5]"] = main_metrics

    save_json(table, os.path.join(out_dir, "ablation_summary.json"))
    write_ablation_md(table, os.path.join(out_dir, "ablation_summary.md"))
    return table


def write_ablation_md(table, path):
    header = "| variant | steps | GAN | TTUR | modes | norm. entropy | off-support | energy dist | MMD |"
    sep = "|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|"
    meta = {
        "1-step, no GAN (fu=3)": (1, "no", "fu=3"),
        "1-step, +GAN (fu=3)": (1, "yes", "fu=3"),
        "4-step, no GAN, fu=1 (-TTUR)": (4, "no", "fu=1"),
        "4-step, no GAN (fu=3)": (4, "no", "fu=3"),
        "4-step, +GAN (fu=3) [main M5]": (4, "yes", "fu=3"),
    }
    lines = ["# M5 ablation: +/-GAN, +/-TTUR, 1-step vs 4-step",
             "", "Ranked by off-support fraction (sharper boundaries), lower is better. "
             "All arms share the teacher-EMA init, step budget and hyperparameters.",
             "", header, sep]
    order = sorted(table, key=lambda k: table[k]["off_support_fraction"])
    for k in order:
        m = table[k]
        n_steps, gan, ttur = meta.get(k, ("?", "?", "?"))
        lines.append(
            f"| {k} | {n_steps} | {gan} | {ttur} | {m['recovered_mode_count']} | "
            f"{m['normalized_mode_entropy']:.3f} | {m['off_support_fraction']:.3f} | "
            f"{m['energy_distance']:.4f} | {m['mmd_rbf']:.4f} |")
    lines.append("")
    with open(path, "w") as f:
        f.write("\n".join(lines))


def robustness_eval(diffusion, student, schedule, centers, side, gap, cfg, device, seed):
    """Off-support / modes at n=2000 (2 seeds) and n=5000, to show the win is robust."""
    n2 = cfg["eval"]["n_samples"]
    n5 = cfg["eval"].get("n_samples_large", 5000)
    results = {}
    student.eval()
    for tag, n, s in (("n2000_seedA", n2, seed), ("n2000_seedB", n2, seed + 7),
                      ("n5000_seedA", n5, seed)):
        rng = np.random.default_rng(s)
        pi1 = dist.sample_pi1(n, rng, side=side, gap=gap)
        gen = torch.Generator().manual_seed(s)
        x_T = torch.randn(n, 2, generator=gen).to(device)
        torch.manual_seed(s)
        with torch.no_grad():
            samples = fs.generate_multistep(diffusion, student, x_T, schedule).cpu().numpy()
        results[tag] = metrics.evaluate_samples(samples, pi1, centers, side=side)
    return results


def main():
    parser = argparse.ArgumentParser(description="M5: few-step DMD2 (sec 4.4 + 4.5).")
    parser.add_argument("--config", default="configs/dmd2_fewstep.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = resolve_device(cfg["device"])
    if device == "cuda":
        print(f"[M5] device: cuda ({torch.cuda.get_device_name(0)})")
    else:
        print(f"[M5] device: {device}")

    seed = cfg["seed"]
    set_reproducibility(seed)
    rng = np.random.default_rng(seed)
    out_dir = ensure_dir(cfg["output_dir"])

    teacher_ckpt = cfg["teacher_checkpoint"]
    for path in (teacher_ckpt, cfg["m3_student_checkpoint"], cfg["m4_student_checkpoint"]):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required checkpoint not found: {path}. Run M1/M3/M4 first.")

    ck = torch.load(teacher_ckpt, map_location=device)
    side, gap = ck["data"]["side"], ck["data"]["gap"]
    n_eval = cfg["eval"]["n_samples"]

    # ---- Train the main few-step student (4-step, +GAN, fu=3) ----
    student, mu_fake, head, diffusion, schedule, curves, diag_curves = train_m5(cfg, device, side, gap)
    T = diffusion.num_timesteps

    # ---- Shared evaluation set ----
    centers = dist.mode_centers(side=side, gap=gap)
    pi1 = dist.sample_pi1(n_eval, rng, side=side, gap=gap)
    pi1_b = dist.sample_pi1(n_eval, rng, side=side, gap=gap)
    pi0 = dist.sample_pi0(n_eval, rng, std=1.0)
    x_T = get_eval_noise(cfg["eval_noise"], n_eval, 2, seed, device)

    teacher, _ = load_teacher(teacher_ckpt, device=device, which="ema")
    with torch.no_grad():
        onestep = diffusion.onestep_x0(teacher, x_T).cpu().numpy()
        torch.manual_seed(seed)
        multistep = diffusion.p_sample_loop(teacher, x_init=x_T).cpu().numpy()

    def ev(samples):
        return metrics.evaluate_samples(samples, pi1, centers, side=side)

    m3_samples, m3 = _eval_onestep_ckpt(cfg["m3_student_checkpoint"], diffusion, device, x_T, pi1, centers, side)
    m4_samples, m4 = _eval_onestep_ckpt(cfg["m4_student_checkpoint"], diffusion, device, x_T, pi1, centers, side)

    student.eval()
    with torch.no_grad():
        m5_samples = fs.generate_multistep(diffusion, student, x_T, schedule).cpu().numpy()
    m5 = ev(m5_samples)

    references = {
        "pi1_vs_pi1_sanity": ev(pi1_b),
        "pi0_vs_pi1_baseline": ev(pi0),
        "teacher_multistep_ema_vs_pi1": ev(multistep),
        "teacher_onestep_ema_vs_pi1": ev(onestep),
        "m3_best_onestep_vs_pi1": m3,
        "m4_dmd2_onestep_vs_pi1": m4,
    }

    robustness = robustness_eval(diffusion, student, schedule, centers, side, gap, cfg, device, seed)

    # ---- Checkpoints ----
    torch.save({"model_state_dict": {k: v.cpu() for k, v in student.state_dict().items()},
                "model_config": student.config, "schedule": schedule},
               os.path.join(out_dir, "fewstep_student.pt"))
    torch.save({"model_state_dict": {k: v.cpu() for k, v in mu_fake.state_dict().items()},
                "model_config": mu_fake.config}, os.path.join(out_dir, "fewstep_mu_fake.pt"))
    torch.save({"model_state_dict": {k: v.cpu() for k, v in head.state_dict().items()},
                "model_config": head.config}, os.path.join(out_dir, "discriminator.pt"))
    save_json({"num_gen_steps": cfg["num_gen_steps"], "schedule": schedule,
               "lambda_gan": cfg["lambda_gan"], "fake_updates_per_gen": cfg["fake_updates_per_gen"],
               "disc_updates_per_gen": cfg["disc_updates_per_gen"],
               "gan_t_min_frac": cfg["gan_t_min_frac"], "gan_t_max_frac": cfg["gan_t_max_frac"],
               "init": cfg.get("init_student_checkpoint") or "teacher_ema",
               "gen_lr": cfg["gen_lr"], "fake_lr": cfg["fake_lr"], "disc_lr": cfg["disc_lr"]},
              os.path.join(out_dir, "config.json"))
    save_json({"vs_pi1": m5, "robustness": robustness}, os.path.join(out_dir, "metrics.json"))
    save_json(diag_curves, os.path.join(out_dir, "diagnostics.json"))

    # ---- Figures ----
    plotting.plot_training_curves(curves, os.path.join(out_dir, "training_curves.png"))
    if any(diag_curves.values()):
        plotting.plot_gan_diagnostics(diag_curves, os.path.join(out_dir, "gan_diagnostics.png"))
    plotting.plot_samples(m5_samples, os.path.join(out_dir, "fewstep_samples.png"),
                          title=f"M5 DMD2 {len(schedule)}-step student", color="tab:purple")
    plotting.plot_scatter_row(
        [(pi1, r"True $\pi_1$", "tab:orange"),
         (m3_samples, "M3 student (1-step)", "tab:blue"),
         (m4_samples, "M4 DMD2 (1-step)", "tab:red"),
         (m5_samples, f"M5 DMD2 ({len(schedule)}-step)", "tab:purple")],
        os.path.join(out_dir, "compare_m3_m4_m5.png"),
    )
    plotting.plot_scatter_row(
        [(pi1, r"True $\pi_1$", "tab:orange"),
         (multistep, "M1 multi-step teacher", "tab:green"),
         (m5_samples, f"M5 DMD2 ({len(schedule)}-step)", "tab:purple")],
        os.path.join(out_dir, "compare_true_multistep_fewstep.png"),
    )
    plotting.plot_mode_counts(
        {"true": metrics.mode_fractions(pi1, centers),
         "M3 (1-step)": metrics.mode_fractions(m3_samples, centers),
         "M4 (1-step)": metrics.mode_fractions(m4_samples, centers),
         "M5 (few-step)": metrics.mode_fractions(m5_samples, centers)},
        os.path.join(out_dir, "mode_counts.png"),
    )
    plotting.plot_metric_bars(
        {"M3 (1-step)": m3["off_support_fraction"],
         "M4 (1-step)": m4["off_support_fraction"],
         f"M5 ({len(schedule)}-step)": m5["off_support_fraction"],
         "teacher (multi)": references["teacher_multistep_ema_vs_pi1"]["off_support_fraction"]},
        os.path.join(out_dir, "offsupport_bars.png"),
        ylabel="off-support fraction (lower = sharper)",
        title="Off-support: M5 few-step vs one-step students",
        hline=references["teacher_multistep_ema_vs_pi1"]["off_support_fraction"],
        hline_label="multi-step teacher",
    )

    # ---- Ablation ----
    ablation = None
    if cfg.get("run_ablation", True):
        ablation = run_ablation(cfg, device, side, gap, x_T, pi1, centers, m5, out_dir)

    # ---- Summary ----
    beats_off = m5["off_support_fraction"] < m3["off_support_fraction"]
    modes_ok = m5["recovered_mode_count"] == 8 and m5["normalized_mode_entropy"] >= 0.99
    ed_not_worse = m5["energy_distance"] <= m3["energy_distance"] * 1.10
    summary = {
        "references": references,
        "m5_fewstep_vs_pi1": m5,
        "robustness": robustness,
        "m5_beats_m3_off_support": beats_off,
        "m5_modes_preserved": modes_ok,
        "m5_ed_not_worse_than_m3": ed_not_worse,
        "clear_win": bool(beats_off and modes_ok and ed_not_worse),
        "ablation": ablation,
    }
    save_json(summary, os.path.join(out_dir, "summary_metrics.json"))
    write_summary_md(summary, schedule, os.path.join(out_dir, "summary_metrics.md"))

    for name, m in references.items():
        print(f"[M5] {name}: {m}")
    print(f"[M5] m5 few-step ({len(schedule)}-step): {m5}")
    print(f"[M5] robustness: {robustness}")
    print(f"[M5] beats M3 off-support: {beats_off}; modes preserved: {modes_ok}; "
          f"ED not worse: {ed_not_worse}; CLEAR WIN: {summary['clear_win']}")


def write_summary_md(summary, schedule, path):
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
        "m3_best_onestep_vs_pi1": "M3 best student (1-step)",
        "m4_dmd2_onestep_vs_pi1": "M4 DMD2 student (1-step)",
    }
    lines = ["# M5 summary metrics (vs true pi_1)", "", header, sep]
    for key in label_map:
        lines.append(row(label_map[key], summary["references"][key]))
    lines.append(row(f"**M5 DMD2 student ({len(schedule)}-step)**", summary["m5_fewstep_vs_pi1"]))
    lines += [
        "",
        f"Schedule (timesteps): {schedule}.",
        "",
        f"Beats M3 one-step on off-support: **{summary['m5_beats_m3_off_support']}**; "
        f"modes preserved (8/8, entropy>=0.99): **{summary['m5_modes_preserved']}**; "
        f"ED not worse than M3: **{summary['m5_ed_not_worse_than_m3']}**; "
        f"**clear win: {summary['clear_win']}**.",
        "",
        "Robustness (off-support fraction):",
        "",
    ]
    for tag, m in summary["robustness"].items():
        lines.append(f"- {tag}: off-support {m['off_support_fraction']:.3f}, "
                     f"modes {m['recovered_mode_count']}, ED {m['energy_distance']:.4f}")
    lines.append("")
    with open(path, "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
