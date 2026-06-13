"""M4 entry point: DMD2 Stage B (DMD + TTUR + GAN) one-step distillation.

Continues training the best M3 student with the regression loss dropped, TTUR
kept (fake_updates_per_gen), and a GAN loss on noised samples added. The critic is
a linear head on mu_fake's shared last hidden layer (DMD2 shared-backbone design),
and the GAN noising timestep range is restricted to low/mid noise so the
discriminator gets a usable signal. Compares against the best M3 student.

`train_m4` is reused by the M4.1 tuning script.

Run via:  ./run.sh m4
"""

import argparse
import os

import numpy as np
import torch
from tqdm import tqdm

from . import distributions as dist
from . import dmd
from . import dmd2
from . import metrics
from . import plotting
from .diffusion import load_teacher
from .models import MLPDenoiser
from .train_dmd import resolve_device
from .utils import ensure_dir, get_eval_noise, load_config, save_json, set_reproducibility


def _grad_global_norm(grads):
    sq = sum((g.detach() ** 2).sum() for g in grads if g is not None)
    return float(torch.sqrt(sq).item()) if torch.is_tensor(sq) else 0.0


def train_m4(cfg, device, side, gap):
    """Train one DMD2 Stage B student; returns models, diffusion, and curves.

    Shared across the clean M4 run and the M4.1 tuning sweep. The discriminator is
    a head on mu_fake's bottleneck; opt_disc updates head + mu_fake jointly.
    """
    student, mu_fake, head, teacher, diffusion = dmd2.build_m4_models(
        cfg["teacher_checkpoint"], cfg["m3_student_checkpoint"],
        cfg.get("m3_fake_checkpoint"), device,
    )
    T = diffusion.num_timesteps
    t_gen = cfg["t_gen"] if cfg["t_gen"] is not None else T - 1
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
    rng = np.random.default_rng(cfg["seed"])

    curves = {"DMD surrogate": [], "fake score": [], "discriminator": [],
              "generator GAN": [], "generator total": []}
    diag_curves = {"mean_real_logit": [], "mean_fake_logit": [], "disc_total_acc": [],
                   "disc_grad_norm": [], "gen_gan_grad_norm": [], "dmd_grad_norm": []}

    for step in tqdm(range(1, cfg["train_steps"] + 1), desc="dmd2"):
        # 1) Fake score updates (TTUR): standard DDPM eps loss on student samples.
        fake_loss_val = 0.0
        for _ in range(fake_updates):
            z = torch.randn(bs, 2, device=device)
            x_fake = dmd.generate(diffusion, student, z, t_gen).detach()
            loss_fake = diffusion.loss(mu_fake, x_fake)
            opt_fake.zero_grad()
            loss_fake.backward()
            opt_fake.step()
            fake_loss_val = loss_fake.item()

        # 2) Discriminator updates (head + shared mu_fake) on noised samples.
        disc_loss_val = 0.0
        disc_gn = 0.0
        for _ in range(disc_updates):
            x_real = torch.from_numpy(
                dist.sample_pi1(bs, rng, side=side, gap=gap)
            ).float().to(device)
            z = torch.randn(bs, 2, device=device)
            x_fake = dmd.generate(diffusion, student, z, t_gen).detach()
            t = dmd2.sample_gan_t(bs, T, device, gan_t_min, gan_t_max)
            loss_d = dmd2.disc_loss(mu_fake, head, diffusion, x_real, x_fake, t)
            opt_disc.zero_grad()
            loss_d.backward()
            disc_gn = _grad_global_norm([p.grad for p in disc_params])
            opt_disc.step()
            disc_loss_val = loss_d.item()

        # 3) Generator update: DMD distribution matching + GAN (student only).
        z = torch.randn(bs, 2, device=device)
        x_fake = dmd.generate(diffusion, student, z, t_gen)
        loss_dmd = dmd.dmd_generator_loss(diffusion, teacher, mu_fake, x_fake, t_min, t_max)
        t = dmd2.sample_gan_t(bs, T, device, gan_t_min, gan_t_max)
        loss_gan = dmd2.gen_gan_loss(mu_fake, head, diffusion, x_fake, t)
        loss_g = loss_dmd + lambda_gan * loss_gan

        is_log = step % cfg["log_every"] == 0
        dmd_gn = gan_gn = 0.0
        if is_log:
            sp = list(student.parameters())
            dmd_gn = _grad_global_norm(
                torch.autograd.grad(loss_dmd, sp, retain_graph=True, allow_unused=True))
            gan_gn = _grad_global_norm(
                torch.autograd.grad(loss_gan, sp, retain_graph=True, allow_unused=True))
        opt_gen.zero_grad()
        loss_g.backward()
        opt_gen.step()

        if is_log:
            curves["DMD surrogate"].append((step, loss_dmd.item()))
            curves["fake score"].append((step, fake_loss_val))
            curves["discriminator"].append((step, disc_loss_val))
            curves["generator GAN"].append((step, loss_gan.item()))
            curves["generator total"].append((step, loss_g.item()))
            x_real = torch.from_numpy(
                dist.sample_pi1(bs, rng, side=side, gap=gap)
            ).float().to(device)
            x_fake_d = dmd.generate(diffusion, student, z, t_gen).detach()
            t = dmd2.sample_gan_t(bs, T, device, gan_t_min, gan_t_max)
            d = dmd2.disc_diagnostics(mu_fake, head, diffusion, x_real, x_fake_d, t)
            diag_curves["mean_real_logit"].append((step, d["mean_real_logit"]))
            diag_curves["mean_fake_logit"].append((step, d["mean_fake_logit"]))
            diag_curves["disc_total_acc"].append((step, d["total_acc"]))
            diag_curves["disc_grad_norm"].append((step, disc_gn))
            diag_curves["gen_gan_grad_norm"].append((step, gan_gn))
            diag_curves["dmd_grad_norm"].append((step, dmd_gn))

    return student, mu_fake, head, diffusion, curves, diag_curves


def build_references(diffusion, teacher, m3_student, x_T, pi1, pi1_b, pi0, centers, side, seed):
    """Reference metrics and sample arrays shared by the clean run and tuning."""
    t_gen = diffusion.num_timesteps - 1
    with torch.no_grad():
        m3_samples = dmd.generate(diffusion, m3_student, x_T, t_gen).cpu().numpy()
        onestep = diffusion.onestep_x0(teacher, x_T).cpu().numpy()
        torch.manual_seed(seed)
        multistep = diffusion.p_sample_loop(teacher, x_init=x_T).cpu().numpy()

    def ev(samples):
        return metrics.evaluate_samples(samples, pi1, centers, side=side)

    references = {
        "pi1_vs_pi1_sanity": ev(pi1_b),
        "pi0_vs_pi1_baseline": ev(pi0),
        "teacher_multistep_ema_vs_pi1": ev(multistep),
        "teacher_onestep_ema_vs_pi1": ev(onestep),
        "m3_best_vs_pi1": ev(m3_samples),
    }
    arrays = {"onestep": onestep, "multistep": multistep, "m3_samples": m3_samples}
    return references, arrays


def main():
    parser = argparse.ArgumentParser(description="M4: DMD2 Stage B.")
    parser.add_argument("--config", default="configs/dmd2.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = resolve_device(cfg["device"])
    if device == "cuda":
        print(f"[M4] device: cuda ({torch.cuda.get_device_name(0)})")
    else:
        print(f"[M4] device: {device}")

    seed = cfg["seed"]
    set_reproducibility(seed)
    rng = np.random.default_rng(seed)
    out_dir = ensure_dir(cfg["output_dir"])

    teacher_ckpt = cfg["teacher_checkpoint"]
    m3_student_ckpt = cfg["m3_student_checkpoint"]
    for path in (teacher_ckpt, m3_student_ckpt):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required checkpoint not found: {path}. Run M1/M3 first.")

    # Geometry from the teacher checkpoint, so M4 matches M1/M2/M3.
    ck = torch.load(teacher_ckpt, map_location=device)
    side, gap = ck["data"]["side"], ck["data"]["gap"]
    n_eval = cfg["eval"]["n_samples"]

    student, mu_fake, head, diffusion, curves, diag_curves = train_m4(cfg, device, side, gap)
    T = diffusion.num_timesteps
    t_gen = cfg["t_gen"] if cfg["t_gen"] is not None else T - 1

    # ---- Evaluation against the shared fixed x_T ----
    centers = dist.mode_centers(side=side, gap=gap)
    pi1 = dist.sample_pi1(n_eval, rng, side=side, gap=gap)
    pi1_b = dist.sample_pi1(n_eval, rng, side=side, gap=gap)
    pi0 = dist.sample_pi0(n_eval, rng, std=1.0)
    x_T = get_eval_noise(cfg["eval_noise"], n_eval, 2, seed, device)

    teacher, _ = load_teacher(teacher_ckpt, device=device, which="ema")

    m3_ck = torch.load(m3_student_ckpt, map_location=device)
    m3_student = MLPDenoiser.from_config(m3_ck["model_config"]).to(device)
    m3_student.load_state_dict(m3_ck["model_state_dict"])
    m3_student.eval()

    references, arrays = build_references(
        diffusion, teacher, m3_student, x_T, pi1, pi1_b, pi0, centers, side, seed)
    onestep, multistep, m3_samples = arrays["onestep"], arrays["multistep"], arrays["m3_samples"]

    student.eval()
    with torch.no_grad():
        m4_samples = dmd.generate(diffusion, student, x_T, t_gen).cpu().numpy()
    m4 = metrics.evaluate_samples(m4_samples, pi1, centers, side=side)
    m3 = references["m3_best_vs_pi1"]

    # ---- Save checkpoints (CPU-loadable) ----
    torch.save({"model_state_dict": {k: v.cpu() for k, v in student.state_dict().items()},
                "model_config": student.config}, os.path.join(out_dir, "dmd2_student.pt"))
    torch.save({"model_state_dict": {k: v.cpu() for k, v in mu_fake.state_dict().items()},
                "model_config": mu_fake.config}, os.path.join(out_dir, "dmd2_mu_fake.pt"))
    torch.save({"model_state_dict": {k: v.cpu() for k, v in head.state_dict().items()},
                "model_config": head.config}, os.path.join(out_dir, "discriminator.pt"))
    save_json({"lambda_gan": cfg["lambda_gan"], "fake_updates_per_gen": cfg["fake_updates_per_gen"],
               "disc_updates_per_gen": cfg["disc_updates_per_gen"], "t_gen": t_gen,
               "gan_t_min_frac": cfg["gan_t_min_frac"], "gan_t_max_frac": cfg["gan_t_max_frac"],
               "gen_lr": cfg["gen_lr"], "fake_lr": cfg["fake_lr"], "disc_lr": cfg["disc_lr"]},
              os.path.join(out_dir, "config.json"))
    save_json({"vs_pi1": m4}, os.path.join(out_dir, "metrics.json"))
    save_json(diag_curves, os.path.join(out_dir, "diagnostics.json"))

    # ---- Figures ----
    plotting.plot_training_curves(curves, os.path.join(out_dir, "training_curves.png"))
    plotting.plot_gan_diagnostics(diag_curves, os.path.join(out_dir, "gan_diagnostics.png"))
    plotting.plot_samples(m4_samples, os.path.join(out_dir, "dmd2_samples.png"),
                          title="M4 DMD2 student", color="tab:purple")
    plotting.plot_scatter_row(
        [(pi1, r"True $\pi_1$", "tab:orange"),
         (m3_samples, "Best M3 student", "tab:blue"),
         (m4_samples, "M4 DMD2 student", "tab:purple")],
        os.path.join(out_dir, "compare_m3_vs_m4.png"),
    )
    plotting.plot_scatter_row(
        [(pi1, r"True $\pi_1$", "tab:orange"),
         (onestep, "M2 one-step teacher", "tab:red"),
         (m3_samples, "Best M3 student", "tab:blue"),
         (m4_samples, "M4 DMD2 student", "tab:purple")],
        os.path.join(out_dir, "compare_true_m2_m3_m4.png"),
    )
    plotting.plot_mode_counts(
        {"true": metrics.mode_fractions(pi1, centers),
         "M3": metrics.mode_fractions(m3_samples, centers),
         "M4": metrics.mode_fractions(m4_samples, centers)},
        os.path.join(out_dir, "mode_counts.png"),
    )

    # ---- Summary ----
    improves = m4["energy_distance"] < m3["energy_distance"]
    final_diag = {k: (v[-1][1] if v else None) for k, v in diag_curves.items()}
    summary = {
        "references": references,
        "m4_dmd2_vs_pi1": m4,
        "m4_improves_over_m3": improves,
        "m4_mmd_improves_over_m3": m4["mmd_rbf"] < m3["mmd_rbf"],
        "final_gan_diagnostics": final_diag,
    }
    save_json(summary, os.path.join(out_dir, "summary_metrics.json"))
    write_summary_md(summary, os.path.join(out_dir, "summary_metrics.md"))

    for name, m in list(references.items()) + [("m4_dmd2_vs_pi1", m4)]:
        print(f"[M4] {name}: {m}")
    print(f"[M4] final GAN diagnostics: {final_diag}")
    print(f"[M4] M4 improves over best M3 (energy distance): {improves}")
    print(f"[M4] M4 improves over best M3 (MMD): {summary['m4_mmd_improves_over_m3']}")


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
        "m3_best_vs_pi1": "M3 best student (no-reg TTUR)",
    }
    lines = ["# M4 summary metrics (vs true pi_1)", "", header, sep]
    for key in ["pi1_vs_pi1_sanity", "pi0_vs_pi1_baseline",
                "teacher_multistep_ema_vs_pi1", "teacher_onestep_ema_vs_pi1",
                "m3_best_vs_pi1"]:
        lines.append(row(label_map[key], summary["references"][key]))
    lines.append(row("**M4 DMD2 student**", summary["m4_dmd2_vs_pi1"]))
    d = summary["final_gan_diagnostics"]
    lines += ["",
              f"M4 improves over best M3 (energy distance): "
              f"**{summary['m4_improves_over_m3']}**; MMD: "
              f"**{summary['m4_mmd_improves_over_m3']}**.",
              "",
              f"Final GAN diagnostics: mean real logit "
              f"{d['mean_real_logit']:.3f}, mean fake logit {d['mean_fake_logit']:.3f}, "
              f"disc total accuracy {d['disc_total_acc']:.3f}.", ""]
    with open(path, "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
