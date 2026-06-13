"""Figures for M0, M1, M2, and M3."""

import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_overlay(pi0, pi1, path):
    """Overlay scatter of source and target, mirroring the brief's Figure 1."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(pi0[:, 0], pi0[:, 1], s=10, alpha=0.45, c="tab:blue", label=r"$\pi_0$ (Gaussian)")
    ax.scatter(pi1[:, 0], pi1[:, 1], s=10, alpha=0.45, c="tab:orange", label=r"$\pi_1$ (checkerboard)")
    ax.set_aspect("equal")
    ax.set_title(r"Source $\pi_0$ and target $\pi_1$")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_hist2d(pi1, path, bins=50):
    """2D histogram of the target, highlighting the 8 modes."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.hist2d(pi1[:, 0], pi1[:, 1], bins=bins, cmap="inferno")
    ax.set_aspect("equal")
    ax.set_title(r"Target $\pi_1$ density (8 modes)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_samples(points, path, title, color="tab:green", lim=3.0):
    """Single scatter of a sample set with fixed limits."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(points[:, 0], points[:, 1], s=6, alpha=0.4, c=color)
    ax.set_aspect("equal")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_title(title)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_loss_curve(losses, path):
    """Training loss vs logged step."""
    steps, values = zip(*losses)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(steps, values, c="tab:blue")
    ax.set_xlabel("training step")
    ax.set_ylabel("DDPM loss (MSE on eps)")
    ax.set_title("Teacher training loss")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_compare(true, teacher, path, lim=3.0):
    """Side-by-side scatter: true pi_1 vs teacher samples, shared limits."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    for ax, pts, title, color in (
        (axes[0], true, r"True $\pi_1$", "tab:orange"),
        (axes[1], teacher, "DDPM teacher (multi-step)", "tab:green"),
    ):
        ax.scatter(pts[:, 0], pts[:, 1], s=6, alpha=0.4, c=color)
        ax.set_aspect("equal")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_title(title)
        ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_compare_triptych(true, multistep, onestep, path, lim=3.0):
    """Three scatters: true pi_1 | multi-step EMA teacher | one-step EMA teacher."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    for ax, pts, title, color in (
        (axes[0], true, r"True $\pi_1$", "tab:orange"),
        (axes[1], multistep, "M1 multi-step teacher (EMA)", "tab:green"),
        (axes[2], onestep, "M2 one-step teacher (EMA)", "tab:red"),
    ):
        ax.scatter(pts[:, 0], pts[:, 1], s=6, alpha=0.4, c=color)
        ax.set_aspect("equal")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_title(title)
        ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_mode_counts(fracs_by_label, path):
    """Grouped bar chart of per-mode mass fractions, to expose mode imbalance.

    `fracs_by_label` maps a label -> array of per-mode fractions (length n_modes).
    """
    labels = list(fracs_by_label)
    n_modes = len(next(iter(fracs_by_label.values())))
    x = np.arange(n_modes)
    width = 0.8 / len(labels)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, label in enumerate(labels):
        ax.bar(x + i * width, fracs_by_label[label], width=width, label=label)
    ax.axhline(1.0 / n_modes, color="gray", ls="--", lw=1, alpha=0.7, label="uniform")
    ax.set_xlabel("mode index")
    ax.set_ylabel("fraction of samples")
    ax.set_title("Per-mode mass: true vs multi-step vs one-step")
    ax.set_xticks(x + width * (len(labels) - 1) / 2)
    ax.set_xticklabels([str(i) for i in range(n_modes)])
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_metric_bars(values_by_label, path, ylabel, title, hline=None, hline_label=None):
    """Single bar chart comparing one scalar metric across labelled models."""
    labels = list(values_by_label)
    values = [values_by_label[k] for k in labels]
    colors = ["tab:orange", "tab:red", "tab:blue", "tab:purple", "tab:green",
              "tab:brown", "tab:gray"][: len(labels)]
    fig, ax = plt.subplots(figsize=(1.4 * len(labels) + 2, 4.5))
    bars = ax.bar(labels, values, color=colors)
    if hline is not None:
        ax.axhline(hline, color="gray", ls="--", lw=1, alpha=0.8, label=hline_label)
        if hline_label:
            ax.legend()
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.2)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_scatter_row(panels, path, suptitle=None, lim=3.0):
    """Generic row of scatters from (points, title, color) tuples, shared limits."""
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(5.0 * n, 5.2))
    if n == 1:
        axes = [axes]
    for ax, (pts, title, color) in zip(axes, panels):
        ax.scatter(pts[:, 0], pts[:, 1], s=6, alpha=0.4, c=color)
        ax.set_aspect("equal")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_title(title)
        ax.grid(True, alpha=0.2)
    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_training_curves(curves, path):
    """Multi-curve plot from a dict label -> list of (step, value)."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label, series in curves.items():
        if not series:
            continue
        steps, values = zip(*series)
        ax.plot(steps, values, label=label)
    ax.set_xlabel("generator step")
    ax.set_ylabel("loss")
    ax.set_title("DMD training curves")
    ax.grid(True, alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_gan_diagnostics(diag, path):
    """Three-panel GAN diagnostics: logits, discriminator accuracy, grad norms.

    `diag` maps each key -> list of (step, value). Mixed scales are split across
    panels so the discriminator's behaviour is readable at a glance.
    """
    def series(key):
        return zip(*diag[key]) if diag.get(key) else ([], [])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    for key, label in (("mean_real_logit", "mean real logit"),
                       ("mean_fake_logit", "mean fake logit")):
        s, v = series(key)
        axes[0].plot(s, v, label=label)
    axes[0].axhline(0.0, color="gray", ls="--", lw=1, alpha=0.7)
    axes[0].set_title("Discriminator logits")
    axes[0].set_xlabel("generator step")
    axes[0].legend()
    axes[0].grid(True, alpha=0.2)

    s, v = series("disc_total_acc")
    axes[1].plot(s, v, color="tab:green", label="total acc")
    axes[1].axhline(0.5, color="gray", ls="--", lw=1, alpha=0.7, label="chance")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_title("Discriminator accuracy")
    axes[1].set_xlabel("generator step")
    axes[1].legend()
    axes[1].grid(True, alpha=0.2)

    for key, label in (("disc_grad_norm", "disc grad norm"),
                       ("gen_gan_grad_norm", "gen GAN grad norm"),
                       ("dmd_grad_norm", "DMD grad norm")):
        s, v = series(key)
        if len(list(s)):
            s, v = series(key)
            axes[2].plot(s, v, label=label)
    axes[2].set_yscale("log")
    axes[2].set_title("Gradient norms")
    axes[2].set_xlabel("generator step")
    axes[2].legend()
    axes[2].grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_gan_diagnostics(diag, path):
    """Three-panel GAN diagnostics: logits, discriminator accuracy, grad norms.

    `diag` maps a key -> list of (step, value). Keys are grouped by scale so the
    very different magnitudes (logits, [0,1] accuracy, grad norms) stay readable.
    """
    panels = [
        ("logits", ["mean_real_logit", "mean_fake_logit"]),
        ("discriminator accuracy", ["disc_total_acc"]),
        ("gradient norms", ["disc_grad_norm", "gen_gan_grad_norm", "dmd_grad_norm"]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, (title, keys) in zip(axes, panels):
        for key in keys:
            series = diag.get(key, [])
            if not series:
                continue
            steps, values = zip(*series)
            ax.plot(steps, values, label=key)
        if title == "discriminator accuracy":
            ax.axhline(0.5, color="gray", ls="--", lw=1, alpha=0.7, label="chance")
        ax.set_xlabel("generator step")
        ax.set_title(title)
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle("M4 GAN diagnostics")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_reverse_snapshots(snapshots, timesteps, path, lim=3.0):
    """Row of scatters showing the same particles at selected reverse steps."""
    n = len(snapshots)
    fig, axes = plt.subplots(1, n, figsize=(2.6 * n, 3.0))
    for ax, pts, t in zip(axes, snapshots, timesteps):
        ax.scatter(pts[:, 0], pts[:, 1], s=4, alpha=0.4, c="tab:purple")
        ax.set_aspect("equal")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_title(f"t = {t}")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("DDPM reverse process (noise -> checkerboard)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _scatter_frame(points, t, lim):
    """Render one scatter frame to an RGB array for the GIF."""
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.scatter(points[:, 0], points[:, 1], s=5, alpha=0.5, c="tab:purple")
    ax.set_aspect("equal")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_title(f"t = {t}")
    fig.tight_layout()
    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return frame


def save_reverse_gif(trajectory, path, num_timesteps, n_frames=60, fps=15, lim=3.0):
    """GIF of the reverse trajectory.

    trajectory[j] is x_{T-j}; we subsample to n_frames evenly spaced frames.
    """
    total = len(trajectory)
    idxs = np.linspace(0, total - 1, num=min(n_frames, total)).astype(int)
    frames = []
    for j in idxs:
        t = num_timesteps - j
        frames.append(Image.fromarray(_scatter_frame(np.asarray(trajectory[j]), t, lim)))
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=round(1000.0 / fps), loop=0)
