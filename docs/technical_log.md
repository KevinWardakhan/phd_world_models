# Technical log (detailed)

This is the detailed engineering and interview log for the project: full method
explanations, the M3.1 and M4.1 tuning stories, ablations, honest failures, and
the per-milestone reasoning. For a quick overview (summary, how to run, the main
metric table, and key findings) see the root [README.md](../README.md).

The end goal: transport an isotropic Gaussian source `pi_0` to a multimodal
target `pi_1` (an 8-mode checkerboard), train a multi-step diffusion teacher, and
distill it into a one/few-step generator with **DMD** then **DMD2**. Priorities:
correctness and explainability over sophistication; everything stays small and
2D; mode coverage is central because the target is multimodal and mode dropping
is the main failure mode.

> Status: **M0 + M1 + M2 + M3 + M4 + M5 complete** (data; DDPM teacher recovers `pi_1`; naive one-step baseline fails; DMD students recover all 8 modes in one step; DMD2 adds a GAN loss but lands approximately equal to M3 in one step, since the discriminator barely learns once distribution matching has solved this 2D toy; **M5 is the clear win** - a 4-step DMD2 student with backward simulation (DMD2 sec 4.4 + 4.5) cuts the off-support fraction ~2.5x vs the M3 one-step student, 0.158 -> 0.064, dropping below the multi-step teacher while keeping all 8 modes). Pipeline complete.

## Repository structure

```
.
├── run.sh              # single entry point (bootstraps venv, runs a stage)
├── requirements.txt
├── configs/            # YAML configs (data.yaml M0, teacher.yaml M1, baseline.yaml M2, dmd.yaml M3, dmd2.yaml M4)
├── src/                # runtime code (does NOT depend on resources/)
│   ├── distributions.py  # pi_0 Gaussian, pi_1 checkerboard, mode_centers
│   ├── metrics.py        # mode coverage + two-sample distances
│   ├── plotting.py       # figures + reverse-process GIF
│   ├── utils.py          # config/seed/IO helpers + fixed eval noise
│   ├── models.py         # MLP noise predictor (M1)
│   ├── ema.py            # EMA of teacher weights (M1)
│   ├── diffusion.py      # DDPM schedule, loss, samplers, one-step, eps->score, DDIM (M1/M2/M3)
│   ├── dmd.py            # DMD primitives: student/fake build, generator surrogate loss, reg pairs (M3)
│   ├── dmd2.py           # DMD2 primitives: discriminator head on mu_fake, GAN losses + diagnostics (M4)
│   ├── dmd2_fewstep.py   # DMD2 multi-step generator + backward simulation primitives (M5)
│   ├── m0_data.py        # M0 entry point
│   ├── train_teacher.py  # M1 entry point
│   ├── baseline_onestep.py  # M2 entry point
│   ├── train_dmd.py      # M3 entry point
│   ├── train_dmd2.py     # M4 entry point
│   └── train_dmd2_fewstep.py  # M5 entry point (few-step DMD2)
├── outputs/            # generated figures + metric files (m0/, m1/, m2/, m3/, m4/, m5/)
├── resources/          # assignment brief + paper markdowns (context only)
│   ├── assignment/take_home_brief.md
│   ├── papers/{ddpm,dmd,dmd2,dimo}.md
│   └── sources_index.md
└── ai_logs/            # honest log of AI-assisted sessions
```

`src/` never imports from `resources/`. The resources folder is for context,
traceability, and method justification only.

## How to run

From a clean checkout, one command per stage:

```bash
./run.sh m0   # data: pi_0, pi_1, plots, sanity metrics
./run.sh m1   # teacher: train DDPM, sample, metrics, figures, GIF
./run.sh m2   # naive one-step baseline from the frozen EMA teacher
./run.sh m3   # DMD Stage A: train one-step students (reg/TTUR ablation)
./run.sh m4   # DMD2 Stage B: continue best M3 student with TTUR + GAN loss
./run.sh m5   # few-step DMD2: 4-step student (sec 4.4 + 4.5) + 1-vs-4-step ablation
```

`./run.sh` defaults to `m0`. The script creates a local `.venv` and installs
`requirements.txt`. M1 adds `torch`, `tqdm`, `pillow`. M1 trains on CPU by
default (`device: auto` uses CUDA when available) in a couple of minutes; set
`load_if_exists: true` in `configs/teacher.yaml` to reuse a saved checkpoint
instead of retraining.

## What M0 produces

In `outputs/m0/`:

- `pi0_pi1_overlay.png` - source `pi_0` (Gaussian) and target `pi_1` (checkerboard) overlaid, mirroring the brief's Figure 1.
- `pi1_hist2d.png` - 2D histogram of `pi_1` showing the 8 modes clearly.
- `metrics.json` - sanity metrics (see below).

### Data design choices

- `pi_0`: isotropic 2D Gaussian, mean 0 (std set in `configs/data.yaml`).
- `pi_1`: classic checkerboard built on a 4x4 grid, keeping the 8 cells where
  `(col + row)` is even (the standard alternating checkerboard). The shipped
  config uses `side=1.0`, `gap=0.0` (touching squares, matching the brief's
  Figure 1); nearest-center assignment to the 8 known cells still makes the
  mode-coverage metrics unambiguous. Each mode is equally weighted and samples are
  uniform within each square. (`src/distributions.py` defaults to `gap=0.5`, but
  `configs/data.yaml` and `configs/teacher.yaml` pass `gap=0.0`.)

## What M1 produces (DDPM teacher)

A standard DDPM (Ho et al., 2020), epsilon-prediction, trained on `pi_1`:

- Network: small MLP with sinusoidal time embedding (`src/models.py`).
- Process: linear `beta` schedule (`T` set in `configs/teacher.yaml`); forward `x_t = sqrt(abar_t) x_0 + sqrt(1-abar_t) eps`; loss `||eps - eps_theta||^2`; ancestral sampler (Algorithm 2) with `sigma_t^2 = beta_t` (`src/diffusion.py`).
- The terminal noise is `N(0, I)`, which coincides with `pi_0` (std 1.0), so the teacher transports `pi_0 -> pi_1`. Trained on raw coordinates (no normalization; the toy is already near unit scale).
- **EMA:** an exponential moving average of the denoiser weights is tracked during training (`src/ema.py`, `decay=0.999`). The **EMA model is the reported, frozen teacher** used for all sampling, metrics, figures, the GIF, and the checkpoint loaded by M2-M4. The raw weights are kept in the checkpoint too (`model_state_dict` vs `ema_state_dict`) for debugging/reproducibility.

In `outputs/m1/`:

- `teacher.pt` - frozen teacher checkpoint (raw + EMA weights, config, metrics; EMA used for eval). Reused in M2-M4.
- `loss_curve.png` - training loss.
- `teacher_samples.png` - multi-step teacher samples (EMA).
- `compare_true_vs_teacher.png` - true `pi_1` vs teacher (EMA), side by side.
- `reverse_snapshots.png` - same particles at a few reverse timesteps.
- `reverse_process.gif` - the full reverse trajectory (noise -> checkerboard).
- `metrics.json` - the comparisons below plus pass/fail acceptance.

### M1 metric table (vs true `pi_1`, n=2000, `T=200`)

| Comparison | modes | norm. entropy | off-support | energy dist | MMD |
|------------|:-----:|:-------------:|:-----------:|:-----------:|:---:|
| `pi_1` vs `pi_1` (sanity floor) | 8 | 0.999 | 0.00 | 0.0011 | 0.0005 |
| `pi_0` vs `pi_1` (baseline) | 8 | 0.948 | 0.54 | 0.0446 | 0.0323 |
| teacher raw vs `pi_1` | 8 | 0.9987 | 0.144 | 0.00207 | 0.00105 |
| **teacher EMA vs `pi_1` (reported)** | **8** | **0.9988** | **0.142** | **0.00148** | **0.00083** |

(`pi_0` covers 8 "modes" only by nearest-center assignment of a broad Gaussian;
its high off-support and large distances show it is not `pi_1`.)

### Does EMA help?

The reported teacher uses EMA. Compared to the raw weights, EMA improves the
distributional distances clearly (energy distance -28%, MMD -21%) and is
marginally better on mode entropy and off-support. So EMA mainly sharpens the
distribution match; the off-support residual barely moves (see below).

### Is the teacher good enough for distillation?

Yes. It recovers all 8 modes with balanced mass, and its energy distance / MMD to
`pi_1` are within ~1.3-1.6x of the same-distribution sanity floor and ~30-40x
better than the `pi_0` baseline. All acceptance criteria pass.

### Limitations

The residual off-support fraction (~14%, vs 54% for `pi_0`) is edge-smoothing:
a continuous diffusion model rounds the hard square boundaries, so modes look
slightly blobby rather than crisp squares. EMA does not remove this (it is a
property of the continuous density, not training noise). This is sharpness, not
mode dropping, and is exactly what the naive one-step baseline (M2) should worsen
and what DMD/DMD2 (M3/M4) should sharpen.

## What M2 produces (naive one-step baseline)

M2 runs the **frozen EMA teacher** as a one-step generator and shows it is much
worse than multi-step sampling. From a fixed Gaussian noise `x_T`, it takes a
single teacher noise prediction at the largest timestep and converts it directly
to a clean estimate (DDPM Eq. 15):

```
x0_hat = (x_T - sqrt(1 - abar_T) * eps_theta(x_T, T)) / sqrt(abar_T)
```

This is **not distillation**: nothing is trained, and there is no student. It is
a diagnostic that motivates DMD. The one-step and multi-step samplers start from
the *same* fixed `x_T` (saved at `outputs/eval_noise.pt` and reused by M3/M4) for
a fair comparison; the stochastic reverse sampler is also seeded.

In `outputs/m2/`:

- `one_step_teacher_samples.png` - one-step samples (collapsed/blurry).
- `compare_true_multistep_onestep.png` - true `pi_1` | multi-step EMA | one-step EMA.
- `mode_counts_comparison.png` - per-mode mass for true vs multi-step vs one-step.
- `metrics.json` - the four comparisons below plus the degradation summary.

### M2 metric table (vs true `pi_1`, n=2000, `T=200`)

| Comparison | modes | norm. entropy | off-support | energy dist | MMD |
|------------|:-----:|:-------------:|:-----------:|:-----------:|:---:|
| `pi_1` vs `pi_1` (sanity floor) | 8 | 0.999 | 0.00 | 0.0011 | 0.0005 |
| `pi_0` vs `pi_1` (baseline) | 8 | 0.948 | 0.54 | 0.0446 | 0.0323 |
| teacher multi-step EMA | 8 | 0.999 | 0.124 | 0.0053 | 0.0014 |
| **teacher one-step EMA** | **7** | **0.558** | **0.443** | **0.311** | **0.244** |

### Does the one-step baseline fail?

Yes, clearly. Going from multi-step to one-step the teacher drops a mode (8 -> 7),
mode entropy collapses (0.999 -> 0.558), and off-support, energy distance, and MMD
all jump by large margins (off-support +0.32, ED +0.31, MMD +0.24). Visually the
one-step samples are a blurry blob biased toward the noise mean rather than the 8
crisp squares. A single jump from pure noise to data cannot capture the
multimodal target; only the iterative reverse process can. This is exactly the
gap distillation must close, motivating **M3 (DMD)**: train a *student* that
matches the teacher's distribution in one/few steps without dropping modes.

## What M3 produces (DMD Stage A)

M3 distills the frozen multi-step teacher into a **one-step student** with
**Distribution Matching Distillation** (Yin et al., 2024), then runs a small
ablation over the regression loss and the two-time-scale update rule (TTUR).
No GAN here (that is M4/DMD2).

### The DMD idea (adapted to 2D)

The generator `G_theta(z)` maps noise `z ~ N(0, I)` to a clean sample in one step:

```
eps = student(z, t_gen);   x = predict_x0(z, t_gen, eps)     # t_gen = T-1
```

DMD trains `G_theta` to minimize `D_KL(p_fake || p_real)`. The gradient of that
KL wrt a generated sample `x` is the difference of two scores at a noised
`x_t = sqrt(abar_t) x + sqrt(1-abar_t) eps`:

```
grad_x = a_t * (score_fake(x_t, t) - score_real(x_t, t)),   a_t = sqrt(abar_t)
```

- **Real score** comes from the frozen EMA teacher (it knows the data/teacher
  distribution).
- **Fake score** comes from a second network `mu_fake` trained *online* on the
  student's own current samples with the standard DDPM eps-loss. It is needed
  because we cannot evaluate the score of the moving student distribution in
  closed form, so we learn it on the fly. As the student improves, `mu_fake`
  chases it, and the gradient pushes the student toward the teacher.
- **eps -> score:** our teacher predicts eps, so we convert
  `score(x_t, t) = -eps_theta(x_t, t) / sqrt(1-abar_t)` for both scores
  (`diffusion.eps_to_score`). The generator update uses a detached dot-product
  surrogate `(x * grad_x.detach()).sum()` whose autograd gradient equals the KL
  gradient above (`src/dmd.py`).

### Design choices (and a deviation from the paper)

- **Student keeps the teacher architecture and time embedding.** The original
  DMD removes time conditioning for a pure one-step generator; we keep it so the
  student is byte-compatible with the teacher and can be **initialized from the
  EMA teacher weights**, and so a few-step variant stays reachable for M3/M4.
  We always call the student at the fixed `t_gen = T-1`, so it behaves as a
  one-step generator. This deviation is intentional and documented here.
- **`mu_fake`** uses the same architecture, also initialized from the EMA
  teacher, and is trained only on detached student samples (never to match the
  teacher directly).
- **Regression loss replaces LPIPS.** DMD adds a paired regression loss for
  stability/mode preservation. LPIPS is meaningless in 2D, so we use a simple
  pointwise loss (default **smooth_l1** at lambda=0.05 after the M3.1 sweep; MSE/L1
  are config options) between `G(z_ref)` and a *deterministic*
  teacher sample `y_ref`. Determinism comes from a **DDIM eta=0** sampler
  (`diffusion.ddim_sample_loop`); the `(z_ref, y_ref)` pairs are saved once at
  `outputs/m3/reg_pairs.pt` and reused across variants.
- The teacher stays frozen; the score difference is detached so the generator
  step never updates the teacher or `mu_fake`.

### Ablation (the hypothesis being tested)

DMD2 argues that when the regression loss is removed, instability comes from
`mu_fake` not tracking the generator distribution well enough, and that updating
the fake score more often (TTUR) can recover it. We test this on the toy:

| Variant | regression | fake updates / gen step |
|---------|:----------:|:-----------------------:|
| `dmd_no_reg` | none | 1 |
| `dmd_reg` | smooth_l1 (lambda=0.05) | 1 |
| `dmd_no_reg_ttur` | none | 3 |

(These settings are the ones selected by the M3.1 sweep below; the `dmd_reg`
loss/strength and the TTUR ratio were originally `mse`/0.25 and 5.)

In `outputs/m3/<variant>/`: `student.pt`, `mu_fake.pt`, `training_curves.png`,
`student_samples.png`, `compare_true_onestep_student.png`,
`compare_true_multistep_student.png`, `mode_counts.png`, `metrics.json`. Plus
`outputs/m3/summary_metrics.{json,md}` with the full comparison table.

### M3 metric table (vs true `pi_1`, tuned config, one-step student)

Numbers below are from the M3.1 sweep eval (n=5000), which is the cleanest
apples-to-apples comparison of the tuned variants. `./run.sh m3` regenerates
`outputs/m3/` (n=2000) with the same `configs/dmd.yaml`; expect the same ordering.

| Comparison | modes | norm. entropy | off-support | energy dist | MMD |
|------------|:-----:|:-------------:|:-----------:|:-----------:|:---:|
| M1 multi-step teacher | 8 | 1.000 | 0.131 | 0.0007 | 0.0003 |
| M2 one-step teacher | 6 | 0.561 | 0.447 | 0.2959 | 0.2331 |
| M3 `dmd_no_reg` (fu=1) | 8 | 0.998 | 0.238 | 0.0074 | 0.0027 |
| M3 `dmd_reg` (smooth_l1, fu=1) | 8 | 0.999 | 0.119 | 0.0031 | 0.0013 |
| **M3 `dmd_no_reg_ttur` (fu=3, best)** | **8** | **0.999** | **0.141** | **0.0016** | **0.0005** |

### Does the DMD student beat the naive one-step teacher?

Yes, decisively. Every DMD variant recovers all **8/8 modes** with balanced mass
(entropy ~0.999 vs 0.561 for M2) and cuts the distributional distances by one to
two orders of magnitude (energy distance 0.30 -> 0.0016-0.0074, MMD 0.23 ->
~0.0005-0.003). The best one-step student (`dmd_no_reg_ttur`) essentially matches
the **multi-step** teacher (ED 0.0016 vs 0.0007) while sampling in a single step.
This is the core M3 result: distribution matching closes the gap that the naive
one-step baseline could not.

### What worked / what failed (honest, after the M3.1 sweep)

- The naive **`dmd_no_reg` (fu=1)** config is actually the **weakest** valid
  variant (highest ED/MMD and off-support). Our first, tiny 3-variant ablation
  made it look best, which was misleading.
- **TTUR helps the most.** Raising the fake-score updates per generator step
  (fu=2/3/5, no regression) clearly beats fu=1; fu=3 is the best overall. This is
  exactly DMD2's argument: when there is no regression stabilizer, the fake score
  model must track the moving generator distribution, and more fake updates make
  it do so.
- **Regression also helps.** Every regression setting beats the naive baseline,
  and `smooth_l1` at lambda=0.05 is the best regression-only variant (and the
  smallest off-support). Regression + TTUR is essentially tied with the best.
- Residual off-support (~0.12-0.24) is the same edge-smoothing seen in M1: the
  continuous student rounds the square boundaries. This is sharpness, not mode
  dropping.

### M3.1 hyperparameter tuning (why, and what it corrected)

**Why we did this.** Our first M3 ablation had only three variants, and on it the
pure `dmd_no_reg` (fu=1) config came out slightly *best*. That was a bit
frustrating: it contradicts the papers. DMD uses a regression loss for stability,
and DMD2 argues that if you drop regression you need a two-time-scale update
(more fake-score steps) to stay stable. A toy where "just remove the stabilizer
and it's fine" would undercut that whole story. So before M4 we ran a small but
broader sweep to check whether that finding was real or an artifact of too small
a search.

**It was an artifact, and the sweep recovers the papers' message.** With a proper
search, `dmd_no_reg` (fu=1) is in fact the **worst** valid configuration, not the
best. Both fixes the papers advocate help: TTUR (more fake-score updates) helps
the most, and regression helps too. So the earlier "no-reg is best" was simply
under-searched.

The sweep was a temporary, isolated study (its config/script/outputs have since
been removed); only the results and the selected hyperparameters are kept. It
tested ~15 curated trials (one factor at a time): `lambda_reg` in
{0, 0.025, 0.05, 0.10, 0.25}, `reg_loss_type` in {mse, smooth_l1, l1},
`fake_updates_per_gen` (TTUR) in {1, 2, 3, 5}, `gen_lr` in {5e-5, 1e-4, 2e-4},
`fake_lr` in {5e-4, 1e-3, 2e-3}, plus one reg+TTUR combo. `t_gen=T-1` and the
noise-time range were fixed. Fixed seed and fixed eval noise; ranking on n=5000
samples.

**Ranking metric.** Mode coverage is primary, distributional quality secondary,
off-support a tie-breaker: (1) reject `recovered_mode_count < 8` or
`normalized_mode_entropy < 0.98`; (2) among valid trials,
`score = energy_distance + mmd_rbf + 0.05 * off_support_fraction`. The brief's
`0.25` off-support weight would dominate ED+MMD at these magnitudes
(`0.25*0.18=0.045` vs `ED+MMD~0.008`), inverting the priority, so we use `0.05`
(a genuine tie-breaker) and reported both for transparency.

**Ranked results (vs true `pi_1`, n=5000), top trials:**

| rank | trial | lambda | reg | fu | off-sup | ED | MMD | score |
|---:|---|---:|---|---:|:---:|:---:|:---:|:---:|
| 1 | **`ttur3`** (selected) | 0 | - | 3 | 0.141 | 0.0016 | 0.0005 | 0.0092 |
| 2 | `reg_mse_l0p05_ttur3` | 0.05 | mse | 3 | 0.159 | 0.0014 | 0.0005 | 0.0099 |
| 3 | `reg_sl1_l0p05` | 0.05 | smooth_l1 | 1 | 0.119 | 0.0031 | 0.0013 | 0.0103 |
| 4 | `ttur2` | 0 | - | 2 | 0.134 | 0.0027 | 0.0008 | 0.0103 |
| 5 | `baseline_ttur5` | 0 | - | 5 | 0.151 | 0.0024 | 0.0008 | 0.0108 |
| ... | ... | | | | | | | |
| 13 | `baseline_reg_mse` | 0.25 | mse | 1 | 0.195 | 0.0058 | 0.0024 | 0.0180 |
| 15 | `baseline_no_reg` (old "best") | 0 | - | 1 | 0.238 | 0.0074 | 0.0027 | 0.0220 |

**Conclusions.**

- **Best hyperparameters:** `fake_updates_per_gen=3`, no regression, `gen_lr=1e-4`,
  `fake_lr=1e-3` (the learning rates were already optimal). Selected as the M3
  default.
- **Improves over the original M3 best?** Yes: score 0.0092 vs 0.0220 (~2.4x lower
  composite; ED 0.0016 vs 0.0074).
- **Did regression help?** Yes, every regression variant beats the naive baseline;
  `smooth_l1`/0.05 is the best regression-only setting, so we adopt it for the
  `dmd_reg` ablation arm (was `mse`/0.25).
- **Did TTUR help?** Yes, clearly and the most: fu=2/3/5 all beat fu=1, with fu=3
  best.
- **Did the conclusion change?** Yes. The earlier "regression/TTUR not needed on
  the toy" was wrong; the papers' message holds once the search is broad enough.
- **`configs/dmd.yaml` was updated** to these settings; `./run.sh m3` regenerates
  `outputs/m3/` with them.

## What M4 produces (DMD2 Stage B)

M4 implements **DMD2** (Yin et al., 2024, NeurIPS) on top of the best M3 student.
It continues training that one-step student with two changes from M3:

- **Regression dropped entirely.** No paired `(z_ref, y_ref)` data, no MSE/L1/
  smooth_l1 term. The student is trained only by distribution matching + GAN.
- **GAN loss added**, keeping the TTUR schedule (`fake_updates_per_gen=3`) that
  the M3.1 sweep selected. This is exactly DMD2's recipe: remove the regression
  stabilizer, lean on TTUR, and add an adversarial term so the student trains
  against *real data*, not only the teacher's (imperfect) score.

### What DMD2 adds over M3

The DMD distribution-matching surrogate is reused unchanged from M3
(`grad_x = a_t * (score_fake - score_real)`, detached). On top of it we add the
GAN term on **noised** samples:

```
F(x, t) = q_sample(x, t, eps)              # diffuse a clean sample to timestep t
L_D     = softplus(-D(F(x_real, t))) + softplus(D(F(x_fake, t)))   # discriminator
L_GAN_G = softplus(-D(F(x_fake, t)))                               # generator
L_G     = L_DMD + lambda_gan * L_GAN_G                             # student step
```

- **`F(x, t)` in 2D** simply means we run a real point `x_real ~ pi_1` and a
  generated point `x_fake = G(z)` through the *forward* diffusion `q_sample` to a
  random timestep before showing them to the discriminator. The GAN timestep is
  drawn from a restricted low/mid band (`gan_t_min_frac`/`gan_t_max_frac`,
  separate from the DMD score range), because at high noise both real and fake
  collapse to `N(0, I)` and the critic cannot separate them (see M4.1).
- **Why noised, not clean, samples.** A GAN on raw 2D points collapses easily;
  noising smooths and overlaps the two supports (the DiffusionGAN trick DMD2
  builds on), which stabilizes the adversarial game.
- **Discriminator: a linear head on `mu_fake`'s shared backbone** (DMD2's actual
  design, `src/dmd2.py`). `D(x_t) = head(mu_fake.forward_features(x_t, t))`, where
  `forward_features` returns `mu_fake`'s last hidden activations and `head` is a
  single `Linear(hidden_dim, 1)` logit. There is **no separate discriminator
  network and no separate time embedding for D** - time enters only through the
  shared `mu_fake` encoder (the head sees `x_t`'s noise level implicitly via those
  features). The `mu_fake` backbone is **updated jointly by the GAN loss**
  (`opt_disc` steps `head + mu_fake`) while the TTUR DDPM eps-loss keeps updating
  `mu_fake` too, exactly as DMD2 shares one backbone for the fake-score model and
  the critic. Raw logits + softplus/BCE-with-logits (no sigmoid). This replaced an
  earlier separate-MLP critic during M4.1.

### Training loop and guards

Per generator step: (1) update `mu_fake` 3x with the standard DDPM eps-loss on
detached student samples (TTUR); (2) update the discriminator (`head + mu_fake`)
once with `L_D` on noised real/detached-fake samples; (3) update the student once
with `L_DMD + lambda_gan * L_GAN_G`. The teacher stays frozen; during the
generator step only the student optimizer is stepped (`head`/`mu_fake` grads from
the GAN term are cleared by the next `zero_grad`); the score difference is
detached; fake samples are detached for the `mu_fake`/`D` updates but **not** for
the generator update. Hyperparameters are the M4.1-tuned values
(`lambda_gan=0.05`, `disc_lr=3e-4`, `gan_t` band `0.0-0.7`; see M4.1). M4 starts
from `outputs/m3/dmd_no_reg_ttur/{student.pt,mu_fake.pt}`.

In `outputs/m4/`: `dmd2_student.pt`, `dmd2_mu_fake.pt`, `discriminator.pt` (head),
`config.json`, `training_curves.png`, `gan_diagnostics.png`, `dmd2_samples.png`,
`compare_m3_vs_m4.png`, `compare_true_m2_m3_m4.png`, `mode_counts.png`,
`metrics.json`, `diagnostics.json`, `summary_metrics.{json,md}`.

### M4 metric table (vs true `pi_1`, n=2000, one-step student, tuned config)

| Comparison | modes | norm. entropy | off-support | energy dist | MMD |
|------------|:-----:|:-------------:|:-----------:|:-----------:|:---:|
| `pi_1` vs `pi_1` (sanity floor) | 8 | 1.000 | 0.000 | 0.0014 | 0.0003 |
| `pi_0` vs `pi_1` (baseline) | 8 | 0.951 | 0.535 | 0.0419 | 0.0298 |
| M1 multi-step teacher | 8 | 0.999 | 0.124 | 0.0053 | 0.0014 |
| M2 one-step teacher | 7 | 0.558 | 0.443 | 0.3106 | 0.2439 |
| M3 best student (no-reg TTUR) | 8 | 0.998 | 0.158 | 0.0059 | 0.0022 |
| **M4 DMD2 student** | **8** | **0.998** | **0.137** | **0.0088** | **0.0029** |

### Does the GAN improve over M3? (honest)

**Headline: M4 is approximately equal to M3, even with the M4.1-tuned
hyperparameters.** Mode coverage is fully preserved (8/8 modes, entropy 0.998 =
M3), training is stable, and the distributional metrics are essentially a tie:

- At the **clean eval (n=2000)** above, M4 improves off-support (0.158 ->
  **0.137**) but is **slightly worse** on ED/MMD (0.0059 -> 0.0088, 0.0022 ->
  0.0029).
- At the **M4.1 ranking eval (n=5000)**, the same tuned setting edges M3 on all
  three (off 0.160 -> 0.145, ED 0.0021 -> 0.0011, MMD 0.0008 -> 0.0004).

The two evals disagree on the sign of the ED/MMD change because the distances are
tiny (~1e-3) and their estimates are noisy at these magnitudes, so the M4-vs-M3
gap is **within evaluation noise**. The honest conclusion: on this already-solved
toy DMD2 lands **on par with M3** (a small, consistent off-support/sharpness gain
at best, no robust ED/MMD win). The root cause is that the GAN never really
contributes (the discriminator does not learn); M4.1 below tried to fix that with
the shared-backbone redesign and tuning, and it did not help much.

### M4.1 GAN diagnostics and tuning

**Why we did this.** The first M4 preserved all 8 modes but did not beat M3 on
ED/MMD, and its GAN losses were suspiciously flat (`L_D ~ 2*ln2 ~ 1.386`,
`L_GAN_G ~ ln2 ~ 0.693`), i.e. the discriminator logits sat at ~0. So before
declaring M4 done we (a) made the discriminator faithful to DMD2, (b) added
diagnostics to check whether the critic actually learns, and (c) ran a small
isolated tuning sweep over the GAN knobs. The tuning was temporary and isolated
(`configs/dmd2_tuning.yaml`, `scripts/tune_m4_dmd2.py`, `outputs/m4_tuning/`); that
code has since been **removed** and only the selected hyperparameters and the
findings below are kept.

**Discriminator redesign (the main fix).** The original critic was a *separate*
MLP. DMD2 instead shares the fake model's backbone, so we replaced it with a
linear head on `mu_fake`'s last hidden layer (see the discriminator bullet above):
no separate network, no separate time embedding, and `mu_fake` is updated jointly
by the GAN loss. This is the faithful 2D analogue of the paper's classifier head
on the fake-UNet bottleneck.

**Diagnostics added** (`outputs/m4/gan_diagnostics.json` + `gan_diagnostics.png`,
printed each run): mean real/fake logit, discriminator real/fake/total accuracy,
and discriminator / generator-GAN / DMD gradient norms. A learning critic should
push real logits positive, fake logits negative, and total accuracy above 0.5.

**GAN timestep band.** The GAN noising range is now configurable separately from
the DMD score range (`gan_t_min_frac`/`gan_t_max_frac`). Diffusing real and fake
over the *full* schedule pushes both toward `N(0, I)` at high `t`, where they are
indistinguishable, so we restrict the GAN to a low/mid band.

**The sweep.** ~11 curated trials (one or two factors at a time) over
`lambda_gan` in {0.02, 0.05, 0.10, 0.20}, `disc_lr` in {1e-4, 3e-4, 1e-3},
`disc_updates_per_gen` in {1, 2, 3}, and `gan_t` band in {0.0-0.5, 0.0-0.7,
0.1-0.6}, plus the full-range baseline. Init fixed to the best M3
student/`mu_fake`, `fake_updates_per_gen=3`, no regression. Ranking (n=5000):
reject `modes<8` or `entropy<0.98`, then sort by `ED+MMD` with off-support as the
tie-breaker, and refuse a trial that improves off-support but clearly worsens
ED/MMD.

**Selected:** `lambda_gan=0.05`, `disc_lr=3e-4`, `disc_updates_per_gen=1`, GAN `t`
band `0.0-0.7` (trial `trange_0p7`). This is now the default in `configs/dmd2.yaml`
and `./run.sh m4` regenerates `outputs/m4/` with it. Top sweep rows (n=5000):

| trial | lambda | disc_lr | du | gan_t | off | ED | MMD | disc acc |
|---|---:|---:|---:|:---:|:---:|:---:|:---:|:---:|
| M3 best (ref) | - | - | - | - | 0.160 | 0.0021 | 0.0008 | - |
| **trange_0p7 (selected)** | 0.05 | 3e-4 | 1 | 0.0-0.7 | 0.145 | 0.0011 | 0.0004 | 0.49 |
| disc_lr1e3 | 0.05 | 1e-3 | 1 | 0.0-0.5 | 0.132 | 0.0011 | 0.0004 | 0.50 |
| disc_du3 | 0.05 | 3e-4 | 3 | 0.0-0.5 | 0.146 | 0.0016 | 0.0006 | 0.47 |
| baseline (full t) | 0.10 | 1e-4 | 1 | 0.0-1.0 | 0.142 | 0.0023 | 0.0007 | 0.49 |

**Did the discriminator learn? Honestly, barely.** Even after the redesign, the
restricted `t` band, and stronger disc settings, total accuracy stayed near chance
(~0.47-0.53) and the logits near 0 across *every* trial; the DMD gradient norm
(~1-2) dwarfs the GAN gradient norm (~0.1-0.2). So the GAN remains weakly active:
the distribution-matching loss has already solved the toy, leaving little for the
critic to exploit once samples are noised. The restricted band and the
shared-backbone redesign help dynamics, but they do not turn the critic into a
strong, clearly-learning discriminator.

**Did M4 improve over M3? Approximately equal.** The whole point of M4.1 was to
fix the discriminator-not-learning problem, and it did **not** help much: after
the redesign + tuning the critic still sits at chance and M4 lands on par with M3
(off-support a touch better, ED/MMD a tie within noise). At the n=5000 ranking eval
the tuned setting edges M3 on all three; at the clean n=2000 eval only off-support
improves while ED/MMD tick up. We kept the tuned config because it is the sweep
winner and the change is small, but we do **not** claim a GAN-driven improvement
the diagnostics do not support. The honest result: **M4 approximately equals M3**,
and adding a GAN buys little once distribution matching has already solved this 2D
toy.

**Temporary files (removed):** the tuning study (`configs/dmd2_tuning.yaml`,
`scripts/tune_m4_dmd2.py`, `scripts/`, `outputs/m4_tuning/`) has been deleted; only
the results and the selected hyperparameters are kept. The permanent pieces stay:
the shared-head discriminator, the GAN-`t` config keys, and the diagnostics.

## What M5 produces (few-step DMD2: the clear win over M3)

M4 showed that on this toy a one-step DMD2 student lands **approximately equal to
M3**: distribution matching already pins ED/MMD to their noise floor (~1e-3), so
the only axis with real, explainable headroom is the **off-support fraction** -
the edge-smoothing of the hard checkerboard boundaries (M3 one-step ~0.158, M4
one-step ~0.137, multi-step teacher ~0.124). A single jump from a Gaussian cannot
carve sharp square edges; the iterative reverse process can.

M5 implements the two DMD2 contributions that M4 skipped - the **multi-step
generator (sec 4.4)** and **backward simulation (sec 4.5)** - and trains a
**4-step** student. With strictly more capacity than a one-step map, it sharpens
the boundaries and clearly beats every one-step student on off-support.

### Method (adapted to 2D)

- **Schedule:** 4 steps at timesteps `[199, 149, 99, 49]` for `T=200` (mirrors the
  paper's `999, 749, 499, 249` for `T=1000`).
- **Inference (sec 4.4):** from `z ~ N(0, I)`, alternate one denoise
  (`x0 = predict_x0(x_t, t, student(x_t, t))`) and one renoise
  (`q_sample(x0, t_next)`); return the final `x0`. A single-entry schedule `[T-1]`
  reduces exactly to the one-step generator (so the same code drives the 1-step
  ablation arm).
- **Backward simulation (sec 4.5):** during training, pick a scheduled step
  uniformly, run the inference loop up to that step under `no_grad` to build the
  *student-generated* input `x_{t_i}`, then take one **differentiable** denoise
  `G(x_{t_i}, t_i)` supervised by the existing DMD surrogate + GAN loss. The
  generator is trained on its own intermediate samples, not on noised real data,
  removing the train/inference input mismatch.
- **Reuse:** the DMD distribution-matching surrogate (`src/dmd.py`) and the GAN
  pieces + diagnostics (`src/dmd2.py`) are reused verbatim; no architecture change
  (the student already conditions on `t`). New primitives live in
  `src/dmd2_fewstep.py`.
- **Init (documented choice):** student and `mu_fake` init from the **EMA teacher**
  (DMD2 initializes the generator from the pretrained diffusion model), not from
  the one-step M3/M4 student. A multi-step student must denoise at every scheduled
  timestep, not just `t=T-1`; the one-step students were only ever trained at
  `t=199`. Configurable via `init_student_checkpoint` in `configs/dmd2_fewstep.yaml`.

In `outputs/m5/`: `fewstep_student.pt`, `fewstep_mu_fake.pt`, `discriminator.pt`,
`config.json`, `training_curves.png`, `gan_diagnostics.png`, `fewstep_samples.png`,
`compare_m3_m4_m5.png`, `compare_true_multistep_fewstep.png`, `mode_counts.png`,
`offsupport_bars.png`, `metrics.json`, `diagnostics.json`,
`summary_metrics.{json,md}`, `ablation_summary.{json,md}`.

### M5 metric table (vs true `pi_1`, n=2000, fixed eval noise)

| Comparison | modes | norm. entropy | off-support | energy dist | MMD |
|------------|:-----:|:-------------:|:-----------:|:-----------:|:---:|
| `pi_1` vs `pi_1` (sanity floor) | 8 | 1.000 | 0.000 | 0.0014 | 0.0003 |
| M1 multi-step teacher | 8 | 0.999 | 0.124 | 0.0053 | 0.0014 |
| M2 one-step teacher | 7 | 0.558 | 0.443 | 0.3106 | 0.2439 |
| M3 best student (1-step) | 8 | 0.998 | 0.158 | 0.0059 | 0.0022 |
| M4 DMD2 student (1-step) | 8 | 0.998 | 0.137 | 0.0088 | 0.0029 |
| **M5 DMD2 student (4-step)** | **8** | **0.996** | **0.064** | **0.0060** | **0.0027** |

### Does M5 beat M3 clearly? (yes)

Yes. M5 cuts off-support **~2.5x** vs the M3 one-step student (0.158 -> 0.064) and
drops it **below the multi-step teacher** (0.124), while preserving all 8 modes
(entropy 0.996) and keeping ED/MMD a tie within noise. The win is robust, not an
artifact of one eval: off-support is **0.061 / 0.066** at n=2000 (two seeds) and
**0.058** at n=5000. Visually (`compare_m3_m4_m5.png`) the M5 squares are visibly
crisper and better separated than the M3/M4 blobs.

### M5 ablation: +/- GAN, +/- TTUR, 1-step vs 4-step

All arms share the teacher-EMA init, 3000-step budget and hyperparameters, so only
the ablated axis differs. Ranked by off-support (lower = sharper):

| variant | steps | GAN | TTUR | modes | off-support | ED | MMD |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **4-step, +GAN (fu=3) [M5]** | 4 | yes | fu=3 | 8 | **0.064** | 0.0060 | 0.0027 |
| 4-step, no GAN (fu=3) | 4 | no | fu=3 | 8 | 0.071 | 0.0171 | 0.0059 |
| 4-step, no GAN, fu=1 (-TTUR) | 4 | no | fu=1 | 8 | 0.080 | 0.0143 | 0.0084 |
| 1-step, no GAN (fu=3) | 1 | no | fu=3 | 8 | 0.144 | 0.0044 | 0.0015 |
| 1-step, +GAN (fu=3) | 1 | yes | fu=3 | 8 | 0.148 | 0.0058 | 0.0020 |

- **Step count dominates.** Every 4-step arm (~0.064-0.080) beats every 1-step arm
  (~0.145) on off-support by ~2x. This is the headline effect.
- **TTUR helps the 4-step student** (fu=3 0.071 < fu=1 0.080), consistent with DMD2.
- **The GAN finally contributes.** At 4 steps it both lowers off-support
  (0.071 -> 0.064) and clearly cuts ED (0.0171 -> 0.0060) - the opposite of the
  inert GAN in M4, because the few-step student leaves real headroom the critic on
  *real* data can exploit.
- **Honest nuance:** the 1-step arms keep the lowest ED (0.0044) but cannot sharpen
  boundaries; few-step trades a tiny ED cost for a large off-support gain. Off-support
  is the metric that exposes the difference, which is exactly why we report it.

## Metrics and why they fit a 2D multimodal target

Evaluation is **sample-based**, not loss-based. M0 computes the mode-coverage
metrics now (the ones central to the multimodal failure mode) and verifies the
two-sample distances on same-distribution data, ready for the M1+ comparisons.

Computed at M0:

- **Recovered mode count** - nearest-center assignment to the 8 known modes; counts modes holding meaningful mass. Directly measures mode dropping. Expect 8/8 for the true target.
- **Normalized mode entropy** - entropy of the per-mode mass, normalized to [0, 1]; 1.0 means balanced coverage. Catches partial collapse even when all modes are nominally present.
- **Off-support fraction** - fraction of samples outside the checkerboard squares; a sanity check on the sampler (expect ~0) and later a fidelity check on generators.
- **pi_0 moments** - empirical mean (~0) and covariance (~I) sanity check.

Implemented and verified now, used fully in M1+:

- **Energy distance** and **MMD (RBF)** - two-sample distances between sets. At M0 they are run on two independent `pi_1` batches (should be ~0) to validate the code; the real use is comparing teacher vs students.

Planned for the final report (teacher vs naive one-step vs DMD vs DMD2):
recovered mode count, normalized mode entropy, off-support fraction, energy
distance, MMD, and optionally a simple geometry-based sharpness/fidelity metric.

## How AI usage is logged

See `ai_logs/`. I use **ChatGPT** as a planning and prompt-refinement assistant
before sending instructions to **Codex** (the coding agent):

- `ai_logs/index.md` - index of sessions.
- `ai_logs/chatgpt_planning.md` - documents the ChatGPT planning workflow.
- `ai_logs/2026-06-12_m0_setup.md` - this M0 setup session.
- `ai_logs/transcripts/` - location for raw Codex/ChatGPT transcripts.

## Roadmap (M1-M4)

- **M0 - Data (done):** `pi_0`, `pi_1` generated and plotted, with sanity metrics.
- **M1 - Teacher (done):** DDPM teacher trained; multi-step sampling recovers all 8 modes of `pi_1` (checkpoint + figures + GIF + metrics).
- **M2 - Naive one-step baseline (done):** running the frozen EMA teacher in a single step drops a mode and collapses entropy/off-support/ED/MMD vs multi-step, motivating distillation. Deliverable: figures + metrics.
- **M3 - DMD (Stage A) (done):** distribution matching distillation into a one-step student that recovers all 8 modes and matches the multi-step teacher's distances, far beating the M2 one-step baseline; with a reg/TTUR ablation. Deliverable: per-variant checkpoints + figures + metrics + summary table.
- **M4 - DMD2 (Stage B) (done):** drop regression, keep TTUR, add a GAN loss on noised samples with a discriminator head sharing `mu_fake`'s backbone (M4.1 redesign); preserves all 8 modes but lands **approximately equal to M3** in one step (small off-support gain, ED/MMD a tie within noise). The discriminator barely learns (accuracy ~0.5) and the M4.1 redesign + tuning did not change that on this already-solved toy. Deliverable: checkpoint + figures + metrics + GAN diagnostics (M4.1 tuning code was temporary and has been removed).
- **M5 - Few-step DMD2 (the clear win) (done):** a 4-step student with the DMD2 multi-step generator (sec 4.4) and backward simulation (sec 4.5). Cuts off-support **~2.5x** vs the M3 one-step student (0.158 -> 0.064), below the multi-step teacher, with all 8 modes and ED/MMD a tie within noise; robust across seeds/sample sizes. Deliverable: checkpoint + figures + metrics + a +/-GAN / +/-TTUR / 1-step-vs-4-step ablation table.
- **Stretch (done):** ablations (+/- GAN, +/- two-time-scale, one-step vs 4-step) - see `outputs/m5/ablation_summary.md`.

Each milestone is documented as it is completed (figures, metric numbers, and an
AI-log entry), not only at the end.
