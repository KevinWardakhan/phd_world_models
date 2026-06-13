# Few-Step Generative Distillation on a 2D Toy

A small, correct, explainable pipeline for few-step generative distillation on a
2D toy problem. The end goal: transport an isotropic Gaussian source `pi_0` to a
multimodal target `pi_1` (an 8-mode checkerboard), train a multi-step
diffusion/flow teacher, and distill it into a one/few-step generator with **DMD**
then **DMD2**.

Priorities: correctness and explainability over sophistication; everything stays
small and 2D; mode coverage is central because the target is multimodal and mode
dropping is the main failure mode.

> Status: **M0 + M1 + M2 + M3 complete** (data; DDPM teacher recovers `pi_1`; naive one-step baseline fails; DMD students recover all 8 modes in one step). M4 not started.

## Repository structure

```
.
├── run.sh              # single entry point (bootstraps venv, runs a stage)
├── requirements.txt
├── configs/            # YAML configs (data.yaml M0, teacher.yaml M1, baseline.yaml M2, dmd.yaml M3)
├── src/                # runtime code (does NOT depend on resources/)
│   ├── distributions.py  # pi_0 Gaussian, pi_1 checkerboard, mode_centers
│   ├── metrics.py        # mode coverage + two-sample distances
│   ├── plotting.py       # figures + reverse-process GIF
│   ├── utils.py          # config/seed/IO helpers + fixed eval noise
│   ├── models.py         # MLP noise predictor (M1)
│   ├── ema.py            # EMA of teacher weights (M1)
│   ├── diffusion.py      # DDPM schedule, loss, samplers, one-step, eps->score, DDIM (M1/M2/M3)
│   ├── dmd.py            # DMD primitives: student/fake build, generator surrogate loss, reg pairs (M3)
│   ├── m0_data.py        # M0 entry point
│   ├── train_teacher.py  # M1 entry point
│   ├── baseline_onestep.py  # M2 entry point
│   └── train_dmd.py      # M3 entry point
├── outputs/            # generated figures + metric files (m0/, m1/, m2/, m3/)
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
```

`./run.sh` defaults to `m0`. The script creates a local `.venv` and installs
`requirements.txt`. M1 adds `torch`, `tqdm`, `imageio`. M1 trains on CPU by
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
  `(col + row)` is even (the standard alternating checkerboard). Squares are
  separated by a small gap (`side=1.0`, `gap=0.5`) so the 8 modes are visually
  and metrically separable; this makes mode-coverage metrics unambiguous. Each
  mode is equally weighted and samples are uniform within each square.

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
  stability/mode preservation. LPIPS is meaningless in 2D, so we use **MSE**
  (SmoothL1/L1 are config options) between `G(z_ref)` and a *deterministic*
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
| `dmd_reg` | MSE (lambda=0.25) | 1 |
| `dmd_no_reg` | none | 1 |
| `dmd_no_reg_ttur` | none | 5 |

In `outputs/m3/<variant>/`: `student.pt`, `mu_fake.pt`, `training_curves.png`,
`student_samples.png`, `compare_true_onestep_student.png`,
`compare_true_multistep_student.png`, `mode_counts.png`, `metrics.json`. Plus
`outputs/m3/summary_metrics.{json,md}` with the full comparison table.

### M3 metric table (vs true `pi_1`, n=2000, `T=200`, one-step student)

| Comparison | modes | norm. entropy | off-support | energy dist | MMD |
|------------|:-----:|:-------------:|:-----------:|:-----------:|:---:|
| `pi_1` vs `pi_1` (sanity floor) | 8 | 1.000 | 0.000 | 0.0014 | 0.0003 |
| `pi_0` vs `pi_1` (baseline) | 8 | 0.951 | 0.535 | 0.0419 | 0.0298 |
| M1 multi-step teacher | 8 | 0.999 | 0.124 | 0.0053 | 0.0014 |
| M2 one-step teacher | 7 | 0.558 | 0.443 | 0.3106 | 0.2439 |
| M3 `dmd_reg` | 8 | 0.999 | 0.147 | 0.0070 | 0.0022 |
| **M3 `dmd_no_reg` (best)** | **8** | **0.999** | **0.180** | **0.0055** | **0.0024** |
| M3 `dmd_no_reg_ttur` | 8 | 0.998 | 0.164 | 0.0080 | 0.0022 |

### Does the DMD student beat the naive one-step teacher?

Yes, decisively. Every DMD variant recovers all **8/8 modes** with balanced mass
(entropy ~0.999 vs 0.558 for M2) and cuts the distributional distances by ~40-55x
(energy distance 0.31 -> ~0.006-0.008, MMD 0.24 -> ~0.002). The one-step student
essentially matches the **multi-step** teacher (ED 0.0053) while sampling in a
single step. This is the core M3 result: distribution matching closes the gap
that the naive one-step baseline could not.

### What worked / what failed (honest)

- On this 2D toy, **regression was not required**: pure distribution matching
  (`dmd_no_reg`) was stable and actually the best on energy distance. We did not
  observe the collapse DMD2 warns about, likely because the target is simple and
  low-dimensional and the student starts from a strong teacher init.
- **TTUR** (more fake updates, no regression) was also stable but did not help
  here, coming out marginally worse on ED/MMD than 1:1 updates. Its motivation
  (rescuing no-regression training) does not bite because no-regression was
  already fine in this setting. The TTUR/GAN machinery is expected to matter more
  on harder targets, which is where M4 (DMD2) goes.
- Residual off-support (~0.15-0.18) is the same edge-smoothing seen in M1: the
  continuous student rounds the square boundaries. This is sharpness, not mode
  dropping.

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
- **M4 - DMD2 (Stage B):** add GAN loss, drop regression loss, two-time-scale update; sharper one-step samples. Deliverable: checkpoint + figure + metric.
- **Stretch:** ablations (+/- GAN, +/- two-time-scale, one-step vs few-step).

Each milestone is documented as it is completed (figures, metric numbers, and an
AI-log entry), not only at the end.
