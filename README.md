# Few-Step Generative Distillation on a 2D Toy

A small, correct, explainable distillation pipeline on a 2D toy. It transports an
isotropic Gaussian source `pi_0` to a multimodal target `pi_1` (an 8-mode
checkerboard): train a multi-step DDPM teacher, then distill it into a one/few-step
generator with **DMD** (distribution matching) and **DMD2** (DMD + two-time-scale
update + GAN).

> For full method details, tuning history, ablations, and honest failure analysis,
> see [docs/technical_log.md](docs/technical_log.md).

## How to run

From a clean checkout, a single command runs the whole pipeline in order. On
first run, `run.sh` creates a local `.venv` and installs `requirements.txt`
automatically (`device: auto` uses CUDA if available, else CPU):

```bash
bash run.sh        # runs every stage: m0 -> m1 -> m2 -> m3 -> m4 -> m5
```

Prerequisites: `python3` with the `venv` module (on Debian/Ubuntu/WSL:
`sudo apt install python3-venv`) and internet access for the one-time install.

Or run one stage at a time:

```bash
./run.sh m0   # data: pi_0, pi_1, plots, sanity metrics
./run.sh m1   # teacher: train DDPM, multi-step sampling, figures + GIF, metrics
./run.sh m2   # naive one-step baseline from the frozen EMA teacher (motivates distillation)
./run.sh m3   # DMD Stage A: one-step student (+ reg/TTUR ablation)
./run.sh m4   # DMD2 Stage B: continue best M3 student with TTUR + GAN
./run.sh m5   # stretch: few-step (4-step) DMD2 + 1-vs-4-step ablation
```

`run.sh` with no argument (or `all`) runs every stage sequentially, reinstalling
deps only when `requirements.txt` changes. Each stage writes figures and a
`metrics.json` / `summary_metrics.md` under `outputs/<stage>/`. Stages run in
minutes on CPU.

## Repository structure

```
.
├── run.sh              # single entry point (bootstraps venv, runs a stage)
├── requirements.txt
├── configs/            # one YAML per stage (data, teacher, baseline, dmd, dmd2, dmd2_fewstep)
├── src/                # runtime code
│   ├── distributions.py, metrics.py, plotting.py, utils.py, models.py, ema.py
│   ├── diffusion.py        # DDPM schedule, loss, samplers, eps->score, DDIM
│   ├── dmd.py / dmd2.py    # DMD (M3) and DMD2 GAN (M4) primitives
│   ├── dmd2_fewstep.py     # multi-step generator and backward simulation (M5)
│   └── m0_data.py, train_teacher.py, baseline_onestep.py, train_dmd.py,
│       train_dmd2.py, train_dmd2_fewstep.py   # stage entry points
├── outputs/            # generated figures and metric files (m0/ .. m5/)
├── resources/          # assignment brief, paper notes and source index (agent context)
├── docs/technical_log.md   # detailed engineering and interview log
└── ai_logs/            # honest log of AI-assisted sessions
```

## Milestones

The brief's ladder is M0-M4 (aim for a clean **M3**, with **M4** as the stretch).
M5 is an **additional stretch experiment** I added, not a milestone from the brief.

- **M0 - Data:** `pi_0`, `pi_1` generated and plotted, with sanity metrics.
- **M1 - Teacher:** DDPM (eps-prediction MLP + EMA); multi-step sampling recovers all 8 modes.
- **M2 - Naive one-step baseline:** running the teacher in one step drops a mode and collapses (motivates distillation).
- **M3 - DMD (Stage A):** one-step student recovers all 8 modes and matches the multi-step teacher's distances; with a reg/TTUR ablation. **(brief target)**
- **M4 - DMD2 (Stage B):** drop regression, keep TTUR, add a GAN; preserves all 8 modes but lands **tied with M3** in one step. **(brief stretch)**
- **M5 - Few-step DMD2 (extra stretch):** a 4-step student (DMD2 sections 4.4 and 4.5). **We get a clear improvement**: off-support `0.158 -> 0.064`, below the multi-step teacher, all 8 modes kept.

**Note**: For both stages of M4, I did a small hyperparameters tunning, because the values given in the papers were a bit off for our toy distribution (for example lambda_reg).

## Main result (vs true `pi_1`, n=2000, fixed eval noise)


| Model                           | steps | modes | norm. entropy | off-support | energy dist | MMD        |
| ------------------------------- | ----- | ----- | ------------- | ----------- | ----------- | ---------- |
| `pi_1` vs `pi_1` (sanity check) | -     | 8     | 1.000         | 0.000       | 0.0014      | 0.0003     |
| `pi_0` vs `pi_1` (baseline)     | -     | 8     | 0.951         | 0.535       | 0.0419      | 0.0298     |
| M1 multi-step teacher           | 200   | 8     | **0.999**     | 0.124       | 0.0053      | 0.0014     |
| M2 one-step teacher (naive)     | 1     | 7     | 0.558         | 0.443       | 0.3106      | 0.2439     |
| M3 DMD student                  | 1     | 8     | 0.998         | 0.158       | 0.0059      | 0.0022     |
| M4 DMD2 student                 | 1     | 8     | 0.998         | 0.137       | 0.0088      | 0.0029     |
| **M5 DMD2 student (stretch)**   | **4** | **8** | 0.996         | **0.064**   | **0.0060**  | **0.0027** |


Evaluation is sample-based: recovered mode count and normalized
mode entropy (mode dropping), off-support fraction (boundary sharpness), and two distances (energy distance, MMD).

**Note**: The results presented here may change on another device or different CPU threads. Nevertheless, it should be close enought to have the same conclusions. I made the choice to not add extra hardening on reproducibility for more lisibility.

## Key findings

- **Distillation is necessary.** The naive one-step teacher (M2) drops a mode and collapses and is not able to recover the true distribution. Only the iterative reverse process captures the multimodal target.
- **M3 DMD works in one step.** It recovers all 8 modes and matches the multi-step
teacher on ED/MMD (~1e-3), far beating M2.
- **M4 one-step DMD2 is approximately tied with M3.** On this 2D toy distribution matching already solves the problem, so the GAN is essentially inert (the discriminator sits at chance accuracy ~0.5). It does not clearly upgrade our M3 results.
- **The clear win is the M5 stretch (few-step DMD2).** A 4-step student cuts off-support by 2.5x over the M3 one-step student (`0.158 -> 0.064`), dropping below even the multi-step teacher, while keeping all 8 modes. Step count  
is the dominant factor here. (see [outputs/m5/ablation_summary.md](outputs/m5/ablation_summary.md)).
- **Off-support is the discriminating metric here.** ED/MMD sit at their noise
floor for every trained student, so off-support (boundary sharpness) is what
separates one-step from few-step.

## AI usage and traceability

This project was built with AI assistance, logged honestly:

- **ChatGPT** (planning / prompt refinement), and  **Cursor** (coding).
- Session-by-session log: [ai_logs/index.md](ai_logs/index.md) 
- Workflow notes in [ai_logs/chatgpt_planning.md](ai_logs/chatgpt_planning.md) 
- Raw transcripts in `ai_logs/transcripts/`.

`resources/` records the exact context the agent had, the assignment brief, paper notes (DDPM, DMD, DMD2, Di[M]O), and a source index mapping each source to how it was used. It is **context and traceability only, runtime code in** `src/` **never imports from it.** See [resources/sources_index.md](resources/sources_index.md).

## More

- Detailed write-up: [docs/technical_log.md](docs/technical_log.md)
- Per-stage tables: [outputs/m3/summary_metrics.md](outputs/m3/summary_metrics.md),
[outputs/m4/summary_metrics.md](outputs/m4/summary_metrics.md),
[outputs/m5/summary_metrics.md](outputs/m5/summary_metrics.md),
[outputs/m5/ablation_summary.md](outputs/m5/ablation_summary.md)

