

## Final-Round Take-Home

## Few-Step Generative Distillation on a 2D Toy

VISTA, LIX, École Polytechnique · final interview round

---

## 1. Overview

You will build, end to end, a small but complete distillation pipeline on a 2D toy problem:

1. make a toy dataset,
2. train a base diffusion / flow-matching model on it, and
3. distill that teacher into a **one- or few-step generator** using **DMD2**.

Keep everything 2D and small. What matters here is understanding, correctness, and judgment, not scale or tuning. A small loop that is *right* and that you can *explain* beats an elaborate one that is neither.

This is the **final interview round**: after you submit, we meet one last time to go through the work together.

## 2. Rules

- **Time: 48 hours, rolling wall-clock, single continuous window.** You choose when to start. Reply to schedule it; this brief is sent to you at your chosen start time, and the 48h clock begins the moment you receive it. Submission is due 48h later.
- **Tools: no restrictions.** Use any AI coding assistant (Claude Code, Codex, Copilot, Cursor, ...), any libraries, any papers, any blog posts. We assume you will, and the bar is set accordingly. Don't hide it; using these tools well is a skill we want to see. If you don't already have access to a paid AI coding assistant, we can reimburse the cost (up to 25 euros) so it isn't a barrier; just keep the receipt.
- **Questions: with caution.** You may email us if something is genuinely ambiguous, but treat this sparingly and with discretion. The brief is meant to stand on its own; leaning on questions to be handed the intended approach does not work in your favour.
- **Deliverable:** send back all your code, runnable from a clean checkout with a single command, together with a short write-up (a README or note) covering your result figures, the milestones you reached, your metric numbers, and a couple of sentences on what you'd do with more time. It does not need to be short for its own sake; a clear write-up that does the job is what we want. Choose your visualisation tools and evaluation metrics wisely, so that they genuinely demonstrate performance and justify your design choices.
- **AI logs:** also upload the logs or transcripts of your AI coding sessions (Claude Code, Codex, and so on). We want to see how you worked with these tools, not only the final code.
- **What happens next:** after you submit, we will meet one final time to go through the work together. You are welcome to prepare a short presentation, but it is not required.

## 3. The task

### Part 1: Toy data

A 2D dataset. Source distribution  $\pi_0$  = isotropic Gaussian. Target distribution  $\pi_1$  = a **multimodal** 2D shape; we suggest a **checkerboard** (8 squares). Multimodality matters here: it exposes the failure mode distillation is famous for (mode dropping). Two-moons / 8-Gaussians / rings are acceptable alternatives. Plot both distributions.

![A 2D scatter plot showing two distributions. The source distribution, pi_0, is an isotropic Gaussian represented by blue dots. The target distribution, pi_1, is a checkerboard pattern represented by orange dots. The checkerboard pattern consists of 8 squares arranged in a 3x3 grid with the center square missing. The plot shows the Gaussian source points and the checkerboard target points overlaid on a light gray grid.](505116873ad67b610dfceb37016d04a3_img.jpg)

A 2D scatter plot showing two distributions. The source distribution, pi\_0, is an isotropic Gaussian represented by blue dots. The target distribution, pi\_1, is a checkerboard pattern represented by orange dots. The checkerboard pattern consists of 8 squares arranged in a 3x3 grid with the center square missing. The plot shows the Gaussian source points and the checkerboard target points overlaid on a light gray grid.

Figure 1: \*

Example of the suggested setup: Gaussian source  $\pi_0 \rightarrow$  checkerboard target  $\pi_1$ .

### Part 2: Base model (the teacher)

Train a diffusion model (DDPM) *or* a flow-matching / rectified-flow model that transports  $\pi_0 \rightarrow \pi_1$ . Show that multi-step sampling (e.g. 100 to 1000 steps) recovers the target faithfully. This trained model is your **teacher** and stays **frozen** during Part 3. Design carefully how you evaluate it, and include a few visualisations that make the quality (and any failure) clear.

### Part 3: Distillation

Distill the teacher into a few-step (ideally **one-step**) generator. Do it in **two stages**, and checkpoint each.

**Stage A: DMD (distribution matching only).** Distill the teacher into a one/few-step student whose *distribution* matches the teacher's. Goal: a one/few-step student that covers all modes of the target.

**Stage B: DMD2.** Upgrade Stage A to DMD2: add the GAN loss, drop the regression loss, and use the two-time-scale update. Goal: cleaner, sharper one-step samples, and optionally a multi-step variant.

Get Stage A working and checkpointed **before** touching Stage B. Stage B's GAN is the part most likely to be unstable, so budget for it.

## 4. Milestones (checkpoint each one)

Climb as far up this ladder as you can, and checkpoint each rung with a working result you can explain. Aim for **M3** with a clean result and honest analysis; **M4** is the stretch.

| #       | Milestone                                                                                               | Deliverable                         |
|---------|---------------------------------------------------------------------------------------------------------|-------------------------------------|
| M0      | Data: $\pi_0$ , $\pi_1$ generated and plotted                                                           | figure                              |
| M1      | Teacher: base diffusion/flow trained; multi-step sampler recovers $\pi_1$                               | checkpoint + figure                 |
| M2      | Naive one-step baseline: run the teacher in a single step (no distillation), show it's blurry/collapsed | figure (motivates everything after) |
| M3      | <b>DMD (Stage A):</b> one/few-step student covers all modes                                             | checkpoint + figure + metric        |
| M4      | <b>DMD2 (Stage B):</b> GAN + two-time-scale; sharper one-step samples                                   | checkpoint + figure + metric        |
| Stretch | Ablations: $\pm$ GAN, $\pm$ two-time-scale, one-step vs 4-step                                          | figures + brief commentary          |

## 5. What we evaluate

- **Correctness over polish:** a simple loop that's right beats an elaborate one that's wrong.
- **Mode coverage** on the multimodal target, judged **from samples**, not just a low training loss.
- **A sample-based metric:** energy distance, MMD, recovered-mode count, or a 2-sample test. Pick one and report it for *teacher multi-step* vs *student one/few-step*.
- **Honesty:** tell us what didn't work. A documented failed M4 is worth more than a silent one.
- **Understanding:** when we meet, be ready to explain and justify your design choices.

## 6. Reading list

Core (read these two carefully):

- **DMD:** Yin et al., *One-step Diffusion with Distribution Matching Distillation*, CVPR 2024. [arXiv:2311.18828](#)
- **DMD2:** Yin et al., *Improved Distribution Matching Distillation for Fast Image Synthesis*, NeurIPS 2024 (Oral). [arXiv:2405.14867](#) · [code](#)

Background (the conceptual core, variational score distillation):

- **ProlificDreamer (VSD):** Wang et al., NeurIPS 2023. [arXiv:2305.16213](#)
- **DreamFusion (SDS):** Poole et al., ICLR 2023. [arXiv:2209.14988](#)

Base model (whichever you build on):

- **DDPM:** Ho et al., NeurIPS 2020. [arXiv:2006.11239](#)
- **Flow Matching:** Lipman et al., ICLR 2023. [arXiv:2210.02747](#)

- **Rectified Flow:** Liu et al., ICLR 2023. [arXiv:2209.03003](#)

## 7. Submission

Reply with a link (or zip) to your code and write-up. Make sure it runs from a clean checkout with **one command**. Then we'll set up our final meeting.

*Good luck, and have fun with it.*