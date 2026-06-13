# AI-assisted sessions index

This project was built with AI assistance, logged honestly here.

Workflow: I use **ChatGPT** as a planning and prompt-refinement assistant first,
then pass refined instructions to **Codex** as the coding agent. See
[chatgpt_planning.md](chatgpt_planning.md) for details.

| Date | Session | Tool(s) | Notes |
|------|---------|---------|-------|
| 2026-06-12 | M0 repo + data setup | ChatGPT (planning) -> Codex (coding) | [2026-06-12_m0_setup.md](2026-06-12_m0_setup.md) |
| 2026-06-12 | M1 DDPM teacher | ChatGPT (planning) -> Codex (coding) | [2026-06-12_m1_teacher.md](2026-06-12_m1_teacher.md) |
| 2026-06-12 | M1 EMA teacher | ChatGPT (planning) -> Codex (coding) | [2026-06-12_m1_ema.md](2026-06-12_m1_ema.md) |
| 2026-06-12 | M2 one-step baseline | ChatGPT (planning) -> Codex (coding) | [2026-06-12_m2_onestep.md](2026-06-12_m2_onestep.md) |
| 2026-06-12 | M3 DMD Stage A | ChatGPT (planning) -> Codex (coding) | [2026-06-12_m3_dmd.md](2026-06-12_m3_dmd.md) |
| 2026-06-13 | M3.1 hparam tuning (results kept, code removed) | ChatGPT (planning) -> Codex (coding) | [2026-06-13_m3_tuning.md](2026-06-13_m3_tuning.md) |
| 2026-06-13 | M4 DMD2 Stage B (GAN) | ChatGPT (planning) -> Codex (coding) | [2026-06-13_m4_dmd2.md](2026-06-13_m4_dmd2.md) |
| 2026-06-13 | M4.1 GAN diagnostics + tuning (disc redesign; results kept, tuning code temporary) | ChatGPT (planning) -> Codex (coding) | [2026-06-13_m4_tuning.md](2026-06-13_m4_tuning.md) |
| 2026-06-13 | M5 few-step DMD2 student (sec 4.4 + 4.5; beats M3 on off-support) | Cursor (Claude) agent | [2026-06-13_m5_fewstep.md](2026-06-13_m5_fewstep.md) |

- Raw transcripts go in [transcripts/](transcripts/).
