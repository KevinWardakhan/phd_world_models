# Sources index

Compact map of what each source in `resources/` is used for in this assignment.
This folder is context and traceability only; the runtime code under `src/` does
not import from it.

| Source | File | Useful for |
|--------|------|------------|
| Assignment brief | `assignment/take_home_brief.md` | Task definition, milestones (M0-M4), deliverables, evaluation criteria, AI-log requirement. |
| DDPM (Ho et al., 2020) | `papers/ddpm.md` | Reference for the teacher: diffusion forward/reverse process and noise-prediction training (M1). |
| DMD (Yin et al., 2024) | `papers/dmd.md` | Stage A distillation: distribution matching via the difference of real/fake scores; the fake score model (M3). |
| DMD2 (Yin et al., 2024) | `papers/dmd2.md` | Stage B distillation: dropping the regression loss, adding the GAN loss, two-time-scale update, training stability (M4). |
| Di[M]O (Zhu et al.) | `papers/dimo.md` | One-step distillation perspective: student-generated states, auxiliary model, and mode-collapse / entropy considerations. |
