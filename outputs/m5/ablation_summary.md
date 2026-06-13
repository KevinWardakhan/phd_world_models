# M5 ablation: +/-GAN, +/-TTUR, 1-step vs 4-step

Ranked by off-support fraction (sharper boundaries), lower is better. All arms share the teacher-EMA init, step budget and hyperparameters.

| variant | steps | GAN | TTUR | modes | norm. entropy | off-support | energy dist | MMD |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 4-step, +GAN (fu=3) [main M5] | 4 | yes | fu=3 | 8 | 0.996 | **0.064** | **0.0060** | **0.0027** |
| 4-step, no GAN (fu=3) | 4 | no | fu=3 | 8 | 0.995 | 0.071 | 0.0171 | 0.0059 |
| 4-step, no GAN, fu=1 (-TTUR) | 4 | no | fu=1 | 8 | 0.995 | 0.080 | 0.0143 | 0.0084 |
| 1-step, no GAN (fu=3) | 1 | no | fu=3 | 8 | 0.998 | **0.144** | **0.0044** | **0.0015** |
| 1-step, +GAN (fu=3) | 1 | yes | fu=3 | 8 | 0.999 | 0.148 | 0.0058 | 0.0020 |
