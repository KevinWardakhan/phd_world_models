# M4 summary metrics (vs true pi_1)

| comparison | modes | norm. entropy | off-support | energy dist | MMD |
|---|:---:|:---:|:---:|:---:|:---:|
| pi_1 vs pi_1 (sanity) | 8 | 1.000 | 0.000 | 0.0014 | 0.0003 |
| pi_0 vs pi_1 (baseline) | 8 | 0.951 | 0.535 | 0.0419 | 0.0298 |
| M1 multi-step teacher | 8 | 0.999 | 0.124 | 0.0053 | 0.0014 |
| M2 one-step teacher | 7 | 0.558 | 0.443 | 0.3106 | 0.2439 |
| M3 best student (no-reg TTUR) | 8 | 0.998 | 0.158 | 0.0059 | 0.0022 |
| **M4 DMD2 student** | 8 | 0.998 | 0.137 | 0.0088 | 0.0029 |

M4 improves over best M3 (energy distance): **False**; MMD: **False**.

Final GAN diagnostics: mean real logit 0.000, mean fake logit -0.006, disc total accuracy 0.519.
