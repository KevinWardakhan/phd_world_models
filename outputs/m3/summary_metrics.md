# M3 summary metrics (vs true pi_1)

| comparison | modes | norm. entropy | off-support | energy dist | MMD |
|---|:---:|:---:|:---:|:---:|:---:|
| M1 multi-step teacher | 8 | 0.999 | 0.124 | 0.0053 | 0.0014 |
| M2 one-step teacher | 7 | 0.558 | 0.443 | 0.3106 | 0.2439 |
| M3 dmd_no_reg | 8 | 0.998 | 0.169 | 0.0091 | 0.0031 |
| M3 dmd_reg | 8 | 0.999 | 0.135 | 0.0080 | 0.0024 |
| M3 dmd_no_reg_ttur (best) | 8 | 0.998 | 0.158 | **0.0059** | **0.0022** |

Best variant: **dmd_no_reg_ttur**; beats M2 one-step: **True**.
