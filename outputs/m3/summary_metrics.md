# M3 summary metrics (vs true pi_1)

| comparison | modes | norm. entropy | off-support | energy dist | MMD |
|---|:---:|:---:|:---:|:---:|:---:|
| pi_1 vs pi_1 (sanity) | 8 | 1.000 | 0.000 | 0.0014 | 0.0003 |
| pi_0 vs pi_1 (baseline) | 8 | 0.951 | 0.535 | 0.0419 | 0.0298 |
| M1 multi-step teacher | 8 | 0.999 | 0.124 | 0.0053 | 0.0014 |
| M2 one-step teacher | 7 | 0.558 | 0.443 | 0.3106 | 0.2439 |
| M3 dmd_reg | 8 | 0.999 | 0.147 | 0.0070 | 0.0022 |
| M3 dmd_no_reg (best) | 8 | 0.999 | 0.180 | 0.0055 | 0.0024 |
| M3 dmd_no_reg_ttur | 8 | 0.998 | 0.164 | 0.0080 | 0.0022 |

Best variant: **dmd_no_reg**; beats M2 one-step: **True**.
