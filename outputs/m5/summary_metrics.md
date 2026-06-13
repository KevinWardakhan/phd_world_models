# M5 summary metrics (vs true pi_1)

| comparison | modes | norm. entropy | off-support | energy dist | MMD |
|---|:---:|:---:|:---:|:---:|:---:|
| pi_1 vs pi_1 (sanity) | 8 | 1.000 | 0.000 | 0.0014 | 0.0003 |
| pi_0 vs pi_1 (baseline) | 8 | 0.951 | 0.535 | 0.0419 | 0.0298 |
| M1 multi-step teacher | 8 | 0.999 | 0.124 | 0.0053 | 0.0014 |
| M2 one-step teacher | 7 | 0.558 | 0.443 | 0.3106 | 0.2439 |
| M3 best student (1-step) | 8 | 0.998 | 0.158 | 0.0059 | 0.0022 |
| M4 DMD2 student (1-step) | 8 | 0.998 | 0.137 | 0.0088 | 0.0029 |
| **M5 DMD2 student (4-step)** | 8 | 0.996 | 0.064 | 0.0060 | 0.0027 |

Schedule (timesteps): [199, 149, 99, 49].

Beats M3 one-step on off-support: **True**; modes preserved (8/8, entropy>=0.99): **True**; ED not worse than M3: **True**; **clear win: True**.

Robustness (off-support fraction):

- n2000_seedA: off-support 0.061, modes 8, ED 0.0114
- n2000_seedB: off-support 0.066, modes 8, ED 0.0040
- n5000_seedA: off-support 0.058, modes 8, ED 0.0066
