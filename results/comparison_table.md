**A. Equal budget, behaviour policies (the headline comparison)**

| checkpoint | battles | eval policy | vs random | vs maxbasepower | vs simpleheuristics |
|---|---|---|---|---|---|
| REINFORCE v1 (seed 0) | 3200 | sampled | 0.870 [0.82, 0.91] | 0.390 [0.33, 0.46] | 0.105 [0.07, 0.16] |
| REINFORCE v1 (seed 1) | 3200 | sampled | 0.940 [0.90, 0.97] | 0.535 [0.47, 0.60] | 0.135 [0.09, 0.19] |
| DQN v1 (seed 0, lr 1e-3) | 3200 | eps 0.05 | 0.800 [0.74, 0.85] | 0.320 [0.26, 0.39] | 0.070 [0.04, 0.11] |

**B. Same checkpoints, deterministic policies (argmax / ε = 0)**

| checkpoint | battles | eval policy | vs random | vs maxbasepower | vs simpleheuristics |
|---|---|---|---|---|---|
| REINFORCE v1 (seed 0) | 3200 | argmax | 0.955 [0.92, 0.98] | 0.545 [0.48, 0.61] | 0.180 [0.13, 0.24] |
| REINFORCE v1 (seed 1) | 3200 | argmax | 0.940 [0.90, 0.97] | 0.515 [0.45, 0.58] | 0.145 [0.10, 0.20] |
| DQN v1 (seed 0, lr 1e-3) | 3200 | eps 0 | 0.840 [0.78, 0.88] | 0.375 [0.31, 0.44] | 0.075 [0.05, 0.12] |

**C. Curriculum vs scratch: both algorithms continued from their v1 checkpoint in A (REINFORCE rows are the checkpoints of the earlier curriculum experiment, re-evaluated in the fixed evaluation format)**

| checkpoint | battles | eval policy | vs random | vs maxbasepower | vs simpleheuristics |
|---|---|---|---|---|---|
| REINFORCE c400 (v1 +200 it vs maxbasepower) | 6400 | sampled | 0.930 [0.89, 0.96] | 0.550 [0.48, 0.62] | 0.155 [0.11, 0.21] |
| DQN c400 (v1 +200 it vs maxbasepower, lr 2.5e-4) | 6400 | eps 0.05 | 0.810 [0.75, 0.86] | 0.250 [0.20, 0.31] | 0.070 [0.04, 0.11] |
| REINFORCE c600 (c400 +200 it vs simpleheuristics) | 9600 | sampled | 0.930 [0.89, 0.96] | 0.545 [0.48, 0.61] | 0.180 [0.13, 0.24] |
| DQN c600 (c400 +200 it vs simpleheuristics, lr 2.5e-4) | 9600 | eps 0.05 | 0.675 [0.61, 0.74] | 0.170 [0.12, 0.23] | 0.015 [0.01, 0.04] |
| REINFORCE smbp (200 it vs maxbasepower, from scratch) | 3200 | sampled | 0.920 [0.87, 0.95] | 0.455 [0.39, 0.52] | 0.090 [0.06, 0.14] |
| REINFORCE s200 (200 it vs simpleheuristics, from scratch) | 3200 | sampled | 0.160 [0.12, 0.22] | 0.030 [0.01, 0.06] | 0.005 [0.00, 0.03] |
| REINFORCE s400 (400 it vs simpleheuristics, from scratch) | 6400 | sampled | 0.140 [0.10, 0.19] | 0.015 [0.01, 0.04] | 0.000 [0.00, 0.02] |
| REINFORCE s600 (600 it vs simpleheuristics, from scratch) | 9600 | sampled | 0.685 [0.62, 0.75] | 0.255 [0.20, 0.32] | 0.060 [0.03, 0.10] |

**D. DQN training parameters: CartPole-derived learning rate vs lr 1e-3 (same budget, same seed)**

| checkpoint | battles | eval policy | vs random | vs maxbasepower | vs simpleheuristics |
|---|---|---|---|---|---|
| DQN lr 2.5e-4 (CartPole settings) | 3200 | eps 0.05 | 0.695 [0.63, 0.75] | 0.175 [0.13, 0.23] | 0.025 [0.01, 0.06] |
| DQN lr 1e-3 (= DQN v1 in A) | 3200 | eps 0.05 | 0.800 [0.74, 0.85] | 0.320 [0.26, 0.39] | 0.070 [0.04, 0.11] |
| DQN lr 2.5e-4 (CartPole settings) | 3200 | eps 0 | 0.695 [0.63, 0.75] | 0.230 [0.18, 0.29] | 0.015 [0.01, 0.04] |
| DQN lr 1e-3 (= DQN v1 in A) | 3200 | eps 0 | 0.840 [0.78, 0.88] | 0.375 [0.31, 0.44] | 0.075 [0.05, 0.12] |
