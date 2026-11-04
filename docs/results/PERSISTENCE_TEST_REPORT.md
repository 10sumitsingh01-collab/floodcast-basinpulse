# BASINSHIFT evaluation: `persistence_baseline`

Model `src.baselines`, commit `33bd13a-dirty`, 2026-11-04T17:19:13+00:00. Fit 4.5 s, predict 5.3 s, total 375.9 s.

Values with 95% bootstrap CIs (resampling basins). Event counts are horizon-specific.

## temporal (239 basins)

| metric | h=1 | h=3 | h=7 |
|---|---|---|---|
| nse_median | 0.509 [0.395, 0.548] | -0.191 [-0.266, -0.117] | -0.485 [-0.534, -0.431] |
| kge_median | 0.754 [0.698, 0.774] | 0.405 [0.367, 0.442] | 0.258 [0.233, 0.285] |
| mae_pooled | 0.489 [0.415, 0.571] | 0.874 [0.758, 0.992] | 1.109 [0.991, 1.267] |
| auprc_high | 0.632 [0.603, 0.660] | 0.301 [0.263, 0.337] | 0.187 [0.160, 0.218] |
| auprc_extreme | 0.201 [0.178, 0.223] | 0.076 [0.061, 0.096] | 0.042 [0.032, 0.054] |
| brier_skill_high | 0.498 [0.470, 0.525] | 0.190 [0.151, 0.229] | 0.094 [0.044, 0.135] |
| recall_high | 0.693 [0.670, 0.716] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| precision_high | 0.682 [0.661, 0.704] | nan | nan |
| false_alarms_per_basin_year_high | 5.834 [5.470, 6.180] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| interval_coverage | 0.804 [0.794, 0.817] | 0.804 [0.790, 0.818] | 0.801 [0.787, 0.816] |
| interval_width | 0.679 [0.588, 0.798] | 1.804 [1.549, 2.071] | 2.909 [2.599, 3.276] |

High-flow events: h=1: 25,573, h=3: 25,571, h=7: 25,568

## geo (60 basins)

| metric | h=1 | h=3 | h=7 |
|---|---|---|---|
| nse_median | 0.474 [0.273, 0.573] | -0.135 [-0.351, -0.027] | -0.445 [-0.541, -0.330] |
| kge_median | 0.737 [0.637, 0.787] | 0.433 [0.325, 0.487] | 0.278 [0.229, 0.335] |
| mae_pooled | 0.495 [0.376, 0.631] | 0.888 [0.628, 1.127] | 1.109 [0.830, 1.446] |
| auprc_high | 0.630 [0.575, 0.673] | 0.295 [0.239, 0.360] | 0.192 [0.159, 0.236] |
| auprc_extreme | 0.194 [0.170, 0.218] | 0.064 [0.047, 0.082] | 0.037 [0.029, 0.048] |
| brier_skill_high | 0.495 [0.443, 0.539] | 0.182 [0.130, 0.237] | 0.097 [0.070, 0.128] |
| recall_high | 0.694 [0.652, 0.729] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| precision_high | 0.681 [0.641, 0.716] | nan | nan |
| false_alarms_per_basin_year_high | 5.880 [5.235, 6.615] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| interval_coverage | 0.796 [0.766, 0.820] | 0.798 [0.767, 0.831] | 0.799 [0.770, 0.832] |
| interval_width | 0.653 [0.483, 0.845] | 1.734 [1.195, 2.187] | 2.794 [2.121, 3.667] |

High-flow events: h=1: 29,271, h=3: 29,271, h=7: 29,266

## geo_temporal (60 basins)

| metric | h=1 | h=3 | h=7 |
|---|---|---|---|
| nse_median | 0.485 [0.374, 0.622] | -0.171 [-0.253, 0.038] | -0.429 [-0.528, -0.379] |
| kge_median | 0.742 [0.687, 0.811] | 0.415 [0.373, 0.519] | 0.283 [0.236, 0.311] |
| mae_pooled | 0.500 [0.358, 0.647] | 0.898 [0.658, 1.199] | 1.139 [0.847, 1.477] |
| auprc_high | 0.609 [0.558, 0.658] | 0.274 [0.227, 0.337] | 0.160 [0.128, 0.197] |
| auprc_extreme | 0.192 [0.164, 0.225] | 0.063 [0.043, 0.086] | 0.034 [0.022, 0.050] |
| brier_skill_high | 0.511 [0.447, 0.585] | 0.221 [0.141, 0.309] | 0.132 [0.045, 0.224] |
| recall_high | 0.677 [0.642, 0.716] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| precision_high | 0.669 [0.633, 0.705] | nan | nan |
| false_alarms_per_basin_year_high | 5.647 [4.731, 6.429] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| interval_coverage | 0.803 [0.774, 0.832] | 0.804 [0.771, 0.832] | 0.800 [0.763, 0.832] |
| interval_width | 0.670 [0.462, 0.883] | 1.779 [1.322, 2.427] | 2.869 [2.046, 3.760] |

High-flow events: h=1: 5,976, h=3: 5,976, h=7: 5,975

## Stress tests (geo_temporal, h=1): relative change vs clean

| scenario | auprc_high | brier_high | interval_coverage | mae_pooled | nse_median |
|---|---|---|---|---|---|
| gaps_3to7d | -8.5% | +10.0% | -2.8% | +10.1% | -14.0% |
| mask_10pct | -4.5% | +5.0% | -1.7% | +5.6% | -5.6% |
| mask_30pct | -15.1% | +18.0% | -6.1% | +18.8% | -33.0% |
| noise_weather | +0.0% | +0.0% | +0.0% | +0.0% | +0.0% |
| obs_delay_1d | -39.6% | +51.9% | -16.2% | +53.0% | -93.9% |

Scenarios: `mask_10pct` = 10% of input values removed at random; `mask_30pct` = 30% of input values removed at random; `gaps_3to7d` = random 3-7 day gaps covering ~10% of days, all inputs missing; `noise_weather` = Gaussian noise, sd = 10% of each basin's sd, on weather inputs; `obs_delay_1d` = latest observed flow arrives 1 day late
