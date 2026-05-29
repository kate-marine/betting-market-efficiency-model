## Calibration Ablation: Raw Model vs. Post-Hoc Calibrated vs. Market

Evaluable seasons: ['2017-2018', '2018-2019', '2019-2020', '2020-2021', '2021-2022']

Bootstrap CIs: 95% percentile, n=1000, match-level resampling.

| estimator   |   brier |   brier_lo |   brier_hi |    ece |   ece_lo |   ece_hi |   log_loss |   n_matches |
|:------------|--------:|-----------:|-----------:|-------:|---------:|---------:|-----------:|------------:|
| Raw model   |  0.2122 |     0.2098 |     0.2146 | 0.0685 |   0.0632 |   0.074  |     1.0744 |       13265 |
| Isotonic    |  0.2052 |     0.2037 |     0.2067 | 0.0124 |   0.009  |   0.0168 |     1.0508 |       13265 |
| Platt       |  0.2049 |     0.2033 |     0.2064 | 0.011  |   0.0075 |   0.0158 |     1.0254 |       13265 |
| Market      |  0.1953 |     0.1935 |     0.1972 | 0.0058 |   0.0041 |   0.0105 |     0.9836 |       13265 |

**ECE gap closed by isotonic calibration:** 89.5% of raw→market gap
