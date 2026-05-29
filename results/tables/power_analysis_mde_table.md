## Power Analysis: Minimum Detectable Effect per League

α = 0.05, target power = 80%. MDE = (z_{α/2} + z_{1-β}) × SE_cluster.

Reference: pooled γ = 0.0459 (robust across bootstrap CIs).

| league   |   gamma |   gamma_se |   n_matches |    mde |   power_at_obs |   power_at_pooled | adequately_powered   |
|:---------|--------:|-----------:|------------:|-------:|---------------:|------------------:|:---------------------|
| Pooled   |  0.0459 |     0.0131 |      18,538 | 0.0368 |         0.9378 |            0.938  | True                 |
| D1       | -0.0228 |     0.037  |       2,142 | 0.1038 |         0.0942 |            0.2363 | False                |
| E0       |  0.0154 |     0.0307 |       2,660 | 0.0861 |         0.0791 |            0.3209 | False                |
| E1       |  0.0877 |     0.0449 |       3,863 | 0.1259 |         0.4966 |            0.1754 | False                |
| E2       |  0.2205 |     0.0891 |       1,104 | 0.2497 |         0.6964 |            0.0809 | False                |
| F1       |  0.0571 |     0.0371 |       2,279 | 0.1038 |         0.338  |            0.2361 | False                |
| I1       |  0.1141 |     0.0322 |       2,279 | 0.0901 |         0.9439 |            0.2975 | True                 |
| N1       |  0.0502 |     0.0541 |         612 | 0.1516 |         0.153  |            0.1356 | False                |
| SC0      |  0.0498 |     0.0411 |       1,319 | 0.1153 |         0.2276 |            0.2003 | False                |
| SP1      |  0.0217 |     0.0348 |       2,280 | 0.0974 |         0.0957 |            0.2617 | False                |

*power@pool*: power at the true effect if γ = pooled estimate.
Leagues with power@pool < 0.80 would miss a pooled-sized effect most of the time.
