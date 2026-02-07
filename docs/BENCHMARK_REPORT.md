# Benchmark Report

## Scope
This report analyzes benchmark artifacts in `results/benchmark/` generated at timestamp `20260207_061725`.

## Artifacts
- `results/benchmark/config.json`
- `results/benchmark/synthetic_20260207_061725.csv`
- `results/benchmark/synthetic_20260207_061725.json`
- `results/benchmark/synthetic_20260207_061725_summary.csv`
- `results/benchmark/real_20260207_061725.csv`
- `results/benchmark/real_20260207_061725.json`

## Configuration
- Synthetic DGPs: ['single', 'mixture_k2', 'mixture_k3', 'mixture_k5']
- Synthetic models: ['single_hill', 'mixture_k2', 'mixture_k3', 'sparse_k5']
- Synthetic seeds: [0, 1, 2, 3, 4]
- Real organizations: 5
- Real models: ['single_hill', 'mixture_k2', 'mixture_k3']
- Real seeds: [0, 1, 2]
- MCMC: warmup=1000, samples=2000, chains=4
- Train ratio: 0.75

## Executive Summary
- Synthetic experiments: 80 runs, 53/80 converged (66.2%).
- Real experiments: 45 runs, 8/45 converged (17.8%).
- For synthetic benchmarks, several catastrophic non-converged runs heavily distort mean ELPD-LOO for `mixture_k2` and `sparse_k5`.
- For real benchmarks, predictive accuracy can still be competitive in some non-converged runs, but variance is very large and includes extreme failures.
- Accuracy-only evaluation should use robust aggregations (median / trimmed mean), not plain mean.

## Synthetic: Convergence Diagnostics
| model | convergence_rate |
| --- | --- |
| mixture_k2 | 30.0% |
| mixture_k3 | 80.0% |
| single_hill | 100.0% |
| sparse_k5 | 55.0% |

Catastrophic runs are defined here as `elpd_loo < -2000`.
| dgp | model | catastrophic_count |
| --- | --- | --- |
| mixture_k2 | mixture_k2 | 2 |
| mixture_k2 | mixture_k3 | 0 |
| mixture_k2 | single_hill | 0 |
| mixture_k2 | sparse_k5 | 2 |
| mixture_k3 | mixture_k2 | 2 |
| mixture_k3 | mixture_k3 | 0 |
| mixture_k3 | single_hill | 0 |
| mixture_k3 | sparse_k5 | 2 |
| mixture_k5 | mixture_k2 | 2 |
| mixture_k5 | mixture_k3 | 0 |
| mixture_k5 | single_hill | 0 |
| mixture_k5 | sparse_k5 | 2 |
| single | mixture_k2 | 2 |
| single | mixture_k3 | 0 |
| single | single_hill | 0 |
| single | sparse_k5 | 2 |

## Synthetic: Predictive and Model-Comparison Metrics
### Means across all runs (includes non-converged runs)
| dgp | model | elpd_loo | test_rmse | train_rmse | test_coverage_90 | effective_k_mean | delta_loo |
| --- | --- | --- | --- | --- | --- | --- | --- |
| mixture_k2 | mixture_k2 | -13736.050 | 4.795 | 4.139 | 84.8% | 1.993 | -13326.344 |
| mixture_k2 | mixture_k3 | -406.354 | 5.157 | 3.586 | 73.6% | 2.513 | 3.353 |
| mixture_k2 | single_hill | -409.707 | 5.382 | 3.574 | 73.2% | 1.000 | 0.000 |
| mixture_k2 | sparse_k5 | -5681.563 | 5.094 | 3.751 | 75.6% | 3.013 | -5271.857 |
| mixture_k3 | mixture_k2 | -34256.100 | 8.355 | 7.900 | 88.4% | 2.000 | -33735.375 |
| mixture_k3 | mixture_k3 | -488.063 | 8.291 | 7.570 | 85.6% | 2.925 | 32.661 |
| mixture_k3 | single_hill | -520.724 | 8.307 | 7.528 | 82.4% | 1.000 | 0.000 |
| mixture_k3 | sparse_k5 | -29021.516 | 8.335 | 7.640 | 87.2% | 3.815 | -28500.792 |
| mixture_k5 | mixture_k2 | -28668.929 | 7.755 | 7.290 | 87.2% | 1.999 | -28159.443 |
| mixture_k5 | mixture_k3 | -498.894 | 7.689 | 6.955 | 86.0% | 2.829 | 10.592 |
| mixture_k5 | single_hill | -509.486 | 7.792 | 6.925 | 85.6% | 1.000 | 0.000 |
| mixture_k5 | sparse_k5 | -25049.015 | 7.667 | 7.096 | 86.8% | 3.686 | -24539.528 |
| single | mixture_k2 | -13973.083 | 4.768 | 3.413 | 76.8% | 1.958 | -13590.730 |
| single | mixture_k3 | -383.380 | 4.820 | 2.990 | 69.6% | 1.952 | -1.027 |
| single | single_hill | -382.353 | 5.139 | 2.982 | 66.0% | 1.000 | 0.000 |
| single | sparse_k5 | -4295.580 | 4.728 | 3.196 | 71.6% | 2.249 | -3913.227 |

### Means across converged runs only
| dgp | model | elpd_loo | test_rmse | train_rmse | test_coverage_90 | effective_k_mean | delta_loo |
| --- | --- | --- | --- | --- | --- | --- | --- |
| mixture_k2 | mixture_k2 | -420.605 | 3.731 | 3.837 | 92.0% | 1.991 | -0.264 |
| mixture_k2 | mixture_k3 | -406.354 | 5.157 | 3.586 | 73.6% | 2.513 | 3.353 |
| mixture_k2 | single_hill | -409.707 | 5.382 | 3.574 | 73.2% | 1.000 | 0.000 |
| mixture_k2 | sparse_k5 | -409.983 | 4.717 | 3.648 | 78.7% | 3.032 | 2.344 |
| mixture_k3 | mixture_k3 | -494.754 | 8.231 | 7.824 | 90.0% | 2.933 | 30.989 |
| mixture_k3 | single_hill | -520.724 | 8.307 | 7.528 | 82.4% | 1.000 | 0.000 |
| mixture_k3 | sparse_k5 | -482.877 | 8.387 | 7.305 | 85.0% | 3.867 | 32.708 |
| mixture_k5 | mixture_k2 | -499.433 | 6.924 | 7.101 | 88.0% | 1.999 | 13.833 |
| mixture_k5 | mixture_k3 | -500.721 | 7.684 | 6.883 | 88.0% | 2.831 | 6.789 |
| mixture_k5 | single_hill | -509.486 | 7.792 | 6.925 | 85.6% | 1.000 | 0.000 |
| mixture_k5 | sparse_k5 | -501.016 | 7.381 | 6.887 | 90.0% | 3.783 | 6.548 |
| single | mixture_k2 | -380.657 | 5.022 | 2.934 | 66.0% | 1.945 | -0.911 |
| single | mixture_k3 | -383.380 | 4.820 | 2.990 | 69.6% | 1.952 | -1.027 |
| single | single_hill | -382.353 | 5.139 | 2.982 | 66.0% | 1.000 | 0.000 |
| single | sparse_k5 | -384.492 | 4.841 | 3.022 | 70.7% | 2.176 | -0.593 |

### Converged vs non-converged RMSE (overall)
| group | n | mean_test_rmse | median_test_rmse | p90_test_rmse | max_test_rmse |
| --- | --- | --- | --- | --- | --- |
| converged | 53 | 6.234 | 5.566 | 8.557 | 9.595 |
| non_converged | 27 | 7.035 | 7.444 | 8.834 | 9.819 |

### Effective K recovery (all runs)
| dgp | model | effective_k_mean | k_abs_err |
| --- | --- | --- | --- |
| mixture_k2 | mixture_k2 | 1.993 | 0.007 |
| mixture_k2 | mixture_k3 | 2.513 | 0.513 |
| mixture_k2 | single_hill | 1.000 | 1.000 |
| mixture_k2 | sparse_k5 | 3.013 | 1.014 |
| mixture_k3 | mixture_k2 | 2.000 | 1.000 |
| mixture_k3 | mixture_k3 | 2.925 | 0.075 |
| mixture_k3 | single_hill | 1.000 | 2.000 |
| mixture_k3 | sparse_k5 | 3.815 | 0.815 |
| mixture_k5 | mixture_k2 | 1.999 | 3.001 |
| mixture_k5 | mixture_k3 | 2.829 | 2.171 |
| mixture_k5 | single_hill | 1.000 | 4.000 |
| mixture_k5 | sparse_k5 | 3.686 | 1.314 |
| single | mixture_k2 | 1.958 | 0.958 |
| single | mixture_k3 | 1.952 | 0.952 |
| single | single_hill | 1.000 | 0.000 |
| single | sparse_k5 | 2.249 | 1.249 |

### Seed-wise RMSE winners
- Non-converged winners: 7 / 20 `(dgp, seed)` cells.

## Real: Convergence Diagnostics
| model | convergence_rate |
| --- | --- |
| mixture_k2 | 13.3% |
| mixture_k3 | 6.7% |
| single_hill | 33.3% |

- `status=success`: 45/45
- `pareto_k_bad` total count: 0
- Runs with `min_ess_bulk <= 10`: 36/45
- Runs with `max_rhat >= 2.0`: 17/45
- Runs with `elpd_loo < -1e6`: 15/45

## Real: Predictive Accuracy
### Mean metrics (all runs)
| model | elpd_loo | test_rmse | train_rmse | test_coverage_90 | time_seconds |
| --- | --- | --- | --- | --- | --- |
| mixture_k2 | -1.052e+07 | 192.886 | 160.654 | 88.9% | 1.883 |
| mixture_k3 | -2.585e+06 | 7478.114 | 8293.108 | 87.0% | 2.188 |
| single_hill | -4.719e+07 | 141.050 | 118.684 | 89.4% | 1.132 |

### Median metrics (all runs)
| model | elpd_loo | test_rmse | train_rmse | test_coverage_90 | time_seconds |
| --- | --- | --- | --- | --- | --- |
| mixture_k2 | -7328.036 | 100.912 | 95.238 | 91.8% | 1.873 |
| mixture_k3 | -7262.585 | 101.000 | 95.296 | 91.7% | 1.911 |
| single_hill | -1.559e+06 | 126.527 | 122.308 | 90.8% | 1.122 |

### 10% trimmed mean of test RMSE
| model | test_rmse_trim10 |
| --- | --- |
| mixture_k2 | 167.161 |
| mixture_k3 | 143.558 |
| single_hill | 127.767 |

### Converged vs non-converged RMSE by model
| model | conv_n | conv_mean_rmse | conv_median_rmse | nonconv_n | nonconv_mean_rmse | nonconv_median_rmse |
| --- | --- | --- | --- | --- | --- | --- |
| mixture_k2 | 2 | 40.606 | 40.606 | 13 | 216.314 | 140.303 |
| mixture_k3 | 1 | 12.407 | 12.407 | 14 | 8011.379 | 101.028 |
| single_hill | 5 | 170.353 | 98.386 | 10 | 126.398 | 126.792 |

### Converged vs non-converged RMSE (overall)
| group | n | mean_test_rmse | median_test_rmse | p90_test_rmse | max_test_rmse |
| --- | --- | --- | --- | --- | --- |
| converged | 8 | 118.173 | 59.090 | 307.628 | 444.095 |
| non_converged | 37 | 3141.497 | 126.527 | 426.747 | 110293.055 |

### Seed-wise RMSE winners
- Non-converged winners: 11 / 15 `(org, seed)` cells.

### Best model by organization (seed-mean, all runs)
| org_id | best_by_loo | best_loo | best_by_test_rmse | best_test_rmse |
| --- | --- | --- | --- | --- |
| 7059e30b528ed5f14ee9921de13248e5 | mixture_k2 | -5916.126 | single_hill | 55.020 |
| 72a86a208d24d68b80be0e44a8a4872d | mixture_k3 | -4768.377 | single_hill | 11.749 |
| 882ce7e286d66facc66518783e2192c7 | mixture_k3 | -1.087e+07 | mixture_k3 | 163.795 |
| ba773ebd7ec0a08f1d042187d086ccb4 | mixture_k3 | -2.034e+06 | single_hill | 340.703 |
| bfb6f6a326141ed6a751fc83ba836984 | mixture_k3 | -7262.358 | mixture_k3 | 101.018 |

## Failure Cases (Top by R-hat)
### Synthetic
| dgp | model | seed | max_rhat | min_ess_bulk | elpd_loo | test_rmse |
| --- | --- | --- | --- | --- | --- | --- |
| mixture_k3 | mixture_k2 | 4 | 2.700 | 5 | -18859.193 | 8.699 |
| mixture_k5 | mixture_k2 | 4 | 2.250 | 6 | -16425.453 | 8.267 |
| mixture_k2 | mixture_k2 | 2 | 2.060 | 5 | -58272.901 | 5.064 |
| mixture_k5 | mixture_k2 | 2 | 1.820 | 6 | -125421.513 | 8.870 |
| mixture_k3 | sparse_k5 | 0 | 1.700 | 7 | -500.134 | 7.118 |
| mixture_k3 | mixture_k3 | 3 | 1.680 | 6 | -480.193 | 8.738 |
| mixture_k2 | mixture_k2 | 1 | 1.600 | 7 | -1723.045 | 3.831 |
| mixture_k2 | sparse_k5 | 3 | 1.600 | 7 | -20239.214 | 5.422 |

### Real
| org_id | model | seed | max_rhat | min_ess_bulk | elpd_loo | test_rmse |
| --- | --- | --- | --- | --- | --- | --- |
| 882ce7e286d66facc66518783e2192c7 | mixture_k2 | 1 | 6.700 | 4 | -3.788e+07 | 449.139 |
| ba773ebd7ec0a08f1d042187d086ccb4 | mixture_k3 | 1 | 4.590 | 4 | -6.086e+06 | 110293.055 |
| ba773ebd7ec0a08f1d042187d086ccb4 | mixture_k2 | 1 | 3.410 | 4 | -1.058e+08 | 707.712 |
| 7059e30b528ed5f14ee9921de13248e5 | mixture_k3 | 2 | 3.300 | 4 | -5977.608 | 68.835 |
| 882ce7e286d66facc66518783e2192c7 | mixture_k3 | 0 | 3.170 | 4 | -3.261e+07 | 175.750 |
| 7059e30b528ed5f14ee9921de13248e5 | mixture_k3 | 1 | 3.030 | 5 | -6517.861 | 54.150 |
| 882ce7e286d66facc66518783e2192c7 | mixture_k3 | 1 | 3.020 | 5 | -8082.885 | 149.748 |
| bfb6f6a326141ed6a751fc83ba836984 | mixture_k2 | 1 | 2.720 | 5 | -1.407e+07 | 140.303 |
| ba773ebd7ec0a08f1d042187d086ccb4 | mixture_k3 | 2 | 2.670 | 5 | -7989.905 | 433.545 |
| ba773ebd7ec0a08f1d042187d086ccb4 | mixture_k2 | 2 | 2.660 | 5 | -8261.100 | 419.039 |

## Interpretation and Recommendations
1. Non-converged runs are not always useless for pure prediction. In this benchmark, they frequently win on RMSE in head-to-head cells.
2. However, non-converged runs dramatically increase tail risk (extreme RMSE / ELPD outliers), especially on real data.
3. For decision-making, use robust ranking (`median`, `trimmed mean`) and report dispersion (IQR/std) in parallel.
4. Keep `converged` as a reliability flag, but do not use it as a hard exclusion when the objective is strictly predictive.
5. For model interpretation (component-level parameters, causal narrative), require stronger convergence and relabeling diagnostics.

## Metric Definitions Used in This Project
- `converged`: `max_rhat < 1.05` from `hill_mmm/inference.py`.
- `delta_loo_significant`: `|delta_loo| > 2 * se` from `hill_mmm/metrics.py`.
- Synthetic summary CSV (`*_summary.csv`) is aggregated over all runs without filtering by convergence.
