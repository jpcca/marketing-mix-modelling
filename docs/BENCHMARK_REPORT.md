# Benchmark Report: Hill Mixture MMM

## Abstract
This report documents a full benchmark run of the Hill Mixture MMM framework executed on **February 7, 2026** (timestamp: `20260207_061725`). The benchmark covers both synthetic and real-data settings under a unified protocol. Overall, synthetic experiments achieved moderate convergence (66.2%), while real-data experiments showed low convergence (17.8%). Despite low convergence in mixture models, non-converged runs frequently produced competitive test RMSE, indicating that strict convergence filtering can discard potentially useful predictive runs. However, non-converged runs also introduced substantial tail risk, especially in real data, with extreme outliers in ELPD-LOO and RMSE. Consequently, this report distinguishes between (i) reliability-oriented diagnostics and (ii) accuracy-oriented model comparison based on robust statistics.

## 1. Experimental Setup

### 1.1 Artifacts Analyzed
- `results/benchmark/config.json`
- `results/benchmark/synthetic_20260207_061725.csv`
- `results/benchmark/synthetic_20260207_061725.json`
- `results/benchmark/synthetic_20260207_061725_summary.csv`
- `results/benchmark/real_20260207_061725.csv`
- `results/benchmark/real_20260207_061725.json`

### 1.2 Configuration
- Synthetic DGPs: `single`, `mixture_k2`, `mixture_k3`, `mixture_k5`
- Synthetic models: `single_hill`, `mixture_k2`, `mixture_k3`, `sparse_k5`
- Synthetic seeds: `0,1,2,3,4`
- Real organizations: `5`
- Real models: `single_hill`, `mixture_k2`, `mixture_k3`
- Real seeds: `0,1,2`
- MCMC: `num_warmup=1000`, `num_samples=2000`, `num_chains=4`
- Train/test split ratio: `0.75/0.25`

### 1.3 Experimental Matrix
- Synthetic: `4 DGP × 4 model × 5 seed = 80` runs
- Real: `5 org × 3 model × 3 seed = 45` runs

## 2. Evaluation Protocol

### 2.1 Primary Metrics
- Predictive: `train_rmse`, `test_rmse`, `train_coverage_90`, `test_coverage_90`
- Model comparison: `elpd_loo`, `p_loo`, `delta_loo`
- Mixture complexity: `effective_k_mean`

### 2.2 Diagnostic Conventions
- `converged` is defined as `max_rhat < 1.05` (`hill_mmm/inference.py`).
- `delta_loo_significant` is defined as `|delta_loo| > 2 * se` (`hill_mmm/metrics.py`).
- Synthetic summary CSV is aggregated over all runs (no convergence filtering).

### 2.3 Accuracy-Centric View
Because MMM and mixture models can be affected by label switching and slow mixing, this report includes robust predictive comparisons using:
- medians,
- 10% trimmed means,
- seed-wise winner counts.

## 3. Results: Synthetic Benchmarks

### 3.1 Convergence and Stability
Synthetic summary:
- Runs: `80`
- Converged: `53/80 (66.2%)`
- Catastrophic runs (`elpd_loo < -2000`): `16/80`

**Table 1. Convergence rate by model (synthetic)**

| model | convergence_rate |
| --- | --- |
| mixture_k2 | 30.0% |
| mixture_k3 | 80.0% |
| single_hill | 100.0% |
| sparse_k5 | 55.0% |

**Table 2. Catastrophic run counts by DGP and model (`elpd_loo < -2000`)**

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

### 3.2 Predictive and LOO Metrics

**Table 3. Means across all synthetic runs (includes non-converged runs)**

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

**Table 4. Means across converged synthetic runs only**

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

### 3.3 Accuracy-Centric Comparison (Converged vs Non-Converged)

**Table 5. Overall synthetic RMSE distribution by convergence status**

| group | n | mean_test_rmse | median_test_rmse | p90_test_rmse | max_test_rmse |
| --- | --- | --- | --- | --- | --- |
| converged | 53 | 6.234 | 5.566 | 8.557 | 9.595 |
| non_converged | 27 | 7.035 | 7.444 | 8.834 | 9.819 |

Seed-wise RMSE winners:
- Non-converged winners: **7 / 20** `(dgp, seed)` cells.

### 3.4 Effective-K Recovery

**Table 6. Effective-K recovery (`|effective_k_mean - K_true|`)**

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

## 4. Results: Real-Data Benchmarks

### 4.1 Convergence and Diagnostic Summary
Real summary:
- Runs: `45`
- Converged: `8/45 (17.8%)`
- `status=success`: `45/45`
- `pareto_k_bad` total: `0`
- `min_ess_bulk <= 10`: `36/45`
- `max_rhat >= 2.0`: `17/45`
- `elpd_loo < -1e6`: `15/45`

**Table 7. Convergence rate by model (real)**

| model | convergence_rate |
| --- | --- |
| mixture_k2 | 13.3% |
| mixture_k3 | 6.7% |
| single_hill | 33.3% |

### 4.2 Predictive Performance (All Runs)

**Table 8. Mean metrics by model (real, all runs)**

| model | elpd_loo | test_rmse | train_rmse | test_coverage_90 | time_seconds |
| --- | --- | --- | --- | --- | --- |
| mixture_k2 | -1.052e+07 | 192.886 | 160.654 | 88.9% | 1.883 |
| mixture_k3 | -2.585e+06 | 7478.114 | 8293.108 | 87.0% | 2.188 |
| single_hill | -4.719e+07 | 141.050 | 118.684 | 89.4% | 1.132 |

**Table 9. Median metrics by model (real, all runs)**

| model | elpd_loo | test_rmse | train_rmse | test_coverage_90 | time_seconds |
| --- | --- | --- | --- | --- | --- |
| mixture_k2 | -7328.036 | 100.912 | 95.238 | 91.8% | 1.873 |
| mixture_k3 | -7262.585 | 101.000 | 95.296 | 91.7% | 1.911 |
| single_hill | -1.559e+06 | 126.527 | 122.308 | 90.8% | 1.122 |

**Table 10. 10% trimmed mean of test RMSE (real)**

| model | test_rmse_trim10 |
| --- | --- |
| mixture_k2 | 167.161 |
| mixture_k3 | 143.558 |
| single_hill | 127.767 |

### 4.3 Converged vs Non-Converged Accuracy

**Table 11. RMSE by convergence status and model (real)**

| model | conv_n | conv_mean_rmse | conv_median_rmse | nonconv_n | nonconv_mean_rmse | nonconv_median_rmse |
| --- | --- | --- | --- | --- | --- | --- |
| mixture_k2 | 2 | 40.606 | 40.606 | 13 | 216.314 | 140.303 |
| mixture_k3 | 1 | 12.407 | 12.407 | 14 | 8011.379 | 101.028 |
| single_hill | 5 | 170.353 | 98.386 | 10 | 126.398 | 126.792 |

**Table 12. Overall RMSE distribution by convergence status (real)**

| group | n | mean_test_rmse | median_test_rmse | p90_test_rmse | max_test_rmse |
| --- | --- | --- | --- | --- | --- |
| converged | 8 | 118.173 | 59.090 | 307.628 | 444.095 |
| non_converged | 37 | 3141.497 | 126.527 | 426.747 | 110293.055 |

Seed-wise RMSE winners:
- Non-converged winners: **11 / 15** `(org, seed)` cells.

### 4.4 Organization-Level Comparison

**Table 13. Best model per organization (seed-mean, all runs)**

| org_id | best_by_loo | best_loo | best_by_test_rmse | best_test_rmse |
| --- | --- | --- | --- | --- |
| 7059e30b528ed5f14ee9921de13248e5 | mixture_k2 | -5916.126 | single_hill | 55.020 |
| 72a86a208d24d68b80be0e44a8a4872d | mixture_k3 | -4768.377 | single_hill | 11.749 |
| 882ce7e286d66facc66518783e2192c7 | mixture_k3 | -1.087e+07 | mixture_k3 | 163.795 |
| ba773ebd7ec0a08f1d042187d086ccb4 | mixture_k3 | -2.034e+06 | single_hill | 340.703 |
| bfb6f6a326141ed6a751fc83ba836984 | mixture_k3 | -7262.358 | mixture_k3 | 101.018 |

## 5. Failure-Mode Characterization

### 5.1 Synthetic Failures (Top by R-hat)

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

### 5.2 Real-Data Failures (Top by R-hat)

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

## 6. Discussion
1. Mixture models can outperform baseline in specific predictive cells, including non-converged runs, which supports an accuracy-oriented evaluation track.
2. Nevertheless, the real-data benchmark exhibits heavy-tailed failure behavior under non-convergence; raw means are therefore misleading.
3. For paper-quality reporting, predictive claims should be based on robust summaries (median/trimmed mean) and complemented by dispersion and failure-rate statistics.
4. Convergence flags should remain mandatory for parameter-level interpretation and causal narratives, but not necessarily for pure forecasting comparison.

## 7. Limitations
- This report is based on a single benchmark timestamp (`20260207_061725`).
- Real-data sample size is limited to five organizations and three seeds per model.
- No hierarchical uncertainty pooling across organizations was applied in this summary.

## 8. Reproducibility
- Regenerate benchmark: `python scripts/run_benchmark.py`
- Output directory: `results/benchmark/`
- This report file: `docs/BENCHMARK_REPORT.md`

