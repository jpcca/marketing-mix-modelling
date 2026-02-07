# Benchmark Evaluation for Hill Mixture MMM

## Abstract
This document provides a paper-style summary of the benchmark run executed on **February 7, 2026** (timestamp: `20260207_061725`) for the Hill Mixture MMM framework. We evaluate synthetic and real-data experiments under a common protocol and separate two analytical lenses: (i) reliability-oriented inference diagnostics and (ii) accuracy-oriented predictive comparison. Synthetic experiments reached moderate convergence (66.2%), while real-data experiments showed low convergence (17.8%). Importantly, non-converged runs were often competitive in test RMSE, but they also introduced heavy-tailed risk, including extreme ELPD-LOO and RMSE failures in real data. The main manuscript narrative should therefore focus on robust predictive summaries (median/trimmed mean) and explicitly distinguish predictive utility from parameter-level interpretability.

## 1. Introduction
The benchmark was designed to assess whether mixture-based Hill models provide practical predictive value over a single-Hill baseline when MCMC convergence is challenging. This is particularly relevant for MMM settings where latent heterogeneity and label switching can destabilize posterior sampling, while forecast-level aggregates may still remain useful.

The central question for paper writing is not only whether models converge in the strict MCMC sense, but also whether non-converged runs can still deliver acceptable out-of-sample accuracy, and under what risk profile.

## 2. Experimental Design

### 2.1 Data and Model Grid
- Synthetic: 4 DGPs (`single`, `mixture_k2`, `mixture_k3`, `mixture_k5`) × 4 models (`single_hill`, `mixture_k2`, `mixture_k3`, `sparse_k5`) × 5 seeds = 80 runs.
- Real: 5 organizations × 3 models (`single_hill`, `mixture_k2`, `mixture_k3`) × 3 seeds = 45 runs.

### 2.2 Inference Configuration
- `num_warmup=1000`, `num_samples=2000`, `num_chains=4`
- Train/test split ratio: `0.75 / 0.25`

### 2.3 Metrics and Conventions
- Predictive: `test_rmse`, `coverage_90`
- Model comparison: `elpd_loo`, `delta_loo`
- Mixture complexity: `effective_k_mean`
- Convergence flag: `max_rhat < 1.05` (`hill_mmm/inference.py`)
- Delta significance: `|delta_loo| > 2 * se` (`hill_mmm/metrics.py`)

## 3. Main Results (Manuscript-Oriented Summary)

### 3.1 High-Level Outcome

| Domain | Runs | Converged | Non-converged RMSE winners | Main Risk Signal |
| --- | --- | --- | --- | --- |
| Synthetic | 80 | 53 (66.2%) | 7 / 20 seed-cells | 16 catastrophic LOO outliers (`elpd_loo < -2000`) |
| Real | 45 | 8 (17.8%) | 11 / 15 seed-cells | Extreme heavy tail (`max RMSE = 110293.055`) |

Interpretation:
- Predictive winners are not restricted to converged runs.
- However, real-data non-converged runs carry severe tail risk and can distort mean-based conclusions.

### 3.2 Synthetic Findings
Synthetic results indicate that mixture models can be competitive in predictive terms, but summary means are sensitive to unstable runs (especially for `mixture_k2` and `sparse_k5` in LOO-based metrics). For synthetic evaluation, a robust interpretation is:
- non-converged runs are not systematically useless,
- but convergence remains a meaningful reliability indicator,
- and robustness checks (median or filtered summaries) are necessary for fair comparison.

### 3.3 Real-Data Findings
Real-data results show the strongest tension between predictive utility and inferential reliability:
- Convergence rates are low for mixture models (`mixture_k2`: 13.3%, `mixture_k3`: 6.7%).
- Nevertheless, non-converged runs frequently win at the `(org, seed)` level in test RMSE.
- Robust aggregates tell a more stable story than means:
  - Median test RMSE: `mixture_k2` (100.912), `mixture_k3` (101.000), `single_hill` (126.527)
  - 10% trimmed mean test RMSE: `single_hill` (127.767), `mixture_k3` (143.558), `mixture_k2` (167.161)

This gap between median and trimmed-mean rankings indicates that model preference depends on tolerance for tail failures.

### 3.4 Recommended Narrative for the Paper
A manuscript-friendly framing is:
1. Mixture models show predictive promise, including in settings where strict MCMC convergence is difficult.
2. Predictive utility and inferential trustworthiness should be explicitly separated.
3. Claims about forecast performance should rely on robust summaries, while parameter interpretation should require stronger convergence evidence.

## 4. Discussion
The benchmark supports a pragmatic but rigorous position: non-convergence is not a binary rejection criterion for prediction, but it is a major warning for risk management and interpretability. This distinction is especially important in MMM, where business decisions may prioritize forecast quality, while scientific claims require stable posteriors.

## 5. Limitations
- Single benchmark timestamp (`20260207_061725`).
- Limited real-data breadth (5 organizations, 3 seeds).
- No hierarchical pooling across organizations in this report-level summary.

## 6. Reproducibility
- Script: `python scripts/run_benchmark.py`
- Outputs: `results/benchmark/`
- Config used: `results/benchmark/config.json`

---

## Appendix A. Detailed Quantitative Tables

### Table A1. Synthetic convergence rate by model

| model | convergence_rate |
| --- | --- |
| mixture_k2 | 30.0% |
| mixture_k3 | 80.0% |
| single_hill | 100.0% |
| sparse_k5 | 55.0% |

### Table A2. Synthetic catastrophic run counts by DGP and model (`elpd_loo < -2000`)

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

### Table A3. Synthetic means across all runs (including non-converged runs)

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

### Table A4. Synthetic means across converged runs only

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

### Table A5. Real convergence rate by model

| model | convergence_rate |
| --- | --- |
| mixture_k2 | 13.3% |
| mixture_k3 | 6.7% |
| single_hill | 33.3% |

### Table A6. Real mean metrics by model (all runs)

| model | elpd_loo | test_rmse | train_rmse | test_coverage_90 | time_seconds |
| --- | --- | --- | --- | --- | --- |
| mixture_k2 | -1.052e+07 | 192.886 | 160.654 | 88.9% | 1.883 |
| mixture_k3 | -2.585e+06 | 7478.114 | 8293.108 | 87.0% | 2.188 |
| single_hill | -4.719e+07 | 141.050 | 118.684 | 89.4% | 1.132 |

### Table A7. Real median metrics by model (all runs)

| model | elpd_loo | test_rmse | train_rmse | test_coverage_90 | time_seconds |
| --- | --- | --- | --- | --- | --- |
| mixture_k2 | -7328.036 | 100.912 | 95.238 | 91.8% | 1.873 |
| mixture_k3 | -7262.585 | 101.000 | 95.296 | 91.7% | 1.911 |
| single_hill | -1.559e+06 | 126.527 | 122.308 | 90.8% | 1.122 |

### Table A8. Real 10% trimmed mean of test RMSE

| model | test_rmse_trim10 |
| --- | --- |
| mixture_k2 | 167.161 |
| mixture_k3 | 143.558 |
| single_hill | 127.767 |

### Table A9. Real RMSE by convergence status and model

| model | conv_n | conv_mean_rmse | conv_median_rmse | nonconv_n | nonconv_mean_rmse | nonconv_median_rmse |
| --- | --- | --- | --- | --- | --- | --- |
| mixture_k2 | 2 | 40.606 | 40.606 | 13 | 216.314 | 140.303 |
| mixture_k3 | 1 | 12.407 | 12.407 | 14 | 8011.379 | 101.028 |
| single_hill | 5 | 170.353 | 98.386 | 10 | 126.398 | 126.792 |

### Table A10. Best model by organization (seed-mean, all runs)

| org_id | best_by_loo | best_loo | best_by_test_rmse | best_test_rmse |
| --- | --- | --- | --- | --- |
| 7059e30b528ed5f14ee9921de13248e5 | mixture_k2 | -5916.126 | single_hill | 55.020 |
| 72a86a208d24d68b80be0e44a8a4872d | mixture_k3 | -4768.377 | single_hill | 11.749 |
| 882ce7e286d66facc66518783e2192c7 | mixture_k3 | -1.087e+07 | mixture_k3 | 163.795 |
| ba773ebd7ec0a08f1d042187d086ccb4 | mixture_k3 | -2.034e+06 | single_hill | 340.703 |
| bfb6f6a326141ed6a751fc83ba836984 | mixture_k3 | -7262.358 | mixture_k3 | 101.018 |

### Table A11. Synthetic failures (top by R-hat)

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

### Table A12. Real-data failures (top by R-hat)

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
