# Experiment Implementation Plan for Paper

**Date**: 2026-02-06  
**Status**: Planning phase  
**Goal**: Define experiments and implementations needed for paper submission

---

## Executive Summary

Based on the validation summary (`docs/VALIDATION_SUMMARY_REPORT.md`), we have identified key findings that require updates to the paper experiments:

1. **K=2 mixture converges on real data** - This is a significant finding not in current paper
2. **K=3 does NOT converge on real data** - Paper currently recommends K=5 sparse, but K=3 issues need addressing
3. **Post-hoc relabeling is essential** - Current paper uses in-MCMC ordering which is suboptimal
4. **Label-invariant diagnostics needed** - Standard R-hat is inappropriate for mixture models

---

## 1. Paper Structure Gap Analysis

### Current Paper Content

| Section | Content | Status |
|---------|---------|--------|
| Introduction | Motivation for mixture models | Complete |
| Model Specification | Hill mixture formulation | Complete |
| Experiments | Synthetic data (K=1,2,3,5) | Needs update |
| Discussion | Recommends Sparse K=5 | Needs revision |

### Missing Content

| Gap | Priority | Action |
|-----|----------|--------|
| Real data validation | **High** | Add new section |
| K=2 model results | **High** | Add to comparison |
| Label-invariant diagnostics | **High** | Update methodology |
| Convergence best practices | Medium | Add to discussion |
| Post-hoc relabeling | Medium | Update model specification |

---

## 2. Experiment Plan

### 2.1 Synthetic Data Experiments (Update)

**Objective**: Demonstrate parameter recovery and model selection across DGP complexity levels.

#### Configuration

```yaml
DGPs:
  - single      # K=1 (null hypothesis)
  - mixture_k2  # K=2
  - mixture_k3  # K=3
  - mixture_k5  # K=5 (sparse discovery)

Models:
  - single_hill
  - mixture_k2      # NEW: Add K=2 model
  - mixture_k3
  - mixture_k5_sparse

MCMC:
  warmup: 1000
  samples: 2000
  chains: 4
  seeds: [0, 1, 2, 3, 4]  # 5 seeds for variance estimation

Data:
  T: 200
  train_ratio: 0.75
```

#### Metrics to Report

| Metric | Purpose |
|--------|----------|
| ELPD-LOO | Predictive accuracy (primary) |
| ELPD-WAIC | Alternative predictive metric |
| Test RMSE | Point prediction accuracy |
| 90% Coverage | Uncertainty calibration |
| Max R-hat (relabeled) | Convergence quality |
| Min ESS | Effective sample size |
| Effective K | Component recovery |
| Parameter recovery MSE | True vs estimated parameters |

#### Expected Results Table

| True K | Model | Expected ELPD-LOO | Expected Convergence |
|--------|-------|-------------------|---------------------|
| 1 | Single Hill | Best | Yes |
| 1 | Mixture K=2 | Slight penalty | Yes |
| 2 | Single Hill | Baseline | Yes |
| 2 | Mixture K=2 | **Best** | Yes |
| 3 | Single Hill | Baseline | Yes |
| 3 | Mixture K=3 | Best | Partial |
| 5 | Mixture K=5 sparse | Best | Partial |

### 2.2 Real Data Validation (NEW)

**Objective**: Demonstrate practical applicability on real marketing data.

#### Data Specification

```yaml
Organisation: ab0b8d655e7140272cab371a515e009a
Vertical: Food & Drink
T: 1180 observations
Channels: 8 (Google + Meta)
Date range: 2020-08-29 to 2023-11-21
```

#### Experiment Design

| Experiment | Models | MCMC Config | Purpose |
|------------|--------|-------------|---------|
| Baseline | Single Hill, K=2, K=3 | Standard (1k/2k/4) | Primary comparison |
| Extended | K=3 | Extended (2k/4k/6) | Convergence rescue |
| Cross-validation | K=2 | Standard | Holdout validation |

#### Expected Results

| Model | ELPD-LOO | Converged | Note |
|-------|----------|-----------|------|
| Single Hill | ~-5126 | Yes | Baseline |
| **Mixture K=2** | ~-4981 | **Yes** | Recommended |
| Mixture K=3 | ~-4954 | No | Not usable |

### 2.3 Convergence Diagnostic Comparison (NEW)

**Objective**: Justify label-invariant diagnostic methodology.

#### Experiment Design

```yaml
Data: Synthetic mixture_k3
Models:
  - constrained (in-MCMC ordering)
  - unconstrained + post-hoc relabeling

Diagnostics:
  - Standard R-hat
  - Rank-normalized R-hat  
  - Log-likelihood R-hat (label-invariant)
  - Relabeled component R-hat
  - Label switching rate
```

#### Expected Results

| Approach | Standard R-hat | Relabeled R-hat | Conclusion |
|----------|----------------|-----------------|------------|
| Constrained | ~1.8 | ~1.6 | Poor |
| **Unconstrained + Relabel** | ~1.7 | **~1.01** | **Recommended** |

### 2.4 Sensitivity Analysis

**Objective**: Assess robustness to hyperprior choices.

#### Hyperprior Comparison

| Prior | Parameters | Test On |
|-------|------------|----------|
| HalfNormal(1.0) | Baseline | Synthetic K=3 |
| HalfNormal(0.1) | Tight | Synthetic K=3 |
| InverseGamma(3,1) | Heavy-tailed | Synthetic K=3 |
| **LogNormal(-1,0.5)** | **Recommended** | Synthetic K=3, Real data |

---

## 3. Implementation Tasks

### 3.1 Model Updates

| Task | File | Priority | Status |
|------|------|----------|--------|
| Add `model_hill_mixture_k2` | `hill_mmm/models.py` | High | TODO |
| Switch to LogNormal hyperpriors | `hill_mmm/models.py` | High | TODO |
| Add unconstrained model as default | `hill_mmm/models.py` | High | Done |
| Update `compute_prior_config()` | `hill_mmm/data.py` | Medium | Review |

### 3.2 Diagnostic Updates

| Task | File | Priority | Status |
|------|------|----------|--------|
| Add relabeled R-hat to default output | `hill_mmm/inference.py` | High | Done |
| Integrate `check_label_switching()` | `hill_mmm/inference.py` | High | Done |
| Add log-likelihood R-hat | `hill_mmm/inference.py` | High | Done |
| Create comprehensive diagnostic report | `hill_mmm/inference.py` | Medium | Done |

### 3.3 Benchmark Infrastructure

| Task | File | Priority | Status |
|------|------|----------|--------|
| Create paper benchmark script | `scripts/run_paper_benchmarks.py` | High | TODO |
| Add real data loader | `hill_mmm/data.py` | High | TODO |
| Create figure generation script | `tests/test_visualization.py` | Medium | Partial |
| Add results export (CSV/JSON) | `hill_mmm/benchmark.py` | Medium | Partial |

### 3.4 Paper Updates

| Task | File | Priority |
|------|------|----------|
| Add K=2 to model comparison | `paper/main.tex` | High |
| Add real data validation section | `paper/main.tex` | High |
| Update convergence methodology | `paper/main.tex` | High |
| Revise recommendations | `paper/main.tex` | High |
| Update figures | `paper/figures/` | Medium |
| Add new references (Vehtari 2021) | `paper/references.bib` | Low |

---

## 4. Execution Timeline

### Phase 1: Infrastructure (1-2 days)

- [ ] Implement `model_hill_mixture_k2`
- [ ] Update hyperpriors to LogNormal
- [ ] Create `scripts/run_paper_benchmarks.py`
- [ ] Add real data loader

### Phase 2: Synthetic Experiments (2-3 days)

- [ ] Run full benchmark suite (4 DGPs × 4 models × 5 seeds)
- [ ] Generate convergence diagnostic comparison
- [ ] Run sensitivity analysis
- [ ] Export results to CSV/JSON

### Phase 3: Real Data Experiments (1-2 days)

- [ ] Run baseline comparison (Single, K=2, K=3)
- [ ] Run extended K=3 experiment (convergence rescue)
- [ ] Validate K=2 with cross-validation
- [ ] Document findings

### Phase 4: Paper Updates (2-3 days)

- [ ] Update experiment section with new results
- [ ] Add real data validation section
- [ ] Update figures
- [ ] Revise discussion and recommendations
- [ ] Proofread and finalize

---

## 5. Success Criteria

### Primary Criteria

| Criterion | Target |
|-----------|--------|
| K=2 converges on real data | R-hat ≤ 1.01 (relabeled) |
| K=2 improves over Single Hill | ΔELPD-LOO > 100 |
| Synthetic experiments complete | 80+ configurations |
| All figures updated | 8+ figures |

### Secondary Criteria

| Criterion | Target |
|-----------|--------|
| Paper compiles without errors | Yes |
| All code reproducible | Script-based |
| Results documented | JSON + CSV |

---

## 6. Risk Mitigation

| Risk | Mitigation |
|------|------------|
| K=2 doesn't generalize to other datasets | Test on multiple organizations |
| Extended MCMC still doesn't converge K=3 | Document limitation, recommend K=2 |
| Synthetic results don't match paper draft | Re-run with updated code |
| Computation time too long | Use quick mode for iteration |

---

## 7. File Reference

### Key Files to Modify

```
hill_mmm/
├── models.py          # Add K=2 model, update hyperpriors
├── inference.py       # Already updated with diagnostics
├── data.py           # Add real data loader
└── benchmark.py      # Update for paper experiments

scripts/
├── run_paper_benchmarks.py   # NEW: Main experiment runner
├── run_real_data_validation.py  # Update for K=2
└── test_ordering_comparison.py  # Already done

paper/
├── main.tex          # Update experiments, add real data
├── references.bib    # Add Vehtari 2021
└── figures/          # Regenerate with new results
```

### Output Files

```
results/
├── paper_benchmarks/
│   ├── synthetic_results.csv
│   ├── synthetic_results.json
│   ├── convergence_comparison.json
│   └── sensitivity_analysis.json
├── real_data_validation/
│   ├── validation_*.json  # Per-organization results
│   └── summary.csv
└── figures/
    ├── fig_elpd_comparison.png
    ├── fig_convergence_heatmap.png
    └── ...
```

---

## Document History

| Date | Author | Change |
|------|--------|--------|
| 2026-02-06 | Session 60a301b3 | Initial plan based on validation findings |
