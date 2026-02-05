# Label Switching and Convergence Diagnostics for Mixture Models

**Date**: 2026-02-05  
**Status**: Investigation completed, improved diagnostics recommended

---

## 1. Executive Summary

This document reports findings from an investigation into **label switching handling strategies** in Bayesian mixture models for Marketing Mix Modeling. The key discovery is that **standard convergence diagnostics (R-hat, ESS) are fundamentally inappropriate for mixture models** due to label switching effects.

### Key Conclusions

1. **Standard R-hat is unreliable** for mixture model convergence assessment
2. **Both ordering approaches** (in-MCMC constraint vs post-hoc relabeling) show similar predictive performance
3. **Improved diagnostics required** before drawing conclusions about convergence

---

## 2. Background: The Label Switching Problem

### What is Label Switching?

In mixture models with K components, the likelihood is invariant to permutations of component labels. For K=3 components, there are K!=6 equivalent posterior modes representing the same model.

```
Component assignment: [A, B, C] ≡ [B, A, C] ≡ [C, B, A] ≡ ...
```

### Why It Matters

- MCMC chains may visit different label permutations
- This creates apparent "disagreement" between chains
- Standard diagnostics misinterpret this as non-convergence
- Parameter posteriors become meaningless without consistent labeling

---

## 3. Two Approaches Investigated

### Approach A: In-MCMC Ordering Constraint (Current Implementation)

**Method**: Enforce `k[0] < k[1] < k[2]` during sampling via cumulative sum transformation:

```python
# In model_hill_mixture
k_increments = numpyro.sample("k_increments", dist.LogNormal(...))
k = jnp.cumsum(jnp.abs(k_increments))  # Ensures k[0] < k[1] < k[2]
```

**Theoretical Concern**: Does `abs()` transformation break the Markov property or violate detailed balance?

### Approach B: Unconstrained + Post-Hoc Relabeling

**Method**: 
1. Sample k values independently without ordering constraint
2. After MCMC, sort each sample by k values
3. Apply same permutation to all component parameters

**Implementation**:
- New model: `model_hill_mixture_unconstrained` in `hill_mmm/models.py`
- Relabeling function: `relabel_samples_by_k()` in `hill_mmm/inference.py`

```python
def relabel_samples_by_k(samples: dict) -> dict:
    """Sort components by k values for each MCMC sample."""
    # For each sample, get permutation that sorts k
    # Apply same permutation to A, k, n, pis
    ...
```

---

## 4. Experimental Results

### Configuration

| Setting | Value |
|---------|-------|
| Data | Synthetic `mixture_k3`, T=200 |
| Warmup | 1000 |
| Samples | 2000 |
| Chains | 4 |

### Results Comparison

| Metric | Constrained | Unconstrained + Relabel | Notes |
|--------|-------------|-------------------------|-------|
| Max R-hat | 1.82 | 2.12 | Both "fail" standard threshold |
| Min ESS | 6 | 5 | Both very low |
| RMSE | 7.45 | 7.44 | Nearly identical |
| 90% Coverage | 92.5% | 92.0% | Both good |
| Time (sec) | 108.4 | 94.5 | Unconstrained faster |
| **Converged** | **No** | **No** | By standard criteria |

### K Values Analysis

```
Constrained k means:    [7.79, 11.76, 19.00]  - ordered by construction
Unconstrained raw k:    [12.06, 11.30, 9.44]  - NOT ordered (label switching!)
Relabeled k means:      [6.83, 10.15, 15.83]  - properly ordered after relabeling
```

### Interpretation

- **Predictive performance is equivalent** (RMSE, coverage nearly identical)
- **Both show "non-convergence"** by R-hat/ESS standards
- **This may be a diagnostic artifact**, not actual non-convergence

---

## 5. Critical Discovery: Convergence Diagnostics Problem

### Standard R-hat is Fundamentally Unreliable for Mixture Models

From Stan User's Guide and academic literature:

> "The R-hat convergence statistic and the computation of effective sample size are both compromised by label switching. The problem is that the posterior mean is affected by label switching, resulting in meaningless values."

### Why Standard Diagnostics Fail

| Issue | Explanation |
|-------|-------------|
| Inflated between-chain variance | Chains visiting different permuted modes appear to "disagree" |
| Underestimated ESS | Apparent "jumps" between modes look like poor mixing |
| Meaningless posterior means | Averaging across permuted labels produces nonsense |
| K! equivalent modes | With K=3, chains explore 6 equivalent posterior regions |

### Example

Chain 1 visits mode: `k = [5, 10, 15]`  
Chain 2 visits mode: `k = [10, 15, 5]` (permuted)  
→ R-hat detects "disagreement" → High R-hat → False "non-convergence"

---

## 6. Recommended Solutions

### 6.1 Use Label-Invariant Quantities for Convergence

Evaluate R-hat on quantities that don't depend on component labeling:

| Quantity | Formula | Why It Works |
|----------|---------|---------------|
| Log-likelihood | `log p(y|θ)` | Same regardless of label permutation |
| Posterior predictive | `p(y_new|y)` | Marginalizes over all permutations |
| Mixture density | `Σ πₖ f(x|θₖ)` | Summation is permutation-invariant |

### 6.2 Use Rank-Normalized R-hat with Stricter Threshold

From Vehtari et al. (2021):

```python
# Use ArviZ with rank normalization
az.rhat(samples, method="rank")

# Stricter threshold
if max_rhat <= 1.01:  # Not 1.05!
    print("Converged")
```

### 6.3 Compute R-hat on Relabeled Parameters

```python
# After relabeling
relabeled = relabel_samples_by_k(samples)
rhat_relabeled = az.rhat(relabeled)
```

### 6.4 Target Higher ESS

| Context | Minimum ESS |
|---------|-------------|
| Standard models | 100 |
| Mixture models | **400** (recommended) |

---

## 7. Implementation Status

### Completed

| Item | Location |
|------|----------|
| Unconstrained model variant | `hill_mmm/models.py::model_hill_mixture_unconstrained` |
| Post-hoc relabeling function | `hill_mmm/inference.py::relabel_samples_by_k()` |
| Comparison test script | `scripts/test_ordering_comparison.py` |
| Experiment results | `results/ordering_comparison/` |

### Pending Implementation

| Item | Priority | Notes |
|------|----------|-------|
| Log-likelihood R-hat computation | High | Label-invariant diagnostic |
| Rank-normalized R-hat | High | Vehtari et al. (2021) |
| R-hat on relabeled samples | Medium | Additional diagnostic |
| Posterior predictive checks | Medium | Alternative validation |

---

## 8. Next Steps

### Immediate Actions

1. **Implement improved convergence diagnostics**
   - Add log-likelihood per-sample computation
   - Compute R-hat on log-likelihood (label-invariant)
   - Add rank-normalized R-hat option

2. **Re-run comparison experiment** with improved diagnostics
   - Determine if apparent non-convergence is real or artifact
   - Compare diagnostic results between approaches

### Longer-Term Considerations

| Approach | Rationale |
|----------|----------|
| LOO-CV for model comparison | Label-invariant model selection |
| Longer chains | If true non-convergence confirmed |
| Posterior predictive checks | Complementary validation |

---

## 9. Key References

1. **Stan User's Guide** - "Label Switching in Mixture Models" section
2. **Vehtari, Gelman, Simpson, Carpenter, Bürkner (2021)** - "Rank-Normalization, Folding, and Localization: An Improved R̂ for Assessing Convergence of MCMC" Bayesian Analysis
3. **Stephens (2000)** - "Dealing with label switching in mixture models" JRSS-B
4. **Jasra, Holmes, Stephens (2005)** - "Markov Chain Monte Carlo Methods and the Label Switching Problem in Bayesian Mixture Modeling" Statistical Science

---

## 10. Files Reference

### Modified Files

| File | Changes |
|------|--------|
| `hill_mmm/models.py` | Added `model_hill_mixture_unconstrained` |
| `hill_mmm/inference.py` | Added `relabel_samples_by_k()` |

### New Files

| File | Purpose |
|------|--------|
| `scripts/test_ordering_comparison.py` | Comparison experiment runner |
| `results/ordering_comparison/results.txt` | Human-readable results |
| `results/ordering_comparison/results.json` | Machine-readable results |
| `results/ordering_comparison/results.csv` | Tabular results |

---

## 11. Contact / Questions

For questions about this investigation:
- Review `hill_mmm/inference.py` for relabeling implementation
- Review `scripts/test_ordering_comparison.py` for experiment methodology
- Check `results/ordering_comparison/results.txt` for detailed experimental output
