# Hill Mixture MMM Paper

## Overview

This folder contains the research paper documenting the Bayesian Hill Mixture Model for Marketing Mix Modeling implemented in this repository.

## Goal

Standard Marketing Mix Models assume all consumers respond identically to marketing spend. This paper proposes a **mixture of Hill saturation functions** to capture heterogeneous response patterns across latent consumer segments, addressing:

- **Segment heterogeneity**: Different consumer groups (heavy vs. light buyers, loyalists vs. switchers) exhibit different saturation curves
- **Model misspecification**: Aggregate response curves mask segment-specific behaviors
- **Identifiability**: Ordered constraints prevent label switching in mixture posteriors

## Key Contributions

1. **Bayesian mixture framework** combining Hill saturation with latent segment membership
2. **Automatic prior scaling** from training data for diverse scales
3. **Sparse Dirichlet priors** for automatic component pruning
4. **NumPyro/JAX implementation** achieving 2-20x faster sampling than TensorFlow alternatives

## Relationship to Code

| Paper Section | Implementation |
|---------------|----------------|
| Model Specification (§2) | `../hill_mixture_mmm.py` - `model_hill_mixture_v3()` |
| Data Generation (§3) | `../hill_mixture_mmm.py` - `generate_data_gmm_hill()` |
| Single Hill baseline | `../hill_mixture_mmm.py` - `model_single_hill_v3()` |
| Sparse mixture | `../hill_mixture_mmm.py` - `model_hill_mixture_sparse()` |

## Files

| File | Description |
|------|-------------|
| `hill_mixture_mmm_paper.md` | Paper source (Markdown + YAML frontmatter) |
| `references.bib` | BibTeX bibliography |
| `hill_mixture_mmm_paper.pdf` | Compiled output |

## Compilation

### Requirements

- [Pandoc](https://pandoc.org/installing.html) (2.11+)
- XeLaTeX (via TeX Live, MacTeX, or MiKTeX)

### Build PDF

    pandoc hill_mixture_mmm_paper.md -o hill_mixture_mmm_paper.pdf --pdf-engine=xelatex --citeproc --bibliography=references.bib
