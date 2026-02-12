# PyData submission

## Proposal title
Bayesian Mixture-of-Hills for Marketing Mix Modelling: A NumPyro Case Study in AI-Augmented Research

## Brief Summary
Marketing mix models typically assume a single consumer response curve per channel, but real audiences are heterogeneous. This talk introduces a Bayesian mixture-of-Hill-curves approach implemented in NumPyro and JAX that captures latent consumer segments with distinct saturation behaviours. We also share how the model was developed through an AI-augmented research workflow, offering a reproducible template for cross-lingual academic-industry collaboration.

## Description

### What and Why
Marketing mix models (MMMs) help companies answer the question: "How much should we spend on each advertising channel?" A core component is the response curve—a function describing how sales respond to increased ad spend. Traditional MMMs fit a single response curve per channel, implicitly assuming all consumers react identically to advertising.

In reality, different consumer segments respond differently. Some convert quickly with minimal exposure; others require sustained investment before responding. When this heterogeneity exists, a single-curve assumption leads to miscalibrated uncertainty and suboptimal budget decisions.

Our approach addresses this by fitting a mixture of response curves—multiple curves per channel, each capturing a distinct consumer segment. The model automatically learns how many segments exist and their relative sizes, using Bayesian inference to quantify uncertainty. Technical innovations (stick-breaking priors, ordering constraints, hierarchical structure) ensure the model remains identifiable and samples efficiently.

### Validation Results
We validate the approach on synthetic datasets that simulate audiences with varying degrees of heterogeneity—from uniform response (one segment) to highly diverse (five distinct segments). When the audience is genuinely heterogeneous, mixture models achieve substantial improvements in predictive uncertainty, while incurring only modest penalties when the audience is actually homogeneous. Interestingly, point prediction accuracy remains similar across models; the gains are in uncertainty calibration—knowing how confident to be in predictions—which is critical for budget-allocation decisions. In practice, we recommend specifying more mixture components than expected, as the model automatically "prunes" unnecessary ones while maintaining stable inference.

We further evaluate the approach on a real-world marketing mix dataset, where the mixture model identifies previously hidden heterogeneous consumer populations—demonstrating that the method surfaces actionable audience structure from observational data.

### AI-Augmented Research Workflow
The second thread of the talk addresses how this model was developed. The project paired a supervisor and student working across different native languages, using LLM-processed meeting transcripts to maintain shared context and formalising requirements as executable pytest suites and MCMC diagnostic thresholds. This created a tight loop: human intent expressed in natural language, LLM-assisted code generation, and quantitative validation via predictive accuracy checks and convergence diagnostics. We share this workflow as a reusable pattern for teams combining domain expertise with probabilistic programming.

### Who Should Attend
This talk is aimed at data scientists, marketing analysts, and research engineers who work with marketing mix models or Bayesian inference. It is also relevant for anyone interested in structuring AI-augmented research workflows for multilingual or cross-sector collaboration.

### Talk Type and Tone
This is a technical talk with moderate mathematical depth. We present model specifications, probabilistic programming concepts, and benchmark results, but focus on intuition and practical applicability rather than proofs. The tone is informative and practical, with emphasis on reproducibility and real-world applicability.

### Background Knowledge
No prior experience with marketing mix modelling is required. Familiarity with Bayesian inference concepts (priors, MCMC, model comparison) is helpful but not essential—key concepts will be briefly introduced.

### Outline (40 minutes including Q&A)
- (0-5 min) The budget-allocation problem and why single-curve MMMs fall short
- (5-15 min) Mixture-of-Hills model specification: Hill transforms, adstock, stick-breaking weights, identifiability constraints
- (15-22 min) Benchmark design and results: synthetic DGPs, predictive accuracy vs uncertainty calibration, when mixtures help and when they don't, real-world dataset results
- (22-30 min) AI-augmented research workflow: transcripts, test-driven iteration, cross-lingual collaboration
- (30-40 min) Q&A

### Key Takeaways
1. **A practical Bayesian approach** to capture heterogeneous consumer response in marketing mix models
2. **When and why** mixture models outperform single-curve baselines (and when they don't)
3. **A reproducible workflow pattern** for AI-augmented probabilistic programming research
4. **An open-source NumPyro/JAX implementation** attendees can apply to their own projects

---
https://pretalx.com/pydata-london-2026-2025/cfp
