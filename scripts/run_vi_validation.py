#!/usr/bin/env python
"""Real data validation using Variational Inference (VI).

Compares MCMC vs SVI (Stochastic Variational Inference) for mixture models.
VI is much faster but provides approximate posteriors.

Usage:
    python scripts/run_vi_validation.py
    python scripts/run_vi_validation.py --quick  # Faster settings for testing
    python scripts/run_vi_validation.py --org ORG_ID  # Specific organization
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoMultivariateNormal, AutoNormal
from numpyro.optim import Adam

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from hill_mmm.data_loader import (  # noqa: E402
    LoadedData,
    TimeSeriesConfig,
    list_timeseries,
    load_timeseries,
)
from hill_mmm.inference import (  # noqa: E402
    compute_convergence_diagnostics,
    compute_loo,
    run_inference,
)
from hill_mmm.models import model_hill_mixture_unconstrained, model_single_hill  # noqa: E402

# Constants
DATA_PATH = project_root / "data" / "conjura_mmm_data.csv"
RESULTS_DIR = project_root / "results" / "vi_validation"


def run_svi(
    model_fn,
    x: np.ndarray,
    y: np.ndarray,
    guide_type: str = "autonormal",
    num_steps: int = 10000,
    learning_rate: float = 0.01,
    seed: int = 42,
    **model_kwargs,
) -> dict:
    """Run Stochastic Variational Inference.

    Args:
        model_fn: NumPyro model function
        x: (T,) spend values
        y: (T,) response values
        guide_type: 'autonormal' or 'automvn'
        num_steps: Number of optimization steps
        learning_rate: Adam learning rate
        seed: Random seed
        **model_kwargs: Additional model arguments (e.g., K for mixture)

    Returns:
        Dict with guide, params, losses, and samples
    """
    rng_key = jax.random.PRNGKey(seed)

    # Select guide
    if guide_type == "autonormal":
        guide = AutoNormal(model_fn)
    elif guide_type == "automvn":
        guide = AutoMultivariateNormal(model_fn)
    else:
        raise ValueError(f"Unknown guide type: {guide_type}")

    # Setup SVI
    optimizer = Adam(learning_rate)
    svi = SVI(model_fn, guide, optimizer, loss=Trace_ELBO())

    # Run optimization
    x_jax = jnp.array(x)
    y_jax = jnp.array(y)

    svi_result = svi.run(
        rng_key,
        num_steps,
        x=x_jax,
        y=y_jax,
        progress_bar=True,
        **model_kwargs,
    )

    params = svi_result.params
    losses = svi_result.losses

    # Get posterior samples
    rng_key, sample_key = jax.random.split(rng_key)
    from numpyro.infer import Predictive

    # First, sample from the guide to get latent variables
    guide_predictive = Predictive(
        guide,
        params=params,
        num_samples=2000,
    )
    guide_samples = guide_predictive(sample_key, x=x_jax, y=y_jax, **model_kwargs)

    # Then, run those samples through the model to get deterministics
    rng_key, model_key = jax.random.split(rng_key)
    model_predictive = Predictive(
        model_fn,
        posterior_samples=guide_samples,
        return_sites=["A", "k", "n", "pis", "alpha", "sigma", "mu_expected"],
    )
    model_samples = model_predictive(model_key, x=x_jax, y=None, **model_kwargs)

    # Merge guide samples with model deterministics
    samples = {k: np.array(v) for k, v in guide_samples.items()}
    for k, v in model_samples.items():
        if k not in samples or np.isnan(samples.get(k, np.array([np.nan]))).all():
            samples[k] = np.array(v)

    return {
        "guide": guide,
        "params": params,
        "losses": np.array(losses),
        "samples": samples,
        "final_loss": float(losses[-1]),
    }


def compute_elbo_stats(losses: np.ndarray) -> dict:
    """Compute ELBO statistics from loss history.

    Convergence is determined by relative stability (coefficient of variation)
    rather than absolute std threshold, which is more appropriate when ELBO
    values are in the thousands.
    """
    final_losses = losses[-100:]
    final_mean = np.mean(final_losses)
    final_std = np.std(final_losses)

    # Use coefficient of variation (CV) - relative measure of stability
    # CV < 0.01 means std is less than 1% of mean
    cv = final_std / (np.abs(final_mean) + 1e-8)
    converged = cv < 0.01  # 1% relative stability

    return {
        "final_elbo": float(-losses[-1]),
        "min_loss": float(losses.min()),
        "final_loss": float(losses[-1]),
        "converged": bool(converged),
        "coefficient_of_variation": float(cv),
        "final_std": float(final_std),
        "loss_reduction": float(losses[0] - losses[-1]),
    }


def run_single_hill_vi(
    data: LoadedData,
    config: dict,
    seed: int = 42,
) -> dict:
    """Run single Hill model with VI."""
    print("\n" + "=" * 60)
    print("Running Single Hill Model (VI)")
    print("=" * 60)

    start_time = time.time()

    svi_result = run_svi(
        model_fn=model_single_hill,
        x=data.x,
        y=data.y,
        guide_type=config["guide_type"],
        num_steps=config["num_steps"],
        learning_rate=config["learning_rate"],
        seed=seed,
    )

    elapsed = time.time() - start_time
    print(f"VI completed in {elapsed:.1f} seconds")

    # Compute ELBO stats
    elbo_stats = compute_elbo_stats(svi_result["losses"])

    # Get parameter summaries
    samples = svi_result["samples"]
    param_summary = {
        "A_mean": float(np.mean(samples.get("A", [np.nan]))),
        "A_std": float(np.std(samples.get("A", [np.nan]))),
        "k_mean": float(np.mean(samples.get("k", [np.nan]))),
        "k_std": float(np.std(samples.get("k", [np.nan]))),
        "n_mean": float(np.mean(samples.get("n", [np.nan]))),
        "n_std": float(np.std(samples.get("n", [np.nan]))),
        "alpha_mean": float(np.mean(samples.get("alpha", [np.nan]))),
        "sigma_mean": float(np.mean(samples.get("sigma", [np.nan]))),
    }

    return {
        "model": "single_hill_vi",
        "method": "svi",
        "guide_type": config["guide_type"],
        "elapsed_seconds": elapsed,
        "elbo": elbo_stats,
        "params": param_summary,
        "num_steps": config["num_steps"],
    }


def reconstruct_mixture_params(samples: dict, K: int) -> dict:
    """Reconstruct A, k, n, pis from raw VI samples.

    The unconstrained model uses non-centered parameterization:
    - log_A_raw ~ Normal(0, 1) -> A = exp(mu_log_A + sigma_log_A * log_A_raw)
    - log_n_raw ~ Normal(0, 1) -> n = exp(mu_log_n + sigma_log_n * log_n_raw)
    - k is built from log_k_base + cumsum(log_k_increments_raw * sigma_log_k)
    - pis is computed from stick_proportions via stick-breaking

    AutoNormal guide only samples latent variables, not deterministics.
    """
    # Check if deterministic values are already available (e.g., from MCMC)
    if "A" in samples and not np.isnan(samples["A"]).all():
        return {
            "A": samples["A"],
            "k": samples["k"],
            "n": samples["n"],
            "pis": samples["pis"],
        }

    # Determine n_samples from any available array
    n_samples_vi = None
    for key in ["log_A_raw", "mu_log_A", "sigma", "alpha"]:
        if key in samples:
            arr = samples[key]
            n_samples_vi = arr.shape[0] if arr.ndim >= 1 else 1
            break
    if n_samples_vi is None:
        n_samples_vi = 1

    # Helper to ensure array has correct shape
    def ensure_shape(arr, default_shape):
        if arr is None:
            return np.full(default_shape, np.nan)
        arr = np.atleast_1d(arr)
        return arr

    # Reconstruct A from raw samples
    log_A_raw = ensure_shape(samples.get("log_A_raw"), (n_samples_vi, K))
    mu_log_A = ensure_shape(samples.get("mu_log_A"), (n_samples_vi,))
    sigma_log_A = ensure_shape(samples.get("sigma_log_A"), (n_samples_vi,))
    if log_A_raw.ndim == 1:
        log_A_raw = (
            log_A_raw.reshape(-1, K) if log_A_raw.size >= K else np.full((n_samples_vi, K), np.nan)
        )
    A = np.exp(mu_log_A[:, None] + sigma_log_A[:, None] * log_A_raw)

    # Reconstruct n from raw samples
    log_n_raw = ensure_shape(samples.get("log_n_raw"), (n_samples_vi, K))
    mu_log_n = ensure_shape(samples.get("mu_log_n"), (n_samples_vi,))
    sigma_log_n = ensure_shape(samples.get("sigma_log_n"), (n_samples_vi,))
    if log_n_raw.ndim == 1:
        log_n_raw = (
            log_n_raw.reshape(-1, K) if log_n_raw.size >= K else np.full((n_samples_vi, K), np.nan)
        )
    n = np.exp(mu_log_n[:, None] + sigma_log_n[:, None] * log_n_raw)

    # Reconstruct k from base + increments
    log_k_base = ensure_shape(samples.get("log_k_base"), (n_samples_vi,))
    sigma_log_k = ensure_shape(samples.get("sigma_log_k"), (n_samples_vi,))

    # Handle log_k_increments_raw carefully
    log_k_increments_raw = samples.get("log_k_increments_raw")
    if log_k_increments_raw is None:
        log_k_increments_raw = np.full((n_samples_vi, K - 1), np.nan)
    else:
        log_k_increments_raw = np.atleast_1d(log_k_increments_raw)
        if log_k_increments_raw.ndim == 1:
            # If 1D and K=2, shape should be (n_samples,) representing single increment
            log_k_increments_raw = log_k_increments_raw.reshape(-1, 1)

    # k[0] = exp(log_k_base), k[i] = k[0] * exp(cumsum(increments * sigma))
    n_k = log_k_base.shape[0]
    log_k = np.zeros((n_k, K))
    log_k[:, 0] = log_k_base
    if K > 1 and log_k_increments_raw.shape[0] == n_k:
        increments = log_k_increments_raw * sigma_log_k[:, None]
        log_k[:, 1:] = log_k_base[:, None] + np.cumsum(increments, axis=1)
    k = np.exp(log_k)

    # Reconstruct pis from stick_proportions (stick-breaking)
    stick_proportions = samples.get("stick_proportions")
    if stick_proportions is None:
        pis = np.full((n_samples_vi, K), np.nan)
    else:
        stick_proportions = np.atleast_1d(stick_proportions)
        if stick_proportions.ndim == 1:
            # For K=2, stick_proportions is shape (n_samples,) - single proportion
            stick_proportions = stick_proportions.reshape(-1, 1)

        n_pis = stick_proportions.shape[0]
        pis = np.zeros((n_pis, K))
        remaining = np.ones(n_pis)
        for i in range(K - 1):
            pis[:, i] = stick_proportions[:, i] * remaining
            remaining = remaining * (1 - stick_proportions[:, i])
        pis[:, K - 1] = remaining  # Last component gets the rest

    return {"A": A, "k": k, "n": n, "pis": pis}


def run_mixture_vi(
    data: LoadedData,
    config: dict,
    K: int = 2,
    seed: int = 42,
) -> dict:
    """Run mixture model with VI."""
    print("\n" + "=" * 60)
    print(f"Running Mixture Model K={K} (VI)")
    print("=" * 60)

    start_time = time.time()

    svi_result = run_svi(
        model_fn=model_hill_mixture_unconstrained,
        x=data.x,
        y=data.y,
        guide_type=config["guide_type"],
        num_steps=config["num_steps"],
        learning_rate=config["learning_rate"],
        seed=seed,
        K=K,
    )

    elapsed = time.time() - start_time
    print(f"VI completed in {elapsed:.1f} seconds")

    # Compute ELBO stats
    elbo_stats = compute_elbo_stats(svi_result["losses"])

    # Get parameter summaries
    samples = svi_result["samples"]

    # Reconstruct parameters from raw samples (VI guide only samples latents)
    reconstructed = reconstruct_mixture_params(samples, K)
    A_samples = reconstructed["A"]
    k_samples = reconstructed["k"]
    n_samples = reconstructed["n"]
    pis_samples = reconstructed["pis"]

    param_summary = {
        "A_means": [float(x) for x in np.mean(A_samples, axis=0)]
        if A_samples.ndim > 1
        else [float(np.mean(A_samples))],
        "k_means": [float(x) for x in np.mean(k_samples, axis=0)]
        if k_samples.ndim > 1
        else [float(np.mean(k_samples))],
        "n_means": [float(x) for x in np.mean(n_samples, axis=0)]
        if n_samples.ndim > 1
        else [float(np.mean(n_samples))],
        "pis_means": [float(x) for x in np.mean(pis_samples, axis=0)]
        if pis_samples.ndim > 1
        else [float(np.mean(pis_samples))],
        "alpha_mean": float(np.mean(samples.get("alpha", [np.nan]))),
        "sigma_mean": float(np.mean(samples.get("sigma", [np.nan]))),
    }

    return {
        "model": f"mixture_k{K}_vi",
        "method": "svi",
        "guide_type": config["guide_type"],
        "K": K,
        "elapsed_seconds": elapsed,
        "elbo": elbo_stats,
        "params": param_summary,
        "num_steps": config["num_steps"],
    }


def run_single_hill_mcmc(
    data: LoadedData,
    config: dict,
    seed: int = 42,
) -> dict:
    """Run single Hill model with MCMC (for comparison)."""
    print("\n" + "=" * 60)
    print("Running Single Hill Model (MCMC)")
    print("=" * 60)

    start_time = time.time()

    mcmc = run_inference(
        model_fn=model_single_hill,
        x=data.x,
        y=data.y,
        seed=seed,
        num_warmup=config["num_warmup"],
        num_samples=config["num_samples"],
        num_chains=config["num_chains"],
    )

    elapsed = time.time() - start_time
    print(f"MCMC completed in {elapsed:.1f} seconds")

    # Compute LOO
    print("Computing LOO-CV...")
    loo_results = compute_loo(mcmc)

    # Compute convergence diagnostics
    print("Computing convergence diagnostics...")
    conv_results = compute_convergence_diagnostics(mcmc)

    # Get parameter summaries
    samples = mcmc.get_samples()
    param_summary = {
        "A_mean": float(np.mean(samples["A"])),
        "A_std": float(np.std(samples["A"])),
        "k_mean": float(np.mean(samples["k"])),
        "k_std": float(np.std(samples["k"])),
        "n_mean": float(np.mean(samples["n"])),
        "n_std": float(np.std(samples["n"])),
        "alpha_mean": float(np.mean(samples["alpha"])),
        "sigma_mean": float(np.mean(samples["sigma"])),
    }

    return {
        "model": "single_hill_mcmc",
        "method": "mcmc",
        "elapsed_seconds": elapsed,
        "loo": loo_results,
        "convergence": conv_results,
        "params": param_summary,
    }


def format_results_table(vi_single: dict, vi_mixture: dict, mcmc_single: dict | None = None) -> str:
    """Format comparison table."""
    lines = []
    lines.append("\n" + "=" * 70)
    lines.append("COMPARISON SUMMARY: VI vs MCMC")
    lines.append("=" * 70)

    # ELBO comparison
    lines.append("\n### VI Results (ELBO)")
    lines.append("-" * 50)
    lines.append(f"{'Model':<30} {'Final ELBO':>15} {'Converged':>12}")
    lines.append("-" * 57)

    lines.append(
        f"{'Single Hill (VI)':<30} {vi_single['elbo']['final_elbo']:>15.1f} "
        f"{'Yes' if vi_single['elbo']['converged'] else 'No':>12}"
    )
    lines.append(
        f"{vi_mixture['model']:<30} {vi_mixture['elbo']['final_elbo']:>15.1f} "
        f"{'Yes' if vi_mixture['elbo']['converged'] else 'No':>12}"
    )

    # MCMC comparison if available
    if mcmc_single:
        lines.append("\n### MCMC Results (ELPD-LOO)")
        lines.append("-" * 50)
        mcmc_elpd = mcmc_single["loo"].get("elpd_loo", np.nan)
        mcmc_conv = mcmc_single["convergence"].get("converged", False)
        lines.append(
            f"{'Single Hill (MCMC)':<30} {mcmc_elpd:>15.1f} {'Yes' if mcmc_conv else 'No':>12}"
        )

    # Timing comparison
    lines.append("\n### Computation Time")
    lines.append("-" * 50)
    lines.append(f"Single Hill (VI):  {vi_single['elapsed_seconds']:>8.1f} seconds")
    lines.append(
        f"Mixture K={vi_mixture['K']} (VI):    {vi_mixture['elapsed_seconds']:>8.1f} seconds"
    )
    if mcmc_single:
        lines.append(f"Single Hill (MCMC): {mcmc_single['elapsed_seconds']:>8.1f} seconds")
        speedup = mcmc_single["elapsed_seconds"] / vi_single["elapsed_seconds"]
        lines.append(f"\nVI Speedup: {speedup:.1f}x faster than MCMC")

    # Parameter comparison
    lines.append("\n### Parameter Estimates")
    lines.append("-" * 50)
    lines.append("Single Hill (VI):")
    lines.append(f"  A = {vi_single['params']['A_mean']:.1f} ± {vi_single['params']['A_std']:.1f}")
    lines.append(f"  k = {vi_single['params']['k_mean']:.1f} ± {vi_single['params']['k_std']:.1f}")
    lines.append(f"  n = {vi_single['params']['n_mean']:.2f} ± {vi_single['params']['n_std']:.2f}")

    if mcmc_single:
        lines.append("\nSingle Hill (MCMC):")
        lines.append(
            f"  A = {mcmc_single['params']['A_mean']:.1f} ± {mcmc_single['params']['A_std']:.1f}"
        )
        lines.append(
            f"  k = {mcmc_single['params']['k_mean']:.1f} ± {mcmc_single['params']['k_std']:.1f}"
        )
        lines.append(
            f"  n = {mcmc_single['params']['n_mean']:.2f} ± {mcmc_single['params']['n_std']:.2f}"
        )

    lines.append(f"\nMixture K={vi_mixture['K']} (VI):")
    for i, (A, k, n, pi) in enumerate(
        zip(
            vi_mixture["params"]["A_means"],
            vi_mixture["params"]["k_means"],
            vi_mixture["params"]["n_means"],
            vi_mixture["params"]["pis_means"],
        )
    ):
        lines.append(f"  Component {i + 1}: A={A:.1f}, k={k:.1f}, n={n:.2f}, pi={pi:.3f}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="VI validation for MMM models")
    parser.add_argument("--quick", action="store_true", help="Use quick settings for testing")
    parser.add_argument("--org", type=str, help="Specific organisation_id to use")
    parser.add_argument("--K", type=int, default=2, help="Number of mixture components")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--skip-mcmc", action="store_true", help="Skip MCMC comparison")
    parser.add_argument(
        "--guide",
        type=str,
        default="autonormal",
        choices=["autonormal", "automvn"],
        help="Guide type for VI",
    )
    args = parser.parse_args()

    # VI configuration
    if args.quick:
        vi_config = {
            "num_steps": 2000,
            "learning_rate": 0.01,
            "guide_type": args.guide,
        }
        mcmc_config = {
            "num_warmup": 200,
            "num_samples": 400,
            "num_chains": 2,
        }
        print("Using QUICK settings (for testing only)")
    else:
        vi_config = {
            "num_steps": 20000,
            "learning_rate": 0.005,
            "guide_type": args.guide,
        }
        mcmc_config = {
            "num_warmup": 1000,
            "num_samples": 2000,
            "num_chains": 4,
        }
        print("Using FULL settings")

    print(
        f"VI config: steps={vi_config['num_steps']}, lr={vi_config['learning_rate']}, guide={vi_config['guide_type']}"
    )

    # Check data exists
    if not DATA_PATH.exists():
        print(f"ERROR: Data file not found: {DATA_PATH}")
        print("Please ensure conjura_mmm_data.csv is in the data/ directory.")
        sys.exit(1)

    # Select organization
    if args.org:
        org_id = args.org
        print(f"\nUsing specified organisation: {org_id}")
    else:
        print("\nDiscovering available time series...")
        ts_info = list_timeseries(DATA_PATH, min_length=200)
        print(f"Found {len(ts_info)} time series with >= 200 days")

        if len(ts_info) == 0:
            print("ERROR: No suitable time series found")
            sys.exit(1)

        ts_info_sorted = ts_info.sort_values(
            by=["n_active_channels", "n_days"], ascending=[False, False]
        )
        org_id = ts_info_sorted.iloc[0]["organisation_id"]
        org_info = ts_info_sorted.iloc[0]
        print(f"\nSelected organisation: {org_id}")
        print(f"  - Days: {org_info['n_days']}")
        print(f"  - Active channels: {org_info['n_active_channels']}")

    # Load data
    print(f"\nLoading data for {org_id}...")
    data = load_timeseries(
        DATA_PATH,
        TimeSeriesConfig(organisation_id=org_id, aggregate_spend=True),
    )
    print(f"Loaded: T={data.meta['T']}, spend range=[{data.x.min():.0f}, {data.x.max():.0f}]")

    # Run VI models
    vi_single = run_single_hill_vi(data, vi_config, seed=args.seed)
    vi_mixture = run_mixture_vi(data, vi_config, K=args.K, seed=args.seed)

    # Optionally run MCMC for comparison
    mcmc_single = None
    if not args.skip_mcmc:
        mcmc_single = run_single_hill_mcmc(data, mcmc_config, seed=args.seed)

    # Print comparison
    summary = format_results_table(vi_single, vi_mixture, mcmc_single)
    print(summary)

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    results = {
        "timestamp": timestamp,
        "vi_config": vi_config,
        "mcmc_config": mcmc_config if not args.skip_mcmc else None,
        "organisation_id": org_id,
        "data_meta": data.meta,
        "vi_single": vi_single,
        "vi_mixture": vi_mixture,
        "mcmc_single": mcmc_single,
    }

    # Save JSON
    json_path = RESULTS_DIR / f"vi_validation_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {json_path}")

    # Save summary
    txt_path = RESULTS_DIR / f"vi_validation_{timestamp}.txt"
    with open(txt_path, "w") as f:
        f.write("VI Validation Results\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Organisation: {org_id}\n")
        f.write(f"Data: T={data.meta['T']}\n")
        f.write(summary)
    print(f"Summary saved to: {txt_path}")


if __name__ == "__main__":
    main()
