"""Core mathematical transforms for Hill Mixture MMM.

This module contains the fundamental transformations:
- Adstock: Geometric decay for carryover effects
- Hill: Saturation function for diminishing returns
"""

from jax import lax


def adstock_geometric(x, alpha):
    """Geometric decay adstock transformation.

    Models how marketing effects carry over time:
        s_t = x_t + alpha * s_{t-1}

    Args:
        x: (T,) array of spend values
        alpha: Decay rate in [0, 1]. 0 = no carryover, 1 = full persistence

    Returns:
        (T,) array of adstocked spend
    """

    def step(carry, x_t):
        carry = x_t + alpha * carry
        return carry, carry

    _, s = lax.scan(step, 0.0, x)
    return s


def hill(x, A, k, n):
    """Hill saturation function.

    Models diminishing returns:
        y = A * x^n / (k^n + x^n)

    Args:
        x: Input (spend or adstocked spend)
        A: Maximum effect (asymptote)
        k: Half-saturation point (spend level at 50% of max effect)
        n: Hill coefficient (controls steepness)

    Returns:
        Saturated effect value
    """
    return A * (x**n) / (k**n + x**n + 1e-12)


def hill_matrix(s, A, k, n):
    """Compute Hill values for all mixture components.

    Args:
        s: (T,) adstocked spend
        A: (K,) max effect per component
        k: (K,) half-saturation per component
        n: (K,) hill coefficient per component

    Returns:
        (T, K) hill effect matrix
    """
    s_col = s[:, None]  # (T, 1)
    return (
        A[None, :]
        * (s_col ** n[None, :])
        / (k[None, :] ** n[None, :] + s_col ** n[None, :] + 1e-12)
    )
