"""Pytest configuration for hill_mixture_mmm tests."""

from pathlib import Path

import pytest

FIGURES_DIR = Path(__file__).parent.parent / "paper" / "figures"
BENCHMARK_ARTIFACTS_DIR = FIGURES_DIR


def pytest_configure(config):
    """Add custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m not slow')")


@pytest.fixture
def output_dir() -> Path:
    """Return the paper figures directory for saving visualization outputs."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    return FIGURES_DIR


@pytest.fixture
def benchmark_output_root() -> Path:
    """Return the paper-submodule figure directory for benchmark test visualizations."""
    BENCHMARK_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    return BENCHMARK_ARTIFACTS_DIR
