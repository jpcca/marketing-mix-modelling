"""Pytest configuration for hill_mmm tests."""

from pathlib import Path

import pytest

FIGURES_DIR = Path(__file__).parent.parent / "paper" / "figures"


def pytest_configure(config):
    """Add custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m not slow')")


@pytest.fixture
def output_dir() -> Path:
    """Return the paper figures directory for saving visualization outputs."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    return FIGURES_DIR
