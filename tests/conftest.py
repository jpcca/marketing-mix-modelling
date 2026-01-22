"""Pytest configuration for hill_mmm tests."""


def pytest_configure(config):
    """Add custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m not slow')")
