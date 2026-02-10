import pytest


@pytest.fixture
def small_sample_size():
    """Small sample size for fast tests."""
    return 1000


@pytest.fixture
def medium_sample_size():
    """Medium sample size for balanced speed/accuracy."""
    return 10000


@pytest.fixture
def large_sample_size():
    """Large sample size for full tests."""
    return 50000
