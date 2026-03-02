"""Setup shared by all test files."""

from pathlib import Path

import pytest


def pytest_configure(config):
    """Add markers that can be used to skip tests depending on certain
    conditions, such as missing dependencies."""
    config.addinivalue_line(
        "markers", "fluid_itg: mark test as requiring flucs_fluid_itg plugin."
    )


def pytest_collection_modifyitems(config, items):
    """Skip tests marked with 'fluid_itg' if the flucs_fluid_itg plugin is not
    installed."""
    try:
        import flucs_fluid_itg  # noqa: F401
    except ImportError:
        skip_fluid_itg = pytest.mark.skip(
            reason="need flucs_fluid_itg plugin to run"
        )
        for item in items:
            if "fluid_itg" in item.keywords:
                item.add_marker(skip_fluid_itg)


@pytest.fixture(scope="session")
def testdata() -> Path:
    """Path to the test data directory."""
    return Path(__file__).parent / "__testdata__"
