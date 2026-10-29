"""
Shared pytest support and optional test-system lookup.
"""

import pytest

import flucs

from .support.fourier import TestFourierSystem

pytest_plugins = ["flucs.testing.pytest_plugin"]


@pytest.fixture(scope="module")
def test_fourier_system():
    """
    Make TestFourierSystem available through normal input construction.
    """
    get_system_type = flucs.get_system_type

    def get_test_system_type(system_name):
        if system_name == "TestFourierSystem":
            return TestFourierSystem
        return get_system_type(system_name)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(flucs, "get_system_type", get_test_system_type)
        yield TestFourierSystem
