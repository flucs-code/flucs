"""
Tests for the top-level FLUCS plugin lookup.
"""

import pytest

import flucs
from tests.support.test_systems import TEST_SYSTEMS

pytestmark = pytest.mark.core


@pytest.mark.parametrize(
    "system_spec",
    TEST_SYSTEMS.values(),
    ids=lambda system_spec: system_spec.system_name,
)
def test_all_test_systems_are_registered_for_test_session(system_spec):
    """
    All standalone test systems are available through normal lookup.
    """
    assert (
        flucs.get_system_type(system_spec.system_name)
        is system_spec.system_type
    )
