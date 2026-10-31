"""
Input freezing and configuration precedence.
"""

import pytest

from flucs.input import FlucsInput

pytestmark = pytest.mark.cpu

################################################################################
# Helpers and fixtures
################################################################################

@pytest.fixture
def input_path(tmp_path):
    """
    Write minimal input for freezing and precedence checks.
    """
    input_path = tmp_path / "input.toml"
    input_path.write_text(
        """\
        [setup]
        solver = "FourierSolver"
        system = "TestFourierSystem"

        [time]
        dt_max = 0.125
        """,
        encoding="utf-8",
    )
    return input_path

################################################################################
# Tests
################################################################################

# ------------------------------------------------------------------------------
# Input freezing
# ------------------------------------------------------------------------------

def test_input_is_read_only(test_fourier_system, input_path):
    """
    Reject assignment after initialization without changing the input.
    """
    flucs_input = FlucsInput(input_path)
    original_dt = flucs_input["time.dt_max"]

    with pytest.raises(RuntimeError, match="read-only"):
        flucs_input["time.dt_max"] = 0.25

    assert flucs_input["time.dt_max"] == original_dt

# ------------------------------------------------------------------------------
# Configuration precedence
# ------------------------------------------------------------------------------

def test_input_precedence(test_fourier_system, input_path):
    """
    File values replace defaults, and explicit overrides take precedence.
    """
    flucs_input = FlucsInput(input_path)

    # An omitted parameter retains its default; the file replaces dt_max=0.01.
    assert flucs_input["time.tfinal"] == 10.0
    assert flucs_input["time.dt_max"] == 0.125

    overridden_input = FlucsInput(
        input_path,
        override=["time.dt_max", 0.25],
    )

    # The explicit override replaces the file value without affecting defaults.
    assert overridden_input["time.tfinal"] == 10.0
    assert overridden_input["time.dt_max"] == 0.25
