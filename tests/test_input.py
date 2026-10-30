"""
Input freezing and configuration precedence.
"""

import pathlib as pl

import pytest

from flucs.input import FlucsInput

pytestmark = pytest.mark.cpu

INPUT_PATH = pl.Path(__file__).parent / "inputs" / "input.toml"

################################################################################
# Tests
################################################################################

# ------------------------------------------------------------------------------
# Input freezing
# ------------------------------------------------------------------------------

def test_input_is_read_only(test_fourier_system):
    """
    Reject assignment after initialization without changing the input.
    """
    flucs_input = FlucsInput(INPUT_PATH)
    original_dt = flucs_input["time.dt_max"]

    with pytest.raises(RuntimeError, match="read-only"):
        flucs_input["time.dt_max"] = 0.25

    assert flucs_input["time.dt_max"] == original_dt

# ------------------------------------------------------------------------------
# Configuration precedence
# ------------------------------------------------------------------------------

def test_input_precedence(test_fourier_system):
    """
    File values replace defaults, and explicit overrides take precedence.
    """
    flucs_input = FlucsInput(INPUT_PATH)

    # An omitted parameter retains its default; the file replaces dt_max=0.01.
    assert flucs_input["time.tfinal"] == 10.0
    assert flucs_input["time.dt_max"] == 0.125

    overridden_input = FlucsInput(
        INPUT_PATH,
        override=["time.dt_max", 0.25],
    )

    # The explicit override replaces the file value without affecting defaults.
    assert overridden_input["time.tfinal"] == 10.0
    assert overridden_input["time.dt_max"] == 0.25
