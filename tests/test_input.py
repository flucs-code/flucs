"""
Tests for loading and validating FLUCS input files.
"""

import pytest
import toml

import flucs
from flucs.input import FlucsInput
from tests.support.support import SINGLE_PRECISION

pytestmark = pytest.mark.core


def _write_input(
    input_path,
    test_system,
    precision=SINGLE_PRECISION,
    updates=None,
):
    """
    Write the smallest useful input for a standalone test system.
    """

    input_data = test_system.create_input_data()
    input_data["setup"]["precision"] = precision.name
    if updates:
        input_data.update(updates)

    input_path.write_text(toml.dumps(input_data), encoding="utf-8")


def test_input_resolves_defaults_and_constructs_selected_system(
    test_system,
    tmp_path,
    precision,
):
    """
    A resolved input combines defaults, user values, and CLI overrides.
    """

    # Start with a small input and override values inherited from two levels
    input_path = tmp_path / "input.toml"
    _write_input(input_path, test_system, precision=precision)
    flucs_input = FlucsInput(
        input_path,
        override=[
            "time.dt_max",
            "0.125",
            "setup.timing",
            "true",
        ],
    )

    # User values, defaults, and overrides all share the dotted-key interface
    assert flucs_input.input_path == input_path
    assert flucs_input.io_path == tmp_path
    assert flucs_input["setup.precision"] == precision.name
    assert flucs_input["time.dt_max"] == 0.125
    assert flucs_input["setup.timing"] is True

    # The printable form exposes the fully resolved input without private state
    resolved_input = toml.loads(str(flucs_input))
    assert resolved_input["setup"]["solver"] == test_system.solver_name
    assert resolved_input["setup"]["system"] == test_system.system_name
    assert resolved_input["time"]["dt_max"] == 0.125

    # Normal construction joins the selected solver and standalone system
    solver, system = flucs_input.create_solver_system()
    assert type(solver) is flucs.get_solver_type(test_system.solver_name)
    assert type(system) is test_system.system_type
    assert solver.input is flucs_input
    assert solver.system is system
    assert system.input is flucs_input
    assert system.solver is solver
    assert system.float is precision.float_type
    assert system.complex is precision.complex_type
    assert system.tolerance == precision.tolerance

    # Once resolved, the input is deliberately read-only
    with pytest.raises(RuntimeError, match="is now read-only"):
        flucs_input["time.dt_max"] = 0.25

    with pytest.raises(
        ValueError,
        match=r"Parameter time[.]missing does not exist",
    ):
        flucs_input["time.missing"]


@pytest.mark.parametrize(
    ("updates", "override", "error", "message"),
    [
        pytest.param(
            {"unknown": {"value": 1}},
            None,
            ValueError,
            "Parameter 'unknown' is invalid",
            id="unknown-parameter",
        ),
        pytest.param(
            {"time": 1.0},
            None,
            ValueError,
            "'time' is a group of parameters",
            id="group-as-value",
        ),
        pytest.param(
            None,
            ["time.dt_max", "not-a-number"],
            TypeError,
            "Error casting 'not-a-number'",
            id="invalid-override-type",
        ),
    ],
)
def test_input_rejects_invalid_parameters(
    test_system,
    tmp_path,
    updates,
    override,
    error,
    message,
):
    """
    Invalid names, structures, and types fail while input is being resolved.
    """

    # Keep every failure behind the same public file-loading boundary
    input_path = tmp_path / "input.toml"
    _write_input(input_path, test_system, updates=updates)

    with pytest.raises(error, match=message):
        FlucsInput(input_path, override=override)
