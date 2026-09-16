"""
Tests for the shared FLUCS solver interfaces.
"""

import signal
from unittest.mock import create_autospec

import pytest

from flucs import get_solver_type
from flucs.input import InvalidFlucsInputFileError
from flucs.solvers import FlucsSolverState
from tests.support.support import create_test_solver_system

pytestmark = pytest.mark.core


def test_solver_constructs_timesteppers_and_handles_signals(
    test_system,
    tmp_path,
    monkeypatch,
):
    """
    Solver construction selects every declared timestepper and installs exits.
    """

    # Observe registration without replacing this test process's signal handlers
    register_signal = create_autospec(signal.signal, spec_set=True)
    monkeypatch.setattr(signal, "signal", register_signal)

    # Keep the expected names and classes independent of the production mapping
    expected_timesteppers = test_system.timestepper_types
    assert expected_timesteppers

    solver_type = get_solver_type(test_system.solver_name)
    assert solver_type._supported_timesteppers == expected_timesteppers

    # Construct every supported method through the normal public input path
    constructed_solvers = []
    for method, expected_timestepper in expected_timesteppers.items():
        flucs_input, solver, system = create_test_solver_system(
            tmp_path / method,
            test_system,
            updates={"timestepping": {"method": method}},
        )
        constructed_solvers.append(solver)

        assert solver.state is FlucsSolverState.NOTINITIALISED
        assert type(solver.timestepper) is expected_timestepper
        assert solver.timestepper.solver is solver
        assert solver.timestepper.system is system
        assert solver.timestepper.input is flucs_input

    # Every construction registers the complete supported signal sequence
    registrations = register_signal.call_args_list
    expected_signals = [
        signal.SIGINT,
        signal.SIGTERM,
        signal.SIGUSR1,
        signal.SIGUSR2,
    ]
    assert [registration.args[0] for registration in registrations] == (
        expected_signals * len(constructed_solvers)
    )

    # One solver-owned callback handles every signal for a construction
    callbacks = {
        registration.args[1]
        for registration in registrations[: len(expected_signals)]
    }
    assert len(callbacks) == 1

    callback = callbacks.pop()
    first_solver = constructed_solvers[0]
    assert first_solver.interrupted is False
    callback(signal.SIGTERM, None)
    assert first_solver.interrupted is True

    # Invalid methods fail against the selected solver's own supported mapping
    with pytest.raises(
        InvalidFlucsInputFileError,
        match="missing is not a supported timestepper",
    ):
        create_test_solver_system(
            tmp_path / "invalid",
            test_system,
            updates={"timestepping": {"method": "missing"}},
        )
