"""
Tests for the shared FLUCS solver interfaces.
"""

import signal
from unittest.mock import create_autospec

import pytest

from flucs.input import InvalidFlucsInputFileError
from flucs.solvers import FlucsSolverState
from tests.support.support import create_test_solver_system

pytestmark = pytest.mark.core


def test_solver_constructs_timestepper_and_handles_signals(
    test_system,
    tmp_path,
    monkeypatch,
):
    """
    Solver construction selects one timestepper and installs clean exits.
    """

    # Observe registration without replacing this test process's signal handlers
    register_signal = create_autospec(signal.signal, spec_set=True)
    monkeypatch.setattr(signal, "signal", register_signal)

    # Construct through the registered TestSystem and public input path
    flucs_input, solver, system = create_test_solver_system(
        tmp_path / "valid",
        test_system,
    )
    method = flucs_input["timestepping.method"]

    assert solver.state is FlucsSolverState.NOTINITIALISED
    assert type(solver.timestepper) is solver._supported_timesteppers[method]
    assert solver.timestepper.solver is solver
    assert solver.timestepper.system is system
    assert solver.timestepper.input is flucs_input

    # Each supported termination signal uses the same solver-owned callback
    registrations = register_signal.call_args_list
    assert [registration.args[0] for registration in registrations] == [
        signal.SIGINT,
        signal.SIGTERM,
        signal.SIGUSR1,
        signal.SIGUSR2,
    ]
    callbacks = {registration.args[1] for registration in registrations}
    assert len(callbacks) == 1

    callback = callbacks.pop()
    assert solver.interrupted is False
    callback(signal.SIGTERM, None)
    assert solver.interrupted is True

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
