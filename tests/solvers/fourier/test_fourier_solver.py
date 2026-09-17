"""
Tests for FourierSolver loop orchestration.
"""

from unittest.mock import Mock, call, create_autospec

import pytest

from flucs.restart import FlucsRestart
from flucs.solvers import FlucsSolverState
from tests.support.support import create_test_solver_system

pytestmark = pytest.mark.solver("FourierSolver")


###############################################################################
# CPU tests
###############################################################################


@pytest.mark.cpu
@pytest.mark.parametrize(
    (
        "state",
        "interrupted",
        "initial_step",
        "initial_time",
        "final_time",
        "timing_steps",
        "expected_steps",
    ),
    [
        pytest.param(
            FlucsSolverState.TIMING,
            False,
            0,
            10.0,
            0.0,
            2,
            2,
            id="timing-step-limit",
        ),
        pytest.param(
            FlucsSolverState.RUNNING,
            False,
            5,
            0.0,
            0.5,
            0,
            2,
            id="production-time-limit",
        ),
        pytest.param(
            FlucsSolverState.RUNNING,
            True,
            5,
            0.0,
            0.5,
            0,
            0,
            id="already-interrupted",
        ),
    ],
)
def test_solver_loop_coordinates_steps_and_final_writes(
    test_system,
    tmp_path,
    monkeypatch,
    state,
    interrupted,
    initial_step,
    initial_time,
    final_time,
    timing_steps,
    expected_steps,
):
    """
    The solver loop obeys its stop condition and coordinates every hook.
    """

    # Construct the real TestSystem while leaving all numerical work uncompiled
    _, solver, system = create_test_solver_system(
        tmp_path,
        test_system,
        updates={"setup": {"timing_steps": timing_steps}},
    )
    solver.state = state
    solver.interrupted = interrupted
    system.current_step = system.int(initial_step)
    system.current_time = system.float(initial_time)
    system.current_dt = system.float(0.25)
    system.final_time = system.float(final_time)

    # Strict mocks isolate loop ownership from CUDA and output implementations
    events = Mock()
    begin_time_step = create_autospec(
        system.begin_time_step,
        spec_set=True,
    )
    execute_timestep = create_autospec(
        solver.timestepper.execute_timestep,
        spec_set=True,
    )
    finish_time_step = create_autospec(
        system.finish_time_step,
        spec_set=True,
    )
    execute_diagnostics = create_autospec(
        system.execute_diagnostics,
        spec_set=True,
    )
    write_output = create_autospec(
        system.write_output,
        spec_set=True,
    )
    restart_manager = create_autospec(
        FlucsRestart,
        instance=True,
        spec_set=True,
    )

    # Set up mock events
    events.attach_mock(begin_time_step, "begin_time_step")
    events.attach_mock(execute_timestep, "execute_timestep")
    events.attach_mock(finish_time_step, "finish_time_step")
    events.attach_mock(execute_diagnostics, "execute_diagnostics")
    events.attach_mock(write_output, "write_output")
    events.attach_mock(restart_manager.write_restart, "write_restart")

    monkeypatch.setattr(system, "begin_time_step", begin_time_step)
    monkeypatch.setattr(
        solver.timestepper,
        "execute_timestep",
        execute_timestep,
    )
    monkeypatch.setattr(system, "finish_time_step", finish_time_step)
    monkeypatch.setattr(system, "execute_diagnostics", execute_diagnostics)
    monkeypatch.setattr(system, "write_output", write_output)
    system.restart_manager = restart_manager

    elapsed = solver._solver_loop()

    # Each completed step follows the complete production lifecycle
    expected_calls = []
    if not interrupted:
        expected_calls.append(call.execute_diagnostics())
        for _ in range(expected_steps):
            expected_calls.extend(
                (
                    call.begin_time_step(),
                    call.execute_timestep(),
                    call.finish_time_step(),
                    call.execute_diagnostics(),
                    call.write_output(),
                    call.write_restart(),
                )
            )
        expected_calls.extend(
            (
                call.execute_diagnostics(force=True),
                call.write_output(force=True),
                call.write_restart(force=True),
            )
        )

    assert events.mock_calls == expected_calls
    assert system.current_step == initial_step + expected_steps
    assert system.current_time == pytest.approx(
        initial_time + expected_steps * 0.25
    )

    if interrupted:
        assert elapsed == 0.0
    else:
        assert elapsed >= 0.0
        assert (
            system.steps_until_next_write == system.input["output.write_steps"]
        )


###############################################################################
# GPU tests
###############################################################################


@pytest.mark.runtime_precision("single")
def test_solver_completes_single_precision_runtime(runtime_run):
    """
    The real CUDA solver completes its production interval cleanly.
    """

    solver = runtime_run.solver
    system = runtime_run.system

    assert solver.state is FlucsSolverState.RUNNING
    assert solver.interrupted is False
    assert system.current_step > 0

    # The final step is precisely the first one to reach or cross final_time
    assert system.current_time >= system.final_time
    assert system.current_time - system.current_dt < system.final_time
