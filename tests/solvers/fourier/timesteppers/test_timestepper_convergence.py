"""
Nonlinear timestep convergence using a common developed Fourier state.
"""

import pathlib as pl
from contextlib import ExitStack

import numpy as np
import pytest

from flucs import cupy as cp
from flucs.solvers import FlucsSolverState
from flucs.testing import create_test_system

pytestmark = [pytest.mark.gpu, pytest.mark.slow]

INPUT_PATH = pl.Path(__file__).parent / "inputs" / "convergence.toml"

################################################################################
# Helper functions
################################################################################

def _setup_system(solver, system):
    """
    Allocate and compile without diagnostics or a solver run.
    """
    system.setup()
    solver.timestepper.setup()
    system.compile_cupy_module()
    system.setup_initial_conditions()


def _reset_system(system, initial_fields, dt):
    """
    Reset fields, counters, and timestepper state before an independent run.
    """
    system.fields_initial = np.asarray(initial_fields, dtype=system.complex)
    system.init_time = system.time_float(0)
    system.init_dt = system.float(dt)
    system.dt_max = system.float(dt)
    system.solver.interrupted = False
    system.solver.state = FlucsSolverState.RUNNING

    for fields in system.fields:
        fields.fill(0)
    system.ready()
    system.solver.timestepper.ready()


def _advance_one_step(system):
    """
    Advance one timestep with the normal solver bookkeeping.
    """
    system.current_step += 1
    system.begin_time_step()
    system.solver.timestepper.execute_timestep()
    system.current_time += system.current_dt
    system.finish_time_step()


def _bootstrap_ab3(system, rk4_system, initial_fields, dt):
    """
    Seed accurate field states while AB3 fills its own RHS history.
    """
    # Advance with RK4 for seeding AB3 history.
    _reset_system(rk4_system, initial_fields, dt)
    _advance_one_step(rk4_system)
    first_fields = rk4_system.get_fields().copy()
    _advance_one_step(rk4_system)
    second_fields = rk4_system.get_fields().copy()

    # Seed AB3 history
    _reset_system(system, initial_fields, dt)
    _advance_one_step(system)
    system.get_fields()[...] = first_fields
    _advance_one_step(system)
    system.get_fields()[...] = second_fields
    _advance_one_step(system)


def _run_solution(system, initial_fields, dt, *, startup_system=None):
    """
    Advance from the common state at a fixed timestep to the common time.
    """
    if startup_system is None:
        _reset_system(system, initial_fields, dt)
    else:
        _bootstrap_ab3(system, startup_system, initial_fields, dt)
    system.solver._solver_loop()
    cp.cuda.get_current_stream().synchronize()

    assert system.current_time == system.final_time
    assert system.current_dt == dt

    return system.get_fields().get()

################################################################################
# Fixtures
################################################################################

@pytest.fixture(scope="module")
def baseline_fourier_state(test_fourier_system, tmp_path_factory):
    """
    Evolve the original single-precision SSPRK3 baseline once.
    """
    io_path = tmp_path_factory.mktemp("convergence_baseline")
    with create_test_system(INPUT_PATH, io_path) as (_, solver, system):

        # Get baseline fields
        _setup_system(solver, system)
        _reset_system(system, system.fields_initial, system.init_dt)
        solver._solver_loop()
        cp.cuda.get_current_stream().synchronize()

        baseline_fields = system.get_fields().get()
        assert system.current_time >= system.final_time
        assert np.all(np.isfinite(baseline_fields))

        # Ensure the developed state exercises nonlinear terms.
        system.compute_nonlinear_terms(
            system.current_dt,
            system.current_time,
            system.current_step,
            system.get_fields(),
            False,
        )
        nonlinear_norm = float(cp.linalg.norm(system.dft_bits))
        field_norm = float(cp.linalg.norm(system.get_fields()))
        assert nonlinear_norm > system.tolerance * field_norm

        # Preserve the physical dissipation rate reached by the adaptive run.
        frozen_hyperdissipation = float(
            system.input["hyperdissipation.kmod"] * system.adaptive_rate
        )
        return baseline_fields, frozen_hyperdissipation


################################################################################
# Tests
################################################################################

# ------------------------------------------------------------------------------
# Timestepper convergence
# ------------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("method", "expected_order"),
    [("rk4", 4), ("ssprk3", 3), ("ssprk3e", 3), ("ab3", 3)],
)
def test_timestepper_convergence(
    baseline_fourier_state,
    tmp_path,
    monkeypatch,
    method,
    expected_order,
):
    """
    Timesteppers recover their formal nonlinear convergence order.
    """
    # Get initial fields (developed turbulence)
    initial_fields, frozen_hyperdissipation = baseline_fourier_state

    # Set up a range of timestep sizes for convergence testing.
    timestep_sizes = 2.0 ** -np.arange(7, 11)
    reference_dt = 2.0**-12
    override = {
        "setup.precision": "double",
        "timestepping.method": method,
        "time.tfinal": 0.5,
        "hyperdissipation.kmod": frozen_hyperdissipation,
        "hyperdissipation.kmod_adaptive": False,
    }

    with ExitStack() as cases:
        # Compile once; each run resets all state and reuses these kernels.
        _, solver, system = cases.enter_context(
            create_test_system(INPUT_PATH, tmp_path, override=override)
        )
        _setup_system(solver, system)
        monkeypatch.setattr(system, "_update_dt", lambda: False)

        # AB3 needs another method to seed its multi-step history (here RK4).
        startup_system = None
        if method == "ab3":
            startup_override = override.copy()
            startup_override["timestepping.method"] = "rk4"
            _, startup_solver, startup_system = cases.enter_context(
                create_test_system(
                    INPUT_PATH,
                    tmp_path / "rk4_startup",
                    override=startup_override,
                )
            )
            _setup_system(startup_solver, startup_system)
            monkeypatch.setattr(startup_system, "_update_dt", lambda: False)

        # Run the reference solutions and compute errors.
        reference = _run_solution(
            system, initial_fields, reference_dt, startup_system=startup_system
        )
        errors = []
        for dt in timestep_sizes:
            solution = _run_solution(
                system, initial_fields, float(dt), startup_system=startup_system
            )
            error = np.linalg.norm(solution - reference) / np.linalg.norm(
                reference
            )
            errors.append(error)

        errors = np.asarray(errors)
        assert np.all(np.isfinite(errors))
        assert np.all(errors[:-1] > errors[1:])
        assert errors[-1] > system.tolerance

        # Fit the full scan and independently check its finest local slope.
        fitted_order = np.polyfit(np.log(timestep_sizes), np.log(errors), 1)[0]
        local_orders = np.log2(errors[:-1] / errors[1:])

        assert fitted_order == pytest.approx(expected_order, abs=0.1), (
            f"errors={errors}, local_orders={local_orders}"
        )
        assert local_orders[-1] == pytest.approx(expected_order, abs=0.1), (
            f"errors={errors}, local_orders={local_orders}"
        )
