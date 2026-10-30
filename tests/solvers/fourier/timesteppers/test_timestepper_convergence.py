"""
Nonlinear timestep convergence using a common developed Fourier state.
"""

import pathlib as pl

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


def _run_solution(system, initial_fields, dt):
    """
    Advance from the common state at a fixed timestep to the common time.
    """
    _reset_system(system, initial_fields, dt)
    system.solver._solver_loop()
    cp.cuda.get_current_stream().synchronize()

    assert not system.solver.interrupted
    assert system.current_step == round(system.final_time / dt)
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
        assert not solver.interrupted
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
# RK4
# ------------------------------------------------------------------------------

def test_rk4_convergence(baseline_fourier_state, tmp_path, monkeypatch):
    """
    RK4 converges at fourth order for the developed nonlinear problem.
    """
    # Get initial fields (developed turbulence)
    initial_fields, frozen_hyperdissipation = baseline_fourier_state

    # Set up a range of timestep sizes for convergence testing.
    timestep_sizes = 2.0 ** -np.arange(7, 11)
    reference_dt = 2.0**-12
    override = {
        "setup.precision": "double",
        "timestepping.method": "rk4",
        "time.tfinal": 0.5,
        "hyperdissipation.kmod": frozen_hyperdissipation,
        "hyperdissipation.kmod_adaptive": False,
    }

    with create_test_system(INPUT_PATH, tmp_path, override=override) as (
        _,
        solver,
        system,
    ):
        # Compile once; each run resets all state and reuses these kernels.
        _setup_system(solver, system)
        monkeypatch.setattr(system, "_update_dt", lambda: False)

        # Run the reference solutions and compute errors.
        reference = _run_solution(system, initial_fields, reference_dt)
        errors = []
        for dt in timestep_sizes:
            solution = _run_solution(system, initial_fields, float(dt))
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
        assert fitted_order == pytest.approx(4, abs=0.1), (
            f"errors={errors}, local_orders={local_orders}"
        )
        assert local_orders[-1] == pytest.approx(4, abs=0.1), (
            f"errors={errors}, local_orders={local_orders}"
        )
