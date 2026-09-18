"""
Numerical convergence tests for the FourierSolver timesteppers.
"""

import numpy as np
import numpy.testing as npt
import pytest

from flucs import cupy as cp
from flucs.solvers import FlucsSolverState
from tests.support.support import (
    DOUBLE_PRECISION,
    SINGLE_PRECISION,
    TEST_SYSTEMS,
    create_test_solver_system,
)

pytestmark = pytest.mark.solver("FourierSolver")

# Overall configuration for the testing
EXPECTED_ORDERS = {
    "ab3": 3,
    "rk4": 4,
    "ssprk3": 3,
}
ORDER_TOLERANCE = 0.1

TIMESTEP_SIZES = 2.0 ** -np.arange(7, 11)
REFERENCE_TIMESTEP_SIZE = 2.0**-12
TIME_INTERVAL = 0.5

SPINUP_TIMESTEP = 0.02


def _system_updates(method, *, baseline_run=False, hyperdissipation=None):
    """
    Return the common nonlinear problem with method-specific integration.
    """

    if baseline_run:
        hyperdissipation = 1.0
    elif hyperdissipation is None:
        raise ValueError("A frozen hyperdissipation rate is required.")

    return {
        "dimensions": {"nz": 32, "nx": 32, "ny": 32},
        "dealiasing": {"method": "two-thirds"},
        "setup": {"check_linear_matrix": False},
        "time": {
            "dt_max": SPINUP_TIMESTEP,
            "max_cfl": 0.2,
            "dt_method": "continuous",
            "tfinal": 100.0 if baseline_run else TIME_INTERVAL,
        },
        "timestepping": {"method": method},
        "parameters": {
            "advection": [0.3, 0.5, -0.1],
            "rotation": [0.2, 0.1, 0.2],
        },
        "init": {"method": "deterministic", "amplitude": 0.01},
        "forcing": {
            "method": "negative_damping",
            "rate": 0.05,
            "range_kmod": [0.5, 2.5],
        },
        "hyperdissipation": {
            "kmod": hyperdissipation,
            "kmod_power": 3,
            "kmod_adaptive": baseline_run,
            "kmod_normalised": True,
        },
        "restart": {"write_restart_file": False},
    }


def _compile_system(
    io_path,
    method,
    *,
    precision,
    baseline_run=False,
    hyperdissipation=None,
):
    """
    Compile one complete nonlinear TestFourierSystem.
    """

    _, solver, system = create_test_solver_system(
        io_path,
        TEST_SYSTEMS["FourierSolver"],
        precision=precision,
        updates=_system_updates(
            method,
            baseline_run=baseline_run,
            hyperdissipation=hyperdissipation,
        ),
    )
    system.setup()
    solver.timestepper.setup()
    system.compile_cupy_module()
    system.setup_initial_conditions()

    if not baseline_run:
        # The scan freezes the controller after the adaptive spin-up
        system._update_dt = lambda: False

    return system


def _reset_system(system, initial_fields, dt):
    """
    Restore fields, counters, forcing, and timestepper state for one run.
    """

    system.fields_initial = np.asarray(initial_fields, dtype=system.complex)
    system.init_time = system.float(0)
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
    Advance one step while constructing full-order AB3 startup history.
    """

    system.current_step += 1
    system.begin_time_step()
    system.solver.timestepper.execute_timestep()
    system.current_time += system.current_dt
    system.finish_time_step()


@pytest.fixture(scope="module")
def baseline_fourier_state(tmp_path_factory):
    """
    Generate one deterministic saturated state for every convergence run.
    """

    system = _compile_system(
        tmp_path_factory.mktemp("fourier-baseline_run"),
        "ssprk3",
        precision=SINGLE_PRECISION,
        baseline_run=True,
    )
    initial_fields = system.fields_initial.copy()

    try:
        _reset_system(system, initial_fields, SPINUP_TIMESTEP)
        system.solver._solver_loop()
        cp.cuda.runtime.deviceSynchronize()

        baseline_fields = cp.asnumpy(system.get_fields()).copy()
        # Finite
        assert np.all(np.isfinite(baseline_fields))

        # Simulation has stepped beyond the initial state
        assert system.current_step > 0

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

        # Freeze the adaptive physical rate reached by the continuous run
        assert system.input["time.dt_method"] == "continuous"
        assert system.input["hyperdissipation.kmod_adaptive"]
        assert system.current_dt <= system.dt_max

        frozen_hyperdissipation = float(
            system.input["hyperdissipation.kmod"] * system.adaptive_rate
        )

        return baseline_fields, frozen_hyperdissipation
    finally:
        system.kernels.unbind()


@pytest.fixture(scope="module")
def convergence_systems(tmp_path_factory, baseline_fourier_state):
    """
    Compile each timestepper once and give all of them the developed state.
    """

    baseline_fields, frozen_hyperdissipation = baseline_fourier_state
    systems = {
        method: _compile_system(
            tmp_path_factory.mktemp(f"fourier-{method}"),
            method,
            precision=DOUBLE_PRECISION,
            hyperdissipation=frozen_hyperdissipation,
        )
        for method in EXPECTED_ORDERS
    }
    for system in systems.values():
        system.fields_initial = baseline_fields.astype(system.complex)

        # Every scan evolves the same forced problem with frozen controls
        assert system.input["time.dt_method"] == "continuous"
        assert system.input["forcing.method"] == "negative_damping"
        assert not system.input["hyperdissipation.kmod_adaptive"]
        assert system.input["hyperdissipation.kmod"] == pytest.approx(
            frozen_hyperdissipation
        )

    try:
        yield systems
    finally:
        for system in systems.values():
            system.kernels.unbind()


def _bootstrap_ab3(systems, initial_fields, dt):
    """
    Seed AB3 history while retaining full-order RK4 startup fields.
    """

    # Obtain full-order u1 and u2 from the already compiled RK4 system
    rk4_system = systems["rk4"]
    _reset_system(rk4_system, initial_fields, dt)
    _advance_one_step(rk4_system)
    fields_1 = rk4_system.get_fields().copy()
    _advance_one_step(rk4_system)
    fields_2 = rk4_system.get_fields().copy()

    ab3_system = systems["ab3"]
    _reset_system(ab3_system, initial_fields, dt)

    # Let production AB3 rotate and propagate its own history. Replacing each
    # startup result makes the next explicit evaluation use the RK4 state.
    _advance_one_step(ab3_system)
    ab3_system.get_fields()[...] = fields_1
    _advance_one_step(ab3_system)
    ab3_system.get_fields()[...] = fields_2

    # The third step is the first one with complete AB3 history
    _advance_one_step(ab3_system)
    coefficients = ab3_system.solver.timestepper.ab3_coefficients.copy()

    return ab3_system, coefficients


def _run_solution(method, systems, initial_fields, dt):
    """
    Evolve one method from the common state over the common interval.
    """

    bootstrap = None

    if method == "ab3":
        system, bootstrap = _bootstrap_ab3(systems, initial_fields, dt)
    else:
        system = systems[method]
        _reset_system(system, initial_fields, dt)

    # All ordinary evolution uses the complete production solver loop
    system.solver._solver_loop()

    cp.cuda.runtime.deviceSynchronize()

    expected_steps = round(TIME_INTERVAL / dt)
    assert system.current_step == expected_steps
    assert system.current_time == pytest.approx(TIME_INTERVAL)
    assert system.current_dt == pytest.approx(dt)

    return cp.asnumpy(system.get_fields()).copy(), bootstrap


###############################################################################
# CPU tests
###############################################################################

# None


###############################################################################
# GPU tests
###############################################################################


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("method", tuple(EXPECTED_ORDERS))
def test_timestepper_convergence(
    method,
    convergence_systems,
    baseline_fourier_state,
):
    """
    Every live timestepper demonstrates its formal nonlinear convergence order.
    """

    baseline_fields, _ = baseline_fourier_state
    system = convergence_systems[method]
    assert set(EXPECTED_ORDERS) == set(system.solver._supported_timesteppers)

    # Each method receives its own finer reference from the identical state
    reference, reference_bootstrap = _run_solution(
        method,
        convergence_systems,
        baseline_fields,
        REFERENCE_TIMESTEP_SIZE,
    )

    # Check that the bootstrap has given the correct AB3 coefficients
    if method == "ab3":
        npt.assert_allclose(
            reference_bootstrap,
            np.asarray((23, -16, 5), dtype=system.float) / 12,
            rtol=0,
            atol=system.tolerance,
        )

    # Iterate over the timestep sizes to measure the convergence of the solution
    errors = []
    for dt in TIMESTEP_SIZES:
        solution, _ = _run_solution(
            method,
            convergence_systems,
            baseline_fields,
            float(dt),
        )

        difference = solution - reference
        error = np.linalg.norm(difference) / np.linalg.norm(reference)
        errors.append(error)

    errors = np.asarray(errors)
    assert np.all(np.isfinite(errors))
    assert np.all(errors[:-1] > errors[1:])
    assert errors[-1] > system.tolerance

    # Fit the complete compact scan and separately guard its two finest slopes
    fitted_order = np.polyfit(np.log(TIMESTEP_SIZES), np.log(errors), 1)[0]
    local_orders = np.log2(errors[:-1] / errors[1:])
    expected_order = EXPECTED_ORDERS[method]

    assert fitted_order == pytest.approx(expected_order, abs=ORDER_TOLERANCE), (
        f"errors={errors}, local_orders={local_orders}"
    )
    assert local_orders[-1] == pytest.approx(
        expected_order,
        abs=ORDER_TOLERANCE,
    )
