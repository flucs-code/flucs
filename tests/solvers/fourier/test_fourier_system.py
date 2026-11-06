"""
Numerical checks of FourierSystem.
"""

import pathlib as pl

import numpy as np
import numpy.testing as npt
import pytest
import toml

from flucs import cupy as cp
from flucs.restart import FlucsRestart
from flucs.testing import create_test_system
from flucs.utilities.dealiasing import dealiased_multiplication_rfft

################################################################################
# Helper functions
################################################################################


def _compute_retained_product_and_reference(system):
    """
    Compare a production product with an independently padded calculation.
    """
    # Fixed-seed fields populate the full retained spectrum.
    random = np.random.default_rng(7412)
    fields_real = random.standard_normal((2, *system.full_tuple)).astype(
        system.float
    )
    fields_fourier = cp.fft.rfftn(
        cp.asarray(fields_real),
        axes=(-3, -2, -1),
        norm="forward",
    )
    solved_modes = system.get_solved_grid_mask().astype(bool)
    fields_fourier[:, ~cp.asarray(solved_modes)] = 0

    # Exercise the production transform/product/transform operation.
    system.check_dealiasing_errors_operation(
        current_dt=system.float(0),
        current_time=system.time_float(0),
        current_step=system.int(0),
        input_array=fields_fourier,
        calculate_cfl=False,
    )
    product = system.check_dealiasing_errors_output[0].get()

    # Independent padding avoids aliases before cropping to this grid.
    reference = dealiased_multiplication_rfft(
        fields_fourier[0],
        fields_fourier[1],
        nz=system.nz,
        nx=system.nx,
        ny=system.ny,
        padded_nz=2 * system.nz,
        padded_nx=2 * system.nx,
        padded_ny=2 * system.ny,
    ).get()

    # Only retained coefficients belong to the evolved solution.
    return product[solved_modes], reference[solved_modes]


################################################################################
# Fixtures
################################################################################


@pytest.fixture(scope="module")
def dealiasing_input(tmp_path_factory):
    """
    Write the common input for dealiased products.
    """
    input_path = tmp_path_factory.mktemp("dealiasing_input") / "dealiasing.toml"
    input_path.write_text(
        """\
        [setup]
        solver = "FourierSolver"
        system = "TestFourierSystem"
        check_linear_matrix = false

        [dimensions]
        nz = 12
        nx = 12
        ny = 12

        [dealiasing]
        check_errors = true

        [forcing]
        method = ""
        """,
        encoding="utf-8",
    )
    return input_path


################################################################################
# Tests
################################################################################

# ------------------------------------------------------------------------------
# Dealiasing: two-thirds
# ------------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("precision", ("single", "double"))
@pytest.mark.parametrize("fft_wrapper", ("flucs", "cupy"))
def test_two_thirds_product(
    test_fourier_system,
    dealiasing_input,
    tmp_path,
    precision,
    fft_wrapper,
):
    """
    Dealiased modes agree with an over-padded reference.
    """

    # Configure test
    override = {
        # Dealiasing
        "dealiasing.method": "two-thirds",
        # Setup
        "setup.precision": precision,
        "setup.fft_wrapper": fft_wrapper,
    }

    with create_test_system(dealiasing_input, tmp_path, override=override) as (
        _,
        solver,
        system,
    ):
        # Set up and compile without running the solver or health checks.
        system.setup()
        solver.timestepper.setup()
        system.compile_cupy_module()

        product, reference = _compute_retained_product_and_reference(system)
        assert product.dtype == np.dtype(system.complex)
        npt.assert_allclose(
            product,
            reference,
            rtol=0,
            atol=system.tolerance,
            equal_nan=False,
        )


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("precision", ("single", "double"))
@pytest.mark.parametrize(
    ("n_unpadded", "is_safe"),
    [
        pytest.param(21, True, id="below-boundary"),
        pytest.param(23, True, id="at-boundary"),
        pytest.param(25, False, id="above-boundary"),
    ],
)
def test_two_thirds_boundary(
    test_fourier_system,
    dealiasing_input,
    tmp_path,
    precision,
    n_unpadded,
    is_safe,
):
    """
    Products alias only beyond the maximum safe retained bandwidth.
    """

    # For this 36-point grid, quadratic products require kmax < 12.
    # Retained sizes 21, 23, and 25 correspond to kmax = 10, 11, and 12.
    override = {
        # Dealiasing
        "dealiasing.method": "two-thirds",
        # Setup
        "setup.precision": precision,
        "setup.fft_wrapper": "flucs",
        # Dimensions
        "dimensions.nz": 36,
        "dimensions.nx": 36,
        "dimensions.ny": 36,
        # Dealiasing
        "dealiasing.nz_unpadded": n_unpadded,
        "dealiasing.nx_unpadded": n_unpadded,
        "dealiasing.ny_unpadded": n_unpadded,
    }

    with create_test_system(dealiasing_input, tmp_path, override=override) as (
        _,
        solver,
        system,
    ):
        # Keep the unsafe bandwidth: the test must detect its aliasing.
        system.setup()
        solver.timestepper.setup()
        system.compile_cupy_module()

        product, reference = _compute_retained_product_and_reference(system)
        error = np.max(np.abs(product - reference))
        assert np.isfinite(error)

        # Safe cases are round-off limited; the next bandwidth must alias.
        if is_safe:
            assert error <= system.tolerance
        else:
            assert error > system.tolerance


# ------------------------------------------------------------------------------
# Dealiasing: phase-shift spherical truncation
# ------------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("precision", ("single", "double"))
@pytest.mark.parametrize(
    ("radius_squared", "is_safe"),
    [
        pytest.param(0.221, True, id="below-boundary"),
        pytest.param(0.223, False, id="above-boundary"),
    ],
)
def test_spherical_phase_shift_boundary(
    test_fourier_system,
    dealiasing_input,
    tmp_path,
    precision,
    radius_squared,
    is_safe,
):
    """
    Spherical phase-shift products alias beyond the safe truncation radius.
    """

    # The quadratic cutoff is radius_squared < 2/9; 0.223 exceeds it.
    override = {
        # Dealiasing
        "dealiasing.method": "phase-shift",
        "dealiasing.truncation": "spherical",
        "dealiasing.radius_squared": radius_squared,
        # Setup
        "setup.precision": precision,
        "setup.fft_wrapper": "flucs",
        # Dimensions
        "dimensions.nz": 36,
        "dimensions.nx": 36,
        "dimensions.ny": 36,
    }

    with create_test_system(dealiasing_input, tmp_path, override=override) as (
        _,
        solver,
        system,
    ):
        # Compile without a health check that would reject the unsafe case.
        system.setup()
        solver.timestepper.setup()
        system.compile_cupy_module()

        product, reference = _compute_retained_product_and_reference(system)
        error = np.max(np.abs(product - reference))
        assert np.isfinite(error)

        # The shell admitted above 2/9 must introduce detectable aliasing.
        if is_safe:
            assert error <= system.tolerance
        else:
            assert error > system.tolerance


# ------------------------------------------------------------------------------
# Dealiasing: phase-shift polyhedral truncation
# ------------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("precision", ("single", "double"))
@pytest.mark.parametrize(
    ("max_sum", "is_safe"),
    [
        pytest.param(0.665, True, id="below-boundary"),
        pytest.param(0.667, False, id="above-boundary"),
    ],
)
def test_polyhedral_phase_shift_boundary(
    test_fourier_system,
    dealiasing_input,
    tmp_path,
    precision,
    max_sum,
    is_safe,
):
    """
    Polyhedral phase-shift products alias beyond the safe pairwise cutoff.
    """

    # The quadratic cutoff is max_sum < 2/3; 0.667 exceeds it.
    override = {
        # Dealiasing
        "dealiasing.method": "phase-shift",
        "dealiasing.truncation": "polyhedral",
        "dealiasing.max_sum": max_sum,
        # Setup
        "setup.precision": precision,
        "setup.fft_wrapper": "flucs",
        # Dimensions
        "dimensions.nz": 36,
        "dimensions.nx": 36,
        "dimensions.ny": 36,
    }

    with create_test_system(dealiasing_input, tmp_path, override=override) as (
        _,
        solver,
        system,
    ):
        # Compile without a health check that would reject the unsafe case.
        system.setup()
        solver.timestepper.setup()
        system.compile_cupy_module()

        product, reference = _compute_retained_product_and_reference(system)
        error = np.max(np.abs(product - reference))
        assert np.isfinite(error)

        # The shell admitted above 2/3 must introduce detectable aliasing.
        if is_safe:
            assert error <= system.tolerance
        else:
            assert error > system.tolerance


# ------------------------------------------------------------------------------
# Dealiasing: phase-shift FFT wrappers and memory models
# ------------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("precision", ("single", "double"))
@pytest.mark.parametrize("truncation", ("spherical", "polyhedral"))
@pytest.mark.parametrize("fft_wrapper", ("flucs", "cupy"))
def test_phase_shift_memory_models(
    test_fourier_system,
    dealiasing_input,
    tmp_path,
    precision,
    truncation,
    fft_wrapper,
):
    """
    Supported FFT wrappers and memory layouts produce the same product.
    """

    # Keep each truncation just inside its safe theoretical cutoff.
    cutoff = (
        {
            # Dealiasing
            "dealiasing.radius_squared": 0.222,
        }
        if truncation == "spherical"
        else {
            # Dealiasing
            "dealiasing.max_sum": 0.666,
        }
    )
    override = {
        # Dealiasing
        "dealiasing.method": "phase-shift",
        "dealiasing.truncation": truncation,
        # Setup
        "setup.precision": precision,
        "setup.fft_wrapper": fft_wrapper,
        # Dimensions
        "dimensions.nz": 30,
        "dimensions.nx": 30,
        "dimensions.ny": 30,
        # Dealiasing
        **cutoff,
    }

    # CuPy does not support the in-place phase-shift layout.
    memory_models = {
        "flucs": ("standard", "low_memory", "in_place"),
        "cupy": ("standard", "low_memory"),
    }[fft_wrapper]

    for memory in memory_models:
        override["dealiasing.memory"] = memory
        with create_test_system(
            dealiasing_input, tmp_path / memory, override=override
        ) as (_, solver, system):
            system.setup()
            solver.timestepper.setup()
            system.compile_cupy_module()

            # Every layout must agree with the independent reference.
            product, reference = _compute_retained_product_and_reference(system)
            npt.assert_allclose(
                product,
                reference,
                rtol=0,
                atol=system.tolerance,
                equal_nan=False,
            )


# ------------------------------------------------------------------------------
# Restart preparation
# ------------------------------------------------------------------------------


@pytest.mark.cpu
@pytest.mark.parametrize("precision", ("single", "double"))
@pytest.mark.parametrize(
    ("restart_resolution", "resolution"),
    [
        pytest.param((18, 24, 30), (32, 40, 48), id="increase-resolution"),
        pytest.param((32, 40, 48), (18, 24, 30), id="decrease-resolution"),
    ],
)
def test_prepare_restart_data_resolution(
    test_fourier_system,
    tmp_path,
    precision,
    restart_resolution,
    resolution,
):
    """
    Preserve common Fourier modes and zero new modes when resizing a restart.
    """
    source_input = pl.Path(__file__).parents[2] / "support/fourier/runtime.toml"
    restart_nz, restart_nx, restart_ny = restart_resolution
    nz, nx, ny = resolution

    # Setup input file
    override = {
        # Setup
        "setup.precision": precision,
        # Dimensions
        "dimensions.nz": nz,
        "dimensions.nx": nx,
        "dimensions.ny": ny,
        # Forcing
        "forcing.method": "",
        # Restart
        "restart.write_restart_file": False,
    }

    with create_test_system(source_input, tmp_path, override=override) as (
        flucs_input,
        _,
        system,
    ):
        # Prescribe signed Fourier modes common to both resolutions.
        # Negative indices select negative kz/kx; ky is stored nonnegative.
        retained_modes = {
            (0, 0, 0): 1,
            (1, 2, 0): 1 + 2j,
            (-1, -2, 0): 1 - 2j,
            (8, -11, 14): 2 + 3j,
            (-8, 11, 1): -1 + 4j,
            (-9, -12, 3): 3 - 2j,
            (0, 0, 15): 4,
        }

        field_amplitudes = np.arange(1, system.number_of_fields + 1)

        restart_shape = (restart_nz, restart_nx, restart_ny // 2 + 1)
        restart_fields = np.zeros(
            (system.number_of_fields, *restart_shape),
            dtype=system.complex,
        )
        expected = np.zeros(
            (system.number_of_fields, nz, nx, ny // 2 + 1),
            dtype=system.complex,
        )

        for (kz, kx, ky), coefficient in retained_modes.items():
            restart_fields[:, kz, kx, ky] = field_amplitudes * coefficient
            expected[:, kz, kx, ky] = field_amplitudes * coefficient

        # The larger restart also contains modes excluded by the smaller grid.
        if restart_resolution == (32, 40, 48):
            discarded_modes = {
                (9, 0, 1): 5 + 6j,
                (-10, 0, 1): 7 - 2j,
                (0, 12, 1): -3 + 8j,
                (0, -13, 1): 9 + 4j,
                (0, 0, 16): 2 - 7j,
            }
            for (kz, kx, ky), coefficient in discarded_modes.items():
                restart_fields[:, kz, kx, ky] = field_amplitudes * coefficient

        # Supply restart data with the original resolution and unchanged box.
        restart_input = toml.loads(str(flucs_input))
        restart_input["dimensions"].update(
            nz=restart_nz,
            nx=restart_nx,
            ny=restart_ny,
        )
        system.restart_manager = FlucsRestart(system)
        system.restart_manager.data = {
            "fields": {
                "data": restart_fields,
                "dimension_names": ("number_of_fields", "nz", "nx", "half_ny"),
            },
            "input_file": {
                "data": np.asarray(toml.dumps(restart_input)),
                "dimension_names": (),
            },
        }

        # Remap through the production path without compiling or evolving.
        actual = system.prepare_restart_data()

        assert actual.dtype == np.dtype(system.complex)
        npt.assert_array_equal(actual, expected)
