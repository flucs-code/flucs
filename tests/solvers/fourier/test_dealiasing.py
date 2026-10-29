"""
Numerical checks of the production Fourier dealiasing operations.
"""

import pathlib as pl

import numpy as np
import numpy.testing as npt
import pytest

from flucs import cupy as cp
from flucs.testing import create_test_system
from flucs.utilities.dealiasing import dealiased_multiplication_rfft

# These are all slow tests
pytestmark = [pytest.mark.gpu, pytest.mark.slow]

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
# Tests
################################################################################

# ------------------------------------------------------------------------------
# Two-thirds
# ------------------------------------------------------------------------------

@pytest.mark.parametrize("precision", ("single", "double"))
@pytest.mark.parametrize("fft_wrapper", ("flucs", "cupy"))
def test_two_thirds_product(
    test_fourier_system,
    tmp_path,
    precision,
    fft_wrapper,
):
    """
    Dealiased modes agree with an over-padded reference.
    """

    # Configure test
    source_input = pl.Path(__file__).parent / "inputs" / "dealiasing.toml"
    override = {
        "dealiasing.method": "two-thirds",
        "setup.precision": precision,
        "setup.fft_wrapper": fft_wrapper,
    }

    with create_test_system(source_input, tmp_path, override=override) as (
        _,
        solver,
        system,
    ):
        # Set up and compile without running the solver or health checks.
        system.setup()
        solver.timestepper.setup()
        system.compile_cupy_module()
        assert system.use_cupy_fft is (fft_wrapper == "cupy")

        product, reference = _compute_retained_product_and_reference(system)
        assert product.dtype == np.dtype(system.complex)
        npt.assert_allclose(
            product,
            reference,
            rtol=0,
            atol=system.tolerance,
        )


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
    source_input = pl.Path(__file__).parent / "inputs" / "dealiasing.toml"
    override = {
        "dealiasing.method": "two-thirds",
        "setup.precision": precision,
        "setup.fft_wrapper": "flucs",
        "dimensions.nz": 36,
        "dimensions.nx": 36,
        "dimensions.ny": 36,
        "dealiasing.nz_unpadded": n_unpadded,
        "dealiasing.nx_unpadded": n_unpadded,
        "dealiasing.ny_unpadded": n_unpadded,
    }

    with create_test_system(source_input, tmp_path, override=override) as (
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
# Phase-shift: spherical truncation
# ------------------------------------------------------------------------------

@pytest.mark.parametrize("precision", ("single", "double"))
@pytest.mark.parametrize(
    ("radius_squared", "is_safe"),
    [
        pytest.param(0.221, True, id="below-boundary"),
        pytest.param(0.222, True, id="at-boundary"),
        pytest.param(0.223, False, id="above-boundary"),
    ],
)
def test_spherical_phase_shift_boundary(
    test_fourier_system,
    tmp_path,
    precision,
    radius_squared,
    is_safe,
):
    """
    Spherical phase-shift products alias beyond the safe truncation radius.
    """

    # The quadratic cutoff is radius_squared < 2/9; 0.223 exceeds it.
    source_input = pl.Path(__file__).parent / "inputs" / "dealiasing.toml"
    override = {
        "dealiasing.method": "phase-shift",
        "dealiasing.truncation": "spherical",
        "dealiasing.radius_squared": radius_squared,
        "setup.precision": precision,
        "setup.fft_wrapper": "flucs",
        "dimensions.nz": 36,
        "dimensions.nx": 36,
        "dimensions.ny": 36,
    }

    with create_test_system(source_input, tmp_path, override=override) as (
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
# Phase-shift: polyhedral truncation
# ------------------------------------------------------------------------------

@pytest.mark.parametrize("precision", ("single", "double"))
@pytest.mark.parametrize(
    ("max_sum", "is_safe"),
    [
        pytest.param(0.665, True, id="below-boundary"),
        pytest.param(0.666, True, id="at-boundary"),
        pytest.param(0.667, False, id="above-boundary"),
    ],
)
def test_polyhedral_phase_shift_boundary(
    test_fourier_system,
    tmp_path,
    precision,
    max_sum,
    is_safe,
):
    """
    Polyhedral phase-shift products alias beyond the safe pairwise cutoff.
    """

    # The quadratic cutoff is max_sum < 2/3; 0.667 exceeds it.
    source_input = pl.Path(__file__).parent / "inputs" / "dealiasing.toml"
    override = {
        "dealiasing.method": "phase-shift",
        "dealiasing.truncation": "polyhedral",
        "dealiasing.max_sum": max_sum,
        "setup.precision": precision,
        "setup.fft_wrapper": "flucs",
        "dimensions.nz": 36,
        "dimensions.nx": 36,
        "dimensions.ny": 36,
    }

    with create_test_system(source_input, tmp_path, override=override) as (
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
# Phase-shift: FFT wrappers and memory models
# ------------------------------------------------------------------------------

@pytest.mark.parametrize("precision", ("single", "double"))
@pytest.mark.parametrize("truncation", ("spherical", "polyhedral"))
@pytest.mark.parametrize("fft_wrapper", ("flucs", "cupy"))
def test_phase_shift_memory_models_agree(
    test_fourier_system,
    tmp_path,
    precision,
    truncation,
    fft_wrapper,
):
    """
    Supported FFT wrappers and memory layouts produce the same product.
    """

    # Keep each truncation just inside its safe theoretical cutoff.
    source_input = pl.Path(__file__).parent / "inputs" / "dealiasing.toml"
    cutoff = (
        {"dealiasing.radius_squared": 0.222}
        if truncation == "spherical"
        else {"dealiasing.max_sum": 0.666}
    )
    override = {
        "dealiasing.method": "phase-shift",
        "dealiasing.truncation": truncation,
        "setup.precision": precision,
        "setup.fft_wrapper": fft_wrapper,
        "dimensions.nz": 30,
        "dimensions.nx": 30,
        "dimensions.ny": 30,
        **cutoff,
    }

    # CuPy does not support the in-place phase-shift layout.
    memory_models = {
        "flucs": ("standard", "low_memory", "in_place"),
        "cupy": ("standard", "low_memory"),
    }[fft_wrapper]

    products = []
    for memory in memory_models:
        override["dealiasing.memory"] = memory
        with create_test_system(
            source_input, tmp_path / memory, override=override
        ) as (_, solver, system):
            system.setup()
            solver.timestepper.setup()
            system.compile_cupy_module()
            assert system.use_cupy_fft is (fft_wrapper == "cupy")

            # Every layout must agree with the independent reference.
            product, reference = _compute_retained_product_and_reference(
                system
            )
            npt.assert_allclose(
                product,
                reference,
                rtol=0,
                atol=system.tolerance,
            )
            products.append(product)

    # Also check agreement between layouts using the same FFT wrapper.
    for product in products[1:]:
        npt.assert_allclose(
            product,
            products[0],
            rtol=0,
            atol=system.tolerance,
        )
