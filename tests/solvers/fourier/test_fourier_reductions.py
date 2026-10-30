"""
Fourier reductions of explicitly prescribed mode contributions.
"""

import pathlib as pl

import numpy as np
import numpy.testing as npt
import pytest

from flucs import cupy as cp
from flucs.solvers.fourier.fourier_system_reductions import FourierReductions
from flucs.testing import create_test_system

pytestmark = [pytest.mark.gpu]

INPUT_PATH = pl.Path(__file__).parent / "inputs" / "reductions.toml"

################################################################################
# Fixtures
################################################################################

@pytest.fixture(scope="module", params=("single", "double"))
def compiled_reductions(request, test_fourier_system, tmp_path_factory):
    """
    Compile registered real and complex reductions once per precision.
    """
    io_path = tmp_path_factory.mktemp(f"reductions_{request.param}")
    override = {"setup.precision": request.param}

    with create_test_system(INPUT_PATH, io_path, override=override) as (
        _,
        solver,
        system,
    ):
        system.setup()
        solver.timestepper.setup()

        # Initialise reductions
        reductions = FourierReductions(system)

        real_reductions = {}
        complex_reductions = {}

        # Scalar reductions
        real_reductions["scalar"] = reductions.get_reduction(
            reduction_output="scalar",
            functor="NOP_Functor<FLUCS_FLOAT>",
            input_args="const FLUCS_FLOAT*",
            complex_output=False,
        )
        complex_reductions["scalar"] = reductions.get_reduction(
            reduction_output="scalar",
            functor="NOP_Functor<FLUCS_COMPLEX>",
            input_args="const FLUCS_COMPLEX*",
            complex_output=True,
        )

        # One-dimensional reductions
        for reduction_output in (
            "kz", "kx", "ky",
            "kz_cumulative", "kx_cumulative", "ky_cumulative",
        ):
            real_reductions[reduction_output] = reductions.get_reduction(
                reduction_output=reduction_output,
                functor="NOP_Functor<FLUCS_FLOAT>",
                input_args="const FLUCS_FLOAT*",
                complex_output=False,
            )
            complex_reductions[reduction_output] = reductions.get_reduction(
                reduction_output=reduction_output,
                functor="NOP_Functor<FLUCS_COMPLEX>",
                input_args="const FLUCS_COMPLEX*",
                complex_output=True,
            )

        # Two-dimensional reductions
        for reduction_output in ("kzkx", "kzky", "kxky"):
            real_reductions[reduction_output] = reductions.get_reduction(
                reduction_output=reduction_output,
                functor="NOP_Functor<FLUCS_FLOAT>",
                input_args="const FLUCS_FLOAT*",
                complex_output=False,
            )
            complex_reductions[reduction_output] = reductions.get_reduction(
                reduction_output=reduction_output,
                functor="NOP_Functor<FLUCS_COMPLEX>",
                input_args="const FLUCS_COMPLEX*",
                complex_output=True,
            )

        # Shell reductions
        for reduction_output in (
            "kperp", "kmod", "kzkperp",
            "kperp_cumulative", "kmod_cumulative",
        ):
            real_reductions[reduction_output] = reductions.get_reduction(
                reduction_output=reduction_output,
                functor="NOP_Functor<FLUCS_FLOAT>",
                input_args="const FLUCS_FLOAT*",
                complex_output=False,
            )
            complex_reductions[reduction_output] = reductions.get_reduction(
                reduction_output=reduction_output,
                functor="NOP_Functor<FLUCS_COMPLEX>",
                input_args="const FLUCS_COMPLEX*",
                complex_output=True,
            )

        # Explicit shell ranges for checking lower/internal/upper boundaries.
        for reduction_output in ("kperp", "kmod"):
            if reduction_output == "kperp":
                shell_options = {
                    "nkperp": 2, "kperp_min": 0.5, "kperp_max": 2.5,
                }
            else:
                shell_options = {
                    "nkmod": 2, "kmod_min": 0.5, "kmod_max": 2.5,
                }
            real_reductions[f"{reduction_output}_bounded"] = (
                reductions.get_reduction(
                    reduction_output=reduction_output,
                    functor="NOP_Functor<FLUCS_FLOAT>",
                    input_args="const FLUCS_FLOAT*",
                    complex_output=False,
                    **shell_options,
                )
            )
            complex_reductions[f"{reduction_output}_bounded"] = (
                reductions.get_reduction(
                    reduction_output=reduction_output,
                    functor="NOP_Functor<FLUCS_COMPLEX>",
                    input_args="const FLUCS_COMPLEX*",
                    complex_output=True,
                    **shell_options,
                )
            )

        # Compile cupy module
        system.compile_cupy_module()

        yield system, real_reductions, complex_reductions

################################################################################
# Tests
################################################################################

# ------------------------------------------------------------------------------
# Scalar reductions
# ------------------------------------------------------------------------------

@pytest.mark.parametrize(
    "complex_output", (False, True), ids=("real", "complex")
)
def test_scalar_reduction(compiled_reductions, complex_output):
    """
    Sum ky=0 contributions once and positive-ky contributions twice.
    """
    system, real_reductions, complex_reductions = compiled_reductions

    # These are prescribed reduction inputs, not physical solution fields.
    if complex_output:
        contributions = np.zeros(system.half_tuple, dtype=system.complex)
        contributions[0, 0, 0] = 1 + 1j
        contributions[1, 0, 0] = 2 - 1j
        contributions[0, 1, 1] = 3 + 2j
        reduction = complex_reductions["scalar"]
        expected = (1 + 1j) + (2 - 1j) + 2 * (3 + 2j)  # 9 + 4j
    else:
        contributions = np.zeros(system.half_tuple, dtype=system.float)
        contributions[0, 0, 0] = 1
        contributions[1, 0, 0] = 2
        contributions[0, 1, 1] = 3
        reduction = real_reductions["scalar"]
        expected = 1 + 2 + 2 * 3  # 9

    result = reduction(cp.asarray(contributions)).get()

    assert result.dtype == contributions.dtype
    npt.assert_allclose(
        result[0], expected, rtol=0, atol=system.tolerance, equal_nan=False
    )

# ------------------------------------------------------------------------------
# One-dimensional reductions
# ------------------------------------------------------------------------------

@pytest.mark.parametrize("reduction_output", ("kz", "kx", "ky"))
@pytest.mark.parametrize(
    "complex_output", (False, True), ids=("real", "complex")
)
def test_axis_reduction(
    compiled_reductions, reduction_output, complex_output
):
    """
    Retain the requested axis and sum the other two with half-grid weighting.
    """
    system, real_reductions, complex_reductions = compiled_reductions

    # Use the same three prescribed contributions as the scalar check.
    if complex_output:
        contributions = np.zeros(system.half_tuple, dtype=system.complex)
        zero_mode = 1 + 1j
        z_mode = 2 - 1j
        xy_mode = 3 + 2j
        reduction = complex_reductions[reduction_output]
    else:
        contributions = np.zeros(system.half_tuple, dtype=system.float)
        zero_mode = 1
        z_mode = 2
        xy_mode = 3
        reduction = real_reductions[reduction_output]

    contributions[0, 0, 0] = zero_mode
    contributions[1, 0, 0] = z_mode
    contributions[0, 1, 1] = xy_mode

    # Summing over ky doubles its positive modes; retaining ky does not.
    if reduction_output == "kz":
        expected = np.zeros(system.nz, dtype=contributions.dtype)
        expected[0] = zero_mode + 2 * xy_mode
        expected[1] = z_mode
    elif reduction_output == "kx":
        expected = np.zeros(system.nx, dtype=contributions.dtype)
        expected[0] = zero_mode + z_mode
        expected[1] = 2 * xy_mode
    else:
        expected = np.zeros(system.half_ny, dtype=contributions.dtype)
        expected[0] = zero_mode + z_mode
        expected[1] = xy_mode

    result = reduction(cp.asarray(contributions)).get()

    assert result.shape == expected.shape
    assert result.dtype == contributions.dtype
    npt.assert_allclose(
        result, expected, rtol=0, atol=system.tolerance, equal_nan=False
    )


@pytest.mark.parametrize(
    "reduction_output", ("kz_cumulative", "kx_cumulative", "ky_cumulative")
)
@pytest.mark.parametrize(
    "complex_output", (False, True), ids=("real", "complex")
)
def test_axis_cumulative_reduction(
    compiled_reductions, reduction_output, complex_output
):
    """
    Fold signed modes before accumulating towards larger absolute wavenumbers.
    """
    system, real_reductions, complex_reductions = compiled_reductions

    if complex_output:
        contributions = np.zeros(system.half_tuple, dtype=system.complex)
        zero_mode = 1 + 1j
        positive_mode = 2 - 1j
        negative_mode = 3 + 2j
        edge_mode = 4 - 3j
        reduction = complex_reductions[reduction_output]
    else:
        contributions = np.zeros(system.half_tuple, dtype=system.float)
        zero_mode, positive_mode, negative_mode, edge_mode = 1, 2, 3, 4
        reduction = real_reductions[reduction_output]

    contributions[0, 0, 0] = zero_mode
    contributions[1, 2, 0] = positive_mode
    contributions[-2, -3, 1] = negative_mode
    # Raw reduction input: probe the last odd-z mode and unpaired x-Nyquist.
    contributions[system.nz // 2, system.nx // 2, 2] = edge_mode

    if reduction_output == "kz_cumulative":
        expected = np.full(
            system.half_nz, zero_mode, dtype=contributions.dtype
        )
        expected[1:] += positive_mode
        expected[2:] += 2 * negative_mode
        expected[system.nz // 2:] += 2 * edge_mode
    elif reduction_output == "kx_cumulative":
        expected = np.full(
            system.half_nx, zero_mode, dtype=contributions.dtype
        )
        expected[2:] += positive_mode
        expected[3:] += 2 * negative_mode
        expected[system.nx // 2:] += 2 * edge_mode
    else:
        expected = np.full(
            system.half_ny, zero_mode + positive_mode,
            dtype=contributions.dtype,
        )
        expected[1:] += 2 * negative_mode
        expected[2:] += 2 * edge_mode

    result = reduction(cp.asarray(contributions)).get()

    assert result.shape == expected.shape
    assert result.dtype == contributions.dtype
    npt.assert_allclose(
        result, expected, rtol=0, atol=system.tolerance, equal_nan=False
    )


# ------------------------------------------------------------------------------
# Two-dimensional reductions
# ------------------------------------------------------------------------------

@pytest.mark.parametrize("reduction_output", ("kzkx", "kzky", "kxky"))
@pytest.mark.parametrize(
    "complex_output", (False, True), ids=("real", "complex")
)
def test_axis_pair_reduction(
    compiled_reductions, reduction_output, complex_output
):
    """
    Retain two signed axes with the appropriate half-grid weighting.
    """
    system, real_reductions, complex_reductions = compiled_reductions

    if complex_output:
        contributions = np.zeros(system.half_tuple, dtype=system.complex)
        zero_mode, z_mode = 1 + 1j, 2 - 1j
        xy_mode, negative_mode = 3 + 2j, 4 - 3j
        reduction = complex_reductions[reduction_output]
    else:
        contributions = np.zeros(system.half_tuple, dtype=system.float)
        zero_mode, z_mode, xy_mode, negative_mode = 1, 2, 3, 4
        reduction = real_reductions[reduction_output]

    contributions[0, 0, 0] = zero_mode
    contributions[1, 0, 0] = z_mode
    contributions[0, 1, 1] = xy_mode
    contributions[-1, -2, 1] = negative_mode

    if reduction_output == "kzkx":
        expected = np.zeros(
            (system.nz, system.nx), dtype=contributions.dtype
        )
        expected[0, 0] = zero_mode
        expected[1, 0] = z_mode
        expected[0, 1] = 2 * xy_mode
        expected[-1, -2] = 2 * negative_mode
    elif reduction_output == "kzky":
        expected = np.zeros(
            (system.nz, system.half_ny), dtype=contributions.dtype
        )
        expected[0, 0] = zero_mode
        expected[1, 0] = z_mode
        expected[0, 1] = xy_mode
        expected[-1, 1] = negative_mode
    else:
        expected = np.zeros(
            (system.nx, system.half_ny), dtype=contributions.dtype
        )
        expected[0, 0] = zero_mode + z_mode
        expected[1, 1] = xy_mode
        expected[-2, 1] = negative_mode

    result = reduction(cp.asarray(contributions)).get().reshape(expected.shape)

    assert result.dtype == contributions.dtype
    npt.assert_allclose(
        result, expected, rtol=0, atol=system.tolerance, equal_nan=False
    )

# ------------------------------------------------------------------------------
# Shell reductions
# ------------------------------------------------------------------------------

@pytest.mark.parametrize(
    "reduction_output",
    ("kperp", "kmod", "kzkperp", "kperp_cumulative", "kmod_cumulative"),
)
@pytest.mark.parametrize(
    "complex_output", (False, True), ids=("real", "complex")
)
def test_shell_reduction(compiled_reductions, reduction_output, complex_output):
    """
    Place known contributions into perpendicular and isotropic shells.
    """
    system, real_reductions, complex_reductions = compiled_reductions

    if complex_output:
        contributions = np.zeros(system.half_tuple, dtype=system.complex)
        zero_mode, oblique_mode = 1 + 1j, 2 - 1j
        positive_ky_mode, negative_z_mode = 3 + 2j, 4 - 3j
        reduction = complex_reductions[reduction_output]
    else:
        contributions = np.zeros(system.half_tuple, dtype=system.float)
        zero_mode, oblique_mode = 1, 2
        positive_ky_mode, negative_z_mode = 3, 4
        reduction = real_reductions[reduction_output]

    # The input boxes give (kz, kx, ky) spacings (1, 1/2, 1/3).
    contributions[0, 0, 0] = zero_mode
    contributions[1, 1, 0] = oblique_mode       # kperp=1/2, |k|=sqrt(5)/2
    contributions[2, 0, 3] = positive_ky_mode  # kperp=1,   |k|=sqrt(5)
    contributions[-1, -2, 0] = negative_z_mode  # kperp=1,  |k|=sqrt(2)

    # Default shells have width 1/3. Missing negative ky contributes a
    # conjugate, not a second copy of a general complex value.
    if reduction_output in ("kperp", "kperp_cumulative"):
        expected = np.zeros(system.shell_nkperp, dtype=contributions.dtype)
        expected[0] = zero_mode
        expected[1] = oblique_mode
        expected[3] = (
            negative_z_mode + positive_ky_mode + np.conj(positive_ky_mode)
        )
    elif reduction_output in ("kmod", "kmod_cumulative"):
        expected = np.zeros(system.shell_nkmod, dtype=contributions.dtype)
        expected[0] = zero_mode
        expected[3] = oblique_mode
        expected[4] = negative_z_mode
        expected[6] = positive_ky_mode + np.conj(positive_ky_mode)
    else:
        # Retaining kz places the missing partner in the opposite kz row.
        expected = np.zeros(
            (system.nz, system.shell_nkperp), dtype=contributions.dtype
        )
        expected[0, 0] = zero_mode
        expected[1, 1] = oblique_mode
        expected[2, 3] = positive_ky_mode
        expected[-2, 3] = np.conj(positive_ky_mode)
        expected[-1, 3] = negative_z_mode

    if reduction_output.endswith("_cumulative"):
        expected = np.cumsum(expected)

    result = reduction(cp.asarray(contributions)).get()
    if reduction_output == "kzkperp":
        result = result.reshape(expected.shape)

    assert result.shape == expected.shape
    assert result.dtype == contributions.dtype
    npt.assert_allclose(
        result, expected, rtol=0, atol=system.tolerance, equal_nan=False
    )


@pytest.mark.parametrize("reduction_output", ("kperp", "kmod"))
@pytest.mark.parametrize(
    "complex_output", (False, True), ids=("real", "complex")
)
def test_shell_boundaries(
    compiled_reductions, reduction_output, complex_output
):
    """
    Include lower edges and exclude upper edges in half-open shell bins.
    """
    system, real_reductions, complex_reductions = compiled_reductions

    if complex_output:
        contributions = np.zeros(system.half_tuple, dtype=system.complex)
        below_range, lower_edge = 1 + 1j, 2 - 1j
        internal_edge, upper_edge = 3 + 2j, 4 - 3j
        reduction = complex_reductions[f"{reduction_output}_bounded"]
    else:
        contributions = np.zeros(system.half_tuple, dtype=system.float)
        below_range, lower_edge, internal_edge, upper_edge = 1, 2, 3, 4
        reduction = real_reductions[f"{reduction_output}_bounded"]

    # For kz=ky=0, both radii are |kx| = ix/2.
    contributions[0, 0, 0] = below_range    # radius 0
    contributions[0, 1, 0] = lower_edge     # radius 0.5
    contributions[0, 3, 0] = internal_edge  # radius 1.5
    contributions[0, 5, 0] = upper_edge     # radius 2.5

    # The registered bins are [0.5, 1.5) and [1.5, 2.5).
    expected = np.asarray(
        (lower_edge, internal_edge), dtype=contributions.dtype
    )
    result = reduction(cp.asarray(contributions)).get()

    assert result.shape == expected.shape
    assert result.dtype == contributions.dtype
    npt.assert_allclose(
        result, expected, rtol=0, atol=system.tolerance, equal_nan=False
    )
