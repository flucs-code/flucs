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

pytestmark = [pytest.mark.gpu, pytest.mark.slow]

INPUT_PATH = pl.Path(__file__).parent / "inputs" / "reductions.toml"

################################################################################
# Fixtures
################################################################################

@pytest.fixture(scope="module", params=("single", "double"))
def compiled_reductions(request, test_fourier_system, tmp_path_factory):
    """
    Compile real and complex scalar reductions once per precision.
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

        # Register both variants before compiling their shared CUDA module.
        reductions = FourierReductions(system)

        # Scalar reductions
        real_scalar = reductions.get_reduction(
            reduction_output="scalar",
            functor="NOP_Functor<FLUCS_FLOAT>",
            input_args="const FLUCS_FLOAT*",
            complex_output=False,
        )
        complex_scalar = reductions.get_reduction(
            reduction_output="scalar",
            functor="NOP_Functor<FLUCS_COMPLEX>",
            input_args="const FLUCS_COMPLEX*",
            complex_output=True,
        )

        # Compile cupy module
        system.compile_cupy_module()

        yield system, real_scalar, complex_scalar

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
    system, real_scalar, complex_scalar = compiled_reductions

    # These are prescribed reduction inputs, not physical solution fields.
    if complex_output:
        contributions = np.zeros(system.half_tuple, dtype=system.complex)
        contributions[0, 0, 0] = 1 + 1j
        contributions[1, 0, 0] = 2 - 1j
        contributions[0, 1, 1] = 3 + 2j
        reduction = complex_scalar
        expected = (1 + 1j) + (2 - 1j) + 2 * (3 + 2j)  # 9 + 4j
    else:
        contributions = np.zeros(system.half_tuple, dtype=system.float)
        contributions[0, 0, 0] = 1
        contributions[1, 0, 0] = 2
        contributions[0, 1, 1] = 3
        reduction = real_scalar
        expected = 1 + 2 + 2 * 3  # 9

    result = reduction(cp.asarray(contributions)).get()

    assert result.dtype == contributions.dtype
    npt.assert_allclose(
        result[0], expected, rtol=0, atol=system.tolerance, equal_nan=False
    )
