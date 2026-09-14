"""
Tests for Fourier dealiasing utilities.
"""

import numpy as np
import numpy.testing as npt
import pytest

from flucs import cupy as cp
from flucs.utilities import dealiasing
from flucs.utilities.dealiasing import (
    dealiased_multiplication_rfft,
    next_smooth_number,
)

pytestmark = pytest.mark.core


@pytest.mark.parametrize(
    ("n", "primes", "expected"),
    [
        pytest.param(0, None, 1, id="below-one"),
        pytest.param(9, None, 9, id="exact-three-smooth"),
        pytest.param(10, None, 12, id="next-three-smooth"),
        pytest.param(121, [2, 3, 5], 125, id="next-five-smooth"),
    ],
)
def test_next_smooth_number(n, primes, expected):
    assert next_smooth_number(n, primes) == expected


def _low_mode_fields(shape):
    """
    Return fields whose product is resolved on the original grid.
    """
    # Unpack tuple
    nz, nx, ny = shape

    # Create meshgrid
    z, x, y = np.meshgrid(
        np.arange(nz),
        np.arange(nx),
        np.arange(ny),
        indexing="ij",
    )

    # Make dummy data
    first = 1.0 + np.cos(2 * np.pi * x / nx)
    second = 0.5 + np.sin(2 * np.pi * y / ny)
    if nz > 1:
        second = second + 0.25 * np.cos(2 * np.pi * z / nz)

    return first, second


@pytest.mark.parametrize(
    ("backend_name", "shape", "dimensions"),
    [
        pytest.param(
            "numpy",
            (5, 7, 9),
            {},
            id="numpy-inferred-3d",
        ),
        pytest.param(
            "numpy",
            (1, 8, 10),
            {"nz": 1, "nx": 8, "ny": 10},
            id="numpy-explicit-2d-even",
        ),
        pytest.param(
            "cupy",
            (5, 7, 9),
            {},
            marks=pytest.mark.gpu,
            id="cupy-inferred-3d",
        ),
    ],
)
def test_dealiased_multiplication(
    monkeypatch,
    precision,
    backend_name,
    shape,
    dimensions,
):
    """
    Dealiased products agree across supported precisions and array backends.
    """

    # Create low-mode fields
    first, second = _low_mode_fields(shape)
    first = first.astype(precision.float_type)
    second = second.astype(precision.float_type)

    # Compute their Fourier components and the expected product
    first_rfft = np.fft.rfftn(first, norm="forward").astype(
        precision.complex_type
    )
    second_rfft = np.fft.rfftn(second, norm="forward").astype(
        precision.complex_type
    )
    expected = np.fft.rfftn(first * second, norm="forward").astype(
        precision.complex_type
    )

    # Select the requested backend without doing GPU work during collection
    if backend_name == "numpy":
        monkeypatch.setattr(dealiasing, "cp", None)
        inputs = (first_rfft, second_rfft)
    else:
        inputs = (cp.asarray(first_rfft), cp.asarray(second_rfft))

    # Exercise the same public multiplication contract for either backend
    result = dealiased_multiplication_rfft(
        *inputs,
        **dimensions,
    )

    # Return to the host before applying the shared precision policy
    if backend_name == "numpy":
        assert isinstance(result, np.ndarray)
        result_host = result
    else:
        assert isinstance(result, cp.ndarray)
        result_host = cp.asnumpy(result)

    assert result_host.dtype == np.dtype(precision.complex_type)
    npt.assert_allclose(
        result_host,
        expected,
        rtol=precision.tolerance,
        atol=precision.tolerance,
    )
