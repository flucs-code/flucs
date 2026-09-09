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
    ("shape", "dimensions"),
    [
        pytest.param((5, 7, 9), {}, id="inferred-3d"),
        pytest.param(
            (1, 8, 10),
            {"nz": 1, "nx": 8, "ny": 10}, # 2D field
            id="explicit-2d-even",
        ),
    ],
)
def test_dealiased_multiplication_numpy(monkeypatch, shape, dimensions):
    # Create low-mode fields
    first, second = _low_mode_fields(shape)

    # Compute their Fourier components and the expected product
    first_rfft = np.fft.rfftn(first, norm="forward")
    second_rfft = np.fft.rfftn(second, norm="forward")
    expected = np.fft.rfftn(first * second, norm="forward")

    # Patch cupy to None to ensure that the numpy implementation is used
    monkeypatch.setattr(dealiasing, "cp", None)
    result = dealiased_multiplication_rfft(
        first_rfft,
        second_rfft,
        **dimensions,
    )

    # Check
    npt.assert_allclose(result, expected, atol=1e-12)


@pytest.mark.gpu
def test_dealiased_multiplication_cupy():
    # Create low-mode fields
    first, second = _low_mode_fields((5, 7, 9))

    # Compute their Fourier components on the GPU and the expected product
    first_gpu = cp.asarray(np.fft.rfftn(first, norm="forward"))
    second_gpu = cp.asarray(np.fft.rfftn(second, norm="forward"))
    expected = np.fft.rfftn(first * second, norm="forward")

    # Compute using the GPU implementation
    result = dealiased_multiplication_rfft(first_gpu, second_gpu)

    # Check
    npt.assert_allclose(cp.asnumpy(result), expected, atol=1e-12)
