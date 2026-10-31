"""
Analytical checks of Fourier padding utilities.
"""

import numpy as np
import numpy.testing as npt
import pytest

from flucs import cupy as cp
from flucs.utilities import dealiasing

################################################################################
# Tests
################################################################################

# ------------------------------------------------------------------------------
# Smooth numbers
# ------------------------------------------------------------------------------

@pytest.mark.cpu
@pytest.mark.parametrize(
    ("n", "primes", "expected"),
    [
        pytest.param(0, None, 1, id="lower-boundary"),
        pytest.param(9, None, 9, id="already-smooth"),
        pytest.param(10, None, 12, id="next-smooth"),
        pytest.param(13, None, 16, id="exclude-other-primes"),
        pytest.param(121, [2, 3, 5], 125, id="custom-primes"),
    ],
)
def test_next_smooth_number(n, primes, expected):
    """
    Find the smallest permitted smooth number at or above the input.
    """
    assert dealiasing.next_smooth_number(n, primes) == expected

# ------------------------------------------------------------------------------
# Padded multiplication
# ------------------------------------------------------------------------------

@pytest.mark.parametrize(
    "backend",
    [
        pytest.param("numpy", marks=pytest.mark.cpu),
        pytest.param("cupy", marks=pytest.mark.gpu),
    ],
)
@pytest.mark.parametrize("precision", ("single", "double"))
@pytest.mark.parametrize("direction", ("z", "x", "y"))
def test_padded_multiplication(monkeypatch, backend, precision, direction):
    """
    Squaring 1 + cos(4c) retains its low modes without aliasing cos(8x).
    """

    # Construct f = 1 + cos(4c) on a 12-point grid in each direction.
    # Here c denotes the selected z, x, or y coordinate.
    complex_type = np.complex64 if precision == "single" else np.complex128
    coordinates = 2 * np.pi * np.arange(12) / 12
    z, x, y = np.meshgrid(
        coordinates, coordinates, coordinates, indexing="ij"
    )

    coordinate = {"z": z, "x": x, "y": y}[direction]
    field_real = 1 + np.cos(4 * coordinate)

    field_fourier = np.fft.rfftn(
        field_real, norm="forward"
    ).astype(complex_type)

    # f^2 = 3/2 + 2*cos(4c) + cos(8c)/2.
    # Cropping removes mode 8; without padding it aliases onto mode -4.
    mode = [0, 0, 0]
    mode["zxy".index(direction)] = 4
    positive_mode = tuple(mode)

    expected = np.zeros_like(field_fourier)
    expected[0, 0, 0] = 1.5
    expected[positive_mode] = 1.0

    # Full Fourier axes store both signs; RFFT omits the negative ky mode.
    if direction != "y":
        mode["zxy".index(direction)] = -4
        negative_mode = tuple(mode)
        expected[negative_mode] = 1.0

    # Select correct backend
    if backend == "numpy":
        monkeypatch.setattr(dealiasing, "cp", None)
        input_array = field_fourier
    else:
        input_array = cp.asarray(field_fourier)

    # Compute the padded product
    product = dealiasing.dealiased_multiplication_rfft(
        input_array,
        input_array,
        nz=12,
        nx=12,
        ny=12,
        padded_nz=24,
        padded_nx=24,
        padded_ny=24,
    )
    if backend == "cupy":
        product = product.get()

    # Check every coefficient, including the zeros where aliases would appear.
    tolerance = 64 * np.finfo(field_fourier.real.dtype).eps
    assert product.dtype == np.dtype(complex_type)
    npt.assert_allclose(
        product,
        expected,
        rtol=0,
        atol=tolerance,
        equal_nan=False,
    )
