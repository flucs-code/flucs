"""
Tests for the shared FourierSystem reduction machinery.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import numpy.testing as npt
import pytest

from flucs import cupy as cp
from flucs.solvers.fourier.fourier_system_reductions import FourierReductions
from tests.support.support import (
    TEST_PRECISIONS,
    TEST_SYSTEMS,
    create_test_solver_system,
)

pytestmark = pytest.mark.solver("FourierSolver")


# An asymmetric grid and unrelated box lengths expose axis and scale mix-ups
GRID_SIZE = (12, 14, 18)
BOX_SIZE = (3.7, 5.2, 6.9)

CARTESIAN_AXES = {
    "scalar": (),
    "kz": (0,),
    "kx": (1,),
    "ky": (2,),
    "kzkx": (0, 1),
    "kzky": (0, 2),
    "kxky": (1, 2),
}
CUMULATIVE_DIMENSIONS = {
    "kz_cumulative": "kz",
    "kx_cumulative": "kx",
    "ky_cumulative": "ky",
}

KPERP_SHELLS = {
    "nkperp": 8,
    "kperp_min": -0.4,
    "kperp_max": 9.2,
}
KMOD_SHELLS = {
    "nkmod": 8,
    "kmod_min": -0.6,
    "kmod_max": 13.0,
}

# Each row declares (kz mode, kx mode, ky mode, kperp bin, kmod bin).
# None marks a mode outside the corresponding test shell range.
SHELL_PROBES = (
    (+0, +0, +0, +0, +0),
    (+1, +0, +0, +0, +1),
    (-1, +0, +0, +0, +1),
    (+0, +1, +0, +1, +1),
    (+0, -1, +0, +1, +1),
    (+0, +0, +1, +1, +0),
    (+1, +1, +1, +1, +1),
    (-1, -1, +1, +1, +1),
    (+2, +1, +2, +2, +2),
    (-2, -1, +2, +2, +2),
    (+3, +2, +1, +2, +3),
    (-3, -2, +1, +2, +3),
    (+1, +3, +4, +4, +3),
    (-1, -3, +4, +4, +3),
    (+4, +1, +3, +2, +4),
    (-4, -1, +3, +2, +4),
    (+2, +5, +1, +5, +4),
    (-2, -5, +1, +5, +4),
    (+5, +2, +6, +5, +6),
    (-5, -2, +6, +5, +6),
    (+3, +6, +7, None, +6),
    (-3, -6, +7, None, +6),
    (+5, +6, +8, None, None),
    (-5, -6, +8, None, None),
    (+4, +5, +7, +7, +6),
    (-4, -5, +7, +7, +6),
)


def _system_updates():
    """
    Return the common reduction-test geometry and minimal runtime options.
    """
    nz, nx, ny = GRID_SIZE
    Lz, Lx, Ly = BOX_SIZE
    return {
        "dimensions": {
            "nz": nz,
            "nx": nx,
            "ny": ny,
            "Lz": Lz,
            "Lx": Lx,
            "Ly": Ly,
        },
        "setup": {
            "linear": True,
            "check_linear_matrix": False,
        },
        "forcing": {"method": ""},
    }


def _reduction_kwargs(output):
    """
    Return the explicitly controlled shell grid for one reduction.
    """
    if output in ("kzkperp", "kperp", "kperp_cumulative"):
        return KPERP_SHELLS
    if output in ("kmod", "kmod_cumulative"):
        return KMOD_SHELLS
    return {}


def _dense_components(system, complex_output):
    """
    Construct two separable components with distinct patterns on every axis.
    """
    iz = np.arange(system.nz)
    ix = np.arange(system.nx)
    iy = np.arange(system.half_ny)

    first = (
        1 + (3 * iz) % 7,
        1 + (5 * ix) % 11,
        1 + (7 * iy) % 13,
    )
    second = (
        1 + (5 * iz) % 9,
        1 + (3 * ix) % 8,
        1 + (2 * iy) % 7,
    )

    # Nyquist planes are not solved by the Fourier systems. Keeping them zero
    # gives the dense test data the same contract as production field arrays.
    full_sizes = (system.nz, system.nx, system.ny)
    for factors in (first, second):
        for factor, full_size in zip(factors, full_sizes, strict=True):
            if full_size % 2 == 0:
                factor[full_size // 2] = 0

    coefficients = (
        (1.0 + 0.5j, -0.25 + 0.75j) if complex_output else (1.0, -0.25)
    )
    dtype = system.complex if complex_output else system.float

    return tuple(
        (
            dtype(coefficient),
            *(np.asarray(factor, dtype=system.float) for factor in factors),
        )
        for coefficient, factors in zip(
            coefficients,
            (first, second),
            strict=True,
        )
    )


def _dense_data(components, dtype):
    """
    Materialise the sum of the separable components used by the GPU kernels.
    """
    shape = tuple(factor.size for factor in components[0][1:])
    data = np.zeros(shape, dtype=dtype)

    for coefficient, z_factor, x_factor, y_factor in components:
        data += (
            coefficient
            * z_factor[:, np.newaxis, np.newaxis]
            * x_factor[np.newaxis, :, np.newaxis]
            * y_factor[np.newaxis, np.newaxis, :]
        )

    return data


def _analytic_cartesian_reduction(components, retained_axes, dtype):
    """
    Reduce separable factors analytically without summing the dense 3D data.
    """
    output_shape = tuple(components[0][axis + 1].size for axis in retained_axes)
    expected = np.zeros(output_shape or (1,), dtype=dtype)

    for coefficient, *factors in components:
        contribution = np.asarray(coefficient, dtype=dtype)

        for axis, factor in enumerate(factors):
            if axis in retained_axes:
                retained_index = retained_axes.index(axis)
                reshape = [1] * len(retained_axes)
                reshape[retained_index] = factor.size
                contribution = contribution * factor.reshape(reshape)
            elif axis == 2:
                contribution = contribution * (
                    factor[0] + 2 * np.sum(factor[1:], dtype=dtype)
                )
            else:
                contribution = contribution * np.sum(factor, dtype=dtype)

        expected += contribution

    return expected


def _expected_cumulative(spectrum, dimension, full_size):
    """
    Fold one known Cartesian spectrum onto absolute wavenumbers and integrate.
    """
    if dimension == "ky":
        folded = spectrum.copy()
        folded[1:] *= 2
        return np.cumsum(folded, dtype=spectrum.dtype)

    paired_size = (full_size - 1) // 2
    folded = np.empty(full_size // 2 + 1, dtype=spectrum.dtype)
    folded[0] = spectrum[0]
    folded[1 : paired_size + 1] = (
        spectrum[1 : paired_size + 1] + spectrum[-1 : -paired_size - 1 : -1]
    )

    if full_size % 2 == 0:
        folded[-1] = spectrum[paired_size + 1]

    return np.cumsum(folded, dtype=spectrum.dtype)


def _probe_value(index, dtype):
    """
    Return a distinct, exactly representable value for one shell probe.
    """
    real = 1 + index % 7
    if np.issubdtype(dtype, np.complexfloating):
        imaginary = (3 * index) % 9 - 4
        return dtype.type(real + 0.5j * imaginary)
    return dtype.type(real)


def _assert_declared_shell(radius, expected_bin, options, prefix):
    """
    Ensure the physical probe radius lies inside its declared test bin.
    """
    count = options[f"n{prefix}"]
    minimum = options[f"{prefix}_min"]
    maximum = options[f"{prefix}_max"]

    if expected_bin is None:
        assert radius < minimum or radius >= maximum
        return

    width = (maximum - minimum) / count
    lower = minimum + expected_bin * width
    upper = lower + width
    assert lower < radius < upper


def _shell_data_and_expected(system, dtype):
    """
    Construct prescribed shell probes and their independently declared sums.
    """
    data = np.zeros(system.half_tuple, dtype=dtype)
    expected_kzkperp = np.zeros(
        (system.nz, KPERP_SHELLS["nkperp"]),
        dtype=dtype,
    )
    expected_kmod = np.zeros(KMOD_SHELLS["nkmod"], dtype=dtype)

    Lz = system.input["dimensions.Lz"]
    Lx = system.input["dimensions.Lx"]
    Ly = system.input["dimensions.Ly"]

    for index, (mz, mx, my, kperp_bin, kmod_bin) in enumerate(SHELL_PROBES):
        iz = mz % system.nz
        ix = mx % system.nx
        value = _probe_value(index, dtype)
        data[iz, ix, my] = value

        # Confirm that the declared cases remain valid if the fixture changes
        kz = 2 * np.pi * mz / Lz
        kx = 2 * np.pi * mx / Lx
        ky = 2 * np.pi * my / Ly
        kperp = np.sqrt(kx**2 + ky**2)
        kmod = np.sqrt(kz**2 + kx**2 + ky**2)
        _assert_declared_shell(
            kperp,
            kperp_bin,
            KPERP_SHELLS,
            "kperp",
        )
        _assert_declared_shell(kmod, kmod_bin, KMOD_SHELLS, "kmod")

        # Shell reductions reconstruct the missing negative-ky coefficient at
        # the reflected (kz, kx) location.
        if kperp_bin is not None:
            expected_kzkperp[iz, kperp_bin] += value
            if my > 0:
                expected_kzkperp[-mz % system.nz, kperp_bin] += np.conj(value)

        if kmod_bin is not None:
            expected_kmod[kmod_bin] += value
            if my > 0:
                expected_kmod[kmod_bin] += np.conj(value)

    expected_kperp = np.sum(expected_kzkperp, axis=0, dtype=dtype)
    return data, expected_kzkperp, expected_kperp, expected_kmod


@pytest.fixture(
    scope="module",
    params=TEST_PRECISIONS,
    ids=lambda precision: precision.name,
)
def compiled_reductions(request, tmp_path_factory):
    """
    Compile all reduction variants once for each supported precision.
    """
    precision = request.param
    io_path = tmp_path_factory.mktemp(f"fourier-reductions-{precision.name}")
    _, solver, system = create_test_solver_system(
        io_path,
        TEST_SYSTEMS["FourierSolver"],
        precision=precision,
        updates=_system_updates(),
    )
    system.setup()
    solver.timestepper.setup()

    reductions = FourierReductions(system)
    reduction_functions = {}
    outputs = (
        *CARTESIAN_AXES,
        *CUMULATIVE_DIMENSIONS,
        "kzkperp",
        "kperp",
        "kmod",
        "kperp_cumulative",
        "kmod_cumulative",
    )

    # Every output type is registered before the single CUDA compilation
    for output_type, cuda_type in (
        ("real", "FLUCS_FLOAT"),
        ("complex", "FLUCS_COMPLEX"),
    ):
        for output in outputs:
            reduction_functions[output_type, output] = reductions.get_reduction(
                reduction_output=output,
                functor=f"NOP_Functor<{cuda_type}>",
                input_args=f"const {cuda_type}*",
                complex_output=output_type == "complex",
                **_reduction_kwargs(output),
            )

    system.compile_cupy_module()

    try:
        yield SimpleNamespace(
            system=system,
            reductions=reductions,
            functions=reduction_functions,
        )
    finally:
        system.kernels.unbind()


###############################################################################
# CPU tests
###############################################################################

# None


###############################################################################
# GPU tests
###############################################################################


@pytest.mark.gpu
def test_cartesian_reductions_use_all_axes(compiled_reductions):
    """
    Dense analytic fields exercise every Cartesian reduction and output type.
    """
    system = compiled_reductions.system

    for output_type, dtype in (
        ("real", np.dtype(system.float)),
        ("complex", np.dtype(system.complex)),
    ):
        # Construct test data
        components = _dense_components(
            system,
            complex_output=output_type == "complex",
        )
        data_cpu = _dense_data(components, dtype)

        # Initialise GPU array
        data = cp.asarray(data_cpu)

        for output, retained_axes in CARTESIAN_AXES.items():
            expected = _analytic_cartesian_reduction(
                components,
                retained_axes,
                dtype,
            )
            result = cp.asnumpy(
                compiled_reductions.functions[output_type, output](data)
            ).reshape(expected.shape)

            # Check that the reduction matches the analytic result
            assert result.dtype == dtype
            npt.assert_allclose(
                result,
                expected,
                rtol=system.tolerance,
                atol=system.tolerance,
                err_msg=f"{output_type} {output} reduction",
            )


@pytest.mark.gpu
def test_cartesian_cumulative_reductions(compiled_reductions):
    """
    Cartesian cumulative reductions fold signed modes before integration.
    """
    system = compiled_reductions.system

    for output_type, dtype in (
        ("real", np.dtype(system.float)),
        ("complex", np.dtype(system.complex)),
    ):
        # Construct test data
        components = _dense_components(
            system,
            complex_output=output_type == "complex",
        )
        data_cpu = _dense_data(components, dtype)

        # Initialise GPU array
        data = cp.asarray(data_cpu)

        # Get a scalar reference for final cumulative value
        scalar = _analytic_cartesian_reduction(components, (), dtype)[0]

        for output, dimension in CUMULATIVE_DIMENSIONS.items():

            # Get axis and spectrum
            axis = {"kz": 0, "kx": 1, "ky": 2}[dimension]
            spectrum = _analytic_cartesian_reduction(
                components,
                (axis,),
                dtype,
            )
            full_size = {
                "kz": system.nz,
                "kx": system.nx,
                "ky": system.ny,
            }[dimension]

            # Perform cumulative reduction
            expected = _expected_cumulative(
                spectrum,
                dimension,
                full_size,
            )
            result = cp.asnumpy(
                compiled_reductions.functions[output_type, output](data)
            )

            # Cumulative output coordinates are ordered absolute wavenumbers
            dimensions = compiled_reductions.reductions.get_dimensions(output)
            assert len(dimensions) == 1

            coordinate = next(iter(dimensions.values()))
            assert coordinate.shape == result.shape
            assert coordinate[0] == 0
            assert np.all(np.diff(coordinate) > 0)

            # Check the results match
            assert result.dtype == dtype
            npt.assert_allclose(
                result,
                expected,
                rtol=system.tolerance,
                atol=system.tolerance,
                err_msg=f"{output_type} {output} reduction",
            )
            npt.assert_allclose(
                result[-1],
                scalar,
                rtol=system.tolerance,
                atol=system.tolerance,
            )


@pytest.mark.gpu
def test_shell_and_shell_cumulative_reductions(compiled_reductions):
    """
    Prescribed probes exercise shell bins, collisions, and conjugate recovery.
    """
    system = compiled_reductions.system

    for output_type, dtype in (
        ("real", np.dtype(system.float)),
        ("complex", np.dtype(system.complex)),
    ):  
        # Construct test data
        data, expected_kzkperp, expected_kperp, expected_kmod = (
            _shell_data_and_expected(system, dtype)
        )
        expected_outputs = {
            "kzkperp": expected_kzkperp,
            "kperp": expected_kperp,
            "kmod": expected_kmod,
            "kperp_cumulative": np.cumsum(expected_kperp,dtype=dtype),
            "kmod_cumulative": np.cumsum(expected_kmod, dtype=dtype),
        }

        # Initialise GPU array
        data_gpu = cp.asarray(data)

        for output, expected in expected_outputs.items():
            result = cp.asnumpy(
                compiled_reductions.functions[output_type, output](data_gpu)
            ).reshape(expected.shape)

            assert result.dtype == dtype
            npt.assert_allclose(
                result,
                expected,
                rtol=system.tolerance,
                atol=system.tolerance,
                err_msg=f"{output_type} {output} reduction",
            )
