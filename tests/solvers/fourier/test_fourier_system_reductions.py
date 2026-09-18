"""
Tests for the shared FourierSystem reduction machinery.

TODO: Define the contract for complex Cartesian reductions that sum over the
R2C ky half-axis. Doubling positive-ky values matches the current
implementation, but a general complex quantity requires the conjugate value at
the reflected (kz, kx) mode. All current production callers request real
outputs, so the complex tests retain the existing behaviour for now.
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


# Mixed grid parity and unrelated box lengths expose branch, axis, and
# scale mix-ups
GRID_SIZE = (13, 14, 19)
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
SHELL_DIMENSIONS = {
    "kzkperp": ("kz", "kperp"),
    "kperp": ("kperp",),
    "kmod": ("kmod",),
    "kperp_cumulative": ("kperp",),
    "kmod_cumulative": ("kmod",),
}

# Each row declares (kz mode, kx mode, ky mode, kperp bin, kmod bin), so tests
# can verify that the declared bins match the expected radii.
SHELL_MODES = (
    (+0, +0, +0, +0, +0),
    (+1, +0, +0, +0, +1),
    (-1, +0, +0, +0, +1),
    (+0, +1, +0, +1, +1),
    (+0, -1, +0, +1, +1),
    (+1, +1, +1, +1, +2),
    (-1, -1, +1, +1, +2),
    (+2, +1, +2, +2, +4),
    (-2, -1, +2, +2, +4),
    (+3, +2, +1, +2, +6),
    (-3, -2, +1, +2, +6),
    (+1, +3, +4, +5, +5),
    (-1, -3, +4, +5, +5),
    (+4, +1, +3, +3, +8),
    (-4, -1, +3, +3, +8),
    (+2, +5, +1, +6, +7),
    (-2, -5, +1, +6, +7),
    (+5, +2, +6, +6, +11),
    (-5, -2, +6, +6, +11),
    (+3, +6, +7, +10, +11),
    (-3, -6, +7, +10, +11),
    (+5, +6, +8, +11, +14),
    (-5, -6, +8, +11, +14),
    (+4, +5, +7, +9, +12),
    (-4, -5, +7, +9, +12),
)

# Sparse modes prescribe known contributions to every cumulative direction.
# The x-Nyquist probe exercises the unpaired even-grid branch, while z is odd.
CUMULATIVE_MODES = (
    (+0, +0, +0),
    (+1, +2, +0),
    (-1, -3, +1),
    (+4, -7, +2),
    (-5, +5, +7),
    (+6, -6, +9),
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


def _cumulative_data_and_expected(system, dtype):
    """
    Construct sparse modes with prescribed absolute-wavenumber contributions.
    """
    data = np.zeros(system.half_tuple, dtype=dtype)
    contributions = {
        "kz": np.zeros(system.half_nz, dtype=dtype),
        "kx": np.zeros(system.half_nx, dtype=dtype),
        "ky": np.zeros(system.half_ny, dtype=dtype),
    }
    scalar = dtype.type(0)

    for index, (mz, mx, my) in enumerate(CUMULATIVE_MODES):
        value = _probe_value(index, dtype)
        data[mz % system.nz, mx % system.nx, my] = value

        # Positive ky represents both signs in the real-to-complex layout
        # Mirrors the current CUDA kernel implementation for complex outputs
        physical_value = value if my == 0 else 2 * value
        contributions["kz"][abs(mz)] += physical_value
        contributions["kx"][abs(mx)] += physical_value
        contributions["ky"][my] += physical_value
        scalar += physical_value

    expected = {
        f"{dimension}_cumulative": np.cumsum(values, dtype=dtype)
        for dimension, values in contributions.items()
    }

    return data, expected, scalar


def _probe_value(index, dtype):
    """
    Return a distinct, exactly representable value for one shell probe.
    """
    real = 1 + index % 7
    if np.issubdtype(dtype, np.complexfloating):
        imaginary = (3 * index) % 9 - 4
        return dtype.type(real + 0.5j * imaginary)
    return dtype.type(real)


def _assert_declared_shell(radius, expected_bin, count, minimum, maximum):
    """
    Ensure the physical probe radius lies inside its declared test bin.
    """
    width = (maximum - minimum) / count
    lower = minimum + expected_bin * width
    upper = lower + width
    assert lower <= radius < upper


def _shell_data_and_expected(system, dtype):
    """
    Construct prescribed shell probes and their independently declared sums.
    """
    data = np.zeros(system.half_tuple, dtype=dtype)
    expected_kzkperp = np.zeros(
        (system.nz, system.shell_nkperp),
        dtype=dtype,
    )
    expected_kmod = np.zeros(system.shell_nkmod, dtype=dtype)

    Lz = system.input["dimensions.Lz"]
    Lx = system.input["dimensions.Lx"]
    Ly = system.input["dimensions.Ly"]

    for index, (mz, mx, my, kperp_bin, kmod_bin) in enumerate(SHELL_MODES):
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
            system.shell_nkperp,
            system.shell_kperp_min,
            system.shell_kperp_max,
        )
        _assert_declared_shell(
            kmod,
            kmod_bin,
            system.shell_nkmod,
            system.shell_kmod_min,
            system.shell_kmod_max,
        )

        # Shell reductions reconstruct the missing negative-ky coefficient at
        # the reflected (kz, kx) location.
        expected_kzkperp[iz, kperp_bin] += value
        if my > 0:
            expected_kzkperp[-mz % system.nz, kperp_bin] += np.conj(value)

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
        # Construct prescribed contributions on signed Fourier modes
        data_cpu, expected_outputs, scalar = _cumulative_data_and_expected(
            system,
            dtype,
        )

        # Initialise GPU array
        data = cp.asarray(data_cpu)

        for output, dimension in CUMULATIVE_DIMENSIONS.items():
            # Perform cumulative reduction
            expected = expected_outputs[output]
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
            "kperp_cumulative": np.cumsum(expected_kperp, dtype=dtype),
            "kmod_cumulative": np.cumsum(expected_kmod, dtype=dtype),
        }

        # Initialise GPU array
        data_gpu = cp.asarray(data)

        for output, expected in expected_outputs.items():
            result = cp.asnumpy(
                compiled_reductions.functions[output_type, output](data_gpu)
            ).reshape(expected.shape)

            # The default shell metadata describes the actual reduction result
            dimensions = compiled_reductions.reductions.get_dimensions(output)

            assert tuple(dimensions) == SHELL_DIMENSIONS[output]
            assert result.shape == tuple(
                coordinate.size for coordinate in dimensions.values()
            )

            assert result.dtype == dtype
            npt.assert_allclose(
                result,
                expected,
                rtol=system.tolerance,
                atol=system.tolerance,
                err_msg=f"{output_type} {output} reduction",
            )
