"""
Tests for the caller-managed cuFFT plan wrapper.
"""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest

from flucs import cupy as cp
from tests.support.support import TEST_SYSTEMS, create_test_solver_system

pytestmark = pytest.mark.solver("FourierSolver")


# Exercise both R2C layouts, including the Nyquist planes on an even grid
GRID_SIZES = (
    pytest.param((7, 9, 11), id="odd"),
    pytest.param((8, 10, 12), id="even-nyquist"),
)
WORKSPACE_GRID_SIZE = (8, 10, 12)
BATCH_SIZE = 2
FFT_WRAPPERS = ("flucs", "cupy")


def _create_fourier_system(
    tmp_path,
    precision,
    fft_wrapper,
    grid_size,
):
    """
    Create a Fourier system with its selected cuFFT interface configured.
    """
    _, _, system = create_test_solver_system(
        tmp_path,
        TEST_SYSTEMS["FourierSolver"],
        precision=precision,
        updates={
            "dimensions": dict(zip(("nz", "nx", "ny"), grid_size, strict=True)),
            "setup": {
                "fft_wrapper": fft_wrapper,
                "linear": True,
            },
        },
    )

    # FFT types are configured during the normal public setup stage
    system.setup()
    return system


def _close_plans(plans):
    """
    Finish queued transforms before releasing their plans and workspaces.
    """
    cp.cuda.get_current_stream().synchronize()
    for plan in plans:
        plan.close()
        assert plan.closed
        assert plan.work_area is None


###############################################################################
# CPU tests
###############################################################################

# None


###############################################################################
# GPU tests
###############################################################################


@pytest.mark.gpu
@pytest.mark.parametrize("fft_wrapper", FFT_WRAPPERS)
@pytest.mark.parametrize(
    "in_place",
    (False, True),
    ids=("out-of-place", "in-place"),
)
@pytest.mark.parametrize("grid_size", GRID_SIZES)
def test_real_fft_round_trip(
    tmp_path,
    precision,
    fft_wrapper,
    in_place,
    grid_size,
):
    """
    Both backends perform batched real transforms in either memory layout.
    """
    from flucs.utilities.flucs_plan_nd import allocate_shared_work_area

    system = _create_fourier_system(
        tmp_path,
        precision,
        fft_wrapper,
        grid_size,
    )

    # Create a matched forward and inverse pair through the production caller
    plans = [
        system.create_standard_real_cufft_plan(
            fft_type,
            BATCH_SIZE,
            in_place=in_place,
        )
        for fft_type in ("r2c", "c2r")
    ]
    plan_r2c, plan_c2r = plans

    try:
        allocate_shared_work_area(plans)
        assert plan_r2c.use_cupy is (fft_wrapper == "cupy")
        assert plan_c2r.use_cupy is (fft_wrapper == "cupy")

        # Use nontrivial deterministic data throughout every transform axis
        rng = np.random.default_rng(314159)
        real_cpu = rng.standard_normal((BATCH_SIZE, *grid_size)).astype(
            precision.float_type
        )
        expected_fourier = np.fft.rfftn(
            real_cpu,
            axes=(-3, -2, -1),
        ).astype(precision.complex_type)

        if in_place:
            # One allocation also exposes cuFFT's padded real view
            fourier_gpu = cp.zeros(
                (BATCH_SIZE, *system.half_tuple),
                dtype=system.complex,
            )
            real_gpu = fourier_gpu.view(system.float).reshape(
                BATCH_SIZE,
                system.nz,
                system.nx,
                2 * system.half_ny,
            )
            real_gpu[..., : system.ny] = cp.asarray(real_cpu)
            forward_input = fourier_gpu
            forward_output = fourier_gpu
        else:
            real_gpu = cp.asarray(real_cpu)
            fourier_gpu = cp.empty(
                (BATCH_SIZE, *system.half_tuple),
                dtype=system.complex,
            )
            forward_input = real_gpu
            forward_output = fourier_gpu

        # The forward transform agrees with an independent NumPy result
        plan_r2c.fft(
            forward_input,
            forward_output,
            system.CUFFT_FORWARD,
        )
        npt.assert_allclose(
            cp.asnumpy(fourier_gpu),
            expected_fourier,
            rtol=system.tolerance,
            atol=system.tolerance,
        )

        if in_place:
            inverse_output = fourier_gpu
        else:
            real_gpu = cp.empty_like(real_gpu)
            inverse_output = real_gpu

        # cuFFT inverses are unnormalised, so divide out the grid volume
        plan_c2r.fft(
            fourier_gpu,
            inverse_output,
            system.CUFFT_INVERSE,
        )
        recovered = real_gpu[..., : system.ny] / np.prod(grid_size)
        npt.assert_allclose(
            cp.asnumpy(recovered),
            real_cpu,
            rtol=system.tolerance,
            atol=system.tolerance,
        )
    finally:
        _close_plans(plans)


@pytest.mark.gpu
@pytest.mark.parametrize("fft_wrapper", FFT_WRAPPERS)
def test_shared_fft_workspace(tmp_path, precision, fft_wrapper):
    """
    Production-style plans use the workspace ownership of their backend.
    """
    from flucs.utilities.flucs_plan_nd import allocate_shared_work_area

    system = _create_fourier_system(
        tmp_path,
        precision,
        fft_wrapper,
        WORKSPACE_GRID_SIZE,
    )

    # Match the four plans needed by in-place phase-shift dealiasing
    plans = [
        system.create_standard_real_cufft_plan(
            fft_type,
            BATCH_SIZE,
            in_place=in_place,
        )
        for in_place in (False, True)
        for fft_type in ("r2c", "c2r")
    ]

    try:
        workspace = allocate_shared_work_area(plans)
        required_size = max(plan.required_work_area_size for plan in plans)

        if fft_wrapper == "cupy":
            # CuPy plans retain their private workspaces and are skipped
            assert workspace is None
            assert all(plan.use_cupy for plan in plans)
        elif required_size == 0:
            # Some cuFFT plans do not require temporary storage
            assert workspace is None
        else:
            # Custom plans all retain the one largest necessary allocation
            assert workspace.size == required_size
            for plan in plans:
                assert plan.work_area is workspace
                assert plan.work_area_ptr == workspace.ptr
                assert plan.work_area_size == workspace.size
    finally:
        _close_plans(plans)
