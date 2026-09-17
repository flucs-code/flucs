"""
Tests for forcing behavior shared by Fourier systems.
"""

import numpy as np
import numpy.testing as npt
import pytest

from flucs import cupy as cp
from tests.support.support import (
    SINGLE_PRECISION,
    create_test_solver_system,
)

pytestmark = pytest.mark.solver("FourierSolver")


# Unequal, non-integral box lengths separate the three fundamental wavenumbers
DIMENSIONS = {
    "nz": 6,
    "nx": 8,
    "ny": 10,
    "Lz": 5.3,
    "Lx": 4.7,
    "Ly": 6.1,
}

# Use the existing OU forcing to test
OU_FORCING = {
    "method": "ornstein_uhlenbeck",
    "amplitude": [0.1, 0.2, 0.3],
    "corr_time": 1.0,
    "rand_seed": 1729,
}

# The isotropic shell 1.0 < kmod < 1.4 contains only the six physical
# fundamental modes. Only positive ky is stored in the Fourier half-grid.
ISOTROPIC_MODES = (
    (+0, +0, +1),
    (+0, +1, +0),
    (+0, -1, +0),
    (+1, +0, +0),
    (-1, +0, +0),
)

# Combining 1.1 < |kz| < 1.25 with 1.0 < kperp < 1.4 selects
# kz = +/-1 and each stored perpendicular fundamental.
ANISOTROPIC_MODES = tuple(
    (iz, ix, iy) for iz in (1, -1) for ix, iy in ((0, 1), (1, 0), (-1, 0))
)


def _analytical_mode_mask(system, modes):
    """
    Mark a stated set of integer Fourier modes in half-grid storage.
    """
    mask = np.zeros(system.half_tuple, dtype=bool)
    for iz, ix, iy in modes:
        mask[iz, ix, iy] = True

    return mask


###############################################################################
# CPU tests
###############################################################################


@pytest.mark.cpu
@pytest.mark.parametrize(
    ("range_updates", "expected_modes", "expected_physical_count"),
    [
        pytest.param(
            {
                "range_kmod": [1.0, 1.4]
            },
            ISOTROPIC_MODES,
            6,
            id="isotropic",
        ),
        pytest.param(
            {
                "range_kmod": [],
                "range_kz": [1.1, 1.25],
                "range_kperp": [1.0, 1.4],
            },
            ANISOTROPIC_MODES,
            8,
            id="anisotropic",
        ),
    ],
)
def test_forcing_range_contract(
    test_system,
    tmp_path,
    precision,
    range_updates,
    expected_modes,
    expected_physical_count,
):
    """
    CUDA forcing acts on only the modes selected by the shared host range.
    """

    _, _, system = create_test_solver_system(
        tmp_path,
        test_system,
        precision=precision,
        updates={
            "dimensions": DIMENSIONS,
            "forcing": {
                **OU_FORCING,
                **range_updates,
            },
        },
    )

    # Enter through the system lifecycle rather than calling range helpers
    system.setup_cuda_definitions()
    forcing = system.forcing_object
    expected_mask = _analytical_mode_mask(system, expected_modes)

    assert forcing.forcing_range_mask.shape == system.half_tuple
    assert forcing.forcing_range_mask.dtype == np.dtype(bool)

    npt.assert_array_equal(forcing.forcing_range_mask, expected_mask)

    # Exclusion masks must not remove any mode in the forcing range
    assert not np.any(
        forcing.forcing_range_mask & forcing.below_forcing_range_mask
    )
    assert not np.any(
        forcing.forcing_range_mask & forcing.above_forcing_range_mask
    )

    # The count includes the omitted negative-ky partners in physical space
    assert forcing.forced_mode_count == expected_physical_count


###############################################################################
# GPU tests
###############################################################################


@pytest.mark.gpu
def test_forcing_updates_only_the_analytical_range(test_system, tmp_path):
    """
    CUDA forcing acts on only the modes selected by the shared host range.
    """

    _, solver, system = create_test_solver_system(
        tmp_path,
        test_system,
        precision=SINGLE_PRECISION,
        updates={
            "dimensions": DIMENSIONS,
            "setup": {
                "linear": True,
                "check_linear_matrix": False,
            },
            "timestepping": {"method": "ssprk3"},
            "parameters": {
                "advection": [0.0, 0.0, 0.0],
                "rotation": [0.0, 0.0, 0.0],
            },
            "init": {
                "method": "deterministic",
                "amplitude": 0.0,
            },
            "forcing": {
                **OU_FORCING,
                "range_kmod": [],
                "range_kz": [1.1, 1.25],
                "range_kperp": [1.0, 1.4],
            },
        },
    )

    # Compile and initialise the ordinary linear, explicitly forced system
    system.setup()
    solver.timestepper.setup()
    system.compile_cupy_module()
    system.setup_initial_conditions()

    try:
        system.ready()
        solver.timestepper.ready()

        # Advance one complete production timestep from identically zero fields
        system.current_step += 1
        system.begin_time_step()
        solver.timestepper.execute_timestep()
        system.current_time += system.current_dt
        system.finish_time_step()

        fields = cp.asnumpy(system.get_fields())
        forcing = system.forcing_object
        expected_mask = _analytical_mode_mask(system, ANISOTROPIC_MODES)
        solved_mask = system.get_solved_grid_mask().astype(bool)

        # Host and CUDA selection agree on the analytically isolated modes
        npt.assert_array_equal(forcing.forcing_range_mask, expected_mask)

        assert not np.any(expected_mask & ~solved_mask)
        assert all(
            np.linalg.norm(field[expected_mask]) > system.tolerance
            for field in fields
        )

        npt.assert_array_equal(
            fields[:, ~expected_mask],
            system.complex(0),
        )

        # Forcing preserves the Fourier reality condition on the ky=0 plane
        conjugate_iz = (-np.arange(system.nz)) % system.nz
        conjugate_ix = (-np.arange(system.nx)) % system.nx
        fields_ky0 = fields[:, :, :, 0]

        npt.assert_allclose(
            fields_ky0,
            np.conj(
                fields_ky0[
                    :,
                    conjugate_iz[:, None],
                    conjugate_ix[None, :],
                ]
            ),
            rtol=0,
            atol=system.tolerance,
        )
    finally:
        system.kernels.unbind()
