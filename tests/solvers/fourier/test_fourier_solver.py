"""
FourierSolver evolution against independently calculated solutions.
"""

import pathlib as pl

import numpy as np
import numpy.testing as npt
import pytest

from flucs.testing import create_test_system

################################################################################
# Tests
################################################################################

# ------------------------------------------------------------------------------
# Linear
# ------------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("precision", ("single", "double"))
@pytest.mark.parametrize("method", ("rk4", "ssprk3", "ssprk3e", "ab3"))
@pytest.mark.parametrize(
    "precompute",
    (False, True),
    ids=("not-precomputed", "precomputed"),
)
def test_linear_evolution(
    test_fourier_system,
    tmp_path,
    precision,
    method,
    precompute,
):
    """
    Evolve TestFourierSystem linearly with mixed hyperdissipation
    """
    source_input = pl.Path(__file__).parents[2] / "support/fourier/runtime.toml"
    dt = 1 / 16
    final_time = 0.5

    # Parameters
    advection = np.array([0.5, -0.25, 0.125])
    rotation = np.array([0.25, 0.5, -0.125])

    # Hyperdissipation values
    nu_z = 1 / 16
    nu_x = 2.0
    nu_y = 0.5
    nu_perp = 0.25
    nu_mod = 4.0

    # Different dimensions
    Lz, Lx, Ly = 2 * np.pi, 4 * np.pi, 6 * np.pi

    # Set a linear run with fixed, binary-exact timesteps.
    override = {
        # Setup
        "setup.precision": precision,
        "setup.linear": True,
        "setup.check_linear_matrix": False,
        # Time
        "time.dt_max": dt,
        "time.tfinal": final_time,
        "time.dt_method": "discrete" if precompute else "continuous",
        # Timestepping
        "timestepping.method": method,
        "timestepping.linear_pade_degree": 4,
        "timestepping.precompute_linear_matrix": precompute,
        # Dimensions
        "dimensions.Lz": Lz,
        "dimensions.Lx": Lx,
        "dimensions.Ly": Ly,
        # Parameters
        "parameters.advection": advection.tolist(),
        "parameters.rotation": rotation.tolist(),
        # Init
        "init.method": "deterministic",
        "init.amplitude": 1.0,
        # Forcing
        "forcing.method": "",
        # Hyperdissipation
        "hyperdissipation.kz": nu_z,
        "hyperdissipation.kz_power": 2,
        "hyperdissipation.kz_normalised": False,
        "hyperdissipation.kz_adaptive": False,
        "hyperdissipation.kx": nu_x,
        "hyperdissipation.kx_power": 1,
        "hyperdissipation.kx_normalised": True,
        "hyperdissipation.kx_adaptive": False,
        "hyperdissipation.ky": nu_y,
        "hyperdissipation.ky_power": 3,
        "hyperdissipation.ky_normalised": False,
        "hyperdissipation.ky_adaptive": False,
        "hyperdissipation.kperp": nu_perp,
        "hyperdissipation.kperp_power": 2,
        "hyperdissipation.kperp_normalised": False,
        "hyperdissipation.kperp_adaptive": False,
        "hyperdissipation.kmod": nu_mod,
        "hyperdissipation.kmod_power": 1,
        "hyperdissipation.kmod_normalised": True,
        "hyperdissipation.kmod_adaptive": False,
        # Output
        "output.time.save_steps": -1,
        "output.0d.save_steps": -1,
        "output.1d.save_steps": -1,
        # Restart
        "restart.write_restart_file": False,
    }

    with create_test_system(source_input, tmp_path, override=override) as (
        _,
        solver,
        system,
    ):
        # Construct prescribed initial condition.
        nz, nx, ny = system.full_tuple
        z = 2 * np.pi * np.arange(nz) / nz
        x = 2 * np.pi * np.arange(nx) / nx
        y = 2 * np.pi * np.arange(ny) / ny

        zz, xx, yy = np.meshgrid(z, x, y, indexing="ij")
        uz = np.sin(zz) + 0.3 * np.cos(xx - yy)
        ux = np.sin(xx) + 0.5 * np.cos(yy + zz)
        uy = np.cos(yy) + 0.4 * np.sin(zz + xx)

        initial_fields = np.fft.rfftn(
            np.stack((uz, ux, uy)),
            axes=(1, 2, 3),
            norm="forward",
        ).astype(system.complex)

        # Compute physical wavenumbers from the box lengths.
        kz = 2 * np.pi * np.fft.fftfreq(nz, d=(Lz / nz))[:, None, None]
        kx = 2 * np.pi * np.fft.fftfreq(nx, d=(Lx / nx))[None, :, None]
        ky = 2 * np.pi * np.fft.rfftfreq(ny, d=(Ly / ny))[None, None, :]

        advection_frequency = (
            advection[0] * kz + advection[1] * kx + advection[2] * ky
        )

        # Two-thirds truncation retains integer modes with 3*abs(m) < n.
        kz_max = 2 * np.pi * ((nz - 1) // 3) / Lz
        kx_max = 2 * np.pi * ((nx - 1) // 3) / Lx
        ky_max = 2 * np.pi * ((ny - 1) // 3) / Ly
        kmod_max_squared = kz_max**2 + kx_max**2 + ky_max**2

        # Sum the prescribed directional, perpendicular and isotropic damping.
        kperp_squared = kx**2 + ky**2
        kmod_squared = kz**2 + kperp_squared
        damping_rate = (
            nu_z * kz**4
            + nu_x * (kx / kx_max)**2
            + nu_y * ky**6
            + nu_perp * kperp_squared**2
            + nu_mod * kmod_squared / kmod_max_squared
        )

        # Rotate through -|Omega|*t using Rodrigues' formula.
        rotation_frequency = np.linalg.norm(rotation)
        rotation_axis = (rotation / rotation_frequency)[:, None, None, None]

        angle = rotation_frequency * final_time

        parallel_fields = rotation_axis * np.sum(
            rotation_axis * initial_fields,
            axis=0,
        )
        cross_fields = np.cross(rotation_axis, initial_fields, axis=0)

        rotated_fields = (
            np.cos(angle) * initial_fields
            - np.sin(angle) * cross_fields
            + (1 - np.cos(angle)) * parallel_fields
        )

        # Apply the exact advective phase and hyperdissipative decay.
        expected = rotated_fields * np.exp(
            -(damping_rate + 1j * advection_frequency) * final_time
        )

        # Evolve through the normal solver loop.
        solver.run()
        actual = system.get_fields().get()

        assert system.current_time == final_time
        assert system.current_dt == dt

        # A small number of steps leave roundoff as the dominant error source.
        tolerance = 64 * np.finfo(system.float).eps
        npt.assert_allclose(
            actual,
            expected,
            rtol=0,
            atol=tolerance,
            equal_nan=False,
        )


# ------------------------------------------------------------------------------
# Nonlinear
# ------------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("precision", ("single", "double"))
@pytest.mark.parametrize(
    "precompute",
    (False, True),
    ids=("not-precomputed", "precomputed"),
)
def test_nonlinear_evolution(
    test_fourier_system,
    tmp_path,
    precision,
    precompute,
):
    """
    Evolve TestFourierSystem nonlinearly with forcing and damping
    """
    source_input = pl.Path(__file__).parents[2] / "support/fourier/runtime.toml"
    dt = 1 / 128
    final_time = 0.5

    # Set a nonlinear run with fixed, binary-exact timesteps.
    override = {
        "setup.precision": precision,
        "setup.linear": False,
        "time.dt_max": dt,
        "time.tfinal": final_time,
        "timestepping.method": "ssprk3e",
        "timestepping.precompute_linear_matrix": precompute,
        "time.dt_method": "discrete" if precompute else "continuous",
        "init.amplitude": 0.25,
        "forcing.method": "negative_damping",
        "forcing.rate": 0.1,
        "hyperdissipation.kmod": 0.05,
        "hyperdissipation.kmod_power": 1,
        "hyperdissipation.kmod_normalised": False,
        "hyperdissipation.kmod_adaptive": False,
        "output.write_steps": 11,
        "output.time.save_steps": 3,
        "output.time.diags": ["free_energy"],
        "output.0d.save_steps": 5,
        "output.1d.save_steps": 5,
        "output.3d.save_steps": 5,
        "output.3d.diags": [
            {
                "name": "fourier_data",
                "options": {"locations": ["0:2, 0:3, 0:4, 0:5"]},
            },
        ],
        "output.realspace.save_steps": 5,
        "output.realspace.diags": [
            {
                "name": "realspace_data",
                "options": {
                    "locations": ["0:2, 0:3, 0:4, 0:5"],
                    "exclude_forcing_range": True,
                },
            },
        ],
        "restart.write_restart_file": True,
        "restart.write_steps": 17,
    }

    # Evolve through the normal solver loop with diagnostics and output.
    with create_test_system(
        source_input,
        tmp_path / "solution",
        override=override,
    ) as (_, solver, system):
        # Run solver fields
        solver.run()
        actual = system.get_fields().get()
        roundoff_tolerance = 64 * np.finfo(system.float).eps

        assert system.current_time == final_time
        assert system.current_dt == dt

    # Refine dt in double precision to limit accumulated roundoff.
    # Disable diagnostics that could alter the evolving fields.
    reference_dt = dt / 4
    reference_override = override | {
        "setup.precision": "double",
        "time.dt_max": reference_dt,
        "timestepping.precompute_linear_matrix": False,
        "time.dt_method": "continuous",
        "output.time.save_steps": -1,
        "output.0d.save_steps": -1,
        "output.1d.save_steps": -1,
        "output.3d.save_steps": -1,
        "output.realspace.save_steps": -1,
        "restart.write_restart_file": False,
    }
    with create_test_system(
        source_input,
        tmp_path / "reference",
        override=reference_override,
    ) as (_, solver, system):
        solver.run()
        reference = system.get_fields().get()

        assert system.current_time == final_time
        assert system.current_dt == reference_dt

    # Allow third-order timestep error and precision-dependent roundoff.
    error = np.linalg.norm(actual - reference) / np.linalg.norm(reference)
    tolerance = max(1e-7, roundoff_tolerance)

    assert error < tolerance
