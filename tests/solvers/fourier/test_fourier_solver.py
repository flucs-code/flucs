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
    Evolve TestFourierSystem linearly with damping
    """
    source_input = pl.Path(__file__).parents[2] / "support/fourier/runtime.toml"
    dt = 1 / 16
    final_time = 0.5

    # Parameters
    advection = np.array([0.5, -0.25, 0.125])
    rotation = np.array([0.25, 0.5, -0.125])
    nu = 0.125

    # Different dimensions
    Lz, Lx, Ly = 2 * np.pi, 4 * np.pi, 6 * np.pi

    # Set a linear run with fixed, binary-exact timesteps.
    override = {
        "setup.precision": precision,
        "setup.linear": True,
        "setup.check_linear_matrix": False,
        "time.dt_max": dt,
        "time.tfinal": final_time,
        "timestepping.method": method,
        "timestepping.linear_pade_degree": 4,
        "timestepping.precompute_linear_matrix": precompute,
        "time.dt_method": "discrete" if precompute else "continuous",
        "dimensions.Lz": Lz,
        "dimensions.Lx": Lx,
        "dimensions.Ly": Ly,
        "parameters.advection": advection.tolist(),
        "parameters.rotation": rotation.tolist(),
        "init.method": "deterministic",
        "init.amplitude": 1.0,
        "forcing.method": "",
        # Use ordinary diffusion: nu*|k|^2, independent of dt and resolution.
        "hyperdissipation.kmod": nu,
        "hyperdissipation.kmod_power": 1,
        "hyperdissipation.kmod_normalised": False,
        "hyperdissipation.kmod_adaptive": False,
        "output.time.save_steps": -1,
        "output.0d.save_steps": -1,
        "output.1d.save_steps": -1,
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
        damping_rate = nu * (kz**2 + kx**2 + ky**2)

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
            +np.cos(angle) * initial_fields
            - np.sin(angle) * cross_fields
            + (1 - np.cos(angle)) * parallel_fields
        )

        # Apply the exact advective phase and diffusive decay.
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
        # Use the same physical dissipation at both timesteps.
        "hyperdissipation.kmod": 0.05,
        "hyperdissipation.kmod_power": 1,
        "hyperdissipation.kmod_normalised": False,
        "hyperdissipation.kmod_adaptive": False,
        # Save at different intervals, including a final partial write.
        "output.write_steps": 11,
        "output.time.save_steps": 3,
        "output.time.diags": ["free_energy"],
        "output.0d.save_steps": 5,
        "output.1d.save_steps": 5,
        # Save Fourier and real-space slices.
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
