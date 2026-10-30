"""
Saved-output integrity through real solver execution.
"""

import pathlib as pl

import numpy as np
import numpy.testing as npt
import pytest
from netCDF4 import Dataset

from flucs.output import FlucsOutput, FlucsOutputNC, FlucsOutputText
from flucs.solvers import FlucsSolverState
from flucs.testing import create_test_system

pytestmark = [pytest.mark.gpu, pytest.mark.slow]

################################################################################
# Helper functions
################################################################################

def record_output_writes(output_class, monkeypatch, write_counts):
    """
    Record cached sample counts without replacing the production writer.
    """
    original_write = output_class.write

    def write_and_record(output):
        # Count cached samples before writing.
        sample_count = len(output.time_cache)
        running = output.system.solver.state == FlucsSolverState.RUNNING

        # Write the cache and record the successful write.
        original_write(output)
        if running and sample_count:
            write_counts.setdefault(output, []).append(sample_count)

    monkeypatch.setattr(output_class, "write", write_and_record)

################################################################################
# Fixtures
################################################################################

@pytest.fixture
def output_samples(monkeypatch):
    """
    Copy diagnostic samples before later execution or writing can change them.
    """
    samples = {}
    write_counts = {}
    original_execute = FlucsOutput.execute

    def execute_and_record(output):
        system = output.system

        # Execute diagnostics and populate their caches.
        original_execute(output)
        if system.solver.state != FlucsSolverState.RUNNING:
            return

        # Record sample times and timestep data.
        if output not in samples:
            samples[output] = {
                "time": [],
                "step": [],
                "dt": [],
                "cfl": [],
                "data": {},
            }
        recorded = samples[output]
        recorded["time"].append(system.current_time)
        recorded["step"].append(system.current_step)
        recorded["dt"].append(system.current_dt)
        recorded["cfl"].append(system.current_cfl)

        # Read each diagnostic's cache, including direct appends.
        for diagnostic in output.diagnostics:
            for variable in diagnostic.vars.values():
                if not variable.is_time_dependent:
                    continue

                variable_path = f"{diagnostic.name}/{variable.name}"
                if variable_path not in recorded["data"]:
                    recorded["data"][variable_path] = []
                variable_samples = recorded["data"][variable_path]
                latest_value = variable.data_cache[-1]

                # Copy the sample before its cache is changed or cleared.
                variable_samples.append(np.array(latest_value, copy=True))

    # Wrap execution and writing; monkeypatch restores the original methods.
    monkeypatch.setattr(FlucsOutput, "execute", execute_and_record)
    record_output_writes(FlucsOutputText, monkeypatch, write_counts)
    record_output_writes(FlucsOutputNC, monkeypatch, write_counts)
    return samples, write_counts

# ------------------------------------------------------------------------------
# Runtime fixture
# ------------------------------------------------------------------------------

@pytest.fixture(params=("single", "double"))
def output_system(request, test_fourier_system, tmp_path):
    """
    Construct the Fourier runtime with short save and write intervals.
    """
    source_input = pl.Path(__file__).parent / "support/fourier/runtime.toml"
    final_step = 9
    dt = 1 / 128  # Binary-exact timing in both precisions.

    # Override the copied runtime input.
    override = {
        "setup.precision": request.param,

        # Set nine fixed steps, deterministic forcing, and fixed dissipation.
        "time.dt_max": dt,
        "time.tfinal": final_step * dt,
        "timestepping.method": "ssprk3e",
        "forcing.method": "negative_damping",
        "hyperdissipation.kmod": 0.1,
        "hyperdissipation.kmod_adaptive": False,

        # Save every 2 or 3 steps; write at steps 4, 8, and 9.
        "output.write_steps": 4,
        "output.time.save_steps": 2,
        "output.time.diags": ["free_energy"],
        "output.0d.save_steps": 3,
        "output.1d.save_steps": 3,
        "output.1d.diags": [
            {
                "name": "free_energy_1d",
                "options": {"spectra": ["kz", "kperp"]},
            },
        ],

        # Save Fourier and real-space slices.
        "output.3d.save_steps": 3,
        "output.3d.diags": [
            {
                "name": "fourier_data",
                "options": {"locations": ["0:2, 0:3, 0:4, 0:5"]},
            },
        ],
        "output.realspace.save_steps": 3,
        "output.realspace.diags": [
            {
                "name": "realspace_data",
                "options": {"locations": ["0:2, 0:3, 0:4, 0:5"]},
            },
        ],

        # Disable restart writing.
        "restart.write_restart_file": False,
    }

    with create_test_system(source_input, tmp_path, override=override) as (
        _, solver, system,
    ):
        yield solver, system, final_step

################################################################################
# Tests
################################################################################

def test_saved_output_integrity(output_system, output_samples):
    """
    Preserve sampled values through repeated writes and the final forced write.
    """
    solver, system, final_step = output_system
    samples, write_counts = output_samples

    # Run the solver.
    solver.run()

    assert system.current_step == final_step
    assert set(samples) == set(system.output_heap)

    text_outputs = [
        output for output in samples if isinstance(output, FlucsOutputText)
    ]
    netcdf_outputs = [
        output for output in samples if isinstance(output, FlucsOutputNC)
    ]
    assert text_outputs
    assert netcdf_outputs

    for output, recorded in samples.items():

        # Calculate save steps from the input, including the first and last.
        expected_steps = list(range(0, final_step + 1, output.save_steps))
        if expected_steps[-1] != final_step:
            expected_steps.append(final_step)
        expected_steps = np.asarray(expected_steps)
        dt = system.input["time.dt_max"]
        expected_times = expected_steps * dt

        recorded_steps = np.asarray(recorded["step"])
        recorded_times = np.asarray(recorded["time"])
        recorded_dts = np.asarray(recorded["dt"])
        npt.assert_array_equal(recorded_steps, expected_steps)
        npt.assert_array_equal(recorded_times, expected_times)
        npt.assert_array_equal(recorded_dts, dt)

        # Count cached samples at each scheduled or forced write.
        write_steps = system.input["output.write_steps"]
        write_boundaries = list(range(write_steps, final_step + 1, write_steps))
        if write_boundaries[-1] != final_step:
            write_boundaries.append(final_step)

        expected_write_counts = []
        previous_boundary = -1
        for boundary in write_boundaries:
            samples_in_cache = (
                (expected_steps > previous_boundary)
                & (expected_steps <= boundary)
            )
            expected_write_counts.append(np.count_nonzero(samples_in_cache))
            previous_boundary = boundary

        assert len(expected_write_counts) >= 3
        assert write_counts[output] == expected_write_counts

        # Check text output.
        if isinstance(output, FlucsOutputText):
            # Assemble timing columns followed by scalar diagnostics.
            expected_names = ["time", "step", "dt", "cfl"]
            expected_columns = [
                recorded_times, recorded_steps, recorded_dts, recorded["cfl"],
            ]
            for diagnostic in output.diagnostics:
                for variable in diagnostic.vars.values():
                    variable_path = f"{diagnostic.name}/{variable.name}"
                    variable_samples = recorded["data"][variable_path]
                    expected_names.append(variable.name)
                    expected_columns.append(variable_samples)

            with output.filepath.open() as file:
                saved_names = file.readline().split()
            saved = np.loadtxt(output.filepath, skiprows=1)
            expected = np.column_stack(expected_columns)

            assert saved_names == expected_names
            assert np.all(np.isfinite(expected))

            # Allow rounding to four significant digits.
            npt.assert_allclose(saved, expected, rtol=5e-4, atol=0)
            continue

        # Check NetCDF output.
        with Dataset(output.filepath) as dataset:
            # Check the run group and timing data. Time is always double.
            assert set(dataset.groups) == {"0"}
            group = dataset["0"]
            assert group["time"].dtype == np.dtype("float64")
            assert group["dt"].dtype == np.dtype(system.float)
            npt.assert_array_equal(group["time"][:], expected_times)
            npt.assert_array_equal(group["dt"][:], recorded_dts)

            for diagnostic in output.diagnostics:
                diagnostic_group = group[diagnostic.name]
                for variable in diagnostic.vars.values():
                    variable_path = f"{diagnostic.name}/{variable.name}"
                    expected = np.asarray(recorded["data"][variable_path])
                    assert np.all(np.isfinite(expected))

                    # Compare dimension coordinates, including nested paths.
                    for name, coordinates in variable.dimensions.items():
                        saved_coordinates = diagnostic_group[name][:]
                        expected_coordinates = np.asarray(
                            coordinates, dtype=system.float,
                        )
                        npt.assert_array_equal(
                            saved_coordinates, expected_coordinates,
                        )

                    # Split complex values into real and imaginary components.
                    if variable.is_complex:
                        components = {
                            "_real": expected.real,
                            "_imag": expected.imag,
                        }
                    else:
                        components = {"": expected}

                    for suffix, expected_component in components.items():
                        saved_name = f"{variable.name}{suffix}"
                        saved = diagnostic_group[saved_name]
                        expected_values = expected_component.astype(
                            system.float,
                        )

                        assert saved.dimensions == ("time", *variable.shape)
                        assert saved.dtype == np.dtype(system.float)

                        # Compare values exactly at the output precision.
                        npt.assert_array_equal(saved[:], expected_values)
