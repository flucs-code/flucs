"""
Output integrity, run history, and NetCDF opening failures.
"""

import pathlib as pl
import shutil

import numpy as np
import numpy.testing as npt
import pytest
from netCDF4 import Dataset

import flucs.output as flucs_output
from flucs.output import FlucsOutput, FlucsOutputNC, FlucsOutputText
from flucs.postprocessing import FlucsPostProcessing
from flucs.solvers import FlucsSolverState
from flucs.testing import create_test_system

################################################################################
# Helpers and fixtures
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


def read_netcdf_variables(group):
    """
    Copy all variables beneath a NetCDF group, retaining their relative paths.
    """
    variables = {}
    for name, variable in group.variables.items():
        variables[name] = np.array(variable[...], copy=True)

    for name, subgroup in group.groups.items():
        for variable_path, values in read_netcdf_variables(subgroup).items():
            variables[f"{name}/{variable_path}"] = values

    return variables


def output_overrides(precision):
    """
    Configure the short Fourier run used by both runtime tests.
    """
    dt = 1 / 128  # Binary-exact timing in both precisions.
    return {
        "setup.precision": precision,

        # Set nine fixed steps, deterministic forcing, and fixed dissipation.
        "time.dt_max": dt,
        "time.tfinal": 9 * dt,
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

        # Save a restart for the output history test.
        "restart.write_restart_file": True,
    }

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

@pytest.fixture(params=("single", "double"))
def output_solver(request, test_fourier_system, tmp_path):
    """
    Construct the Fourier runtime with short save and write intervals.
    """
    source_input = pl.Path(__file__).parent / "support/fourier/runtime.toml"
    override = output_overrides(request.param)

    with create_test_system(source_input, tmp_path, override=override) as (
        _, solver, _,
    ):
        yield solver


@pytest.fixture
def cached_netcdf_output(test_fourier_system, tmp_path):
    """
    Prepare a real NetCDF writer with two known samples, without GPU setup.
    """
    source_input = pl.Path(__file__).parent / "support/fourier/runtime.toml"
    override = {
        "setup.precision": "double",
        "output.3d.save_steps": 1,
        "output.3d.diags": [
            {
                "name": "fourier_data",
                "options": {"locations": ["0, 0:2, 0:3, 0:4"]},
            },
        ],
    }

    with create_test_system(source_input, tmp_path, override=override) as (
        _, solver, system,
    ):
        # Create the output schema without running diagnostics or the solver.
        solver.state = FlucsSolverState.RUNNING
        output = FlucsOutput(name="3d", system=system)
        output.group_number = output.get_next_group()
        output.ready()

        # Populate the cache with two distinguishable complex arrays.
        diagnostic = output.diagnostics[0]
        variable = next(iter(diagnostic.vars.values()))
        shape = tuple(len(values) for values in variable.dimensions.values())
        diagnostic.save_data(variable.name, np.full(shape, 2 + 3j))
        diagnostic.save_data(variable.name, np.full(shape, 5 - 7j))
        output.time_cache.extend([0.125, 0.25])
        output.dt_cache.extend([0.125, 0.125])

        yield output, variable

################################################################################
# Tests
################################################################################

# ------------------------------------------------------------------------------
# Saved data
# ------------------------------------------------------------------------------

@pytest.mark.gpu
@pytest.mark.slow
def test_saved_output_integrity(output_solver, output_samples):
    """
    Preserve sampled values through repeated writes and the final forced write.
    """
    solver = output_solver
    system = solver.system
    samples, write_counts = output_samples

    # Expected saves and cache sizes for writes at steps 4, 8, and 9.
    expected_steps = {
        2: [0, 2, 4, 6, 8, 9],
        3: [0, 3, 6, 9],
    }
    expected_write_counts = {
        2: [3, 2, 1],
        3: [2, 1, 1],
    }

    # Run the solver.
    solver.run()

    assert system.current_step == 9
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
        # Check the save schedule and fixed timestep.
        save_steps = output.save_steps
        dt = system.input["time.dt_max"]
        expected_times = np.asarray(expected_steps[save_steps]) * dt
        
        npt.assert_array_equal(recorded["step"], expected_steps[save_steps])
        npt.assert_array_equal(recorded["time"], expected_times)
        npt.assert_array_equal(recorded["dt"], dt)
        
        assert write_counts[output] == expected_write_counts[save_steps]

    # Check text headers and columns.
    for output in text_outputs:
        recorded = samples[output]
        expected_names = ["time", "step", "dt", "cfl"]
        expected_columns = [
            recorded["time"], recorded["step"], recorded["dt"], recorded["cfl"],
        ]
        for diagnostic in output.diagnostics:
            for variable in diagnostic.vars.values():
                variable_path = f"{diagnostic.name}/{variable.name}"
                expected_names.append(variable.name)
                expected_columns.append(recorded["data"][variable_path])

        with output.filepath.open() as file:
            saved_names = file.readline().split()
        saved = np.loadtxt(output.filepath, skiprows=1)
        expected = np.column_stack(expected_columns)

        assert saved_names == expected_names
        assert np.all(np.isfinite(expected))

        # Allow rounding to four significant digits.
        npt.assert_allclose(saved, expected, rtol=5e-4, atol=0)

    # Check NetCDF timing and diagnostic data.
    output_dtype = np.dtype(system.float)
    for output in netcdf_outputs:
        recorded = samples[output]
        with Dataset(output.filepath) as dataset:
            assert set(dataset.groups) == {"0"}
            group = dataset["0"]

            # Time is double; dt follows the output precision.
            assert group["time"].dtype == np.dtype("float64")
            assert group["dt"].dtype == output_dtype
            npt.assert_array_equal(group["time"][:], recorded["time"])
            npt.assert_array_equal(group["dt"][:], recorded["dt"])

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
                            coordinates, dtype=output_dtype,
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
                            output_dtype,
                        )

                        assert saved.dimensions == ("time", *variable.shape)
                        assert saved.dtype == output_dtype
                        npt.assert_array_equal(saved[:], expected_values)


# ------------------------------------------------------------------------------
# Output history
# ------------------------------------------------------------------------------

@pytest.mark.gpu
@pytest.mark.slow
def test_output_history(output_solver, tmp_path):
    """
    Preserve earlier output and read both runs across a restarted continuation.
    """
    solver = output_solver
    system = solver.system
    restart_time = 9 * system.input["time.dt_max"]
    final_time = 2 * restart_time

    # Run the first segment and copy its saved history.
    solver.run()
    original_data = {}
    original_text = {}
    for output in system.output_heap:
        if isinstance(output, FlucsOutputNC):
            with Dataset(output.filepath) as dataset:
                original_data[output.name] = read_netcdf_variables(dataset["0"])
        else:
            original_text[output.name] = output.filepath.read_bytes()

    # Continue from the restart in a separate directory with existing outputs.
    continued_path = tmp_path / "continued"
    continued_path.mkdir()
    for output in system.output_heap:
        shutil.copy2(output.filepath, continued_path / output.filepath.name)

    continued_override = output_overrides(system.input["setup.precision"])
    continued_override["restart.restart_from"] = str(
        system.input.io_path / "restart.nc"
    )
    source_input = system.input.input_path
    with create_test_system(
        source_input, continued_path, override=continued_override,
    ) as (_, continued_solver, continued_system):
        continued_solver.run()

    # Check the continuation interval.
    assert continued_system.init_time == restart_time
    assert continued_system.current_time == final_time

    postprocessing = FlucsPostProcessing(continued_path, quiet=True)
    for output in continued_system.output_heap:
        # Check that text output was appended without changing earlier rows.
        if isinstance(output, FlucsOutputText):
            saved_text = output.filepath.read_bytes()
            assert saved_text.startswith(original_text[output.name])
            assert len(saved_text) > len(original_text[output.name])
            continue

        # Read both NetCDF run groups, including nested coordinates.
        with Dataset(output.filepath) as dataset:
            assert set(dataset.groups) == {"0", "1"}
            first_group = read_netcdf_variables(dataset["0"])
            second_group = read_netcdf_variables(dataset["1"])

        # Compare the first group with its pre-restart copy.
        expected_first = original_data[output.name]
        assert set(first_group) == set(expected_first)
        for variable_path, expected in expected_first.items():
            npt.assert_array_equal(first_group[variable_path], expected)

        # Check the repeated restart sample and final continuation time.
        assert first_group["time"][-1] == restart_time
        assert second_group["time"][0] == restart_time
        assert second_group["time"][-1] == final_time

        # Compare concatenated times and the run boundary.
        expected_times = np.concatenate([
            first_group["time"], second_group["time"],
        ])
        expected_boundaries = [len(first_group["time"])]
        times, boundaries, _ = postprocessing.load_netcdf_variable(
            output.filepath, "time",
        )
        npt.assert_array_equal(times, expected_times)
        assert boundaries == expected_boundaries

        # Compare public readers with direct reads of each diagnostic.
        for diagnostic in output.diagnostics:
            for variable in diagnostic.vars.values():
                variable_path = f"{diagnostic.name}/{variable.name}"
                if variable.is_complex:
                    first_values = (
                        first_group[f"{variable_path}_real"]
                        + 1j * first_group[f"{variable_path}_imag"]
                    )
                    second_values = (
                        second_group[f"{variable_path}_real"]
                        + 1j * second_group[f"{variable_path}_imag"]
                    )
                    values, boundaries, dimensions = (
                        postprocessing.load_netcdf_variable_complex(
                            output.filepath, variable_path,
                        )
                    )
                else:
                    first_values = first_group[variable_path]
                    second_values = second_group[variable_path]
                    values, boundaries, dimensions = (
                        postprocessing.load_netcdf_variable(
                            output.filepath, variable_path,
                        )
                    )

                expected_values = np.concatenate([first_values, second_values])
                npt.assert_array_equal(values, expected_values)
                assert boundaries == expected_boundaries

                # Check coordinates returned for both run groups.
                assert len(dimensions) == 2
                for group_dimensions in dimensions:
                    assert tuple(group_dimensions) == variable.shape
                    for name, coordinates in variable.dimensions.items():
                        dimension_name = name.split("/")[-1]
                        expected_coordinates = np.asarray(
                            coordinates, dtype=system.float,
                        )
                        npt.assert_array_equal(
                            group_dimensions[dimension_name],
                            expected_coordinates,
                        )


# ------------------------------------------------------------------------------
# NetCDF opening
# ------------------------------------------------------------------------------

@pytest.mark.cpu
def test_netcdf_open_recovers(cached_netcdf_output, monkeypatch):
    """
    Retry temporary opening failures and write cached samples once.
    """
    output, variable = cached_netcdf_output
    expected_times = np.array(output.time_cache)
    expected_dts = np.array(output.dt_cache)
    expected_values = np.array(variable.data_cache)
    attempts = 0
    delays = []

    # Fail the first two opens, then use the real Dataset constructor.
    def open_dataset(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts <= 2:
            raise OSError("Temporary opening failure")
        return Dataset(*args, **kwargs)

    monkeypatch.setattr(flucs_output, "Dataset", open_dataset)
    monkeypatch.setattr(flucs_output, "sleep", delays.append)
    output.write()

    assert attempts == 3
    assert delays == [flucs_output.NC_RETRY_DELAY] * 2
    assert output.time_cache == []
    assert output.dt_cache == []
    assert variable.data_cache == []

    # Check saved samples and split complex components.
    diagnostic = output.diagnostics[0]
    variable_path = f"{diagnostic.name}/{variable.name}"
    with Dataset(output.filepath) as dataset:
        group = dataset["0"]
        npt.assert_array_equal(group["time"][:], expected_times)
        npt.assert_array_equal(group["dt"][:], expected_dts)
        npt.assert_array_equal(
            group[f"{variable_path}_real"][:], expected_values.real,
        )
        npt.assert_array_equal(
            group[f"{variable_path}_imag"][:], expected_values.imag,
        )

    # An empty cache must not open the file or repeat the write.
    output.write()
    assert attempts == 3


@pytest.mark.cpu
def test_netcdf_open_failure_preserves_cache(cached_netcdf_output, monkeypatch):
    """
    Stop after the retry limit and retain all unwritten samples.
    """
    output, variable = cached_netcdf_output
    expected_times = np.array(output.time_cache)
    expected_dts = np.array(output.dt_cache)
    expected_values = np.array(variable.data_cache)
    opening_error = OSError("Persistent opening failure")
    attempts = 0
    delays = []

    # Fail every opening attempt and suppress retry delays.
    def open_dataset(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        raise opening_error

    monkeypatch.setattr(flucs_output, "Dataset", open_dataset)
    monkeypatch.setattr(flucs_output, "sleep", delays.append)
    with pytest.raises(OSError, match=r"after .* retries") as error:
        output.write()

    # Count the initial attempt and the configured retries.
    assert attempts == flucs_output.NC_MAX_RETRIES + 1
    assert delays == [flucs_output.NC_RETRY_DELAY] * flucs_output.NC_MAX_RETRIES
    assert error.value.__cause__ is opening_error

    npt.assert_array_equal(output.time_cache, expected_times)
    npt.assert_array_equal(output.dt_cache, expected_dts)
    npt.assert_array_equal(variable.data_cache, expected_values)

    # Check that no cached sample reached the file.
    with Dataset(output.filepath) as dataset:
        group = dataset["0"]
        assert len(group["time"]) == 0
        assert len(group["dt"]) == 0
