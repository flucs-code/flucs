"""
Restart data, timing, and retained backups from a real solver run.
"""

import pathlib as pl

import numpy as np
import numpy.testing as npt
import pytest
from netCDF4 import Dataset

from flucs import cupy as cp
from flucs.restart import FlucsRestart
from flucs.testing import create_test_system

pytestmark = [pytest.mark.gpu, pytest.mark.slow]

################################################################################
# Helpers and fixtures
################################################################################

@pytest.fixture(scope="module", params=("single", "double"))
def restart_run(request, test_fourier_system, tmp_path_factory):
    """
    Run once per precision and copy the data supplied at each restart write.
    """
    # Input paths
    source_input = pl.Path(__file__).parent / "support/fourier/runtime.toml"
    io_path = tmp_path_factory.mktemp(f"restart_{request.param}")
    dt = 1 / 128

    # Setup input file
    override = {
        "setup.precision": request.param,

        # Set ten fixed steps with deterministic forcing.
        "time.dt_max": dt,
        "time.tfinal": 10 * dt,
        "timestepping.method": "ssprk3e",
        "forcing.method": "negative_damping",
        "hyperdissipation.kmod": 0.1,
        "hyperdissipation.kmod_adaptive": False,

        # Disable diagnostics; their saved data is checked in test_output.py.
        "output.time.save_steps": -1,
        "output.0d.save_steps": -1,
        "output.1d.save_steps": -1,

        # Write at steps 1, 5, 9, and the final forced write at step 10.
        "restart.write_restart_file": True,
        "restart.write_steps": 4,
        "restart.backup_count": 2,
    }

    with create_test_system(source_input, io_path, override=override) as (
        _, solver, system,
    ):
        checkpoints = {}
        get_restart_data = system.get_restart_data

        def record_restart_data():
            # Get restart data without changing what the writer receives.
            restart_data = get_restart_data()
            checkpoint = {
                "time": system.current_time,
                "dt": system.current_dt,
                "data": {},
            }
            for name, variable in restart_data.items():
                checkpoint["data"][name] = {
                    "data": cp.asnumpy(variable["data"]).copy(),
                    "dimension_names": variable["dimension_names"],
                }
            checkpoints[int(system.current_step)] = checkpoint
            return restart_data

        # Record independent copies while running the production restart writer.
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(system, "get_restart_data", record_restart_data)
            solver.run()

        assert system.current_step == 10
        assert list(checkpoints) == [1, 5, 9, 10]
        yield system, checkpoints

################################################################################
# Tests
################################################################################

# ------------------------------------------------------------------------------
# Restart data and timing
# ------------------------------------------------------------------------------

def test_restart_data_and_time(restart_run, tmp_path):
    """
    Restore restart data and saved timing, including the resolved input.
    """
    system, checkpoints = restart_run
    expected = checkpoints[10]
    restart_path = system.input.io_path / "restart.nc"

    # Check the on-disk timing precision and split complex components.
    with Dataset(restart_path) as dataset:
        assert dataset["current_time"].dtype == np.dtype("float64")
        assert dataset["current_dt"].dtype == np.dtype(system.float)
        assert dataset["current_time"][...] == expected["time"]
        assert dataset["current_dt"][...] == expected["dt"]
        assert dataset["input_file"][...] == str(system.input)

        for name, variable in expected["data"].items():
            data = variable["data"]
            if np.iscomplexobj(data):
                components = {"_real": data.real, "_imag": data.imag}
            else:
                components = {"": data}

            for suffix, values in components.items():
                saved = dataset[f"{name}{suffix}"]
                assert saved.dimensions == variable["dimension_names"]
                assert saved.dtype == np.dtype(system.float)
                npt.assert_array_equal(saved[:], values)

    # Load with a different dt_max and an additional half unit of run time.
    override = {
        "setup.precision": system.input["setup.precision"],
        "restart.restart_from": str(restart_path),
        "restart.write_restart_file": False,
        "time.dt_max": 0.125,
        "time.tfinal": 0.5,
    }
    with create_test_system(
        system.input.input_path, tmp_path, override=override,
    ) as (_, _, loaded_system):
        restart = FlucsRestart(loaded_system)

        # Restore dt from the file, not the new input's dt_max.
        assert loaded_system.init_time == expected["time"]
        assert loaded_system.init_dt == expected["dt"]
        assert loaded_system.final_time == expected["time"] + 0.5

        # Keep the resolved input in restart data; timing belongs to the system.
        assert set(restart.data) == set(expected["data"]) | {"input_file"}
        assert restart.data["input_file"]["data"].item() == str(system.input)
        for name, variable in expected["data"].items():
            loaded = restart.data[name]
            assert loaded["dimension_names"] == variable["dimension_names"]
            assert loaded["data"].dtype == variable["data"].dtype
            npt.assert_array_equal(loaded["data"], variable["data"])


def test_restart_reset_time(restart_run, tmp_path):
    """
    Reset the starting time without changing the saved dt or restart data.
    """
    system, checkpoints = restart_run
    expected = checkpoints[10]
    restart_path = system.input.io_path / "restart.nc"
    override = {
        "setup.precision": system.input["setup.precision"],
        "restart.restart_from": str(restart_path),
        "restart.reset_time": True,
        "restart.write_restart_file": False,
        "time.dt_max": 0.125,
        "time.tfinal": 0.5,
    }

    # Load the same file with reset_time enabled.
    with create_test_system(
        system.input.input_path, tmp_path, override=override,
    ) as (_, _, loaded_system):
        restart = FlucsRestart(loaded_system)

        assert loaded_system.init_time == 0.0
        assert loaded_system.init_dt == expected["dt"]
        assert loaded_system.final_time == 0.5

        # Resetting time must not reset the saved state.
        for name, variable in expected["data"].items():
            npt.assert_array_equal(restart.data[name]["data"], variable["data"])

# ------------------------------------------------------------------------------
# Initial state
# ------------------------------------------------------------------------------

def test_restart_initial_state(restart_run, tmp_path):
    """
    Initialise the active system from restart data without advancing it.
    """
    system, checkpoints = restart_run
    expected = checkpoints[10]
    restart_path = system.input.io_path / "restart.nc"

    override = {
        "setup.precision": system.input["setup.precision"],
        "restart.restart_from": str(restart_path),
        "restart.write_restart_file": False,
        "time.tfinal": 0.0,
    }

    # Disable diagnostics in the copied input.
    for name, options in system.input["output"].items():
        if isinstance(options, dict):
            override[f"output.{name}.save_steps"] = -1

    with create_test_system(
        system.input.input_path, tmp_path, override=override,
    ) as (_, solver, restarted_system):
        # Run normal setup and initialisation without taking a timestep.
        solver.run()

        assert restarted_system.current_step == 0
        assert restarted_system.current_time == expected["time"]
        assert restarted_system.current_dt == expected["dt"]

        # Read the active state through the system's restart interface.
        restart_data = restarted_system.get_restart_data()
        assert set(restart_data) == set(expected["data"])

        for name, variable in expected["data"].items():
            restored = restart_data[name]
            actual = cp.asnumpy(restored["data"])
            
            assert restored["dimension_names"] == variable["dimension_names"]
            assert actual.dtype == variable["data"].dtype
            npt.assert_array_equal(actual, variable["data"])

# ------------------------------------------------------------------------------
# Backups
# ------------------------------------------------------------------------------

def test_restart_backups(restart_run):
    """
    Retain the two preceding checkpoints, newest first, after backup rotation.
    """
    system, checkpoints = restart_run
    io_path = system.input.io_path

    # Keep only the requested number of numbered backups.
    backup_names = {path.name for path in io_path.glob("restart.backup.*.nc")}
    assert backup_names == {"restart.backup.00.nc", "restart.backup.01.nc"}

    # Retained backups are steps 9 and 5.
    expected_steps = {
        "restart.backup.00.nc": 9,
        "restart.backup.01.nc": 5,
    }
    for filename, step in expected_steps.items():
        expected = checkpoints[step]
        with Dataset(io_path / filename) as dataset:
            assert dataset["current_time"][...] == expected["time"]
            assert dataset["current_dt"][...] == expected["dt"]

            # Compare restart data with the copy taken at that write.
            for name, variable in expected["data"].items():
                data = variable["data"]
                if np.iscomplexobj(data):
                    components = {"_real": data.real, "_imag": data.imag}
                else:
                    components = {"": data}

                for suffix, values in components.items():
                    saved = dataset[f"{name}{suffix}"]
                    npt.assert_array_equal(saved[:], values)
