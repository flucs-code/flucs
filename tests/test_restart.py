"""
Tests for FLUCS restart persistence and selection.
"""

from types import SimpleNamespace

import numpy as np
import numpy.testing as npt
import pytest
from netCDF4 import Dataset

import flucs.restart as restart_module
from flucs.input import InvalidFlucsInputFileError
from flucs.restart import FlucsRestart
from flucs.solvers import FlucsSolverState

pytestmark = pytest.mark.core


class _RestartInput:
    """
    Minimal input interface for configuring a restart manager.
    """

    def __init__(self, io_path, **updates):
        self.io_path = io_path
        self.values = {
            "restart.restart_if_exists": False,
            "restart.restart_from": "",
            "restart.reset_time": False,
            "restart.write_restart_file": True,
            "restart.write_steps": 2,
            "restart.backup_count": 2,
            "time.tfinal": 4.0,
        }
        self.values.update(updates)

    def __getitem__(self, key):
        return self.values[key]

    def __str__(self):
        return "[setup]\nsolver = 'ExampleSolver'\n"


class _RestartSystem:
    """
    Explicit system boundary used for restart round trips.
    """

    def __init__(self, io_path, **input_updates):
        self.input = _RestartInput(io_path, **input_updates)
        self.float = np.float64
        self.solver = SimpleNamespace(state=FlucsSolverState.RUNNING)
        self.current_time = 1.0
        self.current_dt = 0.125
        self.restart_data = {}

    def get_restart_data(self):
        return self.restart_data


class _GpuArray:
    """
    Placeholder array type for the unexercised CuPy restart branch.
    """


def _read_restart_time(restart_path):
    """
    Return the saved time without depending on manager internals.
    """

    with Dataset(restart_path, "r", format="NETCDF4") as dataset:
        return float(dataset.variables["current_time"][...])


def test_restart_round_trip_scheduling_backups_and_reconstruction(
    tmp_path,
    monkeypatch,
):
    """
    Restart writes rotate safely and can restore state and input metadata.
    """

    # Keep this CPU test on the NumPy branch without requiring a CUDA runtime
    monkeypatch.setattr(
        restart_module,
        "cp",
        SimpleNamespace(ndarray=_GpuArray),
    )

    # Write a mixture of named, implicit, real, and complex restart arrays
    system = _RestartSystem(tmp_path)
    system.restart_data = {
        "real_data": {
            "data": np.array([1.0, 2.0]),
            "dimension_names": ("field",),
        },
        "complex_state": {
            "data": np.array([1.0 + 2.0j, 3.0 + 4.0j]),
            "dimension_names": ("state_component",),
        },
        "implicit": {"data": np.arange(6).reshape(2, 3)},
    }
    restart = FlucsRestart(system)

    # Even a forced write is suppressed outside the production solver state
    system.solver.state = FlucsSolverState.TIMING
    restart.write_restart(force=True)

    assert not (tmp_path / "restart.nc").exists()

    # The first call writes immediately, after which the normal cadence applies
    system.solver.state = FlucsSolverState.RUNNING
    restart.write_restart()

    assert _read_restart_time(tmp_path / "restart.nc") == 1.0

    system.current_time = 2.0
    system.current_dt = 0.0625
    system.restart_data["real_data"]["data"] = np.array([2.0, 3.0])

    restart.write_restart()
    assert _read_restart_time(tmp_path / "restart.nc") == 1.0
    restart.write_restart()
    assert _read_restart_time(tmp_path / "restart.nc") == 2.0

    # A forced third write leaves the two preceding states in newest-first order
    system.current_time = 3.0
    system.restart_data["real_data"]["data"] = np.array([3.0, 4.0])
    restart.write_restart(force=True)

    assert _read_restart_time(tmp_path / "restart.nc") == 3.0
    assert _read_restart_time(tmp_path / "restart.backup.00.nc") == 2.0
    assert _read_restart_time(tmp_path / "restart.backup.01.nc") == 1.0
    assert not (tmp_path / "restart.temp.nc").exists()

    # Loading reconstructs complex values, dimensions, and continuation times
    loaded_system = _RestartSystem(
        tmp_path,
        **{
            "restart.restart_from": "restart.nc",
            "restart.write_restart_file": False,
        },
    )
    loaded = FlucsRestart(loaded_system)

    assert loaded.initial_path == (tmp_path / "restart.nc").resolve()
    assert loaded_system.init_time == 3.0
    assert loaded_system.init_dt == 0.0625
    assert loaded_system.final_time == 7.0

    npt.assert_allclose(loaded.data["real_data"]["data"], [3.0, 4.0])
    npt.assert_allclose(
        loaded.data["complex_state"]["data"],
        [1.0 + 2.0j, 3.0 + 4.0j],
    )
    assert loaded.data["complex_state"]["dimension_names"] == (
        "state_component",
    )
    npt.assert_array_equal(
        loaded.data["implicit"]["data"],
        np.arange(6).reshape(2, 3),
    )
    assert loaded.data["implicit"]["dimension_names"] == (
        "implicit_dim0",
        "implicit_dim1",
    )

    # Reset-time restarts retain the saved timestep but begin a new time window
    reset_system = _RestartSystem(
        tmp_path,
        **{
            "restart.restart_from": "restart.nc",
            "restart.reset_time": True,
            "restart.write_restart_file": False,
        },
    )
    FlucsRestart(reset_system)
    assert reset_system.init_time == 0.0
    assert reset_system.init_dt == 0.0625
    assert reset_system.final_time == 4.0

    # The embedded resolved input can seed a fresh i/o directory
    reconstructed_path = tmp_path / "reconstructed"
    reconstructed_path.mkdir()
    
    FlucsRestart.reconstruct_input_from_restart(
        tmp_path / "restart.nc",
        reconstructed_path,
    )
    assert (reconstructed_path / "input.toml").read_text(
        encoding="utf-8"
    ) == str(system.input)


def test_restart_rejects_ambiguous_or_unsafe_configuration(tmp_path):
    """
    Invalid restart sources and output policies fail before data is changed.
    """

    # Selecting both restart mechanisms is inherently ambiguous
    restart_path = tmp_path / "restart.nc"
    restart_path.touch()
    conflicting = _RestartSystem(
        tmp_path,
        **{
            "restart.restart_if_exists": True,
            "restart.restart_from": "restart.nc",
        },
    )
    with pytest.raises(
        InvalidFlucsInputFileError,
        match="cannot be specified simultaneously",
    ):
        FlucsRestart(conflicting)

    # Explicit sources must exist rather than silently starting from scratch
    restart_path.unlink()
    missing = _RestartSystem(
        tmp_path,
        **{"restart.restart_from": "missing.nc"},
    )
    with pytest.raises(InvalidFlucsInputFileError, match="cannot be found"):
        FlucsRestart(missing)

    # Backup counts are bounded before paths are rotated
    invalid_backups = _RestartSystem(
        tmp_path,
        **{"restart.backup_count": 101},
    )
    with pytest.raises(
        InvalidFlucsInputFileError,
        match="backup_count must be an integer between 0 and 100",
    ):
        FlucsRestart(invalid_backups)

    # Existing state is never overwritten without an explicit restart request
    restart_path.touch()
    unsafe_write = _RestartSystem(tmp_path)
    with pytest.raises(
        InvalidFlucsInputFileError,
        match=r"remove existing 'restart[.]nc' manually",
    ):
        FlucsRestart(unsafe_write)
