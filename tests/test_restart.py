"""
Tests for FLUCS restart persistence and selection.
"""

import shutil
from types import SimpleNamespace

import numpy as np
import numpy.testing as npt
import pytest
import toml
from netCDF4 import Dataset

import flucs.restart as restart_module
from flucs.input import FlucsInput, InvalidFlucsInputFileError
from flucs.restart import FlucsRestart
from flucs.solvers import FlucsSolverState
from tests.support.support import (
    DOUBLE_PRECISION,
    as_numpy,
    create_test_solver_system,
    write_runtime_input,
)

pytestmark = pytest.mark.core


def _create_restart_system(
    io_path,
    test_system,
    precision=DOUBLE_PRECISION,
    **restart_updates,
):
    """
    Construct a real TestSystem configured for restart integration.
    """
    restart_input = {
        "restart_if_exists": False,
        "restart_from": "",
        "reset_time": False,
        "write_restart_file": True,
        "write_steps": 2,
        "backup_count": 2,
    }
    restart_input.update(restart_updates)
    _, _, system = create_test_solver_system(
        io_path,
        test_system,
        precision=precision,
        updates={
            "restart": restart_input,
            "time": {"tfinal": 4.0},
        },
    )

    # Supply the runtime values normally initialized by system setup
    system.solver.state = FlucsSolverState.RUNNING
    system.current_time = 1.0
    system.current_dt = 0.125
    return system


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


def _use_numpy_restart_arrays(monkeypatch):
    """
    Keep host-only restart tests independent of an available CUDA runtime.
    """
    monkeypatch.setattr(
        restart_module,
        "cp",
        SimpleNamespace(ndarray=_GpuArray),
    )


###############################################################################
# CPU tests
###############################################################################


def test_restart_round_trip_scheduling_backups_and_reconstruction(
    test_system,
    tmp_path,
    monkeypatch,
    precision,
):
    """
    Restart writes rotate safely and can restore state and input metadata.
    """

    # Keep this CPU test on the NumPy branch without requiring a CUDA runtime
    _use_numpy_restart_arrays(monkeypatch)

    # An optional default restart may be absent when beginning a fresh run
    optional_system = _create_restart_system(
        tmp_path / "optional",
        test_system,
        precision=precision,
        restart_if_exists=True,
        write_restart_file=False,
    )
    optional_restart = FlucsRestart(optional_system)
    assert optional_restart.initial_path is None
    assert optional_restart.data is None

    # Write a mixture of named, implicit, real, and complex restart arrays
    io_path = tmp_path / "run"
    system = _create_restart_system(
        io_path,
        test_system,
        precision=precision,
    )
    restart_data = {
        "shear_real": {
            "data": np.array([1.0, 2.0], dtype=precision.float_type),
            "dimension_names": ("field",),
        },
        "complex_state": {
            "data": np.array(
                [1.0 + 2.0j, 3.0 + 4.0j],
                dtype=precision.complex_type,
            ),
            "dimension_names": ("state_component",),
        },
        "implicit": {
            "data": np.arange(6, dtype=precision.float_type).reshape(2, 3)
        },
    }
    monkeypatch.setattr(system, "get_restart_data", lambda: restart_data)
    restart = FlucsRestart(system)
    assert restart.netcdf_precision == precision.netcdf_precision

    # Even a forced write is suppressed outside the production solver state
    system.solver.state = FlucsSolverState.TIMING
    restart.write_restart(force=True)

    assert not (io_path / "restart.nc").exists()

    # The first call writes immediately, after which the normal cadence applies
    system.solver.state = FlucsSolverState.RUNNING
    restart.write_restart()

    assert _read_restart_time(io_path / "restart.nc") == 1.0

    system.current_time = 2.0
    system.current_dt = 0.0625
    restart_data["shear_real"]["data"] = np.array(
        [2.0, 3.0],
        dtype=precision.float_type,
    )

    restart.write_restart()
    assert _read_restart_time(io_path / "restart.nc") == 1.0
    restart.write_restart()
    assert _read_restart_time(io_path / "restart.nc") == 2.0

    # A forced third write leaves the two preceding states in newest-first order
    system.current_time = 3.0
    restart_data["shear_real"]["data"] = np.array(
        [3.0, 4.0],
        dtype=precision.float_type,
    )
    restart.write_restart(force=True)

    assert _read_restart_time(io_path / "restart.nc") == 3.0
    assert _read_restart_time(io_path / "restart.backup.00.nc") == 2.0
    assert _read_restart_time(io_path / "restart.backup.01.nc") == 1.0
    assert not (io_path / "restart.temp.nc").exists()

    # The on-disk scalar and array data all use the selected precision
    with Dataset(io_path / "restart.nc", "r", format="NETCDF4") as dataset:
        complex_state_names = (
            f"complex_state{system.netcdf_real_suffix}",
            f"complex_state{system.netcdf_imag_suffix}",
        )
        numerical_variables = [
            dataset.variables["current_time"],
            dataset.variables["current_dt"],
            dataset.variables["shear_real"],
            *(dataset.variables[name] for name in complex_state_names),
            dataset.variables["implicit"],
        ]
        assert all(
            variable.dtype == np.dtype(precision.float_type)
            for variable in numerical_variables
        )

    # The default restart reconstructs values, dimensions, and continuation time
    loaded_system = _create_restart_system(
        io_path,
        test_system,
        precision=precision,
        restart_if_exists=True,
        write_restart_file=False,
    )
    loaded = FlucsRestart(loaded_system)

    assert loaded.initial_path == (io_path / "restart.nc").resolve()
    assert loaded_system.init_time == 3.0
    assert loaded_system.init_dt == 0.0625
    assert loaded_system.final_time == 7.0
    assert type(loaded_system.init_time) is precision.float_type
    assert type(loaded_system.init_dt) is precision.float_type

    assert loaded.data["shear_real"]["data"].dtype == np.dtype(
        precision.float_type
    )
    assert loaded.data["complex_state"]["data"].dtype == np.dtype(
        precision.complex_type
    )
    assert loaded.data["implicit"]["data"].dtype == np.dtype(
        precision.float_type
    )
    npt.assert_allclose(
        loaded.data["shear_real"]["data"],
        [3.0, 4.0],
        rtol=precision.tolerance,
        atol=precision.tolerance,
    )
    npt.assert_allclose(
        loaded.data["complex_state"]["data"],
        [1.0 + 2.0j, 3.0 + 4.0j],
        rtol=precision.tolerance,
        atol=precision.tolerance,
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
    reset_system = _create_restart_system(
        io_path,
        test_system,
        precision=precision,
        restart_from="restart.nc",
        reset_time=True,
        write_restart_file=False,
    )
    FlucsRestart(reset_system)
    assert reset_system.init_time == 0.0
    assert reset_system.init_dt == 0.0625
    assert reset_system.final_time == 4.0

    # The embedded resolved input can seed a fresh i/o directory
    reconstructed_path = tmp_path / "reconstructed"
    reconstructed_path.mkdir()

    FlucsRestart.reconstruct_input_from_restart(
        io_path / "restart.nc",
        reconstructed_path,
    )
    assert (reconstructed_path / "input.toml").read_text(
        encoding="utf-8"
    ) == str(system.input)


@pytest.mark.parametrize(
    ("backup_count", "expected_backup_times"),
    [
        pytest.param(0, [], id="no-backups"),
        pytest.param(1, [1.0], id="one-backup"),
    ],
)
def test_restart_applies_each_backup_policy(
    test_system,
    tmp_path,
    monkeypatch,
    backup_count,
    expected_backup_times,
):
    """
    Zero and one-backup policies replace or retain the preceding restart.
    """

    # Write two generations through the selected TestSystem
    _use_numpy_restart_arrays(monkeypatch)
    system = _create_restart_system(
        tmp_path,
        test_system,
        backup_count=backup_count,
    )
    monkeypatch.setattr(
        system,
        "get_restart_data",
        lambda: {
            "state": {
                "data": np.array(
                    [system.current_time],
                    dtype=system.float,
                )
            }
        },
    )
    restart = FlucsRestart(system)
    restart.write_restart(force=True)
    system.current_time = 2.0
    restart.write_restart(force=True)

    # The current file survives while only the requested history is retained
    assert _read_restart_time(tmp_path / "restart.nc") == 2.0
    assert not (tmp_path / "restart.temp.nc").exists()
    backup_paths = sorted(tmp_path.glob("restart.backup.*.nc"))
    assert [
        _read_restart_time(backup_path) for backup_path in backup_paths
    ] == expected_backup_times


def test_restart_rejects_ambiguous_or_unsafe_configuration(
    test_system,
    tmp_path,
):
    """
    Invalid restart sources and output policies fail before data is changed.
    """

    # Selecting both restart mechanisms is inherently ambiguous
    restart_path = tmp_path / "restart.nc"
    restart_path.touch()
    conflicting = _create_restart_system(
        tmp_path,
        test_system,
        restart_if_exists=True,
        restart_from="restart.nc",
    )
    with pytest.raises(
        InvalidFlucsInputFileError,
        match="cannot be specified simultaneously",
    ):
        FlucsRestart(conflicting)

    # Explicit sources must exist rather than silently starting from scratch
    restart_path.unlink()
    missing = _create_restart_system(
        tmp_path,
        test_system,
        restart_from="missing.nc",
    )
    with pytest.raises(InvalidFlucsInputFileError, match="cannot be found"):
        FlucsRestart(missing)

    # Both sides of the permitted backup-count interval are enforced
    for backup_count in (-1, 101):
        invalid_backups = _create_restart_system(
            tmp_path,
            test_system,
            backup_count=backup_count,
        )
        with pytest.raises(
            InvalidFlucsInputFileError,
            match="backup_count must be an integer between 0 and 100",
        ):
            FlucsRestart(invalid_backups)

    # Existing state is never overwritten without an explicit restart request
    restart_path.touch()
    unsafe_write = _create_restart_system(tmp_path, test_system)
    with pytest.raises(
        InvalidFlucsInputFileError,
        match=r"remove existing 'restart[.]nc' manually",
    ):
        FlucsRestart(unsafe_write)


###############################################################################
# GPU tests
###############################################################################


@pytest.mark.runtime_precision("single")
def test_runtime_restart_restores_the_completed_test_system(
    runtime_run,
    tmp_path,
):
    """
    A real TestSystem restart preserves and restores its complete state.
    """

    # Snapshot the system-owned payload without assuming names or grid shapes
    system = runtime_run.system
    restart = system.restart_manager

    assert restart.write_restart_file
    assert restart.write_path.is_file()

    expected_restart = {}
    for name, entry in system.get_restart_data().items():
        data = as_numpy(entry["data"]).copy()
        dimension_names = entry.get(
            "dimension_names",
            tuple(f"{name}_dim{index}" for index in range(data.ndim)),
        )
        expected_restart[name] = {
            "data": data,
            "dimension_names": tuple(dimension_names),
        }

    assert expected_restart

    # Compare the restart container with the system's public restart contract
    with Dataset(restart.write_path, "r", format="NETCDF4") as dataset:
        assert dataset.type == "flucs_restart"
        assert toml.loads(str(dataset.variables["input_file"][...])) == (
            toml.loads(str(runtime_run.flucs_input))
        )

        npt.assert_allclose(
            dataset.variables["current_time"][...],
            system.current_time,
            rtol=runtime_run.precision.tolerance,
            atol=runtime_run.precision.tolerance,
        )
        npt.assert_allclose(
            dataset.variables["current_dt"][...],
            system.current_dt,
            rtol=runtime_run.precision.tolerance,
            atol=runtime_run.precision.tolerance,
        )

        expected_variable_names = {
            "input_file",
            "current_time",
            "current_dt",
        }
        for name, entry in expected_restart.items():
            expected_data = entry["data"]
            if np.iscomplexobj(expected_data):
                real_name = f"{name}{system.netcdf_real_suffix}"
                imag_name = f"{name}{system.netcdf_imag_suffix}"
                expected_variable_names.update((real_name, imag_name))
                stored_data = np.asarray(
                    dataset.variables[real_name][:]
                ) + 1j * np.asarray(dataset.variables[imag_name][:])
                stored_variables = (
                    dataset.variables[real_name],
                    dataset.variables[imag_name],
                )
            else:
                expected_variable_names.add(name)
                stored_data = np.asarray(dataset.variables[name][:])
                stored_variables = (dataset.variables[name],)

            assert all(
                variable.dimensions == entry["dimension_names"]
                for variable in stored_variables
            )
            assert all(
                variable.dtype == np.dtype(runtime_run.precision.float_type)
                for variable in stored_variables
            )
            npt.assert_allclose(
                stored_data,
                expected_data,
                rtol=runtime_run.precision.tolerance,
                atol=runtime_run.precision.tolerance,
            )

        assert set(dataset.variables) == expected_variable_names

    # Load that file through a fresh instance of the same registered system
    reload_path = tmp_path / "reload"
    reload_path.mkdir()
    shutil.copy2(restart.write_path, reload_path / "restart.nc")

    write_runtime_input(
        reload_path / "input.toml",
        runtime_run.test_system,
        precision=runtime_run.precision,
        updates={
            "restart": {
                "restart_if_exists": True,
                "restart_from": "",
                "write_restart_file": False,
            }
        },
    )

    fresh_input = FlucsInput(reload_path / "input.toml")
    _, fresh_system = fresh_input.create_solver_system()
    fresh_system.setup()
    loaded_restart = fresh_system.restart_manager

    assert loaded_restart.initial_path == (reload_path / "restart.nc").resolve()
    assert loaded_restart.data is not None

    npt.assert_allclose(
        fresh_system.init_time,
        system.current_time,
        rtol=runtime_run.precision.tolerance,
        atol=runtime_run.precision.tolerance,
    )
    npt.assert_allclose(
        fresh_system.init_dt,
        system.current_dt,
        rtol=runtime_run.precision.tolerance,
        atol=runtime_run.precision.tolerance,
    )

    for name, entry in expected_restart.items():
        loaded_entry = loaded_restart.data[name]
        assert loaded_entry["dimension_names"] == entry["dimension_names"]
        npt.assert_allclose(
            loaded_entry["data"],
            entry["data"],
            rtol=runtime_run.precision.tolerance,
            atol=runtime_run.precision.tolerance,
        )

    # Let the TestSystem itself interpret the generic restored mapping
    prepared_data = as_numpy(fresh_system.prepare_restart_data())
    assert prepared_data.size > 0
    assert np.all(np.isfinite(prepared_data))
