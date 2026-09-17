"""
Tests for diagnostic scheduling and output serialization.
"""

from typing import ClassVar

import numpy as np
import numpy.testing as npt
import pytest
import toml
from netCDF4 import Dataset

from flucs.diagnostic import FlucsDiagnostic, FlucsDiagnosticVariable
from flucs.output import (
    FlucsOutput,
    FlucsOutputNC,
    FlucsOutputText,
    get_output_type,
)
from flucs.solvers import FlucsSolverState
from tests.support.support import (
    SINGLE_PRECISION,
    create_test_solver_system,
    get_stored_variable_names,
)

pytestmark = pytest.mark.core


class _ScalarDiagnostic(FlucsDiagnostic):
    """
    Scalar data suitable for the text-output workflow.
    """

    name = "scalar"
    option_defaults: ClassVar[dict[str, object]] = {"factor": 1.0}

    def init_vars(self) -> None:
        self.add_var(
            FlucsDiagnosticVariable(
                name="real_value",
                shape=(),
                dimensions={},
                is_complex=False,
            )
        )
        self.add_var(
            FlucsDiagnosticVariable(
                name="complex_value",
                shape=(),
                dimensions={},
                is_complex=True,
            )
        )

    def execute(self) -> None:
        self.save_data(
            "real_value",
            self.factor * self.system.current_time,
        )
        self.save_data(
            "complex_value",
            self.system.current_step + 1j * self.system.current_cfl,
        )

    def ready(self) -> None:
        pass


class _ArrayDiagnostic(FlucsDiagnostic):
    """
    Array data covering the layouts supported by NetCDF output.
    """

    name = "array"

    def init_vars(self) -> None:
        dimensions = {
            "grid/position": np.array([-1.0, 1.0]),
            "grid/component": np.arange(3),
        }
        for name, is_complex in (
            ("grid/real", False),
            ("grid/complex", True),
        ):
            self.add_var(
                FlucsDiagnosticVariable(
                    name=name,
                    shape=("position", "component"),
                    dimensions=dimensions,
                    is_complex=is_complex,
                )
            )

        self.add_var(
            FlucsDiagnosticVariable(
                name="grid/reference",
                shape=("position", "component"),
                dimensions=dimensions,
                is_complex=True,
                is_time_dependent=False,
            )
        )

    def execute(self) -> None:
        step = self.system.current_step
        real_data = np.arange(6).reshape(2, 3) + step
        self.save_data("grid/real", real_data)
        self.save_data(
            "grid/complex",
            real_data + 1j * (real_data + 10.0),
        )

    def ready(self) -> None:
        self.save_data(
            "grid/reference",
            np.arange(6).reshape(2, 3) + 1.0j,
        )


def _netcdf_variable_paths(group, prefix=""):
    """
    Return all variable paths beneath a NetCDF group.
    """
    paths = {f"{prefix}{name}" for name in group.variables}
    for name, subgroup in group.groups.items():
        paths.update(
            _netcdf_variable_paths(
                subgroup,
                prefix=f"{prefix}{name}/",
            )
        )
    return paths


def _netcdf_variables(group):
    """
    Return all variables beneath a NetCDF group.
    """
    variables = list(group.variables.values())
    for subgroup in group.groups.values():
        variables.extend(_netcdf_variables(subgroup))
    return variables


def _create_output_system(
    io_path,
    test_system,
    monkeypatch,
    output_name,
    output_type,
    diagnostics,
    available,
    precision=SINGLE_PRECISION,
):
    """
    Construct a real TestSystem with controlled output diagnostics.
    """
    _, _, system = create_test_solver_system(
        io_path,
        test_system,
        precision=precision,
        updates={
            "output": {
                output_name: {
                    "type": output_type,
                    "save_steps": 2,
                    "diags": diagnostics,
                }
            }
        },
    )

    # Isolate serialization with small diagnostics valid for every TestSystem
    monkeypatch.setattr(system, "get_available_diags", lambda: available)

    # Supply the ordinary runtime state consumed by output diagnostics
    system.current_time = 0.0
    system.current_step = 0
    system.current_dt = 0.1
    system.current_cfl = 0.0
    return system


###############################################################################
# CPU tests
###############################################################################


@pytest.mark.cpu
def test_text_output_runs_the_diagnostic_and_writes_rows(
    test_system,
    tmp_path,
    monkeypatch,
    precision,
):
    """
    Text output ignores timing data then writes and clears production rows.
    """

    # Construct through the selected TestSystem and public output factory
    output_name = "time"
    system = _create_output_system(
        tmp_path,
        test_system,
        monkeypatch,
        output_name,
        "text",
        [{"name": "scalar", "options": {"factor": 2.0}}],
        {"scalar": _ScalarDiagnostic},
        precision,
    )
    output = FlucsOutput(output_name, system)
    assert type(output) is FlucsOutputText
    assert get_output_type("text") is FlucsOutputText

    # Timing executions may fill caches but must never touch the output file
    output.ready()
    output.execute()
    output.write()
    assert not output.filepath.exists()

    # Starting production clears timing data and creates a fresh header
    system.solver.state = FlucsSolverState.RUNNING
    output.ready()
    for step, time, cfl in ((1, 0.5, 0.2), (2, 1.0, 0.3)):
        system.current_step = step
        system.current_time = time
        system.current_cfl = cfl
        output.execute()
    output.write()

    # Read the user-facing file rather than merely inspecting internal caches
    lines = output.filepath.read_text(encoding="utf-8").splitlines()
    assert lines[0].split() == [
        "time",
        "step",
        "dt",
        "cfl",
        "real_value",
        "complex_value",
    ]
    assert lines[1].split() == [
        "5.000e-01",
        "1",
        "1.000e-01",
        "2.000e-01",
        "1.000e+00",
        "1.0e+00+2.0e-01j",
    ]
    assert len(lines) == 3

    # Successful writes leave every cache ready for the next batch
    assert output.next_save == 4
    assert output.time_cache == []
    assert output.dt_cache == []
    assert output.timing_data == []
    assert all(
        not var.data_cache
        for diagnostic in output.diagnostics
        for var in diagnostic.vars.values()
    )

    # A later run is separated cleanly before repeating the complete header
    output.ready()
    repeated_lines = output.filepath.read_text(encoding="utf-8").splitlines()
    assert repeated_lines[3] == "-" * len(repeated_lines[0])
    assert repeated_lines[4] == repeated_lines[0]

    with pytest.raises(ValueError, match=r"Data type .* is not supported"):
        output.format_data([1.0])


@pytest.mark.cpu
def test_netcdf_output_round_trip_preserves_layout_and_values(
    test_system,
    tmp_path,
    monkeypatch,
    precision,
):
    """
    NetCDF output writes nested, complex, and time-independent variables.
    """

    # Use a real TestSystem to create the first numbered output group
    output_name = "time"
    system = _create_output_system(
        tmp_path,
        test_system,
        monkeypatch,
        output_name,
        "netcdf4",
        ["array"],
        {"array": _ArrayDiagnostic},
        precision,
    )
    output = FlucsOutput(output_name, system)
    assert type(output) is FlucsOutputNC

    # Ready writes static metadata, while executions cache evolving arrays
    system.solver.state = FlucsSolverState.RUNNING
    output.ready()
    for step, time, dt in ((1, 0.25, 0.1), (2, 0.5, 0.05)):
        system.current_step = step
        system.current_time = time
        system.current_dt = dt
        output.execute()
    output.write()

    # Inspect the complete external schema, including split complex values
    with Dataset(output.filepath, "r", format="NETCDF4") as dataset:
        group = dataset.groups["0"]
        diagnostic = group.groups["array"]
        grid = diagnostic.groups["grid"]

        # Resolve storage names from the active system's NetCDF convention
        complex_names = tuple(
            name.rsplit("/", maxsplit=1)[-1]
            for name in get_stored_variable_names(
                system,
                output.diagnostics[0].vars["grid/complex"],
            )
        )
        reference_names = tuple(
            name.rsplit("/", maxsplit=1)[-1]
            for name in get_stored_variable_names(
                system,
                output.diagnostics[0].vars["grid/reference"],
            )
        )

        assert group.type == "flucs_output"
        resolved_input = toml.loads(str(group.variables["input_file"][...]))
        assert resolved_input["setup"]["solver"] == test_system.solver_name
        assert resolved_input["setup"]["system"] == test_system.system_name

        # Every numerical variable follows the selected system precision
        numerical_variables = [
            group.variables["time"],
            group.variables["dt"],
            grid.variables["position"],
            grid.variables["component"],
            grid.variables["real"],
            *(grid.variables[name] for name in complex_names),
            *(grid.variables[name] for name in reference_names),
        ]
        assert all(
            variable.dtype == np.dtype(precision.float_type)
            for variable in numerical_variables
        )

        npt.assert_allclose(
            group.variables["time"][:],
            [0.25, 0.5],
            rtol=precision.tolerance,
            atol=precision.tolerance,
        )
        npt.assert_allclose(
            group.variables["dt"][:],
            [0.1, 0.05],
            rtol=precision.tolerance,
            atol=precision.tolerance,
        )

        npt.assert_allclose(
            grid.variables["position"][:],
            [-1.0, 1.0],
            rtol=precision.tolerance,
            atol=precision.tolerance,
        )
        npt.assert_array_equal(grid.variables["component"][:], np.arange(3))
        npt.assert_allclose(
            grid.variables["real"][:],
            [
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                [[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]],
            ],
            rtol=precision.tolerance,
            atol=precision.tolerance,
        )
        expected_complex = grid.variables["real"][:] + 1j * (
            grid.variables["real"][:] + 10.0
        )
        npt.assert_allclose(
            grid.variables[complex_names[0]][:]
            + 1j * grid.variables[complex_names[1]][:],
            expected_complex,
            rtol=precision.tolerance,
            atol=precision.tolerance,
        )
        npt.assert_allclose(
            grid.variables[reference_names[0]][:]
            + 1j * grid.variables[reference_names[1]][:],
            np.arange(6).reshape(2, 3) + 1.0j,
            rtol=precision.tolerance,
            atol=precision.tolerance,
        )

    assert output.time_cache == []
    assert output.dt_cache == []
    assert all(
        not var.data_cache
        for diagnostic in output.diagnostics
        for var in diagnostic.vars.values()
    )


@pytest.mark.cpu
@pytest.mark.parametrize(
    ("diagnostics", "available", "error", "message"),
    [
        pytest.param(
            [1],
            {"scalar": _ScalarDiagnostic},
            TypeError,
            "Each diagnostic must be specified",
            id="invalid-entry",
        ),
        pytest.param(
            ["missing"],
            {"scalar": _ScalarDiagnostic},
            KeyError,
            "Diagnostic 'missing' is not available",
            id="unknown-diagnostic",
        ),
        pytest.param(
            ["array"],
            {"array": _ArrayDiagnostic},
            ValueError,
            "text output supports only scalar variables",
            id="array-in-text",
        ),
    ],
)
def test_output_rejects_invalid_diagnostic_configuration(
    test_system,
    tmp_path,
    monkeypatch,
    diagnostics,
    available,
    error,
    message,
):
    """
    Invalid diagnostic declarations fail before an output run begins.
    """

    # Resolve each invalid declaration against an actual TestSystem
    output_name = "time"
    system = _create_output_system(
        tmp_path,
        test_system,
        monkeypatch,
        output_name,
        "text",
        diagnostics,
        available,
    )
    with pytest.raises(error, match=message):
        FlucsOutput(output_name, system)


###############################################################################
# GPU tests
###############################################################################


@pytest.mark.runtime_precision("single")
def test_runtime_outputs_preserve_configured_data(runtime_run):
    """
    Real TestSystem outputs preserve their complete declared schemas.
    """

    # Inspect every active output without assuming its name or file format
    system = runtime_run.system
    outputs = tuple(system.output_heap or ())
    assert outputs
    assert system.current_step > 0

    # The public run helper records a complete user-facing runtime log
    log_contents = (runtime_run.io_path / "output.log").read_text(
        encoding="utf-8"
    )
    assert "Finished at time" in log_contents

    for output in outputs:
        assert output.filepath.is_file()

        expected_diagnostic_names = {
            diagnostic.name for diagnostic in output.diagnostics
        }

        if isinstance(output, FlucsOutputText):
            lines = output.filepath.read_text(encoding="utf-8").splitlines()
            expected_header = [
                *output.timing_data_column_names,
                *(
                    variable.name
                    for diagnostic in output.diagnostics
                    for variable in diagnostic.vars.values()
                ),
            ]
            assert lines[0].split() == expected_header

            # The runtime baseline deliberately provides a useful time series
            rows = [line.split() for line in lines[1:]]
            assert len(rows) > 1
            assert all(len(row) == len(expected_header) for row in rows)

            timing_columns = {
                name: index
                for index, name in enumerate(output.timing_data_column_names)
            }
            times = np.asarray(
                [float(row[timing_columns["time"]]) for row in rows]
            )
            timesteps = np.asarray(
                [float(row[timing_columns["dt"]]) for row in rows]
            )
            assert np.all(np.diff(times) > 0.0)
            assert np.all(timesteps > 0.0)
            assert times[-1] == float(
                output.format_data(system.current_time).strip()
            )

        elif isinstance(output, FlucsOutputNC):
            with Dataset(output.filepath, "r", format="NETCDF4") as dataset:
                # A fresh temporary run creates exactly one numbered group
                assert set(dataset.groups) == {output.group_name}
                group = dataset.groups[output.group_name]
                assert group.type == "flucs_output"
                assert set(group.groups) == expected_diagnostic_names

                resolved_input = toml.loads(
                    str(group.variables["input_file"][...])
                )
                assert resolved_input == toml.loads(
                    str(runtime_run.flucs_input)
                )

                # Time coordinates are nontrivial, finite, and fully written
                times = np.asarray(group.variables["time"][:])
                timesteps = np.asarray(group.variables["dt"][:])

                assert times.size > 1
                assert times.shape == timesteps.shape
                assert np.all(np.diff(times) > 0.0)
                assert np.all(np.isfinite(times))
                assert np.all(np.isfinite(timesteps))
                assert np.all(timesteps > 0.0)

                npt.assert_allclose(
                    times[-1],
                    system.current_time,
                    rtol=runtime_run.precision.tolerance,
                    atol=runtime_run.precision.tolerance,
                )

                # Each diagnostic group contains exactly its declared schema
                numerical_variables = [
                    group.variables["time"],
                    group.variables["dt"],
                ]
                for diagnostic in output.diagnostics:
                    diagnostic_group = group.groups[diagnostic.name]
                    expected_paths = set()
                    for variable in diagnostic.vars.values():
                        expected_paths.update(variable.dimensions)
                        expected_paths.update(
                            get_stored_variable_names(system, variable)
                        )

                    assert _netcdf_variable_paths(diagnostic_group) == (
                        expected_paths
                    )

                    numerical_variables.extend(
                        _netcdf_variables(diagnostic_group)
                    )

                assert all(
                    variable.dtype == np.dtype(runtime_run.precision.float_type)
                    for variable in numerical_variables
                )

        else:
            pytest.fail(f"Unsupported runtime output type: {type(output)}")
