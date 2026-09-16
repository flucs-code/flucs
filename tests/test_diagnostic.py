"""
Tests for the base diagnostic classes.
"""

from typing import ClassVar
from unittest.mock import sentinel

import numpy as np
import numpy.testing as npt
import pytest
from netCDF4 import Dataset

from flucs.diagnostic import FlucsDiagnostic, FlucsDiagnosticVariable
from flucs.output import FlucsOutputNC, FlucsOutputText
from tests.support.support import (
    as_numpy,
    create_test_solver_system,
    get_netcdf_variable,
    get_stored_variable_names,
)

pytestmark = pytest.mark.core


class _ExampleDiagnostic(FlucsDiagnostic):
    """
    Small concrete diagnostic used to exercise the shared machinery.
    """

    name = "example"
    option_defaults: ClassVar[dict[str, object]] = {
        "count": 2,
        "scale": 1.5,
        "labels": list(),
    }

    def init_vars(self) -> None:
        self.add_var(
            FlucsDiagnosticVariable(
                name="values",
                shape=("sample",),
                dimensions={"sample": np.arange(2)},
                is_complex=False,
            )
        )
        self.add_var(
            FlucsDiagnosticVariable(
                name="reference",
                shape=(),
                dimensions={},
                is_complex=True,
                is_time_dependent=False,
            )
        )

    def execute(self) -> None:
        self.save_data(
            "values",
            np.array([self.count, self.scale]),
        )

    def ready(self) -> None:
        self.save_data("reference", 1.0 + 2.0j)


###############################################################################
# CPU tests
###############################################################################


def test_diagnostic_initialisation_options_and_cache_lifecycle(
    test_system,
    tmp_path,
):
    """
    Construction prepares typed options, variables, and independent caches.
    """

    # Construct the selected TestSystem through the ordinary plugin path
    _, _, system = create_test_solver_system(tmp_path, test_system)

    # Supply string and tuple values as they might arrive from user input
    diagnostic = _ExampleDiagnostic(
        system=system,
        output=sentinel.output,
        options={
            "count": "4",
            "scale": "2.5",
            "labels": ("first", "second"),
        },
    )

    # Check the common state and option casting established by the base class
    assert diagnostic.system is system
    assert diagnostic.output is sentinel.output

    assert diagnostic.cache_len == 0
    assert diagnostic.count == 4
    assert type(diagnostic.count) is int

    assert diagnostic.scale == 2.5
    assert type(diagnostic.scale) is float

    assert diagnostic.labels == ["first", "second"]
    assert type(diagnostic.labels) is list
    assert hash(diagnostic) == hash(diagnostic.name)

    # The variables retain the metadata needed by the output classes
    assert tuple(diagnostic.vars) == ("values", "reference")
    values = diagnostic.vars["values"]
    reference = diagnostic.vars["reference"]

    assert values.shape == ("sample",)
    npt.assert_array_equal(values.dimensions["sample"], np.arange(2))
    assert values.is_complex is False
    assert values.is_time_dependent is True

    assert reference.shape == ()
    assert reference.dimensions == {}
    assert reference.is_complex is True
    assert reference.is_time_dependent is False

    # Mutable option defaults must not leak from one diagnostic to the next
    first_default = _ExampleDiagnostic(system, sentinel.output)
    second_default = _ExampleDiagnostic(system, sentinel.output)
    first_default.labels.append("private")
    assert second_default.labels == []

    # Misspelled or unsupported options should fail at construction time
    with pytest.raises(
        KeyError,
        match="Unknown option 'unknown' for diagnostic 'example'",
    ):
        _ExampleDiagnostic(
            system,
            sentinel.output,
            options={"unknown": 1},
        )

    # Exercise the configured diagnostic through a complete cache lifecycle
    diagnostic.ready()
    diagnostic.execute()
    diagnostic.execute()

    # Each variable owns its cache and receives data through the normal hooks
    assert diagnostic.vars["reference"].data_cache == [1.0 + 2.0j]
    assert len(diagnostic.vars["values"].data_cache) == 2

    npt.assert_array_equal(
        diagnostic.vars["values"].data_cache[0],
        np.array([4.0, 2.5]),
    )
    assert (
        diagnostic.vars["values"].data_cache
        is not diagnostic.vars["reference"].data_cache
    )

    # Accidentally replacing an existing output variable is forbidden
    with pytest.raises(
        KeyError,
        match="Diagnostic example already has a variable: values",
    ):
        diagnostic.add_var(
            FlucsDiagnosticVariable(
                name="values",
                shape=(),
                dimensions={},
                is_complex=False,
            )
        )

    # One clear prepares every variable for another round of output
    diagnostic.clear()
    assert all(not var.data_cache for var in diagnostic.vars.values())


###############################################################################
# GPU tests
###############################################################################


@pytest.mark.runtime_precision("single")
def test_runtime_diagnostics_follow_their_declared_contract(runtime_run):
    """
    Real TestSystem diagnostics emit data matching their declared metadata.
    """

    # Walk the configured outputs rather than assuming a diagnostic catalogue
    system = runtime_run.system
    available_diagnostics = system.get_available_diags()
    diagnostic_count = 0

    for output in system.output_heap or ():
        configured = system.input[f"output.{output.name}.diags"]
        configured_names = [
            entry if isinstance(entry, str) else entry["name"]
            for entry in configured
        ]
        assert [diagnostic.name for diagnostic in output.diagnostics] == (
            configured_names
        )

        for diagnostic, entry in zip(
            output.diagnostics,
            configured,
            strict=True,
        ):
            diagnostic_count += 1
            assert type(diagnostic) is available_diagnostics[diagnostic.name]

            # Runtime options should reach the genuine diagnostic instance
            options = {} if isinstance(entry, str) else entry.get("options", {})
            for option_name, option_value in options.items():
                actual_value = getattr(diagnostic, option_name)
                if isinstance(actual_value, np.ndarray):
                    npt.assert_array_equal(actual_value, option_value)
                else:
                    assert actual_value == option_value

            for variable in diagnostic.vars.values():
                dimension_names = tuple(
                    name.rsplit("/", maxsplit=1)[-1]
                    for name in variable.dimensions
                )
                assert tuple(variable.shape) == dimension_names

        # Text diagnostics are scalar, with one value in every saved row
        if isinstance(output, FlucsOutputText):
            lines = output.filepath.read_text(encoding="utf-8").splitlines()
            header = lines[0].split()
            variable_names = [
                variable.name
                for diagnostic in output.diagnostics
                for variable in diagnostic.vars.values()
            ]
            assert header[len(output.timing_data_column_names) :] == (
                variable_names
            )
            for line in lines[1:]:
                values = line.split()[len(output.timing_data_column_names) :]
                assert len(values) == len(variable_names)
                assert all(np.isfinite(complex(value)) for value in values)

        # NetCDF diagnostics retain shapes, dimensions, and complex semantics
        if isinstance(output, FlucsOutputNC):
            with Dataset(output.filepath, "r", format="NETCDF4") as dataset:
                run_group = dataset.groups[output.group_name]
                time_count = len(run_group.variables["time"])

                for diagnostic in output.diagnostics:
                    diagnostic_group = run_group.groups[diagnostic.name]
                    for variable in diagnostic.vars.values():
                        dimension_sizes = tuple(
                            len(as_numpy(values))
                            for values in variable.dimensions.values()
                        )
                        expected_shape = (
                            (time_count, *dimension_sizes)
                            if variable.is_time_dependent
                            else dimension_sizes
                        )

                        # Coordinate variables match the diagnostic declaration
                        for (
                            name,
                            expected_values,
                        ) in variable.dimensions.items():
                            coordinate = get_netcdf_variable(
                                diagnostic_group,
                                name,
                            )
                            npt.assert_allclose(
                                coordinate[:],
                                as_numpy(expected_values),
                                rtol=runtime_run.precision.tolerance,
                                atol=runtime_run.precision.tolerance,
                            )

                        stored_names = get_stored_variable_names(
                            system,
                            variable,
                        )
                        stored_data = [
                            np.asarray(
                                get_netcdf_variable(
                                    diagnostic_group,
                                    name,
                                )[:]
                            )
                            for name in stored_names
                        ]
                        assert all(
                            data.shape == expected_shape for data in stored_data
                        )

                        data = (
                            stored_data[0] + 1j * stored_data[1]
                            if variable.is_complex
                            else stored_data[0]
                        )
                        assert data.size > 0
                        assert np.all(np.isfinite(data))

    # A runtime input without a diagnostic would not exercise this contract
    assert diagnostic_count > 0
