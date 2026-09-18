"""
Tests for the base diagnostic classes.
"""

from typing import ClassVar
from unittest.mock import sentinel

import numpy as np
import numpy.testing as npt
import pytest

from flucs.diagnostic import FlucsDiagnostic, FlucsDiagnosticVariable
from tests.support.support import create_test_solver_system

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


@pytest.mark.cpu
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

    assert diagnostic.count == 4
    assert type(diagnostic.count) is int

    assert diagnostic.scale == 2.5
    assert type(diagnostic.scale) is float

    assert diagnostic.labels == ["first", "second"]
    assert type(diagnostic.labels) is list

    # Both declared variables are registered in their original order
    assert tuple(diagnostic.vars) == ("values", "reference")

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
    Real TestSystem diagnostics follow their declared runtime configuration.
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

    # A runtime input without a diagnostic would not exercise this contract
    assert diagnostic_count > 0
