"""
Tests for the base diagnostic classes.
"""

from typing import ClassVar
from unittest.mock import sentinel

import numpy as np
import numpy.testing as npt
import pytest

from flucs.diagnostic import FlucsDiagnostic, FlucsDiagnosticVariable

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
                shape=("mode",),
                dimensions={"mode": np.arange(2)},
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


def test_diagnostic_initialisation_and_options():
    """
    Construction loads typed options and prepares independent variables.
    """

    # Supply string and tuple values as they might arrive from user input
    diagnostic = _ExampleDiagnostic(
        system=sentinel.system,
        output=sentinel.output,
        options={
            "count": "4",
            "scale": "2.5",
            "labels": ("first", "second"),
        },
    )

    # Check the common state and option casting established by the base class
    assert diagnostic.system is sentinel.system
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

    assert values.shape == ("mode",)
    npt.assert_array_equal(values.dimensions["mode"], np.arange(2))
    assert values.is_complex is False
    assert values.is_time_dependent is True

    assert reference.shape == ()
    assert reference.dimensions == {}
    assert reference.is_complex is True
    assert reference.is_time_dependent is False

    # Mutable option defaults must not leak from one diagnostic to the next
    first_default = _ExampleDiagnostic(sentinel.system, sentinel.output)
    second_default = _ExampleDiagnostic(sentinel.system, sentinel.output)
    first_default.labels.append("private")
    assert second_default.labels == []

    # Misspelled or unsupported options should fail at construction time
    with pytest.raises(
        KeyError,
        match="Unknown option 'unknown' for diagnostic 'example'",
    ):
        _ExampleDiagnostic(
            sentinel.system,
            sentinel.output,
            options={"unknown": 1},
        )


def test_diagnostic_variable_cache_lifecycle():
    """
    Diagnostic execution fills distinct caches which can be cleared safely.
    """

    # Prepare both the time-independent and evolving diagnostic data
    diagnostic = _ExampleDiagnostic(sentinel.system, sentinel.output)
    diagnostic.ready()
    diagnostic.execute()
    diagnostic.execute()

    # Each variable owns its cache and receives data through the normal hooks
    assert diagnostic.vars["reference"].data_cache == [1.0 + 2.0j]
    assert len(diagnostic.vars["values"].data_cache) == 2
    
    npt.assert_array_equal(
        diagnostic.vars["values"].data_cache[0],
        np.array([2.0, 1.5]),
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
