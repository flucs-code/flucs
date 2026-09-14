"""
Shared parameter sets and architecture for the FLUCS test suite.
"""

from __future__ import annotations

import importlib
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from importlib.metadata import EntryPoints
from types import SimpleNamespace
from typing import Any

import numpy as np

__test__ = False

FLUCS_TOLERANCE_MULTIPLIER = 64.0


@dataclass(frozen=True)
class PrecisionSpec:
    """
    Types, storage metadata, and tolerance for one FLUCS precision.
    """

    name: str
    float_type: type
    complex_type: type
    netcdf_precision: str

    @property
    def tolerance(self):
        """
        Return the baseline round-off tolerance used by FLUCS.
        """
        return self.float_type(
            np.finfo(self.float_type).eps * FLUCS_TOLERANCE_MULTIPLIER
        )


# Exercise every supported precision through the same behavioural tests
SINGLE_PRECISION = PrecisionSpec(
    name="single",
    float_type=np.float32,
    complex_type=np.complex64,
    netcdf_precision="f4",
)
DOUBLE_PRECISION = PrecisionSpec(
    name="double",
    float_type=np.float64,
    complex_type=np.complex128,
    netcdf_precision="f8",
)
TEST_PRECISIONS = (SINGLE_PRECISION, DOUBLE_PRECISION)


@dataclass(frozen=True)
class TestSystemSpec:
    """
    A standalone system used to exercise one solver.
    """

    # Standard attributes
    solver_name: str
    system_name: str
    system_path: str
    input_data: dict[str, Any]

    def create_input_data(self) -> dict[str, Any]:
        """
        Return an independent, minimally valid input for this test system.
        """
        input_data = deepcopy(self.input_data)
        input_data.setdefault("setup", {}).update(
            {
                "solver": self.solver_name,
                "system": self.system_name,
            }
        )
        return input_data

    @property
    def system_type(self) -> type:
        """
        Load and return the test system class.
        """
        module_name, class_name = self.system_path.split(":", maxsplit=1)
        module = importlib.import_module(module_name)
        return getattr(module, class_name)


# Definitive list of test systems for the test suite
TEST_SYSTEMS = {
    "FourierSolver": TestSystemSpec(
        solver_name="FourierSolver",
        system_name="TestFourierSystem",
        system_path="tests.support.fourier:TestFourierSystem",
        input_data={
            "dimensions": {
                "nx": 8,
                "ny": 10,
                "nz": 6,
            }
        },
    ),
}


@dataclass(frozen=True)
class _TestSystemEntryPoint:
    """
    Minimal entry-point interface backed by a test system class.
    """

    name: str
    value: str
    group: str
    system_type: type
    dist = SimpleNamespace(name="flucs-test-suite")

    def load(self):
        return self.system_type

    def matches(self, **parameters):
        return all(
            getattr(self, key) == value for key, value in parameters.items()
        )


def _test_system_entry_points() -> tuple[_TestSystemEntryPoint, ...]:
    """
    Construct entry points with enough metadata for normal CLI listing.
    """
    entries = []
    for system in TEST_SYSTEMS.values():
        entry = _TestSystemEntryPoint(
            name=system.system_name,
            value=system.system_path,
            group="flucs.systems",
            system_type=system.system_type,
        )
        entries.append(entry)

    return tuple(entries)


@contextmanager
def registered_test_systems():
    """
    Temporarily add standalone systems to FLUCS's runtime registry.
    """
    # Import FLUCS
    flucs = importlib.import_module("flucs")
    flucs_module = importlib.import_module("flucs.flucs")

    # Get possible test systems
    test_system_names = {system.system_name for system in TEST_SYSTEMS.values()}

    # Combine production and test systems for lookup
    production_systems = EntryPoints(
        entry
        for entry in flucs_module.systems
        if entry.name not in test_system_names
    )
    test_systems = _test_system_entry_points()
    registered_systems = EntryPoints((*production_systems, *test_systems))

    # Replace usual registration for tests
    flucs_module.systems = registered_systems
    flucs.systems = registered_systems

    # Attempt to register systems
    try:
        for system in TEST_SYSTEMS.values():
            registered_type = flucs.get_system_type(system.system_name)
            if registered_type is not system.system_type:
                raise RuntimeError(
                    f"Failed to register {system.system_name} for testing"
                )
        yield
    finally:
        # Restore registered production systems
        flucs_module.systems = production_systems
        flucs.systems = production_systems

        # Raise an error if any test systems remain registered
        remaining_names = {entry.name for entry in flucs_module.systems}
        unexpected = test_system_names & remaining_names
        if unexpected:
            names = ", ".join(sorted(unexpected))
            raise RuntimeError(f"Failed to unregister test systems: {names}")
