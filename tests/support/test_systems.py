"""
Test-only solver/system registration.
"""

from __future__ import annotations

import importlib
from contextlib import contextmanager
from dataclasses import dataclass
from importlib.metadata import EntryPoints
from types import SimpleNamespace

__test__ = False


@dataclass(frozen=True)
class TestSystemSpec:
    """
    A standalone system used to exercise one solver.
    """
    # Standard attributes
    solver_name: str
    system_name: str
    system_path: str

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
