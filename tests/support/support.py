"""
Shared parameter sets and architecture for the FLUCS test suite.
"""

from __future__ import annotations

import importlib
import pathlib as pl
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from importlib.metadata import EntryPoints
from types import SimpleNamespace
from typing import Any

import numpy as np
import toml

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
    runtime_input_path: pl.Path
    runtime_requires_gpu: bool = False

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
        runtime_input_path=pl.Path(__file__).parent / "fourier/runtime.toml",
        runtime_requires_gpu=True,
    ),
}


def _update_test_input(
    input_data: dict[str, Any],
    updates: dict[str, Any],
) -> None:
    """
    Recursively apply valid or deliberately invalid test input changes.
    """
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(input_data.get(key), dict):
            _update_test_input(input_data[key], value)
        else:
            input_data[key] = deepcopy(value)


def write_test_input(
    input_path: pl.Path,
    test_system: TestSystemSpec,
    *,
    precision: PrecisionSpec = SINGLE_PRECISION,
    updates: dict[str, Any] | None = None,
) -> None:
    """
    Write a standalone TestSystem input with optional test-specific changes.
    """
    input_data = test_system.create_input_data()
    input_data["setup"]["precision"] = precision.name

    # Non-dictionary values still permit deliberate replacement of a group
    if updates is not None:
        _update_test_input(input_data, updates)

    input_path.write_text(toml.dumps(input_data), encoding="utf-8")


def write_runtime_input(
    input_path: pl.Path,
    test_system: TestSystemSpec,
    *,
    precision: PrecisionSpec = SINGLE_PRECISION,
    updates: dict[str, Any] | None = None,
) -> None:
    """
    Copy a TestSystem runtime input with optional test-specific changes.
    """

    # Load an independent copy so the shared baseline remains immutable
    input_data = toml.load(test_system.runtime_input_path)
    input_data["setup"]["precision"] = precision.name

    # Permit focused runtime tests to vary the common simulation
    if updates is not None:
        _update_test_input(input_data, updates)

    # Runtime inputs must continue to exercise their registered TestSystem
    setup = input_data.get("setup", {})
    if setup.get("solver") != test_system.solver_name:
        raise ValueError(
            f"Runtime input for {test_system.solver_name} selects solver "
            f"{setup.get('solver')!r}."
        )
    if setup.get("system") != test_system.system_name:
        raise ValueError(
            f"Runtime input for {test_system.solver_name} selects system "
            f"{setup.get('system')!r}."
        )

    input_path.parent.mkdir(parents=True, exist_ok=True)
    input_path.write_text(toml.dumps(input_data), encoding="utf-8")


@dataclass(frozen=True)
class RuntimeRun:
    """
    Objects and artifacts retained from one completed TestSystem run.
    """

    test_system: TestSystemSpec
    precision: PrecisionSpec
    io_path: pl.Path
    flucs_input: Any
    solver: Any
    system: Any


def as_numpy(data) -> np.ndarray:
    """
    Move test data to NumPy without assuming its original array backend.
    """
    if hasattr(data, "get"):
        data = data.get()
    return np.asarray(data)


def get_netcdf_variable(group, variable_path: str):
    """
    Resolve a possibly nested variable path beneath a NetCDF group.
    """
    path_parts = variable_path.split("/")
    for group_name in path_parts[:-1]:
        group = group.groups[group_name]
    return group.variables[path_parts[-1]]


def get_stored_variable_names(system, variable) -> tuple[str, ...]:
    """
    Return the NetCDF names used to store one diagnostic or restart variable.
    """
    if variable.is_complex:
        return (
            f"{variable.name}{system.netcdf_real_suffix}",
            f"{variable.name}{system.netcdf_imag_suffix}",
        )
    return (variable.name,)


def create_test_solver_system(
    io_path: pl.Path,
    test_system: TestSystemSpec,
    *,
    precision: PrecisionSpec = SINGLE_PRECISION,
    updates: dict[str, Any] | None = None,
):
    """
    Construct a registered TestSystem through the normal FLUCS input path.
    """
    from flucs.input import FlucsInput

    # Give every construction an ordinary, self-contained i/o directory
    io_path.mkdir(parents=True, exist_ok=True)
    input_path = io_path / "input.toml"
    write_test_input(
        input_path,
        test_system,
        precision=precision,
        updates=updates,
    )

    # Exercise the same construction path used by a solver run
    flucs_input = FlucsInput(input_path)
    solver, system = flucs_input.create_solver_system()
    return flucs_input, solver, system


@dataclass(frozen=True)
class TestOwnership:
    """
    Validated ownership of a test by core or one solver.
    """

    solver_name: str | None = None

    @property
    def is_core(self) -> bool:
        """
        Return whether this test belongs to shared core functionality.
        """
        return self.solver_name is None


def resolve_test_ownership(
    core_markers: int,
    solver_names: tuple[str, ...],
    available_solvers,
) -> TestOwnership:
    """
    Validate test ownership and return its solver-independent description.
    """
    if core_markers == 1 and not solver_names:
        return TestOwnership()

    if core_markers == 0 and len(solver_names) == 1:
        solver_name = solver_names[0]
        if solver_name not in available_solvers:
            available = ", ".join(available_solvers)
            raise ValueError(
                f"Unknown solver marker {solver_name!r}. Available: {available}"
            )
        return TestOwnership(solver_name=solver_name)

    raise ValueError(
        "must have exactly one ownership marker: "
        "core or solver(<entry-point name>)"
    )


def select_test_systems(
    ownership: TestOwnership,
    selected_solvers: tuple[str, ...] | None,
    test_systems=TEST_SYSTEMS,
) -> tuple[TestSystemSpec, ...]:
    """
    Return TestSystems compatible with validated ownership and CLI selection.
    """
    if ownership.is_core:
        solver_names = (
            tuple(test_systems)
            if selected_solvers is None
            else selected_solvers
        )
    else:
        solver_names = (ownership.solver_name,)

    return tuple(test_systems[name] for name in solver_names)


def is_test_selected(
    ownership: TestOwnership,
    core_only: bool,
    selected_solvers: tuple[str, ...] | None,
) -> bool:
    """
    Return whether CLI selection includes a test with this ownership.
    """
    if core_only:
        return ownership.is_core

    return (
        selected_solvers is None
        or ownership.is_core
        or ownership.solver_name in selected_solvers
    )


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

    # Snapshot both public registry references exactly as they were supplied
    original_module_systems = flucs_module.systems
    original_package_systems = flucs.systems

    # Get possible test systems
    test_system_names = {system.system_name for system in TEST_SYSTEMS.values()}

    # Combine production and test systems for lookup
    production_systems = EntryPoints(
        entry
        for entry in original_module_systems
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
        # Restore both objects, including any production name collisions
        flucs_module.systems = original_module_systems
        flucs.systems = original_package_systems

        if (
            flucs_module.systems is not original_module_systems
            or flucs.systems is not original_package_systems
        ):
            raise RuntimeError("Failed to restore the original system registry")
