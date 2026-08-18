"""Setup shared by all test files."""

from pathlib import Path
from unittest.mock import MagicMock, create_autospec

import pytest

import flucs
from flucs.solvers import FlucsSolver
from flucs.systems import FlucsSystem


@pytest.fixture(scope="session")
def testdata() -> Path:
    """Path to the test data directory."""
    return Path(__file__).parent / "__testdata__"


def pytest_configure(config):
    """Add markers that can be used to skip tests depending on certain
    conditions, such as missing dependencies."""
    config.addinivalue_line(
        "markers", "fluid_itg: mark test as requiring flucs_fluid_itg plugin."
    )


def pytest_collection_modifyitems(config, items):
    """Skip tests marked with 'fluid_itg' if the flucs_fluid_itg plugin is not
    installed."""
    try:
        import flucs_fluid_itg  # noqa: F401
    except ImportError:
        skip_fluid_itg = pytest.mark.skip(
            reason="need flucs_fluid_itg plugin to run"
        )
        for item in items:
            if "fluid_itg" in item.keywords:
                item.add_marker(skip_fluid_itg)


@pytest.fixture
def mock_solver(monkeypatch):
    """Monkeypatches ``flucs.get_solver_type`` to
    return a ``MagicMock`` object.

    We can test which methods are called on this object and
    ensure it is being used in the expected way. It will be
    cleaned up after use.
    """
    # Set up fake solver object
    solver = MagicMock(name="MockSolver", spec=FlucsSolver)
    solver_type = create_autospec(
        FlucsSolver, instance=False, name="MockSolver", return_value=solver
    )

    # Replace get_solver_type with a function that returns the mock solver
    def mock_get_solver_type(solver_name: str):
        if solver_name != "MockSolver":
            raise KeyError(f"Solver '{solver_name}' not found.")
        return solver_type

    monkeypatch.setattr(flucs, "get_solver_type", mock_get_solver_type)

    return solver_type


@pytest.fixture
def mock_system(monkeypatch):
    """Monkeypatches ``flucs.get_system_type`` to
    return a ``MagicMock`` object.

    We can test which methods are called on this object and
    ensure it is being used in the expected way. It will be
    cleaned up after use.
    """
    # Set up fake system object
    system = MagicMock(name="MockSystem", spec=FlucsSystem)
    system_type = create_autospec(
        FlucsSystem, instance=False, name="MockSystem", return_value=system
    )

    # Replace get_system_type with a function that returns the mock system
    def mock_get_system_type(system_name: str):
        if system_name != "MockSystem":
            raise KeyError(f"System '{system_name}' not found.")
        return system_type

    monkeypatch.setattr(flucs, "get_system_type", mock_get_system_type)

    return system_type
