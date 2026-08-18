"""Setup shared by all test files."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

import flucs.flucs as flucs
from flucs.solvers import FlucsSolver
from flucs.systems import FlucsSystem


@pytest.fixture(scope="session")
def testdata() -> Path:
    """Path to the test data directory."""
    return Path(__file__).parent / "__testdata__"


@pytest.fixture
def mock_solver(monkeypatch):
    """Monkeypatches ``flucs.get_solver_type`` to
    return a ``MagicMock`` object.

    We can test which methods are called on this object and
    ensure it is being used in the expected way. It will be
    cleaned up after use.
    """
    # Set up fake solver object
    mock_solver = MagicMock(name="MockSolver", spec=FlucsSolver)

    # Replace get_solver_type with a function that returns the mock solver
    def mock_get_solver_type(solver_name: str):
        if solver_name != "MockSolver":
            raise KeyError(f"Solver '{solver_name}' not found.")
        return mock_solver

    monkeypatch.setattr(flucs, "get_solver_type", mock_get_solver_type)

    return mock_solver


@pytest.fixture
def mock_system(monkeypatch):
    """Monkeypatches ``flucs.get_system_type`` to
    return a ``MagicMock`` object.

    We can test which methods are called on this object and
    ensure it is being used in the expected way. It will be
    cleaned up after use.
    """
    # Set up fake system object
    mock_system = MagicMock(name="MockSystem", spec=FlucsSystem)

    # Replace get_system_type with a function that returns the mock system
    def mock_get_system_type(system_name: str):
        if system_name != "MockSystem":
            raise KeyError(f"System '{system_name}' not found.")
        return mock_system

    monkeypatch.setattr(flucs, "get_system_type", mock_get_system_type)

    return mock_system
