"""Tests that plugins are properly registered."""

import pytest

import flucs.flucs as flucs
from flucs.solvers import FlucsSolver
from flucs.systems import FlucsSystem


def test_get_solver_type(mock_solver):
    """
    Test that our mocking of ``get_solver_type`` works as expected.

    Note that we can't import the `get_solver_type` function directly,
    as that would create a new reference to the function that is not affected by
    the monkeypatching.
    """
    solver = flucs.get_solver_type("MockSolver")
    assert "MockSolver" in solver.__repr__()
    assert isinstance(solver, FlucsSolver)


def test_get_system_type(mock_system):
    system = flucs.get_system_type("MockSystem")
    assert "MockSystem" in system.__repr__()
    assert isinstance(system, FlucsSystem)
