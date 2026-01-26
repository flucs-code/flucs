"""Tests that plugins are properly registered."""

import pytest

from flucs import get_solver_type, get_system_type, list_solvers_and_systems


def test_get_solver_type():
    assert get_solver_type("FourierSolver")


def test_unknown_solver_type():
    with pytest.raises(KeyError) as excinfo:
        get_solver_type("FooBar")
    assert "Solver 'FooBar' not found." in str(excinfo.value)


def test_get_system_type():
    assert get_system_type("ColdITG2DFourier")


def test_unknown_system_type():
    with pytest.raises(KeyError) as excinfo:
        get_system_type("FooBar")
    assert "System 'FooBar' not found." in str(excinfo.value)


def test_list_plugins(capfd):
    # Function writes to stdout
    list_solvers_and_systems()
    out, _ = capfd.readouterr()

    assert "Installed solvers:" in out
    assert "FourierSolver" in out
    assert "Installed systems:" in out
    assert "ColdITG2DFourier" in out
