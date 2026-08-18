"""Tests functions in the top-level flucs module (flucs/flucs.py)."""

from unittest.mock import MagicMock

import pytest

import flucs
from flucs.input import FlucsInput
from flucs.solvers import FlucsSolver
from flucs.systems import FlucsSystem


def test_get_solver_type():
    solver = flucs.get_solver_type("FourierSolver")
    assert issubclass(solver, FlucsSolver)


def test_get_unknown_solver_type():
    with pytest.raises(KeyError) as excinfo:
        flucs.get_solver_type("FooBar")
    assert "Solver 'FooBar' not found." in str(excinfo.value)


def test_get_mock_solver_type(mock_solver):
    """
    Test that our mocking of ``get_solver_type`` works as expected.

    Note that we can't import the `get_solver_type` function directly,
    as that would create a new reference to the function that is not affected by
    the monkeypatching.
    """
    solver_type = flucs.get_solver_type("MockSolver")
    assert "MockSolver" in solver_type.__repr__()
    input = MagicMock(name="MockInput", spec=FlucsInput)
    system = MagicMock(name="MockSystem", spec=FlucsSystem)
    solver = solver_type(input, system)
    assert isinstance(solver, FlucsSolver)


@pytest.mark.fluid_itg
def test_get_system_type():
    """Tests that ``get_system_type`` returns a valid system type.

    Only runs if we have the flucs_fluid_itg plugin installed, as
    there is no built-in non-abstract system type.
    """
    system = flucs.get_system_type("ColdITG2DFourier")
    assert isinstance(system, FlucsSystem)


def test_unknown_system_type():
    with pytest.raises(KeyError) as excinfo:
        flucs.get_system_type("FooBar")
    assert "System 'FooBar' not found." in str(excinfo.value)


def test_get_mock_system_type(mock_system):
    """
    Test that our mocking of ``get_system_type`` works as expected.

    Note that we can't import the `get_system_type` function directly,
    as that would create a new reference to the function that is not affected by
    the monkeypatching.
    """
    system_type = flucs.get_system_type("MockSystem")
    assert "MockSystem" in system_type.__repr__()
    input = MagicMock(name="MockInput", spec=FlucsInput)
    system = system_type(input)
    assert isinstance(system, FlucsSystem)


def test_list_plugins(capfd):
    # Function writes to stdout
    flucs.list_solvers_and_systems()
    out, _ = capfd.readouterr()

    assert "Installed solvers:" in out
    assert "FourierSolver" in out
    assert "Installed systems:" in out


def test_run_flucs(mock_input_path, mock_solver, mock_system):
    # Run the flucs.run_flucs function with the test input file.
    (input, solver) = flucs.run_flucs(mock_input_path)

    # Test that the input was set up correctly
    assert isinstance(input, FlucsInput)
    assert input._solver_type == mock_solver
    assert input._system_type == mock_system
    assert input._initialised
    assert input.input_path == mock_input_path

    # Test that the solver.run() method was called.
    solver.run.assert_called_once()
