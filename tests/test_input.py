import re
from textwrap import dedent

import pytest

from flucs.input import FlucsInput


def test_flucs_input(mock_input_path, mock_solver, mock_system):
    """Test that FlucsInput can read the cold_itg_2d_fourier test case."""
    input = FlucsInput(mock_input_path)
    assert input._solver_type == mock_solver
    assert input._system_type == mock_system
    assert input.input_path == mock_input_path
    assert input._initialised
    assert input["parameters.alpha"] == 1.0
    assert input["parameters.beta"] == 2.0
    assert input["dimensions.nx"] == 64
    assert input["dimensions.ny"] == 64
    assert input["dimensions.nz"] == 1


def test_create_solver_system(mock_input_path, mock_solver, mock_system):
    input = FlucsInput(mock_input_path)
    # Test that the correct solver and system types are loaded
    solver, system = input.create_solver_system()
    assert solver == mock_solver.return_value
    assert system == mock_system.return_value


def test_load_dict():
    # Create an uninitialized instance
    input = FlucsInput.__new__(FlucsInput)
    input._input_dict = {}
    input._default_input_dict = {}

    default_dict = {
        "parameters": {"alpha": 1.0, "beta": 1.0},
        "dimensions": {"nx": 1, "ny": 1},
    }

    # Test with default=False
    with pytest.raises(ValueError) as excinfo:
        input.load_dict(default_dict)
    assert re.search(r"Parameter '[a-z]*' is invalid", str(excinfo.value))

    # Test with default=True
    input.load_dict(default_dict, default=True)
    assert input["parameters"] == {"alpha": 1.0, "beta": 1.0}
    assert input["dimensions"] == {"nx": 1, "ny": 1}

    # Test with default=False, which should override the defaults
    new_dict = {
        "parameters": {"alpha": 2.0, "beta": 3.0},
        "dimensions": {"nx": 4, "ny": 5},
    }
    input.load_dict(new_dict)
    assert input["parameters"] == {"alpha": 2.0, "beta": 3.0}
    assert input["dimensions"] == {"nx": 4, "ny": 5}


def test_load_toml_str():
    # Create an uninitialized instance
    input = FlucsInput.__new__(FlucsInput)
    input._input_dict = {}
    input._default_input_dict = {}

    default_toml = dedent("""\
        [parameters]
        alpha = 1.0
        beta = 1.0

        [dimensions]
        nx = 1
        ny = 1
    """)

    # Test with default=False
    with pytest.raises(ValueError) as excinfo:
        input.load_toml_str(default_toml)
    assert re.search(r"Parameter '[a-z]*' is invalid", str(excinfo.value))

    # Test with default=True
    input.load_toml_str(default_toml, default=True)
    assert input["parameters"] == {"alpha": 1.0, "beta": 1.0}
    assert input["dimensions"] == {"nx": 1, "ny": 1}

    # Test with default=False, which should override the defaults
    new_toml = dedent("""\
        [parameters]
        alpha = 2.0
        beta = 3.0

        [dimensions]
        nx = 4
        ny = 5
    """)
    input.load_toml_str(new_toml)
    assert input["parameters"] == {"alpha": 2.0, "beta": 3.0}
    assert input["dimensions"] == {"nx": 4, "ny": 5}
