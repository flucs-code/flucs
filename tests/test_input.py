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

    # Test that the correct solver and system types are loaded
    solver, system = input.create_solver_system()
    assert solver == mock_solver.return_value
    assert system == mock_system.return_value
