from pathlib import Path

import pytest

from flucs.input import FlucsInput


@pytest.fixture
def cold_itg_2d_fourier_toml(testdata: Path) -> Path:
    """FlucsInput for the cold_itg test case."""
    return testdata / "cold_itg_2d_fourier.toml"


@pytest.mark.fluid_itg
def test_flucs_input(cold_itg_2d_fourier_toml: Path):
    """Test that FlucsInput can read the cold_itg_2d_fourier test case."""
    input = FlucsInput(cold_itg_2d_fourier_toml)
    assert input["parameters.kappaT"] == 1.0
    assert input["dimensions.nz"] == 1
    assert input["setup.solver"] == "FourierSolver"
    assert input["setup.system"] == "ColdITG2DFourier"

    # Test that the correct solver and system types are loaded
    solver, system = input.create_solver_system()
    assert solver.__class__.__name__ == "FourierSolver"
    assert system.__class__.__name__ == "ColdITG2DFourier"
