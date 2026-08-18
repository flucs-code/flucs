"""Setup shared by all test files."""

from pathlib import Path
from textwrap import dedent
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


@pytest.fixture
def mock_input_path(tmp_path, mock_solver, mock_system):
    """Create a temporary TOML file for testing FlucsInput.

    Depends on mock_solver and mock_system."""
    toml = dedent("""\
        [setup]
        solver = "MockSolver"
        system = "MockSystem"

        [parameters]
        alpha = 1.0
        beta = 2.0

        [dimensions]
        nx = 64
        ny = 64
        nz = 1
        """)
    toml_path = tmp_path / "setup.toml"
    toml_path.write_text(toml)

    # We also need to modify the behaviour of MockSystem, as loading an input
    # requires the following chain of events:
    #
    # - The [setup] section of the input file is read, and the solver and system
    #   types are determined.
    #   self._system_type.load_defaults(self) is called, with self being the
    #   FlucsInput object.
    # - This looks up an installed TOML file via importlib to set the valid
    #   input fields, and writes these to the FlucsInput
    # - load_dict then uses these input fields to validate the input from
    #   the input file while overriding the defaults.
    #
    # MockSystem does not have an installed TOML file, so we need to bypass this
    # in order to test FlucsInput with our mock system.
    def mock_load_defaults(flucs_input: flucs.FlucsInput):
        default_toml = dedent("""\
            [parameters]
            alpha = 1.0
            beta = 1.0

            [dimensions]
            nx = 1
            ny = 1
            nz = 1

            # Liam: I couldn't get it work without this,
            # but it isn't present in the plugin TOML files,
            # so I don't know what's going wrong here!
            [setup]
            solver = "DefaultSolver"
            system = "DefaultSystem"
            """)
        flucs_input.load_toml_str(default_toml, default=True)

    mock_system.load_defaults.side_effect = mock_load_defaults

    return toml_path
