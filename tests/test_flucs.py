"""
Tests for the top-level FLUCS command-line module.
"""

import sys
from importlib.metadata import EntryPoints
from types import SimpleNamespace
from unittest.mock import create_autospec, sentinel

import pytest

import flucs
import flucs.flucs as flucs_module
from flucs.solvers import FlucsSolver
from tests.support.support import (
    TEST_SYSTEMS,
    is_test_selected,
    registered_test_systems,
    resolve_test_ownership,
    select_test_systems,
)

pytestmark = pytest.mark.core


def test_test_system_registry_is_temporary_and_complete(monkeypatch):
    """
    The complete test overlay restores both original registry objects exactly.
    """

    # Every standalone system is visible through the session-level overlay
    for system_spec in TEST_SYSTEMS.values():
        assert (
            flucs.get_system_type(system_spec.system_name)
            is system_spec.system_type
        )

    # Use separate snapshots containing collisions to exercise exact teardown
    original_module_systems = EntryPoints(tuple(flucs_module.systems))
    original_package_systems = EntryPoints(tuple(flucs.systems))
    monkeypatch.setattr(flucs_module, "systems", original_module_systems)
    monkeypatch.setattr(flucs, "systems", original_package_systems)

    with registered_test_systems():
        assert flucs_module.systems is flucs.systems
        for system_spec in TEST_SYSTEMS.values():
            assert (
                flucs.get_system_type(system_spec.system_name)
                is system_spec.system_type
            )

    assert flucs_module.systems is original_module_systems
    assert flucs.systems is original_package_systems


def test_solver_lookup_and_unknown_plugins(test_system):
    """
    Registered plugins can be loaded and missing plugins give useful errors.
    """

    # Load the solver associated with each available standalone test system
    solver_type = flucs.get_solver_type(test_system.solver_name)
    assert issubclass(solver_type, FlucsSolver)

    # Check errors for both kinds of plugin
    with pytest.raises(KeyError, match="Solver 'Missing' not found"):
        flucs.get_solver_type("Missing")

    with pytest.raises(KeyError, match="System 'Missing' not found"):
        flucs.get_system_type("Missing")


def test_test_ownership_resolves_compatible_systems():
    """
    Ownership validation keeps core and solver parametrization consistent.
    """

    # Use sentinels so this policy test does not depend on today's registry
    systems = {
        "FirstSolver": sentinel.first_system,
        "SecondSolver": sentinel.second_system,
    }
    core_ownership = resolve_test_ownership(1, (), systems)
    solver_ownership = resolve_test_ownership(
        0,
        ("SecondSolver",),
        systems,
    )

    # Core follows CLI selection while solver tests use only their own system
    assert select_test_systems(core_ownership, None, systems) == (
        sentinel.first_system,
        sentinel.second_system,
    )
    assert select_test_systems(
        core_ownership,
        ("FirstSolver",),
        systems,
    ) == (sentinel.first_system,)
    assert select_test_systems(solver_ownership, None, systems) == (
        sentinel.second_system,
    )
    assert is_test_selected(core_ownership, True, None)
    assert not is_test_selected(solver_ownership, True, None)
    assert not is_test_selected(
        solver_ownership,
        False,
        ("FirstSolver",),
    )

    # Missing, duplicated, and mixed ownership all fail through one policy
    invalid_ownership = (
        (0, ()),
        (2, ()),
        (1, ("FirstSolver",)),
        (0, ("FirstSolver", "SecondSolver")),
    )
    for core_markers, solver_names in invalid_ownership:
        with pytest.raises(ValueError, match="exactly one ownership marker"):
            resolve_test_ownership(core_markers, solver_names, systems)

    with pytest.raises(ValueError, match="Unknown solver marker"):
        resolve_test_ownership(0, ("MissingSolver",), systems)


def test_list_solvers_and_systems(monkeypatch, capsys):
    """
    Plugin listings are grouped, sorted, and include their distributions.
    """

    # Create deliberately unsorted plugin registries
    solvers = [
        SimpleNamespace(name="ZetaSolver"),
        SimpleNamespace(name="AlphaSolver"),
    ]
    systems = [
        SimpleNamespace(
            name="ZetaSystem",
            dist=SimpleNamespace(name="zeta-package"),
        ),
        SimpleNamespace(
            name="AlphaSystem",
            dist=SimpleNamespace(name="alpha-package"),
        ),
    ]
    monkeypatch.setattr(flucs_module, "solvers", solvers)
    monkeypatch.setattr(flucs_module, "systems", systems)

    # Generate the user-facing listing
    flucs_module.list_solvers_and_systems()
    output = capsys.readouterr().out

    # Check the headings, ordering, and distribution names
    assert "Installed solvers:" in output
    assert output.index("AlphaSolver") < output.index("ZetaSolver")
    assert "Installed systems:" in output
    assert output.index("AlphaSystem") < output.index("ZetaSystem")
    assert "alpha-package" in output
    assert "zeta-package" in output


@pytest.mark.parametrize(
    ("argv", "expected_flucs", "expected_postprocess"),
    [
        pytest.param(
            ["--run", "--io_path", "case"],
            ["--run", "--io_path", "case"],
            None,
            id="ordinary-arguments",
        ),
        pytest.param(
            ["--io_path", "case", "--postprocess", "2", "--save"],
            ["--io_path", "case", "--postprocess"],
            ["2", "--save"],
            id="long-postprocess-option",
        ),
        pytest.param(
            ["-io", "case", "-p", "1", "--show"],
            ["-io", "case", "-p"],
            ["1", "--show"],
            id="short-postprocess-option",
        ),
    ],
)
def test_parse_cli_arguments(
    argv,
    expected_flucs,
    expected_postprocess,
):
    """
    Post-processing arguments are separated from arguments for FLUCS itself.
    """

    # Split at the post-processing option, if present
    flucs_args, postprocess_args = flucs_module.parse_cli_arguments(argv)

    # Check the arguments destined for each parser
    assert flucs_args == expected_flucs
    assert postprocess_args == expected_postprocess


def test_run_flucs_orchestrates_solver(monkeypatch, tmp_path):
    """
    A run constructs its input, launches its solver, and writes a log.
    """

    # Build a minimal input with the override parameter used
    input_path = tmp_path / "input.toml"
    override = ["time.dt_max", "0.1"]

    # Mock input and solver using their real interfaces
    input_constructor = create_autospec(
        flucs_module.FlucsInput,
        spec_set=True,
    )
    flucs_input = input_constructor.return_value
    solver = create_autospec(
        FlucsSolver,
        instance=True,
        spec_set=True,
    )

    # Construct solver/system pair
    flucs_input.create_solver_system.return_value = (solver, sentinel.system)
    monkeypatch.setattr(flucs_module, "FlucsInput", input_constructor)

    # Run through the public helper
    returned_input, returned_solver = flucs_module.run_flucs(
        input_path,
        override,
    )

    # Check construction, execution, and returned debugging objects
    input_constructor.assert_called_once_with(input_path, override)
    flucs_input.create_solver_system.assert_called_once_with()
    solver.run.assert_called_once_with()
    assert returned_input is flucs_input
    assert returned_solver is solver

    # Check that the run header reached the log
    log_contents = (tmp_path / "output.log").read_text(encoding="utf-8")
    assert flucs_module.FLUCS_HEADER in log_contents


def test_main_defaults_to_run_and_combines_overrides(monkeypatch, tmp_path):
    """
    The CLI defaults to running and combines repeated override options.
    """

    # Supply the input file required by the default run mode
    input_path = tmp_path / "input.toml"
    input_path.touch()

    # Mock run parameters using the real function signature
    run_flucs = create_autospec(
        flucs_module.run_flucs,
        spec_set=True,
    )
    monkeypatch.setattr(flucs_module, "run_flucs", run_flucs)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "flucs",
            "--io_path",
            str(tmp_path),
            "--override",
            "time.dt_max",
            "0.1",
            "-o",
            "setup.precision",
            "double",
        ],
    )

    # Invoke the command-line entry point
    flucs_module.main()

    # Check that the default operation and overrides were forwarded
    run_flucs.assert_called_once_with(
        input_path,
        [
            "time.dt_max",
            "0.1",
            "setup.precision",
            "double",
        ],
    )


@pytest.mark.gpu
def test_main_profiles_memory(monkeypatch, tmp_path, capfd):
    """
    Memory profiling records a real GPU allocation and prints its report.
    """

    # Supply the input file required by the profiling run
    input_path = tmp_path / "input.toml"
    input_path.touch()

    # Replace the full solver run with one small, persistent GPU allocation
    profiled_arrays = []
    run_flucs = create_autospec(flucs_module.run_flucs, spec_set=True)
    run_flucs.side_effect = lambda *_: profiled_arrays.append(
        flucs_module.cupy.zeros(1024)
    )
    monkeypatch.setattr(flucs_module, "run_flucs", run_flucs)

    # Request memory profiling from the command-line entry point
    monkeypatch.setattr(
        sys,
        "argv",
        ["flucs", "--io_path", str(tmp_path), "--memory-profile"],
    )

    # Empty the pool so that the hook observes a fresh device allocation
    memory_pool = flucs_module.cupy.get_default_memory_pool()
    memory_pool.free_all_blocks()
    try:
        flucs_module.main()
    finally:
        profiled_arrays.clear()
        memory_pool.free_all_blocks()

    # Check that the profiled operation ran and produced a nonempty report
    run_flucs.assert_called_once_with(input_path, None)
    output = capfd.readouterr().out

    assert "Memory report from CuPy's LineProfileHook:" in output
    root_report = next(
        line for line in output.splitlines() if line.startswith("_root (")
    )
    assert root_report != "_root (0.00B, 0.00B)"
