"""
Shared pytest configuration for the FLUCS test suite.
"""

from __future__ import annotations

import importlib

import pytest

from tests.support.support import (
    TEST_PRECISIONS,
    TEST_SYSTEMS,
    is_test_selected,
    registered_test_systems,
    resolve_test_ownership,
    select_test_systems,
)


def pytest_addoption(parser):
    """
    Add FLUCS-specific test-selection options.
    """
    group = parser.getgroup("flucs", "FLUCS test selection")
    group.addoption(
        "--flucs-help",
        action="store_true",
        help="Show FLUCS-specific pytest help and exit.",
    )
    group.addoption(
        "--gpu",
        action="store_true",
        help="Include tests that require a working CUDA device.",
    )
    group.addoption(
        "--core",
        action="store_true",
        help="Run only tests of shared FLUCS core functionality.",
    )
    group.addoption(
        "--solvers",
        nargs="+",
        metavar="SOLVER",
        help=(
            "Run core tests and tests for one or more named solver entry "
            "points; use 'all' to select every test solver."
        ),
    )


def pytest_cmdline_main(config):  # Has to be called this for pytest discovery
    """
    Print concise FLUCS-specific help without collecting tests.
    """
    if not config.getoption("--flucs-help"):
        return None

    available_solvers = ", ".join(TEST_SYSTEMS)
    print(
        "FLUCS pytest options:\n"
        "  --core\n"
        "      Run only shared core tests.\n"
        "  --solvers SOLVER [SOLVER ...]\n"
        "      Run core tests and tests for the selected solvers.\n"
        "      Use '--solvers all' to select every test solver.\n"
        "  --gpu\n"
        "      Include GPU tests after checking for a usable CUDA device.\n"
        "  --flucs-help\n"
        "      Show this help and exit.\n"
        "\n"
        f"Available test solvers: {available_solvers}\n"
        "\n"
        "Examples:\n"
        "  pytest\n"
        "  pytest --core\n"
        "  pytest --solvers all\n"
        "  pytest --solvers FourierSolver --gpu"
    )
    return pytest.ExitCode.OK


def _check_gpu() -> None:
    """
    Fail and report underlying CUDA error when GPU tests cannot run.
    """
    # Import flucs
    flucs = importlib.import_module("flucs")
    flucs_module = importlib.import_module("flucs.flucs")

    # Use the cupy import from FLUCS
    if flucs.cupy is None:
        error = flucs_module.CUPY_IMPORT_ERROR
        raise pytest.UsageError(
            "GPU tests were requested, but CuPy could not initialise.\n"
            f"Underlying error: {error!r}"
        )

    # Ensure that FLUCS's import hasn't failed
    try:
        device_count = flucs.cupy.cuda.runtime.getDeviceCount()
        if device_count < 1:
            raise RuntimeError("CUDA reported zero available devices")

        # Check on the first available device
        flucs.cupy.cuda.Device(0).use()
        probe = flucs.cupy.zeros(1)
        flucs.cupy.cuda.runtime.deviceSynchronize()
        del probe

    except Exception as error:
        raise pytest.UsageError(
            "GPU tests were requested, but no usable CUDA device was found.\n"
            f"Underlying error: {error!r}"
        ) from error


def pytest_configure(config):
    """
    Validate command-line selection before test collection begins.
    """
    if config.getoption("--flucs-help"):
        return

    # Core tests are run by default with --solvers, so check for conflict
    requested_solvers = config.getoption("--solvers")
    if config.getoption("--core") and requested_solvers:
        raise pytest.UsageError("--core and --solvers cannot be used together")

    # Expand "all" or validate the requested TestSystems
    if requested_solvers is None:
        selected_solvers = None
    elif "all" in requested_solvers:
        if requested_solvers != ["all"]:
            raise pytest.UsageError(
                "'all' cannot be combined with named solvers in --solvers"
            )
        selected_solvers = tuple(TEST_SYSTEMS)
    else:
        unknown_solvers = [
            name for name in requested_solvers if name not in TEST_SYSTEMS
        ]
        if unknown_solvers:
            unknown = ", ".join(unknown_solvers)
            available = ", ".join(TEST_SYSTEMS)
            raise pytest.UsageError(
                f"Unknown test solver(s): {unknown}. Available: {available}"
            )
        selected_solvers = tuple(dict.fromkeys(requested_solvers))

    if selected_solvers == ():
        available = ", ".join(TEST_SYSTEMS)
        raise pytest.UsageError(
            f"No test solvers were selected. Available: {available}"
        )
    config._flucs_selected_solvers = selected_solvers

    # Make sure we can run the GPU tests if requested
    if config.getoption("--gpu"):
        _check_gpu()


def _solver_marker_name(marker) -> str:
    """
    Validate and return the solver name carried by a marker.
    """

    # Ensure that the markers are one-to-one
    if len(marker.args) != 1 or marker.kwargs:
        raise pytest.UsageError(
            "solver markers require exactly one solver entry-point name"
        )

    return marker.args[0]


def _resolve_node_ownership(node):
    """
    Resolve pytest markers through the shared ownership policy.
    """
    solver_names = tuple(
        _solver_marker_name(marker) for marker in node.iter_markers("solver")
    )

    try:
        return resolve_test_ownership(
            core_markers=len(list(node.iter_markers("core"))),
            solver_names=solver_names,
            available_solvers=TEST_SYSTEMS,
        )
    except ValueError as error:
        nodeid = getattr(node, "nodeid", node.name)
        raise pytest.UsageError(f"{nodeid} {error}") from error


def pytest_collection_modifyitems(config, items):
    """
    Validate ownership and deselect tests outside the requested selection.
    """

    # Validate ownership and deselect tests outside the requested selection
    include_gpu = config.getoption("--gpu")
    core_only = config.getoption("--core")
    selected_solvers = config._flucs_selected_solvers

    # Initalise list
    selected = []
    deselected = []

    for item in items:
        # Resolve ownership once so collection and parametrization agree
        ownership = _resolve_node_ownership(item)
        include_item = include_gpu or item.get_closest_marker("gpu") is None

        # Apply the same shared ownership policy used for parametrization
        include_item = include_item and is_test_selected(
            ownership,
            core_only,
            selected_solvers,
        )

        if include_item:
            selected.append(item)
        else:
            deselected.append(item)

    # Deselect rather than skipping
    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = selected


def pytest_generate_tests(metafunc):
    """
    Parametrize solver-facing core tests over compatible test systems.
    """

    # Make sure we have the necessary fixtures
    if "test_system" not in metafunc.fixturenames:
        return

    # Match core tests to the requested systems and solver tests to their owner
    ownership = _resolve_node_ownership(metafunc.definition)
    selected_solvers = metafunc.config._flucs_selected_solvers
    systems = select_test_systems(ownership, selected_solvers)
    metafunc.parametrize(
        "test_system",
        systems,
        indirect=True,
        ids=lambda system: system.solver_name,
    )


@pytest.fixture
def test_system(request):
    """
    Return one standalone system specification for a core test.
    """
    return request.param


@pytest.fixture(
    params=TEST_PRECISIONS,
    ids=lambda precision: precision.name,
)
def precision(request):
    """
    Return one supported numerical precision specification.
    """
    return request.param


@pytest.fixture(scope="session", autouse=True)
def _register_test_systems():
    """
    Expose standalone systems only for the duration of this test run.
    """
    with registered_test_systems():
        yield
