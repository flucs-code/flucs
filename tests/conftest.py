"""
Shared pytest configuration for the FLUCS test suite.
"""

from __future__ import annotations

import importlib

import pytest

from tests.support.support import (
    TEST_PRECISIONS,
    TEST_SYSTEMS,
    RuntimeRun,
    is_test_selected,
    registered_test_systems,
    resolve_test_ownership,
    select_test_systems,
    write_runtime_input,
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
        "--cpu",
        action="store_true",
        help="Run only CPU tests.",
    )
    group.addoption(
        "--gpu",
        action="store_true",
        help="Run only GPU tests (requires a CUDA device).",
    )
    group.addoption(
        "--long",
        action="store_true",
        help="Include long-running tests in the selected test classes.",
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
        "  --cpu\n"
        "      Run CPU tests only, without probing for a CUDA device.\n"
        "  --gpu\n"
        "      Run GPU tests only after checking for a usable CUDA device.\n"
        "  --long\n"
        "      Include long-running tests in the selected test classes.\n"
        "  --flucs-help\n"
        "      Show this help and exit.\n"
        "\n"
        f"Available test solvers: {available_solvers}\n"
        "\n"
        "Examples:\n"
        "  pytest\n"
        "  pytest --cpu\n"
        "  pytest --gpu\n"
        "  pytest --gpu --long\n"
        "  pytest --core\n"
        "  pytest --solvers all\n"
        "  pytest --solvers <name of solver> --gpu --long"
    )
    return pytest.ExitCode.OK


def _probe_gpu() -> tuple[bool, Exception | None]:
    """
    Return whether a CUDA device is usable and any underlying error.
    """
    # Import flucs
    flucs = importlib.import_module("flucs")
    flucs_module = importlib.import_module("flucs.flucs")

    # Use the cupy import from FLUCS
    if flucs.cupy is None:
        return False, flucs_module.CUPY_IMPORT_ERROR

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
        return False, error

    return True, None


def pytest_configure(config):
    """
    Validate command-line selection before test collection begins.
    """
    if config.getoption("--flucs-help"):
        return

    # Explicit device selections are alternatives
    cpu_only = config.getoption("--cpu")
    gpu_only = config.getoption("--gpu")
    if cpu_only and gpu_only:
        raise pytest.UsageError("--cpu and --gpu cannot be used together")

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

    # Resolve one device mode for collection
    if cpu_only:
        selected_devices = frozenset({"cpu"})
        device_report = "CPU only (explicit --cpu)"
    else:
        gpu_available, gpu_error = _probe_gpu()
        if gpu_only and not gpu_available:
            raise pytest.UsageError(
                "GPU tests were requested, but no usable CUDA device was "
                "found.\n"
                f"Underlying error: {gpu_error!r}"
            )

        if gpu_only:
            selected_devices = frozenset({"gpu"})
            device_report = "GPU only (explicit --gpu)"
        elif gpu_available:
            selected_devices = frozenset({"cpu", "gpu"})
            device_report = "CPU and GPU"
        else:
            selected_devices = frozenset({"cpu"})
            device_report = "CPU only (no usable CUDA device detected)"

    config._flucs_selected_devices = selected_devices
    config._flucs_device_report = device_report


def pytest_report_header(config):
    """
    Report the resolved FLUCS device and duration selections.
    """
    if not hasattr(config, "_flucs_selected_devices"):
        return None

    long_report = "included" if config.getoption("--long") else "excluded"
    return [
        f"FLUCS devices: {config._flucs_device_report}",
        f"FLUCS long tests: {long_report}",
    ]


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


def _resolve_device_class(node) -> str:
    """
    Validate and return the single device class assigned to a test item.
    """
    cpu_markers = list(node.iter_markers("cpu"))
    gpu_markers = list(node.iter_markers("gpu"))
    device_markers = (*cpu_markers, *gpu_markers)
    nodeid = getattr(node, "nodeid", node.name)

    if len(device_markers) != 1:
        raise pytest.UsageError(
            f"{nodeid} must have exactly one of the cpu and gpu markers"
        )

    marker = device_markers[0]
    if marker.args or marker.kwargs:
        raise pytest.UsageError(
            f"{nodeid} device markers do not accept arguments"
        )

    return "cpu" if cpu_markers else "gpu"


def _is_long_test(node) -> bool:
    """
    Validate the optional duration marker and return whether it is present.
    """
    markers = list(node.iter_markers("long"))
    nodeid = getattr(node, "nodeid", node.name)

    if len(markers) > 1:
        raise pytest.UsageError(f"{nodeid} must have at most one long marker")
    if markers and (markers[0].args or markers[0].kwargs):
        raise pytest.UsageError(
            f"{nodeid} long markers do not accept arguments"
        )

    return bool(markers)


def _resolve_runtime_precisions(node):
    """
    Return the requested runtime precision or every supported precision.
    """
    markers = list(node.iter_markers("runtime_precision"))
    if not markers:
        return TEST_PRECISIONS

    nodeid = getattr(node, "nodeid", node.name)
    if len(markers) != 1:
        raise pytest.UsageError(
            f"{nodeid} must have at most one runtime_precision marker"
        )

    marker = markers[0]
    if len(marker.args) != 1 or marker.kwargs:
        raise pytest.UsageError(
            f"{nodeid} runtime_precision requires exactly one precision name"
        )

    precision_name = marker.args[0]
    precisions = {precision.name: precision for precision in TEST_PRECISIONS}
    if precision_name not in precisions:
        available = ", ".join(precisions)
        raise pytest.UsageError(
            f"{nodeid} has unknown runtime precision {precision_name!r}. "
            f"Available: {available}"
        )

    return (precisions[precision_name],)


def pytest_collection_modifyitems(config, items):
    """
    Validate ownership and deselect tests outside the requested selection.
    """

    # Validate ownership and deselect tests outside the requested selection
    selected_devices = config._flucs_selected_devices
    include_long = config.getoption("--long")
    core_only = config.getoption("--core")
    selected_solvers = config._flucs_selected_solvers

    # Initalise list
    selected = []
    deselected = []

    for item in items:
        # Resolve ownership once so collection and parametrization agree
        ownership = _resolve_node_ownership(item)
        device_class = _resolve_device_class(item)
        is_long = _is_long_test(item)

        # Precision restrictions apply only to complete runtime fixtures
        if list(item.iter_markers("runtime_precision")):
            if "runtime_run" not in item.fixturenames:
                raise pytest.UsageError(
                    f"{item.nodeid} uses runtime_precision without "
                    "requesting runtime_run"
                )
            _resolve_runtime_precisions(item)

        include_item = device_class in selected_devices
        include_item = include_item and (include_long or not is_long)

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

    # Share each complete runtime over every module that inspects its results
    fixture_names = set(metafunc.fixturenames)
    if "runtime_run" in fixture_names:
        ownership = _resolve_node_ownership(metafunc.definition)
        selected_solvers = metafunc.config._flucs_selected_solvers
        systems = select_test_systems(ownership, selected_solvers)
        precisions = _resolve_runtime_precisions(metafunc.definition)

        runtime_cases = []
        for system in systems:
            for precision in precisions:
                runtime_cases.append(
                    pytest.param(
                        (system, precision),
                        marks=(
                            pytest.mark.gpu
                            if system.runtime_requires_gpu
                            else pytest.mark.cpu
                        ),
                        id=f"{system.solver_name}-{precision.name}",
                    )
                )

        metafunc.parametrize(
            "runtime_run",
            runtime_cases,
            indirect=True,
            scope="session",
        )
        return

    # Ordinary core tests construct an inexpensive TestSystem without running
    if "test_system" not in fixture_names:
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


@pytest.fixture(scope="session")
def runtime_run(request, tmp_path_factory, _register_test_systems):
    """
    Run one TestSystem and retain its objects and artifacts for inspection.
    """
    from flucs.flucs import run_flucs

    test_system, precision = request.param

    # Give each solver and precision an isolated, persistent session directory
    path_name = f"{test_system.solver_name}-{precision.name}"
    io_path = tmp_path_factory.mktemp(path_name)
    input_path = io_path / "input.toml"

    write_runtime_input(input_path, test_system, precision=precision)

    # Execute the normal public runtime exactly once for this parameter pair
    flucs_input, solver = run_flucs(input_path)
    return RuntimeRun(
        test_system=test_system,
        precision=precision,
        io_path=io_path,
        flucs_input=flucs_input,
        solver=solver,
        system=solver.system,
    )


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
