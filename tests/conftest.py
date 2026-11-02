"""
Shared pytest support and optional test-system lookup.
"""

import pathlib as pl

import pytest

import flucs

from .support.fourier import TestFourierSystem

pytest_plugins = ["flucs.testing.pytest_plugin"]

FLUCS_HELP = """
flucs test suite
===============

Tests are marked as cpu, gpu, or slow, the latter of which can be in addition to
the other two. By default, pytest runs all tests not marked as slow, with gpu
test skipped if a CUDA device is unavailable. The slow tests are currently a
subset of the gpu tests.

{categories}

Selection
---------

pytest                         Default: exclude slow tests
pytest -m cpu                  All CPU tests
pytest -m "gpu and not slow"   Fast GPU tests only
pytest -m gpu                  All GPU tests
pytest -m slow                 Slow tests only
pytest -m ""                   Entire suite

An explicit -m expression replaces the default "not slow" expression. Therefore,
-m gpu includes slow GPU tests.
"""


################################################################################
# Command-line help
################################################################################


def pytest_addoption(parser):
    """
    Add the local test-suite guide to pytest's help options.
    """
    group = parser.getgroup("flucs", "FLUCS testing")
    group.addoption(
        "--flucs-help",
        action="store_true",
        help="Show FLUCS test coverage and selection commands, then exit.",
    )


@pytest.hookimpl(tryfirst=True)
def pytest_cmdline_main(config):
    """
    Collect the full inventory for help, including slow tests.
    """
    if config.getoption("flucs_help"):
        config.option.markexpr = ""
        config.option.keyword = ""


def pytest_collection_finish(session):
    """
    Print test names and markers grouped by category and file.
    """
    if not session.config.getoption("flucs_help") or session.testsfailed:
        return

    # Collapse parameterized cases, retaining their CPU/GPU and slow marks.
    categories = {"cpu": {}, "gpu": {}}
    tests_path = pl.Path(__file__).parent
    for item in session.items:
        path = item.path.relative_to(tests_path)
        name = item.originalname or item.name
        is_slow = item.get_closest_marker("slow") is not None
        for category, files in categories.items():
            if item.get_closest_marker(category) is not None:
                tests = files.setdefault(path, {})
                tests.setdefault(name, set()).add(is_slow)

    # List core files first, then subdirectories; preserve function order.
    indent = 3 * " "
    lines = []
    for category, files in categories.items():
        lines.append(f"{category}:")
        for path in sorted(
            files, key=lambda path: (len(path.parts) > 1, path.as_posix())
        ):
            lines.append(f"\n{indent}{path.as_posix()}:")
            for name, slow_marks in files[path].items():
                suffix = ""
                if slow_marks == {True}:
                    suffix = " (slow)"
                elif True in slow_marks:
                    suffix = " (some cases slow)"
                lines.append(f"{2 * indent}- {name}{suffix}")
        lines.append("")

    print(FLUCS_HELP.format(categories="\n".join(lines).rstrip()), end="")


@pytest.hookimpl(tryfirst=True)
def pytest_runtestloop(session):
    """
    Stop after help collection without running fixtures or tests.
    """
    if session.config.getoption("flucs_help") and not session.testsfailed:
        return True


################################################################################
# Fixtures
################################################################################


@pytest.fixture(scope="module")
def test_fourier_system():
    """
    Make TestFourierSystem available through normal input construction.
    """
    get_system_type = flucs.get_system_type

    def get_test_system_type(system_name):
        if system_name == "TestFourierSystem":
            return TestFourierSystem
        return get_system_type(system_name)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(flucs, "get_system_type", get_test_system_type)
        yield TestFourierSystem
