"""
Shared pytest support and optional test-system lookup.
"""

import pytest

import flucs

from .support.fourier import TestFourierSystem

pytest_plugins = ["flucs.testing.pytest_plugin"]

FLUCS_HELP = """

FLUCS test suite
===============

Tests are marked as cpu, gpu, or slow, the latter of which can be in addition to
the other two. By default, pytest runs all tests not marked as slow, with gpu
test skipped if a CUDA device is unavailable. The slow tests are currently a
subset of the gpu tests.

Categories
----------

Tests are grouped by production module within src/flucs.
Each name represents all of its parameterized cases.

cpu
    flucs.py
        - test_conflicting_operating_modes

    input.py
        - test_input_is_read_only
        - test_input_precedence

    output.py
        - test_netcdf_open_dataset_retries
        - test_netcdf_open_failure_preserves_cache

    solvers/fourier/fourier_system.py
        - test_prepare_restart_data_resolution

    solvers/fourier/timesteppers/
        - test_update_ab3_coefficients

    utilities/dealiasing.py
        - test_next_smooth_number
        - test_dealiased_multiplication_rfft (NumPy)

gpu
    output.py
        - test_output_write_cached_samples (slow)
        - test_output_restart_history (slow)

    restart.py
        - test_restart_data_and_time (slow)
        - test_restart_reset_time (slow)
        - test_restart_initial_state (slow)
        - test_restart_backup_rotation (slow)

    solvers/fourier/fourier_solver.py
        - test_linear_evolution (slow)
        - test_nonlinear_evolution (slow)

    solvers/fourier/fourier_system.py
        - test_two_thirds_product (slow)
        - test_two_thirds_boundary (slow)
        - test_spherical_phase_shift_boundary (slow)
        - test_polyhedral_phase_shift_boundary (slow)
        - test_phase_shift_memory_models (slow)

    solvers/fourier/fourier_system_reductions.py
        - test_scalar_reduction
        - test_axis_reduction
        - test_axis_cumulative_reduction
        - test_axis_pair_reduction
        - test_shell_reduction
        - test_shell_reduction_boundaries

    solvers/fourier/timesteppers/
        - test_timestepper_convergence (slow)

    utilities/dealiasing.py
        - test_dealiased_multiplication_rfft (CuPy)

Selection
---------

pytest                         Default: exclude slow tests
pytest -m cpu                  CPU tests only
pytest -m "gpu and not slow"    Fast GPU tests only
pytest -m gpu                  All GPU tests, including slow
pytest -m slow                 Slow tests only
pytest -m "cpu or gpu"          Entire suite

An explicit -m expression replaces the default "not slow"
expression. Therefore, -m gpu includes slow GPU tests.

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
    Print the guide without collecting or running tests.
    """
    if config.getoption("flucs_help"):
        print(FLUCS_HELP.strip())
        return pytest.ExitCode.OK


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
