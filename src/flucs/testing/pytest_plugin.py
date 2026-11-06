"""
Shared pytest markers and CUDA availability checks.
"""

import pytest

from flucs import cupy as cp
from flucs.flucs import CUPY_IMPORT_ERROR

_GPU_UNAVAILABLE_REASON = pytest.StashKey[str | None]()


def pytest_configure(config):
    """
    Register the shared test categories.
    """
    config.addinivalue_line("markers", "cpu: does not require a CUDA device")
    config.addinivalue_line("markers", "gpu: requires a CUDA device")
    config.addinivalue_line("markers", "slow: expensive tests (cpu or gpu)")


def _probe_gpu():
    """
    Check if there is an available CUDA device.
    """
    if cp is None:
        return f"CuPy is unavailable: {CUPY_IMPORT_ERROR}"

    try:
        if cp.cuda.runtime.getDeviceCount() == 0:
            return "CUDA reported no available devices"

        probe = cp.zeros(1)
        cp.cuda.get_current_stream().synchronize()
        del probe
    except Exception as error:
        return f"CUDA availability check failed: {error}"

    return None


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item):
    """
    Skip GPU tests before setup when a CUDA device is unavailable.
    """
    if item.get_closest_marker("gpu") is None:
        return

    # Keep the probe result on the session, including a successful None.
    if _GPU_UNAVAILABLE_REASON not in item.session.stash:
        item.session.stash[_GPU_UNAVAILABLE_REASON] = _probe_gpu()

    reason = item.session.stash[_GPU_UNAVAILABLE_REASON]
    if reason is not None:
        pytest.skip(reason)
