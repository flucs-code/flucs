"""
Tests for the shared FLUCS system lifecycle.
"""

from unittest.mock import create_autospec

import numpy as np
import pytest

from flucs import cupy as cp
from flucs.systems import FlucsSystem
from tests.support.support import create_test_solver_system

pytestmark = pytest.mark.core


###############################################################################
# CPU tests
###############################################################################


def test_precision_tolerance_follows_machine_precision(precision):
    """
    The shared tolerance remains anchored to each floating-point precision.
    """

    # Calculate the policy independently of the production helper
    expected_tolerance = precision.float_type(
        np.finfo(precision.float_type).eps * 64.0
    )
    tolerance = FlucsSystem.precision_tolerance(precision.float_type)

    # Check both its numerical value and scalar type
    assert type(tolerance) is precision.float_type
    assert tolerance == expected_tolerance


class _ScheduledOutput:
    """
    Small explicit output implementing the interface scheduled by FlucsSystem.
    """

    def __init__(self, system, name, next_save, save_steps):
        self.system = system
        self.name = name
        self.next_save = next_save
        self.save_steps = save_steps
        self.execute_steps = []
        self.ready_calls = 0
        self.write_calls = 0

    def __lt__(self, other):
        return self.next_save < other.next_save

    def execute(self):
        self.execute_steps.append(self.system.current_step)
        self.next_save += self.save_steps

    def ready(self):
        self.ready_calls += 1
        self.next_save = 0

    def write(self):
        self.write_calls += 1


def test_system_coordinates_diagnostics_output_and_interruption(
    test_system,
    tmp_path,
    monkeypatch,
):
    """
    The base system coordinates scheduled outputs and clean interruption.
    """

    # Construct the selected TestSystem without invoking its solver loop
    _, _, system = create_test_solver_system(tmp_path, test_system)
    fast_output = _ScheduledOutput(
        system,
        name="fast",
        next_save=0,
        save_steps=2,
    )
    slow_output = _ScheduledOutput(
        system,
        name="slow",
        next_save=1,
        save_steps=3,
    )

    # Outputs enter a priority queue and execute only on their scheduled steps
    system.add_output(slow_output)
    system.add_output(fast_output)
    assert system.output_heap[0] is fast_output

    for current_step in range(3):
        system.current_step = current_step
        system.execute_diagnostics()

    assert fast_output.execute_steps == [0, 2]
    assert slow_output.execute_steps == [1]

    # A forced pass fills only data not already due on this exact step
    system.current_step = 3
    system.execute_diagnostics(force=True)

    assert fast_output.execute_steps == [0, 2, 3]
    assert slow_output.execute_steps == [1]

    # Ready prepares every output and resets wall-time-estimate bookkeeping
    system.init_time = system.float(0.25)
    FlucsSystem.ready(system)

    assert fast_output.ready_calls == 1
    assert slow_output.ready_calls == 1

    assert system.time_to_finish_last_step == 0
    assert system.time_to_finish_last_time == system.init_time

    # Disk writes occur at the shared cadence unless explicitly forced
    print_time_estimate = create_autospec(
        system.print_time_estimate,
        spec_set=True,
    )
    monkeypatch.setattr(system, "print_time_estimate", print_time_estimate)
    system.steps_until_next_write = 2

    system.write_output()
    assert fast_output.write_calls == 0
    assert slow_output.write_calls == 0

    system.write_output()
    assert fast_output.write_calls == 1
    assert slow_output.write_calls == 1

    assert system.steps_until_next_write == system.input["output.write_steps"]
    print_time_estimate.assert_called_once_with()

    # A STOP file requests a clean exit after completing the forced write
    stop_path = tmp_path / "STOP"
    stop_path.touch()
    system.write_output(force=True)

    assert fast_output.write_calls == 2
    assert slow_output.write_calls == 2

    assert system.solver.interrupted is True
    assert not stop_path.exists()
    print_time_estimate.assert_called_once_with()


###############################################################################
# GPU tests
###############################################################################


@pytest.mark.gpu
def test_system_reuses_temporary_arrays_and_reports_memory(
    test_system,
    tmp_path,
    precision,
):
    """
    Temporary arrays and memory reports follow the active CUDA device.
    """

    # Only construct the TestSystem; these helpers need no CUDA compilation
    _, _, system = create_test_solver_system(
        tmp_path,
        test_system,
        precision=precision,
    )
    initial_device = cp.cuda.Device().id

    # Matching requests share storage while shape or complexity remains distinct
    complex_array = system.get_temp_array(8, is_complex=True)
    repeated_array = system.get_temp_array(8, is_complex=True)
    real_array = system.get_temp_array(8, is_complex=False)
    larger_array = system.get_temp_array(16, is_complex=True)

    assert complex_array is repeated_array
    assert complex_array is not real_array
    assert complex_array is not larger_array

    assert complex_array.shape == (8,)
    assert real_array.shape == (8,)
    assert larger_array.shape == (16,)

    assert complex_array.dtype == np.dtype(precision.complex_type)
    assert real_array.dtype == np.dtype(precision.float_type)

    assert bool(cp.all(complex_array == 0))
    assert bool(cp.all(real_array == 0))

    # Query one device and check the arithmetic rather than volatile byte totals
    memory_usage = system.get_memory_usage(devices=[initial_device])
    device_key = f"device_{initial_device:03d}"
    device_usage = memory_usage[device_key]

    assert memory_usage["number_of_devices"] > initial_device

    assert device_usage["id"] == initial_device
    assert device_usage["global"]["total"] > 0
    assert device_usage["global"]["used"] == (
        device_usage["global"]["total"] - device_usage["global"]["free"]
    )
    assert device_usage["cupy"]["used"] <= device_usage["cupy"]["total"]
    assert device_usage["cupy"]["free"] == (
        device_usage["cupy"]["total"] - device_usage["cupy"]["used"]
    )

    assert cp.cuda.Device().id == initial_device
