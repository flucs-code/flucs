"""
Pseudospectral Fourier-space solver.

Solves a system of PDEs in a periodic box using
pseudospectral Fourier methods.

"""

import datetime
import time
from typing import ClassVar

from flucs.solvers import FlucsSolver, FlucsSolverState
from flucs.solvers.fourier.fourier_system import FourierSystem
from flucs.utilities.messages import (
    flucsprint,
    format_seconds,
)

from .timesteppers.ab3 import FourierAB3Timestepper
from .timesteppers.rk4 import FourierRK4Timestepper
from .timesteppers.ssprk3 import FourierSSPRK3Timestepper


class FourierSolver(FlucsSolver[FourierSystem]):
    """
    A pseudospectral solver for a system of nonlinear fluid PDEs in 2D or 3D
    that are specified by a FourierSystem.

    """

    # Supported time steppers
    _supported_timesteppers: ClassVar = {
        "ab3": FourierAB3Timestepper,
        "rk4": FourierRK4Timestepper,
        "ssprk3": FourierSSPRK3Timestepper,
    }

    def setup_cuda_definitions(self) -> None:
        self.timestepper.setup_cuda_definitions()

    def register_kernels(self) -> None:
        self.timestepper.register_kernels()

    def run(self, timing_steps: int = 0):
        """
        Run the main solver loop.
        """

        # Set solver state
        self.timing_steps = timing_steps

        if self.timing_steps:
            self.state = FlucsSolverState.TIMING
            timing = True
        else:
            self.state = FlucsSolverState.RUNNING
            timing = False

        # Get the system ready
        self.system.setup()
        self.timestepper.setup()

        if not timing:
            self.system.setup_output()

        self.system.compile_cupy_module()
        self.system.setup_initial_conditions()
        self.system.check_health()
        self.system.clean_cupy_memory()
        self.system.get_memory_usage()

        if timing:
            flucsprint(f"\nTiming {timing_steps:.3e} steps...")

        # Ready and exceute the solver loop
        self.system.ready()
        self.timestepper.ready()

        self.system.initial_wallclock_time = datetime.datetime.now()
        time_taken = self._solver_loop()

        # Report final results
        if timing:
            flucsprint(
                f"Timed {timing_steps:.3e} steps, "
                f"taking  {time_taken:.3e} seconds.\n"
            )
            return

        flucsprint(
            f"Finished at time {float(self.system.current_time):.3e}, "
            f"dt {float(self.system.current_dt):.3e}"
        )
        flucsprint(
            f"flucs given in {format_seconds(time_taken, verbose=True)} "
            f"({self.system.current_step} steps).\n"
        )

    def _not_done(self) -> bool:
        if self.interrupted:
            return False

        if self.state == FlucsSolverState.TIMING:
            return (
                self.system.current_step
                < self.timing_steps
            )

        return self.system.current_time < self.system.final_time

    def _solver_loop(self) -> float:
        if self.interrupted:
            return 0.0

        # Diagnostics for the first time step
        self.system.execute_diagnostics()

        start_time = time.time()
        self.system.steps_until_next_write = self.system.input[
            "output.write_steps"
        ]

        while self._not_done():
            # Advance step counter
            # After this, current_step indexes the step to be solved for
            # while current_time still points at the previously known
            # solution for the fields. We cannot advance current_time
            # because we do not yet know dt.
            self.system.current_step += 1

            # System-specific start-of-step hook
            self.system.begin_time_step()

            # Perform a step
            self.timestepper.execute_timestep()

            # Update current_time to reflect the time of current_step
            self.system.current_time += self.system.current_dt

            # System-specific end-of-step hook
            self.system.finish_time_step()

            # Diagnostics, output, etc
            self.system.execute_diagnostics()
            self.system.write_output()
            self.system.restart_manager.write_restart()

        end_time = time.time()

        # Force a final write
        self.system.execute_diagnostics(force=True)
        self.system.write_output(force=True)
        self.system.restart_manager.write_restart(force=True)

        return end_time - start_time
