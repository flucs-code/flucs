"""Pseudospectral Fourier-space solver.

Solves a system of PDEs in a periodic box using
pseudospectral Fourier methods.

"""

import time
from flucs.solvers import FlucsSolver, FlucsSolverState
from flucs.solvers.fourier.fourier_system import FourierSystem


class FourierSolver(FlucsSolver[FourierSystem]):
    """A pseudospectral solver for a system of nonlinear fluid PDEs in 2D or 3D
    that are specified by a FourierSystem.

    """

    def run(self):
        """Run the main solver loop."""

        # We first time the solver
        self.state = FlucsSolverState.TIMING

        # Get the system ready
        self.system.setup()
        self.system.setup_output()
        self.system.compile_cupy_module()
        self.system.get_memory_usage()

        # Timing
        self.system.ready()

        time_taken = self._solver_loop()
        print(f'Timed {self.system.input["setup.timing_steps"]} steps, '
              f'which took {time_taken} s.')

        if self.system.input["setup.timing"]:
            print("Only timing so exiting now!")
            return

        # Reset system and actually run it
        self.state = FlucsSolverState.RUNNING
        self.system.ready()

        time_taken = self._solver_loop()

        print(f"flucs given in {time_taken} seconds.\n")

    def _not_done(self) -> bool:
        if self.interrupted:
            return False

        if self.state == FlucsSolverState.TIMING:
            return self.system.current_step\
                   < self.system.input["setup.timing_steps"]

        return self.system.current_time < self.system.final_time

    def _solver_loop(self) -> float:
        if self.interrupted:
            return 0.0

        is_nonlinear = not self.system.input["setup.linear"]
        # Diagnostics for the first time step
        self.system.execute_diagnostics()

        start_time = time.time()
        self.system.steps_until_next_write = self.system.input["output.write_steps"]

        while self._not_done():
            self.system.begin_time_step()

            if is_nonlinear:
                self.system.calculate_nonlinear_terms()

            self.system.finish_time_step()
            self.system.execute_diagnostics()
            self.system.write_output()
            self.system.restart_manager.write_restart()
        end_time = time.time()

        # One final write
        self.system.write_output(force=True)
        self.system.restart_manager.write_restart(force=True)

        return end_time - start_time
