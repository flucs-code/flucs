"""Pseudospectral Fourier-space solver.

Solves a system of PDEs in a periodic box using
pseudospectral Fourier methods.

"""

import time
from flucs.solvers import FlucsSolver, FlucsSolverState
from flucs.solvers.fourier.fourier_system import FourierSystem


class FourierSolver(FlucsSolver[FourierSystem]):
    def run(self):
        """Run the main solver loop."""

        # We first time the solver
        self.state = FlucsSolverState.TIMING
        # Get the system ready
        self.system.setup()

        # Timing
        self.system.ready()

        # self.system.module_options.define_constant("PRECOMPUTE_LINEAR_MATRIX", 1)


        time_taken = self._solver_loop()
        print(f'Timed {self.system.input["setup.timing_steps"]} steps, '
              f'which took {time_taken} s.')

        if self.system.input["setup.timing"]:
            print("Only timing so exiting now!")
            return

        # Reset system and actually run it
        self.state = FlucsSolverState.RUNNING
        self.system.ready()

        self._solver_loop()

        print("flucs given!")

    def _not_done(self) -> bool:
        if self.state == FlucsSolverState.TIMING:
            return self.system.current_step\
                   < self.system.input["setup.timing_steps"]

        return self.system.current_time < self.system.input["time.tfinal"]


    def _solver_loop(self) -> float:
        start_time = time.time()

        while self._not_done():
            self.system.execute_diagnostics()

            self.system.begin_time_step()

            self.system.calculate_nonlinear_terms()

            self.system.finish_time_step()

        end_time = time.time()

        self.system.execute_diagnostics(ignore_next_save=True)
        self.system.write_output()

        return end_time - start_time


