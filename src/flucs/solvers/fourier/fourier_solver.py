"""Pseudospectral Fourier-space solver.

Solves a system of PDEs in a periodic box using
pseudospectral Fourier methods.

"""

import time
from ..flucs_solver import FlucsSolver, FlucsSolverState
from .fourier_system import FourierSystem


class FourierSolver(FlucsSolver[FourierSystem]):
    def run(self):
        """Run the main solver loop."""

        # Get the system ready
        self.system.setup()

        # Timing
        self.system.ready()

        # self.system.module_options.define_constant("PRECOMPUTE_LINEAR_MATRIX", 1)

        # We first time the solver
        self.state = FlucsSolverState.TIMING

        time_taken = self._solver_loop()
        print(f'Timed {self.system.input["setup.timing_steps"]} steps, '
              f'which took {time_taken} s.')


        # Reset system
        self.system.ready()
        self.state = FlucsSolverState.RUNNING

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
            # print(f"Time step {self.system.current_step}, t = {self.system.current_time}")
            self.system.execute_diagnostics()
            self.system.calculate_nonlinear_terms()

            self.system.finish_time_step()

        end_time = time.time()
        self.system.write_output()

        return end_time - start_time


