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

        # We first time the solver
        # self.system.module_options.define_constant("PRECOMPUTE_LINEAR_MATRIX", 1)

        self.system.ready()

        self.state = FlucsSolverState.TIMING


        start_time = time.time()

        while self.system.current_step < self.system.input["setup.timing_steps"]:
            self.system.calculate_nonlinear_terms()

            self.system.finish_time_step()
            # print("a")
        
        end_time = time.time()
        many_steps_timespan = end_time - start_time
        print(f"Timed {self.system.input["setup.timing_steps"]} steps, which took {many_steps_timespan} s.")


        print("done")
