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
        self.system.setup_restart()

        time_taken = self._solver_loop()

        print(f"flucs given in {time_taken} seconds!")

    def _not_done(self) -> bool:
        if self.state == FlucsSolverState.TIMING:
            return self.system.current_step\
                   < self.system.input["setup.timing_steps"]

        return self.system.current_time < self.system.input["time.tfinal"]

    def _solver_loop(self) -> float:
        is_nonlinear = not self.system.input["setup.linear"]

        # Diagnostics for the first time step
        self.system.execute_diagnostics()

        start_time = time.time()
        while self._not_done():
            self.system.begin_time_step()

            if is_nonlinear:
                self.system.calculate_nonlinear_terms()

            self.system.finish_time_step()
            self.system.execute_diagnostics()
            self.system.write_restart(force=False)
        end_time = time.time()

        self.system.write_output()
        self.system.write_restart(force=True)

        return end_time - start_time
