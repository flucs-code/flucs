"""Pseudospectral Fourier-space solver.

Solves a system of PDEs in a periodic box using
pseudospectral Fourier methods.

"""

from importlib.resources import files
from flucs import FlucsInput, FlucsSolver

class FourierSolver(FlucsSolver):
    def run(self):
        """Run the main solver loop."""

        self.system.ready()
        print("Ready to go")
