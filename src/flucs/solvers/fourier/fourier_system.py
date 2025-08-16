"""Abstract base class for a system that can be solved by FourierSolver.
"""

from abc import ABC
from flucs.systems import FlucsSystem
from flucs import FlucsInput

class FourierSystem(FlucsSystem):
    def initialise(self):
        print("initialising system!")

    def _validate_input(self):
        """Validates the input file
        and sets up the resolution in Fourier space
        """

        for dim in ["x", "y", "z"]:
            n = self.input[f"dimensions.n{dim}"]

            print(f"n{dim} = {n}")
