"""Abstract base class for a system that can be solved by FourierSolver.
"""

from abc import ABC
from flucs.systems import FlucsSystem

class FourierSystem(FlucsSystem):
    def initialise(self):
        print("initialising system!")
