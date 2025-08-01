"""Pseudospectral Fourier implementation of the Ivanov et al. (2020) 2D fluid ITG system.

The nonlinear term is handled explicitly using the Adams-Bashforth 3-step method.

"""

from importlib.resources import files
import numpy as np
from netCDF4 import Dataset
from flucs import FlucsInput
from flucs.solvers.fourier.system import FourierSystem

try:
    import cupy as cp
except ModuleNotFoundError:
    print("CuPy not found!")


class ColdITG2DFourier(FourierSystem):
    def initialise(self):
        self.allocate_memory()
        self.setup_kernels()
        self.setup_initial_conditions()


    def allocate_memory(self):
        pass


    def setup_initial_conditions(self):
        pass


    def setup_kernels(self):
        pass
