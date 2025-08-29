"""Pseudospectral Fourier implementation of the Ivanov et al. (2020) 2D fluid ITG system.

The nonlinear term is handled explicitly using the Adams-Bashforth 3-step method.

"""

from importlib.resources import files
import numpy as np
from netCDF4 import Dataset
from flucs import FlucsInput
from flucs.solvers.fourier import FourierSystem

try:
    import cupy as cp
except ModuleNotFoundError:
    print("CuPy not found!")


class ColdITG2DFourier(FourierSystem):
    """Fourier solver for the 2D system."""
    # GPU memory

    # Direct pointers to the phi and T arrays
    phi: list
    T: list

    # Nonlinear terms at this and at the previous time step
    nonlinear_terms: list = None

    # CPU memory


    def initialise(self):
        self.allocate_memory()
        self.setup_kernels()
        self.set_initial_conditions()


    def allocate_memory(self):
        #GPU arrays

        # For the field arrays, we need to keep the fields
        # at the current time step and the previous one.

        self.fields = [cp.zeros((2, self.nz, self.nx, self.half_ny),
                                dtype=self.complex),
                       cp.zeros((2, self.nz, self.nx, self.half_ny),
                                dtype=self.complex)]

        self.phi = [cp.ndarray((self.nz, self.nx, self.half_ny),
                               dtype=self.complex,
                               memptr=self.fields[0][0, 0, 0, 0].data),
                    cp.ndarray((self.nz, self.nx, self.half_ny),
                               dtype=self.complex,
                               memptr=self.fields[1][0, 0, 0, 0].data),]

        self.T = [cp.ndarray((self.nz, self.nx, self.half_ny),
                               dtype=self.complex,
                               memptr=self.fields[0][1, 0, 0, 0].data),
                    cp.ndarray((self.nz, self.nx, self.half_ny),
                               dtype=self.complex,
                               memptr=self.fields[1][1, 0, 0, 0].data),]

        # For the nonlinear terms, we need to keep terms at the current time
        # step + terms from the past 3 time steps (since we will be using AB3)
        self.nonlinear_terms = [cp.zeros((2, self.nz, self.nx, self.half_ny),
                                         dtype=self.complex)
                                for i in range(4)]


        # CPU arrays


    def set_initial_conditions(self):
        super().set_initial_conditions()

        # Add all custom stuff here


    def setup_kernels(self):
        pass
