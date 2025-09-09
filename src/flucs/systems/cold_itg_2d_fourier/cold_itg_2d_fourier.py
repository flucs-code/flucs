"""Pseudospectral Fourier implementation of the Ivanov et al. (2020) 2D fluid
ITG system. The nonlinear term is handled explicitly using the Adams-Bashforth
3-step method.

"""

import numpy as np
import cupy as cp
from netCDF4 import Dataset
from numpy import dtype
import flucs
from flucs.solvers.fourier.fourier_system import FourierSystem
from flucs.output import FlucsOutput
from .cold_itg_2d_fourier_diagnostics import HeatfluxDiag


class ColdITG2DFourier(FourierSystem):
    """Fourier solver for the 2D system."""
    number_of_fields = 2

    # Direct pointers to the phi and T arrays
    phi: list
    T: list

    # Nonlinear terms at this and at the previous time step
    nonlinear_terms: list

    # Markers for the lists of arrays
    current_field_marker = 0
    previous_field_marker = -1

    # Supported diagnostics
    diags_dict = {"heatflux": HeatfluxDiag}


    def setup(self):
        """Prepares the system for the solver."""

        self.allocate_memory()
        # self.setup_kernels()

        super().setup()

    def ready(self):
        super().ready()
        self.fields[0][:] = cp.array(self.fields_initial.reshape(self.fields[0].shape))


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

        self.R = cp.zeros((4, self.nz, self.nx, self.half_ny), dtype=self.complex)
        self.invL = cp.zeros((4, self.nz, self.nx, self.half_ny), dtype=self.complex)


        # For the nonlinear terms, we need to keep terms at the current time
        # step + terms from the past 3 time steps (since we will be using AB3)
        self.nonlinear_terms = [cp.zeros((2, self.nz, self.nx, self.half_ny),
                                         dtype=self.complex)
                                for i in range(4)]


        # CPU arrays

    def _interpret_input(self):
        """Checks if the input file makes sense"""

        # Make sure to call the parent method to do some standard setup
        # (resolution checks, etc)
        super()._interpret_input()

        # Anything custom goes here

        if self.nz != 1 or self.padded_nz != 1:
            raise ValueError("Both nz and padded_nz should be set to 1 for the 2D system!")


    def ready(self) -> None:
        self.module_options.define_constant("CHI", self.input["parameters.chi"])
        self.module_options.define_constant("A_TIMES_CHI",
                                            self.input["parameters.a"]
                                            * self.input["parameters.chi"])

        self.module_options.define_constant("B_TIMES_CHI",
                                            self.input["parameters.b"]
                                            * self.input["parameters.chi"])

        self.module_options.define_constant("KAPPA_T", self.input["parameters.kappaT"])
        self.module_options.define_constant("KAPPA_N", self.input["parameters.kappaN"])
        self.module_options.define_constant("KAPPA_B", self.input["parameters.kappaB"])

        self.fields[self.current_field_marker][:]\
            = cp.array(np.reshape(self.fields_initial, self.fields[0].shape))

        super().ready() # Call this to compile the module



    def begin_time_step(self) -> None:
        self.current_field_marker = (self.current_field_marker + 1) % 2
        self.previous_field_marker = self.current_field_marker - 1

        super().begin_time_step()


    def calculate_nonlinear_terms(self) -> None:
        pass

    def finish_time_step(self) -> None:
        block_size = 512
        unpadded_kernels_lattice_size = (self.lattice_size // block_size) + 1

        self.linear_kernel((unpadded_kernels_lattice_size,),
                           (block_size,),
                           (self.fields[self.previous_field_marker],
                            self.fields[self.current_field_marker],
                            self.current_dt))

        super().finish_time_step()
