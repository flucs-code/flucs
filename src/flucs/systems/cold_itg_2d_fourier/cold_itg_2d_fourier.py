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
from flucs.solvers.fourier.fourier_system_diagnostics import LinearSpectrumDiag
from flucs.output import FlucsOutput
from .cold_itg_2d_fourier_diagnostics import HeatfluxDiag


class ColdITG2DFourier(FourierSystem):
    """Fourier solver for the 2D system."""
    number_of_fields = 2

    # Direct pointers to the phi and T arrays
    phi: list
    T: list

    # Nonlinear terms
    nonlinear_terms: list
    current_nonlinear_marker: int = 0

    # Supported diagnostics
    diags_dict = {"heatflux": HeatfluxDiag,
                  "linear_spectrum": LinearSpectrumDiag}

    def setup(self):
        """Prepares the system for the solver."""

        self.allocate_memory()
        # self.setup_kernels()
        super().setup()

    def ready(self):
        # Anything system-specific goes here
        super().ready()

    def allocate_memory(self):
        # GPU arrays

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


    def _interpret_input(self):
        """Checks if the input file makes sense"""

        # Make sure to call the parent method to do some standard setup
        # (resolution checks, etc)
        super()._interpret_input()

        # Anything custom goes here

        if self.nz != 1 or self.padded_nz != 1:
            raise ValueError("Both nz and padded_nz should be set to 1 for the 2D system!")


    def compile_cupy_module(self) -> None:
        self.module_options.define_constant("CHI", self.input["parameters.chi"])
        self.module_options.define_constant("A_TIMES_CHI",
                                            self.input["parameters.a"]
                                            * self.input["parameters.chi"])

        self.module_options.define_constant("B_TIMES_CHI",
                                            self.input["parameters.b"]
                                            * self.input["parameters.chi"])

        self.module_options.define_constant("KAPPA_T", self.input["parameters.kappaT"])
        self.module_options.define_constant("KAPPA_N", self.input["parameters.kappan"])
        self.module_options.define_constant("KAPPA_B", self.input["parameters.kappaB"])

        super().compile_cupy_module() # Call this to compile the module

    def begin_time_step(self) -> None:
        # Do anything model-specific here (e.g., advance markers for the
        # nonlinear terms), then call the parent's method
        pass
        super().begin_time_step()


    def calculate_nonlinear_terms(self) -> None:
        pass

    def finish_time_step(self) -> None:
        block_size = 512
        unpadded_kernels_lattice_size = (self.lattice_size // block_size) + 1

        self.finish_step_kernel((unpadded_kernels_lattice_size,),
                           (block_size,),
                           (self.fields[self.current_field_marker - 1],
                            self.fields[self.current_field_marker],
                            self.current_dt))

        super().finish_time_step()

    def compute_complex_omega(self):
        linear_matrix = np.zeros(self.lattice_tuple + (2,2), dtype=self.complex)

        kxs, kys, kzs = self.get_broadcast_wavenumbers()
        kperp2 = kxs**2 + kys**2

        kappaT = self.input["parameters.kappaT"]
        kappaB = self.input["parameters.kappaB"]
        kappan = self.input["parameters.kappan"]
        chi = self.input["parameters.chi"]
        a = self.input["parameters.a"]
        b = self.input["parameters.b"]

        eta = 1 + kperp2
        # zonal response
        eta[0, :, 0] = kperp2[0, :, 0]

        # phi-phi
        linear_matrix[:, :, :, 0, 0] = (
                    a*chi*(kperp2**2)
                    - 1j*(kappaB - kappan)*kys
                    - 1j*kappaT*kperp2*kys) / eta

        # phi-T
        linear_matrix[:, :, :, 0, 1] = (
                    - b*chi*(kperp2**2)
                    - 1j*kappaB*kys) / eta

        # T-phi
        linear_matrix[:, :, :, 1, 0] = 1j*kappaT*kys

        # T-T
        linear_matrix[:, :, :, 1, 1] = chi*kperp2

        # Fix (0,0,0) mode
        linear_matrix[0, 0, 0, :, :] = np.identity(2)

        return -1j*np.linalg.eigvals(linear_matrix)
