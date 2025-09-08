"""Pseudospectral Fourier implementation of the Ivanov et al. (2020) 2D fluid
ITG system. The nonlinear term is handled explicitly using the Adams-Bashforth
3-step method.

"""

from importlib.resources import files
import numpy as np
import cupy as cp
from netCDF4 import Dataset
from numpy import dtype
import flucs
from flucs.solvers.fourier import FourierSystem
from flucs.utilities.cupy import cupy_set_device_pointer
from .cold_itg_2d_fourier_output import HeatfluxDiag
from flucs.diagnostics.output import FlucsOutput


class ColdITG2DFourier(FourierSystem):
    """Fourier solver for the 2D system."""
    number_of_fields = 2
    # GPU memory

    # Direct pointers to the phi and T arrays
    phi: list
    T: list

    # Nonlinear terms at this and at the previous time step
    nonlinear_terms: list

    # Markers for the lists of arrays
    current_field_marker = 0
    previous_field_marker = -1

    # CPU memory


    # CUDA kernels
    linear_kernel: cp.RawKernel # pyright: ignore[reportPossiblyUnboundVariable]

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



    def set_initial_conditions(self):
        super().set_initial_conditions()

        # Add all custom stuff here


    # def setup_kernels(self):
        # resource_path = files(self.__module__) / "cold_itg_2d_fourier.cu"
        # with open(resource_path) as f:
        #     cuda_module = f.read()


        # Now, to set up all the definitions
        # options=("--ptxas-options=-v",
        #          "--use_fast_math",
        #          f"-DTWOPI_OVER_LX=(FLUCS_FLOAT)({2*np.pi/self.input["dimensions.Lx"]})",
        #          f"-DTWOPI_OVER_LY=(FLUCS_FLOAT)({2*np.pi/self.input["dimensions.Ly"]})",
        #          f"-DHALFUNPADDEDSIZE={self.lattice_size}",
        #          f"-DNX={self.nx}",
        #          f"-DHALF_NX={self.half_nx}",
        #          f"-DHALF_NY={self.half_ny}",
        #          f"-DCHI=(FLUCS_FLOAT)({self.input["parameters.chi"]})",
        #          f"-DA_TIMES_CHI=(FLUCS_FLOAT)({self.input["parameters.a"] * self.input["parameters.chi"]})",
        #          f"-DB_TIMES_CHI=(FLUCS_FLOAT)({self.input["parameters.b"] * self.input["parameters.chi"]})",
        #          f"-DKAPPA_T=(FLUCS_FLOAT)({self.input["parameters.kappaT"]})",
        #          f"-DKAPPA_N=(FLUCS_FLOAT)({self.input["parameters.kappaN"]})",
        #          f"-DKAPPA_B=(FLUCS_FLOAT)({self.input["parameters.kappaB"]})",
        #          f"-DALPHA=(FLUCS_FLOAT)({self.input["parameters.alpha"]})",
        #          "-DPRECOMPUTE_LINEAR_MATRIX",
        #          f"-I{files(flucs).parent}")




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


        super().ready()

        cupy_set_device_pointer(self.cupy_module, "invL_precomp", self.invL)
        cupy_set_device_pointer(self.cupy_module, "R_precomp", self.R)

        self.linear_kernel = self.cupy_module.get_function("linear_kernel")

        self.fields[self.current_field_marker][:] = cp.array(np.reshape(self.fields_initial, self.fields[0].shape))

        self.init_output()

    def init_output(self) -> None:
        scalar_output = FlucsOutput(name="0d", system=self)
        scalar_output.ready()
        self.add_output(scalar_output)


    def calculate_nonlinear_terms(self) -> None:
        self.current_field_marker = (self.current_field_marker + 1) % 2
        self.previous_field_marker = self.current_field_marker - 1
        self.current_step += 1

    def finish_time_step(self) -> None:
        block_size = 512
        unpadded_kernels_lattice_size = (self.lattice_size // block_size) + 1

        self.linear_kernel((unpadded_kernels_lattice_size,),
                           (block_size,),
                           (self.fields[self.previous_field_marker],
                            self.fields[self.current_field_marker],
                            self.current_dt))

        self.current_time += self.current_dt

