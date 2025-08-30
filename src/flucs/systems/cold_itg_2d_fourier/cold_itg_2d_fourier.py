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
    number_of_fields = 2
    # GPU memory

    # Direct pointers to the phi and T arrays
    phi: list
    T: list

    # Nonlinear terms at this and at the previous time step
    nonlinear_terms: list = None

    # CPU memory


    # CUDA kernels
    linear_kernel: cp.RawKernel


    def setup(self):
        """Prepares the system for the solver."""
        super().setup()

        self.allocate_memory()
        self.setup_kernels()


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



    def _set_initial_conditions(self):
        super()._set_initial_conditions()

        # Add all custom stuff here


    def setup_kernels(self):
        resource_path = files(self.__module__) / "cold_itg_2d_fourier.cu"
        with open(resource_path) as f:
            cuda_module = f.read()


        # Now, to set up all the definitions
        options=("--ptxas-options=-O3",
                 # "-O3",
                 "--use_fast_math",
                 f"-DTWOPI_OVER_LX=(FLUCS_FLOAT)({2*np.pi/self.input["dimensions.Lx"]})",
                 f"-DTWOPI_OVER_LY=(FLUCS_FLOAT)({2*np.pi/self.input["dimensions.Ly"]})",
                 f"-DHALFUNPADDEDSIZE={self.grid_size}",
                 f"-DNX={self.nx}",
                 f"-DHALF_NX={self.half_nx}",
                 f"-DHALF_NY={self.half_ny}",
                 f"-DCHI=(FLUCS_FLOAT)({self.input["parameters.chi"]})",
                 f"-DA_TIMES_CHI=(FLUCS_FLOAT)({self.input["parameters.a"] * self.input["parameters.chi"]})",
                 f"-DB_TIMES_CHI=(FLUCS_FLOAT)({self.input["parameters.b"] * self.input["parameters.chi"]})",
                 f"-DKAPPA_T=(FLUCS_FLOAT)({self.input["parameters.kappaT"]})",
                 f"-DKAPPA_N=(FLUCS_FLOAT)({self.input["parameters.kappaN"]})",
                 f"-DKAPPA_B=(FLUCS_FLOAT)({self.input["parameters.kappaB"]})",
                 f"-DALPHA=(FLUCS_FLOAT)({self.input["parameters.alpha"]})")

        cupy_module = cp.RawModule(code=cuda_module, options=options)
        cupy_module.compile()

        self.linear_kernel = cupy_module.get_function("linear_kernel")

        print(self.linear_kernel)
