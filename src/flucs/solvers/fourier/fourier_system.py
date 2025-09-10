"""Abstract base class for a system that can be solved by FourierSolver.
"""

from abc import abstractmethod
import numpy as np
import cupy as cp
from flucs.systems import FlucsSystem
from flucs import FlucsInput
from flucs.utilities.smooth_numbers import next_smooth_number
from flucs.utilities.cupy import cupy_set_device_pointer


class FourierSystem(FlucsSystem):
    """A generic system of equations solved using pseudospectral Fourier
    methods."""

    # Number of fields that the solver is solving for
    number_of_fields: int

    # Number of fields to be DFT'ed
    number_of_dfts: int

    # This will hold all the fields. Should be a list of CuPy arrays.
    # It's a list in order to store fields at previous time steps, as required
    # by the algorithm.
    fields: list

    # Linear matrix (used for linear postprocessing)
    linear_matrix: cp.ndarray

    # Iteration matrices (used for precomputing)
    R: cp.ndarray
    invL: cp.ndarray

    # CUDA kernels
    finish_step_kernel: cp.RawKernel
    compute_linear_matrix_kernel: cp.RawKernel

    # Initial conditions, always in CPU memory
    fields_initial: np.ndarray

    # Array sizes
    nx: int
    ny: int
    nz: int
    half_nx: int
    half_ny: int
    half_nz: int

    padded_nx: int
    padded_ny: int
    padded_nz: int
    half_padded_nx: int
    half_padded_ny: int
    half_padded_nz: int

    lattice_size: int
    padded_lattice_size: int
    real_lattice_size: int
    real_padded_lattice_size: int

    # Fourier wavenumbers
    kx: np.ndarray
    ky: np.ndarray
    kz: np.ndarray

    def _interpret_input(self):
        """Validates and sets up the number of lattice points."""

        # Set resolutions appropriately
        for dim in ["x", "y", "z"]:
            n = self.input[f"dimensions.n{dim}"]
            padded_n = self.input[f"dimensions.padded_n{dim}"]

            match (n > 0, padded_n > 0):
                case (True, True):
                    # Check if n is odd
                    if n % 2 == 0:
                        raise ValueError(
                            "Unpadded resolutions must be odd! "
                            f"Please change n{dim} = {n} to an odd number!")

                    half_n = n//2 + 1
                    half_padded_n = padded_n//2 + 1
                    # TODO: add some check that warns the user if their choice
                    # is dumb

                case (True, False):
                    # Check if n is odd
                    if n % 2 == 0:
                        raise ValueError(
                            "Unpadded resolutions must be odd! "
                            f"Please change n{dim} = {n} to an odd number!")

                    half_n = n//2 + 1

                    # Find minimum padded that works
                    padded_n = next_smooth_number(
                        (self.input["dimensions.nonlinear_order"] + 1)*half_n,
                        primes=self.input["dimensions.padded_primes"])

                    half_padded_n = padded_n//2 + 1

                    print(f"Found padded_n{dim} = {padded_n} "
                          "for n{dim} = {n}.")

                case (False, True):
                    # Given a padded_n, it's easiest to figure out half_n

                    factor = self.input["dimensions.nonlinear_order"] + 1
                    _x = padded_n // factor
                    half_padded_n = padded_n // 2

                    # Handle an annoying edge case
                    if padded_n % factor == 0:
                        _x -= 1

                    half_n = _x + 1
                    n = 2*_x + 1

                    print(f"Found n{dim} = {n} for "
                          f"padded_n{dim} = {padded_n}.")

                case (False, False):
                    raise ValueError(f"At least one of n{dim} and "
                                     f"padded_n{dim} must be positive!")

                # This is added only to make pyright happy.
                case _:
                    raise RuntimeError("How the fluc did you get here?")

            # It's useful to have the resolutions as part of the system
            # rather than to access the input dictionary every time
            setattr(self, f"n{dim}", n)
            setattr(self, f"padded_n{dim}", padded_n)
            setattr(self, f"half_n{dim}", half_n)
            setattr(self, f"half_padded_n{dim}", half_padded_n)

        # Set padded and unpadded array sizes
        self.lattice_size = self.nz * self.nx * self.half_ny
        self.padded_lattice_size\
            = self.padded_nz * self.padded_nx * self.half_padded_ny

        self.real_lattice_size = self.nz * self.nx * self.ny
        self.real_padded_lattice_size\
            = self.padded_nz * self.padded_nx * self.padded_ny

        # Finally, precompute wavenumbers (useful for many things)
        self._precompute_wavenumbers()

    def setup(self) -> None:
        """Sets up the system for running the solver. Should be called *after*
        any child class has done its setup, i.e., do not forget to do
        super().setup() in anything that inherits FourierSystem.

        """
        self.set_initial_conditions()

        super().setup()

    def _precompute_wavenumbers(self):
        kx_linear = 2 * np.pi * self.nx * np.fft.fftfreq(self.nx)\
            / self.input["dimensions.Lx"]

        kz_linear = 2 * np.pi * self.nz * np.fft.fftfreq(self.nz)\
            / self.input["dimensions.Lz"]

        # ny is special
        ky_linear = 2 * np.pi * self.ny * np.fft.rfftfreq(self.ny)\
            / self.input["dimensions.Ly"]

        self.kx = np.broadcast_to(kx_linear, (self.nz, self.half_ny,
                                              self.nx)).transpose(0, 2, 1)
        self.kz = np.broadcast_to(kz_linear, (self.half_ny, self.nx,
                                              self.nz)).transpose(2, 1, 0)

        self.ky = np.broadcast_to(ky_linear, (self.nz, self.nx, self.half_ny))


    def ready(self) -> None:
        # Basic setup
        self.current_step = self.int(0)
        self.current_time = self.float(0.0)
        self.current_dt = self.float(self.input["time.dt"])

        super().ready()

        if self.input["setup.precompute_linear_matrix"]:
            if not hasattr(self, "R"):  # allocate matrices if not done yet
                self.R = cp.zeros((2, 2, self.nz, self.nx, self.half_ny), dtype=self.complex)
                self.invL = cp.zeros((2, 2, self.nz, self.nx, self.half_ny), dtype=self.complex)

            cupy_set_device_pointer(self.cupy_module, "invL_precomp", self.invL)
            cupy_set_device_pointer(self.cupy_module, "R_precomp", self.R)

            self.precompute_iteration_matrices_kernel = self.cupy_module.get_function("precompute_iteration_matrices")
            self.precompute_iteration_matrices()

    def precompute_iteration_matrices(self):
        """Precomputes the linear matrix."""
        if not self.input["setup.precompute_linear_matrix"]:
            return

        block_size = 512
        unpadded_kernels_lattice_size = (self.lattice_size // block_size) + 1
        self.precompute_iteration_matrices_kernel(
            (unpadded_kernels_lattice_size,), (block_size,),
            (self.current_dt,))

    def compile_cupy_module(self) -> None:
        # Add module options
        self.module_options.define_constant("TWOPI_OVER_LX",
                                            2*np.pi / self.input["dimensions.Lx"])
        self.module_options.define_constant("TWOPI_OVER_LY",
                                            2*np.pi / self.input["dimensions.Ly"])
        self.module_options.define_constant("TWOPI_OVER_LZ",
                                            2*np.pi / self.input["dimensions.Lz"])

        self.module_options.define_constant("NUMBER_OF_FIELDS",
                                            self.number_of_fields)

        self.module_options.define_constant("HALFUNPADDEDSIZE",
                                            self.lattice_size)

        self.module_options.define_constant("NX", self.nx)
        self.module_options.define_constant("HALF_NX", self.half_nx)
        self.module_options.define_constant("NY", self.ny)
        self.module_options.define_constant("HALF_NY", self.half_ny)
        self.module_options.define_constant("NZ", self.nz)
        self.module_options.define_constant("HALF_NZ", self.half_nz)
        self.module_options.define_constant("ALPHA", self.input["setup.alpha"])

        if self.input["setup.precompute_linear_matrix"]:
            print("Will precompute the linear matrix!")
            self.module_options.define_constant("PRECOMPUTE_LINEAR_MATRIX")

        super().compile_cupy_module()

        self.finish_step_kernel = self.cupy_module.get_function("finish_step")

    def compute_linear_matrix(self) -> None:
        """Computes the linear matrix using the CuPy module and stores the
        result in self.linear_matrix"""
        self.linear_matrix = cp.zeros((2, 2, self.nz, self.nx, self.half_ny),
                                      dtype=self.complex)

        compute_linear_matrix_kernel\
            = self.cupy_module.get_function("compute_linear_matrix")

        block_size = 512
        unpadded_kernels_lattice_size = (self.lattice_size // block_size) + 1
        compute_linear_matrix_kernel(
            (unpadded_kernels_lattice_size,), (block_size,),
            (self.current_dt, self.linear_matrix))

    def set_initial_conditions(self) -> None:
        """Generic setup for the first time step."""

        # Handle known initialisation types
        match self.input["init.type"]:

            case "white_noise":
                np.random.seed(self.input["init.rand_seed"])
                self.fields_initial =\
                    self.input["init.amplitude"]\
                    * np.random.random(self.number_of_fields
                                       * self.lattice_size)

            case _:
                # Exotic initialisation types should be handled by each solver
                # separately.
                pass

    @abstractmethod
    def begin_time_step(self) -> None:
        """Executed in the beginning of the time step. Should be used to
        advance any system-specific counters.

        """
        self.current_step += 1

    @abstractmethod
    def calculate_nonlinear_terms(self) -> None:
        """Computes the nonlinear terms and adjusts the time step if
        necessary.

        Called in the beginning of a time step.

        """
        pass

    @abstractmethod
    def finish_time_step(self) -> None:
        """Combines the nonlinear and linear terms in order to finish the time
        step"""
        self.current_time += self.current_dt
