"""Abstract base class for a system that can be solved by FourierSolver.
"""

from abc import abstractmethod
import numpy as np
import cupy as cp
from flucs.systems import FlucsSystem
from flucs.utilities.smooth_numbers import next_smooth_number
from flucs.utilities.cupy import cupy_set_device_pointer


class FourierSystem(FlucsSystem):
    """A generic system of equations solved using pseudospectral Fourier
    methods."""

    # Number of fields that the solver is solving for
    number_of_fields: int

    # Number of fields to be DFT'ed for the pseudospectral nonlinearity
    # number_of_dfts: int

    # This will hold all the fields. Should be a list of CuPy arrays.
    # It's a list in order to store fields at previous time steps, as required
    # by the algorithm.
    fields: list

    # Fourier-space pieces out of which we construct the nonlinear term at each
    # time step
    dft_bits: cp.ndarray

    # Linear matrix (used for linear postprocessing)
    linear_matrix: cp.ndarray

    # Iteration matrices (used for precomputing)
    R: cp.ndarray
    invL: cp.ndarray

    # CFL coefficient at the current time step
    current_cfl: cp.ndarray

    # CUDA kernels
    precompute_iteration_matrices_kernel: cp.RawKernel
    finish_step_kernel: cp.RawKernel
    compute_linear_matrix_kernel: cp.RawKernel
    cuda_block_size: int = 512

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

    half_unpadded_size: int
    half_unpadded_tuple: tuple
    half_padded_size: int
    half_padded_tuple: tuple
    full_unpadded_size: int
    full_unpadded_tuple: tuple
    full_padded_size: int
    full_padded_tuple: tuple

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
                    half_padded_n = padded_n // 2 + 1

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
        self.half_unpadded_size = self.nz * self.nx * self.half_ny
        self.half_unpadded_tuple = (self.nz, self.nx, self.half_ny)

        self.half_padded_size\
            = self.padded_nz * self.padded_nx * self.half_padded_ny
        self.half_padded_tuple\
            = (self.padded_nz, self.padded_nx, self.half_padded_ny)

        self.full_unpadded_size = self.nz * self.nx * self.ny
        self.full_unpadded_tuple = (self.nz, self.nx, self.ny)
        self.full_padded_size\
            = self.padded_nz * self.padded_nx * self.padded_ny
        self.full_padded_tuple\
            = (self.padded_nz, self.padded_nx, self.padded_ny)

        # Finally, precompute wavenumbers (useful for many things)
        self._precompute_wavenumbers()

    def _setup_system(self) -> None:
        """Sets up the system for running the solver. Should be called *after*
        any child class has done its setup, i.e., do not forget to do
        super().setup() in anything that inherits FourierSystem.

        """
        self.set_initial_conditions()

    def _precompute_wavenumbers(self):
        self.kx = 2 * np.pi * self.nx * np.fft.fftfreq(self.nx)\
            / self.input["dimensions.Lx"]

        self.kz = 2 * np.pi * self.nz * np.fft.fftfreq(self.nz)\
            / self.input["dimensions.Lz"]

        # ny is special
        self.ky = 2 * np.pi * self.ny * np.fft.rfftfreq(self.ny)\
            / self.input["dimensions.Ly"]

        # self.kx = np.broadcast_to(kx_linear, (self.nz, self.half_ny,
        #                                       self.nx)).transpose(0, 2, 1)
        # self.kz = np.broadcast_to(kz_linear, (self.half_ny, self.nx,
        #                                       self.nz)).transpose(2, 1, 0)
        #
        # self.ky = np.broadcast_to(ky_linear, (self.nz, self.nx, self.half_ny))

    def get_broadcast_wavenumbers(self):
        """ Returns wavenumber arrays broadcast to (nz, nx, half_ny)

        Returns
        -------
        kx_broadcast, ky_broadcast, kz_broadcast
            Wavenumber arrays of shape (nz, nx, half_ny)

        """
        kx_broadcast = np.broadcast_to(self.kx, (self.nz, self.half_ny,
                                                 self.nx)).transpose(0, 2, 1)

        ky_broadcast = np.broadcast_to(self.ky, (self.nz, self.nx,
                                                 self.half_ny))

        kz_broadcast = np.broadcast_to(self.kz, (self.half_ny, self.nx,
                                                 self.nz)).transpose(2, 1, 0)

        return kx_broadcast, ky_broadcast, kz_broadcast

    @abstractmethod
    def compute_complex_omega(self):
        """Returns an array of shape (nz, nx, half_ny, number_of_fields) with
        the solutions to the linear dispersion relation. This should be
        calculated using only CPU resources.

        """

    def ready(self) -> None:
        # Basic setup
        self.current_step = self.int(0)
        self.current_time = self.init_time
        self.current_dt = self.init_dt

        # Print message.
        print(f"Starting at time {float(self.current_time):.3e} with timestep {float(self.current_dt):.3e}.")

        # Copy initial condition
        self.fields[0][:]\
            = cp.array(np.reshape(self.fields_initial, self.fields[0].shape))

        super().ready()

        if self.input["setup.precompute_linear_matrix"]:
            if not hasattr(self, "R"):  # allocate matrices if not done yet
                self.R = cp.zeros((2, 2, self.nz, self.nx, self.half_ny),
                                  dtype=self.complex)
                self.invL = cp.zeros((2, 2, self.nz, self.nx, self.half_ny),
                                     dtype=self.complex)

            cupy_set_device_pointer(self.cupy_module,
                                    "invL_precomp", self.invL)
            cupy_set_device_pointer(self.cupy_module,
                                    "R_precomp", self.R)

            self.precompute_iteration_matrices_kernel =\
                self.cupy_module.get_function("precompute_iteration_matrices")

            self.precompute_iteration_matrices()

    def precompute_iteration_matrices(self):
        """Precomputes the linear matrix."""
        if not self.input["setup.precompute_linear_matrix"]:
            return

        unpadded_kernels_lattice_size = (self.half_unpadded_size // self.cuda_block_size) + 1
        self.precompute_iteration_matrices_kernel(
            (unpadded_kernels_lattice_size,), (self.cuda_block_size,),
            (self.float(self.current_dt),))

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
                                            self.half_unpadded_size)
        self.module_options.define_constant("HALFPADDEDSIZE",
                                            self.half_padded_size)
        self.module_options.define_constant("PADDEDSIZE",
                                            self.full_padded_size)

        self.module_options.define_constant("DFT_PADDEDSIZE_FACTOR",
                                            self.float(1.0 / self.full_padded_size))

        self.module_options.define_constant("NX", self.nx)
        self.module_options.define_constant("LX", self.input["dimensions.Lx"])
        self.module_options.define_constant("HALF_NX", self.half_nx)
        self.module_options.define_constant("PADDED_NX",
                                            self.padded_nx)
        self.module_options.define_constant("HALF_PADDED_NX",
                                            self.half_padded_nx)

        self.module_options.define_constant("NY", self.ny)
        self.module_options.define_constant("LY", self.input["dimensions.Ly"])
        self.module_options.define_constant("HALF_NY", self.half_ny)
        self.module_options.define_constant("PADDED_NY",
                                            self.padded_ny)
        self.module_options.define_constant("HALF_PADDED_NY",
                                            self.half_padded_ny)

        self.module_options.define_constant("NZ", self.nz)
        self.module_options.define_constant("LZ", self.input["dimensions.Lz"])
        self.module_options.define_constant("HALF_NZ", self.half_nz)
        self.module_options.define_constant("PADDED_NZ",
                                            self.padded_nz)
        self.module_options.define_constant("HALF_PADDED_NZ",
                                            self.half_padded_nz)

        self.module_options.define_constant("ALPHA", self.input["setup.alpha"])

        if not self.input["setup.linear"]:
            self.module_options.define_constant("NONLINEAR")

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

        unpadded_kernels_lattice_size = (self.half_unpadded_size // self.cuda_block_size) + 1
        compute_linear_matrix_kernel(
            (unpadded_kernels_lattice_size,), (self.cuda_block_size,),
            (self.current_dt, self.linear_matrix))

    def set_initial_conditions(self) -> None:
        """Generic setup for the first time step."""

        # Use restart data if it was read
        if self.restart_manager.data is not None:
            restart_data = self.restart_manager.data

            if "fields" not in restart_data:
                raise ValueError("Restart data does not contain 'fields'.")

            field_data = restart_data["fields"]["data"]

            # TODO: remove when allowing for changing of sizes
            expected_shape = self.fields[0].shape
            if field_data.shape != expected_shape:
                raise ValueError(f"Restart data has incorrect shape: {field_data.shape}, expected: {expected_shape}")

            # Set initial field data
            self.fields_initial = np.asarray(field_data)

            return

        # Handle known initialisation types
        match self.input["init.type"]:

            case "white_noise":
                np.random.seed(self.input["init.rand_seed"])
                self.fields_initial =\
                    self.input["init.amplitude"]\
                    * np.random.random(self.number_of_fields
                                       * self.half_unpadded_size)

            case _:
                # Exotic initialisation types should be handled by each solver
                # separately.
                pass

    def get_restart_data(self) -> dict[str, np.ndarray]:
        """
        Get the complex Fourier data for the fields at the current step.
        """

        index = int(self.current_step)%self.number_of_fields
        current_fields = self.fields[index]

        data = cp.asnumpy(current_fields) if isinstance(current_fields, cp.ndarray) \
            else np.asarray(current_fields)

        return {
            "fields": {
                "data": data,
                "dimension_names": ("number_of_fields", "nz", "nx", "half_ny"),
            }
        }

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
        unpadded_kernels_lattice_size = (self.half_unpadded_size // self.cuda_block_size) + 1

        self.finish_step_kernel((unpadded_kernels_lattice_size,),
                           (self.cuda_block_size,),
                           (self.float(self.current_dt),
                            self.current_step,
                            self.fields[self.current_step%self.number_of_fields - 1],
                            self.dft_bits,
                            self.fields[self.current_step%self.number_of_fields]))

        self.current_time += self.current_dt
