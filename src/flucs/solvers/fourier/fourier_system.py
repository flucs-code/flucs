"""
Abstract base class for a system that can be solved by FourierSolver.
"""

from abc import abstractmethod

import cupy as cp
import numpy as np

from flucs.input import InvalidFlucsInputFileError
from flucs.systems import FlucsSystem
from flucs.utilities.cupy import cupy_set_device_pointer
from flucs.utilities.smooth_numbers import next_smooth_number


class FourierSystem(FlucsSystem):
    """A generic system of equations solved using pseudospectral Fourier
    methods."""

    # Number of fields that the solver is solving for
    number_of_fields: int

    # Total number of time steps for which we hold field data in memory
    # This is typically 2 (previous time step +
    # the current one we are solving for)
    fields_history_size = 2

    # This will hold all the fields. Should be a list of CuPy arrays.
    # It's a list in order to store fields at previous time steps, as required
    # by the algorithm.
    fields: list

    # Real-space fields in CPU memory, used for diagnostics
    realspace_fields: np.ndarray | None = None

    # Fourier-space pieces out of which we construct the nonlinear term at each
    # time step
    dft_bits: cp.ndarray

    # Linear matrix (used for linear postprocessing)
    linear_matrix: cp.ndarray

    # Iteration matrices (used for precomputing)
    rhs: cp.ndarray
    inverse_lhs: cp.ndarray

    # CFL condition variables
    max_cfl: float
    current_cfl: float
    cfl_rate: cp.ndarray
    cfl_rate_float: float
    sub_cfl_steps: float

    # Timestep variables
    dt_max: float
    dt_min: float
    dt_mult_increase: float
    dt_mult_decrease: float
    dt_mult_steps: float
    dt_array: np.ndarray
    ab3_coefficients: np.ndarray

    # CUDA kernels
    precompute_iteration_matrices_kernel: cp.RawKernel
    finish_step_kernel: cp.RawKernel
    compute_linear_matrix_kernel: cp.RawKernel
    cuda_block_size: int = 512

    # CUDA grids
    half_unpadded_cuda_grid_size: int
    half_padded_cuda_grid_size: int
    full_padded_cuda_grid_size: int

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

        # Check for conflicts in time-stepping input parameters
        if self.input["time.dt_method"] == "discrete":
            print("Using discrete time stepping.")

        elif self.input["time.dt_method"] == "continuous":
            if self.input["setup.precompute_linear_matrix"]:
                raise InvalidFlucsInputFileError(
                    "Cannot have setup.precompute_linear_matrix = true if "
                    "time.dt_method = 'continuous'."
                )
            print("Using continuous time stepping.")

        # Check for conflicts in hyperdissipation parameters
        if self.input["hyperdissipation.perp"] > 0.0 and (
            self.input["hyperdissipation.kx"] > 0.0
            or self.input["hyperdissipation.ky"] > 0.0
        ):
            raise InvalidFlucsInputFileError(
                "Cannot enable both hyperdissipation.perp "
                "and hyperdissipation.kx/ky simultaneously. "
                "Use either perp or kx/ky. "
            )

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
                            f"Please change n{dim} = {n} to an odd number!"
                        )

                    half_n = n // 2 + 1
                    half_padded_n = padded_n // 2 + 1
                    # TODO: add some check that warns the user if their choice
                    # is dumb

                case (True, False):
                    # Check if n is odd
                    if n % 2 == 0:
                        raise ValueError(
                            "Unpadded resolutions must be odd! "
                            f"Please change n{dim} = {n} to an odd number!"
                        )

                    half_n = n // 2 + 1

                    # Find minimum padded that works
                    padded_n = next_smooth_number(
                        (self.input["dimensions.nonlinear_order"] + 1) * half_n,
                        primes=self.input["dimensions.padded_primes"],
                    )

                    half_padded_n = padded_n // 2 + 1

                    print(f"Found padded_n{dim} = {padded_n} for n{dim} = {n}")

                case (False, True):
                    # Given a padded_n, it's easiest to figure out half_n

                    factor = self.input["dimensions.nonlinear_order"] + 1
                    _x = padded_n // factor
                    half_padded_n = padded_n // 2 + 1

                    # Handle an annoying edge case
                    if padded_n % factor == 0:
                        _x -= 1

                    half_n = _x + 1
                    n = 2 * _x + 1

                    print(f"Found n{dim} = {n} for padded_n{dim} = {padded_n}")

                case (False, False):
                    raise ValueError(
                        f"At least one of n{dim} and "
                        f"padded_n{dim} must be positive!"
                    )

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

        self.half_padded_size = (
            self.padded_nz * self.padded_nx * self.half_padded_ny
        )
        self.half_padded_tuple = (
            self.padded_nz,
            self.padded_nx,
            self.half_padded_ny,
        )

        self.full_unpadded_size = self.nz * self.nx * self.ny
        self.full_unpadded_tuple = (self.nz, self.nx, self.ny)
        self.full_padded_size = self.padded_nz * self.padded_nx * self.padded_ny
        self.full_padded_tuple = (
            self.padded_nz,
            self.padded_nx,
            self.padded_ny,
        )

        # Finally, precompute wavenumbers (useful for many things)
        self._precompute_wavenumbers()

    def _setup_system(self) -> None:
        """Sets up the system for running the solver. Should be called *after*
        any child class has done its setup, i.e., do not forget to do
        super()._setup_system() in anything that inherits FourierSystem.

        """
        self._set_initial_conditions()
        self._check_initial_conditions()

    def _precompute_wavenumbers(self):
        # Check if we have already done this
        if hasattr(self, "ky"):
            return

        self.kx = (
            2
            * np.pi
            * self.nx
            * np.fft.fftfreq(self.nx)
            / self.input["dimensions.Lx"]
        )

        self.kz = (
            2
            * np.pi
            * self.nz
            * np.fft.fftfreq(self.nz)
            / self.input["dimensions.Lz"]
        )

        # ny is special
        self.ky = (
            2
            * np.pi
            * self.ny
            * np.fft.rfftfreq(self.ny)
            / self.input["dimensions.Ly"]
        )

    def get_broadcast_wavenumbers(self):
        """Returns wavenumber arrays broadcast to (nz, nx, half_ny)

        Returns
        -------
        kx_broadcast, ky_broadcast, kz_broadcast
            Wavenumber arrays of shape (nz, nx, half_ny)

        """
        kx_broadcast = np.broadcast_to(
            self.kx, (self.nz, self.half_ny, self.nx)
        ).transpose(0, 2, 1)

        ky_broadcast = np.broadcast_to(
            self.ky, (self.nz, self.nx, self.half_ny)
        )

        kz_broadcast = np.broadcast_to(
            self.kz, (self.half_ny, self.nx, self.nz)
        ).transpose(2, 1, 0)

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

        self.dt_max = self.input["time.dt_max"]
        self.dt_min = self.input["time.dt_min"]

        # Setup kernel parameters (grid, block, shared memory)
        self.half_unpadded_cuda_grid_size = (
            self.half_unpadded_size + self.cuda_block_size - 1
        ) // self.cuda_block_size

        self.half_padded_cuda_grid_size = (
            self.half_padded_size + self.cuda_block_size - 1
        ) // self.cuda_block_size

        self.full_padded_cuda_grid_size = (
            self.full_padded_size + self.cuda_block_size - 1
        ) // self.cuda_block_size

        # CFL setup
        self.current_cfl = 0.0
        self.max_cfl = self.input["time.max_cfl"]

        # Timestep setup
        self.dt_mult_increase = self.input["time.dt_mult_increase"]
        self.dt_mult_decrease = self.input["time.dt_mult_decrease"]
        self.dt_array = np.array(
            [self.current_dt, 10**10, 10**10], dtype=self.float
        )
        self.ab3_coefficients = np.array([1, 0, 0], dtype=self.float)

        # Determine the time stepping method
        if self.input["time.dt_method"] == "discrete":
            self.sub_cfl_steps = self.int(0)
            self.dt_mult_steps = self.input["time.dt_mult_steps"]
            self._compute_current_dt = self._compute_current_dt_discrete

        elif self.input["time.dt_method"] == "continuous":
            self._compute_current_dt = self._compute_current_dt_continuous

        # Print message.
        print(
            f"Starting at time {float(self.current_time):.3e}, "
            f"dt {float(self.current_dt):.3e}"
        )

        # Copy initial condition
        self.fields[0][:] = cp.array(
            np.reshape(self.fields_initial, self.fields[0].shape)
        )

        super().ready()

        # Allocate precomputation matrices
        if self.input["setup.precompute_linear_matrix"]:
            if not hasattr(self, "rhs"):
                self.rhs = cp.zeros(
                    (
                        self.number_of_fields,
                        self.number_of_fields,
                        self.nz,
                        self.nx,
                        self.half_ny,
                    ),
                    dtype=self.complex,
                )
                self.inverse_lhs = cp.zeros(
                    (
                        self.number_of_fields,
                        self.number_of_fields,
                        self.nz,
                        self.nx,
                        self.half_ny,
                    ),
                    dtype=self.complex,
                )

            cupy_set_device_pointer(
                self.cupy_module, "inverse_lhs_precomp", self.inverse_lhs
            )

            cupy_set_device_pointer(self.cupy_module, "rhs_precomp", self.rhs)

            self.precompute_iteration_matrices_kernel = (
                self.cupy_module.get_function("precompute_iteration_matrices")
            )

            self.precompute_iteration_matrices()

    def precompute_iteration_matrices(self):
        """Precomputes the linear matrix."""
        if not self.input["setup.precompute_linear_matrix"]:
            return

        self.precompute_iteration_matrices_kernel(
            (self.half_unpadded_cuda_grid_size,),
            (self.cuda_block_size,),
            (self.float(self.current_dt),),
        )

    def compile_cupy_module(self) -> None:
        # FourierSystem specific constants
        self.module_options.define_constant(
            "NUMBER_OF_FIELDS", self.number_of_fields
        )

        self.module_options.define_constant(
            "HALFUNPADDEDSIZE", self.half_unpadded_size
        )
        self.module_options.define_constant(
            "HALFPADDEDSIZE", self.half_padded_size
        )
        self.module_options.define_constant("PADDEDSIZE", self.full_padded_size)

        self.module_options.define_constant(
            "DFT_PADDEDSIZE_FACTOR", self.float(1.0 / self.full_padded_size)
        )

        # Dimensions
        for dim in ["x", "y", "z"]:
            box_size = self.float(self.input[f"dimensions.L{dim}"])

            self.module_options.define_constant(
                f"TWOPI_OVER_L{dim.upper()}", 2 * np.pi / box_size
            )

            self.module_options.define_constant(
                f"N{dim.upper()}", getattr(self, f"n{dim}")
            )
            self.module_options.define_constant(f"L{dim.upper()}", box_size)
            self.module_options.define_constant(
                f"HALF_N{dim.upper()}", getattr(self, f"half_n{dim}")
            )
            self.module_options.define_constant(
                f"PADDED_N{dim.upper()}", getattr(self, f"padded_n{dim}")
            )
            self.module_options.define_constant(
                f"HALF_PADDED_N{dim.upper()}",
                getattr(self, f"half_padded_n{dim}"),
            )

        # Hyperdissipation
        for component in ["perp", "kx", "ky", "kz"]:
            if self.input[f"hyperdissipation.{component}"] > 0.0:
                print(f"Using hyperdissipation in {component}.")

                self.module_options.define_constant(
                    f"HYPERDISSIPATION_{component.upper()}",
                    self.input[f"hyperdissipation.{component}"],
                )
                self.module_options.define_constant(
                    f"HYPERDISSIPATION_{component.upper()}_POWER",
                    self.input[f"hyperdissipation.{component}_power"],
                )

        # Setup
        self.module_options.define_constant("ALPHA", self.input["setup.alpha"])

        if not self.input["setup.linear"]:
            self.module_options.define_constant("NONLINEAR")

        if self.input["setup.precompute_linear_matrix"]:
            print("Linear matrices will be precomputed.")
            self.module_options.define_constant("PRECOMPUTE_LINEAR_MATRIX")

        super().compile_cupy_module()

        self.finish_step_kernel = self.cupy_module.get_function("finish_step")

    def compute_linear_matrix(self) -> None:
        """Computes the linear matrix using the CuPy module and stores the
        result in self.linear_matrix"""
        self.linear_matrix = cp.zeros(
            (
                self.number_of_fields,
                self.number_of_fields,
                self.nz,
                self.nx,
                self.half_ny,
            ),
            dtype=self.complex,
        )

        compute_linear_matrix_kernel = self.cupy_module.get_function(
            "compute_linear_matrix"
        )

        compute_linear_matrix_kernel(
            (self.half_unpadded_cuda_grid_size,),
            (self.cuda_block_size,),
            (self.current_dt, self.linear_matrix),
        )

    def _set_initial_conditions(self) -> None:
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
                raise ValueError(
                    f"Restart data has incorrect shape: "
                    f"{field_data.shape}, "
                    f"expected: {expected_shape}"
                )

            # Set initial field data
            self.fields_initial = np.asarray(field_data)

            return

        # Handle known initialisation types
        match self.input["init.type"]:
            case "white_noise":
                np.random.seed(self.input["init.rand_seed"])
                self.fields_initial = self.input[
                    "init.amplitude"
                ] * np.random.random(
                    (self.number_of_fields, *self.half_unpadded_tuple)
                )

            case _:
                # Exotic initialisation types should be handled by each solver
                # separately.
                pass

    def _check_initial_conditions(self) -> None:
        """
        Ensures that the initial conditions satisfy the reality condition
        field[-ikz, -ikx, 0] = conj(field[ikz, ikx, 0]) for all ikx and ikz.
        """

        fields_initial = self.fields_initial.reshape(
            (self.number_of_fields, self.nz, self.nx, self.half_ny)
        )

        # The ky=0 modes are the ones that need to be checked
        fields_initial_ky0 = fields_initial[:, :, :, 0]

        # To make this easier, shift the frequencies so that they are ordered
        # ..., -2, -1, 0, 1, 2, ...
        fields_initial_ky0 = np.fft.fftshift(fields_initial_ky0, axes=(1, 2))

        # If not restarting, enforce the reality condition
        if self.restart_manager.data is None:
            # Enforce conjugate symmetry
            fields_initial_ky0[:] = 0.5 * (
                fields_initial_ky0 + np.conj(fields_initial_ky0[:, ::-1, ::-1])
            )

            # Shift back to original frequency ordering
            fields_initial[:, :, :, 0] = np.fft.ifftshift(
                fields_initial_ky0, axes=(1, 2)
            )

            # Update the stored initial conditions
            self.fields_initial = fields_initial.reshape(
                self.fields_initial.shape
            )

        # Calculate and report error
        error = np.nanmax(
            np.abs(
                fields_initial_ky0 - np.conj(fields_initial_ky0[:, ::-1, ::-1])
            )
        )
        print(f"Init. condition reality error: {error:.3e}")

    def get_restart_data(self) -> dict[str, np.ndarray]:
        """
        Get the complex Fourier data for the fields at the current step.
        """

        index = int(self.current_step) % self.number_of_fields
        current_fields = self.fields[index]

        data = (
            cp.asnumpy(current_fields)
            if isinstance(current_fields, cp.ndarray)
            else np.asarray(current_fields)
        )

        return {
            "fields": {
                "data": data,
                "dimension_names": ("number_of_fields", "nz", "nx", "half_ny"),
            }
        }

    def _compute_current_dt(self) -> None:
        """
        Computes the current time step based on the CFL condition.
        Will be set to either 'compute_current_dt_discrete' or
        'compute_current_dt_continuous' at runtime depending on the
        value of 'time.dt_method'.
        """

    def _compute_current_dt_continuous(self) -> float:
        """
        Computes the current time step based on the CFL condition.
        'dt_multiplier' should be used to limit the increase in the
        time step at each iteration.

        Used if 'time.dt_method' is "continuous".
        """

        # Compute new dt
        new_dt = self.float(
            min(
                (
                    self.max_cfl / self.cfl_rate_float,
                    self.dt_max,
                    self.current_dt * self.dt_mult_increase,
                )
            ),
        )

        # Assign value
        self.current_dt = new_dt

    def _compute_current_dt_discrete(self) -> float:
        """
        Computes the current time step based on the CFL condition.
        'dt_multiplier' should be used to limit the increase in the
        time step at each iteration.

        Used if 'time.dt_method' is "discrete".
        """

        # If CFL condition is violated
        if self.cfl_rate_float * self.current_dt > self.max_cfl:
            new_dt = self.dt_mult_decrease * self.max_cfl / self.cfl_rate_float
            print(
                f"dt: {self.current_dt:.3e} -> "
                f"{new_dt:.3e} (-, {self.current_step:.3e})"
            )

            self.current_dt = new_dt
            self.sub_cfl_steps = self.int(0)
            self.precompute_iteration_matrices()

        # Check to see whether we can increase dt
        elif self.sub_cfl_steps >= self.dt_mult_steps:
            new_dt = self.float(
                min(
                    self.current_dt * self.dt_mult_increase,
                    self.dt_max,
                    self.max_cfl / self.cfl_rate_float,
                )
            )

            if new_dt > self.current_dt:
                print(
                    f"dt: {self.current_dt:.3e} -> {new_dt:.3e} "
                    f"(+, {self.current_step:.3e})"
                )

                self.current_dt = new_dt
                self.sub_cfl_steps = self.int(0)
                self.precompute_iteration_matrices()

        # Otherwise just continue iterating with same current_dt
        else:
            self.sub_cfl_steps += 1

    def _update_dt(self) -> None:
        """
        Updates the time step based on the CFL condition.
        """

        self.cfl_rate_float = self.float(cp.asnumpy(self.cfl_rate[0]))

        self._compute_current_dt()
        if self.current_dt < self.dt_min:
            print(
                f"({self.current_step}) Required time step "
                f"{self.current_dt:.3e} is below dt_min. Exiting."
            )
            self.solver.interrupted = True

        self.current_cfl = self.cfl_rate_float * self.current_dt
        self.dt_array[self.current_step % 3] = self.current_dt

    def _update_ab3_coefficients(self) -> None:
        """
        Updates nonlinear coefficients given changing timestep.
        """

        # Alias for readability
        dt0 = self.dt_array[self.current_step % 3]
        dt1 = self.dt_array[self.current_step % 3 - 1]
        dt2 = self.dt_array[self.current_step % 3 - 2]

        # Compute coefficients.
        # Disabling formatting and linting for readability.
        # fmt: off
        self.ab3_coefficients[0] = 1 + (dt0 / dt1) * ((2.0 / 6.0) * dt0 +               dt1 + (3.0 / 6.0) * dt2) / (dt1 + dt2) # noqa: E501
        self.ab3_coefficients[1] =   - (dt0 / dt1) * ((2.0 / 6.0) * dt0 + (3.0 / 6.0) * dt1 + (3.0 / 6.0) * dt2) / (      dt2) # noqa: E501
        self.ab3_coefficients[2] =   + (dt0 / dt2) * ((2.0 / 6.0) * dt0 + (3.0 / 6.0) * dt1                    ) / (dt1 + dt2) # noqa: E501
        # fmt: on

    def get_realspace_fields(self):
        """
        Calculates the real-space fields at the current time step as a
        NumPy array. The FFTs are done on the CPU in order to save GPU memory.
        This makes them quite time-consuming so use this sparingly!

        The data is saved in FourierSystem.realspace_fields

        """

        # If not None, then we have already called it this time step
        if self.realspace_fields is not None:
            return

        # TODO: this needs to be changed if there's flow shear
        fields_cpu_memory: np.ndarray = self.fields[
            self.current_step % self.fields_history_size].get()

        self.realspace_fields = np.fft.irfftn(fields_cpu_memory,
                                              norm="forward",
                                              axes=(1, 2, 3),
                                              s=self.full_unpadded_tuple)

    @abstractmethod
    def begin_time_step(self) -> None:
        """Executed in the beginning of the time step. Should be used to
        advance any system-specific counters.

        """
        self.current_step += 1

        # Set this to None so that get_realspace_fields() knows
        # whether it has already been called. Saves some time.
        self.realspace_fields = None

    @abstractmethod
    def calculate_nonlinear_terms(self) -> None:
        """Computes the nonlinear terms and adjusts the time step if
        necessary.

        Called in the beginning of a time step.

        """

        self._update_dt()
        self._update_ab3_coefficients()

    @abstractmethod
    def finish_time_step(self) -> None:
        """Combines the nonlinear and linear terms in order to finish the time
        step"""
        self.finish_step_kernel(
            (self.half_unpadded_cuda_grid_size,),
            (self.cuda_block_size,),
            (
                self.float(self.current_dt),
                self.current_step,
                self.ab3_coefficients[0],
                self.ab3_coefficients[1],
                self.ab3_coefficients[2],
                self.fields[self.current_step % self.fields_history_size - 1],
                self.dft_bits,
                self.fields[self.current_step % self.fields_history_size],
            ),
        )

        self.current_time += self.current_dt
