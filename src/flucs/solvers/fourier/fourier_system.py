"""
Abstract base class for a system that can be solved by FourierSolver.
"""

from abc import abstractmethod
from typing import ClassVar

import cupy as cp
import numpy as np
from cupy.cuda import cufft

from flucs.diagnostic import FlucsDiagnostic
from flucs.input import InvalidFlucsInputFileError
from flucs.systems import FlucsSystem
from flucs.utilities.cupy import cupy_set_device_pointer
from flucs.utilities.messages import flucsprint
from flucs.utilities.smooth_numbers import next_smooth_number

from .fourier_system_diagnostics import (
    FourierDataDiag,
    LinearEigensystemDiag,
    RealspaceDataDiag,
)


class FourierSystem(FlucsSystem):
    """A generic system of equations solved using pseudospectral Fourier
    methods."""

    # Number of fields that the solver is solving for
    number_of_fields: int

    # Number of fields whose equations contain nonlinear terms. This is
    # typically the same as number_of_fields, but can be smaller for some
    # systems
    number_of_fields_nonlinear: int

    # Derivatives and bits used for the nonlinear terms
    number_of_dft_derivatives: int
    dft_derivatives: cp.ndarray
    real_derivatives: cp.ndarray

    number_of_dft_bits: int
    dft_bits: cp.ndarray
    real_bits: cp.ndarray

    # DFT plans for the derivatives and bits
    plan_derivatives_c2r: cufft.PlanNd
    plan_bits_r2c: cufft.PlanNd

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

    # Linear quantities (used for linear postprocessing)
    linear_matrix: np.ndarray | None = None
    linear_eigensystem: dict[str, np.ndarray] | None = None

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

    # Diagnostics available to all FourierSystems
    diags: ClassVar[set[type[FlucsDiagnostic]]] = {
        LinearEigensystemDiag,
        FourierDataDiag,
        RealspaceDataDiag,
    }

    # Forcing methods
    solver_forcing_methods: ClassVar[frozenset[str]] = frozenset() # Currently none
    system_forcing_methods: ClassVar[frozenset[str]] = frozenset()

    def _interpret_input(self):
        """Validates inputs and sets up the number of lattice points."""

        # Check for conflicts in time-stepping input parameters
        if self.input["time.dt_method"] == "discrete":
            flucsprint("Using discrete time stepping.")

        elif self.input["time.dt_method"] == "continuous":
            if self.input["setup.precompute_linear_matrix"]:
                raise InvalidFlucsInputFileError(
                    "Cannot have setup.precompute_linear_matrix = true if "
                    "time.dt_method = 'continuous'."
                )
            flucsprint("Using continuous time stepping.")

        else:
            raise InvalidFlucsInputFileError(
                f"Invalid time.dt_method: {self.input['time.dt_method']}. "
                "Must be either 'discrete' or 'continuous'."
            )

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

        # Check for conflicts in forcing parameters
        forcing_method = self.input["forcing.method"]
        if forcing_method and forcing_method not in (
            set(self.solver_forcing_methods) | set(self.system_forcing_methods)
        ):
            raise InvalidFlucsInputFileError(
                f"Invalid forcing.method: {self.input['forcing.method']}."
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

                    flucsprint(
                        f"Found padded_n{dim} = {padded_n} for n{dim} = {n}"
                    )

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

                    flucsprint(
                        f"Found n{dim} = {n} for padded_n{dim} = {padded_n}"
                    )

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

    def setup(self) -> None:
        """
        Sets up the system for running the solver.
        """

        # Base FlucsSystem setup
        super().setup()

        self._set_initial_conditions()
        self._check_initial_conditions()

        # Timestep setup
        self.dt_max = self.input["time.dt_max"]
        self.dt_min = self.input["time.dt_min"]
        self.max_cfl = self.input["time.max_cfl"]
        self.dt_mult_increase = self.input["time.dt_mult_increase"]
        self.dt_mult_decrease = self.input["time.dt_mult_decrease"]

        # Determine the time stepping method
        if self.input["time.dt_method"] == "discrete":
            self.sub_cfl_steps = self.int(0)
            self.dt_mult_steps = self.input["time.dt_mult_steps"]
            self._compute_current_dt = self._compute_current_dt_discrete

        elif self.input["time.dt_method"] == "continuous":
            self._compute_current_dt = self._compute_current_dt_continuous

        # Allocate memory
        self._allocate_memory()

    def _allocate_memory(
        self,
        allocate_derivatives_and_bits=True,
        combine_derivatives_and_bits=False,
    ) -> None:
        """
        Allocates any CPU/GPU memory that is needed by the solver.

        Each system can implement its own version but should always
        call the base one first.

        Parameters
        ----------
        allocate_derivatives_and_bits : bool
            If true, FourierSystem uses self.number_of_dft_derivatives and
            self.number_of_dft_bits to allocate arrays and set up CuFFT plans
            for the necessary Fourier transforms.
        combine_derivatives_and_bits : bool
            If true, the arrays for dft_derivatives and bits are reused
            to save memory.

        """

        # Fields at the current and previous steps as required
        self.fields = [
            cp.zeros(
                (self.number_of_fields, self.nz, self.nx, self.half_ny),
                dtype=self.complex,
            )
            for i in range(self.fields_history_size)
        ]

        if self.input["setup.linear"]:
            # Dummy placeholder that is passed to the kernels
            # when running linearly
            self.dft_bits = cp.zeros(1, dtype=self.complex)
            return

        # For the nonlinear terms, we need to keep terms at the current
        # time step + terms from the past 2 time steps since we are
        # using AB3.
        # The nonlinear terms are indexed as (step, field, kz, kx, ky)
        self.multistep_nonlinear_terms = cp.zeros(
            (
                3,
                self.number_of_fields_nonlinear,
                self.nz,
                self.nx,
                self.half_ny,
            ),
            dtype=self.complex,
        )

        # CFL in GPU memory
        self.cfl_rate = cp.zeros([1], dtype=self.float)

        # Don't do anything if the user wants to handle this manually
        if not allocate_derivatives_and_bits:
            return

        # Combining derivatives and bits is advisable as it saves memory
        if combine_derivatives_and_bits:
            combined_size = max(
                self.number_of_dft_derivatives, self.number_of_dft_bits
            )

            self.dft_derivatives = cp.zeros(
                (
                    combined_size,
                    self.padded_nz,
                    self.padded_nx,
                    self.half_padded_ny,
                ),
                dtype=self.complex,
            )
            self.real_derivatives = cp.zeros(
                (combined_size, self.padded_nz, self.padded_nx, self.padded_ny),
                dtype=self.float,
            )

            self.dft_bits = self.dft_derivatives
            self.real_bits = self.real_derivatives

        else:
            self.dft_derivatives = cp.zeros(
                (
                    self.number_of_dft_derivatives,
                    self.padded_nz,
                    self.padded_nx,
                    self.half_padded_ny,
                ),
                dtype=self.complex,
            )
            self.real_derivatives = cp.zeros(
                (
                    self.number_of_dft_derivatives,
                    self.padded_nz,
                    self.padded_nx,
                    self.padded_ny,
                ),
                dtype=self.float,
            )

            self.dft_bits = cp.zeros(
                (
                    self.number_of_dft_bits,
                    self.padded_nz,
                    self.padded_nx,
                    self.half_padded_ny,
                ),
                dtype=self.complex,
            )
            self.real_bits = cp.zeros(
                (
                    self.number_of_dft_bits,
                    self.padded_nz,
                    self.padded_nx,
                    self.padded_ny,
                ),
                dtype=self.float,
            )

        self.plan_derivatives_c2r = self.create_standard_real_cufft_plan(
            fft_type="c2r",
            padded=True,
            batch_size=self.number_of_dft_derivatives,
        )

        self.plan_bits_r2c = self.create_standard_real_cufft_plan(
            fft_type="r2c",
            padded=True,
            batch_size=self.number_of_dft_bits,
        )

    def create_standard_real_cufft_plan(
        self, fft_type: str, padded: bool, batch_size: int
    ):
        """
        Returns a CuFFT plan for real-to-complex ("r2c") or
        complex-to-real ("c2r") transforms for data of standard FourierSystem
        shape (batch, nz, nx, ny).

        Parameters
        ----------
        type : str
            Type of the FFT. Can be "c2r" or "r2c".
        padded : bool
            If true, switch to complex arrays of shape
                (batch, padded_nz, padded_nx, half_padded_ny)
            that are transformed to real arrays of shape
                (batch, padded_nz, padded_nx, padded_ny)
        batch_size : int
            Numbers of FFTs in the batch.
        """

        if padded:
            nz = self.padded_nz
            nx = self.padded_nx
            ny = self.padded_ny
            half_ny = self.half_padded_ny
        else:
            nz = self.nz
            nx = self.nx
            ny = self.ny
            half_ny = self.half_ny

        shape = (nz, nx, ny)
        istride = 1
        ostride = 1
        compex_embed = (1, nx, half_ny)
        compex_dist = nz * nx * half_ny
        real_embed = (1, nx, ny)
        real_dist = nz * nx * ny

        if fft_type == "c2r":
            inembed = compex_embed
            onembed = real_embed
            idist = compex_dist
            odist = real_dist
            fft_type = self.fft_c2r_plan_type
            last_size = ny
        elif fft_type == "r2c":
            inembed = real_embed
            onembed = compex_embed
            idist = real_dist
            odist = compex_dist
            fft_type = self.fft_r2c_plan_type
            last_size = half_ny
        else:
            raise ValueError("fft_type must be c2r or r2c.")

        return cufft.PlanNd(
            shape=shape,
            istride=istride,
            ostride=ostride,
            inembed=inembed,
            onembed=onembed,
            idist=idist,
            odist=odist,
            fft_type=fft_type,
            batch=batch_size,
            order="C",
            last_axis=3,
            last_size=last_size,
        )

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

    def check_health(self) -> None:
        """Basic consistency/health checks before running.
        Alerts the user if anything needs their attention.

        """

        # Check consistency of linear matrices
        matrix_solver = self.compute_linear_matrix()
        matrix_reference = self.compute_linear_matrix_reference()

        # Check against the reference linear matrix if provided by the user
        if matrix_reference is not None:
            kx, ky, kz = self.get_broadcast_wavenumbers()
            hyperdissipation = np.zeros(
                self.half_unpadded_tuple, dtype=self.float
            )

            for component, k2 in [
                ("perp", kx**2 + ky**2),
                ("kx", kx**2),
                ("ky", ky**2),
                ("kz", kz**2),
            ]:
                coeff = self.input[f"hyperdissipation.{component}"]
                if coeff > 0.0:
                    k2_max = np.max(np.abs(k2))
                    contribution = coeff * (
                        (k2 / k2_max) **
                        self.input[f"hyperdissipation.{component}_power"]
                    )
                    if self.input[f"hyperdissipation.{component}_adaptive"]:
                        contribution /= self.init_dt

                    hyperdissipation += contribution

            diag = np.arange(self.number_of_fields)
            matrix_reference[diag, diag, :, :, :] += hyperdissipation

            if not np.allclose(matrix_reference, matrix_solver):
                raise ValueError(
                    "The linear matrix computed by CUDA disagrees "
                    "with provided reference matrix."
                )

        # Compare relevant linear frequencies to dt_max
        eigvals = self.compute_linear_eigensystem()["eigvals"]
        max_growth = np.max(eigvals.imag)
        max_damping = np.max(-eigvals.imag)
        max_real_frequency = np.max(np.abs(eigvals.real))

        flucsprint(
            "Linear rates (max.):          "
            f"(growth, damping, frequency) = "
            f"({max_growth:.3e}, "
            f"{max_damping:.3e}, "
            f"{max_real_frequency:.3e})"
        )

        flucsprint(
            "Linear rates (max.): dt_max * "
            f"(growth, damping, frequency) = "
            f"({self.dt_max * max_growth:.3e}, "
            f"{self.dt_max * max_damping:.3e}, "
            f"{self.dt_max * max_real_frequency:.3e})"
        )

        # Check whether dt_max is appropriate given the linear properties
        tol = 2.0
        if self.dt_max * max_growth > tol:
            raise InvalidFlucsInputFileError(
                "(dt_max * max growth rate) is too large."
            )

        if self.dt_max * max_real_frequency > tol:
            raise InvalidFlucsInputFileError(
                "(dt_max * max frequency) is too large."
            )

    def ready(self) -> None:

        # Reset time counters
        self.current_step = self.int(0)
        self.current_time = self.init_time

        # Reset time step
        self.current_dt = self.init_dt
        self.dt_array = np.array(
            [self.current_dt, 10**10, 10**10], dtype=self.float
        )
        self.ab3_coefficients = np.array([1, 0, 0], dtype=self.float)

        # Reset CFL
        self.current_cfl = 0.0

        # Copy initial condition
        self.fields[0][:] = cp.array(
            np.reshape(self.fields_initial, self.fields[0].shape)
        )

        # Reset AB3 nonlinear history
        if not self.input["setup.linear"]:
            self.multistep_nonlinear_terms.fill(self.float(0.0))

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

        # Print starting message
        flucsprint(
            f"Starting at time {float(self.init_time):.3e}, "
            f"dt {float(self.init_dt):.3e}"
        )

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
        self.module_options.define_int(
            "NUMBER_OF_FIELDS", self.number_of_fields
        )

        self.module_options.define_int(
            "NUMBER_OF_FIELDS_NONLINEAR", self.number_of_fields_nonlinear
        )

        self.module_options.define_dimension(
            "HALFUNPADDEDSIZE", self.half_unpadded_size
        )
        self.module_options.define_dimension(
            "HALFPADDEDSIZE", self.half_padded_size
        )
        self.module_options.define_dimension(
            "PADDEDSIZE", self.full_padded_size
        )

        self.module_options.define_float(
            "DFT_PADDEDSIZE_FACTOR", self.float(1.0 / self.full_padded_size)
        )

        # Dimensions
        for dim in ["x", "y", "z"]:
            box_size = self.float(self.input[f"dimensions.L{dim}"])

            self.module_options.define_float(
                f"TWOPI_OVER_L{dim.upper()}", 2 * np.pi / box_size
            )

            self.module_options.define_dimension(
                f"N{dim.upper()}", getattr(self, f"n{dim}")
            )
            self.module_options.define_float(f"L{dim.upper()}", box_size)
            self.module_options.define_dimension(
                f"HALF_N{dim.upper()}", getattr(self, f"half_n{dim}")
            )
            self.module_options.define_dimension(
                f"PADDED_N{dim.upper()}", getattr(self, f"padded_n{dim}")
            )
            self.module_options.define_dimension(
                f"HALF_PADDED_N{dim.upper()}",
                getattr(self, f"half_padded_n{dim}"),
            )

        # Hyperdissipation
        for component in ["perp", "kx", "ky", "kz"]:
            if self.input[f"hyperdissipation.{component}"] > 0.0:
                message = f"Using hyperdissipation in {component:<4}"

                self.module_options.define_float(
                    f"HYPERDISSIPATION_{component.upper()}",
                    self.input[f"hyperdissipation.{component}"],
                )
                self.module_options.define_float(
                    f"HYPERDISSIPATION_{component.upper()}_POWER",
                    self.input[f"hyperdissipation.{component}_power"],
                )
                if self.input[f"hyperdissipation.{component}_adaptive"]:
                    self.module_options.define_flag(
                        f"HYPERDISSIPATION_{component.upper()}_ADAPTIVE"
                    )
                    message += " (adaptive)"

                flucsprint(message)

        # Forcing
        if self.input["forcing.method"]:
            flucsprint(f"Using forcing method: {self.input['forcing.method']}")

            self.module_options.define_flag("FORCING")
            self.module_options.define_flag(
                f"FORCING_METHOD_{self.input['forcing.method'].upper()}"
            )

            if self.input["forcing.method"] in self.solver_forcing_methods:
                self.module_options.define_flag("FORCING_FROM_SOLVER")

        # Setup
        self.module_options.define_float("ALPHA", self.input["setup.alpha"])

        if not self.input["setup.linear"]:
            self.module_options.define_flag("NONLINEAR")

        if self.input["setup.precompute_linear_matrix"]:
            flucsprint("Linear matrices will be precomputed.")
            self.module_options.define_flag("PRECOMPUTE_LINEAR_MATRIX")

        super().compile_cupy_module()

        self.finish_step_kernel = self.cupy_module.get_function("finish_step")

    def setup_cuda_grids(self) -> None:
        """Sets up the grids and blocks for CUDA kernels.

        In the future, this may be the place to do some automatic optimisation.
        As it stands, this is sysem-specific.
        """

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

    def compute_linear_matrix(self) -> np.ndarray:
        """
        Computes the linear matrix used by the solver and stores it in
        self.linear_matrix. Note that this is not used directly in the
        solver loop, and so is used entirely for diagnostic purposes.

        """

        if self.linear_matrix is not None:
            return self.linear_matrix

        # Linear matrix in GPU memory
        linear_matrix_cupy = cp.zeros(
            (
                self.number_of_fields,
                self.number_of_fields,
                self.nz,
                self.nx,
                self.half_ny,
            ),
            dtype=self.complex,
        )

        # Get kernel
        compute_linear_matrix_kernel = self.cupy_module.get_function(
            "compute_linear_matrix"
        )

        # Compute
        compute_linear_matrix_kernel(
            (self.half_unpadded_cuda_grid_size,),
            (self.cuda_block_size,),
            (self.init_dt, linear_matrix_cupy),
        )

        self.linear_matrix = cp.asnumpy(linear_matrix_cupy)

        return self.linear_matrix

    def compute_linear_matrix_reference(self) -> np.ndarray | None:
        """
        Returns a user-defined reference linear matrix that should be
        the same shape as self.linear_matrix. This should be calculated
        using only CPU resources, and should be of shape

        (nfields, nfields, nz, nx, half_ny)

        If the user does not provide a reference linear matrix, the default
        value is None.

        """

        return None

    def compute_linear_eigensystem(self) -> dict[str, np.ndarray]:
        """
        Computes both the eigenvalues and (normalised) eigenvectors
        of the linear matrix used by the solver.

        The eigenvalues are the complex frequencies of
        Fourier modes of the form exp(-i*omega*t).

        The eigenvectors are normalised to unit L2 norm and a phase
        where the component with largest absolute value is real and positive.
        """

        if self.linear_eigensystem is not None:
            return self.linear_eigensystem

        # Handle matrix from solver
        linear_matrix = cp.asnumpy(self.compute_linear_matrix()).copy()
        linear_matrix = np.moveaxis(linear_matrix, (0, 1), (-2, -1))
        # (nfields, nfields, nz, nx, half_ny) -> (..., nfields, nfields)

        eigvals, eigvecs = np.linalg.eig(linear_matrix)

        eigvals = (-1j * eigvals).transpose(3, 0, 1, 2)
        eigvecs = eigvecs.transpose(4, 3, 0, 1, 2)
        # (nz, nx, half_ny,          mode) -> (mode,          ...)
        # (nz, nx, half_ny, nfields, mode) -> (mode, nfields, ...)

        # Normalise to unit norm
        eigvecs /= np.linalg.norm(eigvecs, axis=1, keepdims=True)

        # Find field component with largest amplitude for each mode
        indices = np.abs(eigvecs).argmax(axis=1, keepdims=True)
        components = np.take_along_axis(eigvecs, indices, axis=1)

        # Normalise by phase
        phase = np.where(
            np.abs(components) > 0, np.sign(components), 1.0 + 0.0j
        )
        eigvecs *= np.conj(phase)

        # Compute inverse of solver eigenvectors for projection
        eigvecs_inverse = np.linalg.inv(
            eigvecs.transpose(2, 3, 4, 1, 0)
        ).transpose(3, 4, 0, 1, 2)

        # Assign class variable
        self.linear_eigensystem = {
            "eigvals": eigvals,
            "eigvecs": eigvecs,
            "eigvecs_inverse": eigvecs_inverse,
        }

        return self.linear_eigensystem

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
        flucsprint(f"Init. condition reality error: {error:.3e}")

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
            flucsprint(
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
                flucsprint(
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
            flucsprint(
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
            self.current_step % self.fields_history_size
        ].get()

        self.realspace_fields = np.fft.irfftn(
            fields_cpu_memory,
            norm="forward",
            axes=(1, 2, 3),
            s=self.full_unpadded_tuple,
        )

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
