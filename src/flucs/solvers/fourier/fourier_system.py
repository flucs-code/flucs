"""
Abstract base class for a system that can be solved by FourierSolver.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable
from typing import ClassVar

import cupy as cp
import numpy as np
from cupy.cuda import cufft

from flucs.diagnostic import FlucsDiagnostic
from flucs.input import InvalidFlucsInputFileError
from flucs.systems import FlucsSystem
from flucs.utilities.cupy import KernelWrapper
from flucs.utilities.messages import flucsprint
from flucs.utilities.smooth_numbers import next_smooth_number

from .fourier_system_diagnostics import (
    FourierDataDiag,
    LinearEigensystemDiag,
    RealspaceDataDiag,
)
from .fourier_system_forcing import FourierSystemForcing


class FourierSystem(FlucsSystem):
    """
    A generic system of equations solved using pseudospectral Fourier
    methods.
    """

    # Number of fields that the solver is solving for
    number_of_fields: int

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
    linear_propagator: np.ndarray | None = None

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

    # Hyperdissipation variables
    hyperdissipation_components = ("kz", "kx", "ky", "kperp")
    adaptive_rate: float

    # CUDA kernels
    compute_linear_matrix_kernel: KernelWrapper
    compute_propagator_global_kernel: KernelWrapper
    compute_solved_grid_mask_kernel: KernelWrapper
    cuda_block_size: int = 512

    # CUDA grids
    half_cuda_grid_size: int
    full_cuda_grid_size: int

    # Initial conditions, always in CPU memory
    fields_initial: np.ndarray

    # Array sizes
    nx: int
    ny: int
    nz: int
    half_nx: int
    half_ny: int
    half_nz: int

    nx_unpadded: int
    ny_unpadded: int
    nz_unpadded: int
    half_nx_unpadded: int
    half_ny_unpadded: int
    half_nz_unpadded: int

    half_size: int
    half_tuple: tuple
    full_size: int
    full_tuple: tuple

    # Fourier wavenumbers
    kx: np.ndarray
    ky: np.ndarray
    kz: np.ndarray

    # kperp shells
    shell_kperp_min: float
    shell_kperp_max: float
    shell_nkperp: int
    shell_kperp: np.ndarray

    # Solved-grid mask
    solved_grid_mask: np.ndarray

    # Diagnostics available to all FourierSystems
    diags: ClassVar[set[type[FlucsDiagnostic]]] = {
        LinearEigensystemDiag,
        FourierDataDiag,
        RealspaceDataDiag,
    }

    # Forcing methods
    forcing_object: FourierSystemForcing
    solver_forcing_methods: ClassVar[dict[str, type[FourierSystemForcing]]] = {}
    system_forcing_methods: ClassVar[dict[str, type[FourierSystemForcing]]] = {}

    def _interpret_input(self):
        """Validates inputs and sets up the number of lattice points."""

        # Check for conflicts in time-stepping input parameters
        if self.input["time.dt_method"] == "discrete":
            flucsprint("Using discrete time stepping.")

        elif self.input["time.dt_method"] == "continuous":
            if self.input["timestepping.precompute_linear_matrix"]:
                raise InvalidFlucsInputFileError(
                    "Cannot have timestepping.precompute_linear_matrix = "
                    "true if time.dt_method = 'continuous'."
                )
            flucsprint("Using continuous time stepping.")

        else:
            raise InvalidFlucsInputFileError(
                f"Invalid time.dt_method: {self.input['time.dt_method']}. "
                "Must be either 'discrete' or 'continuous'."
            )

        # Check for conflicts in hyperdissipation parameters
        if self.input["hyperdissipation.kperp"] > 0.0 and (
            self.input["hyperdissipation.kx"] > 0.0
            or self.input["hyperdissipation.ky"] > 0.0
        ):
            raise InvalidFlucsInputFileError(
                "Cannot enable both hyperdissipation.kperp "
                "and hyperdissipation.kx/ky simultaneously. "
                "Use either kperp or kx/ky. "
            )

        # Set resolutions appropriately
        for dim in ["x", "y", "z"]:
            n_unpadded = self.input[f"dimensions.n{dim}_unpadded"]
            n = self.input[f"dimensions.n{dim}"]

            match (n_unpadded > 0, n > 0):
                case (True, True):
                    # Check if n is odd
                    if n_unpadded % 2 == 0:
                        raise ValueError(
                            "Unpadded resolutions must be odd! "
                            f"Please change n{dim} = {n} to an odd number!"
                        )

                    half_n_unpadded = n_unpadded // 2 + 1
                    half_n = n // 2 + 1
                    # TODO: add some check that warns the user if their choice
                    # is dumb

                case (True, False):
                    # Check if n is odd
                    if n_unpadded % 2 == 0:
                        raise ValueError(
                            "Unpadded resolutions must be odd! "
                            f"Please change n{dim} = {n} to an odd number!"
                        )

                    half_n_unpadded = n_unpadded // 2 + 1

                    # Find minimum padded that works
                    n = next_smooth_number(
                        (self.input["dimensions.nonlinear_order"] + 1)
                        * half_n_unpadded,
                        primes=self.input["dimensions.padded_primes"],
                    )

                    half_n = n // 2 + 1

                    flucsprint(
                        f"Found n{dim} = {n} for n{dim}_unpadded = {n_unpadded}"
                    )

                case (False, True):
                    # Given a padded_n, it's easiest to figure out half_n

                    factor = self.input["dimensions.nonlinear_order"] + 1
                    _x = n // factor
                    half_n = n // 2 + 1

                    # Handle an annoying edge case
                    if n % factor == 0:
                        _x -= 1

                    half_n_unpadded = _x + 1
                    n_unpadded = 2 * _x + 1

                    flucsprint(
                        f"Found n{dim}_unpadded = {n_unpadded} for n{dim} = {n}"
                    )

                case (False, False):
                    raise ValueError(
                        f"At least one of n{dim}_unpadded and "
                        f"n{dim} must be positive!"
                    )

                # This is added only to make pyright happy.
                case _:
                    raise RuntimeError("How the fluc did you get here?")

            # It's useful to have the resolutions as part of the system
            # rather than to access the input dictionary every time
            setattr(self, f"n{dim}_unpadded", n_unpadded)
            setattr(self, f"n{dim}", n)
            setattr(self, f"half_n{dim}_unpadded", half_n_unpadded)
            setattr(self, f"half_n{dim}", half_n)

        # Set padded and unpadded array sizes

        self.half_size = self.nz * self.nx * self.half_ny
        self.half_tuple = (self.nz, self.nx, self.half_ny)

        self.full_size = self.nz * self.nx * self.ny
        self.full_tuple = (self.nz, self.nx, self.ny)

        # Precompute wavenumbers (useful for many things)
        self._precompute_wavenumbers()

        # Setup forcing
        self._setup_forcing()

    def _setup_forcing(self):
        """Sets up the forcing method."""
        forcing_method = self.input["forcing.method"]
        if not forcing_method:
            return

        if forcing_method not in (
            self.solver_forcing_methods | self.system_forcing_methods
        ):
            raise InvalidFlucsInputFileError(
                f"Invalid forcing.method: {self.input['forcing.method']}."
            )

        self.forcing_object = (
            self.solver_forcing_methods | self.system_forcing_methods
        )[forcing_method](self)

    def setup(self) -> None:
        """
        Sets up the system for running the solver.
        """

        # Base FlucsSystem setup
        super().setup()

        # Initialise shell grids for diagnostics
        self._compute_kperp_shells()

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

    @property
    def requires_explicit_terms(self) -> bool:
        """
        Whether explicit terms need to be allocated and computed for this system
        """

        return not self.input["setup.linear"] or (
            bool(self.input["forcing.method"]) and self.forcing_object.explicit
        )

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
                (self.number_of_fields, *self.half_tuple),
                dtype=self.complex,
            )
            for i in range(self.fields_history_size)
        ]

        if not self.requires_explicit_terms:
            # Dummy placeholder that is passed to the kernels
            # when running with no explicit terms
            self.dft_bits = cp.zeros(1, dtype=self.complex)
            return

        # CFL in GPU memory
        self.cfl_rate = cp.zeros([1], dtype=self.float)

        if self.input["setup.linear"]:
            # Dummy placeholder that is passed to the kernels
            # when running linearly
            self.dft_bits = cp.zeros(1, dtype=self.complex)
            return

        # Don't do anything if the user wants to handle this manually
        if not allocate_derivatives_and_bits:
            return

        # Combining derivatives and bits is advisable as it saves memory
        if combine_derivatives_and_bits:
            combined_size = max(
                self.number_of_dft_derivatives, self.number_of_dft_bits
            )

            self.dft_derivatives = cp.zeros(
                (combined_size, *self.half_tuple),
                dtype=self.complex,
            )
            self.real_derivatives = cp.zeros(
                (combined_size, *self.full_tuple),
                dtype=self.float,
            )

            self.dft_bits = self.dft_derivatives
            self.real_bits = self.real_derivatives

        else:
            self.dft_derivatives = cp.zeros(
                (self.number_of_dft_derivatives, *self.half_tuple),
                dtype=self.complex,
            )
            self.real_derivatives = cp.zeros(
                (self.number_of_dft_derivatives, *self.full_tuple),
                dtype=self.float,
            )

            self.dft_bits = cp.zeros(
                (self.number_of_dft_bits, *self.half_tuple),
                dtype=self.complex,
            )
            self.real_bits = cp.zeros(
                (self.number_of_dft_bits, *self.full_tuple),
                dtype=self.float,
            )

        self.plan_derivatives_c2r = self.create_standard_real_cufft_plan(
            fft_type="c2r",
            batch_size=self.number_of_dft_derivatives,
        )

        self.plan_bits_r2c = self.create_standard_real_cufft_plan(
            fft_type="r2c",
            batch_size=self.number_of_dft_bits,
        )

    def create_dealiased_fourier_to_real(
        self,
        cuda_device_function: str,
        n_in: int | None = None,
        n_out: int | None = None,
        output_fourier_array: cp.ndarray = None,
        shared_mem: int = 0,
    ) -> Callable[..., cp.ndarray]:

        if output_fourier_array is None:
            output_fourier_array = self.dft_derivatives

        if n_in is None:
            n_in = self.number_of_fields

        # Create fft operation
        if n_out is None:
            fft_plan = self.plan_derivatives_c2r
            n_out = self.number_of_dft_derivatives
        else:
            fft_plan = self.create_standard_real_cufft_plan(
                fft_type="c2r",
                batch_size=n_out,
            )

        # Create data kernel
        data_kernel = KernelWrapper(
            system=self,
            cuda_kernel_name=f"dealiased_fourier_operation<{n_in}, {n_out}, {cuda_device_function}>",
            grid=(self.half_cuda_grid_size,),
            block=(self.cuda_block_size,),
            shared_mem=shared_mem,
        )

        def dealiased_fourier_to_real_operation(current_dt, current_time, current_step, input_array, output_real_array):
            data_kernel(
                current_dt,
                current_time,
                current_step,
                input_array,
                output_fourier_array
            )
            fft_plan.fft(
                output_fourier_array,
                output_real_array,
                cufft.CUFFT_INVERSE,
            )

        return dealiased_fourier_to_real_operation

    def create_dealiased_real_to_fourier(
        self,
        cuda_device_function: str,
        n_in: int | None = None,
        n_out: int | None = None,
        output_real_array: cp.ndarray = None,
        shared_mem: int = 0,
    ) -> Callable[..., cp.ndarray]:

        if output_real_array is None:
            output_real_array = self.real_bits

        if n_in is None:
            n_in = self.number_of_dft_derivatives

        # Create fft operation
        if n_out is None:
            fft_plan = self.plan_bits_r2c
            n_out = self.number_of_dft_bits
        else:
            fft_plan = self.create_standard_real_cufft_plan(
                fft_type="r2c",
                batch_size=n_out,
            )

        # Create data kernel
        data_kernel = KernelWrapper(
            system=self,
            cuda_kernel_name=f"real_operation<{n_in}, {n_out}, {cuda_device_function}>",
            grid=(self.full_cuda_grid_size,),
            block=(self.cuda_block_size,),
            shared_mem=shared_mem,
        )

        def dealiased_real_to_fourier_operation(current_dt, current_time, current_step, input_real_array, output_array, calculate_cfl):

            if calculate_cfl:
                self.cfl_rate[0] = 0

            data_kernel(
                current_dt,
                current_time,
                current_step,
                input_real_array,
                output_real_array,
                calculate_cfl,
                self.cfl_rate
            )
            fft_plan.fft(
                output_real_array,
                output_array,
                cufft.CUFFT_FORWARD,
            )

        return dealiased_real_to_fourier_operation

    def create_standard_real_cufft_plan(
        self, fft_type: str, batch_size: int
    ):
        """
        Returns a CuFFT plan for real-to-complex ("r2c") or
        complex-to-real ("c2r") transforms for data of standard FourierSystem
        shape (batch, nz, nx, ny).

        Parameters
        ----------
        type : str
            Type of the FFT. Can be "c2r" or "r2c".
        batch_size : int
            Numbers of FFTs in the batch.
        """

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

    def _compute_kperp_shells(self):
        """
        Sets the default kperp shell grid used.

        The CUDA shell-sum kernels use uniform half-open bins,

            [kperp_min, kperp_max),

        with bin index

            floor((kperp - kperp_min) * nkperp / (kperp_max - kperp_min)).

        The default kperp_max is padded above the largest resolved diagonal mode
        so that summing all bins includes all resolved modes.
        """

        # TODO: if adding NS/isotropic systems, either add to this or add a
        # separate method for isotropic systems (_precompute_shells_isotropic)
        # alongside the appropriate kernels to do the isotropic shell calcs, as
        # well as create_shell_reduction_isotropic

        # Check if we have already done this
        if hasattr(self, "shell_kperp"):
            return

        # kperp grid spacing
        dkperp = min(
            (k[1] for k in (self.kx, self.ky) if k.size > 1),
            default=self.float(1.0),
        )

        # Minimum kperp
        kperp_min = self.float(0.0)

        # Maximum kperp
        kx_max = abs(self.kx[self.half_nx - 1])
        ky_max = abs(self.ky[self.half_ny - 1])

        kperp_max = np.sqrt(kx_max**2 + ky_max**2)
        kperp_max += dkperp  # Adding padding for diagonal

        # Number of kperp shells
        nkperp_from_dkperp = int(np.ceil((kperp_max - kperp_min) / dkperp))
        nkperp = min(nkperp_from_dkperp, self.cuda_block_size)

        # Maximum kperp from bin width
        bin_width = (
            dkperp
            if nkperp_from_dkperp <= self.cuda_block_size
            else (kperp_max - kperp_min) / nkperp
        )

        kperp_max = self.float(kperp_min + nkperp * bin_width)

        # kperp grid
        kperp = kperp_min + bin_width * np.arange(nkperp, dtype=self.float)

        # Assign attributes
        self.shell_kperp_min = kperp_min
        self.shell_kperp_max = kperp_max
        self.shell_nkperp = nkperp
        self.shell_kperp = kperp
        self.shell_last_complete_bin = int(
            (min(kx_max, ky_max) - kperp_min) * nkperp / (kperp_max - kperp_min)
        )

    def get_broadcast_wavenumbers(self):
        """Returns wavenumber arrays broadcast to (nz, nx, half_ny)

        Returns
        -------
        kz_broadcast, kx_broadcast, ky_broadcast
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

        return kz_broadcast, kx_broadcast, ky_broadcast

    def check_health(self) -> None:
        """
        Basic consistency/health checks before running.
        Alerts the user if anything needs their attention.
        """

        self._check_linear_matrix()

    def _check_linear_matrix(self) -> None:
        if not self.input["setup.check_linear_matrix"]:
            flucsprint(
                "Skipping linear matrix check.",
                source=self,
                message_type="warning",
            )
            return

        # Check consistency of linear matrices
        matrix_solver = self.compute_linear_matrix()
        matrix_reference = self.compute_linear_matrix_reference()
        solved_grid_mask = self.get_solved_grid_mask().astype(bool)

        # Check against the reference linear matrix if provided by the user
        if matrix_reference is not None:
            if not np.allclose(
                matrix_reference[..., solved_grid_mask],
                matrix_solver[..., solved_grid_mask],
            ):
                raise ValueError(
                    "The linear matrix computed by CUDA disagrees "
                    "with provided reference matrix."
                )

        # Calculate eigenvalues of linear matrix
        eigvals = self.compute_linear_eigensystem()["eigvals"]
        eigvals = eigvals[:, solved_grid_mask]
        eigvals = np.sort(eigvals, axis=0)
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

        # Evaluate accuracy of Pade approximations for exponential
        propagator = self.compute_linear_propagator(dt=self.dt_max)
        propagator = propagator[..., solved_grid_mask]
        propagator = np.moveaxis(propagator, (0, 1), (-2, -1))

        pade_eigvals = (
            1j * np.log(np.linalg.eigvals(propagator)) / self.dt_max
        ).T
        pade_eigvals = np.sort(pade_eigvals, axis=0)

        abs_errors = np.abs(pade_eigvals - eigvals)
        rel_errors = np.divide(
            abs_errors,
            np.abs(eigvals),
            out=np.zeros_like(abs_errors),
            where=np.abs(eigvals) > self.tolerance,
        )

        flucsprint(
            "Linear error (max.):          "
            f"(absolute, relative)         = "
            f"({np.max(abs_errors):.3e}, "
            f"{np.max(rel_errors):.3e})"
        )

        if np.max(rel_errors) > 1e-2:
            flucsprint(
                "significant linear propagator errors, consider using a smaller"
                " dt_max.",
                source=self,
                message_type="warning",
            )

    def ready(self) -> None:
        # Reset time counters
        self.current_step = self.int(0)
        self.current_time = self.init_time

        # Reset time step
        self.current_dt = self.init_dt
        self.adaptive_rate = self.float(1.0) / self.current_dt

        # Reset CFL
        self.current_cfl = 0.0

        # Copy initial condition
        self.fields[0][:] = cp.array(
            np.reshape(self.fields_initial, self.fields[0].shape)
        )

        # Reset realspace fields
        self.realspace_fields = None

        super().ready()

        # Print starting message
        flucsprint(
            f"Starting at time {float(self.init_time):.3e}, "
            f"dt {float(self.init_dt):.3e}"
        )

    def setup_cuda_definitions(self) -> None:
        # FourierSystem specific constants
        self.module_options.define_int(
            "NUMBER_OF_FIELDS", self.number_of_fields
        )

        self.module_options.define_int(
            "NUMBER_OF_DFT_DERIVATIVES",
            self.number_of_dft_derivatives,
        )
        self.module_options.define_int(
            "NUMBER_OF_DFT_BITS",
            self.number_of_dft_bits,
        )
        self.module_options.define_int(
            "NUMBER_OF_DFT_COMBINED",
            max(self.number_of_dft_bits, self.number_of_dft_derivatives),
        )

        self.module_options.define_dimension(
            "HALFSIZE", self.half_size
        )
        self.module_options.define_dimension(
            "FULLSIZE", self.full_size
        )

        self.module_options.define_float(
            "DFT_FULLSIZE_FACTOR", self.float(1.0 / self.full_size)
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
                f"N{dim.upper()}_UNPADDED", getattr(self, f"n{dim}_unpadded")
            )
            self.module_options.define_dimension(
                f"HALF_N{dim.upper()}_UNPADDED",
                getattr(self, f"half_n{dim}_unpadded"),
            )

        # Hyperdissipation
        for index, component in enumerate(self.hyperdissipation_components):
            self.module_options.define_int(
                f"HYPERDISSIPATION_{component.upper()}_INT", index
            )

            if self.input[f"hyperdissipation.{component}"] > 0.0:
                message = f"Using hyperdissipation in {component:<5}"

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

                if self.input[f"hyperdissipation.{component}_normalised"]:
                    self.module_options.define_flag(
                        f"HYPERDISSIPATION_{component.upper()}_NORMALISED"
                    )
                    message += " (normalised)"

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

            if self.forcing_object.linear:
                self.module_options.define_flag("FORCING_LINEAR")

            if self.forcing_object.explicit:
                self.module_options.define_flag("FORCING_EXPLICIT")

            self.forcing_object.setup_cuda_definitions()

        # Setup
        if not self.input["setup.linear"]:
            self.module_options.define_flag("NONLINEAR")

    def register_kernels(self) -> None:
        """Registers the CUDA kernels."""

        # Setup kernel parameters (grid, block, shared memory)
        self.half_cuda_grid_size = (
            self.half_size + self.cuda_block_size - 1
        ) // self.cuda_block_size

        self.full_cuda_grid_size = (
            self.full_size + self.cuda_block_size - 1
        ) // self.cuda_block_size

        self.compute_linear_matrix_kernel = KernelWrapper(
            system=self,
            cuda_kernel_name="compute_linear_matrix",
            grid=(self.half_cuda_grid_size,),
            block=(self.cuda_block_size,),
        )

        self.compute_solved_grid_mask_kernel = KernelWrapper(
            system=self,
            cuda_kernel_name="compute_solved_grid_mask",
            grid=(self.half_cuda_grid_size,),
            block=(self.cuda_block_size,),
        )

        self.compute_propagator_global_kernel = KernelWrapper(
            system=self,
            cuda_kernel_name="compute_propagator_global",
            grid=(self.half_cuda_grid_size,),
            block=(self.cuda_block_size,),
        )

    def compute_linear_matrix(
        self, dt=None, time=None, step=None
    ) -> np.ndarray:
        """
        Computes the linear matrix used by the solver and stores it in
        self.linear_matrix. Note that this is not used directly in the
        solver loop, and so is used entirely for diagnostic purposes.

        """

        # Set to current values
        if dt is None:
            dt = getattr(self, "current_dt", self.init_dt)
        if time is None:
            time = getattr(self, "current_time", self.init_time)
        if step is None:
            step = getattr(self, "current_step", self.int(0))

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

        # Compute
        self.compute_linear_matrix_kernel(
            self.float(dt),
            self.float(time),
            self.int(step),
            linear_matrix_cupy,
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

        The eigensystem is calculated only on the solved Fourier grid. The
        returned arrays use the full Fourier grid, with padded modes set to
        zero.
        """

        if self.linear_eigensystem is not None:
            return self.linear_eigensystem

        # Handle matrix from solver
        solved_grid_mask = self.get_solved_grid_mask().astype(bool)
        linear_matrix = self.compute_linear_matrix()
        linear_matrix = linear_matrix[..., solved_grid_mask]
        linear_matrix = np.moveaxis(linear_matrix, (0, 1), (-2, -1))
        # (nfields, nfields, n_solved) -> (n_solved, nfields, nfields)

        eigvals, eigvecs = np.linalg.eig(linear_matrix)

        eigvals = (-1j * eigvals).T
        eigvecs = eigvecs.transpose(2, 1, 0)
        # (n_solved,          mode) -> (mode,          n_solved)
        # (n_solved, nfields, mode) -> (mode, nfields, n_solved)

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
            eigvecs.transpose(2, 1, 0)
        ).transpose(2, 1, 0)

        # Embed the solved eigensystem in the full Fourier grid
        eigvals_full = np.zeros(
            (self.number_of_fields, *self.half_tuple),
            dtype=self.complex,
        )
        eigvecs_full = np.zeros(
            (
                self.number_of_fields,
                self.number_of_fields,
                *self.half_tuple,
            ),
            dtype=self.complex,
        )
        eigvecs_inverse_full = np.zeros_like(eigvecs_full)

        eigvals_full[:, solved_grid_mask] = eigvals
        eigvecs_full[..., solved_grid_mask] = eigvecs
        eigvecs_inverse_full[..., solved_grid_mask] = eigvecs_inverse

        # Assign class variable
        self.linear_eigensystem = {
            "eigvals": eigvals_full,
            "eigvecs": eigvecs_full,
            "eigvecs_inverse": eigvecs_inverse_full,
        }

        return self.linear_eigensystem

    def compute_linear_propagator(
        self, dt=None, time=None, step=None
    ) -> np.ndarray:
        """
        Computes the linear propagator used by the solver and stores it in
        self.linear_propagator. Note that this is not used directly in the
        solver loop, and so is used entirely for diagnostic purposes.
        """

        # Set to current values
        if dt is None:
            dt = self.dt_max
        if time is None:
            time = getattr(self, "current_time", self.init_time)
        if step is None:
            step = getattr(self, "current_step", self.int(0))

        # Linear matrix in GPU memory
        propagator_cupy = cp.zeros(
            (
                self.number_of_fields,
                self.number_of_fields,
                self.nz,
                self.nx,
                self.half_ny,
            ),
            dtype=self.complex,
        )

        # Compute
        self.compute_propagator_global_kernel(
            self.float(dt),
            self.float(time),
            self.int(step),
            propagator_cupy,
        )

        self.linear_propagator = cp.asnumpy(propagator_cupy)

        return self.linear_propagator

    def get_solved_grid_mask(self) -> np.ndarray:
        """
        Returns a mask for the solved grid.
        """

        if not hasattr(self, "solved_grid_mask"):
            solved_grid_mask_gpu = cp.empty(
                self.half_tuple,
                dtype=self.float,
            )
            self.compute_solved_grid_mask_kernel(solved_grid_mask_gpu)
            self.solved_grid_mask = solved_grid_mask_gpu.get()

        return self.solved_grid_mask

    def get_solved_wavenumbers(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns flattened wavenumbers for the solved modes.

        Returns
        -------
        kz, kx, ky
            One-dimensional wavenumber arrays containing only solved modes.
        """

        solved_grid_mask = self.get_solved_grid_mask().astype(bool)
        kz, kx, ky = self.get_broadcast_wavenumbers()

        return (
            kz[solved_grid_mask],
            kx[solved_grid_mask],
            ky[solved_grid_mask],
        )

    def setup_initial_conditions(self) -> None:
        """
        Construct initial conditions on the solved Fourier modes.
        """

        solved_grid_mask = self.get_solved_grid_mask().astype(bool)

        # Set the initial conditions on solved modes
        self._set_initial_conditions()

        # Fill remainder with zeros
        self.fields_initial = np.reshape(
            self.fields_initial,
            (self.number_of_fields, *self.half_tuple),
        )
        self.fields_initial[:, ~solved_grid_mask] = 0

        # Check reality condition
        self._check_initial_conditions()

    def _set_initial_conditions(self) -> None:
        """Generic setup for the first time step."""

        # Use restart data if it was read
        if self.restart_manager.data is not None:
            restart_data = self.restart_manager.data

            if "fields" not in restart_data:
                raise ValueError("Restart data does not contain 'fields'.")

            field_data = restart_data["fields"]["data"]

            # TODO: remove when allowing for changing of sizes
            expected_shape = (
                self.number_of_fields,
                self.nz,
                self.nx,
                self.half_ny,
            )
            if field_data.shape != expected_shape:
                raise ValueError(
                    f"Restart data has incorrect shape: "
                    f"{field_data.shape}, "
                    f"expected: {expected_shape}"
                )

            # Set initial field data
            self.fields_initial = np.asarray(field_data)

            return

        solved_grid_mask = self.get_solved_grid_mask().astype(bool)
        number_of_solved_modes = np.count_nonzero(solved_grid_mask)

        # Handle known initialisation methods
        match self.input["init.method"]:
            case "white_noise":
                # Set random seed
                np.random.seed(self.input["init.rand_seed"])

                # Construct fields on the solved modes
                solved_fields = self.input[
                    "init.amplitude"
                ] * np.random.random(
                    (self.number_of_fields, number_of_solved_modes)
                )

            case "gaussian":
                # Construct wavenumbers for the solved modes
                kz, kx, ky = self.get_solved_wavenumbers()

                try:
                    k2 = sum(
                        {"kx": kx**2, "ky": ky**2, "kz": kz**2}[component]
                        for component in self.input["init.components"]
                    )
                except KeyError:
                    raise InvalidFlucsInputFileError(
                        "init.components entries must be one of kx, ky, or kz."
                    )

                # Envelope
                envelope = (k2 ** self.input["init.power"]) * np.exp(
                    -2.0 * (k2 / self.input["init.width"] ** 2)
                )
                envelope[k2 == 0] = 0.0

                # Phase
                phase = self.input["init.phase"]
                if phase == "random":
                    random = np.random.default_rng(self.input["init.rand_seed"])
                    angle = random.uniform(
                        0.0,
                        2.0 * np.pi,
                        size=(self.number_of_fields, number_of_solved_modes),
                    )
                else:
                    angle = self.float(phase) / (2.0 * np.pi)

                # Normalise fields to the requested amplitude
                solved_fields = (
                    envelope[None, :] * np.exp(1j * angle)
                ).astype(self.complex)

                norm = np.sqrt(np.sum(np.abs(solved_fields) ** 2))

                solved_fields *= self.input["init.amplitude"] / norm

            case _:
                raise InvalidFlucsInputFileError(
                    f"Invalid init.method: {self.input['init.method']}."
                )
                pass

        # Embed the solved modes in the full Fourier grid
        self.fields_initial = np.zeros(
            (self.number_of_fields, *self.half_tuple),
            dtype=self.complex,
        )
        self.fields_initial[:, solved_grid_mask] = solved_fields

    def _check_initial_conditions(self) -> None:
        """
        Ensures that the initial conditions satisfy the reality condition
        field[-ikz, -ikx, 0] = conj(field[ikz, ikx, 0]) for all ikx and ikz.
        """

        solved_grid_mask = self.get_solved_grid_mask().astype(bool)

        fields_initial = self.fields_initial.reshape(
            (self.number_of_fields, *self.half_tuple)
        )

        # The ky=0 modes are the ones that need to be checked
        fields_initial_ky0 = fields_initial[:, :, :, 0]
        solved_grid_mask_ky0 = solved_grid_mask[:, :, 0]

        conjugate_ikz = (-np.arange(self.nz)) % self.nz
        conjugate_ikx = (-np.arange(self.nx)) % self.nx
        conjugate_mask_ky0 = solved_grid_mask_ky0[
            conjugate_ikz[:, None], conjugate_ikx[None, :]
        ]

        # If not restarting, enforce the reality condition
        if self.restart_manager.data is None:
            # Enforce conjugate symmetry
            conjugate_fields = np.conj(
                fields_initial_ky0[
                    :,
                    conjugate_ikz[:, None],
                    conjugate_ikx[None, :],
                ]
            )
            fields_initial_ky0[:] = 0.5 * (
                fields_initial_ky0 + conjugate_fields
            )
            fields_initial_ky0[:, ~solved_grid_mask_ky0] = 0
            self.fields_initial = fields_initial

        # Calculate and report error
        conjugate_fields = np.conj(
            fields_initial_ky0[
                :,
                conjugate_ikz[:, None],
                conjugate_ikx[None, :],
            ]
        )
        error = np.nanmax(
            np.abs(fields_initial_ky0 - conjugate_fields)[
                :, solved_grid_mask_ky0
            ]
        )
        flucsprint(f"Init. condition reality error: {error:.3e}")

    def get_restart_data(self) -> dict[str, np.ndarray]:
        """
        Get the complex Fourier data for the fields at the current step.
        """

        current_fields = self.get_fields()

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

    def _compute_current_dt(self) -> bool:
        """
        Computes the current time step based on the CFL condition.
        Will be set to either 'compute_current_dt_discrete' or
        'compute_current_dt_continuous' at runtime depending on the
        value of 'time.dt_method'.

        Returns
        -------
        bool
            True if dt was changed, False if it stayed the same.
        """

    def _compute_current_dt_continuous(self) -> bool:
        """
        Computes the current time step based on the CFL condition.
        'dt_multiplier' should be used to limit the increase in the
        time step at each iteration.

        Used if 'time.dt_method' is "continuous".

        Returns
        -------
        bool
            True (time step always changes)

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

        # Continuously varying time step always changes
        return True

    def _compute_current_dt_discrete(self) -> bool:
        """
        Computes the current time step based on the CFL condition.
        'dt_multiplier' should be used to limit the increase in the
        time step at each iteration.

        Used if 'time.dt_method' is "discrete".

        Returns
        -------
        bool
            True if dt was changed, False if it stayed the same.
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

            return True

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

                return True

        # Otherwise just continue iterating with same current_dt
        else:
            self.sub_cfl_steps += 1

        return False

    def _update_dt(self) -> bool:
        """
        Updates the time step based on the CFL condition.

        Returns
        -------
        dt_changed : bool
            True if dt was changed, False if it stayed the same.

        """

        self.cfl_rate_float = self.float(cp.asnumpy(self.cfl_rate[0]))

        dt_changed = self._compute_current_dt()
        if self.current_dt < self.dt_min:
            flucsprint(
                f"({self.current_step}) Required time step "
                f"{self.current_dt:.3e} is below dt_min. Exiting."
            )
            self.solver.interrupted = True

        self.current_cfl = self.cfl_rate_float * self.current_dt
        self.adaptive_rate = self.float(1.0) / self.current_dt

        return dt_changed

    def get_fields(self, steps_before_current=0) -> cp.ndarray:
        """
        Returns the fields at the specified time step.

        Parameters
        ----------
        steps_before_current : int
            Number of steps before the current step to return.
            0 returns the current step, 1 returns the previous step, etc.

        Returns
        -------
        fields : cp.ndarray
            The fields at the specified time step.

        """

        # TODO: Add call to function to go from shearing-frame to lab-frame
        # fourier data when adding flowshear.
        # Though be careful with global vs copy

        index = (
            self.current_step - int(steps_before_current)
        ) % self.fields_history_size

        return self.fields[index]

    def get_realspace_fields_gpu(self):
        """
        Calculates the real-space fields at the current time step as a
        NumPy array. The FFTs are done on the GPU to save time, but this
        wastes some GPU memory.

        The data is saved in FourierSystem.realspace_fields

        """

        # If not None, then we have already called it this time step
        if self.realspace_fields is not None:
            return

        self.realspace_fields = cp.fft.irfftn(
            self.get_fields(),
            norm="forward",
            axes=(1, 2, 3),
            s=self.full_tuple,
        ).get()

    def get_realspace_fields_cpu(self):
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
        fields_cpu_memory: np.ndarray = self.get_fields().get()

        self.realspace_fields = np.fft.irfftn(
            fields_cpu_memory,
            norm="forward",
            axes=(1, 2, 3),
            s=self.full_tuple,
        )

    def begin_time_step(self) -> None:
        """
        Executed in the beginning of the time step.
        Can be overriden to advance any system-specific counters.

        """
        # Set this to None so that get_realspace_fields_*() knows
        # whether it has already been called. Saves some time.
        self.realspace_fields = None

    @abstractmethod
    def compute_nonlinear_terms(self, fields: cp.ndarray) -> None:
        """
        Computes the nonlinear terms for the supplied fields.

        """

    def finish_time_step(self) -> None:
        """
        Executed at the end of the time step.

        """
        pass
