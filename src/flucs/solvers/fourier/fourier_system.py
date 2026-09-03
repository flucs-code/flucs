"""
Abstract base class for a system that can be solved by FourierSolver.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable
from typing import ClassVar

import numpy as np

from flucs import cupy as cp
from flucs.diagnostic import FlucsDiagnostic
from flucs.input import InvalidFlucsInputFileError
from flucs.systems import FlucsSystem
from flucs.utilities.cupy import KernelWrapper
from flucs.utilities.dealiasing import (
    dealiased_multiplication_rfft,
    next_smooth_number,
)
from flucs.utilities.messages import flucsprint

from .fourier_system_diagnostics import (
    FourierDataDiag,
    LinearEigensystemDiag,
    RealspaceDataDiag,
)
from .fourier_system_forcing import FourierSystemForcing

if cp is not None:
    from cupy.cuda import cufft


class FourierSystem(FlucsSystem):
    """
    A generic system of equations solved using pseudospectral Fourier
    methods.
    """

    # Number of fields that the solver is solving for
    number_of_fields: int

    # Derivatives and bits used for the nonlinear terms
    number_of_dft_derivatives: int
    combine_derivatives_and_bits: bool
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
    hyperdissipation_components_kmax: np.ndarray
    adaptive_rate: float

    # CUDA kernels
    compute_linear_matrix_kernel: KernelWrapper
    compute_propagator_global_kernel: KernelWrapper
    compute_solved_grid_mask_kernel: KernelWrapper
    compute_hyperdissipation_components_kmax_kernel: KernelWrapper
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

    ###########################################################################
    # Input and configuration
    ###########################################################################

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

        # Check for the dialiasing method
        match self.input["dealiasing.method"]:
            case "two-thirds":
                if self.input["dealiasing.low_memory"]:
                    raise InvalidFlucsInputFileError(
                        "Low memory is not available for two-thirds dealiasing."
                    )
                self._setup_two_thirds_dealiasing()
            case "phase-shift":
                self._setup_phase_shift_dealiasing()
            case _:
                raise InvalidFlucsInputFileError(
                    "Invalid dealiasing method: "
                    f"{self.input['dealiasing.method']}"
                )

        # Report dealiasing information
        message = f"Using dealiasing method: {self.input['dealiasing.method']}"

        if self.input["dealiasing.method"] == "phase-shift":
            truncation = self.input["dealiasing.truncation"]
            memory = (
                ", low memory" if self.input["dealiasing.low_memory"] else ""
            )
            message += f" ({truncation}{memory})"

        if (
            self.input["dealiasing.method"],
            self.input["dealiasing.truncation"],
        ) != ("phase-shift", "polyhedral"):
            message += (
                " [equivalent unpadded grid (nz, nx, ny) = "
                f"({self.nz_unpadded}, {self.nx_unpadded}, {self.ny_unpadded})]"
            )

        flucsprint(message)

        # Assign total sizes and half sizes
        self.half_size = self.nz * self.nx * self.half_ny
        self.half_tuple = (self.nz, self.nx, self.half_ny)

        self.full_size = self.nz * self.nx * self.ny
        self.full_tuple = (self.nz, self.nx, self.ny)

        # Precompute wavenumbers (useful for many things)
        self._precompute_wavenumbers()

        # Setup forcing
        self._setup_forcing()

    # -------------------------------------------------------------------------
    # Dealiasing setup
    # -------------------------------------------------------------------------

    def _setup_two_thirds_dealiasing(self):
        self.module_options.define_flag("TWO_THIRDS_DEALIASING")

        # Set grid size
        for dim in ["z", "x", "y"]:
            n_unpadded = self.input[f"dealiasing.n{dim}_unpadded"]
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
                        (self.input["dealiasing.nonlinear_order"] + 1)
                        * half_n_unpadded,
                        primes=self.input["dealiasing.padded_primes"],
                    )

                    half_n = n // 2 + 1

                case (False, True):
                    # Given a padded_n, it's easiest to figure out half_n

                    factor = self.input["dealiasing.nonlinear_order"] + 1
                    _x = n // factor
                    half_n = n // 2 + 1

                    # Handle an annoying edge case
                    if n % factor == 0:
                        _x -= 1

                    half_n_unpadded = _x + 1
                    n_unpadded = 2 * _x + 1

                case (False, False):
                    raise ValueError(
                        f"At least one of n{dim}_unpadded and "
                        f"n{dim} must be positive!"
                    )

                # This is added only to make pyright happy.
                case _:
                    raise RuntimeError("How the fluc did you get here?")

            setattr(self, f"n{dim}_unpadded", n_unpadded)
            setattr(self, f"n{dim}", n)
            setattr(self, f"half_n{dim}_unpadded", half_n_unpadded)
            setattr(self, f"half_n{dim}", half_n)

    def _setup_phase_shift_dealiasing(self):
        self.module_options.define_flag("PHASE_SHIFT_DEALIASING")

        match self.input["dealiasing.truncation"]:
            case "polyhedral":
                if self.input["dealiasing.nonlinear_order"] != 2:
                    raise InvalidFlucsInputFileError(
                        "Polyhedral phase-shift truncation is implemented "
                        "only for quadratic nonlinearities."
                    )
                self.module_options.define_flag("PHASE_SHIFT_POLYHEDRAL")

            case "spherical":
                self.module_options.define_flag("PHASE_SHIFT_SPHERICAL")

                # Set dealiasing radius
                if self.input["dealiasing.radius_squared"] > 0:
                    radius_squared = self.input["dealiasing.radius_squared"]
                else:
                    # Set to largest multiple of 1/scale that is strictly below
                    # the theoretical limit of 2/(nonlinear_order + 1)^2
                    denominator = (
                        self.input["dealiasing.nonlinear_order"] + 1
                    ) ** 2

                    scale = 1000
                    radius_squared = ((2 * scale - 1) // denominator) / scale

                self.module_options.define_float(
                    "DEALIASING_RADIUS_SQUARED",
                    radius_squared,
                )

                dealiasing_radius = np.sqrt(radius_squared)

        # Calculate dimensions in each direction
        for dim in ["z", "x", "y"]:
            n = self.input[f"dimensions.n{dim}"]

            if n < 0:
                raise InvalidFlucsInputFileError(
                    f"Phase-shifted dimension n{dim} must be specified."
                )

            half_n = n // 2 + 1
            setattr(self, f"n{dim}", n)
            setattr(self, f"half_n{dim}", half_n)

            if self.input["dealiasing.truncation"] == "spherical":
                # Find equivalent unpadded (handle edge case for n=1)
                half_n_unpadded = max(
                    1,
                    int(half_n * (2 * dealiasing_radius)),
                )
                n_unpadded = 2 * half_n_unpadded - 1

                setattr(self, f"n{dim}_unpadded", n_unpadded)
                setattr(self, f"half_n{dim}_unpadded", half_n_unpadded)
            else:
                setattr(self, f"n{dim}_unpadded", n)
                setattr(self, f"half_n{dim}_unpadded", half_n)

    # -------------------------------------------------------------------------
    # Wavenumber grids
    # -------------------------------------------------------------------------

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

    # -------------------------------------------------------------------------
    # Forcing setup
    # -------------------------------------------------------------------------

    def _setup_forcing(self):
        """
        Sets up the forcing method.
        """
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

    ###########################################################################
    # Setup
    ###########################################################################

    # -------------------------------------------------------------------------
    # General
    # -------------------------------------------------------------------------

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

    def _allocate_memory(self) -> None:
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

    # -------------------------------------------------------------------------
    # CUDA setup
    # -------------------------------------------------------------------------

    def setup_cuda_definitions(self) -> None:
        """
        Adds any general CUDA definitions. Additional ones may be added after
        executing the initialisation kernels.
        """
        # Layouts
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

        # Sizes
        self.module_options.define_dimension("HALFSIZE", self.half_size)
        self.module_options.define_dimension("FULLSIZE", self.full_size)
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

    def register_initialisation_kernels(self) -> None:
        """
        Registers any initialisation kernels required for computing compile-time
        constants.
        """

        # Normalised hyperdissipation
        if not any(
            self.input[f"hyperdissipation.{component}"] > 0.0
            and self.input[f"hyperdissipation.{component}_normalised"]
            for component in self.hyperdissipation_components
        ):
            return

        self.compute_hyperdissipation_components_kmax_kernel = KernelWrapper(
            system=self,
            cuda_kernel_name="compute_hyperdissipation_components_kmax",
            grid=(1,),
            block=(1,),
        )

    def execute_initialisation_kernels(self) -> None:
        """
        Executes any initialisation kernels
        """

        # Hyperdissipation normalisation
        hyperdissipation_components_kmax = cp.empty(4, dtype=self.float)

        self.compute_hyperdissipation_components_kmax_kernel(
            hyperdissipation_components_kmax
        )
        self.hyperdissipation_components_kmax = (
            hyperdissipation_components_kmax.get()
        )

        # Cleanup kernels (no longer required after initialisation)
        del self.compute_hyperdissipation_components_kmax_kernel

    def setup_cuda_definitions_init(self) -> None:
        """
        Adds any CUDA definitions that were computed during the
        initialisation kernels.
        """

        # Hyperdissipation normalisation
        for index, component in enumerate(self.hyperdissipation_components):
            if (self.input[f"hyperdissipation.{component}"] <= 0.0) or not (
                self.input[f"hyperdissipation.{component}_normalised"]
            ):
                continue

            self.module_options.define_flag(
                f"HYPERDISSIPATION_{component.upper()}_NORMALISED"
            )
            self.module_options.define_float(
                f"HYPERDISSIPATION_{component.upper()}_KMAX",
                self.hyperdissipation_components_kmax[index],
            )

    def register_kernels(self) -> None:
        """
        Registers the CUDA kernels.
        """

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

        # Dealiasing error-checking
        if self.input["dealiasing.check_errors"]:

            def create_first_intermediates(
                current_dt,
                current_time,
                current_step: int,
                input: cp.ndarray,
                memory_dict: dict,
            ) -> None:
                memory_dict["first_intermediates_fourier"][:] = input[:]

            def create_second_intermediates(
                current_dt,
                current_time,
                current_step: int,
                calculate_cfl: bool,
                memory_dict: dict,
            ) -> None:

                memory_dict["second_intermediates_real"][:] = (
                    memory_dict["first_intermediates_real"][0, :]
                    * memory_dict["first_intermediates_real"][1, :]
                ) / self.full_size

            (
                self.check_dealiasing_errors_operation,
                self.check_dealiasing_errors_output,
            ) = self.create_dealiased_operation(
                n_in=2,
                n_out=1,
                create_first_intermediates=create_first_intermediates,
                create_second_intermediates=create_second_intermediates,
                combine_first_and_second_intermediates=True,
            )

    # -------------------------------------------------------------------------
    # Dealiased operations
    # -------------------------------------------------------------------------
    def create_dealiased_operation(
        self,
        n_in: int,
        n_out: int,
        create_first_intermediates: Callable,
        create_second_intermediates: Callable,
        allocate_additional_memory: Callable | None = None,
        combine_first_and_second_intermediates: bool = True,
    ) -> tuple[Callable, cp.ndarray]:
        """
        Builds a dealiased Fourier-to-real-to-Fourier operation.

        create_first_intermediates constructs n_in Fourier-space intermediate
        arrays from the supplied input. After inverse FFTs,
        create_second_intermediates constructs n_out real-space products.
        Forward FFTs place the result in the output array.

        Parameters
        ----------
        n_in : int
            Number of first intermediates

        n_out : int
            Number of second intermediates

        create_first_intermediates : Callable
            Callable to computes the first intermediates with signature
                (dt, time, step, input_array, memory_dict)
            where memory_dict is a dictionary of CuPy arrays.

        create_second_intermediates : Callable
            Callable to computes the second intermediates with signature
                (dt, time, step, calculate_cfl, memory_dict)
            where memory_dict is a dictionary of CuPy arrays.

        allocate_additional_memory : Callable[[], dict] | None
            Callable to allocate any additional memory required
            by first_ or second_intermediates

        combine_first_and_second_intermediates : bool
            Optional flag to combine the arrays of first and second
            intermediates to save memory with no performance hit.

        Returns
        -------
        dealiased_operation : Callable
            The dealiased operation, which is a Callable of signature
                (dt, time, step, input_array, calculate_cfl)
            and no return value.

        output : cp.ndarray
            The output array of the dealiased_operation Callable.

        """

        # Despatch to the correct version of create_dealiased_operation
        if self.input["dealiasing.method"] == "two-thirds":
            return self.create_dealiased_operation_two_thirds(
                n_in=n_in,
                n_out=n_out,
                create_first_intermediates=create_first_intermediates,
                create_second_intermediates=create_second_intermediates,
                allocate_additional_memory=allocate_additional_memory,
                combine_first_and_second_intermediates=combine_first_and_second_intermediates,
            )
        else:
            if self.input["dealiasing.low_memory"]:
                return self.create_dealiased_operation_phase_shift_low_memory(
                    n_in=n_in,
                    n_out=n_out,
                    create_first_intermediates=create_first_intermediates,
                    create_second_intermediates=create_second_intermediates,
                    allocate_additional_memory=allocate_additional_memory,
                    combine_first_and_second_intermediates=combine_first_and_second_intermediates,
                )
            else:
                return self.create_dealiased_operation_phase_shift(
                    n_in=n_in,
                    n_out=n_out,
                    create_first_intermediates=create_first_intermediates,
                    create_second_intermediates=create_second_intermediates,
                    allocate_additional_memory=allocate_additional_memory,
                    combine_first_and_second_intermediates=combine_first_and_second_intermediates,
                )

    def create_dealiased_operation_two_thirds(
        self,
        n_in: int,
        n_out: int,
        create_first_intermediates: Callable,
        create_second_intermediates: Callable,
        allocate_additional_memory: Callable,
        combine_first_and_second_intermediates: bool = True,
    ) -> tuple[Callable, cp.ndarray]:
        """
        See create_dealiased_operation.

        This sets up operations for two-thirds dealiasing where
        the intermediates are created once in arrays with
        appropriate zero padding.

        """
        # Create the cuFFT plans for the forward and backward transforms
        plan_c2r = self.create_standard_real_cufft_plan(
            fft_type="c2r",
            batch_size=n_in,
        )

        plan_r2c = self.create_standard_real_cufft_plan(
            fft_type="r2c",
            batch_size=n_out,
        )

        # Allocate the memory required by the intermediates
        if combine_first_and_second_intermediates:
            combined_size = max(n_in, n_out)

            first_intermediates_fourier = cp.zeros(
                (combined_size, *self.half_tuple),
                dtype=self.complex,
            )
            first_intermediates_real = cp.zeros(
                (combined_size, *self.full_tuple),
                dtype=self.float,
            )

            # Define them with their proper sizes
            first_intermediates_fourier = cp.ndarray(
                (n_in, *self.half_tuple),
                dtype=self.complex,
                memptr=first_intermediates_fourier.data,
            )
            first_intermediates_real = cp.ndarray(
                (n_in, *self.full_tuple),
                dtype=self.float,
                memptr=first_intermediates_real.data,
            )

            second_intermediates_fourier = cp.ndarray(
                (n_out, *self.half_tuple),
                dtype=self.complex,
                memptr=first_intermediates_fourier.data,
            )
            second_intermediates_real = cp.ndarray(
                (n_out, *self.full_tuple),
                dtype=self.float,
                memptr=first_intermediates_real.data,
            )

        else:
            first_intermediates_fourier = cp.zeros(
                (n_in, *self.half_tuple),
                dtype=self.complex,
            )
            first_intermediates_real = cp.zeros(
                (n_in, *self.full_tuple),
                dtype=self.float,
            )

            second_intermediates_fourier = cp.zeros(
                (n_out, *self.half_tuple),
                dtype=self.complex,
            )
            second_intermediates_real = cp.zeros(
                (n_out, *self.full_tuple),
                dtype=self.float,
            )

        memory_dict = {
            "first_intermediates_fourier": first_intermediates_fourier,
            "first_intermediates_real": first_intermediates_real,
            "second_intermediates_fourier": second_intermediates_fourier,
            "second_intermediates_real": second_intermediates_real,
        }

        # Add any additional, user-defined memory
        if allocate_additional_memory is not None:
            memory_dict.update(allocate_additional_memory())

        # Create the dealiased_operation
        def dealiased_operation(
            current_dt, current_time, current_step, input_array, calculate_cfl
        ):

            # Input Fourier fields -> first Fourier intermediate quantities
            create_first_intermediates(
                current_dt,
                current_time,
                current_step,
                input_array,
                memory_dict,
            )

            # Fourier intermediates -> real-space intermediates
            plan_c2r.fft(
                first_intermediates_fourier,
                first_intermediates_real,
                cufft.CUFFT_INVERSE,
            )

            # Real-space first intermediates
            #   -> real-space second intermediates
            create_second_intermediates(
                current_dt,
                current_time,
                current_step,
                calculate_cfl,
                memory_dict,
            )

            # Real-space intermediates -> Fourier intermediates
            plan_r2c.fft(
                second_intermediates_real,
                second_intermediates_fourier,
                cufft.CUFFT_INVERSE,
            )

        return dealiased_operation, second_intermediates_fourier

    def create_dealiased_operation_phase_shift(
        self,
        n_in: int,
        n_out: int,
        create_first_intermediates: Callable,
        create_second_intermediates: Callable,
        allocate_additional_memory: Callable,
        combine_first_and_second_intermediates: bool = False,  # ignored here
    ) -> tuple[Callable, cp.ndarray]:
        """
        See create_dealiased_operation.

        This sets up operations for phase-shift dealiasing where the
        intermediates are created twice with an isotropic phase-shift
        between them.

        This version batches the FFTs of both shifted and unshifted
        intermediates for best performance but worst memory impact.

        combine_first_and_second_intermediates is ignored as it is not
        possible when batching the FFTs unless n_in = n_out.

        """
        # Create the cuFFT plans for the forward and backward transforms
        plan_c2r = self.create_standard_real_cufft_plan(
            fft_type="c2r",
            batch_size=2 * n_in,
        )

        plan_r2c = self.create_standard_real_cufft_plan(
            fft_type="r2c",
            batch_size=2 * n_out,
        )

        # Allocate memory for batched FFTs for both shifted and unshifted data
        first_intermediates_fourier = cp.zeros(
            (2 * n_in, *self.half_tuple),
            dtype=self.complex,
        )
        first_intermediates_real = cp.zeros(
            (2 * n_in, *self.full_tuple),
            dtype=self.float,
        )
        # Assign subarrays accordingly
        unshifted_first_intermediates_fourier = cp.ndarray(
            shape=(n_in, *self.half_tuple),
            dtype=self.complex,
            memptr=first_intermediates_fourier[0].data,
        )
        shifted_first_intermediates_fourier = cp.ndarray(
            shape=(n_in, *self.half_tuple),
            dtype=self.complex,
            memptr=first_intermediates_fourier[n_in].data,
        )

        unshifted_first_intermediates_real = cp.ndarray(
            shape=(n_in, *self.full_tuple),
            dtype=self.float,
            memptr=first_intermediates_real[0].data,
        )
        shifted_first_intermediates_real = cp.ndarray(
            shape=(n_in, *self.full_tuple),
            dtype=self.float,
            memptr=first_intermediates_real[n_in].data,
        )

        # Can combine and keep the FFTs batched iff n_in = n_out
        if combine_first_and_second_intermediates and n_in == n_out:
            second_intermediates_fourier = first_intermediates_fourier
            second_intermediates_real = first_intermediates_real
        else:
            second_intermediates_fourier = cp.zeros(
                (2 * n_out, *self.half_tuple),
                dtype=self.complex,
            )
            second_intermediates_real = cp.zeros(
                (2 * n_out, *self.full_tuple),
                dtype=self.float,
            )

        unshifted_second_intermediates_fourier = cp.ndarray(
            shape=(n_out, *self.half_tuple),
            dtype=self.complex,
            memptr=second_intermediates_fourier[0].data,
        )
        shifted_second_intermediates_fourier = cp.ndarray(
            shape=(n_out, *self.half_tuple),
            dtype=self.complex,
            memptr=second_intermediates_fourier[n_out].data,
        )

        unshifted_second_intermediates_real = cp.ndarray(
            shape=(n_out, *self.full_tuple),
            dtype=self.float,
            memptr=second_intermediates_real[0].data,
        )
        shifted_second_intermediates_real = cp.ndarray(
            shape=(n_out, *self.full_tuple),
            dtype=self.float,
            memptr=second_intermediates_real[n_out].data,
        )

        unshifted_memory_dict = {
            "first_intermediates_fourier": unshifted_first_intermediates_fourier,  # noqa: E501
            "first_intermediates_real": unshifted_first_intermediates_real,
            "second_intermediates_fourier": unshifted_second_intermediates_fourier,  # noqa: E501
            "second_intermediates_real": unshifted_second_intermediates_real,
        }

        shifted_memory_dict = {
            "first_intermediates_fourier": shifted_first_intermediates_fourier,
            "first_intermediates_real": shifted_first_intermediates_real,
            "second_intermediates_fourier": shifted_second_intermediates_fourier,  # noqa: E501
            "second_intermediates_real": shifted_second_intermediates_real,
        }

        # Add any additional, user-defined memory
        if allocate_additional_memory is not None:
            # Separate memory for shifted and unshifted.
            #
            # It is up to the user whether allocate_additional_memory
            # returns the same arrays or allocates new
            # ones every time.
            unshifted_memory_dict.update(allocate_additional_memory())
            shifted_memory_dict.update(allocate_additional_memory())

        # Phase-shifting kernels
        add_phase_factors_kernel = KernelWrapper(
            system=self,
            cuda_kernel_name=f"add_phase_factors<{n_in}>",
            grid=(self.half_cuda_grid_size,),
            block=(self.cuda_block_size,),
        )
        undo_phase_factors_kernel = KernelWrapper(
            system=self,
            cuda_kernel_name=f"undo_phase_factors<{n_out}>",
            grid=(self.half_cuda_grid_size,),
            block=(self.cuda_block_size,),
        )

        # Create the dealiased_operation
        def dealiased_operation(
            current_dt, current_time, current_step, input_array, calculate_cfl
        ):
            # Input Fourier fields -> first Fourier intermediate quantities
            create_first_intermediates(
                current_dt,
                current_time,
                current_step,
                input_array,
                unshifted_memory_dict,
            )

            # Create phase-shifted copies
            add_phase_factors_kernel(
                unshifted_first_intermediates_fourier,
                shifted_first_intermediates_fourier,
            )

            # Fourier intermediates -> real-space intermediates
            # Both shifted and unshifted are done here
            plan_c2r.fft(
                first_intermediates_fourier,
                first_intermediates_real,
                cufft.CUFFT_INVERSE,
            )

            # Real-space first intermediates
            #   -> real-space second intermediates

            # 1. unshifted
            create_second_intermediates(
                current_dt,
                current_time,
                current_step,
                calculate_cfl,
                unshifted_memory_dict,
            )

            # 2. shifted
            create_second_intermediates(
                current_dt,
                current_time,
                current_step,
                False,
                shifted_memory_dict,
            )

            # Real-space intermediates -> Fourier intermediates
            # Both shifted and unshifted are done here
            plan_r2c.fft(
                second_intermediates_real,
                second_intermediates_fourier,
                cufft.CUFFT_INVERSE,
            )

            # Shifted and unshifted Fourier outputs
            #   -> dealiased output
            # stored in unshifted_second_intermediates_fourier
            undo_phase_factors_kernel(
                unshifted_second_intermediates_fourier,
                shifted_second_intermediates_fourier,
            )

        return dealiased_operation, unshifted_second_intermediates_fourier

    def create_dealiased_operation_phase_shift_low_memory(
        self,
        n_in: int,
        n_out: int,
        create_first_intermediates: Callable,
        create_second_intermediates: Callable,
        allocate_additional_memory: Callable,
        combine_first_and_second_intermediates: bool = True,
    ) -> tuple[Callable, cp.ndarray]:
        """
        See create_dealiased_operation.

        This sets up operations for phase-shift dealiasing where the
        intermediates are created twice with an isotropic phase-shift
        between them.

        This version calculates the shifted and unshifted consecutively
        and reuses memory to reduce its memory footprint for the cost
        of a small performance hit.

        """
        # Create the cuFFT plans for the forward and backward transforms
        plan_c2r = self.create_standard_real_cufft_plan(
            fft_type="c2r",
            batch_size=n_in,
        )

        plan_r2c = self.create_standard_real_cufft_plan(
            fft_type="r2c",
            batch_size=n_out,
        )

        # Allocate the memory required by the intermediates
        if combine_first_and_second_intermediates:
            combined_size = max(n_in, n_out)

            first_intermediates_fourier = cp.zeros(
                (combined_size, *self.half_tuple),
                dtype=self.complex,
            )
            first_intermediates_real = cp.zeros(
                (combined_size, *self.full_tuple),
                dtype=self.float,
            )

            # Define them with their proper sizes
            first_intermediates_fourier = cp.ndarray(
                (n_in, *self.half_tuple),
                dtype=self.complex,
                memptr=first_intermediates_fourier.data,
            )
            first_intermediates_real = cp.ndarray(
                (n_in, *self.full_tuple),
                dtype=self.float,
                memptr=first_intermediates_real.data,
            )

            second_intermediates_fourier = cp.ndarray(
                (n_out, *self.half_tuple),
                dtype=self.complex,
                memptr=first_intermediates_fourier.data,
            )
            second_intermediates_real = cp.ndarray(
                (n_out, *self.full_tuple),
                dtype=self.float,
                memptr=first_intermediates_real.data,
            )

        else:
            first_intermediates_fourier = cp.zeros(
                (n_in, *self.half_tuple),
                dtype=self.complex,
            )
            first_intermediates_real = cp.zeros(
                (n_in, *self.full_tuple),
                dtype=self.float,
            )

            second_intermediates_fourier = cp.zeros(
                (n_out, *self.half_tuple),
                dtype=self.complex,
            )
            second_intermediates_real = cp.zeros(
                (n_out, *self.full_tuple),
                dtype=self.float,
            )

        memory_dict = {
            "first_intermediates_fourier": first_intermediates_fourier,
            "first_intermediates_real": first_intermediates_real,
            "second_intermediates_fourier": second_intermediates_fourier,
            "second_intermediates_real": second_intermediates_real,
        }

        # Add any additional, user-defined memory
        if allocate_additional_memory is not None:
            memory_dict.update(allocate_additional_memory())

        # Shifted operations run after unshifted ones are reuse all
        # memory from unshifted apart from the final output array
        shifted_memory_dict = {}
        shifted_memory_dict.update(memory_dict)

        shifted_second_intermediates_fourier = cp.zeros(
            (n_out, *self.half_tuple),
            dtype=self.complex,
        )

        shifted_memory_dict["second_intermediates_fourier"] = (
            shifted_second_intermediates_fourier
        )

        # Phase-shifting kernels
        add_phase_factors_kernel = KernelWrapper(
            system=self,
            cuda_kernel_name=f"add_phase_factors<{n_in}>",
            grid=(self.half_cuda_grid_size,),
            block=(self.cuda_block_size,),
        )
        undo_phase_factors_kernel = KernelWrapper(
            system=self,
            cuda_kernel_name=f"undo_phase_factors<{n_out}>",
            grid=(self.half_cuda_grid_size,),
            block=(self.cuda_block_size,),
        )

        # Create the dealiased_operation
        def dealiased_operation(
            current_dt, current_time, current_step, input_array, calculate_cfl
        ):

            # First do shifted...

            # Input Fourier fields -> first Fourier intermediate quantities
            create_first_intermediates(
                current_dt,
                current_time,
                current_step,
                input_array,
                shifted_memory_dict,
            )

            # Phase shift
            add_phase_factors_kernel(
                first_intermediates_fourier, first_intermediates_fourier
            )

            # Fourier intermediates -> real-space intermediates
            plan_c2r.fft(
                first_intermediates_fourier,
                first_intermediates_real,
                cufft.CUFFT_INVERSE,
            )

            # Real-space first intermediates
            #   -> real-space second intermediates
            create_second_intermediates(
                current_dt,
                current_time,
                current_step,
                False,
                shifted_memory_dict,
            )

            # Real-space intermediates -> Fourier intermediates
            plan_r2c.fft(
                second_intermediates_real,
                shifted_second_intermediates_fourier,
                cufft.CUFFT_INVERSE,
            )

            # ... then unshifted

            # Fourier intermediates -> real-space intermediates
            create_first_intermediates(
                current_dt,
                current_time,
                current_step,
                input_array,
                memory_dict,
            )

            # Fourier intermediates -> real-space intermediates
            plan_c2r.fft(
                first_intermediates_fourier,
                first_intermediates_real,
                cufft.CUFFT_INVERSE,
            )

            # Real-space first intermediates
            #   -> real-space second intermediates
            create_second_intermediates(
                current_dt,
                current_time,
                current_step,
                calculate_cfl,
                memory_dict,
            )

            # Real-space intermediates -> Fourier intermediates
            plan_r2c.fft(
                second_intermediates_real,
                second_intermediates_fourier,
                cufft.CUFFT_INVERSE,
            )

            # Shifted and unshifted Fourier outputs
            #   -> dealiased output
            # stored in second_intermediates_fourier
            undo_phase_factors_kernel(
                second_intermediates_fourier,
                shifted_second_intermediates_fourier,
            )

        return dealiased_operation, second_intermediates_fourier

    def create_standard_real_cufft_plan(self, fft_type: str, batch_size: int):
        """
        Create a reusable batched 3D real cuFFT plan for the FourierSystem grid.

        The plan transforms contiguous arrays with a leading batch dimension:

            c2r: (batch, nz, nx, half_ny) -> (batch, nz, nx, ny)
            r2c: (batch, nz, nx, ny)      -> (batch, nz, nx, half_ny)

        cuFFT transforms are unnormalised; callers must apply any required
        normalisation separately.

        Parameters
        ----------
        fft_type : {"c2r", "r2c"}
            Direction and real/complex layout of the transform.
        batch_size : int
            Number of independent 3D transforms performed by the plan.

        Returns
        -------
        cupy.cuda.cufft.PlanNd
            A reusable plan for the requested batched transform.
        """

        # Sizes
        nz = self.nz
        nx = self.nx
        ny = self.ny
        half_ny = self.half_ny

        # Shapes and tuples for the transform
        shape = (nz, nx, ny)
        istride = 1
        ostride = 1
        compex_embed = (1, nx, half_ny)
        compex_dist = nz * nx * half_ny
        real_embed = (1, nx, ny)
        real_dist = nz * nx * ny

        # Complex to real
        if fft_type == "c2r":
            inembed = compex_embed
            onembed = real_embed
            idist = compex_dist
            odist = real_dist
            fft_type = self.fft_c2r_plan_type
            last_size = ny

        # Real to complex
        elif fft_type == "r2c":
            inembed = real_embed
            onembed = compex_embed
            idist = real_dist
            odist = compex_dist
            fft_type = self.fft_r2c_plan_type
            last_size = half_ny
        else:
            raise ValueError("fft_type must be c2r or r2c.")

        # Create plan
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

    # -------------------------------------------------------------------------
    # Initial conditions
    # -------------------------------------------------------------------------

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
        """
        Generic setup for the first time step.
        """

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
                solved_fields = self.input["init.amplitude"] * np.random.random(
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
                solved_fields = (envelope[None, :] * np.exp(1j * angle)).astype(
                    self.complex
                )

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
        # conjugate_mask_ky0 = solved_grid_mask_ky0[
        #     conjugate_ikz[:, None], conjugate_ikx[None, :]
        # ]

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

    # -------------------------------------------------------------------------
    # Health checks
    # -------------------------------------------------------------------------

    def check_health(self) -> None:
        """
        Basic consistency/health checks before running.
        Alerts the user if anything needs their attention.
        """

        self._check_dealiasing()
        self._check_linear_matrix()

    def _check_dealiasing(self) -> None:
        # Find all solved modes
        all_solved_modes = self.get_number_of_solved_modes(
            only_nonnegative_ky=False
        )
        solved_fraction = 100 * (all_solved_modes / self.full_size)

        print(
            f"Total number of solved modes: {all_solved_modes} / "
            f"{self.full_size} ({solved_fraction:.2f}%)"
        )

        if not self.input["dealiasing.check_errors"]:
            return

        # TODO a test of the dealiasing boundaries should be written, in which
        # the padded values or radius are changed, and all of this code moved
        # into said function.
        # This should be both for two-thirds and phase-shifted dealiasing.

        solved_modes_mask = self.get_solved_grid_mask()
        solved_modes_mask = cp.array(solved_modes_mask)

        nx = self.nx
        ny = self.ny
        nz = self.nz

        input_array = cp.random.rand(2, nz, nx, ny, dtype=self.float)
        input_array_rfft = cp.fft.rfftn(input_array, norm="forward")
        array1_rfft = input_array_rfft[0]
        array2_rfft = input_array_rfft[1]

        array1_rfft[solved_modes_mask < 0.5] = 0
        array2_rfft[solved_modes_mask < 0.5] = 0

        self.check_dealiasing_errors_operation(
            current_dt=0,
            current_time=0,
            current_step=0,
            input_array=input_array_rfft,
            calculate_cfl=False,
        )
        product_operation = self.check_dealiasing_errors_output[0]
        product_operation[solved_modes_mask < 0.5] = 0

        product_dealiased_rfft = dealiased_multiplication_rfft(
            array1_rfft,
            array2_rfft,
            nx=nx,
            ny=ny,
            nz=nz,
            padded_nx=2 * nx,
            padded_ny=2 * ny,
            padded_nz=2 * nz,
        )
        product_dealiased_rfft[solved_modes_mask < 0.5] = 0

        print(
            "Max abs error is ",
            cp.max(cp.abs(product_operation - product_dealiased_rfft)),
        )

        del self.check_dealiasing_errors_operation
        del self.check_dealiasing_errors_output

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

        # Evaluate the accuracy of the CUDA Pade propagator
        propagator = self.compute_linear_propagator(dt=self.dt_max)
        propagator = propagator[..., solved_grid_mask]
        propagator = np.moveaxis(propagator, (0, 1), (-2, -1))

        pade_eigvals = np.linalg.eigvals(propagator).T
        exact_eigvals = np.exp(-1j * self.dt_max * eigvals)

        # Pair each numerical eigenvalue with the nearest exact eigenvalue
        grid_indices = np.arange(pade_eigvals.shape[1])
        pade_eigvals = np.stack(
            [
                pade_eigvals[
                    np.argmin(np.abs(pade_eigvals - exact_eigval), axis=0),
                    grid_indices,
                ]
                for exact_eigval in exact_eigvals
            ]
        )

        # Calculate and report errors
        abs_errors = np.abs(
            1j * np.log(pade_eigvals / exact_eigvals) / self.dt_max
        )

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

    ###########################################################################
    # Solver execution
    ###########################################################################

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

    # -------------------------------------------------------------------------
    # Loop functions
    # -------------------------------------------------------------------------

    def begin_time_step(self) -> None:
        """
        Executed in the beginning of the time step.
        Can be overriden to advance any system-specific counters.

        """
        # Set this to None so that get_realspace_fields_*() knows
        # whether it has already been called. Saves some time.
        self.realspace_fields = None

    @abstractmethod
    def compute_nonlinear_terms(
        self,
        current_dt,
        current_time,
        current_step,
        fields: cp.ndarray,
        calculate_cfl: bool,
    ) -> None:
        """
        Computes the nonlinear terms for the supplied fields.

        """

    def finish_time_step(self) -> None:
        """
        Executed at the end of the time step.

        """
        pass

    # -------------------------------------------------------------------------
    # Time step control
    # -------------------------------------------------------------------------

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

    ###########################################################################
    # Helper functions
    ###########################################################################

    # -------------------------------------------------------------------------
    # Wavenumbers
    # -------------------------------------------------------------------------

    def get_broadcast_wavenumbers(self):
        """
        Returns wavenumber arrays broadcast to (nz, nx, half_ny)

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

    def get_number_of_solved_modes(
        self, only_nonnegative_ky: bool = True
    ) -> int:
        """
        Finds the number of all solved Fourier modes.

        Parameters
        ----------
        only_nonnegative_ky : bool
            If true, takes into account only the modes with nonnegative ky,
            i.e., those that are directly solved for in the Fourier arrays.

        Returns
        -------
        number_of_solved_modes: int
            The number of solved modes
        """

        solved_grid_mask = self.get_solved_grid_mask()
        number_of_nonnegative_ky = np.count_nonzero(solved_grid_mask > 0.5)

        if only_nonnegative_ky:
            return number_of_nonnegative_ky

        number_of_zero_ky = np.count_nonzero(solved_grid_mask[:, :, 0] > 0.5)

        return 2 * number_of_nonnegative_ky - number_of_zero_ky

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

    # -------------------------------------------------------------------------
    # Fields
    # -------------------------------------------------------------------------

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

    # -------------------------------------------------------------------------
    # Linear matrix
    # -------------------------------------------------------------------------

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
        eigvecs_inverse = np.linalg.inv(eigvecs.transpose(2, 1, 0)).transpose(
            2, 1, 0
        )

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
