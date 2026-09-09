from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from flucs import cupy as cp
from flucs.input import InvalidFlucsInputFileError
from flucs.utilities.cupy import KernelWrapper
from flucs.utilities.messages import flucsprint

if TYPE_CHECKING:
    from flucs.solvers.fourier.fourier_system import FourierSystem


class FourierSystemForcing(ABC):
    """
    Base class for optional forcing methods used by FourierSystem solvers.
    """

    linear: bool
    explicit: bool
    forced_mode_count: int
    forcing_range_mask: np.ndarray
    below_forcing_range_mask: np.ndarray
    above_forcing_range_mask: np.ndarray

    def __init__(self, system: FourierSystem):
        self.system = system

    @abstractmethod
    def setup_cuda_definitions(self) -> None:
        pass

    def register_kernels(self) -> None:
        """
        Registers any forcing-specific CUDA kernels.
        """
        pass

    def ready(self) -> None:
        """
        Initialises any forcing state immediately before solver execution.
        """
        pass

    def prepare_time_step(self) -> None:
        """
        Updates any forcing state before executing a complete timestep.
        """
        pass

    def _calculate_forced_mode_count(self) -> None:
        """
        Calculates the number of modes that are being forced
        """

        # Get ky wavenumbers
        _, _, ky = self.system.get_broadcast_wavenumbers()

        # Number of ky=0 modes
        ky0_modes = ky < 0.5 * ky[0, 0, 1]

        # Number of forced modes
        forced_mode_count = 2 * np.sum(self.forcing_range_mask) - np.sum(
            self.forcing_range_mask & ky0_modes
        )

        if forced_mode_count == 0:
            raise InvalidFlucsInputFileError(
                "No modes are being forced. Please check the specified "
                "forcing range."
            )

        # Set number of forced modes
        self.forced_mode_count = int(forced_mode_count)

    def setup_forcing_range_kzkperp(self) -> None:
        """
        Determines the range of wavenumbers to be forced based on the input
        parameters, and calculates the number of modes in the kzperp range.
        """

        # Alias system
        system = self.system

        # Validate ranges
        range_kperp = self.system.input["forcing.range_kperp"]
        range_kz = self.system.input["forcing.range_kz"]

        if len(range_kperp) != 2:
            raise InvalidFlucsInputFileError(
                "forcing.range_kperp must be a list [kperp_min, kperp_max]."
            )

        if len(range_kz) != 2:
            raise InvalidFlucsInputFileError(
                "forcing.range_kz must be a list [kz_min, kz_max]."
            )

        kperp_min = range_kperp[0]
        kperp_max = range_kperp[1]
        if kperp_max < kperp_min:
            raise InvalidFlucsInputFileError(
                "forcing.kperp_max must be larger than forcing.kperp_min."
            )

        kz_min = range_kz[0]
        kz_max = range_kz[1]
        if kz_max < kz_min:
            raise InvalidFlucsInputFileError(
                "forcing.kz_max must be larger than forcing.kz_min."
            )

        # Add module options
        system.module_options.define_flag("FORCING_RANGE_KZKPERP")
        system.module_options.define_float("FORCING_KPERP2_MIN", kperp_min**2)
        system.module_options.define_float("FORCING_KPERP2_MAX", kperp_max**2)
        system.module_options.define_float("FORCING_KZ_MIN", kz_min)
        system.module_options.define_float("FORCING_KZ_MAX", kz_max)

        # Determine forcing range
        system._precompute_wavenumbers()
        kz, kx, ky = system.get_broadcast_wavenumbers()
        kperp2 = kx**2 + ky**2
        kz_abs = np.abs(kz)

        self.forcing_range_mask = (
            (kperp2 > kperp_min**2)
            & (kperp2 < kperp_max**2)
            & (kz_abs > kz_min)
            & (kz_abs < kz_max)
        )

        # Other useful masks
        self.below_forcing_range_mask = kperp2 >= kperp_max**2
        self.above_forcing_range_mask = kperp2 <= kperp_min**2

        # Calculate the number of forced modes
        self._calculate_forced_mode_count()

        flucsprint(
            f"Forcing applied on a total of {self.forced_mode_count} modes.",
            source=self,
        )

    def setup_forcing_range_kmod(self) -> None:
        """
        Determines the range of total wavenumbers to be forced and calculates
        the number of modes in the kmod range.
        """

        # Alias system
        system = self.system

        # Validate range
        range_kmod = system.input["forcing.range_kmod"]
        if len(range_kmod) != 2:
            raise InvalidFlucsInputFileError(
                "forcing.range_kmod must be a list [kmod_min, kmod_max]."
            )

        kmod_min = range_kmod[0]
        kmod_max = range_kmod[1]
        if kmod_max < kmod_min:
            raise InvalidFlucsInputFileError(
                "forcing.kmod_max must be larger than forcing.kmod_min."
            )

        # Add module options
        system.module_options.define_flag("FORCING_RANGE_KMOD")
        system.module_options.define_float("FORCING_KMOD2_MIN", kmod_min**2)
        system.module_options.define_float("FORCING_KMOD2_MAX", kmod_max**2)

        # Determine forcing range
        kz, kx, ky = system.get_broadcast_wavenumbers()
        kmod2 = kz**2 + kx**2 + ky**2

        self.forcing_range_mask = (kmod2 > kmod_min**2) & (kmod2 < kmod_max**2)

        # Other useful masks
        self.below_forcing_range_mask = kmod2 >= kmod_max**2
        self.above_forcing_range_mask = kmod2 <= kmod_min**2

        # Calculate the number of forced modes
        self._calculate_forced_mode_count()

        flucsprint(
            f"Forcing applied on a total of {self.forced_mode_count} modes.",
            source=self,
        )


class FourierOrnsteinUhlenbeckForcing(FourierSystemForcing):
    """
    Additive finite-correlation-time forcing for Fourier systems. This advances 
    the fields via the standard Ornstein-Uhlenbeck process:

        X_{n+1} = X_n * exp(-dt / corr_time) 
                  + sqrt(1 - exp(- dt / corr_time)**2) 
                  * (amplitude) * (unit varianvce Gaussian random number)

    Parameters
    ----------
    amplitude : list[float]
        Stationary RMS forcing amplitude for each field, summed over all
        forced physical modes.
    corr_time : float
        Temporal correlation time of the forcing.
    range_kmod : list[float, float]
        Range of total wavenumbers to force. If non-empty, this takes
        precedence over range_kz and range_kperp.
    range_kperp : list[float, float]
        Range of perpendicular wavenumbers to force when range_kmod is empty.
    range_kz : list[float, float]
        Range of absolute parallel wavenumbers to force when range_kmod is
        empty.
    """

    explicit = True
    linear = False

    amplitude_per_mode: np.ndarray
    amplitude_per_mode_gpu: cp.ndarray
    corr_time: np.floating
    update_forcing_kernel: KernelWrapper

    def setup_cuda_definitions(self) -> None:
        # Alias system
        system = self.system

        # Set ranges and number of forced modes
        if system.input["forcing.range_kmod"]:
            self.setup_forcing_range_kmod() # Take precendence over kzkperp
        else:
            self.setup_forcing_range_kzkperp()

        # Validate amplitudes
        amplitude = np.asarray(
            system.input["forcing.amplitude"], dtype=system.float
        )

        if amplitude.shape != (system.number_of_fields,):
            raise InvalidFlucsInputFileError(
                "forcing.amplitude must contain one finite value per field."
            )
        if np.any(amplitude < 0.0):
            raise InvalidFlucsInputFileError(
                "forcing.amplitude values must be non-negative."
            )

        # Validate correlation time
        corr_time = system.input["forcing.corr_time"]
        if not np.isfinite(corr_time) or corr_time <= 0.0:
            raise InvalidFlucsInputFileError(
                "forcing.corr_time must be finite and positive."
            )

        # Validate random seed
        rand_seed = system.input["forcing.rand_seed"]
        if not isinstance(rand_seed, int) or not 0 <= rand_seed:
            raise InvalidFlucsInputFileError(
                "forcing.rand_seed must be an unsigned integer."
            )

        # Store forcing parameters
        self.amplitude_per_mode = amplitude / np.sqrt(self.forced_mode_count)
        self.corr_time = system.float(corr_time)

    def register_kernels(self) -> None:
        """
        Registers the forcing-state update kernel.
        """
        self.update_forcing_kernel = KernelWrapper(
            system=self.system,
            cuda_kernel_name="update_ornstein_uhlenbeck_forcing",
            grid=(self.system.half_cuda_grid_size,),
            block=(self.system.cuda_block_size,),
        )

    def ready(self) -> None:
        """
        Initialises a stationary forcing state.
        """
        self.amplitude_per_mode_gpu = cp.asarray(
            self.amplitude_per_mode, dtype=self.system.float
        )
        self.update_forcing_kernel(
            self.system.float(0.0),
            self.system.float(1.0),
            self.system.int(0),
            self.system.int(self.system.input["forcing.rand_seed"]),
            self.amplitude_per_mode_gpu,
        )

    def prepare_time_step(self) -> None:
        """
        Advances the forcing state using the current timestep.
        """
        system = self.system

        decay = self.system.float(
            np.exp(-float(self.system.current_dt) / float(self.corr_time))
        )
        innovation = self.system.float(np.sqrt(1.0 - decay**2))

        self.update_forcing_kernel(
            decay,
            innovation,
            self.system.int(self.system.current_step),
            self.system.int(self.system.input["forcing.rand_seed"]),
            self.amplitude_per_mode_gpu,
        )
