from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from flucs.solvers.fourier.fourier_system import FourierSystem

import numpy as np

from flucs.input import InvalidFlucsInputFileError
from flucs.utilities.messages import flucsprint


class FourierSystemForcing(ABC):
    """
    Base class for optional forcing methods used by FourierSystem solvers.
    """

    linear: bool
    explicit: bool
    forced_mode_count: int

    def __init__(self, system: FourierSystem):
        self.system = system

    @abstractmethod
    def setup_cuda_definitions(self) -> None:
        pass

    def setup_forcing_range_kz_kperp(self):
        """
        Determines the range of wavenumbers to be forced based on the input
        parameters, and calculates the number of modes in this range.
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
        system.module_options.define_float("FORCING_KPERP2_MIN", kperp_min**2)
        system.module_options.define_float("FORCING_KPERP2_MAX", kperp_max**2)
        system.module_options.define_float("FORCING_KZ_MIN", kz_min)
        system.module_options.define_float("FORCING_KZ_MAX", kz_max)

        # Determine number of forced modes
        system._precompute_wavenumbers()
        kx, ky, kz = system.get_broadcast_wavenumbers()
        kperp2 = kx**2 + ky**2
        kz_abs = np.abs(kz)

        forced_modes_halfny = (
            (kperp2 > kperp_min**2)
            & (kperp2 < kperp_max**2)
            & (kz_abs > kz_min)
            & (kz_abs < kz_max)
        )
        ky0_modes = ky < 0.5 * ky[0, 0, 1]

        forced_mode_count = 2 * np.sum(forced_modes_halfny) - np.sum(
            forced_modes_halfny & ky0_modes
        )

        if forced_mode_count == 0:
            raise InvalidFlucsInputFileError(
                "No modes are being forced. Please check your forcing.range_kz "
                "and/or forcing.range_kperp."
            )

        # Set number of forced modes
        self.forced_mode_count = int(forced_mode_count)

        flucsprint(
            f"Forcing applied on a total of {self.forced_mode_count} modes.",
            source=self,
        )
