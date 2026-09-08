"""Forcing methods for the Fourier test system."""

from flucs.input import InvalidFlucsInputFileError
from flucs.solvers.fourier.fourier_system_forcing import FourierSystemForcing


class TestFourierNegativeDampingForcing(FourierSystemForcing):
    """
    Explicit negative-damping forcing for all components of u.

    Parameters
    ----------
    rate : float
        Amplitude growth rate of modes.
    range_kperp : list[float, float]
        Range of perpendicular wavenumbers to force.
    range_kz : list[float, float]
        Range of absolute parallel wavenumbers to force.
    """

    explicit = True
    linear = False

    def setup_cuda_definitions(self) -> None:
        # Alias system
        system = self.system

        # Set ranges and number of forced modes
        self.setup_forcing_range_kzkperp()

        # Validate energy injection rate
        rate = system.input["forcing.rate"]
        if rate < 0.0:
            raise InvalidFlucsInputFileError(
                "forcing.rate must be non-negative for negative_damping."
            )

        # Divide the total injection equally between the forced modes.
        system.module_options.define_float("FORCING_RATE", rate)
