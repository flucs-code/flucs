"""
Forcing methods for the Fourier test system.
"""

from flucs.input import InvalidFlucsInputFileError
from flucs.solvers.fourier.fourier_system_forcing import FourierSystemForcing


class TestFourierNegativeDampingForcing(FourierSystemForcing):
    """
    Explicit negative-damping forcing for all components of u.

    Parameters
    ----------
    rate : float
        Amplitude growth rate of modes.
    range_kmod : list[float, float]
        Range of isotropic wavenumbers to force.
    """

    explicit = True
    linear = False

    def setup_cuda_definitions(self) -> None:
        # Alias system
        system = self.system

        # Set ranges and number of forced modes
        self.setup_forcing_range_kmod()

        # Validate forcing rate
        rate = system.input["forcing.rate"]
        if rate < 0.0:
            raise InvalidFlucsInputFileError(
                "forcing.rate must be non-negative for negative_damping."
            )
        system.module_options.define_float("FORCING_RATE", rate)
