"""Defines the base class for diagnostics."""

from __future__ import annotations
from typing import TYPE_CHECKING
from abc import ABC, abstractmethod
import numpy as np

if TYPE_CHECKING:
    from flucs.output import FlucsOutput
    from flucs.systems import FlucsSystem


class FlucsDiagnostic(ABC):
    """ Prepares data to be written by a FlucsOutput. """
    # Name of the diagnostic
    name: str

    # Parent output and system
    output: FlucsOutput
    system: FlucsSystem

    # Shape of the data for a single time step.
    # This is a tuple of strings that correspond to the names of the dimensions
    # used in the netCDF4 file and specified in dimensions_dict.
    shape: tuple

    dimensions_dict: dict[str, np.ndarray | float]

    data_cache: list[np.ndarray]

    # Complex-number diagnostics need to be handled separately
    is_complex: bool = False

    def __init__(self, system: FlucsSystem, output: FlucsOutput):
        self.system = system
        self.output = output
        self.data_cache = []

    def execute(self):
        """Runs the diagnostic."""
        self.data_cache.append(
            self.get_data())

    @abstractmethod
    def ready(self) -> None:
        """Called right before execution of the solver loop begins."""

    @abstractmethod
    def get_data(self) -> np.ndarray | float:
        """Returns the diagnostic data at the current time step."""
