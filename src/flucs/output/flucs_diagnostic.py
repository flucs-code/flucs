"""Defines the base class for diagnostics."""

from __future__ import annotations
from typing import TYPE_CHECKING
from abc import ABC, abstractmethod
import numpy as np

if TYPE_CHECKING:
    from flucs.diagnostics.output import FlucsOutput
    from flucs.systems import FlucsSystem

class FlucsDiagnostic(ABC):
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

    @abstractmethod
    def print_diagnostic(self):
        """Called if the diagnostic's output group is stdout_only."""
        pass


    def execute(self):
        self.data_cache.append(
            self.get_data())

        if self.output.stdout_only:
            self.print_diagnostic()

            # No need to keep stuff in memory if stdout_only
            self.data_cache.clear()

    @abstractmethod
    def ready(self) -> None:
        pass

    @abstractmethod
    def get_data(self) -> np.ndarray | float:
        pass
