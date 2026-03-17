"""
Defines the base class for diagnostics.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from flucs.output import FlucsOutput
    from flucs.systems import FlucsSystem


@dataclass
class FlucsDiagnosticVariable:
    """Data and dimensions for a single output variable.

    A `FlucsDiagnostic` has a set of `FlucsDiagnosticVariable`.
    """

    name: str
    """Name of the data variable."""

    shape: tuple[str, ...]
    """Shape of the data for a single time step.

    This is a tuple of strings that correspond to the names of the dimensions
    used in the netCDF4 file and specified in `dimensions`.
    """

    dimensions: dict[str, np.ndarray | float]

    data_cache: list[np.ndarray] = field(default_factory=list, init=False)

    is_complex: bool = False
    """Complex-number variables are handled separately."""


class FlucsDiagnostic(ABC):
    """Prepares data to be written by a `FlucsOutput`."""

    name: str
    """Name of the diagnostic."""

    output: FlucsOutput
    """Parent output."""

    system: FlucsSystem
    """Parent system."""

    cache_len: int
    """Length of the data cache

    i.e. the number of saves to be written at next write.
    """

    vars: dict[str, FlucsDiagnosticVariable]
    """Output variables."""

    def __init__(self, system: FlucsSystem, output: FlucsOutput):
        self.system = system
        self.output = output
        self.cache_len = 0

        self.vars = {}
        self.init_vars()

    def add_var(self, var: FlucsDiagnosticVariable) -> None:
        if var.name in self.vars:
            raise KeyError(
                f"Diagnostic {self.name} already has a variable: {var.name}"
            )

        self.vars[var.name] = var

    def save_data(self, var_name: str, data):
        self.vars[var_name].data_cache.append(data)

    def clear(self) -> None:
        """Clears the memory cache of the diagnostic."""
        for var in self.vars.values():
            var.data_cache.clear()

    @abstractmethod
    def init_vars(self) -> None:
        """Initialises self.vars."""

    @abstractmethod
    def execute(self) -> None:
        """Runs the diagnostic."""

    @abstractmethod
    def ready(self) -> None:
        """Called right before execution of the solver loop begins."""
