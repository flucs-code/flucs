"""
Defines the base class for diagnostics.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar

import numpy as np

if TYPE_CHECKING:
    from flucs.output import FlucsOutput
    from flucs.systems import FlucsSystem


@dataclass
class FlucsDiagnosticVariable:
    """Data and dimensions for a single output variable.
    A FlucsDiagnostic has a set of FlucsDiagnosticVariable.
    """

    # Name of the data variable
    name: str

    # Shape of the data for a single time step.
    # This is a tuple of strings that correspond to the names of the dimensions
    # used in the netCDF4 file and specified in dimensions_dict.
    shape: tuple

    dimensions: dict[str, np.ndarray | float]

    data_cache: list[np.ndarray] = field(default_factory=list, init=False)

    # Complex-number variables are handled separately
    is_complex: bool = False


class FlucsDiagnostic(ABC):
    """Prepares data to be written by a FlucsOutput."""

    # Name of the diagnostic
    name: str

    # Default values for any options
    option_defaults: ClassVar[dict[str, object]] = {}

    # Parent output and system
    output: FlucsOutput
    system: FlucsSystem

    # Length of the data cache, i.e., number of
    # saves to be written at next write
    cache_len: int

    # Output variables
    vars: dict[str, FlucsDiagnosticVariable]

    def __init__(
        self,
        system: FlucsSystem,
        output: FlucsOutput,
        options: dict | None = None,
    ):
        self.system = system
        self.output = output
        self.cache_len = 0

        self.vars = {}
        self._load_options(options or {})
        self.init_vars()

    def _load_options(self, options: dict) -> None:
        """Loads options for the given diagnostic."""

        # Validate options
        for key in options:
            if key not in self.option_defaults:
                raise KeyError(
                    f"Unknown option '{key}' for diagnostic '{self.name}'."
                )

        # Load options
        for key, default in self.option_defaults.items():
            value = options.get(key, default)
            setattr(self, key, type(default)(value))  # Cast to default type

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
