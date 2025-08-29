"""Definition of the abstract base for any flucs system.

Outlines the basic functionality of any system using
abstract methods.

"""

import numpy as np
from abc import ABC, abstractmethod
from importlib.resources import files
from flucs import FlucsInput

class FlucsSystem(ABC):
    """A generic system of equations for flucs."""
    input: FlucsInput = None

    # Float and complex types
    float: type
    complex: type
    int: type

    @classmethod
    def load_defaults(cls, flucs_input: FlucsInput):
        """Loads default parameters into a flucs input object.

        Parameters
        ----------
        flucs_input : FlucsInput
            Input object that will be initialised with the defaults.
        """

        resource_path = files(cls.__module__) / "defaults.toml"
        with resource_path.open("r") as f:
            contents = f.read()

        flucs_input.load_toml_str(contents, default=True)

    def _set_precision(self):
        """Interprets the precision parameter and sets types accordingly."""
        match self.input["setup.precision"]:
            case "single":
                self.float = np.float32
                self.complex = np.complex64
            case "double":
                self.float = np.float64
                self.complex = np.complex128

        # We always use 32-bit integers
        self.int = np.int32


    @abstractmethod
    def initialise(self) -> None:
        pass

    @abstractmethod
    def set_initial_conditions(self) -> None:
        pass

    @abstractmethod
    def _validate_input(self) -> None:
        pass

    def __init__(self, input : FlucsInput) -> None:
        self.input = input
        self._validate_input()
        self._set_precision()
