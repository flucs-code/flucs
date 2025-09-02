"""Definition of the abstract base for any flucs system.

Outlines the basic functionality of any system using
abstract methods.

"""

import numpy as np
import cupy as cp
from abc import ABC, abstractmethod
from importlib.resources import files
import flucs
from flucs import FlucsInput
from flucs.utilities.cupy import ModuleOptions

class FlucsSystem(ABC):
    """A generic system of equations for flucs."""
    input: FlucsInput = None

    # Float and complex types
    float: type
    complex: type
    int: type

    # CuPy module for the system
    cupy_module: cp.RawModule

    # Compile options for CUDA
    module_options: ModuleOptions

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
                self.module_options.define_constant("DOUBLE_PRECISION")

        # We always use 32-bit integers
        self.int = np.int32

    @abstractmethod
    def setup(self) -> None:
        """The setup method sets up the system of equations for running the
        solver (allocates memory, handles initial conditions, output files,
        etc).

        """

    def ready(self) -> None:
        """This method is called immediately before the solver starts
        execution.

        """

        # The CUDA module for the system should be located in the same
        # directory as its .py file and have a name that matches the .py file,
        # with the .cu extension.

        resource_path = files(self.__module__) / f"{self.__module__.split('.')[-1]}.cu"
        with open(resource_path) as f:
            cuda_module = f.read()

        self.cupy_module = cp.RawModule(code=cuda_module,
                                   options=self.module_options.get_options())

        self.cupy_module.compile()

    @abstractmethod
    def _interpret_input(self) -> None:
        pass

    def __init__(self, input : FlucsInput) -> None:
        self.input = input
        self.module_options = ModuleOptions()
        self.module_options.add_string_option(f"-I{files(flucs).parent}")
        self._interpret_input()
        self._set_precision()
