"""Definition of the abstract base for any flucs system.

Outlines the basic functionality of any system using
abstract methods.

"""


from pathlib import Path
import heapq
import importlib
from typing import Type
import numpy as np
import cupy as cp
from abc import ABC, abstractmethod
import flucs
from flucs import FlucsInput
from flucs.diagnostics.output import FlucsOutput
from flucs.diagnostics.diagnostic import FlucsDiagnostic
from flucs.utilities.cupy import ModuleOptions


class FlucsSystem(ABC):
    """A generic system of equations for flucs."""
    input: FlucsInput = None

    # Float and complex types
    float: type
    complex: type
    int: type

    # Variables to that keep track of time
    current_step: int
    current_dt: float
    current_time: float

    # CuPy module for the system
    cupy_module: cp.RawModule

    # Compile options for CUDA
    module_options: ModuleOptions

    # A priority queue of outputs
    output_heap: list[FlucsOutput]

    # A dict of supported diagnostics
    diags_dict: dict[str, Type[FlucsDiagnostic]]

    @classmethod
    def load_defaults(cls, flucs_input: FlucsInput):
        """Loads default parameters into a flucs input object.

        Parameters
        ----------
        flucs_input : FlucsInput
            Input object that will be initialised with the defaults.
        """

        module = importlib.import_module(cls.__module__)
        resource_path = Path(module.__file__).with_name("defaults.toml")
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

    def add_output(self, output: FlucsOutput):
        heapq.heappush(self.output_heap, output)

    def execute_diagnostics(self):
        while self.output_heap[0].next_save == self.current_step:
            output_to_execute = heapq.heappop(self.output_heap)
            output_to_execute.execute()
            heapq.heappush(self.output_heap, output_to_execute)

    @abstractmethod
    def init_output(self) -> None:
        """Initialises the ouput files."""
        pass

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

        resource_path = Path(importlib.import_module(self.__module__).__file__).parent / f"{self.__module__.split('.')[-1]}.cu"
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
        print(f"-I{Path(flucs.__file__).parent}")
        self.module_options.add_string_option(f"-I{Path(flucs.__file__).parent.parent}")
        self._interpret_input()
        self._set_precision()
