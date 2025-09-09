"""Definition of the abstract base for any flucs system.

Outlines the basic functionality of any system using
abstract methods.

"""


from __future__ import annotations
from typing import TYPE_CHECKING
from pathlib import Path
import heapq
import importlib
from typing import Type
import numpy as np
import cupy as cp
from abc import ABC, abstractmethod
import flucs
from flucs import FlucsInput
from flucs.output import FlucsOutput, FlucsDiagnostic
from flucs.utilities.cupy import ModuleOptions

if TYPE_CHECKING:
    from flucs.solvers import FlucsSolver


class FlucsSystem(ABC):
    """A generic system of equations for flucs."""
    input: FlucsInput = None

    # Solver running the system
    solver: FlucsSolver

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
        Goes recursively through all the parent systems.

        Parameters
        ----------
        flucs_input : FlucsInput
            Input object that will be initialised with the defaults.
        """
        import importlib
        from pathlib import Path

        for parent_cls in reversed(cls.__mro__):
            if not issubclass(parent_cls, FlucsSystem):
                continue

            p = Path(importlib.import_module(parent_cls.__module__).__file__)
            defaults_path = p.with_name(f'{p.stem}.toml')
            print(f"Loading SOLVER defaults for {defaults_path}")
            with defaults_path.open("r") as f:
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
        if not hasattr(self, "output_heap"):
            self.output_heap = []

        heapq.heappush(self.output_heap, output)

    def execute_diagnostics(self):
        while self.output_heap[0].next_save == self.current_step:
            output_to_execute = heapq.heappop(self.output_heap)
            output_to_execute.execute()
            heapq.heappush(self.output_heap, output_to_execute)

    def write_output(self):
        for output in self.output_heap:
            output.write()

    def setup(self) -> None:
        """The setup method sets up the system of equations for running the
        solver (allocates memory, handles initial conditions, output files,
        etc).

        """

        # Initialise outputs
        for output_name, output_opt in self.input["output"].items():
            if not isinstance(output_opt, dict):
                continue

            # If save_steps is negative, don't add the diagnostic
            if output_opt["save_steps"] < 0:
                continue

            self.add_output(FlucsOutput(name=output_name, system=self))

    def ready(self) -> None:
        """This method is called immediately before the solver starts
        execution.

        """

        # Ready up the outputs
        for output in self.output_heap:
            output.ready()

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
