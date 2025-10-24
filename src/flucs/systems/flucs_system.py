"""Definition of the abstract base for any flucs system.

Outlines the basic functionality of any system using
abstract methods.

"""


from __future__ import annotations
from typing import TYPE_CHECKING
import pathlib as pl
import os
import heapq
import datetime
from netCDF4 import Dataset
import importlib
from typing import Type
import numpy as np
import cupy as cp
from abc import ABC, abstractmethod
import flucs
from flucs import FlucsInput
from flucs.output import FlucsOutput, FlucsDiagnostic
from flucs.utilities.cupy import ModuleOptions
from flucs.systems.flucs_restart_manager import FlucsRestartManager

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
    final_time: float

    init_time: float
    init_dt: float

    # Restart manager
    restart_manager: FlucsRestartManager

    # CuPy module for the system
    cupy_module: cp.RawModule

    # Compile options for CUDA
    module_options: ModuleOptions

    # A priority queue of outputs
    output_heap: list[FlucsOutput] | None = None

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
        import pathlib as pl

        for parent_cls in reversed(cls.__mro__):
            if not issubclass(parent_cls, FlucsSystem):
                continue

            p = pl.Path(importlib.import_module(parent_cls.__module__).__file__)
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
        if self.output_heap is None:
            self.output_heap = []

        heapq.heappush(self.output_heap, output)

    def execute_diagnostics(self, ignore_next_save: bool = False):
        """Executes diagnostics based on the current time step.

        Parameters
        ----------
        ignore_next_save : bool
            If ignore_next_save is True, then all diagnostics are executed
            regardless of their next save time.

        """
        if self.output_heap is None:
            return

        if ignore_next_save:
            for output_to_execute in self.output_heap:
                output_to_execute.execute()
            return

        # Execute only those that need to be executed at the current time step
        while self.output_heap[0].next_save == self.current_step:
            output_to_execute = heapq.heappop(self.output_heap)
            output_to_execute.execute()
            heapq.heappush(self.output_heap, output_to_execute)

    def setup(self) -> None:
        """
        Sets up the initial time data, calls the restart manager
        and then delegates to the system-specific setup hook.

        """

        self.init_time = 0.0
        self.init_dt = float(self.input["time.dt"])
        self.final_time = float(self.input["time.tfinal"])

        self.restart_manager = FlucsRestartManager(self)
        self._setup_system()

    @abstractmethod
    def _setup_system(self) -> None:
        """
        System-specific setup (allocate, set initial conditions, etc.).
        """
        pass

    def write_output(self):
        if self.output_heap is not None:
            for output in self.output_heap:
                output.write()

    def setup_output(self) -> None:
        """Initialise outputs."""

        for output_name, output_opt in self.input["output"].items():
            if not isinstance(output_opt, dict):
                continue

            # If save_steps is negative, don't add the diagnostic
            if output_opt["save_steps"] < 0:
                continue

            self.add_output(FlucsOutput(name=output_name, system=self))

    def compile_cupy_module(self) -> None:
        """ Compiles the CuPy CUDA module associated with the system

        Custom CUDA setup should be done by overriding this method. Do not
        forget to call super().compile_cupy_module()!

        The CUDA module for the system should be located in the same
        directory as its .py file and have a name that matches the .py file,
        with the .cu extension.

        """

        import datetime

        # resource_path = Path(importlib.import_module(self.__module__).__file__).parent / f"{self.__module__.split('.')[-1]}.cu"
        p = pl.Path(importlib.import_module(self.__module__).__file__)
        resource_path = p.with_name(f"{p.stem}.cu")
        with open(resource_path) as f:
            cuda_module = f.read()

        # CuPy's caching of compiled kernels is annoying and breaks things.
        # Add the current date at the end of the source to force recompilation
        cuda_module += f"\n// {datetime.datetime.now()}"

        self.cupy_module = cp.RawModule(code=cuda_module,
                                        options=self.module_options.get_options())

        self.cupy_module.compile()

    def ready(self) -> None:
        """
        This method is called immediately before the solver starts
        execution.

        """

        # Ready up the outputs
        if self.output_heap is not None:
            for output in self.output_heap:
                output.ready()

    @abstractmethod
    def _interpret_input(self) -> None:
        pass

    @abstractmethod
    def get_restart_data(self) -> dict[str, dict]:
        """
        Return a dictionary describing restart variables.

        Structure:
        {
            "<var_name>": {
            "data": <ndarray (NumPy or CuPy)>,
            "dimension_names": (<dim1>, <dim2>, ...)  # optional, tuple/list of str
            },
            ...
        }
        """

    def __init__(self, input : FlucsInput) -> None:
        self.input = input
        self.module_options = ModuleOptions()
        print(f"-I{pl.Path(flucs.__file__).parent}")
        self.module_options.add_string_option(f"-I{pl.Path(flucs.__file__).parent.parent}")
        self._interpret_input()
        self._set_precision()
