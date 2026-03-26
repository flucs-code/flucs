"""Definition of the abstract base for any flucs system.

Outlines the basic functionality of any system using
abstract methods.

"""

from __future__ import annotations

import datetime
import heapq
import importlib
import pathlib as pl
import sys
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import cupy as cp
import numpy as np
from cupy.cuda import cufft

from flucs import FlucsInput
from flucs.diagnostic import FlucsDiagnostic
from flucs.output import FlucsOutput
from flucs.restart import FlucsRestart
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
    tolerance: float

    # Variables to that keep track of time
    current_step: int
    current_dt: float
    current_time: float
    final_time: float

    init_time: float
    init_dt: float

    # Restart manager
    restart_manager: FlucsRestart

    # CuPy module for the system
    cupy_module: cp.RawModule

    # Compile options for CUDA
    module_options: ModuleOptions

    # CUFFT plan types
    fft_c2r_plan_type: int
    fft_r2c_plan_type: int

    # A priority queue of outputs
    output_heap: list[FlucsOutput] | None = None
    steps_until_next_write: int

    # A dict of supported diagnostics
    diags: dict[str, type[FlucsDiagnostic]]

    @classmethod
    def get_available_diags(cls) -> dict[str, type[FlucsDiagnostic]]:
        """Returns a dict of available diagnostics.
        Goes recursively through all the parent systems.

        """

        # FlucsDiagnostic uses its name attribute to create a hash
        # so using a set here will give us a set of unique diagnostics
        # where uniqueness is based on their name.
        diags = set()

        for parent_cls in reversed(cls.__mro__):
            if not issubclass(parent_cls, FlucsSystem):
                continue

            if not hasattr(parent_cls, "diags"):
                continue

            diags.update(parent_cls.diags)

        return {diag.name: diag for diag in diags}

    @classmethod
    def load_defaults(cls, flucs_input: FlucsInput):
        """Loads default parameters into a flucs input object.
        Goes recursively through all the parent systems.

        Parameters
        ----------
        flucs_input : FlucsInput
            Input object that will be initialised with the defaults.
        """
        for parent_cls in reversed(cls.__mro__):
            if not issubclass(parent_cls, FlucsSystem):
                continue

            p = pl.Path(importlib.import_module(parent_cls.__module__).__file__)
            defaults_path = p.with_name(f"{p.stem}.toml")
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
                self.module_options.define_flag("DOUBLE_PRECISION")

        # We always use 32-bit integers
        self.int = np.int32

        # Get float error tolerance
        self.tolerance = self.float(np.finfo(self.float).eps) * 64.0

    def add_output(self, output: FlucsOutput):
        if self.output_heap is None:
            self.output_heap = []

        heapq.heappush(self.output_heap, output)

    def execute_diagnostics(self, force: bool = False):
        """Executes diagnostics based on the current time step.

        Parameters
        ----------
        force: bool
            If force is True, then all diagnostics are executed
            regardless of their next save time.

        """
        if self.output_heap is None:
            return

        if force:
            for output_to_execute in self.output_heap:
                output_to_execute.execute()

            # Reset heap for the next save
            heapq.heapify(self.output_heap)

            return

        # Execute only those that need to be executed at the current time step
        while self.output_heap[0].next_save == self.current_step:
            output_to_execute = heapq.heappop(self.output_heap)
            output_to_execute.execute()
            heapq.heappush(self.output_heap, output_to_execute)

    def setup(self) -> None:
        """
        Sets up the initial time data, calls the restart manager,
        sets up Fourier-transform types, and then delegates to
        the system-specific setup hook.

        """

        self.init_time = self.float(0.0)
        self.init_dt = self.float(self.input["time.dt_max"])
        self.final_time = self.float(self.input["time.tfinal"])

        self.restart_manager = FlucsRestart(self)

        if self.input["setup.precision"] == "single":
            self.fft_c2r_plan_type = cufft.CUFFT_C2R
            self.fft_r2c_plan_type = cufft.CUFFT_R2C
        else:
            self.fft_c2r_plan_type = cufft.CUFFT_Z2D
            self.fft_r2c_plan_type = cufft.CUFFT_D2Z

        self._setup_system()

    @abstractmethod
    def _setup_system(self) -> None:
        """
        System-specific setup (allocate, set initial conditions, etc.).
        """
        pass

    def write_output(self, force=False):
        self.steps_until_next_write -= 1
        if self.steps_until_next_write > 0 and not force:
            return

        self.steps_until_next_write = self.input["output.write_steps"]

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
        """Compiles the CuPy CUDA module associated with the system

        Custom CUDA setup should be done by overriding this method. Do not
        forget to call super().compile_cupy_module()!

        The CUDA module for the system should be located in the same
        directory as its .py file and have a name that matches the .py file,
        with the .cu extension.

        """

        import datetime

        p = pl.Path(importlib.import_module(self.__module__).__file__)
        resource_path = p.with_name(f"{p.stem}.cu")
        with open(resource_path) as f:
            cuda_module = f.read()

        # CuPy's caching of compiled kernels is annoying and breaks things.
        # Add the current date at the end of the source to force recompilation
        cuda_module += f"\n// {datetime.datetime.now()}"

        self.cupy_module = cp.RawModule(
            code=cuda_module, options=self.module_options.get_options()
        )

        self.cupy_module.compile(log_stream=sys.stdout)

    def get_memory_usage(self, devices=None, synchronize=True) -> dict:
        """
        Checks the memory usage on the current devices and returns a dictionary
        with the results.

        Parameters
        ----------
        devices: list[int] | None
            Specific device ordinals to query. If None, queries all visible
            devices.

        synchronize: bool
            If True, calls deviceSynchronize() on each device before sampling.

        Returns
        -------
        device_info: dict
            Dictionary with memory usage data

        Notes
        -----
        All of the memory values in device_info are in bytes.

        """

        # Get device count
        n_devices = cp.cuda.runtime.getDeviceCount()
        if devices is None:
            devices = list(range(n_devices))

        # Initialise dictionary
        device_info = {
            "timestamp": datetime.datetime.now(
                datetime.timezone.utc
            ).isoformat(),
            "number_of_devices": n_devices,
        }

        # Save current device to keep context stable
        current_device = cp.cuda.Device()

        for index in devices:
            with cp.cuda.Device(int(index)) as device:
                # Setup dict
                key = f"device_{device.id:03d}"
                device_info[key] = {}

                # Ensure everything is synchronised before getting data
                if synchronize:
                    try:
                        cp.cuda.runtime.deviceSynchronize()
                    except Exception:
                        pass

                # Global device memory
                global_free, global_total = cp.cuda.runtime.memGetInfo()
                global_used = global_total - global_free

                # CuPy pools (device + pinned/host)
                pool = cp.get_default_memory_pool()
                pool_total = pool.total_bytes()
                pool_used = pool.used_bytes()
                pool_free = max(pool_total - pool_used, 0.0)

                # Device properties
                name, compute_capability, multiprocessors = None, None, None
                try:
                    properties = cp.cuda.runtime.getDeviceProperties(device.id)
                    name = properties.get("name")
                    if isinstance(name, (bytes, bytearray)):
                        name = name.decode()
                    major = properties.get("major", "?")
                    minor = properties.get("minor", "?")
                    compute_capability = f"{major}.{minor}"
                    multiprocessors = properties.get("multiProcessorCount")
                except Exception:
                    pass

                # Collate info
                device_info[key] = {
                    "id": int(device.id),
                    "name": name,
                    "compute_capability": compute_capability,
                    "multiprocessors": multiprocessors,
                    "global": {
                        "total": int(global_total),
                        "free": int(global_free),
                        "used": int(global_used),
                    },
                    "cupy": {
                        "total": int(pool_total),
                        "used": int(pool_used),
                        "free": int(pool_free),
                    },
                }

        # Ensure return to original context
        with current_device:
            pass

        bytes_to_gb = 1024**3

        # Print device information
        for key, info in device_info.items():
            if not key.startswith("device_"):
                continue

            global_used_gb = info["global"]["used"] / bytes_to_gb
            global_total_gb = info["global"]["total"] / bytes_to_gb
            cupy_total_gb = info["cupy"]["total"] / bytes_to_gb

            print(
                f"({info['id']}) {info['name']}: {global_used_gb:.3f} / "
                f"{global_total_gb:.3f} GB "
                f"({global_used_gb / global_total_gb * 100:.2f}%), "
                f"CuPy usage: {cupy_total_gb:.3f} GB "
                f"({cupy_total_gb / global_total_gb * 100:.2f}%)"
            )

        return device_info

    def ready(self) -> None:
        """
        This method is called immediately before the solver starts
        execution.

        """

        # Ready up the outputs
        if self.output_heap is not None:
            for output in self.output_heap:
                output.ready()

            # Reset heap for the next save
            heapq.heapify(self.output_heap)

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
            "dimension_names": (<dim1>, <dim2>, ...)  # optional, tuple of str
            },
            ...
        }
        """

    def _add_include_dirs(self) -> None:
        """Adds the base src folder of the projects of each FlucsSystem in the
        inheritance chain of the current instance.

        """
        for parent_cls in type(self).__mro__:
            if not issubclass(parent_cls, FlucsSystem):
                continue

            root_name = parent_cls.__module__.split(".")[0]
            root_mod = importlib.import_module(root_name)
            root_src_path = pl.Path(root_mod.__file__).parent.parent

            self.module_options.add_compiler_option(f"-I{root_src_path}")

    def __init__(self, input: FlucsInput) -> None:
        self.input = input
        self.module_options = ModuleOptions()
        self._add_include_dirs()
        self._interpret_input()
        self._set_precision()
