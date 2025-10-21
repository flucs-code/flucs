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

    # Variables for restarting
    _save_for_restart: bool = False
    _restart_write_steps: int = 0
    _restart_counter: int = 0              
    _restart_path_old: pl.Path | None = None
    _restart_path_new: pl.Path | None = None
    _restart_source: pl.Path | None = None

    restart_time: float 
    restart_dt: float

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
        First sets up restart options and then delegates to the
        system-specific setup hook.
        """

        self._setup_restart()
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
    def _get_restart_data(self) -> dict[str, dict]:
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
        raise NotImplementedError()

    
    def _setup_restart(self) -> None:
        """
        Sets up restart variables and decides whether to restart.
        A specified restart file takes precedence over automatic detection.
        """

        # Flag for whether to save restart data
        self._save_for_restart = bool(self.input["restart.save_for_restart"])
        self._restart_source = None

        # Set up paths
        self._restart_path_old = pl.Path(self.input.io_path / "restart.old.nc")
        self._restart_path_new = pl.Path(self.input.io_path / "restart.new.nc")
        self._restart_path_tmp = pl.Path(self.input.io_path / "restart.tmp.nc")

        # Initialise saving cadence
        self._restart_write_steps = int(self.input["restart.restart_write_steps"])
        self._restart_counter = int(self.input["restart.restart_write_steps"])

        # Check for specified restart file
        restart_file = self.input["restart.restart_file"]

        # Restart from specified file
        if restart_file != "":
            p = pl.Path(restart_file).expanduser()
            if not p.is_absolute(): 
                p = (self.input.io_path / p).resolve()
            if not p.exists():
                print(f"Specified restart file does not exist: {p}")
                exit(1)
            try:
                with Dataset(p, "r") as ds:
                    self._ensure_restart_complete(ds)
            except Exception as e:
                print(f"Invalid restart file {p}: {e}")
                exit(1)
            self._restart_source = p
            print(f"Restarting from specified file: {self._restart_source}")

        # Restart from existing files 
        elif self.input["restart.restart_if_exists"]:

            # Look for possible restart files in the default location
            possible_restart_files = [
                p for p in (self._restart_path_new, self._restart_path_old) if p and p.exists()
            ]

            # Check whether the files are valid for restart
            if possible_restart_files:
                for f in possible_restart_files:
                    try:
                        with Dataset(f, "r") as ds:
                            self._ensure_restart_complete(ds)
                        self._restart_source = f
                        break
                    except Exception as e:
                        print(f"Found restart file {f} but it is invalid: {e}")

                if self._restart_source is not None:
                    print(f"Restart files found in {self.input.io_path}.")
                    print(f"Restarting from {self._restart_source}.")
                else:
                    print("All found restart files are invalid.")
                    exit(1)

        # Initialise using specified method
        if self._restart_source is None:
            print(f"Initialising using type: {self.input['init.type']}")
            self.final_time = float(self.input["time.tfinal"])

        return 
    
    def write_restart(self, force: bool=False) -> None:
        """
        Executes writing restart data if necessary.

        Parameters
        ----------
        force : bool
            If force is True, the restart data is written at that timestep.
        """

        if self._save_for_restart is False:
            return

        # Check whether its time to write
        if not force:
            self._restart_counter -= 1
            if self._restart_counter != 0:
                return
            self._restart_counter = self._restart_write_steps

        # Write the data
        self._write_restart_data()


    def _write_restart_data(self) -> None:
        """
        Writes restart data to netCDF files, rotating old and new files
        as necessary.
        """

        # Filepaths
        old_path = self._restart_path_old
        new_path = self._restart_path_new
        tmp_path = self._restart_path_tmp

        # Get restart data 
        restart_data = self._get_restart_data()

        # Set precision for netCDF variables
        precision = "f4" if self.float is np.float32 else "f8"

        # Write to temporary file
        with Dataset(tmp_path, "w", format="NETCDF4") as ds:

            # Set file attributes
            ds.setncattr("created", datetime.datetime.now(datetime.timezone.utc).isoformat())
            ds.setncattr("location", str(tmp_path.parent))
            ds.setncattr("pid", int(os.getpid()))
            ds.setncattr("type", str("restart file"))
            ds.setncattr("restart_write_steps", np.int32(self._restart_write_steps))
            ds.setncattr("complete", np.int32(0))

            # Scalar values
            ds.createVariable("current_time", precision, ())[...] = float(self.current_time)
            ds.createVariable("current_dt", precision, ())[...] = float(self.current_dt)
            ds.createVariable("current_step", "i8", ())[...] = int(self.current_step)

            # Arrays
            for var_name, var_dict in restart_data.items():
                var_data = var_dict["data"]
                if isinstance(var_data, cp.ndarray):
                    var_data = cp.asnumpy(var_data)

                dim_names = var_dict.get("dimension_names", None)
                if dim_names is not None:
                    for dname, dsize in zip(dim_names, var_data.shape):
                        if dname not in ds.dimensions:
                            ds.createDimension(dname, int(dsize))
                else:
                    dim_names = tuple(f"{var_name}_dim{i}" for i in range(var_data.ndim))
                    for dname, dsize in zip(dim_names, var_data.shape):
                        if dname not in ds.dimensions:
                            ds.createDimension(dname, int(dsize))

                if np.iscomplexobj(var_data):
                    v_r = ds.createVariable(f"{var_name}_real", precision, tuple(dim_names))
                    v_i = ds.createVariable(f"{var_name}_imag", precision, tuple(dim_names))
                    v_r[:] = var_data.real
                    v_i[:] = var_data.imag
                else:
                    v = ds.createVariable(var_name, precision, tuple(dim_names))
                    v[:] = var_data

            # Mark write as complete
            ds.setncattr("complete", np.int32(1))

        # Replace files
        try:
            if new_path.exists():
                os.replace(new_path, old_path)
        except FileNotFoundError:
            pass

        os.replace(tmp_path, new_path)

        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass

    
    def _ensure_restart_complete(self, ds: Dataset) -> None:
        """
        Validates that the restart data is complete, and if so sets the 
        final simulation time accordingly.
        """
        try:
            if int(getattr(ds, "complete", 0)) != 1:
                raise ValueError("incomplete")
        except Exception:
            # Any parsing/type error counts as incomplete
            raise ValueError("incomplete")

        try:
            # Set restart variables
            self.restart_time = float(ds.variables["current_time"][...])
            self.restart_dt = float(ds.variables["current_dt"][...])
            self.final_time = self.restart_time + float(self.input["time.tfinal"])

        except KeyError as e:
            raise ValueError(f"Missing variable in restart file: {e}")
        except Exception as e:
            raise ValueError(f"Failed to read variables in restart file: {e}")


    def __init__(self, input : FlucsInput) -> None:
        self.input = input
        self.module_options = ModuleOptions()
        print(f"-I{pl.Path(flucs.__file__).parent}")
        self.module_options.add_string_option(f"-I{pl.Path(flucs.__file__).parent.parent}")
        self._interpret_input()
        self._set_precision()
