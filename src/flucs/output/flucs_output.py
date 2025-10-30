"""
Defines the FlucsOutput class that handles a group of diagnostics that are
executed together and output to a file of specified format.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
from abc import ABC, abstractmethod
from importlib.metadata import entry_points
import datetime
import pathlib as pl
import numpy as np
from netCDF4 import Dataset, Group
from flucs.solvers import FlucsSolverState

if TYPE_CHECKING:
    from flucs.output import FlucsDiagnostic
    from flucs.systems import FlucsSystem


_registered_outputs = entry_points().select(group="flucs.outputs")

def get_output_type(output_type: str):
    """Returns an output type.

    Parameters
    ----------
    output_type: str
        Name of the output type. Must be registered as an
        entry point in the flucs.outputs group.

    Returns
    -------
    Appropriate FlucsOutput type.

    """
    return _registered_outputs[output_type].load()

class FlucsOutput(ABC):
    """Deals with a single output file. A FlucsSystem typically has several
    FlucsOuputs that handle different kinds of diagnostics and different kinds
    of output formats.

    """

    name: str
    type: str
    filepath: pl.Path
    extension: str
    save_steps: int
    next_save: int

    # Associated system
    system: FlucsSystem

    # List of diagnostics
    diagnostics: list[FlucsDiagnostic]

    # Cache of times at which data was saved
    time_cache: list[float]

    def __new__(cls, name: str, system: FlucsSystem):
        output_type = system.input[f"output.{name}.type"]
        print(_registered_outputs)
        output_class = _registered_outputs[output_type].load()
        return super().__new__(output_class)

    @abstractmethod
    def _setup_output_file(self):
        pass

    @abstractmethod
    def write(self):
        pass

    def _add_diagnostics_from_input(self):
        """Adds all diagnostics from the input of the associated FlucsSystem"""
        for diag_name in self.system.input[f"output.{self.name}.diags"]:
            diag_to_add =\
                self.system.diags_dict[diag_name](system=self.system,
                                                  output=self)
            diag_to_add.output = self
            self.diagnostics.append(diag_to_add)

    def execute(self):
        """Executes each diagnostic. Does not save to disk."""
        for diag in self.diagnostics:
            diag.execute()

        self.next_save += self.save_steps
        self.time_cache.append(self.system.current_time)

    def ready(self):
        """Sets up the diagnostic for running."""
        self.time_cache.clear()
        self.next_save = 0
        for diag in self.diagnostics:
            diag.ready()
            diag.data_cache.clear()

        if self.system.solver.state == FlucsSolverState.RUNNING:
            self._setup_output_file()

    def __init__(self, name: str, system: FlucsSystem) -> None:
        self.name = name
        self.system = system
        self.filepath = (
            self.system.input.io_path / f"output.{name}.{self.extension}"
        )

        self.diagnostics = []
        self.time_cache = []

        # Setup steps and diagnostics from input file and system
        self.next_save = 0
        self.save_steps = self.system.input[f"output.{self.name}.save_steps"]

        self._add_diagnostics_from_input()

    def __lt__(self, other):
        if not isinstance(other, FlucsOutput):
            raise TypeError("Makes no sense to compare a FlucsOutput "
                            f"with a {type(other)}")
        return self.next_save < other.next_save


class FlucsOutputText(FlucsOutput):
    """ Space-separated text output"""
    extension = "txt"

    # The first few columns are always the same and contain simple timing data
    timing_data = []
    timing_data_column_names = ["time", "time step", "dt"]

    # Data formatting options
    column_width = 12
    column_pad = "    "  # 4 spaces
    float_format = "3e"
    complex_format = "1e"

    def _setup_output_file(self):
        column_names = []

        # Add timing data (first few columns)
        for timing_column_name in self.timing_data_column_names:
            column_names.append(
                f"{timing_column_name:>{self.column_width}}"
            )

        for diag in self.diagnostics:
            column_names.append(
                f"{diag.name:>{self.column_width}}"
            )

        if not self.filepath.exists():
            with open(self.filepath, "a", encoding="utf-8") as file:
                file.write(self.column_pad.join(column_names))
                file.write('\n')

    def _add_diagnostics_from_input(self):
        """ Add additional check for scalar diagnostics. """
        super()._add_diagnostics_from_input()

        for diag in self.diagnostics:
            if len(diag.shape) != 0:
                raise ValueError(
                    f"Cannot add diagnostic {diag.name} to text output "
                    f"{self.name} because text output supports only scalar "
                    "diagnostics."
                )

    def ready(self):
        """
        In addition to clearing the usual things, we need to clear the
        timing data, too.
        """

        self.timing_data.clear()
        super().ready()

    def execute(self):
        """Saves timing data in addition to the individual diagnostics."""
        self.timing_data.append([
            self.system.current_time,
            self.system.current_step,
            self.system.current_dt
        ])

        super().execute()

    def format_data(self, data):
        """ Returns an appropriately formatted representation of given data as
        a string of fixed width equal to FlucsOutputText.column_width.

        Parameters
        ----------
        data: str, int, float, or complex
            The data to be formatted

        """
        if isinstance(data, (str, int, np.integer)):
            return f"{data:>{self.column_width}}"

        if isinstance(data, (float, np.floating)):
            return f"{data:>{self.column_width}.{self.float_format}}"

        if isinstance(data, (complex, np.complexfloating)):
            return f"{data:>{self.column_width}.{self.complex_format}}"

        raise ValueError(f"Data type {type(data)}"
                         "is not supported by FlucsOutputText.")

    def write(self):
        """ Writes formatted rows to the text output file. """
        if self.system.solver.state != FlucsSolverState.RUNNING:
            return

        # Don't do anything if we don't have any data
        if not self.time_cache:
            return

        with open(self.filepath, "a", encoding="utf-8") as file:
            for save_index, _ in enumerate(self.time_cache):
                row_to_write = []
                for value in self.timing_data[save_index]:
                    row_to_write.append(self.format_data(value))

                for diag in self.diagnostics:
                    row_to_write.append(
                        self.format_data(diag.data_cache[save_index])
                    )

                file.write(self.column_pad.join(row_to_write))
                file.write('\n')

        # Clear caches
        for diag in self.diagnostics:
            diag.data_cache.clear()
        self.time_cache.clear()
        self.timing_data.clear()


class FlucsOutputNC(FlucsOutput):
    """ netCDF4 output """

    extension = "nc"
    # Dataset for the netCDF4 file
    dataset: Dataset

    # netCDF4 group to write to
    group_name: str
    group: Group

    def _setup_group(self):
        dataset: Dataset = self.dataset
        if not hasattr(self, "group_name"):
            # We are yet to initialise the group
            # Go through the file and pick group_name to be an integer that is
            # equal to the largest one found + 1
            group_number = max([-1] +
                               [int(name) for name in dataset.groups.keys()])
            group_number += 1

            self.group_name = str(group_number)
            grp = dataset.createGroup(self.group_name)
            grp.createDimension("time", None)
            grp.createVariable("time", "f4", ("time",))

        self.group = dataset.groups[self.group_name]

    def _setup_output_file(self):
        """Creates all the necessary groups, dimensions, and variables in the
        netCDF4 file where the output data will be written. Should be called
        before the main solver loop begins (but after any timing/optimisation
        routines).

        """

        with Dataset(self.filepath, "r+", format="NETCDF4") as self.dataset:
            self._setup_group()

            # Set global attributes
            self.dataset.setncattr(
                "created",
                datetime.datetime.now(datetime.timezone.utc).isoformat()
            )
            self.dataset.setncattr("location", str(self.filepath.parent))
            self.dataset.setncattr("type", str("flucs_output"))

            # Check if we already have all necessary dimensions
            # for every diagnostic
            for diagnostic in self.diagnostics:
                diag_incompatible = False
                for dim_name, dim_data in diagnostic.dimensions_dict.items():
                    dim_size = len(dim_data)

                    # If it exists, ensure it's the same
                    if (dim_name in self.group.dimensions and
                            dim_size != self.group.dimensions[dim_name].size):

                        print(
                            f"Dimension {dim_name} in output file"
                            f" {self.filepath.name} has size"
                            f" {self.group.dimensions[dim_name].size}"
                            f" which differs from expected size"
                            f" {len(dim_data)} required by diagnostic"
                            f"  {diagnostic.name}!"
                            "\n"
                            f"Therefore, diagnostic {diagnostic.name}"
                            " will not operate!")
                        diag_incompatible = True
                        break

                    self.group.createDimension(dim_name, dim_size)
                    dim_var = self.group.createVariable(dim_name,
                                                        "f4",
                                                        (dim_name,))
                    dim_var[:] = dim_data[:]

                if diag_incompatible:
                    continue

                # Create variable
                if diagnostic.is_complex:
                    # Complex variables are stored as two separate netCDF4 vars
                    # for the real and imaginary parts with suffixes _real and
                    # _imag, respectively.
                    self.group.createVariable(f"{diagnostic.name}_real", "f4",
                                              ("time", ) + diagnostic.shape)
                    self.group.createVariable(f"{diagnostic.name}_imag", "f4",
                                              ("time", ) + diagnostic.shape)
                else:
                    self.group.createVariable(diagnostic.name, "f4",
                                              ("time", ) + diagnostic.shape)

    def write(self):
        """Saves any cached diagnostic data to disk and clears the cache.
        Saves data only if we are not timing.

        """
        if self.system.solver.state != FlucsSolverState.RUNNING:
            return

        # Don't do anything if we don't have any data
        if not self.time_cache:
            return

        with Dataset(self.filepath, "r+", format="NETCDF4") as self.dataset:
            self._setup_group()

            times_to_write = len(self.time_cache)
            first_index = self.group["time"].shape[0]
            last_index = first_index + times_to_write
            # Write time data
            self.group["time"][first_index:last_index]\
                = np.array(self.time_cache)[:]

            self.time_cache.clear()

            # Write individual diagnostics and clear data cache
            for diag in self.diagnostics:
                # If scalar diagnostic, this is much easier
                if len(diag.shape) == 0:
                    if diag.is_complex:
                        self.group[f"{diag.name}_real"][first_index:last_index]\
                            = np.array(diag.data_cache).real
                        self.group[f"{diag.name}_imag"][first_index:last_index]\
                            = np.array(diag.data_cache).imag
                    else:
                        self.group[diag.name][first_index:last_index]\
                            = np.array(diag.data_cache)
                else:
                    # TODO: Rewrite with np.array
                    if diag.is_complex:
                        for i in range(times_to_write):
                            self.group[f"{diag.name}_real"][first_index + i, :]\
                                = diag.data_cache[i][:].real
                            self.group[f"{diag.name}_imag"][first_index + i, :]\
                                = diag.data_cache[i][:].imag
                    else:
                        for i in range(times_to_write):
                            self.group[diag.name][first_index + i, :]\
                                = diag.data_cache[i][:]

                diag.data_cache.clear()
