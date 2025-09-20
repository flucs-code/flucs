from __future__ import annotations
from typing import TYPE_CHECKING
import heapq
import numpy as np
from netCDF4 import Dataset, Group
from flucs.solvers import FlucsSolverState

if TYPE_CHECKING:
    from flucs.output import FlucsDiagnostic
    from flucs.systems import FlucsSystem

class FlucsOutput:
    """Deals with a single output file. A FlucsSystem typically has several
    FlucsOuputs that handle different kinds of diagnostics

    """
    name: str
    filename: str
    save_steps: int
    next_save: int

    # If true, no netCDF4 file is created
    stdout_only: bool = False

    # Associated system
    system: FlucsSystem

    # List of diagnostics
    diagnostics: list[FlucsDiagnostic]

    # Cache of times at which data was saved
    time_cache: list[float]

    # Dataset for the netCDF4 file
    dataset: Dataset

    # netCDF4 group to write to
    group_name: str
    group: Group

    def __init__(self, name: str, system: FlucsSystem) -> None:
        self.name = name
        self.system = system
        self.filename = f"output.{name}.nc"

        self.diagnostics = []
        self.time_cache = []

        # Setup steps and diagnostics from input file and system
        self.next_save = 0
        self.save_steps = self.system.input[f"output.{self.name}.save_steps"]

        # if stdout_only, add diagnostics but do not create a netcdf4 file
        if ("stdout_only" in system.input[f"output.{name}"] and
                system.input[f"output.{name}.stdout_only"]):
            self.stdout_only = True

        self._add_diagnostics_from_input()

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

    def _add_diagnostics_from_input(self):
        """Adds all diagnostics from the input of the associated FlucsSystem"""
        for diag_name in self.system.input[f"output.{self.name}.diags"]:
            diag_to_add =\
                self.system.diags_dict[diag_name](system=self.system,
                                                  output=self)
            diag_to_add.output = self
            self.diagnostics.append(diag_to_add)

    def _setup_output_file(self):
        """Creates all the necessary groups, dimensions, and variables in the
        netCDF4 file where the output data will be written. Should be called
        before the main solver loop begins (but after any timing/optimisation
        routines).

        """
        if self.stdout_only:
            return

        with Dataset(self.filename, "r+", format="NETCDF4") as self.dataset:
            self._setup_group()

            # Check if we already have all necessary dimensions
            # for every diagnostic
            for diagnostic in self.diagnostics:
                for dim_name, dim_data in diagnostic.dimensions_dict.items():
                    dim_size = len(dim_data)

                    # If it exists, ensure it's the same
                    if (dim_name in self.group.dimensions and
                            dim_size != self.group.dimensions[dim_name].size):

                        print(
                            f"Dimension {dim_name} in output file"
                            f" {self.filename} has size"
                            f" {self.group.dimensions[dim_name].size}"
                            f" which differs from expected size"
                            f" {len(dim_data)} required by diagnostic"
                            f"  {diagnostic.name}!"
                            "\n"
                            "Therefore, diagnostic {diagnostic.name}"
                            "will not operate!")
                        return

                    self.group.createDimension(dim_name, dim_size)
                    dim_var = self.group.createVariable(dim_name,
                                                        "f4",
                                                        (dim_name,))
                    dim_var[:] = dim_data[:]

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

    def ready(self):
        """Sets up the diagnostic for running."""
        self.time_cache.clear()
        self.next_save = 0
        for diag in self.diagnostics:
            diag.ready()
            diag.data_cache.clear()

        if self.system.solver.state == FlucsSolverState.RUNNING:
            self._setup_output_file()

    def execute(self):
        """Executes each diagnostic. Does not save to disk."""
        for diag in self.diagnostics:
            diag.execute()

        self.next_save += self.save_steps
        self.time_cache.append(self.system.current_time)

    def write(self):
        """Saves any cached diagnostic data to disk and clears the cache.
        Saves data only if we are not timing.

        """
        if (self.stdout_only or
                self.system.solver.state != FlucsSolverState.RUNNING):
            return

        with Dataset(self.filename, "r+", format="NETCDF4") as self.dataset:
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
                    # Could probably rewrite with np.array
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

    def __lt__(self, other):
        if not isinstance(other, FlucsOutput):
            raise TypeError(f"Makes no sense to compare a FlucsOutput with a {type(other)}")
        return self.next_save < other.next_save
