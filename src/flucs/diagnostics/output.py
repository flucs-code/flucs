from __future__ import annotations
from typing import TYPE_CHECKING
import heapq
import numpy as np
from netCDF4 import Dataset, Group

if TYPE_CHECKING:
    from flucs.diagnostics.diagnostic import FlucsDiagnostic
    from flucs.systems.flucs_system import FlucsSystem

class FlucsOutput:
    """Deals with a single output file. A FlucsSystem typically has several
    FlucsOuputs that handle different kinds of diagnostics

    """
    name: str
    filename: str
    save_steps: int
    next_save: int

    # Associated system
    system: FlucsSystem

    # List of diagnostics
    diagnostics: list[FlucsDiagnostic]

    # Cache of times at which data was saved
    time_cache: list[float]

    # Handle to the output netCDF4 file
    dataset: Dataset

    # netCDF4 group to write to
    group_name: str
    group: Group

    def __init__(self, name: str, system: FlucsSystem) -> None:
        self.name = name
        self.system = system

        # Setup steps and diagnostics from input file and system
        self.next_save = 0
        self.save_steps = self.system.input[f"output.{self.name}.save_steps"]

        with Dataset(self.filename, "r+", format="NETCDF4") as self.dataset:
            self._setup_group()

            for diag_name in self.system.input[f"output.{self.name}.diags"]:
                diag_to_add = self.system.diags_dict[diag_name](self.system)
                self._add_diagnostic(diag_to_add)

    def _setup_group(self):
        dataset: Dataset = self.dataset
        if hasattr(self, "group_name"):
            # We are yet to initialise the group
            # Go through the file and pick group_name to be an integer that is
            # equal to the largest one found + 1
            group_number = max([-1] +
                               [int(name) for name in dataset.groups.keys()])
            group_number += 1

            self.group_name = str(group_number)
            dataset.createGroup(self.group_name)

        self.group = dataset.groups[self.group_name]

    def _add_diagnostic(self, diagnostic: FlucsDiagnostic):
        # Check if we already have all necessary dimensions
        for dim_name, dim_data in diagnostic.dimensions_dict:
            dim_size = len(dim_data)

            # If it exists, ensure it's the same
            if (dim_name in self.group.dimensions
                    and dim_size != self.group.dimensions[dim_name].size):

                print(
                    f"Dimension {dim_name} in output file"
                    f" {self.filename} has size"
                    f" {self.group.dimensions[dim_name].size}"
                    f" which differs from expected size {len(dim_data)}"
                    f" required by diagnostic {diagnostic.name}!"
                    "\n"
                    "Therefore, diagnostic {diagnostic.name}"
                    "will not operate!")
                return

            self.group.createDimension(dim_name, dim_size)
            dim_var = self.group.createVariable(dim_name, "f4", (dim_size, ))
            dim_var[:] = dim_data[:]

        # Create variable
        self.group.createVariable(diagnostic.name, "f4",
                                  ("time", ) + diagnostic.shape)

        diagnostic.output = self
        self.diagnostics.append(diagnostic)

    def ready(self):
        """Sets up the diagnostic for running."""
        for diag in self.diagnostics:
            diag.ready()

    def execute(self):
        """Executes each diagnostic. Does not save to disk."""
        for diag in self.diagnostics:
            diag.execute()

        self.next_save += self.save_steps
        self.time_cache.append(self.system.current_time)

    def write(self):
        """Saves any cached diagnostic data to disk and clears the cache."""

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
                for i in range(times_to_write):
                    self.group[diag.name][first_index + i, :]\
                        = diag.data_cache[i][:]

                diag.data_cache.clear()

    def __lt__(self, other):
        if not isinstance(other, FlucsOutput):
            raise TypeError(f"Makes no sense to compare a FlucsOutput with a {type(other)}")
        return self.next_save < other.next_save
