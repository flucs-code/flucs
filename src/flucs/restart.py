from __future__ import annotations

import shutil
import datetime
import cupy as cp
import numpy as np
import pathlib as pl
from netCDF4 import Dataset
from typing import TYPE_CHECKING

from flucs.input import InvalidFlucsInputFileError
from flucs.solvers import FlucsSolverState
if TYPE_CHECKING:
    from flucs.systems import FlucsSystem

class FlucsRestart:
    """
    Helper class that handles writing and reading restart files.
    """

    # Parent system
    system: FlucsSystem

    # Reading a restart file
    initial_path: pl.Path | None = None
    data: dict | None = None

    # Writing a restart file
    write_restart_file: bool = False
    write_path: pl.Path
    backup_path: pl.Path
    steps_until_write: int = 0

    # Flag to reset simulation
    reset_time: bool = False

    def __init__(self, system: FlucsSystem):
        self.system = system

        self._decide_initial_path()
        self._load_restart_data()
        self._setup_restart_output()

    def _decide_initial_path(self):
        """
        Decides the location of the restart file for loading initial data.

        """

        restart_if_exists = self.system.input["restart.restart_if_exists"]
        restart_from = self.system.input["restart.restart_from"]
        is_restart_from_specified = len(restart_from) > 0

        if not (restart_if_exists or is_restart_from_specified):
            # Not restarting
            return

        # Make sure only one way of restart-file initialisation is specified
        if restart_if_exists and is_restart_from_specified:
            raise InvalidFlucsInputFileError(
                "'restart_from' and 'restart_if_exists' cannot be specified"
                " simultaneously."
            )

        # Restart from default restart file
        if restart_if_exists:
            self.initial_path = self.system.input.io_path / "restart.nc"

            if not self.initial_path.exists():
                # Well, it does not exist
                self.initial_path = None
                return

        if is_restart_from_specified:
            self.initial_path = pl.Path(restart_from).expanduser()

            # Deal with relative paths
            if not self.initial_path.is_absolute():
                self.initial_path = (
                    self.system.input.io_path / restart_from
                ).resolve()

            if not self.initial_path.exists():
                raise InvalidFlucsInputFileError(
                    f"The restart_from file {self.initial_path}"
                    " cannot be found."
                )

        print(f"Restarting from file: {self.initial_path}")

    def _load_restart_data(self) -> None:
        """
        Load restart array data from self.initial_path and stores a dict in
        self.data that is identical to that returned by
        FlucsSystem._get_restart_data().

        The structure of the data dict is as follows:
        {
            "<var_name>": {
                "data": np.ndarray,  # NumPy arrays (host)
                "dimension_names": (<dim1>, <dim2>, ...),  # tuple[str]
            },
            ...
        }

        """

        if self.initial_path is None:
            # Nothing to do
            return

        self.data = {}
        system = self.system

        with Dataset(self.initial_path, "r") as ds:
            # Set system's time variables to continue from the restart file
            system.init_time = self.system.float(ds.variables["current_time"][...]) if \
                                    not system.input["restart.reset_time"] else 0.0
            system.init_dt = self.system.float(ds.variables["current_dt"][...])
            system.final_time = (
                system.init_time
                + self.system.float(system.input["time.tfinal"])
            )

            # Load all the restart data
            var_names = [
                v for v in ds.variables.keys()
                if v not in {"current_time", "current_dt"}
            ]

            for name in var_names:
                # Imaginary part handled simulatneously with real part
                if name.endswith("_imag"):
                    continue

                # Complex arrays stored as <base>_real and <base>_imag
                if name.endswith("_real"):
                    base_name = name.rstrip("_real")
                    imag_name = base_name + "_imag"

                    v_r = ds.variables[name]
                    if imag_name in ds.variables:
                        v_i = ds.variables[imag_name]
                        real = np.asarray(v_r[...])
                        imag = np.asarray(v_i[...])
                        data = real + 1j * imag
                        dims = tuple(v_r.dimensions)
                        self.data[base_name] = {
                            "data": data, "dimension_names": dims
                        }
                    else:
                        # No matching imag: treat as a regular real-valued
                        # variable with its own name
                        data = np.asarray(v_r[...])
                        dims = tuple(v_r.dimensions)
                        self.data[name] = {
                            "data": data, "dimension_names": dims
                        }
                    continue

                # Regular real-valued array
                var = ds.variables[name]
                data = np.asarray(var[...])
                dims = tuple(var.dimensions)
                self.data[name] = {"data": data, "dimension_names": dims}

    def _setup_restart_output(self):
        """ Gets the FlucsRestart ready to write restart data. """

        system_input = self.system.input

        self.write_restart_file = system_input["restart.write_restart_file"]

        if not self.write_restart_file:
            # Nothing to do here
            return

        self.write_path = system_input.io_path / "restart.nc"
        self.backup_path = (
            system_input.io_path / "restart.backup.nc"
        )

        if (self.write_path.exists()
                and not system_input["restart.restart_if_exists"]):
            raise InvalidFlucsInputFileError(
                "You must remove existing 'restart.nc' manually if write_restart_file"
                "is 'True' but restart_if_exists is 'False'."
            )

    def write_restart(self, force: bool = False) -> None:
        """
        Executes writing restart data if necessary.
        Must be called every time step to work properly.

        Parameters
        ----------
        force : bool
            If force is True, the restart data is written at that timestep.
        """

        if not self.write_restart_file:
            # Nothing to do here
            return

        if not self.system.solver.state == FlucsSolverState.RUNNING:
            return

        if force:
            # Write the data regardless of the time step we are on
            self._write_restart_data()
            return

        # Check whether its time to write
        self.steps_until_write -= 1
        if self.steps_until_write > 0:
            return

        # Reset the counter and write the data
        self.steps_until_write = self.system.input["restart.write_steps"]
        self._write_restart_data()

    def _backup_restart_file(self):
        """
        Called before starting to write to a restart file.
        Copies the old restart file (if it exists) to a backup file.
        If the backup file already exists, it is deleted.

        """

        if not self.write_path.exists():
            return

        if self.backup_path.exists():
            self.backup_path.unlink()

        shutil.move(self.write_path, self.backup_path)

    def _write_restart_data(self) -> None:
        """
        Writes restart data to netCDF files, rotating old and new files
        as necessary.
        """

        # Backup restart file
        self._backup_restart_file()

        # Get restart data
        restart_data = self.system.get_restart_data()

        # Set precision for netCDF variables
        precision = "f4" if self.system.float is np.float32 else "f8"

        # Write to temporary file
        with Dataset(self.write_path, "w", format="NETCDF4") as ds:

            # Set file attributes
            ds.setncattr(
                "created",
                datetime.datetime.now(datetime.timezone.utc).isoformat()
            )
            ds.setncattr("location", str(self.write_path.parent))
            ds.setncattr("type", str("flucs_restart"))

            # Add input file as a string
            ds.setncattr("input_file", str(self.system.input))

            # Scalar values
            ds.createVariable("current_time", precision, ())[...] =\
                self.system.float(self.system.current_time)

            ds.createVariable("current_dt", precision, ())[...] =\
                self.system.float(self.system.current_dt)

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

        # Remove backup file after successful write
        if self.backup_path.exists():
            self.backup_path.unlink()

    @staticmethod
    def reconstruct_input_from_restart(restart_file_path: str | pl.Path,
                                       io_path: str | pl.Path) -> None:
        """
        Reconstructs an input file from a restart file.

        Parameters
        ----------
        filepath: str | Path
            Path to the restart file.

        io_path: str | Path
            Path where the reconstructed input file will be written.

        """

        # Check supplied path
        restart_file_path = pl.Path(restart_file_path).expanduser().resolve()

        if not restart_file_path.exists():
            raise FileNotFoundError(
                f"Restart file not found: {restart_file_path}"
            )

        # Get input file from restart file
        with Dataset(restart_file_path, "r") as ds:
            if getattr(ds, "type", None) != "flucs_restart":
                raise ValueError(
                    f"File {restart_file_path} is not a restart file."
                )
            try:
                input_file = ds.getncattr("input_file")
            except Exception as e:
                raise ValueError(
                    f"Restart file {restart_file_path} does not contain"
                    " an input file stored as a string."
                ) from e

        # Check whether an input file of the same name already exists
        input_file_path = pl.Path(io_path) / "input.toml"
        if input_file_path.exists():
            raise FileExistsError(
                f"Input file already exists: {input_file_path}"
            )

        input_file_path.write_text(input_file)
        print(f"Reconstructed input file: {input_file_path}")
