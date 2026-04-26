import argparse
import inspect
import pathlib as pl
from collections import OrderedDict
from collections.abc import Sequence
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import toml
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from netCDF4 import Dataset

import flucs


class FlucsPostProcessing:
    """
    Class that handles post-processing of output data.
    """

    # Postprocessing-specific attributes
    io_paths: list[pl.Path]
    output_files: list[str] | None
    _output_paths: dict[pl.Path, list[pl.Path]]
    save_directory: pl.Path | None
    _script_paths: list[tuple[int, str, pl.Path]]

    # Solver and system for the outputs
    solver_names: dict[pl.Path, str]
    system_names: dict[pl.Path, str]
    solver_types: dict[pl.Path, type]
    system_types: dict[pl.Path, type]

    # Formatting for printing
    _indent = 3 * " "

    def _get_solver_and_system_types(self) -> None:
        """
        Sets the solver and system types for each provided i/o directory based
        on its corresponding input file.
        """

        self.solver_names = {}
        self.system_names = {}
        self.solver_types = {}
        self.system_types = {}

        for io_path in self.io_paths:
            input_file_dict = toml.load(io_path / "input.toml")

            solver_name = input_file_dict["setup"]["solver"]
            system_name = input_file_dict["setup"]["system"]

            self.solver_names[io_path] = solver_name
            self.system_names[io_path] = system_name

            self.solver_types[io_path] = flucs.get_solver_type(solver_name)
            self.system_types[io_path] = flucs.get_system_type(system_name)

    def _get_script_paths(self) -> list[tuple[int, str, pl.Path]]:
        """
        Gathers the paths to the relevant postprocessing scripts for each
        solver and system used across all provided i/o directories and returns
        them in a stable integer-addressable order.
        """

        self._script_paths = []

        # Keep solver scripts ahead of system scripts in the printed order.
        ordered_types = []
        for flucs_types in (
            self.solver_types.values(),
            self.system_types.values(),
        ):
            for flucs_type in sorted(
                set(flucs_types), key=lambda f: f.__name__.lower()
            ):
                if flucs_type not in ordered_types:
                    ordered_types.append(flucs_type)

        # Find postprocessing directory and collect Python scripts.
        for flucs_type in ordered_types:
            type_name = flucs_type.__name__
            path = inspect.getfile(flucs_type)

            scripts_dir = pl.Path(path).parent / "postprocessing"
            if scripts_dir.exists():
                for script in sorted(
                    scripts_dir.glob("*.py"), key=lambda p: p.name.lower()
                ):
                    self._script_paths.append(
                        (len(self._script_paths), type_name, pl.Path(script))
                    )

        return self._script_paths

    def get_script_path(self, script_integer: int) -> pl.Path:
        """
        Returns the postprocessing script corresponding to a given integer.
        """

        script_paths = self._get_script_paths()

        if not script_paths:
            raise ValueError(
                "No postprocessing scripts are available for the specified "
                "i/o directory."
            )

        if not 0 <= script_integer < len(script_paths):
            raise ValueError(
                f"Invalid postprocessing script integer: {script_integer}. "
                f"Expected a value between 0 and {len(script_paths) - 1}."
            )

        return script_paths[script_integer][2]

    def list_script_paths(self) -> None:
        """
        Prints information about the postprocessing scripts to the
        standard output for all solver/system types referenced by the
        provided i/o directories.
        """

        script_paths = self._get_script_paths()

        print("Available postprocessing scripts:")
        if not script_paths:
            print(f"{self._indent}None")
        else:
            integer_width = len(str(len(script_paths) - 1))
            current_type = None
            for integer, type_name, path in script_paths:
                if type_name != current_type:
                    print(f"{self._indent}{type_name}:")
                    current_type = type_name
                label = f"[{integer:>{integer_width}}]"
                print(f"{2 * self._indent}{label} {path}")
        print(
            "To run a specific script: 'flucs -p <integer> <script arguments>'."
        )

    def _get_output_paths(self) -> dict[pl.Path, list[pl.Path]]:
        """
        For each i/o directory, gathers the paths to the output files
        specified by self.output_files.

        Returns
        -------
        dict[pl.Path, list[pl.Path]]
            Mapping io_path -> list of filepaths corresponding to
            self.output_files.
        """

        output_paths: dict[pl.Path, list[pl.Path]] = {}

        for io_path in self.io_paths:
            matched_paths = []
            seen = set()

            for output_file in self.output_files or []:
                for path in sorted(io_path.glob(output_file)):
                    if not path.is_file():
                        continue
                    if path in seen:
                        continue
                    seen.add(path)
                    matched_paths.append(path)

            output_paths[io_path] = matched_paths

        return output_paths

    @staticmethod
    def get_netcdf_variables(
        nc_path: pl.Path, ignore=None
    ) -> dict[str, list[int]]:
        """
        Given a netCDF filepath, returns a mapping of variable names to the
        list of groups that they appear in.

        Parameters
        ----------
        nc_path : pl.Path
            Path to the NetCDF file.
        ignore : Iterable[str] | None
            Variable names to ignore.

        Returns
        -------
        dict[str, list[int]]
            Variable name -> sorted list of groups.
        """

        netcdf_variables: dict[str, list[int]] = {}

        # Helper function to add variable and group number
        def _add(name: str, grp_number: int) -> None:
            if ignore is not None and name in ignore:
                return
            netcdf_variables.setdefault(name, [])
            if grp_number not in netcdf_variables[name]:
                netcdf_variables[name].append(grp_number)

        # Helper function to add variables from nested groups
        def _add_nested(grp, grp_number, base_name="") -> None:
            for var_name in grp.variables.keys():
                _add(base_name + var_name, grp_number)

            for subgrp_name, subgrp in grp.groups.items():
                _add_nested(
                    subgrp, grp_number, base_name=base_name + f"{subgrp_name}/"
                )

        # Iterate over groups in netCDF file
        with Dataset(pl.Path(nc_path), "r", format="NETCDF4") as ds:
            for grp_name, grp in ds.groups.items():
                grp_number = int(grp_name)
                _add_nested(grp, grp_number)

        return {v: sorted(ids) for v, ids in netcdf_variables.items()}

    def _get_all_netcdf_variables(
        self, ignore=None
    ) -> dict[pl.Path, dict[pl.Path, dict[str, list[int]]]]:
        """
        For each i/o directory, collect the variables present in the netCDF file
        specified by self.output_files, and the groups that they appear in.

        Parameters
        ----------
        ignore : Iterable[str] | None
            Variable names to ignore.

        Returns
        -------
        dict[pl.Path, dict[pl.Path, dict[str, list[int]]]]
            Mapping io_path -> {nc_path -> {variable: [group_ids], ... }, ... }
        """

        if self.output_files is None:
            raise ValueError(
                "'output_files' must be set to derive netCDF paths from "
                "i/o directories."
            )

        # Get netCDF paths from all output paths
        netcdf_paths = {
            io_path: [path for path in paths if path.suffix == ".nc"]
            for io_path, paths in self._output_paths.items()
        }

        result: dict[pl.Path, dict[pl.Path, dict[str, list[int]]]] = {}
        for io_path, nc_paths in netcdf_paths.items():
            result[io_path] = {}

            for nc_path in nc_paths:
                result[io_path][nc_path] = self.get_netcdf_variables(
                    nc_path, ignore=ignore
                )

        return result

    def list_netcdf_variables(
        self, ignore=("time", "dt", "input_file")
    ) -> None:
        """
        Prints the available netCDF variables to the standard output for each
        of the provided i/o directories given a specific output type.
        """

        print("Available netCDF variables:")
        netcdf_variables = self._get_all_netcdf_variables(ignore=ignore)
        for io_path, file_map in netcdf_variables.items():
            all_variables = set()

            for variables_dict in file_map.values():
                all_variables.update(variables_dict.keys())

            listed_variables = []
            unique_variables = set()

            for variable in sorted(all_variables):
                if variable.startswith(("realspace_data/", "fourier_data/")):
                    variable = variable.rsplit("/", 1)[0]

                if variable in unique_variables:
                    continue

                unique_variables.add(variable)
                listed_variables.append(variable)

            print(rf"{self._indent}{io_path}: {listed_variables}")

    def get_valid_netcdf_paths(self, variable: str) -> list[pl.Path]:
        """
        Return the netCDF filepaths that contain a given variable.

        Returns
        -------
        list[pathlib.Path]
            Filepaths that contain the given variable.
        """

        mapping = self._get_all_netcdf_variables(ignore=())

        # Check files for variable
        found = []
        missing = []

        for io_path, file_map in mapping.items():
            io_found = False

            for nc_path, variables in file_map.items():
                if variables.get(variable):
                    found.append(nc_path)
                    io_found = True

            if not io_found:
                missing.extend(file_map.keys())

        if missing:
            print(f"Variable '{variable}' not found in:")
            for path in missing:
                print(f"{self._indent}{path}")

        return found

    def load_netcdf_variable(
        self,
        nc_path: pl.Path,
        variable: str,
        fill_value: float = np.nan,
        group: int | None = None,
    ):
        """
        Load a variable from a netCDF file.

        Time-dependent variables are (by default) concatenated across all groups
        along time (zeroth axis). Groups missing the variable are filled with
        'fill_value'.

        Time-independent variables are (by default) loaded from the latest non-
        empty group.

        Parameters
        ----------
        nc_path : pathlib.Path
            Path to the netCDF file to read.
        variable : str
            Name of the variable to load.
        fill_value : float
            Value to use for groups that do not contain 'variable'.
        group : int | None
            If specified, load data only from this output group. If None,
            time-dependent variables are concatenated across all groups, while
            time-independent variables are loaded from the latest non-empty
            group.

        Returns
        -------
        tuple
            (values, boundary_indices, dims_dict) where
            - values is an np.ndarray with shape (sum(time_lengths), ...)
              after concatenation across groups
            - boundary_indices is a list of integer indices marking the
              boundaries between groups in the concatenated time axis
            - dims_dicts is a list of ordered dictionaries of the variable's
              dimensions and their data. The length of dims_dicts is the
              total number of restart groups.
        """

        # Helper function to get variable from group
        def _get_var(grp, name: str) -> Any | None:
            # Group variables
            if "/" not in name:
                if name in grp.variables:
                    return grp.variables[name]
                return None

            # Diagnostic variables
            subgrp, var = name.split("/", 1)
            if subgrp in grp.groups:
                return _get_var(grp[subgrp], var)
            return None

        # Read data from netCDF file
        with Dataset(str(nc_path), "r", format="NETCDF4") as ds:
            # Get output groups sorted by group id
            groups = [(int(name), grp) for name, grp in ds.groups.items()]

            # Check whether the groups are in the correct order
            grp_numbers = [grp_number for grp_number, _ in groups]
            if grp_numbers != sorted(grp_numbers):
                raise ValueError(
                    "Output groups are not in order; check netCDF file"
                )

            # Get variable shapes to determine fill values
            sample_var = None
            for _, grp in groups:
                sample_var = _get_var(grp, variable)
                if sample_var is not None:
                    break

            if sample_var is None:
                raise ValueError(
                    f"Variable '{variable}' not found in any group of {nc_path}"
                )

            time_dependent = (sample_var.dimensions[:1] == ("time",))
            res_shape = tuple(
                size
                for dim, size in zip(sample_var.dimensions, sample_var.shape)
                if dim != "time"
            )
            var_dtype = sample_var.dtype

            # Either read specified group, or all of them
            groups_to_read = ([groups[group]]if group is not None else groups)

            # Set up lists
            group_data = []
            boundaries = []
            dims_dicts = []

            # Get data from each group
            for grp_number, grp in groups_to_read:
                time_length = int(grp.variables["time"].shape[0])
                boundaries.append(time_length)
                dims_dicts.append(OrderedDict())

                fill_shape = (
                    (time_length, *res_shape)
                    if time_dependent
                    else res_shape
                )

                var_obj = _get_var(grp, variable)
                if var_obj is not None:
                    arr = np.asarray(var_obj[:]).astype(var_dtype, copy=False)
                    group_data.append(arr)

                    # Add dimensions
                    for dim in var_obj.dimensions:
                        if dim == "time":
                            continue
                        dims_dicts[-1][dim] = np.asarray(
                            var_obj.group()[dim][:]
                        )
                else:
                    # Fill missing group segment with zeros of appropriate shape
                    group_data.append(
                        np.full(
                            fill_shape,
                            fill_value,
                            dtype=var_dtype,
                        )
                    )

        # Determine how to handle time axis, if present
        if group is not None:
            return group_data[0], [], dims_dicts
        
        elif time_dependent:
            values = np.concatenate(group_data, axis=0)
            boundary_indices = list(np.cumsum(boundaries)[:-1])

            return values, boundary_indices, dims_dicts
    
        else:
            latest = next(
                i
                for i in range(len(groups_to_read) - 1, -1, -1)
                if boundaries[i] > 0
            )
            return group_data[latest], [], [dims_dicts[latest]]


    def load_netcdf_variable_complex(
        self,
        nc_path: pl.Path,
        variable: str,
        fill_value: complex = np.nan + 1j * np.nan,
        group: int | None = None,
    ):
        """
        Load a complex variable stored as '<variable>_real' and
        '<variable>_imag' from a netCDF file.

        This is a thin wrapper around load_netcdf_variable and follows the same
        group-selection rules.

        Parameters
        ----------
        nc_path : pathlib.Path
            Path to the netCDF file to read.
        variable : str
            Base name of the complex variable.
        fill_value : complex
            Value to use for groups that do not contain the variable.
        group : int | None
            If specified, load data only from this output group. If None,
            behaviour matches load_netcdf_variable.

        Returns
        -------
        tuple
            (values, boundary_indices, dims_dicts), matching the return
            signature of load_netcdf_variable.
        """

        # Load data
        real, boundary_indices_real, dims_dicts_real = (
            self.load_netcdf_variable(
                nc_path,
                f"{variable}_real",
                fill_value=np.real(fill_value),
                group=group,
            )
        )

        imag, boundary_indices_imag, dims_dicts_imag = (
            self.load_netcdf_variable(
                nc_path,
                f"{variable}_imag",
                fill_value=np.imag(fill_value),
                group=group,
            )
        )

        # Quick consistency check
        if (boundary_indices_real != boundary_indices_imag) or (
            len(dims_dicts_real) != len(dims_dicts_imag)
        ):
            raise ValueError(
                f"Real and imaginary parts of complex variable "
                f"'{variable}' have mismatched dimensions."
            )

        # Combine into complex object
        values = real + 1j * imag

        return values, boundary_indices_real, dims_dicts_real

    def save(
        self,
        obj,
        *,
        name: str,
        suffix: str,
        conflict_strategy: Literal[
            "overwrite", "preserve", "error"
        ] = "overwrite",
        save_kwargs: dict | None = None,
    ) -> None:
        """
        Save the result of post-processing to 'self.save_directory'.

        Parameters
        ----------
        obj : Any
            The object to save. Its type will determine the save handler used.
        name : str | None
            The desired filename stem (without suffix).
        suffix : str | None
            File extension.
        conflict_strategy : {"overwrite", "preserve", "error"}
            Behaviour when the target save filepath already exists.
        save_kwargs : dict | None
            Arguments forwarded to the type-specific save function.
        """

        # Do nothing is there is no save directory
        if self.save_directory is None:
            return None

        # Validate conflict strategy
        if conflict_strategy not in ("overwrite", "preserve", "error"):
            raise ValueError("Invalid value for 'conflict_strategy'.")

        # Ensure directory exists
        directory = self.save_directory
        directory.mkdir(parents=True, exist_ok=True)

        # Construct base save filepath
        ext = f".{suffix.lstrip('.')}" if suffix else ""
        base_save_filepath = directory / f"{name}{ext}"

        # Handle conflict strategies
        if base_save_filepath.exists():
            if conflict_strategy == "overwrite":
                pass
            elif conflict_strategy == "preserve":
                return
            elif conflict_strategy == "error":
                raise OSError(
                    f"Target save path already exists: {base_save_filepath}"
                )

        # Call type-specific save function
        if isinstance(obj, Figure):
            self._save_matplotlib_figure(
                fig=obj,
                save_filepath=base_save_filepath,
                save_kwargs=save_kwargs,
            )
        elif isinstance(obj, Axes):
            self._save_matplotlib_figure(
                fig=obj.figure,
                save_filepath=base_save_filepath,
                save_kwargs=save_kwargs,
            )
        else:
            name = type(obj).__name__
            raise NotImplementedError(
                f"Saving objects of type '{name}' is not yet implemented."
            )

        return

    def _save_matplotlib_figure(
        self,
        fig: Figure,
        save_filepath: pl.Path,
        save_kwargs: dict | None,
    ) -> None:
        """
        Save a single Matplotlib figure to self.save_directory.
        """

        # Parse kwargs
        kwargs = dict(save_kwargs or {})
        close_fig = bool(kwargs.pop("close", False))

        # Save each figure
        fig.savefig(save_filepath, **kwargs)

        if close_fig:
            plt.close(fig)

        return

    @staticmethod
    def parser() -> argparse.ArgumentParser:
        """
        A common parser for postprocessing scripts that use FlucsPostProcessing.
        """

        parser = argparse.ArgumentParser(add_help=False)

        parser.add_argument(
            "--io_path",
            "-io",
            nargs="+",
            type=str,
            default=pl.Path.cwd(),
            required=False,
            help=(
                "Paths to the i/o directories, which must contain "
                "'input.toml'. If no path is specified, will assume the "
                "current working directory."
            ),
        )

        parser.add_argument(
            "--save_directory",
            "-s",
            nargs="?",
            type=lambda s: pl.Path(s).expanduser().resolve(),
            const=pl.Path.cwd(),
            default=None,
            help=(
                "Directory to which postprocessing outputs are saved. If "
                "omitted, nothing is saved. If no path is specified, will "
                "assume the current working directory."
            ),
        )

        return parser

    def __init__(
        self,
        io_paths: pl.Path | Sequence[pl.Path],
        *,
        save_directory: pl.Path | None = None,
        output_files: str | Sequence[str] | None = None,
        constraint: Literal["none", "solver", "system", "both"] = "none",
        quiet: bool = False,
    ) -> None:
        """
        Given one or more i/o directories, sets up the relevant paths, and
        resolves the solver and system types referenced by their input files.

        Parameters
        ----------
        io_paths : pl.Path | Sequence[pl.Path]
            Path or paths to i/o directories containing 'input.toml'.

        save_directory : pl.Path | None
            Optional path where to save results. If None, nothing will be saved.

        output_files: str | Sequence[str] | None
            Output files being analysed for this instance of post-processing.
            If None, no specific output type is assumed.

        constraint : {"none", "solver", "system", "both"}
            Constraint on mixing solvers/systems across provided i/o
            directories. If "solver", all solvers must match. If "system", all
            systems must match. If "both", both solvers and systems must match.
        quiet : bool
            Whether to suppress the short summary printed when the
            postprocessing object is initialised.
        """

        # Parse io_paths input
        if isinstance(io_paths, (str, pl.Path)):
            io_paths = [io_paths]

        self.io_paths = []
        for path in io_paths:
            resolved_path = pl.Path(path).expanduser().resolve()
            input_file = resolved_path / "input.toml"
            if not input_file.exists():
                raise ValueError(f"Path {path} is not a valid i/o directory.")
            self.io_paths.append(resolved_path)

        # Set output files
        if isinstance(output_files, str):
            self.output_files = [output_files]
        else:
            self.output_files = (
                list(output_files) if output_files is not None else None
            )

        self._output_paths = self._get_output_paths()

        # Set save directory
        self.save_directory = (
            pl.Path(save_directory).expanduser().resolve()
            if save_directory
            else None
        )

        # Determine solver and system types across all i/o directories
        self._get_solver_and_system_types()

        # Enforce constraint across provided inputs
        if constraint not in ("none", "solver", "system", "both"):
            raise ValueError("Invalid value for 'constraint'.")

        solver_types = set(self.solver_types.values())
        system_types = set(self.system_types.values())

        if constraint in ("solver", "both") and len(solver_types) > 1:
            raise ValueError(
                "All i/o directories must contain output from the same "
                "solver when 'constraint' is 'solver' or 'both'."
            )
        if constraint in ("system", "both") and len(system_types) > 1:
            raise ValueError(
                "All i/o directories must contain output from the same "
                "system when 'constraint' is 'system' or 'both'."
            )

        if not quiet:
            print(
                f"FlucsPostProcessing "
                f"({len(self.io_paths)}, "
                f"{self.output_files}, "
                f"{self.save_directory})"
            )
