import toml
import flucs
import inspect
import argparse
import numpy as np  
import pathlib as pl 
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from typing import Sequence, Literal, Any

class FlucsPostProcessing:
    """
    Class that handles post-processing of output data.
    """

    # Postprocessing-specific attributes
    io_paths: list[pl.Path] 
    output_file: str | None
    save_directory: pl.Path | None
    _script_paths: dict[str, list[pl.Path]]

    # Solver and system for the outputs
    _solver_names: dict[pl.Path, str]
    _system_names: dict[pl.Path, str]
    _solver_types: dict[pl.Path, type]
    _system_types: dict[pl.Path, type]

    # Formatting for printing
    _indent = 3 * " "

    def _get_solver_and_system_types(self) -> None:
        """
        Sets the solver and system types for each provided i/o directory based
        on its corresponding input file.
        """

        self._solver_names = {}
        self._system_names = {}
        self._solver_types = {}
        self._system_types = {}

        for io_path in self.io_paths:
            input_file_dict = toml.load(io_path / "input.toml")

            solver_name = input_file_dict["setup"]["solver"]
            system_name = input_file_dict["setup"]["system"]

            self._solver_names[io_path] = solver_name
            self._system_names[io_path] = system_name

            self._solver_types[io_path] = flucs.get_solver_type(solver_name)
            self._system_types[io_path] = flucs.get_system_type(system_name)


    def _get_script_paths(self) -> None:
        """
        Gathers the paths to the relevant postprocessing scripts for each
        solver and system used across all provided i/o directories.
        """

        self._script_paths = {}

        # Collect unique types across all io_paths
        unique_types: set[type] = set()
        unique_types.update(self._solver_types.values())
        unique_types.update(self._system_types.values())

        for flucs_type in unique_types:
            name = flucs_type.__name__
            path = inspect.getfile(flucs_type)

            self._script_paths[name] = []
            scripts_dir = pl.Path(path).parent / "postprocessing"
            if scripts_dir.exists():
                for script in scripts_dir.glob("*.py"):
                    self._script_paths[name].append(pl.Path(script))


    def list_script_paths(self) -> None:
        """
        Prints information about the postprocessing scripts to the standard output
        for all solver/system types referenced by the provided i/o directories.
        """

        self._get_script_paths()

        print("Available postprocessing scripts:")
        for type_name, paths in self._script_paths.items():
            print(f"{self._indent}{type_name}:")
            for path in paths:
                print(f"{2*self._indent}{path}")
        print("For information on a specific script, run '<script path> --help'.")


    @staticmethod
    def get_netcdf_variables(nc_path: pl.Path, ignore=("time", "dt")) -> dict[str, list[int]]:
        """
        Given a netCDF filepath, returns a mapping of variable names to the
        list of groups that they appear in.

        Parameters
        ----------
        nc_path : pl.Path
            Path to the NetCDF file.
        ignore : Iterable[str]
            Variable names to ignore. Defaults to ('time', 'dt').

        Returns
        -------
        dict[str, list[int]]
            Variable name -> sorted list of groups.
        """

        netcdf_variables: dict[str, list[int]] = {}

        # Helper function to add variable and group number
        def _add(name: str, grp_number: int) -> None:
            if name in ignore:
                return
            netcdf_variables.setdefault(name, [])
            if grp_number not in netcdf_variables[name]:
                netcdf_variables[name].append(grp_number)

        # Iterate over groups in netCDF file
        with Dataset(pl.Path(nc_path), "r", format="NETCDF4") as ds:
            for grp_name, grp in ds.groups.items():
                grp_number = int(grp_name)

                # Group variables
                for var_name in grp.variables.keys():
                    _add(var_name, grp_number)

                # Diagnostic variables
                for diag_name, diag_grp in grp.groups.items():
                    for var_name in diag_grp.variables.keys():
                        _add(f"{diag_name}/{var_name}", grp_number)

        return {v: sorted(ids) for v, ids in netcdf_variables.items()}

    def _get_all_netcdf_variables(self, ignore=("time", "dt")) -> dict[str, dict[str, list[int]]]:
        """
        For each i/o directory, collect the variables present in the netCDF file 
        specified by self.output_file, and the groups that they appear in.

        Parameters
        ----------
        ignore : Iterable[str]
            Variable names to ignore. Defaults to ('time', 'dt').

        Returns
        -------
        dict[str, dict[str, list[int]]]
            Mapping io_path (as a string) -> {variable: [group_ids], ... }
        """

        if self.output_file is None:
            raise ValueError("'output_file' must be set to derive netCDF paths from i/o directories.")

        result: dict[str, dict[str, list[int]]] = {}
        for io_path in self.io_paths:
            nc_path = io_path / self.output_file
            if not nc_path.exists():
                result[str(io_path)] = {}
                continue
            result[str(io_path)] = FlucsPostProcessing.get_netcdf_variables(nc_path, ignore=ignore)

        return result
    

    def list_netcdf_variables(self, ignore=("time", "dt")) -> None:
        """
        Prints the available netCDF variables to the standard output for each 
        of the provided i/o directories given a specific output type.
        """

        print(f"Available netCDF variables:")
        netcdf_variables = self._get_all_netcdf_variables(ignore=ignore)
        for io_path, variables_dict in netcdf_variables.items():
            print(rf"{self._indent}{io_path}: {sorted(variables_dict.keys())}")


    def get_valid_files(self, variable: str) -> list[pl.Path]:
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

        for io_path, variables in mapping.items():
            nc_path = pl.Path(io_path) / self.output_file
            if variable in variables and variables[variable]:
                found.append(nc_path)
            else:
                missing.append(nc_path)
        # Report files that are missing the variable
        if missing:
            print(f"Variable '{variable}' not found in:")
            for path in missing:
                print(f"{self._indent}{path}")

        return found


    def load_netcdf_variable(self, nc_path: pl.Path, variable: str, fill_value: float = np.nan):
        """
        Load a variable across all groups in a netCDF file and concatenate
        along time (zeroth axis). Groups missing the variable are filled with
        'fill_value'.

        Parameters
        ----------
        nc_path : pathlib.Path
            Path to the netCDF file to read.
        variable : str
            Name of the variable to load.
        fill_value : float
            Value to use for groups that do not contain 'variable'.

        Returns
        -------
        tuple
            (values, boundary_indices) where
            - values is an np.ndarray with shape (sum(time_lengths), ...) after
              concatenation across groups
            - boundary_indices is a list of integer indices marking the boundaries
              between groups in the concatenated time axis
        """

        # Helper function to get variable from group
        def _get_var(grp, name: str) -> Any | None:

            # Group variables
            if "/" not in name:
                if name in grp.variables:
                    return grp.variables[name]
                return None

            # Diagnostic variables
            diag, var = name.split("/", 1)
            if diag in grp.groups and var in grp.groups[diag].variables:
                return grp.groups[diag].variables[var]
            return None

        # Read data from netCDF file
        with Dataset(str(nc_path), "r", format="NETCDF4") as ds:

            # Get output groups sorted by group id
            groups = [(int(name), grp) for name, grp in ds.groups.items()]

            # Check whether the groups are in the correct order
            grp_numbers = [grp_number for grp_number, _ in groups]
            if grp_numbers != sorted(grp_numbers):
                raise ValueError("Output groups are not in order; check netCDF file.")

            # Determine residual shape and dtype of output variable
            var_iter = (
                (grp_number, var_obj)
                for grp_number, grp in groups
                if (var_obj := _get_var(grp, variable)) is not None
            )
            try:
                grp_number, var = next(var_iter)
            except StopIteration:
                raise ValueError(f"Variable '{variable}' not found in any group of {nc_path}")
            
            res_shape = var.shape[1:]
            var_dtype = var.dtype

            # Get data from each group
            group_data = []
            boundaries = []
            for grp_number, grp in groups:
                time_length = int(grp.variables["time"].shape[0])
                boundaries.append(time_length)

                var_obj = _get_var(grp, variable)
                if var_obj is not None:
                    arr = np.asarray(var_obj[:])
                    if arr.shape[0] != time_length:
                        raise ValueError(
                            f"Time dimension mistmatch for variable '{variable}' in group "
                            f"{grp_number} (has {arr.shape} but expected {time_length}). "
                        )
                    group_data.append(arr.astype(var_dtype, copy=False))
                else:
                    # Fill missing group segment with zeros of appropriate shape
                    group_data.append(np.full((time_length, *res_shape), fill_value, dtype=var_dtype))

        # Concatenate along time (zeroth axis)
        values = np.concatenate(group_data, axis=0)
        boundary_indices = list(np.cumsum(boundaries)[:-1])

        return values, boundary_indices


    def save(
        self,
        obj,
        *,
        name: str,
        suffix: str,
        conflict_strategy: Literal["overwrite", "error"] = "overwrite",
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
        conflict_strategy : {"overwrite", "error"}
            Behaviour when the target save filepath already exists.
        save_kwargs : dict | None
            Arguments forwarded to the type-specific save function.
        """

        # Do nothing is there is no save directory
        if self.save_directory is None:
            return None
        
        # Validate conflict strategy
        if conflict_strategy not in ("overwrite", "error"):
            raise ValueError("Invalid value for 'conflict_strategy'.")

        # Ensure directory exists
        directory = self.save_directory
        directory.mkdir(parents=True, exist_ok=True)

        # Construct base save filepath
        ext = f".{suffix.lstrip('.')}" if suffix else ""
        base_save_filepath = directory / f"{name}{ext}"

        # If conflict_strategy = 'error', raise an error if any existing files
        # match '{name}_*{ext}'.
        if conflict_strategy == "error":
            pattern = f"{name}_*{ext}"
            if any(directory.glob(pattern)):
                raise OSError(
                    f"Conflicting files matching '{pattern}' already exist: {directory}"
                )
        
        # Call type-specific save function
        if isinstance(obj, (Figure, Axes)):
            self._save_matplotlib_figures(
                base_save_filepath=base_save_filepath,
                save_kwargs=save_kwargs,
            )
        else:
            raise NotImplementedError(
                f"Saving objects of type '{type(obj).__name__}' is not yet implemented."
            )

        return
    
    def _save_matplotlib_figures(
        self,
        base_save_filepath: pl.Path,
        save_kwargs: dict | None,
    ) -> None:
        """
        Save all open Matplotlib figures to self.save_directory.
        """

        # Get all figures to save
        fignums = plt.get_fignums()
        if not fignums:
            raise RuntimeError("No Matplotlib figures available to save.")

        # Parse kwargs
        kwargs = dict(save_kwargs or {})
        close_fig = bool(kwargs.pop("close", False))

        # Save each figure
        f = base_save_filepath
        for fignum in fignums:

            fig = plt.figure(fignum)
            number = f"_{int(fig.number):03d}" if len(fignums) > 1 else ""
            filename = f"{f.with_suffix('').name}{number}{f.suffix}"
            save_path = f.parent / filename

            fig.savefig(save_path, **kwargs)
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
            "--io_path", "-io",
            nargs='+',
            type=str,
            default=pl.Path.cwd(),
            required=False,
            help="Paths to the i/o directories, which must contain 'input.toml'. "
                "If no path is specified, will assume the current working directory.",
        )

        parser.add_argument(
            "--save_directory", "-s",
            nargs="?",
            type=lambda s: pl.Path(s).expanduser().resolve(),
            const=pl.Path.cwd(),
            default=None,
            help=(
                "Directory to which postprocessing outputs are saved. If omitted, nothing "
                "is saved. If no path is specified, will assume the current working directory."
            ),
        )

        return parser

    
    def __init__(self,
            io_paths: pl.Path | Sequence[pl.Path],
            *, 
            save_directory: pl.Path | None = None,
            output_file: str | None = None,
            constraint: Literal["none", "solver", "system", "both"] = "none",
        ) -> None:
        
        """
        Given one or more i/o directories, sets up the relevant paths, and resolves
        the solver and system types referenced by their input files.

        Parameters
        ----------
        io_paths : pl.Path | Sequence[pl.Path]
            Path or paths to i/o directories containing 'input.toml'.

        save_directory : pl.Path | None
            Optional path where to save results. If None, nothing will be saved.

        output_file: str | None
            Type of output being analysed for this instance of post-processing.
            If None, no specific output type is assumed.

        constraint : {"none", "solver", "system", "both"}
            Constraint on mixing solvers/systems across provided i/o directories. 
            If "solver", all solvers must match. If "system", all systems must
            match. If "both", both solvers and systems must match.
        """

        # Parse io_paths input
        if isinstance(io_paths, (str, pl.Path)):
            io_paths = [io_paths]

        self.io_paths = []
        for path in io_paths:
            input_file = pl.Path(path) / "input.toml"
            if not input_file.exists():
                raise ValueError(
                    f"Path {path} is not a valid i/o directory."
                )
            self.io_paths.append(pl.Path(path))

        # Set output file and save directory
        self.output_file = output_file 
        self.save_directory = pl.Path(save_directory).resolve() if save_directory else None

        # Determine solver and system types across all i/o directories
        self._get_solver_and_system_types()

        # Enforce constraint across provided inputs
        if constraint not in ("none", "solver", "system", "both"):
            raise ValueError("Invalid value for 'constraint'.")

        solver_types = set(self._solver_types.values())
        system_types = set(self._system_types.values())

        if constraint in ("solver", "both") and len(solver_types) > 1:
            raise ValueError("All i/o directories must contain output from the same "
                             "solver when 'constraint' is 'solver' or 'both'.")
        if constraint in ("system", "both") and len(system_types) > 1:
            raise ValueError("All i/o directories must contain output from the same "
                             "system when 'constraint' is 'system' or 'both'.")
        
        print(f"FlucsPostProcessing "
              f"({len(self.io_paths)}, "
              f"{self.output_file}, "
              f"{self.save_directory})")