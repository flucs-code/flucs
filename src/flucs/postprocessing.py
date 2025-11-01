import argparse
import toml
import glob
import flucs
import inspect
import pathlib as pl
from typing import Sequence, Literal
from netCDF4 import Dataset, Group

class FlucsPostProcessing:
    """
    Class that handles post-processing of output data.
    """

    # Postprocessing-specific attributes
    io_paths: list[pl.Path] 
    output_type: str | None
    save_directory: pl.Path | None
    _script_paths: dict[str, list[pl.Path]]

    # Solver and system for the outputs
    _solver_names: dict[pl.Path, str]
    _system_names: dict[pl.Path, str]
    _solver_types: dict[pl.Path, type]
    _system_types: dict[pl.Path, type]

    # Formatting for printing
    _indent = "   "


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

        with Dataset(pl.Path(nc_path), "r", format="NETCDF4") as ds:
            for grp_name, grp in ds.groups.items():
                grp_id = int(grp_name)
                for var in grp.variables.keys():
                    if var in ignore:
                        continue
                    if var not in netcdf_variables:
                        netcdf_variables[var] = []
                    if grp_id not in netcdf_variables[var]:
                        netcdf_variables[var].append(grp_id)

        return {v: sorted(ids) for v, ids in netcdf_variables.items()}

    def _get_all_netcdf_variables(self, ignore=("time", "dt")) -> dict[str, dict[str, list[int]]]:
        """
        For each i/o directory, collect the variables present in the netCDF file 
        specified by self.output_type, and the groups that they appear in.

        Parameters
        ----------
        ignore : Iterable[str]
            Variable names to ignore. Defaults to ('time', 'dt').

        Returns
        -------
        dict[str, dict[str, list[int]]]
            Mapping nc_path (as a string) -> {variable: [group_ids], ... }
        """

        if self.output_type is None:
            raise ValueError("'output_type' must be set to derive netCDF paths from i/o directories.")

        result: dict[str, dict[str, list[int]]] = {}
        for io_path in self.io_paths:
            nc_path = io_path / self.output_type
            if not nc_path.exists():
                result[str(nc_path)] = {}
                continue
            result[str(nc_path)] = FlucsPostProcessing.get_netcdf_variables(nc_path, ignore=ignore)

        return result
    

    def list_netcdf_variables(self, ignore=("time", "dt")) -> None:
        """
        Prints the available netCDF variables to the standard output for each 
        of the provided i/o directories given a specific output type.
        """

        print("Available netCDF variables:")
        netcdf_variables = self._get_all_netcdf_variables(ignore=ignore)
        for nc_path, variables_dict in netcdf_variables.items():
            print(rf"{self._indent}{nc_path}: {sorted(variables_dict.keys())}")


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

        for nc_path, variables in mapping.items():
            nc_path = pl.Path(nc_path)
            if variable in variables and variables[variable]:
                found.append(nc_path)
            else:
                missing.append(nc_path)

        # Report files that are missing the variable
        print(f"Variable {variable} not found in:")
        for path in missing:
            print(f"{self._indent}{path}")

        return found


    def __init__(self, 
            io_paths: pl.Path | Sequence[pl.Path],
            output_type: str | None = None,
            save_directory: pl.Path | None = None,
            consistent: Literal["none", "solver", "system", "both"] = "none",
        ) -> None:
        
        """
        Given one or more i/o directories, sets up the relevant paths, and resolves
        the solver and system types referenced by their input files.

        Parameters
        ----------
        io_paths : pl.Path | Sequence[pl.Path]
            Path or paths to i/o directories (each must contain 'input.toml')

        output_type : str | None
            Type of output being analysed for this instance of post-processing.
            If None, no specific output type is assumed.

        save_directory : pl.Path | None
            Optional path where to save results. If None, nothing will be saved.

        consistent : {"none", "solver", "system", "both"}
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

        # Set output type and save directory
        self.output_type = output_type 
        self.save_directory = pl.Path(save_directory).resolve() if save_directory else None

        # Determine solver and system types across all i/o directories
        self._get_solver_and_system_types()

        # Enforce consistency across provided inputs
        if consistent not in ("none", "solver", "system", "both"):
            raise ValueError("Invalid value for 'consistent'.")

        solver_types = set(self._solver_types.values())
        system_types = set(self._system_types.values())

        if consistent in ("solver", "both") and len(solver_types) > 1:
            raise ValueError("All inputs must use the same solver when "
                             "'consistent' is 'solver' or 'both'.")
        if consistent in ("system", "both") and len(system_types) > 1:
            raise ValueError("All inputs must use the same system when "
                             "'consistent' is 'system' or 'both'.")
