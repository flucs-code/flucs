import toml
import glob
import flucs
import inspect
import pathlib as pl

class FlucsPostProcessing:
    """
    Class that handles post-processing of output data from flucs.
    """

    # Paths
    _input_path: pl.Path
    _script_paths: dict[str, list[pl.Path]]

    # Solver and system for this output
    _solver_name: str 
    _solver_type: type
    _system_name: str           
    _system_type: type  


    def _get_solver_and_system_types(self) -> None:
        """
        Helper function to get the solver and system types
        for this output based on the input file.
        """

        input_file_dict = toml.load(self._input_path)

        self._solver_name = input_file_dict["setup"]["solver"]
        self._system_name = input_file_dict["setup"]["system"]

        self._solver_type =\
            flucs.get_solver_type(self._solver_name)
        self._system_type =\
            flucs.get_system_type(self._system_name)


    def _get_script_paths(self) -> None:
        """
        Helper function to gather the paths to the relevant postprocessing
        scripts for the solver and system used in this output.
        """
        self._script_paths = {}

        for type in [self._solver_type, self._system_type]:
            name = type.__name__
            path = inspect.getfile(type)

            self._script_paths[name] = []
            for script in (pl.Path(path).parent / "postprocessing").glob("*.py"):
                self._script_paths[name].append(pl.Path(script))


    def print_script_paths(self) -> None:
        """
        Prints information about the postprocessing scripts to the standard output.
        """
        self._get_script_paths()

        print("Available postprocessing scripts:")
        for type, paths in self._script_paths.items():
            print(f"--- {type}:")
            for path in paths:
                print(f"     {path}")
    

    def __init__(self, io_path: pl.Path) -> None:
        
        """
        Given an io_path, sets up the relevant paths, solver, and
        system for this instance of FlucsPostProcessing.

        Parameters
        ----------
        io_path : pl.Path
            Path to the i/o directory
        """

        # Get input path
        if not pl.Path(io_path, "input.toml").exists():
            raise ValueError(
                f"Path {io_path} is not a valid i/o directory."
            )
        
        self._input_path = io_path / "input.toml"
        self._get_solver_and_system_types()
