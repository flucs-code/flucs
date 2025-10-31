import pathlib as pl
from importlib.metadata import entry_points

class FlucsPostProcessing:
    """
    Helper class that handles post-processing of output data. 
    """

    # Paths
    _input_path: pl.Path
    _script_paths: dict[str, list[pl.Path]]

    # Solver and system for this output
    _solver_name: str          
    _system_name: str       

    def __init__(self, io_path: pl.Path) -> None:
        
        """
        Given an io_path, sets up the relevant paths, solver, and
        system for this instance of post-processing.

        Parameters
        ----------
        io_path : pl.Path
            Path to the i/o directory
        """

        if not pl.Path(io_path, "input.toml").exists():
            raise ValueError(
                f"Path {io_path} is not a valid i/o directory."
            )

        # Use entry points to get file locations given solver and system types
