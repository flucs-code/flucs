"""
Main flucs script.
Used to run simulations.
"""

import argparse
import pathlib as pl
from importlib.metadata import entry_points
from flucs.input import FlucsInput
from flucs.utilities.clean_directory import clean_directory


# Load lists of registered solvers and systems
solvers = entry_points().select(group="flucs.solvers")
systems = entry_points().select(group="flucs.systems")


def get_solver_type(solver_name: str):
    """Returns a solver type.

    Parameters
    ----------
    solver_name: str
        Name of the solver. Must be registered as an
        entry point in the flucs.solvers group.

    Returns
    -------
    Appropriate FlucsSolver type.

    """
    return solvers[solver_name].load()


def get_system_type(system_name: str):
    """Returns a system type.

    Parameters
    ----------
    system_name: str
        Name of the system. Must be registered as an
        entry point in the flucs.systems group.

    Returns
    -------
    Appropriate FlucsSystem type.

    """
    return systems[system_name].load()

def run_flucs():
    """Main starting point for flucs.

    This function interprets command-line arguments, feeds them
    to FlucsInput, then finally calls the appropriate solver.

    """

    parser = argparse.ArgumentParser(
        description="Runs the appropriate flucs solver using an input file"
    )

    parser.add_argument(
        "--io_path", "-io",
        type=str,
        default=pl.Path.cwd(),
        required=False,
        help="Path to the i/o directory, which must contain 'input.toml'."
             "If no path is specified, will assume the current working directory."
    )

    parser.add_argument(
        "--override", "-o",
        nargs="+",
        required=False,
        help="Additional arguments to override input-file parameters. Must be "
             "specified in TOML grouping format: e.g., to override the value "
             "of dt in group time to be 0.01, specify 'time.dt 0.01'."
    )

    parser.add_argument(
        "--clean", "-c",
        action="store_true",
        default=False,
        required=False,
        help="Remove 'output.*' and 'restart.*' files in the current directory"
             "and exit."
    )

    parser.add_argument(
        "--reconstruct", "-r",
        type=str,
        required=False,
        help="Reconstruct the input file from the specified restart file."
    )

    parser.add_argument(#TODO
        "--test", "-t",
        action="store_true",
        default=False,
        required=False,
        help="NOT YET IMPLEMENTED: run setup/timing tests and then exit"
    )

    args = parser.parse_args()

    io_path = args.io_path

    # Run possible helpers
    if args.clean:
        clean_directory(io_path, ("restart.*", "output.*"))
        return

    if args.reconstruct is not None:
        # Import here to avoid circular imports at module load time
        from flucs.systems.flucs_restart_manager import FlucsRestartManager
        FlucsRestartManager.reconstruct_input_from_restart(args.reconstruct,
                                                           io_path)
        return

    input_path = pl.Path(io_path) / "input.toml"
    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found in {io_path} "
        )

    flucs_input = FlucsInput(input_path, args.override)

    solver, system = flucs_input.create_solver_system()

    # Start execution
    solver.run()


if __name__ == "__main__":
    run_flucs()
