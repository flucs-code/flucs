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
        "--input", "-i",
        type=str,
        default=None,
        required=False,
        help="Path to the input file. If not specified, looks in the current "
             "working directory for 'input.toml' and, if that is not found, "
             "looks for 'NAME.toml' file where NAME matches the name of the "
             "current working directory."
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

    parser.add_argument(#TODO
        "--test", "-t",
        action="store_true",
        default=False,
        required=False,
        help="NOT YET IMPLEMENTED: run setup/timing tests and then exit"
    )

    args = parser.parse_args()

    if args.clean:
        clean_directory(pl.Path.cwd(), ("restart.*", "output.*"))
        return

    if args.input is None:

        cwd = pl.Path.cwd()
        candidates = [
            cwd / "input.toml",
            cwd / f"{cwd.name}.toml"
            ]
        args.input = next((str(c) for c in candidates if c.exists()), None)

        if args.input is None:
            raise FileNotFoundError(
                "Input file not found. See 'flucs -- help'."
                )

    flucs_input = FlucsInput(args.input, args.override)

    solver, system = flucs_input.create_solver_system()

    # Start execution
    solver.run()


if __name__ == "__main__":
    run_flucs()
