"""
Main flucs script.
Used to run simulations.
"""

import argparse
import importlib.metadata
import pathlib as pl
from datetime import datetime
from importlib.metadata import entry_points

from flucs.input import FlucsInput
from flucs.utilities.clean_directory import clean_directory
from flucs.utilities.log_handler import FlucsLogHandler

FLUCS_HEADER = rf"""
***************************************************

       ██████  ████
      ███░░███░░███
     ░███ ░░░  ░███  █████ ████  ██████   █████
    ███████    ░███ ░░███ ░███  ███░░███ ███░░
   ░░░███░     ░███  ░███ ░███ ░███ ░░░ ░░█████
     ░███      ░███  ░███ ░███ ░███  ███ ░░░░███
     █████     █████ ░░████████░░██████  ██████
    ░░░░░     ░░░░░   ░░░░░░░░  ░░░░░░  ░░░░░░

***************************************************

{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Version: {importlib.metadata.version("flucs")}
"""

# Load lists of registered solvers and systems
solvers = entry_points().select(group="flucs.solvers")
systems = entry_points().select(group="flucs.systems")


def get_solver_type(solver_name: str):
    """
    Returns a solver type.

    Parameters
    ----------
    solver_name: str
        Name of the solver. Must be registered as an
        entry point in the flucs.solvers group.

    Returns
    -------
    Appropriate FlucsSolver type.

    """

    try:
        s = solvers[solver_name]
    except KeyError as e:
        raise KeyError(
            f"Solver '{solver_name}' not found. "
            "Use 'flucs --list' to see installed solvers."
        ) from e

    return s.load()


def get_system_type(system_name: str):
    """
    Returns a system type.

    Parameters
    ----------
    system_name: str
        Name of the system. Must be registered as an
        entry point in the flucs.systems group.

    Returns
    -------
    Appropriate FlucsSystem type.

    """

    try:
        s = systems[system_name]
    except KeyError as e:
        raise KeyError(
            f"System '{system_name}' not found. "
            "Use 'flucs --list' to see installed systems."
        ) from e

    return s.load()


def list_solvers_and_systems():
    """
    Prints the available solvers and systems to stdout.
    """

    _indent = 3 * " "

    print("Installed solvers:")
    for s in sorted(solvers, key=lambda e: e.name.lower()):
        print(f"{_indent}{s.name}")

    print("Installed systems:")
    if not systems:
        print(f"{_indent}None")
    else:
        for s in sorted(systems, key=lambda e: e.name.lower()):
            print(f"{_indent}{s.name:20} ({s.dist.name})")

    print("For more information, see https://github.com/flucs-code")


def run_flucs(input_path: pl.Path, override: list | None = None):
    """
    Construct FlucsInput then call the appropriate solver.

    Parameters
    ----------
    input_path : pl.Path
        Path to the input file
    override : list
        Additional override parameters specified by the --override flag in the
        command line.

    """

    # Set up redirection of stdout and stderr to an additional log file
    input_path = pl.Path(input_path)
    log_path = input_path.parent / "output.log"

    with open(log_path, "a", encoding="utf-8") as log_file:
        with FlucsLogHandler(log_file, keep_stdout=True):
            print(f"{FLUCS_HEADER}")

            flucs_input = FlucsInput(input_path, override)

            solver, _ = flucs_input.create_solver_system()

            solver.run()


def main():
    """
    Main starting point for flucs.

    This function interprets command-line arguments and decides what to do
    next.

    """

    parser = argparse.ArgumentParser(description="flucs = fluid cuda solver.")

    parser.add_argument(
        "--io_path",
        "-io",
        type=str,
        default=pl.Path.cwd(),
        required=False,
        help="Path to the i/o directory, which must contain 'input.toml'. "
        "If no path is specified, will assume the current working directory.",
    )

    parser.add_argument(
        "--override",
        "-o",
        nargs="+",
        required=False,
        help="Additional arguments to override input-file parameters. Must be "
        "specified in TOML grouping format: e.g., to override the value "
        "of dt_max in group time to be 0.01, specify 'time.dt_max 0.01'.",
    )

    # The script can only do one thing at a time.
    # Here are the options.
    operation_modes = parser.add_mutually_exclusive_group()

    operation_modes.add_argument(
        "--run",
        action="store_true",
        default=False,
        required=False,
        help="Runs the appropriate solver using input.toml from --io_path.",
    )

    operation_modes.add_argument(
        "--list",
        "-l",
        action="store_true",
        default=False,
        required=False,
        help="Lists the solvers and systems that can be run in the "
        "current installation.",
    )

    operation_modes.add_argument(  # TODO
        "--test",
        "-t",
        action="store_true",
        default=False,
        required=False,
        help="NOT YET IMPLEMENTED: run setup/timing tests and then exit.",
    )

    operation_modes.add_argument(
        "--clean",
        "-c",
        action="store_true",
        default=False,
        required=False,
        help="Remove 'output.*' and 'restart.*' files in the current directory "
        "and exit.",
    )

    operation_modes.add_argument(
        "--postprocess",
        "-p",
        action="store_true",
        default=False,
        required=False,
        help="Provides information about post-processing options for a given "
        "i/o directory (specified via '--io_path') and exit.",
    )

    operation_modes.add_argument(
        "--reconstruct",
        "-r",
        type=str,
        required=False,
        help="Reconstruct the input file from the specified restart file. "
        "Note that --override is ignored.",
    )

    args = parser.parse_args()
    io_path = pl.Path(args.io_path).resolve()

    # If nothing is specified, assume --run
    if not any(
        (
            args.run,
            args.list,
            args.test,
            args.clean,
            args.reconstruct,
            args.postprocess,
        )
    ):
        args.run = True

    # Actually solve something
    if args.run:
        input_path = io_path / "input.toml"

        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found in {io_path} ")

        run_flucs(input_path, args.override)
        return

    # List installed solvers and systems
    if args.list:
        list_solvers_and_systems()
        return

    # Cleanup
    if args.clean:
        clean_directory(io_path, ("restart.*", "output.*"))
        return

    # Input-file reconstruction
    if args.reconstruct is not None:
        # Import here to avoid circular imports at module load time
        from flucs.restart import FlucsRestart

        FlucsRestart.reconstruct_input_from_restart(args.reconstruct, io_path)
        return

    # Post-processing
    if args.postprocess:
        from flucs.postprocessing import FlucsPostProcessing

        FlucsPostProcessing(io_path).list_script_paths()
        return
