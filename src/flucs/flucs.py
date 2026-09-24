from __future__ import annotations

import argparse
import importlib.metadata
import os
import pathlib as pl
import shutil
import subprocess
import sys
from datetime import datetime
from importlib.metadata import entry_points
from typing import TYPE_CHECKING

from flucs.input import FlucsInput
from flucs.utilities.clean_directory import clean_directory
from flucs.utilities.log_handler import FlucsLogHandler
from flucs.utilities.messages import HORIZONTAL_SEPARATOR, flucsprint

try:
    import cupy as cupy

    cupy.fft.fft(cupy.zeros(1))  # quickly test if CuPy actually works
except Exception as exc:
    cupy = None
    CUPY_IMPORT_ERROR = exc
    print(f"CuPy not found! {CUPY_IMPORT_ERROR}")
else:
    CUPY_IMPORT_ERROR = None

if TYPE_CHECKING:
    from flucs.solvers import FlucsSolver

FLUCS_HEADER = rf"""
{HORIZONTAL_SEPARATOR}

             ██████  ████
            ███░░███░░███
           ░███ ░░░  ░███  █████ ████  ██████   █████
          ███████    ░███ ░░███ ░███  ███░░███ ███░░
         ░░░███░     ░███  ░███ ░███ ░███ ░░░ ░░█████
           ░███      ░███  ░███ ░███ ░███  ███ ░░░░███
           █████     █████ ░░████████░░██████  ██████
          ░░░░░     ░░░░░   ░░░░░░░░  ░░░░░░  ░░░░░░

{HORIZONTAL_SEPARATOR}

{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Version: {importlib.metadata.version("flucs")}
"""

NSYS_ENV_VAR = "FLUCS_UNDER_NSYS"

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

    flucsprint("Installed solvers:")
    for s in sorted(solvers, key=lambda e: e.name.lower()):
        flucsprint(f"{_indent}{s.name}")

    flucsprint("Installed systems:")
    if not systems:
        flucsprint(f"{_indent}None")
    else:
        for s in sorted(systems, key=lambda e: e.name.lower()):
            flucsprint(f"{_indent}{s.name:22} ({s.dist.name})")

    flucsprint("For more information, see https://github.com/flucs-code")


def parse_cli_arguments(argv: list[str]) -> tuple[list[str], list[str] | None]:
    """
    Split command-line arguments so that everything following
    '-p/--postprocess' can be passed through directly to a selected
    postprocessing script.
    """

    for index, arg in enumerate(argv):
        if arg in ("-p", "--postprocess"):
            return argv[: index + 1], argv[index + 1 :]

    return argv, None


def run_flucs(
    input_path: pl.Path,
    override: list | None = None,
    timing_steps: int = 0,
) -> tuple[FlucsInput, FlucsSolver]:
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
            flucsprint(f"{FLUCS_HEADER}")

            flucs_input = FlucsInput(input_path, override)

            solver, _ = flucs_input.create_solver_system()

            solver.run(timing_steps=timing_steps)

    # Return the input and solver for debugging purposes
    return flucs_input, solver


def run_flucs_under_nsys(io_path: pl.Path) -> None:
    """
    Call run_flucs under Nsight Systems and prints a summary of GPU kernel
    execution to the log file

    Parameters:
    ----------
    io_path : pl.Path
        Path to the i/o directory where the input file is located.

    """
    # Set the correct env variable for NSight Systems
    env = os.environ.copy()
    env[NSYS_ENV_VAR] = "1"

    # Set up temporary directory and file paths for nsys output
    temp_path = io_path / ".temp_nsys"
    nsys_report = temp_path / "flucs_profile.nsys-rep"
    nsys_stats = temp_path / "flucs_stats"
    gpu_kernel_summary = temp_path / "flucs_stats_cuda_gpu_kern_sum.csv"

    if temp_path.exists():
        shutil.rmtree(temp_path)
    temp_path.mkdir()

    # Run and then clean up the temporary directory
    try:
        # Run profiling
        profile_cmd = [
            "nsys",
            "profile",
            "--trace=cuda,nvtx,osrt",
            f"--output={nsys_report}",
            *sys.argv[:],
        ]
        subprocess.run(profile_cmd, env=env, check=True)

        # Run stats to get the GPU kernel summary
        stats_cmd = [
            "nsys",
            "stats",
            "--force-export=true",
            "--force-overwrite=true",
            "--report",
            "cuda_gpu_kern_sum",
            "--format",
            "csv",
            "--output",
            nsys_stats,
            str(nsys_report),
        ]
        subprocess.run(stats_cmd, check=True)

        # Append summary to log file
        log_path = io_path / "output.log"
        with open(log_path, "a", encoding="utf-8") as log_file:
            with FlucsLogHandler(log_file, keep_stdout=True):
                flucsprint(format_nsys_gpu_kernel_summary(gpu_kernel_summary))
    finally:
        shutil.rmtree(temp_path)


def format_nsys_gpu_kernel_summary(filename: pl.Path) -> str:
    """
    Formats a summary of GPU kernel execution from Nsight Systems CSV output.

    Parameters
    ----------
    filename : pl.Path
        Path to the Nsight Systems CSV output.

    Returns
    -------
    str
        Formatted GPU kernel execution summary.

    """
    # Lazily import pythons native csv tools
    import csv

    # Columns to display in the summary
    columns = [
        "Time (%)",
        "Total Time (us)",
        "Avg (us)",
        "Max (us)",
        "Std (us)",
        "Name",
    ]

    # Read in data from csv file and convert times from ns to us
    ns_to_us = 1e-3
    with open(filename, newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        rows = []
        for row in reader:
            rows.append(
                {
                    "Time (%)": row["Time (%)"],
                    "Total Time (us)": (
                        f"{float(row['Total Time (ns)']) * ns_to_us:.3f}"
                    ),
                    "Avg (us)": f"{float(row['Avg (ns)']) * ns_to_us:.3f}",
                    "Max (us)": f"{float(row['Max (ns)']) * ns_to_us:.3f}",
                    "Std (us)": f"{float(row['StdDev (ns)']) * ns_to_us:.3f}",
                    "Name": row["Name"],
                }
            )

    # Determine the maximum width of each column
    widths = {
        column: max(
            len(column),
            *(len(row[column]) for row in rows),
        )
        for column in columns
    }

    # Construct summary lines
    lines = [
        "GPU kernel execution summary (Nsight Systems)",
        "",
        "  ".join(f"{column:<{widths[column]}}" for column in columns),
        "  ".join("-" * widths[column] for column in columns),
    ]

    for row in rows:
        lines.append(
            "  ".join(
                f"{row[column]:>{widths[column]}}"
                if column != "Name"
                else f"{row[column]:<{widths[column]}}"
                for column in columns
            )
        )

    return "\n".join(lines)


def write_default_input(system_name: str, io_path: pl.Path):
    """
    Creates a default input file.

    Parameters
    ----------
    system_name : str
        Name of the FlucsSystem.
    io_path : pl.Path
        Path to the i/o directory where the input file will be created.

    """
    input_file_path = io_path / "input.toml"

    if input_file_path.exists():
        raise ValueError(
            "input.toml already exists in the specified directory. "
            "Please remove it manually if you want to write a default-input "
            "file."
        )

    # Get system type
    system_type = get_system_type(system_name)

    # Get default inputs
    default_input = FlucsInput(filepath=None)
    system_type.load_defaults(default_input)

    # Set the system to be the requested one, and the solver automatically
    default_input["setup.system"] = system_name
    default_input["setup.solver"] = system_type.solver_name

    # Write to file
    input_file_path.write_text(str(default_input), encoding="utf-8")


def main():
    """
    Main starting point for flucs.

    This function interprets command-line arguments and decides what to do
    next.

    """

    parser = argparse.ArgumentParser(description="FLUCS = fluid cuda solver.")

    parser.add_argument(
        "--io_path",
        "-io",
        type=str,
        default=pl.Path.cwd(),
        required=False,
        help=(
            "Path to the i/o directory, which must contain 'input.toml'. "
            "If no path is specified, assumes the current working directory."
        ),
    )

    parser.add_argument(
        "--override",
        "-o",
        nargs="+",
        action="extend",
        required=False,
        help=(
            "Additional arguments to override input-file parameters. Must be "
            "specified in TOML grouping format: e.g., to override the value "
            "of dt_max in group time to be 0.01, specify 'time.dt_max 0.01'."
        ),
    )

    parser.add_argument(
        "--memory",
        "-m",
        action="store_true",
        default=False,
        required=False,
        help=(
            "If specified, --run will execute with CuPy's LineProfileHook. "
            "This can be used to profile GPU memory allocations."
        ),
    )

    operation_modes = parser.add_mutually_exclusive_group()

    operation_modes.add_argument(
        "--run",
        action="store_true",
        default=False,
        required=False,
        help=("Runs the appropriate solver using input.toml from --io_path."),
    )

    operation_modes.add_argument(
        "--init",
        "-i",
        type=str,
        metavar="SYSTEM_NAME",
        required=False,
        help=(
            "Writes input.toml that contains the defaults for the specified "
            "FlucsSystem to --io_path."
        ),
    )

    operation_modes.add_argument(
        "--list",
        "-l",
        action="store_true",
        default=False,
        required=False,
        help=(
            "Lists the solvers and systems that can be run in the "
            "current installation."
        ),
    )

    operation_modes.add_argument(  # TODO
        "--timing",
        "-t",
        nargs="?",
        type=int,
        metavar="STEPS_TO_TIME",
        const=1000,
        default=False,
        required=False,
        help=(
            "Runs STEPS_TO_TIME time steps (default of 1000) then exits. "
            "No output is produced."
        ),
    )

    operation_modes.add_argument(
        "--clean",
        "-c",
        action="store_true",
        default=False,
        required=False,
        help=(
            "Remove 'output.*' and 'restart.*' files in the current directory "
            "and exit."
        ),
    )

    operation_modes.add_argument(
        "--postprocess",
        "-p",
        action="store_true",
        default=False,
        required=False,
        help=(
            "List post-processing scripts for the specified i/o directory, "
            "or run a given script using '-p <integer> <script arguments>'."
        ),
    )

    operation_modes.add_argument(
        "--reconstruct",
        "-r",
        type=str,
        required=False,
        help=(
            "Reconstruct the input file from the specified restart file. "
            "Note that --override is ignored."
        ),
    )

    # Parse command-line arguments
    flucs_args, postprocess_args = parse_cli_arguments(sys.argv[1:])
    args = parser.parse_args(flucs_args)
    io_path = pl.Path(args.io_path).resolve()

    # If nothing is specified, assume --run
    if not any(
        (
            args.run,
            args.init,
            args.list,
            args.clean,
            args.reconstruct,
            args.postprocess,
            args.timing,
        )
    ):
        args.run = True

    if args.timing:
        # Run under nsys
        if os.environ.get("FLUCS_UNDER_NSYS") != "1":
            run_flucs_under_nsys(io_path)
            return

        args.run = True
        timing_steps = int(args.timing)
    else:
        timing_steps = 0

    # Run the solver
    if args.run:
        input_path = io_path / "input.toml"

        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found in {io_path} ")

        if args.memory:
            # Local imports
            from cupy.cuda.memory_hooks import LineProfileHook

            from flucs.utilities.cupy import format_memory_report

            # Run with profiler
            hook = LineProfileHook()
            with hook:
                run_flucs(input_path, args.override, timing_steps)
            cupy.cuda.get_current_stream().synchronize()

            # Append to log
            log_path = io_path / "output.log"
            with open(log_path, "a", encoding="utf-8") as log_file:
                with FlucsLogHandler(log_file, keep_stdout=True):
                    flucsprint(format_memory_report(hook, verbose=False))
            return

        run_flucs(input_path, args.override, timing_steps)
        return

    # Write a default input file
    if args.init:
        write_default_input(args.init, io_path)

    # List installed solvers and systems
    if args.list:
        list_solvers_and_systems()
        return

    # Cleanup
    if args.clean:
        clean_directory(io_path, ("restart.*", "output.*", "STOP"))
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

        # List the possible script paths
        if not postprocess_args:
            postprocessing = FlucsPostProcessing(io_path, quiet=True)
            postprocessing.list_script_paths()
            return

        # Forward the arguments to the selected script as a subprocess
        try:
            script_integer = int(postprocess_args[0])
        except ValueError as e:
            raise ValueError(
                "The first argument after '-p/--postprocess' must be the "
                "integer of one of the listed postprocessing scripts."
            ) from e
        subprocess_args = postprocess_args[1:]

        io_paths = [io_path]
        if any(arg in ("-io", "--io_path") for arg in subprocess_args):
            parser = FlucsPostProcessing.parser()
            args, _ = parser.parse_known_args(subprocess_args)
            io_paths = args.io_path

        postprocessing = FlucsPostProcessing(io_paths, quiet=True)

        script_path = postprocessing.get_script_path(script_integer)
        subprocess.run(
            [sys.executable, str(script_path), *subprocess_args],
            check=True,
        )
        return
