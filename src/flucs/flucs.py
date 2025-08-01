"""
Main flucs script.
Used to run simulations.
"""

import argparse
from importlib.metadata import entry_points
from flucs.input import FlucsInput


# Load lists of registered solvers and systems
solvers = entry_points().select(group="flucs.solvers")
systems = entry_points().select(group="flucs.systems")


def get_solver_type(solver_name : str):
    """Returns a solver type.

    Parameters
    ----------
    solver_name : str
        Name of the solver. Must be registered as an
        entry point in the flucs.solvers group.

    Returns
    -------
    Appropriate FlucsSolver type.

    """
    return solvers[solver_name].load()


def load_solver_defaults(solver_name : str, flucs_input : FlucsInput):
    """Loads the default input file for a given solver.

    Parameters
    ----------
    solver_name : str
        Name of the solver.
    flucs_input : FlucsInput
        FlucsInput object to which the input will be saved.

    """
    solver_type = get_solver_type(str)
    if hasattr(solver_type, 'load_defaults') and callable(getattr(solver_type, 'load_defaults')):
        solver_type.load_defaults(flucs_input)
    else:
        print(f"{str} has not implemented a load_defaults method!")


def get_system_type(system_name : str):
    """Returns a system type.

    Parameters
    ----------
    system_name : str
        Name of the system. Must be registered as an
        entry point in the flucs.systems group.

    Returns
    -------
    Appropriate FlucsSystem type.

    """
    return systems[system_name].load()


def load_system_defaults(system_name : str, flucs_input : FlucsInput):
    """Loads the default input file for a given system.

    Parameters
    ----------
    system_name : str
        Name of the system.
    flucs_input : FlucsInput
        FlucsInput object to which the input will be saved.

    """
    system_type = get_system_type(str)
    if hasattr(system_type, 'load_defaults') and callable(getattr(system_type, 'load_defaults')):
        system_type.load_defaults(flucs_input)
    else:
        print(f"{str} has not implemented a load_defaults method!")


def run_flucs():
    """Main starting point for flucs.

    This function interprets command-line arguments, feeds them
    to FlucsInput, then finally calls the appropriate solver.

    """

    parser = argparse.ArgumentParser(description="Runs the appropriate temp_name solver using an input file")
    parser.add_argument("input", type=str, help="Path to the input file.")
    parser.add_argument("--override", nargs="+", help="Additional arguments to override input-file parameters. "
                                                      "Must be specified in TOML grouping format, e.g., to override "
                                                      "the value of dt in group time to be 0.01, specify 'time.dt 0.01'.")
    args = parser.parse_args()

    flucs_input = FlucsInput(args.input, args.override)

    solver, system = flucs_input.create_solver_system()

    system.initialise()
    solver.initialise()

    # Start execution
    solver.run()


if __name__ == "__main__":
    run_flucs()
