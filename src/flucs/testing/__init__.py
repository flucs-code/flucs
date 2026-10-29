"""
Helpers for constructing isolated FLUCS test systems.
"""

import pathlib as pl
import shutil
import signal
from contextlib import contextmanager

from flucs.input import FlucsInput


@contextmanager
def create_test_system(source_input, io_path, *, override=None):
    """
    Construct a system from a copied input without running its solver.

    Parameters
    ----------
    source_input : pathlib.Path or str
        Source input file.
    io_path : pathlib.Path or str
        Isolated input/output directory. Files are not removed after exit.
    override : dict, optional
        Dotted parameter names mapped to their replacement values.

    Yields
    ------
    flucs_input : FlucsInput
        Resolved input for this case.
    solver : FlucsSolver
        Constructed solver.
    system : FlucsSystem
        Constructed system, linked to the solver.
    """

    # Convert the dictionary to the override format accepted by FlucsInput.
    override_args = []
    if override is not None:
        for parameter, value in override.items():
            override_args.extend([parameter, value])

    # Copy the input without replacing an existing case.
    io_path = pl.Path(io_path)
    io_path.mkdir(parents=True, exist_ok=True)

    input_path = io_path / "input.toml"
    with open(source_input, "rb") as source_file:
        with input_path.open("xb") as input_file:
            shutil.copyfileobj(source_file, input_file)

    # Solver construction installs handlers for these signals.
    signal_handlers = {
        sig: signal.getsignal(sig)
        for sig in (
            signal.SIGINT,
            signal.SIGTERM,
            signal.SIGUSR1,
            signal.SIGUSR2,
        )
    }

    # Attempt to construct the solver and system, then cleanup
    system = None
    try:
        flucs_input = FlucsInput(input_path, override_args)
        solver, system = flucs_input.create_solver_system()
        yield flucs_input, solver, system
    finally:
        try:
            if system is not None:
                system.kernels.unbind()
        finally:
            for sig, handler in signal_handlers.items():
                signal.signal(sig, handler)
