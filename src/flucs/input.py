"""
Contains the definition of the FlucsInput class
that deals with interpreting TOML input files.
"""

import pathlib as pl
from typing import Any

import toml

import flucs


class InvalidFlucsInputFileError(ValueError):
    """Raised when the input file is a valid TOML file but has invalid
    contents.

    """

    pass


class FlucsInput:
    """
    Deals with interpreting TOML input files for
    flucs solvers. It is essentially a cleverdict
    with added functionality.
    """

    input_path: pl.Path  # Path to the input file
    io_path: pl.Path  # Input/output directory

    _input_dict: dict[str, Any]  # Holds all the input parameters
    _default_input_dict: dict[str, Any]  # Holds all the defaults

    _solver_type: type  # Solver type for this input
    _system_type: type  # System type for this input
    _initialised: bool = False  # if True, __setitem__ throws an exception

    def create_solver_system(self):
        """Creates the solver and system for this input.

        Returns
        -------
        (solver: FlucsSolver, system: FlucsSystem)
            Tuple of solver and system.

        """
        system = self._system_type(self)
        solver = self._solver_type(self, system)
        system.solver = solver
        return solver, system

    def __getitem__(self, arg: str):
        """
        Access the dict of input data directly using
        the TOML format aa.bb.cc for nested dicts.
        """
        if not isinstance(arg, str):
            raise ValueError("The key should be a string!")

        split_arg = arg.split(".")
        _dict = self._input_dict
        for i in range(len(split_arg) - 1):
            _dict = _dict[split_arg[i]]
            if not isinstance(_dict, dict):
                raise ValueError(f"Parameter {arg} does not exist!")

        if split_arg[-1] not in _dict:
            raise ValueError(f"Parameter {arg} does not exist!")

        return _dict[split_arg[-1]]

    def __setitem__(self, arg: str, value):
        if self._initialised:
            raise RuntimeError(
                "Input class has finished its initialisation "
                "and is now read-only!"
            )

        if not isinstance(arg, str):
            raise ValueError("The key should be a string!")

        split_arg = arg.split(".")
        _dict = self._input_dict
        for i in range(len(split_arg) - 1):
            _dict = _dict[split_arg[i]]
            if not isinstance(_dict, dict):
                raise ValueError(f"Parameter {arg} does not exist!")

        try:
            _dict[split_arg[-1]] = type(_dict[split_arg[-1]])(value)
        except KeyError as e:
            raise ValueError(f"Parameter {arg} does not exist!") from e
        except ValueError as e:
            raise TypeError(
                f"Error casting '{value}' to type "
                f"'{type(_dict[split_arg[-1]])}' for "
                f"parameter '{arg}'!"
            ) from e

    def load_toml_str(self, toml_str: str, default=False):
        """
        Loads parameters from a TOML string.
        """
        self.load_dict(toml.loads(toml_str), default)

    def load_dict(self, _dict: dict, default=False):
        """
        Loads a dict into _input_dict or _default_input_dict
        depending on whether default is False or True.
        If default is False, the dict is checked to make sure it conforms
        to the set of parameters outlined in _default_input_dict.
        """
        if default:
            self._update_dict(self._default_input_dict, _dict, allow_new=True)
            self._update_dict(self._input_dict, _dict, allow_new=True)
        else:
            self._update_dict(self._input_dict, _dict)

    @staticmethod
    def _update_dict(_dict: dict, _updates: dict, allow_new=False):
        """
        Goes recursively through a dict of dicts, updating the values of keys
        (or 'parameters' in the context of the solvers) as specified by
        _updates. If allow_new is False, every (key, value) pair _updates must
        already exist in _dict; otherwise, an exception is raised. Values are
        cast to their types in _dict in order to keep the type structure of
        _dict unchanged.

        Parameters
        ----------
        _dict : dict
            The dict to be updated.
        _updates : dict
            Data to update the values in _dict.
        """

        for k, v in _updates.items():
            if k not in _dict:
                if not allow_new:
                    raise ValueError(f"Parameter '{k}' is invalid!")

                _dict[k] = v
                continue

            if isinstance(v, dict):
                # If the value is a dict, go into it recursively
                if not isinstance(_dict[k], dict):
                    raise ValueError(
                        f"'{k}' is a parameter, not a group of "
                        f"parameters! It cannot be set to {v!s}!"
                    )

                FlucsInput._update_dict(_dict[k], v, allow_new=allow_new)
            else:
                if isinstance(_dict[k], dict):
                    raise ValueError(
                        f"'{k}' is a group of parameters, not a parameter "
                        f"itself! It cannot be set to {v!s}!"
                    )

                try:
                    _dict[k] = type(_dict[k])(v)
                except ValueError as e:
                    raise ValueError(
                        f"Error casting '{v}' to type '{type(_dict[k])}' "
                        f"for parameter '{k}'!"
                    ) from e

    def __init__(self, filepath: pl.Path, override: list | None = None):
        """
        Initialises defaults and loads from file.
        """

        # Initialise initial state
        self._input_dict = {}
        self._default_input_dict = {}
        self._initialised = False

        # Store input filepath
        self.input_path = pl.Path(filepath)

        # All input and output happens in the same directory
        self.io_path = self.input_path.parent

        # Loads the dict for the user-defined inputs
        input_file_dict = toml.load(filepath)

        # Loads the solver
        self._solver_type = flucs.get_solver_type(
            input_file_dict["setup"]["solver"]
        )

        # Loads the system
        self._system_type = flucs.get_system_type(
            input_file_dict["setup"]["system"]
        )
        self._system_type.load_defaults(self)

        # Load from the input file
        self.load_dict(input_file_dict, default=False)

        # Finally, override parameters if necessary
        if override is not None:
            for parameter, value in zip(override[::2], override[1::2]):
                self[parameter] = value

        self._initialised = True

    def __str__(self):
        """
        The string representation of the input object
        is just a TOML string for the input dict.
        """
        return toml.dumps(self._input_dict)
