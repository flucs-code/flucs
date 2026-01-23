"""A selection of useful functions and classes for dealing with CuPy"""

from typing import Any

import cupy as cp
import numpy as np


def cupy_set_device_pointer(
    module: cp.RawModule, ptr_name: str, data_array: cp.ndarray
):
    """Assigns a device memory pointer to point to a given device array.

    Parameters
    ----------
    module : CuPy.RawModule
        CuPy module that declares the pointer to be assigned.
    ptr_name : str
        Name of the pointer variable.
    data_array : CuPy.array
        Device memory to which ptr_name should point.

    """

    ptr_to_ptr = module.get_global(ptr_name)
    cp.ndarray((1,), dtype=cp.uint64, memptr=ptr_to_ptr)[0] = (
        data_array.data.ptr
    )


class ModuleOptions:
    """Helper class that builds the tuple of options needed to compule CuPy's
    RawModule. Useful for defining compile-time macros and definitions.

    Attributes
    ----------
    string_options : tuple
        A manually specified tuple of string options to be passed to the
        compiler. By default, this is
        ("--ptxas-options=-O3", "--use_fast_math").
    """

    _defs: dict
    string_options = ("--ptxas-options=-O3", "--use_fast_math")

    def __init__(self) -> None:
        self._defs = {}

    def add_string_option(self, option: str) -> None:
        """Adds a compiler option."""
        self.string_options += (str(option),)

    def define_constant(
        self, name: str, value: Any = "", float_convert: bool = False
    ):
        """Adds a definition to the compiler flags.
        Effectively, this is equivalent to adding

        #define name value

        to the source files.

        Parameters
        ----------
        name : str
            Name of the macro/constant to be defined.

        value
            Converted to a string if needed. If value is any of (float,
            np.float16, np.float32, np.float64), "(FLUCS_FLOAT)" is added in
            front of it in order to cast it to the correct type.

        """

        if float_convert or type(value) in (
            float,
            np.float16,
            np.float32,
            np.float64,
        ):
            value_to_add = f"((FLUCS_FLOAT)({value!s}))"
        else:
            value_to_add = f"({value!s})"

        self._defs[name] = value_to_add

    def get_options(self) -> tuple:
        """Returns the tuple of options to be passed to CuPy's RawModule/"""

        ret = ()
        ret += self.string_options

        for key, value in self._defs.items():
            if len(value) > 0:
                ret += (f"-D{key}={value}",)
            else:
                ret += (f"-D{key}",)

        return ret
