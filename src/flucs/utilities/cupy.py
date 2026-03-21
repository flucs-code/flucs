"""A selection of useful functions and classes for dealing with CuPy"""

import cupy as cp


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
    options : tuple[str]
        A manually specified tuple of string options to be passed to the
        compiler. By default, this is
        ("--ptxas-options=-O3", "--use_fast_math").
    """

    _defs: dict
    options = ("--ptxas-options=-O3", "--use_fast_math")

    def __init__(self) -> None:
        self._defs = {}

    def add_compiler_option(self, option: str) -> None:
        """Adds a compiler option."""
        self.options += (str(option),)

    def _define_constant(
        self, name: str, value=None, value_type: str | None = None
    ):
        """Adds a definition to the compiler flags.
        Equivalent to

            #define name (value_type)(value)

        Parameters
        ----------
        name: str
            Name of the macro/constant to be defined.

        value:
            Converted to a string if needed. If value is any of (float,
            np.float16, np.float32, np.float64), "(FLUCS_FLOAT)" is added in
            front of it in order to cast it to the correct type.

        value_type:
            Type to which the value is cast.

        """

        if value is None:
            _value_to_add = ""
        else:
            _value_to_add = f"(({value_type})({value!s}))"

        self._defs[name] = _value_to_add

    def define_flag(
        self,
        name: str,
    ):
        """Adds a flag-like macro to the compiler flags.
        Equivalent to

            #define name

        Parameters
        ----------
        name: str
            Name of the macro/constant to be defined.

        """
        self._define_constant(name)

    def define_float(self, name: str, value):
        """Adds a definition to the compiler flags.
        Equivalent to

            #define name ((FLUCS_FLOAT)(value))

        Parameters
        ----------
        name: str
            Name of the macro/constant to be defined.

        value:
            Value of the constant

        """
        self._define_constant(name, value, "FLUCS_FLOAT")

    def define_int(self, name: str, value):
        """Adds a definition of a 32-bit int to the compiler flags.
        Equivalent to

            #define name ((int)(value))

        Parameters
        ----------
        name: str
            Name of the macro/constant to be defined.

        value:
            Value of the constant

        """
        self._define_constant(name, value, "int")

    def define_dimension(self, name: str, value):
        """Adds a definition of a size_t value to the compiler flags.
        Equivalent to

            #define name ((size_t)(value))

        Parameters
        ----------
        name: str
            Name of the macro/constant to be defined.

        value:
            Value of the constant

        """
        self._define_constant(name, value, "size_t")

    def get_options(self) -> tuple:
        """Returns the tuple of options to be passed to CuPy's RawModule/"""

        ret = ()
        ret += self.options

        for key, value in self._defs.items():
            if len(value) > 0:
                ret += (f"-D{key}={value}",)
            else:
                ret += (f"-D{key}",)

        return ret
