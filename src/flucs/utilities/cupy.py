"""A selection of useful functions and classes for dealing with CuPy"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cupy as cp

if TYPE_CHECKING:
    from flucs.systems import FlucsSystem


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


def cupy_get_device_array(
    module: cp.RawModule, array_name: str, shape: tuple, dtype
) -> cp.ndarray:
    """Gets a CuPy array associated with a __device__ array.

    Parameters
    ----------
    module : cp.RawModule
        CuPy module that declares the array.
    array_name : str
        Name of the __device__ array.
    shape : tuple
        Shape of the array.
    dtype
        Data type of the array.

    Returns
    -------
    cp.ndarray
        A CuPy array associated with array_name.

    """
    array_ptr = module.get_global(array_name)
    return cp.ndarray(
        shape=shape,
        dtype=dtype,
        memptr=array_ptr,
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
    options = ("--ptxas-options=-O3", "--use_fast_math", "-std=c++17")
    name_expressions: list

    def __init__(self) -> None:
        self._defs = {}
        self.name_expressions = []

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
        value: str = "",
    ):
        """Adds a flag-like macro to the compiler flags.
        Equivalent to

            #define name value

        Parameters
        ----------
        name: str
            Name of the macro/constant to be defined.
        value: str
            Optional value for the macro/constant.

        """
        self._defs[name] = value

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


class KernelWrapper:
    system: FlucsSystem
    kernel: cp.RawKernel
    cuda_kernel_name: str
    grid: tuple[int]
    block: tuple[int]
    shared_mem: int

    def bind(self) -> None:
        """Binds the wrapper to the compiled kernel."""
        self.kernel = self.system.cupy_module.get_function(
            self.cuda_kernel_name
        )

    def __call__(self, *args) -> None:
        self.kernel(self.grid, self.block, args, shared_mem=self.shared_mem)

    def __eq__(self, other) -> bool:
        if not isinstance(other, KernelWrapper):
            return NotImplemented

        return (
            self.cuda_kernel_name,
            self.grid,
            self.block,
            self.shared_mem,
        ) == (other.cuda_kernel_name, other.grid, other.block, other.shared_mem)

    def __init__(
        self,
        system: FlucsSystem,
        cuda_kernel_name: str,
        grid: tuple[int],
        block: tuple[int],
        shared_mem: int = 0,
    ) -> None:
        system.kernels._kernels.append(self)
        system.module_options.name_expressions.append(cuda_kernel_name)
        self.system = system
        self.cuda_kernel_name = cuda_kernel_name
        self.grid = grid
        self.block = block
        self.shared_mem = shared_mem


class KernelCollection:
    _kernels: list[KernelWrapper]
    system: FlucsSystem

    def bind(self):
        """Binds the KernelWrappers to the compiled symbols."""
        for kernel in self._kernels:
            kernel.bind()

    def __init__(self, system: FlucsSystem):
        self._kernels = []
        self.system = system

    def __getitem__(self, i):
        return self._kernels[i]

    def __iter__(self):
        return iter(self._kernels)
