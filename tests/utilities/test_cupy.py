"""
Tests for CUDA configuration and kernel-wrapper utilities.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from flucs.utilities.cupy import KernelCollection, KernelWrapper, ModuleOptions

pytestmark = pytest.mark.core


def test_module_options_public_workflow():
    # Register options
    options = ModuleOptions()
    options.add_compiler_option("--fmad=false")
    options.define_flag("ENABLED")
    options.define_flag("METHOD", "explicit")
    options.define_float("RATE", 1.5)
    options.define_int("COUNT", 3)
    options.define_dimension("SIZE", 64)

    # Verify that the options are correctly set
    assert options.get_options() == (
        "--ptxas-options=-O3",
        "--use_fast_math",
        "-std=c++17",
        "--fmad=false",
        "-DENABLED",
        "-DMETHOD=explicit",
        "-DRATE=((FLUCS_FLOAT)(1.5))",
        "-DCOUNT=((int)(3))",
        "-DSIZE=((size_t)(64))",
    )

    # Create a second instance and verify that modifying the first instance does
    # not alter the global defaults.
    assert "--fmad=false" not in ModuleOptions().get_options()


def test_kernel_collection_lifecycle():
    # Construct minimal object resembling a FLUCS system
    system = SimpleNamespace(module_options=ModuleOptions())
    system.kernels = KernelCollection(system)

    # Make a mock cupy.RawModule and associated kernel
    cuda_kernel = Mock()
    system.cupy_module = Mock(spec_set=["get_function"])
    system.cupy_module.get_function.return_value = cuda_kernel

    # Create KernelWrapper
    wrapper = KernelWrapper(
        system=system,
        cuda_kernel_name="advance",
        grid=(4,),
        block=(32,),
        shared_mem=128,
    )

    # Check that it is added to system.kernels and that its name has been added
    assert system.kernels
    assert system.kernels[0] is wrapper
    assert list(system.kernels) == [wrapper]
    assert system.module_options.name_expressions == ["advance"]

    # Bind kernel and simulates a call to the underlying CUDA kernel
    system.kernels.bind()
    wrapper("field", 2)
    system.cupy_module.get_function.assert_called_once_with("advance")
    cuda_kernel.assert_called_once_with(
        (4,),
        (32,),
        ("field", 2),
        shared_mem=128,
    )

    # Remove refrence to kernel and check it is unbound
    system.kernels.unbind()
    assert not hasattr(wrapper, "kernel")
