from collections.abc import Callable

import cupy as cp

from flucs.solvers.fourier.fourier_system import FourierSystem
from flucs.utilities.cupy import KernelWrapper

THREADS_PER_WARP = 32


def create_reduction(
    system: FourierSystem,
    shape: tuple[int],
    functor: str,
    input_args: str,
    complex_output: bool,
    reduce_axis: tuple[bool] = (True, True, True),
    is_half_axis: bool = False,
    shared_mem: int = 0,
) -> Callable[..., cp.ndarray]:
    """
    Creates a reduction function, typically used for diagnostics,
    that constructs an implicit array of specified shape using a functor
    with arbitrary parameters.

    The reduction itself is a function that can be called as
    reduction(*args), where *args are passed onto the functor constructor,
    and returns the final sum a CuPy array.

    Parameters
    ----------
    system : FourierSystem
        The system that hosts the relevant CUDA code.
    shape : tuple[int]
        Input shape used to sample the functor.
    functor : str
        Name of the functor to be sampled from.
    input_args : str
        Types of the input parameters of the functor, separated by commas.
    complex_output : bool
        True if the functor returns FLUCS_COMPLEX. Otherwise,
        FLUCS_FLOAT is assumed.
    reduce_axis : tuple[bool]
        A 3-tuple that specifies which axes are to be reduced.
    is_half_axis : tuple[bool], default: (False, False)
        Specifies if half_axis reductions should be used
        for the last axis. If reduce_axis[-1] = False, this has not effect.
    shared_mem : int
        Bytes of shared memory required by the data kernel in addition
        to the standard reduction requirements.

    Returns
    -------
    reduction : Callable[..., cp.ndarray]
        The reduction function.

    """

    if len(shape) != 3:
        raise ValueError("Only 3D reductions are supported.")

    # Create kernels for intermediate steps if needed
    if complex_output:
        output_type = "FLUCS_COMPLEX"
        last_axis_reduction_shared_mem = (
            THREADS_PER_WARP * system.complex().nbytes
        )
    else:
        output_type = "FLUCS_FLOAT"
        last_axis_reduction_shared_mem = (
            THREADS_PER_WARP * system.float().nbytes
        )

    # Decide what kernels need to be called based on what the reduction is

    match reduce_axis:
        case (True, True, True):
            # Full 3D reduction, use contiguous reductions only
            shared_mem += last_axis_reduction_shared_mem
            data_kernel = KernelWrapper(
                system=system,
                cuda_kernel_name=(
                    f"simple_last_axis_sum<{shape[2]},"
                    f"{str(is_half_axis).lower()},"
                    f"{output_type},"
                    f"{functor},"
                    f"{input_args}>"
                ),
                grid=(shape[0] * shape[1],),
                block=(system.cuda_block_size,),
                shared_mem=shared_mem,
            )

            reduction_kernel_2d = KernelWrapper(
                system=system,
                cuda_kernel_name=(
                    f"simple_last_axis_sum<{shape[1]},"
                    "false,"
                    f"{output_type},"
                    f"NOP_Functor<{output_type}>,"
                    f"{output_type}*>"
                ),
                grid=(shape[0],),
                block=(system.cuda_block_size,),
                shared_mem=last_axis_reduction_shared_mem,
            )
            reduction_kernel_1d = KernelWrapper(
                system=system,
                cuda_kernel_name=(
                    f"simple_last_axis_sum<{shape[0]},"
                    "false,"
                    f"{output_type},"
                    f"NOP_Functor<{output_type}>,"
                    f"{output_type}*>"
                ),
                grid=(1,),
                block=(system.cuda_block_size,),
                shared_mem=last_axis_reduction_shared_mem,
            )

            # Arrays for intermediate steps
            temp_2d = system.get_temp_array(
                shape[0] * shape[1], is_complex=complex_output
            )
            temp_1d = system.get_temp_array(shape[0], is_complex=complex_output)
            temp_0d = system.get_temp_array(1, is_complex=complex_output)

            def reduction(*args):
                data_kernel(temp_2d, *args)
                reduction_kernel_2d(temp_1d, temp_2d)
                reduction_kernel_1d(temp_0d, temp_1d)
                return temp_0d

        case (False, True, True):
            # 2D contiguous reduction
            shared_mem += last_axis_reduction_shared_mem
            data_kernel = KernelWrapper(
                system=system,
                cuda_kernel_name=(
                    f"simple_last_axis_sum<{shape[2]},"
                    f"{str(is_half_axis).lower()},"
                    f"{output_type},"
                    f"{functor},"
                    f"{input_args}>"
                ),
                grid=(shape[0] * shape[1],),
                block=(system.cuda_block_size,),
                shared_mem=shared_mem,
            )

            reduction_kernel_2d = KernelWrapper(
                system=system,
                cuda_kernel_name=(
                    f"simple_last_axis_sum<{shape[1]},"
                    "false,"
                    f"{output_type},"
                    f"NOP_Functor<{output_type}>,"
                    f"{output_type}*>"
                ),
                grid=(shape[0],),
                block=(system.cuda_block_size,),
                shared_mem=last_axis_reduction_shared_mem,
            )

            # Arrays for intermediate steps
            temp_2d = system.get_temp_array(
                shape[0] * shape[1], is_complex=complex_output
            )
            temp_1d = system.get_temp_array(shape[0], is_complex=complex_output)

            def reduction(*args):
                data_kernel(temp_2d, *args)
                reduction_kernel_2d(temp_1d, temp_2d)
                return temp_1d

        case (True, False, True):
            # One contiguous followed by one non-contiguous
            shared_mem += last_axis_reduction_shared_mem
            data_kernel = KernelWrapper(
                system=system,
                cuda_kernel_name=(
                    f"simple_last_axis_sum<{shape[2]},"
                    f"{str(is_half_axis).lower()},"
                    f"{output_type},"
                    f"{functor},"
                    f"{input_args}>"
                ),
                grid=(shape[0] * shape[1],),
                block=(system.cuda_block_size,),
                shared_mem=shared_mem,
            )

            # (M, N, K) -> (M, K) reduction
            # for M = 1, N = shape[0], K = shape[1]
            nblocks_per_K = (
                shape[1] + system.cuda_block_size - 1
            ) // system.cuda_block_size
            reduction_kernel_2d = KernelWrapper(
                system=system,
                cuda_kernel_name=(
                    f"simple_middle_axis_sum<{shape[0]},{shape[1]},"
                    f"{output_type},"
                    f"NOP_Functor<{output_type}>,"
                    f"{output_type}*>"
                ),
                grid=(nblocks_per_K, 1),
                block=(system.cuda_block_size,),
                shared_mem=last_axis_reduction_shared_mem,
            )

            # Arrays for intermediate steps
            temp_2d = system.get_temp_array(
                shape[0] * shape[1], is_complex=complex_output
            )
            temp_1d = system.get_temp_array(shape[1], is_complex=complex_output)

            def reduction(*args):
                data_kernel(temp_2d, *args)
                reduction_kernel_2d(temp_1d, temp_2d)
                return temp_1d
        case (False, False, True):
            # One contiguous reduction and that's it
            shared_mem += last_axis_reduction_shared_mem
            data_kernel = KernelWrapper(
                system=system,
                cuda_kernel_name=(
                    f"simple_last_axis_sum<{shape[2]},"
                    f"{str(is_half_axis).lower()},"
                    f"{output_type},"
                    f"{functor},"
                    f"{input_args}>"
                ),
                grid=(shape[0] * shape[1],),
                block=(system.cuda_block_size,),
                shared_mem=shared_mem,
            )

            temp_2d = system.get_temp_array(
                shape[0] * shape[1], is_complex=complex_output
            )

            def reduction(*args):
                data_kernel(temp_2d, *args)
                return temp_2d

        case (True, True, False):
            # Two non-contiguous axis reductions

            # (M, N, K) -> (M, K) reduction
            # for M = shape[0], N = shape[1], K = shape[2]
            M = shape[0]
            N = shape[1]
            K = shape[2]
            nblocks_per_K = (
                K + system.cuda_block_size - 1
            ) // system.cuda_block_size
            data_kernel = KernelWrapper(
                system=system,
                cuda_kernel_name=(
                    f"simple_middle_axis_sum<{N},{K},"
                    f"{output_type},"
                    f"{functor},"
                    f"{input_args}>"
                ),
                grid=(nblocks_per_K, M),
                block=(system.cuda_block_size,),
                shared_mem=shared_mem,
            )

            # (M, N, K) -> (M, K) reduction
            # for M = 1, N = shape[0], K = shape[2]
            M = 1
            N = shape[0]
            K = shape[2]
            nblocks_per_K = (
                K + system.cuda_block_size - 1
            ) // system.cuda_block_size
            reduction_kernel_2d = KernelWrapper(
                system=system,
                cuda_kernel_name=(
                    f"simple_middle_axis_sum<{N},{K},"
                    f"{output_type},"
                    f"NOP_Functor<{output_type}>,"
                    f"{output_type}*>"
                ),
                grid=(nblocks_per_K, M),
                block=(system.cuda_block_size,),
                shared_mem=last_axis_reduction_shared_mem,
            )

            # Arrays for intermediate steps
            temp_2d = system.get_temp_array(
                shape[0] * shape[2], is_complex=complex_output
            )
            temp_1d = system.get_temp_array(shape[2], is_complex=complex_output)

            def reduction(*args):
                data_kernel(temp_2d, *args)
                reduction_kernel_2d(temp_1d, temp_2d)
                return temp_1d
        case (False, True, False):
            # One non-contiguous axis reductions

            # (M, N, K) -> (M, K) reduction
            # for M = shape[0], N = shape[1], K = shape[2]
            M = shape[0]
            N = shape[1]
            K = shape[2]
            nblocks_per_K = (
                K + system.cuda_block_size - 1
            ) // system.cuda_block_size
            data_kernel = KernelWrapper(
                system=system,
                cuda_kernel_name=(
                    f"simple_middle_axis_sum<{N},{K},"
                    f"{output_type},"
                    f"{functor},"
                    f"{input_args}>"
                ),
                grid=(nblocks_per_K, M),
                block=(system.cuda_block_size,),
                shared_mem=shared_mem,
            )

            temp_2d = system.get_temp_array(
                shape[0] * shape[2], is_complex=complex_output
            )

            def reduction(*args):
                data_kernel(temp_2d, *args)
                return temp_2d
        case (True, False, False):
            # One non-contiguous axis reductions

            # (M, N, K) -> (M, K) reduction
            # for M = 1, N = shape[0], K = shape[1]*shape[2]
            M = 1
            N = shape[0]
            K = shape[1] * shape[2]
            nblocks_per_K = (
                K + system.cuda_block_size - 1
            ) // system.cuda_block_size
            data_kernel = KernelWrapper(
                system=system,
                cuda_kernel_name=(
                    f"simple_middle_axis_sum<{N},{K},"
                    f"{output_type},"
                    f"{functor},"
                    f"{input_args}>"
                ),
                grid=(nblocks_per_K, M),
                block=(system.cuda_block_size,),
                shared_mem=shared_mem,
            )

            temp_2d = system.get_temp_array(
                shape[1] * shape[2], is_complex=complex_output
            )

            def reduction(*args):
                data_kernel(temp_2d, *args)
                return temp_2d

        case _:
            raise ValueError(
                f"{reduce_axis} is an invalid specification of reduction axes."
            )

    return reduction


def reduce_unpadded_to_scalar(
    system: FourierSystem,
    functor: str,
    input_args: str,
    complex_output: bool,
    shared_mem: int = 0,
):
    return create_reduction(
        system=system,
        shape=system.half_unpadded_tuple,
        functor=functor,
        input_args=input_args,
        complex_output=complex_output,
        reduce_axis=(True, True, True),
        is_half_axis=True,
        shared_mem=shared_mem,
    )


def reduce_unpadded_to_kz(
    system: FourierSystem,
    functor: str,
    input_args: str,
    complex_output: bool,
    shared_mem: int = 0,
):
    return create_reduction(
        system=system,
        shape=system.half_unpadded_tuple,
        functor=functor,
        input_args=input_args,
        complex_output=complex_output,
        reduce_axis=(False, True, True),
        is_half_axis=True,
        shared_mem=shared_mem,
    )


def reduce_unpadded_to_kx(
    system: FourierSystem,
    functor: str,
    input_args: str,
    complex_output: bool,
    shared_mem: int = 0,
):
    return create_reduction(
        system=system,
        shape=system.half_unpadded_tuple,
        functor=functor,
        input_args=input_args,
        complex_output=complex_output,
        reduce_axis=(True, False, True),
        is_half_axis=True,
        shared_mem=shared_mem,
    )


def reduce_unpadded_to_ky(
    system: FourierSystem,
    functor: str,
    input_args: str,
    complex_output: bool,
    shared_mem: int = 0,
):
    return create_reduction(
        system=system,
        shape=system.half_unpadded_tuple,
        functor=functor,
        input_args=input_args,
        complex_output=complex_output,
        reduce_axis=(True, True, False),
        is_half_axis=True,
        shared_mem=shared_mem,
    )


def reduce_unpadded_to_kzkx(
    system: FourierSystem,
    functor: str,
    input_args: str,
    complex_output: bool,
    shared_mem: int = 0,
):
    return create_reduction(
        system=system,
        shape=system.half_unpadded_tuple,
        functor=functor,
        input_args=input_args,
        complex_output=complex_output,
        reduce_axis=(False, False, True),
        is_half_axis=True,
        shared_mem=shared_mem,
    )


def reduce_unpadded_to_kzky(
    system: FourierSystem,
    functor: str,
    input_args: str,
    complex_output: bool,
    shared_mem: int = 0,
):
    return create_reduction(
        system=system,
        shape=system.half_unpadded_tuple,
        functor=functor,
        input_args=input_args,
        complex_output=complex_output,
        reduce_axis=(False, True, False),
        is_half_axis=True,
        shared_mem=shared_mem,
    )


def reduce_unpadded_to_kxky(
    system: FourierSystem,
    functor: str,
    input_args: str,
    complex_output: bool,
    shared_mem: int = 0,
):
    return create_reduction(
        system=system,
        shape=system.half_unpadded_tuple,
        functor=functor,
        input_args=input_args,
        complex_output=complex_output,
        reduce_axis=(True, False, False),
        is_half_axis=True,
        shared_mem=shared_mem,
    )
