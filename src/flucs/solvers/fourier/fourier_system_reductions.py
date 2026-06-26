from collections.abc import Callable

import cupy as cp

from flucs.solvers.fourier.fourier_system import FourierSystem
from flucs.utilities.cupy import KernelWrapper


class FourierReductions:
    """
    A wrapper class for registering reductions and other functionality required
    for diagnostics
    """

    system: FourierSystem
    threads_per_warp: int = 32

    def __init__(self, system: FourierSystem):
        self.system = system

    def get_reduction(
        self,
        reduction_output: str,
        functor: str,
        input_args: str,
        complex_output: bool,
        **kwargs,
    ):
        """
        Returns the appropriate reduction to obtain the specified output.
        """
        match reduction_output:
            case "scalar":
                return self._reduce_unpadded_to_scalar(
                    functor, input_args, complex_output, **kwargs
                )
            case "kx":
                return self._reduce_unpadded_to_kx(
                    functor, input_args, complex_output, **kwargs
                )
            case "ky":
                return self._reduce_unpadded_to_ky(
                    functor, input_args, complex_output, **kwargs
                )
            case "kz":
                return self._reduce_unpadded_to_kz(
                    functor, input_args, complex_output, **kwargs
                )
            case "kperp":
                return self._reduce_unpadded_to_kperp(
                    functor, input_args, complex_output, **kwargs
                )
            case "kzkx":
                return self._reduce_unpadded_to_kzkx(
                    functor, input_args, complex_output, **kwargs
                )
            case "kzky":
                return self._reduce_unpadded_to_kzky(
                    functor, input_args, complex_output, **kwargs
                )
            case "kxky":
                return self._reduce_unpadded_to_kxky(
                    functor, input_args, complex_output, **kwargs
                )
            case "kzkperp":
                return self._reduce_unpadded_to_kzkperp(
                    functor, input_args, complex_output, **kwargs
                )
            case "kx_cumulative":
                return self._reduce_unpadded_to_kx_cumulative(
                    functor, input_args, complex_output, **kwargs
                )
            case "ky_cumulative":
                return self._reduce_unpadded_to_ky_cumulative(
                    functor, input_args, complex_output, **kwargs
                )
            case "kz_cumulative":
                return self._reduce_unpadded_to_kz_cumulative(
                    functor, input_args, complex_output, **kwargs
                )
            case "kperp_cumulative":
                return self._reduce_unpadded_to_kperp_cumulative(
                    functor, input_args, complex_output, **kwargs
                )
            case _:
                raise ValueError(
                    f"Unknown reduction output {reduction_output!r}."
                )

    def get_dimensions(self, reduction_output: str) -> dict[str, cp.ndarray]:
        """
        Returns the dimensions corresponding to the specified reduction output.
        """

        match reduction_output:
            case "scalar":
                return {}

            case "kx":
                return {"kx": self.system.kx}

            case "ky":
                return {"ky": self.system.ky}

            case "kz":
                return {"kz": self.system.kz}

            case "kperp":
                return {"kperp": self.system.shell_kperp}

            case "kzkx":
                return {
                    "kz": self.system.kz,
                    "kx": self.system.kx,
                }

            case "kzky":
                return {
                    "kz": self.system.kz,
                    "ky": self.system.ky,
                }

            case "kxky":
                return {
                    "kx": self.system.kx,
                    "ky": self.system.ky,
                }

            case "kzkperp":
                return {
                    "kz": self.system.kz,
                    "kperp": self.system.shell_kperp,
                }

            case "kx_cumulative":
                return {
                    "kx_abs": self.system.kx[: self.system.half_nx],
                }

            case "ky_cumulative":
                return {
                    "ky_abs": self.system.ky,
                }

            case "kz_cumulative":
                return {
                    "kz_abs": self.system.kz[: self.system.half_nz],
                }

            case "kperp_cumulative":
                return {
                    "kperp": self.system.shell_kperp,
                }

            case _:
                raise ValueError(
                    f"Unknown reduction output {reduction_output!r}."
                )

    def _create_reduction(
        self,
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
        and returns the final sum as a CuPy array.

        Parameters
        ----------
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
            for the last axis. If reduce_axis[-1] = False, this has no effect.
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
                self.threads_per_warp * self.system.complex().nbytes
            )
        else:
            output_type = "FLUCS_FLOAT"
            last_axis_reduction_shared_mem = (
                self.threads_per_warp * self.system.float().nbytes
            )

        # Decide what kernels need to be called based on what the reduction is

        match reduce_axis:
            case (True, True, True):
                # Full 3D reduction, use contiguous reductions only
                shared_mem += last_axis_reduction_shared_mem
                data_kernel = KernelWrapper(
                    system=self.system,
                    cuda_kernel_name=(
                        f"simple_last_axis_sum<{shape[2]},"
                        f"{str(is_half_axis).lower()},"
                        f"{output_type},"
                        f"{functor},"
                        f"{input_args}>"
                    ),
                    grid=(shape[0] * shape[1],),
                    block=(self.system.cuda_block_size,),
                    shared_mem=shared_mem,
                )

                reduction_kernel_2d = KernelWrapper(
                    system=self.system,
                    cuda_kernel_name=(
                        f"simple_last_axis_sum<{shape[1]},"
                        "false,"
                        f"{output_type},"
                        f"NOP_Functor<{output_type}>,"
                        f"{output_type}*>"
                    ),
                    grid=(shape[0],),
                    block=(self.system.cuda_block_size,),
                    shared_mem=last_axis_reduction_shared_mem,
                )
                reduction_kernel_1d = KernelWrapper(
                    system=self.system,
                    cuda_kernel_name=(
                        f"simple_last_axis_sum<{shape[0]},"
                        "false,"
                        f"{output_type},"
                        f"NOP_Functor<{output_type}>,"
                        f"{output_type}*>"
                    ),
                    grid=(1,),
                    block=(self.system.cuda_block_size,),
                    shared_mem=last_axis_reduction_shared_mem,
                )

                # Arrays for intermediate steps
                temp_2d = self.system.get_temp_array(
                    shape[0] * shape[1], is_complex=complex_output
                )
                temp_1d = self.system.get_temp_array(
                    shape[0], is_complex=complex_output
                )
                temp_0d = self.system.get_temp_array(
                    1, is_complex=complex_output
                )

                def reduction(*args):
                    data_kernel(temp_2d, *args)
                    reduction_kernel_2d(temp_1d, temp_2d)
                    reduction_kernel_1d(temp_0d, temp_1d)
                    return temp_0d

            case (False, True, True):
                # 2D contiguous reduction
                shared_mem += last_axis_reduction_shared_mem
                data_kernel = KernelWrapper(
                    system=self.system,
                    cuda_kernel_name=(
                        f"simple_last_axis_sum<{shape[2]},"
                        f"{str(is_half_axis).lower()},"
                        f"{output_type},"
                        f"{functor},"
                        f"{input_args}>"
                    ),
                    grid=(shape[0] * shape[1],),
                    block=(self.system.cuda_block_size,),
                    shared_mem=shared_mem,
                )

                reduction_kernel_2d = KernelWrapper(
                    system=self.system,
                    cuda_kernel_name=(
                        f"simple_last_axis_sum<{shape[1]},"
                        "false,"
                        f"{output_type},"
                        f"NOP_Functor<{output_type}>,"
                        f"{output_type}*>"
                    ),
                    grid=(shape[0],),
                    block=(self.system.cuda_block_size,),
                    shared_mem=last_axis_reduction_shared_mem,
                )

                # Arrays for intermediate steps
                temp_2d = self.system.get_temp_array(
                    shape[0] * shape[1], is_complex=complex_output
                )
                temp_1d = self.system.get_temp_array(
                    shape[0], is_complex=complex_output
                )

                def reduction(*args):
                    data_kernel(temp_2d, *args)
                    reduction_kernel_2d(temp_1d, temp_2d)
                    return temp_1d

            case (True, False, True):
                # One contiguous followed by one non-contiguous
                shared_mem += last_axis_reduction_shared_mem
                data_kernel = KernelWrapper(
                    system=self.system,
                    cuda_kernel_name=(
                        f"simple_last_axis_sum<{shape[2]},"
                        f"{str(is_half_axis).lower()},"
                        f"{output_type},"
                        f"{functor},"
                        f"{input_args}>"
                    ),
                    grid=(shape[0] * shape[1],),
                    block=(self.system.cuda_block_size,),
                    shared_mem=shared_mem,
                )

                # (M, N, K) -> (M, K) reduction
                # for M = 1, N = shape[0], K = shape[1]
                nblocks_per_K = (
                    shape[1] + self.system.cuda_block_size - 1
                ) // self.system.cuda_block_size
                reduction_kernel_2d = KernelWrapper(
                    system=self.system,
                    cuda_kernel_name=(
                        f"simple_middle_axis_sum<{shape[0]},{shape[1]},"
                        f"{output_type},"
                        f"NOP_Functor<{output_type}>,"
                        f"{output_type}*>"
                    ),
                    grid=(nblocks_per_K, 1),
                    block=(self.system.cuda_block_size,),
                    shared_mem=last_axis_reduction_shared_mem,
                )

                # Arrays for intermediate steps
                temp_2d = self.system.get_temp_array(
                    shape[0] * shape[1], is_complex=complex_output
                )
                temp_1d = self.system.get_temp_array(
                    shape[1], is_complex=complex_output
                )

                def reduction(*args):
                    data_kernel(temp_2d, *args)
                    reduction_kernel_2d(temp_1d, temp_2d)
                    return temp_1d
            case (False, False, True):
                # One contiguous reduction and that's it
                shared_mem += last_axis_reduction_shared_mem
                data_kernel = KernelWrapper(
                    system=self.system,
                    cuda_kernel_name=(
                        f"simple_last_axis_sum<{shape[2]},"
                        f"{str(is_half_axis).lower()},"
                        f"{output_type},"
                        f"{functor},"
                        f"{input_args}>"
                    ),
                    grid=(shape[0] * shape[1],),
                    block=(self.system.cuda_block_size,),
                    shared_mem=shared_mem,
                )

                temp_2d = self.system.get_temp_array(
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
                    K + self.system.cuda_block_size - 1
                ) // self.system.cuda_block_size
                data_kernel = KernelWrapper(
                    system=self.system,
                    cuda_kernel_name=(
                        f"simple_middle_axis_sum<{N},{K},"
                        f"{output_type},"
                        f"{functor},"
                        f"{input_args}>"
                    ),
                    grid=(nblocks_per_K, M),
                    block=(self.system.cuda_block_size,),
                    shared_mem=shared_mem,
                )

                # (M, N, K) -> (M, K) reduction
                # for M = 1, N = shape[0], K = shape[2]
                M = 1
                N = shape[0]
                K = shape[2]
                nblocks_per_K = (
                    K + self.system.cuda_block_size - 1
                ) // self.system.cuda_block_size
                reduction_kernel_2d = KernelWrapper(
                    system=self.system,
                    cuda_kernel_name=(
                        f"simple_middle_axis_sum<{N},{K},"
                        f"{output_type},"
                        f"NOP_Functor<{output_type}>,"
                        f"{output_type}*>"
                    ),
                    grid=(nblocks_per_K, M),
                    block=(self.system.cuda_block_size,),
                    shared_mem=last_axis_reduction_shared_mem,
                )

                # Arrays for intermediate steps
                temp_2d = self.system.get_temp_array(
                    shape[0] * shape[2], is_complex=complex_output
                )
                temp_1d = self.system.get_temp_array(
                    shape[2], is_complex=complex_output
                )

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
                    K + self.system.cuda_block_size - 1
                ) // self.system.cuda_block_size
                data_kernel = KernelWrapper(
                    system=self.system,
                    cuda_kernel_name=(
                        f"simple_middle_axis_sum<{N},{K},"
                        f"{output_type},"
                        f"{functor},"
                        f"{input_args}>"
                    ),
                    grid=(nblocks_per_K, M),
                    block=(self.system.cuda_block_size,),
                    shared_mem=shared_mem,
                )

                temp_2d = self.system.get_temp_array(
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
                    K + self.system.cuda_block_size - 1
                ) // self.system.cuda_block_size
                data_kernel = KernelWrapper(
                    system=self.system,
                    cuda_kernel_name=(
                        f"simple_middle_axis_sum<{N},{K},"
                        f"{output_type},"
                        f"{functor},"
                        f"{input_args}>"
                    ),
                    grid=(nblocks_per_K, M),
                    block=(self.system.cuda_block_size,),
                    shared_mem=shared_mem,
                )

                temp_2d = self.system.get_temp_array(
                    shape[1] * shape[2], is_complex=complex_output
                )

                def reduction(*args):
                    data_kernel(temp_2d, *args)
                    return temp_2d

            case _:
                raise ValueError(
                    f"{reduce_axis} is an invalid specification "
                    "of reduction axes."
                )

        return reduction

    def _reduce_unpadded_to_scalar(
        self,
        functor: str,
        input_args: str,
        complex_output: bool,
        shared_mem: int = 0,
    ):
        return self._create_reduction(
            shape=self.system.half_unpadded_tuple,
            functor=functor,
            input_args=input_args,
            complex_output=complex_output,
            reduce_axis=(True, True, True),
            is_half_axis=True,
            shared_mem=shared_mem,
        )

    def _reduce_unpadded_to_kz(
        self,
        functor: str,
        input_args: str,
        complex_output: bool,
        shared_mem: int = 0,
    ):
        return self._create_reduction(
            shape=self.system.half_unpadded_tuple,
            functor=functor,
            input_args=input_args,
            complex_output=complex_output,
            reduce_axis=(False, True, True),
            is_half_axis=True,
            shared_mem=shared_mem,
        )

    def _reduce_unpadded_to_kx(
        self,
        functor: str,
        input_args: str,
        complex_output: bool,
        shared_mem: int = 0,
    ):
        return self._create_reduction(
            shape=self.system.half_unpadded_tuple,
            functor=functor,
            input_args=input_args,
            complex_output=complex_output,
            reduce_axis=(True, False, True),
            is_half_axis=True,
            shared_mem=shared_mem,
        )

    def _reduce_unpadded_to_ky(
        self,
        functor: str,
        input_args: str,
        complex_output: bool,
        shared_mem: int = 0,
    ):
        return self._create_reduction(
            shape=self.system.half_unpadded_tuple,
            functor=functor,
            input_args=input_args,
            complex_output=complex_output,
            reduce_axis=(True, True, False),
            is_half_axis=True,
            shared_mem=shared_mem,
        )

    def _reduce_unpadded_to_kzkx(
        self,
        functor: str,
        input_args: str,
        complex_output: bool,
        shared_mem: int = 0,
    ):
        return self._create_reduction(
            shape=self.system.half_unpadded_tuple,
            functor=functor,
            input_args=input_args,
            complex_output=complex_output,
            reduce_axis=(False, False, True),
            is_half_axis=True,
            shared_mem=shared_mem,
        )

    def _reduce_unpadded_to_kzky(
        self,
        functor: str,
        input_args: str,
        complex_output: bool,
        shared_mem: int = 0,
    ):
        return self._create_reduction(
            shape=self.system.half_unpadded_tuple,
            functor=functor,
            input_args=input_args,
            complex_output=complex_output,
            reduce_axis=(False, True, False),
            is_half_axis=True,
            shared_mem=shared_mem,
        )

    def _reduce_unpadded_to_kxky(
        self,
        functor: str,
        input_args: str,
        complex_output: bool,
        shared_mem: int = 0,
    ):
        return self._create_reduction(
            shape=self.system.half_unpadded_tuple,
            functor=functor,
            input_args=input_args,
            complex_output=complex_output,
            reduce_axis=(True, False, False),
            is_half_axis=True,
            shared_mem=shared_mem,
        )

    def _create_shell_reduction(
        self,
        functor: str,
        input_args: str,
        complex_output: bool,
        reduce_kz: bool = False,
        nkperp: int | None = None,
        kperp_min: float | None = None,
        kperp_max: float | None = None,
    ) -> Callable[..., cp.ndarray]:
        """
        Creates a kperp shell-reduction function, typically used for diagnostics
        that constructs an implicit unpadded Fourier-space array using
        a functor with arbitrary parameters.

        The shell reduction itself is a function that can be called as
        reduction(*args), where *args are passed onto the functor constructor,
        and returns the shell sum as a CuPy array.

        The CUDA shell-sum kernel bins the unpadded Fourier grid in kperp,
        where kperp = sqrt(kx**2 + ky**2). It uses uniform half-open bins,

            [kperp_min, kperp_max),

        with bin index

            floor((kperp - kperp_min) * nkperp / (kperp_max - kperp_min)).

        If reduce_kz is False, the returned array is flattened data with shape
        (nz, nkperp). If reduce_kz is True, the intermediate shell sum is also
        reduced over kz, and the returned array has shape (nkperp,).

        Parameters
        ----------
        functor : str
            Name of the functor to be sampled from.
        input_args : str
            Types of the input parameters of the functor, separated by commas.
        complex_output : bool
            True if the functor returns FLUCS_COMPLEX. Otherwise,
            FLUCS_FLOAT is assumed.
        reduce_kz : bool
            Whether to reduce the kperp shell sums over kz.
        nkperp : int, optional
            Number of kperp bins. If not provided, system.shell_nkperp is used.
        kperp_min : float, optional
            Lower edge of the shell range. If not provided,
            system.shell_kperp_min is used.
        kperp_max : float, optional
            Upper edge of the shell range. If not provided,
            system.shell_kperp_max is used.

        Returns
        -------
        reduction : Callable[..., cp.ndarray]
            The shell-reduction function.

        """

        # Precompute shells if not already done
        self.system._compute_kperp_shells()

        # Overwrite defaults if specified
        nkperp = self.system.shell_nkperp if nkperp is None else int(nkperp)

        kperp_min = (
            self.system.shell_kperp_min
            if kperp_min is None
            else self.system.float(kperp_min)
        )
        kperp_max = (
            self.system.shell_kperp_max
            if kperp_max is None
            else self.system.float(kperp_max)
        )

        # Validate parameters
        if nkperp < 1:
            raise ValueError("nkperp must be positive.")

        if nkperp > self.system.cuda_block_size:
            raise ValueError("nkperp must not exceed system.cuda_block_size.")

        if not kperp_max > kperp_min:
            raise ValueError("kperp_max must be larger than kperp_min.")

        # Output type
        if complex_output:
            output_type = "FLUCS_COMPLEX"
            item_nbytes = self.system.complex().nbytes
        else:
            output_type = "FLUCS_FLOAT"
            item_nbytes = self.system.float().nbytes

        # Shell averaging kernel
        shell_kernel = KernelWrapper(
            system=self.system,
            cuda_kernel_name=(
                f"simple_shell_sum<{output_type},{functor},{input_args}>"
            ),
            grid=(self.system.nz,),
            block=(self.system.cuda_block_size,),
            shared_mem=nkperp * item_nbytes,
        )

        temp_2d = self.system.get_temp_array(
            self.system.nz * nkperp,
            is_complex=complex_output,
        )

        # (kz, kperp) data
        if not reduce_kz:

            def reduction(*args):
                shell_kernel(nkperp, kperp_min, kperp_max, temp_2d, *args)
                return temp_2d

            return reduction

        # Temporary array for holding kperp data
        temp_kperp = self.system.get_temp_array(
            nkperp, is_complex=complex_output
        )

        reduce_kz_kernel = KernelWrapper(
            system=self.system,
            cuda_kernel_name=(
                f"simple_middle_axis_sum<{self.system.nz},{nkperp},"
                f"{output_type},NOP_Functor<{output_type}>,{output_type}*>"
            ),
            grid=(
                (nkperp + self.system.cuda_block_size - 1)
                // self.system.cuda_block_size,
                1,
            ),
            block=(self.system.cuda_block_size,),
            shared_mem=0,
        )

        # (kperp) data
        def reduction(*args):
            shell_kernel(nkperp, kperp_min, kperp_max, temp_2d, *args)
            reduce_kz_kernel(temp_kperp, temp_2d)
            return temp_kperp

        return reduction

    def _reduce_unpadded_to_kzkperp(
        self,
        functor: str,
        input_args: str,
        complex_output: bool,
        **shell_kwargs,
    ):
        return self._create_shell_reduction(
            functor=functor,
            input_args=input_args,
            complex_output=complex_output,
            reduce_kz=False,
            **shell_kwargs,
        )

    def _reduce_unpadded_to_kperp(
        self,
        functor: str,
        input_args: str,
        complex_output: bool,
        **shell_kwargs,
    ):
        return self._create_shell_reduction(
            functor=functor,
            input_args=input_args,
            complex_output=complex_output,
            reduce_kz=True,
            **shell_kwargs,
        )

    def _create_cumulative_reduction(
        self,
        base_reduction: Callable[..., cp.ndarray],
        dimension: str,
        output_size: int,
        complex_output: bool,
    ) -> Callable[..., cp.ndarray]:
        """
        Creates a cumulative reduction from an existing 1D reduction using
        cp.cumsum. The resulting quantities are a function of the relevant
        absolute wavenumber.

        Parameters
        ----------
        base_reduction : Callable[..., cp.ndarray]
            The 1D reduction to be cumulatively summed.
        dimension : str
            Dimension retained by the base reduction.
        output_size : int
            Number of points in the cumulative output.
        complex_output : bool
            True if the reduction returns complex values. Otherwise,
            real values are assumed.

        Returns
        -------
        reduction : Callable[..., cp.ndarray]
            The cumulative reduction function.

        """

        # Use a separate array from the shared reduction work arrays
        dtype = self.system.complex if complex_output else self.system.float
        cumulative = cp.empty(output_size, dtype=dtype)

        match dimension:
            case "kx" | "kz":
                # Fold onto absolute wavenumbers
                def reduction(*args):
                    spectrum = base_reduction(*args)
                    cumulative[0] = spectrum[0]
                    cumulative[1:] = (
                        spectrum[1:output_size]
                        + spectrum[-1 : output_size - 1 : -1]
                    )
                    cp.cumsum(cumulative, out=cumulative)
                    return cumulative

            case "ky":
                # Handle the half grid
                def reduction(*args):
                    spectrum = base_reduction(*args)
                    cumulative[0] = spectrum[0]
                    cumulative[1:] = 2 * spectrum[1:]
                    cp.cumsum(cumulative, out=cumulative)
                    return cumulative

            case "kperp":
                # kperp is already an absolute value
                def reduction(*args):
                    spectrum = base_reduction(*args)
                    cp.cumsum(spectrum, out=cumulative)
                    return cumulative

            case _:
                raise ValueError(
                    f"{dimension!r} does not support cumulative reduction."
                )

        return reduction

    def _reduce_unpadded_to_kx_cumulative(
        self,
        functor: str,
        input_args: str,
        complex_output: bool,
        shared_mem: int = 0,
    ):
        base_reduction = self._reduce_unpadded_to_kx(
            functor,
            input_args,
            complex_output,
            shared_mem,
        )

        return self._create_cumulative_reduction(
            base_reduction=base_reduction,
            dimension="kx",
            output_size=self.system.half_nx,
            complex_output=complex_output,
        )

    def _reduce_unpadded_to_ky_cumulative(
        self,
        functor: str,
        input_args: str,
        complex_output: bool,
        shared_mem: int = 0,
    ):
        base_reduction = self._reduce_unpadded_to_ky(
            functor,
            input_args,
            complex_output,
            shared_mem,
        )

        return self._create_cumulative_reduction(
            base_reduction=base_reduction,
            dimension="ky",
            output_size=self.system.half_ny,
            complex_output=complex_output,
        )

    def _reduce_unpadded_to_kz_cumulative(
        self,
        functor: str,
        input_args: str,
        complex_output: bool,
        shared_mem: int = 0,
    ):
        base_reduction = self._reduce_unpadded_to_kz(
            functor,
            input_args,
            complex_output,
            shared_mem,
        )

        return self._create_cumulative_reduction(
            base_reduction=base_reduction,
            dimension="kz",
            output_size=self.system.half_nz,
            complex_output=complex_output,
        )

    def _reduce_unpadded_to_kperp_cumulative(
        self,
        functor: str,
        input_args: str,
        complex_output: bool,
        **shell_kwargs,
    ):
        base_reduction = self._reduce_unpadded_to_kperp(
            functor,
            input_args,
            complex_output,
            **shell_kwargs,
        )

        self.system._compute_kperp_shells()
        nkperp = shell_kwargs.get("nkperp")
        output_size = (
            self.system.shell_nkperp if nkperp is None else int(nkperp)
        )

        return self._create_cumulative_reduction(
            base_reduction=base_reduction,
            dimension="kperp",
            output_size=output_size,
            complex_output=complex_output,
        )
