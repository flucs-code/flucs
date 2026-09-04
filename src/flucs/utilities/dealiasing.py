import heapq

import numpy as np

from flucs import cupy as cp


def next_smooth_number(n: int, primes: list | None = None) -> int:
    """
    Returns the smallest number that is greater than or equal to a given number
    n and divisible only by the prime numbers specified in primes.

    Parameters
    ----------
    n : int
    primes : list
        List of primes. Defaults to [2, 3].
        N.B. The algorithm assumes but does not check
        that these numbers are prime.

    Returns
    -------
    int
        The smallest 3-smooth number greater than or equal to n.

    """

    if primes is None:
        primes = [2, 3]

    # Use a heap to keep track of the smallest smooth number
    # we have seen so far, and generate them in ascending order.
    heap = [1]

    while True:
        guess = heapq.heappop(heap)

        if n <= guess:
            return guess

        for p in primes:
            # It might be a good idea to check whether p * guess
            # is way too large to even consider. Might speed things
            # a bit but not really worth it given that it's unlikely
            # we will ever use this for n > 10^4 or so.
            heapq.heappush(heap, p * guess)


def dealiased_multiplication_rfft(*args, **kwargs):
    """
    Compute the RFFT of a real-space product using Fourier-space padding.

    Each input is a 3D CuPy RFFT array with shape

        (nz, nx, half_ny)

    The input Fourier arrays are padded, inverse Fourier transformed, multiplied
    pointwise in real space, and transformed back to Fourier space. The fields
    are then cropped to the original grid.

    This is slow, unoptimised, and should therefore not be used
    in performance-critical code.

    If CuPy is not available, NumPy is used instead.

    Parameters
    ----------
    *args
        Fourier-space fields to multiply. All fields must have the same shape
        and dtype.
    **kwargs
        Optional nx, ny, and nz real-space dimensions and their corresponding
        padded_nx, padded_ny, and padded_nz values. If omitted, dimensions and
        sufficiently large padding are inferred from the array input shape
    """

    # Figure out if CuPy is present
    if cp is not None:
        cp_np = cp
    else:
        cp_np = np

    # Number of fields in the real-space product
    n = len(args)

    # Get the original real-space dimensions
    try:
        nx = kwargs["nx"]
    except KeyError:
        nx = args[0].shape[1]

    try:
        ny = kwargs["ny"]
    except KeyError:
        ny = 2 * args[0].shape[2] - 1

    try:
        nz = kwargs["nz"]
    except KeyError:
        nz = args[0].shape[0]

    half_nx = nx // 2 + 1
    half_ny = ny // 2 + 1
    half_nz = nz // 2 + 1

    # Choose padded dimensions large enough for the n-field convolution
    try:
        padded_nx = kwargs["padded_nx"]
    except KeyError:
        padded_nx = int(np.ceil((1.1 + n) * nx / 2))

    try:
        padded_ny = kwargs["padded_ny"]
    except KeyError:
        padded_ny = int(np.ceil((1.1 + n) * ny / 2))

    try:
        padded_nz = kwargs["padded_nz"]
    except KeyError:
        padded_nz = int(np.ceil((1.1 + n) * nz / 2))

    # All Fourier arrays are assumed to use the same complex dtype
    complex_type = args[0].dtype
    padded_half_ny = padded_ny // 2 + 1

    # Allows us to deal with nz = 1 correctly
    neg_z = slice(-half_nz + 1, None) if nz > 1 else slice(0, 0)

    # Accumulate the product on the padded real-space grid
    product = 1
    for i in range(n):
        padded_array = cp_np.zeros(
            (padded_nz, padded_nx, padded_half_ny), dtype=complex_type
        )

        # Handle corners as RFFT returns only nonnegative ky modes.
        padded_array[:half_nz, :half_nx, :half_ny] = args[i][
            :half_nz, :half_nx, :half_ny
        ]
        padded_array[neg_z, :half_nx, :half_ny] = args[i][
            neg_z, :half_nx, :half_ny
        ]
        padded_array[:half_nz, -half_nx + 1 :, :half_ny] = args[i][
            :half_nz, -half_nx + 1 :, :half_ny
        ]
        padded_array[neg_z, -half_nx + 1 :, :half_ny] = args[i][
            neg_z, -half_nx + 1 :, :half_ny
        ]

        # Transform each padded field
        # and multiply it into the real-space product
        product *= cp_np.fft.irfftn(
            padded_array, s=(padded_nz, padded_nx, padded_ny), norm="forward"
        )

    # Transform the product back to the padded Fourier grid
    product_rfft_padded = cp_np.fft.rfftn(
        product, norm="forward", s=(padded_nz, padded_nx, padded_ny)
    )

    # Crop the same four Fourier-space corners back to the original grid
    product_rfft = cp_np.zeros((nz, nx, half_ny), dtype=complex_type)

    product_rfft[:half_nz, :half_nx, :half_ny] = product_rfft_padded[
        :half_nz, :half_nx, :half_ny
    ]
    product_rfft[neg_z, :half_nx, :half_ny] = product_rfft_padded[
        neg_z, :half_nx, :half_ny
    ]
    product_rfft[:half_nz, -half_nx + 1 :, :half_ny] = product_rfft_padded[
        :half_nz, -half_nx + 1 :, :half_ny
    ]
    product_rfft[neg_z, -half_nx + 1 :, :half_ny] = product_rfft_padded[
        neg_z, -half_nx + 1 :, :half_ny
    ]

    return product_rfft
