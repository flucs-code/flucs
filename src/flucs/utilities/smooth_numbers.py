import heapq


def next_smooth_number(n: int, primes: list | None = None) -> int:
    """Returns the smallest number that is strictly bigger than a given number n
    and divisible only by the prime numbers specified in primes.

    Parameters
    ----------
    n
        The number to find the next smooth number for.
    primes
        List of primes. Defaults to ``[2, 3]``.  The algorithm assumes but does
        not check that these numbers are prime.

    Returns
    -------
        The smallest 3-smooth number bigger than ``n``.

    """

    if primes is None:
        primes = [2, 3]

    # Use a heap to keep track of the smallest smooth number
    # we have seen so far, and generate them in ascending order.
    heap = [1]

    while True:
        guess = heapq.heappop(heap)

        if n < guess:
            return guess

        for p in primes:
            # It might be a good idea to check whether p * guess
            # is way too large to even consider. Might speed things
            # a bit but not really worth it given that it's unlikely
            # we will ever use this for n > 10^4 or so.
            heapq.heappush(heap, p * guess)
