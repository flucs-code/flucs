"""A collection of functions for dealing with 3-smooth numbers,
i.e., numbers divisible only by 2 and 3.
"""
import heapq


def smallest_larger_threesmooth(n : int) -> int:
    """Returns the smallest 3-smooth number that is bigger
    than a given number n.

    Parameters
    ----------
    n : int

    Returns
    -------
    int
        The smallest 3-smooth number bigger than n.

    """

    # Use a heap to keep track of the smallest 3-smooth number
    # we have seen so far, and generate them in ascending order.
    heap = [1]

    while True:
        guess = heapq.heappop(heap)

        if n <= guess:
            return guess

        heapq.heappush(heap, 2 * guess)
        heapq.heappush(heap, 3 * guess)
