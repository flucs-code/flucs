import pytest

from flucs.utilities.smooth_numbers import next_smooth_number


@pytest.mark.parametrize(
    "n, primes, expected",
    [
        pytest.param(0, None, 1, id="3-smooth, n=0"),
        pytest.param(3, None, 4, id="3-smooth, n=3"),
        pytest.param(4, None, 6, id="3-smooth, n=4"),
        pytest.param(9, None, 12, id="3-smooth, n=9"),
        pytest.param(10, None, 12, id="3-smooth, n=10"),
        pytest.param(121, None, 128, id="3-smooth, n=121"),
        pytest.param(0, [2, 3, 5], 1, id="5-smooth, n=0"),
        pytest.param(3, [2, 3, 5], 4, id="5-smooth, n=3"),
        pytest.param(4, [2, 3, 5], 5, id="5-smooth, n=4"),
        pytest.param(9, [2, 3, 5], 10, id="5-smooth, n=9"),
        pytest.param(10, [2, 3, 5], 12, id="5-smooth, n=10"),
        pytest.param(121, [2, 3, 5], 125, id="5-smooth, n=121"),
    ],
)
def test_next_smooth_number(n: int, primes: list[int] | None, expected: int):
    assert next_smooth_number(n, primes=primes) == expected
