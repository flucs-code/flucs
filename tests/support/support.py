"""
Shared parameter sets and numerical policy for the FLUCS test suite.
"""

from dataclasses import dataclass

import numpy as np

FLUCS_TOLERANCE_MULTIPLIER = 64.0


@dataclass(frozen=True)
class TestPrecision:
    """
    Types, storage metadata, and tolerance for one FLUCS precision.
    """

    name: str
    float_type: type
    complex_type: type
    netcdf_precision: str

    @property
    def tolerance(self):
        """
        Return the baseline round-off tolerance used by FLUCS.
        """
        return self.float_type(
            np.finfo(self.float_type).eps * FLUCS_TOLERANCE_MULTIPLIER
        )


# Exercise every supported precision through the same behavioural tests
SINGLE_PRECISION = TestPrecision(
    name="single",
    float_type=np.float32,
    complex_type=np.complex64,
    netcdf_precision="f4",
)
DOUBLE_PRECISION = TestPrecision(
    name="double",
    float_type=np.float64,
    complex_type=np.complex128,
    netcdf_precision="f8",
)
TEST_PRECISIONS = (SINGLE_PRECISION, DOUBLE_PRECISION)
