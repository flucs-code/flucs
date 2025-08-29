"""Abstract base class for a system that can be solved by FourierSolver.
"""

import numpy as np
from flucs.systems import FlucsSystem
from flucs import FlucsInput
from flucs.utilities.smooth_numbers import next_smooth_number

try:
    import cupy as cp
except ModuleNotFoundError:
    print("CuPy not found!")

class FourierSystem(FlucsSystem):
    """A generic system of equations solved using pseudospectral Fourier methods."""
    # This will hold all the fields. Should be a list of NumPy-like arrays.
    # It's a list in order to store fields at previous time steps, as required
    # by the algorithm.
    fields: list = None


    # Array sizes
    nx: int
    ny: int
    nz: int
    half_nx: int
    half_ny: int
    half_nz: int

    padded_nx: int
    padded_ny: int
    padded_nz: int
    half_padded_nx: int
    half_padded_ny: int
    half_padded_nz: int

    grid_size: int
    padded_grid_size: int
    real_grid_size: int
    real_padded_grid_size: int

    # Variables to that keep track of time
    current_step: int
    current_dt: float
    current_time: float


    def _validate_input(self):
        """Validates the input file
        and sets up the resolution in Fourier space
        """

        # Set resolutions appropriately
        for dim in ["x", "y", "z"]:
            n = self.input[f"dimensions.n{dim}"]
            padded_n = self.input[f"dimensions.padded_n{dim}"]

            match (n > 0, padded_n > 0):
                case (True, True):
                    # Check if n is odd
                    if n % 2 == 0:
                        raise ValueError(f"Unpadded resolutions must be odd! Please change n{dim} = {n} to an odd number!")

                    # TODO: add some check that warns the user if their choice was dumb

                case (True, False):
                    # Check if n is odd
                    if n % 2 == 0:
                        raise ValueError(f"Unpadded resolutions must be odd! Please change n{dim} = {n} to an odd number!")

                    half_n = n // 2

                    # Find minimum padded that works
                    padded_n = next_smooth_number(
                        (self.input["dimensions.nonlinear_order"] + 1) * half_n,
                        primes=self.input["dimensions.padded_primes"])

                    half_padded_n = padded_n // 2

                    print(f"Found padded_n{dim} = {padded_n} for n{dim} = {n}.")

                case (False, True):
                    # Given a padded_n, it's easiest to figure out half_n

                    factor = self.input["dimensions.nonlinear_order"] + 1
                    half_n = padded_n // factor

                    # Handle an annoying edge case
                    if padded_n % factor == 0:
                        half_n -= 1

                    n = 2*half_n + 1

                    print(f"Found n{dim} = {n} for padded_n{dim} = {padded_n}.")

                case (False, False):
                    raise ValueError(f"At least one of n{dim} and padded_n{dim} must be positive!")

            # It's useful to have the resolutions as part of the system
            # rather than to access the input dictionary every time
            setattr(self, f"n{dim}", n)
            setattr(self, f"padded_n{dim}", padded_n)
            setattr(self, f"half_n{dim}", half_n)
            setattr(self, f"half_padded_n{dim}", half_padded_n)

        # Set padded and unpadded array sizes
        self.padded_grid_size = self.padded_nz * self.padded_nx * self.half_padded_ny
        self.grid_size = self.nz * self.nx * self.half_ny

        self.real_padded_grid_size = self.padded_nz * self.padded_nx * self.padded_ny
        self.real_grid_size = self.nz * self.nx * self.ny


    def set_initial_conditions(self):
        # Basic setup
        self.current_step = self.int(0)
        self.current_time = self.float(0.0)
        self.current_dt = self.float(self.input["time.dt"])

        # Handle known initialisation types

        # Depending on whether we are running with NumPy or Cupy, we might need
        # to convert the data to different array objects.
        if type(self.fields[0]) is cp.ndarray:
            to_array = cp.array
        elif type(self.fields[0]) is np.ndarray:
            to_array = np.array
        else:
            raise TypeError(f"Solver {type(self)} uses an unknown type of data arrays!")

        match self.input["init.type"]:

            case "white_noise":
                np.random.seed(self.input["init.rand_seed"])
                self.fields[0][:] = to_array(self.input["init.amplitude"] *\
                    np.random.random(self.fields[0].shape))

            case _:
                # Exotic initialisation types should be handled by each solver
                # separately.
                pass
