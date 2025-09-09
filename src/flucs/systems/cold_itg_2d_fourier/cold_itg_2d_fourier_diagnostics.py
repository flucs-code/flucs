import numpy as np
import cupy as cp
from flucs.output import FlucsDiagnostic


class HeatfluxDiag(FlucsDiagnostic):
    name = "heatflux"
    shape = ()
    dimensions_dict = {}

    # d/dy operator in Fourier space
    dy: cp.ndarray

    def ready(self):
        # Copy ky wavenumbers to GPU memory for faster multiplication
        self.dy = cp.array(1j * self.system.ky)

    def get_data(self):
        phi = self.system.phi[self.system.current_field_marker]
        T = self.system.T[self.system.current_field_marker]

        return -2 * cp.sum(self.dy * phi * cp.conjugate(T)).item().real

    def print_diagnostic(self):
        print(f"Heat flux at time step {self.system.current_step} (time {self.system.current_time}) is {self.data_cache[-1]}")
