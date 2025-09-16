import numpy as np
import cupy as cp
from flucs.output import FlucsDiagnostic
from .fourier_system import FourierSystem


class LinearSpectrumDiag(FlucsDiagnostic):
    name = "linear_spectrum"
    system: FourierSystem
    is_complex = True

    # Speficies which field to use for estimating linear frequencies
    field_index: int = 0

    def ready(self):
        self.shape = ("kz", "kx", "ky")
        self.dimensions_dict = {"kx": self.system.kx,
                                "ky": self.system.ky,
                                "kz": self.system.kz}

    def get_data(self):
        # Do not execute at first time step
        if self.system.current_step == 0:
            return np.zeros(self.system.lattice_tuple, dtype=self.system.complex)

        alpha = self.system.input["setup.alpha"]
        current_field = \
            self.system.fields[self.system.current_field_marker][self.field_index, :]

        previous_field =\
            self.system.fields[self.system.current_field_marker - 1][self.field_index, :]

        return cp.asnumpy(
            (1j/self.system.current_dt)
            * (current_field - previous_field)
            / (alpha*current_field + (1 - alpha)*previous_field))

    def print_diagnostic(self):
        print("TODO: implement something for the linear-spectrum diagnostic")
