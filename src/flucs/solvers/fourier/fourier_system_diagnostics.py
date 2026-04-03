from collections.abc import Callable
from typing import TYPE_CHECKING, ClassVar

import cupy as cp
import numpy as np

from flucs.diagnostic import FlucsDiagnostic, FlucsDiagnosticVariable
from flucs.solvers import FlucsSolverState

if TYPE_CHECKING:
    from .fourier_system import FourierSystem


class LinearEigensystemDiag(FlucsDiagnostic):
    """
    to be added

    """

    name = "linear_eigensystem"
    option_defaults: ClassVar[dict[str, object]] = {
        "field_index": 0, 
        "tolerance": -1.0,
        "amplitude_overflow": 1e+16, 
        "amplitude_underflow": 1e-16,
        "init_only": False 
        }
    system: "FourierSystem"

    def init_vars(self):
        mode = np.arange(self.system.number_of_fields)
        field = np.arange(self.system.number_of_fields)

        self.add_var(
            FlucsDiagnosticVariable(
                name="eigvals_solver",
                shape=("mode", "kz", "kx", "ky"),
                dimensions={
                    "mode": mode,
                    "kz": self.system.kz,
                    "kx": self.system.kx,
                    "ky": self.system.ky,
                },
                is_complex=True,
            )
        )

        self.add_var(
            FlucsDiagnosticVariable(
                name="eigvecs_solver",
                shape=("mode", "field", "kz", "kx", "ky"),
                dimensions={
                    "mode": mode,
                    "field": field,
                    "kz": self.system.kz,
                    "kx": self.system.kx,
                    "ky": self.system.ky,
                },
                is_complex=True,
            )
        )

        self.add_var(
            FlucsDiagnosticVariable(
                name="eigvals_reference",
                shape=("mode", "kz", "kx", "ky"),
                dimensions={
                    "mode": mode,
                    "kz": self.system.kz,
                    "kx": self.system.kx,
                    "ky": self.system.ky,
                },
                is_complex=True,
            )
        )

        self.add_var(
            FlucsDiagnosticVariable(
                name="eigvecs_reference",
                shape=("mode", "field", "kz", "kx", "ky"),
                dimensions={
                    "mode": mode,
                    "field": field,
                    "kz": self.system.kz,
                    "kx": self.system.kx,
                    "ky": self.system.ky,
                },
                is_complex=True,
            )
        )

        self.add_var(
            FlucsDiagnosticVariable(
                name="eigvals",
                shape=("mode", "kz", "kx", "ky"),
                dimensions={
                    "mode": mode,
                    "kz": self.system.kz,
                    "kx": self.system.kx,
                    "ky": self.system.ky,
                },
                is_complex=True,
            )
        )

        self.add_var(
            FlucsDiagnosticVariable(
                name="eigvals_tolerance",
                shape=("mode", "kz", "kx", "ky"),
                dimensions={
                    "mode": mode,
                    "kz": self.system.kz,
                    "kx": self.system.kx,
                    "ky": self.system.ky,
                },
                is_complex=False,
            )
        )

    def ready(self):
        pass

    def execute(self):

        # Get eigensystem from solver and save to diagnostics
        eigensystem = self.system.compute_linear_eigensystem()

        self.save_data(
            "eigvals_solver", eigensystem["eigvals_solver"]
        )
        self.save_data(
            "eigvecs_solver", eigensystem["eigvecs_solver"]
        )
        self.save_data(
            "eigvals_reference", eigensystem["eigvals_reference"]
        )
        self.save_data(
            "eigvecs_reference", eigensystem["eigvecs_reference"]
        )

        # Placeholder values until runtime omega diagnostics is implemented
        nan_omega = np.full(
            (self.system.number_of_fields, *self.system.half_unpadded_tuple),
            np.nan + 1j * np.nan,
            dtype=self.system.complex,
        )
        nan_tol = np.full(
            (self.system.number_of_fields, *self.system.half_unpadded_tuple),
            np.nan,
            dtype=self.system.float,
        )

        self.save_data("eigvals", nan_omega)
        self.save_data("eigvals_tolerance", nan_tol)

        # Interrupt the solver if it is actually running and the user only wants
        # to inspect the eigensystem associated with the CUDA matrix, and not 
        # how it is reproduced by the timestepping scheme. 
        if (
            self.init_only
            and self.system.solver.state == FlucsSolverState.RUNNING
        ):
            self.system.solver.interrupted = True

    # def execute(self):
    #     # Do not execute at first time step
    #     if self.system.current_step == 0:
    #         return np.zeros(
    #             self.system.half_unpadded_tuple, dtype=self.system.complex
    #         )

    #     alpha = self.system.input["setup.alpha"]
    #     current_field = self.system.fields[self.system.current_step % 2][
    #         self.field_index, :
    #     ]

    #     previous_field = self.system.fields[self.system.current_step % 2 - 1][
    #         self.field_index, :
    #     ]

    #     self.vars["eigvals"].data_cache.append(
    #         cp.as_numpy(
    #             (1j / self.system.current_dt)
    #             * (current_field - previous_field)
    #             / (alpha * current_field + (1 - alpha) * previous_field)
    #         )
    #     )


class FourierDataDiag(FlucsDiagnostic):
    """
    Outputs 1D, 2D, or 3D `locations` of the Fourier-space data.

    Requires specifying diagnostic options with the following structure:
    {
        locations = [location0, location1, ...]
    }

    where each of location0, location1, etc, are strings with the format

        'ifield, ikz, ikx, iky'

    where ifield, ikz, ikx, and iky specify the Fourier-space indices at
    which is saved. Simple NumPy-like slicing of format a:b:c is allowed.
    For example, '0, :, :, :' saves all Fourier data for field 0.
    """

    name = "fourier_data"
    option_defaults: ClassVar[dict[str, object]] = {"locations": list()}
    type: str
    slices: dict
    slice_calculators: list[Callable[[], None]]

    def init_vars(self):
        system = self.system  # Save some space
        self.slice_calculators = []

        def parse_slice(s: str) -> slice:
            # Check if slice syntax or just a single index
            if ":" not in s:
                index = int(s.strip())
                return slice(index, index + 1)

            parts = s.split(":")
            if len(parts) > 3:
                raise ValueError(f"Invalid slice string: {s}")

            def get_index(i):
                return None if i == "" else int(i)

            return slice(*(get_index(p) for p in parts))

        for location in self.locations:
            loc = ",".join(part.strip() for part in location.split(","))
            loc_name = f"location_{loc}"

            try:
                loc_parts = loc.split(",")

                ifield = parse_slice(loc_parts[0])
                ikz = parse_slice(loc_parts[1])
                ikx = parse_slice(loc_parts[2])
                iky = parse_slice(loc_parts[3])

            except (IndexError, ValueError):
                raise ValueError(
                    f"'{loc}' is not a valid Fourier location. "
                    r"The correct location format is '(ifield, ikz, ikx, iky)' "
                    r"where the indices are integers or slices 'a:b:c'."
                )

            self.system._precompute_wavenumbers()

            dimensions = {
                f"{loc_name}/field": np.arange(system.number_of_fields)[ifield],
                f"{loc_name}/kz": system.kz[ikz],
                f"{loc_name}/kx": system.kx[ikx],
                f"{loc_name}/ky": system.ky[iky],
            }

            self.add_var(
                FlucsDiagnosticVariable(
                    name=f"{loc_name}/data",
                    shape=("field", "kz", "kx", "ky"),
                    dimensions=dimensions,
                    is_complex=True,
                )
            )

            def slice_calculator(
                loc_name=loc_name,
                ifield=ifield,
                ikz=ikz,
                ikx=ikx,
                iky=iky,
            ):
                self.vars[f"{loc_name}/data"].data_cache.append(
                    system.fields[
                        system.current_step % system.fields_history_size
                    ][ifield, ikz, ikx, iky].get()
                )

            self.slice_calculators.append(slice_calculator)

    def ready(self):
        pass

    def execute(self):
        for slice_calculator in self.slice_calculators:
            slice_calculator()


class RealspaceDataDiag(FlucsDiagnostic):
    """
    Outputs 1D, 2D, or 3D `locations` of the real-space data.

    Requires specifying diagnostic options with the following structure:
    {
        locations = [location0, location1, ...]
    }

    where each of location0, location1, etc, are strings with the format

        'ifield, iz, ix, iy'

    where ifield, iz, ix, and iy specify the real-space indices at
    which is saved. Simple NumPy-like slicing of format a:b:c is allowed.
    For example, '0, 0, :, :' produces a cut of field 0 in the z=0 plane.
    """

    name = "realspace_data"
    option_defaults: ClassVar[dict[str, object]] = {"locations": list()}
    type: str
    slices: dict
    slice_calculators: list[Callable[[], None]]

    def init_vars(self):
        self.slice_calculators = []

        def parse_slice(s: str) -> slice:
            # Check if slice syntax or just a single index
            if ":" not in s:
                index = int(s.strip())
                return slice(index, index + 1)

            parts = s.split(":")
            if len(parts) > 3:
                raise ValueError(f"Invalid slice string: {s}")

            def get_index(x):
                return None if x == "" else int(x)

            return slice(*(get_index(p) for p in parts))

        for location in self.locations:
            loc = ",".join(part.strip() for part in location.split(","))
            loc_name = f"location_{loc}"

            try:
                loc_parts = loc.split(",")

                ifield = parse_slice(loc_parts[0])
                iz = parse_slice(loc_parts[1])
                ix = parse_slice(loc_parts[2])
                iy = parse_slice(loc_parts[3])

            except (IndexError, ValueError):
                raise ValueError(
                    f"'{loc}' is not a valid realspace location. "
                    r"The correct location format is '(ifield, iz, ix, iy)' "
                    r"where the indices are integers or slices 'a:b:c'."
                )

            dimensions = {
                f"{loc_name}/field": np.arange(self.system.number_of_fields)[
                    ifield
                ],
                f"{loc_name}/z": np.linspace(
                    0, self.system.input["dimensions.Lz"], self.system.nz
                )[iz],
                f"{loc_name}/x": np.linspace(
                    0, self.system.input["dimensions.Lx"], self.system.nx
                )[ix],
                f"{loc_name}/y": np.linspace(
                    0, self.system.input["dimensions.Ly"], self.system.ny
                )[iy],
            }

            self.add_var(
                FlucsDiagnosticVariable(
                    name=f"{loc_name}/data",
                    shape=("field", "z", "x", "y"),
                    dimensions=dimensions,
                    is_complex=False,
                )
            )

            def slice_calculator(
                loc_name=loc_name,
                ifield=ifield,
                iz=iz,
                ix=ix,
                iy=iy,
            ):
                self.vars[f"{loc_name}/data"].data_cache.append(
                    self.system.realspace_fields[ifield, iz, ix, iy]
                )

            self.slice_calculators.append(slice_calculator)

    def ready(self):
        pass

    def execute(self):
        self.system.get_realspace_fields()

        for slice_calculator in self.slice_calculators:
            slice_calculator()
