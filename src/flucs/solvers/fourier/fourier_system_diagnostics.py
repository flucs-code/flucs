from collections.abc import Callable
from typing import TYPE_CHECKING, ClassVar

import cupy as cp
import numpy as np

from flucs.diagnostic import FlucsDiagnostic, FlucsDiagnosticVariable

if TYPE_CHECKING:
    from .fourier_system import FourierSystem


class LinearSpectrumDiag(FlucsDiagnostic):
    """
    Calculates the linear frequency at the current time step by comparing with
    the fields at the previous one.

    """

    name = "linear_spectrum"
    is_complex = True
    system: "FourierSystem"

    # Speficies which field to use for estimating linear frequencies
    field_index: int = 0

    def ready(self):
        dimensions_dict = {
            "kx": self.system.kx,
            "ky": self.system.ky,
            "kz": self.system.kz,
        }

        self.add_var(
            FlucsDiagnosticVariable(
                name="omega",
                shape=("kz", "kx", "ky"),
                dimensions_dict=dimensions_dict,
                is_complex=True,
            )
        )

    def execute(self):
        # Do not execute at first time step
        if self.system.current_step == 0:
            return np.zeros(
                self.system.half_unpadded_tuple, dtype=self.system.complex
            )

        alpha = self.system.input["setup.alpha"]
        current_field = self.system.fields[self.system.current_step % 2][
            self.field_index, :
        ]

        previous_field = self.system.fields[self.system.current_step % 2 - 1][
            self.field_index, :
        ]

        self.vars["omega"].data_cache.append(
            cp.as_numpy(
                (1j / self.system.current_dt)
                * (current_field - previous_field)
                / (alpha * current_field + (1 - alpha) * previous_field)
            )
        )


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
