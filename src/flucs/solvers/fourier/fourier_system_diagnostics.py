from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, ClassVar

import cupy as cp
import numpy as np

from flucs.diagnostic import FlucsDiagnostic, FlucsDiagnosticVariable
from flucs.solvers import FlucsSolverState
from flucs.utilities.messages import flucsprint

if TYPE_CHECKING:
    from flucs.solvers.fourier.fourier_system import FourierSystem


class LinearEigensystemDiag(FlucsDiagnostic):
    """
    Outputs the linear eigensystem of a given FourierSystem. This includes the
    eigenvalues and eigenvectors computed directly from the linear matrix used
    by the solver, as well as a reference matrix if provided by the user.

    The runtime diagnostic calculates the eigenvalues directly from the time-
    evolution of the fields by projecting the solutions onto the precomputed
    eigenvectors. This diagnostic will exit early if any amplitudes reach an
    overflow threshold, or if the eigenvalues converge within a specified
    tolerance, defined as:

        tolerance := abs(eigvals_{n} - eigvals_{n-1}) / abs(eigvals_{n})

    where eigvals_{n} are the eigenvalues calculated from the current time step.
    If tolerance is set to be negative, this exit condition is disabled.

    Options
    ----------
    tolerance : float
        Tolerance for the eigenvalue calculation.
    init_only : bool
        If True, the diagnostic will execute for a single timestep to load the
        solver/reference linear eigensystem.
    save_eigvecs : bool
        Whether to save the eigenvectors of the solver/reference eigensystems.

    """

    name = "linear_eigensystem"
    option_defaults: ClassVar[dict[str, object]] = {
        "tolerance": 1e-6,
        "init_only": False,
        "save_eigvecs": False,
    }
    system: FourierSystem

    def init_vars(self):
        if not self.system.input["setup.linear"]:
            flucsprint(
                "running nonlinearly.",
                source=self,
                message_type="warning"
            )

        field = np.arange(self.system.number_of_fields)

        self.add_var(
            FlucsDiagnosticVariable(
                name="eigvals_solver",
                shape=("mode", "kz", "kx", "ky"),
                dimensions={
                    "mode": field,
                    "kz": self.system.kz,
                    "kx": self.system.kx,
                    "ky": self.system.ky,
                },
                is_complex=True,
                is_time_dependent=False,
            )
        )

        self.add_var(
            FlucsDiagnosticVariable(
                name="eigvals",
                shape=("mode", "kz", "kx", "ky"),
                dimensions={
                    "mode": field,
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
                    "mode": field,
                    "kz": self.system.kz,
                    "kx": self.system.kx,
                    "ky": self.system.ky,
                },
                is_complex=False,
            )
        )

        self.add_var(
            FlucsDiagnosticVariable(
                name="eigvals_amplitude",
                shape=("mode", "kz", "kx", "ky"),
                dimensions={
                    "mode": field,
                    "kz": self.system.kz,
                    "kx": self.system.kx,
                    "ky": self.system.ky,
                },
                is_complex=False,
            )
        )

        # Optionally save eigenvectors
        if self.save_eigvecs:
            self.add_var(
                FlucsDiagnosticVariable(
                    name="eigvecs_solver",
                    shape=("mode", "field", "kz", "kx", "ky"),
                    dimensions={
                        "mode": field,
                        "field": field,
                        "kz": self.system.kz,
                        "kx": self.system.kx,
                        "ky": self.system.ky,
                    },
                    is_complex=True,
                    is_time_dependent=False,
                )
            )

    def ready(self):
        # Cache inverses for mode projection
        eigensystem = self.system.compute_linear_eigensystem()
        self.eigvecs_inverse = cp.asarray(eigensystem["eigvecs_inverse"])

        self.save_data("eigvals_solver", eigensystem["eigvals"])
        if self.save_eigvecs:
            self.save_data("eigvecs_solver", eigensystem["eigvecs"])

        # Initialise fill values
        shape = (self.system.number_of_fields, *self.system.half_unpadded_tuple)
        fill_value = cp.nan

        self.eigvals_fill = cp.full(
            shape, fill_value + 1j * fill_value, dtype=self.system.complex
        )
        self.eigvals_tolerance_fill = cp.full(
            shape, fill_value, dtype=self.system.float
        )

        # Runtime diagnostic variables
        self.previous_eigvals = None
        self.amplitude_overflow = self.system.float(
            1e-1 * np.sqrt(np.finfo(self.system.float).max)
        )

    def execute(self):
        # Get fields
        current_fields = self.system.fields[
            self.system.current_step % self.system.fields_history_size
        ]

        previous_fields = self.system.fields[
            self.system.current_step % self.system.fields_history_size - 1
        ]

        # Project onto solver eigenvectors
        current_amplitude = cp.einsum(
            "mfzxy,fzxy->mzxy",
            self.eigvecs_inverse,
            current_fields,
        )

        abs_current_amplitude = cp.abs(current_amplitude)

        # Get previous time data
        initial_execution = self.system.current_step == 0

        previous_amplitude = (
            current_amplitude
            if initial_execution
            else cp.einsum(
                "mfzxy,fzxy->mzxy",
                self.eigvecs_inverse,
                previous_fields,
            )
        )
        previous_eigvals = (
            self.eigvals_fill if initial_execution else self.previous_eigvals
        )
        time_interval = (
            self.system.float(1.0)
            if initial_execution
            else self.system.current_dt
        )

        # Compute eigenvalues and tolerance
        current_eigvals = self.eigvals_fill.copy()
        valid = (abs_current_amplitude > self.system.float(0.0)) & (
            cp.abs(previous_amplitude) > self.system.float(0.0)
        )
        current_eigvals[valid] = (1j / time_interval) * cp.log(
            current_amplitude[valid] / previous_amplitude[valid]
        )

        eigvals_tolerance = self.eigvals_tolerance_fill.copy()
        valid_tolerance = (
            valid
            & cp.isfinite(previous_eigvals)
            & (cp.abs(current_eigvals) > self.system.float(0.0))
        )

        eigvals_tolerance[valid_tolerance] = cp.abs(
            current_eigvals[valid_tolerance] - previous_eigvals[valid_tolerance]
        ) / cp.abs(current_eigvals[valid_tolerance])

        # Save data
        self.save_data("eigvals", cp.asnumpy(current_eigvals))
        self.save_data("eigvals_tolerance", cp.asnumpy(eigvals_tolerance))
        self.save_data("eigvals_amplitude", cp.asnumpy(abs_current_amplitude))

        # Reset previous eigval data for tolerance calculation
        self.previous_eigvals = current_eigvals.copy()

        # Exit conditions
        if self.system.solver.state == FlucsSolverState.RUNNING:
            overflow = bool(
                cp.any(abs_current_amplitude > self.amplitude_overflow).get()
            )

            converged = (
                self.system.float(self.tolerance) > self.system.float(0.0)
                and bool(cp.any(valid_tolerance).get())
                and bool(
                    cp.all(
                        eigvals_tolerance[valid_tolerance]
                        < self.system.float(self.tolerance)
                    ).get()
                )
            )

            if self.init_only or overflow or converged:
                if self.init_only:
                    message = "init only"
                if overflow:
                    message = "amplitude overflow"
                if converged:
                    message = "converged"

                flucsprint(message, source=self)
                self.system.solver.interrupted = True


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
