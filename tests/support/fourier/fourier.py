r"""
Three-field system for testing FourierSolver. 

The system evolves:

    \boldsymbol{u} = (u_z, u_x, u_y),

with components stored in the same order as the Fourier grid
(kz, kx, ky). It obeys

    \partial_t \boldsymbol{u}
    + \boldsymbol{c}\mathbin{\cdot}\nabla\boldsymbol{u}
    + \boldsymbol{\Omega}\times\boldsymbol{u}
    + \boldsymbol{N}(\boldsymbol{u}) = \boldsymbol{f},

where

    N_z &= \partial_y(u_y^2 / 2) + \partial_z(u_x * u_z), \\
    N_x &= \partial_x(u_x * u_y) + \partial_z(u_z^2 / 2), \\
    N_y &= \partial_x(u_x^2 / 2) + \partial_y(u_y * u_z).

The equations are a numerical test fixture rather than a physical model.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from flucs import cupy as cp
from flucs.diagnostic import FlucsDiagnostic
from flucs.input import InvalidFlucsInputFileError
from flucs.solvers.fourier.fourier_system import FourierSystem
from flucs.solvers.fourier.fourier_system_forcing import FourierSystemForcing
from flucs.utilities.cupy import KernelWrapper

from .fourier_diagnostics import FreeEnergyDiag, FreeEnergyDiag1D
from .fourier_forcing import TestFourierNegativeDampingForcing


class TestFourierSystem(FourierSystem):
    """
    Fourier solver for the test system for FourierSolver.

    """

    # Prevent pytest from trying to collect the support class as a test class.
    __test__ = False

    number_of_fields = 3
    number_of_dft_derivatives = 3
    number_of_dft_bits = 6

    # Vector parameters use Fourier-axis order (z, x, y).
    advection: np.ndarray
    rotation: np.ndarray

    # CUDA kernels
    find_derivatives_kernel: KernelWrapper
    find_nonlinear_bits_kernel: KernelWrapper

    # Supported diagnostics
    diags: ClassVar[set[type[FlucsDiagnostic]]] = {
        FreeEnergyDiag,
        FreeEnergyDiag1D
    }

    # Supported forcing
    system_forcing_methods: ClassVar[dict[str, type[FourierSystemForcing]]] = {
        "negative_damping": TestFourierNegativeDampingForcing,
    }

    def ready(self) -> None:
        super().ready()

    def register_kernels(self) -> None:
        super().register_kernels()

        nonlinear_bits_shared_mem = self.cuda_block_size * self.float().nbytes

        # Register kernels
        self.find_derivatives_kernel = KernelWrapper(
            system=self,
            cuda_kernel_name="find_derivatives",
            grid=(self.half_cuda_grid_size,),
            block=(self.cuda_block_size,),
        )
        self.find_nonlinear_bits_kernel = KernelWrapper(
            system=self,
            cuda_kernel_name="find_nonlinear_bits",
            grid=(self.full_cuda_grid_size,),
            block=(self.cuda_block_size,),
            shared_mem=nonlinear_bits_shared_mem,
        )

        # Register functions
        def find_derivatives_function(
            current_dt,
            current_time,
            current_step,
            fields: cp.ndarray,
            memory_dict: dict,
        ) -> None:
            self.find_derivatives_kernel(
                fields,
                memory_dict["first_intermediates_fourier"],
            )

        def find_nonlinear_bits_function(
            current_dt,
            current_time,
            current_step,
            calculate_cfl: bool,
            memory_dict: dict,
        ) -> None:
            self.find_nonlinear_bits_kernel(
                memory_dict["first_intermediates_real"],
                memory_dict["second_intermediates_real"],
                calculate_cfl,
                self.cfl_rate,
            )

        if not self.input["setup.linear"]:
            self.dft_derivatives_operation, self.dft_bits = (
                self.create_dealiased_operation(
                    n_in=self.number_of_dft_derivatives,
                    n_out=self.number_of_dft_bits,
                    create_first_intermediates=find_derivatives_function,
                    create_second_intermediates=find_nonlinear_bits_function,
                    combine_first_and_second_intermediates=True,
                )
            )

    def _interpret_input(self) -> None:
        """
        Check the system-specific input parameters.
        """

        # Perform the standard FourierSystem input setup first
        super()._interpret_input()

        # Read and validate vector parameters
        for parameter in ("advection", "rotation"):

            # Try to parse as an array
            try:
                value = np.asarray(
                    self.input[f"parameters.{parameter}"], dtype=self.float
                )
            except (TypeError, ValueError) as exc:
                raise InvalidFlucsInputFileError(
                    f"parameters.{parameter} must contain three real numbers."
                ) from exc

            # Check that the array is a finite 3-vector
            if value.shape != (3,) or not np.all(np.isfinite(value)):
                raise InvalidFlucsInputFileError(
                    f"parameters.{parameter} must contain three finite real "
                    f"numbers."
                )
            
            # Assign value
            self.__setattr__(parameter, value)

    def _set_initial_conditions(self) -> None:
        """
        Construct smooth deterministic initial conditions.
        """

        # Use restart data if it was read
        if self.restart_manager.data is not None:
            super()._set_initial_conditions()
            return

        # Handle known initialisation methods
        match self.input["init.method"]:

            case "deterministic":

                # Create realspace meshgrid
                x = 2.0 * np.pi * np.arange(self.nx) / self.nx
                y = 2.0 * np.pi * np.arange(self.ny) / self.ny
                z = 2.0 * np.pi * np.arange(self.nz) / self.nz
                zz, xx, yy = np.meshgrid(z, x, y, indexing="ij")

                # Amplitude of initial perturbation
                amplitude = self.float(self.input["init.amplitude"])

                # Initial wave solutions
                ux = amplitude * (np.sin(xx) + 0.5 * np.cos(yy + zz))
                uy = amplitude * (np.cos(yy) + 0.4 * np.sin(zz + xx))
                uz = amplitude * (np.sin(zz) + 0.3 * np.cos(xx - yy))

                # Transform to Fourier space and store
                self.fields_initial = np.fft.rfftn(
                    np.stack((uz, ux, uy)),
                    axes=(-3, -2, -1),
                    norm="forward",
                ).astype(self.complex)

            case _:
                # Fall back to the generic FourierSystem initial conditions
                super()._set_initial_conditions()

    def setup_cuda_definitions(self) -> None:
        """
        Add system-specific CUDA definitions.
        """

        # Advection vector
        for component, value in zip("ZXY", self.advection, strict=True):
            self.module_options.define_float(f"ADVECTION_{component}", value)

        # Rotation vector
        for component, value in zip("ZXY", self.rotation, strict=True):
            self.module_options.define_float(f"ROTATION_{component}", value)

        super().setup_cuda_definitions()

    def begin_time_step(self) -> None:
        super().begin_time_step()

    def compute_nonlinear_terms(
        self,
        current_dt,
        current_time,
        current_step,
        fields: cp.ndarray,
        calculate_cfl: bool,
    ) -> None:
        """
        Compute the six dealiased quadratic fluxes.
        """

        self.dft_derivatives_operation(
            current_dt,
            current_time,
            current_step,
            fields,
            calculate_cfl,
        )

    def finish_time_step(self) -> None:
        super().finish_time_step()

    def compute_linear_matrix_reference(self) -> np.ndarray:
        """
        Return the exact CPU representation of the linear operator.
        """

        # Initialise linear matrix
        linear_matrix = np.zeros(
            (
                self.number_of_fields,
                self.number_of_fields,
                *self.half_tuple,
            ),
            dtype=self.complex,
        )

        # Uniform advection
        advection_frequency = self._compute_advection_frequency()

        for field in range(self.number_of_fields):
            linear_matrix[field, field] = 1j * advection_frequency

        # Rotation in field order (uz, ux, uy)
        omega_z, omega_x, omega_y = self.rotation
        linear_matrix[0, 1] = - omega_y
        linear_matrix[0, 2] = + omega_x
        linear_matrix[1, 0] = + omega_y
        linear_matrix[1, 2] = - omega_z
        linear_matrix[2, 0] = - omega_x
        linear_matrix[2, 1] = + omega_z

        return linear_matrix

    def compute_linear_frequencies_reference(self) -> np.ndarray:
        """
        Return the analytical dispersion relation on the Fourier grid.
        """

        # Frequencies 
        advection_frequency = self._compute_advection_frequency()
        rotation_frequency = np.linalg.norm(self.rotation)

        # Return solutions
        return np.stack(
            (
                advection_frequency,
                advection_frequency + rotation_frequency,
                advection_frequency - rotation_frequency,
            ),
            axis=0,
        ).astype(self.complex)

    def _compute_advection_frequency(self) -> np.ndarray:
        """
        Compute the advective frequency on the Fourier grid.
        """

        # Wavenumbers
        kz, kx, ky = self.get_broadcast_wavenumbers()

        return (
            + self.advection[0] * kz
            + self.advection[1] * kx
            + self.advection[2] * ky
        )
