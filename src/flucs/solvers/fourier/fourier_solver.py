"""
Pseudospectral Fourier-space solver.

Solves a system of PDEs in a periodic box using
pseudospectral Fourier methods.

"""

import time
from typing import ClassVar

import cupy as cp
import numpy as np

from flucs.solvers import FlucsSolver, FlucsSolverState, FlucsTimestepper
from flucs.solvers.fourier.fourier_system import FourierSystem
from flucs.utilities.cupy import KernelWrapper, cupy_set_device_pointer
from flucs.utilities.messages import flucsprint


class FourierAB3Timestepper(FlucsTimestepper[FourierSystem]):
    ab3_type: str
    is_nonlinear: bool

    dt_array: np.ndarray
    ab3_coefficients: np.ndarray

    # CUDA kernels
    precompute_iteration_matrices_kernel: KernelWrapper
    finish_step_kernel: KernelWrapper

    def _update_ab3_coefficients(self) -> None:
        """
        Updates nonlinear coefficients given changing timestep.
        """
        self.dt_array[self.system.current_step % 3] = self.system.current_dt

        # Alias for readability
        current_step = self.system.current_step
        dt0 = self.dt_array[current_step % 3]
        dt1 = self.dt_array[current_step % 3 - 1]
        dt2 = self.dt_array[current_step % 3 - 2]

        # Compute coefficients.
        # Disabling formatting and linting for readability.
        # fmt: off
        self.ab3_coefficients[0] = 1 + (dt0 / dt1) * ((2.0 / 6.0) * dt0 +               dt1 + (3.0 / 6.0) * dt2) / (dt1 + dt2) # noqa: E501
        self.ab3_coefficients[1] =   - (dt0 / dt1) * ((2.0 / 6.0) * dt0 + (3.0 / 6.0) * dt1 + (3.0 / 6.0) * dt2) / (      dt2) # noqa: E501
        self.ab3_coefficients[2] =   + (dt0 / dt2) * ((2.0 / 6.0) * dt0 + (3.0 / 6.0) * dt1                    ) / (dt1 + dt2) # noqa: E501
        # fmt: on

    def setup(self):
        if not self.input["setup.precompute_linear_matrix"]:
            self.precompute_iteration_matrices = lambda: None

        self._allocate_memory()

    def ready(self):
        system = self.system
        self.dt_array = np.array(
            [system.current_dt, 10**10, 10**10], dtype=system.float
        )
        self.ab3_coefficients = np.array([1, 0, 0], dtype=system.float)

        # Reset AB3 history
        if system.requires_explicit_terms:
            self.multistep_explicit_terms.fill(system.complex(0.0))
            cupy_set_device_pointer(
                system.cupy_module,
                "multistep_explicit_terms",
                self.multistep_explicit_terms,
            )

        if self.input["setup.precompute_linear_matrix"]:
            if self.ab3_type == "ab3":
                cupy_set_device_pointer(
                    system.cupy_module, "inverse_lhs_precomp", self.inverse_lhs
                )
                cupy_set_device_pointer(
                    system.cupy_module, "rhs_precomp", self.rhs
                )
            elif self.ab3_type == "ab3_if":
                cupy_set_device_pointer(
                    system.cupy_module, "propagator_precomp", self.propagator
                )

            self.precompute_iteration_matrices()

    def _allocate_memory(self):
        # For the explicit terms, we need to keep terms at the current
        # time step + terms from the past 2 time steps since we are
        # using AB3.
        # The explicit terms are indexed as (step, field, kz, kx, ky)
        system = self.system
        if self.system.requires_explicit_terms:
            self.multistep_explicit_terms = cp.zeros(
                (
                    3,
                    system.number_of_fields,
                    system.nz,
                    system.nx,
                    system.half_ny,
                ),
                dtype=system.complex,
            )

        # Allocate precomputation matrices
        if self.input["setup.precompute_linear_matrix"]:
            # Allocate according to method
            if self.ab3_type == "ab3":
                if not hasattr(self, "rhs"):
                    self.rhs = cp.zeros(
                        (
                            system.number_of_fields,
                            system.number_of_fields,
                            system.nz,
                            system.nx,
                            system.half_ny,
                        ),
                        dtype=system.complex,
                    )
                    self.inverse_lhs = cp.zeros(
                        (
                            system.number_of_fields,
                            system.number_of_fields,
                            system.nz,
                            system.nx,
                            system.half_ny,
                        ),
                        dtype=system.complex,
                    )

            elif self.ab3_type == "ab3_if":
                if not hasattr(self, "propagator"):
                    self.propagator = cp.zeros(
                        (
                            system.number_of_fields,
                            system.number_of_fields,
                            system.nz,
                            system.nx,
                            system.half_ny,
                        ),
                        dtype=system.complex,
                    )

    def precompute_iteration_matrices(self):
        """Precomputes the linear matrix."""
        self.precompute_iteration_matrices_kernel(
            self.system.float(self.system.current_dt)
        )

    def setup_cuda_definitions(self):
        self.system.module_options.define_flag(self.ab3_type.upper())

        self.system.module_options.define_flag(
            "LINEAR_PADE_DEGREE", str(self.input["setup.linear_pade_degree"])
        )

        if self.input["setup.precompute_linear_matrix"]:
            flucsprint("Linear matrices will be precomputed.", source=self)
            self.system.module_options.define_flag("PRECOMPUTE_LINEAR_MATRIX")

    def register_kernels(self) -> None:
        """Registers the CUDA kernels."""
        self.precompute_iteration_matrices_kernel = KernelWrapper(
            system=self.system,
            cuda_kernel_name="precompute_iteration_matrices",
            grid=(self.system.half_unpadded_cuda_grid_size,),
            block=(self.system.cuda_block_size,),
        )

        self.finish_step_kernel = KernelWrapper(
            system=self.system,
            cuda_kernel_name="finish_step",
            grid=(self.system.half_unpadded_cuda_grid_size,),
            block=(self.system.cuda_block_size,),
        )

    def perform_timestep(self):
        system = self.system

        if self.is_nonlinear:
            fields = system.fields[
                (system.current_step - 1) % system.fields_history_size
            ]
            system.compute_nonlinear_terms(fields)

            if system._update_dt():
                self.precompute_iteration_matrices()

            self._update_ab3_coefficients()

        self.finish_step_kernel(
            system.float(system.current_dt),
            system.float(system.current_time),
            system.int(system.current_step),
            system.float(system.adaptive_rate),
            self.ab3_coefficients[0],
            self.ab3_coefficients[1],
            self.ab3_coefficients[2],
            system.fields[system.current_step % system.fields_history_size - 1],
            system.dft_bits,
            system.fields[system.current_step % system.fields_history_size],
        )

    def __init__(self, solver: FlucsSolver[FourierSystem]):
        super().__init__(solver)
        self.ab3_type = self.input["setup.timestepper"]
        self.is_nonlinear = not self.system.input["setup.linear"]

    def __str__(self):
        if self.ab3_type == "ab3_if":
            return "Adams-Bashforth 3 with Pade integrating factors"

        return "Adams-Bashforth 3"


class FourierSolver(FlucsSolver[FourierSystem]):
    """
    A pseudospectral solver for a system of nonlinear fluid PDEs in 2D or 3D
    that are specified by a FourierSystem.

    """

    # Supported time steppers
    _supported_timesteppers: ClassVar = {
        "ab3": FourierAB3Timestepper,
        "ab3_if": FourierAB3Timestepper,
    }

    def setup_cuda_definitions(self) -> None:
        self.timestepper.setup_cuda_definitions()

    def register_kernels(self) -> None:
        self.timestepper.register_kernels()

    def run(self):
        """Run the main solver loop."""

        # We first time the solver
        self.state = FlucsSolverState.TIMING

        # Get the system ready
        self.system.setup()
        self.timestepper.setup()

        self.system.setup_output()
        self.system.compile_cupy_module()
        self.system.check_health()
        self.system.get_memory_usage()

        # Timing
        self.system.ready()
        self.timestepper.ready()

        time_taken = self._solver_loop()
        flucsprint(
            f"Timed {self.system.input['setup.timing_steps']:.3e} steps, "
            f"taking  {time_taken:.3e} seconds."
        )

        if self.system.input["setup.timing"]:
            flucsprint("Timing completed. Exiting.\n")
            return

        # Reset system and actually run it
        self.state = FlucsSolverState.RUNNING
        self.system.ready()
        self.timestepper.ready()

        time_taken = self._solver_loop()

        flucsprint(
            f"Finished at time {float(self.system.current_time):.3e}, "
            f"dt {float(self.system.current_dt):.3e}"
        )
        flucsprint(f"flucs given in {time_taken} seconds.\n")

    def _not_done(self) -> bool:
        if self.interrupted:
            return False

        if self.state == FlucsSolverState.TIMING:
            return (
                self.system.current_step
                < self.system.input["setup.timing_steps"]
            )

        return self.system.current_time < self.system.final_time

    def _solver_loop(self) -> float:
        if self.interrupted:
            return 0.0

        # Diagnostics for the first time step
        self.system.execute_diagnostics()

        start_time = time.time()
        self.system.steps_until_next_write = self.system.input[
            "output.write_steps"
        ]

        while self._not_done():
            # Advance step counter
            # After this, current_step indexes the step to be solved for
            # while current_time still points at the previously known
            # solution for the fields. We cannot advance current_time
            # because we do not yet know dt.
            self.system.current_step += 1

            # System-specific start-of-step hook
            self.system.begin_time_step()

            # Perform a step
            self.timestepper.perform_timestep()

            # Update current_time to reflect the time
            # of current_step
            self.system.current_time += self.system.current_dt

            # System-specific end-of-step hook
            self.system.finish_time_step()

            # Diagnostics, output, etc
            self.system.execute_diagnostics()
            self.system.write_output()
            self.system.restart_manager.write_restart()
        end_time = time.time()

        # One final write
        self.system.execute_diagnostics(force=True)
        self.system.write_output(force=True)
        self.system.restart_manager.write_restart(force=True)

        return end_time - start_time
