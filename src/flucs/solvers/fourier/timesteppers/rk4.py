import cupy as cp
import numpy as np

from flucs.solvers import FlucsSolver, FlucsTimestepper
from flucs.solvers.fourier.fourier_system import FourierSystem
from flucs.utilities.cupy import KernelWrapper, cupy_set_device_pointer
from flucs.utilities.messages import flucsprint


class FourierRK4Timestepper(FlucsTimestepper[FourierSystem]):
    is_nonlinear: bool

    # CUDA kernels
    precompute_iteration_matrices_kernel: KernelWrapper
    finish_step_kernel: KernelWrapper

    def setup(self):
        if not self.input["timestepping.precompute_linear_matrix"]:
            self.precompute_iteration_matrices = lambda: None

        self._allocate_memory()

    def ready(self):
        system = self.system

        if self.input["timestepping.precompute_linear_matrix"]:
            cupy_set_device_pointer(
                system.cupy_module,
                "propagator_half_precomp",
                self.propagator_half,
            )
            cupy_set_device_pointer(
                system.cupy_module,
                "propagator_full_precomp",
                self.propagator_full,
            )

            self.precompute_iteration_matrices()

    def _allocate_memory(self):
        # RK4 needs one temporary field array for intermediate stages.
        system = self.system
        self.stage_fields = cp.zeros(
            (
                system.number_of_fields,
                system.nz,
                system.nx,
                system.half_ny,
            ),
            dtype=system.complex,
        )

        # Allocate precomputation matrices
        if self.input["timestepping.precompute_linear_matrix"]:
            matrix_shape = (
                system.number_of_fields,
                system.number_of_fields,
                system.nz,
                system.nx,
                system.half_ny,
            )
            if not hasattr(self, "propagator_half"):
                self.propagator_half = cp.zeros(
                    matrix_shape,
                    dtype=system.complex,
                )
            if not hasattr(self, "propagator_full"):
                self.propagator_full = cp.zeros(
                    matrix_shape,
                    dtype=system.complex,
                )

    def precompute_iteration_matrices(self):
        """Precomputes the linear matrix."""
        self.precompute_iteration_matrices_kernel(
            self.system.float(self.system.current_dt)
        )

    def setup_cuda_definitions(self):
        self.system.module_options.define_flag("RK4")

        self.system.module_options.define_flag(
            "LINEAR_PADE_DEGREE", str(
                self.input["timestepping.linear_pade_degree"]
            )
        )

        if self.input["timestepping.precompute_linear_matrix"]:
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

        self.finish_stage1_kernel = KernelWrapper(
            system=self.system,
            cuda_kernel_name="finish_stage<1>",
            grid=(self.system.half_unpadded_cuda_grid_size,),
            block=(self.system.cuda_block_size,),
        )
        self.finish_stage2_kernel = KernelWrapper(
            system=self.system,
            cuda_kernel_name="finish_stage<2>",
            grid=(self.system.half_unpadded_cuda_grid_size,),
            block=(self.system.cuda_block_size,),
        )
        self.finish_stage3_kernel = KernelWrapper(
            system=self.system,
            cuda_kernel_name="finish_stage<3>",
            grid=(self.system.half_unpadded_cuda_grid_size,),
            block=(self.system.cuda_block_size,),
        )
        self.finish_stage4_kernel = KernelWrapper(
            system=self.system,
            cuda_kernel_name="finish_stage<4>",
            grid=(self.system.half_unpadded_cuda_grid_size,),
            block=(self.system.cuda_block_size,),
        )

    def execute_timestep(self):
        system = self.system

        # Get fields
        current_fields = system.get_fields()
        previous_fields = system.get_fields(1)

        # Stage 1

        if self.is_nonlinear:
            system.compute_nonlinear_terms(previous_fields)

            # Update dt as necessary
            if system._update_dt():
                self.precompute_iteration_matrices()

        self.finish_stage1_kernel(
            system.float(system.current_dt),
            system.float(system.current_time),
            system.int(system.current_step),
            system.float(system.adaptive_rate),
            previous_fields,
            system.dft_bits,
            self.stage_fields,
            current_fields,
        )

        # Stage 2

        if self.is_nonlinear:
            system.compute_nonlinear_terms(self.stage_fields)

        self.finish_stage2_kernel(
            system.float(system.current_dt),
            system.float(system.current_time),
            system.int(system.current_step),
            system.float(system.adaptive_rate),
            previous_fields,
            system.dft_bits,
            self.stage_fields,
            current_fields,
        )

        # Stage 3

        if self.is_nonlinear:
            system.compute_nonlinear_terms(self.stage_fields)

        self.finish_stage3_kernel(
            system.float(system.current_dt),
            system.float(system.current_time),
            system.int(system.current_step),
            system.float(system.adaptive_rate),
            previous_fields,
            system.dft_bits,
            self.stage_fields,
            current_fields,
        )

        # Stage 4

        if self.is_nonlinear:
            system.compute_nonlinear_terms(self.stage_fields)

        self.finish_stage4_kernel(
            system.float(system.current_dt),
            system.float(system.current_time),
            system.int(system.current_step),
            system.float(system.adaptive_rate),
            previous_fields,
            system.dft_bits,
            self.stage_fields,
            current_fields,
        )

    def __init__(self, solver: FlucsSolver[FourierSystem]):
        super().__init__(solver)

        self.is_nonlinear = not self.system.input["setup.linear"]

    def __str__(self):
        return "classic Runge-Kutta 4 with Lawson integrating factors"
