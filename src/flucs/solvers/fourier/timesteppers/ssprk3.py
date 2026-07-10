import cupy as cp

from flucs.solvers import FlucsSolver, FlucsTimestepper
from flucs.solvers.fourier.fourier_system import FourierSystem
from flucs.utilities.cupy import KernelWrapper, cupy_set_device_pointer
from flucs.utilities.messages import flucsprint


class FourierSSPRK3Timestepper(FlucsTimestepper[FourierSystem]):
    is_nonlinear: bool

    precompute_iteration_matrices_kernel: KernelWrapper

    def setup(self):
        if not self.input["timestepping.precompute_linear_matrix"]:
            self.precompute_iteration_matrices = lambda: None

        self._allocate_memory()

    def ready(self):
        system = self.system

        if self.input["timestepping.precompute_linear_matrix"]:
            cupy_set_device_pointer(
                system.cupy_module,
                "propagator_full_precomp",
                self.propagator_full,
            )
            cupy_set_device_pointer(
                system.cupy_module,
                "propagator_half_precomp",
                self.propagator_half,
            )
            cupy_set_device_pointer(
                system.cupy_module,
                "propagator_minus_half_precomp",
                self.propagator_minus_half,
            )

            self.precompute_iteration_matrices()

    def _allocate_memory(self):
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

        if self.input["timestepping.precompute_linear_matrix"]:
            matrix_shape = (
                system.number_of_fields,
                system.number_of_fields,
                system.nz,
                system.nx,
                system.half_ny,
            )
            self.propagator_full = cp.zeros(
                matrix_shape, dtype=system.complex
            )
            self.propagator_half = cp.zeros(
                matrix_shape, dtype=system.complex
            )
            self.propagator_minus_half = cp.zeros(
                matrix_shape, dtype=system.complex,
            )

    def precompute_iteration_matrices(self):
        self.precompute_iteration_matrices_kernel(
            self.system.float(self.system.current_dt)
        )

    def setup_cuda_definitions(self):
        self.system.module_options.define_flag("SSPRK3")

        self.system.module_options.define_flag(
            "LINEAR_PADE_DEGREE",
            str(self.input["timestepping.linear_pade_degree"]),
        )

        if self.input["timestepping.precompute_linear_matrix"]:
            flucsprint("Linear matrices will be precomputed.", source=self)
            self.system.module_options.define_flag("PRECOMPUTE_LINEAR_MATRIX")

    def register_kernels(self) -> None:
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

    def execute_timestep(self):
        system = self.system

        previous_fields = system.fields[
            (system.current_step - 1) % system.fields_history_size
        ]
        current_fields = system.fields[
            system.current_step % system.fields_history_size
        ]

        # Stage 1
        if self.is_nonlinear:
            system.compute_nonlinear_terms(previous_fields)

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

    def __init__(self, solver: FlucsSolver[FourierSystem]):
        super().__init__(solver)

        self.is_nonlinear = not self.system.input["setup.linear"]

    def __str__(self):
        return "Shu–Osher Runge-Kutta 3 with Lawson integrating factors"