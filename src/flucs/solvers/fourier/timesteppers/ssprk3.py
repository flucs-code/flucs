import cupy as cp

from flucs.solvers import FlucsSolver, FlucsTimestepper
from flucs.solvers.fourier.fourier_system import FourierSystem
from flucs.utilities.cupy import KernelWrapper
from flucs.utilities.messages import flucsprint


class FourierSSPRK3Timestepper(FlucsTimestepper[FourierSystem]):
    is_nonlinear: bool

    precompute_iteration_matrices_kernel: KernelWrapper

    def setup(self):
        if not self.input["timestepping.precompute_linear_matrix"]:
            self.precompute_iteration_matrices = lambda: None

        self._allocate_memory()

    def ready(self):
        if self.input["timestepping.precompute_linear_matrix"]:
            self.precompute_iteration_matrices()

    def _allocate_memory(self):
        system = self.system

        self.stage_fields = [
            cp.zeros(
                (
                    system.number_of_fields,
                    system.nz,
                    system.nx,
                    system.half_ny,
                ),
                dtype=system.complex,
            ),
            cp.zeros(
                (
                    system.number_of_fields,
                    system.nz,
                    system.nx,
                    system.half_ny,
                ),
                dtype=system.complex,
            ),
        ]

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
            grid=(self.system.half_cuda_grid_size,),
            block=(self.system.cuda_block_size,),
        )

        self.finish_stage1_kernel = KernelWrapper(
            system=self.system,
            cuda_kernel_name="finish_stage<1>",
            grid=(self.system.half_cuda_grid_size,),
            block=(self.system.cuda_block_size,),
        )
        self.finish_stage2_kernel = KernelWrapper(
            system=self.system,
            cuda_kernel_name="finish_stage<2>",
            grid=(self.system.half_cuda_grid_size,),
            block=(self.system.cuda_block_size,),
        )
        self.finish_stage3_kernel = KernelWrapper(
            system=self.system,
            cuda_kernel_name="finish_stage<3>",
            grid=(self.system.half_cuda_grid_size,),
            block=(self.system.cuda_block_size,),
        )

    def execute_timestep(self):
        system = self.system

        # Get fields
        current_fields = system.get_fields()
        previous_fields = system.get_fields(1)

        # Stage 1
        if self.is_nonlinear:
            system.cfl_rate[0] = 0
            system.compute_nonlinear_terms(
                system.float(system.current_dt),
                system.float(system.current_time),
                system.int(system.current_step),
                previous_fields,
                True,
            )

            # Update dt as necessary
            if system._update_dt():
                self.precompute_iteration_matrices()

        self.finish_stage1_kernel(
            system.float(system.current_dt),
            system.float(system.current_time),
            system.int(system.current_step),
            previous_fields,
            system.dft_bits,
            previous_fields,
            self.stage_fields[1],
            current_fields,
        )

        # Stage 2
        if self.is_nonlinear:
            system.compute_nonlinear_terms(
                system.float(system.current_dt),
                system.float(system.current_time + system.current_dt),
                system.int(system.current_step),
                self.stage_fields[1],
                False,
            )

        self.finish_stage2_kernel(
            system.float(system.current_dt),
            system.float(system.current_time),
            system.int(system.current_step),
            previous_fields,
            system.dft_bits,
            self.stage_fields[1],
            self.stage_fields[0],
            current_fields,
        )

        # Stage 3
        if self.is_nonlinear:
            system.compute_nonlinear_terms(
                system.float(system.current_dt),
                system.float(system.current_time + 0.5 * system.current_dt),
                system.int(system.current_step),
                self.stage_fields[0],
                False,
            )

        self.finish_stage3_kernel(
            system.float(system.current_dt),
            system.float(system.current_time),
            system.int(system.current_step),
            previous_fields,
            system.dft_bits,
            self.stage_fields[0],
            self.stage_fields[1],
            current_fields,
        )

    def __init__(self, solver: FlucsSolver[FourierSystem]):
        super().__init__(solver)

        self.is_nonlinear = not self.system.input["setup.linear"]

    def __str__(self):
        return "Shu-Osher Runge-Kutta 3"
