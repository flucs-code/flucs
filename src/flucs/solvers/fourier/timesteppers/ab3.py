import numpy as np

from flucs.solvers import FlucsSolver, FlucsTimestepper
from flucs.solvers.fourier.fourier_system import FourierSystem
from flucs.utilities.cupy import KernelWrapper, cupy_get_device_array
from flucs.utilities.messages import flucsprint


class FourierAB3Timestepper(FlucsTimestepper[FourierSystem]):
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
        if not self.input["timestepping.precompute_linear_matrix"]:
            self.precompute_iteration_matrices = lambda: None

    def ready(self):
        system = self.system
        self.dt_array = np.array(
            [10**10, 10**10, 10**10], dtype=system.float
        )
        self.ab3_coefficients = np.array([1, 0, 0], dtype=system.float)

        # Reset AB3 history
        if system.requires_explicit_terms:
            self.multistep_explicit_terms = cupy_get_device_array(
                module=system.cupy_module,
                array_name="multistep_explicit_terms_global",
                shape=(
                    3,
                    system.number_of_fields,
                    system.nz,
                    system.nx,
                    system.half_ny,
                ),
                dtype=system.complex,
            )
            self.multistep_explicit_terms.fill(system.complex(0.0))

        if self.input["timestepping.precompute_linear_matrix"]:
            self.precompute_iteration_matrices()

    def precompute_iteration_matrices(self):
        """Precomputes the linear matrix."""
        self.precompute_iteration_matrices_kernel(
            self.system.float(self.system.current_dt)
        )

    def setup_cuda_definitions(self):
        self.system.module_options.define_flag("AB3")

        self.system.module_options.define_flag(
            "LINEAR_PADE_DEGREE",
            str(self.input["timestepping.linear_pade_degree"]),
        )

        if self.input["timestepping.precompute_linear_matrix"]:
            flucsprint("Linear matrices will be precomputed.", source=self)
            self.system.module_options.define_flag("PRECOMPUTE_LINEAR_MATRIX")

    def register_kernels(self) -> None:
        """Registers the CUDA kernels."""
        self.precompute_iteration_matrices_kernel = KernelWrapper(
            system=self.system,
            cuda_kernel_name="precompute_iteration_matrices",
            grid=(self.system.half_cuda_grid_size,),
            block=(self.system.cuda_block_size,),
        )

        self.finish_step_kernel = KernelWrapper(
            system=self.system,
            cuda_kernel_name="finish_step",
            grid=(self.system.half_cuda_grid_size,),
            block=(self.system.cuda_block_size,),
        )

    def execute_timestep(self):
        system = self.system

        # Get fields
        current_fields = system.get_fields()
        previous_fields = system.get_fields(1)

        if self.is_nonlinear:
            system.cfl_rate[0] = 0
            system.compute_nonlinear_terms(
                system.float(system.current_dt),
                system.float(system.current_time),
                system.int(system.current_step),
                previous_fields,
                True,
            )

            if system._update_dt():
                self.precompute_iteration_matrices()

            self._update_ab3_coefficients()

        self.finish_step_kernel(
            system.float(system.current_dt),
            system.float(system.current_time),
            system.int(system.current_step),
            self.ab3_coefficients[0],
            self.ab3_coefficients[1],
            self.ab3_coefficients[2],
            previous_fields,
            system.dft_bits,
            current_fields,
        )

    def __init__(self, solver: FlucsSolver[FourierSystem]):
        super().__init__(solver)
        self.is_nonlinear = not self.system.input["setup.linear"]

    def __str__(self):
        return "Adams-Bashforth 3"
