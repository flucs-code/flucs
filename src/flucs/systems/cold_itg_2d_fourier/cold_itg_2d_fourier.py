"""Pseudospectral Fourier implementation of the Ivanov et al. (2020) 2D fluid
ITG system. The nonlinear term is handled explicitly using the Adams-Bashforth
3-step method.

"""

import cupy as cp
import numpy as np
from cupy.cuda import cufft

from .cold_itg_2d_fourier_diagnostics import HeatfluxDiag
from flucs.utilities.cupy import cupy_set_device_pointer
from flucs.solvers.fourier.fourier_system import FourierSystem
from flucs.solvers.fourier.fourier_system_diagnostics import LinearSpectrumDiag


class ColdITG2DFourier(FourierSystem):
    """Fourier solver for the 2D system."""
    number_of_fields = 2

    # Direct pointers to the phi and T arrays
    phi: list
    T: list

    # Nonlinear terms
    nonlinear_terms: list

    # Derivatives and 'bits' used for the nonlinear terms
    dft_derivatives: cp.ndarray
    real_derivatives: cp.ndarray

    dft_bits: cp.ndarray
    real_bits: cp.ndarray
    real_dxphi: cp.ndarray
    real_dxphi_zonal: cp.ndarray

    # DFT plans
    plan_r2c: cufft.PlanNd
    plan_c2r: cufft.PlanNd

    # CUDA grids
    zonal_average_cuda_block: tuple
    zonal_average_cuda_grid: tuple
    zonal_average_shared_mem: int
    nonlinear_bits_shared_mem: int

    # CUDA FFTs
    fft_c2r_plan_type: int
    fft_r2c_plan_type: int

    find_derivatives_kernel: cp.RawKernel
    find_nonlinear_bits_kernel: cp.RawKernel
    zonal_average_kernel: cp.RawKernel

    # Supported diagnostics
    diags_dict = {"heatflux": HeatfluxDiag,
                  "linear_spectrum": LinearSpectrumDiag}

    def setup(self):
        """Prepares the system for the solver."""

        self.allocate_memory()
        # self.setup_kernels()
        super().setup()

    def ready(self):
        # Anything system-specific goes here

        if not self.input["setup.linear"]:
            cupy_set_device_pointer(self.cupy_module,
                                    "multistep_nonlinear_terms",
                                    self.multistep_nonlinear_terms)
        # Setup kernel parameters (grid, block, shared memory)
        self.zonal_average_cuda_block = (32, 16)
        self.zonal_average_cuda_grid = (
            (self.padded_nx + self.zonal_average_cuda_block[0] - 1)
            // self.zonal_average_cuda_block[0],
            (self.padded_ny + self.zonal_average_cuda_block[1] - 1)
            // self.zonal_average_cuda_block[1]
        )
        self.zonal_average_shared_mem = (
            self.zonal_average_cuda_block[0]
            * self.zonal_average_cuda_block[1]
            * self.float().nbytes
        )

        self.nonlinear_bits_shared_mem = (
            self.cuda_block_size * self.float().nbytes
        )

        super().ready()

    def allocate_memory(self):
        # GPU arrays

        # For the field arrays, we need to keep the fields
        # at the current time step and the previous one.

        self.fields = [cp.zeros((2, self.nz, self.nx, self.half_ny),
                                dtype=self.complex),
                       cp.zeros((2, self.nz, self.nx, self.half_ny),
                                dtype=self.complex)]

        self.phi = [cp.ndarray((self.nz, self.nx, self.half_ny),
                               dtype=self.complex,
                               memptr=self.fields[0][0, 0, 0, 0].data),
                    cp.ndarray((self.nz, self.nx, self.half_ny),
                               dtype=self.complex,
                               memptr=self.fields[1][0, 0, 0, 0].data),]

        self.T = [cp.ndarray((self.nz, self.nx, self.half_ny),
                             dtype=self.complex,
                             memptr=self.fields[0][1, 0, 0, 0].data),
                  cp.ndarray((self.nz, self.nx, self.half_ny),
                             dtype=self.complex,
                             memptr=self.fields[1][1, 0, 0, 0].data),]

        # when running linearly, need something to pass to the kernels
        # this is unused
        self.dft_bits = cp.zeros(1, dtype=self.complex)

        if not self.input["setup.linear"]:
            # For the nonlinear terms, we need to keep terms at the current
            # time step + terms from the past 2 time steps (since we will be
            # using AB3)
            # The nonlinear terms are indexed as (step, field, kz, kx, ky)
            self.multistep_nonlinear_terms = cp.zeros((3, 2, self.nz, self.nx,
                                                       self.half_ny),
                                                      dtype=self.complex)

            # All fields and derivatives to be transformed to real space
            # are kept in one huge array (dft_derivatives).
            # The first index indexes the fields and it's meaning is
            # 0 dxphi,
            # 1 dyphi,
            # 2 (dx^2 - dy^2) phi,
            # 3 dxdyphi
            # 4 p
            self.dft_derivatives = cp.zeros([5, self.padded_nx,
                                             self.half_padded_ny],
                                            dtype=self.complex)
            self.real_derivatives = cp.zeros([5, self.padded_nx,
                                              self.padded_ny],
                                             dtype=self.float)

            # These 'NL bits' are the terms which are calculated in real space.
            # They are transformed back to Fourier space, where any additional
            # derivatives are taken by multiplying the NL bits by the
            # appropriate powers of k. The NL bits here are
            # 0 dxphi0 * dyphi
            # 1 (dx^2 - dy^2)phi * p
            # 2 dxdyphi * p
            # 3 dxphi * p
            # 4 dyphi * p
            # E.g., we calculate {phi, T} by calculating
            # {phi, p} = dy (dxphi * u) - dx (dyphi * u)
            self.dft_bits = cp.zeros([5, self.padded_nx, self.half_padded_ny],
                                     dtype=self.complex)
            self.real_bits = cp.zeros([5, self.padded_nx, self.padded_ny],
                                      dtype=self.float)

            self.cfl_rate = cp.zeros([1], dtype=self.float)

            # The first derivative in real_derivatives is dx phi.
            # We need this for computing the zonal flow.
            self.real_dxphi = cp.ndarray((self.padded_nx, self.padded_ny),
                                         dtype=self.float,
                                         memptr=self.real_derivatives.data)

            self.real_dxphi_zonal = cp.zeros((self.padded_nx,),
                                             dtype=self.float)

            if self.input["setup.precision"] == "single":
                self.fft_c2r_plan_type = cufft.CUFFT_C2R
                self.fft_r2c_plan_type = cufft.CUFFT_R2C
            else:
                self.fft_c2r_plan_type = cufft.CUFFT_Z2D
                self.fft_r2c_plan_type = cufft.CUFFT_D2Z

            self.plan_c2r = cufft.PlanNd(
                shape=tuple([self.padded_nx, self.padded_ny]),
                istride=1,
                ostride=1,
                inembed=tuple([1, self.half_padded_ny]),
                onembed=tuple([1, self.padded_ny]),
                idist=self.padded_nx*self.half_padded_ny,
                odist=self.padded_nx*self.padded_ny,
                fft_type=self.fft_c2r_plan_type,
                batch=5,
                order='C',
                last_axis=2,
                last_size=self.padded_ny)

            self.plan_r2c = cufft.PlanNd(
                shape=tuple([self.padded_nx, self.padded_ny]),
                istride=1,
                ostride=1,
                inembed=tuple([1, self.padded_ny]),
                onembed=tuple([1, self.half_padded_ny]),
                idist=self.padded_nx*self.padded_ny,
                odist=self.padded_nx*self.half_padded_ny,
                fft_type=self.fft_r2c_plan_type,
                batch=5,
                order='C',
                last_axis=2,
                last_size=self.half_padded_ny)

    def _interpret_input(self):
        """Checks if the input file makes sense"""

        # Make sure to call the parent method to do some standard setup
        # (resolution checks, etc)
        super()._interpret_input()

        # Anything custom goes here

        if self.nz != 1 or self.padded_nz != 1:
            raise ValueError("Both nz and padded_nz should be "
                             "set to 1 for the 2D system!")

    def compile_cupy_module(self) -> None:
        # System-specific constants for the kernels
        self.module_options.define_constant("CHI",
                                            self.input["parameters.chi"])
        self.module_options.define_constant("A_TIMES_CHI",
                                            self.input["parameters.a"]
                                            * self.input["parameters.chi"])

        self.module_options.define_constant("B_TIMES_CHI",
                                            self.input["parameters.b"]
                                            * self.input["parameters.chi"])

        self.module_options.define_constant("KAPPA_T",
                                            self.input["parameters.kappaT"])
        self.module_options.define_constant("KAPPA_N",
                                            self.input["parameters.kappan"])
        self.module_options.define_constant("KAPPA_B",
                                            self.input["parameters.kappaB"])

        # Call this to compile the module
        super().compile_cupy_module()

        # System-specific kernels
        self.find_derivatives_kernel =\
            self.cupy_module.get_function("find_derivatives")

        self.find_nonlinear_bits_kernel =\
            self.cupy_module.get_function("find_nonlinear_bits")

        self.zonal_average_kernel =\
            self.cupy_module.get_function("zonal_average")

    def begin_time_step(self) -> None:
        # Do anything model-specific here, then call the parent's method
        super().begin_time_step()

    def calculate_nonlinear_terms(self) -> None:
        """
        Calculates the nonlinear terms. This is the most computationaly
        intensive part of taking a time step. Here, we also determine the
        nonlinear CFL coefficient.

        """
        self.find_derivatives_kernel((self.half_padded_cuda_grid_size,),
                                     (self.cuda_block_size,),
                                     (self.fields[self.current_step % 2 - 1],
                                      self.dft_derivatives,
                                      self.real_dxphi_zonal,
                                      self.cfl_rate))

        self.plan_c2r.fft(self.dft_derivatives,
                          self.real_derivatives,
                          cufft.CUFFT_INVERSE)

        # self.real_derivatives = cp.fft.irfftn(self.dft_derivatives, s=(self.padded_nx, self.padded_ny), norm="forward")
        # cp.mean(self.real_dxphi, axis=[-1], out=self.real_dxphi_zonal)

        self.zonal_average_kernel(self.zonal_average_cuda_grid,
                                  self.zonal_average_cuda_block,
                                  (self.real_dxphi, self.real_dxphi_zonal),
                                  shared_mem=self.zonal_average_shared_mem)

        self.find_nonlinear_bits_kernel(
            (self.full_padded_cuda_grid_size,),
            (self.cuda_block_size,),
            (self.real_derivatives, self.real_dxphi_zonal,
             self.real_bits, self.cfl_rate),
            shared_mem=self.nonlinear_bits_shared_mem
        )

        self.plan_r2c.fft(self.real_bits, self.dft_bits, cufft.CUFFT_FORWARD)
        # self.dft_bits = cp.fft.rfftn(self.real_bits,
        #                              s=(self.padded_nx, self.padded_ny),
        #                              norm="forward")

        super().calculate_nonlinear_terms()

    def finish_time_step(self) -> None:
        super().finish_time_step()

    def compute_complex_omega(self):
        linear_matrix = np.zeros(self.half_unpadded_tuple + (2, 2),
                                 dtype=self.complex)

        kxs, kys, kzs = self.get_broadcast_wavenumbers()
        kperp2 = kxs**2 + kys**2

        kappaT = self.input["parameters.kappaT"]
        kappaB = self.input["parameters.kappaB"]
        kappan = self.input["parameters.kappan"]
        chi = self.input["parameters.chi"]
        a = self.input["parameters.a"]
        b = self.input["parameters.b"]

        eta = 1 + kperp2
        # zonal response
        eta[0, :, 0] = kperp2[0, :, 0]

        # phi-phi
        linear_matrix[:, :, :, 0, 0] = (
                    a*chi*(kperp2**2)
                    - 1j*(kappaB - kappan)*kys
                    - 1j*kappaT*kperp2*kys) / eta

        # phi-T
        linear_matrix[:, :, :, 0, 1] = (
                    - b*chi*(kperp2**2)
                    - 1j*kappaB*kys) / eta

        # T-phi
        linear_matrix[:, :, :, 1, 0] = 1j*kappaT*kys

        # T-T
        linear_matrix[:, :, :, 1, 1] = chi*kperp2

        # Fix (0,0,0) mode
        linear_matrix[0, 0, 0, :, :] = np.identity(2)

        return -1j*np.linalg.eigvals(linear_matrix)
