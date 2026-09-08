"""Diagnostics for the Fourier test system."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, ClassVar

from flucs import cupy as cp
from flucs.diagnostic import FlucsDiagnostic, FlucsDiagnosticVariable
from flucs.solvers.fourier.fourier_system_reductions import FourierReductions

if TYPE_CHECKING:
    from .fourier import TestFourierSystem


class FreeEnergyDiag(FlucsDiagnostic):
    """
    Computes the free energy and each contribution to its conservation budget.
    """

    name = "free_energy"
    system: TestFourierSystem

    get_W: Callable[..., cp.ndarray]
    get_dWdt_nonlinear: Callable[..., cp.ndarray]
    get_dWdt_forcing: Callable[..., cp.ndarray]
    get_dWdt_hyperdissipation_component: Callable[..., cp.ndarray]

    def init_vars(self) -> None:
        reductions = FourierReductions(self.system)

        # Add variables for the free energy and its time derivative
        for name in [
            "W",
            "dWdt",
            "dWdt_nonlinear",
            "dWdt_forcing",
            "dWdt_error",
        ]:
            self.add_var(
                FlucsDiagnosticVariable(
                    name=name,
                    shape=(),
                    dimensions={},
                    is_complex=False,
                )
            )

        # Add variables for each hyperdissipation component
        for component in self.system.hyperdissipation_components:
            self.add_var(
                FlucsDiagnosticVariable(
                    name=f"dWdt_hyperdissipation_{component}",
                    shape=(),
                    dimensions={},
                    is_complex=False,
                )
            )

        # Register reductions
        self.get_W = reductions.get_reduction(
            reduction_output="scalar",
            functor="FreeEnergy_Functor",
            input_args="const FLUCS_COMPLEX*",
            complex_output=False,
        )
        self.get_dWdt_nonlinear = reductions.get_reduction(
            reduction_output="scalar",
            functor="FreeEnergyNonlinear_Functor",
            input_args=(
                "const FLUCS_COMPLEX (*)[HALFSIZE],FLUCS_FLOAT,"
                "FLUCS_FLOAT,long long,"
                "const FLUCS_COMPLEX (*)[HALFSIZE]"
            ),
            complex_output=False,
        )
        self.get_dWdt_forcing = reductions.get_reduction(
            reduction_output="scalar",
            functor="FreeEnergyForcing_Functor",
            input_args=(
                "const FLUCS_COMPLEX (*)[HALFSIZE],FLUCS_FLOAT,"
                "FLUCS_FLOAT,long long"
            ),
            complex_output=False,
        )
        self.get_dWdt_hyperdissipation_component = reductions.get_reduction(
            reduction_output="scalar",
            functor="FreeEnergyHyperdissipationComponent_Functor",
            input_args="const FLUCS_COMPLEX*,FLUCS_FLOAT,int",
            complex_output=False,
        )

    def ready(self) -> None:
        pass

    def execute(self) -> None:
        # Useful aliases
        current_dt = self.system.float(self.system.current_dt)
        current_time = self.system.float(self.system.current_time)
        current_step = self.system.int(self.system.current_step)
        adaptive_rate = self.system.float(self.system.adaptive_rate)

        fields = self.system.get_fields()
        fields_prev = self.system.get_fields(1)

        # W and dW/dt
        W = self.get_W(fields).get().item()
        W_prev = self.get_W(fields_prev).get().item()
        dWdt = (W - W_prev) / current_dt

        self.save_data("W", W)
        self.save_data("dWdt", dWdt)

        # Nonlinear contribution
        if not self.system.input["setup.linear"]:
            self.system.compute_nonlinear_terms(
                current_dt,
                current_time,
                current_step,
                fields,
                False,
            )

        dWdt_nonlinear = (
            self.get_dWdt_nonlinear(
                fields,
                current_dt,
                current_time,
                current_step,
                self.system.dft_bits,
            )
            .get()
            .item()
        )
        self.save_data("dWdt_nonlinear", dWdt_nonlinear)

        # Forcing contribution
        dWdt_forcing = (
            self.get_dWdt_forcing(
                fields,
                current_dt,
                current_time,
                current_step,
            )
            .get()
            .item()
        )
        self.save_data("dWdt_forcing", dWdt_forcing)

        # Hyperdissipation contributions
        dWdt_hyperdissipation_total = 0.0
        for index, component in enumerate(
            self.system.hyperdissipation_components
        ):
            result = self.get_dWdt_hyperdissipation_component(
                fields,
                adaptive_rate,
                index,
            )
            dWdt_hyperdissipation_component = -result.get().item()
            self.save_data(
                f"dWdt_hyperdissipation_{component}",
                dWdt_hyperdissipation_component,
            )
            dWdt_hyperdissipation_total += dWdt_hyperdissipation_component

        # Error in the free-energy balance
        self.save_data(
            "dWdt_error",
            dWdt - dWdt_nonlinear - dWdt_forcing - dWdt_hyperdissipation_total,
        )


class FreeEnergyDiag1D(FlucsDiagnostic):
    """
    Computes 1D spectra of free-energy quantities and budget terms.
    """

    name = "free_energy_1d"
    system: TestFourierSystem
    option_defaults: ClassVar[dict[str, object]] = {
        "spectra": ["kmod"],
        "save_contributions": False,
    }

    get_W: dict[str, Callable[..., cp.ndarray]]
    get_dWdt_forcing: dict[str, Callable[..., cp.ndarray]]
    get_dWdt_hyperdissipation: dict[str, Callable[..., cp.ndarray]]

    get_W_contribution: dict[str, Callable[..., cp.ndarray]]

    def init_vars(self) -> None:
        reductions = FourierReductions(self.system)

        # Parse valid spectra (enforce 1D)
        valid_spectra = ("kz", "kx", "ky", "kperp", "kmod")
        spectra = self.spectra
        if isinstance(spectra, str):
            spectra = [spectra]
        spectra = tuple(dict.fromkeys(spectra))

        invalid_spectra = set(spectra) - set(valid_spectra)
        if invalid_spectra:
            raise ValueError(
                f"{self.name} only supports 1D spectra {valid_spectra}."
            )

        # Initialise dicts
        self.get_W = {}
        self.get_dWdt_forcing = {}
        self.get_dWdt_hyperdissipation = {}

        self.get_W_contribution = {}

        # Iterate over spectra types and initialise variables
        for spectrum in spectra:
            dimensions = reductions.get_dimensions(spectrum)
            shape = tuple(dimensions)

            for name in ["W", "dWdt_forcing", "dWdt_hyperdissipation"]:
                self.add_var(
                    FlucsDiagnosticVariable(
                        name=f"{spectrum}_spectra/{name}",
                        shape=shape,
                        dimensions=dimensions,
                        is_complex=False,
                    )
                )

            self.get_W[spectrum] = reductions.get_reduction(
                reduction_output=spectrum,
                functor="FreeEnergy_Functor",
                input_args="const FLUCS_COMPLEX*",
                complex_output=False,
            )
            self.get_dWdt_forcing[spectrum] = reductions.get_reduction(
                reduction_output=spectrum,
                functor="FreeEnergyForcing_Functor",
                input_args=(
                    "const FLUCS_COMPLEX (*)[HALFSIZE],FLUCS_FLOAT,"
                    "FLUCS_FLOAT,long long"
                ),
                complex_output=False,
            )
            self.get_dWdt_hyperdissipation[spectrum] = (
                reductions.get_reduction(
                    reduction_output=spectrum,
                    functor="FreeEnergyHyperdissipation_Functor",
                    input_args="const FLUCS_COMPLEX*,FLUCS_FLOAT",
                    complex_output=False,
                )
            )

            # Save contributions if required
            if self.save_contributions:
                for name in ["W_uz", "W_ux", "W_uy"]:
                    self.add_var(
                        FlucsDiagnosticVariable(
                            name=f"{spectrum}_spectra/{name}",
                            shape=shape,
                            dimensions=dimensions,
                            is_complex=False,
                        )
                    )

                self.get_W_contribution[spectrum] = reductions.get_reduction(
                    reduction_output=spectrum,
                    functor="Abs2_Functor",
                    input_args="const FLUCS_COMPLEX*,FLUCS_FLOAT",
                    complex_output=False,
                )

    def ready(self) -> None:
        pass

    def execute(self) -> None:
        # Useful aliases
        current_dt = self.system.float(self.system.current_dt)
        current_time = self.system.float(self.system.current_time)
        current_step = self.system.int(self.system.current_step)
        adaptive_rate = self.system.float(self.system.adaptive_rate)
        half = self.system.float(0.5)
        fields = self.system.get_fields()

        # Iterate over spectra to save
        for spectrum in self.get_W:

            # Free energy
            W = self.get_W[spectrum](fields).get()

            self.save_data(
                f"{spectrum}_spectra/W",
                W,
            )

            # Forcing contribution
            dWdt_forcing = self.get_dWdt_forcing[spectrum](
                fields,
                current_dt,
                current_time,
                current_step,
            ).get()

            self.save_data(
                f"{spectrum}_spectra/dWdt_forcing",
                dWdt_forcing,
            )

            # Hyperdissipation contribution
            dWdt_hyperdissipation = -self.get_dWdt_hyperdissipation[spectrum](
                fields,
                adaptive_rate,
            ).get()

            self.save_data(
                f"{spectrum}_spectra/dWdt_hyperdissipation",
                dWdt_hyperdissipation,
            )

            # Free-energy contributions
            if self.save_contributions:
                self.save_data(
                    f"{spectrum}_spectra/W_uz",
                    self.get_W_contribution[spectrum](fields[0], half).get(),
                )
                self.save_data(
                    f"{spectrum}_spectra/W_ux",
                    self.get_W_contribution[spectrum](fields[1], half).get(),
                )
                self.save_data(
                    f"{spectrum}_spectra/W_uy",
                    self.get_W_contribution[spectrum](fields[2], half).get(),
                )
