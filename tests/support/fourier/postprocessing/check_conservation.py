import argparse
import pathlib as pl

import matplotlib.pyplot as plt
import numpy as np

from flucs.postprocessing import FlucsPostProcessing


def check_conservation(post, args):

    # Get valid files for the specified variable
    nc_paths = post.get_valid_netcdf_paths("free_energy/dWdt")

    # Iterate over output files
    for index, nc_path in enumerate(nc_paths):

        # Separate figure for each output
        fig, axs = plt.subplots(3, 1, layout="constrained", sharex=True)
        ax_energy, ax_balance, ax_error = axs

        # Set figure title
        figure_name = f"check_conservation_energy_{pl.Path(nc_path).parent.name}"
        fig.canvas.manager.set_window_title(figure_name)

        # Read data from netCDF file
        variables = post.get_netcdf_variables(nc_path)

        # Mask out initial data at restart boundaries given the calculation
        # of dW/dt is not valid at the first time step for a given run
        time, boundaries, _ = post.load_netcdf_variable(
            nc_path, "time", groups=args.groups
        )
        time[0] = np.nan
        for boundary in boundaries:
            time[boundary] = np.nan

        # Load data
        dt = post.load_netcdf_variable(
            nc_path, "dt", groups=args.groups
        )[0]
        free_energy = post.load_netcdf_variable(
            nc_path, "free_energy/W", groups=args.groups
        )[0]
        dWdt = post.load_netcdf_variable(
            nc_path, "free_energy/dWdt", groups=args.groups
        )[0]
        dWdt_nonlinear = post.load_netcdf_variable(
            nc_path, "free_energy/dWdt_nonlinear", groups=args.groups
        )[0]
        dWdt_forcing = post.load_netcdf_variable(
            nc_path, "free_energy/dWdt_forcing", groups=args.groups
        )[0]
        dWdt_error = post.load_netcdf_variable(
            nc_path, "free_energy/dWdt_error", groups=args.groups
        )[0]

        # Injection and dissipation
        injection = dWdt_forcing
        dissipation = np.zeros_like(dWdt)

        # Add hyperdissipation
        for variable in variables:
            if variable.startswith("free_energy/dWdt_hyperdissipation_"):
                dissipation += post.load_netcdf_variable(
                    nc_path, variable, groups=args.groups
                )[0]

        # Add vertical lines to mark restart boundaries
        for ax in axs:
            for index in boundaries:
                ax.axvline(time[index], color="black", linestyle="dotted")

        # Plot free energy
        ax_energy.plot(
            time,
            free_energy,
            label="W (free energy)",
            linewidth=1.5,
            color="black",
        )

        # Plot free-energy balance
        ax_balance.plot(
            time,
            dWdt,
            label="dW/dt",
            linewidth=1.5,
            color="black",
            linestyle="solid",
        )
        ax_balance.plot(
            time,
            dWdt_nonlinear,
            label="Nonlinear",
            linewidth=1.5,
            color="green",
            linestyle="solid",
        )
        ax_balance.plot(
            time,
            injection,
            label="Injection",
            linewidth=1.5,
            color="red",
            linestyle="solid",
        )
        ax_balance.plot(
            time,
            dissipation,
            label="Dissipation",
            linewidth=1.5,
            color="blue",
            linestyle="solid",
        )
        ax_balance.plot(
            time,
            dWdt_nonlinear + injection + dissipation,
            label="Nonlinear + injection + dissipation",
            linewidth=1.5,
            color="black",
            linestyle="dashed",
        )

        # Compute and plot measures of the error in the free-energy balance
        integrand = (
            0.5
            * (dWdt_error[1:] + dWdt_error[:-1])
            * np.diff(time)
        )  # Manual trapezoidal rule
        normalisation = np.abs(
            np.maximum(free_energy[0], np.average(free_energy[1:]))
        )  # For either decaying or steady-state problems

        accumulated_error = np.full_like(time, np.nan, dtype=float)
        accumulated_error[0] = 0.0
        accumulated_error[1:] = np.nancumsum(integrand) / normalisation

        instantaneous_error = dWdt_error / (
            + np.abs(dWdt)
            + np.abs(dWdt_nonlinear)
            + np.abs(injection)
            + np.abs(dissipation)
        )

        ax_error.plot(
            time,
            np.abs(accumulated_error),
            label="Accumulated",
            linewidth=1.5,
            color="black",
            linestyle="solid",
        )
        ax_error.plot(
            time,
            np.abs(instantaneous_error),
            label="Instantaneous",
            linewidth=1.5,
            color="blue",
            linestyle="solid",
        )
        ax_error.plot(
            time,
            dt,
            label="dt",
            linewidth=1.5,
            color="red",
            linestyle="solid",
        )

        # Setting plot options
        ax_error.set_xlim(np.nanmin(time), np.nanmax(time))
        ax_error.set_xlabel("Time")
        ax_error.set_yscale("log")

        ax_energy.legend()
        ax_balance.legend(ncols=2)
        ax_error.legend(ncols=2)

        # Save figures if required
        post.save(
            fig,
            name=figure_name,
            suffix="png",
            save_kwargs={"dpi": 300, "close": True},
        )

    plt.show()

    return


if __name__ == "__main__":

    # Setup parser
    parser = argparse.ArgumentParser(
        parents=[FlucsPostProcessing.parser()],
        description="Check TestFourierSystem free-energy conservation.",
    )
    args = parser.parse_args()

    # Initialise post-processing object
    post = FlucsPostProcessing(
        io_paths=args.io_path,
        save_directory=args.save_directory,
        output_files=["output.0d.nc"],
        constraint="both",
    )

    # Call function
    check_conservation(post, args)
