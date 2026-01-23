import argparse
import pathlib as pl

import matplotlib.pyplot as plt
import numpy as np

from flucs.postprocessing import FlucsPostProcessing


def plot_0d_vs_time(post, variable=None):
    # Get valid files for the specified variable
    nc_paths = post.get_valid_files(str(variable))

    # Initialise plotting
    fig, ax = plt.subplots(1, 1, layout="constrained")

    figure_name = f"{str(variable).split('/', 1)[-1]}_vs_time"
    fig.canvas.manager.set_window_title(figure_name)

    # Iterate over output files
    for index, nc_path in enumerate(nc_paths):
        # Assign identifiers
        sim_label = pl.Path(nc_path)
        sim_color = plt.cm.rainbow(np.linspace(0, 1, len(nc_paths)))[index]

        # Read data from netCDF file
        time, _ = post.load_netcdf_variable(nc_path, "time")
        data, _ = post.load_netcdf_variable(nc_path, variable)

        # Plot data
        ax.plot(
            time,
            data,
            label=sim_label,
            linewidth=1.5,
            color=sim_color,
            linestyle="solid",
        )

    # Setting plot options
    ax.set_xlabel("Time")
    ax.set_ylabel(variable)

    ax.set_xlim(np.min(time), np.max(time))
    ax.set_ylim(ymin=0.0)

    ax.legend()

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
        description=(
            "Plots any of the variables from 'output.0d.nc' against time."
        ),
    )

    operation_modes = parser.add_mutually_exclusive_group(required=True)

    operation_modes.add_argument(
        "--list",
        "-l",
        action="store_true",
        default=False,
        help="List all available variables to plot and exit.",
    )

    operation_modes.add_argument(
        "--variable",
        "-v",
        type=str,
        default=None,
        help="Name of variable to plot.",
    )

    args = parser.parse_args()

    # Initialise post-processing object
    post = FlucsPostProcessing(
        io_paths=args.io_path,
        save_directory=args.save_directory,
        output_file="output.0d.nc",
        constraint="none",
    )

    if args.list:
        post.list_netcdf_variables()
        exit()

    # Call function
    plot_0d_vs_time(post, variable=args.variable)
