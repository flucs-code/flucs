import argparse
import pathlib as pl

import matplotlib.pyplot as plt
import numpy as np

from flucs.postprocessing import FlucsPostProcessing


def plot_0d_vs_time(post, args):
    # Get variable
    variable = args.variable
    if variable is None:
        raise ValueError(
            "Please specify a variable to plot using --variable/-v."
        )
    variable = str(variable)

    # Get valid files for the specified variable
    nc_paths = post.get_valid_netcdf_paths(variable)

    # Initialise plotting
    fig, ax = plt.subplots(1, 1, layout="constrained")

    figure_name = f"{variable.split('/', 1)[-1]}_vs_time"
    fig.canvas.manager.set_window_title(figure_name)

    min_time = np.nan
    max_time = np.nan

    # Iterate over output files
    for index, nc_path in enumerate(nc_paths):
        # Assign identifiers
        sim_label = pl.Path(nc_path).parent.name
        sim_color = plt.cm.rainbow(np.linspace(0, 1, len(nc_paths)))[index]

        # Read data from netCDF file
        time = post.load_netcdf_variable(nc_path, "time")[0]
        data = post.load_netcdf_variable(nc_path, variable)[0]

        # Validate dimension
        if data.ndim != 1:
            raise ValueError(
                f"Expected a 0D time-dependent variable, but '{variable}' "
                f"loaded with shape {data.shape}."
            )

        # Plot data
        ax.plot(
            time,
            data,
            label=sim_label,
            linewidth=1.5,
            color=sim_color,
            linestyle="solid",
        )

        # Change x axis lims if needed
        min_time = np.nanmin((min_time, np.min(time)))
        max_time = np.nanmax((max_time, np.max(time)))

    # Setting plot options
    ax.set_xlabel("Time")
    ax.set_ylabel(variable)

    ax.set_xlim((min_time, max_time))

    ax.legend()

    # Save figures if required
    post.save(
        fig,
        name=figure_name,
        suffix="png",
        save_kwargs={"dpi": 300},
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

    parser.add_argument(
        "--list",
        "-l",
        action="store_true",
        default=False,
        help="List all available variables to plot and exit.",
    )

    parser.add_argument(
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
        output_files=["output.0d.nc"],
        constraint="none",
    )

    if args.list:
        post.list_netcdf_variables()
        exit()

    # Call function
    plot_0d_vs_time(post, args)
