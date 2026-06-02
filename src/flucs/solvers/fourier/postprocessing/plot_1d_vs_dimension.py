import argparse
import pathlib as pl

import matplotlib.pyplot as plt
import numpy as np

from flucs.postprocessing import FlucsPostProcessing


def plot_1d_vs_dimension(post, args):

    # Alias arguments
    variable = str(args.variable)
    groups = args.groups
    fraction = args.fraction

    variable_name = variable.rsplit("/", 1)[-1]

    # Get valid files for the specified variable
    nc_paths = post.get_valid_netcdf_paths(variable)

    # Initialise plotting
    fig, ax = plt.subplots(1, 1, layout="constrained")
    figure_name = None

    # Iterate over output files
    for index, nc_path in enumerate(nc_paths):
        
        # Assign identifiers
        sim_label = pl.Path(nc_path).parent.name
        sim_color = plt.cm.rainbow(np.linspace(0, 1, len(nc_paths)))[index]

        # Read data from netCDF file
        time = post.load_netcdf_variable(
            nc_path, 
            "time", 
            groups=groups,
        )[0]
        data, _, dims_dicts = post.load_netcdf_variable(
            nc_path,
            variable,
            groups=groups,
        )

        # Validate dimension
        dims = next(dims for dims in reversed(dims_dicts) if dims)
        if len(dims) != 1:
            raise ValueError(
                f"Expected a 1D variable, but '{variable}' has "
                f"dimensions {list(dims)}."
            )
        dimension_name, dimension = next(iter(dims.items()))

        # Mask for logarithmic axes
        mask = dimension >= 0.0
        dimension = dimension[mask]
        data = data[:, mask]

        # Time average
        mask = time >= (
            np.min(time) + (1.0 - fraction) * (np.max(time) - np.min(time))
        )
        data_avg = np.nanmean(data[mask], axis=0)

        # Plot data
        ax.plot(
            dimension,
            np.abs(data_avg),
            label=sim_label,
            linewidth=1.5,
            color=sim_color,
            linestyle="solid",
        )

    # Setting plot options
    ax.set_xlabel(dimension_name)
    ax.set_ylabel(f"|{variable_name}|")

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.legend()

    # Assign figure name based on last file
    figure_name = f"{variable_name}_vs_{dimension_name}"
    fig.canvas.manager.set_window_title(figure_name)

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
            "Plots any of the variables from 'output.1d.nc' against their "
            "coordinate dimension."
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

    parser.add_argument(
        "--fraction",
        "-f",
        type=float,
        default=0.2,
        help=(
            "Final fraction of the selected time series over which to average. "
            "Default is 0.2, i.e. the final 20 percent."
        ),
    )

    parser.add_argument(
        "--groups",
        "-g",
        nargs="+",
        type=str,
        default=None,
        required=False,
        help="Names of groups to load. Loads all groups by default.",
    )

    args = parser.parse_args()

    # Initialise post-processing object
    post = FlucsPostProcessing(
        io_paths=args.io_path,
        save_directory=args.save_directory,
        output_files=["output.1d.nc"],
        constraint="none",
    )

    if args.list:
        post.list_netcdf_variables()
        exit()

    # Call function
    plot_1d_vs_dimension(post, args)
