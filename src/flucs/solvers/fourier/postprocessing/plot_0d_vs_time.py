import glob
import argparse
import numpy as np
import pathlib as pl
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from flucs.postprocessing import FlucsPostProcessing, FlucsPostProcessing_parser


def plot_0d_vs_time(post, variable=None):

    # Get valid files for the specified variable 
    nc_paths = post.get_valid_files(str(variable))

    # Initialise plotting
    fig, ax = plt.subplots(1, 1, layout='constrained')
    fig.canvas.manager.set_window_title(f"test")

    # Iterate over output files
    for index, nc_path in enumerate(nc_paths):

        # Assign identifiers
        sim_label = pl.Path(nc_path)
        sim_color = plt.cm.rainbow(np.linspace(0, 1, len(nc_paths)))[index]

        # Read data from netCDF file
        time, _ = post.load_netcdf_variable(nc_path, "time")
        data, _ = post.load_netcdf_variable(nc_path, variable)

        # Plot data
        ax.plot(time, data, label=sim_label, linewidth=1.5, color=sim_color, linestyle='solid')

    # Setting plot options
    ax.set_xlabel("Time")
    ax.set_ylabel(variable)

    ax.set_xlim(np.min(time), np.max(time))
    ax.set_ylim(ymin=0.0)

    ax.legend()

    plt.show()

    save_directory = post.save_directory
    if save_directory is not None:
        save_directory.mkdir(parents=True, exist_ok=True)
        save_path = save_directory / f"test.png"
        fig.savefig(save_path)
        print(f"Save location: {save_path}")

    return

if __name__ == "__main__":

    # Setup parser
    parser = argparse.ArgumentParser(
        parents=[FlucsPostProcessing_parser()], 
        description="Plots any of the variables from 'output.0d.nc' against time.", 
    )

    operation_modes = parser.add_mutually_exclusive_group(required=True)

    operation_modes.add_argument(
        "--list", "-l",
        action="store_true",
        default=False,
        help="List all available variables to plot and exit."
    )

    operation_modes.add_argument(
        "--variable", "-v",
        type=str,
        default=None,
        help="Name of variable to plot."
    )

    args = parser.parse_args()

    # Initialise post-processing object
    post = FlucsPostProcessing(
        io_paths=args.io_path,
        save_directory=args.save_directory,
        output_type="output.0d.nc",
        constraint="none"
    )

    if args.list:
        post.list_netcdf_variables()
        exit()

    # Run script
    plot_0d_vs_time(post, variable=args.variable)

        
  