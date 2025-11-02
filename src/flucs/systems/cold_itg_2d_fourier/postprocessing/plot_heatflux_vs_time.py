import argparse
import numpy as np
import pathlib as pl
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from flucs.postprocessing import FlucsPostProcessing, FlucsPostProcessing_parser


def plot_heatflux_vs_time(post):

    # Get valid files for the specified variable 
    nc_paths = post.get_valid_files("heatflux")

    # Initialise plotting
    fig, ax = plt.subplots(1, 1, layout='constrained')

    figure_name = "heatflux_vs_time"
    fig.canvas.manager.set_window_title(figure_name)

    # Iterate over output files
    for index, nc_path in enumerate(nc_paths):

        # Assign identifiers
        sim_label = pl.Path(nc_path)
        sim_color = plt.cm.rainbow(np.linspace(0, 1, len(nc_paths)))[index]

        # Read data from netCDF file
        time, _ = post.load_netcdf_variable(nc_path, "time")
        data, _ = post.load_netcdf_variable(nc_path, "heatflux")

        # Plot data
        ax.plot(time, data, label=sim_label, linewidth=1.5, color=sim_color, linestyle='solid')

    # Setting plot options
    ax.set_xlabel(r"$(2c_s/L_B)t$")
    ax.set_ylabel(r"$Q_i/[4 n_e T_e c_s (\rho_s/L_B)^2]$")

    ax.set_xlim(np.min(time), np.max(time))
    ax.set_ylim(ymin=0.0)

    ax.legend()

    # Save figures if required
    post.save(fig, name=figure_name, suffix="png", save_kwargs={"dpi":300, "close": True})

    plt.show()

    return

if __name__ == "__main__":

    # Setup parser
    parser = argparse.ArgumentParser(
        parents=[FlucsPostProcessing_parser()], 
        description="Plots any of the variables from 'output.0d.nc' against time.", 
    )

    args = parser.parse_args()

    # Initialise post-processing object
    post = FlucsPostProcessing(
        io_paths=args.io_path,
        save_directory=args.save_directory,
        output_type="output.0d.nc",
        constraint="both"
    )

    # Call function
    plot_heatflux_vs_time(post)