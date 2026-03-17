import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from flucs.postprocessing import FlucsPostProcessing

def plot_1d(ax, data, dims):
    # 1D plots here
    pass

def plot_2d(ax, data, dims):
    n_levels = 200
    ampl = np.max(np.abs(data))
    levels = np.linspace(-ampl, ampl, n_levels)
    ax.contourf(dims[0], dims[1], data[0, :, :].transpose(), cmap="seismic", levels=levels)

def plot_3d(ax, data, dims):
    # 3D cube plots here
    pass

def interative_slider_plot(post, loc):
    loc_str = f"realspace_slice/location_{loc}/"
    nc_paths = post.get_valid_files(loc_str + "data")

    plot_fun = {1: plot_1d, 2: plot_2d, 3: plot_3d}[loc.count(":")]

    for index, nc_path in enumerate(nc_paths):
        fig, ax = plt.subplots(1, 1)
        time, boundaries, _ = post.load_netcdf_variable(nc_path, "time")
        data, _, dims_dicts = post.load_netcdf_variable(nc_path, loc_str + "data")

        fig.subplots_adjust(bottom=0.22)
        slider = Slider(
            plt.axes([0.2, 0.1, 0.6, 0.03]),
            "Time",
            0,
            time.shape[0],
            valinit=0,
            valstep=1
        )

        dims = [dims_dicts[0]["x"], dims_dicts[0]["y"]]

        def update(fig=fig, ax=ax, slider=slider, time=time, data=data, dims=dims):
            time_index = int(slider.val)
            plot_fun(ax, data[time_index, 0, :, :], dims)
            slider.valtext.set_text(f"t = {time[time_index]}")

        slider.on_changed(update)

    plt.show()

if __name__ == "__main__":
    # Setup parser
    parser = argparse.ArgumentParser(
        parents=[FlucsPostProcessing.parser()],
        description=(
            "Plots any of the variables from 'output.realspacend.nc' against time."
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
        "n",
        help="Specifies which output file to use. "
             "Data is to be read from output.realspace[n]d.nc"
    )

    parser.add_argument(
        "location",
        help="Location in the format ifield,iz,ix,iy that specifies "
             "the location of the data to plot.",
    )

    args = parser.parse_args()

    # Initialise post-processing object
    post = FlucsPostProcessing(
        io_paths=args.io_path,
        save_directory=args.save_directory,
        output_file=f"output.realspace{args.n}d.nc",
        constraint="none",
    )

    if args.list:
        post.list_netcdf_variables()
        exit()

    # Call functions
    interative_slider_plot(post, args.location)
