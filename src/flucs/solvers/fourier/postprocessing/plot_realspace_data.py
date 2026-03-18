import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from collections import OrderedDict
from bisect import bisect_right

from flucs.postprocessing import FlucsPostProcessing

def plot_1d(ax, data, dims):
    # 1D plots here
    pass

def plot_2d(axs, data, plot_dims):
    ampl = np.max(np.abs(data))

    try:
        for ifield, ax in enumerate(axs):
            ax.clear()
            ax.imshow(
                data[ifield, :, :].transpose(),
                origin="lower",
                extent=[plot_dims[0].min(), plot_dims[0].max(), plot_dims[1].min(), plot_dims[1].max()],
                aspect="auto",
                cmap="seismic",
                vmin=-ampl,
                vmax=ampl,
            )
    except IndexError:
        print("Could not plot, likely missing data.")


def plot_3d(ax, data, dims):
    # 3D cube plots here
    pass


def parse_slice(s: str) -> slice:
    # Check if slice syntax or just a single index
    if ":" not in s:
        index = int(s.strip())
        return slice(index, index + 1)

    parts = s.split(":")
    if len(parts) > 3:
        raise ValueError(f"Invalid slice string: {s}")

    def get_index(i):
        return None if i == "" else int(i)

    return slice(*(get_index(p) for p in parts))

def interative_slider_plot(post, loc):
    loc_str = f"realspace_data/location_{loc}/"
    nc_paths = post.get_valid_netcdf_paths(loc_str + "data")
    figure_name = (
        "realspace_data_"
        + loc.replace(",", "_").replace(":", "all").replace("/", "_")
    )

    for index, nc_path in enumerate(nc_paths):
        time, boundaries, _ = post.load_netcdf_variable(nc_path, "time")
        data, _, dims_dicts = post.load_netcdf_variable(nc_path, loc_str + "data")
        initial_time_index = max(0, time.shape[0] - 1)

        fig, axs = plt.subplots(1, data.shape[1])
        if data.shape[1] == 1:
            axs = (axs,)
        fig.canvas.manager.set_window_title(f"{figure_name}_{index}")

        fig.subplots_adjust(bottom=0.22)
        slider = Slider(
            plt.axes([0.2, 0.1, 0.6, 0.03]),
            "Time",
            0,
            initial_time_index,
            valinit=initial_time_index,
            valstep=1
        )

        plot_dims_groups = []
        for i in range(len(dims_dicts)):
            plot_dims_groups.append([])
            for num_coord, coord in enumerate(dims_dicts[i].keys()):
                if num_coord == 0:
                    continue  # The first coord is the field index
                if len(dims_dicts[i][coord]) == 1:
                    continue
                plot_dims_groups[-1].append(dims_dicts[i][coord])

        axes_to_remove = tuple(i for i, n in enumerate(data.shape) if n == 1 and i > 1)
        plot_fun = {1: plot_1d, 2: plot_2d, 3: plot_3d}[3 - len(axes_to_remove)]

        def update(val, fig=fig, axs=axs, slider=slider, time=time, data=np.squeeze(data, axes_to_remove), plot_dims_groups=plot_dims_groups):
            time_index = min(int(slider.val), time.shape[0] - 1)
            restart_index = bisect_right(boundaries, time_index)
            plot_fun(axs, data[time_index, :], plot_dims_groups[restart_index])
            slider.valtext.set_text(f"t = {time[time_index]:.2f}")
            fig.canvas.draw_idle()

        slider.on_changed(update)
        update(initial_time_index)

    if plt.get_fignums():
        post.save(
            plt.figure(plt.get_fignums()[0]),
            name=figure_name,
            suffix="png",
            save_kwargs={"dpi": 300},
        )
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
        output_files=[f"output.realspace{args.n}d.nc"],
        constraint="none",
    )

    if args.list:
        post.list_netcdf_variables()
        exit()

    # Call functions
    interative_slider_plot(post, args.location)
