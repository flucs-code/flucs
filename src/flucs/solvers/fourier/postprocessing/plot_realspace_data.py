import argparse
import numpy as np
import pathlib as pl
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from bisect import bisect_right

from flucs.postprocessing import FlucsPostProcessing

### TODO needs significant cleaning up everywhere


# Helper functions
def increment_time_index(fig, inc):
    """Increments the time index of a figure by inc.

    Parameters
    ----------
    fig: matplotlib figure
    inc: integer steps to increment

    """
    fig._time_index += inc
    if fig._time_index < 0:
        fig._time_index = 0
    elif fig._time_index >= fig._time_array_len:
        fig._time_index = fig._time_array_len - 1


def set_relative_time_index(fig, frac):
    """Sets the time index of a figure to a fraction of its total time.

    Parameters
    ----------
    fig: matplotlib figure
    frac: fraction (0 to 1) to set the time index

    """
    fig._time_index = int(frac * (fig._time_array_len - 1))


# Key event actions
actions = {
    "left": lambda fig: increment_time_index(fig, -1),
    "right": lambda fig: increment_time_index(fig, +1),
    "ctrl+left": lambda fig: increment_time_index(fig, -5),
    "ctrl+right": lambda fig: increment_time_index(fig, +5),
    "0": lambda fig: set_relative_time_index(fig, 0.0),
    "1": lambda fig: set_relative_time_index(fig, 0.1),
    "2": lambda fig: set_relative_time_index(fig, 0.2),
    "3": lambda fig: set_relative_time_index(fig, 0.3),
    "4": lambda fig: set_relative_time_index(fig, 0.4),
    "5": lambda fig: set_relative_time_index(fig, 0.5),
    "6": lambda fig: set_relative_time_index(fig, 0.6),
    "7": lambda fig: set_relative_time_index(fig, 0.7),
    "8": lambda fig: set_relative_time_index(fig, 0.8),
    "9": lambda fig: set_relative_time_index(fig, 0.9),
}

actions_help_text = (
"""
To interact with the plots, use the following key binds:

Increment the time index with
left: -1
right: +1
Ctrl + left: -5
Ctrl + right: +5

Use 0--9 to set the time index to 0%, 10%, ..., 90% of the final time.
"""
)


# Key event handler
def on_key_pressed(event):
    action = actions.get(event.key, None)
    if action is not None:
        action(event.canvas.figure)
        event.canvas.figure._update_plot()


def plot_1d(ax, data, dims):
    # 1D plots here
    pass


def plot_2d(axs, data, plot_dims):
    for ifield, ax in enumerate(axs):
        ax.clear()
        try:
            ampl = np.max(np.abs(data[ifield, :, :]))
            ax.imshow(
                data[ifield, :, :].transpose(),
                origin="lower",
                extent=[plot_dims[0].min(),
                        plot_dims[0].max(),
                        plot_dims[1].min(),
                        plot_dims[1].max()],
                aspect="equal",
                cmap="seismic",
                vmin=-ampl,
                vmax=ampl,
            )
        except IndexError:
            print("Could not plot, likely missing data.")


def plot_3d(ax, data, dims):
    # 3D cube plots here
    pass


def plot_realspace_data(post, location, time_to_plot):
    location_parts = [part.strip() for part in location.split(",")]
    location_parts = [part for part in location_parts if part]

    if len(location_parts) != 4:
        raise ValueError(
            "Location must have four comma-separated entries: ifield,iz,ix,iy"
        )

    loc = ",".join(location_parts)
    loc_str = f"realspace_data/location_{loc}/"
    nc_paths = post.get_valid_netcdf_paths(loc_str + "data")

    print(actions_help_text)
    for index, nc_path in enumerate(nc_paths):
        print(index, nc_path)
        time, boundaries, _ = post.load_netcdf_variable(nc_path, "time")
        data, _, dims_dicts = post.load_netcdf_variable(nc_path,
                                                        loc_str + "data")

        initial_render_time_index = -1

        if time_to_plot is not None:
            initial_render_time_index = bisect_right(time, time_to_plot)

        # If not specifiec, the initial render time is
        # the last time for which we have data
        elif len(boundaries) > 1:
            # Deal with multiple groups if they exist
            initial_render_time_index = len(time) - 1
            last_group_with_data = -1
            while not dims_dicts[last_group_with_data]:
                last_group_with_data -= 1

            initial_render_time_index = boundaries[last_group_with_data + 1] - 1

        fig, axs = plt.subplots(1, data.shape[1], sharex=True, sharey=True)

        # It's easier to do stuff later on if the axes are always a list
        if data.shape[1] == 1:
            axs = (axs,)

        # This is used as the initial part of the figure window title
        # to which we append the selected current time
        fig._figure_name_initial = (
            f"realspace_data_{loc}"
            # + loc
            + f"_{pl.Path(nc_path).parent.name}"
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

        # Remove axes of length 1 and decide
        # what kind of a plot we are going to do
        axes_to_remove = tuple(
            i for i, n in enumerate(data.shape)
            if n == 1 and i > 1
        )
        plot_rank = 3 - len(axes_to_remove)
        plot_fun = {1: plot_1d, 2: plot_2d, 3: plot_3d}[plot_rank]

        def update(
            fig=fig,
            axs=axs,
            time=time,
            data=np.squeeze(data, axes_to_remove),
            plot_dims_groups=plot_dims_groups,
            boundaries=boundaries,
        ):
            restart_index = bisect_right(boundaries, fig._time_index)

            plot_fun(axs,
                     data[fig._time_index, :],
                     plot_dims_groups[restart_index])

            fig.canvas.manager.set_window_title(
                f"{fig._figure_name_initial} "
                f"at t = {time[fig._time_index]:.2f}"
            )
            fig.canvas.draw_idle()

        fig._time_index = initial_render_time_index
        fig._time_array_len = len(time)
        fig._update_plot = update
        fig.canvas.mpl_connect("key_press_event", on_key_pressed)

        update()

        post.save(
            fig,
            name=(
                fig._figure_name_initial
                + f"_{time[initial_render_time_index]:.2f}"
            ),
            suffix="png",
            save_kwargs={"dpi": 300},
        )

    plt.show()


if __name__ == "__main__":
    # Setup parser
    parser = argparse.ArgumentParser(
        parents=[FlucsPostProcessing.parser()],
        description=(
            "Plots any of the variables from 'output.realspace*.nc' against time."
        ),
    )

    parser.add_argument(
        "--list",
        "-l",
        action="store_true",
        default=False,
        help="List all available realspace locations and exit.",
    )

    parser.add_argument(
        "--location",
        "-loc",
        type=str,
        help=(
            "Realspace location in the format ifield,iz,ix,iy. "
            "Commas are required; quote the argument if you include spaces."
        ),
    )

    parser.add_argument(
        "--time",
        "-t",
        type=float,
        default=None,
        help=(
            "Time at which to (initially) plot the data. If not provided, the last time step will be plotted."
        ),
    )

    args = parser.parse_args()

    # Initialise post-processing object
    post = FlucsPostProcessing(
        io_paths=args.io_path,
        save_directory=args.save_directory,
        output_files=[f"output.realspace*.nc"],
        constraint="none",
    )

    if args.list:
        post.list_netcdf_variables()
        exit()

    # Call function
    plot_realspace_data(post, args.location, args.time)
