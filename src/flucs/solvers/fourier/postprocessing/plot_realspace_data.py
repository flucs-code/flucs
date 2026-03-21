import argparse
import numpy as np
import pathlib as pl
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from bisect import bisect_right

from flucs.postprocessing import FlucsPostProcessing

### TODO needs significant cleaning up everywhere


# Helper functions
def increment_time_index(fig, inc):
    """Increments the time index of a figure by some integer increment.

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
    """Sets the time index of a figure to a fraction of its total range.

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
Interact with the plots using the following key binds.

Increment the time index with
left: -1
right: +1
ctrl + left: -5
ctrl + right: +5

Use 0--9 to set the time index to 0%, 10%, ..., 90% of the final time.
"""
)


# Key event handler
def on_key_pressed(event):
    """"
    Handles key press events on the figure.

    Parameters    
    ----------
    event: matplotlib key press event
    
    """

    action = actions.get(event.key, None)
    if action is not None:
        action(event.canvas.figure)
        event.canvas.figure._update_plot()


def plot_1d(axs, data, plot_dims, coord_names=None):
    
    # Iterate over fields and plot data
    for ifield, ax in enumerate(axs):
        ax.clear()
        try:
            ax.plot(
                plot_dims[0],
                data[ifield, :],
                color="black",
                linewidth=1.5,
            )

            # Set plot options
            ax.set_title(f"field_{ifield}")
            if coord_names is not None:
                ax.set_xlabel(coord_names[0])
            ax.set_xlim(plot_dims[0].min(), plot_dims[0].max())
            ax.axhline(y=0, color="gray", linestyle="solid", linewidth=1.0)

        except IndexError:
            print("Could not render plot, likely due to missing data.")


def plot_2d(axs, data, plot_dims, coord_names=None):

    # Iterate over fields and plot data. 
    for ifield, ax in enumerate(axs):
        ax.clear()
        try:
            amplitude = np.max(np.abs(data[ifield, :, :]))
            ax.imshow(
                data[ifield, :, :].transpose(),
                origin="lower",
                extent=[plot_dims[0].min(),
                        plot_dims[0].max(),
                        plot_dims[1].min(),
                        plot_dims[1].max()],
                aspect="equal",
                cmap="seismic",
                vmin=-amplitude,
                vmax=amplitude,
            )

            # Set plot options
            ax.set_title(f"field_{ifield}")
            if coord_names is not None:
                ax.set_xlabel(coord_names[0])
                if ifield == 0:
                    ax.set_ylabel(coord_names[1])

        except IndexError:
            print("Could not render plot, likely due to missing data.")


def plot_3d(axs, data, plot_dims, coord_names=None):

    # Extract dimensions
    z, x, y = plot_dims

    # Set colormap
    cmap = cm.seismic

    # Iterate over fields and plot data
    for ifield, ax in enumerate(axs):
        ax.clear()

        try:
            # Extract full cube for current field
            field_data = data[ifield, :, :, :]   # shape = (nz, nx, ny)

            # Normalise to maximum
            amplitude = np.nanmax(np.abs(field_data))
            norm = colors.Normalize(vmin=-amplitude, vmax=amplitude)

            # Plot the physical x/y face at fixed z = z[-1].
            # Display axes: X->z, Y->x, Z->y
            Y_xy, Z_xy = np.meshgrid(x, y, indexing="ij")
            X_xy = np.full_like(Y_xy, z[-1])
            data_xy = field_data[-1, :, :]
            ax.plot_surface(
                X_xy, Y_xy, Z_xy,
                facecolors=cmap(norm(data_xy)),
                edgecolor="none",
                rstride=1,
                cstride=1,
                shade=False,
                linewidth=0,
                antialiased=False,
            )

            # Plot the physical z/y face at fixed x = x[0].
            # Display axes: X->z, Y->x, Z->y
            X_zy, Z_zy = np.meshgrid(z, y, indexing="ij")
            Y_zy = np.full_like(X_zy, x[0])
            data_zy = field_data[:, 0, :]
            ax.plot_surface(
                X_zy, Y_zy, Z_zy,
                facecolors=cmap(norm(data_zy)),
                edgecolor="none",
                rstride=1,
                cstride=1,
                shade=False,
                linewidth=0,
                antialiased=False,
            )

            # Plot the physical z/x face at fixed y = y[-1].
            # Display axes: X->z, Y->x, Z->y
            X_zx, Y_zx = np.meshgrid(z, x, indexing="ij")
            Z_zx = np.full_like(X_zx, y[-1])
            data_zx = field_data[:, :, -1]
            ax.plot_surface(
                X_zx, Y_zx, Z_zx,
                facecolors=cmap(norm(data_zx)),
                edgecolor="none",
                rstride=1,
                cstride=1,
                shade=False,
                linewidth=0,
                antialiased=False,
            )

            # Set plot options
            ax.set_title(f"field_{ifield}")
            ax.set_xlabel("z")
            ax.set_ylabel("x")
            ax.set_zlabel("y")

            ax.set_xlim(z.min(), z.max())
            ax.set_ylim(x.min(), x.max())
            ax.set_zlim(y.min(), y.max())
            ax.set_box_aspect((1, 1, 1))

            ax.grid(False)
            ax.tick_params(labelsize=6, pad=0)

            ax.set_proj_type("ortho")
            ax.view_init(elev=18, azim=-30)

        except IndexError:
            print("Could not render plot, likely due to missing data.")


def plot_realspace_data(post, location, time_to_plot):

    # Parse user input location
    if location is None:
        raise ValueError("No location provided. See --help/-h for details.")
    
    location_parts = [part.strip() for part in location.split(",")]
    location_parts = [part for part in location_parts if part]

    if len(location_parts) != 4:
        raise ValueError(
            "Location must have four comma-separated entries: ifield,iz,ix,iy"
        )

    # Get netCDF paths for the specified location
    loc = ",".join(location_parts)
    loc_str = f"realspace_data/location_{loc}/"
    nc_paths = post.get_valid_netcdf_paths(loc_str + "data")

    print(actions_help_text)

    # Iterate over netCDF files and plot data
    for index, nc_path in enumerate(nc_paths):

        # Load time and data from netCDF file
        time, boundaries, _ = post.load_netcdf_variable(nc_path, "time")
        data, _, dims_dicts = post.load_netcdf_variable(nc_path,
                                                        loc_str + "data")

        # Determine the initial render time
        if time_to_plot is not None:
            initial_render_time_index = min(
                bisect_right(time, time_to_plot),
                len(time) - 1,
            )
        else:
            last_group_with_data = len(dims_dicts) - 1
            while last_group_with_data >= 0 and not dims_dicts[last_group_with_data]:
                last_group_with_data -= 1

            if last_group_with_data < 0:
                raise ValueError(f"No data found for {loc} in {nc_path}.")

            if last_group_with_data == len(dims_dicts) - 1:
                initial_render_time_index = len(time) - 1
            else:
                initial_render_time_index = boundaries[last_group_with_data] - 1

        # Remove any axes corresponding to dimensions of length 1. 
        axes_to_remove = tuple(
            i for i, n in enumerate(data.shape)
            if n == 1 and i > 1
        )
        plot_rank = 3 - len(axes_to_remove)
        plot_function = {1: plot_1d, 2: plot_2d, 3: plot_3d}[plot_rank]

        # Initialise plotting and cast to list
        if plot_rank == 3:
            fig = plt.figure()
            axs = tuple(
                fig.add_subplot(1, data.shape[1], i + 1, projection="3d")
                for i in range(data.shape[1])
            )
        else:
            fig, axs = plt.subplots(
                1,
                data.shape[1],
                sharex=True,
                sharey=True,
                layout="constrained",
            )

        if data.shape[1] == 1:
            axs = (axs,)

        fig._base_name = (
            f"realspace_data_{loc}"
            + f"_{pl.Path(nc_path).parent.name}"
        )

        # Get the minimal dimensions for plotting, and adjust axs accordingly
        plot_dims_groups = []
        plot_coord_names_groups = []

        for i in range(len(dims_dicts)):
            plot_dims_groups.append([])
            plot_coord_names_groups.append([])

            for num_coord, coord in enumerate(dims_dicts[i].keys()):
                if num_coord == 0:
                    continue  # The first coord is the field index
                if len(dims_dicts[i][coord]) == 1:
                    continue
                plot_dims_groups[-1].append(dims_dicts[i][coord])
                plot_coord_names_groups[-1].append(coord)

        # Define updating function, connect key events, and update
        def update(
            fig=fig,
            axs=axs,
            time=time,
            data=np.squeeze(data, axes_to_remove),
            plot_dims_groups=plot_dims_groups,
            boundaries=boundaries,
        ):
            restart_index = bisect_right(boundaries, fig._time_index)

            plot_function(
                axs,
                data[fig._time_index, :],
                plot_dims_groups[restart_index],
                plot_coord_names_groups[restart_index] if plot_coord_names_groups else None
            )

            fig.canvas.manager.set_window_title(
                f"{fig._base_name} "
                f"at t = {time[fig._time_index]:.2f}"
            )
            fig.canvas.draw_idle()

        fig._time_index = initial_render_time_index
        fig._time_array_len = len(time)
        fig._update_plot = update
        fig.canvas.mpl_connect("key_press_event", on_key_pressed)

        update()

        # Save figure if required
        post.save(
            fig,
            name=(
                fig._base_name
                + f"_{time[initial_render_time_index]:.2e}"
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
