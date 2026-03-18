import argparse
import numpy as np
import pathlib as pl
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from bisect import bisect_right

from flucs.postprocessing import FlucsPostProcessing

### TODO needs significant cleaning up everywhere

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
    
    for index, nc_path in enumerate(nc_paths):
        print(index, nc_path)
        time, boundaries, _ = post.load_netcdf_variable(nc_path, "time")
        data, _, dims_dicts = post.load_netcdf_variable(nc_path, loc_str + "data")
        initial_render_time = (
            time[np.argmin(np.abs(time - time_to_plot))]
            if time_to_plot is not None
            else time[-1]
        )

        fig, axs = plt.subplots(1, data.shape[1], sharex=True, sharey=True)
        if data.shape[1] == 1:
            axs = (axs,)
        figure_name = (
            "realspace_data_"
            + loc.replace(",", "_").replace(":", "-").replace("/", "_")
            + f"_{pl.Path(nc_path).parent.name}"
        )
        fig.canvas.manager.set_window_title(f"{figure_name}")

        fig.subplots_adjust(bottom=0.22)
        slider = Slider(
            plt.axes([0.2, 0.1, 0.6, 0.03]),
            "Time",
            float(time[0]),
            float(time[-1]),
            valinit=initial_render_time,
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
        plot_rank = 3 - len(axes_to_remove)
        plot_fun = {1: plot_1d, 2: plot_2d, 3: plot_3d}[plot_rank]

        valid_time_mask = np.array(
            [
                len(plot_dims_groups[bisect_right(boundaries, i)]) == plot_rank
                for i in range(len(time))
            ]
        )
        valid_time_indices = np.flatnonzero(valid_time_mask)

        if valid_time_indices.size == 0:
            print(f"No plottable data found for {loc} in {nc_path}.")
            continue

        snapping = False
        last_missing_index = None

        def update(
            val,
            fig=fig,
            axs=axs,
            slider=slider,
            time=time,
            data=np.squeeze(data, axes_to_remove),
            plot_dims_groups=plot_dims_groups,
        ):
            nonlocal snapping, last_missing_index

            if snapping:
                return

            time_index = int(np.argmin(np.abs(time - slider.val)))

            if not valid_time_mask[time_index]:
                nearest_index = int(
                    valid_time_indices[
                        np.argmin(np.abs(time[valid_time_indices] - slider.val))
                    ]
                )

                if last_missing_index != time_index:
                    print(
                        f"No data available at t = {time[time_index]:.2f}; "
                        f"snapping to nearest available t = {time[nearest_index]:.2f}."
                    )
                    last_missing_index = time_index

                snapped_time = float(time[nearest_index])
                if slider.val != snapped_time:
                    snapping = True
                    slider.set_val(snapped_time)
                    snapping = False

                time_index = nearest_index
            else:
                last_missing_index = None

            restart_index = bisect_right(boundaries, time_index)
            plot_fun(axs, data[time_index, :], plot_dims_groups[restart_index])
            slider.valtext.set_text(f"t = {time[time_index]:.2f}")
            fig.canvas.draw_idle()

        slider.on_changed(update)
        update(initial_render_time)

        post.save(
            fig,
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
