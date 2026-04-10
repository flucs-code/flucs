import argparse
import pathlib as pl

import matplotlib.pyplot as plt
import numpy as np

from flucs.postprocessing import FlucsPostProcessing


def plot_eigensystem(post):

    # Get valid files for the specified variable
    nc_paths = post.get_valid_netcdf_paths(
        "linear_eigensystem/eigvals_solver_real"
    )

    # Initialise plotting
    fig, axs = plt.subplots(2, 1, layout="constrained", sharex=True)

    figure_name = f"linear_eigensystem"
    fig.canvas.manager.set_window_title(figure_name)

    # Iterate over output files
    for index, nc_path in enumerate(nc_paths):

        # Assign identifiers
        sim_label = pl.Path(nc_path).parent.name
        sim_color = plt.cm.rainbow(np.linspace(0, 1, len(nc_paths)))[index]

        # Get eigenvalues and grids
        eigvals_sol, _, dims_dicts = post.load_netcdf_variable_complex(
            nc_path, "linear_eigensystem/eigvals_solver"
        )
        eigvals_ref = post.load_netcdf_variable_complex(
            nc_path, "linear_eigensystem/eigvals_reference"
        )[0]

        dims = next(d for d in reversed(dims_dicts) if d)
        kz = np.asarray(dims["kz"])
        kx = np.asarray(dims["kx"])
        ky = np.asarray(dims["ky"])

        # Calculate and report errors between solver and reference eigenvalues
        finite = np.isfinite(eigvals_sol) & np.isfinite(eigvals_ref)

        if np.any(finite):
            abs_diff = np.abs(eigvals_sol[finite] - eigvals_ref[finite])
            rel_norm = np.maximum(
                np.abs(eigvals_sol[finite]), np.abs(eigvals_ref[finite])
            )
            rel_diff = np.divide(
                abs_diff, 
                rel_norm, 
                out=np.zeros_like(abs_diff), 
                where=rel_norm!=0
            )
        else:
            abs_diff = eigvals_ref[0]
            rel_diff = eigvals_ref[0]

        print(
            f"{np.max(rel_diff):>9.3e}, {np.max(abs_diff):>9.3e} "
            f"({sim_label})"
        )

        # Select data to plot
        it = -1
        ikz = int(np.argmin(np.abs(kz - 0.0)))
        ikx = int(np.argmin(np.abs(kx - 0.0)))

        eigvals_sol_plot = eigvals_sol[it, :, ikz, ikx, :]
        eigvals_ref_plot = eigvals_ref[it, :, ikz, ikx, :]

        gammas_sol = eigvals_sol_plot.imag
        gammas_max = np.max(gammas_sol, axis=0)
        most_unstable_mode = np.argmax(gammas_sol, axis=0) 

        selected_mode = np.where(
            gammas_max > 0.0,
            most_unstable_mode,
            0,
        )[None, :]

        eigvals_sol_plot = np.take_along_axis(
            eigvals_sol_plot, selected_mode, axis=0
        )[0]
        eigvals_ref_plot = np.take_along_axis(
            eigvals_ref_plot, selected_mode, axis=0
        )[0]

        # Plotting
        data_to_plot = (eigvals_sol_plot, eigvals_ref_plot)
        labels = (sim_label, None)
        markers = ("o", "s")

        for data, label, marker in zip(data_to_plot, labels, markers):

            axs[0].plot(
                ky,
                data.imag,
                label=label,
                color=sim_color,
                linestyle="none",
                marker=marker,
                markerfacecolor="none",
            )
            
            axs[1].plot(
                ky,
                data.real,
                label=label,
                color=sim_color,
                linestyle="none",
                marker=marker,
                markerfacecolor="none",
            )

    # Setting plot options
    axs[0].set_ylabel(r"$\mathrm{Im}(\omega)$")
    axs[1].set_ylabel(r"$\mathrm{Re}(\omega)$")
    axs[1].set_xlabel(r"$k_y$")

    for ax in axs:
        ax.axhline(y=0.0, color="gray", linewidth=1.0)
        
    axs[1].legend()

    # Save figures if required
    post.save(fig, name=figure_name, suffix="png", save_kwargs={"dpi": 300})

    plt.show()

    return


if __name__ == "__main__":
    # Setup parser
    parser = argparse.ArgumentParser(
        parents=[FlucsPostProcessing.parser()],
        description=(
            "Plots the linear eigensystem from `output.eigensystem.nc`."
        ),
    )

    args = parser.parse_args()

    # Initialise post-processing object
    post = FlucsPostProcessing(
        io_paths=args.io_path,
        save_directory=args.save_directory,
        output_files=["output.eigensystem.nc"],
        constraint="both",
    )

    # Call function
    plot_eigensystem(post)