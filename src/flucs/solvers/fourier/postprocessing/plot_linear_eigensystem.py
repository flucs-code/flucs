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

    # Extract number of modes
    system_type = post.system_types[post.io_paths[0]]
    n_modes = system_type.number_of_fields

    # Initialise plotting
    fig, axs = plt.subplots(
        3, n_modes, 
        layout="constrained", 
        sharex=True, sharey="row",
        figsize=(6 * n_modes, 8)
    )

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
        eigvals_run = post.load_netcdf_variable_complex(
            nc_path, "linear_eigensystem/eigvals"
        )[0]
        eigvals_tol = post.load_netcdf_variable(
            nc_path, "linear_eigensystem/eigvals_tolerance"
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
                where=(rel_norm != 0.0)
            )
        else:
            abs_diff = eigvals_ref[0]
            rel_diff = eigvals_ref[0]

        print(
            f"{np.max(rel_diff):>9.3e}, {np.max(abs_diff):>9.3e} "
            f"({sim_label})"
        )

        # Get data at final time and user-specified indices
        it = -1
        ikz, ikx = args.indices

        if kz.size == 1: # Default to ikz=0 if the system is 2D
            ikz = 0

        eigvals_sol_plot = eigvals_sol[it, :, ikz, ikx, :]
        eigvals_ref_plot = eigvals_ref[it, :, ikz, ikx, :]
        eigvals_run_plot = eigvals_run[it, :, ikz, ikx, :]
        eigvals_tol_plot = eigvals_tol[it, :, ikz, ikx, :]

        # Plotting
        data_to_plot = (eigvals_sol_plot, eigvals_ref_plot, eigvals_run_plot)
        markers = ("o", "s", "x")

        for mode in range(n_modes):
            for data, marker in zip(data_to_plot, markers):

                axs[0, mode].plot(
                    ky,
                    data[mode, :].imag,
                    label=None,
                    color=sim_color,
                    linestyle="none",
                    marker=marker,
                    markerfacecolor="none",
                )
                
                axs[1, mode].plot(
                    ky,
                    data[mode, :].real,
                    label=None,
                    color=sim_color,
                    linestyle="none",
                    marker=marker,
                    markerfacecolor="none",
                )

            axs[2, mode].plot(
                ky,
                eigvals_tol_plot[mode, :],
                label=sim_label,
                color=sim_color,
                linestyle="none",
                marker=markers[-1],
                markerfacecolor="none",
            )
        
    # Setting plot options
    axs[0, 0].set_ylabel(r"$\mathrm{Im}(\omega)$")
    axs[1, 0].set_ylabel(r"$\mathrm{Re}(\omega)$")
    axs[2, 0].set_ylabel(r"$\mathrm{Tolerance}$")

    for i in [0, 1]:
        for j in range(n_modes):
            axs[i, j].axhline(y=0.0, color="gray", linewidth=1.0)
        axs[-1, i].set_xlabel(r"$k_y$")
        axs[-1, i].set_yscale("log")

    axs[-1, 1].legend()

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

    parser.add_argument(
        "--indices", "-idx",
        nargs=2,
        type=int,
        default=(1, 0),
        help=(
            "Indices (ikz, ikx) to plot. Default is (1, 0)"
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