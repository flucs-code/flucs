"""
Dynamically plot variables from output.time.txt.
"""

from __future__ import annotations

import argparse
import pathlib as pl

import matplotlib.pyplot as plt
import numpy as np

from flucs.postprocessing import FlucsPostProcessing
from flucs.utilities.messages import flucsprint

# Global settings
POLL_INTERVAL_SECONDS = 1.0
SOURCE = "dynamic_time_plots"
INDENT = 3 * " "


def _is_separator(line: str) -> bool:
    """
    Return whether a line separates two text-output groups.
    """
    stripped = line.strip()
    return bool(stripped) and set(stripped) == {"-"}


class TimeOutputReader:
    """
    Read the latest complete group from output.time.txt.
    """

    def __init__(self, io_path: pl.Path) -> None:
        # Setup paths
        self.io_path = pl.Path(io_path)
        self.path = self.io_path / "output.time.txt"
        self.columns: tuple[str, ...] = ()
        self.values: dict[str, list[complex]] = {}
        self.complex_columns: set[str] = set()
        self.generation = 0

        # The signature avoids rereading an unchanged file. The group key
        # identifies the latest appended output group within that file.
        self._signature: tuple[int, int, int, int] | None = None
        self._group_key: tuple[int, int, int] | None = None
        self._malformed_lines: set[int] = set()

    def poll(self) -> bool:
        """
        Reload the latest group if the output file has changed.
        """
        try:
            path_stat = self.path.stat()
        except FileNotFoundError:
            self._signature = None
            self._group_key = None
            self._malformed_lines.clear()
            return False

        signature = (
            path_stat.st_dev,
            path_stat.st_ino,
            path_stat.st_size,
            path_stat.st_mtime_ns,
        )
        if signature == self._signature:
            return False

        # A run can replace its output file or truncate it in place. Treat a
        # truncation as a new generation even if its header is unchanged.
        file_was_truncated = (
            self._signature is not None
            and signature[:2] == self._signature[:2]
            and signature[2] < self._signature[2]
        )
        self._signature = signature

        try:
            contents = self.path.read_bytes()
        except FileNotFoundError:
            self._signature = None
            self._group_key = None
            self._malformed_lines.clear()
            return False

        # Ignore a final row while it is still being written.
        if not contents.endswith(b"\n"):
            if b"\n" not in contents:
                return False
            contents = contents.rsplit(b"\n", 1)[0]

        lines = contents.decode("utf-8").splitlines()

        # FlucsOutputText appends a separator and a fresh header for each run.
        # Only the newest group should be shown in the live plots.
        group_start = 0
        for index, line in enumerate(lines):
            if _is_separator(line):
                group_start = index + 1

        group_lines = [line for line in lines[group_start:] if line.strip()]
        if not group_lines:
            return False

        columns = tuple(group_lines[0].split())
        if "time" not in columns:
            raise ValueError(
                f"Invalid header in {self.path}: expected a 'time' column, "
                f"found {list(columns)}."
            )
        if len(columns) != len(set(columns)):
            raise ValueError(
                f"Invalid header in {self.path}: column names must be unique."
            )

        group_key = (path_stat.st_dev, path_stat.st_ino, group_start)
        if file_was_truncated or group_key != self._group_key:
            # Missing-variable and malformed-row warnings are reported once
            # per output group.
            self.generation += 1
            self._group_key = group_key
            self._malformed_lines.clear()

        values = {column: [] for column in columns}
        complex_columns = set()

        # Store every scalar as complex so real and complex diagnostic columns
        # can follow the same parsing and plotting path.
        for line in group_lines[1:]:
            tokens = line.split()
            if len(tokens) != len(columns):
                line_hash = hash(line)
                if line_hash not in self._malformed_lines:
                    flucsprint(
                        f"Skipping malformed row in {self.path}: expected "
                        f"{len(columns)} values, found {len(tokens)}.",
                        source=SOURCE,
                        message_type="warning",
                    )
                    self._malformed_lines.add(line_hash)
                continue

            try:
                row = [complex(token) for token in tokens]
            except ValueError:
                line_hash = hash(line)
                if line_hash not in self._malformed_lines:
                    flucsprint(
                        f"Skipping non-numeric row in {self.path}: {line!r}.",
                        source=SOURCE,
                        message_type="warning",
                    )
                    self._malformed_lines.add(line_hash)
                continue

            for column, token, value in zip(columns, tokens, row, strict=True):
                values[column].append(value)
                if "j" in token.lower():
                    complex_columns.add(column)

        self.columns = columns
        self.values = values
        self.complex_columns = complex_columns
        return True


def list_variables(readers: list[TimeOutputReader]) -> None:
    """
    List the headings in each available output.time.txt file.
    """
    flucsprint("Available output.time.txt headings:", source=SOURCE)

    for reader in readers:
        reader.poll()
        if not reader.columns:
            flucsprint(
                f"No complete output group found in {reader.path}.",
                source=SOURCE,
                message_type="warning",
            )
        flucsprint(
            f"{INDENT}{reader.io_path}: {list(reader.columns)}",
            source=SOURCE,
        )


def print_webagg_connection_instructions() -> None:
    """
    Start the WebAgg listener and report how to reach it over SSH.

    Initialising the application here makes Matplotlib select and bind its
    port before plt.show() starts the blocking server event loop. 
    """
    from matplotlib.backends.backend_webagg import WebAggApplication

    WebAggApplication.initialize()
    port = WebAggApplication.port
    address = WebAggApplication.address

    # A wildcard listening address is not a valid SSH forwarding target.
    # The loopback interface still reaches a server listening on all interfaces.
    if address in {"0.0.0.0", "::"}:
        address = "127.0.0.1"
    forward_address = f"[{address}]" if ":" in address else address

    flucsprint(
        "If using SSH to a remote machine, first run this in a local "
        "terminal (replacing USER and REMOTE_HOST):\n"
        f"{INDENT}ssh -N -L "
        f"{port}:{forward_address}:{port} USER@REMOTE_HOST\n",
    )


def dynamic_time_plots(
    readers: list[TimeOutputReader], variables: list[str]
) -> None:
    """
    Follow selected variables from one or more i/o directories.
    """

    def _poll_readers() -> bool:
        """
        Poll every directory and report whether any data changed.
        """
        changed = False
        for reader in readers:
            changed = reader.poll() or changed
            if not reader.columns and reader.path not in waiting_paths:
                flucsprint(
                    f"Waiting for {reader.path} ...",
                    source=SOURCE,
                )
                waiting_paths.add(reader.path)
        return changed

    def _update_plots() -> None:
        """
        Copy the current reader data into the existing plot lines.
        """
        for reader in readers:
            if not reader.columns:
                continue

            time_values = np.asarray(reader.values["time"]).real

            for variable in variables:
                real_line, imaginary_line = lines[(reader.io_path, variable)]

                if variable not in reader.columns:
                    warning_key = (reader.path, reader.generation, variable)
                    if warning_key not in reported_missing:
                        flucsprint(
                            f"Variable '{variable}' not found in "
                            f"{reader.path}.",
                            source=SOURCE,
                            message_type="warning",
                        )
                        reported_missing.add(warning_key)

                    real_line.set_data([], [])
                    real_line.set_label("_nolegend_")
                    imaginary_line.set_data([], [])
                    imaginary_line.set_label("_nolegend_")
                    continue

                values = np.asarray(reader.values[variable])
                label = labels[reader.io_path]
                real_line.set_data(time_values, values.real)

                # Complex columns use matched solid/dashed lines. The
                # imaginary line stays out of the legend for real columns.
                if variable in reader.complex_columns:
                    real_line.set_label(f"{label} (real)")
                    imaginary_line.set_data(time_values, values.imag)
                    imaginary_line.set_label(f"{label} (imaginary)")
                else:
                    real_line.set_label(label)
                    imaginary_line.set_data([], [])
                    imaginary_line.set_label("_nolegend_")

        # Recalculate limits only after every directory has been updated.
        for ax in axs_dict.values():
            ax.relim()
            ax.autoscale_view()

            # Only add a legend to the final plot
            if ax == list(axs_dict.values())[-1]:
                handles, legend_labels = ax.get_legend_handles_labels()
                if handles:
                    ax.legend(handles, legend_labels)
                elif ax.get_legend() is not None:
                    ax.get_legend().remove()

    def _refresh_plots() -> bool:
        """
        Update the figure from a GUI-backend timer callback.
        """
        if not plt.fignum_exists(figure.number):
            return False

        if _poll_readers():
            _update_plots()

        # WebAgg sends draw requests over a browser WebSocket. The request made
        # when a new output file first appears can precede that connection and
        # is then lost. Requesting a redraw on every inexpensive timer tick
        # lets a newly connected or reconnected browser catch up with the
        # latest artist data without rereading an unchanged output file.
        figure.canvas.draw_idle()

        # Matplotlib removes a timer callback when it returns False. Returning
        # True keeps this callback active while the figure remains open.
        return True

    # Initialise overall figure and axes
    figure, axs_array = plt.subplots(
        len(variables),
        1,
        sharex=len(variables) > 1,
        squeeze=False,
        layout="constrained",
    )
    figure.canvas.manager.set_window_title("FLUCS dynamic time plots")

    axs_dict = dict(zip(variables, axs_array[:, 0], strict=True))

    # Directory basenames keep legends compact. Fall back to full paths when
    # two requested directories have the same basename.
    directory_names = [reader.io_path.name for reader in readers]
    labels = {
        reader.io_path: (
            reader.io_path.name
            if directory_names.count(reader.io_path.name) == 1
            else str(reader.io_path)
        )
        for reader in readers
    }

    # Create one panel per variable and one pair of lines per directory.
    colors = plt.cm.rainbow(np.linspace(0, 1, len(readers)))
    lines = {}
    for variable, ax in axs_dict.items():
        ax.set_ylabel(variable)
        ax.grid(True)

        for reader, color in zip(readers, colors, strict=True):
            (real_line,) = ax.plot([], [], color=color, label="_nolegend_")
            (imaginary_line,) = ax.plot(
                [],
                [],
                color=color,
                linestyle="--",
                label="_nolegend_",
            )
            lines[(reader.io_path, variable)] = (
                real_line,
                imaginary_line,
            )

    axs_array[-1, 0].set_xlabel("time")

    # Avoid repeating waiting and missing-variable messages on every poll.
    waiting_paths = set()
    reported_missing: set[tuple[pl.Path, int, str]] = set()

    # A canvas timer schedules polling inside the active event loop and works
    # for both WebAgg and desktop GUI backends.
    timer = figure.canvas.new_timer(
        interval=round(POLL_INTERVAL_SECONDS * 1000)
    )
    timer.add_callback(_refresh_plots)

    # Start the event loop
    try:
        if _poll_readers():
            _update_plots()

        timer.start()
        if plt.get_backend().lower() == "webagg":
            print_webagg_connection_instructions()
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        timer.stop()
        plt.close(figure)


if __name__ == "__main__":
    # Setup parser
    parser = argparse.ArgumentParser(
        parents=[FlucsPostProcessing.parser()],
        description=(
            "Dynamically plot variables from one or more output.time.txt files."
        ),
    )

    parser.add_argument(
        "--list",
        "-l",
        action="store_true",
        default=False,
        help="List the available output.time.txt headings and exit.",
    )

    parser.add_argument(
        "--variable",
        "-v",
        nargs="+",
        default=None,
        help=(
            "Variables to plot, with one panel per variable. Defaults to dt."
        ),
    )

    parser.add_argument(
        "--webagg",
        "-w",
        action="store_true",
        default=False,
        help="Use Matplotlib's WebAgg backend for the live plots.",
    )

    args = parser.parse_args()

    # Change to MPLBACKEND=WebAgg before generating plots
    if args.webagg:
        plt.switch_backend("WebAgg")
        plt.rcParams["webagg.open_in_browser"] = False

    # Initialise postprocessing object
    post = FlucsPostProcessing(
        io_paths=args.io_path,
        save_directory=args.save_directory,
        output_files=["output.time.txt"],
        constraint="none",
    )
    output_readers = [TimeOutputReader(path) for path in post.io_paths]

    if args.list:
        list_variables(output_readers)
    else:
        selected_variables = list(dict.fromkeys(args.variable or ["dt"]))
        dynamic_time_plots(output_readers, selected_variables)
