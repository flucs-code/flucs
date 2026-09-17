"""Live plotting for the append-only ``output.time.txt`` file."""

from __future__ import annotations

import os
import pathlib as pl
import time
import warnings
from dataclasses import dataclass, field
from typing import BinaryIO

import matplotlib.pyplot as plt
import numpy as np

from flucs.utilities.messages import flucsprint

POLL_INTERVAL_SECONDS = 1.0
RESERVED_COLUMNS = ("time", "dt", "cfl")
DEFAULT_EXCLUDED_COLUMNS = (*RESERVED_COLUMNS, "step")


def _is_separator(line: str) -> bool:
    """Return whether *line* is a FlucsOutputText group separator."""
    stripped = line.strip()
    return bool(stripped) and set(stripped) == {"-"}


@dataclass
class TextOutputGroup:
    """The header and accumulated rows for one text-output group."""

    columns: tuple[str, ...]
    values: dict[str, list[complex]] = field(init=False)
    complex_columns: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        self.values = {column: [] for column in self.columns}


class TextOutputReader:
    """Incrementally read the latest group from a text-output file."""

    def __init__(self, path: pl.Path) -> None:
        self.path = pl.Path(path)
        self.group: TextOutputGroup | None = None
        self.generation = 0
        self._file: BinaryIO | None = None
        self._identity: tuple[int, int] | None = None
        self._buffer = b""
        self._expect_header = True

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
        self._file = None
        self._identity = None

    def poll(self) -> bool:
        """Read appended complete lines and report whether data changed."""
        if not self._ensure_open_file():
            return False

        assert self._file is not None
        chunk = self._file.read()
        if not chunk:
            return False

        self._buffer += chunk
        lines = self._buffer.split(b"\n")
        self._buffer = lines.pop()

        changed = False
        for raw_line in lines:
            line = raw_line.rstrip(b"\r").decode("utf-8")
            changed = self._consume_line(line) or changed
        return changed

    def _ensure_open_file(self) -> bool:
        try:
            path_stat = self.path.stat()
        except FileNotFoundError:
            return False

        identity = (path_stat.st_dev, path_stat.st_ino)
        reset = self._file is None or identity != self._identity
        if self._file is not None and path_stat.st_size < self._file.tell():
            reset = True

        if reset:
            self.close()
            try:
                self._file = open(self.path, "rb")
            except FileNotFoundError:
                return False
            file_stat = os.fstat(self._file.fileno())
            self._identity = (file_stat.st_dev, file_stat.st_ino)
            self._buffer = b""
            self._expect_header = True
            self.group = None

        return True

    def _consume_line(self, line: str) -> bool:
        if _is_separator(line):
            self._expect_header = True
            return False

        if not line.strip():
            return False

        if self._expect_header:
            self._start_group(line)
            return True

        assert self.group is not None
        tokens = line.split()
        if len(tokens) != len(self.group.columns):
            warnings.warn(
                "Skipping malformed row in "
                f"{self.path}: expected {len(self.group.columns)} values, "
                f"found {len(tokens)}.",
                RuntimeWarning,
                stacklevel=2,
            )
            return False

        try:
            row = [complex(token) for token in tokens]
        except ValueError:
            warnings.warn(
                f"Skipping non-numeric row in {self.path}: {line!r}.",
                RuntimeWarning,
                stacklevel=2,
            )
            return False

        time_index = self.group.columns.index("time")
        if row[time_index].imag != 0:
            warnings.warn(
                f"Skipping row with complex time in {self.path}: {line!r}.",
                RuntimeWarning,
                stacklevel=2,
            )
            return False

        for column, token, value in zip(
            self.group.columns, tokens, row, strict=True
        ):
            self.group.values[column].append(value)
            if "j" in token.lower():
                self.group.complex_columns.add(column)
        return True

    def _start_group(self, line: str) -> None:
        columns = tuple(line.split())
        if not columns or "time" not in columns:
            raise ValueError(
                f"Invalid header in {self.path}: expected a 'time' column, "
                f"found {list(columns)}."
            )
        if len(columns) != len(set(columns)):
            raise ValueError(
                f"Invalid header in {self.path}: column names must be unique."
            )

        self.group = TextOutputGroup(columns)
        self.generation += 1
        self._expect_header = False


class FollowPlotter:
    """Own and update the figures for a followed text-output group."""

    def __init__(
        self,
        requested_columns: list[str] | None,
        *,
        plot_dt_cfl: bool,
        one_panel: bool,
    ) -> None:
        self.requested_columns = requested_columns
        self.plot_dt_cfl = plot_dt_cfl
        self.one_panel = one_panel
        self.generation = -1
        self.figures: list[plt.Figure] = []
        self.lines: dict[str, tuple[plt.Line2D, plt.Line2D | None]] = {}
        self.axes: dict[str, plt.Axes] = {}

    def update(self, group: TextOutputGroup, generation: int) -> None:
        if generation != self.generation:
            self._configure(group, generation)

        time_values = np.asarray(group.values["time"]).real
        for column, (real_line, imaginary_line) in list(self.lines.items()):
            values = np.asarray(group.values[column])
            real_line.set_data(time_values, values.real)

            if column in group.complex_columns and imaginary_line is None:
                imaginary_line = self._add_imaginary_line(column, real_line)
                self.lines[column] = (real_line, imaginary_line)
            if imaginary_line is not None:
                imaginary_line.set_data(time_values, values.imag)

        for axis in set(self.axes.values()):
            axis.relim()
            axis.autoscale_view()

        for figure in self.figures:
            if plt.fignum_exists(figure.number):
                figure.canvas.draw_idle()

    def has_open_figures(self) -> bool:
        return any(plt.fignum_exists(figure.number) for figure in self.figures)

    def _configure(self, group: TextOutputGroup, generation: int) -> None:
        for figure in self.figures:
            plt.close(figure)
        self.figures.clear()
        self.lines.clear()
        self.axes.clear()

        columns = self._select_columns(group.columns)
        if self.one_panel:
            self._create_column_figure(columns, 1, "FLUCS follow")
        else:
            for first in range(0, len(columns), 4):
                batch = columns[first : first + 4]
                number = first // 4 + 1
                title = f"FLUCS follow {number}"
                self._create_column_figure(batch, len(batch), title)

        if self.plot_dt_cfl:
            self._create_dt_cfl_figure(group.columns)

        self.generation = generation
        plt.show(block=False)

    def _select_columns(self, available: tuple[str, ...]) -> list[str]:
        if self.requested_columns:
            if len(self.requested_columns) != len(set(self.requested_columns)):
                raise ValueError("Requested column names must be unique.")
            reserved = [
                column
                for column in self.requested_columns
                if column in RESERVED_COLUMNS
            ]
            if reserved:
                names = ", ".join(reserved)
                raise ValueError(
                    f"Cannot select reserved column(s): {names}. 'time' is "
                    "always the x-axis; use --dt-cfl for 'dt' and 'cfl'."
                )
            missing = [
                column
                for column in self.requested_columns
                if column not in available
            ]
            if missing:
                raise ValueError(
                    f"Column(s) not found: {', '.join(missing)}. Available "
                    f"columns: {', '.join(available)}."
                )
            return list(self.requested_columns)

        return [
            column
            for column in available
            if column not in DEFAULT_EXCLUDED_COLUMNS
        ]

    def _create_column_figure(
        self, columns: list[str], panel_count: int, title: str
    ) -> None:
        if not columns:
            return
        figure, axes_array = plt.subplots(
            panel_count,
            1,
            sharex=panel_count > 1,
            squeeze=False,
            layout="constrained",
        )
        axes = axes_array[:, 0]
        figure.canvas.manager.set_window_title(title)
        self.figures.append(figure)

        for index, column in enumerate(columns):
            axis = axes[0] if self.one_panel else axes[index]
            label = column if self.one_panel else None
            (line,) = axis.plot([], [], label=label)
            self.lines[column] = (line, None)
            self.axes[column] = axis
            if not self.one_panel:
                axis.set_ylabel(column)
            axis.grid(True)

        axes[-1].set_xlabel("time")
        if self.one_panel:
            axes[0].legend()

    def _create_dt_cfl_figure(self, available: tuple[str, ...]) -> None:
        missing = [
            column for column in ("dt", "cfl") if column not in available
        ]
        if missing:
            raise ValueError(
                "Cannot use --dt-cfl because the latest output group is "
                f"missing: {', '.join(missing)}."
            )

        figure, axes_array = plt.subplots(
            2,
            1,
            sharex=True,
            squeeze=False,
            layout="constrained",
        )
        axes = axes_array[:, 0]
        figure.canvas.manager.set_window_title("FLUCS follow: dt and cfl")
        self.figures.append(figure)
        for axis, column in zip(axes, ("dt", "cfl"), strict=True):
            (line,) = axis.plot([], [])
            axis.set_ylabel(column)
            axis.grid(True)
            self.lines[column] = (line, None)
            self.axes[column] = axis
        axes[-1].set_xlabel("time")

    def _add_imaginary_line(
        self, column: str, real_line: plt.Line2D
    ) -> plt.Line2D:
        axis = self.axes[column]
        real_line.set_label(f"{column} (real)" if self.one_panel else "real")
        label = f"{column} (imaginary)" if self.one_panel else "imaginary"
        (imaginary_line,) = axis.plot([], [], label=label)
        axis.legend()
        return imaginary_line


def follow(
    io_path: pl.Path,
    columns: list[str] | None = None,
    *,
    plot_dt_cfl: bool = False,
    one_panel: bool = False,
) -> None:
    """Follow and plot the latest group in ``output.time.txt``."""
    output_path = pl.Path(io_path) / "output.time.txt"
    reader = TextOutputReader(output_path)
    plotter = FollowPlotter(
        columns,
        plot_dt_cfl=plot_dt_cfl,
        one_panel=one_panel,
    )

    waiting_message_printed = False
    try:
        while reader.group is None:
            reader.poll()
            if reader.group is None:
                if not waiting_message_printed:
                    flucsprint(f"Waiting for {output_path} ...")
                    waiting_message_printed = True
                time.sleep(POLL_INTERVAL_SECONDS)

        plotter.update(reader.group, reader.generation)
        while plotter.has_open_figures():
            plt.pause(POLL_INTERVAL_SECONDS)
            reader.poll()
            if reader.group is not None:
                plotter.update(reader.group, reader.generation)
    except KeyboardInterrupt:
        pass
    finally:
        reader.close()
        plt.close("all")
