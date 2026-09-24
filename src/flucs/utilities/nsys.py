import csv
import pathlib as pl


def format_nsys_gpu_kernel_summary(filepath: pl.Path) -> str:
    """
    Formats a summary of GPU kernel execution from NVIDIA Nsight Systems
    .csv output.

    Parameters
    ----------
    filepath : pl.Path
        Path to the NVIDIA Nsight Systems .csv output.

    Returns
    -------
    str
        Formatted GPU kernel execution summary.

    """
    # Columns to display in the summary
    columns = [
        "Time (%)",
        "Total Time (us)",
        "Avg (us)",
        "Max (us)",
        "Std (us)",
        "Name",
    ]

    # Read in data from csv file and convert times from ns to us
    ns_to_us = 1e-3
    with open(filepath, newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        rows = []
        for row in reader:
            rows.append(
                {
                    "Time (%)": row["Time (%)"],
                    "Total Time (us)": (
                        f"{float(row['Total Time (ns)']) * ns_to_us:.3f}"
                    ),
                    "Avg (us)": f"{float(row['Avg (ns)']) * ns_to_us:.3f}",
                    "Max (us)": f"{float(row['Max (ns)']) * ns_to_us:.3f}",
                    "Std (us)": f"{float(row['StdDev (ns)']) * ns_to_us:.3f}",
                    "Name": row["Name"],
                }
            )

    # Determine the maximum width of each column
    widths = {
        column: max(
            len(column),
            *(len(row[column]) for row in rows),
        )
        for column in columns
    }

    # Construct summary lines
    lines = [
        "GPU kernel execution summary (Nsight Systems)",
        "",
        "  ".join(f"{column:<{widths[column]}}" for column in columns),
        "  ".join("-" * widths[column] for column in columns),
    ]

    for row in rows:
        lines.append(
            "  ".join(
                f"{row[column]:>{widths[column]}}"
                if column != "Name"
                else f"{row[column]:<{widths[column]}}"
                for column in columns
            )
        )

    return "\n".join(lines)
