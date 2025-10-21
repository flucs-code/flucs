import pathlib as pl

def clean_directory(path: pl.Path, patterns: tuple[str, ...]) -> None:
    """
    Cleans up files matching given patterns in the specified directory.
    """

    # Ensure typing
    path = pl.Path(path).resolve()
    abort_msg = "Cleaning aborted."

    # Collect unique files matching patterns
    candidates = sorted(
        {p for pat in patterns for p in path.glob(pat) if p.is_file()},
        key=lambda p: p.name,
    )
    format_text = (
        " or ".join(patterns) if len(patterns) <= 2
        else ", ".join(patterns[:-1]) + f", or {patterns[-1]}"
    )
    if not candidates:
        print(f"No {format_text} files found.")
        return

    # Print candidates and confirm
    print("Candidates for deletion:")
    for p in candidates:
        print(f" - {p.name}")

    try:
        resp = input("Confirm with <ENTER>")
    except (KeyboardInterrupt, EOFError):
        print("\n" + abort_msg)
        return

    if resp == "":
        for p in candidates:
            try:
                p.unlink(missing_ok=True)
            except Exception as e:
                print(f"Could not delete {p.name}: {e}")
        print("Cleanup complete.")
    else:
        print(abort_msg)