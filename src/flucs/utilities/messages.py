import textwrap

FLUCS_LINE_WIDTH = 100
HORIZONTAL_SEPARATOR = 64 * "*"


def flucsprint(*parts, source=None, message_type=None):
    """
    Standard print function for FLUCS.
    """

    # Determine source name for prefix
    if source is None:
        source_prefix = ""
    else:
        if isinstance(source, str):
            source_name = source
        elif hasattr(source, "__name__"):
            source_name = source.__name__
        else:
            source_name = source.__class__.__name__

        source_prefix = f"[{source_name}]"

    # Determine message type for prefix
    if message_type is not None:
        if message_type not in ["info", "warning", "error"]:
            raise ValueError(
                f"Invalid message_type '{message_type}'. "
                "Must be one of 'info', 'warning', or 'error'."
            )
        type_prefix = f"{message_type.upper()}:"
    else:
        type_prefix = ""

    # Construct prefix and message
    prefix = " ".join(part for part in [source_prefix, type_prefix] if part)
    message = " ".join(str(part) for part in parts)

    # Combine with prefix
    message = " ".join(part for part in [prefix, message] if part)


    message = "\n".join(textwrap.fill(
        line,
        width=FLUCS_LINE_WIDTH,
        break_long_words=False,
        break_on_hyphens=False,
    ) for line in message.split("\n"))

    # Warnings get extra spaces for extra attention
    if message_type == "warning":
        message = f"\n{message}\n"

    # Print
    print(message)

def format_seconds(seconds: float, verbose: bool = False) -> str:
    """
    Formats a duration in seconds to be human readable
    """
    total_seconds = int(seconds)
    days, remainder = divmod(total_seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)


    if verbose:
        parts = {
            "days": days,
            "hours": hours,
            "minutes": minutes,
            "seconds": seconds,
        } 

        # Handles singular and plural
        def format_part_value(part, value):
            if value > 1:
                return f"{value} {part}"
            else:
                return f"{value} {part[:-1]}"

        return ", ".join(
            format_part_value(part, value)
            for part, value in parts.items()
            if part == "seconds" or value > 0
        )

    return f"{days:02d}:{hours:02d}:{minutes:02d}:{seconds:02d}"

