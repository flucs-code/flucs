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
    prefix = " ".join(
        part for part in [source_prefix, type_prefix] if part
    )
    message = " ".join(str(part) for part in parts)

    # Print
    print(" ".join(part for part in [prefix, message] if part))
