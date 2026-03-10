import sys
import traceback
from contextlib import AbstractContextManager
from io import TextIOBase


class FlucsLogHandler(TextIOBase, AbstractContextManager):
    """Redirects stdout and stderr to a specified set of streams.
    Used primarily to redirect print() statements to an additional log file.
    """

    streams: list
    """Streams where data is written."""

    _old_stdout: TextIOBase
    """Backup of the old stdout to restore at __exit__."""
    _old_stderr: TextIOBase
    """Backup of the old stderr to restore at __exit__."""

    # Hard-coded UTF-8 encoding, strict errors, and NOT a TTY
    encoding = "utf-8"
    errors = "strict"

    def isatty(self):
        """Hard-coded to NOT be a tty."""
        return False

    def __init__(self, *streams, keep_stdout=True):
        self.streams = list(streams)

        if keep_stdout and sys.stdout not in self.streams:
            # Append to stdout in addition to the streams specified above
            self.streams.append(sys.stdout)

        # Save the old streams, which we replace later on
        self._old_stdout = sys.stdout
        self._old_stderr = sys.stderr

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

        return len(data)

    def flush(self):
        for s in self.streams:
            s.flush()

    # Context-manager stuff
    def __enter__(self):
        sys.stdout = self
        sys.stderr = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            # Print the traceback while still redirected
            # Print only to non-stdout
            for s in self.streams:
                if s is not self._old_stdout:
                    traceback.print_exception(exc_type, exc_val, exc_tb, file=s)

        sys.stdout = self._old_stdout
        sys.stderr = self._old_stderr
