import sys
from io import TextIOBase
from contextlib import AbstractContextManager


class FlucsLogHandler(TextIOBase, AbstractContextManager):
    """ Redirects stdout and stderr to a specified set of streams.
    Used primarily to redirect print() statements to an additional log file.

    Attributes
    ----------
    streams : Streams where data is written.

    """
    # List of streams to which we write data
    streams: list

    # Backup of the old stdout and stderr to restore at __exit__
    _old_stdout: TextIOBase
    _old_stderr: TextIOBase

    # Hard-coded UTF-8 encoding, strict errors, and NOT a TTY
    encoding = "utf-8"
    errors = "strict"

    def isatty(self):
        """Hard-coded to NOT be a tty. """
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
        sys.stdout = self._old_stdout
        sys.stderr = self._old_stderr
