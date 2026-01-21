# flucs
flucs is a framework for solving simple system of fluid PDEs.

## Installation

The following dependencies must be installed prior to installing flucs:

- Python version 3.10 or higher.
- [Cuda toolkit](https://developer.nvidia.com/cuda-downloads) version 11 or
  higher.

flucs is currently not available on PyPI, and so must be installed
from the source code. This will install both the Python library
and the `flucs` command line tool.

To begin, clone the GitHub repository and enter the directory:

```console
$ git clone https://github.com/flucs-code/flucs
$ cd flucs
```

It is recommended to install flucs to a fresh virtual environment:

```console
$ python -m venv venv
$ source venv/bin/activate
```

flucs may then be installed using `pip`:

```console
$ pip install -e .[cuda13]
```

Including `[cuda13]` will install [`cupy`](https://cupy.dev/) for CUDA v13.
This may instead be replaced with `[cuda12]` or `[cuda11]` for those using
older CUDA toolkits. It is strongly recommended to include one of these to get
the most out of the library.

When you are finished, the virtual environment can be deactivated using:

```console
$ deactivate
```

For users of `uv`, the recommended steps are:

```console
$ uv venv
$ source .venv/bin/activate
$ uv sync --extra cuda13  # Or cuda12, or cuda11
$ deactivate  # When done
```
