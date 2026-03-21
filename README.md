# flucs
$\texttt{FLUCS}$ is a general GPU-native framework for solving systems of partial 
differential equations. This base $\texttt{FLUCS}$ repository contains 
the `solvers` that can be used to evolve the available `systems` that are housed 
in separate repositories within the [flucs-code](https://github.com/flucs-code)
organisation. These repositories must be installed separately; see the instructions 
below.

## Installation

The following dependencies must be installed prior to installing $\texttt{flucs}$:

- Python (version 3.10, or higher)
- [CUDA toolkit](https://developer.nvidia.com/cuda-downloads) (version 11, or
  higher)

$\texttt{FLUCS}$ is currently not available on PyPI, and so must be installed
from the source code. This will install both the Python library
and the `flucs` command line tool.

To begin, clone the GitHub repository and enter the directory:

```console
$ git clone https://github.com/flucs-code/flucs
$ cd flucs
```

It is recommended to install $\texttt{FLUCS}$ to a fresh virtual environment:

```console
$ python -m venv venv
$ source venv/bin/activate
```

$\texttt{FLUCS}$ may then be installed using `pip`:

```console
$ pip install -e .[cuda13]
```

Including `[cuda13]` in the above command will install [`cupy`](https://cupy.dev/) 
for CUDA version 13. This may instead be replaced with `[cuda12]` or `[cuda11]` for 
those using older CUDA toolkits. A CUDA installation is required in order to access
most of the functionality of the library.

When you are finished, the virtual environment can be deactivated using:

```console
$ deactivate
```

For users of `uv`, the equivalent recommended steps are:

```console
$ uv venv
$ source .venv/bin/activate
$ uv sync --extra cuda13    # Or cuda12, cuda11
$ deactivate
```

## Installation for `systems`

After installing $\texttt{FLUCS}$ as described above, `systems` can be installed as 
plugins in the same virtual environment.

For example, if using `pip`:

```console
$ git clone https://github.com/flucs-code/flucs_fluid_itg
$ cd flucs_fluid_itg
$ pip install -e .
```

If using `uv`, the steps are identical except that `uv pip install` should be
used instead.

After installing a plugin, verify it was registered correctly:

```console
$ flucs --list
```

This will display all installed `solvers` and `systems`. 

## Developer tools

To install all developer tools, the project should be installed using the `dev`
dependency group:

```console
$ pip install --upgrade pip            # You may need a later version of pip
$ pip install -e .[cuda13] --group dev
```

The `dev` group will be installed automatically when installing with `uv sync`.

The project is formatted and linted using [Ruff](https://docs.astral.sh/ruff/). 
It is recommended that developers make use of these tools, as any pull-requests 
failing these checks will be blocked from merging.

To format a specific file:

```console
$ ruff format <path to file>
```

To lint the a specific file:

```console
$ ruff check <path to file> [--fix]
```

The `--fix` flag is optional, and will automatically correct many issues. Note that 
both the `format` and `check` commands will also apply recursively if run on a directory 
(such as `flucs/src`). 
