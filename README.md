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

Including `[cuda13]` in the above command will install
[`cupy`](https://cupy.dev/) and
[`nvmath-python`](https://docs.nvidia.com/cuda/nvmath-python/) for CUDA version
13. The latter provides the low-level cuFFT bindings used by the default FLUCS
FFT-plan wrapper, which allows multiple plans to share a caller-managed work
area. Users of CUDA version 12 should instead install with `[cuda12]`.

The `[cuda11]` extra installs the corresponding CuPy package, but does not
currently include `nvmath-python`. For the complete dependency set used by the
default native FLUCS FFT-plan wrapper, use the `[cuda12]` or `[cuda13]` extra.
A CUDA installation is required in order to access most of the functionality
of the library.

When you are finished, the virtual environment can be deactivated using:

```console
$ deactivate
```

For users of `uv`, the equivalent recommended steps are:

```console
$ uv venv
$ source .venv/bin/activate
$ uv sync --extra cuda13    # Or cuda12
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

To lint a specific file:

```console
$ ruff check <path to file> [--fix]
```

The `--fix` flag is optional, and will automatically correct many issues. Note that 
both the `format` and `check` commands will also apply recursively if run on a directory 
(such as `flucs/src`). 

[Pytest](https://docs.pytest.org/en/stable/) is used to test the project. By
default, the test suite checks for a usable CUDA device. It runs the standard
CPU and GPU tests when one is available and otherwise runs only the standard
CPU tests:

```console
$ pytest
```

Every test is marked explicitly as either `cpu` or `gpu`. The corresponding
flags select only that class; they are mutually exclusive, and explicit GPU
selection fails clearly if CUDA is unavailable:

```console
$ pytest --cpu
$ pytest --gpu
```

Slow-running tests are marked separately and excluded from standard runs. The
`--slow` flag includes them without changing the selected device or ownership
classes. For example, this runs all GPU tests, including the timestepper
convergence test:

```console
$ pytest --gpu --slow
```

Tests for shared FLUCS functionality are marked as `core`, while tests owned by
a particular solver are associated with that solver's entry-point name. These
groups can be selected using:

```console
$ pytest --core
$ pytest --solvers FourierSolver
$ pytest --solvers all
```

The `--solvers` option accepts one or more available solver names. Selecting
solvers also runs the applicable core tests. Add `--cpu` or `--gpu` to select a
device class and `--slow` to include slow-running tests. Core tests that require
a system are exercised against each selected solver's standalone test system;
without a solver selection, every available test system is used.

Individual files or directories can still be supplied using the usual Pytest
syntax, for example:

```console
$ pytest tests/utilities
$ pytest tests/test_flucs.py --gpu
$ pytest tests/solvers/fourier --gpu --slow
```

For a concise list of the FLUCS-specific options and the currently available
test solvers, run:

```console
$ pytest --flucs-help
```
