# flucs
$\texttt{flucs}$ is a general GPU-native framework for solving systems of partial 
differential equations. This base $\texttt{flucs}$ repository contains 
the `solvers` that can be used to evolve the available `systems` that are housed 
in separate repositories within the [flucs-code](https://github.com/flucs-code)
organisation. These repositories must be installed separately; see the instructions 
below.

## Installation

The following dependencies must be installed prior to installing $\texttt{flucs}$:

- Python (version 3.10, or higher)
- [CUDA toolkit](https://developer.nvidia.com/cuda-downloads) (version 11, or
  higher)

$\texttt{flucs}$ is currently not available on PyPI, and so must be installed
from the source code. This will install both the Python library
and the `flucs` command line tool.

To begin, clone the GitHub repository and enter the directory:

```console
$ git clone https://github.com/flucs-code/flucs
$ cd flucs
```

It is recommended to install $\texttt{flucs}$ to a fresh virtual environment:

```console
$ python -m venv venv
$ source venv/bin/activate
```

$\texttt{flucs}$ may then be installed using `pip`:

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
$ uv sync --extra cuda13  # Or cuda12, or cuda11
$ deactivate  # When done
```

## Installation for `systems`

After installing $\texttt{flucs}$ as described above, `systems` can be installed as 
plugins in the same virtual environment.

For example, if using `pip`:

```console
$ git clone https://github.com/flucs-code/flucs_fluid_itg
$ cd flucs_fluid_itg
$ pip install -e .
```
After installing a plugin, verify it was registered correctly:

```console
$ flucs --list
```

This will display all installed `solvers` and `systems`. 
