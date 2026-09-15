"""
Tests for shared FLUCS post-processing support.
"""

import matplotlib.pyplot as plt
import numpy as np
import numpy.testing as npt
import pytest
import toml
from netCDF4 import Dataset

import flucs
from flucs.postprocessing import FlucsPostProcessing
from tests.support.support import write_test_input

pytestmark = pytest.mark.core


def _write_output(nc_path, test_system, precision):
    """
    Write two output groups with one deliberately absent variable.
    """

    with Dataset(nc_path, "w", format="NETCDF4") as dataset:
        for group_number, times in ((0, [0.0, 0.5]), (1, [1.0])):
            group = dataset.createGroup(str(group_number))
            group.createDimension("time", None)
            group.createVariable(
                "time",
                precision.netcdf_precision,
                ("time",),
            )[:] = times
            group.createVariable(
                "dt",
                precision.netcdf_precision,
                ("time",),
            )[:] = 0.5

            input_file = group.createVariable("input_file", str)
            input_data = test_system.create_input_data()
            input_data["run"] = {"number": group_number}
            input_file[...] = toml.dumps(input_data)

            diagnostic = group.createGroup("diagnostic")
            grid = diagnostic.createGroup("grid")
            grid.createDimension("position", 2)
            grid.createDimension("component", 3)
            grid.createVariable(
                "position",
                precision.netcdf_precision,
                ("position",),
            )[:] = [-1.0, 1.0]
            grid.createVariable(
                "component",
                precision.netcdf_precision,
                ("component",),
            )[:] = np.arange(3)

            # The real variable is absent from the second run on purpose
            if group_number == 0:
                grid.createVariable(
                    "value",
                    precision.netcdf_precision,
                    ("time", "position", "component"),
                )[:] = np.arange(1.0, 13.0).reshape(2, 2, 3)

            real_values = np.arange(6.0).reshape(1, 2, 3) + group_number
            if group_number == 0:
                real_values = np.concatenate(
                    [real_values, real_values + 6.0],
                    axis=0,
                )
            complex_values = real_values + 1j * (real_values + 20.0)
            grid.createVariable(
                "state_real",
                precision.netcdf_precision,
                ("time", "position", "component"),
            )[:] = complex_values.real
            grid.createVariable(
                "state_imag",
                precision.netcdf_precision,
                ("time", "position", "component"),
            )[:] = complex_values.imag


def test_postprocessing_discovers_and_loads_netcdf_data(
    test_system,
    tmp_path,
    precision,
):
    """
    Post-processing resolves plugins and combines numbered output groups.
    """

    # Build one ordinary i/o directory and use overlapping output patterns
    io_path = tmp_path / "run"
    io_path.mkdir()
    write_test_input(io_path / "input.toml", test_system)

    nc_path = io_path / "output.data.nc"
    _write_output(nc_path, test_system, precision)

    post = FlucsPostProcessing(
        io_path,
        output_files=("output.*.nc", "*.nc"),
        quiet=True,
    )

    # Construction resolves the registered solver/system and deduplicates files
    resolved_io_path = io_path.resolve()

    assert post.io_paths == [resolved_io_path]
    assert post._output_paths == {resolved_io_path: [nc_path.resolve()]}
    assert post.solver_types[resolved_io_path] is flucs.get_solver_type(
        test_system.solver_name
    )
    assert post.system_types[resolved_io_path] is test_system.system_type

    # Recursive discovery records the numbered groups containing each variable
    variables = post.get_netcdf_variables(
        nc_path,
        ignore=(
            "time",
            "dt",
            "input_file",
            "diagnostic/grid/position",
            "diagnostic/grid/component",
        ),
    )
    assert variables == {
        "diagnostic/grid/value": [0],
        "diagnostic/grid/state_real": [0, 1],
        "diagnostic/grid/state_imag": [0, 1],
    }
    assert post.get_valid_netcdf_paths("diagnostic/grid/state_real") == [
        nc_path.resolve()
    ]

    # Missing group segments are filled while run boundaries remain visible
    values, boundaries, dimensions = post.load_netcdf_variable(
        nc_path,
        "diagnostic/grid/value",
        fill_value=-1.0,
    )
    npt.assert_allclose(
        values,
        np.concatenate(
            [
                np.arange(1.0, 13.0).reshape(2, 2, 3),
                -np.ones((1, 2, 3)),
            ]
        ),
        rtol=precision.tolerance,
        atol=precision.tolerance,
    )
    assert values.dtype == np.dtype(precision.float_type)
    assert boundaries == [2]
    npt.assert_allclose(
        dimensions[0]["position"],
        [-1.0, 1.0],
        rtol=precision.tolerance,
        atol=precision.tolerance,
    )
    assert dimensions[0]["position"].dtype == np.dtype(precision.float_type)
    npt.assert_array_equal(dimensions[0]["component"], np.arange(3))
    assert dimensions[0]["component"].dtype == np.dtype(precision.float_type)
    assert dimensions[1] == {}

    # The complex wrapper and group selectors share the same loading rules
    complex_values, complex_boundaries, _ = post.load_netcdf_variable_complex(
        nc_path,
        "diagnostic/grid/state",
    )
    expected_real = np.concatenate(
        [
            np.arange(12.0).reshape(2, 2, 3),
            np.arange(1.0, 7.0).reshape(1, 2, 3),
        ]
    )
    npt.assert_allclose(
        complex_values,
        expected_real + 1j * (expected_real + 20.0),
        rtol=precision.tolerance,
        atol=precision.tolerance,
    )
    assert complex_values.dtype == np.dtype(precision.complex_type)
    assert complex_boundaries == [2]

    latest, latest_boundaries, _ = post.load_netcdf_variable(
        nc_path,
        "diagnostic/grid/state_real",
        groups=-1,
    )
    npt.assert_allclose(
        latest,
        np.arange(1.0, 7.0).reshape(1, 2, 3),
        rtol=precision.tolerance,
        atol=precision.tolerance,
    )
    assert latest.dtype == np.dtype(precision.float_type)
    assert latest_boundaries == []

    # Stored inputs are returned in the same selected group order
    input_files = post.load_netcdf_input_files(nc_path)
    assert [input_file["run"]["number"] for input_file in input_files] == [0, 1]

    with pytest.raises(ValueError, match="Variable 'missing' not found"):
        post.load_netcdf_variable(nc_path, "missing")


def test_postprocessing_saves_figures_and_parses_common_arguments(
    tmp_path,
):
    """
    Figure saving honours conflicts and the common parser resolves paths.
    """

    # Saving only needs a destination; plugin discovery is exercised above
    io_path = tmp_path / "run"
    io_path.mkdir()

    save_directory = tmp_path / "figures"

    post = object.__new__(FlucsPostProcessing)
    post.save_directory = save_directory

    # Axes are promoted to their figure and the optional close flag is consumed
    figure, axes = plt.subplots()
    post.save(
        axes,
        name="result",
        suffix=".png",
        save_kwargs={"close": True, "dpi": 40},
    )
    save_path = save_directory / "result.png"

    assert save_path.is_file()
    assert not plt.fignum_exists(figure.number)

    # Existing results can be preserved or treated as an explicit error
    original_contents = save_path.read_bytes()
    second_figure = plt.figure()
    try:
        post.save(
            second_figure,
            name="result",
            suffix="png",
            conflict_strategy="preserve",
        )
        assert save_path.read_bytes() == original_contents

        with pytest.raises(OSError, match="Target save path already exists"):
            post.save(
                second_figure,
                name="result",
                suffix="png",
                conflict_strategy="error",
            )
    finally:
        plt.close(second_figure)

    # The shared parser supplies paths and group identifiers to all scripts
    arguments = post.parser().parse_args(
        [
            "--io_path",
            str(io_path),
            "--save_directory",
            str(save_directory),
            "--groups",
            "0",
            "2",
        ]
    )
    assert arguments.io_path == [str(io_path)]
    assert arguments.save_directory == save_directory.resolve()
    assert arguments.groups == ["0", "2"]

    with pytest.raises(
        ValueError,
        match="Invalid value for 'conflict_strategy'",
    ):
        post.save(
            object(),
            name="invalid",
            suffix="png",
            conflict_strategy="invalid",
        )
