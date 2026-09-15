"""
CPU tests for FourierSystem configuration and state management.
"""

from types import SimpleNamespace

import numpy as np
import numpy.testing as npt
import pytest
import toml

import flucs.solvers.fourier.fourier_system as fourier_system_module
from flucs.input import InvalidFlucsInputFileError
from tests.support.support import create_test_solver_system

pytestmark = pytest.mark.solver("FourierSolver")


@pytest.mark.parametrize(
    ("updates", "expected_method", "expected_truncation", "expected_memory"),
    [
        pytest.param(
            {"dimensions": {"nz": 24, "nx": 24, "ny": 24}},
            "two-thirds",
            "rectangular",
            "standard",
            id="two-thirds-padded",
        ),
        pytest.param(
            {
                "dimensions": {"nz": -1, "nx": -1, "ny": -1},
                "dealiasing": {
                    "nz_unpadded": 15,
                    "nx_unpadded": 15,
                    "ny_unpadded": 15,
                },
            },
            "two-thirds",
            "rectangular",
            "standard",
            id="two-thirds-unpadded",
        ),
        *[
            pytest.param(
                {
                    "dimensions": {"nz": 18, "nx": 18, "ny": 18},
                    "dealiasing": {
                        "method": "phase-shift",
                        "truncation": truncation,
                        "memory": memory,
                    },
                },
                "phase-shift",
                truncation,
                memory,
                id=f"phase-shift-{truncation}-{memory}",
            )
            for truncation in ("spherical", "polyhedral")
            for memory in ("standard", "low_memory", "in_place")
        ],
    ],
)
def test_dealiasing_configuration(
    test_system,
    tmp_path,
    updates,
    expected_method,
    expected_truncation,
    expected_memory,
):
    """
    Every supported dealiasing configuration resolves its effective grid.
    """

    # Construct each option through the same input path used by a real run
    _, _, system = create_test_solver_system(
        tmp_path,
        test_system,
        updates=updates,
    )

    assert system.input["dealiasing.method"] == expected_method
    assert system.dealiasing_truncation == expected_truncation
    assert system.input["dealiasing.memory"] == expected_memory

    # All three Fourier directions obey the same padding convention
    for dimension in "zxy":
        n = getattr(system, f"n{dimension}")
        half_n = getattr(system, f"half_n{dimension}")
        n_unpadded = getattr(system, f"n{dimension}_unpadded")
        half_n_unpadded = getattr(system, f"half_n{dimension}_unpadded")

        assert half_n == n // 2 + 1
        assert n_unpadded == 2 * half_n_unpadded - 1
        assert n_unpadded % 2 == 1
        assert n_unpadded <= n

    definitions = system.module_options._defs

    if expected_method == "two-thirds":
        assert definitions["TWO_THIRDS_DEALIASING"] == ""
        assert (system.nz, system.nx, system.ny) == (24, 24, 24)
        assert (
            system.nz_unpadded,
            system.nx_unpadded,
            system.ny_unpadded,
        ) == (15, 15, 15)
    else:
        assert definitions["PHASE_SHIFT_DEALIASING"] == ""
        truncation_flag = f"PHASE_SHIFT_{expected_truncation.upper()}"
        assert definitions[truncation_flag] == ""
        constant = {
            "spherical": "DEALIASING_RADIUS_SQUARED",
            "polyhedral": "DEALIASING_MAX_SUM",
        }[expected_truncation]
        assert constant in definitions


@pytest.mark.parametrize(
    ("updates", "error_type", "message"),
    [
        pytest.param(
            {"dimensions": {"Lx": 0.0}},
            InvalidFlucsInputFileError,
            "must be positive",
            id="nonpositive-box",
        ),
        pytest.param(
            {"time": {"dt_method": "missing"}},
            InvalidFlucsInputFileError,
            "Invalid time.dt_method",
            id="unknown-dt-method",
        ),
        pytest.param(
            {
                "time": {"dt_method": "continuous"},
                "timestepping": {"precompute_linear_matrix": True},
            },
            InvalidFlucsInputFileError,
            "Cannot have timestepping.precompute_linear_matrix",
            id="continuous-precomputed-matrix",
        ),
        pytest.param(
            {"dealiasing": {"method": "missing"}},
            InvalidFlucsInputFileError,
            "Invalid dealiasing.method",
            id="unknown-dealiasing-method",
        ),
        pytest.param(
            {"dealiasing": {"truncation": "spherical"}},
            InvalidFlucsInputFileError,
            "Invalid dealiasing.truncation",
            id="two-thirds-truncation",
        ),
        pytest.param(
            {"dealiasing": {"memory": "low_memory"}},
            InvalidFlucsInputFileError,
            "Invalid dealiasing.memory",
            id="two-thirds-memory",
        ),
        pytest.param(
            {"dealiasing": {"nx_unpadded": 6}},
            ValueError,
            "Unpadded resolutions must be odd",
            id="even-unpadded-grid",
        ),
        pytest.param(
            {
                "dimensions": {"nx": -1},
                "dealiasing": {"method": "phase-shift"},
            },
            InvalidFlucsInputFileError,
            "Phase-shifted dimension nx must be specified",
            id="missing-phase-shift-grid",
        ),
        pytest.param(
            {"forcing": {"method": "missing"}},
            InvalidFlucsInputFileError,
            "Invalid forcing.method",
            id="unknown-forcing-method",
        ),
    ],
)
def test_fourier_configuration_rejects_invalid_options(
    test_system,
    tmp_path,
    updates,
    error_type,
    message,
):
    """
    Related Fourier input errors fail during construction with useful messages.
    """

    with pytest.raises(error_type, match=message):
        create_test_solver_system(
            tmp_path,
            test_system,
            updates=updates,
        )


def test_fourier_geometry_shells_and_solved_modes(
    test_system,
    tmp_path,
    precision,
):
    """
    Fourier coordinates, diagnostic shells, and solved modes remain consistent.
    """

    # Unequal box lengths make accidental axis swaps immediately visible
    lengths = {"Lz": 3.0, "Lx": 4.0, "Ly": 5.0}
    _, _, system = create_test_solver_system(
        tmp_path,
        test_system,
        precision=precision,
        updates={
            "dimensions": {
                "nz": 12,
                "nx": 18,
                "ny": 24,
                **lengths,
            }
        },
    )

    expected_kz = 2 * np.pi * np.fft.fftfreq(system.nz) * system.nz / 3.0
    expected_kx = 2 * np.pi * np.fft.fftfreq(system.nx) * system.nx / 4.0
    expected_ky = 2 * np.pi * np.fft.rfftfreq(system.ny) * system.ny / 5.0

    npt.assert_allclose(system.kz, expected_kz, atol=precision.tolerance)
    npt.assert_allclose(system.kx, expected_kx, atol=precision.tolerance)
    npt.assert_allclose(system.ky, expected_ky, atol=precision.tolerance)

    # Diagnostic shell grids cover even the largest retained diagonal
    system._compute_kperp_shells()
    system._compute_kmod_shells()

    assert system.shell_kperp.dtype == np.dtype(precision.float_type)
    assert system.shell_kmod.dtype == np.dtype(precision.float_type)

    assert np.all(np.diff(system.shell_kperp) > 0)
    assert np.all(np.diff(system.shell_kmod) > 0)

    kx_max = abs(system.kx[system.half_nx - 1])
    ky_max = abs(system.ky[system.half_ny - 1])
    kz_max = abs(system.kz[system.half_nz - 1])

    assert system.shell_kperp_max > np.hypot(kx_max, ky_max)
    assert system.shell_kmod_max > np.sqrt(kz_max**2 + kx_max**2 + ky_max**2)

    # Reconstruct the rectangular solved region without invoking CUDA
    solved_z = np.ones(system.nz, dtype=bool)
    solved_x = np.ones(system.nx, dtype=bool)
    solved_y = np.arange(system.half_ny) < system.half_ny_unpadded

    solved_z[
        system.half_nz_unpadded : system.half_nz_unpadded
        + system.nz
        - system.nz_unpadded
    ] = False
    solved_x[
        system.half_nx_unpadded : system.half_nx_unpadded
        + system.nx
        - system.nx_unpadded
    ] = False
    solved_mask = (
        solved_z[:, np.newaxis, np.newaxis]
        & solved_x[np.newaxis, :, np.newaxis]
        & solved_y[np.newaxis, np.newaxis, :]
    )
    system.solved_grid_mask = solved_mask.astype(precision.float_type)

    # Public helpers must report coordinates and physical mode counts together
    kz, kx, ky = system.get_broadcast_wavenumbers()
    solved_kz, solved_kx, solved_ky = system.get_solved_wavenumbers()

    npt.assert_array_equal(solved_kz, kz[solved_mask])
    npt.assert_array_equal(solved_kx, kx[solved_mask])
    npt.assert_array_equal(solved_ky, ky[solved_mask])

    nonnegative_count = int(np.count_nonzero(solved_mask))
    zero_ky_count = int(np.count_nonzero(solved_mask[:, :, 0]))
    assert system.get_number_of_solved_modes() == nonnegative_count
    assert system.get_number_of_solved_modes(False) == (
        2 * nonnegative_count - zero_ky_count
    )


@pytest.mark.parametrize(
    ("current_dt", "cfl_rate", "expected_dt"),
    [
        pytest.param(0.05, 10.0, 0.02, id="cfl-limited"),
        pytest.param(0.05, 1.0, 0.055, id="increase-limited"),
        pytest.param(0.10, 1.0, 0.10, id="maximum-limited"),
    ],
)
def test_continuous_timestep_control(
    test_system,
    tmp_path,
    precision,
    current_dt,
    cfl_rate,
    expected_dt,
):
    """
    Continuous timestep control chooses the strictest active limit.
    """

    _, _, system = create_test_solver_system(
        tmp_path,
        test_system,
        precision=precision,
    )
    system.current_dt = system.float(current_dt)
    system.cfl_rate_float = system.float(cfl_rate)
    system.max_cfl = system.float(0.2)
    system.dt_max = system.float(0.1)
    system.dt_mult_increase = system.float(1.1)

    assert system._compute_current_dt_continuous() is True
    assert system.current_dt == pytest.approx(
        expected_dt,
        rel=precision.tolerance,
        abs=precision.tolerance,
    )


def test_discrete_timestep_control(test_system, tmp_path, precision):
    """
    Discrete timestep control reduces, increases, or retains one timestep.
    """

    _, _, system = create_test_solver_system(
        tmp_path,
        test_system,
        precision=precision,
    )
    system.current_time = system.float(0.0)
    system.max_cfl = system.float(0.2)
    system.dt_max = system.float(0.1)
    system.dt_mult_increase = system.float(1.1)
    system.dt_mult_decrease = system.float(0.75)
    system.dt_mult_steps = system.int(2)

    # A CFL violation immediately reduces the timestep and resets the counter
    system.current_dt = system.float(0.05)
    system.cfl_rate_float = system.float(10.0)
    system.sub_cfl_steps = system.int(2)

    assert system._compute_current_dt_discrete() is True
    assert system.current_dt == pytest.approx(
        0.015,
        rel=precision.tolerance,
        abs=precision.tolerance,
    )
    assert system.sub_cfl_steps == 0

    # Enough quiet steps permit one bounded increase
    system.current_dt = system.float(0.05)
    system.cfl_rate_float = system.float(1.0)
    system.sub_cfl_steps = system.int(2)

    assert system._compute_current_dt_discrete() is True
    assert system.current_dt == pytest.approx(
        0.055,
        rel=precision.tolerance,
        abs=precision.tolerance,
    )
    assert system.sub_cfl_steps == 0

    # Reaching dt_max suppresses an otherwise permitted increase
    system.current_dt = system.float(0.1)
    system.sub_cfl_steps = system.int(2)

    assert system._compute_current_dt_discrete() is False
    assert system.current_dt == pytest.approx(
        0.1,
        rel=precision.tolerance,
        abs=precision.tolerance,
    )
    assert system.sub_cfl_steps == 2

    # Otherwise the timestep is retained and only the counter advances
    system.current_dt = system.float(0.05)
    system.sub_cfl_steps = system.int(0)

    assert system._compute_current_dt_discrete() is False
    assert system.current_dt == pytest.approx(
        0.05,
        rel=precision.tolerance,
        abs=precision.tolerance,
    )
    assert system.sub_cfl_steps == 1


def test_timestep_update_interrupts_below_minimum(
    test_system,
    tmp_path,
    precision,
    monkeypatch,
):
    """
    The complete timestep update records CFL state and stops below dt_min.
    """

    _, solver, system = create_test_solver_system(
        tmp_path,
        test_system,
        precision=precision,
    )

    # A tiny NumPy stand-in keeps this control-flow test entirely CPU-only
    monkeypatch.setattr(
        fourier_system_module,
        "cp",
        SimpleNamespace(asnumpy=np.asarray),
    )
    system.cfl_rate = np.asarray([100.0], dtype=precision.float_type)
    system.max_cfl = system.float(0.2)

    system.current_time = system.float(0.0)

    system.current_dt = system.float(0.01)
    system.dt_min = system.float(0.005)
    system.dt_max = system.float(0.1)
    system.dt_mult_increase = system.float(1.1)

    system._compute_current_dt = system._compute_current_dt_continuous
    solver.interrupted = False

    assert system._update_dt() is True
    assert system.current_dt == pytest.approx(
        0.002,
        rel=precision.tolerance,
        abs=precision.tolerance,
    )
    assert system.current_cfl == pytest.approx(
        0.2,
        rel=precision.tolerance,
        abs=precision.tolerance,
    )
    assert system.adaptive_rate == pytest.approx(
        500.0,
        rel=precision.tolerance,
        abs=precision.tolerance,
    )
    assert solver.interrupted is True


def _restart_data(system, fields):
    """
    Package Fourier fields with the box metadata required by restart loading.
    """
    dimensions = {
        f"L{dimension}": system.input[f"dimensions.L{dimension}"]
        for dimension in "zxy"
    }
    return {
        "fields": {"data": fields},
        "input_file": {
            "data": np.asarray(toml.dumps({"dimensions": dimensions}))
        },
    }


def _assert_restart_modes_match(source_fields, result):
    """
    Check restart data by its integer Fourier coordinates.
    """

    def fft_modes(size):
        return np.rint(np.fft.fftfreq(size) * size).astype(int)

    _, source_nz, source_nx, source_half_ny = source_fields.shape
    _, target_nz, target_nx, target_half_ny = result.shape

    # Label source coefficients by physical mode rather than storage index
    source_by_mode = {
        (int(kz), int(kx), ky): source_fields[:, iz, ix, ky]
        for iz, kz in enumerate(fft_modes(source_nz))
        for ix, kx in enumerate(fft_modes(source_nx))
        for ky in range(source_half_ny)
    }
    zero = np.zeros(source_fields.shape[0], dtype=source_fields.dtype)

    # A target coefficient is either the same physical mode or newly zeroed
    for iz, kz in enumerate(fft_modes(target_nz)):
        for ix, kx in enumerate(fft_modes(target_nx)):
            for ky in range(target_half_ny):
                expected = source_by_mode.get((int(kz), int(kx), ky), zero)
                npt.assert_array_equal(result[:, iz, ix, ky], expected)


@pytest.mark.parametrize(
    "source_shape",
    [
        pytest.param((6, 8, 6), id="unchanged"),
        pytest.param((4, 6, 4), id="refined"),
        pytest.param((8, 10, 7), id="coarsened"),
    ],
)
def test_restart_grid_remapping(
    test_system,
    tmp_path,
    precision,
    source_shape,
):
    """
    Restart loading preserves common modes across unchanged or resized grids.
    """

    _, _, system = create_test_solver_system(
        tmp_path,
        test_system,
        precision=precision,
    )

    # Seeded random values expose any axis, sign, or field-index mix-up
    source_size = system.number_of_fields * np.prod(source_shape)
    random = np.random.default_rng(4821)
    
    source_fields = (
        random.standard_normal(source_size)
        + 1j * random.standard_normal(source_size)
    ).astype(precision.complex_type)
    source_fields = source_fields.reshape(
        system.number_of_fields,
        *source_shape,
    )
    system.restart_manager = SimpleNamespace(
        data=_restart_data(system, source_fields)
    )

    result = system.prepare_restart_data()
    target_shape = (system.number_of_fields, *system.half_tuple)

    assert result.shape == target_shape
    assert result.dtype == np.dtype(precision.complex_type)
    _assert_restart_modes_match(source_fields, result)


@pytest.mark.parametrize(
    ("restart_data", "message"),
    [
        pytest.param({}, "does not contain 'fields'", id="missing-fields"),
        pytest.param(
            {"fields": {"data": np.zeros((1, 1, 1, 1))}},
            "does not contain an input file",
            id="missing-input",
        ),
    ],
)
def test_restart_grid_rejects_missing_data(
    test_system,
    tmp_path,
    restart_data,
    message,
):
    """
    Restart preparation rejects incomplete payloads before indexing fields.
    """

    _, _, system = create_test_solver_system(tmp_path, test_system)
    system.restart_manager = SimpleNamespace(data=restart_data)

    with pytest.raises(InvalidFlucsInputFileError, match=message):
        system.prepare_restart_data()


def test_restart_grid_rejects_incompatible_field_count(
    test_system,
    tmp_path,
):
    """
    Restart remapping refuses data for a different number of fields.
    """

    _, _, system = create_test_solver_system(tmp_path, test_system)
    fields = np.zeros(
        (system.number_of_fields + 1, 4, 6, 4),
        dtype=system.complex,
    )
    system.restart_manager = SimpleNamespace(data=_restart_data(system, fields))

    with pytest.raises(
        InvalidFlucsInputFileError,
        match="but the current system requires",
    ):
        system.prepare_restart_data()
