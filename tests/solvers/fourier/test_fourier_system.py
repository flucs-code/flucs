"""
Tests for FourierSystem configuration, state management, and CUDA lifecycle.
"""

from types import SimpleNamespace

import numpy as np
import numpy.testing as npt
import pytest
import toml

import flucs.solvers.fourier.fourier_system as fourier_system_module
from flucs import cupy as cp
from flucs.input import InvalidFlucsInputFileError
from flucs.utilities.dealiasing import dealiased_multiplication_rfft
from tests.support.support import (
    TEST_PRECISIONS,
    TEST_SYSTEMS,
    create_test_solver_system,
)

pytestmark = pytest.mark.solver("FourierSolver")


@pytest.fixture(
    scope="module",
    params=TEST_PRECISIONS,
    ids=lambda precision: precision.name,
)
def compiled_fourier_system(request, tmp_path_factory):
    """
    Compile one minimal TestFourierSystem for each supported precision.
    """

    precision = request.param
    io_path = tmp_path_factory.mktemp(f"fourier-system-{precision.name}")

    # This shared system needs the linear kernels but no nonlinear workspace
    _, solver, system = create_test_solver_system(
        io_path,
        TEST_SYSTEMS["FourierSolver"],
        precision=precision,
        updates={
            "setup": {
                "linear": True,
                "check_linear_matrix": True,
            },
            "forcing": {"method": ""},
        },
    )

    # Follow the production setup order up to the point needed by these tests
    system.setup()
    solver.timestepper.setup()
    system.compile_cupy_module()
    system.setup_initial_conditions()

    try:
        yield system
    finally:
        system.kernels.unbind()


@pytest.fixture
def ready_fourier_system(compiled_fourier_system):
    """
    Reset mutable runtime state around the shared compiled system.
    """

    system = compiled_fourier_system

    # No field history or derived CPU data may leak between the GPU tests
    for fields in system.fields:
        fields.fill(0)
    system.linear_matrix = None
    system.linear_eigensystem = None
    system.linear_propagator = None
    system.realspace_fields = None
    system.solver.interrupted = False

    # Use the same reset hooks as the two production solver passes
    system.ready()
    system.solver.timestepper.ready()

    return system


def _dealiasing_product(
    io_path,
    test_system,
    precision,
    grid_size,
    dealiasing_updates,
    setup_updates=None,
):
    """
    Compare one compiled Fourier operation with the padded reference product.
    """

    # Compile the real nonlinear operation selected by this input
    nz, nx, ny = grid_size
    _, solver, system = create_test_solver_system(
        io_path,
        test_system,
        precision=precision,
        updates={
            "dimensions": {
                "nz": nz,
                "nx": nx,
                "ny": ny,
            },
            "setup": {
                "check_linear_matrix": False,
                **(setup_updates or {}),
            },
            "forcing": {"method": ""},
            "dealiasing": {
                "check_errors": True,
                **dealiasing_updates,
            },
        },
    )
    system.setup()
    solver.timestepper.setup()
    system.compile_cupy_module()

    try:
        # Every configuration receives the same fixed-seed physical fields
        random = np.random.default_rng(7412)
        fields = random.standard_normal((2, *system.full_tuple)).astype(
            system.float
        )
        fields_fourier = cp.fft.rfftn(
            cp.asarray(fields),
            axes=(-3, -2, -1),
            norm="forward",
        )

        solved_mask = system.get_solved_grid_mask().astype(bool)
        solved_mask_gpu = cp.asarray(solved_mask)
        fields_fourier[:, ~solved_mask_gpu] = 0

        # Exercise the configured two-thirds or phase-shift implementation
        system.check_dealiasing_errors_operation(
            current_dt=0,
            current_time=0,
            current_step=0,
            input_array=fields_fourier,
            calculate_cfl=False,
        )
        product = system.check_dealiasing_errors_output[0].copy()
        product[~solved_mask_gpu] = 0

        # Twice-padding supplies an independent alias-free reference
        reference = dealiased_multiplication_rfft(
            fields_fourier[0],
            fields_fourier[1],
            nx=system.nx,
            ny=system.ny,
            nz=system.nz,
            padded_nx=2 * system.nx,
            padded_ny=2 * system.ny,
            padded_nz=2 * system.nz,
        )
        reference[~solved_mask_gpu] = 0

        error = float(cp.max(cp.abs(product - reference)))
        return (
            error,
            cp.asnumpy(product),
            cp.asnumpy(reference),
            system,
        )
    finally:
        system.kernels.unbind()


###############################################################################
# CPU tests
###############################################################################


@pytest.mark.cpu
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

    if expected_method == "two-thirds":
        assert (system.nz, system.nx, system.ny) == (24, 24, 24)
        assert (
            system.nz_unpadded,
            system.nx_unpadded,
            system.ny_unpadded,
        ) == (15, 15, 15)
    else:
        assert (system.nz, system.nx, system.ny) == (18, 18, 18)


@pytest.mark.cpu
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
            {"setup": {"fft_wrapper": "missing"}},
            InvalidFlucsInputFileError,
            "'missing' is not a valid cuFFT wrapper",
            id="unknown-fft-wrapper",
        ),
        pytest.param(
            {
                "setup": {"fft_wrapper": "cupy"},
                "dealiasing": {
                    "method": "phase-shift",
                    "memory": "in_place",
                },
            },
            InvalidFlucsInputFileError,
            "Cannot use cupy fft_wrapper with in_place memory setup",
            id="cupy-in-place",
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


@pytest.mark.cpu
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

    expected_kz = (
        2
        * np.pi
        * np.fft.fftfreq(system.nz)
        * system.nz
        / system.input["dimensions.Lz"]
    )
    expected_kx = (
        2
        * np.pi
        * np.fft.fftfreq(system.nx)
        * system.nx
        / system.input["dimensions.Lx"]
    )
    expected_ky = (
        2
        * np.pi
        * np.fft.rfftfreq(system.ny)
        * system.ny
        / system.input["dimensions.Ly"]
    )

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

    # Prescribe unrelated solved positions so the helpers are tested without
    # reconstructing any production dealiasing region
    solved_mask = np.zeros(system.half_tuple, dtype=bool)
    solved_indices = (
        (+0, +0, +0),
        (+1, +2, +0),
        (-1, -2, +0),
        (+2, +3, +1),
        (-3, +4, +2),
    )
    for index in solved_indices:
        solved_mask[index] = True
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


@pytest.mark.cpu
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


@pytest.mark.cpu
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


@pytest.mark.cpu
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


@pytest.mark.cpu
@pytest.mark.parametrize(
    "grid_change",
    [
        pytest.param("unchanged", id="unchanged"),
        pytest.param("refined", id="refined"),
        pytest.param("coarsened", id="coarsened"),
    ],
)
def test_restart_grid_remapping(
    test_system,
    tmp_path,
    precision,
    grid_change,
):
    """
    Restart loading preserves common modes across unchanged or resized grids.
    """

    _, _, system = create_test_solver_system(
        tmp_path,
        test_system,
        precision=precision,
    )

    # Vary full-grid axes independently to expose swaps and odd/even mistakes
    grid_changes = {
        "unchanged": (0, 0, 0),
        "refined": (-1, -2, -3),
        "coarsened": (3, 2, 1),
    }[grid_change]
    source_nz, source_nx, source_ny = (
        size + change
        for size, change in zip(
            (system.nz, system.nx, system.ny),
            grid_changes,
            strict=True,
        )
    )
    source_shape = (
        source_nz,
        source_nx,
        source_ny // 2 + 1,
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


@pytest.mark.cpu
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


@pytest.mark.cpu
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


@pytest.mark.cpu
def test_initial_conditions_are_projected_onto_fourier_grid(
    test_system,
    tmp_path,
    precision,
    monkeypatch,
):
    """
    Initial data is reshaped, truncated, and made Fourier-real.
    """

    _, _, system = create_test_solver_system(
        tmp_path,
        test_system,
        precision=precision,
        updates={"forcing": {"method": ""}},
    )

    # Prescribe a sparse conjugate-symmetric mask without reconstructing any
    # production dealiasing region
    solved_mask = np.zeros(system.half_tuple, dtype=bool)
    solved_indices = (
        (+0, +0, +0),
        (+1, +2, +0),
        (-1, -2, +0),
        (+2, +3, +1),
        (-3, +4, +2),
    )
    for index in solved_indices:
        solved_mask[index] = True
    system.solved_grid_mask = solved_mask.astype(system.float)
    system.restart_manager = SimpleNamespace(data=None)

    # Provide random initial data so the check is independent of the method
    random = np.random.default_rng(1729)
    generated_fields = (
        random.standard_normal((system.number_of_fields, system.half_size))
        + 1j
        * random.standard_normal((system.number_of_fields, system.half_size))
    ).astype(system.complex)

    monkeypatch.setattr(
        system,
        "_set_initial_conditions",
        lambda: setattr(system, "fields_initial", generated_fields.copy()),
    )

    # Set initial conditions and check the data is well-formed
    system.setup_initial_conditions()
    fields = system.fields_initial

    assert fields.shape == (system.number_of_fields, *system.half_tuple)
    assert fields.dtype == np.dtype(system.complex)
    npt.assert_array_equal(fields[:, ~solved_mask], system.complex(0))

    # Solved positive-ky modes do not need projection and remain untouched
    positive_ky_mask = solved_mask.copy()
    positive_ky_mask[:, :, 0] = False
    generated_fields = generated_fields.reshape(fields.shape)

    npt.assert_array_equal(
        fields[:, positive_ky_mask],
        generated_fields[:, positive_ky_mask],
    )

    # The ky=0 plane is projected onto the Fourier reality condition
    conjugate_iz = (-np.arange(system.nz)) % system.nz
    conjugate_ix = (-np.arange(system.nx)) % system.nx
    fields_ky0 = fields[:, :, :, 0]

    npt.assert_allclose(
        fields_ky0,
        np.conj(
            fields_ky0[
                :,
                conjugate_iz[:, None],
                conjugate_ix[None, :],
            ]
        ),
        rtol=0,
        atol=system.tolerance,
    )


###############################################################################
# GPU tests
###############################################################################


@pytest.mark.gpu
def test_solved_grid_and_initial_conditions(ready_fourier_system):
    """
    CUDA grid selection and initial fields obey the Fourier representation.
    """

    system = ready_fourier_system

    # Treat the production mask as the description of the active Fourier grid
    solved_grid_mask = system.get_solved_grid_mask()
    solved_mask = solved_grid_mask.astype(bool)

    assert solved_grid_mask.shape == system.half_tuple
    assert solved_grid_mask.dtype == np.dtype(system.float)
    assert np.all((solved_grid_mask == 0) | (solved_grid_mask == 1))
    assert np.any(solved_mask)
    assert np.any(~solved_mask)

    # Initial data has the configured layout and no energy in padded modes
    initial_fields = system.fields_initial
    expected_shape = (system.number_of_fields, *system.half_tuple)

    assert initial_fields.shape == expected_shape
    assert initial_fields.dtype == np.dtype(system.complex)
    assert np.all(np.isfinite(initial_fields))
    assert np.any(initial_fields[:, solved_mask] != 0)
    npt.assert_array_equal(
        initial_fields[:, ~solved_mask],
        system.complex(0),
    )

    # The ready hook copies the initial state onto the current device field
    assert isinstance(system.get_fields(), cp.ndarray)
    assert system.get_fields().shape == expected_shape
    assert system.get_fields().dtype == np.dtype(system.complex)

    npt.assert_array_equal(cp.asnumpy(system.get_fields()), initial_fields)
    npt.assert_array_equal(
        cp.asnumpy(system.get_fields(1)),
        np.zeros_like(initial_fields),
    )


@pytest.mark.gpu
def test_linear_matrix_eigensystem_and_propagator(ready_fourier_system):
    """
    CUDA linear quantities agree with references and are released after checks.
    """

    system = ready_fourier_system
    solved_mask = system.get_solved_grid_mask().astype(bool)
    expected_matrix = system.compute_linear_matrix_reference()

    # The CUDA matrix matches the CPU model only where modes are retained
    matrix = system.compute_linear_matrix()
    expected_shape = (
        system.number_of_fields,
        system.number_of_fields,
        *system.half_tuple,
    )

    assert matrix.shape == expected_shape
    assert matrix.dtype == np.dtype(system.complex)
    npt.assert_allclose(
        matrix[..., solved_mask],
        expected_matrix[..., solved_mask],
        rtol=0,
        atol=system.tolerance,
    )
    npt.assert_array_equal(
        matrix[..., ~solved_mask],
        system.complex(0),
    )

    # Eigenfrequencies have an analytical reference for TestFourierSystem
    eigensystem = system.compute_linear_eigensystem()
    eigvals = eigensystem["eigvals"]
    eigvecs = eigensystem["eigvecs"]
    eigvecs_inverse = eigensystem["eigvecs_inverse"]
    expected_eigvals = system.compute_linear_frequencies_reference()

    for quantity in (eigvals, eigvecs, eigvecs_inverse):
        assert quantity.dtype == np.dtype(system.complex)

    npt.assert_allclose(
        np.sort(eigvals[:, solved_mask], axis=0),
        np.sort(expected_eigvals[:, solved_mask], axis=0),
        rtol=0,
        atol=system.tolerance,
    )
    npt.assert_array_equal(eigvals[:, ~solved_mask], system.complex(0))
    npt.assert_array_equal(eigvecs[..., ~solved_mask], system.complex(0))

    # Normalisation, phase choice, and inverse are all part of the public data
    solved_eigvecs = eigvecs[..., solved_mask]
    npt.assert_allclose(
        np.linalg.norm(solved_eigvecs, axis=1),
        1,
        rtol=0,
        atol=system.tolerance,
    )

    largest_indices = np.abs(solved_eigvecs).argmax(axis=1, keepdims=True)
    largest_components = np.take_along_axis(
        solved_eigvecs,
        largest_indices,
        axis=1,
    )
    npt.assert_allclose(
        largest_components.imag,
        0,
        rtol=0,
        atol=system.tolerance,
    )
    assert np.all(largest_components.real >= -system.tolerance)

    vectors = solved_eigvecs.transpose(2, 1, 0)
    inverse = eigvecs_inverse[..., solved_mask].transpose(2, 1, 0)
    identity = np.broadcast_to(
        np.eye(system.number_of_fields, dtype=system.complex),
        (vectors.shape[0], system.number_of_fields, system.number_of_fields),
    )
    npt.assert_allclose(
        inverse @ vectors,
        identity,
        rtol=0,
        atol=system.tolerance,
    )

    # Compare the CUDA Pade result with a NumPy exponential of the CPU matrix
    dt = system.dt_max
    reference_matrices = np.moveaxis(
        expected_matrix[..., solved_mask],
        (0, 1),
        (-2, -1),
    )
    rates, reference_vectors = np.linalg.eig(reference_matrices)
    reference_inverse = np.linalg.inv(reference_vectors)
    exact_propagator = (
        reference_vectors * np.exp(-dt * rates)[:, np.newaxis, :]
    ) @ reference_inverse

    propagator = system.compute_linear_propagator(dt=dt)
    solved_propagator = np.moveaxis(
        propagator[..., solved_mask],
        (0, 1),
        (-2, -1),
    )

    assert propagator.shape == expected_shape
    assert propagator.dtype == np.dtype(system.complex)
    npt.assert_allclose(
        solved_propagator,
        exact_propagator,
        rtol=0,
        atol=system.tolerance,
    )
    npt.assert_array_equal(
        propagator[..., ~solved_mask],
        system.complex(0),
    )

    # The complete health check releases its temporary linear quantities
    system._check_linear_matrix()
    assert system.linear_matrix is None
    assert system.linear_eigensystem is None
    assert system.linear_propagator is None


@pytest.mark.gpu
def test_linear_matrix_health_check_rejects_inconsistent_reference(
    ready_fourier_system,
    monkeypatch,
):
    """
    The linear health check rejects a disagreeing system reference matrix.
    """

    # Create an incorrect reference by adding a constant
    system = ready_fourier_system
    reference = system.compute_linear_matrix_reference().copy()
    reference += system.complex(1.0)

    # Replace the independent TestSystem reference with a known disagreement
    monkeypatch.setattr(
        system,
        "compute_linear_matrix_reference",
        lambda: reference,
    )

    with pytest.raises(
        ValueError,
        match="linear matrix computed by CUDA disagrees",
    ):
        system._check_linear_matrix()


@pytest.mark.gpu
def test_field_history_realspace_and_restart_data(ready_fourier_system):
    """
    Field history feeds real-space reconstruction and restart serialization.
    """

    system = ready_fourier_system

    # Distinct valid states expose both directions of the circular history
    history_values = [
        system.fields_initial * system.complex(index + 1)
        for index in range(system.fields_history_size)
    ]
    for fields, values in zip(system.fields, history_values, strict=True):
        fields.set(values)

    for current_step in range(2 * system.fields_history_size):
        system.current_step = system.int(current_step)
        current_index = current_step % system.fields_history_size
        previous_index = (current_step - 1) % system.fields_history_size

        assert system.get_fields() is system.fields[current_index]
        assert system.get_fields(1) is system.fields[previous_index]

    current_fields = cp.asnumpy(system.get_fields())

    # The CPU and GPU inverse transforms must reconstruct the same real fields
    system.realspace_fields = None
    system.get_realspace_fields_cpu()
    realspace_cpu = system.realspace_fields.copy()
    cached_realspace = system.realspace_fields

    system.get_realspace_fields_gpu()
    assert system.realspace_fields is cached_realspace

    system.begin_time_step()
    assert system.realspace_fields is None

    system.get_realspace_fields_gpu()
    realspace_gpu = system.realspace_fields

    assert realspace_gpu.shape == (system.number_of_fields, *system.full_tuple)
    assert np.all(np.isfinite(realspace_gpu))
    npt.assert_allclose(
        realspace_gpu,
        realspace_cpu,
        rtol=0,
        atol=system.tolerance,
    )

    # Restart data is a host snapshot of the same current Fourier state
    restart_data = system.get_restart_data()["fields"]

    assert restart_data["dimension_names"] == (
        "number_of_fields",
        "nz",
        "nx",
        "half_ny",
    )
    assert restart_data["data"].shape == current_fields.shape
    assert restart_data["data"].dtype == np.dtype(system.complex)
    npt.assert_array_equal(restart_data["data"], current_fields)

    # Later device changes must not mutate the serialized host snapshot
    system.get_fields().fill(0)
    npt.assert_array_equal(restart_data["data"], current_fields)


@pytest.mark.gpu
@pytest.mark.parametrize("fft_wrapper", ("flucs", "cupy"))
def test_fft_wrappers_execute_dealiased_operations(
    test_system,
    tmp_path,
    precision,
    fft_wrapper,
):
    """
    Both FFT wrappers produce the same alias-free Fourier product.
    """

    # Use same initial conditions and dealiasing
    error, product, reference, system = _dealiasing_product(
        tmp_path,
        test_system,
        precision,
        grid_size=(12, 12, 12),
        dealiasing_updates={"method": "two-thirds"},
        setup_updates={"fft_wrapper": fft_wrapper},
    )

    assert system.use_cupy_fft is (fft_wrapper == "cupy")
    assert error <= system.tolerance

    npt.assert_allclose(
        product,
        reference,
        rtol=0,
        atol=system.tolerance,
    )


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize(
    ("n_unpadded", "is_safe"),
    [
        pytest.param(21, True, id="below-boundary"),
        pytest.param(23, True, id="at-boundary"),
        pytest.param(25, False, id="above-boundary"),
    ],
)
def test_two_thirds_dealiasing_boundary(
    test_system,
    tmp_path,
    precision,
    n_unpadded,
    is_safe,
):
    """
    Two-thirds padding succeeds through its maximum retained bandwidth.
    """

    error, _, _, system = _dealiasing_product(
        tmp_path,
        test_system,
        precision,
        grid_size=(36, 36, 36),
        dealiasing_updates={
            "method": "two-thirds",
            "nz_unpadded": n_unpadded,
            "nx_unpadded": n_unpadded,
            "ny_unpadded": n_unpadded,
        },
    )

    # Safe cases are round-off limited; the next bandwidth must visibly alias
    if is_safe:
        assert error <= system.tolerance
    else:
        assert error > system.tolerance


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize(
    ("radius_squared", "is_safe"),
    [
        pytest.param(0.221, True, id="below-boundary"),
        pytest.param(0.222, True, id="at-boundary"),
        pytest.param(0.223, False, id="above-boundary"),
    ],
)
def test_spherical_phase_shift_dealiasing_boundary(
    test_system,
    tmp_path,
    precision,
    radius_squared,
    is_safe,
):
    """
    Spherical phase shifting fails only beyond its safe theoretical shell.
    """

    error, _, _, system = _dealiasing_product(
        tmp_path,
        test_system,
        precision,
        grid_size=(36, 36, 36),
        dealiasing_updates={
            "method": "phase-shift",
            "truncation": "spherical",
            "radius_squared": radius_squared,
        },
    )

    # The adjacent shell admitted above 2/9 produces resolvable aliasing
    if is_safe:
        assert error <= system.tolerance
    else:
        assert error > system.tolerance


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize(
    ("max_sum", "is_safe"),
    [
        pytest.param(0.665, True, id="below-boundary"),
        pytest.param(0.666, True, id="at-boundary"),
        pytest.param(0.667, False, id="above-boundary"),
    ],
)
def test_polyhedral_phase_shift_dealiasing_boundary(
    test_system,
    tmp_path,
    precision,
    max_sum,
    is_safe,
):
    """
    Polyhedral phase shifting fails only beyond its safe theoretical limit.
    """

    error, _, _, system = _dealiasing_product(
        tmp_path,
        test_system,
        precision,
        grid_size=(36, 36, 36),
        dealiasing_updates={
            "method": "phase-shift",
            "truncation": "polyhedral",
            "max_sum": max_sum,
        },
    )

    # The adjacent shell admitted above 2/3 produces resolvable aliasing
    if is_safe:
        assert error <= system.tolerance
    else:
        assert error > system.tolerance


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("truncation", ("spherical", "polyhedral"))
@pytest.mark.parametrize("fft_wrapper", ("flucs", "cupy"))
def test_phase_shift_memory_models_agree(
    test_system,
    tmp_path,
    precision,
    truncation,
    fft_wrapper,
):
    """
    Supported FFT and memory models implement the same safe product.
    """

    cutoff = (
        {"radius_squared": 0.222}
        if truncation == "spherical"
        else {"max_sum": 0.666}
    )
    results = []
    memory_models = {
        "flucs": ("standard", "low_memory", "in_place"),
        "cupy": ("standard", "low_memory"),
    }[fft_wrapper]
    for memory in memory_models:
        operation_path = tmp_path / fft_wrapper / memory
        error, product, reference, system = _dealiasing_product(
            operation_path,
            test_system,
            precision,
            grid_size=(30, 30, 30),
            dealiasing_updates={
                "method": "phase-shift",
                "truncation": truncation,
                "memory": memory,
                **cutoff,
            },
            setup_updates={"fft_wrapper": fft_wrapper},
        )

        assert error <= system.tolerance
        npt.assert_allclose(
            product,
            reference,
            rtol=0,
            atol=system.tolerance,
        )
        results.append(product)

    # Agreement with one reference also makes cross-model drift explicit
    for result in results[1:]:
        npt.assert_allclose(
            result,
            results[0],
            rtol=0,
            atol=system.tolerance,
        )
