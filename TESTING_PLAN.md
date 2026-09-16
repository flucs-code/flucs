# FLUCS Test-Suite Infrastructure Plan

## Summary

Organize tests according to the FLUCS source hierarchy. Classify every test
explicitly with exactly one of `pytest.mark.cpu` and `pytest.mark.gpu`.

Plain `pytest` detects whether a usable CUDA device is available. It runs the
standard CPU and GPU suite when one is available and otherwise runs the
standard CPU suite. Tests deliberately classified as long are excluded unless
`--long` is supplied. `pytest --cpu` and `pytest --gpu`
explicitly select only that device class.

## Layout and Selection

- Mirror `src/flucs` beneath `tests`, for example:
  - `tests/test_flucs.py`
  - `tests/test_input.py`
  - `tests/test_diagnostic.py`
  - `tests/solvers/fourier/test_fourier_solver.py`
  - `tests/solvers/fourier/timesteppers/`
  - `tests/systems/`
- Retain `tests/support/` as the exception for standalone test systems, shared
  fixtures, and test-only assets.
- Colocate CPU and GPU tests within the relevant module. Mark every test,
  class, module, or generated parameter with exactly one of
  `@pytest.mark.cpu` and `@pytest.mark.gpu`. Use a module-level device mark only
  when every test in that module belongs to the same class.
- Mark shared functionality with `pytest.mark.core` and solver-owned
  functionality with `pytest.mark.solver("EntryPointName")`. Every test must
  have exactly one of these ownership markers.
- Mark deliberately long tests with the additional `pytest.mark.long` marker.
  This classifies elapsed time rather than computational resources and is
  independent of device and ownership, not an alternative to the required
  markers.
- Provide these commands:
  - `pytest --flucs-help`: concise FLUCS-specific options and examples.
  - `pytest`: all standard CPU tests, plus all standard GPU tests when
    automatic GPU detection succeeds.
  - `pytest --cpu`: CPU tests only, without probing for a GPU.
  - `pytest --gpu`: GPU tests only, after strict GPU validation.
  - `pytest --long`: include long-running tests within the otherwise selected
    device and ownership classes.
  - `pytest --core [--cpu|--gpu]`: shared core functionality in the selected
    device class.
  - `pytest --solvers FourierSolver [--cpu|--gpu]`: core tests applicable to
    FourierSolver plus FourierSolver-specific tests in the selected device
    class.
  - `pytest --solvers FourierSolver ThirdSolver [--cpu|--gpu]`: core tests plus
    tests for each selected solver in the selected device class.
  - `pytest --solvers all [--cpu|--gpu]`: core tests plus tests for every
    solver in the selected device class.
- Treat `--cpu` and `--gpu` as mutually exclusive. These flags select a device
  class; they do not alter core/solver ownership selection.
- Treat `--long` as an orthogonal inclusion flag. Neither automatic GPU
  discovery nor explicit `--gpu` enables long tests by itself.
- Treat `--core` and `--solvers` as mutually exclusive. Use exact public
  entry-point names and reject unknown names with an actionable list of valid
  choices.
- Use the mirrored path for organization and explicit ownership markers for
  reliable selection and validation.

## Test Strictness and Readability

- Use the simplest test double that enforces the intended interface. Prefer
  `create_autospec(..., spec_set=True)` for stable Python classes and functions,
  and assert important call counts and arguments explicitly.
- Use small explicit fakes or sentinels when an object is only a value or
  implements a tiny protocol. Do not use a permissive `Mock` where it could
  silently accept an unintended attribute or method.
- Avoid autospeccing compiled, GPU-dependent, or highly dynamic interfaces when
  doing so would make CPU collection unsafe or misrepresent the real callable.
  In those cases, mock only the narrow boundary used by the test and verify its
  calls directly.
- Make numerical tolerances explicit, including both relative and absolute
  tolerances. Check array backend, shape, or dtype when these form part of the
  behaviour under test.
- Compare complete output when it is stable. Use focused substring assertions
  only for variable output such as tracebacks, timestamps, and filesystem
  paths.
- Keep tests behavioural rather than creating a one-to-one mapping between
  functions and tests. Use short comments to separate setup, execution, and
  verification, and place the opening and closing triple quotes of docstrings
  on their own lines.
- Prefer broad integration tests that carry one behaviour through its public
  workflow. When that workflow supports materially different options,
  parameterize the same test over those options rather than testing only the
  default or creating separate test functions for each case.
- Exercise data-producing and data-loading workflows over every supported
  precision. Assert the stored and reconstructed dtypes as well as values, and
  derive explicit numerical tolerances from the precision under test.
- Keep shared test architecture in `tests/support/support.py`, including the
  standalone TestSystem registry and reusable parameter sets. Define the
  single- and double-precision names, NumPy types, and NetCDF storage types
  there, and expose them through a shared parametrized fixture. Obtain the
  general comparison tolerance from `FlucsSystem` rather than maintaining a
  second test-only tolerance policy.
- Anchor the production tolerance policy in a separate lightweight test that
  independently checks the intended value of 64 times machine epsilon and its
  scalar type for every supported precision. Integration tests should still
  confirm that constructed systems receive the shared tolerance, but should
  not repeat its formula. This keeps policy failures local and easy to identify
  without allowing an accidental increase in the production tolerance to make
  numerical tests silently more permissive.
- Prefer constructing registered TestSystems through the normal
  `FlucsInput.create_solver_system()` path whenever a core test needs a system.
  Do not introduce reduced system subclasses merely to satisfy one core
  interface; use narrow controlled payloads only when the data itself is the
  behaviour under test.
- Use the shared baseline directly for round-off comparisons. Tests of
  operations that accumulate additional numerical error must express their
  allowance as an explicit, documented multiple of that baseline; exact
  operations should continue to use exact assertions.
- Keep independent option axes in the integration test where each matters.
  Avoid an uninformative Cartesian product when one option can be exercised in
  another existing integration test without weakening coverage.
- Distinguish shared configuration from behavioural expectations. Derive
  incidental metadata such as solver-specific shapes, diagnostic variables,
  NetCDF suffixes, and active tolerances from the relevant production object.
  Keep expected numerical results, public schemas, method-to-class mappings,
  convergence orders, and other behavioural contracts independent of the code
  being exercised.
- Do not calculate an expected result by calling the implementation under test
  or by reading back the same registry that selected the production behaviour.
  Hard-coded expected values are desirable when they form a small independent
  oracle; duplicated incidental data is not.
- For finite production registries, maintain an independent expected manifest
  and assert exact agreement between their keys. Parameterize the behavioural
  test over the expected manifest so that a newly registered precision,
  timestepper, solver, dealiasing method, or similar option cannot silently
  escape coverage.
- Prefer one focused, lightweight test for a shared policy when this gives a
  substantially clearer failure than embedding the policy formula in a larger
  integration test. Retain integration assertions that verify the policy is
  actually wired into constructed runtime objects.
- When several options share setup and behaviour, exercise them in one
  parametrized or loop-based integration test. Do not repeat option-independent
  checks for every case merely because the objects were constructed together;
  verify invariant behaviour once on a representative instance.
- Make dtype conversions explicit where precision is part of the contract. Do
  not rely on NumPy scalar-promotion details that differ between supported
  dependency versions; tests should assert the resulting dtype as well as the
  values.

## Collection, Duration Selection, and Device Preflight

- Register the `cpu`, `gpu`, and `long` markers in pytest configuration
  and retain strict marker validation for the device markers.
- During collection, require every test item to resolve to exactly one device
  marker. Reject an item with neither marker or both markers with its node ID
  and an actionable error. This makes device classification a checked part of
  the suite rather than a decorative convention.
- Apply the same rule to parametrized runtime cases. Mark each generated
  parameter `gpu` when its TestSystem requires CUDA and `cpu` otherwise, so a
  single runtime test function can produce valid cases in both classes.
- Resolve a session device mode before collection:
  - `--cpu` selects CPU-only mode and does not probe CUDA;
  - `--gpu` performs strict GPU validation and selects GPU-only mode;
  - with neither flag, a successful non-fatal GPU probe selects both classes;
  - with neither flag and no usable GPU, select CPU-only mode.
- Deselect items outside the resolved device mode rather than skipping them.
  Deselect `long` items as well unless `--long` is present.
  Apply ownership, device, and duration filtering independently, then select
  only items satisfying every filter.
- Report whether long tests are excluded or enabled in the pytest header
  alongside the resolved device mode.
- Report the resolved mode in the pytest header, for example `FLUCS devices:
  CPU and GPU` or `FLUCS devices: CPU only (no usable CUDA device detected)`.
- Require GPU test modules to avoid CUDA initialization, allocation, or
  compilation at import time; all GPU work belongs in fixtures or test bodies.
  This keeps CPU-only and automatic no-GPU collection safe on machines without
  CuPy or CUDA.
- Use one shared GPU probe which:
  - imports CuPy successfully;
  - detects at least one CUDA device;
  - selects a device;
  - performs a minimal allocation and synchronization.
- Cache the probe result for the session. Automatic mode treats a failed probe
  as ordinary CPU-only operation, while explicit `--gpu` fails before test
  collection and exposes the underlying CuPy/CUDA exception.
- The probe establishes runtime device usability, not the complete CUDA build
  environment. GPU CI and cluster environments must also provide the matching
  CUDA toolkit/compiler needed by runtime kernel compilation.

## Core Tests Across Solvers

- Provide an internal registry mapping each public solver name to its
  standalone test system, initially:
  - `FourierSolver` → `TestFourierSystem`
- Assert that the TestSystem registry keys exactly match the solver entry-point
  names owned by the repository's distribution. Filter out entry points from
  unrelated installed distributions so local plugin installations do not alter
  the core repository's coverage contract.
- Let each TestSystem specification declare an independent, lazily loaded
  mapping of its expected timestepper names to concrete classes. The shared
  solver test should compare this mapping exactly with the selected production
  solver and construct every declared method through the normal input path.
  This keeps the core test solver-agnostic while detecting missing, additional,
  or accidentally swapped timestepper registrations.
- Core tests that exercise solver-facing functionality explicitly request a
  shared parametrized test-system fixture.
- Without `--solvers`, those tests run once for every compatible registered
  test system.
- With `--solvers`, they run once for each selected solver's test system;
  `--solvers all` selects every registered test system.
- Pure core tests that do not interact with a solver remain ordinary tests and
  are not unnecessarily parametrized.
- Future solvers add a standalone test system and registry entry, allowing
  existing shared core tests to exercise them automatically.
- Store each TestSystem's baseline runtime input and GPU requirement alongside
  its registry specification. Copy the input to a temporary directory for each
  run, override precision centrally, and allow focused future tests to apply
  further recursive overrides without modifying the shared baseline.
- Run each selected `(TestSystem, precision)` runtime combination once per
  pytest session and share its completed objects and artifacts between the
  diagnostic, output, and restart tests. Apply the GPU marker to each runtime
  parameter independently, so CPU-capable TestSystems continue to run by
  default even when another solver requires CUDA.
- Allow a runtime test to select one precision with the validated
  `runtime_precision(name)` marker. Use single precision for general runtime
  integration when the precision-specific data paths are already exercised by
  synthetic tests, while retaining both precisions for tests where runtime
  numerical precision is itself material.
- Discover runtime outputs, diagnostic variables, dimensions, complex-variable
  suffixes, and restart payloads from the completed TestSystem rather than
  listing expected files or assuming solver-specific names and shapes.
- Keep one compact real runtime workflow for every TestSystem. It should use
  the registered plug-in to exercise genuine diagnostics, output, and restart
  data together, complementing CPU tests that isolate serialization and edge
  cases with small controlled payloads. Load and construct this same runtime
  input in the input test without executing the GPU solver.

## TestFourierSystem Lifecycle

- Temporary development registration must not survive completion of the test
  suite. Before the suite is considered finished, remove the
  `TestFourierSystem` entry from `[project.entry-points."flucs.systems"]`,
  remove the accompanying exposure of `tests.support` through setuptools, and
  restore normal `src`-only package discovery. Verify that it is no longer
  discoverable during ordinary FLUCS execution.
- At pytest session start, temporarily register one `TestFourierSystem` entry
  in FLUCS's in-process system registry.
- Restore the original registry during session teardown, including teardown
  following test failures.
- Keep registration session-scoped, while mutable solver and system instances
  remain isolated between tests.
- Test the normal FLUCS argument parsing and execution path in-process.
  Supporting a separately launched `flucs` subprocess is not required.
- Keep the existing `flucs --test` option separate from pytest; it does not
  become the test-suite runner.

## Planned Fourier Numerical Validation

- Keep numerical coverage behavioural and compact. Exercise lower-level CUDA
  helpers through solver-level tests when those tests already establish the
  relevant behaviour, rather than duplicating tests for each implementation
  function.
- Test every supported dealiasing method using random data with a small scan
  around its theoretical truncation boundary. Include points immediately
  below, at, and immediately above the boundary so that each test demonstrates
  both the valid range and where aliasing begins.
- Demonstrate timestepper convergence with one parametrized GPU test covering
  AB3, RK4, and SSPRK3. Use a short nonlinear evolution and three timestep
  refinements initially; this need only establish the expected local order,
  rather than reproduce a long standalone convergence campaign.
- Generate the developed initial state in-process during the test session:
  - derive the spin-up configuration from
    `~/data/flucs_testing/testing/damping/input.toml`, but copy the required
    values into test support so the suite does not depend on an external file;
  - override `tfinal = 100` and disable restart, diagnostic, text, and NetCDF
    writes that are irrelevant to producing the developed state;
  - perform the deterministic spin-up using its negative-damping forcing;
  - retain an immutable host copy of the resulting Fourier fields; and
  - initialize every timestepper and timestep refinement from an identical
    copy of that state.
- Treat the existing runs under `flucs_testing/testing/ou`,
  `flucs_testing/testing/dt`, and `flucs_testing/testing/damping` as sources
  for sensible resolution, forcing, spin-up, and timestep parameters. Do not
  make the test suite depend on these external files or commit a generated
  restart fixture.
- Keep the convergence problem deterministic, the timestep fixed, and the
  physical end time identical between refinements. Compare the complete solved
  Fourier state using an explicit relative norm, synchronize the GPU before
  comparison, and check for non-finite values and invalid padded modes.
- Give AB3 a test-only full-order bootstrap rather than measuring its normal
  zero-history startup:
  - generate the first two advances with a higher-order one-step method;
  - evaluate the explicit terms at the required time levels;
  - populate the timestep and explicit-term histories in the exact propagated
    representation consumed by the AB3 CUDA kernel; and
  - begin the convergence interval only after all three AB3 history levels are
    valid.
- Keep the AB3 bootstrap in test-support code; it is a measurement fixture and
  does not change the production startup behaviour. Fold checks of its history,
  coefficients, and initial advance into the convergence test rather than
  creating several narrow bootstrap tests.
- Dedicated forcing and shared Fourier-system diagnostic tests may be added if
  solver-level runtime tests leave meaningful behaviour uncovered. Do not test
  post-processing scripts as part of the core suite.

## FourierSystem Test Implementation

Keep the Fourier tests owned by `FourierSolver`, with CPU and GPU tests
colocated beneath `tests/solvers/fourier/`. Place the timestep convergence test
beneath the corresponding `timesteppers/` subdirectory. Mark tests that
initialise CUDA with `pytest.mark.gpu` and every remaining test with
`pytest.mark.cpu`.

### CPU coverage

Use separate behavioural tests rather than one large integration test:

- Dealiasing configuration and validation: parameterize the valid two-thirds,
  spherical phase-shift, and polyhedral phase-shift configurations and their
  memory models. Check the derived padded and unpadded dimensions, effective
  truncation, and compile definitions. Use a second parametrized test for
  invalid methods, truncations, memory models, resolutions, and radii.
- Fourier geometry and shells: compare `kx`, `ky`, and `kz` with NumPy's FFT
  conventions and check broadcast shapes, monotonic `kperp` and `kmod` shells,
  complete-grid coverage, active-precision shell dtypes, and solved-mode
  coordinate extraction. Construct shell widths explicitly in the active
  precision so results do not depend on NumPy's scalar-promotion version.
- Timestep control: cover continuous and discrete timestep reduction,
  permitted increases, maximum-timestep limiting, unchanged timesteps, and
  interruption below `dt_min`.
- Restart-grid remapping: cover unchanged, refined, and coarsened Fourier
  grids. Preserve common signed modes, zero new modes, use the active
  precision, and reject missing or incompatible restart fields.

### Shared-setup GPU coverage

Compile one `TestFourierSystem` per precision at module scope. Reset its fields,
counters, caches, and timestepper state before each test so that the following
tests remain independent without repeating CUDA compilation:

- Solved grid and initial conditions: compare the CUDA mask with an independent
  CPU mask and check mode counts, array metadata, zero padded modes, and the
  `ky=0` reality condition.
- Linear quantities: compare the CUDA linear matrix and frequencies with the
  TestFourierSystem CPU references, then check eigensystem normalization and
  inversion and the Pade propagator on solved modes.
- Fields and serialization inputs: check current/previous field-ring indexing,
  CPU and GPU real-space reconstruction, cache invalidation, and the Fourier
  restart payload returned for the current device fields.

### CUDA dealiasing boundaries

Use the same calculation as `_check_dealiasing`. Generate fixed-seed random
real fields, transform them using `norm="forward"`, zero modes outside the
solved mask, and compare the configured operation with a reference from
`dealiased_multiplication_rfft` using twice-padded dimensions. Zero both outputs
outside the solved mask and define the error exactly as

```python
error = cp.max(cp.abs(product_operation - product_dealiased_rfft))
```

Run safe paths in both precisions and use explicit precision-derived absolute
thresholds. Deliberately unsafe configurations must produce an error above a
separate lower bound.

- For two-thirds dealiasing, fix the padded grid at 36 cubed and scan unpadded
  sizes 21, 23, and 25. The first two are below and at the largest safe retained
  bandwidth; the final case is immediately above it and must reveal aliasing.
- For spherical and polyhedral phase shifting, use a 30 cubed grid. For the
  spherical truncation, scan
  `radius_squared = 0.221`, `0.222`, and `0.223`. The first two are within the
  production-safe cutoff immediately below the theoretical `2/9` limit; the
  final value admits the dangerous boundary shell.
- For polyhedral phase shifting, scan `max_sum = 0.665`, `0.666`, and `0.667`.
  The first two are within the production-safe cutoff immediately below the
  theoretical `2/3` limit; the final value admits the dangerous boundary.
- At safe spherical and polyhedral cutoffs, compare `standard`, `low_memory`,
  and `in_place` operation results with the same reference and with one another.

Retain the utility-level multiplication test. It checks the independent
reference primitive and array backends, while these tests check the actual
FourierSystem CUDA implementations and their truncation boundaries.

### AB3 bootstrap

Bootstrap AB3 entirely through the production timestepper. Execute its first
two updates so that it rotates and propagates its own explicit-term history,
but replace each resulting startup field immediately with the corresponding
full-order RK4 field. The second AB3 history evaluation therefore sees the RK4
state, while no test code reproduces or mutates CUDA history internals. This is
the same scratch-update principle used by the earlier convergence prototype,
implemented here with the separately compiled RK4 TestFourierSystem.

### Timestepper convergence

Use one double-precision GPU test parametrized over the complete live
FourierSolver timestepper registry. Maintain the expected-order mapping
`ab3 = 3`, `rk4 = 4`, and `ssprk3 = 3`, and assert that its keys exactly match
the registered methods so a future timestepper cannot silently escape
coverage. Mark this convergence test with both `pytest.mark.gpu` and
`pytest.mark.long`. It is excluded from every standard invocation, including
`pytest --gpu`, and included only when `--long` is also present.

Generate one common developed TestFourierSystem state in-process from the
configuration in `~/data/flucs_testing/testing/damping/input.toml`: a 32-cubed
two-thirds grid, deterministic amplitude `0.01`, timestep `0.02`,
negative-damping rate `0.05`, `range_kmod = [0.5, 1.5]`, and normalized
adaptive `kmod` hyperdissipation with coefficient `2.0` and power `3`. Copy
these values into a repository-owned test input, set `tfinal = 100`, and
disable restart, diagnostic, text, and NetCDF writes during spin-up. Retain an
immutable host copy and require finite values, nonzero nonlinear content,
material departure from the initial transient, and zero padded modes.

For each method, start from an identical copy of this state, retain the same
deterministic forcing, use fixed rather than adaptive hyperdissipation so the
physical problem is independent of timestep, and evolve over
`T = 0.4` with timesteps `0.04`, `0.02`, `0.01`, and `0.005`. Use a
method-specific reference at `0.00125`. Advance exact integer step counts and
disable diagnostics, output, and restart work.

For AB3, obtain the first two accepted fields from RK4 at the selected
timestep. Run the corresponding AB3 updates to build production history, but
replace each AB3 startup result with its RK4 counterpart before evaluating the
next history term. Verify that the first post-bootstrap update uses
coefficients `(23, -16, 5) / 12`. All advances must use the unchanged
production timesteppers.

Compare every solved complex Fourier coefficient with the method's own fine
reference using a relative L2 norm. Require finite and strictly decreasing
errors, exact zero padded modes, a fitted order within `0.5` of the formal
order, the two finest local orders within `0.75`, and errors above the
double-precision noise floor.

### Verification

- On a machine without a usable GPU, plain pytest must select every CPU test
  in the standard suite and collect then deselect every GPU and long
  test.
- On a machine with a usable GPU, plain pytest must select the standard tests
  from both device classes while continuing to deselect long tests.
- Explicit `pytest --cpu` and `pytest --gpu` runs must select disjoint sets
  whose union is the complete standard suite.
- `pytest --gpu --long` must additionally select the timestepper convergence
  test. `--long` must not change device or ownership
  selection.
- Run the targeted CPU Fourier tests, the complete default suite, and
  `ruff check .` before handoff.
- Validate GPU behavior with
  `pytest tests/solvers/fourier --solvers FourierSolver --gpu` when a GPU run
  is explicitly requested.
- Validate the comprehensive long-running path separately with
  `pytest --solvers FourierSolver --gpu --long`.
- Do not omit a supported timestepper from the parametrized convergence test
  merely to shorten it; control its runtime impact by excluding the complete
  test from the standard suite.

## Automation and Repository Contract

- Pull-request CI runs `pytest --cpu`, making every CPU test required while
  keeping the job deterministic even if runner hardware changes.
- The CPU job installs FLUCS and its test dependencies without a CUDA toolkit
  or CuPy extra, demonstrating genuine CPU independence.
- Current GitHub-hosted `ubuntu-latest` runners have no usable NVIDIA device;
  plain automatic mode would therefore resolve to CPU-only, but the workflow
  should still use explicit `--cpu` to state and preserve its contract.
- No GPU CI job is initially required. A future self-hosted or explicitly
  GPU-equipped runner installs a compatible NVIDIA driver, CUDA toolkit,
  CuPy/nvmath-python environment, and test dependencies, then runs
  `pytest --gpu` for the standard GPU suite. A scheduled or manually triggered
  comprehensive job runs `pytest --gpu --long`. One supported Python
  version is sufficient for GPU jobs while the CPU matrix retains broad
  Python-version coverage.
- Automatic mode is primarily for interactive use: plain `pytest` on an
  allocated Hydra GPU node runs the standard CPU and GPU suite, whereas plain
  `pytest` on a login node or CPU-only machine runs only standard CPU tests.
  The long timestepper convergence scan always remains an explicit
  `--long` opt-in.
- Each `flucs_*` repository owns its tests and CI while following the same
  conventions:
  - source-mirroring test layout;
  - explicit core or solver ownership markers;
  - exactly one explicit `cpu` or `gpu` marker per collected item;
  - the same pytest flags, long-test policy, and GPU preflight behavior;
  - exact public solver names for selection.
- Before implementing the first `flucs_*` test suite, extract the generic
  command-line flags, marker validation, selection rules, and GPU preflight
  into an explicitly loaded reusable pytest plugin such as
  `flucs.testing.pytest_plugin`. Keep each repository's TestSystem registry and
  fixtures local, and do not register the plugin globally.
- Benchmark, regression-data, cross-repository orchestration, and
  documentation policies remain subjects for later plans.

## Infrastructure Acceptance Criteria

- Every collected test has exactly one of the `cpu` and `gpu` markers; missing
  or conflicting device markers fail collection clearly.
- Plain `pytest` executes both classes when a usable GPU is detected and only
  the CPU class otherwise, always excluding long tests.
- `pytest --cpu` executes only standard CPU tests without requiring CUDA.
- `pytest --gpu` executes only standard GPU tests after successful GPU
  validation.
- `pytest --long` includes long tests without changing the
  resolved device or ownership selection.
- `pytest --gpu --long` includes the GPU timestepper convergence test,
  while `pytest --gpu` alone does not.
- `--cpu` and `--gpu` together fail with an actionable usage error.
- A missing or unusable GPU is non-fatal in automatic mode but causes explicit
  `pytest --gpu` to fail before collecting test modules and expose the
  underlying error.
- Standard CPU-only and GPU-only item sets are disjoint, and their union equals
  the automatic standard-suite selection on a usable GPU machine. Repeating
  those selections with `--long` produces the corresponding
  comprehensive sets.
- Full, core-only, and solver-specific selection produce the intended sets.
- Marker typos, conflicting selectors, and unknown solvers fail clearly.
- Missing or conflicting ownership markers fail during collection.
- `TestFourierSystem` is discoverable through normal FLUCS lookup during
  pytest and absent after teardown.
- Shared core parametrization selects each applicable test system exactly
  once.
- GPU-marked tests perform no CUDA work during module import.

## Assumptions

- "Registered once per test run" means one temporary registry entry per
  pytest session, not one shared mutable system instance.
- Device classification is always explicit. Tests that do not initialise CUDA
  are marked `cpu`; tests that require a working CUDA device are marked `gpu`.
- Duration classification is additional and deliberately sparse. Only tests
  whose runtime materially impedes ordinary development, initially the
  timestepper convergence scan, receive `pytest.mark.long`.
- A `cpu` marker describes test execution requirements, not machine
  compatibility: CPU tests still run in automatic standard-suite mode on a GPU
  machine, but are intentionally excluded by explicit `--gpu`.
- A `long` marker never implies CPU, GPU, core, or solver ownership. `--long`
  only removes the default duration exclusion; every other selector continues
  to apply normally.
- Pytest must import a test module to discover function-level markers.
  Therefore, CPU-only and automatic no-GPU runs deselect GPU items after
  discovery, while the no-import-time-GPU-work rule keeps collection safe.
- Exact numerical parameters and tolerances will be fixed alongside each test
  implementation. Persistent regression datasets and their policies remain a
  later decision.
