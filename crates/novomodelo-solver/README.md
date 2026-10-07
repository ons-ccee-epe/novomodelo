# cobre-solver

LP/MIP solver abstraction for the [Cobre](https://github.com/cobre-rs/cobre)
power systems ecosystem.

Defines a backend-agnostic `SolverInterface` trait for LP and MIP problem
construction, solving, dual extraction, and basis warm-starting. The default
backend is [HiGHS](https://highs.dev), a production-grade open-source solver
well-suited to the iterative LP workloads of power system optimization. The
crate includes a 12-level retry escalation strategy for numerically difficult
LPs: when HiGHS returns an infeasible or numerically unstable status, the
solver retries with progressively more aggressive scaling, presolve, and
simplex strategy options before propagating failure. An optional CLP backend
exists behind the `clp` feature (off by default); it implements the same
`SolverInterface` and is conformance-validated as a drop-in for HiGHS.

## When to Use

Depend on `cobre-solver` directly when you are writing an optimization
algorithm that needs to build and solve LP subproblems and you want
backend-portability without coupling to HiGHS internals. If you only need to
run the full SDDP pipeline, depend on `cobre-sddp` instead, which manages the
solver lifecycle for you.

## Features

| Feature        | Default | Description                                                                                                        |
| -------------- | ------- | ------------------------------------------------------------------------------------------------------------------ |
| `highs`        | on      | Gates `HighsSolver` and `HighsProfile`. On by default.                                                             |
| `clp`          | off     | Gates `ClpSolver`, `ClpProfile`, and `clp_version`. Mutually exclusive with `highs`. See build requirements below. |
| `test-support` | off     | Exposes FFI option-setting helpers for integration tests. Must not be enabled in production builds.                |

## Backend selection

Exactly one LP backend is compiled in. The backends are mutually exclusive:

- `--features highs` (the default) selects HiGHS.
- `--no-default-features --features clp` selects CLP.
- Enabling both `highs` and `clp` is a compile error, and enabling neither is
  also a compile error. Building with `--all-features` therefore fails by
  design.

### `clp` build requirements

The `clp` feature builds CLP and CoinUtils from vendored source. Before
enabling it, initialize the submodules:

```
git submodule update --init --recursive
```

This fetches the Clp (`releases/1.17.11`) and CoinUtils (`releases/2.11.13`)
sources into `crates/cobre-solver/vendor/`.

The first build with `--features clp` runs a CoinUtils + Clp cmake superbuild
(approximately 150 C++ translation units), which takes several minutes. The
cmake output directory is cached across rebuilds, so subsequent builds are
fast.

### CI behavior

Because the backends are mutually exclusive, CI runs a two-backend matrix
rather than a single `--all-features` pass:

- A primary HiGHS job runs the full suite (`check`, `test`, `clippy`, `docs`,
  `coverage`) with the default feature set plus the shared non-solver features.
- A lean CLP job checks out submodules recursively and runs
  `check`/`clippy`/`build`/`test` under
  `--no-default-features --features clp` plus the same non-solver features.

`docs`, `fmt`, and `coverage` run once (HiGHS only). Expect the first CLP CI
build after a cache miss to be slower than a HiGHS-only build due to the C++
superbuild.

### Per-solver test invocations

Each backend's test suite can be run in isolation, building only that solver:

```
# HiGHS backend (default features)
cargo test -p cobre-solver --features highs

# CLP backend only (HiGHS excluded)
cargo test -p cobre-solver --no-default-features --features clp
```

Both invocations compile, run, and lint clean on their own: the test suite is
backend-agnostic where possible, and each backend's tests are gated on its
feature so neither build pulls in the other solver. The CLP-only build always
includes at least one runnable end-to-end integration test.

## Module overview

| Module                    | Purpose                                                                                                                     |
| ------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| `trait_def`               | `SolverInterface` trait definition — the backend-agnostic method contracts                                                  |
| `types`                   | Canonical data types: `StageTemplate`, `RowBatch`, `Basis`, `LpSolution`, `SolutionView`, `SolverError`, `SolverStatistics` |
| `profile`                 | Shared profile sentinel constants (`DEFAULT_PROFILE_HEURISTIC_SENTINEL`, `DEFAULT_PROFILE_IPM_UNBOUNDED_SENTINEL`)          |
| `basis_status`            | `BasisStatus` — canonical basis status codes shared across backends                                                         |
| `freeze`                  | Freeze/thaw helpers for solver state                                                                                        |
| `backends::highs`         | `HighsSolver` and `HighsProfile` — the HiGHS backend (feature-gated behind `highs`)                                         |
| `backends::clp`           | `ClpSolver` and `ClpProfile` — the CLP backend (feature-gated behind `clp`)                                                 |
| `backends::profiled`      | `ProfiledSolver<S>` — generic per-phase profile-tracking wrapper over any `SolverInterface`                                 |
| `ffi::highs` / `ffi::clp` | Raw `unsafe` FFI bindings to the `cobre_highs_*` / `cobre_clp_*` C wrapper functions                                        |

The `backends::highs` and `ffi::highs` modules compile only with the `highs`
feature; `backends::clp` and `ffi::clp` only with `clp`. `trait_def`, `types`,
and `basis_status` are always compiled — the canonical `BasisStatus` mapping is
feature-independent, built regardless of which backend feature is selected —
so algorithm code can be written against `SolverInterface` without depending
on either concrete backend.

## `SolverInterface` trait

```rust
pub trait SolverInterface: Send {
    type Profile: Copy + PartialEq + Default + Send;
    // ...
}
```

Resolved as a **generic type parameter at compile time** (never `dyn
SolverInterface`), keeping virtual dispatch off the hot path — the same
compile-time monomorphization pattern used by `Communicator` in
[cobre-comm](../cobre-comm/README.md). Requires `Send` but not `Sync`: a
solver instance holds mutable C-library state (factorization workspace) that
is not thread-safe, so each worker thread owns exactly one instance.

### Method summary

| Method                           | `&self` / `&mut self` | Returns                                 | Description                                                                   |
| -------------------------------- | --------------------- | --------------------------------------- | ----------------------------------------------------------------------------- |
| `apply_profile`                  | `&mut self`           | `()`                                    | Applies every field of `Self::Profile` to the underlying solver in one call   |
| `load_model`                     | `&mut self`           | `()`                                    | Bulk-loads a structural LP from a `StageTemplate`; replaces any prior model   |
| `add_rows`                       | `&mut self`           | `()`                                    | Appends a `RowBatch` of constraint rows to the dynamic region                 |
| `set_row_bounds`                 | `&mut self`           | `()`                                    | Updates row lower/upper bounds at indexed positions                           |
| `set_col_bounds`                 | `&mut self`           | `()`                                    | Updates column lower/upper bounds at indexed positions                        |
| `solve`                          | `&mut self`           | `Result<SolutionView<'_>, SolverError>` | Solves the LP, optionally warm-starting; `basis: Option<&Basis>` — see below  |
| `get_basis`                      | `&mut self`           | `()`                                    | Writes basis status codes into a caller-owned `&mut Basis`                    |
| `statistics` / `statistics_into` | `&self`               | `SolverStatistics` / `()`               | Returns (or copies into a reused buffer) accumulated monotonic solve counters |
| `name`                           | `&self`               | `&'static str`                          | Returns a static string identifying the backend                               |
| `solver_name_version`            | `&self`               | `String`                                | Returns `"name vX.Y.Z"` (e.g. `"HiGHS v1.8.1"`) for metadata output           |

### `solve` merges cold-start and warm-start

Unlike an earlier design with separate `solve`/`solve_with_basis` methods, a
single `fn solve(&mut self, basis: Option<&Basis>) -> Result<SolutionView<'_>, SolverError>`
covers both paths: `basis = Some(&b)` installs `b` before running the simplex
(returning `SolverError::BasisInconsistent` if `b` fails the solver's
consistency check, or `BasisRowCountMismatch` if `b.row_status.len()` is
smaller than the current LP's row count); `basis = None` warm-starts from
whatever basis the instance currently holds from its own prior `solve`
history. Implementations may retain internal state (factorization, simplex
basis) between consecutive calls as a performance optimization — callers
needing a reproducible reset must call `load_model` or pass an explicit
`Basis`.

### Thread safety

`SolverInterface` requires `Send` but not `Sync`. `Send` allows a solver
instance to be transferred to a worker thread at startup. The absence of `Sync`
prevents concurrent access from multiple threads, which matches the reality of
C-library solver handles: they maintain mutable factorization workspaces that
are not thread-safe. Each worker thread owns exactly one solver instance.

## `SolverError`

Nine variants, returned after all retry attempts are exhausted:

| Variant                 | Fields                                             | Hard stop?                                          |
| ----------------------- | -------------------------------------------------- | --------------------------------------------------- |
| `Infeasible`            | —                                                  | Yes                                                 |
| `Unbounded`             | —                                                  | Yes                                                 |
| `NumericalDifficulty`   | `message`                                          | Yes                                                 |
| `TimeLimitExceeded`     | `elapsed_seconds`                                  | No                                                  |
| `IterationLimit`        | `iterations`                                       | No                                                  |
| `InternalError`         | `message`, `error_code: Option<i32>`               | Yes                                                 |
| `Unsupported`           | `&'static str`                                     | No — fall back to an alternate code path            |
| `BasisInconsistent`     | (basis rejected: basic-variable count mismatch)    | Yes — offered basis violates the LP basis invariant |
| `BasisRowCountMismatch` | (offered basis has fewer rows than the current LP) | Yes                                                 |

`BasisInconsistent` and `BasisRowCountMismatch` occur **only** when
`solve(Some(&b))` is called with an incompatible `b`.

## HiGHS backend (`HighsSolver`)

`HighsSolver::new()` allocates a HiGHS handle and applies the performance-tuned
default `HighsProfile` (below) before returning.

### 12-level retry escalation

When the initial solve does not reach `OPTIMAL`, `HighsSolver::solve` escalates
through twelve retry levels (`backends/highs/retry.rs`) in two phases: levels
0–4 are cumulative (each adds options on top of the previous state, within a
15s per-level wall-clock budget), and levels 5–11 each start fresh from
restored defaults (within a 30s per-level budget), with a 120s overall budget
across the whole escalation. The first level that reaches `OPTIMAL` exits the
loop; `UNBOUNDED`, `ITERATION_LIMIT`, or a budget-exceeded attempt is treated
as possibly spurious and retried at the next level, while any other terminal
status (e.g. `INFEASIBLE`) stops the escalation immediately. Default settings
and the caller's profile are unconditionally restored after the retry loop
regardless of outcome. If all twelve levels are exhausted, the method returns
`SolverError::NumericalDifficulty` (or `SolverError::Unbounded` if the initial
failure was an unbounded status).

### `HighsProfile` fields

| Field                               | Type  | Default                                                           |
| ----------------------------------- | ----- | ----------------------------------------------------------------- |
| `primal_feasibility_tolerance`      | `f64` | `1e-9`                                                            |
| `dual_feasibility_tolerance`        | `f64` | `1e-9`                                                            |
| `simplex_iteration_limit`           | `u32` | `0` (heuristic: `num_cols * 50`, capped at `100_000`)             |
| `ipm_iteration_limit`               | `u32` | `10_000`                                                          |
| `simplex_dual_edge_weight_strategy` | `i32` | `1` (Devex)                                                       |
| `simplex_scale_strategy`            | `i32` | `0` (off — the cobre prescaler already normalizes matrix entries) |
| `simplex_price_strategy`            | `i32` | `1` (Row)                                                         |

`ProfiledSolver<S>` (`backends::profiled`) wraps any `SolverInterface`
implementor with per-phase profile tracking, resolved at compile time via
monomorphization. `set_profile` compares the requested profile to the
currently-applied one with a single whole-struct `PartialEq` check and issues
zero inner calls when they match.

## Build requirements

### Git submodule

HiGHS is vendored as a git submodule at `crates/cobre-solver/vendor/HiGHS/`. Before building
`cobre-solver` for the first time (or after a fresh clone), initialize the
submodule:

```
git submodule update --init --recursive
```

The build script checks for `crates/cobre-solver/vendor/HiGHS/CMakeLists.txt` and panics with a
clear error message if the submodule is not initialized.

### System dependencies

| Dependency   | Minimum version | Notes                                                       |
| ------------ | --------------- | ----------------------------------------------------------- |
| cmake        | 3.15            | Required by the HiGHS build system                          |
| C compiler   | C11             | gcc or clang; HiGHS and the C wrapper are C/C++             |
| C++ compiler | C++17           | Required by HiGHS internals                                 |
| ~~zlib~~     | ~~any~~         | Not needed — disabled via `CMAKE_DISABLE_FIND_PACKAGE_ZLIB` |

## Testing

```
cargo test -p cobre-solver --features highs
```

This requires cmake, a C/C++ compiler, and an initialized
`crates/cobre-solver/vendor/HiGHS/` submodule (see
[Build requirements](#build-requirements)). See
[Per-solver test invocations](#per-solver-test-invocations) above for running
the CLP backend's suite in isolation.

## Key Types

- **`SolverInterface`** — the core trait every backend must implement; defines
  problem construction, solve, and dual/basis extraction methods
- **`HighsSolver`** — the default HiGHS backend; feature-gated behind `highs`
  (enabled by default)
- **`ClpSolver`** — the optional CLP/CoinUtils backend; feature-gated behind
  `clp` (off by default)
- **`Basis`** — LP basis snapshot for warm-starting a subsequent solve on a
  structurally related problem
- **`LpSolution`** — solved LP result carrying primal values, duals, and
  objective value
- **`SolverStatistics`** — per-solve diagnostics including iteration count,
  wall-clock time, basis rejections, and retry escalation level reached
- **`StageTemplate`** — pre-built LP structure for a single time stage,
  cloned and patched each iteration to avoid repeated matrix assembly

## Links

| Resource   | URL                                                        |
| ---------- | ---------------------------------------------------------- |
| Docs site  | <https://docs.cobre-rs.dev/>                               |
| API Docs   | <https://docs.rs/cobre-solver/latest/cobre_solver/>        |
| Repository | <https://github.com/cobre-rs/cobre>                        |
| CHANGELOG  | <https://github.com/cobre-rs/cobre/blob/main/CHANGELOG.md> |

## Status

**Alpha** — API is functional but not yet stable.

## License

Apache-2.0
