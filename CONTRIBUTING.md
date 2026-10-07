# Contributing to Novomodelo

Thanks for your interest in contributing. Novomodelo is an ecosystem for power system computation, and contributions of all kinds are welcome — code, documentation, bug reports, test cases, and domain expertise.

## Getting Started

### Prerequisites

- **Rust** (stable, latest): https://rustup.rs
- **C compiler** (for HiGHS solver FFI): `gcc` or `clang`
- **CMake** (for building HiGHS from source): `cmake >= 3.15`

Optional (needed for specific crates):

- **MPICH** (for `novomodelo-comm` MPI backend): `libmpich-dev` on Debian/Ubuntu
- **Python 3.12+** and **maturin** (for `novomodelo-python` builds): `pip install maturin`

### Building

```bash
git clone https://github.com/ons-ccee-epe/novomodelo.git
cd novomodelo

# Build all crates
cargo build --workspace

# Run the test suite (default HiGHS backend)
cargo test --workspace

# Run tests for a specific crate
cargo test -p novomodelo-sddp

# Build Rust API documentation
cargo doc --workspace --no-deps --open
```

### Reclaiming disk space

Every integration-test binary statically links the HiGHS/CLP/qhull C++ solver,
so `target/debug` grows large, and Cargo does not garbage-collect the
hash-suffixed binaries left behind by earlier rebuilds — stale artifacts
accumulate until pruned. When `target/` outgrows your disk, prune with
`cargo-sweep` rather than a full wipe, so the current build stays warm:

```bash
cargo install cargo-sweep      # one-time
cargo sweep --time 7           # remove artifacts not touched in the last 7 days
cargo sweep --installed        # also drop artifacts from removed toolchains
```

`cargo clean` is the from-scratch reset (it forces a full rebuild, including the
vendored C++ solver). The maturin-built `novomodelo-python` crate is excluded from the
workspace and builds into a separate `target/` directory of its own under
`crates/novomodelo-python/` — sweep or clean that one separately.

### Solver Backend Selection

`novomodelo-solver` ships two LP backends behind **mutually exclusive** Cargo features:
**HiGHS** (`highs`, on by default) and **CLP/CoinUtils** (`clp`, off by default).
Exactly one may be enabled per build — enabling both (including via
`--all-features`) is rejected at compile time. Any command that compiles
`novomodelo-solver` (the whole workspace, or `-p novomodelo-solver`/`-p novomodelo-sddp`/`-p
novomodelo-cli`) must therefore select a single backend instead of passing
`--all-features`:

```bash
# Default HiGHS backend
cargo test --workspace

# CLP backend (requires the Clp and CoinUtils submodules)
cargo test --workspace --no-default-features --features clp
```

To exercise the same surface CI does — every non-solver optional feature, run
once per backend — add the non-solver feature set explicitly (this needs MPICH
and `flatc` installed, just as `--all-features` previously did):

```bash
# HiGHS (default) backend
cargo test --workspace --features "mpi numa shared-memory serde schema slow-tests flatc-conformance test-support"

# CLP backend
cargo test --workspace --no-default-features \
  --features "clp mpi numa shared-memory serde schema slow-tests flatc-conformance test-support"
```

The same backend split applies to `cargo clippy` and `cargo build`. Crates that
do not depend on `novomodelo-solver` (e.g. `novomodelo-core`, `novomodelo-io`) are unaffected
and may still use `--all-features`.

### Testing novomodelo-core

```bash
# Run all tests including serde round-trip tests
cargo test -p novomodelo-core --all-features

# Run only the serde-gated tests
cargo test -p novomodelo-core --features serde
```

### Testing novomodelo-io

```bash
# Run all tests (requires --all-features)
cargo test -p novomodelo-io --all-features

# Run only integration tests
cargo test -p novomodelo-io --all-features --test '*'
```

novomodelo-io also declares a `test-support` feature (rolled into `--all-features` above),
exposing its crate-internal fixture builders — entity, `ParsedData`, and `Config`
construction helpers used by the semantic-validation unit tests — to `tests/`
integration binaries and downstream crates' tests.

Sample case directories are in `tests/data/`. Each follows the standard Novomodelo layout:
`config.json` at the root, entity files under `buses/`, `hydros/`, `thermals/`, etc.
When adding new parsers, add sample input files to `tests/data/` and reference them
from the integration test.

### Testing novomodelo-solver

Initialize the solver submodules first:

```bash
git submodule update --init --recursive

# Default HiGHS backend (add `test-support` for the FFI option-setting tests)
cargo test -p novomodelo-solver --features test-support

# CLP backend
cargo test -p novomodelo-solver --no-default-features --features "clp test-support"
```

The `highs` feature triggers cmake-based HiGHS compilation (requires cmake >= 3.15);
the `clp` feature builds the vendored CLP/CoinUtils superbuild. The two backends are
mutually exclusive — see [Solver Backend Selection](#solver-backend-selection). The
conformance suite validates the `SolverInterface` contract against hand-computed LP
fixtures for whichever backend is active.

### Testing novomodelo-comm

**Without MPI** (default):

```bash
cargo test -p novomodelo-comm
```

**With MPI**:

```bash
cargo test -p novomodelo-comm --features mpi
```

MPI installation required: Debian/Ubuntu: `sudo apt install libmpich-dev`; Fedora:
`sudo dnf install mpich-devel`; macOS: `brew install mpich`. The conformance suite
validates the `Communicator` contract and runs without the `mpi` feature.

### Testing novomodelo-stochastic

```bash
cargo test -p novomodelo-stochastic
```

No external dependencies or feature flags needed. Conformance tests verify PAR(p)
preprocessing (tolerance 1e-10); reproducibility tests verify seed determinism and
declaration-order invariance.

novomodelo-stochastic also declares a `test-support` feature, exposing its crate-internal
fixture builders — `OpeningTree`, `SeasonMap`, and `InflowModel` construction helpers
used by its own unit tests — to `tests/` integration binaries and downstream crates'
tests.

### Testing novomodelo-sddp

Initialize the HiGHS submodule first:

```bash
git submodule update --init --recursive

# Default HiGHS backend (`slow-tests` un-ignores the D-case parity sweep)
cargo test -p novomodelo-sddp --features slow-tests

# CLP backend
cargo test -p novomodelo-sddp --no-default-features --features "clp slow-tests"
```

Pass exactly one backend — `--all-features` enables both and fails to compile (see
[Solver Backend Selection](#solver-backend-selection)). The test suite includes unit
tests (forward/backward pass, cut management, simulation),
conformance tests (algorithm contracts against hand-computed fixtures), and integration
tests (end-to-end pipelines, Parquet output validation). The genericity gate asserts
no algorithm-specific references in infrastructure crates.

### Testing novomodelo-cli

```bash
# Default HiGHS backend
cargo test -p novomodelo-cli

# CLP backend
cargo test -p novomodelo-cli --no-default-features --features clp
```

The CLI forwards the active backend to both `novomodelo-solver` and `novomodelo-sddp`, so it
takes one backend at a time — not `--all-features` (see
[Solver Backend Selection](#solver-backend-selection)). Integration tests exercise
the binary via `assert_cmd`, organized by subcommand
(`tests/cli_run.rs`, `tests/cli_validate.rs`,
`tests/cli_smoke.rs`). Each verifies exit codes, output, and file creation.

### Testing novomodelo-python

`novomodelo-python` is excluded from the workspace, so address it with
`--manifest-path crates/novomodelo-python/Cargo.toml` (not `-p novomodelo-python`) for any
cargo command — this includes `cargo fmt` and `cargo clippy`, which the
workspace-wide invocations skip. The crate is built with PyO3's
`extension-module` + `abi3` features, which deliberately omit libpython linkage,
so `cargo test` on this crate fails to link by design. The supported test path —
the one CI runs — builds the extension into a virtualenv and runs pytest:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install maturin pytest pyarrow
(cd crates/novomodelo-python && maturin develop --release)
pytest crates/novomodelo-python/tests/
```

GIL-bound conversion helpers (Python↔JSON value mapping, result dict shapes) are
covered by the pytest suite rather than Rust unit tests — extend the pytest
suite when touching them.

### Manual boundary-reduction regression

`policy.boundary` reconciliation has one acceptance criterion that cannot run in CI:
whether the total cost of a boundary-reconciled truncated study agrees with what the same
scenarios cost against the full-horizon study it was sliced from. The deck this check needs
is not in this repository and is not redistributable, so a maintainer runs it by hand
against a deck of their own. Point `NOVOMODELO_BOUNDARY_DECK_ROOT` at a directory holding two
case directories, `full/` and `reduced/`, where `reduced/config.json` sets
`policy.boundary.path` to the checkpoint `full/` will produce. A valid deck has all four of
these properties:

- The reduced case's horizon is a strict prefix of the full case's horizon — sliced to fewer
  stages, not an independently authored short study.
- The reduced case's terminal stage `end_date` is the boundary date the reconciliation
  prices against.
- At least one thermal's `anticipated_config` (`LeadStages` or `LeadTime`) reaches past the
  reduced horizon, so the reconciliation folds in an anticipated-family coupling, not just
  storage state.
- The reduced case declares `post_study_stages.json`, covering at least the ring depth of
  its anticipated commitments — the count of simultaneously open commitments the removed
  stages still need to represent. Without the file, case validation rejects the deck
  outright: a thermal whose lead reaches past the horizon can never deliver, so nothing
  loads. With the file present but short of the ring depth, the load succeeds and the
  zero-dropped-couplings invariant below is what fails; correct the deck before drawing
  any conclusion from the run.

```bash
export NOVOMODELO_BOUNDARY_DECK_ROOT=/path/to/your/boundary-reduction-deck  # not in this repo

# 1. Train the full-horizon case first — its checkpoint is what the reduced
#    case's policy.boundary.path reconciles against.
novomodelo run "$NOVOMODELO_BOUNDARY_DECK_ROOT/full" --output "$NOVOMODELO_BOUNDARY_DECK_ROOT/full/output"

# 2. Validate the reduced case: reconciles policy.boundary without solving.
novomodelo validate "$NOVOMODELO_BOUNDARY_DECK_ROOT/reduced"

# 3. Same check, machine-readable — assert both drop/straddle lists are empty
#    and print the priced boundary date.
novomodelo validate --json "$NOVOMODELO_BOUNDARY_DECK_ROOT/reduced" | python3 -c '
import json, sys
out = json.load(sys.stdin)
assert out["configured"] is True, out
report = out["report"]
assert report["dropped_source_slots"] == [], report["dropped_source_slots"]
assert report["straddling_slots"] == [], report["straddling_slots"]
print("boundary_date:", out["boundary_date"])
'

# 4. Run the reduced case, then read both runs' mean_cost for comparison.
novomodelo run "$NOVOMODELO_BOUNDARY_DECK_ROOT/reduced" --output "$NOVOMODELO_BOUNDARY_DECK_ROOT/reduced/output"
for label in full reduced; do
  python3 -c '
import json, sys, os
label = sys.argv[1]
path = os.path.join(os.environ["NOVOMODELO_BOUNDARY_DECK_ROOT"], label, "output/simulation/metadata.json")
with open(path) as f:
    meta = json.load(f)
assert meta["cost"] is not None, f"missing cost block in {label} run"
cost_mean = meta["cost"]["mean_cost"]
print(f"{label} mean_cost: {cost_mean}")
' "$label"
done
```

Confirm all four before trusting the run:

- The `boundary policy priced at <date>` line `novomodelo validate` printed (step 2) equals the
  reduced case's last stage `end_date`, and the `--json` object's `boundary_date` (step 3)
  agrees with it.
- The `--json` report's `dropped_source_slots` and `straddling_slots` are both empty — the
  same invariant [`crates/novomodelo-sddp/tests/boundary_horizon_reduction.rs`](crates/novomodelo-sddp/tests/boundary_horizon_reduction.rs)
  already covers in CI, restated here as a live-deck spot check.
- Neither `novomodelo validate` nor `novomodelo run` printed a `warning:` line attributed to the
  boundary load. One there is itself a regression: the reconciliation path only ever
  records drops in the report or rejects outright under `policy.boundary.strict`; it never
  warns.
- The reduced run's total cost (`cost.mean_cost` from the run's
  `simulation/metadata.json`, step 4) agrees with
  the full-horizon run's cost restricted to the same stage count: from the full run's
  `simulation/costs/` output, sum each scenario's `immediate_cost` over the stages the
  reduced case also covers, add that last included stage's `future_cost`, and average
  scenario-weighted — that sum is what the reduced run's boundary-priced future cost stands
  in for.

The automated suite already proves the mechanical half of this: a trained full-horizon
checkpoint reconciled into a truncated study with zero dropped couplings, both with and
without a season map. This manual run is the residue — expected-cost fidelity on a deck
large enough for the comparison to be meaningful — not a re-check of the tallies above.

A pre-change baseline cannot be reconstructed to diff against: the
`policy.boundary.source_stage` key the previous implementation read was removed and is now
rejected by `deny_unknown_fields`, and a checkpoint that implementation wrote carries a
`format_version` that `read_policy_checkpoint` now rejects before parsing any payload. The
check is forward-only; the off-by-one this work removed has nothing left to reproduce it
against.

### Project Structure

```
novomodelo/
├── crates/
│   ├── novomodelo-core/         # Entity model (buses, hydros, thermals, lines…)
│   ├── novomodelo-io/           # JSON/Parquet input, FlatBuffers/Parquet output
│   ├── novomodelo-stochastic/   # PAR(p) models, scenario generation
│   ├── novomodelo-solver/       # LP solver abstraction (HiGHS backend)
│   ├── novomodelo-comm/         # Communication abstraction (MPI, local)
│   ├── novomodelo-sddp/         # SDDP training loop, simulation, cut management
│   ├── novomodelo-cli/          # Binary: run/validate/init/schema/version
│   ├── novomodelo-mcp/          # Binary: MCP server for AI agent integration (reserved)
│   ├── novomodelo-python/       # cdylib: PyO3 Python bindings
│   ├── novomodelo-tui/          # Library: ratatui terminal UI (reserved)
│   ├── novomodelo-flow/         # Library: power flow algorithms (reserved)
│   ├── novomodelo-uc/           # Library: MILP unit commitment for hydrothermal dispatch (reserved)
│   └── novomodelo-emt/          # Library: electromagnetic transient analysis (reserved)
├── assets/                  # Logos and diagrams
└── docs/                    # Internal project documentation
```

The full specification corpus lives in the separate [novomodelo-docs](https://github.com/ons-ccee-epe/novomodelo-docs) repository ([deployed docs](https://docs.novomodelo.invalid/)).

## How to Contribute

### Reporting Bugs

Open an issue with:

1. What you did (steps to reproduce, input data if possible)
2. What you expected
3. What actually happened
4. Novomodelo version (`cargo --version`, `rustc --version`, and the crate version)

For numerical issues (wrong results, convergence failures), include:

- The study configuration (`config.json`)
- System size (number of hydros, thermals, stages, scenarios)
- Expected values and source (e.g., "produces X for this input")

### Suggesting Features

Open an issue describing:

- The use case — what problem are you trying to solve?
- Which crate(s) it would affect
- Whether you'd be willing to implement it

For algorithmic enhancements, a reference to the relevant paper or implementation is very helpful.

### Submitting Code

1. **Fork** the repository
2. **Create a branch** from `main`: `git checkout -b feat/my-feature`
3. **Make your changes** — see coding guidelines below
4. **Test**: `cargo test --workspace` (and the CLP backend — see [Solver Backend Selection](#solver-backend-selection))
5. **Lint**: `cargo clippy --workspace --all-targets -- -D warnings`
6. **Format**: `cargo fmt --all`
7. **Push** and open a pull request

#### Commit Messages

We use [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]
```

Types:

- `feat` — new feature
- `fix` — bug fix
- `docs` — documentation only
- `refactor` — code change that neither fixes a bug nor adds a feature
- `test` — adding or correcting tests
- `perf` — performance improvement
- `ci` — CI/CD changes
- `chore` — maintenance (dependencies, tooling)

Scope is the crate name without the `novomodelo-` prefix (use `ferrompi` for the MPI bindings):

```
feat(sddp): implement multi-cut strategy
fix(core): correct reservoir volume bounds validation
docs(io): document Parquet schema for hydro inflows
test(stochastic): add PAR(p) coefficient estimation tests
perf(solver): reduce allocations in basis reuse path
refactor(comm): extract MPI backend into separate module
ci: add cargo-deny license check
chore(ferrompi): update to v0.3.0
```

### Improving Documentation

Documentation improvements are always welcome. Novomodelo's documentation is split
by audience — each kind of doc has one authoritative home, and the others point
to it rather than restating it:

| Documentation            | Home                                                                     | Authoritative for                                        |
| ------------------------ | ------------------------------------------------------------------------ | -------------------------------------------------------- |
| Rust API reference       | rustdoc (inline in source)                                               | Public types, traits, functions                          |
| Per-crate reference      | `crates/*/README.md`                                                     | A crate's responsibility and internals                   |
| Workspace map            | `ARCHITECTURE.md`                                                        | Crate boundaries and the dependency graph                |
| Contributor guide        | `CONTRIBUTING.md` (this file)                                            | Build, test, and PR workflow                             |
| User guides & tutorials  | [docs.novomodelo.invalid](https://docs.novomodelo.invalid/)                          | Installation, CLI and Python usage                       |
| Methodology & theory     | [novomodelo-docs](https://github.com/ons-ccee-epe/novomodelo-docs)                     | SDDP formulation, cut derivation, math specs             |
| Design specs & decisions | `docs/design/` (`docs/design/README.md` is the status index)             | Shipped-behavior specs, decision records, proposals      |
| Coding conventions       | `CLAUDE.md`, `.claude/rules/*`                                           | Enforced comment/import/testing/SDDP contracts           |
| Release history          | `CHANGELOG.md`                                                           | Per-version changes                                      |
| License & notices        | `LICENSE`, `NOTICE`, `THIRD_PARTY_LICENSES.md`, `THIRD_PARTY_NOTICES.md` | Legal obligations (see each file's header for its scope) |

The methodology specification corpus lives in the separate
[novomodelo-docs](https://github.com/ons-ccee-epe/novomodelo-docs) repository; user-facing
guides and reference pages are published from it at
[docs.novomodelo.invalid](https://docs.novomodelo.invalid/).

#### JSON schemas and the novomodelo-docs vendored copy

The input JSON schemas are generated from the `novomodelo-io` Rust types — code is
the source of truth. They are exported with `novomodelo schema export`, committed
under `schemas/`, and CI guards their freshness (`scripts/ci/check_schemas.sh`,
via the `schemas` job in `.github/workflows/ci.yml`). If a schema-bearing type
changes — including adding or renaming a field, changing a
`#[derive(JsonSchema)]` type, or editing a schemars-visible doc comment —
regenerate them:

```bash
cargo build --release --bin novomodelo
./target/release/novomodelo schema export --output-dir schemas
```

The novomodelo-docs site **vendors** these schemas for its Reference pages. When the
schemas change in a release, refresh the vendored copy in novomodelo-docs
(`npm run refresh:schemas` there) so the published reference stays in sync. The
freshness gate stays here in `novomodelo`, next to the generating types; novomodelo-docs
only consumes the exported output.

## Coding Guidelines

### General

- **Run the full check before pushing** (default HiGHS backend; repeat the
  clippy/test steps with `--no-default-features --features clp` to cover CLP —
  see [Solver Backend Selection](#solver-backend-selection)):
  ```bash
  cargo fmt --all
  cargo clippy --workspace --all-targets -- -D warnings
  cargo test --workspace
  ```
- **No `unsafe` without justification.** If you need `unsafe`, add a `// SAFETY:` comment explaining the invariants.
- **No `unwrap()` in library code.** Use `Result` or `Option` with proper error types. `unwrap()` is acceptable in tests and examples.
- **Minimize allocations in hot paths.** The SDDP solver runs millions of LP solves — allocation-heavy code in the inner loop is a performance problem.

### Python Parity

Every output file written by the CLI (`write_training_outputs` / `write_simulation_outputs` in `crates/novomodelo-cli/src/commands/run/outputs.rs`)
must also be written by the Python bindings (`write_training_outputs` / `run_simulation_phase_py` in `crates/novomodelo-python/src/run.rs`).
When adding a new output:

1. Add the `novomodelo_io::write_*` call to the phase's writer on both the CLI and Python paths, before that phase's `write_success_marker` call
2. Run `python3 scripts/ci/check_python_parity.py` to verify parity. The script also fails when any write follows `write_success_marker`, or when a phase writer listed in its `PHASE_WRITERS` is missing
3. The pre-commit hook runs this check automatically

See `.claude/architecture-rules.md` for the full Python parity checklist.

### Crate-Specific Guidelines

#### novomodelo-core

- Types here are shared across all solvers. Changes require careful consideration of downstream impact.
- All public types must implement `Clone`, `Debug`. Implement `serde::Serialize`/`Deserialize` where appropriate.
- Entity collections must be stored in ID-sorted (canonical) order. **Declaration-order invariance** is a hard requirement: results must be bit-for-bit identical regardless of input file ordering.
- Validation logic lives here. A resolved system should be self-consistent — invalid states should be caught at load time, not at solve time.
- The `serde` feature enables JSON serialization for core types. Enable it with `--features serde` or `--all-features` when running tests that cover serialization round-trips.

#### novomodelo-io

- Every parser must have round-trip tests: parse → serialize → parse should produce identical data.
- Include sample input files in `tests/data/` for each supported format.
- The layered validation pipeline (structural → schema → referential → dimensional → semantic) must collect all errors before failing; never short-circuit on the first error.
- Always run `cargo test -p novomodelo-io --all-features` to include the full test suite. Tests gated behind the `serde` feature (from `novomodelo-core`) are required for integration tests.

#### novomodelo-sddp

- Algorithmic changes must reference the relevant literature (paper, section, equation number).
- Numerical changes require validation against reference outputs. Include the test case and expected bounds in the PR.
- The four algorithm parameterization points (risk measure, cut formulation, horizon mode, sampling scheme) must remain generic — no hard-coding of specific strategies.

#### novomodelo-solver

- The `SolverInterface` trait must remain backend-agnostic. HiGHS-specific code stays behind the `highs` feature flag.
- Basis warm-starting affects correctness, not only performance — validate it in tests.

#### novomodelo-comm

- The `Communicator` trait must remain implementable by both backends (MPI and local).
- The local backend's calls compile to no-ops on the hot path — do not add indirection that penalizes single-process users.
- MPI code requires an MPI installation to test; gate MPI tests appropriately.

#### novomodelo-mcp and novomodelo-python

- These crates are **single-process only** — they must never initialize MPI or depend on `ferrompi`.
- `novomodelo-python` must release the GIL (`py.allow_threads()`) during all Rust computation.

### Testing

- **Unit tests** go in the same file as the code (`#[cfg(test)] mod tests`).
- **Integration tests** go in `tests/` at the crate root.
- **Use `approx` for floating-point comparisons:** `assert_relative_eq!(actual, expected, epsilon = 1e-6)`.
- **Property-based tests** (proptest) are encouraged for numerical code.
- **Order-invariance tests:** for any function that processes entity collections, test that reordering the input produces identical output.

### Dependencies

- Prefer well-maintained crates with minimal transitive dependencies.
- New dependencies are checked for license compliance by `cargo-deny` in CI.
- Feature-gate optional heavy dependencies (solver backends, MPI, PyO3, ratatui).

## Domain Knowledge

Novomodelo sits at the intersection of power systems engineering, stochastic optimization, and systems programming. Not all contributors will have expertise in all three areas. That's fine.

If you're a **power systems engineer** new to Rust:

- The [Rust Book](https://doc.rust-lang.org/book/) is the standard learning resource
- Focus on `novomodelo-core` and `novomodelo-io` — these are the most domain-heavy crates

If you're a **Rust developer** new to power systems:

- The [docs site](https://docs.novomodelo.invalid/) has algorithm documentation and a user guide
- Ask questions in [Discussions](https://github.com/ons-ccee-epe/novomodelo/discussions) — no question is too basic

If you're a **researcher** with algorithmic improvements:

- Open an issue describing the algorithm, with references
- We can help translate the math into Rust

## Decision Making

Novomodelo is currently maintained by [@rjmalves](https://github.com/rjmalves). Major design decisions (new crates, breaking API changes, new solver backends) are made through GitHub issues with discussion. As the contributor base grows, we'll formalize this with an RFC process.

## Release Checklist

Before tagging a new release:

1. Update `CHANGELOG.md` with the new version's changes
2. If the release changed deterministic numerical output (solver tolerances,
   scaling, cut selection, basis handling, …), regenerate the parity baselines
   and update the guard table **in the same change** — otherwise the `Test`
   job goes red on the stale baselines:
   ```bash
   cargo nextest run -p novomodelo-sddp --features slow-tests --test parity \
     -E 'test(parity_regen)' --run-ignored ignored-only
   cargo nextest run -p novomodelo-sddp --no-default-features --features "clp slow-tests" \
     --test parity -E 'test(parity_regen)' --run-ignored ignored-only
   # These rewrite the committed baselines under
   # crates/novomodelo-sddp/tests/fixtures/parity_baselines/ (HiGHS) and
   # crates/novomodelo-sddp/tests/fixtures/parity_baselines_clp/ (CLP). The two dirs
   # are the single source of truth and are re-baselined TOGETHER — updating
   # only one lets the other backend's slow-gated suite rot silently. Review
   # the .sha256 diffs and commit them in the same change.
   ```
3. Run quality checks:
   ```bash
   python3 scripts/ci/check_python_parity.py --max 0
   ```
4. If `schemas/` changed since the last release, refresh the vendored copy in
   the `novomodelo-docs` repository (`npm run refresh:schemas` there) so the
   published reference stays in sync — see
   [JSON schemas and the novomodelo-docs vendored copy](#json-schemas-and-the-novomodelo-docs-vendored-copy)
5. Run `cargo fmt --all && cargo clippy --workspace --all-targets -- -D warnings`,
   then repeat clippy for the CLP backend:
   `cargo clippy --workspace --all-targets --no-default-features --features clp -- -D warnings`
   (the two backends are mutually exclusive — see [Solver Backend Selection](#solver-backend-selection))
6. If `policy.boundary` reconciliation changed since the last release, run the
   [Manual boundary-reduction regression](#manual-boundary-reduction-regression) procedure
   against your own deck before tagging
7. Tag: `git tag v<version>`

## License

By contributing to Novomodelo, you agree that your contributions will be licensed under the [Apache License 2.0](LICENSE).
