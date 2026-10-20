# novomodelo-sddp

Stochastic Dual Dynamic Programming (SDDP) algorithm for the [Novomodelo](https://github.com/ons-ccee-epe/novomodelo) power systems ecosystem.

Implements the SDDP algorithm (Pereira & Pinto, 1991) for long-term hydrothermal
dispatch and energy planning. The crate covers the full solve cycle: forward
pass scenario simulation, backward pass Benders cut generation, cut management
with cut selection (the `level1` method and the lazy `dynamic` method), CVaR risk
measures, convergence monitoring, policy warm-start and resume from checkpoint,
and annual discount rate support. Designed for hybrid MPI + thread-level
parallelism via [ferrompi](https://github.com/cobre-rs/ferrompi).

## When to Use

Use `novomodelo-sddp` when you need programmatic access to the SDDP algorithm —
embedding it in a custom orchestration layer, running parameter sweeps, or
integrating the solver into a larger application. For single-study command-line
use, prefer `novomodelo-cli`, which wraps this crate.

## Key Types

- **`TrainingConfig`** — algorithm parameters in three parts: loop settings
  (`loop_config`), cut management (`cut_management`), and the event channel
  (`events`)
- **`TrainingContext`** — runtime state shared across all iterations of the
  training loop (cut pools, workspaces, convergence monitor)
- **`CutPool`** — pre-allocated storage for Benders cuts with active/inactive
  bookkeeping
- **`CutSelectionStrategy`** — enum controlling cut selection, with two variants:
  - `Level1 { check_frequency, lookback_iterations }` — at each stored trial state the cut
    with the highest value is active, ties go to the lowest slot, and a cut that owns no
    state is deactivated, never deleted; it runs every `check_frequency` iterations, and
    `lookback_iterations` limits the states to a window of the last iterations
  - `Dynamic { candidate_recency, seed_window, max_added_per_round, violation_tolerance,
    start_iteration }` — a lazy solve that starts from the binding cuts (`seed_window`) and
    adds violated candidates (`candidate_recency`, `violation_tolerance`), up to
    `max_added_per_round` per round, until none is violated; it begins at `start_iteration`
    and never deactivates a cut

- **`SimulationConfig`** — parameters for the post-training simulation run
- **`ConvergenceMonitor`** — tracks lower bound, statistical upper bound, and
  gap closure across iterations

## Cut selection

`level1` follows Level 1 of de Matos, Philpott and Finardi (2015), with two departures: the
root pool is never selected, and selection runs every `check_frequency` iterations rather
than whenever a cut is added. The rustdoc of `cut::cut_selection` states the rule, the
invariant and what is said of convergence. `dynamic` is this software's own lazy row
generation, not the method of that paper.

`level1` stores the trial state of every cut a run generates, in every interior pool (stage 1
or later, with a successor). Each rank holds the same full copy. A pool reserves up to
`capacity × state_dimension × 8` bytes plus 20 bytes per slot. Here `capacity` is the loaded
cut count plus the iteration limit times the trial states routed to the node per iteration,
and `state_dimension` is the pool's projected dimension. The root pool stores states only
when `exports.states` is on. With `lookback_iterations = w` the store holds
`min((w + check_frequency) × visit_stride, capacity)` rows, where `visit_stride` is the
number of trial states routed to the node per iteration, so the window bounds the rows, not
the cuts: a cut whose state left the window stays in the pool and competes at the states that
remain. To measure a window on your own case, run `scripts/measure_cut_selection.py`; its
docstring lists the variants and the report columns.

A resume continues the uninterrupted run. The cuts, the activity flags, the binding counters,
the stored trial states and the bases are identical, bit for bit, to those of the run that was
never interrupted. The lower-bound series agree within solver tolerance, so a stopping rule
that reads the lower bound can stop the two runs one iteration apart on its exact threshold.
The guarantee holds for the same build of the program, and the rank count may change between
the interrupted run and the resume. A resume refuses a checkpoint that cannot continue the
run: one written with another forward-pass count, one without the trial-state history or
window that `level1` needs, one without the binding counters that a budget or `dynamic`
reads, one whose bases are incomplete, and one written by a run that ended in an error. A run
that declares `policy.boundary` keeps the terminal binding counters on a resume. A resume
that declares `policy.boundary` is refused when the reconciled boundary rows differ from the
terminal pool of the checkpoint; a changed `modeling.cost_scale_factor` is one way that
difference can arise. The resume applies its own cost scale from then on.

A `policy.boundary` pool contributes its active rows to training, and the terminal pool is
never selected. Under `simulation.cut_set = all` the pool also contributes the rows its
writer scored, whatever their flag. `dynamic` refuses
`simulation.cut_set = all` when the run simulates and a loaded pool, or the boundary pool,
holds an inactive row that `all` includes, because the `all` set then differs from the
active set and a `dynamic` solve reads only the active cuts. For a policy loaded to resume
or only to simulate, those are the rows selection deactivated or the cap evicted. For a
warm-start policy and the boundary pool, only scored inactive rows are included, so inactive
rows that were never scored, such as those a capped source leaves, do not trigger the refusal.

## Error handling (`SddpError`)

All fallible operations return `Result<T, SddpError>` (`Send + Sync + 'static`).
The rustdoc of each `SddpError` variant states when it is raised; the API docs
in [Links](#links) list every variant.

## Feature flags

| Feature      | Default | Description                                                                                                                                                                     |
| ------------ | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `highs`      | on      | Selects the HiGHS LP solver backend (`novomodelo-solver/highs`); gates the per-phase `HighsProfile` impl and the `FORWARD_PROFILE`/`BACKWARD_PROFILE`/`SIMULATION_PROFILE` constants |
| `clp`        | off     | Selects the CLP/CoinUtils LP solver backend (`novomodelo-solver/clp`); gates the per-phase `ClpProfile` impl                                                                         |
| `dhat-heap`  | off     | Enables DHAT heap profiling — replaces the global allocator, so never enable by default; only for the `dhat_baseline` example                                                   |
| `slow-tests` | off     | Opts in to the slow test suite (D-case sweep, FPHA plane-selection, forward-sampler convergence), ignored by default so `cargo test --workspace` stays fast                     |

## Testing

```
cargo test -p novomodelo-sddp
```

No external system libraries are required beyond the workspace default (HiGHS
is always available; MPI is optional via the `mpi` feature of `novomodelo-comm`).
The suite covers unit tests for each module's core logic, integration tests
using `LocalBackend` (single-rank) for the communication-involving modules,
and doc-tests for all public types with constructible examples.

## Links

| Resource                    | URL                                                        |
| --------------------------- | ---------------------------------------------------------- |
| Docs site                   | <https://docs.modelossetoreletrico.org.br/>                      |
| Repository                  | <https://github.com/ons-ccee-epe/novomodelo>                        |
| Changelog                   | <https://github.com/ons-ccee-epe/novomodelo/blob/main/CHANGELOG.md> |

## Status

**Alpha** — API is functional but not yet stable. See the [main repository](https://github.com/ons-ccee-epe/novomodelo) for the current release.

## License

Apache-2.0
