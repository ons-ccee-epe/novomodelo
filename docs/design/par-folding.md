# PAR folding

> **Status**: Decision record. The stage LP carries no PAR(p) inflow model: the
> realized inflow and the lag terms of the cut rows enter as right-hand-side constants
> computed before each solve, and each new cut's lag coefficients are rebuilt from the
> duals by the chain rule. This record states what the LP folds, the decisions and the
> alternatives they rule out, where each invariant is pinned, what was measured, and
> what is still open. The formulation, its exactness and its limits are in the *PAR(p)
> Folding* chapter of the user documentation (`math/par-folding`); the correctness
> contracts in `.claude/rules/sddp.md`; the builder's obligations in
> `docs/design/lp-builder-contract.md`.

## 1. The folded LP

- **Absent:** each hydro's incoming inflow-lag columns, its realized-inflow column and
  that column's definition row. No column pins a lag (`fill_col_state_patches` skips the
  lag region) and no cut row has a lag entry.
- **The coupling:** a base row that reads an inflow keeps its other entries; its inflow
  coefficients form the unscaled matrix G (`InflowCoupling`, per stage in
  `StageContext::inflow_coupling`), recorded at build time by `InflowCouplingSink`.
  `PrecomputedPar` owns ψ, b and σ.
- **Cut rows** carry θ and the outgoing storage, bucket and commitment-hold columns; the
  truncation inflow-slack column stays an LP column.
- **Before each solve** (`StageSolvePrep::pin_and_bracket`, then `patch_opening`): per
  trial point, p(â) and each cut row's bracket B; per solve, ẑ = p + σ·η
  (`realize_inflow`), the coupled rows' bounds shifted by −Gẑ
  (`PatchBuffer::fill_inflow_shift_patches`), the inflow-slack bounds from l\*(ẑ), and the
  cut-row lower bounds B + W in one `set_row_lower_bounds_range` push.
- **After each solve:** `assemble_outgoing_state` copies the columned state and advances
  the lag slice from â and ẑ (`advance_lag_chain`). A new cut's columned coefficients are
  reduced costs; its lag coefficients come from Gᵀπ and the resident cut duals
  (`cut/chain_rule.rs`).

**The policy** is defined in state space, not in LP columns. The state keeps its lag
slice (`StateSpace::state_dim_lag_range`), so trial states and cuts have lag
coordinates, and the policy file, the cut wire (`cut/wire.rs`), the state exchange and
the export manifest carry them. The partial sum of an open lag period is rank-local and is
not state. Stored bases use the compact layout; a stored basis that does not fit the LP is
dropped with the `stored bases not used` warning and the solve starts cold.

**The expanded LP** — the same stage with the lag columns, the realized-inflow columns
and their rows reinserted — exists only behind `test-support`
(`test_support/expander.rs`), rebuilt from the compact template, G and `PrecomputedPar`,
as the oracle of §3.

## 2. Decisions

### 2.1 Fold the PAR model out of the LP

The incoming lags are the trial state, so ẑ is known before every solve and every LP term
that multiplies a lag or ẑ is a constant. With N hydros and lag depth L
(`StateSpace::max_par_order`), folding removes N(L + 1) columns and N rows from every stage
LP and every lag entry from the cut rows — most of a lag-bearing cut row on a
NEWAVE-converted study — and keeps Q, the optimal dual face and the attainable cuts in
exact arithmetic. A study whose interior pools are storage-only, such as a converted
DECOMP deck, gains the smaller LP only. Work runs per solve only if it depends on the
noise, the trial state or the duals; everything else runs once per stage.

Rejected: the PAR model as LP rows, which prices any inflow model linear in the state with
no model-specific step, where the fold needs a chain rule per model and is specialized to
the PAR(p) family (user docs, "The flexibility trade-off"); the expanded LP as a runtime
option or cargo feature (two production layouts and baselines for one model); deriving the
compact LP from the expanded one by submatrix, presolve or a table between geometries (the
builder emits it with one `StageGeometry` and one address formula per family; builder
contract, "3. Address").

### 2.2 Natural scaling

The compact template is scaled like every template (`postprocess_templates`,
`lp/builder/scaling.rs`). A column scale depends only on its column, so it is the same in
both LPs; a coupled row has one entry fewer and can take another row scale, so the
compact LP is not a scaled submatrix of the expanded one. G and the pool are unscaled, in
cost-scaled units; the row scale is applied where a shifted bound enters the solver and
where a coupled row's dual leaves it. At a degenerate optimum the two LPs can stop at
different vertices, and computed values differ at rounding level: the permitted
hot ≠ cold class. Tests compare invariants, or quantities derived from one LP's own duals
or from one shared basis, never the two formulations bit for bit.

### 2.3 One route for the realized inflow

Every coupling of an inflow to a row is an `InflowCouplingSink::record` call, and no
matrix coefficient depends on ẑ, â or the incoming storage ("Inflow enters only through
the coupling sink", "The realized inflow has one route into the LP"), so the chain rule
differentiates the LP that is solved. A row family that skips the sink defines a wrong LP,
which the coupling tests of §3 catch, never a cut inconsistent with its LP. A new
coupling, such as an FPHA plane reading the inflow, needs no chain-rule change.

### 2.4 The cut-row RHS split by what it depends on

The association and its writers are in `.claude/rules/sddp.md`, "The cut-row RHS has one
association". B depends on the trial point, the child and the cut; W on the opening and
the cut. Every backward opening at a trial point shares one B and adds its W, so the
per-opening work is one contiguous add and one range push. The blocked GEMM kernel is not
used for S: it does not sum a row in ascending slot order. The batched form of
`cut_brackets_into` is kept because the `rhs_kernel_all_cores` group of
`benches/fold_kernels.rs` measures it at least 1.3× faster than the plain row loop with
every core busy; under full-socket load the S pass is bandwidth-bound, so a single-thread
gain does not decide it. Revisit when an all-cores run falls below 1.3× at any row count.

### 2.5 Caches

| Cache | Content | Valid for |
|---|---|---|
| `FrozenBrackets` slab in each `TrajectoryRecord` (stages ≥ 1) | B of the frozen rows, written by the sampled forward, read by the frozen backward | one iteration and node, tagged `(iteration, NodeId)`, at the reader's frozen row count |
| `StageNoiseTable` | σ·η per outcome and W per (outcome, resident row), outcome-major | one node, filled before the claim loop on ranks with trials there |
| Pin once per unit | state pins and B, set per unit, not per opening | one unit |

B is computed from the record's raw trial state, never from the scaled pin.
`freeze_active_cuts_into_templates` runs once per iteration (`run_cut_management`) and
once before the first (`prime_frozen_templates`), so the iteration names the frozen set.
The slab lives from the end of the sampled forward to the start of
`run_cut_management`, for the child the forward visited, and is never persisted. Only the
sampled forward writes it and only the frozen backward reads it; every other site
computes B itself, and the replicated backward and the lower bound also compute W per
solve. A miss recomputes B: a performance event, expected for children the forward did
not visit, counted by `slab_misses` under `test-support` only. Why a hit is right is the
positional premise of "The cut-row RHS has one association".

Rejected, each for a second owner or path against a small saving: a setup table of σ·η
per (stage, opening, hydro); a pre-pass for the delta rows' bracket, which splits
`by_node` from `by_scenario`; a slot-keyed slab under DCS, in the replicated backward or
under enumerated traversal; a W table in the replicated backward, where each W feeds one
solve (revisit if that path gains a trial-point axis); caching p from the forward, the
lower bound's full RHS or the forward's lag product; splitting the base-row shift into
Gp + G(σ·η), which re-associates Gẑ; solving the forward's own opening first, an
in-sample warm-start effect. The noise table stays outcome-major; revisit if its fill
exceeds 1% of backward time on a profiled deck.

### 2.6 The chain rule aggregates first where the payload allows

The rule is linear in the duals and the risk weights depend only on the opening
objectives, so forming ȳ = Σ_ω μ_ω y_ω first and applying the rule once gives the
aggregate of the per-opening cuts in exact arithmetic. `by_node` and `by_scenario` (DCS
included, which `resolve_backward_scheduler` forces to `by_scenario`) aggregate first;
the replicated backward and the fused terminal slice, whose payload is per opening, call
`extract_opening_cut`. The kernels, their zero-sign rules and their pins are in
`.claude/rules/sddp.md` ("Benders cut sign & subgradient extraction", "The cut intercept
dots the trial state through the projection, never positionally", "Chain-rule
accumulators start at negative zero"). The walk Σ_k ȳ_k β_k is row-major: each non-zero
cut row streams once across the lag width into one accumulator per lag slot (the
`chain_rule_walk` bench group measures it against a tiled nest). The opening loop stays in
the ȳ formation: inside the walk it multiplies the work by Ω and changes the association.

### 2.7 Reading cut duals

The read, its zero filter and the binding predicate are in "Binding activity is recorded
only for a configured reader". The read is unconditional where the resident projection
has lag slots because the NEWAVE conversion emits `level1` with no budget, which tracks
no binding activity.

The records are sized by a structural bound. In a basic optimal solution a row whose
logical is basic has dual 0, and the rows with a nonbasic logical are linearly
independent on the basic structural columns. Cut rows touch only θ and the columned
outgoing state, so at most `CutStateProjection::structural_dual_bound` cut duals are
non-zero, whatever the pool size. The bound sizes reservations and is never a capacity,
because its premise is solver behaviour (§4). Rejected: records sized by the resident
cut-row count, which grow with the pool; dense y with a shared index list, which exists
neither under DCS, whose row map is per child, nor on a fan, where each child has its
own slot space.

### 2.8 The cut pool stays flat

Rows are slot-major and append-only; the kernels read each row's lag sub-slice in place.
Rejected: a tiled or compact copy of the active set (a small gain on a small share of the
stage); separate storage and lag blocks (breaks the full-row `gemm_block` contract of the
ownership scan and DCS scoring); an f32 shadow (adds error to every RHS); padding or
alignment (padded lanes can flip the sign of a zero sum).

### 2.9 The by-node finish runs in parallel

The finish reads each opening's record in place through a (position, ω) location table.
Positions are split into one contiguous range per worker; each cut is computed by one
thread into its own staged slot and no value is reduced across threads, so the cuts are
bit-identical for any worker count, and the ascending insert stays serial
(`by_node_finish`). Budget: at most 1 ms per trial position and 5% of the stage's by-node
backward wall time; past either, the next lever is the serial insert.

### 2.10 A lower-bounds-only range push

`SolverInterface::set_row_lower_bounds_range` sets a contiguous range to `[lower, +∞)`;
every cut row, deactivated rows included, has that shape. It is required, with no default:
every solver is wrapped in `ProfiledSolver`, and a default through `set_row_bounds` would
fall back silently and need an index list. Every implementor, test doubles included,
carries it. HiGHS, whose C API has no lower-only call, passes a prefix of an internal +∞
vector (`row_upper_infinity`); CLP writes its bound copies and pushes them through
`push_row_bounds_to_clp`, the helper it shares with `set_row_bounds`. The coupled-row
shifts and the load rows stay on `set_row_bounds`.

### 2.11 Memory and determinism

Buffers are allocated once and reserved serially, growth-only: at construction
(`WorkspacePool::reserve_backward_scratch`), where the pools grow
(`WorkspacePool::reserve_pool_scratch`), and for the slab at freeze. Rayon binds no
workspace to a thread, so locality comes from one rank per NUMA domain with core and memory
binding at launch, not from `madvise`. Each output element is computed by one thread in a
fixed order. No guard against fused multiply-add is needed: rustc never contracts
`a * b + c`, so "no `mul_add`" is a review item.

## 3. Invariants and their pins

Each contract, with its wrong-but-compiling alternatives and its named tests, is a section
of `.claude/rules/sddp.md`:

| Invariant | Section |
|---|---|
| Lag coefficient ψ·g + w, unscaled; Gᵀπ from unscaled duals | "Benders cut sign & subgradient extraction" |
| Intercept after the lag coefficients, through the projection | "The cut intercept dots the trial state through the projection, never positionally" |
| Sign of zero in the chain rule and the RHS dots | "Chain-rule accumulators start at negative zero; cut-RHS dots start at positive zero" |
| Every resident row's RHS at every solve | "Every resident cut row's RHS is written at every solve" |
| A trial's lag chain is its own | "A sampled forward trial owns its lag chain" |
| Lag-point modes, the partial sum, the slab tag | "The cut-evaluation lag point has one owner" |
| One RHS association; caches hold only its values | "The cut-row RHS has one association" |
| No lag column; outgoing state by region | "State pinning uses column bounds, not equality rows" |
| Inflow through the sink; the shift | "Inflow enters only through the coupling sink" |
| ẑ raw; l\* not differentiated | "Truncation keeps the PAR state raw; the correction is a bounded column" |
| One route; one owner of ψ, b, σ | "The realized inflow has one route into the LP" |
| Mandatory dual read; zero filter; structural bound | "Binding activity is recorded only for a configured reader" |
| Misfit stored bases dropped | "Cut pool is append-only; basis matches by slot identity" |
| Fused slice projected to the parent pool | "Fused terminal slice projects with the parent pool, not the leaf pool" |

**Oracles against the expanded LP** (`tests/cut_oracles/`): the mapped compact basis is
optimal there (`mapped_basis_warm_start.rs`); at that vertex the fold meets the expanded
KKT conditions at every cut-row-set shape (`kkt_per_shape.rs`,
`expanded_solve_assembly.rs`); Gᵀπ from the expanded LP's own duals, through the
production kernel, equals the lag and z reduced costs recomputed from its columns, never
the reported ones, which HiGHS sets to exactly 0 on basic columns
(`shared_dual_identity.rs`); independent cold solves agree in objective and each side's
cuts minorize the other's Q (`cross_formulation.rs`). Power guards
(`power_guard_checks.rs`, `fpha_lag_study_checks.rs` on the in-code `fpha_lag_study`)
fail unless a binding cut carries a lag slot, the lag and z column scales differ from 1,
and a binding coupled row's compact row scale differs from its expanded one and from 1.
Tolerances are in unscaled units and every compared solve asserts it needed no retry
(`tests/common/oracle_tolerance.rs`).

**The coupling itself** is outside the oracles, which share G and ψ with production; it
rests on derived expected values: `lp/builder/inflow_coupling/tests.rs`,
`every_study_reads_z_inflow_on_its_water_rows`, `analytical_fold_cuts.rs`,
`chronological_inflow_noise_moves_only_its_own_hydro`, `tests/hydro_inflow_travel_time.rs`
and the d16, d29, d30, d39 and d40 cases of `tests/deterministic.rs`.

**Determinism and allocation:** `compact_fold_determinism` in `tests/mpi_wire.rs`,
`tests/template_permutation.rs`, the cache-bypass runs of `tests/cut_basis.rs`
(`every_lagged_deck_trains_to_the_same_bits_with_the_solve_caches_bypassed` and the
`fpha_lag_study_` variants), `tests/forward_sampler_no_alloc.rs` and
`tests/backward_iteration_no_alloc.rs`.

## 4. Premises that are solver behaviour

HiGHS zeroes the duals of basic variables in `HEkk::getSolution`, and every result the
backend accepts is optimal and basic (simplex rungs return a basis, presolve is followed
by a postsolved basis, IPM rungs keep crossover on). CLP runs simplex on every rung and
refactorizes before its final status check whenever the solve pivoted
(`ClpSimplexDual::statusOfProblemInDual`, unless special option 2048, which no backend
sets). So a row with a basic logical reports ±0.0 on both. Pinned against a solver
upgrade by the `*_basic_rows_report_a_zero_dual_*` tests of
`crates/novomodelo-solver/tests/conformance.rs`, and the bound of §2.7 by
`cut_dual_counts_stay_within_the_structural_bound` and its slow sweep over every lagged
deck. CLP's perturbation, scaling and primal retry rungs are not exercised; a miss there
costs records past the reservation, never a wrong cut.

## 5. What was measured

One local comparison, kept outside the repository and not reproduced by a committed test,
ran the fold against a same-version build whose LP defines the inflow in rows, on three
reduced converted decks (sampled NEWAVE, enumerated DECOMP, deterministic NEWAVE), each
trained for a short run:

- every lower bound stayed below its upper bound;
- on the sampled deck the fold's lower bound fell within the other build's two-seed
  spread; on the other two that spread is zero and the bounds differed, because the two
  LPs stop at different vertices of a degenerate optimal face at the first iteration,
  before any cut, and then train different valid cuts;
- each build simulating the other's policy reproduced the owner's stage-0 objective within
  1e-9 relative on every scenario of every deck, in both directions, after dropping the
  other build's stored bases with the `stored bases not used` warning;
- profiled per `docs/profiling.md`, LP time per solve fell on both passes at an unchanged
  simplex count per solve, the fold's kernels stayed far below 1% of cycles with no new
  hotspot, the by-node finish stayed far inside the budget of §2.9, the slab never missed,
  and `rhs_kernel_all_cores` met its 1.3× gate at every row count benched.

## 6. Open checks and limits

- **Production comparison, pre-registered, not run.** On the converted NEWAVE PMO deck
  and converted DECOMP decks (the latter measure the smaller LP only), with identical
  configuration, build settings and rank placement, against seeds A and B of a build whose
  LP defines the inflow in rows: |LB_fold − LB_A| ≤ |LB_B − LB_A|; the paired per-scenario
  cost difference D̄ ± z·s_D/√S within the A-vs-B statistic; each policy simulated by both
  builds, which separates the formulation from the training path and decides where the
  seed spread is zero (§5); timing at equal iteration counts with cold loads, warm solves
  and cut selection apart; determinism within each build only. A policy written by
  another version is refused for every load kind ("A policy written by other software, or
  another version, is refused first"), so the cross-simulation needs two builds of one
  version; two releases compare on training and timing only. The run also confirms the
  finish budget of §2.9 at production scale.
- **Partial sums that differ between paths.** A cut at a stage closing a lag period begun
  earlier is exact only where the partial sum is the same on every path ("The
  cut-evaluation lag point has one owner", "The limit"); exact cuts would carry the partial
  sum in the state, changing the state space and every cut and policy layout. The same
  section states what lifting `TrainingSession::new`'s rejection of interior branching
  under enumerated traversal at world ≥ 2 requires.
- **Truncation.** l\* is a column bound the chain rule does not differentiate; the
  inflow-slack column is not folded.
- **FPHA lateral inflow.** No FPHA plane reads the inflow; the lateral-flow LP wiring is
  open (`production/fpha_fitting/secant.rs`) and couples through the sink (§2.3).
- **`dynamic` cut selection** has correctness coverage on the folded path, not a
  performance comparison.
