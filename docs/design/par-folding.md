# PAR folding

> **Status**: Proposal — not implemented. This is the target design for removing the
> PAR(p) inflow model from the stage LP: the builder stops emitting the lag columns,
> the z columns and the z rows, and their effect enters as right-hand-side constants
> computed outside the solver. The work lives on the long-lived branch `folding`,
> where the compact LP replaces today's LP outright (no runtime switch, no cargo
> feature). The branch is evaluated against main in production before any merge is
> considered. Figures quoted here are calibration-time estimates for the converted
> NEWAVE PMO deck, not standing claims.
>
> **Companions**: `docs/design/lp-builder-contract.md` (the four jobs of the builder,
> which this design amends in Jobs 1, 2 and 4, §8.2); `.claude/rules/sddp.md` (the
> correctness contracts restated and added in §8.1);
> `crates/novomodelo-sddp/src/cut/basis_reconstruct.rs` (the two basis entry points,
> unchanged in kind).

## 1. Context

Today each stage LP carries the PAR(p) model itself:

- **Incoming lag columns** `a_{h,l}` for every hydro h and lag l < L, pinned by column
  bounds to the trial state. There are N·L of them (L = 12 under PAR(p)-A).
- **One z column and one z row per hydro**, `z_h − Σ_l ψ_{h,l} a_{h,l} = b_h + σ_h η_h`.
  The RHS is rewritten every solve by `stochastic/noise.rs::transform_inflow_noise` and
  `lp/builder/patch.rs::fill_z_inflow_patches`.
- **z feeds** the hydro's water-balance rows (or the first non-PreFilling downstream
  target's), generic constraints with a `HydroInflow` term, and every cut row as lag 0.
- **Every cut row carries lag coefficients.** Through
  `StateSpace::state_to_lp_column`, lag slot (h, 0) sits on z_h and lag slot (h, l ≥ 1)
  on the incoming column `a_{h,l−1}`. On the converted NEWAVE PMO deck the cut-state
  dimension is about 2,184, of which 2,016 (92%) are lag slots, so the lag part of a cut
  row has 12× the entries of its storage part.
- **Cut coefficients are reduced costs** of the pinned incoming columns divided by the
  column scale (`training/backward/duals_extraction.rs::extract_duals_from_view`).

The lags are known before every solve: they are the trial state. So ẑ can be computed
outside the LP, and every term that multiplies a lag or ẑ is a constant. Folding moves
those constants into the RHS and rebuilds the lag coefficients of each new cut by the
chain rule. Converted DECOMP decks gain from the smaller LP only: their interior pools
are storage-only, so their cut rows carry no lag terms today.

`novomodelo-stochastic/src/par/evaluate.rs::evaluate_par_batch` already evaluates PAR
outside the LP for truncation, and agrees with the LP z to 2.8e-14 on case D30.

## 2. Goal, success criteria and decisions

**Goal.** An exact fold: the same optimal value and the same set of attainable cuts as
today, a smaller LP at every solve site, and all derivable work moved to setup.

**The per-solve rule.** Work runs per solve only if it depends on per-solve data: the
noise η (per opening), the trial state â (per trial point), or the duals. Everything
else is computed once per stage, by the builder or at setup.

**Success criteria.**

- Every test in §9 passes on HiGHS and CLP, including the determinism gates with their
  power preconditions.
- The local performance check (§10) shows LP time dropping, the out-of-LP kernels as a
  small share, and no new hotspot.
- The production A/B (§11, run outside the implementation plan) meets the acceptance
  rules written down in §11.3 before any run.

**Decisions already taken.**

| Topic | Decision |
|---|---|
| Fold mode | Replaces today's LP on `folding`; no switch, no feature flag |
| Shape | The builder emits the compact LP natively (§4). During the branch an internal builder parameter can still emit today's LP as the transition oracle; it is never config and never a runtime switch, production solves one LP at every commit, and the parameter is deleted at the switch-over (§4.7) |
| Scaling | The compact LP gets the natural geometric-mean scaling of every template (§3.2) |
| Solve sites | One shared folded solve primitive, after today's duplicates are removed on today's LP (§5.2) |
| Branch | Named `folding`, no prefix; does not track main; main is merged in only at releases, so the branch inherits main's `CARGO_PKG_VERSION` |
| Solver backends | HiGHS is the performance target; CLP is correct, not tuned (no shim) |
| Cut-row push | A required lower-bounds-only range push; the rows become `[lower, +∞)` (§7.1) |
| Paging and NUMA | Two code rules plus launcher settings (§6.6); no `madvise` helper |
| Production acceptance | Seed-spread yardstick: the branch must sit within main's own seed-to-seed spread (§11.3) |
| Cross-build check | Same-policy cross-simulation in the production A/B; no cross-build resume |
| Performance reporting | Existing `timing`, `solver_stats` and `convergence` parquet columns only; no new output columns |

## 3. The folded formulation

### 3.1 Notation

All quantities are in cost-scaled units (divided by `cost_scale_factor` κ), the units
the pool stores. A tilde marks solver (scaled) space: `c_j` is the column scale, `r_i`
the row scale, `Ã = D_r A D_c`, `π_i = r_i π̃_i`. Cut rows have row scale 1, so their
duals are `y_k = ỹ_k`. Row duals are ∂obj/∂(row bound) and reduced costs are
`rc_j = c_j − Σ_i A_ij π_i`, the convention of `.claude/rules/sddp.md` "Benders cut sign".

Three LPs appear below. **Today's LP** is the stage LP main builds, with the lag
columns, the z columns and the z rows. The **compact LP** is the one this design
builds and solves. The **expanded LP** is today's LP rebuilt from the compact one by a
test-support expander, for the oracles (§4.7).

### 3.2 What the solver sees

At stage t with trial state (v̂, â) and opening η:

```text
p_h   = b_h + Σ_l ψ_{h,l} â_{h,l}             once per trial point
ẑ_h   = p_h + σ_h η_h                          once per solve
base row r:  bound̃_r = r_r · (bound_r − Σ_h G_{r,h} ẑ_h)    both finite bounds; bound_r is the unscaled template bound
cut row k:   row_lower_k = α_k + Σ_{j ∈ lag slots of P_t} β_{k,j} Φ_{t,j}(â, ẑ),  row_upper_k = +∞
```

`G_{r,h} = A_{r,z_h}`: the unscaled coefficient the builder records for hydro h's
inflow in row r through the inflow-coupling sink (§4.2), at the sites that write z
today. It is in row units per m³/s, and ẑ has no column scale, because the compact LP
has no z column. The shift is accumulated from +0.0 in ascending hydro id, and the row
scale is applied only where the bound crosses the solver boundary, with the compact
template's r_r, as `PatchBuffer::fill_load_patches` scales a load bound. In unscaled
units `Σ_r G_{r,h} π_r` is the quantity today's LP gives through its z column,
`Σ_r A_{r,z_h} π_r`. In solver units each LP pairs its own scaled entries with its own
duals: coupled rows carry different row scales in the compact and the expanded LP
(below), so a compact scaled entry times an expanded dual is wrong. Before scaling,
the compact LP is today's LP with each z column fixed at ẑ_h and eliminated into the
bounds of the rows it touches, and with the lag columns and the z rows dropped: the
elimination a presolve would perform, done by the builder.

Φ_t is the lag point at which stage t's cut rows evaluate the successor cut. It is the
alias that `StateSpace::state_to_lp_column` encodes in today's LP: Φ_{(h,0)} = ẑ_h and
Φ_{(h,l)} = â_{h,l−1} for l ≥ 1. It does not depend on the multi-resolution
accumulator, which only builds the outgoing state vector. So reproducing today's
cuts needs no accumulator weights.

Today's lag and z columns have objective coefficient exactly 0, and the shift adds no
constant offset, so the compact LP's optimal value Q equals today's.

**Scaling.** `setup/template_postprocess.rs::postprocess_templates` scales the compact
template as it scales every template (`lp/builder/scaling.rs`): geometric-mean column
scales `1/sqrt(max·min)` over each column's entries, then row scales over each row's
column-scaled entries. A column scale depends only on its own column, so every kept
column keeps today's scale. A row that carried a z entry (water balance, generic
`HydroInflow`) loses that entry and can get a different row scale; every other row
keeps today's. The compact LP is therefore not a scaled submatrix of today's LP, and
nothing derived from today's scaled matrix survives in production. The cut
coefficients the pool stores and the coupling G are unscaled, in cost-scaled units, so
neither depends on the scaling. Q and the optimal dual face are unchanged in exact
arithmetic. At a common basis, computed values move at rounding level. Near a
tolerance tie the declared-optimal basis can differ, because the solver's tolerances
apply to coupled-row residuals that are scaled differently (the class of §3.8). The
parity re-baseline (§9.6) covers both.

### 3.3 The chain rule

Let R be the kept base rows and π their unscaled duals, `π_r = r_r π̃_r` as
`solve/stage_solve.rs::fill_unscaled_dual` forms them, and y the resident cut-row duals
(cut rows have row scale 1, so `y_k = ỹ_k`). For each hydro h, in unscaled cost-scaled
units:

```text
g_h        = − Σ_{r ∈ R} G_{r,h} π_r  +  Σ_k y_k β_{k,(h,0)}
β^new_{h,l} =   ψ_{h,l} g_h          +  Σ_k y_k β_{k,(h,l+1)}      (the second sum is absent for l = L−1)
```

- Every term is already in the unit the pool stores. Do not divide by any `col_scale`
  and do not multiply by κ.
- g_h equals today's z-row dual. In today's LP the z-row dual carries the solver's
  `rc_z` (at most the dual tolerance), so `β^today − β^fold = −ψ rc_z^today`: the fold
  is at least as accurate as today.
- Storage, bucket and commitment coefficients are unchanged: they remain
  `r̃c / c` at their incoming columns, over the same kept rows.
- **No other term exists.** The incoming state enters the compact LP only through
  right-hand sides and column bounds: no matrix coefficient depends on ẑ, â or the
  incoming storage (FPHA planes, evaporation slopes, backwater levels and filling
  targets are fixed at setup, and no coefficient is set per solve). The realized
  inflow has no column in the compact LP, so a row can depend on it only through an
  entry recorded by the inflow-coupling sink (§4.2), and G is exactly the set of
  those entries. A row family with no coupling entry, such as FPHA or evaporation,
  reaches the lag coefficients only through π and y, and the storage coefficient
  through `r̃c / c` at `storage_in`. The chain rule therefore differentiates the LP
  that is solved: a row family that should couple to inflow and does not call the sink
  gives a wrong LP, which the transition oracle (§4.7) and the closed-form tests (§9.2)
  catch, never a cut inconsistent with its LP. l\* (§3.6) is the one state-dependent
  quantity outside the LP, and it is not differentiated.
- **Slots map by global state dimension, never by position.** The resident pool's
  projection P_t (stage t+1's `state_config`) and the extracted pool's projection
  P_{t−1} can differ (d43; DECOMP boundary leaves with `FULL_STATE_CONFIG`). β^new is
  written at P_{t−1}'s slot for global dim (h, l); the w term reads P_t's slot for
  (h, l+1) and contributes 0 if P_t lacks it; g_h reads P_t's slot for (h, 0) if present.
- **General form,** through the Φ_t owner (§4.4): with `M_z = ∂Φ_t/∂ẑ`,
  `M_a = ∂Φ_t/∂â` and `w = Σ_k y_k β_k^lag`, `g = −Gᵀπ + M_zᵀ w` and
  `β^new,lag = Ψᵀ g + M_aᵀ w`. Today `M_z = E_0` and `M_a` is the shift.

### 3.4 Intercept

`α^new = Q − Σ_j β^new_j x̂_j`, through `CutStateProjection::dot_trial_state`, computed
only after the lag coefficients are rebuilt. Computing it while the lag β are still 0
gives a cut that is tight at x̂ with the wrong slope.

### 3.5 Primals and the outgoing state

- ẑ has one owner: the PAR evaluator in `novomodelo-stochastic`. It is split into p
  (per trial point) and ẑ = p + σ·η (per solve), accumulated in today's
  `evaluate_par_batch` order (base, then ψ·lag in ascending l, then σ·η), so the split is
  bitwise equal to today's single call. The split lives in the evaluator, not in a
  second implementation.
- That one ẑ feeds the base-row shifts, the truncation bound l\*, the outgoing lag 0, DCS
  scoring and the simulation inflow outputs.
- The outgoing state takes storage, buckets and commitments from the compact primal at
  their columns (§4.3), and the lag slice from the unchanged accumulator fed from â and
  ẑ instead of from LP primals.

### 3.6 Truncation

The inflow-slack column stays in the LP with bounds from l\*(ẑ) = max(−ẑ, 0), computed
from the same ẑ. The chain rule does not differentiate l\*; today's LP does not either,
because l\* enters as a column bound. The fold reproduces today's cut, including its
known limitation.

### 3.7 CVaR

`convergence/risk_measure.rs::aggregate_cut_into` weights openings by
μ = (1 − λ)p + λν, with ν from sorting the opening objectives. The aggregated cut is
linear in the per-opening duals and μ depends only on the objectives, so aggregating
the duals first and running the chain rule once gives the same cut in exact arithmetic
(§5.3, the aggregate-first call shape).

### 3.8 What is identical and what is not

Identical in exact arithmetic: the feasible set projected on the kept columns and θ;
Q; the optimal dual face on the kept rows, in unscaled units; hence the set of
attainable cuts; the non-lag coefficients at a given vertex; the intercept given β.

Not identical: the vertex the simplex selects at a degenerate optimum (a smaller LP
with different row scales follows a different pivot path), and therefore every
downstream iterate, the CVaR ordering at ties and the DCS resident set near tolerance;
and every computed value at rounding level, because coupled rows are scaled
differently (§3.2). This is the permitted hot ≠ cold class; tests assert invariants or
compare quantities derived from the same duals or the same basis.

## 4. Architecture: the compact LP, built natively

### 4.1 What leaves the builder

The builder stops emitting the incoming lag columns (`columns.rs::fill_ar_lag_columns`),
the z columns (`columns.rs::fill_z_inflow_columns`), the z rows
(`rows.rs::fill_z_inflow_rows`) and their entries (`entries.rs::fill_z_inflow_entries`).
Their per-solve patches go with them: `PatchBuffer::fill_z_inflow_patches` and the lag
pins in `PatchBuffer::fill_col_state_patches`. Nothing derives the compact LP from a
larger one: the template the builder emits is the template the solver loads.

### 4.2 The inflow-coupling sink

- **The writers.** The two sites that write z today record the inflow coupling
  instead: `entries.rs::push_local_inflow_coupling` (the water-balance rows of the
  hydro or of its PreFilling target: one stage-total −ζ under parallel blocks, one −τ_k
  per block row under chronological blocks) and
  `generic_constraints.rs::push_local_inflow_rate` (a generic `HydroInflow` term). Each
  records (base row, hydro, coefficient) into one inflow-coupling sink, unscaled, as
  the builder writes every matrix entry. `generic_constraints.rs::resolve_hydro_inflow`
  returns a generic row's hydro-keyed inflow terms apart from its column terms (slack,
  diversion), and the unchanged `entries.rs::push_merged_row_entries` merges each list
  on its own, as it merges a column's entries today (summed in push order, exact zeros
  dropped), so each coupling entry is bitwise today's z-column entry. Its
  `z_inflow.is_empty()` guard becomes the condition it stands for, a study with no
  hydro: on the compact LP, where `z_inflow` is empty, it would drop every
  `HydroInflow` term, slack and diversion included.
- **Sink first, on today's LP.** The sink arrives before any compact LP exists, in the
  refactor of §5.2. The writers always fill it, and today's emission builds each z
  column, `col_entries[z_h]`, from the sink's entries for h before the per-column sort;
  rows are unique per column, so the order before the sort does not matter. Parity
  hashes that do not move then show that the sink equals main's coupling. This is the
  anchor that ties G to main; no golden from main is kept.
- **The only route.** The compact LP has no z column to write against, so the sink is
  the only way the builder can reference the realized inflow.
- **Its output is G,** a hydro-major CSR, hydro → (base row `u32`, coefficient `f64`),
  in canonical hydro-id order with rows ascending within a hydro. G is stored
  unscaled, beside the unscaled template bounds of the rows it touches; there is no
  postprocess pass on G. The row scale is applied only at the solver boundary: the
  base-row shift scales the whole shifted bound (§3.2, §6.4), and the extraction
  unscales the duals before forming Gᵀπ (§5.2). The same G serves production, the
  expander and every oracle, so no scale-mixed pair can be formed.
- **A debug assertion in the sink** checks each entry: a base row of the stage being
  built, never a cut row; a hydro index in range; a finite coefficient. No setup
  refusal is needed. With no z row, the LP holds no copy of ψ that could disagree with
  the evaluator's: ψ, b and σ have one owner, `PrecomputedPar`, which the PAR evaluator
  and the chain rule both read. With no lag or z column, there is no column whose
  objective coefficient could be nonzero.
- **Cut rows never reach the sink.** `cut/row.rs::push_cut_row` and
  `training/forward/delta_cut_batch.rs::build_delta_cut_row_batch_into` write only the
  slots that have an LP column (storage, buckets, commitments). The lag slots, which
  today alias to z and to the lag columns, go to the RHS (§5.1).

### 4.3 One layout

- **The state slice and the LP columns are split.** `StateSpace.inflow_lags` names
  both the lag columns of today's LP and the lag slice of the state vector. The state
  slice is `StateSpace::state_dim_lag_range`, which already exists; the LP-column range
  is deleted with the columns, and each user of `inflow_lags` moves to the one it
  means. `StateSpace::z_inflow`, `StateSpace::z_inflow_rows` and the `z_inflow_row`
  accessors are deleted.
- **The geometry is native.** With those ranges gone, the cursors in
  `lp/builder/layout.rs` allocate the compact layout directly: base rows start at
  row 0 instead of after `z_inflow_rows()`, and `StageGeometry` describes the LP that
  is solved. There is one geometry and one address formula per family (builder
  contract Job 3), and nothing to translate: no second geometry, no boundary table, no
  typed pairs of logical and solver indices, no restricted accessors.
- **State columns.** Storage columns still sit at their state index. The outgoing
  bucket and commitment columns follow storage directly, so they no longer sit at
  their state index; `state_to_lp_column` loses its lag arm and resolves them.
  `assemble_outgoing_state` copies the storage block and the bucket-and-commitment
  block from the primal per region, through an `outgoing_block_start` that mirrors
  `StateSpace::incoming_block_start`, and writes the lag slice from the accumulator
  (§3.5). It never copies a primal prefix: today's `unscaled_primal[..n_state]` copy
  assumes that LP column equals state index over the whole state.
- **Cut emission and slot maps live on `CutStateProjection`.** It already holds each
  slot's incoming and outgoing column, `render_pairs` and `global_state_index`. It
  gains, built at setup, the lag-slot ranges of §4.4 and the chain rule's slot maps
  between the resident projection P_t and the extracted projection P_{t−1}, keyed by
  global state dimension. `render_pairs`, `incoming_column` and `outgoing_column` cover
  only the slots that have an LP column, so `push_cut_row`,
  `build_delta_cut_row_batch_into` and `extract_duals_from_view` keep their shape.
- **The policy export reads by state index.** `policy/policy_export.rs` classifies a
  slot through `StateSpace::classify(projection.global_state_index(slot))` instead of
  `classify_incoming_column(projection.incoming_column(slot))`, so it no longer depends
  on the LP layout. The manifest is defined in state space, so its bytes do not change.
  `classify_incoming_column` loses its last caller and is deleted.

### 4.4 One owner for the cut-evaluation lag point

Φ_t, M_z and M_a are owned in one place, and both the cut-RHS function (§5.1) and the
chain rule (§3.3) read it. Today it encodes the fixed alias. The owner resolves the
alias at setup into slot ranges per projection: the lag-0 slots, read against p and
σ·η, and the lag ≥ 1 slots, read against â one lag back. The kernels read those ranges
and write into the caller's buffer; no kernel consults the owner per element.

A later change that evaluates lag 0 as a month average when a month spans several
stages starts at this owner but does not end there. At a non-closing stage M_z = 0 and
M_a = I, so the chain rule reads slot (h, l) instead of (h, l+1) and the RHS loses its
P and W terms: the kernels' operands change, not only the owner. That change also has
open questions of its own (§12).

### 4.5 Buffers of record

Each worker holds the out-of-LP values in existing `ScratchBuffers` fields, renamed for
their role: `lag_matrix_buf` holds â and `par_inflow_buf` holds ẑ (today's bits, §3.5).
In the backward, â is the lag slice of the pinned trial state, borrowed rather than
copied. p, written once per trial point, is the one added vector. The base-row shifts
are written straight into the `PatchBuffer` region that replaces the z-row patches, so
there is no separate shifted-bounds buffer, and `z_inflow_rhs_buf` leaves production.
That region is sized from the maximum over stages of the distinct rows G touches, not
from N: under chronological blocks G touches every block row of a hydro's water
balance, and PreFilling target rows and generic `HydroInflow` rows add more. No consumer reads a lag or inflow value from anywhere else.

### 4.6 One-way doors

Unchanged: the state space and the state vector (dimensions, region order, `n_state`);
the cut pool, its rows and its append-only slot identity; the policy file format; the
MPI cut wire (`cut/wire.rs`) and the state exchange; the policy-export manifest. The
stored and broadcast basis changes by length only (§5.9), and the solver trait gains
one required method (§7.1).

Not doors: `StageGeometry` is rebuilt for the compact layout (§4.3). The canonical
template digest (`lp/builder/template/canonical.rs`) is compiled only for tests and
`test-support`, where it guards the declaration-order tests; it is not a stored format,
and it changes with the template.

### 4.7 Transition: the temporary builder arm

- **The arm.** During the branch the builder takes an internal parameter that selects
  today's emission or the compact one. It is never config and never a runtime switch;
  outside the crate it is reachable only behind the `test-support` feature, so the
  transition oracle and the folded-path integration tests can build a compact-LP
  study. Production builds today's LP until the switch-over
  and the compact LP from then on, so production solves one LP at every commit. Until
  the switch-over the folded solve path (§5) runs only in tests, on the compact LP the
  arm emits.
- **The transition oracle.** A transient test builds both emissions of every study
  (the committed decks and `tests/common/in_code_studies.rs::structural_studies`) and
  asserts that the compact template equals today's template with the lag columns, the
  z columns and the z rows removed, bitwise before scaling (dimensions, CSC, bounds,
  objective), through an order-preserving deletion of today's removed ranges, never
  through the compact `StageGeometry`; that `StageGeometry` decodes every kept row and
  column to the same entity in both; and that, after `postprocess_templates`, every
  kept column's scale and every uncoupled row's scale equal today's bitwise. The
  scaling check runs on a study with a non-empty `commit_out`, because the
  commitment-hold column-scale override addresses `StateSpace` commitment column
  ranges, which move with the column region. The sink needs no comparison here: both
  emissions read it, and the parity hashes tie it to main (§4.2).
- **The expander.** A test-support expander rebuilds today's LP from the compact
  template, the sink's unscaled G and `PrecomputedPar`, reinserting the lag columns,
  the z columns and the z rows at today's positions, and scales itself through
  `postprocess_templates`. It has three parts, each checked bitwise against today's
  emission on every study before that emission is deleted:
  - the template, unscaled and after `postprocess_templates`;
  - the cut rows: a lag-bearing pool frozen into the templates, against
    `cut/row.rs::build_cut_row_batch_into`, with a power guard that requires a
    lag-bearing cut at every stage that has a successor;
  - the per-solve patches: the z-row RHS and the lag pins at sampled (â, η), against
    today's `PatchBuffer`.

  Its lag alias, (h, 0) → z_h and (h, l) → a_{h,l−1}, is its own code, moved from the
  lag arm of `state_to_lp_column`. It never reads the Φ_t owner, the lag-slot ranges of
  `CutStateProjection`, the canonical RHS function, the chain-rule kernel or the compact
  scales, so the oracles built on it are not common-mode on the lag alias.
- **The switch-over** is one commit. It flips production to the compact LP, deletes
  the old emission with the arm, the transition oracle and the expander check, and
  re-baselines `parity_baselines/` and `parity_baselines_clp/` after every tier passes
  on its tree (§9.6), so every commit stays green. Its commit message records why every
  hash moved.
- **Deleted at the switch-over:** today's emission (`fill_ar_lag_columns`,
  `fill_z_inflow_columns`, `fill_z_inflow_rows`, `fill_z_inflow_entries`, the building
  of z columns from the sink, the lag arm of `state_to_lp_column`) and the arm; `fill_z_inflow_patches` and the lag pins;
  `z_inflow_rhs_buf`; the `StateSpace` ranges and `classify_incoming_column` (§4.3);
  the pieces of today's solve path that the folded primitive replaces (§5.2). Nothing
  of today's emission survives behind `test-support`: the expander is the only route to
  today's LP.
- **After the switch-over** the expanded LP is the logical oracle of §9.3. It is built
  from the same G and ψ as the compact LP, so it tests the fold's algebra and each
  site's assembly, not the coupling itself. The coupling is tied to main by the
  sink-first parity hashes (§4.2) while today's emission exists, and after it by the
  coupling-independence set of §9.1. A one-time mutation-kill table at the switch-over
  shows that this set catches coupling errors.

## 5. Per-solve data flow

### 5.1 The canonical cut-row RHS

One function computes every cut row's RHS at every site:

```text
S_k = Σ β_{k,j} Φ_{t,j}(â)              over the lag ≥ 1 slots, ascending
P_k = Σ_h β_{k,(h,0)} p_h               ascending h
W_k = Σ_h β_{k,(h,0)} (σ_h η_h)          ascending h
B_k = fl(fl(α_k + S_k) + P_k)            the bracket: depends on (â, child, cut), not on the opening
RHS_k = fl(B_k + W_k)
```

- Each dot product is sequential in ascending slot order, accumulators start at +0.0,
  and no `mul_add` is used. Vectorization runs across rows or trial points, never
  along the inner dimension, so an element's bits never depend on which rows or trials
  share its call.
- The function reads the slot ranges the Φ_t owner resolves at setup (§4.4), not
  hard-coded slots.
- A cache stores only values this function produced, so cache on and cache off give
  the same bits.
- This is a new definition: β₀·ẑ is split into P + W, which is not bit-equal to one
  dot over ẑ. ẑ = fl(p + σ·η) is still the value the base-row shifts and l\* use.

### 5.2 One shared folded solve primitive

Today every site prepares through `training/stage_solve_prep.rs::StageSolvePrep::run`,
the single owner of LP patching (`.claude/rules/sddp.md`), solves through
`solve/stage_solve.rs::run_stage_solve` or `cut/dcs.rs::lazy_solve_preloaded`, extracts
through `extract_duals_from_view` or `extract_state_duals_only`, and builds the
outgoing state through `assemble_outgoing_state`. The fold keeps that shape and changes
each piece once:

- **Folded prep**, in two phases. The prep stays the single patch owner.
  - The **pin phase** runs once per unit, a trial point on a loaded LP: pin the state
    columns, compute p, and assemble the cut-row bracket B from its source (§5.8).
  - The **opening phase** runs once per solve: ẑ = p + σ·η, the base-row shifts, load
    rows, NCS and slack bounds as today, the cut RHS B + W, then one set-based push
    and one range push.

  Today's per-opening re-pin writes values identical to the first pin, so every
  multi-opening loop runs the pin phase once per unit. A single-opening site runs both
  phases back to back.
- **Solve** through one frozen-or-lazy dispatch: the `StageOpeningSolver` enum
  (`training/backward/by_scenario.rs`) moves into `solve/` and every site goes through
  it. DCS is a mode of the shared solve, not a separate site.
- **Folded extraction**, one function in place of `extract_duals_from_view` and
  `extract_state_duals_only`. It returns Q, the non-lag state duals (`r̃c / c` at the
  storage, bucket and commitment incoming columns), the N-vector Gᵀπ, formed from the
  coupled rows' unscaled duals `π_r = r_r π̃_r` with the compact template's row scales,
  and the cut duals as zero-free (slot, y) pairs, appended straight from the solver's
  dual slice with no dense copy (§5.7). The fused terminal slice is the same call with
  the (leaf, parent) projection pair.
- **Outgoing state** through one `assemble_outgoing_state` (§3.5, §4.3).

Sites keep only their loops and their real differences: the cut-row set (frozen rows;
frozen plus delta rows in the backward; the lower bound's whole `CutRowMap`,
deactivated rows included; the DCS resident set), the warm-start rule (the terminal
stage's 1:1 apply, the lower bound's solve without a stored basis, DCS's uniform-basic
reconstruction), the output record and the opening loop.

**Refactor first.** Before any fold code lands, today's duplicates are removed and the
pieces the compact LP needs that today's LP can already carry are moved onto it, each
change green on today's LP with the parity hashes unchanged:

- the inflow-coupling sink (§4.2): the coupling writers always fill it, the generic
  `HydroInflow` terms are split from the column terms before `push_merged_row_entries`,
  and today's emission builds its z columns from it, so unchanged hashes show that the
  sink equals main's coupling;
- `assemble_outgoing_state` and its twin in `test_support.rs` copy the outgoing state
  per region through `outgoing_block_start` instead of copying a primal prefix (§4.3);
- `by_node`'s inline per-opening body (`training/backward/by_node.rs`) and
  `by_scenario.rs::solve_frozen` become one backward per-opening solve;
- one resolver of External or Generated opening noise replaces the copies in
  `by_node`, `by_scenario`, `replicated` and the lower bound;
- the frozen-or-lazy fork, written today in `solve/stage_solve.rs`,
  `training/forward/enumerated.rs`, `simulation/pipeline.rs` and the
  `StageOpeningSolver` enum, becomes the one dispatch above;
- `enumerated.rs::solve_forward_node` and `forward/stage_solve.rs::run_forward_stage`
  share one forward node-solve core, with the terminal 1:1 basis apply as a parameter;
- the two cut-dual-to-slot paths of `training/backward/outcome_aggregation.rs` become
  one (slot, y) extraction, read by both binding-activity counts.

### 5.3 The chain rule: one kernel, two call shapes

The chain rule of §3.3 runs in one kernel that takes the openings of a trial point with
their weights.

- **Aggregate-first** (`by_node` and `by_scenario`, DCS included). Each opening stores its
  folded extraction: Q_ω, its non-lag state duals, Gᵀπ_ω and its cut duals sparse by
  pool slot. Then
  `ḡ_h = −Σ_ω μ_ω (Gᵀπ_ω)_h + Σ_k ȳ_k β_{k,(h,0)}`,
  `β̄_{h,l} = ψ_{h,l} ḡ_h + Σ_k ȳ_k β_{k,(h,l+1)}` with `ȳ = Σ_ω μ_ω y_ω` in slot
  space, and `ᾱ = Σ_ω μ_ω Q_ω − dot_trial_state(β̄, x̂)`. ȳ is formed per (trial point,
  child) before the β product, in canonical ω order, in a per-worker slot-space
  accumulator with a touched bitmap. The bitmap's word scan yields the touched slots in
  ascending order and resets in O(nnz), independent of the pool size. One walk
  `Σ_k ȳ_k β_k` follows; the opening loop never sits inside it. Binding-activity
  tracking reads each opening's (slot, y) pairs before aggregation, because under pure
  CVaR some μ are 0. On a fan, μ comes from one weight-kernel call over the flattened
  successor × opening set; each child's ȳ_c and Ḡ_c are partial sums under those joint
  weights, never renormalised per child, and β̄ = Σ_c β̄_c. Openings are taken in
  canonical order.
- **Per opening**: the same call with one opening and μ = [1.0]. Accumulators start
  from their first term rather than from +0.0, so this call returns the unweighted
  formula exactly (fl(1·x) = x, and a +0.0 start would turn −0.0 into +0.0). It serves
  the replicated and fused-slice payloads, which are per opening
  (`objective ‖ subgradient`), and T-KKT (§9.3).

The kernel takes Gᵀπ and a borrowed slice of (slot, y) pairs of any length, in strictly
ascending slot order, and never reads the G store, so T-ID (§9.3) can feed it values
formed on the expanded LP. It skips an exact-zero weight (y, or ȳ) and accumulates each
ḡ_h in one accumulator: the Gᵀπ terms in canonical ω order, then the ȳ_k β_{k,(h,0)}
terms in ascending slot. With both rules its bits do not depend on whether zero duals
are present. A Σ_k summed separately from +0.0 and added afterwards breaks this:
+0.0 + (−0.0) is +0.0, and the sign the per-opening shape preserves is lost.

The aggregate-first shape goes into production only after the per-opening shape passes
the chain-rule oracles; the tolerance test of §9.4 pins that aggregating first commutes
with `aggregate_cut_into`. There is no second implementation.

DCS always runs `by_scenario`:
`training/backward_pass_state.rs::resolve_backward_scheduler` forces it whenever DCS
is active, so no `by_node` path exists under DCS. Under DCS the cut duals are read
through `scratch.row_map` over the final resident set and stored sparse by pool slot,
so the DCS finish is the same aggregate-first call as the frozen one: `by_scenario` has
one finish.

### 5.4 Sampled forward (out-of-sample on the PMO deck)

At stage t for pass m, with trial state x̂ = `records[m, t−1].state`:

1. p, then ẑ from the forward's own noise draw.
2. B over the frozen rows; store it in the bracket slab at `[m][t][k]` (§5.8).
3. W for the forward's draw, computed on the fly.
4. Pin the state columns; RHS = B + W; base-row shifts; one set-based push for load and
   shifted base rows; one range push for the cut rows.
5. Solve.
6. Outgoing state per §3.5.

`run_forward_stage` keeps its stage-cost formula; only the LP shrinks.

### 5.5 Backward `by_node`, per stage

`process_stage_backward_by_node` claims units (trial point m, opening block, child)
from a shared cursor; the two blocks of one trial point usually run on different
workers. Each rank's backward runs only its own forward trials, so every trial point's
slab entry is rank-local.

1. **Per-stage pre-pass,** parallel over rows, ended by one join before the claim
   loop: σ·η for every opening of the child, from the evaluator's noise term into a
   stage-scoped Ω×N scratch, then W `[ω][k]` for every resident row of the child LP
   (frozen rows, then delta rows).
2. **Per unit (pin phase):**
   - `load_backward_lp` (frozen template plus the delta batch);
   - pin the state columns once;
   - compute p;
   - assemble B: frozen rows from the forward slab, delta rows computed in the unit
     by the canonical function.
3. **Per opening (opening phase):**
   - ẑ_ω = p + σ·η_ω, from the stage scratch;
   - base-row shifts −G·ẑ_ω, scaled as the bound is written, load rows, NCS and slack
     bounds as today;
   - cut RHS = B + W[ω], one contiguous add, then one range push;
   - solve; append the opening's folded extraction to the worker's record buffer
     (§6.3).
4. **Per trial point, once both units finish:** the aggregate-first chain rule (§5.3),
   reading each opening's record in place through the location table.

### 5.6 Other sites

| Site | Data flow |
|---|---|
| `by_scenario` (`process_by_scenario_backward`) | As `by_node`, with the per-stage W pass; each trial point is one unit, so its delta-row bracket is computed in the unit, as in `by_node` |
| `replicated` (`solve_replicated_outcome_slice`) | B computed in the unit once per child load, plus W; per-opening chain rule, because its payload is per-opening `objective ‖ subgradient` |
| Lower bound (`lb_evaluate_stage_0`) | Every row in `CutRowMap`, including pool-deactivated rows that stay in the LP, gets the canonical RHS at x₀ per root opening, recomputed at every evaluation |
| Simulation (`solve_simulation_stage`) | The canonical function runs on the fly (out-of-sample, no caches); the inflow outputs read ẑ and the `inflow_lags` output reads â |
| DCS (`score_violated_candidates`, `lazy_solve_preloaded`, `solve_lazy`) | Always under `by_scenario` (§5.3). Scoring is B + W over all active slots, B computed in the unit, plus the non-lag product through `gemm_block`; appended rows carry the same RHS bits that selected them; the aggregate-first chain rule (§5.3) reads cut duals through `scratch.row_map` over the final resident set |
| Fused terminal slice (`capture_fused_terminal_slice`) | The folded extraction with the (leaf, parent) projection pair: the per-opening chain rule from the leaf pool (resident) to the parent pool (extracted) by global dimension |
| Enumerated forward, terminal static (`run_stage_solve_terminal_static`) | As the sampled forward, with no slab write; 1:1 basis in the compact layout |

### 5.7 Reading cut duals

`extract_duals_from_view` copies the cut-row duals only when binding activity is
tracked (`tracks_binding_activity`: a budget or `dynamic`). The converted NEWAVE deck
runs `level1` with no budget, so today that slice is empty. Under the fold it is
mandatory chain-rule input; gating its read on the existing predicate would silently
drop every Σyβ term on the production path.

A per-stage setup flag, true when the resident pool's projection has lag slots, decides
whether the folded extraction reads the cut-row duals. It reads them once, as (slot, y)
pairs, for both the chain rule and binding-activity recording, which keeps its own
predicate for whether it records. On the DCS path the pairs come through
`scratch.row_map`, by the ascending walk over the populated slots, so they are sorted
with no sort; a resident row outside the dual slice is an invariant breach, never a
skip. The slice spans the cut rows after the compact base rows: frozen rows in
`active_cuts()` order, then delta rows in `active_delta_cuts(iteration)` order, mapped
to slots through the precomputed per-iteration list, never by slot arithmetic. That
list ascends in slot, because the pool is append-only.

Two predicates, never conflated:

- **Storage keeps `y != 0.0`.** The IEEE comparison drops −0.0, which HiGHS reports on
  every basic row. A bit test (`to_bits() != 0`) keeps every −0.0, so nnz becomes the
  row count. A tolerance drops real Σyβ terms and changes the cut.
- **The binding count keeps `y > row_activity_tolerance`.** It runs per opening, before
  μ exists. The tolerance is validated ≥ 0, so an omitted zero never counted, and the
  counts are bitwise unchanged.

**The non-zero count is bounded independently of the cut count.** In a basic optimal
solution, a row whose logical is basic has dual 0. The rows with a nonbasic logical are
linearly independent on the basic structural columns. Cut rows touch only θ and the
outgoing non-lag state columns (a lag slot has no LP column). So at most
`B_t = 1 + n_x^LP` cut duals are non-zero: 169 on the PMO compact LP, whatever the
number of cut rows. `B_t` sizes reservations (§6.3). It is never a capacity, because
its premise is solver behaviour that no backend API contracts (§13), and an expanded-LP
caller (T-ID) reaches up to 2,185.

### 5.8 Caches

The fold's structural saving is the split into B and W (§5.1). On top of it, the items
below stay; the aggregate-first chain rule (§5.3) and DCS scoring from B + W (§5.6) are
the other two kept items.

| Cache | Content | Lifetime and key | Size per rank |
|---|---|---|---|
| Forward bracket slab | B over the frozen rows, written by the sampled forward at `[m][t][k]`, read by the frozen backward of the same trial point and child | One iteration; tagged (iteration, node_id) | 46 MB ÷ total ranks |
| Per-stage W | σ·η per opening in an Ω×N scratch and W `[ω][k]`, both in the pre-pass | One stage | 96 KB, reused |
| Pin once per unit | The state pins and B are set once per unit, not once per opening, in every multi-opening loop | One unit | none |

Rules that bind them:

- **Raw x̂.** B is computed from the raw trial state (`records[m, s].state`, the
  exchanged f64 values), never from the scaled column pin.
- **Validity.** `freeze_active_cuts_into_templates` runs once per iteration, in
  `run_cut_management`, and once before the first iteration, in
  `prime_frozen_templates`. The iteration therefore identifies the frozen set, and no
  freeze generation is kept. The slab is valid from the end of the sampled forward to
  the start of `run_cut_management`, and only for the child the forward visited (on
  the PMO chain, every child). Checkpoints are written only at iteration end, so the
  slab is never persisted; a resume refills it on its first forward.
- **Scope.** Only the sampled forward writes the slab, and only the frozen backward
  reads it. Under DCS, in `replicated` and in the enumerated forward, B is computed in
  the unit.
- **Misses.** A tag mismatch or a missing entry recomputes B and increments a miss
  counter that exists only under `test-support`, where tests read it. A miss is a
  performance event, never a correctness event. Misses are expected on a multi-child
  node for the children the forward did not visit.

Rejected:

- a setup table of σ·η per (stage, opening, hydro): a second owner of the noise term,
  bound to stay bitwise equal to the evaluator's, to save one multiply per hydro and
  opening;
- a pre-pass for the delta-row bracket: it makes `by_node` and `by_scenario` take
  different paths, to save one bracket per extra opening block;
- a slab under DCS, in `replicated` or in the enumerated forward: a second, slot-keyed
  layout to save a fraction of B;
- caching p from the forward (it costs 1–2 µs to recompute);
- splitting the base-row shift into Gp + Gσ·η (it would re-associate Gẑ for under
  1 µs per opening);
- caching the full lower-bound RHS (about 0.01 s per iteration);
- reusing the forward's lag product in the ownership scan (under 4% of that scan);
- building the enumerated External interior child cache (nothing to reuse);
- solving the forward's opening first (warm-start only, in-sample only).

**Expected out-of-LP time** on the NEWAVE PMO deck, all ranks, per iteration: about
8.9 / 9.4 / 12.6 CPU-s at 5 / 30 / 300 non-zero cut duals per opening, against
36 / 40 / 75 for the fold with no caches. The 300 column is an upper envelope: the
compact PMO LP admits at most 169 (§5.7). That is about 0.6% of LP time at 5 ms per solve and 2% at
1.5 ms. Computing the delta-row bracket in every unit adds one bracket per extra
opening block of a trial point. The remainder is dominated by the forward's own
bracket pass, which no cache removes because every forward state is new.

### 5.9 Warm start and stored bases

- `CapturedBasis` uses the compact layout throughout: `base_row_count` is today's
  count minus N and column statuses cover the compact columns. `to_broadcast_payload`
  and `try_from_broadcast_payload` change together; only lengths change.
- `reconstruct_basis`, `ReconstructionTarget::from_template` and the terminal 1:1 apply
  read the compact template. Cut rows still match by slot identity.
- The forward basis seeds the backward, as today. RHS patches never invalidate a basis
  or a retained factorization.
- A basis of today's LP restricts to a compact one by dropping the basic z columns, the
  nonbasic z rows and the nonbasic fixed lag columns, so warm-start quality is
  equivalent in exact arithmetic; T-BASIS (§9.3) maps the other way. Pricing state
  differs; pivot counts are similar but not equal, and no test compares them across
  formulations.
- Checkpointed bases are stored compact. A same-build resume is bitwise as today. A
  cross-build load passes the version check (same `CARGO_PKG_VERSION`), then
  `admit_stored_basis` rejects every basis on its column count and the run continues
  with an `UnusedStoredBases` warning.

## 6. Memory layout and kernels

### 6.1 Rules

- Large buffers are allocated once and reused; nothing allocates on the hot path.
- New data goes into the existing context structs, not new function parameters
  (`.claude/architecture-rules.md`).
- Kernels are safe Rust; enum dispatch, no `Box<dyn Trait>`.
- New per-worker buffers are sized in the existing serial helpers
  (`WorkspacePool::resize_scratch_bases`, `WorkspacePool::reserve_dcs_scratch`), one
  sizing shape for the struct.
- What changes in `novomodelo-stochastic` (the p/ẑ split) and `novomodelo-solver` (the
  range push) stays algorithm-neutral, per the infrastructure-crate rule.

### 6.2 The cut pool is unchanged

Flat slot-major rows, append-only. The kernels read each row's lag sub-slice in place,
visiting the frozen active slots in ascending order, then the delta slots. Rejected,
with measurements as evidence:

- a tiled or compact copy of the active set: 1.10–1.26× faster on a term of 0.2–0.3 ms;
- splitting the pool into storage and lag blocks: breaks the full-row `gemm_block`
  contract that ownership relies on;
- an f32 shadow copy: adds error to every cut RHS;
- padding or alignment changes: padded lanes can flip the sign of a zero sum.

### 6.3 New buffers

| Buffer | Owner | Lifetime | Size per rank |
|---|---|---|---|
| G CSR, hydro → (base row `u32`, unscaled coef `f64`), canonical hydro order, with the unscaled template bounds of the rows it touches | `StageContext` | run | small |
| Lag-slot ranges and chain-rule slot maps | `CutStateProjection` | run | small |
| Bracket slab `[m][t][k]`, tagged (iteration, node_id) | the forward's trajectory records | iteration | 46 MB ÷ total ranks |
| σ·η per opening (Ω×N) and W `[ω][k]` in LP cut-row order (frozen rows, then delta) | `BackwardPassState` | stage | 96 KB, reused |
| p, the assembled B, the RHS | `ScratchBuffers` (per worker) | reused | about 5 KB each |
| Per opening: Q, non-lag state duals, Gᵀπ (N-vector), zero-free (slot, y) | per-worker record buffers (below) | stage under `by_node`, one unit under `by_scenario` | at most about 5.4 KB per record on PMO whatever the cut count, against 17.5 KB today |

ψ, b and σ are read from `PrecomputedPar`, their one owner; â and ẑ live in renamed
existing buffers (§4.5). The slab replaces `TrajectoryRecord.primal` and
`TrajectoryRecord.dual`, which today are only cleared and never read.

**Per-opening records.**

- Each worker appends its records to its own flat buffers.
- Under `by_node`, a (trial position, ω) → (worker, record) location table, built in
  canonical order, lets the finish read every record in place in (position, ω) order,
  whichever worker wrote it. No stage-long arena is kept, and nothing is copied into one.
- Today's arena is a second dense copy of every outcome, committed in full at sizing
  (about 2 × 70 MB per stage at 1 rank), with a serial scatter into it (about 7.6 ms per
  stage).
- The fixed part of a record is 2.7 KB, and the cut-dual part is at most 2.7 KB at
  `B_t = 169`. So a `by_node` stage holds at most about 22 MB at 1 rank, typically
  11–13 MB.
- The buffers reserve from `B_t` (§5.7), never from the cut-row count, and grow
  monotonically past it, as `BackwardAccumulators` does. A record past `B_t` is stored
  whole: nothing truncates, clamps or errors.
- Sizing by the resident cut-row count is the rejected alternative. It costs 16 B per
  cut row per record, twice a dense f64 per row, and grows with the pool. Storing dense
  y with the indices kept separately is rejected with it:
  - the cut-dual part alone exceeds today's whole record above 2,184 cut rows;
  - the shared index list does not exist under DCS, where `row_map` is insertion-ordered
    and reset per child;
  - on a fan, each child has its own slot space.

New memory is about 46 MB at 1 rank and 12 MB at 4 ranks, under 0.4% of a 13.5 GB
per-rank budget. The slab is sized at freeze: when a freeze leaves more frozen rows than
its capacity, it grows there, and it is never resized anywhere else.

The slab is `[m][t][k]` because the forward already splits its records into disjoint
worker windows: it is writable in safe Rust with no false sharing and local first
touch, and each backward unit reads one contiguous run (3.2 KB at 400 frozen rows).
It is indexed by global pass id, never by worker id, so its contents are independent of
thread and rank shape.

### 6.4 Kernels

- **PAR evaluation:** `evaluate_par_batch` split into p and ẑ, today's summation order.
  Its noise term also fills the backward's per-stage σ·η scratch, so W and ẑ read the
  same product bits.
- **The canonical RHS kernel:** one production kernel for every site and every row
  count: the forward, the backward's in-unit brackets, the lower bound, simulation and
  DCS. Accumulators start at +0.0; no FMA; no skipping of zero coefficients; lanes
  across rows or trials only. It replaces `gemm_block` here because S has an inner
  dimension of 1,848, above the 513 to which `gemm_block` is pinned (`INNER_DIMS`).
  - It starts as a plain loop over rows of the scalar canonical function.
  - A batched form, 8 rows at a time in 256-element chunks, replaces the loop only if
    the bench of §9.7 shows it at least 1.3× faster with every local core busy (the
    core count is recorded with the result), not only at 1 thread. Each β element read from the pool feeds one
    multiply and one add, about 0.25 flop per byte, so under full-socket load the S pass
    is bandwidth-bound and the 8-row advantage can vanish.
  - If the batched form ships, the scalar function moves behind `test-support` as its
    reference, as `cut/selection_reference.rs` is for cut selection. There is no
    trial-point-lane variant.
- **Per opening:** RHS = B + W[ω], one contiguous add, then one range push.
- **Base-row shifts:** through the CSR, accumulating each row's shift Σ_h G_{r,h} ẑ_h
  from +0.0 in ascending canonical hydro id, subtracting it from the row's unscaled
  template bound, and multiplying by the compact row scale as the bound is written
  straight into `PatchBuffer` (§4.5), the idiom of `fill_load_patches`; finite bounds
  only. The order keeps the result independent of declaration order, and no step
  divides by a scale.
- **Chain rule:** one kernel, two call shapes (§5.3).
  - It reads the pool rows of the non-zero cut duals only. Skipping an exact zero gives
    the same bits as the dense sum under the single-accumulator rule of §5.3.
  - Order is ascending slot within each output element.
  - The walk is row-major: each non-zero row streams once across the full lag width,
    into a 2,016-element accumulator (16 KB, resident in L1d). It is bit-identical to a
    256-wide j-tile nest and measured 10–31% faster on a performance core.
  - The opening loop belongs to ȳ formation (§5.3). Nesting it inside the walk costs
    Ω times the multiply-adds and changes the association.
  - Cost is about 0.4–0.5 µs per non-zero row with the pool warm, and 1.2 µs cold from a
    pool much larger than the last-level cache. The aggregate-first shape runs it once
    per trial point and child.

### 6.5 Determinism

Each output element is computed by one thread in a fixed order; no floating-point
reduction crosses threads. Remote cuts keep ascending peer-rank order. Tests (§9.4):
the kernel property test when the batched form ships, and the cache on/off bitwise
test. No FMA guard covers the hand-written kernels, and none is needed: rustc never
contracts `a * b + c` into a fused multiply-add, so no build flag can introduce one,
and "no `mul_add`" is a review item. The existing
`gemm.rs::resolved_kernel_is_the_unfused_baseline` checks which `matrixmultiply`
kernel is selected and says nothing about hand-written Rust.

### 6.6 Paging and NUMA

- **Code:** allocate once (the slab grows only at freeze); size new per-worker buffers
  in the existing serial helpers (§6.1). Rayon does not bind a workspace to a thread,
  so first touch inside a worker would guarantee nothing; locality comes from one rank
  per NUMA domain. No `madvise` helper.
- **Launcher:** transparent huge pages via `GLIBC_TUNABLES`, one rank per NUMA domain,
  core and memory binding. Both builds use identical settings, recorded per run.

## 7. Solver-crate changes

### 7.1 A lower-bounds-only range push

```rust
fn set_row_lower_bounds_range(&mut self, first_row: usize, lower: &[f64]);
```

- Rows `[first_row, first_row + lower.len())` become `[lower, +∞)`. Name and doc are
  generic, with no mention of cuts. Every cut row is `[rhs, +∞)` (`push_cut_row`),
  including the lower bound's deactivated rows, so no caller needs a finite upper and
  the sddp crate holds no +∞ buffer.
- **A required trait method, no default.** Every workspace solver is wrapped in the
  `ProfiledSolver<S>` forwarder (`SolverWorkspace.solver`): a default delegating to
  `set_row_bounds` would let that forwarder fall back silently, and it would need an
  index list, either a hot-path allocation or one FFI call per row. Every implementor
  is updated: HiGHS, CLP, `ProfiledSolver`, and the test doubles in `novomodelo-sddp`
  and the solver crate.
- **HiGHS:** a new `novomodelo_highs_change_rows_bounds_by_range` in
  `csrc/highs_wrapper.{c,h}` wrapping `Highs_changeRowsBoundsByRange`, with its FFI
  declaration. HiGHS has no lower-only C call, so the backend keeps an internal +∞
  vector, resized in `load_model` and `add_rows` beside the per-row vectors it already
  sizes there (`row_value`, `row_dual`, `row_activity`, `row_magnitude`,
  `basis_row_i32`), and passes its prefix as the upper slice. It checks that indices fit
  in `i32`, asserts a non-error status, and adds to the existing
  `total_set_bounds_time_seconds`. Measured at 600 rows: 2.1 µs against 39 µs for
  today's set-based call.
- **CLP (correct, not tuned):** writes the lower bounds and `row_upper = +∞` into the
  Rust-side bound copies over the range, then makes the whole-array
  `novomodelo_clp_chg_row_lower` / `novomodelo_clp_chg_row_upper` push through one
  helper shared with `set_row_bounds`, so the unsafe block and its SAFETY text exist
  once. No shim.

The cut rows use the range push, once per solve. The base-row shifts and load rows stay
on the set-based `set_row_bounds`: the coupled base rows replace today's z rows at a
similar count.

### 7.2 Infinite bounds

The `set_row_bounds` and `set_col_bounds` docs (`trait_def.rs`) say bounds must be
finite, but both backends already accept infinities: `tests/sentinel_inf_row_probe.rs`
pushes [−∞, +∞] row bounds on HiGHS, `tests/conformance.rs` pushes a +∞ column upper,
and the CLP C wrapper translates ±∞ to ±`DBL_MAX`. Both docs are reworded to
"±`f64::INFINITY` means no bound; the backend translates it". The range push's doc
states its +∞ upper.

### 7.3 Unchanged

`StageTemplate` (the compact template is an ordinary one), `add_rows`, `load_model`,
the basis APIs, and `HighsProfile` (presolve stays On by default and is skipped on warm
starts).

## 8. Contract and documentation changes

### 8.1 `.claude/rules/sddp.md`

Every contract keeps its named regression test; pinning tests are retargeted, never
deleted.

Restated:

1. **Benders cut sign.** A lag coefficient is ψ·g + w, mapped by global state
   dimension, in cost-scaled units, with no `col_scale` division. Storage, bucket and
   ring coefficients stay r̃c/c at the incoming column. Cut rows keep −β·c on the
   columns they have; the lag slots go to the RHS.
2. **Intercept through the projection.** Computed only after the lag β are rebuilt;
   under aggregate-first it is Σμ·Q − β̄·x̂.
3. **State pinning by column bounds.** Lags have no LP column; their values enter only
   through the ẑ shifts and the cut RHS. There is still no state-fixing row.
4. **"Inflow noise enters only through the z rows"** becomes **"Inflow enters only
   through the coupling sink"**: the base rows the sink records, with the coefficients
   it records, shifted by −Gẑ with G unscaled and the row scale applied as the bound is
   written, plus the cut-row lag-0 term. Retargeted tests:
   `chronological_inflow_noise_moves_only_its_own_hydro`,
   `every_noise_dimension_patches_only_its_own_entity`,
   `inflow_slack_follows_z_routing_for_prefilling`.
5. **Truncation keeps the PAR state raw.** One evaluator call produces ẑ, which feeds
   the shifts, l\* and outgoing lag 0. The chain rule never differentiates l\*.
6. **Binding activity only for a configured reader.** Cut-row duals gain a second
   reader, the chain rule, decided by the per-stage lag-slot flag; both read one
   zero-free (slot, y) extraction, filtered by `y != 0.0`: never a bit test and never a
   tolerance. Its non-zero bound sizes reservations and never caps a record (§5.7).
7. **Append-only pool and slot-identity basis.** Bases use the compact layout; the base
   row count is today's − N. Across builds, stored bases are dropped with
   `UnusedStoredBases`.
8. **Fused terminal slice.** The folded extraction with the (leaf, parent) projection
   pair runs the chain rule from the leaf pool to the parent pool by global dimension.
9. **A cut's value at a state is canonical.** Unchanged, plus: the RHS bracket and the
   ownership values are separate quantities, and neither is computed from the other.

New:

10. **Every resident cut row's RHS is patched at every solve**: frozen and delta rows,
    DCS appends, lower-bound rows including deactivated ones, the simulation cut set,
    and terminal boundary rows.
11. **The cut-evaluation lag point has one owner.** Φ_t, M_z and M_a live in one place,
    resolved at setup into slot ranges; the RHS function and the chain rule both read
    them. A cached bracket is valid only for the (iteration, node_id, â bits, child) it
    was computed for.
12. **The cut-row RHS has one association.** ((α + S) + P) + W, each dot ascending,
    vectorized across rows or trials only, no `mul_add`, independent of batch shape;
    caches store only values this function produced. Pinned by the cache on/off
    bitwise test, the thread-shape gate with its power preconditions, and the kernel
    property test when the batched form ships.
13. **The realized inflow has one route into the LP.** The builder references it only
    through the inflow-coupling sink, and ψ, b and σ have one owner, `PrecomputedPar`.
    Pinned by the closed forms T4-1, T4-3 and T4-4 and the sink's structural tests
    (§9.1).

The Kiro mirror is regenerated in the same commit
(`python scripts/gen/sync-kiro-steering.py`).

### 8.2 `docs/design/lp-builder-contract.md`

- **Job 1 (build):** the builder emits the compact LP, and the unscaled inflow coupling
  G through the one sink (§4.2). New must-not: reference the realized inflow other than
  through the sink.
- **Job 2 (per-solve edits):** they become the pins of the incoming storage, bucket and
  commitment columns (no lag pins), the bounds of the coupled base rows and of the
  load-balance rows, the NCS and inflow-slack column bounds, θ at the terminal stage,
  and the cut rows' lower bounds through the range push. `fill_z_inflow_patches` and
  the lag pins are deleted.
- **Job 3 (address):** kept as it stands: one address home per family. The compact
  layout is the only layout, so no fact gains a second home. Output decoding loses
  `classify_incoming_column`; the policy export classifies by state index (§4.3).
- **Job 4 (layout conventions):** "the inflow-definition rows occupy the same range at
  every stage" is deleted. "A state dimension's outgoing column index equals its state
  index" holds for storage only; the bucket and commitment columns follow storage,
  resolved through `StateSpace::state_to_lp_column`, and the lag slice comes from the
  accumulator. The base row count shrinks by N. The one-route rule of contract 13
  (§8.1) is added to the invariants list.

### 8.3 Other documents

- **This spec** lives at `docs/design/par-folding.md` with status Proposal. If the
  branch merges, its durable content moves into `sddp.md` and `lp-builder-contract.md`
  and this file is deleted, per `docs/design/README.md`.
- **CHANGELOG:** one `[Unreleased]` entry under Changed stating that results change,
  written final under `.claude/rules/doc-integrity.md` §6, in the commit that moves the
  numbers.
- **User docs, at merge only** (docs repo): `math/state-augmentation.md` §2 (its case for
  pinning over RHS constants reverses for lags), `math/lp-layout-and-scaling.md`,
  `math/_impl/_cut-management.configure.mdx`, and their pt-br copies.
- **Not needed:** schema regeneration (no config change), Python parity work (no new
  output file), `ARCHITECTURE.md`, `.claude/architecture-rules.md` (the new buffers
  follow its placement rules).
- Plan identifiers stay out of shipped comments, rustdoc and the CHANGELOG. Stale
  docstrings are fixed when the surrounding code is touched.

## 9. Testing

Every new test goes into an existing test binary as a `mod`
(`.claude/rules/testing.md`, cost discipline). Every tolerance is stated in unscaled
units, converted from a solver tolerance by the scale of the row or column it bounds in
the LP that produced it, because the compact and expanded LPs scale coupled rows
differently: row residual ε_p/r_i, column bound c_j·ε_p, reduced cost ε_d/c_j, row
dual r_i·ε_d. Kept columns and uncoupled rows have bitwise-equal scales in both LPs;
coupled rows use the compact scale for compact-produced values and the expanded scale
for expanded ones. A check that mixes both LPs sums both sides' converted terms. The
conversion is exact only for first-try solves, since retries change solver options
(CLP retries switch internal scaling on), so every solve an oracle compares asserts
`retry_count == 0` and fails loudly otherwise; a tolerance is never loosened instead.

### 9.1 Tier 3, LP structure (`template_integration.rs`)

- **Transition oracle** (§4.7, deleted at the switch-over): compact template against
  today's template minus the removed rows and columns, bitwise before scaling;
  geometry decoding; kept-column and uncoupled-row scales after `postprocess_templates`
  on a study with a non-empty `commit_out`; on every study.
- **Expander check** (§4.7, deleted at the switch-over): the three expander parts
  against today's emission, bitwise, on every study: the template, unscaled and after
  `postprocess_templates`; the cut rows of a frozen lag-bearing pool; the z-row RHS
  and lag pins at sampled (â, η).
- **The sink's structural tests:** hand-derived entries for each coupling family on an
  in-code fixture (parallel −ζ, chronological −τ_k per block row, the PreFilling
  route, a generic `HydroInflow` term merged with a second term on the same hydro), and
  one `should_panic` test per clause of the sink's debug assertion.
- **Coupling-independence set.** After the switch-over no reference LP from main
  survives, so the coupling rests on tests that derive their expected values:
  - parallel −ζ: the retargeted `every_study_reads_z_inflow_on_its_water_rows`
    sweep, T4-1, the sink test and d16;
  - chronological −τ_k: the sweep (d49 and d50 have unequal block hours), the sink
    test, and a block-0 storage assertion added to
    `chronological_inflow_noise_moves_only_its_own_hydro`, whose fixture otherwise
    reads only end-of-stage storage, which a τ_k swap leaves unchanged;
  - the PreFilling route: the sweep's routing assertions, T4-3, d39, d40 and
    `inflow_slack_follows_z_routing_for_prefilling`;
  - generic `HydroInflow`: the sink test, T4-4 and the `hydro_inflow_travel_time.rs`
    equivalence;
  - the Φ alias: T4-2, T4-7, the d16 lag-shift twin, d29 and d30.

  The sweep and the travel-time equivalence are retargeted onto the sink's entries
  before the switch-over and run on both emissions while the arm exists. The sweep
  fails loudly unless it visits a chronological stage with unequal hours, a
  PreFilling-routed hydro and a generic `HydroInflow` entry. The twin tests and
  `run_cut_oracles` do not count: they are built on the same G and ψ as production.
- **Mutation-kill table,** run once at the switch-over against the coupling-independence
  set alone and recorded with that commit's ticket, with no permanent code: flip −ζ,
  swap τ_k, route PreFilling to the owner's row, skip the generic merge, skip the
  row-scale multiply in the shift, l versus l+1 in the Φ_t owner. Each mutation must
  make at least one test fail. A T4-8 chronological closed form is added only if a
  τ_k swap survives.
- **Scaling:** every kept column's scale and every row scale of a row with no coupling
  entry equal the expanded LP's exactly.
- The compact template has N(L+1) fewer columns and N fewer base rows than the expanded
  LP.
- `build_delta_cut_row_batch_into` and `push_cut_row` emit identical compact rows, and
  neither emits a lag entry.
- `par_a_lag12_lp_coefficient.rs` asserts ψ_eff on `PrecomputedPar`, its one owner,
  instead of on the CSC.

### 9.2 Tier 4, analytical

Each test is labeled `/// derived:` and has a non-degeneracy twin.

| Test | Fixture | Mutation it catches |
|---|---|---|
| T4-1 | 1 hydro, PAR(1), no future | Sign of the G term; κ |
| T4-2 | PAR(2) with one injected binding cut | Missing w_{(h,0)}; l vs l+1; an alias on the oldest lag; negated Σyβ; col_scale multiplied instead of divided (ψ₁ = 0.5) |
| T4-3 | PreFilling upstream → target | Using the owner's row instead of the target's |
| T4-4 | Binding generic `hydro_inflow` row (d54) | Missing −w·π_gen; equality or range rows shifting only one bound |
| T4-5 | Storage-only projection | Lag slots leaking into a storage-only cut |
| T4-6 | Truncation with ẑ < 0 | Differentiating l\* |
| T4-7 | Multi-resolution non-finalize stage | Breaking today's alias |

### 9.3 Twin tests on the production path

The expanded LP (§4.7) is the logical oracle, behind `test-support`.

- **T-BASIS, basis transfer.** Map the compact optimal basis to the expanded LP (z
  basic, z rows and lag columns nonbasic) and solve warm. It fails loudly unless the
  simplex takes 0 iterations, the fixture has a binding lag-bearing cut, and c_z ≠ 1
  and c_a ≠ 1 on the expanded LP.
- **T-KKT (primary).** Takes its duals from the T-BASIS warm solve of the expanded LP:
  the same vertex, with duals the fold did not compute. It checks that the expanded
  LP's lag reduced costs divided by c_a equal the fold's β, computed from the compact
  solve's own duals; that the compact primal, completed with â and ẑ, meets the
  expanded LP's z rows and cut rows within tolerance, with complementarity against the
  expanded duals; and that the objectives are equal. Degeneracy cannot make it pass or
  fail spuriously, because both sides read one vertex.
- **One T-KKT configuration per cut-row-set shape:** frozen rows; frozen plus delta
  rows; the lower bound's whole `CutRowMap`; the DCS resident set. One more
  configuration covers the projection pair: the fused terminal slice, where the
  resident (leaf) projection differs from the extracted (parent) one and slots map by
  global dimension. The configurations, and the full-run recorder's per-kind ones, are
  built by an exhaustive match over the production enums the shared folded primitive
  dispatches on to choose its cut-row set and projection pair: the frozen-or-lazy
  dispatch, the source of B, and the extraction's projection pair. The test-side table
  of solve-site kinds in `test_support.rs` is keyed from those variants, so a new
  production variant does not compile under `test-support` until it has a
  configuration.
- **The recorder.** `BoundRecordingSolver` (`test_support.rs`) is extended to record the
  bounds of the rows `add_rows` appends, `set_row_lower_bounds_range`, and a per-solve
  log. T-KKT and the full-run recorder use it; no new `SolverInterface` implementor is
  added.
- **Power guards.** One shared helper in `tests/common/` that every gate calls: at least
  one binding cut carries a lag slot; c_z ≠ 1 and c_a ≠ 1 on the expanded LP; a coupled
  row with a nonzero dual whose compact row scale differs from its expanded one, which
  exercises the per-LP conversion; a coupled row with a nonzero dual whose compact row
  scale is ≠ 1, which makes a missing or doubled row-scale multiply in the shift or the
  extraction visible; and, on the lag study, the study's own preconditions below. One T-KKT instance runs on a
  `level1`, no-budget fixture and so catches the cut-dual trap of §5.7.
- **T-ID, shared-dual identity,** on the expanded LP. Forms Gᵀπ^exp from G and the
  expanded LP's own unscaled duals, π^exp = r^exp·π̃^exp, feeds it with the expanded cut
  duals into the production chain-rule kernel, and compares to ρ_a + ψ·ρ_z within
  2γ_n·Σ|terms|, indexing lags through the `StateSpace` regions. ρ_a and ρ_z are the lag
  and z reduced costs c − aᵀπ that the test recomputes from the expanded LP's own
  columns (z row, coupled rows, cut rows) and the reported duals, unscaled with the
  expanded scales. The identity never reads the solver's reported reduced costs: HiGHS
  reports exactly 0 for every basic column, z_h included, so the reported values carry
  the final BTRAN's backward error, which no rounding bound covers. A separate check ties
  ρ_a to the reported r̃c_a/c_a within ε_dual/c_a. It never pairs compact duals with
  expanded ones. It replaces
  `basis_trajectory_probe.rs::lag_fold_check_stage`, which assumes the state is storage
  plus lags and feeds the z-row dual straight in, so it never tests reconstruction from
  the water and generic duals.
- **Lag study with FPHA.** An in-code study in `structural_studies()` gives T-ID and
  T-KKT binding FPHA and evaporation rows next to lag-bearing cuts: 3 stages, 2 hydros
  with PAR(2), FPHA on one and evaporation on the other, a generic `hydro_inflow` range
  row and load above the hydro capacity. Its power guard fails loudly unless, at
  stage 1, an FPHA plane, the evaporation row and the generic row have nonzero duals, a
  binding cut carries a nonzero lag coefficient, Gᵀπ ≠ 0 on each hydro, c_z ≠ 1 and
  c_a ≠ 1, and a binding coupled row has different compact and expanded row scales. A
  geometric-mean row scale changes only when the removed z entry was the row's largest
  or smallest |entry|, so the study is built for it.
- **T-OBJ and T-VALID, independent cold solves** of the compact and the expanded LP.
  Objectives agree within `ORACLE_ABS_TOL + ORACLE_REL_TOL·|Q|`
  (`tests/common/cut_oracles.rs`), and each formulation's cuts minorize the other's Q
  at sampled points. These never compare coefficients or primals.
- **Deleted once T-ID and T-KKT pass:** the ignored probes
  `lag_fold_chain_rule_identity` and `lag_fold_static_ab` in
  `tests/basis_trajectory_probe.rs`, with `lag_fold_check_stage` and their other
  helpers. After the switch-over they would
  check an LP that production no longer solves.

### 9.4 Caching and determinism

- **RHS kernel property test,** only if the batched form ships: bitwise against the
  scalar canonical function at 2,016 lag slots, up to 10⁴ rows, batch widths 1..W and
  random row subsets.
- **Cache on vs off:** bitwise-identical results with the slab and W enabled versus
  recomputed, over one backward stage and over a full small training run. The one
  test-support control forces a tag mismatch, which must recompute B, also compute W in
  the unit through the canonical function, increment the miss counter and give
  identical bits. Chain fixtures assert zero misses.
- **Aggregate-first:** matches the per-opening shape followed by `aggregate_cut_into`
  to tolerance. A pure-CVaR fixture (some μ = 0) checks that binding counts still come
  from the per-opening duals.
- **`mpi_wire.rs` gates under the fold:** thread and rank invariance, opening order,
  and the opening-block scheduler with at least 2 blocks, on the uniform fixture, on
  d43, and on a leaf ≠ parent projection fixture for the fused slice. Power
  preconditions, failing loudly when unmet: frozen rows exist, the forward windows
  differ between the two shapes, and at least one binding cut is lag-bearing.
- **Declaration order:** a sort-contract unit test builds the G CSR from permuted
  hydros and asserts the same canonical order, and the FPHA lag study joins the
  permutation test in `template_permutation.rs`.
- **No allocation:** one no-alloc `mod` in `forward_sampler_no_alloc.rs` drives the
  folded path single-threaded through a solve (prep, RHS assembly, pushes, solve,
  extraction, chain rule) after warm-up; its `thread_local` counter cannot see rayon
  workers. A capacity check, after the precedent of
  `mpi_wire.rs::assert_by_node_scratch_capacity_invariant`, asserts that the slab grows
  only at freeze.

### 9.5 Solver crate

In `tests/conformance.rs`, per backend: a tiny LP with a ≥ row pushed through
`set_row_lower_bounds_range`, checked against the hand-derived optimum; a row with a
finite upper loses it to the push; a push at a nonzero `first_row` changes only its
rows; `set_row_bounds` accepts a +∞ upper; and, at an optimum, every row whose logical
is basic reports a dual equal to 0.0 (either sign), which pins the premise of the
non-zero bound of §5.7 against a solver upgrade.

### 9.6 Behavioral and golden

- `deterministic.rs` keeps its assertions. A value that moves is a finding to derive or
  explain, never one to re-record.
- `run_cut_oracles` runs on every deck with lag state, plus a cross-formulation oracle:
  the folded pool is checked against the expanded successor LP.
- `extensive_form_oracle.rs`, `inflow_nonnegativity.rs` and the three retargeted pinning
  tests keep running.
- **Parity hashes:** nearly every hash moves, since even a PAR(0) case loses its z rows
  and natural scaling moves coupled rows at rounding level. Until the switch-over no
  hash moves at any commit: the sink-first refactor and the arm leave today's LP and
  its results unchanged. `parity_baselines/` and `parity_baselines_clp/` are
  re-baselined together in the switch-over commit itself, after every tier above
  passes on its tree, so every commit stays green; the commit message records why
  every hash moved.

### 9.7 Benches (Criterion, not run in CI)

One bench target in `crates/novomodelo-sddp/benches/`, with one group per measurement:

- the RHS kernel: the plain row loop, the batched form and `dgemm`, at 1 thread and
  with every local core busy, the count recorded (the gate of §6.4);
- the per-stage W pass;
- the chain rule at 5, 30 and 169 non-zero cut duals per opening, plus a degraded case
  at a tenth of the cut rows, per opening versus aggregated, with the pool warm and
  with a pool much larger than the last-level cache;
- the solver push, set-based versus range.

### 9.8 Gate

The full workspace test command from `CLAUDE.md`, the CLP suite, clippy pedantic with
zero warnings, `cargo fmt --check`, and `bash scripts/ci/check-deny.sh`.

## 10. Local performance check

The last step of the implementation, after all correctness work.

- Build main and `folding` with `scripts/build-profiling.sh` (GNU ld, so DWARF unwinding
  works) and profile both on a local converted deck for a few iterations with
  `perf stat` and `perf record -e cycles:u` under `taskset`, following
  `docs/profiling.md`. Both files live on the `docs/profiling-guide` branch (commit
  `151ef3d3`), not yet on main; the check uses that branch's copies.
- Check that LP time drops, that the out-of-LP kernels stay a small share, and that no
  new hotspot appears. The benches of §9.7 cover the kernels.
- Output: a short findings note.

No new performance counters are added. Production relies on the columns main already
writes; out-of-LP cost shows there as time no existing column accounts for, and `perf`
attributes it locally if it looks wrong.

## 11. Production A/B

Run by the product owner on the production cluster, outside the implementation plan.
The analysis of the returned parquets happens later; the rules are fixed here, before
any run.

### 11.1 Cases

- **NEWAVE PMO jan/26 (converted):** 168 hydros, 12 lag axes, about 72 stages, 200
  forward passes, 20 openings, 2,000 simulation series.
- **DECOMP jan/26 rv0 and out/26 rv0 (converted):** storage-only interior pools, so they
  measure the LP shrink and the fixed per-solve cost, not the cut kernels.

### 11.2 Runbook

1. **Pre-flight.** Convert each deck with a pinned bridge version, read the converted
   `config.json`, and confirm: `level1` cut selection (check_frequency 1, lookback 2)
   with no `max_active_per_stage`; `by_node` backward with block size 10;
   `out_of_sample` training-forward and simulation inflows; simulation
   `cut_set = active`; a `finite_horizon` chain graph. Record each config's hash.
2. **Build** both binaries identically: same toolchain and profile, unfused baseline
   kernel, no `target-cpu=native`. Record the build info.
3. **Runs per deck:**
   - main, forward seed A;
   - main, forward seed B (the seed-spread yardstick);
   - `folding`, forward seed A;
   - two cross-simulations, each build simulating the other's policy with the same
     simulation seed; the `UnusedStoredBases` warning must appear.
4. **Environment:** identical rank count, placement and launcher settings (§6.6) for
   both builds; record `numactl -H`, `lscpu`, the THP settings, `ldd --version`, the
   Slurm binding, the instance type and ranks per node with every run.
5. **Never resume from the other build's checkpoint.** The version check would let it
   through; only the basis-drop warning would show it.
6. **Bring back** the output parquets of every run.
7. **Optional, any time:** a short main-only run to measure solve time, load time,
   simplex iterations and cut rows per LP on the deck.

### 11.3 Analysis plan

- **Lower bound:** acceptable when |LB_folding − LB_mainA| ≤ |LB_mainB − LB_mainA|.
- **Simulated cost:** paired by scenario, D_s = C^folding_s − C^mainA_s over S = 2,000,
  reported as D̄ ± z·s_D/√S; acceptable when within main B vs main A on the same
  statistic.
- **Cross-simulation:** each policy simulated by both builds separates the formulation
  effect from the training-path effect.
- **Timing:** compared at equal iteration counts; cold loads and warm solves separately
  (HiGHS presolve runs only on cold solves); `cut_selection_ms` reported on its own,
  because the fold does not touch it.
- **Determinism** is checked within each build by the test gates, never across builds.

## 12. Out of scope

- Evaluating lag 0 of the cuts as a month average when a month spans several stages.
  The Φ_t owner of §4.4 is where that change starts; it has its own open questions on
  the state space and on the noise grouping.
- FPHA reading z for tailrace lateral inflow (a new coupling goes through the
  inflow-coupling sink, so the chain rule picks it up with no special case).
- Folding the truncation slack.
- A CLP range-bounds shim.
- A `dynamic` cut-selection A/B (it gets correctness tests only).
- A `madvise` helper, a tiled cut operand, and pool layout changes.
- A shared no-op `SolverInterface` test double replacing today's empty mocks.

## 13. Facts this design rests on

Verified in code:

- In today's LP, z entries are written only by `push_local_inflow_coupling` (the
  water-balance rows, own or PreFilling target), `push_local_inflow_rate` (generic
  `HydroInflow` terms) and `fill_z_inflow_entries` (its own z row); lag columns have
  entries only in their z row and in cut rows; lag and z columns have objective
  coefficient 0. No production code reads the z-row dual.
- `postprocess_templates` computes each column scale from that column's entries alone
  and each row scale from that row's column-scaled entries.
- The backward pins the same raw state the forward pinned for that trial point and
  stage; nothing touches the frozen rows or pool activity between the forward and the
  backward of one iteration; the opening tree is fixed at setup; each rank's backward
  runs only its own forward trials.
- `freeze_active_cuts_into_templates` runs once per iteration in `run_cut_management`
  and once in `prime_frozen_templates`, before the first iteration.
- `resolve_backward_scheduler` forces `by_scenario` whenever DCS is active.
- The canonical template digest is compiled only under
  `cfg(any(test, feature = "test-support"))`.
- HiGHS's C API has no lower-only row-bound call; `Highs_changeRowsBoundsByRange`
  takes both slices.
- HiGHS 1.13.1 (vendored) zeroes the duals of basic variables in
  `HEkk::getSolution` before negating, so a row whose logical is basic reports
  exactly −0.0. Every result the backend accepts is OPTIMAL and basic:
  - its simplex rungs return a basis;
  - presolve is followed by a postsolved basis;
  - its IPM rungs keep `run_crossover` at its default, which is on.

  This is implementation behaviour, not an API contract.
- CLP 1.17.11 runs simplex on every rung. That a row whose logical is basic reports an
  exact zero is probable (one BTRAN, slack columns pivot as identity columns) but
  unverified; §9.5 pins it.
- Checkpoints are written only at iteration end.
- The NEWAVE converter emits `level1`, `by_node` with block size 10, `out_of_sample`
  training and simulation inflows, simulation `cut_set = active`, and a chain graph.
  The converted `config.json` itself has not been read; the pre-flight (§11.2) closes
  that.

Assumed, to be measured: LP time per solve (1.5–5 ms used above); the non-zero cut
duals per opening on PMO (structurally at most 169, §5.7) and their union across a
stage's trial points, which places the chain-rule walk between its warm and cold costs;
the batched RHS kernel's gain with every core busy (§6.4); CLP's range-push cost.

Measured on small decks only: an instrumented frozen-path training run on four decks
found no opening with more non-zero cut duals than `1 + n_x^LP`, and one deck attained
the bound.
