# PAR folding

> **Status**: Proposal — not implemented. This is the target design for removing the
> PAR(p) inflow model from the stage LP: the lag columns, the z columns and the z rows
> leave the solver, and their effect enters as right-hand-side constants computed
> outside it. The work lives on the long-lived branch `folding`, where the fold
> replaces today's LP outright (no runtime switch, no cargo feature). The branch is
> evaluated against main in production before any merge is considered. Figures quoted
> here are calibration-time estimates for the converted NEWAVE PMO deck, not standing
> claims.
>
> **Companions**: `docs/design/lp-builder-contract.md` (the four jobs of the builder,
> which this design amends in Jobs 2 and 4); `.claude/rules/sddp.md` (the correctness
> contracts restated and added in §8); `crates/novomodelo-sddp/src/cut/basis_reconstruct.rs`
> (the two basis entry points, unchanged in kind).

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
else is computed once per stage, by the builder or by the fold derivation straight after
it.

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
| Shape | A fold layer over the unchanged builder (§4), keeping the builder as the formulation of record |
| Branch | Named `folding`, no prefix; does not track main; main is merged in only at releases, so the branch inherits main's `CARGO_PKG_VERSION` |
| Solver backends | HiGHS is the performance target; CLP is correct, not tuned (no shim) |
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

### 3.2 What the solver sees

At stage t with trial state (v̂, â) and opening η:

```text
p_h   = b_h + Σ_l ψ_{h,l} â_{h,l}             once per trial point
ẑ_h   = p_h + σ_h η_h                          once per solve
base row r:  bound_r = template_bound_r − Σ_h G̃_{r,h} ẑ_h   (both finite bounds)
cut row k:   row_lower_k = α_k + Σ_{j ∈ lag slots of P_t} β_{k,j} Φ_{t,j}(â, ẑ),  row_upper_k = +∞
```

`G̃_{r,h} = Ã_{r,z_h} / c_{z_h}`: the z column of the logical template's CSC divided by
the z column scale, in compact row positions. The base-row shift is exactly what a
presolve does with a column fixed at `ẑ_h / c_{z_h}`.

Φ_t is the lag point at which stage t's cut rows evaluate the successor cut. Today it
is a fixed alias, the one `state_to_lp_column` encodes: Φ_{(h,0)} = ẑ_h and
Φ_{(h,l)} = â_{h,l−1} for l ≥ 1. It does not depend on the multi-resolution
accumulator, which only builds the outgoing state vector. So reproducing today's
cuts needs no accumulator weights.

The removed columns have objective coefficient exactly 0 and the fold adds no constant
offset, so the folded optimal value Q equals the logical one.

### 3.3 The chain rule

Let R be the kept base rows, π̃ their duals and ỹ the resident cut-row duals. For each
hydro h, in solver units:

```text
g_h        = − Σ_{r ∈ R} G̃_{r,h} π̃_r  +  Σ_k ỹ_k β_{k,(h,0)}
β^new_{h,l} =   ψ_{h,l} g_h            +  Σ_k ỹ_k β_{k,(h,l+1)}      (the second sum is absent for l = L−1)
```

- Every term is already in the unit the pool stores. Do not divide by any `col_scale`
  and do not multiply by κ.
- g_h equals today's z-row dual. In today's LP the z-row dual carries the solver's
  `rc_z` (at most the dual tolerance), so `β^today − β^fold = −ψ rc_z^today`: the fold
  is at least as accurate as today.
- Storage, bucket and commitment coefficients are unchanged: they remain
  `r̃c / c` at their incoming columns, over the same kept rows.
- **Slots map by global state dimension, never by position.** The resident pool's
  projection P_t (stage t+1's `state_config`) and the extracted pool's projection
  P_{t−1} can differ (d43; DECOMP boundary leaves with `FULL_STATE_CONFIG`). β^new is
  written at P_{t−1}'s slot for global dim (h, l); the w term reads P_t's slot for
  (h, l+1) and contributes 0 if P_t lacks it; g_h reads P_t's slot for (h, 0) if present.
- **General form,** through the Φ_t owner (§4.4): with `M_z = ∂Φ_t/∂ẑ`,
  `M_a = ∂Φ_t/∂â` and `w = Σ_k ỹ_k β_k^lag`, `g = −G̃ᵀπ̃ + M_zᵀ w` and
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
- The outgoing state is the kept leading block of the compact primal (storage, buckets,
  commitments) read at compact positions, with the lag part produced by the unchanged
  accumulator fed from â and ẑ instead of from LP primals.

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
(§5.3, the aggregate-first chain rule).

### 3.8 What is identical and what is not

Identical in exact arithmetic: the feasible set projected on the kept columns and θ;
Q; the optimal dual face on the kept rows; hence the set of attainable cuts; the
non-lag coefficients at a given vertex; the intercept given β.

Not identical: the vertex the simplex selects at a degenerate optimum (a smaller LP
follows a different pivot path), and therefore every downstream iterate, the CVaR
ordering at ties and the DCS resident set near tolerance. This is the permitted
hot ≠ cold class; tests assert invariants or compare quantities derived from the same
duals or the same basis.

## 4. Architecture: the fold layer

### 4.1 One new module, `lp::fold`

Built once per stage at setup from four existing inputs: the logical `StageTemplate`,
`StateSpace`, `StageGeometry` and `PrecomputedPar`. For each stage it produces a
`FoldedStage` holding:

- **The compact template**, an ordinary `StageTemplate`: the logical CSC, bounds and
  objective without the N·L incoming lag columns, the N z columns and the N z rows,
  with the logical column and row scales unchanged. It is a submatrix of the logical
  template.
- **G̃** as a hydro-major CSR, hydro → (compact row as `u32`, coefficient `f64`), in
  canonical hydro-id order. It is read from the CSC, so PreFilling routing, generic
  `HydroInflow` terms and any future row family that reads z come along with no special
  case.
- **Every per-solve write position**: the base-row shift positions, the contiguous
  cut-row range, and a shared +∞ upper-bound buffer.
- **Read maps for every consumer** of primals and duals, in compact positions. Removing
  `inflow_lags` and `z_inflow` leaves three column regions, each with one fixed offset:
  storage at 0, the outgoing buckets and commitments at −N·L, and everything from
  `storage_in` onward (θ and the whole control region) at −(N·L + N). Base rows shift
  uniformly by −N, because the z rows lead the row space (`z_inflow_rows()`).
- **The cut-row emission map**: for each pool dimension, either a compact column or a
  lag slot for the RHS. Cut rows are written in compact form directly from the pool
  row; lag coefficients never reach the LP.
- **Chain-rule maps**: slot maps between the resident projection P_t and the extracted
  projection P_{t−1}, keyed by global state dimension, plus ψ, b and σ in `[h][l]`
  order.

Typed indices `LogicalCol` / `SolverCol` and `LogicalRow` / `SolverRow` keep the two
spaces apart in code.

### 4.2 Setup refusal

`fold::derive` refuses, with a setup error, a template in which a removed column has a
nonzero objective coefficient, or a lag column has an entry in any row other than its
own z row. Both are true of today's builder (`columns.rs` zero-initializes the
objective; lag columns appear only in `entries.rs::fill_z_inflow_entries`); the refusal
turns that fact into a checked one. A structural test pins it.

### 4.3 Who sees what

- **Everything that talks to the solver uses the compact template**: loading, patch
  positions, the base-row count, θ, and the basis layout.
- **Patch positions are translated once, at setup.** `fill_z_inflow_patches` and the
  lag column pins leave the production path and stay only as `test-support` drivers
  for the oracles.
- **No per-solve scatter into logical vectors.** Consumers receive compact positions at
  setup; values of removed columns come from the â and ẑ buffers. A logical scatter
  exists only behind `test-support`, for the logical-replay KKT test (§9.3).
- **The logical template stays** as the formulation of record and the test oracle. It
  holds base rows only; its memory is measured, not assumed.

### 4.4 One owner for the cut-evaluation lag point

Φ_t, M_z and M_a are owned in one place, and both the cut-RHS function (§5.1) and the
chain rule (§3.3) call it. Today it encodes the fixed alias. A later change that
evaluates lag 0 as a month average when a month spans several stages changes only this
owner: at a non-closing stage M_z = 0 and M_a = I (the chain rule then reads slot
(h, l), not (h, l+1)), and the β₀·p and W terms vanish.

### 4.5 Buffers of record

Each worker holds authoritative out-of-LP buffers in `ScratchBuffers`: â (the incoming
lags), p (once per trial point) and ẑ (once per solve). No consumer reads a removed
column's value from anywhere else.

### 4.6 Unchanged

`StateSpace` and the state vector; the cut pool, its rows and its append-only slot
identity; the policy file format; the MPI cut wire (`cut/wire.rs`) and state exchange;
`StageGeometry`; the policy-export manifest; the canonical template digest (both keep
reading the logical layout).

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
- The function goes through the Φ_t owner (§4.4), not hard-coded slots.
- A cache stores only values this function produced, so cache on and cache off give
  the same bits.
- This is a new definition: β₀·ẑ is split into P + W, which is not bit-equal to one
  dot over ẑ. ẑ = fl(p + σ·η) is still the value the base-row shifts and l\* use.

### 5.2 Sampled forward (out-of-sample on the PMO deck)

At stage t for pass m, with trial state x̂ = `records[m, t−1].state`:

1. p, then ẑ from the forward's own noise draw.
2. B over the frozen rows; store it in the bracket slab at `[m][t][k]` (§5.6).
3. W for the forward's draw, computed on the fly.
4. Pin the state columns; RHS = B + W; base-row shifts; one set-based push for load and
   shifted base rows; one range push for the cut rows.
5. Solve.
6. Outgoing state per §3.5.

`run_forward_stage` keeps its stage-cost formula; only the LP shrinks.

### 5.3 Backward `by_node`, per stage

`process_stage_backward_by_node` claims units (trial point m, opening block, child)
from a shared cursor; the two blocks of one trial point usually run on different
workers. Each rank's backward runs only its own forward trials, so every trial point's
slab entry is rank-local.

1. **Per-stage pre-pass,** parallel over local trial points and rows, ended by one join
   before the claim loop:
   - W `[ω][k]` for every resident row of the child LP (frozen rows, then delta rows),
     from the setup σ·η table;
   - the delta-row bracket `[m][k]` for every local trial point.
2. **Per unit:**
   - `load_backward_lp` (frozen template plus the delta batch);
   - pin the state columns once (the per-unit phase of a two-phase patch: today's
     per-opening re-pin writes identical values);
   - compute p;
   - assemble B from the forward slab (frozen rows) and the pre-pass (delta rows).
3. **Per opening:**
   - ẑ_ω = p + σ·η_ω;
   - base-row shifts −G̃·ẑ_ω, load rows, NCS and slack bounds as today;
   - cut RHS = B + W[ω], one contiguous add, then one range push;
   - solve; store the opening's outcome.
4. **Per trial point, once both units finish:**
   - **Step 1 (per opening):** run the chain rule for each opening, then aggregate with
     today's `aggregate_cut_into`. This path is a shared function.
   - **Step 2 (aggregate-first):** aggregate the duals with the CVaR weights μ first,
     then run the chain rule once. Each opening stores its objective Q_ω, its non-lag
     state duals, the N-vector G̃ᵀπ̃_ω and its cut duals sparse by pool slot. Then
     `ḡ_h = −Σ_ω μ_ω (G̃ᵀπ̃_ω)_h + Σ_k ȳ_k β_{k,(h,0)}`,
     `β̄_{h,l} = ψ_{h,l} ḡ_h + Σ_k ȳ_k β_{k,(h,l+1)}` with `ȳ = Σ_ω μ_ω y_ω` in slot
     space, and `ᾱ = Σ_ω μ_ω Q_ω − dot_trial_state(β̄, x̂)`. Binding-activity tracking
     reads each opening's y before aggregation, because under pure CVaR some μ are 0.
     On a fan, aggregate per child. Openings are taken in canonical order.

Step 2 replaces step 1 for `by_node` and `by_scenario` only after step 1 passes the
chain-rule oracles; step 1 stays as the reference and for the sites in §5.4.

### 5.4 Other sites

| Site | Data flow |
|---|---|
| `by_scenario` (`process_by_scenario_backward`) | As `by_node`, with the per-stage W pass; each trial point is one unit, so its delta-row bracket is computed in the unit |
| `replicated` (`solve_replicated_outcome_slice`) | B once per child load (the slab hits only on the rank that ran that forward visit), plus W; chain rule per opening (step 1), because its payload is per-opening `objective ‖ subgradient` |
| Lower bound (`lb_evaluate_stage_0`) | Every row in `CutRowMap`, including pool-deactivated rows that stay in the LP, gets the canonical RHS at x₀ per root opening, recomputed at every evaluation |
| Simulation (`solve_simulation_stage`) | The canonical function runs on the fly (out-of-sample, no caches); the inflow outputs read ẑ and the `inflow_lags` output reads â |
| DCS (`score_violated_candidates`, `lazy_solve_preloaded`, `solve_lazy`) | Scoring is B + W over all active slots plus the non-lag product through `gemm_block`; appended rows carry the same RHS bits that selected them; the chain rule reads cut duals through `scratch.row_map` over the final resident set |
| Fused terminal slice (`capture_fused_terminal_slice`) | The terminal forward solve runs the chain rule from the leaf pool (resident) to the parent pool (extracted) by global dimension |
| Enumerated forward, terminal static (`run_stage_solve_terminal_static`) | As the sampled forward; 1:1 basis in the compact layout |

### 5.5 Reading cut duals

`extract_duals_from_view` copies the cut-row duals only when binding activity is
tracked (`tracks_binding_activity`: a budget or `dynamic`). The converted NEWAVE deck
runs `level1` with no budget, so today that slice is empty. Under the fold it is
mandatory chain-rule input; gating its read on the existing predicate would silently
drop every Σyβ term on the production path.

A per-stage setup flag, true when the resident pool's projection has lag slots, decides
whether the cut-row duals are read. Binding-activity recording keeps its own predicate.
`extract_state_duals_only` (DCS, replicated, fused slice) gains the same read, through
`scratch.row_map` on the DCS path. The slice spans the cut rows after the compact base
rows: frozen rows in `active_cuts()` order, then delta rows in
`active_delta_cuts(iteration)` order, mapped to slots through the precomputed
per-iteration list, never by slot arithmetic.

### 5.6 Caches

The fold's structural saving is the split into B and W (§5.1). On top of it, the items
below stay; the aggregate-first chain rule (§5.3) and DCS scoring from B + W (§5.4) are
the other two kept items.

| Cache | Content | Lifetime and key | Size per rank |
|---|---|---|---|
| Forward bracket slab | B over the frozen rows, written by the forward at `[m][t][k]`, read by the backward of the same trial point and child | One iteration; tagged (iteration, freeze generation) | 46 MB ÷ total ranks |
| Opening-noise table and per-stage W | σ·η per (stage, opening, hydro) at setup; W `[ω][k]` per stage in the pre-pass | Run (table); one stage (W) | 1.9 MB + 96 KB reused |
| Delta-row bracket | B over the delta rows for every local trial point, in the pre-pass | One stage | 0.3 MB |
| Pin once per unit | The state pins and B are set once per backward unit, not once per opening | One unit | none |

Rules that bind them:

- **Raw x̂.** B is computed from the raw trial state (`records[m, s].state`, the
  exchanged f64 values), never from the scaled column pin.
- **Validity.** The slab is valid from the end of the forward to the start of
  `run_cut_management`, which is the only re-freeze site, and only for the child the
  forward visited (on the PMO chain, every child). Checkpoints are written only at
  iteration end, so the slab is never persisted; a resume refills it on its first
  forward.
- **Misses.** A tag mismatch or a missing entry recomputes B and increments an
  in-memory miss counter that tests read. A miss is a performance event, never a
  correctness event. Misses are expected on a multi-child node for the children the
  forward did not visit.
- **DCS.** Under `dynamic` the forward does not load the frozen template, so the slab is
  keyed by slot over all active slots.

Rejected: caching p from the forward (it costs 1–2 µs to recompute); splitting the
base-row shift into G̃p + G̃σ·η (it would re-associate G̃ẑ for under 1 µs per opening);
caching the full lower-bound RHS (about 0.01 s per iteration); reusing the forward's
lag product in the ownership scan (under 4% of that scan); building the enumerated
External interior child cache (nothing to reuse); and solving the forward's opening
first (warm-start only, in-sample only).

**Expected out-of-LP time** on the NEWAVE PMO deck, all ranks, per iteration: about
8.9 / 9.4 / 12.6 CPU-s at 5 / 30 / 300 binding cuts per opening, against 36 / 40 / 75
for the fold with no caches. That is about 0.6% of LP time at 5 ms per solve and 2% at
1.5 ms. The remainder is dominated by the forward's own bracket pass, which no cache
removes because every forward state is new.

### 5.7 Warm start and stored bases

- `CapturedBasis` uses the compact layout throughout: `base_row_count` is the logical
  count minus N and column statuses cover the compact columns. `to_broadcast_payload`
  and `try_from_broadcast_payload` change together; only lengths change.
- `reconstruct_basis`, `ReconstructionTarget::from_template` and the terminal 1:1 apply
  read the compact template. Cut rows still match by slot identity.
- The forward basis seeds the backward, as today. RHS patches never invalidate a basis
  or a retained factorization.
- A logical basis restricts to a compact one by dropping the basic z columns, the
  nonbasic z rows and the nonbasic fixed lag columns, so warm-start quality is
  equivalent in exact arithmetic. Pricing state differs; pivot counts are similar but
  not equal, and no test compares them across formulations.
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
- Per-worker scratch is first touched inside its own worker thread.
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
| G̃ CSR, hydro → (compact row `u32`, coef `f64`), canonical hydro order | `StageContext` | run | small |
| ψ, b, σ in `[h][l]` order | `StageContext` | run | 16 KB per stage |
| σ·η per (stage, opening, hydro) | `StageContext` | run | 1.9 MB |
| Shared +∞ upper buffer; chain-rule slot maps; cut-row emission map; read maps | `StageContext` | run | small |
| Bracket slab `[m][t][k]`, tagged (iteration, freeze generation) | the forward's trajectory records | iteration | 46 MB ÷ total ranks |
| W `[ω][k]` in LP cut-row order (frozen rows, then delta) | `BackwardPassState` | stage | 96 KB, reused |
| Delta-row bracket `[m][k]` | `BackwardPassState` | stage | 0.3 MB |
| â, p, ẑ, the assembled B, the RHS | `ScratchBuffers` (per worker) | reused | about 5 KB each |
| Per opening: Q, G̃ᵀπ̃ (N-vector), sparse cut duals by slot | outcome arena | stage | about 6× smaller than today's dense per-opening coefficients |

New memory is about 48 MB at 1 rank and 14 MB at 4 ranks, under 0.4% of a 13.5 GB
per-rank budget. The slab is sized at freeze: when a freeze leaves more frozen rows than
its capacity, it grows there, and it is never resized anywhere else.

The slab is `[m][t][k]` because the forward already splits its records into disjoint
worker windows: it is writable in safe Rust with no false sharing and local first
touch, and each backward unit reads one contiguous run (3.2 KB at 400 frozen rows).
It is indexed by global pass id, never by worker id, so its contents are independent of
thread and rank shape.

### 6.4 Kernels

- **PAR evaluation:** `evaluate_par_batch` split into p and ẑ, today's summation order.
- **The canonical RHS kernel:** hand-written, 8 rows at a time in 256-element chunks.
  Accumulators start at +0.0; no FMA; no skipping of zero coefficients; lanes across
  rows or trials only. It replaces `gemm_block` here because S has an inner dimension of
  1,848, above the 513 to which `gemm_block` is pinned (`INNER_DIMS`). It serves the
  forward, the per-stage pre-pass, the lower bound, simulation and DCS.
- **Per opening:** RHS = B + W[ω], one contiguous add, then one range push.
- **Base-row shifts:** through the CSR, accumulating each row's shift from +0.0 in
  ascending canonical hydro id, then subtracting it from the template bound. This keeps
  the result independent of declaration order.
- **Chain rule:** reads only the rows of binding cuts; skipping an exact y = 0.0 gives
  the same bits as the dense sum. Order is ascending slot within each output element.
  Loops nest as 256-wide j-tiles, then binding rows, then openings, so accumulators stay
  in L1. Cost is about 0.44 µs per binding row per opening; step 2 runs it once per
  trial point.

### 6.5 Determinism

Each output element is computed by one thread in a fixed order; no floating-point
reduction crosses threads. Remote cuts keep ascending peer-rank order. Tests (§9.4):
the kernel property test, the cache on/off bitwise test, and the existing guard that
the unfused baseline kernel is selected, which fails if a build enables FMA.

### 6.6 Paging and NUMA

- **Code:** allocate once (the slab grows only at freeze); first-touch per-worker
  scratch in its own worker thread. No `madvise` helper.
- **Launcher:** transparent huge pages via `GLIBC_TUNABLES`, one rank per NUMA domain,
  core and memory binding. Both builds use identical settings, recorded per run.

## 7. Solver-crate changes

### 7.1 A contiguous-range row-bound push

```rust
fn set_row_bounds_range(&mut self, first_row: usize, lower: &[f64], upper: &[f64]);
```

- Covers rows `[first_row, first_row + lower.len())`. Name and doc are generic, with no
  mention of cuts.
- **A required trait method, no default.** A default delegating to `set_row_bounds`
  would need an index list: either a hot-path allocation or a silent slow path when a
  backend forgets to override. Every implementor is updated: HiGHS, CLP, and the test
  mocks in `novomodelo-sddp` and the solver crate.
- **HiGHS:** a new `novomodelo_highs_change_rows_bounds_by_range` in
  `csrc/highs_wrapper.{c,h}` wrapping `Highs_changeRowsBoundsByRange`, with its FFI
  declaration. It checks that indices fit in `i32`, asserts a non-error status, and adds
  to the existing `total_set_bounds_time_seconds`. Measured at 600 rows: 2.1 µs against
  39 µs for today's set-based call.
- **CLP (correct, not tuned):** patches the Rust-side bound copies over the range, then
  makes the same whole-array `chg_row_lower` / `chg_row_upper` calls `set_row_bounds`
  makes today. No shim.

The cut rows use the range push, once per solve, with the shared +∞ upper buffer. The
base-row shifts and load rows stay on the set-based `set_row_bounds`: the water rows
replace today's z rows at a similar count.

### 7.2 Infinite bounds

The `set_row_bounds` and `set_col_bounds` docs (`trait_def.rs`) say bounds must be
finite, but both backends already accept infinities: `tests/sentinel_inf_row_probe.rs`
pushes [−∞, +∞] row bounds on HiGHS, `tests/conformance.rs` pushes a +∞ column upper,
and the CLP C wrapper translates ±∞ to ±`DBL_MAX`. The docs of all three methods are
reworded to "±`f64::INFINITY` means no bound; the backend translates it".

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
   remaining columns; the lag part moves to the RHS.
2. **Intercept through the projection.** Computed only after the lag β are rebuilt;
   under aggregate-first it is Σμ·Q − β̄·x̂.
3. **State pinning by column bounds.** Lags have no LP column; their values enter only
   through the ẑ shifts and the cut RHS. There is still no state-fixing row.
4. **"Inflow noise enters only through the z rows"** becomes **"Inflow enters only as
   the ẑ shift of the z-coupled rows"**: the rows where the logical z column has
   entries, with the template's coefficients, plus the cut-row lag-0 term. Retargeted
   tests: `chronological_inflow_noise_moves_only_its_own_hydro`,
   `every_noise_dimension_patches_only_its_own_entity`,
   `inflow_slack_follows_z_routing_for_prefilling`.
5. **Truncation keeps the PAR state raw.** One evaluator call produces ẑ, which feeds
   the shifts, l\* and outgoing lag 0. The chain rule never differentiates l\*.
6. **Binding activity only for a configured reader.** Cut-row duals gain a second
   reader, the chain rule, decided by the per-stage lag-slot flag (§5.5).
7. **Append-only pool and slot-identity basis.** Bases use the compact layout; the base
   row count is logical − N. Across builds, stored bases are dropped with
   `UnusedStoredBases`.
8. **Fused terminal slice.** The terminal forward solve runs the chain rule from the
   leaf pool to the parent pool by global dimension.
9. **A cut's value at a state is canonical.** Unchanged, plus: the RHS bracket and the
   ownership values are separate quantities, and neither is computed from the other.

New:

10. **Every resident cut row's RHS is patched at every solve**: frozen and delta rows,
    DCS appends, lower-bound rows including deactivated ones, the simulation cut set,
    and terminal boundary rows.
11. **The cut-evaluation lag point has one owner.** Φ_t, M_z and M_a live in one place;
    the RHS function and the chain rule both go through it. A cached bracket is valid
    only for the (iteration, freeze generation, â bits, child) it was computed for.
12. **The cut-row RHS has one association.** ((α + S) + P) + W, each dot ascending,
    vectorized across rows or trials only, no `mul_add`, independent of batch shape;
    caches store only values this function produced. Pinned by the kernel property
    test, the cache on/off bitwise test and the thread-shape gate with its power
    preconditions.

The Kiro mirror is regenerated in the same commit
(`python scripts/gen/sync-kiro-steering.py`).

### 8.2 `docs/design/lp-builder-contract.md`

- **Job 2 (per-solve edits):** they become the RHS of the z-coupled rows and of the
  resident lag-bearing cut rows. `fill_z_inflow_patches` and the lag pins leave
  production and stay in `test-support`.
- **Job 4 (layout conventions):** in solver space, "outgoing column index = state
  index" no longer holds for lags, the z-row range disappears, and the base row count
  shrinks by N. The logical layout stays the formulation of record, so the digest and
  the manifest are unchanged.
- **New subsection, the fold layer:** the two index spaces and their typed indices, the
  precomputed maps, and the setup refusal.

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
  docstrings, such as the one in `tests/basis_trajectory_probe.rs`, are fixed when the
  surrounding code is touched.

## 9. Testing

Every new test goes into an existing test binary as a `mod`
(`.claude/rules/testing.md`, cost discipline).

### 9.1 Tier 3, LP structure (`template_integration.rs`)

- The compact template has N(L+1) fewer columns and N fewer base rows than the logical
  one, and the expected nnz difference.
- G̃ equals the logical CSC's z-column entries divided by c_z, in compact rows.
- The compact column and row scales are exactly the logical ones, restricted.
- Removed columns have cost exactly 0 and no lag column has an entry outside its z row;
  a structural test pins the setup refusal.
- `build_delta_cut_row_batch_into` and `push_cut_row` emit identical compact rows, and
  neither emits a lag entry.
- Each read map sends a consumer to the same logical entity, over every committed deck.
- `par_a_lag12_lp_coefficient.rs` keeps its logical-template assertions and gains a twin
  on the fold's ψ table.

### 9.2 Tier 4, analytical

Each test is labeled `/// derived:` and has a non-degeneracy twin.

| Test | Fixture | Mutation it catches |
|---|---|---|
| T4-1 | 1 hydro, PAR(1), no future | Sign of the G̃ term; κ |
| T4-2 | PAR(2) with one injected binding cut | Missing w_{(h,0)}; l vs l+1; an alias on the oldest lag; negated Σyβ; col_scale multiplied instead of divided (ψ₁ = 0.5) |
| T4-3 | PreFilling upstream → target | Using the owner's row instead of the target's |
| T4-4 | Binding generic `hydro_inflow` row (d54) | Missing −w·π_gen; equality or range rows shifting only one bound |
| T4-5 | Storage-only projection | Lag slots leaking into a storage-only cut |
| T4-6 | Truncation with ẑ < 0 | Differentiating l\* |
| T4-7 | Multi-resolution non-finalize stage | Breaking today's alias |

### 9.3 Twin tests on the production path

The logical template is the oracle, behind `test-support`.

- **T-KKT, logical replay (primary).** At every solve site of §5, scatter the folded
  solution into logical vectors and check: rc of z is 0 within rounding; the lag reduced
  costs divided by c_a equal the fold's β; z-row and cut-row residuals are within the
  primal tolerance, with complementarity; the objectives are equal. It uses the solve's
  own duals, so degeneracy cannot make it pass or fail spuriously.
- **T-KKT power guard.** One instance runs on a `level1`, no-budget fixture and fails
  loudly unless at least one binding cut carries a lag slot. It catches the cut-dual
  trap of §5.5.
- **T-ID, shared-dual identity.** Feeds the logical LP's duals into the chain rule and
  compares to its lag reduced costs, within |ψ|·ε_dual/c_z + 64u·Σ|terms|. It extends
  `basis_trajectory_probe.rs::lag_fold_check_stage`, which today feeds the z-row dual
  straight in and so never tests reconstruction from the water and generic duals.
- **T-BASIS, basis transfer.** Map the folded optimal basis to the logical layout and
  solve warm. It fails loudly unless the simplex takes 0 iterations, the fixture has a
  binding lag-bearing cut, and c_z ≠ 1 and c_a ≠ 1.
- **T-OBJ and T-VALID, independent cold solves.** Objectives agree within
  `ORACLE_ABS_TOL + ORACLE_REL_TOL·|Q|` (`tests/common/cut_oracles.rs`), and each
  formulation's cuts minorize the other's Q at sampled points. These never compare
  coefficients or primals.

### 9.4 Caching and determinism

- **RHS kernel property test:** bitwise against the scalar canonical function at 2,016
  lag slots, up to 10⁴ rows, batch widths 1..W and random row subsets.
- **Cache on vs off:** bitwise-identical results with the slab, W and the delta-row
  bracket enabled versus recomputed, over one backward stage and over a full small
  training run. A forced tag mismatch must recompute, increment the miss counter and
  give identical bits. Chain fixtures assert zero misses.
- **Aggregate-first:** matches the per-opening path to tolerance. A pure-CVaR fixture
  (some μ = 0) checks that binding counts still come from the per-opening duals.
- **`mpi_wire.rs` gates under the fold:** thread and rank invariance, opening order,
  and the opening-block scheduler with at least 2 blocks, on the uniform fixture, on
  d43, and on a leaf ≠ parent projection fixture for the fused slice. Power
  preconditions, failing loudly when unmet: frozen rows exist, the forward windows
  differ between the two shapes, and at least one binding cut is lag-bearing.
- **Declaration order:** a sort-contract unit test builds the G̃ CSR from permuted
  hydros and asserts the same canonical order.
- **No allocation:** the existing no-alloc harnesses (`forward_sampler_no_alloc.rs`,
  `stop_decision_no_alloc.rs`) extend to the folded per-solve path (prep, RHS assembly,
  pushes, chain rule) after warm-up.

### 9.5 Solver crate

In `tests/conformance.rs`, per backend: a tiny LP with a ≥ row pushed with a +∞ upper
through both `set_row_bounds` and `set_row_bounds_range`, checked against the
hand-derived optimum; a range push at a nonzero `first_row` changes only its rows.

### 9.6 Behavioral and golden

- `deterministic.rs` keeps its assertions. A value that moves is a finding to derive or
  explain, never one to re-record.
- `run_cut_oracles` runs on every deck with lag state, plus a cross-formulation oracle:
  the folded pool is checked against the logical successor LP.
- `extensive_form_oracle.rs`, `inflow_nonnegativity.rs` and the three retargeted pinning
  tests keep running.
- **Parity hashes:** nearly every hash moves, since even a PAR(0) case loses its z rows.
  `parity_baselines/` and `parity_baselines_clp/` are re-baselined together, in one
  commit, after every tier above passes; the commit message records why every hash
  moved.

### 9.7 Benches (Criterion, not run in CI)

The RHS kernel against `dgemm`; the per-stage pre-pass; the chain rule at 5–300 binding
cuts per opening, per opening versus aggregated; the solver push, set-based versus
range.

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

- Evaluating lag 0 of the cuts as a month average when a month spans several stages
  (the Φ_t owner of §4.4 is the seam for it).
- FPHA reading z for tailrace lateral inflow (G̃ is read from the CSC, so a new z
  coupling is picked up without special casing).
- Folding the truncation slack.
- A CLP range-bounds shim.
- A `dynamic` cut-selection A/B (it gets correctness tests only).
- A `madvise` helper, a tiled cut operand, and pool layout changes.

## 13. Facts this design rests on

Verified in code:

- Incoming lag columns appear only in their z row and in cut rows; z appears only in
  its z row, the water-balance rows (own or PreFilling target), generic `HydroInflow`
  rows and cut rows. No production code reads the z-row dual.
- The backward pins the same raw state the forward pinned for that trial point and
  stage; nothing touches the frozen rows or pool activity between the forward and the
  backward of one iteration; the opening tree is fixed at setup; each rank's backward
  runs only its own forward trials.
- Checkpoints are written only at iteration end.
- The NEWAVE converter emits `level1`, `by_node` with block size 10, `out_of_sample`
  training and simulation inflows, simulation `cut_set = active`, and a chain graph.
  The converted `config.json` itself has not been read; the pre-flight (§11.2) closes
  that.

Assumed, to be measured: LP time per solve (1.5–5 ms used above); binding cuts per
opening (5–300); the logical template's memory; CLP's range-push cost.
