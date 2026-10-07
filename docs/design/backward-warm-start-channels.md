# Backward-pass warm-start channels

> **Terminology**: this note predates a naming pass. In current code the "PN"
> scheduler is the opening-block scheduler
> (`training.parallelism.backward_scheduler`,
> `training/backward/opening_block.rs`), "LPT" is the hardest-first claim order
> (`hardest_first_block_order`), and the "TSP" opening order is the intrinsic
> shortest-chain order (`noise_key::apply_chain_order`; no config field).

> **Status**: Design evaluation — no implementation in this document. It records
> one measured-and-closed channel (H3) and evaluates two open, cluster-gated
> channels (H1, H2) for lowering backward-pass solve cost by warm-starting from a
> better basis. The measured finding is stated first and governs the rest: at
> plant-level network granularity a backward opening cannot be made free, so the
> remaining upside is fewer _pivots per solve_, never fewer _solves_.
>
> **Companions** (the two standalone design notes were retired with the naming
> pass; the code and rules below are the surviving anchors):
>
> - `crates/novomodelo-sddp/src/stochastic/noise_key.rs` (`apply_chain_order`) — the
>   warm-start-friendly opening solve order (shortest-chain path; σ-key kept
>   below three openings) that H1 rotates; the mechanism H1 builds on.
> - `crates/novomodelo-sddp/src/training/backward/opening_block.rs` — the
>   opening-block claim scheduler (hardest-first claim order) whose
>   per-`(stage, block)` structure H2 keys its cache on. Its retired note was
>   also the house precedent for the pre-registered, matched-epoch
>   decision-rule register used below.
> - `.claude/rules/sddp.md` — the opening-order, opening-block-scheduler, and
>   slot-identity basis-reconstruction contracts H1 and H2 must not disturb.

## The motivating question

The backward pass solves one stage LP per opening `ω ∈ Ω` at every visited trial
point: `|Ω|` solves per node. Those solves differ only in the opening's inflow
RHS and a few bounds on one resident LP, warm-continuing the dual simplex from
the previous opening's basis. This raises a structural question: could the
backward pass cost _far less_ than `|Ω|` solver calls per node?

The premise for a "yes" is basis persistence. A warm dual re-solve after an RHS
change does work proportional to how far the new optimal basis sits from the
current one. If consecutive openings were close enough that the optimal basis did
not move — zero pivots — then those openings would need no genuine solve: the
dual `π_ω` depends only on `(basis, costs)`, so an unchanged basis yields the same
`π_ω` in closed form, and `|Ω|` openings could collapse toward one solve plus
`|Ω| − 1` closed-form reads.

The measured answer is no, at plant-level granularity. Production-scale telemetry
over 12.1 million backward solves finds no persistence region to exploit: **0.00%
of chain solves finish with zero simplex iterations** (still 0.00% at ≤ 10
pivots; 0.01% at ≤ 50), the **median chain hop is 629 pivots**, and pivots per
hop **rise** as the cut pool grows. Every opening is a genuine solve; none is
free. Small backward/forward solve-time ratios seen elsewhere are not a solver
technique — they are a horizon effect (see Resolved attribution). What is left on
the table is the _pivot count_ of each of those `|Ω|` genuine solves, which a
better seed basis can lower. H1 and H2 pursue that pivot count; H3 pursued the
empty persistence region and is closed.

## H3 — basis-persistence short-circuit (measured and closed)

**Do not re-propose.** This channel was measured at production scale and refuted
before this evaluation; it is recorded here so the mechanism is not rediscovered
and re-attempted.

**Mechanism.** The dual solution of a stage LP depends only on `(basis, costs)`.
Between consecutive openings only a fixed-sparsity patch changes — the inflow RHS
and a handful of bounds. If the incumbent basis provably stayed optimal across
that patch, the solver call could be replaced by a cached feasibility check on
the patched RHS plus a closed-form intercept from the unchanged `π` — no simplex
iterations, no factorization update. The backward pass would then pay one genuine
solve per persistence run instead of one per opening.

**Refutation.** The short-circuit needs a non-empty persistence region: openings
whose optimal basis does not move under the patch. Production-scale telemetry over
12.1 million backward solves shows that region empty at plant-level network
granularity — 0.00% zero-iteration chain solves, 0.01% at ≤ 50 pivots, a median
of 629 pivots per hop, and a rising trend as cuts accumulate. A cached-basis
short-circuit that never triggers is pure overhead. The channel is **closed**:
there is no feature to design, so this document records the mechanism and the
measured outcome and stops there — no feasibility cache, no dual-getter
inventory, no tolerance or audit machinery.

The same 629-pivot median that closes H3 is why H1 and H2 remain worth measuring:
a seed basis hundreds of pivots from the optimum still has hundreds of pivots to
give back, even though it can never give back the _solve_.

## Resolved attribution

A backward pass that reports a small backward/forward solve-time ratio invites the
inference that its backward openings are somehow cheap — the persistence effect H3
chased. The measured attribution is different: a horizon effect, not a solver
trick.

**Hybrid horizons.** A production study can mix granularities across its horizon:
near stages carry the full plant-level network, while far stages carry
low-dimensional aggregate reservoir models. An aggregate-stage LP is small, so its
solve is cheap per opening regardless of warm starting. A horizon dominated by
aggregate stages therefore shows a low aggregate backward/forward ratio — an
artifact of _what most stages model_, not of any basis-reuse mechanism on the
plant-level stages.

**Matched granularity is equivalent.** Compared at matched plant-level
granularity — like against like — the marginal cost of one backward opening is
the same across implementations: **0.56 versus 0.53 forward-solve equivalents**.
Neither implementation makes a plant-level backward opening cheaper than roughly
half a forward solve; the gap between whole-run ratios is the horizon mix, not the
per-opening mechanism.

**Honest residuals.** Two measurable effects remain and are _not_ claimed closed:

- **Resident-cut-row trajectories under cut selection.** Pivots per hop rise with
  the cut pool; how the active cut set is selected and retained changes the
  resident row count each backward solve carries, and thus its pivots. This is a
  cut-management lever, orthogonal to the seed basis.
- **Solver-side steepest-edge / factorization persistence across re-solves.** The
  dual simplex carries steepest-edge (DSE) pricing weights and a factorization
  that could in principle persist across the opening chain. The backward solver
  profile's `steepest_edge_devex_fallback_threshold` field is the related lever (it governs
  when DSE weights are dropped for Devex pricing). Quantifying its effect on chain
  pivots is left as measurement, not asserted here.

## H1 — chain anchoring at the sampled opening

**Mechanism.** In the frozen backward arm, each trial point's opening chain is
solved in the installed `solve_order` permutation (`build_noise_key_table`;
`BackwardOpeningOrder::Tsp` by default computes a lowest-cost Hamiltonian path,
`tsp_path`, over the openings' noise-prefix L2 distances). The chain's _head_ —
`solve_order` position 0 — is warm-started from the forward-captured basis for
that `(trial point, stage)` slot (`CapturedBasis`, written at the forward solve).
But the forward pass solved and captured at exactly one opening: the _sampled_
opening `ξ`. Its basis is near-exact for `ξ` and only for `ξ`. Because the TSP
path starts at position 0, generally not `ξ`, the head pays a `ξ → ω₀`
perturbation — it applies a near-exact basis to the wrong opening.

H1 removes that jump. Compute the opening order as a _cycle_ (Hamiltonian cycle)
rather than an open path, and rotate the cycle per trial point so the chain
_starts at `ξ`_. The head then warm-starts from the forward capture applied to its
own opening — near-exact — and the chain walks the cycle from there. Only the
solve order changes; each opening's outcome is still written and aggregated by
canonical `ω` (the opening-order contract in `.claude/rules/sddp.md`), so the cut
stays an expectation over `ω` in canonical order.

**Gain bound.** The head is one of `|Ω|` solves, and anchoring it at `ξ` can only
recover the head's `ξ → ω₀` excess pivots, so the ceiling is **≈ one solve in
`|Ω|`** per chain — a `1/|Ω|` effect that shrinks as `|Ω|` grows. It is a modest
channel, bounded by construction; the head is, however, the chain's coldest solve,
so the wall fraction recovered can exceed the naive pivot fraction.

**Output shift and re-baseline cost.** Unlike a result-neutral reorder, H1 is
_not_ byte-neutral. Rotating the chain changes the basis each opening warm-starts
from; at a degenerate optimum a differently-warmed solve can settle on a
different-but-equally-valid vertex with different duals — the hot ≠ cold
divergence the determinism contract permits. The generated cut can therefore
differ bit-for-bit from the current baseline, so enabling H1 forces a golden
parity re-baseline. It does **not** relax the determinism contract: the anchor
`ξ` is a deterministic function of the reproducible forward sample, so the rotated
order is identical across thread and rank counts, and run-to-run reproducibility
and declaration-order invariance are preserved — the determinism suites remain the
correctness gate.

**Interaction with PN.** Under the opening-block scheduler each block is its own
sub-chain with its own head, so only the block containing `ξ` can be anchored at
`ξ`; the other blocks' heads still pay a jump. H1 is therefore cleanest on the
single-chain trial-point path; the other block heads are H2's target.

**Pre-registered go/no-go (cluster-measured).** Arms: two builds — H1-on (cycle +
per-trial rotation to `ξ`) versus H1-off (current path from position 0) — same
backward scheduler and case. Primary metric: total backward `simplex_iterations`
on a production-scale plant-level case, a deterministic,
work-distribution-invariant function of the configuration; corroborated by the
matched-epoch backward wall. **Go** if H1-on lowers total backward pivots beyond
the cluster wall-noise band with the thread/rank determinism suites green and the
re-baseline reproducible across rank counts. **No-go** if the reduction sits
inside noise (the `1/|Ω|` ceiling is small by construction) or if reproducibility
or rank-invariance regresses.

## H2 — cross-iteration backward-basis cache per `(stage, block)`

**Mechanism.** Under the opening-block scheduler (`process_stage_backward_opening_block`)
each work unit is an opening-block of one trial point, its head anchored on the
forward capture and warm-continued across the block. The forward capture carries
the primal geometry of the sampled `ξ` but not the `ω`-specific dual geometry of
the block's other openings — the forward pass never solved them. H2 supplies that
geometry from history: cache, per `(stage, block)`, the backward basis produced
for that block in the _previous_ iteration, and warm the block head from it
instead of from the forward capture. Same block, adjacent iteration — the openings
are identical and the cut pool has grown by only that one iteration's cuts.

**Slot-identity absorbs the one-iteration delta.** The cached basis was captured
against the previous iteration's cut pool; this iteration's LP carries the cuts
added since. That delta is exactly what slot-identity reconstruction handles:
`reconstruct_basis` matches stored cut rows to current LP rows by `CutPool` slot
identity, copies their status verbatim, and seeds each new cut row `BASIC` (the
append-only cut-pool contract). A one-iteration-older backward basis is therefore
directly reconstructible onto the current LP with no shape mismatch — the delta is
absorbed, not rejected.

**Memory sizing.** Cache size is (number of cached bases) × (per-basis size). A
per-`(stage, block)` cache holds `Σ_stages n_blocks(stage)` bases
(`pn_block_count`), which at production block counts is **order tens of MB per
rank**. Keying per-`(stage, ω)` instead multiplies the count by the block size —
`Σ_stages n_openings(stage)` bases — pushing the cache to **order hundreds of MB**
for no dual information the block head needs. Per-`(stage, ω)` is therefore
**rejected**; the block granularity is the sizing this cache is built on. (The
per-basis term itself grows with the cut pool, so the cache grows over iterations
under either keying; block-keying is what holds the absolute footprint to tens of
MB.)

**Fallback.** On the first iteration, or any `(stage, block)` with no cached entry
(a cache miss), the block head falls back to the forward-capture anchor — the
current behavior. H2 is strictly additive: it replaces a forward-basis seed with a
same-block backward seed only when one exists, and never removes the fallback.

**Output shift.** As with H1, a different seed can land a degenerate solve on a
different vertex, so H2 is output-shifting and forces a re-baseline; the cache key
`(stage, block)` and its contents are deterministic functions of reproducible
state, so thread/rank determinism and run-to-run reproducibility are preserved and
remain the correctness gate.

**Pre-registered go/no-go (cluster-measured).** Arms: H2-on (per-`(stage, block)`
cross-iteration cache, forward-capture fallback) versus H2-off (forward-capture
anchor only), under the opening-block scheduler on a production-scale plant-level
case. Primary metric: total backward `simplex_iterations` and matched-epoch
backward wall. **Go** if H2-on lowers backward pivots and wall beyond the cluster
noise band, the tens-of-MB/rank footprint holds, the determinism suites stay
green, and the re-baseline is reproducible across rank counts. **No-go** if the
reduction sits inside noise, the footprint exceeds the block-keyed sizing, or
reproducibility or rank-invariance regresses.

## Decision record

**H3 — closed (measured, refuted, no-go).** The basis-persistence short-circuit
requires a persistence region that production-scale telemetry over 12.1 million
backward solves shows is empty at plant-level granularity (0.00% zero-iteration
chain solves; median 629 pivots per hop; rising with the cut pool). Recorded so it
is not re-proposed; no implementation follows.

**H1 — open, modest upside, cluster-gated.** Anchor each backward chain at the
sampled opening `ξ` by rotating the opening _cycle_. Ceiling ≈ one solve in `|Ω|`.
Go/no-go pre-registered above.

**H2 — open, cluster-gated.** Cache the previous iteration's backward basis per
`(stage, block)` and warm PN block heads from it, with a forward-capture fallback
on miss. Footprint order tens of MB per rank at block granularity. Go/no-go
pre-registered above.

Both open channels lower _pivots per solve_, never the _solve count_ — that
distinction is the whole finding. With a median of hundreds of pivots per chain
hop, a better seed basis has real work to recover; with 0.00% of hops idle, no
scheme that tries to _skip_ the solve can. H1 and H2 are the seed-basis levers
that survive that fact.
