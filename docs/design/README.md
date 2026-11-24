# Design docs

Design specifications, decision records, and proposals for the Novomodelo workspace.
Each doc carries a **status** at its top; this index is the map. Status vocabulary:

- **Live spec** — documents shipped behavior. The cited symbols exist in the tree;
  verify them before acting, but the described behavior is real.
- **Decision record** — a settled evaluation: what was measured, what was decided,
  and what not to re-attempt. Includes pre-registered experiments where noted.
- **Living register** — tracks the workspace's current reserved/deferred state; each
  entry is self-guarding and re-derived against the live tree.
- **Proposal** — a target design that is **not yet implemented**. Snapshot figures in
  a proposal are calibration-time context, not standing claims.
- **Design brief** — a precise problem statement and goal that precedes a design:
  the situation, the confirmed constraints, and what a solution must achieve. It
  proposes no solution; the design that answers it is written separately.

| Doc                                                                                              | Status                                                 | Summary                                                                                                                                                                                                                                                           |
| ------------------------------------------------------------------------------------------------ | ------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`policy-graph-limitations.md`](policy-graph-limitations.md)                                     | Live spec                                              | Where the node-native engine is a deliberate subset of a general Markovian policy graph, plus the probability/discount semantics and SDDP.jl convention mapping                                                                                                   |
| [`enumerated-traversal-distribution.md`](enumerated-traversal-distribution.md)                   | Live spec                                              | The current "replicate" MPI work-distribution for enumerated traversal, and the deferred intent to migrate to "broadcast"                                                                                                                                         |
| [`wire-format-evolution.md`](wire-format-evolution.md)                                           | Living register                                        | Append-only postcard discriminant ledger for tail-appended `VariableRef` and `BroadcastComputedParameter` variants, each tied to its guarding pin/round-trip tests, plus the no-reject-old-version-test rationale                                                |
| [`backward-warm-start-channels.md`](backward-warm-start-channels.md)                             | Decision record                                        | Backward-pass warm-start channels: one measured-and-closed (H3), two pre-registered, cluster-gated (H1, H2)                                                                                                                                                       |
| [`testing-architecture.md`](testing-architecture.md)                                             | Partially adopted (§5.2); the rest Proposal            | Workspace-wide target testing standard — layering, per-crate structure, the `test-support` convention, and the migration path                                                                                                                                     |
| [`anticipated-thermals-and-water-travel-time.md`](anticipated-thermals-and-water-travel-time.md) | Live spec                                              | How the anticipated-thermal commitment ring and the water-travel-time bucket ring share one lagged-delivery-ring substrate — input parsing, slot-count sizing, and LP entry for both — and the two post-study calendars (`thermal_stages`, `water_bucket_stages`) that date them                                                                              |
| [`hydro-productivity-and-stored-energy.md`](hydro-productivity-and-stored-energy.md)             | Live spec                                              | The scope x evaluator productivity model, the physical/operative storage-range split, the per-column productivity-override reach, `max_stored_energy`/stored-energy units, and the security-curve authoring recipe                                              |
| [`lp-builder-contract.md`](lp-builder-contract.md) | Live spec (normative) | The four jobs the stage-LP builder exists for (build, update, address, keep the algorithm's invariants) and the questions every change to the LP code must answer. |
| [`par-folding.md`](par-folding.md) | Decision record | Why the stage LP carries no PAR(p) inflow model (ẑ and the cut-row lag terms are RHS constants, lag coefficients come from the chain rule): the caches, kernels and dual reads it chose and rejected, where each invariant is pinned, the local comparison, and the pre-registered production comparison |

## Maintenance convention

A proposal that ships is **deleted from this directory** once its content lands in
its authoritative home (a `.claude/rules/*` rule, a code contract, an
`ARCHITECTURE.md` section, a live spec in this directory, or the user
documentation) — git history preserves the original. This keeps
`docs/design/` a set of live references, not an archive of superseded plans. When a
proposal is implemented, fold any durable rationale into the home it describes, then
remove the doc.
