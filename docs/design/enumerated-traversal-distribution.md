# Enumerated-traversal work distribution: replicate today, broadcast tomorrow

**Status:** the current engines use the _replicate_ distribution described below.
This document records a deliberate, deferred intent to migrate the enumerated
forward, backward, and simulation passes onto the _broadcast_ distribution, and
explains the difference in enough detail to execute that migration as one
sweep. Nothing here is implemented beyond what the "Current state" section
states.

## Problem

Enumerated (exhaustive) traversal solves a finite acyclic scenario graph by
visiting every root→leaf path. The nodes of that graph must be distributed
across MPI ranks and, within a rank, across worker threads. Two distribution
models are correct; they differ in what they replicate versus what they
communicate. The choice is invisible to results — every model is required to
be declaration-order-invariant and bit-identical across thread and rank shapes
— but it drives compute cost, communication, memory, and which graph shapes are
representable at all.

The unit of parallelism, in both models, is the same and is not in question: a
stage-synchronous outer loop (solve stage `t` for the whole graph before stage
`t+1`), parallelizing across the distinct nodes within a stage. The models
differ only in how those within-stage nodes, and their solved states, are
placed across ranks.

## Model A — replicate (current)

Distribute the **leaf paths** across ranks: rank `r` owns a contiguous slice of
the canonical root→leaf paths and marks every node on those paths as its own.
Because a path includes all of its ancestors, a rank owns complete subtrees
under its path slice.

- At each stage, the rank's worker threads claim its own distinct stage-`t`
  nodes from a shared cursor, solve each **once**, and scatter results into a
  rank-local, node-keyed arena in a canonical (claim-order-independent) order.
- A node at stage `t+1` reads its parent's outgoing state from that same
  rank-local arena — the parent was necessarily solved earlier on this rank.
- A node shared by two ranks' path slices (any ancestor common to both — in the
  limit, a node on every path) is **replicated**: each rank solves it
  independently. A determinacy contract guarantees both ranks reach the
  identical vertex, so replication is exact, not approximate.

There is **no inter-rank communication during the sweep**, and per-path output
is assembled entirely locally (every node on a rank's path is in its own
arena).

**Strengths.**

- Zero sweep-time communication; trivially network-scalable.
- Simple determinism: independent rank-local arenas, one within-rank canonical
  scatter, replication exact by contract.
- Local output assembly — no cross-rank gather to emit per-path results.
- Near-optimal for a **short deterministic trunk with a wide terminal fan**
  (the DECOMP shape): the fan leaves, which dominate the work, are partitioned
  across ranks with no redundancy; only the short trunk is re-solved per rank.

**Weaknesses.**

- **Redundant compute on shared ancestors.** The extra work is the sum over
  nodes of `(ranks sharing the node − 1)` solves. Negligible for a short shared
  prefix; it grows with a **long shared prefix** (a deep deterministic stem that
  branches late), in the limit re-solving the whole prefix on every rank.
- **Does not extend to recombining DAGs.** "Own complete subtrees" assumes a
  single-predecessor forest. A node reachable by paths owned by different ranks
  is either re-solved per path (blowup) or needs a dedup that dissolves the
  clean rank-local model. Replicate is fundamentally tree-shaped.
- **Path-slice load imbalance** when subtrees are uneven in cost.
- Per-rank memory holds the full arena for the rank's owned subtree, resident
  until per-path output re-expansion.

## Model B — broadcast (future)

Distribute the **distinct nodes** of each stage across ranks: each node is
solved exactly **once** globally. After each stage, the freshly solved node
states are exchanged (an all-gather in canonical order) so that every rank
holds every stage-`t` state before stage `t+1` begins.

**Strengths.**

- **No redundant solves** — optimal compute regardless of graph shape; this is
  the decisive win over replicate for long-shared-prefix and deep graphs.
- **Natural for recombining DAGs** — global node dedup is exactly what a
  multiple-predecessor node wants: solve once, everyone consumes it.
- Even node-level load balance, independent of tree shape.
- Every rank ends each stage holding every node's state — see "Why broadcast
  also closes the backward gap".

**Weaknesses.**

- **A per-stage collective** — one canonical-order all-gather per stage, whose
  bandwidth scales with the stage's node count times the state size, and whose
  latency is one round per stage. This is exactly the exchange replicate avoids.
- **Order-invariant reduction discipline** — the collective must deliver in a
  canonical order and any downstream reduction must be order-independent to keep
  bit-identical results across shapes. More coordination surface than
  replicate's independent arenas (the backward's existing exchange already meets
  this bar, so the machinery is precedented, not novel).
- **Cross-rank output assembly** — a path's nodes are scattered across ranks, so
  emitting per-path output needs a gather that replicate does not.

## Comparison

| Axis                     | Replicate (A)                   | Broadcast (B)                           |
| ------------------------ | ------------------------------- | --------------------------------------- |
| Shared-ancestor compute  | re-solved once per sharing rank | solved once globally                    |
| Sweep-time communication | none                            | one canonical all-gather per stage      |
| Recombining DAGs         | not representable               | natural                                 |
| Load balance             | by path slice (shape-sensitive) | by node (shape-independent)             |
| Per-path output          | assembled locally               | needs a cross-rank gather               |
| Determinism surface      | small (rank-local arenas)       | larger (ordered collective + reduction) |
| Best-fit shape           | short trunk / wide fan          | long prefix, deep, or DAG               |

Memory is shape-dependent and not a clean discriminator: replicate holds each
rank's owned subtree resident; broadcast holds each rank's assigned nodes plus
what the exchange and output-assembly require. Neither dominates in general.

## Current state

- The enumerated **forward** pass (`run_enumerated_forward`) uses **replicate**:
  path-partitioned across ranks, per-rank dedup, shared ancestors replicated,
  no sweep-time exchange.
- The enumerated **simulation** pass uses **replicate**, mirroring the forward
  step-for-step (a fork of the sampled simulation on the traversal axis; the
  sampled arm is byte-identical). It shares the forward's distribution
  primitives (the claim cursor, the canonical scatter, the own-set marking and
  node arena) rather than forking its own copy.
- The enumerated **backward** pass uses a **third** variant that is neither A nor
  B: it splits a single node's openings across all ranks to parallelize the cut
  aggregation, but consumes each node's incoming state without a matching
  exchange. Under replicate's forward, a rank that does not own a node holds a
  zero-filled state for it, so at world size ≥ 2 on a graph with interior
  branching the split reads an invalid state. That configuration is therefore
  **hard-rejected at study construction** (`TrainingSession::new`) rather than
  silently mis-computed; the full cross-rank interior-node state exchange that
  would lift the rejection is deferred.

## Why replicate now

- The graph shapes where replicate is weak are gated off today: recombining DAGs
  are rejected at study construction, and deep multi-stage branching hits the
  `Kᵀ` overflow guard. Replicate is therefore not a compromise for any workload
  that is currently runnable.
- DECOMP — a short deterministic trunk with a wide terminal fan — is
  replicate-optimal.
- It preserves forward ≡ simulation architectural symmetry with the least code,
  reusing one shared distribution substrate across both passes.

## The desired change and why it is one sweep

The broadcast model is the general one: it is the only model that admits
recombining DAGs and that keeps compute optimal on long-prefix and deep graphs.
The intent is to migrate the forward, backward, and simulation passes onto it
**together, in one deliberate sweep**, for three reasons:

1. **A shared substrate.** All three passes already route their distribution
   through the same hoisted primitives (the claim cursor, the canonical scatter,
   the own-set/arena pattern). Broadcast is introduced by changing that one
   substrate — the per-stage canonical all-gather and the order-invariant
   reduction — in a single place, not three.

2. **Broadcast also closes the backward gap.** Under a broadcast forward, every
   rank holds every node's outgoing state at the end of each stage. That is
   exactly the precondition the node-native backward's openings-split needs and
   lacks today. Migrating the forward to broadcast makes the backward's
   world-size-≥-2 interior-branching rejection unnecessary — the deferred
   interior-node state exchange becomes a special case of the broadcast the
   forward already performs, rather than a separate mechanism. The backward is
   thus not an afterthought in this migration; it is a primary beneficiary, and
   folding it in is the point of doing all three at once.

3. **One determinism proof.** The ordered-collective + order-invariant-reduction
   contract is proven and gated once, uniformly, instead of re-derived per pass.

## Migration seam and trigger

The seam is the shared distribution substrate named above; keeping every pass's
distribution logic there (never in per-pass forks) is what keeps the conversion
a single change. The migration is warranted when any of the following becomes
real: a workload needs recombining DAGs; a workload needs deep multi-stage
branching beyond the current `Kᵀ` guard; the backward's world-size-≥-2
interior-branching case must run; or the redundant shared-ancestor compute is
measured to be a bottleneck on a genuine graph. Until then, replicate is the
correct, simpler engine, and this document is the standing record of the
direction.
