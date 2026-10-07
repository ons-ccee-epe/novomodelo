//! Cut synchronization across MPI ranks after the backward pass.
//!
//! Each rank's newly generated cuts are exchanged via a per-stage `allgatherv`
//! of serialized records in the [`cut::wire`] format, so the FCF is bit-for-bit
//! identical across all ranks at the end of each iteration (the next forward
//! pass rebuilds the LP from it).
//!
//! `sync_level_records` inserts only **remote** cuts: the backward pass already
//! inserted the local rank's own cuts, so the local segment of the receive
//! buffer is skipped — re-inserting it would double-count cuts.
//!
//! The `allgatherv` acts as an implicit barrier; no explicit `comm.barrier()`
//! is needed. Buffers pre-allocated in [`CutSyncBuffers::new`] are reused so the
//! per-stage exchange is allocation-free.
//!
//! Serialization/version handling is delegated to [`cut::wire`]; the
//! version-reject contract (and the per-record generating-node key) lives
//! there, not here.
//!
//! [`cut::wire`]: crate::cut::wire

use cobre_comm::{Communicator, per_rank_counts, prefix_displs};

use crate::{
    FutureCostFunction, SddpError,
    cut::wire::{CutWireHeader, cut_wire_size, deserialize_cuts_from_buffer_into, serialize_cut},
    setup::NodeId,
};

/// Pre-allocated byte buffers for gathering cut wire records across all MPI
/// ranks via [`Communicator::allgatherv`] with `T = u8`.
///
/// # Buffer layout
///
/// | Buffer      | Capacity                                             | Description                                                   |
/// |-------------|------------------------------------------------------|---------------------------------------------------------------|
/// | `send_buf`  | `max_cuts_per_rank * cut_wire_size(n_state)`         | This rank's serialized cut records                            |
/// | `recv_buf`  | `max_cuts_per_rank * num_ranks * cut_wire_size(n_state)` | All ranks' serialized cut records in rank-major order     |
/// | `counts`    | `num_ranks`                                          | Per-rank byte count (`actual_cuts * record_size`)             |
/// | `displs`    | `num_ranks`                                          | Per-rank byte displacement (`sum of preceding counts`)        |
///
#[derive(Debug, Clone)]
pub struct CutSyncBuffers {
    /// Only the leading `actual_cuts * record_size` bytes are sent each call.
    send_buf: Vec<u8>,

    /// Rank `r`'s records occupy `recv_buf[displs[r]..displs[r] + counts[r]]`
    /// after `allgatherv`.
    recv_buf: Vec<u8>,

    /// Per-rank byte count for `allgatherv`, recomputed each `sync_level_records` call.
    counts: Vec<usize>,

    /// Per-rank byte displacement for `allgatherv`: entry `r` = sum of
    /// `counts[0..r]`.
    displs: Vec<usize>,

    /// Maximum cut-coefficient count across all pools (the global
    /// `StateSpace::n_state`); sizes buffer capacity only.
    ///
    /// The wire stride for a pool's exchange is
    /// `cut_wire_size(fcf.pools[pool].state_dimension)` — the pool's own
    /// dimension — never this value. A reduced pool carries fewer
    /// coefficients than this max, so packing at `max_record_size` while reading
    /// per-pool coefficient slices would mis-stride every record and corrupt
    /// remote ranks' deserialized cuts.
    max_n_state: usize,

    num_ranks: usize,

    /// Maximum wire record size `cut_wire_size(max_n_state)`; the buffer-capacity
    /// upper bound, not the per-stage stride (see [`max_n_state`](Self::max_n_state)).
    max_record_size: usize,

    /// Per-rank expected cut counts: entry `r` is the number of cuts rank `r`
    /// generates per stage per iteration, sizing each rank's `allgatherv` slot.
    per_rank_cuts: Vec<usize>,

    /// Deserialization scratch for cut headers; grown lazily, never shrunk.
    deserialize_headers_buf: Vec<CutWireHeader>,

    /// Deserialization scratch for coefficients (flat layout); grown lazily,
    /// never shrunk.
    deserialize_coefficients_buf: Vec<f64>,

    /// Per-pool `(pool, record_size, n_local)` packing plan built by
    /// [`Self::sync_level_records`], reused across calls via
    /// [`std::mem::take`] (never a fresh `Vec::new` per call) so the
    /// level-batched cut exchange stays allocation-free.
    level_plan_scratch: Vec<(usize, usize, usize)>,

    /// Gathered per-`(rank, pool)` local cut counts for the current level,
    /// `num_ranks * pools.len()` entries in rank-major/pool-minor order. Sizing
    /// each peer's per-pool segment by ITS OWN count is what lets a multi-node
    /// level spread across >1 rank; a single pool collapses it to the uniform
    /// per-rank plan, keeping the chain path byte-identical. Counts ride the wire
    /// as `u64` (`usize` is not an `MpiDatatype`). Grown lazily inside
    /// [`Self::sync_level_records`], reused across calls (never a hot-path alloc).
    per_pool_rank_counts: Vec<u64>,

    /// This rank's own per-pool counts, the send buffer for the per-`(rank, pool)`
    /// count `allgatherv` (length `pools.len()`); reused across calls.
    pool_count_send_scratch: Vec<u64>,
}

impl CutSyncBuffers {
    /// Construct pre-allocated cut synchronization buffers for the given
    /// topology, assuming a uniform distribution of `max_cuts_per_rank` cuts
    /// per rank.
    ///
    /// # Arguments
    ///
    /// - `n_state` — state dimension (number of cut coefficients per cut).
    /// - `max_cuts_per_rank` — maximum number of cuts any rank generates per
    ///   stage per iteration. Used to pre-allocate buffer capacity.
    /// - `num_ranks` — total number of MPI ranks (`comm.size()`).
    #[must_use]
    pub fn new(n_state: usize, max_cuts_per_rank: usize, num_ranks: usize) -> Self {
        Self::with_distribution(
            n_state,
            max_cuts_per_rank,
            num_ranks,
            max_cuts_per_rank * num_ranks,
        )
    }

    /// Construct buffers for non-uniform work distribution.
    ///
    /// When the total number of forward passes does not divide evenly among
    /// ranks, the first `total_forward_passes % num_ranks` ranks each handle
    /// one extra forward pass. This constructor sizes buffers for the maximum
    /// per-rank count and records each rank's expected count for correct
    /// `allgatherv` displacements.
    ///
    /// # Arguments
    ///
    /// - `n_state` — state dimension (number of cut coefficients per cut).
    /// - `max_cuts_per_rank` — maximum cuts any rank generates per stage per
    ///   iteration. Used to size the send buffer.
    /// - `num_ranks` — total number of MPI ranks.
    /// - `total_forward_passes` — total forward passes across all ranks. Used
    ///   to compute per-rank expected cut counts.
    #[must_use]
    pub fn with_distribution(
        n_state: usize,
        max_cuts_per_rank: usize,
        num_ranks: usize,
        total_forward_passes: usize,
    ) -> Self {
        let max_record_size = cut_wire_size(n_state);
        let send_cap = max_cuts_per_rank * max_record_size;

        let per_rank_cuts = per_rank_counts(total_forward_passes, num_ranks);
        let recv_cap: usize = per_rank_cuts.iter().sum::<usize>() * max_record_size;

        let counts: Vec<usize> = per_rank_cuts.iter().map(|&c| c * max_record_size).collect();
        let displs = prefix_displs(&counts);

        Self {
            send_buf: vec![0u8; send_cap],
            recv_buf: vec![0u8; recv_cap],
            counts,
            displs,
            max_n_state: n_state,
            num_ranks,
            max_record_size,
            per_rank_cuts,
            deserialize_headers_buf: Vec::new(),
            deserialize_coefficients_buf: Vec::new(),
            level_plan_scratch: Vec::new(),
            per_pool_rank_counts: Vec::new(),
            pool_count_send_scratch: Vec::new(),
        }
    }

    /// Serialize `pool`'s active cuts generated at `iteration` into
    /// `send_buf[send_offset..]` at the pool's own wire stride, returning the
    /// count packed. Used by the level-batched packing in
    /// [`Self::sync_level_records`] (contiguous per-pool offsets).
    #[allow(clippy::cast_possible_truncation)]
    fn pack_pool_into(
        &mut self,
        fcf: &FutureCostFunction,
        pool: usize,
        iteration: u64,
        record_size: usize,
        send_offset: usize,
    ) -> usize {
        let cut_pool = &fcf.pools[pool];

        // Wire stride is the pool's own dimension, never the cached
        // `max_record_size` (see the `max_n_state` field doc): packing at the global
        // max while the slice below reads `cut_pool.state_dimension` coefficients
        // mis-strides every record once a pool is reduced.
        debug_assert!(
            record_size <= self.max_record_size,
            "record size {record_size} exceeds buffer-capacity max {}",
            self.max_record_size
        );

        let mut n_cuts = 0usize;
        for slot in 0..cut_pool.populated() {
            if !cut_pool.is_active(slot) {
                continue;
            }
            let meta = cut_pool.metadata(slot);
            if meta.iteration_generated != iteration {
                continue;
            }

            let start = send_offset + n_cuts * record_size;
            debug_assert!(
                start + record_size <= self.send_buf.len(),
                "pack_pool_into: {} bytes required, exceeds send_buf capacity {}",
                start + record_size,
                self.send_buf.len()
            );

            let coeffs = cut_pool.coefficient_row(slot);
            serialize_cut(
                &mut self.send_buf[start..start + record_size],
                slot as u32,
                meta.node.0,
                iteration as u32,
                meta.forward_pass_index,
                cut_pool.intercept(slot),
                coeffs,
            );
            n_cuts += 1;
        }

        n_cuts
    }

    /// Exchange every pool of one reverse-topological cut-sharing level in a
    /// SINGLE `allgatherv`, inserting only remote cuts (each pool's local cuts
    /// were already inserted by the node's backward compute). Batching per level
    /// — never one collective per node — keeps the collective count scaling with
    /// level count, not node count. Returns `(local_total, remote_total)` cut
    /// counts summed over the level's pools.
    ///
    /// A small integer `allgatherv` first exchanges every rank's per-pool local
    /// count, so each rank sizes each peer's per-pool segment by ITS OWN count —
    /// per-pool routing lets a pool's local count differ from the uniform
    /// per-rank plan (a sibling pool at a multi-node level takes only its own
    /// visitors). With a single pool the gathered counts collapse to that plan
    /// and the send bytes, counts, displacements, and remote inserts are exactly
    /// the uniform single-pool exchange —
    /// the generalization is reached by the counts, never by a shape branch.
    ///
    /// `replicated_pools` names the pools whose backward cut was aggregated by
    /// the replicated per-node solve (`training::backward::replicated`): every
    /// rank already appended the bit-identical cut locally, so re-distributing it
    /// here would insert it once per rank — the cut multiplied by rank count, a
    /// silent wrong bound. Such a pool is EXCLUDED structurally, by aggregation
    /// kind and never by a cut count (a count heuristic would drop a legitimately
    /// exchanged cut on a non-replicated pool): it is not packed, not counted in
    /// the per-`(rank, pool)` exchange, and not deserialized from any peer —
    /// generalizing the local-rank-segment skip (a replicated pool's segment is
    /// skipped for EVERY rank, not only the local one). The sampled level driver
    /// passes `&[]` (its pools are all trial-point-distributed), so the exchange
    /// is byte-for-byte unchanged there.
    ///
    /// # Errors
    ///
    /// Returns `Err(SddpError::Validation(_))` only for a MALFORMED count
    /// exchange — a gathered per-`(rank, pool)` count that overruns the grown
    /// receive buffer, or a local total disagreeing with the packed total — so a
    /// corrupt exchange fails loudly rather than deserializing at the wrong
    /// stride. A well-formed multi-node level whose pools carry differing
    /// per-`(rank, pool)` counts is exchanged correctly. Returns
    /// `Err(SddpError::Communication(_))` if either underlying `allgatherv` fails.
    pub fn sync_level_records<C: Communicator>(
        &mut self,
        pools: &[usize],
        replicated_pools: &[usize],
        fcf: &mut FutureCostFunction,
        iteration: u64,
        comm: &C,
    ) -> Result<(usize, usize), SddpError> {
        let my_rank = comm.rank();
        let expected_for_me = self.per_rank_cuts[my_rank];
        let is_replicated = |pool: usize| replicated_pools.contains(&pool);

        // Grow the send/recv buffers to the level's total across its pools when a
        // multi-pool level exceeds the per-pool capacity the constructor sized —
        // a no-op on the single-pool levels the shipped shape produces, so the
        // chain path never resizes and stays byte-identical. `expected_for_me`
        // and `total_cuts_all_ranks` are safe UPPER bounds: a rank's per-pool
        // counts sum to at most its own total, so the actual send/recv lengths
        // never exceed these.
        let total_cuts_all_ranks: usize = self.per_rank_cuts.iter().sum();
        let mut needed_send = 0usize;
        let mut needed_recv = 0usize;
        for &pool in pools {
            if is_replicated(pool) {
                continue;
            }
            let record_size = cut_wire_size(fcf.pools[pool].state_dimension);
            needed_send += expected_for_me * record_size;
            needed_recv += total_cuts_all_ranks * record_size;
        }
        if self.send_buf.len() < needed_send {
            self.send_buf.resize(needed_send, 0);
        }
        if self.recv_buf.len() < needed_recv {
            self.recv_buf.resize(needed_recv, 0);
        }

        // Pack every non-replicated pool's local records contiguously
        // (pool-major), recording each pool's (record_size, n_local) for the
        // count/deserialize passes. Taken out of `level_plan_scratch` (never a
        // fresh `Vec::new`) so `pack_pool_into` can borrow `self` mutably below;
        // restored just before returning `Ok`. `n_pools` is the packed
        // (non-replicated) count — the width every downstream pass indexes by.
        let mut plan = std::mem::take(&mut self.level_plan_scratch);
        plan.clear();
        let mut send_len = 0usize;
        let mut local_total = 0usize;
        for &pool in pools {
            if is_replicated(pool) {
                continue;
            }
            let pool_n_state = fcf.pools[pool].state_dimension;
            debug_assert!(
                pool_n_state <= self.max_n_state,
                "pool {pool} dimension {pool_n_state} exceeds buffer-capacity max {}",
                self.max_n_state
            );
            let record_size = cut_wire_size(pool_n_state);
            let n_local = self.pack_pool_into(fcf, pool, iteration, record_size, send_len);
            send_len += n_local * record_size;
            local_total += n_local;
            plan.push((pool, record_size, n_local));
        }
        let n_pools = plan.len();

        // Exchange every rank's per-pool local count (a small uniform integer
        // `allgatherv`, `n_pools` per rank) so each peer's per-pool byte segment
        // below is sized by ITS OWN count. Counts ride the wire as `u64`; the
        // per-rank byte `counts`/`displs` are reused as the uniform integer
        // layout here, then recomputed for the byte `allgatherv`.
        self.pool_count_send_scratch.clear();
        self.pool_count_send_scratch
            .extend(plan.iter().map(|&(_, _, n_local)| n_local as u64));
        let gathered = self.num_ranks * n_pools;
        if self.per_pool_rank_counts.len() < gathered {
            self.per_pool_rank_counts.resize(gathered, 0);
        }
        for r in 0..self.num_ranks {
            self.counts[r] = n_pools;
            self.displs[r] = r * n_pools;
        }
        comm.allgatherv(
            &self.pool_count_send_scratch,
            &mut self.per_pool_rank_counts[..gathered],
            &self.counts,
            &self.displs,
        )?;

        // A corrupt exchange must fail loudly, never deserialize at the wrong
        // stride: this rank's own gathered segment must echo what it packed.
        let mut my_gathered_total = 0usize;
        for pool_idx in 0..n_pools {
            my_gathered_total +=
                usize::try_from(self.per_pool_rank_counts[my_rank * n_pools + pool_idx]).map_err(
                    |_| malformed_count_exchange(&format!("rank {my_rank} count exceeds usize")),
                )?;
        }
        if my_gathered_total != local_total {
            return Err(malformed_count_exchange(&format!(
                "rank {my_rank} gathered total {my_gathered_total} disagrees with packed local total {local_total}"
            )));
        }

        // Size the byte `allgatherv` per `(rank, pool)` from the gathered counts,
        // rejecting a malformed count (overflow, or a total that overruns the
        // grown receive buffer). A single pool collapses this to
        // `per_rank_cuts[r] * record_size` — byte-identical to the chain path.
        for r in 0..self.num_ranks {
            let mut bytes = 0usize;
            for (pool_idx, &(_, record_size, _)) in plan.iter().enumerate() {
                let cuts_for_r = usize::try_from(self.per_pool_rank_counts[r * n_pools + pool_idx])
                    .map_err(|_| {
                        malformed_count_exchange(&format!("rank {r} count exceeds usize"))
                    })?;
                let seg = cuts_for_r.checked_mul(record_size).ok_or_else(|| {
                    malformed_count_exchange(&format!("rank {r} per-pool byte count overflows"))
                })?;
                bytes = bytes.checked_add(seg).ok_or_else(|| {
                    malformed_count_exchange(&format!("rank {r} byte total overflows"))
                })?;
            }
            self.counts[r] = bytes;
        }
        self.displs[0] = 0;
        for r in 1..self.num_ranks {
            self.displs[r] = self.displs[r - 1] + self.counts[r - 1];
        }

        let mut recv_len = 0usize;
        for r in 0..self.num_ranks {
            recv_len = recv_len.checked_add(self.counts[r]).ok_or_else(|| {
                malformed_count_exchange(&format!("receive total overflows at rank {r}"))
            })?;
        }
        if recv_len > self.recv_buf.len() {
            return Err(malformed_count_exchange(&format!(
                "gathered counts need {recv_len} bytes, exceeding receive buffer capacity {}",
                self.recv_buf.len()
            )));
        }
        debug_assert!(
            send_len <= self.send_buf.len(),
            "send_len {send_len} exceeds send_buf capacity {}",
            self.send_buf.len()
        );

        comm.allgatherv(
            &self.send_buf[..send_len],
            &mut self.recv_buf[..recv_len],
            &self.counts,
            &self.displs,
        )?;

        // Remote inserts are rank-major over ascending peer rank, pool-major
        // within each peer, wire order within a segment. Each pool is a distinct
        // append target, so for a given pool the inserts arrive in ascending peer
        // rank regardless of this loop nesting — the rank-count-invariant order
        // the append-only slot identity needs.
        let mut remote_total = 0usize;
        for r in 0..self.num_ranks {
            if r == my_rank {
                continue;
            }
            let mut cursor = self.displs[r];
            for (pool_idx, &(pool, record_size, _)) in plan.iter().enumerate() {
                let pool_n_state = fcf.pools[pool].state_dimension;
                let seg_cuts = usize::try_from(self.per_pool_rank_counts[r * n_pools + pool_idx])
                    .map_err(|_| {
                    malformed_count_exchange(&format!("rank {r} count exceeds usize"))
                })?;
                let seg_len = seg_cuts * record_size;
                let slice = &self.recv_buf[cursor..cursor + seg_len];
                deserialize_cuts_from_buffer_into(
                    slice,
                    pool_n_state,
                    &mut self.deserialize_headers_buf,
                    &mut self.deserialize_coefficients_buf,
                )?;
                for (i, header) in self.deserialize_headers_buf.iter().enumerate() {
                    let coeff_start = i * pool_n_state;
                    fcf.add_cut(
                        NodeId(header.node_id),
                        pool,
                        u64::from(header.iteration),
                        header.forward_pass_index,
                        header.intercept,
                        &self.deserialize_coefficients_buf[coeff_start..coeff_start + pool_n_state],
                    );
                    remote_total += 1;
                }
                cursor += seg_len;
            }
        }

        self.level_plan_scratch = plan;
        Ok((local_total, remote_total))
    }

    /// Return the send buffer capacity in bytes.
    #[must_use]
    pub fn send_capacity(&self) -> usize {
        self.send_buf.len()
    }

    /// Return the receive buffer capacity in bytes.
    #[must_use]
    pub fn recv_capacity(&self) -> usize {
        self.recv_buf.len()
    }
}

/// Build the `SddpError::Validation` for a malformed per-`(rank, pool)` count
/// exchange in [`CutSyncBuffers::sync_level_records`]. Only reached on the error
/// return, so its `format!` never allocates on the hot path.
fn malformed_count_exchange(detail: &str) -> SddpError {
    SddpError::Validation(format!(
        "sync_level_records: malformed per-(rank, pool) count exchange: {detail}"
    ))
}

/// Reusable buffers for the replicated-aggregation outcome `allgatherv` at a
/// singleton-trial-state group, where ranks split the successor outcome set and
/// every rank must reassemble the whole set to run the identical flat
/// aggregation. Grown lazily; never on the trial-point-distributed path.
#[derive(Debug, Default, Clone)]
pub struct OutcomeExchangeScratch {
    send: Vec<f64>,
    recv: Vec<f64>,
    counts: Vec<usize>,
    displs: Vec<usize>,
}

impl OutcomeExchangeScratch {
    /// `allgatherv` every rank's canonical outcome slice into the full outcome
    /// set, in rank order — which equals canonical `(m, ψ)` order because
    /// `partition` assigns each rank a contiguous canonical range. Each outcome
    /// is `1 + n_state` doubles (objective, then subgradient). Returns the
    /// reassembled full set; the caller runs the identical `aggregate_cut_into`
    /// over it on every rank.
    ///
    /// Mirrors the cut `allgatherv` buffer discipline: per-rank counts/displs
    /// and a per-rank-count mismatch guard.
    ///
    /// # Errors
    ///
    /// Returns `Err(SddpError::Validation(_))` when `local`'s length is not
    /// `outcome_counts[my_rank] * (1 + n_state)` (a divergence that would
    /// corrupt the reassembly); returns `Err(SddpError::Communication(_))` if
    /// the `allgatherv` fails.
    pub fn allgather_outcomes<C: Communicator>(
        &mut self,
        local: &[f64],
        n_state: usize,
        outcome_counts: &[usize],
        comm: &C,
    ) -> Result<&[f64], SddpError> {
        let my_rank = comm.rank();
        let stride = 1 + n_state;
        let expected = outcome_counts[my_rank] * stride;
        if local.len() != expected {
            return Err(SddpError::Validation(format!(
                "allgather_outcomes invariant violated: rank {my_rank} produced \
                 {} doubles, expected {expected} ({} outcomes * {stride}) per the \
                 outcome partition. Releasing this divergence to allgatherv would \
                 corrupt remote ranks' reassembled outcome set.",
                local.len(),
                outcome_counts[my_rank],
            )));
        }

        self.counts.clear();
        self.counts
            .extend(outcome_counts.iter().map(|&c| c * stride));
        self.displs.clear();
        let mut acc = 0usize;
        for &c in &self.counts {
            self.displs.push(acc);
            acc += c;
        }
        let total = acc;

        self.send.clear();
        self.send.extend_from_slice(local);
        if self.recv.len() < total {
            self.recv.resize(total, 0.0);
        }

        comm.allgatherv(
            &self.send,
            &mut self.recv[..total],
            &self.counts,
            &self.displs,
        )?;
        Ok(&self.recv[..total])
    }
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::float_cmp
    )]

    use cobre_comm::{CommData, CommError, Communicator, LocalBackend, ReduceOp};

    use super::CutSyncBuffers;
    use crate::{
        SddpError,
        cut::{
            fcf::FutureCostFunction,
            wire::{CutWireTuple, cut_wire_size, deserialize_cuts_from_buffer, serialize_cut},
        },
        setup::NodeId,
    };

    // ── Unit tests ────────────────────────────────────────────────────────────

    #[test]
    fn new_deserialize_scratch_bufs_start_empty() {
        // deserialize_headers_buf and deserialize_coefficients_buf both
        // have capacity == 0 immediately after construction (grown lazily).
        let bufs = CutSyncBuffers::new(2, 3, 4);
        assert_eq!(
            bufs.deserialize_headers_buf.capacity(),
            0,
            "deserialize_headers_buf must start with capacity 0"
        );
        assert_eq!(
            bufs.deserialize_coefficients_buf.capacity(),
            0,
            "deserialize_coefficients_buf must start with capacity 0"
        );
    }

    #[test]
    fn new_send_buf_capacity_is_max_cuts_times_record_size() {
        let bufs = CutSyncBuffers::new(2, 3, 1);
        let expected = 3 * cut_wire_size(2);
        assert_eq!(bufs.send_capacity(), expected);
    }

    #[test]
    fn new_recv_buf_capacity_is_max_cuts_times_num_ranks_times_record_size() {
        // 10 * 4 * cut_wire_size(3) = 40 * 53 = 2120
        let bufs = CutSyncBuffers::new(3, 10, 4);
        let expected = 10 * 4 * cut_wire_size(3);
        assert_eq!(bufs.recv_capacity(), expected);
        assert_eq!(expected, 2120);
    }

    #[test]
    fn new_counts_length_equals_num_ranks() {
        let bufs = CutSyncBuffers::new(3, 10, 4);
        assert_eq!(bufs.counts.len(), 4);
    }

    #[test]
    fn new_displs_length_equals_num_ranks() {
        let bufs = CutSyncBuffers::new(3, 10, 4);
        assert_eq!(bufs.displs.len(), 4);
    }

    #[test]
    fn new_counts_and_displs_initialized_to_max_uniform_values() {
        // Construction sets max uniform capacity; sync_level_records recomputes per call.
        let bufs = CutSyncBuffers::new(2, 3, 2);
        let per_rank = 3 * cut_wire_size(2); // 123
        assert_eq!(bufs.counts[0], per_rank);
        assert_eq!(bufs.counts[1], per_rank);
        assert_eq!(bufs.displs[0], 0);
        assert_eq!(bufs.displs[1], per_rank);
    }

    #[test]
    fn new_n_state_zero_record_size_is_29() {
        // Edge case: n_state = 0, record_size = 29.
        let bufs = CutSyncBuffers::new(0, 5, 1);
        assert_eq!(bufs.send_capacity(), 5 * 29);
        assert_eq!(bufs.recv_capacity(), 5 * 29);
    }

    #[test]
    fn send_buf_serialization_round_trip_two_cuts() {
        let mut bufs = CutSyncBuffers::new(2, 2, 1);
        let local_cuts: &[CutWireTuple<'_>] = &[
            (0, 0, 1, 0, 10.0, &[1.0, 2.0]),
            (1, 0, 1, 1, 20.0, &[3.0, 4.0]),
        ];

        let record_size = cut_wire_size(2);
        let send_len = local_cuts.len() * record_size;
        assert_eq!(send_len, 90);

        // Serialize manually into send_buf, mirroring the pack path's serialize_cut logic.
        for (i, &(slot_index, node_id, iteration, forward_pass_index, intercept, coefficients)) in
            local_cuts.iter().enumerate()
        {
            let start = i * record_size;
            serialize_cut(
                &mut bufs.send_buf[start..start + record_size],
                slot_index,
                node_id,
                iteration,
                forward_pass_index,
                intercept,
                coefficients,
            );
        }

        let recovered = deserialize_cuts_from_buffer(&bufs.send_buf[..send_len], 2).unwrap();
        assert_eq!(recovered.len(), 2);

        let (h0, c0) = &recovered[0];
        assert_eq!(h0.slot_index, 0);
        assert_eq!(h0.iteration, 1);
        assert_eq!(h0.forward_pass_index, 0);
        assert_eq!(h0.intercept, 10.0);
        assert_eq!(c0, &[1.0, 2.0]);

        let (h1, c1) = &recovered[1];
        assert_eq!(h1.slot_index, 1);
        assert_eq!(h1.iteration, 1);
        assert_eq!(h1.forward_pass_index, 1);
        assert_eq!(h1.intercept, 20.0);
        assert_eq!(c1, &[3.0, 4.0]);
    }

    #[test]
    fn counts_and_displs_computation_for_various_cut_counts() {
        // 2 local cuts, n_state=2: per_rank_bytes = 2 * 45 = 90; 3 ranks →
        // counts = [90, 90, 90], displs = [0, 90, 180].
        let mut bufs = CutSyncBuffers::new(2, 5, 3);

        let n_local = 2usize;
        let record_size = cut_wire_size(2); // 45
        let per_rank = n_local * record_size; // 90

        // Simulate the per-rank counts/displs computation the exchange performs.
        for r in 0..3 {
            bufs.counts[r] = per_rank;
            bufs.displs[r] = r * per_rank;
        }

        assert_eq!(bufs.counts, vec![90, 90, 90]);
        assert_eq!(bufs.displs, vec![0, 90, 180]);
    }

    // ── replicated aggregation (outcome allgather + skip) ─────────────────

    /// A singleton-trial-state group's successor outcome set, split across two
    /// rank slices, allgathered in canonical `(m, ψ)` order and aggregated on
    /// each slice, yields a cut bitwise identical (`to_bits`) to the single-rank
    /// aggregation over the whole set — and the replicated group's cut exchange
    /// is skipped (an empty pool list exchanges nothing; a naive exchange would
    /// multiply the cut by the rank count). `CVaR` makes the aggregation
    /// index-order-sensitive, so a reassembly that lost canonical order would
    /// fail the bitwise check.
    #[test]
    fn d2_replicated_split_equals_single_rank_and_skips_exchange() {
        use super::OutcomeExchangeScratch;
        use crate::risk_measure::{BackwardOutcome, RiskMeasure, RiskMeasureScratch};

        /// Rank 0 of a 2-rank world; `allgatherv` writes only rank 0's own slice
        /// (`displs[0] = 0`), leaving rank 1's pre-populated segment intact.
        struct Rank0Of2Outcome;
        impl Communicator for Rank0Of2Outcome {
            fn allgatherv<T: CommData>(
                &self,
                send: &[T],
                recv: &mut [T],
                counts: &[usize],
                _displs: &[usize],
            ) -> Result<(), CommError> {
                recv[..counts[0]].clone_from_slice(&send[..counts[0]]);
                Ok(())
            }
            fn allreduce<T: CommData>(
                &self,
                _s: &[T],
                _r: &mut [T],
                _o: ReduceOp,
            ) -> Result<(), CommError> {
                unreachable!()
            }
            fn broadcast<T: CommData>(&self, _b: &mut [T], _root: usize) -> Result<(), CommError> {
                unreachable!()
            }
            fn barrier(&self) -> Result<(), CommError> {
                unreachable!()
            }
            fn rank(&self) -> usize {
                0
            }
            fn size(&self) -> usize {
                2
            }
            fn abort(&self, code: i32) -> ! {
                std::process::exit(code)
            }
        }

        let n_state = 2usize;
        let stride = 1 + n_state;
        let x_hat = [1.0_f64, 2.0_f64];
        // Four distinct outcomes (objective, subgradient); intercept =
        // objective − subgradient·x_hat, matching the replicated reconstruct.
        let raw: [(f64, [f64; 2]); 4] = [
            (30.0, [1.0, 0.5]),
            (10.0, [0.2, -0.3]),
            (50.0, [-1.0, 2.0]),
            (20.0, [0.7, 0.1]),
        ];
        let probabilities = [0.25_f64, 0.25, 0.25, 0.25];
        let risk = RiskMeasure::CVaR {
            alpha: 0.5,
            lambda: 1.0,
        };

        let build = |sub: &[(f64, [f64; 2])]| -> Vec<BackwardOutcome> {
            sub.iter()
                .map(|&(obj, g)| {
                    let intercept = obj - (g[0] * x_hat[0] + g[1] * x_hat[1]);
                    BackwardOutcome {
                        intercept,
                        coefficients: g.to_vec(),
                        objective_value: obj,
                    }
                })
                .collect()
        };

        // Single-rank aggregation over the whole canonical set.
        let whole = build(&raw);
        let mut int_a = 0.0_f64;
        let mut coeffs_a = vec![0.0_f64; n_state];
        let mut scratch_a = RiskMeasureScratch::default();
        risk.aggregate_cut_into(
            &whole,
            &probabilities,
            &mut int_a,
            &mut coeffs_a,
            &mut scratch_a,
        );

        // Split: rank 0 owns outcomes [0, 2), rank 1 owns [2, 4). Each rank's
        // local flat buffer is `objective ‖ subgradient` per outcome.
        let flat = |sub: &[(f64, [f64; 2])]| -> Vec<f64> {
            let mut v = Vec::new();
            for &(obj, g) in sub {
                v.push(obj);
                v.extend_from_slice(&g);
            }
            v
        };
        let rank0_local = flat(&raw[..2]);
        let rank1_local = flat(&raw[2..]);
        let total = raw.len() * stride;

        let mut ex = OutcomeExchangeScratch::default();
        // Pre-populate rank 1's segment; `allgather_outcomes` preserves it (its
        // resize only grows) and the stub overwrites only rank 0's prefix.
        ex.recv.resize(total, 0.0);
        ex.recv[2 * stride..total].copy_from_slice(&rank1_local);

        let full = ex
            .allgather_outcomes(&rank0_local, n_state, &[2, 2], &Rank0Of2Outcome)
            .unwrap();

        // Canonical order preserved: reassembled == rank0 ‖ rank1 == whole.
        let expected_full = flat(&raw);
        assert_eq!(
            full,
            &expected_full[..],
            "allgather must reassemble in canonical (m, ψ) order"
        );

        // Reconstruct and aggregate on the reassembled full set.
        let mut reassembled = Vec::new();
        for o in 0..raw.len() {
            let base = o * stride;
            let objective = full[base];
            let coefficients = full[base + 1..base + 1 + n_state].to_vec();
            let intercept = objective - (coefficients[0] * x_hat[0] + coefficients[1] * x_hat[1]);
            reassembled.push(BackwardOutcome {
                intercept,
                coefficients,
                objective_value: objective,
            });
        }
        let mut int_b = 0.0_f64;
        let mut coeffs_b = vec![0.0_f64; n_state];
        let mut scratch_b = RiskMeasureScratch::default();
        risk.aggregate_cut_into(
            &reassembled,
            &probabilities,
            &mut int_b,
            &mut coeffs_b,
            &mut scratch_b,
        );

        assert_eq!(
            int_a.to_bits(),
            int_b.to_bits(),
            "split replicated aggregation intercept must be bitwise identical to single-rank"
        );
        for (a, b) in coeffs_a.iter().zip(&coeffs_b) {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "split replicated aggregation coefficient must be bitwise identical to single-rank"
            );
        }

        // Cut-exchange skip: a replicated group contributes no pool, so the
        // level's batched exchange runs over an empty pool list — nothing sent,
        // nothing inserted.
        let mut fcf = FutureCostFunction::new(1, n_state, 1, 10, &[0; 1]);
        let mut bufs = CutSyncBuffers::new(n_state, 1, 1);
        let (local, remote) = bufs
            .sync_level_records(&[], &[], &mut fcf, 1, &LocalBackend)
            .unwrap();
        assert_eq!((local, remote), (0, 0));
        assert_eq!(fcf.total_active_cuts(), 0);
    }

    /// A level's two differing-dimension pools exchanged in ONE `sync_level_records`
    /// call (not one collective per node): both pools' local cuts pack and pass
    /// the per-pool guard; single-rank inserts no remote and never double-inserts
    /// the local cuts.
    #[test]
    fn sync_level_records_batches_two_pools_in_one_exchange() {
        let dims = [2usize, 3usize];
        let mut fcf = FutureCostFunction::new_per_pool(&dims, 3, 1, 10, &[0; 2], &[6; 2]);
        let mut bufs = CutSyncBuffers::new(3, 1, 1);
        fcf.add_cut(NodeId(0), 0, 1, 0, 10.0, &[1.0, 2.0]);
        fcf.add_cut(NodeId(0), 1, 1, 0, 20.0, &[3.0, 4.0, 5.0]);

        let (local, remote) = bufs
            .sync_level_records(&[0, 1], &[], &mut fcf, 1, &LocalBackend)
            .unwrap();
        assert_eq!(
            local, 2,
            "both pools' single local cut packed in one exchange"
        );
        assert_eq!(remote, 0, "single-rank: no remote cuts");
        assert_eq!(
            fcf.total_active_cuts(),
            2,
            "local cuts must not be re-inserted"
        );
    }

    /// Rank 0 of 2; `allgatherv` writes only rank 0's own segment
    /// (`recv[displs[0]..displs[0] + counts[0]]`), leaving each peer's
    /// pre-populated segment intact — the faithful single-invocation view of the
    /// per-`(rank, pool)` count exchange and the byte exchange from rank 0.
    struct Rank0Of2Preserve;

    impl Communicator for Rank0Of2Preserve {
        fn allgatherv<T: CommData>(
            &self,
            send: &[T],
            recv: &mut [T],
            counts: &[usize],
            displs: &[usize],
        ) -> Result<(), CommError> {
            let start = displs[0];
            recv[start..start + counts[0]].clone_from_slice(&send[..counts[0]]);
            Ok(())
        }

        fn allreduce<T: CommData>(
            &self,
            _send: &[T],
            _recv: &mut [T],
            _op: ReduceOp,
        ) -> Result<(), CommError> {
            unreachable!()
        }

        fn broadcast<T: CommData>(&self, _buf: &mut [T], _root: usize) -> Result<(), CommError> {
            unreachable!()
        }

        fn barrier(&self) -> Result<(), CommError> {
            unreachable!()
        }

        fn rank(&self) -> usize {
            0
        }

        fn size(&self) -> usize {
            2
        }

        fn abort(&self, error_code: i32) -> ! {
            std::process::exit(error_code)
        }
    }

    /// A genuinely multi-node level whose two pools carry differing
    /// per-`(rank, pool)` counts (pool 0 visited only by rank 0, pool 1 only by
    /// rank 1) is exchanged at each peer's OWN per-pool stride: rank 1's pool-1
    /// cuts land in pool 1, and rank 0's pool-0 local cuts are never re-inserted.
    #[test]
    fn sync_level_records_per_rank_pool_divergence_inserts_remote() {
        let n_state = 2usize;
        let record_size = cut_wire_size(n_state); // 45
        let dims = [n_state, n_state];
        let mut fcf = FutureCostFunction::new_per_pool(&dims, n_state, 2, 10, &[0; 2], &[4; 2]);
        let mut bufs = CutSyncBuffers::with_distribution(n_state, 2, 2, 4);

        // Rank 0's own pool-0 cuts (the backward pass inserted these before sync);
        // pool 1 has no local cut on rank 0.
        fcf.add_cut(NodeId(0), 0, 1, 0, 10.0, &[1.0, 2.0]);
        fcf.add_cut(NodeId(0), 0, 1, 1, 20.0, &[3.0, 4.0]);

        // Pre-seed the gathered counts rank-major/pool-minor: rank 0 = [2, 0]
        // (overwritten by the stub's echo), rank 1 = [0, 2] (preserved). The lazy
        // resize to num_ranks * n_pools = 4 is a no-op at this length.
        bufs.per_pool_rank_counts = vec![0, 0, 0, 2];

        // Rank 1's pool-1 cuts at its byte segment (displs[1] = counts[0] = 90;
        // pool 0 contributes 0 bytes, so pool 1's two records fill [90, 180)).
        let r1 = 2 * record_size; // 90
        serialize_cut(
            &mut bufs.recv_buf[r1..r1 + record_size],
            50,
            0,
            1,
            0,
            100.0,
            &[5.0, 6.0],
        );
        serialize_cut(
            &mut bufs.recv_buf[r1 + record_size..r1 + 2 * record_size],
            51,
            0,
            1,
            1,
            200.0,
            &[7.0, 8.0],
        );

        let (local, remote) = bufs
            .sync_level_records(&[0, 1], &[], &mut fcf, 1, &Rank0Of2Preserve)
            .unwrap();

        assert_eq!(local, 2, "rank 0 packs its two pool-0 cuts");
        assert_eq!(remote, 2, "rank 1's two pool-1 cuts insert remotely");
        assert_eq!(
            fcf.pools[0].active_count(),
            2,
            "pool 0: rank 0's own cuts, not re-inserted"
        );
        assert_eq!(
            fcf.pools[1].active_count(),
            2,
            "pool 1: rank 1's remote cuts inserted at pool 1's own stride"
        );
        // The larger pool-1 cut evaluates to 200 + 7 + 8 at the unit state.
        assert_eq!(fcf.evaluate_at_state(1, &[1.0, 1.0]), 215.0);
    }

    /// A fault-injected per-`(rank, pool)` count that overruns the grown receive
    /// buffer fails loudly (`SddpError::Validation`) rather than deserializing at
    /// the wrong stride. Rank 0 of 2; rank 1's count segment is pre-seeded with an
    /// oversized value the stub's rank-0-only echo preserves.
    #[test]
    fn sync_level_records_rejects_malformed_count_exchange_overrun() {
        let n_state = 2usize;
        let mut fcf = FutureCostFunction::new(1, n_state, 2, 10, &[0; 1]);
        let mut bufs = CutSyncBuffers::with_distribution(n_state, 1, 2, 2);
        // Rank 0's own single pool-0 cut.
        fcf.add_cut(NodeId(0), 0, 1, 0, 10.0, &[1.0, 2.0]);

        // num_ranks * n_pools = 2: rank 0 = [<echoed>], rank 1 = [u32::MAX] — a
        // count whose byte total overruns the grown receive buffer.
        bufs.per_pool_rank_counts = vec![0, u64::from(u32::MAX)];

        let result = bufs.sync_level_records(&[0], &[], &mut fcf, 1, &Rank0Of2Preserve);
        match result {
            Err(SddpError::Validation(ref msg)) => {
                assert!(
                    msg.contains("malformed per-(rank, pool) count exchange"),
                    "message missing the malformed-exchange marker: {msg}"
                );
            }
            other => panic!("expected SddpError::Validation for the overrun, got: {other:?}"),
        }
    }

    /// A replicated-aggregation pool is EXCLUDED from the batched exchange: it is
    /// not packed, so `local_total` counts only the non-replicated pool. The
    /// `replicated_pools = &[]` contrast packs both pools, so the skip is
    /// selective (by aggregation kind), never a blanket effect.
    #[test]
    fn sync_level_records_excludes_replicated_pool_from_packing() {
        let n_state = 2usize;
        let dims = [n_state, n_state];
        let mut fcf = FutureCostFunction::new_per_pool(&dims, n_state, 1, 10, &[0; 2], &[1; 2]);
        let mut bufs = CutSyncBuffers::with_distribution(n_state, 2, 1, 1);
        let it = 1u64;

        // The backward compute inserted each pool's own local cut before the sync.
        fcf.add_cut(NodeId(0), 0, it, 0, 10.0, &[1.0, 2.0]);
        fcf.add_cut(NodeId(1), 1, it, 0, 20.0, &[3.0, 4.0]);

        // Pool 1 replicated -> excluded: only the non-replicated pool 0 is packed.
        let (local, remote) = bufs
            .sync_level_records(&[0, 1], &[1], &mut fcf, it, &LocalBackend)
            .unwrap();
        assert_eq!(local, 1, "only the non-replicated pool 0 is packed");
        assert_eq!(remote, 0, "single rank inserts no remote cuts");

        // No replicated pools -> both packed.
        let (local_all, _) = bufs
            .sync_level_records(&[0, 1], &[], &mut fcf, it, &LocalBackend)
            .unwrap();
        assert_eq!(local_all, 2, "both pools packed when none is replicated");
    }

    /// World >= 2: a replicated-aggregation pool is skipped even while a
    /// non-replicated sibling pool IS exchanged. Rank 1's pool-0 cut inserts
    /// remotely (the legitimate exchange), while the replicated pool 1 — already
    /// held identically on every rank — is never re-inserted, so it keeps its one
    /// local cut: `cuts_added` stays the world=1 count, no rank-multiplication.
    #[test]
    fn sync_level_records_skips_replicated_pool_under_two_ranks() {
        let n_state = 2usize;
        let record_size = cut_wire_size(n_state); // 45
        let dims = [n_state, n_state];
        let mut fcf = FutureCostFunction::new_per_pool(&dims, n_state, 2, 10, &[0; 2], &[4; 2]);
        let mut bufs = CutSyncBuffers::with_distribution(n_state, 2, 2, 4);

        // Rank 0's own local cuts (inserted by the backward compute before sync):
        // one trial-point-distributed cut in pool 0, one replicated cut in pool 1.
        fcf.add_cut(NodeId(0), 0, 1, 0, 10.0, &[1.0, 2.0]);
        fcf.add_cut(NodeId(1), 1, 1, 0, 20.0, &[3.0, 4.0]);

        // Pool 1 is skipped, so the filtered plan holds only pool 0; the gathered
        // per-(rank, pool) counts are num_ranks * 1 = 2: rank 0 = [<echoed 1>],
        // rank 1 = [1] (preserved by the rank-0-only stub).
        bufs.per_pool_rank_counts = vec![0, 1];

        // Rank 1's distinct pool-0 cut at its byte segment (displs[1] = counts[0] =
        // 45, pool 0 the only packed pool). fp=1 gives it a distinct slot.
        serialize_cut(
            &mut bufs.recv_buf[record_size..2 * record_size],
            50,
            0,
            1,
            1,
            100.0,
            &[5.0, 6.0],
        );

        let (local, remote) = bufs
            .sync_level_records(&[0, 1], &[1], &mut fcf, 1, &Rank0Of2Preserve)
            .unwrap();

        assert_eq!(local, 1, "only the non-replicated pool 0 is packed");
        assert_eq!(remote, 1, "rank 1's pool-0 cut inserts remotely");
        assert_eq!(
            fcf.pools[0].active_count(),
            2,
            "pool 0: rank 0's own cut plus rank 1's remote cut"
        );
        assert_eq!(
            fcf.pools[1].active_count(),
            1,
            "pool 1 (replicated) keeps its single local cut — no rank-multiplication"
        );
    }
}
