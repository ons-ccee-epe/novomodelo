//! Binding-cut counting for the backward pass: bump slot increments from (slot, y) pairs and
//! fold them into the stage's metadata contribution.

/// Bumps `slot_increments[metadata_offset + slot]` for each (slot, y) pair whose
/// raw cut-row dual exceeds `threshold`.
///
/// The bump lands in the child's OWN pool region (`metadata_offset + slot`), so a
/// fan's sibling pools never collide on a shared slot index. The criterion is the
/// raw dual, strictly, not its magnitude. The bump is a deterministic function of
/// the pairs only — no worker id, rank, or trace — so the order-insensitive
/// metadata allreduce stays rank-invariant.
///
/// `slot_increments` accumulates (summed) across the trial point's openings; the
/// per-trial-point reset happens in the stage loop before the openings run.
pub(crate) fn count_binding_slots(
    cut_dual_slots: &[(usize, f64)],
    threshold: f64,
    slot_increments: &mut [u64],
    metadata_offset: usize,
) {
    for &(slot, y) in cut_dual_slots {
        if y > threshold {
            slot_increments[metadata_offset + slot] += 1;
        }
    }
}

/// Fold this trial point's binding-cut `slot_increments` into the accumulating
/// `metadata_sync_contribution` over the successor pools' populated region only.
///
/// `populated_len` is `SuccessorOutcomes::total_metadata_len`, never the buffers'
/// capacity: only `[0, populated_len)` is zeroed each stage and only each pool's own
/// populated slice is reduced by `sync_stage_metadata`, so a `slot_increments.len()`
/// bound would fold never-zeroed, never-read tail slots into the contribution.
pub(crate) fn fold_slot_increments_into_sync(
    slot_increments: &[u64],
    metadata_sync_contribution: &mut [u64],
    populated_len: usize,
) {
    for slot in 0..populated_len {
        let increment = slot_increments[slot];
        if increment > 0 {
            metadata_sync_contribution[slot] += increment;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::count_binding_slots;

    #[test]
    fn count_binding_slots_bumps_only_strictly_binding_pairs_in_the_child_region() {
        let mut slot_increments = vec![0_u64; 10];

        count_binding_slots(
            &[(0, 0.0), (2, 1e-9), (3, 5.0)],
            1e-9,
            &mut slot_increments,
            4,
        );

        assert_eq!(slot_increments, [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]);
    }
}
