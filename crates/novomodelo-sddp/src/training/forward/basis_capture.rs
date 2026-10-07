//! Captured-basis metadata population after a forward stage solve.

use cobre_solver::BasisStatus;

use crate::cut::pool::CutPool;
use crate::setup::NodeId;
use crate::workspace::CapturedBasis;

/// Populate `CapturedBasis` metadata after a stage solve.
///
/// Records each active cut's slot as `u32` in `active_cuts()` order so the next
/// iteration's warm-start matches stored cut rows to LP rows by slot identity,
/// not row count (the append-only-pool warm-start contract; `reconstruct_basis`).
///
/// `row_status` is resized to `base_row_count + cut_row_count` so the invariant
/// holds even when `get_basis` is a no-op (test mocks); real solvers write the
/// correct length, making this a no-op.
///
/// `node_id` is the declared id of the node being solved
/// (`NodeGraph::node_ids[node]`); this is the single owner that sets
/// `CapturedBasis::node_id` — both the forward and backward capture routes
/// reach it through this function, never a second call site.
#[allow(clippy::cast_possible_truncation)]
pub(crate) fn write_capture_metadata(
    captured: &mut CapturedBasis,
    pool: &CutPool,
    base_row_count: usize,
    cut_row_count: usize,
    current_state: &[f64],
    node_id: NodeId,
) {
    captured.cut_row_slots.clear();
    for (slot, _intercept, _coeffs) in pool.active_cuts().take(cut_row_count) {
        captured.cut_row_slots.push(slot as u32);
    }
    captured.state_at_capture.clear();
    captured.state_at_capture.extend_from_slice(current_state);
    captured.base_row_count = base_row_count;
    captured.node_id = node_id;
    let expected_len = base_row_count + cut_row_count;
    if captured.basis.row_status.len() != expected_len {
        captured
            .basis
            .row_status
            .resize(expected_len, BasisStatus::Basic);
    }
}
