//! Captured-basis metadata population after a forward stage solve.

use novomodelo_solver::{BasisStatus, SolverInterface, StageTemplate};

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
    node_id: NodeId,
) {
    captured.cut_row_slots.clear();
    for (slot, _intercept, _coeffs) in pool.active_cuts().take(cut_row_count) {
        captured.cut_row_slots.push(slot as u32);
    }
    captured.base_row_count = base_row_count;
    captured.node_id = node_id;
    captured
        .basis
        .row_status
        .resize(base_row_count + cut_row_count, BasisStatus::Basic);
}

/// Captures the solver's current basis into `slot`, reusing its buffer when
/// present. Called only for a frozen solve: a lazy solve's basis describes the
/// resident subset, not the frozen layout (`StageOpeningSolver::captures_basis`).
///
/// `basis_row_capacity` is the frozen template's row count.
pub(crate) fn capture_forward_basis<S: SolverInterface>(
    solver: &mut S,
    slot: &mut Option<CapturedBasis>,
    template: &StageTemplate,
    basis_row_capacity: usize,
    pool: &CutPool,
    node_id: NodeId,
) {
    let cut_row_count = basis_row_capacity.saturating_sub(template.num_rows);
    let captured = slot.get_or_insert_with(|| {
        CapturedBasis::new(
            template.num_cols,
            basis_row_capacity,
            template.num_rows,
            cut_row_count,
            node_id,
        )
    });
    solver.get_basis(&mut captured.basis);
    write_capture_metadata(captured, pool, template.num_rows, cut_row_count, node_id);
}
