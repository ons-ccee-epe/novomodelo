//! Per-(scenario, stage) LP solve kernel for the forward pass.
//!
//! All scratch is owned by the workspace and reused across scenarios; no
//! allocation occurs on this hot path.

use novomodelo_solver::SolverInterface;

use crate::{
    context::{StageContext, TrainingContext},
    error::SddpError,
    solve::StageOpeningSolver,
    trajectory::TrajectoryRecord,
    workspace::{BasisStoreSliceMut, SolverWorkspace},
};

use super::{
    StageKey, capture_forward_basis,
    node_solve::{ForwardBasisApply, ForwardNodeSolve, solve_forward_node_core},
};

/// Execute the stage-level LP solve for one (scenario, stage) pair.
///
/// Returns the stage cost on success.
///
/// # Errors
///
/// Returns `Err(SddpError::Infeasible)` when the stage LP is infeasible, or
/// `Err(SddpError::Solver)` for any other terminal solver failure.
pub(crate) fn run_forward_stage<S: SolverInterface + Send>(
    ws: &mut SolverWorkspace<S>,
    basis_slice: &mut BasisStoreSliceMut<'_>,
    ctx: &StageContext<'_>,
    training_ctx: &TrainingContext<'_>,
    key: &StageKey<'_>,
    worker_records: &mut [TrajectoryRecord],
) -> Result<f64, SddpError> {
    let StageKey {
        t,
        m,
        local_m,
        iteration,
        raw_noise,
        basis_row_capacity,
        pool,
        dcs,
        node,
    } = *key;
    let node_id = training_ctx.node_graph.node_ids[node];
    let template = ctx.template(t);

    // DCS path: load the cut-free base template here (the caller skips its frozen
    // load). Loading the frozen template instead would make the lazy loop's fresh
    // CutRowMap double-append the embedded cut rows.
    if dcs.is_some() {
        ws.solver.load_model(template);
    }

    let opening_solver = StageOpeningSolver::from_dcs_params(dcs);
    let stage_cost = solve_forward_node_core(
        ws,
        ctx,
        training_ctx,
        &ForwardNodeSolve {
            t,
            node,
            scenario_index: m,
            iteration,
            raw_noise,
            pool,
            // Warm-start basis keyed by NODE, matching the backward pass and the
            // store's node axis; keying by stage would alias sibling nodes that
            // share a stage, forcing every branching warm-start cold.
            stored_basis: if opening_solver.captures_basis() {
                basis_slice.get_mut(m, node).as_ref()
            } else {
                None
            },
            opening_solver: &opening_solver,
            basis_apply: ForwardBasisApply::SlotReconstruct,
            fusion_cut_state: None,
        },
        None,
    )
    .inspect_err(|e| {
        // Invalidate the stored basis on Infeasible so the next warm-start
        // attempt cold-solves.
        if opening_solver.captures_basis() && matches!(e, SddpError::Infeasible { .. }) {
            *basis_slice.get_mut(m, node) = None;
        }
    })?;

    let rec = &mut worker_records[local_m * training_ctx.horizon.num_stages() + t.0];
    rec.stage_cost = stage_cost;
    rec.node_id = node_id;
    rec.state.clear();
    rec.state.extend_from_slice(&ws.current_state);
    if t.0 > 0 && matches!(opening_solver, StageOpeningSolver::Frozen) {
        rec.frozen_brackets
            .write(iteration, node_id, &ws.scratch.cut_bracket_buf);
    }

    if opening_solver.captures_basis() {
        capture_forward_basis(
            &mut ws.solver,
            basis_slice.get_mut(m, node),
            template,
            basis_row_capacity,
            pool,
            &ws.current_state[..training_ctx.state.n_state],
            node_id,
        );
    }
    Ok(stage_cost)
}
