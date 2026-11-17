//! The frozen-or-lazy stage solve: one dispatch for every site that solves a
//! stage LP against its cuts.

use novomodelo_solver::{SolutionView, SolverInterface};

use crate::{
    cut::{CutPool, CutRowMap},
    dcs::{
        ActiveCutRhs, DcsCutView, DcsParams, DcsSolveContext, build_initial_resident_set,
        lazy_solve_preloaded,
    },
    error::SddpError,
    lp::indexer::{CutStateProjection, StateSpace},
    workspace::SolverWorkspace,
};

use super::stage_solve::{StageInputs, run_stage_solve_over};

/// How a stage LP is solved against its cuts, dispatched by `match` — never a
/// trait object.
///
/// - [`StageOpeningSolver::Frozen`]: the frozen all-cuts LP, warm-started from a
///   stored basis that the caller captures.
/// - [`StageOpeningSolver::Lazy`]: the resident-set LP of the `dynamic` method, a
///   cut-free core plus the cuts the solve finds violated.
pub(crate) enum StageOpeningSolver {
    Frozen,
    Lazy(DcsParams),
}

/// The inputs only the lazy mode reads; the frozen mode ignores them.
#[derive(Clone, Copy)]
pub(crate) struct LazyInputs<'a> {
    pub(crate) state: &'a StateSpace,
    pub(crate) cut_layout: &'a CutStateProjection,
    pub(crate) continue_carry: bool,
}

impl StageOpeningSolver {
    /// `Some(params)` → [`StageOpeningSolver::Lazy`], `None` → [`StageOpeningSolver::Frozen`].
    pub(crate) fn from_dcs_params(dcs_params: Option<DcsParams>) -> Self {
        dcs_params.map_or(StageOpeningSolver::Frozen, StageOpeningSolver::Lazy)
    }

    /// Fills the lazy mode's initial resident set from `pool`; the frozen mode
    /// has none.
    pub(crate) fn seed_resident_set<S: SolverInterface>(
        &self,
        ws: &mut SolverWorkspace<S>,
        pool: &CutPool,
        iteration: u64,
    ) {
        if let StageOpeningSolver::Lazy(params) = self {
            build_initial_resident_set(
                pool,
                iteration,
                params.seed_window,
                &mut ws.backward_accum.dcs_initial_resident,
            );
        }
    }

    /// Whether the solve's basis may be captured: a lazy solve's basis describes
    /// the resident subset, not the frozen layout.
    pub(crate) fn captures_basis(&self) -> bool {
        matches!(self, StageOpeningSolver::Frozen)
    }

    /// Whether the loaded LP holds the frozen template's rows: the pool's active
    /// cuts in active-slot order, with the brackets the forward pinned for them. A
    /// lazy solve holds a resident subset in row-map order instead.
    pub(crate) fn solves_frozen_rows(&self) -> bool {
        matches!(self, StageOpeningSolver::Frozen)
    }

    /// Fills `out` with the zero-free `(slot, y)` cut-row duals of the lazy solve
    /// that just finished, in ascending slot order, read through its row map; the
    /// frozen mode leaves `out` untouched, since its extraction reads them in row
    /// order.
    pub(crate) fn read_resident_cut_duals<S: SolverInterface>(
        &self,
        ws: &SolverWorkspace<S>,
        pool: &CutPool,
        out: &mut Vec<(usize, f64)>,
    ) {
        match self {
            StageOpeningSolver::Frozen => {}
            StageOpeningSolver::Lazy(_) => {
                let dcs_solve = &ws.backward_accum.dcs_solve;
                fill_resident_cut_dual_slots(
                    dcs_solve.result_view().dual,
                    &dcs_solve.row_map,
                    pool,
                    out,
                );
            }
        }
    }

    /// Hands the cut rows' `rhs`, which the opening phase has filled in either
    /// mode, to the solver as one range from `first_row`; the lazy mode pushes
    /// each row it makes resident instead.
    pub(crate) fn push_cut_rhs<S: SolverInterface>(
        &self,
        solver: &mut S,
        first_row: usize,
        rhs: &[f64],
    ) {
        match self {
            StageOpeningSolver::Frozen => solver.set_row_lower_bounds_range(first_row, rhs),
            StageOpeningSolver::Lazy(_) => {}
        }
    }

    /// Solves the LP loaded in `ws` for `inputs` in this mode.
    ///
    /// `cut_rows` are the frozen LP's cut rows in row order; the lazy mode
    /// ignores them and reads `inputs.pool`. `lazy` is read only by the lazy mode.
    /// The caller's opening phase has left the active cuts'
    /// values in the scratch, which the lazy mode pushes to the rows it makes
    /// resident.
    ///
    /// # Errors
    ///
    /// Same as [`run_stage_solve_over`] in either mode.
    pub(crate) fn solve_over<'ws, 'c, S: SolverInterface>(
        &self,
        ws: &'ws mut SolverWorkspace<S>,
        inputs: &StageInputs<'_>,
        cut_rows: impl Iterator<Item = (usize, f64, &'c [f64])>,
        lazy: LazyInputs<'_>,
    ) -> Result<SolutionView<'ws>, SddpError> {
        match self {
            StageOpeningSolver::Frozen => run_stage_solve_over(ws, inputs, cut_rows),
            StageOpeningSolver::Lazy(params) => {
                // The cut-free core, never the frozen template: that one already
                // carries the active cuts, which the lazy loop would append again.
                let core = inputs.stage_context.template(inputs.stage_index);
                let cut_rhs = ActiveCutRhs {
                    rhs: &ws.scratch.cut_rhs_buf,
                };
                lazy_solve_preloaded(
                    &mut ws.solver,
                    core,
                    DcsCutView {
                        pool: inputs.pool,
                        state: lazy.state,
                        cut_state: lazy.cut_layout,
                        col_scale: &core.col_scale,
                    },
                    None,
                    &ws.backward_accum.dcs_initial_resident,
                    cut_rhs,
                    params,
                    &mut ws.backward_accum.dcs_solve,
                    DcsSolveContext {
                        stage_index: inputs.stage_index,
                        scenario_index: inputs.scenario_index,
                        iteration: inputs.iteration,
                        continue_carry: lazy.continue_carry,
                        node_id: inputs.node_id,
                    },
                )?;
                Ok(ws.backward_accum.dcs_solve.result_view())
            }
        }
    }
}

/// Fills `out` with (pool slot, raw cut-row dual) pairs for the lazy solve's
/// resident cuts, in ascending slot order.
///
/// Under a lazy solve the resident rows are a row-map-ordered subset, not the
/// frozen cut-row order, so a resident slot's dual is
/// `dual[row_map.lp_row_for_slot(slot)]`. A non-resident slot did not bind (by
/// exactness; else the lazy loop would have added it), so leaving it out is
/// correct.
///
/// A zero dual (either sign) yields no pair.
///
/// `dual` must be the FINAL all-satisfied solve's dual vector and `row_map` the
/// residency that produced it.
fn fill_resident_cut_dual_slots(
    dual: &[f64],
    row_map: &CutRowMap,
    pool: &CutPool,
    out: &mut Vec<(usize, f64)>,
) {
    out.clear();
    out.extend((0..pool.populated()).filter_map(|slot| {
        let lp_row = row_map.lp_row_for_slot(slot)?;
        let y = dual[lp_row];
        (y != 0.0).then_some((slot, y))
    }));
}

#[cfg(test)]
mod tests {
    use super::fill_resident_cut_dual_slots;
    use crate::cut::{CutPool, CutRowMap};
    use crate::setup::NodeId;

    /// Slots 3 then 1 are resident, so the row map's order (3, 1) is not slot
    /// order; the pairs follow ascending slot and slot 1's zero dual yields no pair.
    #[test]
    fn fill_resident_cut_dual_slots_skips_non_resident_slots() {
        let mut pool = CutPool::new(8, 1, 8, 0);
        for forward_pass_index in 0..4 {
            pool.add_cut(NodeId(0), 0, forward_pass_index, 0.0, &[0.0]);
        }
        let mut row_map = CutRowMap::new(8, 2);
        row_map.insert(3);
        row_map.insert(1);
        let dual = [9.0, 9.0, 0.5, 0.0];
        let mut out = vec![(99, 99.0)];

        fill_resident_cut_dual_slots(&dual, &row_map, &pool, &mut out);

        assert_eq!(out, [(3, 0.5)]);
    }

    #[test]
    #[should_panic(expected = "index out of bounds: the len is 3 but the index is 3")]
    fn resident_cut_dual_pairs_index_the_dual_slice_directly() {
        let mut pool = CutPool::new(8, 1, 8, 0);
        for forward_pass_index in 0..4 {
            pool.add_cut(NodeId(0), 0, forward_pass_index, 0.0, &[0.0]);
        }
        let mut row_map = CutRowMap::new(8, 2);
        row_map.insert(3);
        row_map.insert(1);
        let dual = [9.0, 9.0, 0.5];
        let mut out = Vec::new();

        fill_resident_cut_dual_slots(&dual, &row_map, &pool, &mut out);
    }
}
