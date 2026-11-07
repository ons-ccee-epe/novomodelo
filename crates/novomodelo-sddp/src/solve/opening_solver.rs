//! The frozen-or-lazy stage solve: one dispatch for every site that solves a
//! stage LP against its cuts.

use novomodelo_solver::{SolutionView, SolverInterface};

use crate::{
    cut::CutPool,
    dcs::{DcsParams, DcsSolveContext, build_initial_resident_set, lazy_solve_preloaded},
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
        match dcs_params {
            Some(params) => StageOpeningSolver::Lazy(params),
            None => StageOpeningSolver::Frozen,
        }
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

    /// Solves the LP loaded in `ws` for `inputs` in this mode.
    ///
    /// `cut_rows` are the frozen LP's cut rows in row order; the lazy mode
    /// ignores them and reads `inputs.pool`. `lazy` is read only by the lazy mode.
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
                lazy_solve_preloaded(
                    &mut ws.solver,
                    core,
                    inputs.pool,
                    lazy.state,
                    lazy.cut_layout,
                    &core.col_scale,
                    None,
                    &ws.backward_accum.dcs_initial_resident,
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
