//! Shared stage-LP solve-preparation pipeline (pin and bracket, then patch the
//! opening): the single owner the forward pass, backward pass, simulation
//! pipeline, and lower-bound evaluation all route through, so every solve site
//! patches the LP identically (the "patch NCS identically" contract, D15).
//!
//! Home is `training/`, not `lp/builder/`: the pipeline needs `crate::context`,
//! `crate::workspace`, and `crate::noise`, which `training/` already depends on,
//! whereas `lp/builder/` sits below `training/` in the crate layering — hosting it
//! there would invert the dependency direction.

use novomodelo_solver::SolverInterface;
use novomodelo_stochastic::evaluate_par_conditional_mean_batch;

use crate::{
    context::{StageContext, TrainingContext},
    cut::{
        CutPool,
        rhs::{cut_brackets_into, cut_rhs_from_noise_terms_into, cut_rhs_into},
    },
    lp::builder::{PatchBuffer, StateBox},
    lp::indexer::{BlockGrid, CutStateProjection},
    noise::{
        apply_inflow_slack_col_bounds, apply_ncs_col_bounds, realize_inflow, transform_load_noise,
        transform_ncs_noise,
    },
    setup::node_graph::StageIdx,
    solve::StageOpeningSolver,
    workspace::ScratchBuffers,
};

/// The state slice this stage solve pins as incoming state; the caller resolves
/// it per site (forward `current_state`, backward `x_hat`, lower-bound
/// `initial_state`, simulation `current_state`).
#[derive(Debug, Clone, Copy)]
pub(crate) struct StateSource<'a>(pub &'a [f64]);

/// Inputs of the single-solve prep: the pinned state, this solve's raw
/// noise, the resident pool and its projection, and the opening solver.
pub(crate) struct SingleSolvePrepParams<'a> {
    pub(crate) state_source: StateSource<'a>,
    pub(crate) raw_noise: &'a [f64],
    pub(crate) pool: &'a CutPool,
    pub(crate) projection: &'a CutStateProjection,
    pub(crate) opening_solver: &'a StageOpeningSolver,
}

/// Where the pin phase takes the cut-row bracket B from.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum BracketSource<'a> {
    /// The pin phase computes every cut row's bracket from the trial state.
    Computed,
    /// The frozen-prefix brackets the forward wrote, or `None` for a miss; the rows after the prefix are computed.
    Slab(Option<&'a [f64]>),
}

/// Where the opening phase takes the noise term σ·η and the cut-row terms W from.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum NoiseTermSource<'a> {
    /// The opening computes σ·η and each W from its own raw noise.
    OnTheFly,
    /// σ·η per hydro, and `cut_noise[i]` the W of `cut_rows.slots[i]`, both computed once for the stage.
    Stage {
        noise_term: &'a [f64],
        cut_noise: &'a [f64],
    },
}

/// The cut rows of the loaded LP whose brackets the pin phase computes.
pub(crate) struct LoadedCutRows<'a> {
    /// The resident pool the cut rows come from.
    pub(crate) pool: &'a CutPool,
    /// That pool's projection, the owner of the lag point B reads.
    pub(crate) projection: &'a CutStateProjection,
    /// `slots[k]` is the pool slot of LP row `first_row + k`: frozen rows in `active_cuts()` order, then delta rows in `active_delta_cuts` order.
    pub(crate) slots: &'a [usize],
    /// The loaded LP's structural row count, which is the first cut row.
    pub(crate) first_row: usize,
}

/// The single owner of the shared stage-LP solve-preparation pipeline.
pub(crate) struct StageSolvePrep;

impl StageSolvePrep {
    /// The previous stage's admissible box, which bounds the pinned incoming state;
    /// `None` at stage 0, whose incoming state is the study's initial condition.
    fn producer_box<'a>(ctx: &'a StageContext<'_>, stage: StageIdx) -> Option<&'a StateBox> {
        if stage.0 == 0 {
            None
        } else {
            Some(ctx.state_box(StageIdx(stage.0 - 1)))
        }
    }

    /// Pins the incoming state on the loaded LP; runs once per loaded LP and state.
    fn pin<S>(
        solver: &mut S,
        patch_buf: &mut PatchBuffer,
        ctx: &StageContext<'_>,
        training_ctx: &TrainingContext<'_>,
        stage: StageIdx,
        pinned_state: &[f64],
    ) where
        S: SolverInterface,
    {
        patch_buf.fill_col_state_patches(
            training_ctx.state,
            pinned_state,
            &ctx.template(stage).col_scale,
            Self::producer_box(ctx, stage),
        );
        let cp = patch_buf.active_state_col_patch_count();
        solver.set_col_bounds(
            &patch_buf.col_indices[..cp],
            &patch_buf.col_lower[..cp],
            &patch_buf.col_upper[..cp],
        );
    }

    /// Pin phase: runs once per unit (a trial point on a loaded LP). Pins the
    /// columned state and computes p and the cut-row brackets B from the raw
    /// `trial_state`, never the scaled pin. Allocates nothing once the scratch
    /// buffers are sized.
    pub(crate) fn pin_and_bracket<S>(
        solver: &mut S,
        patch_buf: &mut PatchBuffer,
        scratch: &mut ScratchBuffers,
        ctx: &StageContext<'_>,
        training_ctx: &TrainingContext<'_>,
        stage: StageIdx,
        trial_state: StateSource<'_>,
        cut_rows: &LoadedCutRows<'_>,
        source: BracketSource<'_>,
    ) where
        S: SolverInterface,
    {
        Self::pin(solver, patch_buf, ctx, training_ctx, stage, trial_state.0);

        let incoming_lags = &trial_state.0[training_ctx.state.state_dim_lag_range()];
        let par_lp = training_ctx.stochastic.par();
        if par_lp.n_stages() > 0 {
            evaluate_par_conditional_mean_batch(
                par_lp,
                stage.0,
                incoming_lags,
                &mut scratch.par_mean_buf,
            );
        } else {
            scratch.par_mean_buf.fill(0.0);
        }

        let prefix: &[f64] = match source {
            BracketSource::Computed => &[],
            BracketSource::Slab(Some(prefix)) => prefix,
            BracketSource::Slab(None) => {
                #[cfg(any(test, feature = "test-support"))]
                {
                    scratch.slab_misses += 1;
                }
                &[]
            }
        };
        let n = prefix.len();
        debug_assert!(
            n <= cut_rows.slots.len(),
            "the slab prefix exceeds the cut rows of the loaded LP"
        );
        scratch.cut_bracket_buf.resize(cut_rows.slots.len(), 0.0);
        scratch.cut_bracket_buf[..n].copy_from_slice(prefix);
        cut_brackets_into(
            cut_rows.pool,
            &cut_rows.slots[n..],
            cut_rows.projection.lag_point(),
            incoming_lags,
            &scratch.par_mean_buf,
            &mut scratch.cut_bracket_buf[n..],
        );
    }

    /// Opening phase: runs once per solve, after the pin phase of the unit.
    /// Allocates nothing once the scratch buffers are sized.
    // Rationale (too_many_arguments): solver, patch_buf, and scratch are three
    // disjoint mutable borrows of the workspace that must stay separate
    // parameters — a bundling struct would reborrow the whole workspace and
    // defeat the split; the rest are distinct immutable inputs.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn patch_opening<S>(
        solver: &mut S,
        patch_buf: &mut PatchBuffer,
        scratch: &mut ScratchBuffers,
        ctx: &StageContext<'_>,
        training_ctx: &TrainingContext<'_>,
        stage: StageIdx,
        raw_noise: &[f64],
        cut_rows: &LoadedCutRows<'_>,
        noise: NoiseTermSource<'_>,
        opening_solver: &StageOpeningSolver,
    ) where
        S: SolverInterface,
    {
        let stage_noise_term = match noise {
            NoiseTermSource::OnTheFly => None,
            NoiseTermSource::Stage { noise_term, .. } => Some(noise_term),
        };
        realize_inflow(raw_noise, stage, training_ctx, scratch, stage_noise_term);
        Self::patch_load_rows(patch_buf, scratch, ctx, training_ctx, stage, raw_noise);
        patch_buf.fill_inflow_shift_patches(
            &ctx.inflow_coupling[stage.0],
            &scratch.realized_inflow_buf,
            &ctx.template(stage).row_scale,
        );

        let pc = patch_buf.forward_patch_count();
        solver.set_row_bounds(
            &patch_buf.indices[..pc],
            &patch_buf.lower[..pc],
            &patch_buf.upper[..pc],
        );

        Self::apply_noise_col_bounds(solver, scratch, ctx, training_ctx, stage, raw_noise);

        debug_assert_eq!(
            scratch.cut_bracket_buf.len(),
            cut_rows.slots.len(),
            "the opening phase reads one bracket per cut row of the pin phase"
        );
        scratch.cut_rhs_buf.resize(cut_rows.slots.len(), 0.0);
        match noise {
            NoiseTermSource::OnTheFly => cut_rhs_into(
                cut_rows.pool,
                cut_rows.slots,
                cut_rows.projection.lag_point(),
                &scratch.par_noise_term_buf,
                &scratch.cut_bracket_buf,
                &mut scratch.cut_rhs_buf,
            ),
            NoiseTermSource::Stage { cut_noise, .. } => {
                debug_assert_eq!(
                    cut_noise.len(),
                    cut_rows.slots.len(),
                    "one stage W per cut row"
                );
                cut_rhs_from_noise_terms_into(
                    &scratch.cut_bracket_buf,
                    cut_noise,
                    &mut scratch.cut_rhs_buf,
                );
            }
        }
        opening_solver.push_cut_rhs(solver, cut_rows.first_row, &scratch.cut_rhs_buf);
    }

    /// The prep of a site that solves one opening per loaded LP and
    /// pinned state: the pin phase, then the opening phase through
    /// `params.opening_solver`. `slots` are the pool slots of the loaded LP's
    /// cut rows in row order. Allocates nothing once the slot buffer has grown.
    pub(crate) fn prepare_single<S>(
        solver: &mut S,
        patch_buf: &mut PatchBuffer,
        scratch: &mut ScratchBuffers,
        ctx: &StageContext<'_>,
        training_ctx: &TrainingContext<'_>,
        stage: StageIdx,
        params: &SingleSolvePrepParams<'_>,
        slots: impl Iterator<Item = usize>,
    ) where
        S: SolverInterface,
    {
        let mut slot_buf = std::mem::take(&mut scratch.cut_slots_buf);
        slot_buf.clear();
        slot_buf.extend(slots);
        let rows = LoadedCutRows {
            pool: params.pool,
            projection: params.projection,
            slots: &slot_buf,
            first_row: ctx.template(stage).num_rows,
        };
        Self::pin_and_bracket(
            solver,
            patch_buf,
            scratch,
            ctx,
            training_ctx,
            stage,
            params.state_source,
            &rows,
            BracketSource::Computed,
        );
        Self::patch_opening(
            solver,
            patch_buf,
            scratch,
            ctx,
            training_ctx,
            stage,
            params.raw_noise,
            &rows,
            NoiseTermSource::OnTheFly,
            params.opening_solver,
        );
        scratch.cut_slots_buf = slot_buf;
    }

    fn patch_load_rows(
        patch_buf: &mut PatchBuffer,
        scratch: &mut ScratchBuffers,
        ctx: &StageContext<'_>,
        training_ctx: &TrainingContext<'_>,
        stage: StageIdx,
        raw_noise: &[f64],
    ) {
        let load_blocks = if ctx.load_bus_indices.is_empty() {
            0
        } else {
            ctx.block_count(stage)
        };
        transform_load_noise(
            raw_noise,
            training_ctx.stochastic,
            stage,
            load_blocks,
            &mut scratch.load_rhs_buf,
        );

        if !ctx.load_bus_indices.is_empty() {
            let grid = BlockGrid::new(load_blocks, training_ctx.study_dims.max_deficit_segments);
            patch_buf.fill_load_patches(
                ctx.geometry_per_stage[stage.0].load_balance,
                grid,
                &scratch.load_rhs_buf,
                ctx.load_bus_indices,
                &ctx.template(stage).row_scale,
            );
        }
    }

    fn apply_noise_col_bounds<S>(
        solver: &mut S,
        scratch: &mut ScratchBuffers,
        ctx: &StageContext<'_>,
        training_ctx: &TrainingContext<'_>,
        stage: StageIdx,
        raw_noise: &[f64],
    ) where
        S: SolverInterface,
    {
        if training_ctx.stochastic.n_stochastic_ncs() > 0 {
            transform_ncs_noise(
                raw_noise,
                training_ctx.stochastic,
                stage,
                ctx.block_count(stage),
                ctx.ncs_max_gen,
                ctx.ncs_allow_curtailment,
                &mut scratch.ncs_col_lower_buf,
                &mut scratch.ncs_col_upper_buf,
            );
            // Stage id is the dormancy key (NOT the index `stage`; filtered
            // placeholder stages can shift the id off the index).
            #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
            let stage_id = training_ctx
                .stages
                .get(stage.0)
                .map_or(stage.0 as i32, |s| s.id);
            apply_ncs_col_bounds(
                solver,
                scratch,
                &ctx.geometry_per_stage[stage.0],
                ctx.ncs_stochastic_dense_col,
                ctx.ncs_stochastic_windows,
                stage_id,
            );
        }

        if training_ctx.inflow_method.truncates() {
            apply_inflow_slack_col_bounds(
                solver,
                scratch,
                &ctx.geometry_per_stage[stage.0],
                &ctx.template(stage).col_scale,
            );
        }
    }
}

#[cfg(test)]
mod tests;
