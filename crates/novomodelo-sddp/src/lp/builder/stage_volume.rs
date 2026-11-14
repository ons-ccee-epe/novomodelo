//! Stage-volume (hm³) left-hand sides, keyed by system index.
//!
//! A stage volume is `Σ_k τ_k · flow_k` over every block in ascending order; τ comes
//! only from `BlockClock::tau`. Generic-constraint and native rows both call these
//! functions, so they carry identical coefficients.

use super::layout::{StageLayout, TemplateBuildCtx};
use crate::indexer::{BlockIdx, HydroCell, HydroSys, PumpingSys};

/// Stage volume (hm³): every block's flow columns from `per_block`, each weighted by
/// that block's `tau`, in ascending block order.
pub(super) fn resolve_stage_volume(
    layout: &StageLayout<'_>,
    per_block: impl Fn(BlockIdx) -> Vec<(usize, f64)>,
) -> Vec<(usize, f64)> {
    (0..layout.clock.n_blks())
        .map(BlockIdx::new)
        .flat_map(|blk| {
            let tau = layout.clock.tau(blk);
            per_block(blk)
                .into_iter()
                .map(move |(col, m)| (col, m * tau))
        })
        .collect()
}

/// Every cell's turbine column of `h` in cell order, then its spillage column, each
/// at `1.0`; diversion is excluded.
pub(super) fn hydro_outflow_columns(
    h: HydroSys,
    blk: BlockIdx,
    ctx: &TemplateBuildCtx<'_>,
    layout: &StageLayout<'_>,
) -> Vec<(usize, f64)> {
    let mut columns: Vec<(usize, f64)> = ctx
        .hydro_cell_index
        .cells_of(h)
        .map(|c| (layout.geometry.turbine_col(HydroCell::new(c), blk), 1.0))
        .collect();
    columns.push((layout.geometry.spillage_col(h, blk), 1.0));
    columns
}

pub(super) fn hydro_outflow_volume_terms(
    h: HydroSys,
    ctx: &TemplateBuildCtx<'_>,
    layout: &StageLayout<'_>,
) -> Vec<(usize, f64)> {
    resolve_stage_volume(layout, |blk| hydro_outflow_columns(h, blk, ctx, layout))
}

pub(super) fn hydro_diversion_volume_terms(
    h: HydroSys,
    layout: &StageLayout<'_>,
) -> Vec<(usize, f64)> {
    resolve_stage_volume(layout, |blk| {
        vec![(layout.geometry.diversion_col(h, blk), 1.0)]
    })
}

pub(super) fn pumping_volume_terms(p: PumpingSys, layout: &StageLayout<'_>) -> Vec<(usize, f64)> {
    resolve_stage_volume(layout, |blk| {
        vec![(layout.geometry.pumping_flow_col(p, blk), 1.0)]
    })
}
