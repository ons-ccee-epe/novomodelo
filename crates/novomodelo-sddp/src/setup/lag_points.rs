//! Classifies each cut pool's lag point from the stage calendar.
//!
//! The weight recursion of the lag kernel never reads a realized value, so a
//! pool's class follows from the stage transitions and the stage-0 seed alone.

use novomodelo_core::temporal::StageLagTransition;
use novomodelo_stochastic::DerivedInflowSeeds;
use novomodelo_stochastic::par::lag_kernel::{LagWeightState, advance_lag_weights};

use super::NodeGraph;
use crate::lp::indexer::{CutStateProjection, LagPointMode};

pub(super) struct LagPointClass {
    pub(super) mode: LagPointMode,
    pub(super) inverse_weight: Vec<f64>,
    pub(super) share: Vec<f64>,
    pub(super) incoming_weight: Vec<f64>,
}

impl LagPointClass {
    fn without_coefficients(mode: LagPointMode) -> Self {
        Self {
            mode,
            inverse_weight: Vec::new(),
            share: Vec::new(),
            incoming_weight: Vec::new(),
        }
    }
}

#[expect(
    clippy::float_cmp,
    reason = "the closing multiply reproduces the realized value only when w·(1/w) is exactly 1.0"
)]
fn scales_to_unity(weight: f64) -> bool {
    weight * (1.0 / weight) == 1.0
}

/// One class per stage, in stage order, replaying the lag-kernel weights from `seed`.
///
/// Warns once for each stage that rebuilds its lags from the downstream ring.
pub(super) fn classify_lag_points(
    stage_ids: &[i32],
    transitions: &[StageLagTransition],
    downstream_par_order: usize,
    seed: &DerivedInflowSeeds,
) -> Vec<LagPointClass> {
    let mut weights = LagWeightState::from_seed(&seed.weight);
    let mut seed_accumulator_open = seed.accum.iter().any(|&accumulated| accumulated != 0.0);

    transitions
        .iter()
        .zip(stage_ids)
        .map(|(transition, &stage_id)| {
            let mut after_own_downstream_update = weights.clone();
            advance_lag_weights(
                &mut after_own_downstream_update,
                &StageLagTransition {
                    rebuild_from_downstream: false,
                    ..*transition
                },
                downstream_par_order,
            );
            let rebuilds =
                transition.rebuild_from_downstream && after_own_downstream_update.n_completed > 0;

            let w = transition.accumulate_weight;
            let class = if rebuilds {
                tracing::warn!(
                    "stage {stage_id}: the inflow lags step from months to quarters here; cuts at \
                     this stage evaluate inflow lags as at a single-resolution stage"
                );
                LagPointClass::without_coefficients(LagPointMode::Shifted)
            } else if transition.finalize_period {
                let monthly_identity = scales_to_unity(w)
                    && !seed_accumulator_open
                    && weights.primary.iter().all(|&held| held == 0.0);
                if monthly_identity {
                    LagPointClass::without_coefficients(LagPointMode::Shifted)
                } else {
                    let inverse_weight: Vec<f64> = weights
                        .primary
                        .iter()
                        .map(|&held| {
                            let total = held + w;
                            if total > 0.0 { 1.0 / total } else { 1.0 }
                        })
                        .collect();
                    LagPointClass {
                        mode: LagPointMode::Closing,
                        share: inverse_weight.iter().map(|&inverse| w * inverse).collect(),
                        inverse_weight,
                        incoming_weight: weights.primary.clone(),
                    }
                }
            } else {
                LagPointClass::without_coefficients(LagPointMode::Carry)
            };

            if transition.finalize_period && !rebuilds {
                seed_accumulator_open = false;
            }
            advance_lag_weights(&mut weights, transition, downstream_par_order);
            class
        })
        .collect()
}

/// Installs on every pool the class of its stage, leaf pools included; a pool
/// without lag slots keeps the shifted point.
pub(super) fn install_lag_points(
    layouts: &mut [CutStateProjection],
    node_graph: &NodeGraph,
    stage_ids: &[i32],
    transitions: &[StageLagTransition],
    downstream_par_order: usize,
    seed: &DerivedInflowSeeds,
) {
    let classes = classify_lag_points(stage_ids, transitions, downstream_par_order, seed);
    for (layout, stage) in layouts.iter_mut().zip(&node_graph.pool_stage) {
        if layout.has_lag_slots() {
            let class = &classes[stage.0];
            layout.lag_point_mut().install_mode(
                class.mode,
                &class.inverse_weight,
                &class.share,
                &class.incoming_weight,
            );
        }
    }
}
