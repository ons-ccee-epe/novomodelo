//! The aggregate-first cut over a successor node's whole opening set, shared by both backward
//! schedulers.

use crate::{
    context::TrainingContext,
    cut::chain_rule::{ChainRuleInputs, ChildOpenings, OpeningDuals, aggregate_first_cut_into},
    risk_measure::{RiskMeasure, RiskMeasureScratch},
    workspace::workspace::{SuccessorAggregateScratch, recycle},
};

use super::{SuccessorOutcomes, SuccessorSpec};

/// Writes the aggregate-first cut over `succ`'s whole opening set into `coefficients` and returns
/// its intercept: one joint weight call over the canonical-ω costs, then one
/// [`aggregate_first_cut_into`] over the children. `view_of` maps a canonical ω to its duals.
pub(super) fn aggregate_successor_cut<'r>(
    training_ctx: &TrainingContext<'_>,
    succ: &SuccessorSpec<'_>,
    outcomes: &SuccessorOutcomes<'_>,
    risk_measure: &RiskMeasure,
    x_hat: &[f64],
    view_of: impl Fn(usize) -> OpeningDuals<'r>,
    scratch: &mut SuccessorAggregateScratch,
    risk_scratch: &mut RiskMeasureScratch,
    coefficients: &mut [f64],
) -> f64 {
    let SuccessorAggregateScratch {
        aggregate,
        costs,
        views,
        children,
        ..
    } = scratch;
    let n_openings = succ.probabilities.len();
    let mut opening_views: Vec<OpeningDuals<'r>> = recycle(std::mem::take(views));
    costs.clear();
    for omega in 0..n_openings {
        let view = view_of(omega);
        costs.push(view.objective);
        opening_views.push(view);
    }
    let weights = risk_measure.opening_weights_into(costs, succ.probabilities, risk_scratch);

    let mut child_list: Vec<ChildOpenings<'_>> = recycle(std::mem::take(children));
    for ci in 0..outcomes.n_children() {
        let child = outcomes.child(ci);
        child_list.push(ChildOpenings {
            inputs: ChainRuleInputs {
                slots: child.chain_rule_slots,
                resident_pool: child.successor_pool,
                extracted: succ.cut_state,
                par: training_ctx.stochastic.par(),
                stage: succ.successor.0,
            },
            openings: &opening_views[child.outcome_range.clone()],
        });
    }
    let intercept = aggregate_first_cut_into(&child_list, weights, x_hat, aggregate, coefficients);
    *children = recycle(child_list);
    *views = recycle(opening_views);
    intercept
}
