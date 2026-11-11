use novomodelo_core::System;
use novomodelo_sddp::indexer::{CutSlot, StateDim};
use novomodelo_sddp::test_support::decks::{SLOW_DECKS, committed_decks};
use novomodelo_sddp::test_support::expander::{ExpandedStudy, expand_stage_templates};
use novomodelo_sddp::test_support::stage_state_box_bounds;
use novomodelo_sddp::{RiskMeasure, StudySetup};
use novomodelo_solver::{ActiveSolver, SolverError, SolverInterface};

use crate::common::expanded_solve::{
    CutRowSet, assemble_expanded_solve, lag_dims, load_expanded,
    stages_whose_producer_drops_a_state_dimension, stored_incoming_states,
};
use crate::common::in_code_studies::FPHA_LAG_STUDY_KEY as LAG_STUDY;
use crate::common::in_code_studies::{FphaLagOptions, fpha_lag_study_with};
use crate::common::oracle_tolerance::{assert_first_try, value_tolerance};
use crate::common::stored_cut_oracles::apply_oracle_config;
use crate::common::{
    build_setup_in_code_with_models, fresh_system_and_setup_with, is_chain,
    train_storing_trial_states,
};

const OPENINGS: usize = 3;
const SWEEP_STATES_PER_STAGE: usize = 2;
const MIN_FEASIBLE_POINTS: usize = 3;
/// A cut value of `m` terms rounds at most `m + 2` times along one term: its difference and its
/// product, the `m - 1` adds of the dot, and the closing add of the value it starts from.
const ROUNDINGS_BEYOND_THE_TERMS: usize = 2;

fn rounding_bound(terms: usize) -> f64 {
    let n = terms + ROUNDINGS_BEYOND_THE_TERMS;
    let n_u = f64::from(u32::try_from(n).expect("n fits u32")) * (f64::EPSILON / 2.0);
    n_u / (1.0 - n_u)
}

fn as_f64(count: usize) -> f64 {
    f64::from(u32::try_from(count).expect("count fits u32"))
}

#[derive(Clone, Copy)]
struct Margin {
    count: usize,
    fraction: f64,
    gap: f64,
    bound: f64,
}

impl Default for Margin {
    fn default() -> Self {
        Self {
            count: 0,
            fraction: f64::NEG_INFINITY,
            gap: 0.0,
            bound: 0.0,
        }
    }
}

impl Margin {
    fn record(&mut self, gap: f64, bound: f64) {
        self.count += 1;
        if gap / bound > self.fraction {
            *self = Self {
                count: self.count,
                fraction: gap / bound,
                gap,
                bound,
            };
        }
    }

    fn describe(&self) -> String {
        if self.count == 0 {
            return "0 checks".to_string();
        }
        format!(
            "{} checks, tightest gap {:.3e} against bound {:.3e}",
            self.count, self.gap, self.bound
        )
    }
}

/// `value + Σ slope · (y[dim] − anchor[dim])` over `slopes`, with every dimension a global one.
struct Cut {
    value: f64,
    slopes: Vec<(usize, f64)>,
}

impl Cut {
    fn at(&self, anchor: &[f64], y: &[f64]) -> (f64, f64) {
        let (mut sum, mut magnitude) = (0.0, 0.0);
        for &(dim, slope) in &self.slopes {
            let term = slope * (y[dim] - anchor[dim]);
            sum += term;
            magnitude += term.abs();
        }
        (self.value + sum, magnitude)
    }

    fn has_lag_slope(&self, lag: &[bool]) -> bool {
        self.slopes
            .iter()
            .any(|&(dim, slope)| lag[dim] && slope != 0.0)
    }
}

struct ColdSolve {
    compact: Cut,
    expanded: Cut,
}

struct Fixture<'a> {
    label: &'a str,
    compact: &'a StudySetup,
    expanded: &'a ExpandedStudy,
    risk_measures: Vec<RiskMeasure>,
    lag: Vec<bool>,
}

impl<'a> Fixture<'a> {
    fn new(
        label: &'a str,
        system: &System,
        compact: &'a StudySetup,
        expanded: &'a ExpandedStudy,
    ) -> Self {
        Self {
            label,
            compact,
            expanded,
            risk_measures: stage_risk_measures(system),
            lag: lag_dims(&expanded.layout)
                .iter()
                .map(Option::is_some)
                .collect(),
        }
    }
}

fn stage_risk_measures(system: &System) -> Vec<RiskMeasure> {
    system
        .stages()
        .iter()
        .filter(|stage| stage.id >= 0)
        .map(|stage| RiskMeasure::from(stage.risk_config))
        .collect()
}

#[derive(Default)]
struct Totals {
    stages: usize,
    states: usize,
    solves: usize,
    widest_stage_states: usize,
    pools: Vec<(usize, usize, usize)>,
    skipped_pools: Vec<(usize, &'static str)>,
    dropped_stages: Vec<usize>,
    objectives: Margin,
    compact_validity: Margin,
    expanded_validity: Margin,
    folded_pool: Margin,
    expected_pair_checks: usize,
    expected_pool_checks: usize,
    compact_lag_cuts: usize,
    pool_lag_cuts: usize,
}

/// Solves the compact LP and the expanded LP of `stage` at `x` and `raw_noise`, each from no basis;
/// `None` when the expanded LP is infeasible.
fn solve_cold(
    fx: &Fixture,
    rows: &CutRowSet,
    stage: usize,
    x: &[f64],
    raw_noise: &[f64],
) -> Option<ColdSolve> {
    let solve = assemble_expanded_solve(
        fx.compact,
        fx.expanded,
        &fx.compact.fcf,
        rows,
        stage,
        x,
        raw_noise,
    );
    let mut solver = ActiveSolver::new().expect("ActiveSolver::new must succeed");
    load_expanded(&mut solver, &solve);
    let before = solver.statistics();
    let outcome = solver.solve(None).map(|view| view.to_owned());
    let after = solver.statistics();
    let expanded = match outcome {
        Ok(solution) => solution,
        Err(SolverError::Infeasible) => return None,
        Err(error) => panic!(
            "{}: stage {stage}: the expanded solve failed: {error:?}",
            fx.label
        ),
    };
    assert_first_try(
        &format!("{}: stage {stage}: expanded cold solve", fx.label),
        &before,
        &after,
    );

    let layout = &fx.expanded.layout;
    let col_scale = &fx.expanded.scaled[stage].col_scale;
    let expanded_slopes = (0..layout.state().n_state)
        .map(|dim| {
            let col = layout.incoming_state_col(StateDim::new(dim));
            let scale = if col_scale.is_empty() {
                1.0
            } else {
                col_scale[col]
            };
            (dim, expanded.reduced_costs[col] / scale)
        })
        .collect();
    let projection = &fx.compact.training_ctx().cut_state_layouts[rows.extracted_pool];
    let compact_slopes = solve
        .compact_cut
        .coefficients
        .iter()
        .enumerate()
        .map(|(j, &slope)| (projection.global_state_index(CutSlot::new(j)).get(), slope))
        .collect();
    Some(ColdSolve {
        compact: Cut {
            value: solve.compact_cut.objective,
            slopes: compact_slopes,
        },
        expanded: Cut {
            value: expanded.objective,
            slopes: expanded_slopes,
        },
    })
}

fn check_cold_objectives(context: &str, cold: &ColdSolve, margin: &mut Margin) {
    let (compact, expanded) = (cold.compact.value, cold.expanded.value);
    let tolerance = value_tolerance(compact, expanded);
    let gap = (compact - expanded).abs();
    margin.record(gap, tolerance);
    assert!(
        gap <= tolerance,
        "{context}: compact objective {compact} vs expanded objective {expanded}: gap {gap} > \
         value tolerance {tolerance}"
    );
}

fn assert_below(
    context: &str,
    cut: &Cut,
    anchor: &[f64],
    at: &[f64],
    ceiling: f64,
    margin: &mut Margin,
) {
    let (value, magnitude) = cut.at(anchor, at);
    let tolerance = value_tolerance(value, ceiling);
    let rounding = 2.0 * rounding_bound(cut.slopes.len()) * magnitude;
    let bound = tolerance + rounding;
    margin.record(value - ceiling, bound);
    assert!(
        value <= ceiling + bound,
        "{context}: cut value {value} (value at its anchor {}, {} terms of total magnitude \
         {magnitude}) exceeds {ceiling} by {} > value tolerance {tolerance} + rounding {rounding}",
        cut.value,
        cut.slopes.len(),
        value - ceiling
    );
}

/// At every ordered pair of distinct stored states, each formulation's per-opening cut stays below
/// the other formulation's value at the other state.
fn check_cross_validity(
    fx: &Fixture,
    stage: usize,
    states: &[Vec<f64>],
    cold: &[Vec<ColdSolve>],
    totals: &mut Totals,
) {
    for (i, from) in states.iter().enumerate() {
        for (k, to) in states.iter().enumerate() {
            if i == k {
                continue;
            }
            for (opening, (a, b)) in cold[i].iter().zip(&cold[k]).enumerate() {
                let context = format!(
                    "{}: stage {stage} state {i} cut at state {k}, opening {opening}",
                    fx.label
                );
                assert_below(
                    &format!("{context}: compact cut against the expanded value"),
                    &a.compact,
                    from,
                    to,
                    b.expanded.value,
                    &mut totals.compact_validity,
                );
                assert_below(
                    &format!("{context}: expanded cut against the compact value"),
                    &a.expanded,
                    from,
                    to,
                    b.compact.value,
                    &mut totals.expanded_validity,
                );
            }
        }
    }
}

/// Every populated cut of pool `t` stays below the expanded successor's expected value at the
/// stored states, the midpoint and the lower corner of the state box.
fn check_folded_pool(
    fx: &Fixture,
    t: usize,
    rows: &CutRowSet,
    states: &[Vec<f64>],
    cold: &[Vec<ColdSolve>],
    totals: &mut Totals,
) {
    let s = t + 1;
    let training_ctx = fx.compact.training_ctx();
    let tree = training_ctx.stochastic.tree_view();
    let n_openings = tree.n_openings(s);
    let mean = |values: Vec<f64>| values.iter().sum::<f64>() / as_f64(n_openings);

    let mut points: Vec<(String, Vec<f64>, f64)> = states
        .iter()
        .zip(cold)
        .enumerate()
        .map(|(i, (x, solves))| {
            let q_hat = mean(solves.iter().map(|c| c.expanded.value).collect());
            (format!("stored state {i}"), x.clone(), q_hat)
        })
        .collect();
    let mut infeasible = 0;
    let (lo, hi) = stage_state_box_bounds(fx.compact, t);
    for (name, pick) in [
        ("midpoint", f64::midpoint as fn(f64, f64) -> f64),
        ("lower corner", |lower, _upper| lower),
    ] {
        let x: Vec<f64> = (0..lo.len())
            .map(|dim| {
                if lo[dim].is_finite() && hi[dim].is_finite() {
                    pick(lo[dim], hi[dim])
                } else {
                    states[0][dim]
                }
            })
            .collect();
        let per_opening: Option<Vec<ColdSolve>> = (0..n_openings)
            .map(|opening| solve_cold(fx, rows, s, &x, tree.opening(s, opening)))
            .collect();
        let Some(cold_solves) = per_opening else {
            infeasible += 1;
            continue;
        };
        let q_hat = mean(cold_solves.iter().map(|c| c.expanded.value).collect());
        points.push((name.to_string(), x, q_hat));
    }
    assert!(
        points.len() >= MIN_FEASIBLE_POINTS,
        "{}: pool {t}: {} feasible points ({infeasible} infeasible), fewer than {MIN_FEASIBLE_POINTS}",
        fx.label,
        points.len()
    );

    let pool = &fx.compact.fcf.pools[t];
    let projection = &training_ctx.cut_state_layouts[t];
    let cuts: Vec<Cut> = (0..pool.populated())
        .map(|slot| Cut {
            value: pool.intercept(slot),
            slopes: pool
                .coefficient_row(slot)
                .iter()
                .enumerate()
                .map(|(j, &slope)| (projection.global_state_index(CutSlot::new(j)).get(), slope))
                .collect(),
        })
        .collect();
    totals.pool_lag_cuts += cuts.iter().filter(|cut| cut.has_lag_slope(&fx.lag)).count();
    let origin = vec![0.0; fx.expanded.layout.state().n_state];
    for (idx, (name, x, q_hat)) in points.iter().enumerate() {
        for (slot, cut) in cuts.iter().enumerate() {
            assert_below(
                &format!(
                    "{}: pool {t} point {idx} ({name}) slot {slot}: against the expanded \
                     successor's expected value over {n_openings} openings",
                    fx.label
                ),
                cut,
                &origin,
                x,
                *q_hat,
                &mut totals.folded_pool,
            );
        }
    }
    totals.expected_pool_checks += points.len() * cuts.len();
    totals.pools.push((t, points.len(), infeasible));
}

fn check_stage(fx: &Fixture, t: usize, states_per_stage: Option<usize>, totals: &mut Totals) {
    let s = t + 1;
    assert!(
        matches!(fx.risk_measures[s], RiskMeasure::Expectation),
        "{}: stage {s}: risk measure {:?} is not Expectation",
        fx.label,
        fx.risk_measures[s]
    );
    let Some(mut states) = stored_incoming_states(fx.compact, s) else {
        totals.dropped_stages.push(s);
        return;
    };
    states.truncate(states_per_stage.unwrap_or(usize::MAX));

    let training_ctx = fx.compact.training_ctx();
    let tree = training_ctx.stochastic.tree_view();
    let n_openings = tree.n_openings(s);
    let rows = CutRowSet::active(&fx.compact.fcf, s, t);
    let mut cold = Vec::with_capacity(states.len());
    for (i, x) in states.iter().enumerate() {
        let mut per_opening = Vec::with_capacity(n_openings);
        for opening in 0..n_openings {
            let context = format!("{}: stage {s} state {i} opening {opening}", fx.label);
            let solved =
                solve_cold(fx, &rows, s, x, tree.opening(s, opening)).unwrap_or_else(|| {
                    panic!("{context}: the expanded LP is infeasible at a stored trial state")
                });
            check_cold_objectives(&context, &solved, &mut totals.objectives);
            totals.compact_lag_cuts += usize::from(solved.compact.has_lag_slope(&fx.lag));
            per_opening.push(solved);
        }
        cold.push(per_opening);
    }
    check_cross_validity(fx, s, &states, &cold, totals);
    check_folded_pool(fx, t, &rows, &states, &cold, totals);

    totals.stages += 1;
    totals.states += states.len();
    totals.solves += states.len() * n_openings;
    totals.expected_pair_checks += states.len() * (states.len() - 1) * n_openings;
    totals.widest_stage_states = totals.widest_stage_states.max(states.len());
}

fn check_fixture(fx: &Fixture, states_per_stage: Option<usize>) -> Totals {
    let mut totals = Totals::default();
    let n_pools = fx.compact.fcf.pools.len();
    for t in 0..n_pools {
        if t + 1 == n_pools {
            totals.skipped_pools.push((t, "no successor stage"));
        } else if fx.compact.fcf.active_cuts(t + 1).next().is_none() {
            totals
                .skipped_pools
                .push((t, "the successor holds no active cut"));
        } else {
            check_stage(fx, t, states_per_stage, &mut totals);
        }
    }

    let label = fx.label;
    assert_eq!(
        totals.dropped_stages,
        stages_whose_producer_drops_a_state_dimension(fx.compact),
        "{label}: stages skipped vs stages whose producer drops a state dimension"
    );
    eprintln!(
        "{label}: {} stages, {} states, {} openings solved, {} pools checked {:?} (pool, feasible, \
         infeasible), skipped pools {:?}, stages skipped for a dropped state dimension {:?}; cold \
         objectives: {}; compact cuts against the expanded values: {}; expanded cuts against the \
         compact values: {}; folded pools against the expanded successor: {}; {} compact cuts and \
         {} pool cuts with a non-zero lag coefficient",
        totals.stages,
        totals.states,
        totals.solves,
        totals.pools.len(),
        totals.pools,
        totals.skipped_pools,
        totals.dropped_stages,
        totals.objectives.describe(),
        totals.compact_validity.describe(),
        totals.expanded_validity.describe(),
        totals.folded_pool.describe(),
        totals.compact_lag_cuts,
        totals.pool_lag_cuts
    );
    assert!(
        totals.widest_stage_states >= 2,
        "{label}: no checked stage has two or more stored states"
    );
    assert!(!totals.pools.is_empty(), "{label}: no pool was checked");
    for (name, ran, expected) in [
        ("cold objective", totals.objectives.count, totals.solves),
        (
            "compact cut validity",
            totals.compact_validity.count,
            totals.expected_pair_checks,
        ),
        (
            "expanded cut validity",
            totals.expanded_validity.count,
            totals.expected_pair_checks,
        ),
        (
            "folded pool",
            totals.folded_pool.count,
            totals.expected_pool_checks,
        ),
    ] {
        assert!(ran >= 1, "{label}: no {name} check ran");
        assert_eq!(ran, expected, "{label}: {name} checks run vs expected");
    }
    totals
}

/// Independent cold solves of the compact and the expanded LP reach one objective, and each
/// formulation's cuts, the stored folded pools included, stay below the other formulation's value.
#[test]
fn cold_solves_and_cuts_agree_across_the_formulations() {
    let options = FphaLagOptions {
        openings: OPENINGS,
        ..FphaLagOptions::default()
    };
    let (system, mut config, hydro_models) = fpha_lag_study_with(options);
    apply_oracle_config(&mut config);
    let mut compact = build_setup_in_code_with_models(system, &config, hydro_models);
    train_storing_trial_states(LAG_STUDY, &mut compact);
    let (system, _, _) = fpha_lag_study_with(options);
    let expanded = expand_stage_templates(&system, &compact);

    let totals = check_fixture(&Fixture::new(LAG_STUDY, &system, &compact, &expanded), None);
    assert!(
        totals.compact_lag_cuts >= 1,
        "{LAG_STUDY}: no checked compact cut has a non-zero lag coefficient"
    );
    assert!(
        totals.pool_lag_cuts >= 1,
        "{LAG_STUDY}: no checked pool cut has a non-zero lag coefficient"
    );
}

/// The same agreement on the first two stored states of every stage of every committed deck with
/// inflow-lag state and a risk-neutral objective.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "trains every committed deck with inflow lags"
)]
fn cold_solves_and_cuts_agree_across_the_formulations_on_every_lag_deck() {
    let (mut visited, mut checked) = (Vec::new(), Vec::new());
    let (mut not_chain, mut not_risk_neutral) = (Vec::new(), Vec::new());
    let (mut compact_lag_cuts, mut pool_lag_cuts) = (0, 0);
    for deck in committed_decks() {
        if SLOW_DECKS.contains(&deck.key.as_str()) {
            continue;
        }
        let (system, mut compact) = fresh_system_and_setup_with(&deck.dir, apply_oracle_config);
        if compact.stage_state().max_par_order == 0 {
            continue;
        }
        if !is_chain(&compact) {
            not_chain.push(deck.key);
            continue;
        }
        visited.push(deck.key.clone());
        if !stage_risk_measures(&system)
            .iter()
            .all(|risk| matches!(risk, RiskMeasure::Expectation))
        {
            eprintln!("{}: skipped, its risk measure is not Expectation", deck.key);
            not_risk_neutral.push(deck.key);
            continue;
        }
        train_storing_trial_states(&deck.key, &mut compact);
        let expanded = expand_stage_templates(&system, &compact);
        let totals = check_fixture(
            &Fixture::new(&deck.key, &system, &compact, &expanded),
            Some(SWEEP_STATES_PER_STAGE),
        );
        compact_lag_cuts += totals.compact_lag_cuts;
        pool_lag_cuts += totals.pool_lag_cuts;
        checked.push(deck.key);
    }
    eprintln!(
        "visited {} lag decks: {visited:?}; not chains: {not_chain:?}; not risk-neutral: \
         {not_risk_neutral:?}; checked: {checked:?}",
        visited.len()
    );
    assert!(
        visited.len() >= 2,
        "fewer than two committed lag decks were visited: {visited:?}"
    );
    assert!(
        !checked.is_empty(),
        "every visited lag deck was skipped for its risk measure: {not_risk_neutral:?}"
    );
    assert!(
        compact_lag_cuts >= 1 && pool_lag_cuts >= 1,
        "no checked cut has a non-zero lag coefficient ({compact_lag_cuts} compact cuts, \
         {pool_lag_cuts} pool cuts) on {checked:?}"
    );
}
