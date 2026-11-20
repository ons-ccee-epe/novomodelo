//! Reads which noise term a solve of a study under `"lognormal_3p"` took, from the bits of its realized inflow.

#![allow(clippy::panic)]

use novomodelo_sddp::StudySetup;
use novomodelo_sddp::setup::NodePos;
use novomodelo_sddp::test_support::node_opening_noise;
use novomodelo_stochastic::{
    add_noise_term_batch, evaluate_lognormal_noise_term_batch, evaluate_par_conditional_mean_batch,
    evaluate_par_noise_term_batch,
};

use super::try_stored_trial_states;

/// The noise term a realized inflow carries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Branch {
    /// The lognormal term, with a conditional mean above zero.
    Lognormal,
    /// The normal term, with a conditional mean at or below zero.
    Fallback,
    /// The normal term, with a conditional mean above zero.
    Normal,
    /// A term the lognormal and normal kernels both give, so the bits do not tell them apart.
    Tied,
    /// A value no opening of the node gives under either kernel.
    Unmatched,
}

#[derive(Clone, Copy, Default)]
struct Seen {
    normal: bool,
    lognormal: bool,
    tied: bool,
}

/// The PAR conditional mean per hydro at `stage`, from the lags of `incoming_state`.
pub fn conditional_means(setup: &StudySetup, stage: usize, incoming_state: &[f64]) -> Vec<f64> {
    let space = setup.stage_state();
    let n_hydros = space.hydro_count;
    let lags = &incoming_state[n_hydros..n_hydros * (1 + space.max_par_order)];
    let mut means = vec![0.0; n_hydros];
    evaluate_par_conditional_mean_batch(
        setup.training_ctx().stochastic.par(),
        stage,
        lags,
        &mut means,
    );
    means
}

/// Calls `visit(normal, lognormal)` with the realized inflows both kernels give at each opening of `node`.
fn for_each_candidate(
    setup: &StudySetup,
    node: NodePos,
    conditional_mean: &[f64],
    mut visit: impl FnMut(&[f64], &[f64]),
) {
    let training_ctx = setup.training_ctx();
    let par = training_ctx.stochastic.par();
    let n_hydros = par.n_hydros();
    let stage = setup.inputs.node_graph.nodes[node].stage.0;
    let (mut term, mut normal, mut lognormal) = (
        vec![0.0; n_hydros],
        vec![0.0; n_hydros],
        vec![0.0; n_hydros],
    );
    for opening in 0..setup.inputs.node_graph.nodes[node].openings.len {
        let noise = node_opening_noise(setup, node, opening);
        let eta = &noise[..n_hydros];
        evaluate_par_noise_term_batch(par, stage, eta, &mut term);
        add_noise_term_batch(conditional_mean, &term, &mut normal);
        evaluate_lognormal_noise_term_batch(par, stage, conditional_mean, eta, &mut term);
        add_noise_term_batch(conditional_mean, &term, &mut lognormal);
        visit(&normal, &lognormal);
    }
}

/// The branch each hydro's `realized` inflow took at `node`, given the conditional mean it was solved at.
pub fn classify(
    setup: &StudySetup,
    node: NodePos,
    conditional_mean: &[f64],
    realized: &[f64],
) -> Vec<Branch> {
    let mut seen = vec![Seen::default(); conditional_mean.len()];
    for_each_candidate(setup, node, conditional_mean, |normal, lognormal| {
        for (((seen, &z), &n), &l) in seen.iter_mut().zip(realized).zip(normal).zip(lognormal) {
            let (z, n, l) = (z.to_bits(), n.to_bits(), l.to_bits());
            seen.normal |= z == n && n != l;
            seen.lognormal |= z == l && n != l;
            seen.tied |= z == n && n == l;
        }
    });
    conditional_mean
        .iter()
        .zip(&seen)
        .map(|(&p, seen)| {
            if p > 0.0 {
                match (seen.lognormal, seen.normal, seen.tied) {
                    (true, false, _) => Branch::Lognormal,
                    (false, true, _) => Branch::Normal,
                    (false, false, true) => Branch::Tied,
                    _ => Branch::Unmatched,
                }
            } else if seen.normal || seen.tied {
                Branch::Fallback
            } else {
                Branch::Unmatched
            }
        })
        .collect()
}

/// Panics naming `label` unless the incoming states of the stored pins include a hydro whose
/// conditional mean is at or below zero with a nonzero sigma, and one whose conditional mean is above
/// zero with an opening where the two kernels differ.
pub fn assert_the_stored_pins_reach_both_branches(label: &str, setup: &StudySetup) {
    let training_ctx = setup.training_ctx();
    let par = training_ctx.stochastic.par();
    let (mut reaches_fallback, mut reaches_lognormal) = (false, false);
    for stage in 0..setup.num_stages() {
        let pins = if stage == 0 {
            vec![training_ctx.initial_state.to_vec()]
        } else {
            try_stored_trial_states(setup, NodePos(stage - 1)).unwrap_or_else(|| {
                panic!(
                    "{label}: the pool of node {} drops a state dimension",
                    stage - 1
                )
            })
        };
        for pin in &pins {
            let means = conditional_means(setup, stage, pin);
            reaches_fallback |= means
                .iter()
                .enumerate()
                .any(|(h, &p)| p <= 0.0 && par.sigma(stage, h) > 0.0);
            for_each_candidate(setup, NodePos(stage), &means, |normal, lognormal| {
                reaches_lognormal |= means
                    .iter()
                    .zip(normal)
                    .zip(lognormal)
                    .any(|((&p, n), l)| p > 0.0 && n.to_bits() != l.to_bits());
            });
        }
    }
    assert!(
        reaches_fallback,
        "{label}: no stored pin has a conditional mean at or below zero with a nonzero sigma"
    );
    assert!(
        reaches_lognormal,
        "{label}: no stored pin has a positive conditional mean and an opening where the lognormal term differs from the normal one"
    );
}
