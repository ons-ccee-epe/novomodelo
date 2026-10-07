//! Shared η-inversion driver for [`standardize_external_inflow`] and
//! [`standardize_historical_windows`].
//!
//! [`standardize_external_inflow`]: super::external::standardize_external_inflow
//! [`standardize_historical_windows`]: super::historical::standardize_historical_windows

use cobre_core::temporal::StageLagTransition;

use crate::par::{
    DownstreamLagAccum, EntityMajor, PrimaryLagAccum, advance_lag_chain, evaluate::solve_par_noise,
    precompute::PrecomputedPar, resolve_stage_lag_transition,
};
use crate::seeds::DerivedSeed;

/// Run the PAR(p) η-inversion loop shared by the external-scenario and
/// historical-window sampling schemes: seed the lag chain from `seed`
/// ([`DerivedSeed`]), then for every `(outer, stage, hydro)` triple invert
/// the PAR(p) model via [`solve_par_noise`] and advance the lag chain via
/// [`advance_lag_chain`], resetting to the same derived seed at each `outer`
/// boundary.
///
/// `raw_target(t, outer, h)` supplies the raw value to invert;
/// `write_eta(t, outer, h, eta)` stores the result — each caller supplies its
/// own raw-value lookup and its own library's `eta_slice_mut` addressing;
/// this driver owns only the scratch buffers and the loop nest.
///
/// # Panics
///
/// Panics in debug builds if dimension mismatches are detected.
// Rationale: the lag-state and accumulator buffers thread through the
// (outer × stage × entity) loop in strict sequence; extracting further
// helpers would pass them by mutable ref across several call boundaries.
#[allow(clippy::too_many_lines)]
// Rationale: `seed` folds the former lag/accum/weight arguments into one
// aggregate; the remaining dimension counts, the PAR model, the
// lag-transition table, and the two per-scheme callbacks are independent
// inputs with no further shared aggregate.
#[allow(clippy::too_many_arguments)]
pub(super) fn run_eta_inversion<F, W>(
    n_stages: usize,
    outer_count: usize,
    n_hydros: usize,
    max_order: usize,
    par: &PrecomputedPar,
    seed: DerivedSeed<'_>,
    stage_lag_transitions: &[StageLagTransition],
    downstream_par_order: usize,
    raw_target: F,
    mut write_eta: W,
) where
    F: Fn(usize, usize, usize) -> f64,
    W: FnMut(usize, usize, usize, f64),
{
    // past_lag_buf[h_idx * safe_max_order + lag]
    let safe_max_order = max_order.max(1);
    let mut past_lag_buf = vec![0.0_f64; n_hydros * safe_max_order];
    for h in 0..n_hydros {
        // Fill up to `max_order` slots so PAR(p)-A annual contributions
        // (widened across the `psi` slice) see real lag values, not zeros.
        // `max_order.min(seed.l_state)` truncates silently when `max_order`
        // under-covers `seed.l_state` — the caller must pass a `max_order`
        // already widened to the intended lag depth, never rely on this `min`
        // to raise it.
        for lag in 0..max_order.min(seed.l_state) {
            past_lag_buf[h * safe_max_order + lag] = seed.lag_values[h * seed.l_state + lag];
        }
    }

    // lag_state[h * safe_max_order + l] = lag-l value for hydro h.
    let mut lag_state = vec![0.0_f64; n_hydros * safe_max_order];
    let mut lag_buf = vec![0.0_f64; safe_max_order];
    let mut lag_accum = vec![0.0_f64; n_hydros];
    let mut lag_weight_accum = vec![0.0_f64; n_hydros];
    // Per-stage raw rate, one entry per hydro; read by both the eta-solve and
    // the accumulate/spillover steps below so the lag-state advancement
    // mirrors the forward pass's accumulation of `z_inflow`.
    let mut raw_rate_buf = vec![0.0_f64; n_hydros];
    // advance_lag_chain (D1) reads the pre-shift lag values from a buffer
    // separate from the one it writes; this outer-scratch snapshot fills that
    // role since `lag_state` is shifted in place.
    let mut incoming_scratch = vec![0.0_f64; n_hydros * safe_max_order];
    let mut downstream_accumulator = if downstream_par_order > 0 {
        vec![0.0_f64; n_hydros]
    } else {
        Vec::new()
    };
    let mut downstream_completed_lags = if downstream_par_order > 0 {
        vec![0.0_f64; n_hydros * downstream_par_order]
    } else {
        Vec::new()
    };

    for outer in 0..outer_count {
        lag_state.copy_from_slice(&past_lag_buf);
        if seed.accum.is_empty() {
            lag_accum.fill(0.0);
            lag_weight_accum.fill(0.0);
        } else {
            lag_accum[..seed.accum.len()].copy_from_slice(seed.accum);
            lag_weight_accum[..seed.weight.len()].copy_from_slice(seed.weight);
        }
        downstream_accumulator.fill(0.0);
        downstream_completed_lags.fill(0.0);
        let mut downstream_weight_accum = 0.0_f64;
        let mut downstream_n_completed = 0_usize;

        for t in 0..n_stages {
            let stage_lag = resolve_stage_lag_transition(stage_lag_transitions, t);

            for h in 0..n_hydros {
                let raw_value = raw_target(t, outer, h);
                raw_rate_buf[h] = raw_value;

                // Full-length lag_buf so PAR(p)-A annual contributions (widened
                // across the `psi` slice) participate in the η inversion.
                for (l, slot) in lag_buf.iter_mut().enumerate() {
                    *slot = lag_state[h * safe_max_order + l];
                }

                let det_base = par.deterministic_base(t, h);
                let psi = par.psi_slice(t, h);
                let sigma = par.sigma(t, h);

                let eta = solve_par_noise(det_base, psi, &lag_buf, sigma, raw_value);

                write_eta(t, outer, h, eta);
            }

            incoming_scratch.copy_from_slice(&lag_state);
            let mut primary = PrimaryLagAccum {
                accumulator: &mut lag_accum,
                weight_accum: &mut lag_weight_accum,
            };
            let mut downstream = DownstreamLagAccum {
                accumulator: &mut downstream_accumulator,
                weight_accum: &mut downstream_weight_accum,
                completed_lags: &mut downstream_completed_lags,
                n_completed: &mut downstream_n_completed,
                par_order: downstream_par_order,
            };
            advance_lag_chain(
                EntityMajor {
                    entity_count: n_hydros,
                    max_order: safe_max_order,
                },
                &mut lag_state,
                &incoming_scratch,
                &raw_rate_buf[..n_hydros],
                &stage_lag,
                &mut primary,
                &mut downstream,
            );
        }
    }
}
