//! Per-season statistics of the rolling 12-month average for the PAR-A model.

use std::collections::{HashMap, HashSet};

use chrono::NaiveDate;

use cobre_core::{
    EntityId,
    temporal::{SeasonMap, Stage},
};

use super::seasonal_stats::{build_stage_index, find_season_for_date};
use crate::StochasticError;

// ---------------------------------------------------------------------------
// Annual seasonal stats helper (PAR-A eqs. 17, 18)
// ---------------------------------------------------------------------------

/// Per-season sample statistics of the rolling 12-month average, one entry per
/// `(entity, season)`.
///
/// `mean_m3s`/`std_m3s` are raw m³/s, **not** standardised; standardisation
/// happens in the cross-correlation helpers
/// ([`build_extended_periodic_yw_matrix`](super::build_extended_periodic_yw_matrix))
/// and the `ψ̂` conversion at `PrecomputedPar::build`.
#[must_use]
#[derive(Debug, Clone, PartialEq)]
pub struct AnnualSeasonalStats {
    /// Entity identifier.
    pub hydro_id: EntityId,
    /// 0-based season ID.
    pub season_id: usize,
    /// Sample mean of the rolling 12-month average.
    pub mean_m3s: f64,
    /// Population-divisor (`1/N`) standard deviation — the Maceira-Damazio
    /// PAR(p)-A convention, not the Bessel-corrected sample std.
    pub std_m3s: f64,
}

/// Estimate the per-season `(μ^A_m, σ^A_m)` of the rolling 12-month average
/// `A_t = (1/12) · Σ_{j=0..11} z[t-j]`, returning rows sorted by
/// `(hydro_id, season_id)`.
///
/// `σ^A_m` uses the population (`1/N`) divisor — required because the runtime
/// `ψ̂ = ψ · s_m / σ^A_m` needs `σ^A_m > 0` (enforced by the `cobre-io` output
/// validator) and self-consistent partitioned-covariance FACPs.
///
/// # Errors
///
/// Returns [`StochasticError::InsufficientData`] when any requested `entity_id`
/// has fewer than 13 observations (zero rolling-window `A_t` values); no silent
/// fallback to the classical PAR path is performed.
pub fn estimate_annual_seasonal_stats(
    observations: &[(EntityId, NaiveDate, f64)],
    stages: &[Stage],
    entity_ids: &[EntityId],
    season_map: Option<&SeasonMap>,
) -> Result<Vec<AnnualSeasonalStats>, StochasticError> {
    let stage_index = build_stage_index(stages);

    let entity_set: HashSet<EntityId> = entity_ids.iter().copied().collect();

    let mut entity_obs: HashMap<EntityId, Vec<(NaiveDate, f64)>> = HashMap::new();
    for &(entity_id, date, value) in observations {
        if entity_set.contains(&entity_id) {
            entity_obs.entry(entity_id).or_default().push((date, value));
        }
    }
    for obs_vec in entity_obs.values_mut() {
        obs_vec.sort_unstable_by_key(|(d, _)| *d);
    }

    for &entity_id in entity_ids {
        let n_obs = entity_obs.get(&entity_id).map_or(0, Vec::len);
        if n_obs < 13 {
            return Err(StochasticError::InsufficientData {
                context: format!(
                    "entity {entity_id} has {n_obs} observation(s); \
                     at least 13 are required to form one rolling 12-month average"
                ),
            });
        }
    }

    // Storage convention (callers depend on this): `A_{t-1} = mean(z[t-12..t-1])`
    // is keyed under the season of its own PDF time-index `t-1` — `group[i + 11]`
    // when `t = i + 12`. Yule-Walker callers
    // (`build_extended_periodic_yw_matrix`, `assemble_partitioned_covariance`)
    // retrieve it via `prev_season = (m - 1) mod n_seasons` for the equation at
    // current season `m`.
    let mut group_map: HashMap<(EntityId, usize), Vec<f64>> = HashMap::new();

    for &entity_id in entity_ids {
        let Some(group) = entity_obs.get(&entity_id) else {
            continue;
        };

        for i in 0..group.len().saturating_sub(12) {
            let target_date = group[i + 11].0;

            let Some(season_id) = find_season_for_date(&stage_index, target_date)
                .or_else(|| season_map.and_then(|sm| sm.season_for_date(target_date)))
            else {
                continue;
            };

            let mean_a: f64 = group[i..i + 12].iter().map(|(_, v)| v).sum::<f64>() / 12.0;
            group_map
                .entry((entity_id, season_id))
                .or_default()
                .push(mean_a);
        }
    }

    // Population (1/N) divisor, not Bessel (1/(N-1)): the sample-vs-population
    // scale factor would otherwise leak through every Z⊗A cross-correlation.
    let mut result: Vec<AnnualSeasonalStats> = Vec::with_capacity(group_map.len());
    for ((entity_id, season_id), values) in &group_map {
        let n = values.len();
        #[allow(clippy::cast_precision_loss)]
        let mean_m3s = values.iter().copied().sum::<f64>() / n as f64;
        #[allow(clippy::cast_precision_loss)]
        let var = values
            .iter()
            .map(|&v| (v - mean_m3s) * (v - mean_m3s))
            .sum::<f64>()
            / n as f64;
        let std_m3s = var.sqrt();

        result.push(AnnualSeasonalStats {
            hydro_id: *entity_id,
            season_id: *season_id,
            mean_m3s,
            std_m3s,
        });
    }

    // Canonical order, independent of HashMap traversal (declaration-order
    // determinism).
    result.sort_unstable_by_key(|s| (s.hydro_id.0, s.season_id));

    Ok(result)
}
