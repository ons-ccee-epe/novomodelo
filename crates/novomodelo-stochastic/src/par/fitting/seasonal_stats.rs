//! Seasonal mean/std estimation, history classification, and date-to-season
//! lookup primitives for PAR model fitting.

use std::collections::HashMap;

use chrono::NaiveDate;

use cobre_core::{
    EntityId,
    temporal::{SeasonMap, Stage},
};

use crate::StochasticError;

/// Build the season-aware stage index `(start_date, end_date, stage_id, season_id)`.
///
/// Only stages carrying a `season_id` are included; sorted by `start_date` so
/// [`find_season_for_date`] can binary-search it.
pub(super) fn build_stage_index(stages: &[Stage]) -> Vec<(NaiveDate, NaiveDate, i32, usize)> {
    let mut stage_index: Vec<(NaiveDate, NaiveDate, i32, usize)> = stages
        .iter()
        .filter_map(|s| s.season_id.map(|sid| (s.start_date, s.end_date, s.id, sid)))
        .collect();
    stage_index.sort_unstable_by_key(|(start, _, _, _)| *start);
    stage_index
}

/// Seasonal mean and standard deviation for one entity–season pair.
///
/// `stage_id` is the id (not index) of the **first** stage whose `season_id`
/// matches this row's season.
#[must_use]
#[derive(Debug, Clone, PartialEq)]
pub struct SeasonalStats {
    /// Entity (e.g., hydro plant) identifier.
    pub entity_id: EntityId,
    /// Id of the first stage belonging to this season.
    pub stage_id: i32,
    /// Sample mean of observed values (caller's unit, typically m³/s).
    pub mean: f64,
    /// Population-divisor (1/N) standard deviation — the Maceira-Damazio
    /// PAR(p)-A convention, not Bessel 1/(N−1).
    pub std: f64,
}

/// Classification of a per-(entity, season) historical observation series.
///
/// The degenerate classes force `std = 0`, which drives order 0 on both fitting
/// paths (PAR(p)-A via the structural-zero short-circuit, classical PAR(p) via
/// zeroed periodic autocorrelations), so a degenerate bucket cannot inject
/// spurious AR structure into adjacent months' PACFs.
#[must_use]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HistoryClass {
    /// No specific behaviour — run the standard fit.
    Default,
    /// Every observation is the same value (or all zero/null); forces order 0.
    /// Common for regulated/transposed-flow plants with structurally constant
    /// monthly inflow.
    Constant {
        /// The constant value to use as the seasonal mean.
        value: f64,
    },
    /// More than 10% of observations strictly negative — flags unphysical
    /// upstream-incremental construction. Diagnostic only: **does not override
    /// fitting**.
    ManyNegative {
        /// Empirical mean of the observation series (fallback constant; std 0).
        sample_mean: f64,
    },
    /// More than 50% of observations at the modal value (flow cap or low-flow
    /// constant); treated like `Constant`. No P99 guard — low caps (0.0, 1.0)
    /// classify as readily as high caps.
    Saturated {
        /// The cap value (modal value of the series).
        cap: f64,
    },
}

impl HistoryClass {
    /// The `(mean, std)` override replacing the empirical stats, or `None` when
    /// the class does not override fitting (`Default`, `ManyNegative`). The
    /// degenerate classes force `std = 0`.
    #[must_use]
    pub fn stats_override(self) -> Option<(f64, f64)> {
        match self {
            HistoryClass::Default | HistoryClass::ManyNegative { .. } => None,
            HistoryClass::Constant { value } => Some((value, 0.0)),
            HistoryClass::Saturated { cap } => Some((cap, 0.0)),
        }
    }

    /// Whether the class forces a degenerate (order-0) fit: `Constant` and
    /// `Saturated`, but not the diagnostic-only `ManyNegative`.
    #[must_use]
    pub fn is_degenerate(self) -> bool {
        matches!(
            self,
            HistoryClass::Constant { .. } | HistoryClass::Saturated { .. }
        )
    }
}

/// Classify a single (entity, season) observation series per the
/// [`HistoryClass`] taxonomy.
///
/// Priority order `Constant` → `ManyNegative` → `Saturated` → `Default`.
/// Mode counting rounds to the nearest integer (the historical inflow format
/// stores m³/s as integers); the constancy check tolerates `1e-6` to absorb the
/// parquet float round-trip. Empty and single-observation inputs return
/// `Constant`, so downstream fitters short-circuit predictably.
pub fn classify_history(observations: &[f64]) -> HistoryClass {
    if observations.is_empty() {
        return HistoryClass::Constant { value: 0.0 };
    }

    let first = observations[0];
    let const_tol = 1e-6;

    if observations.iter().all(|&v| (v - first).abs() < const_tol) {
        return HistoryClass::Constant { value: first };
    }

    let n = observations.len();
    let n_neg = observations.iter().filter(|&&v| v < 0.0).count();
    #[allow(clippy::cast_precision_loss)]
    if (n_neg as f64) / (n as f64) > 0.10 {
        #[allow(clippy::cast_precision_loss)]
        let sample_mean = observations.iter().sum::<f64>() / n as f64;
        return HistoryClass::ManyNegative { sample_mean };
    }

    #[allow(clippy::cast_possible_truncation)]
    let mut sorted: Vec<i64> = observations.iter().map(|v| v.round() as i64).collect();
    sorted.sort_unstable();
    let mut best_count = 0_usize;
    let mut best_value: i64 = sorted[0];
    let mut run = 1_usize;
    for i in 1..sorted.len() {
        if sorted[i] == sorted[i - 1] {
            run += 1;
        } else {
            if run > best_count {
                best_count = run;
                best_value = sorted[i - 1];
            }
            run = 1;
        }
    }
    if run > best_count {
        best_count = run;
        best_value = sorted[sorted.len() - 1];
    }
    #[allow(clippy::cast_precision_loss)]
    if (best_count as f64) / (n as f64) > 0.50 {
        return HistoryClass::Saturated {
            cap: best_value as f64,
        };
    }

    HistoryClass::Default
}

/// Estimate seasonal means and population-divisor (1/N) standard deviations from
/// historical observations, grouped by `(entity_id, season_id)`.
///
/// Only entities in `entity_ids` are processed; others are silently ignored.
/// Stages with `season_id = None` are skipped.
///
/// When `season_map` is `Some`, historical observation dates that fall outside
/// the study horizon are resolved to a season using the calendar-based cycle
/// definition. This allows PAR estimation from inflow history that predates
/// the study period.
///
/// # Parameters
///
/// - `observations` — `(entity_id, date, value)` triples, sorted by
///   `(entity_id, date)` (parser guarantee).
/// - `stages` — all stages in canonical index order; `end_date` is exclusive.
/// - `entity_ids` — canonical sorted list of entity IDs to estimate for.
/// - `season_map` — optional [`SeasonMap`] fallback for dates outside the study horizon.
///
/// # Errors
///
/// Returns [`StochasticError::InsufficientData`] when an observation date
/// cannot be mapped to any season, or when fewer than 2 observations exist
/// for any `(entity, season)` group.
///
/// # Examples
///
/// ```
/// use chrono::NaiveDate;
/// use cobre_core::{EntityId, temporal::{Stage, Block, BlockMode, StageStateConfig, StageRiskConfig, ScenarioSourceConfig, NoiseMethod}};
/// use cobre_stochastic::par::fitting::estimate_seasonal_stats_with_season_map;
///
/// fn stage(id: i32, y0: i32, m0: u32, y1: i32, m1: u32, season: usize) -> Stage {
///     Stage {
///         index: 0,
///         id,
///         start_date: NaiveDate::from_ymd_opt(y0, m0, 1).unwrap(),
///         end_date: NaiveDate::from_ymd_opt(y1, m1, 1).unwrap(),
///         season_id: Some(season),
///         blocks: vec![Block { index: 0, name: "S".to_string(), duration_hours: 744.0 }],
///         block_mode: BlockMode::Parallel,
///         state_config: StageStateConfig { storage: true, inflow_lags: false },
///         risk_config: StageRiskConfig::Expectation,
///         scenario_config: ScenarioSourceConfig { branching_factor: 1, noise_method: NoiseMethod::Saa },
///     }
/// }
///
/// let stages = vec![
///     stage(1, 2020, 1, 2020, 2, 0),
///     stage(2, 2020, 2, 2020, 3, 1),
/// ];
/// let obs = vec![
///     (EntityId::from(1), NaiveDate::from_ymd_opt(2020, 1, 15).unwrap(), 100.0),
///     (EntityId::from(1), NaiveDate::from_ymd_opt(2020, 1, 20).unwrap(), 200.0),
///     (EntityId::from(1), NaiveDate::from_ymd_opt(2020, 2, 10).unwrap(), 150.0),
///     (EntityId::from(1), NaiveDate::from_ymd_opt(2020, 2, 20).unwrap(), 250.0),
/// ];
/// let entity_ids = vec![EntityId::from(1)];
/// let stats = estimate_seasonal_stats_with_season_map(&obs, &stages, &entity_ids, None).unwrap();
/// assert_eq!(stats.len(), 2);
/// assert!((stats[0].mean - 150.0).abs() < 1e-10);
/// ```
pub fn estimate_seasonal_stats_with_season_map(
    observations: &[(EntityId, NaiveDate, f64)],
    stages: &[Stage],
    entity_ids: &[EntityId],
    season_map: Option<&SeasonMap>,
) -> Result<Vec<SeasonalStats>, StochasticError> {
    if observations.is_empty() {
        return Ok(Vec::new());
    }

    let entity_set: std::collections::HashSet<EntityId> = entity_ids.iter().copied().collect();

    let stage_index = build_stage_index(stages);

    let mut group_map: HashMap<(EntityId, usize), (Vec<f64>, i32)> = HashMap::new();

    // first_stage_id per season = stage.id of the lowest-start_date stage with
    // that season_id (stage_index is start_date-sorted).
    let mut season_first_stage: HashMap<usize, i32> = HashMap::new();
    for &(_, _, stage_id, season_id) in &stage_index {
        season_first_stage.entry(season_id).or_insert(stage_id);
    }

    for &(entity_id, date, value) in observations {
        if !entity_set.contains(&entity_id) {
            continue;
        }

        // Exact stage-date containment first; fall back to the SeasonMap for
        // observations predating the study horizon.
        let season_id = find_season_for_date(&stage_index, date)
            .or_else(|| season_map.and_then(|sm| sm.season_for_date(date)))
            .ok_or_else(|| StochasticError::InsufficientData {
                context: format!(
                    "observation date {date} for entity {entity_id} \
                     does not match any stage date range or season definition"
                ),
            })?;

        // Skip a resolved season with no study stage: its stats are never
        // consumed, and indexing it would panic for partial-year studies whose
        // history spans the full cycle.
        let Some(&first_stage_id) = season_first_stage.get(&season_id) else {
            continue;
        };
        let entry = group_map
            .entry((entity_id, season_id))
            .or_insert_with(|| (Vec::new(), first_stage_id));
        entry.0.push(value);
    }

    // 1/N population divisor, not Bessel 1/(N−1): required for self-consistent
    // conditional FACP values; a sample-vs-population scale factor would
    // otherwise propagate through every cross-correlation.
    let mut result: Vec<SeasonalStats> = Vec::with_capacity(group_map.len());
    for ((entity_id, _season_id), (values, stage_id)) in group_map {
        let n = values.len();
        if n < 2 {
            return Err(StochasticError::InsufficientData {
                context: format!(
                    "entity {entity_id} season mapped to stage {stage_id} \
                     has {n} observation(s); need at least 2 for std estimation"
                ),
            });
        }

        // Rationale: n (observation count) is far within f64's 2^53 exact range,
        // so the precision-loss lint cannot fire in practice.
        #[allow(clippy::cast_precision_loss)]
        let mean = values.iter().copied().sum::<f64>() / n as f64;
        #[allow(clippy::cast_precision_loss)]
        let variance = values.iter().map(|&v| (v - mean) * (v - mean)).sum::<f64>() / n as f64;
        let std = variance.sqrt();

        let (final_mean, final_std) = classify_history(&values)
            .stats_override()
            .unwrap_or((mean, std));

        result.push(SeasonalStats {
            entity_id,
            stage_id,
            mean: final_mean,
            std: final_std,
        });
    }

    // group_map iteration is non-deterministic; sort by (entity_id, stage_id) to
    // enforce declaration-order bit-determinism regardless of input ordering.
    result.sort_unstable_by_key(|s| (s.entity_id.0, s.stage_id));

    Ok(result)
}

/// Find the `season_id` for `date` by binary-searching `stage_index`.
///
/// `stage_index` must be sorted by `start_date`. Returns `None` when `date`
/// falls outside every stage's `[start_date, end_date)` range.
#[must_use]
pub fn find_season_for_date(
    stage_index: &[(NaiveDate, NaiveDate, i32, usize)],
    date: NaiveDate,
) -> Option<usize> {
    let pos = stage_index.partition_point(|(start, _, _, _)| *start <= date);
    if pos == 0 {
        return None;
    }
    let (_, end_date, _, season_id) = stage_index[pos - 1];
    if date < end_date {
        Some(season_id)
    } else {
        None
    }
}
