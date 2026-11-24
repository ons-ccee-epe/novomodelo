//! Layer 5b — scenario, penalty, and probability-data validation.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::path::Path;

use chrono::Datelike;
use novomodelo_core::scenario::SamplingScheme;
use novomodelo_core::{EntityId, Hydro, HydroGenerationModel, Stage};
use novomodelo_stochastic::par::{
    AnnualParams, ClosureRejection, check_stationarity, check_stationarity_annual,
};

use crate::config::HistoricalYearRange;
use crate::extensions::{FphaColumnLayout, FphaHyperplaneRow};
use crate::resolution::{PenaltiesEntitySlices, PenaltiesOverrides, resolve_penalties};
use crate::scenarios::estimation::estimation_samples;
use crate::{LoadError, StageIdResolver};

use super::super::{ValidationContext, rules, schema::ParsedData};
use super::envelope_tolerance;

// ── Rules 8-10: Penalty ordering ──────────────────────────────────────────────

/// Rules 8-10: checks the penalty hierarchy ordering across all hydros and buses.
///
/// Only costs that share a unit are compared. Deficit and
/// `generation_violation_below_cost` are $/`MWh`; the flow-violation and resource
/// costs are $/(m³/s·h). A $/hm³ storage cost or a $/(m³/s·h) flow cost is never
/// compared with a $/`MWh` cost, because the conversion needs plant productivity,
/// which this layer does not have; that is why rules 6 and 7 are retired. Rule 9
/// reads the directional evaporation and withdrawal costs because those are the
/// ones the LP prices; the symmetric pair is only their fallback.
///
/// Emits one warning per violated ordering check, aggregating
/// all violating entities into a single warning with the count and worst-case ID.
pub(super) fn check_penalty_ordering(data: &ParsedData, ctx: &mut ValidationContext) {
    let max_deficit_cost = max_bus_deficit_cost(data);

    {
        let mut violations: Vec<(i32, f64)> = Vec::new();
        for hydro in &data.hydros {
            let generation = hydro.penalties.generation_violation_below_cost;
            if generation >= max_deficit_cost {
                violations.push((hydro.id.0, generation));
            }
        }
        if let Some(worst) = violations
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        {
            let count = violations.len();
            ctx.emit(
                &rules::SEMANTIC_PENALTY_DEFICIT_NOT_ABOVE_GENERATION_VIOLATION,
                "penalties.json",
                None::<&str>,
                format!(
                    "Penalty ordering violation: max(deficit_segment_costs) \
                     ({max_deficit_cost}) should be > generation_violation_below_cost ({}) \
                     (both $/MWh) -- {count} hydro(s) affected, worst case: Hydro {}",
                    worst.1, worst.0
                ),
            );
        }
    }

    {
        let min_flow_cost = |h: &Hydro| {
            let p = &h.penalties;
            p.turbined_violation_below_cost
                .min(p.outflow_violation_below_cost)
                .min(p.outflow_violation_above_cost)
                .min(p.evaporation_violation_pos_cost)
                .min(p.evaporation_violation_neg_cost)
                .min(p.water_withdrawal_violation_pos_cost)
                .min(p.water_withdrawal_violation_neg_cost)
        };

        let max_resource_cost: f64 = data
            .hydros
            .iter()
            .map(|h| h.penalties.spillage_cost.max(h.penalties.diversion_cost))
            .fold(f64::NEG_INFINITY, f64::max)
            .max(0.0);

        let mut violations: Vec<(i32, f64)> = Vec::new();
        for hydro in &data.hydros {
            let min_flow = min_flow_cost(hydro);
            if min_flow <= max_resource_cost {
                violations.push((hydro.id.0, min_flow));
            }
        }
        if let Some(worst) = violations
            .iter()
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        {
            let count = violations.len();
            ctx.emit(
                &rules::SEMANTIC_PENALTY_FLOW_VIOLATION_NOT_ABOVE_RESOURCE,
                "penalties.json",
                None::<&str>,
                format!(
                    "Penalty ordering violation: min(flow_violation_costs) ({}) should be > \
                     max(resource_costs) ({max_resource_cost}) (both $/(m³/s·h)) -- {count} \
                     hydro(s) affected, worst case: Hydro {}",
                    worst.1, worst.0
                ),
            );
        }
    }

    {
        let mut violations: Vec<(i32, f64)> = Vec::new();
        for hydro in &data.hydros {
            let min_resource = hydro
                .penalties
                .spillage_cost
                .min(hydro.penalties.diversion_cost);
            if min_resource <= 0.0 {
                violations.push((hydro.id.0, min_resource));
            }
        }
        if let Some(worst) = violations
            .iter()
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        {
            let count = violations.len();
            ctx.emit(
                &rules::SEMANTIC_PENALTY_RESOURCE_COST_NOT_POSITIVE,
                "penalties.json",
                None::<&str>,
                format!(
                    "Penalty ordering violation: min(resource_costs) ({}) should be > 0 \
                     (regularization costs must be positive to prevent LP degeneracy) -- \
                     {count} hydro(s) affected, worst case: Hydro {}",
                    worst.1, worst.0
                ),
            );
        }
    }
}

fn max_bus_deficit_cost(data: &ParsedData) -> f64 {
    data.buses
        .iter()
        .flat_map(|b| b.deficit_segments.iter().map(|s| s.cost_per_mwh))
        .fold(f64::NEG_INFINITY, f64::max)
        .max(0.0)
}

// ── Rule 53: FPHA price dominance ─────────────────────────────────────────────

/// Rule 53: warns when an FPHA hydro's `fpha_violation_cost` is at or below
/// `max(deficit_segment_costs) + generation_violation_below_cost` (all $/`MWh`),
/// both prices resolved per stage through [`resolve_penalties`] so the stage
/// override tier counts.
///
/// Only study stages where the hydro's `hydro_production_models.json` entry
/// selects FPHA ([`uses_fpha_at`](crate::extensions::ProductionModelConfig::uses_fpha_at))
/// contribute; a hydro with no entry never does.
///
/// Every violating `(hydro, stage)` pair aggregates into one warning naming the
/// smallest-margin pair by `(margin, hydro id, stage id)`; the bridge's
/// `convert --validate` output relies on that single line.
pub(super) fn check_fpha_price_dominance(data: &ParsedData, ctx: &mut ValidationContext) {
    struct Violation {
        margin: f64,
        hydro_id: i32,
        stage_id: i32,
        p: f64,
        g: f64,
    }

    if data.production_models.is_empty() {
        return;
    }
    let study_stages: Vec<&Stage> = data.stages.stages.iter().filter(|s| s.id >= 0).collect();
    let study_stage_ids: Vec<i32> = study_stages.iter().map(|s| s.id).collect();
    let stage_resolver = StageIdResolver::from_study_stage_ids(&study_stage_ids);
    let resolved = resolve_penalties(
        &PenaltiesEntitySlices {
            hydros: &data.hydros,
            buses: &data.buses,
            lines: &data.lines,
            ncs_sources: &data.non_controllable_sources,
        },
        study_stages.len(),
        stage_resolver.index_map(),
        &PenaltiesOverrides {
            hydro: &data.penalty_overrides_hydro,
            bus: &data.penalty_overrides_bus,
            line: &data.penalty_overrides_line,
            ncs: &data.penalty_overrides_ncs,
        },
    );
    let d = max_bus_deficit_cost(data);

    let mut violations: Vec<Violation> = Vec::new();
    for (hi, hydro) in data.hydros.iter().enumerate() {
        let Some(config) = data
            .production_models
            .iter()
            .find(|c| c.hydro_id == hydro.id)
        else {
            continue;
        };
        for (ti, stage) in study_stages.iter().enumerate() {
            if !config.uses_fpha_at(stage) {
                continue;
            }
            let cell = resolved.hydro_penalties(hi, ti);
            let p = cell.fpha_violation_cost;
            let g = cell.generation_violation_below_cost;
            if p <= d + g {
                violations.push(Violation {
                    margin: p - (d + g),
                    hydro_id: hydro.id.0,
                    stage_id: stage.id,
                    p,
                    g,
                });
            }
        }
    }

    let Some(worst) = violations.iter().min_by(|a, b| {
        a.margin
            .total_cmp(&b.margin)
            .then(a.hydro_id.cmp(&b.hydro_id))
            .then(a.stage_id.cmp(&b.stage_id))
    }) else {
        return;
    };
    let n_pairs = violations.len();
    let n_hydros = violations
        .iter()
        .map(|v| v.hydro_id)
        .collect::<BTreeSet<_>>()
        .len();
    let Violation {
        hydro_id: id,
        stage_id,
        p,
        g,
        ..
    } = *worst;
    ctx.emit(
        &rules::SEMANTIC_FPHA_PRICE_NOT_ABOVE_DEFICIT_PLUS_GENERATION,
        "penalties.json",
        None::<&str>,
        format!(
            "FPHA price ordering violation: fpha_violation_cost ({p}) should be > \
             max(deficit_segment_costs) ({d}) + generation_violation_below_cost ({g}) \
             (all $/MWh) -- {n_pairs} (hydro, stage) pair(s) at FPHA stages, \
             {n_hydros} hydro(s) affected, smallest margin: Hydro {id} at stage {stage_id}"
        ),
    );
}

// ── Rule 55: precomputed FPHA planes at zero flow ─────────────────────────────

/// Rule 55: evaluates each precomputed FPHA plane the LP reads at a study stage at
/// `(v_min, q = 0, s = 0)`, with `v_min` the lower bound of the stage's fitting
/// window ([`FphaColumnLayout::resolve_fitting_window`]) and κ on γ₀ only, as the
/// LP scales it. One warning per hydro with a plane below 0 MW there names its
/// lowest.
pub(super) fn check_fpha_precomputed_plane_at_zero_turbining(
    data: &ParsedData,
    ctx: &mut ValidationContext,
) {
    struct Shortfall {
        power_mw: f64,
        stage_id: i32,
        plane_id: i32,
        v_min: f64,
    }

    if data.production_models.is_empty() || data.fpha_hyperplanes.is_empty() {
        return;
    }
    let mut planes: HashMap<(EntityId, Option<i32>), Vec<&FphaHyperplaneRow>> = HashMap::new();
    for row in &data.fpha_hyperplanes {
        planes
            .entry((row.hydro_id, row.stage_id))
            .or_default()
            .push(row);
    }

    for hydro in &data.hydros {
        let Some(config) = data
            .production_models
            .iter()
            .find(|c| c.hydro_id == hydro.id)
        else {
            continue;
        };
        if !config.uses_fpha() || config.uses_computed_fpha() || !hydro.has_turbine_capacity() {
            continue;
        }

        let mut shortfalls: Vec<Shortfall> = Vec::new();
        for stage in data.stages.stages.iter().filter(|s| s.id >= 0) {
            let res = config.resolve_stage(stage);
            if res.model != Some(HydroGenerationModel::Fpha) {
                continue;
            }
            let Ok((v_min, _)) = FphaColumnLayout::resolve_fitting_window(
                res.fpha_config,
                hydro.min_storage_hm3,
                hydro.max_storage_hm3,
            ) else {
                continue;
            };
            let Some(rows) = planes
                .get(&(hydro.id, Some(stage.id)))
                .or_else(|| planes.get(&(hydro.id, None)))
            else {
                continue;
            };
            for row in rows {
                let intercept = row.gamma_0 * row.kappa;
                let storage_term = row.gamma_v * v_min;
                let power_mw = intercept + storage_term;
                if power_mw < -envelope_tolerance(intercept.abs().max(storage_term.abs())) {
                    shortfalls.push(Shortfall {
                        power_mw,
                        stage_id: stage.id,
                        plane_id: row.plane_id,
                        v_min,
                    });
                }
            }
        }

        let Some(lowest) = shortfalls.iter().min_by(|x, y| {
            x.power_mw
                .total_cmp(&y.power_mw)
                .then(x.stage_id.cmp(&y.stage_id))
                .then(x.plane_id.cmp(&y.plane_id))
        }) else {
            continue;
        };
        let n_pairs = shortfalls.len();
        let stages = compress_stage_ids(&shortfalls.iter().map(|c| c.stage_id).collect());
        let Shortfall {
            power_mw,
            stage_id,
            plane_id,
            v_min,
        } = *lowest;
        let entity = format!("Hydro {}", hydro.id.0);
        let message = format!(
            "FPHA plane below zero at zero turbining: {entity} -- {n_pairs} (stage, plane) \
             pair(s) at stages {stages} fall below 0 MW with no turbined flow and no \
             spillage; the lowest reads {power_mw} MW (stage {stage_id}, plane {plane_id}, \
             storage {v_min} hm³), so the hydro turbines water at 0 MW, or reports \
             fpha_violation_mw where turbining cannot lift the plane to 0 MW"
        );
        ctx.emit(
            &rules::SEMANTIC_FPHA_PRECOMPUTED_PLANE_NEGATIVE_AT_ZERO_TURBINING,
            "system/fpha_hyperplanes.parquet",
            Some(&entity),
            message,
        );
    }
}

fn compress_stage_ids(ids: &BTreeSet<i32>) -> String {
    let mut runs: Vec<(i32, i32)> = Vec::new();
    for &id in ids {
        match runs.last_mut() {
            Some((_, end)) if end.checked_add(1) == Some(id) => *end = id,
            _ => runs.push((id, id)),
        }
    }
    runs.iter()
        .map(|&(start, end)| {
            if start == end {
                start.to_string()
            } else {
                format!("{start}-{end}")
            }
        })
        .collect::<Vec<_>>()
        .join(", ")
}

// ── Rule 11: FPHA penalty rule ─────────────────────────────────────────────────

/// Rule 11: checks that FPHA hydros have `turbined_cost >= 0`.
///
/// A zero cost is valid for constant-head plants (e.g., `gamma_v = 0`) where the
/// LP has no incentive to spill rather than turbine. Negative values are rejected
/// because they would make turbining artificially profitable and distort dispatch.
pub(super) fn check_fpha_penalty_rule(data: &ParsedData, ctx: &mut ValidationContext) {
    for hydro in &data.hydros {
        if hydro.generation_model != HydroGenerationModel::Fpha {
            continue;
        }
        let fpha_cost = hydro.penalties.turbined_cost;
        if fpha_cost < 0.0 {
            let entity_str = format!("Hydro {}", hydro.id.0);
            ctx.emit(
                &rules::SEMANTIC_FPHA_TURBINED_COST_NEGATIVE,
                "penalties.json",
                Some(&entity_str),
                format!(
                    "{entity_str}: turbined_cost ({fpha_cost}) must be non-negative (>= 0) \
                     for FPHA hydros; negative values distort LP dispatch"
                ),
            );
        }
    }
}

// ── Rule 12: Scenario model rules ───────────────────────────────────────────

/// Rule 12: validates inflow model standard deviation.
pub(super) fn check_scenario_models(data: &ParsedData, ctx: &mut ValidationContext) {
    // Rule 12: the parser rejects std_m3s < 0; this layer only warns on == 0.0
    // (valid but unusual deterministic inflow) -- suppressed when every applicable
    // scenario source resolves inflow to the External scheme, which never runs PAR
    // generation and for which the warning is meaningless.
    if inflow_scheme_is_external_everywhere(data) {
        return;
    }
    for row in &data.inflow_seasonal_stats {
        if row.std_m3s == 0.0 {
            ctx.emit(
                &rules::SEMANTIC_INFLOW_STD_ZERO,
                "scenarios/inflow_seasonal_stats.parquet",
                Some(format!("Hydro {}", row.hydro_id.0)),
                format!(
                    "Hydro {} stage {}: std_m3s is 0.0, indicating deterministic inflow \
                     (no stochastic component); verify this is intentional",
                    row.hydro_id.0, row.stage_id
                ),
            );
        }
    }
}

/// Resolves whether every scenario source that reads `inflow_seasonal_stats.parquet`
/// (training, and simulation when it declares its own `scenario_source`) uses the
/// `External` inflow scheme -- the same config source [`check_external_scheme_has_files`]
/// reads. A config-read failure keeps rule 12 live, since Layer 2 already guarantees
/// these reads succeed in practice.
fn inflow_scheme_is_external_everywhere(data: &ParsedData) -> bool {
    let Ok(training) = data
        .config
        .training_scenario_source(Path::new("config.json"))
    else {
        return false;
    };
    if training.inflow_scheme != SamplingScheme::External {
        return false;
    }
    if data.config.simulation.scenario_source.is_some() {
        let Ok(simulation) = data
            .config
            .simulation_scenario_source(Path::new("config.json"))
        else {
            return false;
        };
        if simulation.inflow_scheme != SamplingScheme::External {
            return false;
        }
    }
    true
}

// ── Rule 35: Hard stationarity gate on user-supplied AR coefficients ─────────

/// Rule 35: gates user-supplied `inflow_ar_coefficients.parquet` rows for stationarity
/// via the periodic-ACF closure (`novomodelo_stochastic::par::closure`).
///
/// Runs only when `data.inflow_ar_coefficients` is non-empty -- the
/// external-input path. The internal fitting/estimation path (triggered from
/// `inflow_history.parquet` when no coefficient rows are supplied) keeps its
/// own fallbacks and is not gated here.
///
/// For each hydro present in `inflow_ar_coefficients` (ascending `hydro_id`),
/// resolves seasons via [`crate::scenarios::resolve_stage_seasons`] --
/// including its no-`season_map` fallback (dense-ranked distinct per-stage
/// `season_id`s) -- then groups lag rows by season: one representative stage
/// per season, mirroring
/// [`crate::scenarios::populate_derived_residual_ratios`]'s grouping, since
/// stages sharing a season carry identical `ψ*`. Calls
/// [`check_stationarity_annual`] when any season carries an annual component
/// (`inflow_annual_components.parquet`), else [`check_stationarity`]. Every
/// [`ClosureRejection`] becomes an error naming the offending
/// season/lag and the failing quantity.
///
/// A hydro whose order-bearing coefficients reference a stage with no
/// resolvable season (no `season_map` AND no usable per-stage `season_id` --
/// the same condition under which
/// [`crate::scenarios::populate_derived_residual_ratios`] itself errors) gets
/// an error naming the unresolved stage(s), and is skipped
/// for the stationarity check (other hydros still run). Bare per-stage
/// `season_id`s with no `season_map` (the fallback) resolve cleanly and ARE
/// gated. Never silently skipped.
pub(super) fn check_par_stationarity(data: &ParsedData, ctx: &mut ValidationContext) {
    if data.inflow_ar_coefficients.is_empty() {
        return;
    }

    let (stage_to_season, n_seasons) = crate::scenarios::resolve_stage_seasons(
        &data.stages.stages,
        data.stages.policy_graph.season_map.as_ref(),
    );

    let seasonal_std_by_key: HashMap<(i32, i32), f64> = data
        .inflow_seasonal_stats
        .iter()
        .map(|row| ((row.hydro_id.0, row.stage_id), row.std_m3s))
        .collect();
    let annual_by_key: HashMap<(i32, i32), AnnualParams> = data
        .inflow_annual_components
        .iter()
        .map(|row| {
            (
                (row.hydro_id.0, row.stage_id),
                AnnualParams {
                    coefficient: row.annual_coefficient,
                    sigma_a: row.annual_std_m3s,
                },
            )
        })
        .collect();

    let mut psi_by_hydro_stage: BTreeMap<i32, BTreeMap<i32, Vec<f64>>> = BTreeMap::new();
    for row in &data.inflow_ar_coefficients {
        psi_by_hydro_stage
            .entry(row.hydro_id.0)
            .or_default()
            .entry(row.stage_id)
            .or_default()
            .push(row.coefficient);
    }

    for (hydro_id, stage_psi) in psi_by_hydro_stage {
        let unresolved: Vec<i32> = stage_psi
            .keys()
            .filter(|stage_id| !stage_to_season.contains_key(stage_id))
            .copied()
            .collect();
        if !unresolved.is_empty() {
            ctx.emit(
                &rules::SEMANTIC_AR_COEFFICIENT_SEASON_UNRESOLVED,
                "scenarios/inflow_ar_coefficients.parquet",
                Some(format!("Hydro {hydro_id}")),
                format!(
                    "Hydro {hydro_id}: PAR stationarity gate cannot resolve a season \
                     for stage(s) {unresolved:?} with order-bearing AR coefficients \
                     (no season_definitions and no per-stage season_id); add \
                     season_definitions to stages.json or a season_id on the \
                     affected stage(s)"
                ),
            );
            continue;
        }

        let mut orders = vec![0_usize; n_seasons];
        let mut psi_by_season = vec![Vec::new(); n_seasons];
        let mut seasonal_std = vec![0.0_f64; n_seasons];
        let mut annual: Vec<Option<AnnualParams>> = vec![None; n_seasons];
        let mut season_seen = vec![false; n_seasons];

        for (stage_id, psi) in stage_psi {
            let Some(&season) = stage_to_season.get(&stage_id) else {
                continue;
            };
            if season_seen[season] {
                continue;
            }
            season_seen[season] = true;
            orders[season] = psi.len();
            seasonal_std[season] = seasonal_std_by_key
                .get(&(hydro_id, stage_id))
                .copied()
                .unwrap_or(0.0);
            annual[season] = annual_by_key.get(&(hydro_id, stage_id)).copied();
            psi_by_season[season] = psi;
        }

        let has_annual = annual.iter().any(Option::is_some);
        let result = if has_annual {
            check_stationarity_annual(&psi_by_season, &orders, &annual, &seasonal_std, n_seasons)
        } else {
            check_stationarity(&psi_by_season, &orders, n_seasons)
        };

        if let Err(rejection) = result {
            let entity_str = format!("Hydro {hydro_id}");
            ctx.emit(
                &rules::SEMANTIC_AR_COEFFICIENT_NOT_STATIONARY,
                "scenarios/inflow_ar_coefficients.parquet",
                Some(&entity_str),
                describe_par_rejection(hydro_id, &rejection),
            );
        }
    }
}

/// Formats a [`ClosureRejection`] into a message naming the offending
/// season/lag and the failing quantity, for [`check_par_stationarity`].
fn describe_par_rejection(hydro_id: i32, rejection: &ClosureRejection) -> String {
    match rejection {
        ClosureRejection::SingularClosure => format!(
            "Hydro {hydro_id}: PAR stationarity gate found the periodic-ACF closure \
             singular for the AR coefficients in inflow_ar_coefficients.parquet"
        ),
        ClosureRejection::AutocorrelationOutOfRange { season, lag, rho } => format!(
            "Hydro {hydro_id} season {season}: PAR stationarity gate rejected \
             inflow_ar_coefficients.parquet -- implied autocorrelation ρ(lag {lag}) \
             = {rho} is outside [-1, 1]"
        ),
        ClosureRejection::NonPositiveResidualVariance { season, r_squared } => format!(
            "Hydro {hydro_id} season {season}: PAR stationarity gate rejected \
             inflow_ar_coefficients.parquet -- implied residual variance r² \
             = {r_squared} is at or below the numerical floor"
        ),
        ClosureRejection::NonStationaryMonodromy { spectral_radius } => format!(
            "Hydro {hydro_id}: PAR stationarity gate rejected \
             inflow_ar_coefficients.parquet -- periodic monodromy spectral radius \
             = {spectral_radius} is at or above 1"
        ),
    }
}

// ── External scheme requires external scenario files ─────────────────────────

/// Rule 44a: validates that when a class uses the `External` sampling scheme, the
/// corresponding external scenario file data is non-empty.
pub(super) fn check_external_scheme_has_files(data: &ParsedData, ctx: &mut ValidationContext) {
    // validate_config resolves both sources at load, so these reads cannot fail here.
    let Ok(training_source) = data
        .config
        .training_scenario_source(Path::new("config.json"))
    else {
        return;
    };
    let Ok(simulation_source) = data
        .config
        .simulation_scenario_source(Path::new("config.json"))
    else {
        return;
    };

    // Check simulation only when it defines its own scenario_source; otherwise it
    // falls back to training (already checked) and would duplicate errors.
    let sources: &[(&str, &_)] = if data.config.simulation.scenario_source.is_some() {
        &[
            ("training", &training_source),
            ("simulation", &simulation_source),
        ]
    } else {
        &[("training", &training_source)]
    };

    let mut check_external =
        |section: &str, scheme: SamplingScheme, class_name: &str, is_empty: bool| {
            if scheme == SamplingScheme::External && is_empty {
                ctx.emit(
                    &rules::SEMANTIC_EXTERNAL_SCHEME_WITHOUT_DATA,
                    "config.json",
                    Some(format!("{section}.scenario_source.{class_name}")),
                    format!(
                        "{class_name} class uses 'external' scheme but no \
                     external_{class_name}_scenarios.parquet data was found; \
                     external scheme requires corresponding scenario file"
                    ),
                );
            }
        };

    for (section, source) in sources {
        check_external(
            section,
            source.inflow_scheme,
            "inflow",
            data.external_scenarios.is_empty(),
        );
        check_external(
            section,
            source.load_scheme,
            "load",
            data.external_load_scenarios.is_empty(),
        );
        check_external(
            section,
            source.ncs_scheme,
            "ncs",
            data.external_ncs_scenarios.is_empty(),
        );
    }
}

// ── Rules 45-48: External-library coherence ──────────────────────────────────

/// Per-class extract of a slot-occupying external scenario file: the raw cell
/// values (keyed by resolved study index, `scenario_id`, and entity id), the
/// entity ids present, and the per-stage raw column count `raw_c(t)` (the
/// distinct `scenario_id` count at that stage — the canonical, cross-class-agreed
/// count [`check_node_graph`](super::stages::check_node_graph)'s pointer bound quantifies over).
struct ClassExternal {
    name: &'static str,
    file: &'static str,
    /// `(stage_idx, scenario_id, entity_id) -> value` over rows whose `stage_id`
    /// resolves; a repeated key is an A1 duplicate, caught on insert. Left
    /// empty when [`extract_class`]'s `populate_values` is `false`.
    cells: HashMap<(usize, i32, i32), f64>,
    entities: BTreeSet<i32>,
    raw_c: Vec<usize>,
}

/// Rules 45-48: external-library coherence across the slot-occupying external
/// classes of the training scenario source — the shared per-stage raw
/// column-count vector `raw_c(t)` (rule 45), the exact `scenario_id` set
/// per (class, stage) (rule 46), out-of-range `stage_id` rejection (rule
/// 47) and the prefix-coherence warning (rule 48). Rules 45-47
/// fire for every study; the prefix-coherence warning only when `nodes[]` is
/// declared. Reads raw parsed values only — the standardized-library width
/// assertion runs at study setup, where the standardized libraries exist.
pub(super) fn check_external_library_coherence(data: &ParsedData, ctx: &mut ValidationContext) {
    let Ok(source) = data
        .config
        .training_scenario_source(Path::new("config.json"))
    else {
        return;
    };

    let study_ids: Vec<i32> = data
        .stages
        .stages
        .iter()
        .filter(|s| s.id >= 0)
        .map(|s| s.id)
        .collect();
    let resolver = StageIdResolver::from_study_stage_ids(&study_ids);
    let n_stages = study_ids.len();

    // check_prefix_coherence (the sole reader of ClassExternal.cells' values)
    // runs only under a declared nodes[] graph; skip populating values otherwise.
    let policy_graph_declared = !data.stages.policy_graph.nodes.is_empty();

    let mut classes: Vec<ClassExternal> = Vec::new();
    if source.inflow_scheme == SamplingScheme::External && !data.external_scenarios.is_empty() {
        classes.push(extract_class(
            "inflow",
            "scenarios/external_inflow_scenarios.parquet",
            "ExternalScenarioRow",
            data.external_scenarios
                .iter()
                .map(|r| (r.hydro_id.0, r.stage_id, r.scenario_id, r.value_m3s)),
            &resolver,
            n_stages,
            policy_graph_declared,
            ctx,
        ));
    }
    if source.load_scheme == SamplingScheme::External && !data.external_load_scenarios.is_empty() {
        classes.push(extract_class(
            "load",
            "scenarios/external_load_scenarios.parquet",
            "ExternalLoadRow",
            data.external_load_scenarios
                .iter()
                .map(|r| (r.bus_id.0, r.stage_id, r.scenario_id, r.value_mw)),
            &resolver,
            n_stages,
            policy_graph_declared,
            ctx,
        ));
    }
    if source.ncs_scheme == SamplingScheme::External && !data.external_ncs_scenarios.is_empty() {
        classes.push(extract_class(
            "ncs",
            "scenarios/external_ncs_scenarios.parquet",
            "ExternalNcsRow",
            data.external_ncs_scenarios
                .iter()
                .map(|r| (r.ncs_id.0, r.stage_id, r.scenario_id, r.value)),
            &resolver,
            n_stages,
            policy_graph_declared,
            ctx,
        ));
    }

    check_raw_c_agreement(&classes, &study_ids, ctx);

    if policy_graph_declared {
        check_prefix_coherence(data, &classes, &resolver, ctx);
    }
}

/// Extract one external class, running A2 (`stage_id` resolution, rule 47) and
/// A1 (exact `scenario_id` set, rule 46) as it builds the [`ClassExternal`].
///
/// `populate_values`: when `false` (no declared `nodes[]`), the returned
/// [`ClassExternal::cells`] carries no values -- only [`check_prefix_coherence`]
/// reads them, and it does not run in that dialect. A1 duplicate detection
/// still runs, over a key-only set.
fn extract_class(
    name: &'static str,
    file: &'static str,
    row_label: &'static str,
    rows: impl Iterator<Item = (i32, i32, i32, f64)>,
    resolver: &StageIdResolver,
    n_stages: usize,
    populate_values: bool,
    ctx: &mut ValidationContext,
) -> ClassExternal {
    let mut cells: HashMap<(usize, i32, i32), f64> = HashMap::new();
    let mut seen_keys: HashSet<(usize, i32, i32)> = HashSet::new();
    let mut entities: BTreeSet<i32> = BTreeSet::new();
    let mut union_by_stage: Vec<BTreeSet<i32>> = vec![BTreeSet::new(); n_stages];
    let mut entity_scen: HashMap<(usize, i32), BTreeSet<i32>> = HashMap::new();

    for (i, (entity_id, stage_id, scenario_id, value)) in rows.enumerate() {
        // A2 (rule 47): an out-of-range stage_id is rejected through the shared
        // StageIdResolver constructor, never silently dropped.
        let Some(stage_idx) = resolver.resolve(stage_id) else {
            emit_resolver_error(
                ctx,
                resolver.unresolved_stage_id_error(
                    file,
                    format!("{row_label}[{i}].stage_id"),
                    stage_id,
                ),
            );
            continue;
        };
        let is_duplicate = if populate_values {
            cells
                .insert((stage_idx, scenario_id, entity_id), value)
                .is_some()
        } else {
            !seen_keys.insert((stage_idx, scenario_id, entity_id))
        };
        if is_duplicate {
            ctx.emit(
                &rules::SEMANTIC_EXTERNAL_SCENARIO_ID_SET,
                file,
                Some(format!("{row_label}[{i}]")),
                format!(
                    "{name} external library has a duplicate row at stage {stage_id} for \
                     scenario_id {scenario_id}, entity {entity_id}; each (entity, scenario_id) \
                     must appear exactly once per stage"
                ),
            );
        }
        entities.insert(entity_id);
        union_by_stage[stage_idx].insert(scenario_id);
        entity_scen
            .entry((stage_idx, entity_id))
            .or_default()
            .insert(scenario_id);
    }

    let mut raw_c = vec![0usize; n_stages];
    for (t, union) in union_by_stage.iter().enumerate() {
        let c = union.len();
        raw_c[t] = c;
        if c == 0 {
            continue;
        }
        let stage_id = resolver.id_at(t).unwrap_or_default();
        let c_i32 = i32::try_from(c).unwrap_or(i32::MAX);
        // A1 (rule 46): every entity present at this stage must carry exactly the
        // scenario_id set {0..raw_c-1} — a set check, not a bound check, so a
        // 1-based deck (which is in-bounds below raw_c but shifted) is rejected.
        for &e in &entities {
            let Some(es) = entity_scen.get(&(t, e)) else {
                continue;
            };
            let out_of_range: Vec<i32> = es
                .iter()
                .copied()
                .filter(|&s| s < 0 || s >= c_i32)
                .collect();
            let missing: Vec<i32> = (0..c_i32).filter(|m| !es.contains(m)).collect();
            if !out_of_range.is_empty() || !missing.is_empty() {
                ctx.emit(
                    &rules::SEMANTIC_EXTERNAL_SCENARIO_ID_SET,
                    file,
                    Some(format!("{name} entity {e} stage {stage_id}")),
                    format!(
                        "{name} external library at stage {stage_id}, entity {e}: scenario_id set \
                         must be exactly {{0..{}}} ({c} realizations); out-of-range {out_of_range:?}, \
                         missing {missing:?}",
                        c_i32 - 1
                    ),
                );
            }
        }
    }

    ClassExternal {
        name,
        file,
        cells,
        entities,
        raw_c,
    }
}

/// Feed [`StageIdResolver::unresolved_stage_id_error`] into the
/// validation context so A2 reports the identical message shape the
/// `noise_openings.parquet` resolver uses — one message shape, not two.
fn emit_resolver_error(ctx: &mut ValidationContext, err: LoadError) {
    if let LoadError::SchemaError {
        path,
        field,
        message,
    } = err
    {
        ctx.emit(
            &rules::SEMANTIC_EXTERNAL_STAGE_UNRESOLVED,
            path,
            Some(field),
            message,
        );
    }
}

/// P-B1 (rule 45): all slot-occupying external classes must agree on the
/// per-stage raw column-count vector `raw_c(t)`; a disagreement is a hard error
/// naming both classes, the stage and both counts — with or without `nodes[]`.
/// No element-wise-minimum reconciliation: a truncated scenario set is a wrong
/// answer, so the deck is rejected instead of repaired.
fn check_raw_c_agreement(
    classes: &[ClassExternal],
    study_ids: &[i32],
    ctx: &mut ValidationContext,
) {
    let Some((base, rest)) = classes.split_first() else {
        return;
    };
    for other in rest {
        for (t, (&bc, &oc)) in base.raw_c.iter().zip(other.raw_c.iter()).enumerate() {
            if bc != oc {
                let stage_id = study_ids.get(t).copied().unwrap_or_default();
                ctx.emit(
                    &rules::SEMANTIC_EXTERNAL_COLUMN_COUNT_DISAGREEMENT,
                    other.file,
                    Some(format!("stage {stage_id}")),
                    format!(
                        "external classes '{}' and '{}' disagree on the raw column count at \
                         stage {stage_id}: {bc} vs {oc}; all slot-occupying external classes must \
                         declare one shared realization axis",
                        base.name, other.name
                    ),
                );
            }
        }
    }
}

/// Prefix-coherence warning (rule 48): for every edge `n → m` and every
/// slot-occupying external class, the raw cells of columns `scenario_id(n)`
/// and `scenario_id(m)` must agree bitwise at every stage `s <= t(n)` — the
/// shared prefix along the root path. Disagreement warns (never rejects), naming
/// the edge, the class, the stage and both values; a bridge-shaped column pair
/// with identical trunk cells is silent.
fn check_prefix_coherence(
    data: &ParsedData,
    classes: &[ClassExternal],
    resolver: &StageIdResolver,
    ctx: &mut ValidationContext,
) {
    let graph = &data.stages.policy_graph;
    let mut node_info: HashMap<i32, (usize, Option<i32>)> = HashMap::new();
    for node in &graph.nodes {
        if let Some(idx) = resolver.resolve(node.stage_id) {
            node_info.insert(node.id, (idx, node.scenario_id));
        }
    }

    // Several edges can share the same (class, cn, cm) column pair (e.g. a
    // fan-in of source nodes onto one column); memoise the walk per pair so
    // it runs once for the deepest sn seen, not once per edge. Every edge
    // still gets its own warning, attributed only when the cached first
    // disagreement falls within that edge's own prefix depth.
    let mut walks: HashMap<(usize, i32, i32), PrefixWalk> = HashMap::new();

    for tr in &graph.transitions {
        let (Some(&(sn, Some(cn))), Some(&(_, Some(cm)))) =
            (node_info.get(&tr.source_id), node_info.get(&tr.target_id))
        else {
            continue;
        };
        for (class_idx, class) in classes.iter().enumerate() {
            let walk = walks.entry((class_idx, cn, cm)).or_default();
            walk.extend_to(class, cn, cm, sn);
            if let Some((s, e, va, vb)) = walk.disagreement.filter(|&(s, ..)| s <= sn) {
                let stage_id = resolver.id_at(s).unwrap_or_default();
                ctx.emit(
                    &rules::SEMANTIC_EXTERNAL_PREFIX_INCOHERENT,
                    class.file,
                    Some(format!("edge {}->{}", tr.source_id, tr.target_id)),
                    format!(
                        "prefix-coherence: external class '{}' columns {cn} and {cm} \
                         (edge {}->{}) disagree at stage {stage_id}, entity {e}: {va} vs {vb}; \
                         a node reproduces its pointed column's own history, not the mixed path",
                        class.name, tr.source_id, tr.target_id
                    ),
                );
            }
        }
    }
}

/// Incremental prefix walk for one `(class, cn, cm)` column pair, shared by
/// every edge that reaches it: extends forward from the last stage walked,
/// never re-scanning a stage whose disagreement outcome is already known.
#[derive(Default)]
struct PrefixWalk {
    /// Inclusive stage depth walked so far, or `None` before the first call.
    walked_through: Option<usize>,
    /// The first (lowest-stage) disagreement found within `0..=walked_through`.
    disagreement: Option<(usize, i32, f64, f64)>,
}

impl PrefixWalk {
    /// Extends the walk to cover `0..=sn` unless a disagreement is already
    /// cached (it is necessarily the first one, valid for any larger `sn`)
    /// or that depth was already walked. Stops at the first disagreement
    /// found.
    fn extend_to(&mut self, class: &ClassExternal, cn: i32, cm: i32, sn: usize) {
        if self.disagreement.is_some() {
            return;
        }
        let start = self.walked_through.map_or(0, |d| d + 1);
        if start > sn {
            return;
        }
        for s in start..=sn {
            let found = class.entities.iter().find_map(|&e| {
                match (class.cells.get(&(s, cn, e)), class.cells.get(&(s, cm, e))) {
                    (Some(&va), Some(&vb)) if va.to_bits() != vb.to_bits() => Some((s, e, va, vb)),
                    _ => None,
                }
            });
            if found.is_some() {
                self.disagreement = found;
                break;
            }
        }
        self.walked_through = Some(sn);
    }
}

// ── Rule 17: Load factor consistency ──────────────────────────────────────────

/// Validates cross-file consistency between `load_factors.json` and
/// `load_seasonal_stats.parquet`.
///
/// Rule 17: For every `LoadFactorEntry`, each `block_factors[j].block_id` must
/// match a `Block.index` in the corresponding stage's `blocks` array.
///
/// Silently skips when `data.load_factors` is empty.
pub(super) fn check_load_factor_consistency(data: &ParsedData, ctx: &mut ValidationContext) {
    if data.load_factors.is_empty() {
        return;
    }

    let stage_block_indices: HashMap<i32, HashSet<usize>> = data
        .stages
        .stages
        .iter()
        .filter(|s| s.id >= 0)
        .map(|s| {
            let indices: HashSet<usize> = s.blocks.iter().map(|b| b.index).collect();
            (s.id, indices)
        })
        .collect();

    for (i, entry) in data.load_factors.iter().enumerate() {
        let Some(valid_indices) = stage_block_indices.get(&entry.stage_id) else {
            continue;
        };
        for bf in &entry.block_factors {
            let block_idx = usize::try_from(bf.block_id).unwrap_or(usize::MAX);
            if !valid_indices.contains(&block_idx) {
                let mut sorted: Vec<usize> = valid_indices.iter().copied().collect();
                sorted.sort_unstable();
                ctx.emit(
                    &rules::SEMANTIC_LOAD_FACTOR_BLOCK_ID,
                    "scenarios/load_factors.json",
                    Some(format!("LoadFactorEntry[{i}]")),
                    format!(
                        "LoadFactorEntry[{i}] has block_id {} which is not in the block set \
                         {sorted:?} for stage {}",
                        bf.block_id, entry.stage_id
                    ),
                );
            }
        }
    }
}

// ── Rules 19-21: Estimation prerequisites ─────────────────────────────────────

/// Validates prerequisites for the history-based PAR(p) estimation path.
///
/// Runs only when `inflow_history.parquet` is present and
/// `inflow_seasonal_stats.parquet` is absent — i.e., when the estimation path
/// will be triggered (same condition used by the estimation pipeline).
///
/// Rule 19: `season_definitions` must be present in `stages.json` so that
/// observations can be grouped by season.
///
/// Rule 20: Each `(hydro_id, season_id)` group must have at least
/// `config.estimation.min_observations_per_season` of the samples estimation fits
/// from ([`estimation_samples`]).
///
/// Rule 21: Every hydro in the system must have at least one observation in
/// `inflow_history.parquet`; missing hydros cannot be estimated.
pub(super) fn check_estimation_prerequisites(data: &ParsedData, ctx: &mut ValidationContext) {
    if !super::season::estimation_active(data) {
        return;
    }

    if data.stages.policy_graph.season_map.is_none() {
        ctx.emit(
            &rules::SEMANTIC_ESTIMATION_WITHOUT_SEASON_DEFINITIONS,
            "scenarios/inflow_history.parquet",
            None::<&str>,
            "season_definitions is required in stages.json when estimating from \
             inflow_history.parquet; add a season_definitions section to stages.json",
        );
    }

    let hydro_ids_in_history: HashSet<i32> =
        data.inflow_history.iter().map(|r| r.hydro_id.0).collect();
    let mut missing_hydros: Vec<i32> = data
        .hydros
        .iter()
        .filter(|h| !hydro_ids_in_history.contains(&h.id.0))
        .map(|h| h.id.0)
        .collect();
    missing_hydros.sort_unstable();
    for id in missing_hydros {
        ctx.emit(
            &rules::SEMANTIC_ESTIMATION_HYDRO_WITHOUT_HISTORY,
            "scenarios/inflow_history.parquet",
            Some(format!("Hydro {id}")),
            format!(
                "hydro {id} has no observations in inflow_history.parquet but estimation \
                 is required; add historical inflow data for this hydro"
            ),
        );
    }

    // Skipped when season_map is None: Rule 19 already errored, and running here
    // would cascade a confusing second diagnostic.
    if let Some(season_map) = &data.stages.policy_graph.season_map {
        let min_obs = data.config.estimation.min_observations_per_season as usize;

        let stage_index = super::season::build_stage_index(data);
        // An unaggregatable history fails estimation itself, with its own error.
        let Ok((samples, _skipped_partial)) = estimation_samples(
            &data.inflow_history,
            &data.stages.stages,
            Some(season_map),
            data.config.estimation.history_years,
        ) else {
            return;
        };
        let mut counts: HashMap<(i32, usize), usize> = HashMap::new();
        for (hydro_id, start_date, _) in samples {
            if let Some(season_id) =
                super::season::estimation_season_for_date(&stage_index, season_map, start_date)
            {
                *counts.entry((hydro_id.0, season_id)).or_insert(0) += 1;
            }
        }

        let mut violations: Vec<(i32, usize, usize)> = counts
            .iter()
            .filter(|&(_, n)| *n < min_obs)
            .map(|(&(hid, sid), &n)| (hid, sid, n))
            .collect();
        // Sort for deterministic output order.
        violations.sort_unstable_by_key(|&(hid, sid, _)| (hid, sid));
        for (hid, sid, n) in violations {
            ctx.emit(
                &rules::SEMANTIC_ESTIMATION_FEW_OBSERVATIONS,
                "scenarios/inflow_history.parquet",
                Some(format!("Hydro {hid}")),
                format!(
                    "hydro {hid} season {sid} has {n} observations \
                     (minimum recommended: {min_obs}); estimation accuracy may be \
                     insufficient with so few observations"
                ),
            );
        }
    }
}

/// Refuses an `estimation.history_years` range reaching outside the start-date
/// years of `inflow_history.parquet`.
pub(super) fn check_estimation_history_years(data: &ParsedData, ctx: &mut ValidationContext) {
    let Some(HistoricalYearRange { from, to }) = data.config.estimation.history_years else {
        return;
    };
    if !super::season::estimation_active(data) {
        return;
    }
    let years = data.inflow_history.iter().map(|row| row.start_date.year());
    let (Some(first), Some(last)) = (years.clone().min(), years.max()) else {
        return;
    };
    if from < first || to > last {
        ctx.emit(
            &rules::SEMANTIC_ESTIMATION_HISTORY_YEARS_OUTSIDE_HISTORY,
            "config.json",
            None::<&str>,
            format!(
                "estimation.history_years ({from} to {to}) reaches outside the years of \
                 scenarios/inflow_history.parquet ({first} to {last}); narrow the range to \
                 those years"
            ),
        );
    }
}

// ── Filling-schedule sufficiency ──────────────────────────────────────────────

/// m³/s → hm³ per stage-hour: `3600 s/h ÷ 1e6 m³/hm³`. Duplicated from the
/// solver-side copy on purpose — a shared dependency would break novomodelo-io's
/// infrastructure-genericity rule; this is redundancy-with-purpose, not drift.
const M3S_TO_HM3: f64 = 3_600.0 / 1_000_000.0;

/// Rule 33: checks that each filling hydro's minimum accumulation schedule can reach its
/// dead volume before the entry stage:
///
/// ```text
/// Σ_{s ∈ [start_stage_id, entry)} ζ_s · rate_s  ≥  (min_storage_hm3 − seed)
/// ```
///
/// where `ζ_s = (Σ_b blocks[s].duration_hours) · M3S_TO_HM3` and `rate_s` is the
/// per-stage `filling_min_rate_m3s` override from `hydro_bounds` when present,
/// else the entity-level rate.
///
/// One-sided: only genuine under-provisioning is rejected
/// (`capacity < required - tolerance`, never float equality) — a
/// relative-with-floor tolerance (`envelope_tolerance`, the same idiom
/// `check_hydro_unit_groups` uses) absorbs the round-off `Σ ζ_s·rate_s`
/// accumulates against `min_storage_hm3 - seed`; surplus capacity merely
/// relaxes the earliest floors to slack, so a two-sided / exact-equality test
/// would reject valid schedules.
///
/// Stage ids index `data.stages.stages` by `Stage::id`, not array position: the
/// override key (`HydroBoundsRow.stage_id`) and `start_stage_id`/`entry_stage_id`
/// are all domain identifiers.
pub(super) fn check_filling_sufficiency(data: &ParsedData, ctx: &mut ValidationContext) {
    let zeta_by_stage: HashMap<i32, f64> = data
        .stages
        .stages
        .iter()
        .map(|s| (s.id, s.total_hours() * M3S_TO_HM3))
        .collect();

    let rate_override: HashMap<(i32, i32), f64> = data
        .hydro_bounds
        .iter()
        .filter_map(|row| {
            row.filling_min_rate_m3s
                .map(|rate| ((row.hydro_id.0, row.stage_id), rate))
        })
        .collect();

    let seed_by_hydro: HashMap<i32, f64> = data
        .initial_conditions
        .filling_storage
        .iter()
        .map(|s| (s.hydro_id.0, s.value_hm3))
        .collect();

    for hydro in &data.hydros {
        let (Some(filling), Some(entry)) = (hydro.filling.as_ref(), hydro.entry_stage_id) else {
            continue;
        };

        let mut capacity = 0.0;
        for stage_id in filling.start_stage_id..entry {
            let Some(&zeta) = zeta_by_stage.get(&stage_id) else {
                continue;
            };
            let rate = rate_override
                .get(&(hydro.id.0, stage_id))
                .copied()
                .unwrap_or(filling.filling_min_rate_m3s);
            capacity += zeta * rate;
        }

        let seed = seed_by_hydro.get(&hydro.id.0).copied().unwrap_or(0.0);
        let required = hydro.min_storage_hm3 - seed;
        let tolerance = envelope_tolerance(required);

        if capacity < required - tolerance {
            let entity_str = format!("Hydro {}", hydro.id.0);
            ctx.emit(
                &rules::SEMANTIC_FILLING_SCHEDULE_SHORT_OF_DEAD_VOLUME,
                "system/hydros.json",
                Some(&entity_str),
                format!(
                    "{entity_str}: filling schedule is insufficient to reach the dead volume \
                     before stage {entry}; cumulative minimum-rate capacity over stages \
                     [{}, {entry}) is {capacity} hm3 but {required} hm3 \
                     (min_storage {} - seed {seed}) is required",
                    filling.start_stage_id, hydro.min_storage_hm3
                ),
            );
        }
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::panic,
    clippy::too_many_lines,
    clippy::doc_markdown,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss
)]
mod tests {
    use super::super::validate_semantic_stages_penalties_scenarios;
    use super::M3S_TO_HM3;
    use crate::test_support::*;
    use crate::{
        config::HistoricalYearRange,
        constraints::HydroPenaltyOverrideRow,
        extensions::{
            FittingWindow, FphaColumnLayout, FphaHyperplaneRow, ProductionModelConfig,
            SeasonConfig, SelectionMode, StageRange,
        },
        scenarios::{
            BlockFactor, InflowAnnualComponentRow, InflowArCoefficientRow, InflowHistoryRow,
            InflowSeasonalStatsRow, LoadFactorEntry, LoadSeasonalStatsRow,
        },
        stages::StagesData,
        validation::{ErrorKind, ValidationContext, ValidationEntry, schema::ParsedData},
    };
    use novomodelo_core::{
        EntityId, HorizonGraph, Hydro,
        entities::HydroGenerationModel,
        scenario::{ExternalLoadRow, ExternalNcsRow, ExternalScenarioRow, NcsModel},
        temporal::{
            Block, Node, PolicyGraphType, SeasonCycleType, SeasonDefinition, SeasonMap, Transition,
        },
    };

    // ── Local helpers ─────────────────────────────────────────────────────────

    /// Build a `StagesData` with one stage that has one block (index = 0).
    fn make_stages_with_block(stage_id: i32) -> StagesData {
        let mut stage = make_stage(stage_id);
        stage.blocks = vec![Block {
            index: 0,
            name: "FLAT".to_string(),
            duration_hours: 744.0,
        }];
        StagesData {
            openings_declared: std::collections::HashSet::new(),
            stages: vec![stage],
            policy_graph: HorizonGraph {
                stage_discount_rate_overrides: std::collections::BTreeMap::new(),
                graph_type: PolicyGraphType::FiniteHorizon,
                annual_discount_rate: 0.06,
                transitions: vec![],
                nodes: Vec::new(),
                season_map: None,
            },
        }
    }

    // ── Rules 8-10: Penalty ordering ──────────────────────────────────────────

    /// A $/hm³ storage cost is never compared with a $/MWh deficit cost.
    #[test]
    fn test_5b_penalty_ordering_never_compares_storage_costs_with_deficit_costs() {
        let mut hydro = make_hydro_ordered_penalties(7);
        hydro.penalties.filling_target_violation_cost = 100.0;
        hydro.penalties.storage_violation_below_cost = 5.0;
        let data = make_data_5b(
            vec![hydro],
            make_stages_5b(vec![0]),
            vec![make_bus_with_deficit(1, 60.0)],
            vec![],
            vec![],
            None,
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(&data, &mut ctx);
        assert!(!ctx.has_errors());
        let all = ctx.warnings();
        let ordering: Vec<_> = all
            .iter()
            .filter(|w| w.message.starts_with("Penalty ordering violation:"))
            .collect();
        assert!(
            ordering.is_empty(),
            "no cross-unit penalty-ordering warning expected, got: {ordering:?}"
        );
    }

    /// Deficit costs are compared only with `generation_violation_below_cost`,
    /// the one other $/MWh cost.
    #[test]
    fn test_5b_penalty_ordering_compares_deficit_only_with_generation_violation_cost() {
        let rule8 = |ctx: &ValidationContext| -> Vec<String> {
            ctx.warnings()
                .iter()
                .filter(|w| {
                    w.message
                        .starts_with("Penalty ordering violation: max(deficit_segment_costs)")
                })
                .map(|w| w.message.clone())
                .collect()
        };

        let mut hydro = make_hydro_ordered_penalties(7);
        hydro.penalties.evaporation_violation_cost = 5000.0;
        hydro.penalties.evaporation_violation_pos_cost = 5000.0;
        hydro.penalties.evaporation_violation_neg_cost = 5000.0;
        let data = make_data_5b(
            vec![hydro],
            make_stages_5b(vec![0]),
            vec![make_bus_with_deficit(1, 60.0)],
            vec![],
            vec![],
            None,
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(&data, &mut ctx);
        assert!(
            rule8(&ctx).is_empty(),
            "a $/(m3/s.h) cost must not be compared with a deficit cost, got: {:?}",
            rule8(&ctx)
        );

        let mut hydro_a = make_hydro_ordered_penalties(3);
        hydro_a.penalties.generation_violation_below_cost = 2000.0;
        let mut hydro_b = make_hydro_ordered_penalties(9);
        hydro_b.penalties.generation_violation_below_cost = 3000.0;
        let data = make_data_5b(
            vec![hydro_a, hydro_b],
            make_stages_5b(vec![0]),
            vec![make_bus_with_deficit(1, 60.0)],
            vec![],
            vec![],
            None,
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(&data, &mut ctx);
        let warnings = rule8(&ctx);
        assert_eq!(
            warnings.len(),
            1,
            "exactly one rule-8 warning: {warnings:?}"
        );
        let msg = &warnings[0];
        for expected in [
            "generation_violation_below_cost (3000)",
            "(both $/MWh)",
            "2 hydro(s) affected",
            "worst case: Hydro 9",
        ] {
            assert!(msg.contains(expected), "missing {expected:?} in: {msg}");
        }
    }

    /// Resource costs are compared only with the $/(m³/s·h) flow-violation
    /// costs, so `generation_violation_below_cost` ($/MWh) is not among them.
    #[test]
    fn test_5b_penalty_ordering_compares_resource_costs_only_with_flow_violation_costs() {
        let rule9 = |ctx: &ValidationContext| -> Vec<String> {
            ctx.warnings()
                .iter()
                .filter(|w| {
                    w.message
                        .starts_with("Penalty ordering violation: min(flow_violation_costs)")
                })
                .map(|w| w.message.clone())
                .collect()
        };

        let mut hydro = make_hydro_ordered_penalties(7);
        hydro.penalties.generation_violation_below_cost = 0.5;
        let data = make_data_5b(
            vec![hydro],
            make_stages_5b(vec![0]),
            vec![make_bus_with_deficit(1, 60.0)],
            vec![],
            vec![],
            None,
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(&data, &mut ctx);
        assert!(
            rule9(&ctx).is_empty(),
            "a $/MWh cost must not be compared with a resource cost, got: {:?}",
            rule9(&ctx)
        );

        let mut hydro = make_hydro_ordered_penalties(4);
        hydro.penalties.turbined_violation_below_cost = 0.5;
        let data = make_data_5b(
            vec![hydro],
            make_stages_5b(vec![0]),
            vec![make_bus_with_deficit(1, 60.0)],
            vec![],
            vec![],
            None,
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(&data, &mut ctx);
        let warnings = rule9(&ctx);
        assert_eq!(
            warnings.len(),
            1,
            "exactly one rule-9 warning: {warnings:?}"
        );
        let msg = &warnings[0];
        for expected in [
            "min(flow_violation_costs) (0.5)",
            "max(resource_costs) (1)",
            "(both $/(m³/s·h))",
            "1 hydro(s) affected",
            "worst case: Hydro 4",
        ] {
            assert!(msg.contains(expected), "missing {expected:?} in: {msg}");
        }
    }

    /// A non-positive spillage or diversion cost is still flagged.
    #[test]
    fn test_5b_penalty_ordering_warns_on_non_positive_resource_cost() {
        let mut hydro = make_hydro_ordered_penalties(7);
        hydro.penalties.spillage_cost = 0.0;
        let data = make_data_5b(
            vec![hydro],
            make_stages_5b(vec![0]),
            vec![make_bus_with_deficit(1, 60.0)],
            vec![],
            vec![],
            None,
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(&data, &mut ctx);
        let all = ctx.warnings();
        let warnings: Vec<_> = all
            .iter()
            .filter(|w| {
                w.message
                    .starts_with("Penalty ordering violation: min(resource_costs)")
            })
            .collect();
        assert_eq!(warnings.len(), 1, "exactly one rule-10 warning");
        assert!(
            warnings[0]
                .message
                .contains("min(resource_costs) (0) should be > 0"),
            "got: {}",
            warnings[0].message
        );
    }

    /// The flow-violation costs are the directional evaporation and withdrawal
    /// costs the LP prices, not the symmetric costs that only fill them in.
    #[test]
    fn test_5b_penalty_ordering_compares_the_priced_directional_costs_not_their_symmetric_fallbacks()
     {
        let flow_warnings = |ctx: &ValidationContext| -> Vec<String> {
            ctx.warnings()
                .iter()
                .filter(|w| w.message.starts_with("Penalty ordering violation: min("))
                .map(|w| w.message.clone())
                .collect()
        };

        let hydros: Vec<Hydro> = (1..=4)
            .map(|id| {
                let mut h = make_hydro_ordered_penalties(id);
                h.penalties.evaporation_violation_cost = 5000.0;
                h.penalties.water_withdrawal_violation_cost = 5000.0;
                match id {
                    1 => h.penalties.water_withdrawal_violation_pos_cost = 0.5,
                    2 => h.penalties.water_withdrawal_violation_neg_cost = 0.6,
                    3 => h.penalties.evaporation_violation_pos_cost = 0.7,
                    _ => h.penalties.evaporation_violation_neg_cost = 0.8,
                }
                h
            })
            .collect();
        let data = make_data_5b(
            hydros,
            make_stages_5b(vec![0]),
            vec![make_bus_with_deficit(1, 60.0)],
            vec![],
            vec![],
            None,
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(&data, &mut ctx);
        let warnings = flow_warnings(&ctx);
        assert_eq!(warnings.len(), 1, "exactly one warning: {warnings:?}");
        let msg = &warnings[0];
        for expected in [
            "min(flow_violation_costs) (0.5)",
            "max(resource_costs) (1)",
            "4 hydro(s) affected",
            "worst case: Hydro 1",
        ] {
            assert!(msg.contains(expected), "missing {expected:?} in: {msg}");
        }

        let mut hydro = make_hydro_ordered_penalties(7);
        hydro.penalties.evaporation_violation_cost = 0.5;
        hydro.penalties.water_withdrawal_violation_cost = 0.5;
        let data = make_data_5b(
            vec![hydro],
            make_stages_5b(vec![0]),
            vec![make_bus_with_deficit(1, 60.0)],
            vec![],
            vec![],
            None,
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(&data, &mut ctx);
        assert!(
            flow_warnings(&ctx).is_empty(),
            "a symmetric fallback no LP column prices must not warn, got: {:?}",
            flow_warnings(&ctx)
        );
    }

    // ── Rule 11: FPHA penalty rule ────────────────────────────────────────────

    /// Hydro 3 with Fpha model, `turbined_cost = -0.01` produces a
    /// `BusinessRuleViolation` error with "Hydro 3" and "turbined_cost".
    #[test]
    fn test_5b_fpha_penalty_violated() {
        let mut hydro = make_hydro_ordered_penalties(3);
        hydro.generation_model = HydroGenerationModel::Fpha;
        hydro.penalties.turbined_cost = -0.01; // invalid: must be >= 0
        let data = make_data_5b(
            vec![hydro],
            make_stages_5b(vec![0]),
            vec![make_bus_with_deficit(1, 10.0)],
            vec![],
            vec![],
            None,
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(&data, &mut ctx);
        assert!(ctx.has_errors());
        let errors = ctx.errors();
        let relevant: Vec<_> = errors
            .iter()
            .filter(|e| e.kind == ErrorKind::BusinessRuleViolation)
            .collect();
        assert_eq!(
            relevant.len(),
            1,
            "exactly 1 BusinessRuleViolation expected"
        );
        let msg = &relevant[0].message;
        assert!(
            msg.contains("Hydro 3"),
            "message should contain 'Hydro 3', got: {msg}"
        );
        assert!(
            msg.contains("turbined_cost"),
            "message should contain 'turbined_cost', got: {msg}"
        );
    }

    /// FPHA hydro with `turbined_cost == 0.0` (constant-head) produces no error.
    #[test]
    fn test_5b_fpha_penalty_zero_valid() {
        let mut hydro = make_hydro_ordered_penalties(3);
        hydro.generation_model = HydroGenerationModel::Fpha;
        hydro.penalties.turbined_cost = 0.0; // valid: constant-head plant
        let data = make_data_5b(
            vec![hydro],
            make_stages_5b(vec![0]),
            vec![make_bus_with_deficit(1, 10.0)],
            vec![],
            vec![],
            None,
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(&data, &mut ctx);
        let errors: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| e.kind == ErrorKind::BusinessRuleViolation)
            .collect();
        assert!(
            errors.is_empty(),
            "turbined_cost == 0.0 should be valid for constant-head plants, \
             got: {errors:?}"
        );
    }

    /// FPHA hydro with `turbined_cost == spillage_cost` produces no error.
    #[test]
    fn test_5b_fpha_penalty_equal_spillage_valid() {
        let mut hydro = make_hydro_ordered_penalties(3);
        hydro.generation_model = HydroGenerationModel::Fpha;
        hydro.penalties.turbined_cost = 1.0;
        hydro.penalties.spillage_cost = 1.0; // turbined_cost == spillage_cost is valid
        let data = make_data_5b(
            vec![hydro],
            make_stages_5b(vec![0]),
            vec![make_bus_with_deficit(1, 10.0)],
            vec![],
            vec![],
            None,
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(&data, &mut ctx);
        let errors: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| e.kind == ErrorKind::BusinessRuleViolation)
            .collect();
        assert!(
            errors.is_empty(),
            "turbined_cost == spillage_cost should be valid, got: {errors:?}"
        );
    }

    /// FPHA hydro with `turbined_cost > spillage_cost` produces no error.
    #[test]
    fn test_5b_fpha_penalty_valid() {
        let mut hydro = make_hydro_ordered_penalties(4);
        hydro.generation_model = HydroGenerationModel::Fpha;
        hydro.penalties.turbined_cost = 2.0;
        hydro.penalties.spillage_cost = 1.0;
        let data = make_data_5b(
            vec![hydro],
            make_stages_5b(vec![0]),
            vec![make_bus_with_deficit(1, 10.0)],
            vec![],
            vec![],
            None,
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(&data, &mut ctx);
        let errors: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| e.kind == ErrorKind::BusinessRuleViolation)
            .collect();
        assert!(
            errors.is_empty(),
            "valid FPHA penalty ordering should produce no BusinessRuleViolation, \
             got: {errors:?}"
        );
    }

    // ── Rule 53: FPHA price dominance ─────────────────────────────────────────

    const FPHA_PRICE_PREFIX: &str = "FPHA price ordering violation:";

    fn fpha_price_warnings(data: &ParsedData) -> Vec<String> {
        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(data, &mut ctx);
        ctx.warnings()
            .iter()
            .filter(|w| w.message.starts_with(FPHA_PRICE_PREFIX))
            .map(|w| w.message.clone())
            .collect()
    }

    fn fpha_everywhere(hydro_id: i32) -> ProductionModelConfig {
        fpha_from(hydro_id, 0, None)
    }

    fn fpha_from(
        hydro_id: i32,
        start_stage_id: i32,
        fpha_config: Option<FphaColumnLayout>,
    ) -> ProductionModelConfig {
        ProductionModelConfig {
            hydro_id: EntityId::from(hydro_id),
            selection_mode: SelectionMode::StageRanges {
                ranges: vec![StageRange {
                    start_stage_id,
                    end_stage_id: None,
                    model: HydroGenerationModel::Fpha,
                    fpha_config,
                    reference_volume: None,
                    productivity_mw_per_m3s: None,
                }],
            },
        }
    }

    fn fpha_price_override(hydro_id: i32, stage_id: i32, price: f64) -> HydroPenaltyOverrideRow {
        HydroPenaltyOverrideRow {
            hydro_id: EntityId::from(hydro_id),
            stage_id,
            spillage_cost: None,
            turbined_cost: None,
            diversion_cost: None,
            storage_violation_below_cost: None,
            filling_target_violation_cost: None,
            turbined_violation_below_cost: None,
            outflow_violation_below_cost: None,
            outflow_violation_above_cost: None,
            generation_violation_below_cost: None,
            fpha_violation_cost: Some(price),
            evaporation_violation_cost: None,
            water_withdrawal_violation_cost: None,
            water_withdrawal_violation_pos_cost: None,
            water_withdrawal_violation_neg_cost: None,
            evaporation_violation_pos_cost: None,
            evaporation_violation_neg_cost: None,
            inflow_nonnegativity_cost: None,
        }
    }

    const DEFICIT_COST: f64 = 500.0;
    const GENERATION_VIOLATION_COST: f64 = 1000.0;
    const DEFICIT_PLUS_GENERATION: f64 = DEFICIT_COST + GENERATION_VIOLATION_COST;

    fn priced_fpha_hydro(id: i32, fpha_violation_cost: f64) -> Hydro {
        let mut hydro = make_hydro_ordered_penalties(id);
        hydro.penalties.generation_violation_below_cost = GENERATION_VIOLATION_COST;
        hydro.penalties.fpha_violation_cost = fpha_violation_cost;
        hydro
    }

    fn fpha_price_data(hydros: Vec<Hydro>, stage_ids: Vec<i32>) -> ParsedData {
        let production_models = hydros.iter().map(|h| fpha_everywhere(h.id.0)).collect();
        ParsedData {
            production_models,
            ..make_data_5b(
                hydros,
                make_stages_5b(stage_ids),
                vec![make_bus_with_deficit(1, DEFICIT_COST)],
                vec![],
                vec![],
                None,
            )
        }
    }

    #[test]
    fn fpha_price_at_sum_warns() {
        let data = fpha_price_data(vec![priced_fpha_hydro(7, DEFICIT_PLUS_GENERATION)], vec![0]);
        let warnings = fpha_price_warnings(&data);
        assert_eq!(warnings.len(), 1, "exactly one warning: {warnings:?}");
        let msg = &warnings[0];
        for expected in [
            "fpha_violation_cost (1500)",
            "max(deficit_segment_costs) (500)",
            "generation_violation_below_cost (1000)",
            "1 (hydro, stage) pair(s) at FPHA stages",
            "1 hydro(s) affected",
            "smallest margin: Hydro 7 at stage 0",
        ] {
            assert!(msg.contains(expected), "missing {expected:?} in: {msg}");
        }
    }

    #[test]
    fn fpha_price_below_sum_warns() {
        let data = fpha_price_data(
            vec![priced_fpha_hydro(7, DEFICIT_PLUS_GENERATION - 1.0)],
            vec![0],
        );
        let warnings = fpha_price_warnings(&data);
        assert_eq!(warnings.len(), 1, "exactly one warning: {warnings:?}");
        assert!(
            warnings[0].contains("fpha_violation_cost (1499)"),
            "got: {}",
            warnings[0]
        );
    }

    #[test]
    fn fpha_price_above_sum_is_silent() {
        let data = fpha_price_data(
            vec![priced_fpha_hydro(7, DEFICIT_PLUS_GENERATION + 1.0)],
            vec![0],
        );
        let warnings = fpha_price_warnings(&data);
        assert!(
            warnings.is_empty(),
            "no warning expected, got: {warnings:?}"
        );
    }

    #[test]
    fn fpha_price_rule_counts_only_fpha_stages() {
        let mut data = fpha_price_data(
            vec![priced_fpha_hydro(7, DEFICIT_PLUS_GENERATION - 1.0)],
            vec![0, 1],
        );
        data.stages.stages[0].season_id = Some(1);
        data.stages.stages[1].season_id = Some(2);
        data.production_models = vec![ProductionModelConfig {
            hydro_id: EntityId::from(7),
            selection_mode: SelectionMode::Seasonal {
                default_model: HydroGenerationModel::Fpha,
                seasons: vec![SeasonConfig {
                    season_id: 1,
                    model: HydroGenerationModel::ConstantProductivity,
                    fpha_config: None,
                    reference_volume: None,
                    productivity_mw_per_m3s: Some(1.0),
                }],
            },
        }];
        let warnings = fpha_price_warnings(&data);
        assert_eq!(warnings.len(), 1, "exactly one warning: {warnings:?}");
        let msg = &warnings[0];
        for expected in ["1 (hydro, stage) pair(s)", "at stage 1"] {
            assert!(msg.contains(expected), "missing {expected:?} in: {msg}");
        }
    }

    #[test]
    fn fpha_price_rule_reads_the_stage_override_tier() {
        let mut data = fpha_price_data(
            vec![priced_fpha_hydro(7, DEFICIT_PLUS_GENERATION + 1.0)],
            vec![0, 1],
        );
        data.penalty_overrides_hydro = vec![fpha_price_override(7, 1, 1200.0)];
        let warnings = fpha_price_warnings(&data);
        assert_eq!(warnings.len(), 1, "exactly one warning: {warnings:?}");
        let msg = &warnings[0];
        for expected in [
            "fpha_violation_cost (1200)",
            "1 (hydro, stage) pair(s)",
            "smallest margin: Hydro 7 at stage 1",
        ] {
            assert!(msg.contains(expected), "missing {expected:?} in: {msg}");
        }
    }

    #[test]
    fn fpha_price_rule_aggregates_into_one_warning() {
        let hydros = vec![
            priced_fpha_hydro(3, DEFICIT_PLUS_GENERATION - 100.0),
            priced_fpha_hydro(1, DEFICIT_PLUS_GENERATION - 500.0),
            priced_fpha_hydro(2, DEFICIT_PLUS_GENERATION - 500.0),
        ];
        let message_for = |hydros: Vec<Hydro>| -> String {
            let mut data = fpha_price_data(hydros, vec![0, 1]);
            data.penalty_overrides_hydro = vec![fpha_price_override(3, 1, 1300.0)];
            let warnings = fpha_price_warnings(&data);
            assert_eq!(warnings.len(), 1, "exactly one warning: {warnings:?}");
            warnings.into_iter().next().unwrap()
        };

        let given = message_for(hydros.clone());
        let reversed = message_for(hydros.into_iter().rev().collect());

        for expected in [
            "6 (hydro, stage) pair(s)",
            "3 hydro(s) affected",
            "smallest margin: Hydro 1 at stage 0",
        ] {
            assert!(given.contains(expected), "missing {expected:?} in: {given}");
        }
        assert_eq!(given, reversed);
    }

    // ── Rule 55: precomputed FPHA planes at zero flow ─────────────────────────

    const ZERO_TURBINING_PREFIX: &str = "FPHA plane below zero at zero turbining:";

    fn zero_turbining_warnings(data: &ParsedData) -> Vec<ValidationEntry> {
        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(data, &mut ctx);
        ctx.warnings()
            .into_iter()
            .filter(|w| w.message.starts_with(ZERO_TURBINING_PREFIX))
            .cloned()
            .collect()
    }

    fn spill_row(
        hydro_id: i32,
        stage_id: Option<i32>,
        plane_id: i32,
        gamma_s: f64,
    ) -> FphaHyperplaneRow {
        FphaHyperplaneRow {
            gamma_s,
            ..make_fpha_row(hydro_id, stage_id, plane_id)
        }
    }

    fn intercept_row(
        hydro_id: i32,
        stage_id: Option<i32>,
        plane_id: i32,
        gamma_0: f64,
    ) -> FphaHyperplaneRow {
        FphaHyperplaneRow {
            gamma_0,
            ..make_fpha_row(hydro_id, stage_id, plane_id)
        }
    }

    fn precomputed_layout() -> FphaColumnLayout {
        FphaColumnLayout {
            source: "precomputed".to_string(),
            volume_discretization_points: None,
            turbine_discretization_points: None,
            fitting_window: None,
            spillage_fit_max_m3s: None,
            turbined_fit_max_m3s: None,
        }
    }

    fn zero_turbining_data(
        hydros: Vec<Hydro>,
        stage_ids: Vec<i32>,
        production_models: Vec<ProductionModelConfig>,
        fpha_hyperplanes: Vec<FphaHyperplaneRow>,
    ) -> ParsedData {
        ParsedData {
            production_models,
            fpha_hyperplanes,
            ..make_data_5b(
                hydros,
                make_stages_5b(stage_ids),
                vec![make_bus_with_deficit(1, DEFICIT_COST)],
                vec![],
                vec![],
                None,
            )
        }
    }

    /// derived: at v_min 0 and s = 0 the planes read 100, 100, 0 and 2 · 50 = 100,
    /// κ scaling γ₀ only; at 2 000 m³/s of spillage, twice `max_turbined_m3s`,
    /// planes 0, 2 and 3 read 100 − 0.125 · 2 000 = −150, −0.5 · 2 000 = −1 000 and
    /// 100 − 0.25 · 2 000 = −400.
    #[test]
    fn precomputed_plane_below_zero_only_while_spilling_is_silent() {
        let rows = vec![
            spill_row(1, None, 0, -0.125),
            make_fpha_row(1, None, 1),
            FphaHyperplaneRow {
                gamma_0: 0.0,
                gamma_v: 0.0,
                ..spill_row(1, None, 2, -0.5)
            },
            FphaHyperplaneRow {
                gamma_0: 50.0,
                gamma_v: 0.0,
                kappa: 2.0,
                ..spill_row(1, None, 3, -0.25)
            },
        ];
        let data = zero_turbining_data(
            vec![make_hydro(1, None)],
            vec![0],
            vec![fpha_everywhere(1)],
            rows,
        );

        let warnings = zero_turbining_warnings(&data);

        assert!(
            warnings.is_empty(),
            "no warning expected, got: {warnings:?}"
        );
    }

    /// derived: at v_min 0 and s = 0 hydro 1's plane 1 reads γ₀ = −10 at stages 0-3,
    /// `make_fpha_row`'s 100 at stage 4 and −40 at stage 5, the lowest. Hydro 2's
    /// global plane reads 2 · −5 = −10 at all six stages, so stage and plane ids
    /// break the tie.
    #[test]
    fn precomputed_plane_below_zero_at_zero_flow_warns_once_per_hydro() {
        let mut rows = Vec::new();
        for s in 0..=5 {
            rows.push(make_fpha_row(1, Some(s), 0));
            rows.push(match s {
                4 => make_fpha_row(1, Some(4), 1),
                5 => intercept_row(1, Some(5), 1, -40.0),
                _ => intercept_row(1, Some(s), 1, -10.0),
            });
        }
        rows.push(FphaHyperplaneRow {
            kappa: 2.0,
            ..intercept_row(2, None, 0, -5.0)
        });
        let data_for = |rows: Vec<FphaHyperplaneRow>| {
            zero_turbining_data(
                vec![make_hydro(1, None), make_hydro(2, None)],
                (0..=5).collect(),
                vec![fpha_everywhere(1), fpha_everywhere(2)],
                rows,
            )
        };

        let warnings = zero_turbining_warnings(&data_for(rows.clone()));
        let reversed = zero_turbining_warnings(&data_for(rows.into_iter().rev().collect()));

        assert_eq!(warnings.len(), 2, "one warning per hydro: {warnings:?}");
        for (entry, entity, expected) in [
            (
                &warnings[0],
                "Hydro 1",
                [
                    "Hydro 1 -- 5 (stage, plane) pair(s) at stages 0-3, 5",
                    "the lowest reads -40 MW (stage 5, plane 1, storage 0 hm³)",
                ],
            ),
            (
                &warnings[1],
                "Hydro 2",
                [
                    "Hydro 2 -- 6 (stage, plane) pair(s) at stages 0-5",
                    "the lowest reads -10 MW (stage 0, plane 0,",
                ],
            ),
        ] {
            assert_eq!(
                entry.file,
                std::path::PathBuf::from("system/fpha_hyperplanes.parquet")
            );
            assert_eq!(entry.entity.as_deref(), Some(entity));
            for needle in expected {
                assert!(
                    entry.message.contains(needle),
                    "missing {needle:?} in: {}",
                    entry.message
                );
            }
        }
        let messages = |entries: &[ValidationEntry]| -> Vec<String> {
            entries.iter().map(|e| e.message.clone()).collect()
        };
        assert_eq!(messages(&warnings), messages(&reversed));
    }

    #[test]
    fn precomputed_plane_warning_names_both_outcomes() {
        let data = zero_turbining_data(
            vec![make_hydro(1, None)],
            vec![0],
            vec![fpha_everywhere(1)],
            vec![intercept_row(1, None, 0, -10.0)],
        );

        let warnings = zero_turbining_warnings(&data);

        assert_eq!(warnings.len(), 1, "exactly one warning: {warnings:?}");
        let msg = &warnings[0].message;
        assert!(
            msg.ends_with(
                "so the hydro turbines water at 0 MW, or reports fpha_violation_mw where \
                 turbining cannot lift the plane to 0 MW"
            ),
            "got: {msg}"
        );
    }

    #[test]
    fn precomputed_plane_check_reads_stage_rows_before_global_rows_at_fpha_stages() {
        let data = zero_turbining_data(
            vec![make_hydro(1, None)],
            vec![0, 1, 2],
            vec![fpha_from(1, 1, None)],
            vec![
                intercept_row(1, None, 0, -10.0),
                make_fpha_row(1, Some(2), 0),
            ],
        );

        let warnings = zero_turbining_warnings(&data);

        assert_eq!(warnings.len(), 1, "exactly one warning: {warnings:?}");
        let msg = &warnings[0].message;
        assert!(
            msg.contains("1 (stage, plane) pair(s) at stages 1 "),
            "got: {msg}"
        );
    }

    /// derived: with γᵥ 0.5 the plane reads −150 + 0.5 · 400 = 50 at v_min 400 and
    /// −150 + 0.5 · 200 = −50 at v_min 200.
    #[test]
    fn precomputed_plane_check_reads_v_min_from_the_fitting_window() {
        let window_from = |volume_min_hm3: f64| FphaColumnLayout {
            fitting_window: Some(FittingWindow {
                volume_min_hm3: Some(volume_min_hm3),
                volume_max_hm3: None,
                volume_min_percentile: None,
                volume_max_percentile: None,
            }),
            ..precomputed_layout()
        };
        let warnings_with = |layout: Option<FphaColumnLayout>| {
            zero_turbining_warnings(&zero_turbining_data(
                vec![make_hydro(1, None)],
                vec![0],
                vec![fpha_from(1, 0, layout)],
                vec![intercept_row(1, None, 0, -150.0)],
            ))
        };

        let high = warnings_with(Some(window_from(400.0)));
        let low = warnings_with(Some(window_from(200.0)));
        let unwindowed = warnings_with(None);

        assert!(high.is_empty(), "no warning expected, got: {high:?}");
        assert_eq!(low.len(), 1, "exactly one warning: {low:?}");
        assert!(
            low[0]
                .message
                .contains("reads -50 MW (stage 0, plane 0, storage 200 hm³)"),
            "got: {}",
            low[0].message
        );
        assert_eq!(unwindowed.len(), 1, "exactly one warning: {unwindowed:?}");
        assert!(
            unwindowed[0].message.contains("storage 0 hm³"),
            "got: {}",
            unwindowed[0].message
        );
    }

    #[test]
    fn precomputed_plane_check_skips_a_hydro_with_a_computed_entry() {
        let config_with_first_source = |source: &str| ProductionModelConfig {
            hydro_id: EntityId::from(1),
            selection_mode: SelectionMode::StageRanges {
                ranges: vec![
                    StageRange {
                        start_stage_id: 0,
                        end_stage_id: Some(0),
                        model: HydroGenerationModel::Fpha,
                        fpha_config: Some(FphaColumnLayout {
                            source: source.to_string(),
                            ..precomputed_layout()
                        }),
                        reference_volume: None,
                        productivity_mw_per_m3s: None,
                    },
                    StageRange {
                        start_stage_id: 1,
                        end_stage_id: None,
                        model: HydroGenerationModel::Fpha,
                        fpha_config: Some(precomputed_layout()),
                        reference_volume: None,
                        productivity_mw_per_m3s: None,
                    },
                ],
            },
        };
        let warnings_with = |source: &str| {
            zero_turbining_warnings(&zero_turbining_data(
                vec![make_hydro(1, None)],
                vec![0, 1],
                vec![config_with_first_source(source)],
                vec![intercept_row(1, None, 0, -10.0)],
            ))
        };

        let computed = warnings_with("computed");
        let precomputed = warnings_with("precomputed");

        assert!(
            computed.is_empty(),
            "no warning expected, got: {computed:?}"
        );
        assert_eq!(precomputed.len(), 1, "exactly one warning: {precomputed:?}");
    }

    #[test]
    fn precomputed_plane_check_skips_a_hydro_without_turbine_capacity() {
        let warnings_for = |hydro: Hydro| {
            zero_turbining_warnings(&zero_turbining_data(
                vec![hydro],
                vec![0],
                vec![fpha_everywhere(1)],
                vec![intercept_row(1, None, 0, -10.0)],
            ))
        };
        let mut dormant = make_hydro(1, None);
        dormant.max_generation_mw = 0.0;

        let dormant = warnings_for(dormant);
        let capable = warnings_for(make_hydro(1, None));

        assert!(dormant.is_empty(), "no warning expected, got: {dormant:?}");
        assert_eq!(capable.len(), 1, "exactly one warning: {capable:?}");
    }

    // ── Rule 12: Inflow std_m3s = 0.0 warning ────────────────────────────────

    /// `std_m3s = 0.0` produces a `ModelQuality` warning (deterministic inflow).
    #[test]
    fn test_5b_inflow_std_zero_warning() {
        let stats = vec![InflowSeasonalStatsRow {
            hydro_id: EntityId::from(1),
            stage_id: 0,
            mean_m3s: 100.0,
            std_m3s: 0.0, // triggers ModelQuality warning
        }];
        let data = make_data_5b(
            vec![make_hydro_ordered_penalties(1)],
            make_stages_5b(vec![0]),
            vec![make_bus_with_deficit(1, 10.0)],
            stats,
            vec![],
            None,
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(&data, &mut ctx);
        assert!(
            !ctx.has_errors(),
            "std_m3s=0.0 should produce warning, not error, got: {:?}",
            ctx.errors()
        );
        let warnings = ctx.warnings();
        assert!(
            !warnings.is_empty(),
            "std_m3s=0.0 should produce at least 1 ModelQuality warning"
        );
        assert!(
            warnings.iter().any(|w| w.kind == ErrorKind::ModelQuality),
            "should have ModelQuality warning"
        );
    }

    // ── Rule 35: PAR stationarity gate ────────────────────────────────────────

    /// Build a `SeasonMap` with exactly `n` seasons, ids `0..n`.
    fn season_map_n(n: usize) -> SeasonMap {
        SeasonMap {
            cycle_type: SeasonCycleType::Monthly,
            seasons: (0..n)
                .map(|i| SeasonDefinition {
                    id: i,
                    label: format!("Season{i}"),
                    month_start: (i % 12 + 1) as u32,
                    day_start: None,
                    month_end: None,
                    day_end: None,
                })
                .collect(),
        }
    }

    /// Build a `StagesData` with `n` stages (ids `0..n`), each pinned 1:1 to
    /// its own season (`stage.id == season_id`), and a matching `n`-season
    /// `SeasonMap`.
    fn stages_one_stage_per_season(n: usize) -> StagesData {
        let stages = (0..n as i32)
            .map(|id| {
                let mut stage = make_stage(id);
                stage.season_id = Some(id as usize);
                stage
            })
            .collect();
        StagesData {
            openings_declared: std::collections::HashSet::new(),
            stages,
            policy_graph: HorizonGraph {
                stage_discount_rate_overrides: std::collections::BTreeMap::new(),
                graph_type: PolicyGraphType::FiniteHorizon,
                annual_discount_rate: 0.06,
                transitions: vec![],
                nodes: Vec::new(),
                season_map: Some(season_map_n(n)),
            },
        }
    }

    /// An explosive AR(1) (`ψ* = 1.2`) on a single-season hydro produces an
    /// `InvalidValue` error naming the hydro, season 0, and the offending
    /// autocorrelation.
    #[test]
    fn par_gate_rejects_explosive() {
        let ar_rows = vec![InflowArCoefficientRow {
            hydro_id: EntityId::from(1),
            stage_id: 0,
            lag: 1,
            coefficient: 1.2,
        }];
        let data = make_data_5b(
            vec![make_hydro_ordered_penalties(1)],
            stages_one_stage_per_season(1),
            vec![make_bus_with_deficit(1, 10.0)],
            vec![],
            ar_rows,
            None,
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(&data, &mut ctx);
        let rejections: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| e.kind == ErrorKind::InvalidValue && e.message.contains("stationarity"))
            .collect();
        assert_eq!(
            rejections.len(),
            1,
            "expected exactly 1 stationarity rejection, got: {:?}",
            ctx.errors()
        );
        assert!(
            rejections[0].message.contains("Hydro 1") && rejections[0].message.contains("season 0"),
            "message should name the hydro and season, got: {}",
            rejections[0].message
        );
    }

    /// A stationary mixed-order (`orders = [3, 1, 2, 1]`) hydro -- the same
    /// fixture `mixed_orders_pass_stationarity_gate` verifies stationary --
    /// produces no stationarity rejection.
    #[test]
    fn par_gate_accepts_stationary() {
        let psi: [Vec<f64>; 4] = [
            vec![
                0.398_915_659_532_620_8,
                0.062_802_463_704_355_48,
                -0.014_634_714_422_504_587,
            ],
            vec![0.35],
            vec![0.294_017_094_017_094, 0.017_094_017_094_017_092],
            vec![0.38],
        ];
        let mut ar_rows = Vec::new();
        for (season, coeffs) in psi.iter().enumerate() {
            for (i, &coefficient) in coeffs.iter().enumerate() {
                ar_rows.push(InflowArCoefficientRow {
                    hydro_id: EntityId::from(1),
                    stage_id: season as i32,
                    lag: (i + 1) as i32,
                    coefficient,
                });
            }
        }
        let data = make_data_5b(
            vec![make_hydro_ordered_penalties(1)],
            stages_one_stage_per_season(4),
            vec![make_bus_with_deficit(1, 10.0)],
            vec![],
            ar_rows,
            None,
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(&data, &mut ctx);
        let rejections: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| e.message.contains("stationarity"))
            .collect();
        assert!(
            rejections.is_empty(),
            "stationary mixed-order coefficients should produce no stationarity \
             rejection, got: {rejections:?}"
        );
    }

    /// Empty `inflow_ar_coefficients` (history-only / estimation path) skips
    /// the gate even when unrelated history data is present.
    #[test]
    fn par_gate_skips_when_no_coefficients() {
        let data = make_data_5b(
            vec![make_hydro_ordered_penalties(1)],
            stages_one_stage_per_season(1),
            vec![make_bus_with_deficit(1, 10.0)],
            vec![],
            vec![], // no AR coefficients -- estimation/history path
            None,
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(&data, &mut ctx);
        let rejections: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| e.message.contains("stationarity"))
            .collect();
        assert!(
            rejections.is_empty(),
            "empty inflow_ar_coefficients should skip the gate entirely, \
             got: {rejections:?}"
        );
    }

    /// Order-bearing AR coefficients whose stage carries NO resolvable season
    /// context (no `season_map` AND no per-stage `season_id` -- genuinely
    /// unresolvable, the same condition under which
    /// `populate_derived_residual_ratios` itself errors) produce a
    /// `BusinessRuleViolation` naming the unresolved stage, instead of
    /// silently skipping the gate.
    #[test]
    fn par_gate_requires_season_map() {
        let ar_rows = vec![InflowArCoefficientRow {
            hydro_id: EntityId::from(1),
            stage_id: 0,
            lag: 1,
            coefficient: 0.5,
        }];
        let data = make_data_5b(
            vec![make_hydro_ordered_penalties(1)],
            make_stages_5b(vec![0]), // stage.season_id: None, no season_map
            vec![make_bus_with_deficit(1, 10.0)],
            vec![],
            ar_rows,
            None,
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(&data, &mut ctx);
        let matching: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| {
                e.kind == ErrorKind::BusinessRuleViolation
                    && e.message.contains("stationarity")
                    && e.message.contains("season_definitions")
            })
            .collect();
        assert!(
            !matching.is_empty(),
            "genuinely unresolvable season context with order-bearing \
             coefficients should error, got: {:?}",
            ctx.errors()
        );
    }

    /// Bare per-stage `season_id`s with no `season_map` resolve via
    /// `resolve_stage_seasons`'s fallback: no
    /// missing-season-context error is added, and the stationarity check
    /// actually runs -- proven here by an explosive AR(1) still being
    /// rejected on this path, not merely "no error".
    #[test]
    fn par_gate_uses_season_id_fallback() {
        let mut stages = make_stages_5b(vec![0]);
        stages.stages[0].season_id = Some(0); // bare season_id, no season_map
        let ar_rows = vec![InflowArCoefficientRow {
            hydro_id: EntityId::from(1),
            stage_id: 0,
            lag: 1,
            coefficient: 1.2, // explosive -- proves the check actually ran
        }];
        let data = make_data_5b(
            vec![make_hydro_ordered_penalties(1)],
            stages,
            vec![make_bus_with_deficit(1, 10.0)],
            vec![],
            ar_rows,
            None,
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(&data, &mut ctx);

        let missing_context: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| {
                e.kind == ErrorKind::BusinessRuleViolation && e.message.contains("stationarity")
            })
            .collect();
        assert!(
            missing_context.is_empty(),
            "a bare per-stage season_id should resolve via the fallback, \
             got: {missing_context:?}"
        );

        let rejections: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| e.kind == ErrorKind::InvalidValue && e.message.contains("stationarity"))
            .collect();
        assert_eq!(
            rejections.len(),
            1,
            "expected the fallback-resolved gate to run and reject the \
             explosive AR(1), got: {:?}",
            ctx.errors()
        );
    }

    /// A PAR-A hydro whose classical part is stationary but whose effective
    /// 12-lag system (widened by the annual term) is explosive -- the same
    /// parameterization as `par_a_explosive_effective_rejected` --
    /// produces a stationarity rejection (the annual gate fired, not merely
    /// a singular closure).
    #[test]
    fn par_gate_rejects_explosive_annual() {
        let ar_rows = vec![InflowArCoefficientRow {
            hydro_id: EntityId::from(1),
            stage_id: 0,
            lag: 1,
            coefficient: 0.3,
        }];
        let stats = vec![InflowSeasonalStatsRow {
            hydro_id: EntityId::from(1),
            stage_id: 0,
            mean_m3s: 100.0,
            std_m3s: 20.0,
        }];
        let mut data = make_data_5b(
            vec![make_hydro_ordered_penalties(1)],
            stages_one_stage_per_season(1),
            vec![make_bus_with_deficit(1, 10.0)],
            stats,
            ar_rows,
            None,
        );
        data.inflow_annual_components = vec![InflowAnnualComponentRow {
            hydro_id: EntityId::from(1),
            stage_id: 0,
            annual_coefficient: 5.0,
            annual_mean_m3s: 100.0,
            annual_std_m3s: 1.0,
        }];

        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(&data, &mut ctx);
        let rejections: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| e.kind == ErrorKind::InvalidValue && e.message.contains("stationarity"))
            .collect();
        assert_eq!(
            rejections.len(),
            1,
            "expected exactly 1 stationarity rejection from the annual gate, got: {:?}",
            ctx.errors()
        );
        assert!(
            !rejections[0].message.contains("singular"),
            "the explosive effective annual system should be a typed rejection, \
             not a singular closure, got: {}",
            rejections[0].message
        );
    }

    // ── Rule 17: Load factor consistency ──────────────────────────────────────

    /// `LoadFactorEntry` with a `block_id` not present in the stage's blocks
    /// still produces 1 rule-17 `BusinessRuleViolation` error.
    #[test]
    fn test_rule17_invalid_block_id_errors() {
        let mut data = make_data_5b(
            vec![],
            make_stages_with_block(0),
            vec![make_bus_with_deficit(1, 10.0)],
            vec![],
            vec![],
            None,
        );
        // Stage 0 has block index 0 only; block_id=99 is invalid.
        data.load_factors = vec![LoadFactorEntry {
            bus_id: EntityId::from(1),
            stage_id: 0,
            block_factors: vec![BlockFactor {
                block_id: 99,
                factor: 1.0,
            }],
        }];
        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(&data, &mut ctx);
        assert!(ctx.has_errors());
        let errors: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| e.kind == ErrorKind::BusinessRuleViolation)
            .collect();
        assert_eq!(
            errors.len(),
            1,
            "expected 1 BusinessRuleViolation, got: {errors:?}"
        );
        assert!(
            errors[0].file.to_string_lossy().contains("load_factors"),
            "error should reference load_factors.json"
        );
        assert!(
            errors[0].message.contains("99"),
            "message should mention invalid block_id 99"
        );
    }

    /// A deterministic load (`std_mw == 0.0`) with defined block factors
    /// produces zero ModelQuality warnings mentioning "deterministic" or "no
    /// effect" — block factors are applied at σ = 0 (see
    /// `test_block_factors_applied_at_zero_sigma` in `novomodelo-stochastic`).
    #[test]
    fn test_deterministic_load_emits_no_factor_warning() {
        let mut data = make_data_5b(
            vec![],
            make_stages_with_block(0),
            vec![make_bus_with_deficit(1, 10.0)],
            vec![],
            vec![],
            None,
        );
        data.load_seasonal_stats = vec![LoadSeasonalStatsRow {
            bus_id: EntityId::from(1),
            stage_id: 0,
            mean_mw: 100.0,
            std_mw: 0.0,
        }];
        data.load_factors = vec![LoadFactorEntry {
            bus_id: EntityId::from(1),
            stage_id: 0,
            block_factors: vec![BlockFactor {
                block_id: 0,
                factor: 1.0,
            }],
        }];
        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(&data, &mut ctx);
        assert!(
            !ctx.has_errors(),
            "deterministic load should not produce an error, got: {:?}",
            ctx.errors()
        );
        let warnings = ctx.warnings();
        let relevant: Vec<_> = warnings
            .iter()
            .filter(|w| w.kind == ErrorKind::ModelQuality)
            .filter(|w| {
                let msg = w.message.to_lowercase();
                msg.contains("deterministic") || msg.contains("no effect")
            })
            .collect();
        assert!(
            relevant.is_empty(),
            "expected zero ModelQuality warnings mentioning \"deterministic\"/\"no effect\", \
             got: {relevant:?}"
        );
    }

    /// Empty `load_factors` produces zero load-related diagnostics.
    #[test]
    fn test_5b_load_factors_empty_no_errors() {
        let data = make_data_5b(
            vec![],
            make_stages_5b(vec![0]),
            vec![make_bus_with_deficit(1, 10.0)],
            vec![],
            vec![],
            None,
        );
        // load_factors is already empty in make_data_5b
        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(&data, &mut ctx);
        let load_factor_errors: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| e.file.to_string_lossy().contains("load_factors"))
            .collect();
        let load_factor_warnings: Vec<_> = ctx
            .warnings()
            .into_iter()
            .filter(|w| w.file.to_string_lossy().contains("load_factors"))
            .collect();
        assert!(
            load_factor_errors.is_empty() && load_factor_warnings.is_empty(),
            "empty load_factors should produce no load-related diagnostics; \
             errors: {load_factor_errors:?}, warnings: {load_factor_warnings:?}"
        );
    }

    // ── Estimation prerequisites (Rules 19-21) ────────────────────────────────

    /// Given `inflow_history` present, `inflow_seasonal_stats` absent, and
    /// `stages.json` WITHOUT `season_definitions`, validation produces a
    /// `BusinessRuleViolation` mentioning "season_definitions is required".
    #[test]
    fn test_estimation_requires_season_definitions() {
        let history = make_history_rows(1, 12);
        let stages = make_stages_with_seasons(12, /*with_season_map=*/ false);
        let data = make_data_estimation(vec![make_hydro(1, None)], stages, history);

        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(&data, &mut ctx);

        let matching: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| {
                e.kind == ErrorKind::BusinessRuleViolation
                    && e.message.contains("season_definitions is required")
            })
            .collect();
        assert!(
            !matching.is_empty(),
            "expected a BusinessRuleViolation about season_definitions, got errors: {:?}",
            ctx.errors()
        );
    }

    /// Given `inflow_history` with only 3 observations for one `(hydro, season)`,
    /// validation produces a `ModelQuality` warning containing "has 3 observations".
    #[test]
    fn test_estimation_warns_low_observations() {
        // 3 full-coverage observations for hydro 1: one per January (season 0)
        // over 3 years — full-month windows so each counts under the
        // coverage gate, matching `make_stages_with_seasons`'s
        // own [1st, next-1st) January stage bounds exactly.
        let history: Vec<InflowHistoryRow> = (0..3)
            .map(|y| {
                let start_date = chrono::NaiveDate::from_ymd_opt(2000 + y, 1, 1).unwrap();
                let end_date = chrono::NaiveDate::from_ymd_opt(2000 + y, 2, 1).unwrap();
                InflowHistoryRow {
                    hydro_id: EntityId::from(1),
                    start_date,
                    end_date,
                    value_m3s: 100.0,
                }
            })
            .collect();

        // 3 years × 12 months = 36 stages, with season_map present.
        let stages = make_stages_with_seasons(36, true);
        let data = make_data_estimation(vec![make_hydro(1, None)], stages, history);

        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(&data, &mut ctx);

        let matching: Vec<_> = ctx
            .warnings()
            .into_iter()
            .filter(|w| {
                w.kind == ErrorKind::ModelQuality && w.message.contains("has 3 observations")
            })
            .collect();
        assert!(
            !matching.is_empty(),
            "expected a ModelQuality warning about 3 observations, got warnings: {:?}",
            ctx.warnings()
        );
    }

    /// Given `inflow_history` with observations for hydro 1 only, but `hydros`
    /// containing hydro 1 and hydro 2, validation produces a
    /// `BusinessRuleViolation` for hydro 2.
    #[test]
    fn test_estimation_error_missing_hydro() {
        let history = make_history_rows(1, 36); // only hydro 1
        let stages = make_stages_with_seasons(36, true);
        let hydros = vec![make_hydro(1, None), make_hydro(2, None)];
        let data = make_data_estimation(hydros, stages, history);

        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(&data, &mut ctx);

        let matching: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| {
                e.kind == ErrorKind::BusinessRuleViolation
                    && e.message.contains("hydro 2 has no observations")
            })
            .collect();
        assert!(
            !matching.is_empty(),
            "expected a BusinessRuleViolation for hydro 2, got errors: {:?}",
            ctx.errors()
        );
    }

    /// When BOTH `inflow_seasonal_stats` and `inflow_ar_coefficients` are
    /// non-empty, no estimation-related errors or warnings are produced.
    #[test]
    fn test_no_estimation_when_stats_and_coefficients_present() {
        let history = make_history_rows(1, 12);
        let stages = make_stages_with_seasons(12, false); // no season_map

        // Provide both stats AND AR coefficients to fully deactivate estimation.
        let stats = vec![InflowSeasonalStatsRow {
            hydro_id: EntityId::from(1),
            stage_id: 0,
            mean_m3s: 500.0,
            std_m3s: 50.0,
        }];
        let ar_coefficients = vec![make_ar_row(1, 0, 1)];

        let mut data = make_data_estimation(vec![make_hydro(1, None)], stages, history);
        data.inflow_seasonal_stats = stats;
        data.inflow_ar_coefficients = ar_coefficients;

        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(&data, &mut ctx);

        // No estimation errors — the estimation path is not active.
        let estimation_errors: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| e.file.to_string_lossy().contains("inflow_history.parquet"))
            .collect();
        let estimation_warnings: Vec<_> = ctx
            .warnings()
            .into_iter()
            .filter(|w| w.file.to_string_lossy().contains("inflow_history.parquet"))
            .collect();
        assert!(
            estimation_errors.is_empty() && estimation_warnings.is_empty(),
            "stats+coefficients present should disable estimation checks; \
             errors: {estimation_errors:?}, warnings: {estimation_warnings:?}"
        );
    }

    fn history_years_errors(data: &ParsedData) -> Vec<String> {
        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(data, &mut ctx);
        ctx.errors()
            .into_iter()
            .filter(|e| e.message.contains("estimation.history_years"))
            .map(|e| e.message.clone())
            .collect()
    }

    #[test]
    fn estimation_history_years_reaching_outside_the_history_is_an_error() {
        // History covers 2000-2002.
        for (from, to, refused) in [
            (2000, 2002, false),
            (2001, 2001, false),
            (1999, 2002, true),
            (2000, 2003, true),
        ] {
            let mut data = make_data_estimation(
                vec![make_hydro(1, None)],
                make_stages_with_seasons(36, true),
                make_history_rows(1, 36),
            );
            data.config.estimation.history_years = Some(HistoricalYearRange { from, to });

            let errors = history_years_errors(&data);
            assert_eq!(
                !errors.is_empty(),
                refused,
                "history_years {from}..={to}: {errors:?}"
            );
        }
    }

    #[test]
    fn estimation_history_years_is_not_checked_when_stats_and_coefficients_are_supplied() {
        let mut data = make_data_estimation(
            vec![make_hydro(1, None)],
            make_stages_with_seasons(12, false),
            make_history_rows(1, 12),
        );
        data.inflow_seasonal_stats = vec![InflowSeasonalStatsRow {
            hydro_id: EntityId::from(1),
            stage_id: 0,
            mean_m3s: 500.0,
            std_m3s: 50.0,
        }];
        data.inflow_ar_coefficients = vec![make_ar_row(1, 0, 1)];
        data.config.estimation.history_years = Some(HistoricalYearRange {
            from: 1931,
            to: 2024,
        });

        let errors = history_years_errors(&data);
        assert!(errors.is_empty(), "{errors:?}");
    }

    /// Three full-January rows, from `first_year` on.
    fn january_rows(first_year: i32) -> Vec<InflowHistoryRow> {
        (0..3)
            .map(|y| InflowHistoryRow {
                hydro_id: EntityId::from(1),
                start_date: chrono::NaiveDate::from_ymd_opt(first_year + y, 1, 1).unwrap(),
                end_date: chrono::NaiveDate::from_ymd_opt(first_year + y, 2, 1).unwrap(),
                value_m3s: 100.0,
            })
            .collect()
    }

    fn assert_season_0_count(data: &ParsedData, n: usize) {
        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(data, &mut ctx);
        let expected = format!("hydro 1 season 0 has {n} observations");
        assert!(
            ctx.warnings().iter().any(|w| w.message.contains(&expected)),
            "{:?}",
            ctx.warnings()
        );
    }

    #[test]
    fn estimation_observation_count_covers_only_the_history_years() {
        let mut data = make_data_estimation(
            vec![make_hydro(1, None)],
            make_stages_with_seasons(36, true),
            january_rows(2000),
        );
        data.config.estimation.history_years = Some(HistoricalYearRange {
            from: 2000,
            to: 2001,
        });

        assert_season_0_count(&data, 2);
    }

    #[test]
    fn estimation_observation_count_includes_history_before_the_study() {
        let data = make_data_estimation(
            vec![make_hydro(1, None)],
            make_stages_with_seasons(12, true),
            january_rows(1990),
        );

        assert_season_0_count(&data, 3);
    }

    #[test]
    fn estimation_observation_count_skips_a_partially_covered_occurrence() {
        let mut history = january_rows(1990);
        history[0].end_date = chrono::NaiveDate::from_ymd_opt(1990, 1, 20).unwrap();
        let data = make_data_estimation(
            vec![make_hydro(1, None)],
            make_stages_with_seasons(12, true),
            history,
        );

        assert_season_0_count(&data, 2);
    }

    /// When `inflow_seasonal_stats` is present but `inflow_ar_coefficients`
    /// is absent, estimation IS active and season_definitions is required.
    #[test]
    fn test_estimation_active_when_stats_present_but_coefficients_absent() {
        let history = make_history_rows(1, 12);
        let stages = make_stages_with_seasons(12, false); // no season_map

        let stats = vec![InflowSeasonalStatsRow {
            hydro_id: EntityId::from(1),
            stage_id: 0,
            mean_m3s: 500.0,
            std_m3s: 50.0,
        }];

        let mut data = make_data_estimation(vec![make_hydro(1, None)], stages, history);
        data.inflow_seasonal_stats = stats;
        // AR coefficients NOT provided — estimation should be active.

        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(&data, &mut ctx);

        // Should produce a season_definitions error (Rule 19).
        let estimation_errors: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| {
                e.file.to_string_lossy().contains("inflow_history.parquet")
                    && e.message.contains("season_definitions")
            })
            .collect();
        assert!(
            !estimation_errors.is_empty(),
            "stats present without coefficients should trigger estimation checks; \
             got errors: {:?}",
            ctx.errors()
        );
    }

    // ── External scheme requires external scenario files ──────────────

    /// `config.training.scenario_source.inflow.scheme = "external"` with no
    /// `external_scenarios` data produces an error referencing `"config.json"` and
    /// field `"training.scenario_source.inflow"`.
    #[test]
    fn test_training_external_inflow_without_file_is_error() {
        let mut data = make_data_5b(
            vec![make_hydro_ordered_penalties(1)],
            make_stages_5b(vec![0]),
            vec![make_bus_with_deficit(1, 75.0)],
            vec![],
            vec![],
            None,
        );
        data.config = config_with_training_external_inflow();
        data.external_scenarios = vec![]; // no external inflow file

        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(&data, &mut ctx);

        let errors = ctx.errors();
        let matching: Vec<_> = errors
            .iter()
            .filter(|e| {
                e.kind == ErrorKind::BusinessRuleViolation
                    && e.file == std::path::Path::new("config.json")
                    && e.entity
                        .as_deref()
                        .is_some_and(|f| f.contains("training.scenario_source.inflow"))
            })
            .collect();
        assert_eq!(
            matching.len(),
            1,
            "expected 1 error for missing external inflow file (training), \
             got: {errors:?}"
        );
    }

    /// `config.simulation.scenario_source.load.scheme = "external"` with no
    /// `external_load_scenarios` data produces an error referencing `"config.json"`
    /// and field `"simulation.scenario_source.load"`.
    #[test]
    fn test_simulation_external_load_without_file_is_error() {
        let mut data = make_data_5b(
            vec![make_hydro_ordered_penalties(1)],
            make_stages_5b(vec![0]),
            vec![make_bus_with_deficit(1, 75.0)],
            vec![],
            vec![],
            None,
        );
        data.config = config_with_simulation_external_load();
        data.external_load_scenarios = vec![]; // no external load file

        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(&data, &mut ctx);

        let errors = ctx.errors();
        let matching: Vec<_> = errors
            .iter()
            .filter(|e| {
                e.kind == ErrorKind::BusinessRuleViolation
                    && e.file == std::path::Path::new("config.json")
                    && e.entity
                        .as_deref()
                        .is_some_and(|f| f.contains("simulation.scenario_source.load"))
            })
            .collect();
        assert_eq!(
            matching.len(),
            1,
            "expected 1 error for missing external load file (simulation), \
             got: {errors:?}"
        );
    }

    /// Training uses External inflow and the external file is present: no error.
    #[test]
    fn test_training_external_inflow_with_file_is_ok() {
        use novomodelo_core::scenario::ExternalScenarioRow;
        let mut data = make_data_5b(
            vec![make_hydro_ordered_penalties(1)],
            make_stages_5b(vec![0]),
            vec![make_bus_with_deficit(1, 75.0)],
            vec![],
            vec![],
            None,
        );
        data.config = config_with_training_external_inflow();
        // Provide at least one row so the file is considered non-empty.
        data.external_scenarios = vec![ExternalScenarioRow {
            hydro_id: EntityId::from(1),
            stage_id: 0,
            scenario_id: 1,
            value_m3s: 10.0,
        }];

        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(&data, &mut ctx);

        let errors = ctx.errors();
        let external_errors: Vec<_> = errors
            .iter()
            .filter(|e| {
                e.kind == ErrorKind::BusinessRuleViolation
                    && e.file == std::path::Path::new("config.json")
            })
            .collect();
        assert!(
            external_errors.is_empty(),
            "no external-file errors expected when file is present, \
             got: {external_errors:?}"
        );
    }

    // ── Rule 33: Filling-schedule sufficiency ─────────────────────────────────

    /// Build a `StagesData` whose stages have ids `ids`, each carrying a single
    /// block of `duration_hours`. The ζ for every stage is therefore
    /// `duration_hours · M3S_TO_HM3`.
    fn make_stages_with_block_duration(ids: &[i32], duration_hours: f64) -> StagesData {
        let stages = ids
            .iter()
            .map(|&id| {
                let mut stage = make_stage(id);
                stage.blocks = vec![Block {
                    index: 0,
                    name: "FLAT".to_string(),
                    duration_hours,
                }];
                stage
            })
            .collect();
        StagesData {
            openings_declared: std::collections::HashSet::new(),
            stages,
            policy_graph: HorizonGraph {
                stage_discount_rate_overrides: std::collections::BTreeMap::new(),
                graph_type: PolicyGraphType::FiniteHorizon,
                annual_discount_rate: 0.06,
                transitions: vec![],
                nodes: Vec::new(),
                season_map: None,
            },
        }
    }

    /// Build a filling hydro with dead volume `min_storage_hm3`, filling window
    /// `[start_stage_id, entry_stage_id)`, and entity-level `filling_min_rate_m3s`.
    fn make_filling_hydro(
        id: i32,
        min_storage_hm3: f64,
        start_stage_id: i32,
        entry_stage_id: i32,
        filling_min_rate_m3s: f64,
    ) -> Hydro {
        use novomodelo_core::entities::FillingConfig;
        let mut h = make_hydro_ordered_penalties(id);
        h.min_storage_hm3 = min_storage_hm3;
        h.entry_stage_id = Some(entry_stage_id);
        h.filling = Some(FillingConfig {
            start_stage_id,
            filling_min_rate_m3s,
        });
        h
    }

    /// Under-provisioned: two filling stages (ζ = 0.0036·720 = 2.592 each) at
    /// rate 1.0 give capacity 5.184 < 60.0 → one `BusinessRuleViolation`.
    #[test]
    fn test_filling_sufficiency_underprovisioned_errors() {
        let hydro = make_filling_hydro(7, 60.0, 2, 4, 1.0);
        let data = make_data_5b(
            vec![hydro],
            make_stages_with_block_duration(&[2, 3], 720.0),
            vec![make_bus_with_deficit(1, 10.0)],
            vec![],
            vec![],
            None,
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(&data, &mut ctx);

        let relevant: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| {
                e.kind == ErrorKind::BusinessRuleViolation
                    && e.file == std::path::Path::new("system/hydros.json")
                    && e.message.contains("filling schedule is insufficient")
            })
            .collect();
        assert_eq!(
            relevant.len(),
            1,
            "expected exactly 1 sufficiency error, got: {:?}",
            ctx.errors()
        );
        let msg = &relevant[0].message;
        assert!(
            msg.contains("Hydro 7"),
            "message should name Hydro 7, got: {msg}"
        );
        assert!(
            msg.contains("5.184"),
            "message should contain the capacity 5.184, got: {msg}"
        );
        assert!(
            msg.contains("60"),
            "message should contain the required 60 shortfall, got: {msg}"
        );
    }

    /// Sufficient: same two stages at rate 20.0 give capacity 103.68 >= 60.0 →
    /// no sufficiency error.
    #[test]
    fn test_filling_sufficiency_sufficient_no_error() {
        let hydro = make_filling_hydro(7, 60.0, 2, 4, 20.0);
        let data = make_data_5b(
            vec![hydro],
            make_stages_with_block_duration(&[2, 3], 720.0),
            vec![make_bus_with_deficit(1, 10.0)],
            vec![],
            vec![],
            None,
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(&data, &mut ctx);

        let relevant: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| e.message.contains("filling schedule is insufficient"))
            .collect();
        assert!(
            relevant.is_empty(),
            "capacity 103.68 >= 60.0 should emit no sufficiency error, got: {relevant:?}"
        );
    }

    /// Over-provisioned: capacity strictly greater than `min_storage − seed`
    /// must not be rejected (one-sided check).
    #[test]
    fn test_filling_sufficiency_overprovisioned_no_error() {
        // Capacity = 2·2.592·100 = 518.4, far above min_storage 60.0.
        let hydro = make_filling_hydro(7, 60.0, 2, 4, 100.0);
        let data = make_data_5b(
            vec![hydro],
            make_stages_with_block_duration(&[2, 3], 720.0),
            vec![make_bus_with_deficit(1, 10.0)],
            vec![],
            vec![],
            None,
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(&data, &mut ctx);

        let relevant: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| e.message.contains("filling schedule is insufficient"))
            .collect();
        assert!(
            relevant.is_empty(),
            "over-provisioning must never be rejected (one-sided), got: {relevant:?}"
        );
        assert!(
            !ctx.has_errors(),
            "an over-provisioned filling schedule should produce no errors at all, got: {:?}",
            ctx.errors()
        );
    }

    /// A `capacity` a hair below `required` — by less than the
    /// relative-with-floor tolerance — must not be rejected: the round-off
    /// false-reject this check exists to fix (reported Δ ≈ 5e-14 hm3 on a
    /// ~232 hm3 requirement, ~2e-16 relative).
    #[test]
    fn test_filling_sufficiency_within_relative_tolerance_no_error() {
        let zeta = 720.0 * M3S_TO_HM3;
        let rate = 30.0;
        let capacity = zeta * rate + zeta * rate;
        // A relative gap an order of magnitude below the tolerance.
        let min_storage_hm3 = capacity + capacity.abs().max(1.0) * 1e-10;
        let hydro = make_filling_hydro(7, min_storage_hm3, 2, 4, rate);
        let data = make_data_5b(
            vec![hydro],
            make_stages_with_block_duration(&[2, 3], 720.0),
            vec![make_bus_with_deficit(1, 10.0)],
            vec![],
            vec![],
            None,
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(&data, &mut ctx);

        let relevant: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| e.message.contains("filling schedule is insufficient"))
            .collect();
        assert!(
            relevant.is_empty(),
            "a capacity within relative tolerance of required must not be rejected, \
             got: {relevant:?}"
        );
    }

    /// A `capacity` short of `required` by an order of magnitude MORE than the
    /// tolerance is still rejected — the tolerance must have power, not just
    /// admit the reported round-off.
    #[test]
    fn test_filling_sufficiency_beyond_tolerance_errors() {
        let zeta = 720.0 * M3S_TO_HM3;
        let rate = 30.0;
        let capacity = zeta * rate + zeta * rate;
        // A relative gap an order of magnitude above the tolerance.
        let min_storage_hm3 = capacity + capacity.abs().max(1.0) * 1e-8;
        let hydro = make_filling_hydro(7, min_storage_hm3, 2, 4, rate);
        let data = make_data_5b(
            vec![hydro],
            make_stages_with_block_duration(&[2, 3], 720.0),
            vec![make_bus_with_deficit(1, 10.0)],
            vec![],
            vec![],
            None,
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(&data, &mut ctx);

        let relevant: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| e.message.contains("filling schedule is insufficient"))
            .collect();
        assert_eq!(
            relevant.len(),
            1,
            "a shortfall beyond the tolerance must still be rejected, got: {:?}",
            ctx.errors()
        );
    }

    /// A non-filling hydro (`filling: None`) is ignored by the sufficiency check
    /// even when its `min_storage_hm3` is large.
    #[test]
    fn test_filling_sufficiency_ignores_non_filling_hydro() {
        let mut hydro = make_hydro_ordered_penalties(7);
        // Large dead volume, but no filling config: the check must skip it.
        hydro.min_storage_hm3 = 60.0;
        hydro.entry_stage_id = Some(4);
        assert!(hydro.filling.is_none());
        let data = make_data_5b(
            vec![hydro],
            make_stages_with_block_duration(&[2, 3], 720.0),
            vec![make_bus_with_deficit(1, 10.0)],
            vec![],
            vec![],
            None,
        );
        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(&data, &mut ctx);

        let relevant: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| e.message.contains("filling schedule is insufficient"))
            .collect();
        assert!(
            relevant.is_empty(),
            "non-filling hydro must emit no sufficiency diagnostic, got: {relevant:?}"
        );
    }

    // ── Rules 45-48: External-library coherence ───────────────────────────────

    /// Build `ParsedData` with the given study stage ids and external classes,
    /// wired for the training external scheme of each supplied non-empty vector.
    fn external_data(
        stage_ids: Vec<i32>,
        inflow: Vec<ExternalScenarioRow>,
        load: Vec<ExternalLoadRow>,
        ncs: Vec<ExternalNcsRow>,
    ) -> ParsedData {
        let mut data = make_data_5b(
            vec![],
            make_stages_5b(stage_ids),
            vec![make_bus_with_deficit(1, 10.0)],
            vec![],
            vec![],
            None,
        );
        data.config =
            config_with_training_external(!inflow.is_empty(), !load.is_empty(), !ncs.is_empty());
        data.external_scenarios = inflow;
        data.external_load_scenarios = load;
        data.external_ncs_scenarios = ncs;
        data
    }

    /// One inflow row for hydro `hydro`, `(stage_id, scenario_id, value)`.
    fn inflow_row(
        hydro: i32,
        stage_id: i32,
        scenario_id: i32,
        value_m3s: f64,
    ) -> ExternalScenarioRow {
        ExternalScenarioRow {
            stage_id,
            scenario_id,
            hydro_id: EntityId::from(hydro),
            value_m3s,
        }
    }

    fn run(data: &ParsedData) -> ValidationContext {
        let mut ctx = ValidationContext::new();
        validate_semantic_stages_penalties_scenarios(data, &mut ctx);
        ctx
    }

    fn error_msgs(ctx: &ValidationContext) -> Vec<String> {
        ctx.errors().iter().map(|e| e.message.clone()).collect()
    }

    // ── A1: exact scenario_id set (rule 46) ───────────────────────────────────

    /// A 1-based deck (`{1, 2}`) is rejected — the case that loads today,
    /// aliases stage `t+1`'s realization 0, and zero-fills realization 0.
    #[test]
    fn a1_rejects_one_based_deck() {
        let data = external_data(
            vec![0],
            vec![inflow_row(1, 0, 1, 10.0), inflow_row(1, 0, 2, 20.0)],
            vec![],
            vec![],
        );
        let ctx = run(&data);
        assert!(
            error_msgs(&ctx)
                .iter()
                .any(|m| m.contains("scenario_id set must be exactly") && m.contains("inflow")),
            "1-based deck must be rejected as a set violation, got: {:?}",
            error_msgs(&ctx)
        );
    }

    /// A gapped deck (`{0, 2}`) is rejected — a set check, not a bound check.
    #[test]
    fn a1_rejects_gap() {
        let data = external_data(
            vec![0],
            vec![inflow_row(1, 0, 0, 10.0), inflow_row(1, 0, 2, 20.0)],
            vec![],
            vec![],
        );
        let ctx = run(&data);
        assert!(
            error_msgs(&ctx)
                .iter()
                .any(|m| m.contains("scenario_id set must be exactly")),
            "a gap must be rejected, got: {:?}",
            error_msgs(&ctx)
        );
    }

    /// A duplicated `(entity, scenario_id)` pair is rejected.
    #[test]
    fn a1_rejects_duplicate() {
        let data = external_data(
            vec![0],
            vec![inflow_row(1, 0, 0, 10.0), inflow_row(1, 0, 0, 11.0)],
            vec![],
            vec![],
        );
        let ctx = run(&data);
        assert!(
            error_msgs(&ctx).iter().any(|m| m.contains("duplicate row")),
            "a duplicate must be rejected, got: {:?}",
            error_msgs(&ctx)
        );
    }

    /// On a chain-dialect deck (no `nodes[]`), `extract_class` tracks
    /// duplicates in a key-only set instead of the value map -- the rejection
    /// must still fire, byte-identical to the value-carrying path.
    #[test]
    fn a1_rejects_duplicate_chain_dialect_key_only() {
        let data = external_data(
            vec![0],
            vec![inflow_row(1, 0, 0, 10.0), inflow_row(1, 0, 0, 11.0)],
            vec![],
            vec![],
        );
        assert!(
            data.stages.policy_graph.nodes.is_empty(),
            "chain-dialect fixture"
        );
        let ctx = run(&data);
        assert!(
            error_msgs(&ctx).iter().any(|m| {
                m.contains(
                    "inflow external library has a duplicate row at stage 0 for scenario_id 0, \
                     entity 1",
                )
            }),
            "a key-only duplicate check must still reject the duplicate row, got: {:?}",
            error_msgs(&ctx)
        );
    }

    /// An out-of-range member (`5`) is rejected naming the offending value.
    #[test]
    fn a1_rejects_out_of_range() {
        let data = external_data(
            vec![0],
            vec![
                inflow_row(1, 0, 0, 10.0),
                inflow_row(1, 0, 1, 20.0),
                inflow_row(1, 0, 5, 30.0),
            ],
            vec![],
            vec![],
        );
        let ctx = run(&data);
        assert!(
            error_msgs(&ctx)
                .iter()
                .any(|m| m.contains("scenario_id set must be exactly") && m.contains('5')),
            "an out-of-range member must be rejected naming the value, got: {:?}",
            error_msgs(&ctx)
        );
    }

    // ── A2: out-of-range stage_id (rule 47) ───────────────────────────────────

    /// Reuses the shared error shape ("resolves to no declared study
    /// stage") for the inflow file.
    #[test]
    fn a2_rejects_out_of_range_stage_id_inflow() {
        let data = external_data(vec![0], vec![inflow_row(1, 5, 0, 10.0)], vec![], vec![]);
        let ctx = run(&data);
        assert!(
            ctx.errors().iter().any(|e| {
                e.message.contains("resolves to no declared study stage")
                    && e.file
                        .to_string_lossy()
                        .contains("external_inflow_scenarios")
            }),
            "an out-of-range inflow stage_id must be rejected, got: {:?}",
            error_msgs(&ctx)
        );
    }

    #[test]
    fn a2_rejects_out_of_range_stage_id_load() {
        let load = vec![ExternalLoadRow {
            stage_id: 5,
            scenario_id: 0,
            bus_id: EntityId::from(1),
            value_mw: 10.0,
        }];
        let data = external_data(vec![0], vec![], load, vec![]);
        let ctx = run(&data);
        assert!(
            ctx.errors().iter().any(|e| {
                e.message.contains("resolves to no declared study stage")
                    && e.file.to_string_lossy().contains("external_load_scenarios")
            }),
            "an out-of-range load stage_id must be rejected, got: {:?}",
            error_msgs(&ctx)
        );
    }

    #[test]
    fn a2_rejects_out_of_range_stage_id_ncs() {
        let ncs = vec![ExternalNcsRow {
            stage_id: 5,
            scenario_id: 0,
            ncs_id: EntityId::from(1),
            value: 0.5,
        }];
        let data = external_data(vec![0], vec![], vec![], ncs);
        let ctx = run(&data);
        assert!(
            ctx.errors().iter().any(|e| {
                e.message.contains("resolves to no declared study stage")
                    && e.file.to_string_lossy().contains("external_ncs_scenarios")
            }),
            "an out-of-range ncs stage_id must be rejected, got: {:?}",
            error_msgs(&ctx)
        );
    }

    // ── P-B1: cross-class raw_c agreement (rule 45) ───────────────────────────

    /// inflow (`raw_c = 2`) and load (`raw_c = 3`) disagree on the per-stage raw
    /// column count on a chain (no `nodes[]`) — rejected, naming both classes and
    /// both counts, with no element-wise-minimum reconciliation.
    #[test]
    fn p_b1_rejects_raw_c_disagreement_chain() {
        let load = vec![
            ExternalLoadRow {
                stage_id: 0,
                scenario_id: 0,
                bus_id: EntityId::from(1),
                value_mw: 1.0,
            },
            ExternalLoadRow {
                stage_id: 0,
                scenario_id: 1,
                bus_id: EntityId::from(1),
                value_mw: 2.0,
            },
            ExternalLoadRow {
                stage_id: 0,
                scenario_id: 2,
                bus_id: EntityId::from(1),
                value_mw: 3.0,
            },
        ];
        let data = external_data(
            vec![0],
            vec![inflow_row(1, 0, 0, 10.0), inflow_row(1, 0, 1, 20.0)],
            load,
            vec![],
        );
        assert!(data.stages.policy_graph.nodes.is_empty(), "chain fixture");
        let ctx = run(&data);
        assert!(
            error_msgs(&ctx)
                .iter()
                .any(|m| m.contains("disagree on the raw column count")
                    && m.contains("inflow")
                    && m.contains("load")
                    && m.contains('2')
                    && m.contains('3')),
            "cross-class raw_c disagreement on a chain must be rejected, got: {:?}",
            error_msgs(&ctx)
        );
    }

    /// The same disagreement under a declared `nodes[]` graph is rejected too —
    /// P-B1 fires with or without a graph.
    #[test]
    fn p_b1_rejects_raw_c_disagreement_graph() {
        let load = vec![
            ExternalLoadRow {
                stage_id: 0,
                scenario_id: 0,
                bus_id: EntityId::from(1),
                value_mw: 1.0,
            },
            ExternalLoadRow {
                stage_id: 0,
                scenario_id: 1,
                bus_id: EntityId::from(1),
                value_mw: 2.0,
            },
            ExternalLoadRow {
                stage_id: 0,
                scenario_id: 2,
                bus_id: EntityId::from(1),
                value_mw: 3.0,
            },
        ];
        let mut data = external_data(
            vec![0],
            vec![inflow_row(1, 0, 0, 10.0), inflow_row(1, 0, 1, 20.0)],
            load,
            vec![],
        );
        data.stages.policy_graph.nodes = vec![Node {
            id: 0,
            stage_id: 0,
            scenario_id: Some(0),
            label: None,
        }];
        let ctx = run(&data);
        assert!(
            error_msgs(&ctx)
                .iter()
                .any(|m| m.contains("disagree on the raw column count")),
            "cross-class raw_c disagreement under a graph must be rejected, got: {:?}",
            error_msgs(&ctx)
        );
    }

    // ── Prefix-coherence warning (rule 48) ────────────────────────────────────

    /// Two stages, one hydro, columns 0 and 1. Edge `0 -> 2` points at columns 0
    /// and 1; a disagreeing shared-prefix cell (stage 0) warns and the run
    /// proceeds. The bridge edge `0 -> 1` (same column) is silent.
    fn prefix_graph_data(stage0_col0: f64, stage0_col1: f64) -> ParsedData {
        let inflow = vec![
            inflow_row(1, 0, 0, stage0_col0),
            inflow_row(1, 0, 1, stage0_col1),
            inflow_row(1, 1, 0, 30.0),
            inflow_row(1, 1, 1, 40.0),
        ];
        let mut data = external_data(vec![0, 1], inflow, vec![], vec![]);
        data.stages.policy_graph.nodes = vec![
            Node {
                id: 0,
                stage_id: 0,
                scenario_id: Some(0),
                label: None,
            },
            Node {
                id: 1,
                stage_id: 1,
                scenario_id: Some(0),
                label: None,
            },
            Node {
                id: 2,
                stage_id: 1,
                scenario_id: Some(1),
                label: None,
            },
        ];
        data.stages.policy_graph.transitions = vec![
            Transition {
                source_id: 0,
                target_id: 1,
                probability: 0.5,
                annual_discount_rate_override: None,
            },
            Transition {
                source_id: 0,
                target_id: 2,
                probability: 0.5,
                annual_discount_rate_override: None,
            },
        ];
        data
    }

    #[test]
    fn prefix_coherence_warns_on_disagreeing_column() {
        let data = prefix_graph_data(10.0, 20.0);
        let ctx = run(&data);
        let prefix_warnings: Vec<_> = ctx
            .warnings()
            .into_iter()
            .filter(|w| w.message.contains("prefix-coherence"))
            .collect();
        assert_eq!(
            prefix_warnings.len(),
            1,
            "the disagreeing edge 0->2 must warn exactly once, got: {:?}",
            ctx.warnings()
        );
        let msg = &prefix_warnings[0].message;
        assert!(
            msg.contains("0->2") && msg.contains("10") && msg.contains("20"),
            "the warning names the edge and both values, got: {msg}"
        );
    }

    #[test]
    fn prefix_coherence_silent_on_identical_trunk() {
        let data = prefix_graph_data(10.0, 10.0);
        let ctx = run(&data);
        assert!(
            !ctx.warnings()
                .iter()
                .any(|w| w.message.contains("prefix-coherence")),
            "identical trunk columns must not warn, got: {:?}",
            ctx.warnings()
        );
    }

    /// A second source node (id 3) shares stage 0, column 0 with node 0, so
    /// edges `0->2` and `3->2` share the same `(cn, cm, sn)` triple
    /// [`PrefixWalk`] memoises. Both edges must still warn independently, in
    /// transition order, each with its own edge attribution -- the memo
    /// removes the redundant re-walk, never a warning.
    #[test]
    fn prefix_coherence_shared_pair_warns_independently_per_edge() {
        let mut data = prefix_graph_data(10.0, 20.0);
        data.stages.policy_graph.nodes.push(Node {
            id: 3,
            stage_id: 0,
            scenario_id: Some(0),
            label: None,
        });
        data.stages.policy_graph.transitions.push(Transition {
            source_id: 3,
            target_id: 2,
            probability: 0.5,
            annual_discount_rate_override: None,
        });
        let ctx = run(&data);
        let prefix_warnings: Vec<_> = ctx
            .warnings()
            .into_iter()
            .filter(|w| w.message.contains("prefix-coherence"))
            .collect();
        assert_eq!(
            prefix_warnings.len(),
            2,
            "both edges sharing (cn,cm,sn) must warn independently, got: {:?}",
            ctx.warnings()
        );
        assert!(
            prefix_warnings[0].message.contains("0->2"),
            "edge 0->2 (declared first) must warn first, got: {prefix_warnings:?}"
        );
        assert!(
            prefix_warnings[1].message.contains("3->2"),
            "edge 3->2 (declared second) must warn second, got: {prefix_warnings:?}"
        );
        for w in &prefix_warnings {
            assert!(
                w.message.contains("10") && w.message.contains("20"),
                "each edge's own warning must name both disagreeing values, got: {}",
                w.message
            );
        }
    }

    /// Two edges share the same `(cn, cm)` column pair but differ in `sn`: a
    /// DEEP edge (`0->2`, `sn=2`) and a SHALLOW edge (`1->2`, `sn=0`). Columns
    /// agree at stage 0 and disagree at stage 1 -- so the deep edge's own
    /// prefix reaches the disagreement (must warn) while the shallow edge's
    /// own prefix (`0..=0`) never does (must not warn), even though the deep
    /// edge is declared FIRST and caches the disagreement before the shallow
    /// edge is evaluated. Without the depth-gated attribution, the shallow
    /// edge would spuriously inherit the deep edge's cached disagreement.
    #[test]
    fn prefix_coherence_differing_sn_shallow_edge_does_not_inherit_deep_disagreement() {
        let inflow = vec![
            inflow_row(1, 0, 0, 10.0),
            inflow_row(1, 0, 1, 10.0),
            inflow_row(1, 1, 0, 10.0),
            inflow_row(1, 1, 1, 20.0),
            inflow_row(1, 2, 0, 30.0),
            inflow_row(1, 2, 1, 30.0),
        ];
        let mut data = external_data(vec![0, 1, 2], inflow, vec![], vec![]);
        data.stages.policy_graph.nodes = vec![
            Node {
                id: 0,
                stage_id: 2,
                scenario_id: Some(0),
                label: None,
            },
            Node {
                id: 1,
                stage_id: 0,
                scenario_id: Some(0),
                label: None,
            },
            Node {
                id: 2,
                stage_id: 1,
                scenario_id: Some(1),
                label: None,
            },
        ];
        data.stages.policy_graph.transitions = vec![
            Transition {
                source_id: 0,
                target_id: 2,
                probability: 0.5,
                annual_discount_rate_override: None,
            },
            Transition {
                source_id: 1,
                target_id: 2,
                probability: 0.5,
                annual_discount_rate_override: None,
            },
        ];

        let ctx = run(&data);
        let prefix_warnings: Vec<_> = ctx
            .warnings()
            .into_iter()
            .filter(|w| w.message.contains("prefix-coherence"))
            .collect();
        assert_eq!(
            prefix_warnings.len(),
            1,
            "only the deep edge (sn >= the disagreement's stage) may warn; the \
             shallow edge (sn < that stage) must not inherit the cached \
             disagreement, got: {prefix_warnings:?}"
        );
        let msg = &prefix_warnings[0].message;
        assert!(
            msg.contains("0->2") && !msg.contains("1->2"),
            "the warning must be attributed to the deep edge 0->2, not the \
             shallow edge 1->2, got: {msg}"
        );
        assert!(
            msg.contains("stage 1") && msg.contains("10") && msg.contains("20"),
            "the warning must name stage 1 and both disagreeing values, got: {msg}"
        );
    }

    /// The reverse declaration order (shallow edge first) reaches the same
    /// outcome: the shallow edge's own walk never reaches the disagreement, and
    /// the deep edge extends the walk past it afterward.
    #[test]
    fn prefix_coherence_differing_sn_shallow_edge_does_not_inherit_deep_disagreement_reverse_order()
    {
        let inflow = vec![
            inflow_row(1, 0, 0, 10.0),
            inflow_row(1, 0, 1, 10.0),
            inflow_row(1, 1, 0, 10.0),
            inflow_row(1, 1, 1, 20.0),
            inflow_row(1, 2, 0, 30.0),
            inflow_row(1, 2, 1, 30.0),
        ];
        let mut data = external_data(vec![0, 1, 2], inflow, vec![], vec![]);
        data.stages.policy_graph.nodes = vec![
            Node {
                id: 0,
                stage_id: 2,
                scenario_id: Some(0),
                label: None,
            },
            Node {
                id: 1,
                stage_id: 0,
                scenario_id: Some(0),
                label: None,
            },
            Node {
                id: 2,
                stage_id: 1,
                scenario_id: Some(1),
                label: None,
            },
        ];
        data.stages.policy_graph.transitions = vec![
            Transition {
                source_id: 1,
                target_id: 2,
                probability: 0.5,
                annual_discount_rate_override: None,
            },
            Transition {
                source_id: 0,
                target_id: 2,
                probability: 0.5,
                annual_discount_rate_override: None,
            },
        ];

        let ctx = run(&data);
        let prefix_warnings: Vec<_> = ctx
            .warnings()
            .into_iter()
            .filter(|w| w.message.contains("prefix-coherence"))
            .collect();
        assert_eq!(
            prefix_warnings.len(),
            1,
            "declaration order must not change the outcome, got: {prefix_warnings:?}"
        );
        assert!(
            prefix_warnings[0].message.contains("0->2"),
            "the warning must still be attributed to the deep edge, got: {:?}",
            prefix_warnings[0]
        );
    }

    /// A coherent single-class deck (no `nodes[]`) produces no coherence
    /// diagnostics, so the coherence checks never fire on a deck without a node
    /// graph.
    #[test]
    fn coherent_external_deck_has_no_coherence_errors() {
        let data = external_data(
            vec![0, 1],
            vec![
                inflow_row(1, 0, 0, 10.0),
                inflow_row(1, 0, 1, 20.0),
                inflow_row(1, 1, 0, 30.0),
                inflow_row(1, 1, 1, 40.0),
            ],
            vec![],
            vec![],
        );
        let ctx = run(&data);
        assert!(
            !error_msgs(&ctx).iter().any(|m| {
                m.contains("scenario_id set must be exactly")
                    || m.contains("disagree on the raw column count")
                    || m.contains("resolves to no declared study stage")
                    || m.contains("duplicate row")
            }),
            "a coherent deck must produce no coherence errors, got: {:?}",
            error_msgs(&ctx)
        );
    }

    // ── External σ = 0 columns ────────────────────────────────────────────────

    /// Two-stage graph: stage 0 is a single-node trunk (column 0 only, always
    /// a constant column); stage 1 fans into two children pointing at columns
    /// 0 and 1, whose values the caller controls.
    fn trunk_and_fan_graph(hydro: i32, fan_col0: f64, fan_col1: f64) -> ParsedData {
        let inflow = vec![
            inflow_row(hydro, 0, 0, 10.0),
            inflow_row(hydro, 1, 0, fan_col0),
            inflow_row(hydro, 1, 1, fan_col1),
        ];
        let mut data = external_data(vec![0, 1], inflow, vec![], vec![]);
        data.stages.policy_graph.nodes = vec![
            Node {
                id: 0,
                stage_id: 0,
                scenario_id: Some(0),
                label: None,
            },
            Node {
                id: 1,
                stage_id: 1,
                scenario_id: Some(0),
                label: None,
            },
            Node {
                id: 2,
                stage_id: 1,
                scenario_id: Some(1),
                label: None,
            },
        ];
        data.stages.policy_graph.transitions = vec![
            Transition {
                source_id: 0,
                target_id: 1,
                probability: 0.5,
                annual_discount_rate_override: None,
            },
            Transition {
                source_id: 0,
                target_id: 2,
                probability: 0.5,
                annual_discount_rate_override: None,
            },
        ];
        data
    }

    fn ar_coefficient_row(hydro: i32, stage_id: i32) -> InflowArCoefficientRow {
        InflowArCoefficientRow {
            hydro_id: EntityId::from(hydro),
            stage_id,
            lag: 1,
            coefficient: 0.5,
        }
    }

    /// A constant external inflow column under an autoregressive model is
    /// accepted at the trunk and at the fan stage: the noise inversion uses the
    /// model's own σ, which is positive, so the constant replays as recorded.
    #[test]
    fn external_constant_inflow_column_under_an_ar_model_is_accepted() {
        let mut data = trunk_and_fan_graph(1, 20.0, 20.0);
        for stage in &mut data.stages.stages {
            stage.season_id = Some(0);
        }
        data.inflow_ar_coefficients = vec![ar_coefficient_row(1, 0), ar_coefficient_row(1, 1)];
        let ctx = run(&data);
        assert!(
            !ctx.has_errors(),
            "a constant inflow column under an AR model must be accepted: {:?}",
            error_msgs(&ctx)
        );
    }

    /// An AR(0) hydro's constant external inflow column is accepted: its
    /// deterministic base is the column's mean, so the constant is the base.
    #[test]
    fn external_constant_inflow_column_without_an_ar_model_is_accepted() {
        let data = external_data(
            vec![0],
            vec![inflow_row(1, 0, 0, 100.0), inflow_row(1, 0, 1, 100.0)],
            vec![],
            vec![],
        );
        assert!(
            data.inflow_ar_coefficients.is_empty(),
            "fixture precondition: hydro 1 is AR(0)"
        );
        let ctx = run(&data);
        assert!(
            !ctx.has_errors(),
            "a σ = 0 AR(0) inflow column must be accepted: {:?}",
            error_msgs(&ctx)
        );
    }

    /// An External-scheme deck with no seasonal-stats file for that class (no
    /// `load_seasonal_stats.parquet`) validates without error — seasonal-stats
    /// files are optional under External.
    #[test]
    fn external_load_deck_validates_without_error_when_seasonal_stats_absent() {
        let load = vec![
            ExternalLoadRow {
                stage_id: 0,
                scenario_id: 0,
                bus_id: EntityId::from(1),
                value_mw: 50.0,
            },
            ExternalLoadRow {
                stage_id: 0,
                scenario_id: 1,
                bus_id: EntityId::from(1),
                value_mw: 50.0,
            },
        ];
        let data = external_data(vec![0], vec![], load, vec![]);
        assert!(
            data.load_seasonal_stats.is_empty(),
            "fixture precondition: no seasonal-stats file declared"
        );
        let ctx = run(&data);
        assert!(
            !ctx.has_errors(),
            "an External load deck with no seasonal-stats file must validate cleanly: {:?}",
            error_msgs(&ctx)
        );
    }

    /// A σ = 0 external NCS column whose values all equal μ is accepted; the
    /// seasonal μ plays no part under External (see the "differs" companion
    /// below).
    #[test]
    fn external_ncs_zero_sigma_accepts_when_value_equals_mean() {
        let ncs = vec![
            ExternalNcsRow {
                stage_id: 0,
                scenario_id: 0,
                ncs_id: EntityId::from(1),
                value: 0.5,
            },
            ExternalNcsRow {
                stage_id: 0,
                scenario_id: 1,
                ncs_id: EntityId::from(1),
                value: 0.5,
            },
        ];
        let mut data = external_data(vec![0], vec![], vec![], ncs);
        data.ncs_models = vec![NcsModel {
            ncs_id: EntityId::from(1),
            stage_id: 0,
            mean: 0.5,
            std: 0.0,
        }];
        let ctx = run(&data);
        assert!(
            !ctx.has_errors(),
            "a deterministic NCS column (every value == μ) with σ = 0 must be accepted: {:?}",
            error_msgs(&ctx)
        );
    }

    /// The same σ = 0 NCS column with one value diverging from μ is likewise
    /// accepted: under External, NCS's μ is defined by the external file
    /// itself, so there is no seasonal μ left to disagree with.
    #[test]
    fn external_ncs_zero_sigma_accepts_when_value_differs_from_seasonal_mean() {
        let ncs = vec![
            ExternalNcsRow {
                stage_id: 0,
                scenario_id: 0,
                ncs_id: EntityId::from(1),
                value: 0.5,
            },
            ExternalNcsRow {
                stage_id: 0,
                scenario_id: 1,
                ncs_id: EntityId::from(1),
                value: 0.9,
            },
        ];
        let mut data = external_data(vec![0], vec![], vec![], ncs);
        data.ncs_models = vec![NcsModel {
            ncs_id: EntityId::from(1),
            stage_id: 0,
            mean: 0.5,
            std: 0.0,
        }];
        let ctx = run(&data);
        assert!(
            !ctx.has_errors(),
            "a σ = 0 NCS column with a value differing from the seasonal μ must be accepted: {:?}",
            error_msgs(&ctx)
        );
    }

    /// A σ = 0 external load library whose values all equal μ is accepted the
    /// same way as NCS.
    #[test]
    fn external_load_zero_sigma_accepts_when_value_equals_mean() {
        let load = vec![
            ExternalLoadRow {
                stage_id: 0,
                scenario_id: 0,
                bus_id: EntityId::from(1),
                value_mw: 50.0,
            },
            ExternalLoadRow {
                stage_id: 0,
                scenario_id: 1,
                bus_id: EntityId::from(1),
                value_mw: 50.0,
            },
        ];
        let mut data = external_data(vec![0], vec![], load, vec![]);
        data.load_seasonal_stats = vec![LoadSeasonalStatsRow {
            bus_id: EntityId::from(1),
            stage_id: 0,
            mean_mw: 50.0,
            std_mw: 0.0,
        }];
        let ctx = run(&data);
        assert!(
            !ctx.has_errors(),
            "a deterministic load column (every value == μ) with σ = 0 must be accepted: {:?}",
            error_msgs(&ctx)
        );
    }

    /// A σ = 0 external load column with a value differing from the seasonal
    /// μ is accepted: under External, load's μ is defined by the external
    /// file itself, so the seasonal μ in `load_seasonal_stats.parquet` does not
    /// constrain a deterministic load value.
    #[test]
    fn external_load_zero_sigma_accepts_when_value_differs_from_seasonal_mean() {
        let load = vec![
            ExternalLoadRow {
                stage_id: 0,
                scenario_id: 0,
                bus_id: EntityId::from(1),
                value_mw: 50.0,
            },
            ExternalLoadRow {
                stage_id: 0,
                scenario_id: 1,
                bus_id: EntityId::from(1),
                value_mw: 50.0,
            },
        ];
        let mut data = external_data(vec![0], vec![], load, vec![]);
        data.load_seasonal_stats = vec![LoadSeasonalStatsRow {
            bus_id: EntityId::from(1),
            stage_id: 0,
            mean_mw: 999.0,
            std_mw: 0.0,
        }];
        let ctx = run(&data);
        assert!(
            !ctx.has_errors(),
            "a σ = 0 load column diverging from a declared seasonal μ must be accepted: {:?}",
            error_msgs(&ctx)
        );
    }

    // ── Rule 12: External scheme suppresses the deterministic-inflow warning ─────

    /// An External inflow class with `std_m3s == 0.0` does not emit rule 12's
    /// warning — the external library never runs PAR generation, so the
    /// "deterministic inflow" framing is meaningless.
    #[test]
    fn rule_12_suppressed_for_external_inflow_scheme() {
        let mut data = make_data_5b(
            vec![make_hydro_ordered_penalties(1)],
            make_stages_5b(vec![0]),
            vec![make_bus_with_deficit(1, 10.0)],
            vec![InflowSeasonalStatsRow {
                hydro_id: EntityId::from(1),
                stage_id: 0,
                mean_m3s: 100.0,
                std_m3s: 0.0,
            }],
            vec![],
            None,
        );
        data.config = config_with_training_external_inflow();
        data.external_scenarios = vec![ExternalScenarioRow {
            hydro_id: EntityId::from(1),
            stage_id: 0,
            scenario_id: 0,
            value_m3s: 10.0,
        }];
        let ctx = run(&data);
        assert!(
            !ctx.warnings()
                .iter()
                .any(|w| w.message.contains("deterministic inflow")),
            "rule 12 must be suppressed for an External inflow class: {:?}",
            ctx.warnings()
        );
    }

    /// The same `std_m3s == 0.0` row under the default (generated) inflow
    /// scheme still emits rule 12's warning.
    #[test]
    fn rule_12_fires_for_generated_inflow_scheme() {
        let data = make_data_5b(
            vec![make_hydro_ordered_penalties(1)],
            make_stages_5b(vec![0]),
            vec![make_bus_with_deficit(1, 10.0)],
            vec![InflowSeasonalStatsRow {
                hydro_id: EntityId::from(1),
                stage_id: 0,
                mean_m3s: 100.0,
                std_m3s: 0.0,
            }],
            vec![],
            None,
        );
        let ctx = run(&data);
        assert!(
            ctx.warnings()
                .iter()
                .any(|w| w.kind == ErrorKind::ModelQuality
                    && w.message.contains("deterministic inflow")),
            "rule 12 must still fire for a generated class: {:?}",
            ctx.warnings()
        );
    }

    /// A valid generated-only study (no external rows, positive σ everywhere)
    /// produces no error and no rule 12 warning.
    #[test]
    fn generated_only_study_with_positive_sigma_has_no_findings() {
        let data = make_data_5b(
            vec![make_hydro_ordered_penalties(1)],
            make_stages_5b(vec![0, 1]),
            vec![make_bus_with_deficit(1, 10.0)],
            vec![
                InflowSeasonalStatsRow {
                    hydro_id: EntityId::from(1),
                    stage_id: 0,
                    mean_m3s: 100.0,
                    std_m3s: 10.0,
                },
                InflowSeasonalStatsRow {
                    hydro_id: EntityId::from(1),
                    stage_id: 1,
                    mean_m3s: 100.0,
                    std_m3s: 10.0,
                },
            ],
            vec![],
            None,
        );
        let ctx = run(&data);
        assert!(
            !ctx.has_errors(),
            "a valid generated-only study must produce no errors: {:?}",
            ctx.errors()
        );
        assert!(
            !ctx.warnings()
                .iter()
                .any(|w| w.message.contains("deterministic inflow")),
            "a nonzero std_m3s must not warn: {:?}",
            ctx.warnings()
        );
    }
}
