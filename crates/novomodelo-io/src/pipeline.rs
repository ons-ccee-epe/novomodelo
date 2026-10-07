//! Loading pipeline: validation, resolution, scenario assembly, and
//! `SystemBuilder::build` for a case directory.

use std::path::Path;

use chrono::NaiveDate;
use cobre_core::SystemBuilder;

use crate::{
    CaseArtifacts, LoadError, LoadedCase, StageIdResolver,
    extensions::load_tailrace_curves,
    report::{ValidationReport, generate_report},
    resolution::{
        BoundsEntitySlices, BoundsOverrides, PenaltiesEntitySlices, PenaltiesOverrides,
        resolve_bounds, resolve_generic_constraint_bounds, resolve_hydro_unit_group_bounds,
        resolve_load_factors, resolve_ncs_bounds, resolve_ncs_factors, resolve_penalties,
    },
    scenarios::assembly::{assemble_inflow_models, assemble_load_models},
    scenarios::residual_derivation::{populate_derived_residual_ratios, resolve_stage_seasons},
    stages::normalize_out_edge_probabilities,
    validation::{
        ValidationContext,
        dimensional::validate_dimensional_consistency,
        productivity_resolution::validate_productivity_resolution,
        referential::validate_referential_integrity,
        scalar_parameters::validate_scalar_parameters,
        schema::validate_schema,
        semantic::{validate_semantic_hydro_thermal, validate_semantic_stages_penalties_scenarios},
        structural::{InputFile, validate_structure},
    },
};

/// The canonical pipeline: returns the validated [`System`](cobre_core::System) in a
/// [`LoadedCase`] bundle with the [`CaseArtifacts`] rows and the [`ValidationReport`].
///
/// # Errors
///
/// - [`LoadError::IoError`] / [`LoadError::ParseError`] — file read or JSON/Parquet
///   parse failure.
/// - [`LoadError::ConstraintError`] — collected validation errors or
///   `SystemBuilder::build` rejection.
/// - [`LoadError::SchemaError`] — AR coefficient count mismatch in scenario assembly.
// Rationale: a single linear cascade where each stage's output feeds the next; splitting at layer
// boundaries would scatter the global ordering guarantee across call sites.
#[allow(clippy::too_many_lines)]
pub(crate) fn run_pipeline_with_artifacts(
    path: &Path,
) -> Result<(LoadedCase, ValidationReport), LoadError> {
    let mut ctx = ValidationContext::new();

    let manifest = validate_structure(path, &mut ctx);

    let Some(mut data) = validate_schema(path, &manifest, &mut ctx) else {
        return ctx.into_result().and(Err(LoadError::ConstraintError {
            description: "schema validation failed but no errors were collected".to_string(),
        }));
    };

    validate_referential_integrity(&data, &mut ctx);
    validate_dimensional_consistency(&data, &mut ctx);
    validate_semantic_hydro_thermal(&data, &mut ctx);
    validate_semantic_stages_penalties_scenarios(&data, &mut ctx);

    validate_productivity_resolution(&data, &mut ctx);
    // Pre-study stages have negative IDs; this study-stage count must match what
    // `build_resolved_parameters` consumes downstream.
    let study_stages: Vec<_> = data.stages.stages.iter().filter(|s| s.id >= 0).collect();
    let n_stages = study_stages.len();
    validate_scalar_parameters(&data.scalar_parameters, &data.hydros, n_stages, &mut ctx);

    let report = generate_report(&ctx);
    ctx.into_result()?;

    // Normalize once here — after the semantic prob-sum gate and before the graph
    // reaches any bound; ahead of validation a grossly-wrong vector would normalize
    // silently instead of being rejected.
    normalize_out_edge_probabilities(&mut data.stages.policy_graph, &path.join("stages.json"))?;

    // Every resolver below indexes its output table by position in these slices
    // (`entity_idx`), which must already equal the position `SystemBuilder::build`'s
    // `sort_canonical` assigns; sorting only after validation keeps validation's
    // error order on the parsed (id) order, unaffected by this resort.
    sort_into_canonical_order(&mut data.buses, |b| b.operational_start_date, |b| b.id.0);
    sort_into_canonical_order(&mut data.lines, |l| l.operational_start_date, |l| l.id.0);
    sort_into_canonical_order(&mut data.hydros, |h| h.operational_start_date, |h| h.id.0);
    sort_into_canonical_order(&mut data.thermals, |t| t.operational_start_date, |t| t.id.0);
    sort_into_canonical_order(
        &mut data.pumping_stations,
        |p| p.operational_start_date,
        |p| p.id.0,
    );
    sort_into_canonical_order(
        &mut data.energy_contracts,
        |c| c.operational_start_date,
        |c| c.id.0,
    );
    sort_into_canonical_order(
        &mut data.non_controllable_sources,
        |n| n.operational_start_date,
        |n| n.id.0,
    );

    let study_stage_ids: Vec<i32> = study_stages.iter().map(|s| s.id).collect();
    let stage_resolver = StageIdResolver::from_study_stage_ids(&study_stage_ids);
    let stage_index = stage_resolver.index_map();

    let penalties = resolve_penalties(
        &PenaltiesEntitySlices {
            hydros: &data.hydros,
            buses: &data.buses,
            lines: &data.lines,
            ncs_sources: &data.non_controllable_sources,
        },
        n_stages,
        stage_index,
        &PenaltiesOverrides {
            hydro: &data.penalty_overrides_hydro,
            bus: &data.penalty_overrides_bus,
            line: &data.penalty_overrides_line,
            ncs: &data.penalty_overrides_ncs,
        },
    );

    let k_max: usize = data
        .thermals
        .iter()
        .filter_map(|t| t.anticipated_config.as_ref())
        .map(|c| {
            // A LeadTime plant contributes 0 here: this crate cannot compute the
            // resolver-derived ring depth (that's downstream, solver-side state), and
            // the delivery-anchored ring only ever reads a plant's bounds at
            // delivery_stage < n_stages, so the padded [n_stages, n_stages + k_max)
            // region resolve_bounds sizes from this value is unread for anticipated
            // plants regardless of lead mode.
            usize::try_from(c.lead_stages().unwrap_or(0)).unwrap_or(usize::MAX)
        })
        .max()
        .unwrap_or(0);
    let blocks_per_stage: Vec<usize> = study_stages.iter().map(|s| s.blocks.len()).collect();
    let mut bounds = resolve_bounds(
        &BoundsEntitySlices {
            hydros: &data.hydros,
            thermals: &data.thermals,
            lines: &data.lines,
            pumping_stations: &data.pumping_stations,
            contracts: &data.energy_contracts,
        },
        n_stages,
        k_max,
        stage_index,
        &BoundsOverrides {
            hydro: &data.hydro_bounds,
            thermal: &data.thermal_bounds,
            line: &data.line_bounds,
            pumping: &data.pumping_bounds,
            contract: &data.contract_bounds,
        },
        &blocks_per_stage,
    );
    bounds.set_group_overlay(resolve_hydro_unit_group_bounds(
        &data.hydros,
        n_stages,
        stage_index,
        &data.hydro_unit_group_bounds,
        &blocks_per_stage,
    ));

    let resolved_generic_bounds = resolve_generic_constraint_bounds(
        &data.generic_constraints,
        &data.generic_constraint_bounds,
    );

    let resolved_load_factors =
        resolve_load_factors(&data.load_factors, &data.buses, &data.stages.stages);

    let resolved_ncs_bounds = resolve_ncs_bounds(
        &data.ncs_bounds,
        &data.non_controllable_sources,
        n_stages,
        stage_index,
    );

    let resolved_ncs_factors = resolve_ncs_factors(
        &data.non_controllable_factors,
        &data.non_controllable_sources,
        &data.stages.stages,
    );

    let mut inflow_models = assemble_inflow_models(
        data.inflow_seasonal_stats,
        data.inflow_ar_coefficients,
        data.inflow_annual_components,
    )?;
    let (stage_to_season, n_seasons) = resolve_stage_seasons(
        &data.stages.stages,
        data.stages.policy_graph.season_map.as_ref(),
    );
    populate_derived_residual_ratios(&mut inflow_models, &stage_to_season, n_seasons)?;
    let load_models = assemble_load_models(data.load_seasonal_stats);

    // Without tailrace curves the fit collapses to the constant entity tailrace,
    // which zeroes γ_S and emits sub-ULP LP coefficients.
    let tailrace_curves = if manifest.present(InputFile::SystemTailraceCurvesParquet) {
        let tailrace_path = path.join("system").join("tailrace_curves.parquet");
        load_tailrace_curves(Some(tailrace_path.as_path()))?
    } else {
        Vec::new()
    };

    let artifacts = CaseArtifacts {
        file_manifest: manifest,
        hydro_geometry: data.hydro_geometry,
        production_models: data.production_models,
        plane_reduction: data.plane_reduction,
        hydro_energy_productivity: data.hydro_energy_productivity_rows,
        fpha_hyperplanes: data.fpha_hyperplanes,
        scalar_parameters: data.scalar_parameters,
        tailrace_curves,
    };

    let system = SystemBuilder::new()
        .buses(data.buses)
        .lines(data.lines)
        .hydros(data.hydros)
        .thermals(data.thermals)
        .pumping_stations(data.pumping_stations)
        .contracts(data.energy_contracts)
        .non_controllable_sources(data.non_controllable_sources)
        .stages(data.stages.stages)
        .policy_graph(data.stages.policy_graph)
        .penalties(penalties)
        .bounds(bounds)
        .resolved_generic_bounds(resolved_generic_bounds)
        .resolved_load_factors(resolved_load_factors)
        .resolved_ncs_bounds(resolved_ncs_bounds)
        .resolved_ncs_factors(resolved_ncs_factors)
        .inflow_models(inflow_models)
        .load_models(load_models)
        .ncs_models(data.ncs_models)
        .correlation(data.correlation.unwrap_or_default())
        .initial_conditions(data.initial_conditions)
        .generic_constraints(data.generic_constraints)
        .inflow_history(data.inflow_history)
        .external_scenarios(data.external_scenarios)
        .external_load_scenarios(data.external_load_scenarios)
        .external_ncs_scenarios(data.external_ncs_scenarios)
        .post_study_stages(data.post_study_stages)
        .build()
        .map_err(|errs| LoadError::ConstraintError {
            description: errs
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join("\n"),
        })?;

    Ok((LoadedCase { system, artifacts }, report))
}

/// Sorts `entities` into the order
/// [`SystemBuilder::build`](cobre_core::SystemBuilder::build) establishes —
/// not `(id, date)` or `id` alone — so a resolver's `entity_idx` matches the
/// position `System` will expose it at.
fn sort_into_canonical_order<T>(
    entities: &mut [T],
    date: impl Fn(&T) -> NaiveDate,
    id: impl Fn(&T) -> i32,
) {
    entities.sort_by_key(|e| (date(e), id(e)));
}
