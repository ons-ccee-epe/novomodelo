//! # novomodelo-io
//!
//! Case directory loading, validation, and result writing for the
//! [Novomodelo](https://github.com/ons-ccee-epe/novomodelo) power systems ecosystem.
//!
//! This crate provides two top-level entry points:
//!
//! - [`load_case`] — reads a case directory and produces a fully-validated
//!   [`novomodelo_core::System`] ready for use by the solver.
//! - [`write_results`] — writes the training result tables, the training
//!   dictionaries, and the training/simulation completion metadata under a
//!   root directory; the remaining artifacts are written by the callers through
//!   the individual [`output`] writer modules.
//!
//! ## Loading pipeline
//!
//! [`load_case`] executes a six-layer validation pipeline:
//!
//! 1. **Structural validation** — checks that required files exist on disk and records
//!    which optional files are present ([`validation::structural`]).
//! 2. **Schema validation** — verifies required fields, types, and value ranges.
//! 3. **Referential integrity** — checks all entity ID cross-references are resolvable.
//! 4. **Dimensional consistency** — cross-file coverage checks (e.g., inflow params
//!    cover all hydros).
//! 5. **Semantic validation** — domain business rules (acyclic cascade, penalty ordering,
//!    PAR stationarity, etc.).
//! 6. **Cross-file resolution and cross-validation** — multi-file consistency checks that
//!    span the parsed data assembled by earlier layers (productivity source conflict
//!    detection, scalar-parameter hydro-ID existence, per-stage length checks).
//!
//! All validation diagnostics are collected by [`validation::ValidationContext`] before
//! failing, so users see every problem in a single report.  Final errors are reported
//! via [`LoadError`], which carries enough context for diagnostic messages without
//! re-reading input files.

// Intentional for maintainer docs with --document-private-items.
#![allow(rustdoc::private_intra_doc_links)]

#[cfg(feature = "schema")]
pub mod schema;

pub mod broadcast;
pub mod config;
pub mod constraints;
pub mod error;
pub(crate) mod expression;
pub mod extensions;
pub mod initial_conditions;
pub mod output;
pub(crate) mod parquet_helpers;
pub mod penalties;
pub(crate) mod pipeline;
pub(crate) mod post_study_stages;
pub mod report;
pub mod resolution;
pub mod scenarios;
pub mod stage_resolve;
pub mod stages;
pub mod system;
#[cfg(any(test, feature = "test-support"))]
pub mod test_support;
pub mod validation;
pub(crate) mod windowed_history;

pub use broadcast::{BroadcastComputedParameter, BroadcastParameterKind, BroadcastScalarParameter};
pub use config::{
    BoundaryPolicy, Config, EstimationConfig, OrderSelectionMethod, PolicyMode, parse_config,
};
pub use constraints::{
    AvailabilityError, BusPenaltyOverrideRow, ContractBoundsRow, GenericConstraintBoundsRow,
    HydroBoundsRow, HydroPenaltyOverrideRow, LineBoundsRow, LineBusPairIndex,
    LinePenaltyOverrideRow, NcsPenaltyOverrideRow, PumpingBoundsRow, ThermalBoundsRow,
    build_line_bus_pair_index, load_contract_bounds, load_generic_constraint_bounds,
    load_generic_constraints, load_hydro_bounds, load_line_bounds, load_penalty_overrides_bus,
    load_penalty_overrides_hydro, load_penalty_overrides_line, load_penalty_overrides_ncs,
    load_pumping_bounds, load_thermal_bounds, parse_contract_bounds,
    parse_generic_constraint_bounds, parse_generic_constraints, parse_hydro_bounds,
    parse_line_bounds, parse_penalty_overrides_bus, parse_penalty_overrides_hydro,
    parse_penalty_overrides_line, parse_penalty_overrides_ncs, parse_pumping_bounds,
    parse_thermal_bounds, resolve_hydro_availability,
};
pub use error::LoadError;
pub use extensions::{
    EvaporationModelRow, FittingWindow, FphaColumnLayout, FphaDeviationPointRow, FphaHyperplaneRow,
    HydroEnergyProductivityRow, HydroGeometryRow, HydroReferenceVolumeFractions,
    PlaneReductionConfig, ProductionModelConfig, ProductionModelFile, SeasonConfig, SelectionMode,
    StageRange, build_hydro_reference_volumes_resolved, load_fpha_hyperplanes,
    load_hydro_energy_productivity, load_hydro_geometry, load_production_models,
    parse_evaporation_models, parse_fpha_deviation_points, parse_fpha_hyperplanes,
    parse_hydro_energy_productivity, parse_hydro_geometry, parse_production_models,
    parse_scalar_parameters_json,
};
pub use initial_conditions::parse_initial_conditions;
pub use output::policy::codec::{deserialize_checkpoint_manifest, serialize_checkpoint_manifest};
pub use output::policy::records::CheckpointManifest;
pub use output::policy::{
    BasisEntry, ENTITY_SLOT_DATE_SENTINEL, EntitySlot, FORMAT_VERSION, GraphManifest,
    HydroSeasonOrders, ManifestEdge, ManifestNode, OwnedBasisEntry, OwnedPolicyBasisRecord,
    OwnedPolicyCutRecord, PolicyBasisRecord, PolicyCheckpoint, PolicyCutRecord, ProducerBlock,
    ResolvedCheckpoint, SEASON_CYCLE_CODE_ABSENT, SEASON_CYCLE_CODE_CUSTOM,
    SEASON_CYCLE_CODE_MONTHLY, SEASON_CYCLE_CODE_WEEKLY, STAGE_CUTS_GRAPH_STAGE_ID_SENTINEL,
    STAGE_CUTS_NODE_ID_SENTINEL, STAGE_CUTS_PRICED_STATE_DATE_SENTINEL,
    STAGE_STATES_NODE_ID_SENTINEL, STATE_HISTORY_SCOPE_NONE, STATE_HISTORY_SCOPE_WHOLE,
    STATE_HISTORY_SCOPE_WINDOW, SeasonManifest, StageCutsPayload, StageCutsReadResult,
    StageStatesPayload, StageStatesReadResult, StateFamily, StateHistory,
    check_checkpoint_payloads, decode_slot_date, deserialize_stage_basis, deserialize_stage_cuts,
    deserialize_stage_states, encode_slot_date, read_policy_checkpoint, resolve_policy_checkpoint,
    serialize_stage_basis, serialize_stage_cuts, serialize_stage_states, write_policy_checkpoint,
};
pub use output::{
    DeviationSummary, DeviationWorstEntry, DistributionInfo, EVAPORATION_MODELS_FILE,
    FPHA_DEVIATION_POINTS_FILE, FPHA_HYPERPLANES_FILE, FixedDeliveryRow,
    GENERIC_CONSTRAINT_ECHO_FILE, GenericConstraintEchoRow, HostLayout, IterationRecord,
    MetadataBounds, MetadataConfiguration, MetadataConvergence, MetadataCost, MetadataIterations,
    MetadataProblemDimensions, MetadataRowPool, MetadataScenarios, MetadataSimulationSolveStats,
    MetadataTrainingSolveStats, OutputContext, OutputError, RowPoolStatistics, RowSelectionRecord,
    RunStatus, SOFTWARE_NAME, SOFTWARE_VERSION, SetupTimings, SimulationMetadata, SimulationOutput,
    SoftwareIdentity, SolverStatsRow, TrainingMetadata, TrainingOutput, TrainingParquetWriter,
    WorkerTimingRecord, get_hostname, now_iso8601, policy_checkpoint_remedy,
    read_simulation_metadata, read_training_metadata, remove_conditional_training_outputs,
    remove_simulation_outputs, remove_success_marker, simulation_family_subpaths,
    write_dictionaries, write_evaporation_models, write_fixed_delivery,
    write_fpha_deviation_points, write_fpha_hyperplanes, write_generic_constraint_echo,
    write_hydro_model_summary, write_provenance_report, write_results, write_row_selection_records,
    write_scaling_report, write_simulation_metadata, write_simulation_results,
    write_simulation_solver_stats, write_skipped_simulation_results, write_solver_stats,
    write_success_marker, write_training_metadata, write_training_results,
};
pub use penalties::parse_penalties;
pub use report::{ReportEntry, ValidationReport, generate_report};
pub use resolution::{resolve_bounds, resolve_penalties};
pub use scenarios::{
    BlockFactor, ExternalLoadRow, ExternalNcsRow, ExternalScenarioRow, InflowArCoefficientRow,
    InflowHistoryRow, InflowSeasonalStatsRow, LoadFactorEntry, LoadSeasonalStatsRow,
    NoiseOpeningRow, assemble_inflow_models, assemble_load_models, load_correlation,
    load_external_inflow_scenarios, load_external_load_scenarios, load_external_ncs_scenarios,
    load_inflow_ar_coefficients, load_inflow_history, load_inflow_seasonal_stats,
    load_load_factors, load_load_seasonal_stats, load_noise_openings, parse_correlation,
    parse_external_inflow_scenarios, parse_external_load_scenarios, parse_external_ncs_scenarios,
    parse_inflow_ar_coefficients, parse_inflow_history, parse_inflow_seasonal_stats,
    parse_load_factors, parse_load_seasonal_stats,
};
pub use stage_resolve::StageIdResolver;
pub use stages::{StagesData, parse_stages};
pub use system::{
    load_energy_contracts, load_lines, load_non_controllable_sources, load_pumping_stations,
    load_thermals, parse_buses, parse_energy_contracts, parse_hydros, parse_lines,
    parse_non_controllable_sources, parse_pumping_stations, parse_thermals,
};
pub use validation::scalar_parameters::validate_scalar_parameters;
pub use validation::semantic::seed_lag_state_depth;
pub use validation::structural::{FileManifest, InputFile, validate_structure};
pub use validation::{ErrorKind, Severity, ValidationContext, ValidationEntry};

use novomodelo_core::{ScalarParameter, System};
use std::path::Path;

/// Auxiliary rows produced by the load pipeline alongside [`System`].
///
/// `CaseArtifacts` is the single-source delivery of the already-parsed-and-validated
/// parquet/JSON rows, so downstream solver crates do not re-open the same files
/// from disk after [`load_case`] returns.
///
/// Each field is a row table carrying its own key column (e.g. `hydro_id`),
/// ordered per that row type's own parser, not by a shared entity index.
/// Empty vectors indicate the optional file was absent on disk.
#[derive(Debug, Clone, Default)]
pub struct CaseArtifacts {
    /// File-presence manifest produced by Layer 1 (structural). Lets
    /// downstream code avoid re-running `validate_structure` to check
    /// optional-file presence.
    pub file_manifest: FileManifest,

    /// Rows from `system/hydro_geometry.parquet`.
    pub hydro_geometry: Vec<extensions::HydroGeometryRow>,

    /// Entries from `system/hydro_production_models.json`.
    pub production_models: Vec<extensions::ProductionModelConfig>,

    /// File-level FPHA plane-reduction block from
    /// `system/hydro_production_models.json`. `None` when the file is absent or
    /// carries no `fpha_plane_reduction` key. Carried for the post-fit
    /// plane-reduction pass; no behavior depends on it yet.
    pub plane_reduction: Option<extensions::PlaneReductionConfig>,

    /// Rows from `system/hydro_energy_productivity.parquet`.
    pub hydro_energy_productivity: Vec<extensions::HydroEnergyProductivityRow>,

    /// Rows from `system/fpha_hyperplanes.parquet`.
    pub fpha_hyperplanes: Vec<extensions::FphaHyperplaneRow>,

    /// Assembled scalar parameters from `constraints/generic_parameters.json`.
    pub scalar_parameters: Vec<ScalarParameter>,

    /// The data, `if()` selectors, conditions and endpoint shifts the generic constraints of
    /// `constraints/generic_constraints.json` read per scenario; empty when no constraint reads
    /// scenario data.
    pub generic_scenario: constraints::ScenarioLayer,

    /// Rows from `system/tailrace_curves.parquet`.
    pub tailrace_curves: Vec<extensions::TailraceCurveRow>,
}

/// Fully-loaded case bundle: the validated [`System`] plus the auxiliary
/// row sets that downstream consumers need without re-reading the case
/// directory.
#[derive(Debug)]
pub struct LoadedCase {
    /// Validated, ready-to-solve system.
    pub system: System,
    /// Auxiliary rows (parsed and validated by the load pipeline).
    pub artifacts: CaseArtifacts,
}

/// Load a case directory and return a fully-validated [`System`].
///
/// `path` must point to the root case directory containing `config.json` and the
/// standard subdirectories (`system/`, `scenarios/`, `constraints/`, `policy/`).
///
/// The function executes a six-layer validation pipeline; see the
/// [crate-level docs](crate) for the layer-by-layer breakdown.
///
/// After all layers pass, three-tier penalty/bound resolution and scenario assembly
/// are performed before constructing the [`System`].
///
/// Warnings collected during validation are silently discarded. Use [`validate_case`]
/// when you need to inspect or display warnings alongside the loaded [`System`].
///
/// Prefer [`load_case_with_artifacts`] when the downstream consumer also needs the
/// auxiliary parquet/JSON rows this pipeline already parsed: it returns them as a
/// [`CaseArtifacts`] bundle so downstream code can skip the duplicate disk reads.
///
/// # Errors
///
/// - [`LoadError::IoError`] — a required file is missing or cannot be read.
/// - [`LoadError::ParseError`] — a file contains malformed JSON or invalid Parquet.
/// - [`LoadError::SchemaError`] — a domain constraint violation detected
///   post-deserialization (e.g., AR coefficient count mismatch).
/// - [`LoadError::ConstraintError`] — one or more validation errors collected
///   across Layers 1-5, or `SystemBuilder` rejected the assembled data.
pub fn load_case(path: &Path) -> Result<System, LoadError> {
    pipeline::run_pipeline_with_artifacts(path).map(|(loaded, _report)| loaded.system)
}

/// Load a case directory and return the validated [`System`] together with
/// the [`CaseArtifacts`] bundle of pre-parsed auxiliary rows.
///
/// This is the preferred entry point for solver pipelines that need the
/// production-model / hydro-geometry / FPHA hyperplane / scalar-parameter
/// rows: returning them here avoids the duplicate disk re-reads and parallel
/// validation paths in downstream crates.
///
/// The function runs the six-layer validation pipeline described in [`load_case`].
///
/// # Errors
///
/// Same error conditions as [`load_case`].
pub fn load_case_with_artifacts(path: &Path) -> Result<LoadedCase, LoadError> {
    pipeline::run_pipeline_with_artifacts(path).map(|(loaded, _report)| loaded)
}

/// Load a case directory and return both the fully-validated [`System`] and a
/// [`ValidationReport`] containing all warnings collected during the pipeline.
///
/// This function runs the same six-layer validation pipeline as [`load_case`] but
/// preserves warnings so that callers can display them to the user. Errors still
/// cause the function to return `Err`; warnings never block loading.
///
/// # Errors
///
/// Same error conditions as [`load_case`].
pub fn validate_case(path: &Path) -> Result<(System, ValidationReport), LoadError> {
    pipeline::run_pipeline_with_artifacts(path).map(|(loaded, report)| (loaded.system, report))
}

/// Load a case directory and return the validated [`LoadedCase`] together with a
/// [`ValidationReport`] containing all warnings collected during the pipeline.
///
/// This is the preferred entry point for callers that need both the auxiliary
/// [`CaseArtifacts`] bundle (for downstream prep phases such as
/// `prepare_hydro_models_from_artifacts`) **and** the warning report.
///
/// The function runs the same six-layer validation pipeline as [`load_case`]. Errors
/// still cause the function to return `Err`; warnings never block loading.
///
/// # Errors
///
/// Same error conditions as [`load_case`].
pub fn validate_case_with_artifacts(
    path: &Path,
) -> Result<(LoadedCase, ValidationReport), LoadError> {
    pipeline::run_pipeline_with_artifacts(path)
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]
mod tests {
    use crate::constraints::scenario::{ConstraintScenario, DataAtom};
    use crate::test_support::{make_minimal_case, write_file, write_parquet};
    use crate::{CaseArtifacts, InputFile, LoadError, LoadedCase, load_case_with_artifacts};
    use arrow::array::{ArrayRef, Float64Array, Int32Array};
    use arrow::record_batch::RecordBatch;
    use novomodelo_core::resolved::GenericConstraintBoundEntry;
    use novomodelo_core::{EntityId, LinearTerm, ScalarExpr, VariableRef};
    use std::sync::Arc;
    use tempfile::TempDir;

    #[test]
    fn case_artifacts_plane_reduction_defaults_to_none() {
        let artifacts = CaseArtifacts::default();
        assert!(artifacts.plane_reduction.is_none());
    }

    #[test]
    fn a_generic_case_carries_an_empty_scenario_layer() {
        let case = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../examples/deterministic/d13-generic-constraint");
        let file: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(case.join("constraints/generic_constraints.json"))
                .expect("the constraints file reads"),
        )
        .expect("the constraints file is JSON");
        let LoadedCase { system, artifacts } =
            load_case_with_artifacts(&case).expect("the case loads");
        assert!(artifacts.generic_scenario.is_empty());
        assert_eq!(
            system.generic_constraints().len(),
            file["constraints"]
                .as_array()
                .expect("a constraints array")
                .len()
        );
    }

    const THERMALS_JSON: &str = r#"{ "thermals": [
        { "id": 2, "name": "T2", "operational_start_date": "2024-01-01", "bus_id": 1,
          "cost_per_mwh": 10.0, "generation": { "min_mw": 0.0, "max_mw": 100.0 } },
        { "id": 3, "name": "T3", "operational_start_date": "2024-01-01", "bus_id": 1,
          "cost_per_mwh": 20.0, "generation": { "min_mw": 0.0, "max_mw": 100.0 } }
    ] }"#;

    fn constraint_error(dir: &TempDir) -> String {
        match load_case_with_artifacts(dir.path()) {
            Err(LoadError::ConstraintError { description }) => description,
            other => panic!("expected a constraint error, got {other:?}"),
        }
    }

    const GENERIC_FILE: &str = "constraints/generic_constraints.json";

    /// A case with thermals 2 and 3, `generic` as its generic constraints file and a bounds table
    /// with one stage-0 row per `(constraint_id, bound_upper)` of `rows`.
    fn generic_case(generic: &str, rows: &[(i32, Option<f64>)]) -> TempDir {
        let dir = TempDir::new().expect("temp dir");
        make_minimal_case(&dir);
        write_file(dir.path(), "system/thermals.json", THERMALS_JSON);
        write_file(dir.path(), GENERIC_FILE, generic);
        let (ids, uppers): (Vec<i32>, Vec<Option<f64>>) = rows.iter().copied().unzip();
        let batch = RecordBatch::try_from_iter([
            ("constraint_id", Arc::new(Int32Array::from(ids)) as ArrayRef),
            (
                "stage_id",
                Arc::new(Int32Array::from(vec![0; rows.len()])) as ArrayRef,
            ),
            (
                "bound_upper",
                Arc::new(Float64Array::from(uppers)) as ArrayRef,
            ),
        ])
        .expect("bounds batch");
        std::fs::copy(
            write_parquet(&batch).path(),
            dir.path()
                .join("constraints/generic_constraint_bounds.parquet"),
        )
        .expect("write the bounds table");
        dir
    }

    /// derived: no parameter is declared, so the load of bus 1 both conditions read is id 0 and
    /// the `if()` selector is id 1.
    #[test]
    fn a_case_carries_its_generic_constraints_and_their_scenario_layer() {
        let dir = generic_case(
            r#"{ "constraints": [
                { "id": 405, "name": "constraint_405",
                  "expression": "@FNESE <= if(bus_load_value(1) <= 85000, 100, 200)",
                  "active_when": "82000 < bus_load_value(1) & bus_load_value(1) <= 95000",
                  "slack": { "enabled": false } },
                { "id": 401, "name": "constraint_401", "expression": "thermal_generation(2)",
                  "slack": { "enabled": false } }
            ],
            "expressions": [{ "name": "FNESE",
                              "expression": "thermal_generation(2) - thermal_generation(3)" }] }"#,
            &[(401, Some(6300.0)), (405, None)],
        );
        let LoadedCase { system, artifacts } =
            load_case_with_artifacts(dir.path()).expect("the case loads");
        for file in [
            InputFile::ConstraintsGenericConstraintsJson,
            InputFile::ConstraintsGenericConstraintBoundsParquet,
        ] {
            assert!(artifacts.file_manifest.present(file), "{file:?}");
        }

        let thermal = |id: i32| VariableRef::ThermalGeneration {
            thermal_id: EntityId(id),
            block_id: None,
        };
        let constraints = system.generic_constraints();
        let ids: Vec<EntityId> = constraints.iter().map(|constraint| constraint.id).collect();
        assert_eq!(ids, [EntityId(401), EntityId(405)]);
        assert_eq!(
            constraints[0].expression.terms,
            [LinearTerm::literal(1.0, thermal(2))]
        );
        assert_eq!(
            constraints[1].expression.terms,
            [
                LinearTerm::literal(1.0, thermal(2)),
                LinearTerm::literal(-1.0, thermal(3)),
            ]
        );
        assert_eq!(
            constraints[1].bound_upper,
            Some(ScalarExpr::literal(0.0) + ScalarExpr::param(EntityId(1)))
        );
        assert_eq!(
            system.resolved_generic_bounds().bounds_for_stage(0, 0),
            [GenericConstraintBoundEntry {
                block_id: None,
                bound_lower: None,
                bound_upper: Some(6300.0),
            }]
        );

        let layer = &artifacts.generic_scenario;
        assert_eq!(
            layer.data_atoms,
            [(
                EntityId(0),
                DataAtom::BusLoadValue {
                    bus_id: EntityId(1),
                    block_id: None,
                }
            )]
        );
        assert_eq!(
            layer
                .selectors
                .iter()
                .map(|(id, _)| *id)
                .collect::<Vec<_>>(),
            [EntityId(1)]
        );
        let [
            ConstraintScenario {
                constraint_id: EntityId(405),
                active_when: Some(_),
                endpoint_shift: None,
            },
        ] = layer.constraints.as_slice()
        else {
            panic!(
                "expected the condition of 405, got: {:?}",
                layer.constraints
            );
        };
    }

    #[test]
    fn a_generic_constraint_on_an_undeclared_plant_fails_the_load() {
        let json = r#"{ "constraints": [
            { "id": 701, "name": "constraint_701", "expression": "hydro_generation(14) >= 10",
              "slack": { "enabled": false } },
            { "id": 402, "name": "constraint_402", "expression": "hydro_generation(14)",
              "slack": { "enabled": false } }
        ] }"#;
        let expected: Vec<String> = [402, 701]
            .map(|id| {
                let label = format!("GenericConstraint {id} term[0]");
                format!(
                    "[InvalidReference] {GENERIC_FILE} ({label}): {label} references \
                     non-existent Hydro 14"
                )
            })
            .into();
        assert_eq!(
            constraint_error(&generic_case(json, &[(402, Some(6300.0)), (701, None)])),
            expected.join("\n")
        );
    }

    #[test]
    fn an_active_when_that_does_not_parse_fails_the_load() {
        let condition = "if unparseable (";
        let json = format!(
            r#"{{ "constraints": [
                {{ "id": 401, "name": "constraint_401", "expression": "thermal_generation(2)",
                   "slack": {{ "enabled": false }} }},
                {{ "id": 400, "name": "constraint_400", "expression": "thermal_generation(3) <= 100",
                   "active_when": "{condition}", "slack": {{ "enabled": false }} }}
            ] }}"#
        );
        let diagnostic = crate::expression::parser::parse_condition(condition)
            .expect_err("the condition does not parse");
        assert_eq!(
            constraint_error(&generic_case(&json, &[(401, Some(6300.0)), (400, None)])),
            format!(
                "[SchemaViolation] {GENERIC_FILE}: field constraints[1].active_when: {}",
                diagnostic.render(condition)
            )
        );
    }

    #[test]
    fn an_active_when_reading_a_decision_fails_the_load() {
        let json = r#"{ "constraints": [
            { "id": 401, "name": "constraint_401", "expression": "thermal_generation(2)",
              "active_when": "hydro_generation(7) > 100", "slack": { "enabled": false } }
        ] }"#;
        assert_eq!(
            constraint_error(&generic_case(json, &[(401, Some(6300.0))])),
            format!(
                "[SchemaViolation] {GENERIC_FILE}: field constraints[0].active_when: \
                 `hydro_generation(7)`: an LP variable cannot appear in a scalar-parameter \
                 expression"
            )
        );
    }

    #[test]
    #[ignore = "needs a converted case; set NOVOMODELO_CONVERTED_CASE to its directory"]
    fn the_converted_case_carries_its_scenario_layer() {
        let dir = std::env::var("NOVOMODELO_CONVERTED_CASE").expect("NOVOMODELO_CONVERTED_CASE");
        let LoadedCase { artifacts, .. } =
            load_case_with_artifacts(std::path::Path::new(&dir)).expect("the case loads");
        assert!(!artifacts.generic_scenario.constraints.is_empty());
    }
}
