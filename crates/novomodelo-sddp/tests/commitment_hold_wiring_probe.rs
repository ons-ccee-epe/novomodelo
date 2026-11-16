//! Guard test for two wiring assumptions the terminal (post-horizon)
//! commitment-hold lane depends on: the anticipated-thermal manifest's
//! slot-identity match keys on a generic `subindex`, and the commitment-hold
//! ring's LP columns / stored cuts ride `col_scale`/`cost_scale` exactly like
//! every other state family.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap
)]

mod common;

use chrono::{NaiveDate, TimeDelta};
use novomodelo_core::entities::thermal::AnticipatedConfig;
use novomodelo_core::{EntityId, System, SystemBuilder};
use novomodelo_io::config::{
    Config, EstimationConfig, ExportsConfig, InflowNonNegativityConfig, InflowNonNegativityMethod,
    ModelingConfig, ParallelismConfig, PolicyConfig, RowSelectionConfig, SimulationConfig,
    StoppingMode, StoppingRuleConfig, TrainingConfig, TrainingSelection, TrainingSolverConfig,
    UpperBoundEvaluationConfig,
};
use novomodelo_io::test_support::{piece_defaults, stage_cuts_payload_defaults};
use novomodelo_io::{
    EntitySlot, GraphManifest, PolicyCutRecord, ProducerBlock, StageCutsPayload, encode_slot_date,
    write_policy_checkpoint,
};
use novomodelo_sddp::{
    BoundaryLoadRequest, SddpError, compare_manifest_slot_identity, load_boundary_cuts,
};

use common::build_setup_in_code;
use common::builders::{BusSpec, StageSpec, ThermalSpec, make_bus, make_stage, make_thermal};

const K_MAX: usize = 2;
const N_STAGES: usize = 3;
const COST_PER_MWH: f64 = 42.0;
const MAX_GEN_MW: f64 = 100.0;

/// One bus, one anticipated thermal (`lead_stages = K_MAX`), zero load, zero
/// hydros — the minimal fixture that gives the anticipated ring a non-empty
/// `n_state` without exercising any other state family.
fn build_system() -> System {
    let bus = make_bus(EntityId(1), BusSpec::default());

    let thermal = make_thermal(
        EntityId(2),
        ThermalSpec {
            name: "T_ant".to_string(),
            bus_id: EntityId(1),
            anticipated_config: Some(AnticipatedConfig::LeadStages(K_MAX as u32)),
            ..Default::default()
        },
    );

    let stage_start = |offset_days: i64| {
        NaiveDate::from_ymd_opt(2024, 1, 1).unwrap() + TimeDelta::days(offset_days)
    };
    let stages: Vec<_> = (0..N_STAGES)
        .map(|i| {
            make_stage(
                i,
                StageSpec {
                    start_date: stage_start(i as i64),
                    end_date: stage_start(i as i64 + 1),
                    ..Default::default()
                },
            )
        })
        .collect();

    novomodelo_sddp::test_support::build_with_resolved_tables(
        || {
            SystemBuilder::new()
                .buses(vec![bus.clone()])
                .thermals(vec![thermal.clone()])
                .stages(stages.clone())
        },
        K_MAX,
        |_, bounds, _| {
            for s in 0..N_STAGES + K_MAX {
                bounds.thermal_block_base_mut(0, s).cost_per_mwh = COST_PER_MWH;
                bounds.thermal_block_base_mut(0, s).max_generation_mw = MAX_GEN_MW;
            }
        },
    )
}

fn build_config() -> Config {
    Config {
        schema: None,
        modeling: ModelingConfig {
            inflow_non_negativity: InflowNonNegativityConfig {
                method: InflowNonNegativityMethod::Penalty,
            },
            cost_scale_factor: None,
        },
        training: TrainingConfig {
            enabled: true,
            tree_seed: Some(42),
            stopping_rules: Some(vec![StoppingRuleConfig::IterationLimit { limit: 1 }]),
            stopping_mode: StoppingMode::Any,
            cut_selection: RowSelectionConfig::default(),
            solver: TrainingSolverConfig::default(),
            parallelism: ParallelismConfig::default(),
            scenario_source: None,
            selection: Some(TrainingSelection::Sampled { forward_passes: 1 }),
        },
        upper_bound_evaluation: UpperBoundEvaluationConfig::default(),
        policy: PolicyConfig::default(),
        simulation: SimulationConfig::default(),
        exports: ExportsConfig::default(),
        estimation: EstimationConfig::default(),
    }
}

/// Invariant 1: `compare_manifest_slot_identity` keys on the plain
/// `(entity_type, entity_id, subindex)` triple — `subindex` is opaque to it, so
/// a window-index slot the commitment block assigns (a numbering unrelated to
/// the ring's own `[0, k_max)` domain) binds under the identical mechanism the
/// anticipated ring already relies on, with no schema change.
#[test]
fn manifest_slot_identity_binds_on_subindex_with_no_schema_change() {
    let system = build_system();
    let setup = build_setup_in_code(build_system(), &build_config());
    let manifest = setup.build_terminal_entity_manifest(&system);

    assert!(
        !manifest.is_empty(),
        "fixture's terminal manifest must be non-empty (n_anticipated=1, k_max={K_MAX})",
    );
    // n_hydros = 0 and no travel-time buckets are declared, so every terminal
    // manifest slot on this fixture belongs to the anticipated-thermal family —
    // there is no storage/lag/bucket slot competing for the entity-type byte.
    let real_slot = manifest[0].clone();

    let window_indexed_slot = EntitySlot::anticipated(real_slot.entity_id, 7, real_slot.was_active)
        .with_interval(real_slot.interval_start, real_slot.interval_end);
    let source = vec![window_indexed_slot.clone()];
    let matching_current = vec![window_indexed_slot.clone()];
    compare_manifest_slot_identity(&source, &matching_current, &mut |_| {})
        .expect("identical (entity_type, entity_id, subindex) slots must bind");

    let mut mismatched_current = matching_current;
    mismatched_current[0].subindex = window_indexed_slot.subindex + 1;
    let err = compare_manifest_slot_identity(&source, &mismatched_current, &mut |_| {}).expect_err(
        "a subindex-only mismatch must be rejected, or a boundary-injected cut could \
             silently attach to the wrong state variable",
    );
    assert!(
        matches!(err, SddpError::Validation(_)),
        "subindex mismatch must reject as SddpError::Validation; got {err:?}",
    );
}

/// Invariant 2: `commit_out` and `commit_in` carry
/// `col_scale == 1.0` on every stage template after construction-time LP
/// postprocessing — the ring's pin/re-read round-trip depends on this.
#[test]
fn anticipated_ring_columns_carry_unit_col_scale() {
    let setup = build_setup_in_code(build_system(), &build_config());
    let state = setup.stage_state();
    let commit_out = state.commit_out.clone();
    let commit_in = state.commit_in.clone();

    assert!(
        !commit_out.is_empty() && !commit_in.is_empty(),
        "fixture must carry a non-empty anticipated ring (n_anticipated=1, k_max={K_MAX})",
    );

    for (stage_idx, template) in setup
        .inputs
        .stage_data
        .stage_templates
        .templates
        .iter()
        .enumerate()
    {
        for c in commit_out.clone() {
            assert_eq!(
                template.col_scale[c], 1.0,
                "stage {stage_idx}: commit_out column {c} must carry col_scale == 1.0",
            );
        }
        for c in commit_in.clone() {
            assert_eq!(
                template.col_scale[c], 1.0,
                "stage {stage_idx}: commit_in column {c} must carry col_scale == 1.0",
            );
        }
    }
}

const SAMPLE_CUTS: [(f64, [f64; 3]); 2] = [(10.0, [1.0, -2.0, 3.5]), (-4.0, [0.0, 7.25, -1.0])];

/// Invariant 3: the intercept and every coefficient of a loaded cut scale by
/// the identical `source / loading_cost_scale_factor` factor — no per-family
/// branch a commitment-block coefficient could fall outside of.
#[test]
fn boundary_load_scales_every_coefficient_uniformly() {
    let tmp = tempfile::tempdir().unwrap();
    let date = novomodelo_sddp::test_support::fixture_priced_date(
        novomodelo_sddp::test_support::ymd(2030, 1, 1),
        0,
    );
    let cuts: Vec<PolicyCutRecord> = SAMPLE_CUTS
        .iter()
        .enumerate()
        .map(|(slot, (intercept, coefficients))| PolicyCutRecord {
            slot_index: slot as u32,
            iteration: 0,
            intercept: *intercept,
            coefficients,
            is_active: true,
            ..piece_defaults()
        })
        .collect();
    let payload = StageCutsPayload {
        stage_id: 0,
        state_dimension: 3,
        capacity: 2,
        cuts: &cuts,
        populated_count: 2,
        entity_manifest: &[],
        cost_scale_factor: 1.0,
        node_id: 0,
        graph_stage_id: 0,
        priced_state_date: encode_slot_date(date),
        ..stage_cuts_payload_defaults()
    };
    let metadata = novomodelo_sddp::test_support::checkpoint_metadata(
        1,
        GraphManifest::default(),
        ProducerBlock {
            completed_iterations: 1,
            max_iterations: 1,
            forward_passes: 1,
            ..novomodelo_sddp::test_support::producer_block()
        },
    );
    write_policy_checkpoint(tmp.path(), &[payload], &[], &metadata, &[]).unwrap();
    let loading_cost_scale_factor = 2_000_000.0;

    let loaded = load_boundary_cuts(&BoundaryLoadRequest::new(
        tmp.path(),
        date,
        3,
        &[],
        loading_cost_scale_factor,
    ))
    .unwrap();

    assert_eq!(loaded.len(), SAMPLE_CUTS.len());
    for ((intercept, coefficients), cut) in SAMPLE_CUTS.iter().zip(loaded.iter()) {
        assert_eq!(cut.intercept, intercept / loading_cost_scale_factor);
        for (before, after) in coefficients.iter().zip(&cut.coefficients) {
            assert_eq!(
                *after,
                before / loading_cost_scale_factor,
                "every coefficient must scale by the identical factor as the intercept",
            );
        }
    }
}
