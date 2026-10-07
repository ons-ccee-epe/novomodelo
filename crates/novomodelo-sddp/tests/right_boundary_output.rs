//! Integration coverage for the per-lane post-horizon commitment output
//! (`simulation/anticipated_lanes`): the deposited decision and the ring's
//! carried value, keyed `(thermal_id, delivery_date)`, riding the shared
//! per-scenario simulation writer.
//!
//! The fixture mirrors `right_boundary_cost_semantics.rs`'s minimal shape:
//! zero hydros, one bus, one anticipated thermal declared `LeadTime` over two
//! study stages. Its OWN in-study self-delivery (stage 1, decided at stage 0)
//! keeps `lead_stages >= 1` alive so the in-study ring sizes
//! non-degenerately; the ONE declared `post_study_stages` stage (decider
//! stage 1) is where the ring deposits its carried commitment (`dest = 0`).

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::too_many_lines
)]

use chrono::NaiveDate;
use cobre_core::entities::thermal::AnticipatedConfig;
use cobre_core::{
    BoundsCountsSpec, BoundsDefaults, ContractBlockBounds, EntityId, HydroBlockBounds,
    HydroStageBounds, InitialConditions, LineBlockBounds, PostStudyStage, PostStudyStages,
    PostStudyThermalBound, PumpingBlockBounds, ResolvedBounds, System, SystemBuilder,
    ThermalBlockBounds, ThermalStageBounds,
};
use cobre_io::config::SimulationSelection;
use cobre_io::output::simulation_writer::{ScenarioWritePayload, SimulationParquetWriter};
use cobre_sddp::{SimulationScenarioResult, StudySetup};

mod common;
use common::build_setup_in_code;
use common::builders::{BusSpec, StageSpec, ThermalSpec, make_bus, make_stage, make_thermal};
use common::run_simulation;

const BUS_ID: EntityId = EntityId(1);
const THERMAL_ID: EntityId = EntityId(2);
/// `LeadTime` delta (hours), resolved over the extended delivery axis whose
/// cumulative stage boundaries are `[0, 744, 1464, 2184]` (Jan study stage,
/// Feb study stage, then the post-study April stage). A lead in `(744, 1440]`
/// makes the in-study delivery (`m = 1`) decide at stage 0 and the post-study
/// delivery (`m = 2`) decide at stage 1, a single decision per stage — the
/// single-decider shape the fan-out reject requires.
const DELTA_HOURS: f64 = 1100.0;
/// The window's in-study decider stage — the only stage whose LP carries both
/// the sparse decision column and the `anticipated_lanes` row.
const DECIDER_STAGE: u32 = 1;
/// Fuel cost `$/MWh` at the resolved post-study cell (unused by the pinned
/// decision fixtures below beyond keeping the LP well-posed).
const COST_PER_MWH: f64 = 37.5;
/// The post-study commitment, pinned by a degenerate post-study capability
/// (`min_k == max_k == PINNED_MW`): the ring's post-study decision column reads
/// its bound from the post-study thermal capability, so pinning that capability
/// leaves the LP no freedom and the deposited/carried value is deterministic.
const PINNED_MW: f64 = 30.0;
/// `YYYYMM01` anchor of the declared post-study destination stage
/// (`2024-04-01`) — the expected `delivery_date` on every emitted row.
const EXPECTED_DELIVERY_DATE: i32 = 2024_0401;

fn study_start() -> NaiveDate {
    NaiveDate::from_ymd_opt(2024, 1, 1).expect("valid date")
}

/// Two stages matching their real calendar span exactly (required for
/// `StageCalendar` coverage): stage 0 is January (744h), stage 1 is 30 days
/// (720h) so the post-study window below overlaps it exactly at `dest = 0`.
fn stages() -> Vec<cobre_core::temporal::Stage> {
    let start = study_start();
    let stage0_end = start + chrono::TimeDelta::days(31);
    let stage1_end = stage0_end + chrono::TimeDelta::days(30);
    vec![
        make_stage(
            0,
            StageSpec {
                start_date: start,
                end_date: stage0_end,
                blocks: vec![cobre_core::temporal::Block {
                    index: 0,
                    name: "S0".to_string(),
                    duration_hours: 744.0,
                }],
                ..Default::default()
            },
        ),
        make_stage(
            1,
            StageSpec {
                start_date: stage0_end,
                end_date: stage1_end,
                blocks: vec![cobre_core::temporal::Block {
                    index: 0,
                    name: "S1".to_string(),
                    duration_hours: 720.0,
                }],
                ..Default::default()
            },
        ),
    ]
}

/// The one declared post-study stage: `[2024-04-01, 2024-05-01)` — 30 days
/// (720h), the EXACT span the post-study delivery (via `DELTA_HOURS`) resolves
/// to, so `StageCalendar::resolve_window` covers it at destination index 0.
fn post_study_stages(min_k: f64, max_k: f64) -> PostStudyStages {
    PostStudyStages {
        stages: vec![PostStudyStage {
            start_date: NaiveDate::from_ymd_opt(2024, 4, 1).expect("valid date"),
            duration_hours: 720.0,
        }],
        thermal_bounds: vec![PostStudyThermalBound {
            thermal_id: THERMAL_ID,
            post_study_stage_index: 0,
            cost_per_mwh: COST_PER_MWH,
            min_mw: min_k,
            max_mw: max_k,
        }],
    }
}

fn bounds() -> ResolvedBounds {
    ResolvedBounds::new(
        &BoundsCountsSpec {
            n_hydros: 0,
            n_thermals: 1,
            n_lines: 0,
            n_pumping: 0,
            n_contracts: 0,
            n_stages: 2,
            k_max: 1,
        },
        &BoundsDefaults {
            hydro: HydroStageBounds {
                min_storage_hm3: 0.0,
                max_storage_hm3: 0.0,
                filling_min_rate_m3s: 0.0,
                water_withdrawal_m3s: 0.0,
            },
            hydro_block: HydroBlockBounds::default(),
            thermal: ThermalStageBounds { cost_per_mwh: 1.0 },
            thermal_block: ThermalBlockBounds {
                min_generation_mw: 0.0,
                max_generation_mw: 0.0,
            },
            line_block: LineBlockBounds {
                direct_mw: 0.0,
                reverse_mw: 0.0,
            },
            pumping_block: PumpingBlockBounds {
                min_flow_m3s: 0.0,
                max_flow_m3s: 0.0,
            },
            contract_block: ContractBlockBounds {
                min_mw: 0.0,
                max_mw: 0.0,
                price_per_mwh: 0.0,
            },
        },
    )
}

fn penalties() -> cobre_core::resolved::ResolvedPenalties {
    use cobre_core::HydroPenalties;
    use cobre_core::resolved::{
        BusStagePenalties, LineStagePenalties, NcsStagePenalties, PenaltiesCountsSpec,
        PenaltiesDefaults, ResolvedPenalties,
    };
    ResolvedPenalties::new(
        &PenaltiesCountsSpec {
            n_hydros: 0,
            n_buses: 1,
            n_lines: 0,
            n_ncs: 0,
            n_stages: 2,
        },
        &PenaltiesDefaults {
            hydro: HydroPenalties {
                spillage_cost: 0.0,
                diversion_cost: 0.0,
                turbined_cost: 0.0,
                storage_violation_below_cost: 0.0,
                filling_target_violation_cost: 0.0,
                turbined_violation_below_cost: 0.0,
                outflow_violation_below_cost: 0.0,
                outflow_violation_above_cost: 0.0,
                generation_violation_below_cost: 0.0,
                evaporation_violation_cost: 0.0,
                water_withdrawal_violation_cost: 0.0,
                water_withdrawal_violation_pos_cost: 0.0,
                water_withdrawal_violation_neg_cost: 0.0,
                evaporation_violation_pos_cost: 0.0,
                evaporation_violation_neg_cost: 0.0,
                inflow_nonnegativity_cost: 0.0,
            },
            bus: BusStagePenalties { excess_cost: 0.0 },
            line: LineStagePenalties { exchange_cost: 0.0 },
            ncs: NcsStagePenalties {
                curtailment_cost: 0.0,
            },
        },
    )
}

/// `with_commitment` toggles the one declared `post_study_stages` stage — the
/// re-sourced `anticipated_lanes` gate's presence predicate; `(min_k, max_k)`
/// are read only when `with_commitment` is `true`.
fn build_system(with_commitment: bool, min_k: f64, max_k: f64) -> System {
    let bus = make_bus(BUS_ID, BusSpec::default());
    let thermal = make_thermal(
        THERMAL_ID,
        ThermalSpec {
            bus_id: BUS_ID,
            cost_per_mwh: 1.0,
            min_generation_mw: 0.0,
            max_generation_mw: 0.0,
            anticipated_config: Some(AnticipatedConfig::LeadTime(DELTA_HOURS)),
            ..Default::default()
        },
    );
    let mut builder = SystemBuilder::new()
        .buses(vec![bus])
        .thermals(vec![thermal])
        .stages(stages())
        .bounds(bounds())
        .penalties(penalties())
        .initial_conditions(InitialConditions::default());
    if with_commitment {
        builder = builder.post_study_stages(Some(post_study_stages(min_k, max_k)));
    }
    builder.build().expect("fixture System must build")
}

fn config(num_scenarios: u32) -> cobre_io::config::Config {
    use cobre_io::config::{
        Config, EstimationConfig, ExportsConfig, InflowNonNegativityConfig,
        InflowNonNegativityMethod, ModelingConfig, ParallelismConfig, PolicyConfig,
        RowSelectionConfig, SimulationConfig, StoppingMode, StoppingRuleConfig, TrainingConfig,
        TrainingSelection, TrainingSolverConfig, UpperBoundEvaluationConfig,
    };
    Config {
        schema: None,
        modeling: ModelingConfig {
            inflow_non_negativity: InflowNonNegativityConfig {
                method: InflowNonNegativityMethod::Penalty,
            },
            cost_scale_factor: Some(1.0),
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
        simulation: SimulationConfig {
            enabled: true,
            io_channel_capacity: 16,
            selection: Some(SimulationSelection::Sampled { num_scenarios }),
            ..SimulationConfig::default()
        },
        exports: ExportsConfig::default(),
        estimation: EstimationConfig::default(),
    }
}

/// Train + simulate `setup`, returning results ordered by ascending `scenario_id`
/// (`run_simulation`'s drain order is arrival order, not necessarily ascending).
fn simulate_sorted(setup: &mut StudySetup, n_scenarios: u32) -> Vec<SimulationScenarioResult> {
    let mut results = run_simulation(setup, 1);
    assert_eq!(
        results.len(),
        n_scenarios as usize,
        "must produce exactly one SimulationScenarioResult per scenario"
    );
    results.sort_by_key(|r| r.scenario_id);
    results
}

// -- A solved post-horizon commitment produces a keyed row --

mod partition_row_keying {
    use super::*;

    #[test]
    fn decider_stage_emits_a_row_keyed_by_thermal_and_delivery_date() {
        let system = build_system(true, PINNED_MW, PINNED_MW);
        let mut setup = build_setup_in_code(system, &config(1));

        let results = simulate_sorted(&mut setup, 1);
        let scenario = &results[0];

        let decider_stage = scenario
            .stages
            .iter()
            .find(|s| s.stage_id == DECIDER_STAGE)
            .expect("the decider stage must be present in the per-stage results");
        assert_eq!(
            decider_stage.anticipated_lanes.len(),
            1,
            "exactly one declared window decides at this stage"
        );

        let row = &decider_stage.anticipated_lanes[0];
        assert_eq!(
            row.thermal_id, THERMAL_ID.0,
            "row must key on the owning thermal"
        );
        assert_eq!(
            row.delivery_date, EXPECTED_DELIVERY_DATE,
            "delivery_date must be the resolved post-study destination stage's YYYYMM01 anchor"
        );
        assert!(
            (row.deposited_decision_mw - PINNED_MW).abs() < 1e-6,
            "deposited_decision_mw must equal the pinned commitment: expected {PINNED_MW}, got {}",
            row.deposited_decision_mw
        );
        assert!(
            (row.carried_committed_mw - PINNED_MW).abs() < 1e-6,
            "carried_committed_mw must equal the same pinned commitment (the deposit row pins \
             commit_out to the decision column): expected {PINNED_MW}, got {}",
            row.carried_committed_mw
        );

        for stage in &scenario.stages {
            if stage.stage_id != DECIDER_STAGE {
                assert!(
                    stage.anticipated_lanes.is_empty(),
                    "stage {}: only the decider stage emits a commitment-lane row",
                    stage.stage_id
                );
            }
        }
    }

    /// Per-lane extraction cross-check: the emitted row's deposited decision
    /// reads the SAME in-study anticipated decision column
    /// (`StageGeometry::anticipated_decision`, `col_anticipated_decision_start +
    /// local`) the LP pins to the delivery-anchored commitment interval — the
    /// column the ring deposit latches, now that the post-study delivery flows
    /// through the anticipated ring rather than the retired lane block.
    #[test]
    fn emitted_decision_matches_the_deciders_own_decision_column_bounds() {
        let system = build_system(true, PINNED_MW, PINNED_MW);
        let setup = build_setup_in_code(system, &config(1));

        let range = setup.stage_ctx().geometry_per_stage[DECIDER_STAGE as usize]
            .anticipated_decision
            .clone();
        assert_eq!(
            range.len(),
            1,
            "fixture declares exactly one anticipated plant, so one decision column per stage"
        );
        let template = &setup.stage_ctx().templates[DECIDER_STAGE as usize];
        assert_eq!(template.col_lower[range.start], PINNED_MW);
        assert_eq!(template.col_upper[range.start], PINNED_MW);
    }
}

// -- Per-scenario carried value: S terminal scenarios -> S rows, never one --

mod per_scenario_shape {
    use super::*;

    #[test]
    fn fanned_scenario_count_produces_one_row_per_scenario_never_collapsed() {
        const S: u32 = 3;
        let system = build_system(true, PINNED_MW, PINNED_MW);
        let mut setup = build_setup_in_code(system, &config(S));

        let results = simulate_sorted(&mut setup, S);

        let mut seen_scenarios: Vec<u32> = Vec::new();
        for scenario in &results {
            let decider_stage = scenario
                .stages
                .iter()
                .find(|s| s.stage_id == DECIDER_STAGE)
                .expect("decider stage must be present");
            assert_eq!(
                decider_stage.anticipated_lanes.len(),
                1,
                "scenario {}: exactly one row for the sole declared window",
                scenario.scenario_id
            );
            let row = &decider_stage.anticipated_lanes[0];
            assert_eq!(row.thermal_id, THERMAL_ID.0);
            assert_eq!(row.delivery_date, EXPECTED_DELIVERY_DATE);
            seen_scenarios.push(scenario.scenario_id);
        }

        seen_scenarios.sort_unstable();
        assert_eq!(
            seen_scenarios,
            (0..S).collect::<Vec<_>>(),
            "S={S} terminal scenarios must produce S per-scenario rows for the SAME \
             (thermal_id, delivery_date) key -- never collapsed to a single value"
        );
    }
}

// -- Inert without a declared post-horizon commitment --

mod inert_without_commitment {
    use super::*;

    #[test]
    fn no_post_study_stages_emits_no_rows_and_leaves_per_plant_columns_intact() {
        let system = build_system(false, 0.0, 0.0);
        let mut setup = build_setup_in_code(system, &config(1));

        let results = simulate_sorted(&mut setup, 1);
        let scenario = &results[0];

        for stage in &scenario.stages {
            assert!(
                stage.anticipated_lanes.is_empty(),
                "stage {}: no declared window exists, so no commitment-lane row may exist",
                stage.stage_id
            );
            // The per-plant anticipated_* columns stay populated regardless --
            // additive, not replaced.
            assert!(
                !stage.thermals.is_empty(),
                "stage {}: the per-plant thermal row must still be emitted",
                stage.stage_id
            );
            assert!(
                stage.thermals[0].is_anticipated,
                "the thermal is still declared anticipated even with no post-study stages"
            );
        }
    }

    #[test]
    fn writer_creates_no_anticipated_lanes_directory() {
        let system = build_system(false, 0.0, 0.0);
        let mut setup = build_setup_in_code(system, &config(1));
        let results = simulate_sorted(&mut setup, 1);

        let tmp = tempfile::tempdir().expect("tempdir must succeed");
        let mut writer = SimulationParquetWriter::new(tmp.path(), &build_system(false, 0.0, 0.0))
            .expect("SimulationParquetWriter::new must succeed");

        for scenario in results {
            writer
                .write_scenario(ScenarioWritePayload::from(scenario))
                .expect("write_scenario must succeed");
        }

        assert!(
            !tmp.path().join("simulation/anticipated_lanes").exists(),
            "anticipated_lanes/ must not exist for a study with no declared window"
        );
    }
}

// -- Shared-writer parity: byte-identical output for identical inputs --
//
// `cobre-python` is not linkable from a `cobre-sddp` integration test, so
// parity is verified structurally: two independent runs of the identical
// deterministic fixture, each converted and written through the EXACT
// production path both the CLI and Python bindings call
// (`ScenarioWritePayload::from` then `SimulationParquetWriter::write_scenario`
// -- see `crates/cobre-cli/src/commands/run/simulation.rs` and
// `crates/cobre-python/src/run.rs`), must produce byte-identical Parquet.
mod shared_writer_parity {
    use super::*;

    #[test]
    fn identical_runs_produce_byte_identical_anticipated_lanes_parquet() {
        let write_once = |tmp_path: &std::path::Path| {
            let system = build_system(true, PINNED_MW, PINNED_MW);
            let mut setup =
                build_setup_in_code(build_system(true, PINNED_MW, PINNED_MW), &config(1));
            let results = simulate_sorted(&mut setup, 1);

            let mut writer = SimulationParquetWriter::new(tmp_path, &system)
                .expect("SimulationParquetWriter::new must succeed");
            for scenario in results {
                writer
                    .write_scenario(ScenarioWritePayload::from(scenario))
                    .expect("write_scenario must succeed");
            }
        };

        let tmp_a = tempfile::tempdir().expect("tempdir must succeed");
        let tmp_b = tempfile::tempdir().expect("tempdir must succeed");
        write_once(tmp_a.path());
        write_once(tmp_b.path());

        let rel = "simulation/anticipated_lanes/scenario_id=0000/data.parquet";
        let path_a = tmp_a.path().join(rel);
        let path_b = tmp_b.path().join(rel);
        assert!(
            path_a.exists(),
            "anticipated_lanes parquet must exist in run A"
        );
        assert!(
            path_b.exists(),
            "anticipated_lanes parquet must exist in run B"
        );

        let bytes_a = std::fs::read(&path_a).expect("run A parquet must be readable");
        let bytes_b = std::fs::read(&path_b).expect("run B parquet must be readable");
        assert_eq!(
            bytes_a, bytes_b,
            "two independent runs of the identical fixture through the shared writer path \
             must produce byte-identical anticipated_lanes output -- the mechanism that \
             makes CLI/Python parity automatic"
        );
    }
}
