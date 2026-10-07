//! Integration coverage for the commitment-hold family of
//! `crate::lp::builder::state_box::build_state_box` — the one family a hand-built
//! layout cannot cover, since the delivery-stage resolution needs the full
//! `StudySetup` construction pipeline.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::doc_markdown,
    clippy::too_many_lines
)]
// `..Default::default()` in the make_* Spec calls is the intentional future-field
// seam from `common::builders` — a no-op today, not dead code.
#![allow(clippy::needless_update)]

mod common;

use chrono::{NaiveDate, TimeDelta};
use cobre_core::entities::{
    bus::DeficitSegment, hydro::HydroGenerationModel, thermal::AnticipatedConfig,
};
use cobre_core::scenario::{InflowModel, LoadModel};
use cobre_core::temporal::{
    Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig, StageStateConfig,
};
use cobre_core::{
    AnticipatedCommitmentHistory, BoundsCountsSpec, BoundsDefaults, BusStagePenalties,
    ContractBlockBounds, EntityId, HydroBlockBounds, HydroPenalties, HydroStageBounds,
    HydroStorage, InitialConditions, LineBlockBounds, LineStagePenalties, NcsStagePenalties,
    PenaltiesCountsSpec, PenaltiesDefaults, PumpingBlockBounds, ResolvedBounds, ResolvedPenalties,
    SystemBuilder, ThermalBlockBounds, ThermalStageBounds,
};
use cobre_io::config::{
    Config, EstimationConfig, ExportsConfig, InflowNonNegativityConfig,
    InflowNonNegativityMethod as CfgInflowMethod, ModelingConfig, PolicyConfig, RowSelectionConfig,
    SimulationConfig as IoSimulationConfig, StoppingRuleConfig, TrainingConfig, TrainingSelection,
    TrainingSolverConfig, UpperBoundEvaluationConfig,
};

use common::build_setup_in_code;
use common::builders::{
    BusSpec, HydroSpec, StageSpec, ThermalSpec, make_bus, make_hydro, make_stage, make_thermal,
};

const N_STAGES: usize = 5;
const K_MAX: usize = 2;
const DELIVERY_STAGE: usize = 2;
const OVERRIDE_MIN_MW: f64 = 10.0;
const OVERRIDE_MAX_MW: f64 = 77.0;
/// A pre-study (`decider == None`) delivery target, seeded via
/// `past_anticipated_commitments`, still in flight (carried, not yet matured) at
/// decision stage 0 — the regression case for a k_max > 1 interior slot.
const INTERIOR_DELIVERY_STAGE: usize = 1;
const INTERIOR_MIN_MW: f64 = 5.0;
const INTERIOR_MAX_MW: f64 = 42.0;

fn daily_stage_dates(
    anchor: NaiveDate,
    n_stages: usize,
    days_per_stage: i64,
) -> Vec<(NaiveDate, NaiveDate)> {
    (0..n_stages)
        .map(|i| {
            (
                anchor + TimeDelta::days(days_per_stage * i as i64),
                anchor + TimeDelta::days(days_per_stage * (i as i64 + 1)),
            )
        })
        .collect()
}

fn windowed_commitments_daily(
    thermal_id: EntityId,
    anchor: NaiveDate,
    days_per_stage: i64,
    values: &[f64],
) -> Vec<AnticipatedCommitmentHistory> {
    values
        .iter()
        .enumerate()
        .map(|(i, &value_mw)| AnticipatedCommitmentHistory {
            thermal_id,
            start_date: anchor + TimeDelta::days(days_per_stage * i as i64),
            end_date: anchor + TimeDelta::days(days_per_stage * (i as i64 + 1)),
            value_mw,
        })
        .collect()
}

fn default_hydro_bounds() -> HydroStageBounds {
    HydroStageBounds {
        min_storage_hm3: 0.0,
        max_storage_hm3: 200.0,
        filling_min_rate_m3s: 0.0,
        water_withdrawal_m3s: 0.0,
    }
}

fn default_hydro_block_bounds() -> HydroBlockBounds {
    HydroBlockBounds {
        max_turbined_m3s: 100.0,
        max_generation_mw: 250.0,
        ..Default::default()
    }
}

fn default_hydro_penalties() -> HydroPenalties {
    HydroPenalties {
        spillage_cost: 0.01,
        diversion_cost: 0.0,
        turbined_cost: 0.0,
        storage_violation_below_cost: 500.0,
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
        inflow_nonnegativity_cost: 1000.0,
    }
}

/// One bus, one K=2 anticipated thermal, one non-anticipated backup thermal, one
/// hydro — the same shape `anticipated_scenarios.rs`'s `anticipated_5stage_k2_smoke`
/// fixture uses, plus `thermal_block_base` overrides at `DELIVERY_STAGE` (the
/// fresh-deposit case) and `INTERIOR_DELIVERY_STAGE` (the carried-interior case)
/// so neither assertion can pass on a stage-invariant default value. Returns the
/// system and the exact `ResolvedBounds` baked into it, so the test can read
/// `thermal_block_base` back from the same table `build_state_box` reads.
fn build_system_and_bounds() -> (cobre_core::System, ResolvedBounds) {
    let bus = make_bus(
        EntityId(1),
        BusSpec {
            name: "B1".to_string(),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 500.0,
            }],
            excess_cost: 0.0,
            ..Default::default()
        },
    );

    let anticipated_id = EntityId(2);
    let thermal_ant = make_thermal(
        anticipated_id,
        ThermalSpec {
            name: "T_ant".to_string(),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            bus_id: EntityId(1),
            min_generation_mw: 0.0,
            max_generation_mw: 100.0,
            cost_per_mwh: 50.0,
            anticipated_config: Some(AnticipatedConfig::LeadStages(K_MAX as u32)),
            entry_stage_id: None,
            exit_stage_id: None,
            ..Default::default()
        },
    );

    let thermal_backup = make_thermal(
        EntityId(4),
        ThermalSpec {
            name: "T_backup".to_string(),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            bus_id: EntityId(1),
            min_generation_mw: 0.0,
            max_generation_mw: 200.0,
            cost_per_mwh: 500.0,
            anticipated_config: None,
            entry_stage_id: None,
            exit_stage_id: None,
            ..Default::default()
        },
    );

    let hydro = make_hydro(
        EntityId(3),
        HydroSpec {
            name: "H1".to_string(),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            bus_id: EntityId(1),
            downstream_id: None,
            entry_stage_id: None,
            exit_stage_id: None,
            min_storage_hm3: 0.0,
            max_storage_hm3: 200.0,
            min_outflow_m3s: 0.0,
            max_outflow_m3s: None,
            generation_model: HydroGenerationModel::ConstantProductivity,
            min_turbined_m3s: 0.0,
            max_turbined_m3s: 100.0,
            specific_productivity_mw_per_m3s_per_m: None,
            min_generation_mw: 0.0,
            max_generation_mw: 250.0,
            tailrace: None,
            hydraulic_losses: None,
            efficiency: None,
            evaporation_coefficients_mm: None,
            evaporation_reference_volumes_hm3: None,
            diversion: None,
            filling: None,
            penalties: default_hydro_penalties(),
            ..Default::default()
        },
    );

    let calendar_anchor = NaiveDate::from_ymd_opt(2024, 1, 1).unwrap();
    let stage_dates = daily_stage_dates(calendar_anchor, N_STAGES, 31);
    let stages: Vec<Stage> = (0..N_STAGES)
        .map(|i| {
            make_stage(
                i,
                StageSpec {
                    start_date: stage_dates[i].0,
                    end_date: stage_dates[i].1,
                    season_id: None,
                    blocks: vec![Block {
                        index: 0,
                        name: "S".to_string(),
                        duration_hours: 744.0,
                    }],
                    block_mode: BlockMode::Parallel,
                    state_config: StageStateConfig {
                        storage: true,
                        inflow_lags: false,
                    },
                    risk_config: StageRiskConfig::Expectation,
                    scenario_config: ScenarioSourceConfig {
                        branching_factor: 1,
                        noise_method: NoiseMethod::Saa,
                    },
                    ..Default::default()
                },
            )
        })
        .collect();

    let inflow_models: Vec<InflowModel> = (0..N_STAGES)
        .map(|i| InflowModel {
            hydro_id: EntityId(3),
            stage_id: i as i32,
            mean_m3s: 80.0,
            std_m3s: 20.0,
            ar_coefficients: vec![],
            residual_std_ratio: 1.0,
            annual: None,
        })
        .collect();

    let load_models: Vec<LoadModel> = (0..N_STAGES)
        .map(|i| LoadModel {
            bus_id: EntityId(1),
            stage_id: i as i32,
            mean_mw: 150.0,
            std_mw: 0.0,
        })
        .collect();

    let mut bounds = ResolvedBounds::new(
        &BoundsCountsSpec {
            n_hydros: 1,
            n_thermals: 2,
            n_lines: 0,
            n_pumping: 0,
            n_contracts: 0,
            n_stages: N_STAGES,
            k_max: K_MAX,
        },
        &BoundsDefaults {
            hydro: default_hydro_bounds(),
            hydro_block: default_hydro_block_bounds(),
            thermal: ThermalStageBounds { cost_per_mwh: 0.0 },
            thermal_block: ThermalBlockBounds {
                min_generation_mw: 0.0,
                max_generation_mw: 100.0,
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
    );

    // The anticipated thermal is system thermal index 0 (`.thermals(vec![thermal_ant,
    // thermal_backup])`). Overriding only DELIVERY_STAGE's and
    // INTERIOR_DELIVERY_STAGE's cells — leaving every other stage at the uniform
    // default — makes both assertions fail if `build_state_box` read any stage
    // other than each slot's own genuine held delivery target.
    let ant_cell = bounds.thermal_block_base_mut(0, DELIVERY_STAGE);
    ant_cell.min_generation_mw = OVERRIDE_MIN_MW;
    ant_cell.max_generation_mw = OVERRIDE_MAX_MW;
    let interior_cell = bounds.thermal_block_base_mut(0, INTERIOR_DELIVERY_STAGE);
    interior_cell.min_generation_mw = INTERIOR_MIN_MW;
    interior_cell.max_generation_mw = INTERIOR_MAX_MW;

    let penalties = ResolvedPenalties::new(
        &PenaltiesCountsSpec {
            n_hydros: 1,
            n_buses: 1,
            n_lines: 0,
            n_ncs: 0,
            n_stages: N_STAGES,
        },
        &PenaltiesDefaults {
            hydro: default_hydro_penalties(),
            bus: BusStagePenalties { excess_cost: 0.0 },
            line: LineStagePenalties { exchange_cost: 0.0 },
            ncs: NcsStagePenalties {
                curtailment_cost: 0.0,
            },
        },
    );

    let initial_conditions = InitialConditions {
        storage: vec![HydroStorage {
            hydro_id: EntityId(3),
            value_hm3: 100.0,
        }],
        filling_storage: vec![],
        past_anticipated_commitments: windowed_commitments_daily(
            anticipated_id,
            calendar_anchor,
            31,
            &[100.0, 50.0],
        ),
        recent_observations: vec![],
        past_defluences: vec![],
    };

    let system = SystemBuilder::new()
        .buses(vec![bus])
        .thermals(vec![thermal_ant, thermal_backup])
        .hydros(vec![hydro])
        .stages(stages)
        .inflow_models(inflow_models)
        .load_models(load_models)
        .bounds(bounds.clone())
        .penalties(penalties)
        .initial_conditions(initial_conditions)
        .build()
        .expect("build_system_and_bounds: valid");

    (system, bounds)
}

fn build_config() -> Config {
    Config {
        schema: None,
        modeling: ModelingConfig {
            inflow_non_negativity: InflowNonNegativityConfig {
                method: CfgInflowMethod::Penalty,
            },
            cost_scale_factor: None,
        },
        training: TrainingConfig {
            enabled: true,
            tree_seed: Some(42),
            stopping_rules: Some(vec![StoppingRuleConfig::IterationLimit { limit: 8 }]),
            stopping_mode: cobre_io::config::StoppingMode::Any,
            cut_selection: RowSelectionConfig::default(),
            solver: TrainingSolverConfig::default(),
            parallelism: cobre_io::config::ParallelismConfig::default(),
            scenario_source: None,
            selection: Some(TrainingSelection::Sampled { forward_passes: 1 }),
        },
        upper_bound_evaluation: UpperBoundEvaluationConfig::default(),
        policy: PolicyConfig::default(),
        simulation: IoSimulationConfig::default(),
        exports: ExportsConfig::default(),
        estimation: EstimationConfig::default(),
    }
}

/// The commit_out slot decision stage 0 latches (K=2, delivering at stage 2) takes
/// its box from `thermal_block_base(thermal_idx, delivery_stage)` — the same
/// stage-level resolved bound `fill_anticipated_columns` reads for the decision
/// column — not decision stage 0's own (unoverridden) bound.
#[test]
fn state_box_commitment_slot_takes_the_delivery_stage_resolved_bound() {
    let (system, bounds) = build_system_and_bounds();
    let config = build_config();
    let setup = build_setup_in_code(system, &config);

    let state = setup.stage_state();
    let n_anticipated = state.n_anticipated;
    let k_max = state.k_max;
    assert_eq!(
        n_anticipated, 1,
        "fixture declares exactly one anticipated thermal"
    );
    assert_eq!(k_max, K_MAX, "fixture's K_MAX must match the declared lead");

    let slot = DELIVERY_STAGE % k_max;
    let local_idx = 0;
    let j = state.commit_out.start + slot * n_anticipated + local_idx;

    let (lower, upper) = cobre_sddp::test_support::stage_state_box_bounds(&setup, 0);

    let expected = bounds.thermal_block_base(0, DELIVERY_STAGE);
    assert_eq!(
        lower[j], expected.min_generation_mw,
        "commit_out slot lower bound must equal thermal_block_base's min at the delivery stage"
    );
    assert_eq!(
        upper[j], expected.max_generation_mw,
        "commit_out slot upper bound must equal thermal_block_base's max at the delivery stage"
    );
    assert_eq!(lower[j], OVERRIDE_MIN_MW);
    assert_eq!(upper[j], OVERRIDE_MAX_MW);
}

/// At decision stage 0, the K=2 ring's slot for
/// `INTERIOR_DELIVERY_STAGE` (a pre-study, `decider == None`, seeded target still
/// in flight — not this stage's fresh deposit, which targets `DELIVERY_STAGE`
/// instead) must carry its OWN held target's resolved bound, not the `[0, 0]`
/// default a fresh-decision-only sweep would leave it at.
#[test]
fn state_box_commitment_carried_interior_slot_takes_its_own_held_delivery_target_bound() {
    let (system, bounds) = build_system_and_bounds();
    let config = build_config();
    let setup = build_setup_in_code(system, &config);

    let state = setup.stage_state();
    let n_anticipated = state.n_anticipated;
    let k_max = state.k_max;

    let slot = INTERIOR_DELIVERY_STAGE % k_max;
    let local_idx = 0;
    let j = state.commit_out.start + slot * n_anticipated + local_idx;
    // Sanity: the interior slot and the fresh-deposit slot from the sibling test
    // must land in DIFFERENT commit_out columns, or the two overrides above would
    // not isolate each case.
    assert_ne!(
        slot,
        DELIVERY_STAGE % k_max,
        "interior and fresh-deposit slots must differ for this fixture to isolate them"
    );

    let (lower, upper) = cobre_sddp::test_support::stage_state_box_bounds(&setup, 0);

    let expected = bounds.thermal_block_base(0, INTERIOR_DELIVERY_STAGE);
    assert_eq!(
        lower[j], expected.min_generation_mw,
        "carried interior slot's lower bound must equal thermal_block_base's min at its own held delivery target"
    );
    assert_eq!(
        upper[j], expected.max_generation_mw,
        "carried interior slot's upper bound must equal thermal_block_base's max at its own held delivery target"
    );
    assert_eq!(lower[j], INTERIOR_MIN_MW);
    assert_eq!(upper[j], INTERIOR_MAX_MW);
    assert_ne!(
        upper[j], 0.0,
        "the old fresh-decision-only sweep left every carried interior slot at [0, 0]"
    );
}
