//! Integration tests for inflow non-negativity enforcement via the penalty method.
//!
//! All tests share a 2-hydro, 1-bus, 3-stage fixture whose PAR(0) inflow model
//! (`mean_m3s = 0.0`, `std_m3s = 30.0`, 10 openings/stage from seed 42) makes
//! roughly half of sampled noise values produce negative effective inflows —
//! the precondition the slack-value assertions rely on. `ActiveSolver` is used
//! throughout so the LP is solved and slack columns receive real primal values.
//! `truncation_handles_zero_sigma` replaces the inflow model through `InflowShape`.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::too_many_lines
)]
// `..Default::default()` in the make_* Spec calls is the intentional future-field
// seam from `common::builders` — a no-op today, not dead code.
#![allow(clippy::needless_update)]

use std::collections::{BTreeMap, HashMap};
use std::sync::mpsc;

use chrono::NaiveDate;
use novomodelo_core::{
    BoundsCountsSpec, BoundsDefaults, BusStagePenalties, ContractBlockBounds, DeficitSegment,
    EntityId, HydroBlockBounds, HydroPenalties, HydroStageBounds, LineBlockBounds,
    LineStagePenalties, NcsStagePenalties, PenaltiesCountsSpec, PenaltiesDefaults,
    PumpingBlockBounds, ResolvedBounds, ResolvedPenalties, System, SystemBuilder,
    ThermalBlockBounds, ThermalStageBounds,
    scenario::{
        CorrelationEntity, CorrelationGroup, CorrelationModel, CorrelationProfile, SamplingScheme,
    },
    temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
        StageStateConfig,
    },
};
use novomodelo_io::config::{InflowNonNegativityMethod as CfgInflowMethod, SimulationSelection};
use novomodelo_sddp::{
    LoadedPolicy, Phase, ResolvedParameters, SimulationHydroResult, SimulationScenarioResult,
    SolverProfiles, StoppingMode, StoppingRule, StoppingRuleSet, TrainingConfig,
    build_stage_templates_resolving_layout,
    config::{CutManagementConfig, EventConfig, LoopConfig},
    context::TrainingContext,
    cut::FutureCostFunction,
    energy_conversion::{EnergyConversion, EnergyConversionSet},
    horizon_mode::HorizonMode,
    hydro_models::PrepareHydroModelsResult,
    indexer::{AnticipatedPlants, CutStateProjection, StateSpace, StudyDimensions},
    inflow_method::InflowNonNegativityMethod,
    lead_time::AnticipatedResolution,
    lp::builder::{PatchBuffer, StageGeometry, StateBox},
    risk_measure::RiskMeasure,
    setup::{NodeId, NodePos, node_graph::Traversal},
    simulate,
    simulation::{EntityCounts, SimulationConfig, SimulationOutputSpec},
    test_support::{node_opening_noise, permissive_state_boxes},
    train,
    workspace::{SolverWorkspace, WorkspaceSizing},
};
use novomodelo_solver::ActiveSolver;
use novomodelo_stochastic::{
    ClassSchemes, OpeningTreeInputs, PrecomputedNormal, PrecomputedPar, StochasticContext,
    build_stochastic_context,
};

mod common;
use common::builders::{BusSpec, HydroSpec, StageSpec, make_bus, make_hydro, make_stage};
use common::in_code_studies::{
    CHRONOLOGICAL_NOISE_BLOCK_HOURS, NEGATIVE_DRAW_LAG_SEED_M3S, chronological_noise_study,
    negative_draw_par1_spec, negative_draw_raw_inflow,
};
use common::{StubComm, build_setup_in_code, run_simulation};

/// External test crate cannot see parent's `test-support` surface; constructs
/// from explicit dimensions.
fn state_layout_for(hydro_count: usize, max_par_order: usize) -> StateSpace {
    StateSpace::new(
        hydro_count,
        max_par_order,
        Vec::new(),
        vec![],
        AnticipatedResolution::default(),
        &vec![max_par_order; hydro_count],
    )
}

/// External test crate cannot see `test-support`; sets fields directly.
fn study_dims_for(inflow_method: InflowNonNegativityMethod) -> StudyDimensions {
    StudyDimensions {
        max_deficit_segments: 1,
        inflow_method,
        anticipated_plants: AnticipatedPlants::default(),
        downstream_par_order: 0,
    }
}

// ===========================================================================
// System and stochastic context fixture
// ===========================================================================

const N_STAGES: usize = 3;
const N_HYDROS: usize = 2;

const COST_SCALE: f64 = 1_000_000.0;
const STAGE_HOURS: f64 = 744.0;
const M3S_TO_HM3: f64 = 0.0036;
const STORAGE_MAX_HM3: f64 = 50.0;
const INFLOW_PENALTY_COST: f64 = 1000.0;
const STAGE_HM3_PER_M3S: f64 = M3S_TO_HM3 * STAGE_HOURS;
/// derived: a unit of slack costs `c_inf` per m³/s·h and stores `M3S_TO_HM3` hm³ per m³/s·h,
/// so slack and stored water break even at `w = c_inf / M3S_TO_HM3`.
const TIE_WATER_VALUE_PER_HM3: f64 = INFLOW_PENALTY_COST / M3S_TO_HM3;

const ZERO_SIGMA_MEAN_M3S: f64 = -10.0;
const ZERO_SIGMA_AR_COEFFICIENT: f64 = 0.5;
const ZERO_SIGMA_LAG_SEEDS: [f64; N_HYDROS] = [-30.0, -50.0];

struct InflowShape {
    mean_m3s: f64,
    std_m3s: f64,
    ar_coefficient: Option<f64>,
    residual_std_ratio: f64,
}

const WHITE_NOISE_SHAPE: InflowShape = InflowShape {
    mean_m3s: 0.0,
    std_m3s: 30.0,
    ar_coefficient: None,
    residual_std_ratio: 1.0,
};

fn build_system() -> novomodelo_core::System {
    build_system_with_shape(&WHITE_NOISE_SHAPE)
}

fn build_system_with_shape(shape: &InflowShape) -> novomodelo_core::System {
    use novomodelo_core::entities::hydro::{HydroGenerationModel, HydroPenalties};
    use novomodelo_core::scenario::InflowModel;

    let zero_entity_penalties = HydroPenalties {
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
        inflow_nonnegativity_cost: INFLOW_PENALTY_COST,
    };

    let bus = make_bus(
        EntityId(0),
        BusSpec {
            name: "B0".to_string(),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 1000.0,
            }],
            excess_cost: 0.0,
            ..Default::default()
        },
    );

    let build_hydro = |id_val: i32, name: &str| {
        make_hydro(
            EntityId(id_val),
            HydroSpec {
                name: name.to_string(),
                operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                bus_id: EntityId(0),
                downstream_id: None,
                entry_stage_id: None,
                exit_stage_id: None,
                min_storage_hm3: 0.0,
                max_storage_hm3: STORAGE_MAX_HM3,
                min_outflow_m3s: 0.0,
                max_outflow_m3s: None,
                generation_model: HydroGenerationModel::ConstantProductivity,
                min_turbined_m3s: 0.0,
                max_turbined_m3s: 50.0,
                specific_productivity_mw_per_m3s_per_m: None,
                min_generation_mw: 0.0,
                max_generation_mw: 50.0,
                tailrace: None,
                hydraulic_losses: None,
                efficiency: None,
                evaporation_coefficients_mm: None,
                evaporation_reference_volumes_hm3: None,
                diversion: None,
                filling: None,
                penalties: zero_entity_penalties,
                ..Default::default()
            },
        )
    };

    let hydros = vec![build_hydro(1, "H1"), build_hydro(2, "H2")];

    let stages: Vec<Stage> = (0..N_STAGES)
        .map(|idx| {
            make_stage(
                idx,
                StageSpec {
                    start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                    end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
                    season_id: Some(0),
                    blocks: vec![Block {
                        index: 0,
                        name: "S".to_string(),
                        duration_hours: STAGE_HOURS,
                    }],
                    block_mode: BlockMode::Parallel,
                    state_config: StageStateConfig {
                        storage: true,
                        inflow_lags: shape.ar_coefficient.is_some(),
                    },
                    risk_config: StageRiskConfig::Expectation,
                    scenario_config: ScenarioSourceConfig {
                        branching_factor: 10,
                        noise_method: NoiseMethod::Saa,
                    },
                    ..Default::default()
                },
            )
        })
        .collect();

    let inflow_models: Vec<InflowModel> = [EntityId(1), EntityId(2)]
        .iter()
        .flat_map(|&hydro_id| {
            (0..N_STAGES).map(move |stage_idx| InflowModel {
                hydro_id,
                stage_id: stage_idx as i32,
                mean_m3s: shape.mean_m3s,
                std_m3s: shape.std_m3s,
                ar_coefficients: shape.ar_coefficient.into_iter().collect(),
                residual_std_ratio: shape.residual_std_ratio,
                annual: None,
            })
        })
        .collect();

    let mut profiles = BTreeMap::new();
    profiles.insert(
        "default".to_string(),
        CorrelationProfile {
            groups: vec![CorrelationGroup {
                name: "g1".to_string(),
                entities: vec![
                    CorrelationEntity {
                        entity_type: "inflow".to_string(),
                        id: EntityId(1),
                    },
                    CorrelationEntity {
                        entity_type: "inflow".to_string(),
                        id: EntityId(2),
                    },
                ],
                matrix: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
            }],
        },
    );
    let correlation = CorrelationModel {
        method: "spectral".to_string(),
        profiles,
        schedule: vec![],
    };

    let hydro_bounds_default = HydroStageBounds {
        min_storage_hm3: 0.0,
        max_storage_hm3: STORAGE_MAX_HM3,
        filling_min_rate_m3s: 0.0,
        water_withdrawal_m3s: 0.0,
    };
    let hydro_bounds_default_block = HydroBlockBounds {
        max_turbined_m3s: 50.0,
        max_generation_mw: 50.0,
        ..Default::default()
    };
    let resolved_bounds = ResolvedBounds::new(
        &BoundsCountsSpec {
            n_hydros: N_HYDROS,
            n_thermals: 0,
            n_lines: 0,
            n_pumping: 0,
            n_contracts: 0,
            n_stages: N_STAGES,
            k_max: 0,
        },
        &BoundsDefaults {
            hydro: hydro_bounds_default,
            hydro_block: hydro_bounds_default_block,
            thermal: ThermalStageBounds { cost_per_mwh: 0.0 },
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
    );

    let hydro_penalties_default = HydroPenalties {
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
        inflow_nonnegativity_cost: INFLOW_PENALTY_COST,
    };
    let resolved_penalties = ResolvedPenalties::new(
        &PenaltiesCountsSpec {
            n_hydros: N_HYDROS,
            n_buses: 1,
            n_lines: 0,
            n_ncs: 0,
            n_stages: N_STAGES,
        },
        &PenaltiesDefaults {
            hydro: hydro_penalties_default,
            bus: BusStagePenalties { excess_cost: 0.0 },
            line: LineStagePenalties { exchange_cost: 0.0 },
            ncs: NcsStagePenalties {
                curtailment_cost: 0.0,
            },
        },
    );

    SystemBuilder::new()
        .buses(vec![bus])
        .hydros(hydros)
        .stages(stages)
        .inflow_models(inflow_models)
        .correlation(correlation)
        .bounds(resolved_bounds)
        .penalties(resolved_penalties)
        .build()
        .unwrap()
}

fn build_stochastic(system: &System) -> StochasticContext {
    build_stochastic_context(
        system,
        42,
        None,
        &[],
        &[],
        OpeningTreeInputs::default(),
        ClassSchemes {
            inflow: Some(SamplingScheme::InSample),
            load: Some(SamplingScheme::InSample),
            ncs: Some(SamplingScheme::InSample),
        },
    )
    .unwrap()
}

// ===========================================================================
// Shared fixture builder
// ===========================================================================

/// All resources needed to run training and simulation.
struct Fixture {
    stage_templates: novomodelo_sddp::StageTemplates,
    stochastic: StochasticContext,
    /// Production stage-0 geometry (the role-(b) equipment/slack column ranges),
    /// cloned from `stage_templates.geometry_per_stage[0]`.
    geometry: StageGeometry,
    study_dims: StudyDimensions,
    state: StateSpace,
    initial_state: Vec<f64>,
    horizon: HorizonMode,
    risk_measures: Vec<RiskMeasure>,
    entity_counts: EntityCounts,
    inflow_method: InflowNonNegativityMethod,
}

fn build_fixture() -> Fixture {
    build_fixture_with_method(InflowNonNegativityMethod::Penalty)
}

fn build_fixture_with_method(inflow_method: InflowNonNegativityMethod) -> Fixture {
    build_fixture_with_shape(inflow_method, &WHITE_NOISE_SHAPE)
}

fn build_fixture_with_shape(
    inflow_method: InflowNonNegativityMethod,
    shape: &InflowShape,
) -> Fixture {
    let system = build_system_with_shape(shape);

    let par_lp = PrecomputedPar::build(
        system.inflow_models(),
        &system
            .stages()
            .iter()
            .filter(|s| s.id >= 0)
            .cloned()
            .collect::<Vec<_>>(),
        &system.hydros().iter().map(|h| h.id).collect::<Vec<_>>(),
        None,
    )
    .unwrap();

    let hydro_models = PrepareHydroModelsResult::default_from_system(&system);
    let stage_templates = build_stage_templates_resolving_layout(
        &system,
        inflow_method,
        &par_lp,
        &PrecomputedNormal::default(),
        &hydro_models.production,
        &hydro_models.evaporation,
        &ResolvedParameters::default(),
    )
    .expect("no FPHA plants in integration test fixture");
    let stochastic = build_stochastic(&system);

    let n_stages = stage_templates.templates.len();
    let n_h = system.hydros().len();
    let max_par_order = par_lp.max_order();
    let study_dims = study_dims_for(inflow_method);
    let geometry = stage_templates
        .geometry_per_stage
        .first()
        .expect("at least 1 stage geometry")
        .clone();

    let state = state_layout_for(n_h, max_par_order);
    let initial_state = vec![0.0_f64; state.n_state];
    let horizon = HorizonMode::Finite {
        num_stages: n_stages,
    };
    let risk_measures = vec![RiskMeasure::Expectation; n_stages];

    let entity_counts = EntityCounts {
        hydro_ids: system.hydros().iter().map(|h| h.id.0).collect(),
        hydro_productivities: vec![1.0; system.hydros().len()],
        thermal_ids: vec![],
        line_ids: vec![],
        bus_ids: system.buses().iter().map(|b| b.id.0).collect(),
        pumping_station_ids: vec![],
        contract_ids: vec![],
        non_controllable_ids: vec![],
    };

    Fixture {
        stage_templates,
        stochastic,
        geometry,
        study_dims,
        state,
        initial_state,
        horizon,
        risk_measures,
        entity_counts,
        inflow_method,
    }
}

// ===========================================================================
// Shared test helpers
// ===========================================================================

fn base_stage_context<'a>(
    fx: &'a Fixture,
    state_boxes: &'a [StateBox],
) -> novomodelo_sddp::test_support::StageContextFixture<'a> {
    novomodelo_sddp::test_support::StageContextFixture::from_stage_templates(
        &fx.stage_templates,
        state_boxes,
    )
}

fn train_fixture(
    fx: &Fixture,
    iterations: u64,
) -> Result<novomodelo_sddp::TrainingOutcome, novomodelo_sddp::SddpError> {
    let n_stages = fx.stage_templates.templates.len();
    let mut fcf = FutureCostFunction::new(n_stages, fx.state.n_state, 1, 20, &vec![0; n_stages]);
    let mut solver = ActiveSolver::new().expect("ActiveSolver::new must succeed");
    let comm = StubComm;

    let state_boxes = permissive_state_boxes(fx.state.n_state, n_stages);
    let stage_ctx_fixture = base_stage_context(fx, &state_boxes);
    let stage_ctx = stage_ctx_fixture.ctx();
    train(
        &mut solver,
        TrainingConfig {
            loop_config: LoopConfig {
                forward_passes: 1,
                training_enumerated: false,
                max_iterations: 10,
                start_iteration: 0,
                resume_lower_bound_history: Vec::new(),
                n_fwd_threads: 1,
                stopping_rules: StoppingRuleSet {
                    rules: vec![StoppingRule::IterationLimit { limit: iterations }],
                    mode: StoppingMode::Any,
                },
            },
            cut_management: CutManagementConfig {
                cut_selection: None,
                budget: None,
                cut_activity_tolerance: 0.0,
                risk_measures: fx.risk_measures.clone(),
            },
            events: EventConfig {
                event_sender: None,
                periodic_checkpoint: None,
                shutdown_flag: None,
                export_states: false,
            },
        },
        &mut fcf,
        &stage_ctx,
        &TrainingContext {
            node_graph: &novomodelo_sddp::test_support::chain_node_graph(&fx.stochastic),
            horizon: &fx.horizon,
            state: &fx.state,
            cut_state_layouts: &all_enabled_cut_state_layouts(&fx.state, n_stages),
            study_dims: &fx.study_dims,
            inflow_method: &fx.inflow_method,
            stochastic: &fx.stochastic,
            initial_state: &fx.initial_state,
            inflow_scheme: SamplingScheme::InSample,
            load_scheme: SamplingScheme::InSample,
            ncs_scheme: SamplingScheme::InSample,
            historical_library: None,
            external_inflow_library: None,
            external_load_library: None,
            external_ncs_library: None,
            stages: &[],
            lag_accum_seed: &[],
            lag_weight_seed: &[],
            dcs: None,
        },
        &comm,
        ActiveSolver::new,
        LoadedPolicy::default(),
        SolverProfiles::default(),
    )
}

fn simulate_fixture(
    fx: &Fixture,
    fcf: &FutureCostFunction,
) -> Result<Vec<novomodelo_sddp::SimulationScenarioResult>, novomodelo_sddp::SimulationError> {
    let (result_tx, result_rx) = mpsc::sync_channel(32);

    let collector_thread = std::thread::spawn(move || {
        let mut all_results = Vec::new();
        while let Ok(r) = result_rx.recv() {
            all_results.push(r);
        }
        all_results
    });

    let sim_training_ctx = TrainingContext {
        node_graph: &novomodelo_sddp::test_support::chain_node_graph(&fx.stochastic),
        horizon: &fx.horizon,
        state: &fx.state,
        cut_state_layouts: &all_enabled_cut_state_layouts(&fx.state, N_STAGES),
        study_dims: &fx.study_dims,
        inflow_method: &fx.inflow_method,
        stochastic: &fx.stochastic,
        initial_state: &fx.initial_state,
        inflow_scheme: SamplingScheme::InSample,
        load_scheme: SamplingScheme::InSample,
        ncs_scheme: SamplingScheme::InSample,
        historical_library: None,
        external_inflow_library: None,
        external_load_library: None,
        external_ncs_library: None,
        stages: &[],
        lag_accum_seed: &[],
        lag_weight_seed: &[],
        dcs: None,
    };
    let state_boxes_sim = permissive_state_boxes(fx.state.n_state, N_STAGES);
    let stage_ctx_fixture_sim = base_stage_context(fx, &state_boxes_sim);
    let stage_ctx_sim = stage_ctx_fixture_sim.ctx();
    let mut sim_workspaces = vec![SolverWorkspace::new(
        0,
        0,
        ActiveSolver::new().expect("ActiveSolver::new must succeed"),
        PatchBuffer::new(&fx.state, &[], &[]),
        &sim_training_ctx,
        &stage_ctx_sim,
        WorkspaceSizing::default(),
    )];
    let comm = StubComm;

    let zero_ec = EnergyConversion {
        equivalent_productivity_mw_per_m3s: 0.0,
        reference_volume_hm3: 0.0,
        reference_outflow_m3s: 0.0,
    };
    let ec = EnergyConversionSet::new(
        vec![vec![zero_ec; N_STAGES]; N_HYDROS],
        vec![vec![0.0_f64; N_STAGES]; N_HYDROS],
        &novomodelo_sddp::test_support::minimal_hydros(N_HYDROS),
        N_STAGES,
    );

    simulate(
        &mut sim_workspaces,
        &stage_ctx_sim,
        fcf,
        &sim_training_ctx,
        &SimulationConfig {
            n_scenarios: 20,
            io_channel_capacity: 32,
            profile: Phase::Simulation.profile(),
            forward_seed: None,
        },
        SimulationOutputSpec {
            result_tx: &result_tx,
            hydro_cell_index: &novomodelo_sddp::test_support::identity_hydro_cell_index(256),
            block_hours_per_stage: &fx.stage_templates.block_hours_per_stage,
            entity_counts: &fx.entity_counts,
            generic_constraint_row_entries: &fx.stage_templates.generic_constraint_row_entries,
            pumping_consumption_mw_per_m3s: &[],
            contract_prices_per_stage: &vec![Vec::new(); N_STAGES],
            contract_slots: &[],
            diversion_upstream: &HashMap::new(),
            hydro_productivities_per_stage: &fx.stage_templates.hydro_productivities_per_stage,
            energy_conversion: &ec,
            hydro_min_storage_hm3: &[0.0; N_HYDROS],
            event_sender: None,
            extended_delivery_anchors: &[],
            transit_seed_arcs: &[],
            past_defluences: &[],
            study_stage_dates: &[],
        },
        None,
        &[],
        &comm,
        &Traversal::default(),
    )?;

    drop(result_tx);
    Ok(collector_thread
        .join()
        .expect("collector thread must not panic"))
}

fn has_nonzero_slack(scenario_results: &[novomodelo_sddp::SimulationScenarioResult]) -> bool {
    scenario_results.iter().any(|scenario| {
        scenario.stages.iter().any(|stage| {
            stage
                .hydros
                .iter()
                .any(|h| h.inflow_nonnegativity_slack_m3s > 0.0)
        })
    })
}

// ===========================================================================
// Test 1: Penalty method prevents LP infeasibility
// ===========================================================================

/// Training with `Penalty` completes without `SddpError::Infeasible` despite the
/// fixture's negative effective inflows; the slack columns keep the LP feasible.
#[test]
fn test_penalty_method_prevents_infeasibility() {
    let fx = build_fixture();
    let result = train_fixture(&fx, 5);
    assert!(
        result.is_ok(),
        "training must succeed without SddpError::Infeasible with penalty method, got: {result:?}"
    );
}

// ===========================================================================
// Test 2: Penalty slack absorbs negative inflow in simulation
// ===========================================================================

/// With the penalty method active, simulation produces at least one
/// `SimulationHydroResult` with `inflow_nonnegativity_slack_m3s > 0.0`.
#[test]
fn test_penalty_slack_value_matches_negative_inflow() {
    let fx = build_fixture();
    let n_stages = fx.stage_templates.templates.len();
    let fcf = FutureCostFunction::new(n_stages, fx.state.n_state, 1, 20, &vec![0; n_stages]);

    train_fixture(&fx, 3).expect("training must succeed before simulation");
    let scenario_results = simulate_fixture(&fx, &fcf).expect("simulate must succeed");

    assert!(
        has_nonzero_slack(&scenario_results),
        "at least one hydro must have inflow_nonnegativity_slack_m3s > 0.0 across 20 scenarios \
         with mean_m3s=0 and std_m3s=30; none found"
    );
}

// ===========================================================================
// Test 3: Simulation slack output field is populated
// ===========================================================================

/// `SimulationHydroResult.inflow_nonnegativity_slack_m3s` is populated in
/// simulation output (non-zero in at least one hydro stage) under the penalty method.
#[test]
fn test_simulation_slack_output_populated() {
    let fx = build_fixture();
    let n_stages = fx.stage_templates.templates.len();
    let fcf = FutureCostFunction::new(n_stages, fx.state.n_state, 1, 20, &vec![0; n_stages]);

    train_fixture(&fx, 3).expect("training must succeed");
    let scenario_results = simulate_fixture(&fx, &fcf).expect("simulate must succeed");

    assert_eq!(
        scenario_results.len(),
        20,
        "expected 20 simulation results, got {}",
        scenario_results.len()
    );

    assert!(
        has_nonzero_slack(&scenario_results),
        "inflow_nonnegativity_slack_m3s must be > 0.0 in at least one hydro stage result"
    );
}

/// `TruncationWithPenalty` (raw PAR value, slack capped per solve and priced)
/// trains end-to-end and the LP template carries inflow slack columns.
#[test]
fn truncation_with_penalty_training_completes() {
    let fx = build_fixture_with_method(InflowNonNegativityMethod::TruncationWithPenalty);

    assert!(
        fx.inflow_method.has_slack_columns(),
        "TruncationWithPenalty must have slack columns"
    );

    assert!(
        !fx.geometry.inflow_slack.is_empty(),
        "geometry.inflow_slack must be non-empty for TruncationWithPenalty"
    );

    let outcome = train_fixture(&fx, 5).expect("training must succeed");
    assert!(
        outcome.error.is_none(),
        "TruncationWithPenalty: training error: {:?}",
        outcome.error
    );
    assert!(
        outcome.result.iterations <= 10,
        "TruncationWithPenalty: iterations={} (expected <= 10)",
        outcome.result.iterations
    );
}

/// The two-hydro system with per-plant `inflow_nonnegativity_cost` (H1 = 100,
/// H2 = 5000 $/(m³/s·h)).
fn per_plant_inflow_penalty_system() -> System {
    let hydro_penalties_default = HydroPenalties {
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
        inflow_nonnegativity_cost: 100.0,
    };
    let mut resolved_penalties = ResolvedPenalties::new(
        &PenaltiesCountsSpec {
            n_hydros: N_HYDROS,
            n_buses: 1,
            n_lines: 0,
            n_ncs: 0,
            n_stages: N_STAGES,
        },
        &PenaltiesDefaults {
            hydro: hydro_penalties_default,
            bus: BusStagePenalties { excess_cost: 0.0 },
            line: LineStagePenalties { exchange_cost: 0.0 },
            ncs: NcsStagePenalties {
                curtailment_cost: 0.0,
            },
        },
    );
    for stage_idx in 0..N_STAGES {
        resolved_penalties
            .hydro_penalties_mut(1, stage_idx)
            .inflow_nonnegativity_cost = 5000.0;
    }

    let base_system = build_system();
    SystemBuilder::new()
        .buses(base_system.buses().to_vec())
        .hydros(base_system.hydros().to_vec())
        .stages(base_system.stages().to_vec())
        .inflow_models(base_system.inflow_models().to_vec())
        .correlation(base_system.correlation().clone())
        .bounds(base_system.bounds().clone())
        .penalties(resolved_penalties)
        .build()
        .unwrap()
}

/// Per-plant `inflow_nonnegativity_cost` (H1 = 100, H2 = 5000 $/(m³/s·h)) produces
/// distinct inflow-slack objective coefficients in the LP template: H1's equals
/// `100 * block_hours`, H2's `5000 * block_hours` (justifies the magic asserts).
#[test]
fn per_plant_inflow_penalty_differentiates_objective_coefficients() {
    let system = per_plant_inflow_penalty_system();

    let inflow_method = InflowNonNegativityMethod::Penalty;
    let par_lp = PrecomputedPar::build(
        system.inflow_models(),
        &system
            .stages()
            .iter()
            .filter(|s| s.id >= 0)
            .cloned()
            .collect::<Vec<_>>(),
        &system.hydros().iter().map(|h| h.id).collect::<Vec<_>>(),
        None,
    )
    .unwrap();
    let hydro_models = PrepareHydroModelsResult::default_from_system(&system);
    let templates = build_stage_templates_resolving_layout(
        &system,
        inflow_method,
        &par_lp,
        &PrecomputedNormal::default(),
        &hydro_models.production,
        &hydro_models.evaporation,
        &ResolvedParameters::default(),
    )
    .expect("build_stage_templates_resolving_layout must succeed");

    let tmpl0 = &templates.templates[0];
    let block_hours = 744.0_f64;

    let geometry = &templates.geometry_per_stage[0];

    assert_eq!(geometry.inflow_slack.len(), N_HYDROS);
    let h1_col = geometry.inflow_slack.start;
    let h2_col = geometry.inflow_slack.start + 1;

    let h1_obj = tmpl0.objective[h1_col];
    let h2_obj = tmpl0.objective[h2_col];

    // LP builder divides by COST_SCALE_FACTOR for conditioning.
    let cost_scale = 1_000_000.0_f64;
    let expected_h1 = 100.0 * block_hours / cost_scale;
    let expected_h2 = 5000.0 * block_hours / cost_scale;

    assert!(
        (h1_obj - expected_h1).abs() < 1e-6,
        "H1 inflow slack objective: expected {expected_h1}, got {h1_obj}"
    );
    assert!(
        (h2_obj - expected_h2).abs() < 1e-6,
        "H2 inflow slack objective: expected {expected_h2}, got {h2_obj}"
    );
    assert!(
        (h2_obj / h1_obj - 50.0).abs() < 1e-6,
        "H2/H1 objective ratio must be 50 (5000/100), got {}",
        h2_obj / h1_obj
    );
}

/// Stage-0 inflow-slack objective and template bounds per method: `Penalty`
/// carries `c_h * hours / cost_scale` on `[0, +inf)`; the truncation methods are
/// fixed at `[0, 0]` in the template (patched per solve), priced only under
/// `TruncationWithPenalty`; `None` has no slack family.
#[test]
fn inflow_slack_objective_and_template_bounds_follow_the_method() {
    let system = per_plant_inflow_penalty_system();
    let par_lp = PrecomputedPar::build(
        system.inflow_models(),
        &system
            .stages()
            .iter()
            .filter(|s| s.id >= 0)
            .cloned()
            .collect::<Vec<_>>(),
        &system.hydros().iter().map(|h| h.id).collect::<Vec<_>>(),
        None,
    )
    .unwrap();
    let hydro_models = PrepareHydroModelsResult::default_from_system(&system);
    let costs = [100.0_f64, 5000.0];
    let cost_scale = 1_000_000.0_f64;
    let block_hours = 744.0_f64;

    for method in [
        InflowNonNegativityMethod::None,
        InflowNonNegativityMethod::Penalty,
        InflowNonNegativityMethod::Truncation,
        InflowNonNegativityMethod::TruncationWithPenalty,
    ] {
        let templates = build_stage_templates_resolving_layout(
            &system,
            method,
            &par_lp,
            &PrecomputedNormal::default(),
            &hydro_models.production,
            &hydro_models.evaporation,
            &ResolvedParameters::default(),
        )
        .expect("build_stage_templates_resolving_layout must succeed");
        let tmpl0 = &templates.templates[0];
        let slack = &templates.geometry_per_stage[0].inflow_slack;

        if method == InflowNonNegativityMethod::None {
            assert!(slack.is_empty(), "{method:?}: no slack family");
            continue;
        }
        assert_eq!(slack.len(), N_HYDROS, "{method:?}: one slack per hydro");
        for (h, c_h) in costs.iter().enumerate() {
            let col = slack.start + h;
            let expected_obj = if method == InflowNonNegativityMethod::Truncation {
                0.0
            } else {
                c_h * block_hours / cost_scale
            };
            let expected_upper = if matches!(
                method,
                InflowNonNegativityMethod::Truncation
                    | InflowNonNegativityMethod::TruncationWithPenalty
            ) {
                0.0
            } else {
                f64::INFINITY
            };
            assert_eq!(
                tmpl0.objective[col], expected_obj,
                "{method:?} h{h} objective"
            );
            assert_eq!(tmpl0.col_lower[col], 0.0, "{method:?} h{h} lower bound");
            assert_eq!(
                tmpl0.col_upper[col], expected_upper,
                "{method:?} h{h} upper bound"
            );
        }
    }
}

/// External test crate cannot see parent's `#[cfg(test)]` surface; mirrors
/// `test_support::all_enabled_cut_state_layouts` via public API.
fn all_enabled_cut_state_layouts(global: &StateSpace, n_stages: usize) -> Vec<CutStateProjection> {
    let full = StageStateConfig {
        storage: true,
        inflow_lags: true,
    };
    (0..n_stages)
        .map(|_| CutStateProjection::new(global, full))
        .collect()
}

// ===========================================================================
// Closed-form slack pins per inflow method
// ===========================================================================

/// One cut per pool: `theta >= (w / COST_SCALE) * (N_HYDROS * STORAGE_MAX_HM3 - v_1 - v_2)`,
/// non-negative over the whole storage box, so every stored hm3 is worth `w` dollars.
fn water_value_fcf(fx: &Fixture, water_value_per_hm3: f64) -> FutureCostFunction {
    assert_eq!(
        COST_SCALE, fx.stage_templates.cost_scale_factor,
        "the injected cuts are written in the fixture's scaled units"
    );
    let slope = water_value_per_hm3 / COST_SCALE;
    let mut fcf = FutureCostFunction::new(N_STAGES, fx.state.n_state, 1, 20, &[0; N_STAGES]);
    for pool in 0..N_STAGES {
        fcf.add_cut(
            NodeId(0),
            pool,
            0,
            0,
            slope * STORAGE_MAX_HM3 * N_HYDROS as f64,
            &[-slope; N_HYDROS],
        );
    }
    fcf
}

fn hydro_records(
    results: &[SimulationScenarioResult],
) -> impl Iterator<Item = (u32, &SimulationHydroResult)> {
    results.iter().flat_map(|scenario_result| {
        scenario_result.stages.iter().flat_map(move |stage_result| {
            stage_result
                .hydros
                .iter()
                .map(move |record| (scenario_result.scenario_id, record))
        })
    })
}

fn count_records(
    results: &[SimulationScenarioResult],
    holds: impl Fn(&SimulationHydroResult) -> bool,
) -> usize {
    hydro_records(results)
        .filter(|(_, record)| holds(record))
        .count()
}

fn tol(x: f64) -> f64 {
    1e-6 * x.abs().max(1.0)
}

/// derived: `l* = max(-r, 0)`, and the solved z equals the raw PAR value `r` to LP tolerance.
fn truncation_amount(z: f64) -> f64 {
    (-z).max(0.0)
}

/// derived: water worth less than the penalty is bought only where an empty reservoir forces it:
/// `v_out = v_in + STAGE_HM3_PER_M3S * (z + slack) >= 0` gives
/// `slack = max(-z - v_in / STAGE_HM3_PER_M3S, 0)`.
fn forced_slack(z: f64, storage_initial_hm3: f64) -> f64 {
    (-z - storage_initial_hm3 / STAGE_HM3_PER_M3S).max(0.0)
}

fn assert_slack_equals(
    results: &[SimulationScenarioResult],
    expected_slack: impl Fn(&SimulationHydroResult) -> f64,
) {
    for (scenario, record) in hydro_records(results) {
        let z = record.incremental_inflow_m3s;
        let slack = record.inflow_nonnegativity_slack_m3s;
        let expected = expected_slack(record);
        assert!(
            (slack - expected).abs() <= tol(z),
            "scenario {scenario} stage {} hydro {}: slack {slack} != {expected} \
             (z = {z}, storage_initial = {})",
            record.stage_id,
            record.hydro_id,
            record.storage_initial_hm3,
        );
    }
}

fn assert_slack_within_truncation_amount(results: &[SimulationScenarioResult]) {
    for (scenario, record) in hydro_records(results) {
        let z = record.incremental_inflow_m3s;
        let slack = record.inflow_nonnegativity_slack_m3s;
        assert!(
            slack <= truncation_amount(z) + tol(z),
            "scenario {scenario} stage {} hydro {}: slack {slack} exceeds l* = {} (z = {z})",
            record.stage_id,
            record.hydro_id,
            truncation_amount(z),
        );
    }
}

#[test]
fn truncation_slack_is_fixed_at_the_negated_raw_inflow() {
    let fx = build_fixture_with_method(InflowNonNegativityMethod::Truncation);
    let fcf = FutureCostFunction::new(N_STAGES, fx.state.n_state, 1, 20, &[0; N_STAGES]);

    let results = simulate_fixture(&fx, &fcf).expect("simulate must succeed");

    assert_slack_equals(&results, |record| {
        truncation_amount(record.incremental_inflow_m3s)
    });
    let negative = count_records(&results, |record| record.incremental_inflow_m3s < -1.0);
    let positive = count_records(&results, |record| record.incremental_inflow_m3s > 1.0);
    let near_zero = count_records(&results, |record| {
        record.incremental_inflow_m3s.abs() <= 1e-3
    });
    assert!(
        negative > 0,
        "no record has z < -1: the fixed slack is never exercised"
    );
    assert!(
        positive > 0,
        "no record has z > 1: the zero slack is never exercised"
    );
    assert_eq!(near_zero, 0, "records with |z| <= 1e-3 sit on the l* kink");
}

#[test]
fn truncation_with_penalty_slack_is_capped_by_the_truncation_amount() {
    let fx = build_fixture_with_method(InflowNonNegativityMethod::TruncationWithPenalty);

    let above_tie = simulate_fixture(&fx, &water_value_fcf(&fx, 10.0 * TIE_WATER_VALUE_PER_HM3))
        .expect("simulate must succeed");
    assert_slack_equals(&above_tie, |record| {
        truncation_amount(record.incremental_inflow_m3s)
    });
    assert_slack_within_truncation_amount(&above_tie);

    let below_tie = simulate_fixture(&fx, &water_value_fcf(&fx, 0.1 * TIE_WATER_VALUE_PER_HM3))
        .expect("simulate must succeed");
    assert_slack_equals(&below_tie, |record| {
        forced_slack(record.incremental_inflow_m3s, record.storage_initial_hm3)
    });
    assert_slack_within_truncation_amount(&below_tie);
    let keeps_negative_inflow = count_records(&below_tie, |record| {
        record.incremental_inflow_m3s < -1.0 && record.storage_initial_hm3 >= STAGE_HM3_PER_M3S
    });
    let slack_free = count_records(&below_tie, |record| {
        record.incremental_inflow_m3s < -1.0
            && record.storage_initial_hm3
                >= STAGE_HM3_PER_M3S * record.incremental_inflow_m3s.abs() + 0.1
    });
    assert!(
        keeps_negative_inflow > 0,
        "no negative-inflow record starts with >= 1 m3/s of stored water: the cap is never below l*"
    );
    assert!(
        slack_free > 0,
        "no negative-inflow record is covered by stored water alone: the zero regime is never exercised"
    );
}

#[test]
fn truncation_with_penalty_slack_is_zero_for_non_negative_inflow() {
    let water_value = 10.0 * TIE_WATER_VALUE_PER_HM3;

    let fx = build_fixture_with_method(InflowNonNegativityMethod::TruncationWithPenalty);
    let results =
        simulate_fixture(&fx, &water_value_fcf(&fx, water_value)).expect("simulate must succeed");
    for (scenario, record) in hydro_records(&results) {
        let z = record.incremental_inflow_m3s;
        if z >= 0.0 {
            assert!(
                record.inflow_nonnegativity_slack_m3s <= tol(z),
                "scenario {scenario} stage {} hydro {}: slack {} on z = {z} >= 0",
                record.stage_id,
                record.hydro_id,
                record.inflow_nonnegativity_slack_m3s,
            );
        }
    }
    assert!(
        count_records(&results, |record| record.incremental_inflow_m3s > 1.0) > 0,
        "no record has z > 1: the zero slack is never exercised"
    );

    let twin_fx = build_fixture_with_method(InflowNonNegativityMethod::Penalty);
    let twin = simulate_fixture(&twin_fx, &water_value_fcf(&twin_fx, water_value))
        .expect("simulate must succeed");
    assert!(
        count_records(&twin, |record| {
            record.incremental_inflow_m3s >= 0.0 && record.inflow_nonnegativity_slack_m3s > 1.0
        }) > 0,
        "the uncapped penalty never buys water on a non-negative inflow: the cap has no power"
    );
}

/// derived: `z_0 = b + phi * seed`, `z_t = b + phi * z_{t-1}`, with `b = mean * (1 - phi)`.
fn zero_sigma_inflow(hydro: usize, stage: usize) -> f64 {
    let intercept = ZERO_SIGMA_MEAN_M3S * (1.0 - ZERO_SIGMA_AR_COEFFICIENT);
    (0..=stage).fold(ZERO_SIGMA_LAG_SEEDS[hydro], |lag, _| {
        intercept + ZERO_SIGMA_AR_COEFFICIENT * lag
    })
}

#[test]
fn truncation_handles_zero_sigma() {
    let shape = InflowShape {
        mean_m3s: ZERO_SIGMA_MEAN_M3S,
        ar_coefficient: Some(ZERO_SIGMA_AR_COEFFICIENT),
        residual_std_ratio: 0.0,
        ..WHITE_NOISE_SHAPE
    };
    let mut fx = build_fixture_with_shape(InflowNonNegativityMethod::Truncation, &shape);
    for (hydro, seed) in ZERO_SIGMA_LAG_SEEDS.iter().enumerate() {
        fx.initial_state[fx.state.inflow_lags.start + hydro] = *seed;
    }
    let fcf = FutureCostFunction::new(N_STAGES, fx.state.n_state, 1, 20, &[0; N_STAGES]);

    let results = simulate_fixture(&fx, &fcf).expect("a zero-sigma lagged hydro must simulate");
    assert!(
        !results.is_empty(),
        "the zero-sigma simulation returned no scenario"
    );

    let mut checked = 0;
    for scenario_result in &results {
        for stage_result in &scenario_result.stages {
            for (hydro, record) in stage_result.hydros.iter().enumerate() {
                let expected_z = zero_sigma_inflow(hydro, stage_result.stage_id as usize);
                assert!(
                    expected_z < -1.0,
                    "stage {} hydro {hydro}: expected z = {expected_z} is not negative enough",
                    stage_result.stage_id,
                );
                let z = record.incremental_inflow_m3s;
                let slack = record.inflow_nonnegativity_slack_m3s;
                assert!(
                    (z - expected_z).abs() <= tol(expected_z),
                    "scenario {} stage {} hydro {hydro}: z {z} != {expected_z}",
                    scenario_result.scenario_id,
                    stage_result.stage_id,
                );
                assert!(
                    (slack + expected_z).abs() <= tol(expected_z),
                    "scenario {} stage {} hydro {hydro}: slack {slack} != {}",
                    scenario_result.scenario_id,
                    stage_result.stage_id,
                    -expected_z,
                );
                checked += 1;
            }
        }
    }
    assert_eq!(checked, results.len() * N_STAGES * N_HYDROS);
}

#[test]
fn truncation_keeps_the_raw_par_value_as_the_next_lag() {
    let spec = negative_draw_par1_spec(CfgInflowMethod::Truncation);
    let (system, mut config) = chronological_noise_study(&spec);
    config.simulation.enabled = true;
    config.simulation.selection = Some(SimulationSelection::Sampled { num_scenarios: 1 });
    let hydro_ids: Vec<i32> = system.hydros().iter().map(|hydro| hydro.id.0).collect();
    let mut setup = build_setup_in_code(system, &config);

    let hydro_range = setup.inputs.stochastic.class_dimensions().hydro_range();
    let eta_0 = node_opening_noise(&setup, NodePos(0), 0)[hydro_range.clone()].to_vec();
    let eta_1 = node_opening_noise(&setup, NodePos(1), 0)[hydro_range].to_vec();
    assert_eq!(eta_0.len(), hydro_ids.len());
    assert_eq!(eta_1.len(), hydro_ids.len());
    let raw_0: Vec<f64> = (0..hydro_ids.len())
        .map(|h| negative_draw_raw_inflow(h, NEGATIVE_DRAW_LAG_SEED_M3S[h], eta_0[h]))
        .collect();
    let raw_1: Vec<f64> = (0..hydro_ids.len())
        .map(|h| negative_draw_raw_inflow(h, raw_0[h], eta_1[h]))
        .collect();
    for (h, (r_0, r_1)) in raw_0.iter().zip(&raw_1).enumerate() {
        assert!(
            *r_0 < -1.0 && *r_1 < -1.0,
            "hydro {h}: raw PAR values {r_0} and {r_1} are not both below -1 \
             (eta_0 = {eta_0:?}, eta_1 = {eta_1:?})"
        );
    }
    let raw = [raw_0, raw_1];

    let results = run_simulation(&mut setup, 1);
    assert_eq!(results.len(), 1);
    let scenario = &results[0];
    assert_eq!(scenario.stages.len(), raw.len());

    let within = |actual: f64, expected: f64| (actual - expected).abs() <= tol(expected);
    let lag_0_values = |stage: usize, hydro_id: i32| -> Vec<f64> {
        scenario
            .stages
            .iter()
            .filter(|stage_result| stage_result.stage_id as usize == stage)
            .flat_map(|stage_result| stage_result.inflow_lags.iter())
            .filter(|record| record.hydro_id == hydro_id && record.lag_index == 0)
            .map(|record| record.inflow_m3s)
            .collect()
    };
    let hydro_rows = |stage: usize, hydro_id: i32| -> Vec<(f64, f64)> {
        scenario
            .stages
            .iter()
            .filter(|stage_result| stage_result.stage_id as usize == stage)
            .flat_map(|stage_result| stage_result.hydros.iter())
            .filter(|record| record.hydro_id == hydro_id)
            .map(|record| {
                (
                    record.incremental_inflow_m3s,
                    record.inflow_nonnegativity_slack_m3s,
                )
            })
            .collect()
    };

    for (h, &hydro_id) in hydro_ids.iter().enumerate() {
        let lags = lag_0_values(1, hydro_id);
        assert_eq!(
            lags.len(),
            1,
            "stage 1 hydro {hydro_id}: lag-0 records {lags:?}"
        );
        assert!(
            within(lags[0], raw[0][h]),
            "stage 1 hydro {hydro_id}: incoming lag 0 is {}, expected the raw stage-0 PAR value {}",
            lags[0],
            raw[0][h],
        );
    }
    for (h, &hydro_id) in hydro_ids.iter().enumerate() {
        let lags = lag_0_values(0, hydro_id);
        assert_eq!(
            lags.len(),
            1,
            "stage 0 hydro {hydro_id}: lag-0 records {lags:?}"
        );
        assert!(
            within(lags[0], NEGATIVE_DRAW_LAG_SEED_M3S[h]),
            "stage 0 hydro {hydro_id}: incoming lag 0 is {}, expected the lag seed {}",
            lags[0],
            NEGATIVE_DRAW_LAG_SEED_M3S[h],
        );
    }
    for (stage, raw_stage) in raw.iter().enumerate() {
        for (h, &hydro_id) in hydro_ids.iter().enumerate() {
            let rows = hydro_rows(stage, hydro_id);
            assert_eq!(rows.len(), CHRONOLOGICAL_NOISE_BLOCK_HOURS.len());
            for (z, _) in rows {
                assert!(
                    within(z, raw_stage[h]),
                    "stage {stage} hydro {hydro_id}: z is {z}, expected the raw PAR value {}",
                    raw_stage[h],
                );
            }
        }
    }
    for (stage, raw_stage) in raw.iter().enumerate() {
        for (h, &hydro_id) in hydro_ids.iter().enumerate() {
            let expected_slack = truncation_amount(raw_stage[h]);
            for (_, slack) in hydro_rows(stage, hydro_id) {
                assert!(
                    within(slack, expected_slack),
                    "stage {stage} hydro {hydro_id}: slack is {slack}, expected {expected_slack}",
                );
            }
        }
    }
}
