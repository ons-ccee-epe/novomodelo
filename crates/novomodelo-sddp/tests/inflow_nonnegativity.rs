//! Integration tests for inflow non-negativity enforcement via the penalty method.
//!
//! All tests share a 2-hydro, 1-bus, 3-stage fixture whose PAR(0) inflow model
//! (`mean_m3s = 0.0`, `std_m3s = 30.0`, 10 openings/stage from seed 42) makes
//! roughly half of sampled noise values produce negative effective inflows —
//! the precondition the slack-value assertions rely on. `ActiveSolver` is used
//! throughout so the LP is solved and slack columns receive real primal values.

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
    PumpingBlockBounds, ResolvedBounds, ResolvedPenalties, SystemBuilder, ThermalBlockBounds,
    ThermalStageBounds,
    scenario::{
        CorrelationEntity, CorrelationGroup, CorrelationModel, CorrelationProfile, SamplingScheme,
    },
    temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
        StageStateConfig,
    },
};
use novomodelo_sddp::{
    LoadedPolicy, Phase, ResolvedParameters, SolverProfiles, StoppingMode, StoppingRule,
    StoppingRuleSet, TrainingConfig, build_stage_templates_resolving_layout,
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
    setup::node_graph::Traversal,
    simulate,
    simulation::{EntityCounts, SimulationConfig, SimulationOutputSpec},
    test_support::permissive_state_boxes,
    train,
    workspace::{SolverWorkspace, WorkspaceSizing},
};
use novomodelo_solver::ActiveSolver;
use novomodelo_stochastic::{
    ClassSchemes, OpeningTreeInputs, PrecomputedNormal, PrecomputedPar, StochasticContext,
    build_stochastic_context,
};

mod common;
use common::StubComm;
use common::builders::{BusSpec, HydroSpec, StageSpec, make_bus, make_hydro, make_stage};

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
fn study_dims_for(method: InflowNonNegativityMethod) -> StudyDimensions {
    StudyDimensions {
        max_deficit_segments: 1,
        inflow_method: method,
        anticipated_plants: AnticipatedPlants::default(),
        downstream_par_order: 0,
    }
}

// ===========================================================================
// System and stochastic context fixture
// ===========================================================================

const N_STAGES: usize = 3;
const N_HYDROS: usize = 2;

/// The fixture's varying axes, shared by both hydros and every stage.
#[derive(Clone)]
struct FixtureSpec {
    mean_m3s: f64,
    std_m3s: f64,
    /// Non-empty makes the inflow lag a state variable.
    ar_coefficients: Vec<f64>,
    residual_std_ratio: f64,
    inflow_nonnegativity_cost: f64,
    /// Incoming inflow lag \[m³/s\] of every hydro in the initial state.
    initial_lag_m3s: f64,
    max_storage_hm3: f64,
    initial_storage_hm3: f64,
    /// Spillage and turbined unit cost.
    release_cost: f64,
}

impl Default for FixtureSpec {
    fn default() -> Self {
        Self {
            mean_m3s: 0.0,
            std_m3s: 30.0,
            ar_coefficients: Vec::new(),
            residual_std_ratio: 1.0,
            inflow_nonnegativity_cost: 1000.0,
            initial_lag_m3s: 0.0,
            max_storage_hm3: 50.0,
            initial_storage_hm3: 0.0,
            release_cost: 0.0,
        }
    }
}

fn build_system() -> novomodelo_core::System {
    build_system_from(&FixtureSpec::default())
}

fn build_system_from(spec: &FixtureSpec) -> novomodelo_core::System {
    use novomodelo_core::entities::hydro::{HydroGenerationModel, HydroPenalties};
    use novomodelo_core::scenario::InflowModel;

    let zero_entity_penalties = HydroPenalties {
        spillage_cost: spec.release_cost,
        diversion_cost: 0.0,
        turbined_cost: spec.release_cost,
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
        inflow_nonnegativity_cost: spec.inflow_nonnegativity_cost,
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
                max_storage_hm3: spec.max_storage_hm3,
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
                        duration_hours: 744.0,
                    }],
                    block_mode: BlockMode::Parallel,
                    state_config: StageStateConfig {
                        storage: true,
                        inflow_lags: !spec.ar_coefficients.is_empty(),
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
                mean_m3s: spec.mean_m3s,
                std_m3s: spec.std_m3s,
                ar_coefficients: spec.ar_coefficients.clone(),
                residual_std_ratio: spec.residual_std_ratio,
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
        max_storage_hm3: spec.max_storage_hm3,
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
        spillage_cost: spec.release_cost,
        diversion_cost: 0.0,
        turbined_cost: spec.release_cost,
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
        inflow_nonnegativity_cost: spec.inflow_nonnegativity_cost,
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

fn build_stochastic(system: &novomodelo_core::System) -> StochasticContext {
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
    build_fixture_from(&FixtureSpec::default(), inflow_method)
}

fn build_fixture_from(spec: &FixtureSpec, inflow_method: InflowNonNegativityMethod) -> Fixture {
    let system = build_system_from(spec);

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
    let mut initial_state = vec![0.0_f64; state.n_state];
    initial_state[state.inflow_lags.clone()].fill(spec.initial_lag_m3s);
    initial_state[state.storage.clone()].fill(spec.initial_storage_hm3);
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
    let mut fcf = empty_fcf(fx);
    train_fixture_into(fx, &mut fcf, iterations)
}

fn empty_fcf(fx: &Fixture) -> FutureCostFunction {
    let n_stages = fx.stage_templates.templates.len();
    FutureCostFunction::new(n_stages, fx.state.n_state, 1, 20, &vec![0; n_stages])
}

fn train_fixture_into(
    fx: &Fixture,
    fcf: &mut FutureCostFunction,
    iterations: u64,
) -> Result<novomodelo_sddp::TrainingOutcome, novomodelo_sddp::SddpError> {
    let n_stages = fx.stage_templates.templates.len();
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
        fcf,
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

/// `TruncationWithPenalty` (raw PAR value, capped and priced slack) trains end-to-end and the
/// LP template carries inflow slack columns.
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

/// The two-hydro system with per-plant `inflow_nonnegativity_cost` H1 = 100 and
/// H2 = 5000 $/(m³/s·h).
fn per_plant_penalty_system() -> novomodelo_core::System {
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

fn templates_for(
    system: &novomodelo_core::System,
    inflow_method: InflowNonNegativityMethod,
) -> novomodelo_sddp::StageTemplates {
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
    let hydro_models = PrepareHydroModelsResult::default_from_system(system);
    build_stage_templates_resolving_layout(
        system,
        inflow_method,
        &par_lp,
        &PrecomputedNormal::default(),
        &hydro_models.production,
        &hydro_models.evaporation,
        &ResolvedParameters::default(),
    )
    .expect("build_stage_templates_resolving_layout must succeed")
}

/// Per-plant `inflow_nonnegativity_cost` (H1 = 100, H2 = 5000 $/(m³/s·h)) produces
/// distinct inflow-slack objective coefficients in the LP template: H1's equals
/// `100 * block_hours`, H2's `5000 * block_hours` (justifies the magic asserts).
#[test]
fn per_plant_inflow_penalty_differentiates_objective_coefficients() {
    let system = per_plant_penalty_system();
    let templates = templates_for(&system, InflowNonNegativityMethod::Penalty);

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

/// Per method, the inflow-slack family's objective and template bounds: absent under
/// `None`; priced at `c_h * hours` under `Penalty` and `TruncationWithPenalty`, cost 0
/// under `Truncation`; bounds `[0, +inf)` under `Penalty`, and `[0, 0]` under the
/// truncation methods, which patch them per solve.
#[test]
fn inflow_slack_objective_and_template_bounds_follow_the_method() {
    let system = per_plant_penalty_system();
    let block_hours = 744.0_f64;
    let cost_scale = 1_000_000.0_f64;
    let costs = [100.0_f64, 5000.0];
    for method in [
        InflowNonNegativityMethod::None,
        InflowNonNegativityMethod::Penalty,
        InflowNonNegativityMethod::Truncation,
        InflowNonNegativityMethod::TruncationWithPenalty,
    ] {
        let templates = templates_for(&system, method);
        let tmpl0 = &templates.templates[0];
        let slack = &templates.geometry_per_stage[0].inflow_slack;
        if method == InflowNonNegativityMethod::None {
            assert!(slack.is_empty(), "{method:?}: no slack family");
            continue;
        }
        assert_eq!(slack.len(), N_HYDROS, "{method:?}");
        for (h, &c_h) in costs.iter().enumerate() {
            let col = slack.start + h;
            let (objective, lower, upper) = (
                tmpl0.objective[col],
                tmpl0.col_lower[col],
                tmpl0.col_upper[col],
            );
            if method == InflowNonNegativityMethod::Truncation {
                assert_eq!(objective, 0.0, "{method:?} hydro {h} objective");
            } else {
                let expected = c_h * block_hours / cost_scale;
                assert!(
                    (objective - expected).abs() < 1e-6,
                    "{method:?} hydro {h} objective: expected {expected}, got {objective}"
                );
            }
            let expected_upper = if method == InflowNonNegativityMethod::Penalty {
                f64::INFINITY
            } else {
                0.0
            };
            assert_eq!(
                (lower, upper),
                (0.0, expected_upper),
                "{method:?} hydro {h} bounds"
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

mod default_method_equivalence {
    use novomodelo_io::Config;
    use novomodelo_io::config::{
        InflowNonNegativityMethod as CfgInflowMethod, SimulationSelection,
    };
    use novomodelo_sddp::SimulationScenarioResult;

    use super::common::in_code_studies::{
        ChronologicalNoiseSpec, chronological_noise_study, negative_draw_par1_spec,
    };
    use super::common::{build_setup_in_code, run_simulation};

    const TRAINING_ITERATIONS: usize = 3;
    const SIMULATION_SCENARIOS: u32 = 20;
    const SCARCE_WATER_LOAD_MW: f64 = 400.0;
    const SCARCE_WATER_INITIAL_STORAGE_HM3: f64 = 100.0;

    struct RunFingerprint {
        cut_bits: Vec<Vec<u64>>,
        simulation_json: String,
        results: Vec<SimulationScenarioResult>,
    }

    /// `None` omits the method from the config, so it resolves through the
    /// deserialization default.
    fn study_config(method: Option<CfgInflowMethod>) -> (novomodelo_core::System, Config) {
        let spec = ChronologicalNoiseSpec {
            load_mw: [SCARCE_WATER_LOAD_MW; 2],
            initial_storage_hm3: SCARCE_WATER_INITIAL_STORAGE_HM3,
            inflow_nonnegativity_cost: 1.0,
            ..negative_draw_par1_spec(method.unwrap_or(CfgInflowMethod::None))
        };
        let (system, mut config) = chronological_noise_study(&spec);
        config.simulation.enabled = true;
        config.simulation.selection = Some(SimulationSelection::Sampled {
            num_scenarios: SIMULATION_SCENARIOS,
        });
        if method.is_some() {
            return (system, config);
        }
        let mut value = serde_json::to_value(&config).expect("Config serializes");
        value["modeling"]["inflow_non_negativity"]
            .as_object_mut()
            .expect("the study config carries an inflow_non_negativity object")
            .remove("method")
            .expect("the study config declares the method explicitly");
        let omitted =
            serde_json::from_value(value).expect("Config without the method deserializes");
        (system, omitted)
    }

    fn run(method: Option<CfgInflowMethod>) -> RunFingerprint {
        let (system, config) = study_config(method);
        let mut setup = build_setup_in_code(system, &config);
        let results = run_simulation(&mut setup, TRAINING_ITERATIONS);
        let cut_bits = setup
            .fcf
            .pools
            .iter()
            .map(|pool| {
                pool.active_cuts()
                    .flat_map(|(slot, intercept, beta)| {
                        std::iter::once(slot as u64)
                            .chain(std::iter::once(intercept.to_bits()))
                            .chain(beta.iter().map(|b| b.to_bits()))
                    })
                    .collect()
            })
            .collect();
        let simulation_json =
            serde_json::to_string(&results).expect("simulation results serialize");
        RunFingerprint {
            cut_bits,
            simulation_json,
            results,
        }
    }

    /// A config that omits the inflow non-negativity method trains and simulates
    /// bit-for-bit like `truncation_with_penalty` declared explicitly, on a study
    /// with negative draws and an active correction; `penalty` differs.
    #[test]
    fn omitted_method_trains_and_simulates_like_explicit_truncation_with_penalty() {
        let omitted = run(None);
        let explicit = run(Some(CfgInflowMethod::TruncationWithPenalty));
        let penalty = run(Some(CfgInflowMethod::Penalty));

        let hydro_rows = || {
            explicit
                .results
                .iter()
                .flat_map(|scenario| &scenario.stages)
                .flat_map(|stage| &stage.hydros)
        };
        assert!(
            hydro_rows().any(|row| row.incremental_inflow_m3s < 0.0),
            "vacuity guard: no simulated row draws a negative inflow"
        );
        assert!(
            hydro_rows().any(|row| row.inflow_nonnegativity_slack_m3s > 0.0),
            "vacuity guard: no simulated row uses the inflow correction"
        );
        assert!(
            explicit.cut_bits.iter().any(|pool| !pool.is_empty()),
            "vacuity guard: training stored no cut"
        );

        assert_eq!(
            omitted.cut_bits, explicit.cut_bits,
            "the omitted method must train the same cuts as explicit truncation_with_penalty"
        );
        assert_eq!(
            omitted.simulation_json, explicit.simulation_json,
            "the omitted method must simulate like explicit truncation_with_penalty"
        );
        assert_ne!(
            penalty.simulation_json, explicit.simulation_json,
            "non-degeneracy twin: penalty must simulate differently"
        );
    }
}

mod slack_closed_forms {
    use novomodelo_sddp::cut::FutureCostFunction;
    use novomodelo_sddp::inflow_method::InflowNonNegativityMethod;
    use novomodelo_sddp::setup::NodePos;
    use novomodelo_sddp::{SimulationHydroResult, SimulationScenarioResult};

    use super::{Fixture, FixtureSpec, N_STAGES, build_fixture_from, empty_fcf, simulate_fixture};

    const COST_SCALE: f64 = 1e6;

    /// Adds to every stage's pool one cut `θ ≥ (2·max(w, 0)·v_max·n − w·Σ_h v_h)/cost_scale`,
    /// which binds over the whole storage box and prices every hydro's stored water at
    /// `w` $/hm³ at every stage (a negative `w` makes stored water a burden). Pricing every
    /// stage, not only the last, keeps the solution independent of training, so of the
    /// dual vertex a backend picks.
    fn inject_water_value(
        fcf: &mut FutureCostFunction,
        fx: &Fixture,
        max_storage_hm3: f64,
        w_per_hm3: f64,
    ) {
        let node_ids = novomodelo_sddp::test_support::chain_node_graph(&fx.stochastic).node_ids;
        let mut coefficients = vec![0.0; fx.state.n_state];
        for j in fx.state.storage.clone() {
            coefficients[j] = -w_per_hm3 / COST_SCALE;
        }
        let n_hydros = fx.state.storage.len() as f64;
        let intercept = 2.0 * w_per_hm3.max(0.0) * max_storage_hm3 * n_hydros / COST_SCALE;
        for stage in 0..N_STAGES {
            fcf.add_cut(
                node_ids[NodePos(stage)],
                stage,
                0,
                0,
                intercept,
                &coefficients,
            );
        }
    }

    fn records(
        results: &[SimulationScenarioResult],
    ) -> impl Iterator<Item = &SimulationHydroResult> {
        results
            .iter()
            .flat_map(|scenario| &scenario.stages)
            .flat_map(|stage| &stage.hydros)
    }

    const WATER_VALUE_PER_HM3: f64 = 1000.0;
    /// `0.0036 · w = 3.6` $/(m³/s·h): the slack is cheaper than stored water below it.
    const CHEAP_SLACK_COST: f64 = 1.0;
    const DEAR_SLACK_COST: f64 = 10.0;
    /// Storage room for every draw: three stages of `ζ·|z|` stay inside `[0, max]`.
    const ROOMY_MAX_STORAGE_HM3: f64 = 2000.0;
    const ROOMY_INITIAL_STORAGE_HM3: f64 = 1000.0;
    /// Above `0.0036·w = 3.6`, so releasing stored water costs more than keeping it.
    const BURDEN_RELEASE_COST: f64 = 10.0;

    fn tol(z: f64) -> f64 {
        1e-9 * z.abs().max(1.0)
    }

    fn truncation_amount(z: f64) -> f64 {
        (-z).max(0.0)
    }

    fn simulate_untrained(
        spec: &FixtureSpec,
        method: InflowNonNegativityMethod,
    ) -> Vec<SimulationScenarioResult> {
        let fx = build_fixture_from(spec, method);
        simulate_fixture(&fx, &empty_fcf(&fx)).expect("simulate must succeed")
    }

    /// Simulates the roomy-storage fixture with the water value injected at every
    /// stage.
    fn simulate_with_terminal_water_value(
        method: InflowNonNegativityMethod,
        inflow_nonnegativity_cost: f64,
    ) -> Vec<SimulationScenarioResult> {
        simulate_roomy(
            method,
            FixtureSpec {
                inflow_nonnegativity_cost,
                ..FixtureSpec::default()
            },
            WATER_VALUE_PER_HM3,
        )
    }

    fn simulate_roomy(
        method: InflowNonNegativityMethod,
        base: FixtureSpec,
        w_per_hm3: f64,
    ) -> Vec<SimulationScenarioResult> {
        let spec = FixtureSpec {
            max_storage_hm3: ROOMY_MAX_STORAGE_HM3,
            initial_storage_hm3: ROOMY_INITIAL_STORAGE_HM3,
            ..base
        };
        let fx = build_fixture_from(&spec, method);
        let mut fcf = empty_fcf(&fx);
        inject_water_value(&mut fcf, &fx, ROOMY_MAX_STORAGE_HM3, w_per_hm3);
        simulate_fixture(&fx, &fcf).expect("simulate must succeed")
    }

    /// The injected cut binds: every terminal record has interior storage and
    /// prices stored water at `w`.
    fn assert_terminal_water_value_binds(results: &[SimulationScenarioResult], label: &str) {
        assert_terminal_water_value_is(results, WATER_VALUE_PER_HM3, label);
    }

    fn assert_terminal_water_value_is(
        results: &[SimulationScenarioResult],
        w_per_hm3: f64,
        label: &str,
    ) {
        let terminal = u32::try_from(N_STAGES - 1).expect("stage index fits u32");
        let mut terminal_records = 0usize;
        for r in records(results).filter(|r| r.stage_id == terminal) {
            terminal_records += 1;
            assert!(
                r.storage_final_hm3 > 0.0 && r.storage_final_hm3 < ROOMY_MAX_STORAGE_HM3,
                "{label}: precondition: terminal storage {} must be interior",
                r.storage_final_hm3
            );
            // The output reports the water-balance dual, which is −w here.
            assert!(
                (r.water_value_per_hm3 + w_per_hm3).abs() <= 1e-6 * w_per_hm3.abs(),
                "{label}: precondition: terminal water value {} must be {}",
                r.water_value_per_hm3,
                -w_per_hm3
            );
        }
        assert!(
            terminal_records > 0,
            "{label}: precondition: no terminal-stage record"
        );
    }

    /// Under `truncation` the correction is fixed at `max(−z, 0)` on every record,
    /// also where stored water is a burden and releasing it costs, so the solve would
    /// rather take no correction.
    #[test]
    fn truncation_slack_is_fixed_at_the_negated_raw_inflow() {
        let burden = simulate_roomy(
            InflowNonNegativityMethod::Truncation,
            FixtureSpec {
                release_cost: BURDEN_RELEASE_COST,
                ..FixtureSpec::default()
            },
            -WATER_VALUE_PER_HM3,
        );
        assert_terminal_water_value_is(&burden, -WATER_VALUE_PER_HM3, "burden");
        let terminal = u32::try_from(N_STAGES - 1).expect("stage index fits u32");
        assert!(
            records(&burden).any(|r| r.stage_id == terminal && r.incremental_inflow_m3s < -1.0),
            "burden: vacuity guard: no terminal record with z < −1"
        );
        for r in records(&burden) {
            let z = r.incremental_inflow_m3s;
            assert!(
                (r.inflow_nonnegativity_slack_m3s - truncation_amount(z)).abs() <= tol(z),
                "burden, stage {} hydro {}: slack {} must equal max(−z, 0) for z = {z}",
                r.stage_id,
                r.hydro_id,
                r.inflow_nonnegativity_slack_m3s
            );
        }

        let results = simulate_untrained(
            &FixtureSpec::default(),
            InflowNonNegativityMethod::Truncation,
        );
        assert!(
            records(&results).any(|r| r.incremental_inflow_m3s < -1.0),
            "vacuity guard: no record with z < −1"
        );
        assert!(
            records(&results).any(|r| r.incremental_inflow_m3s > 1.0),
            "vacuity guard: no record with z > 1"
        );
        for r in records(&results) {
            let z = r.incremental_inflow_m3s;
            assert!(
                (r.inflow_nonnegativity_slack_m3s - truncation_amount(z)).abs() <= tol(z),
                "stage {} hydro {}: slack {} must equal max(−z, 0) for z = {z}",
                r.stage_id,
                r.hydro_id,
                r.inflow_nonnegativity_slack_m3s
            );
        }
    }

    /// Under `truncation_with_penalty` the correction never exceeds `max(−z, 0)`; on
    /// the terminal stage, where stored water is worth `w`, it is the whole amount
    /// when `c_inf < 0.0036·w` and zero when `c_inf > 0.0036·w`.
    #[test]
    fn truncation_with_penalty_slack_is_capped_by_the_truncation_amount() {
        let terminal = u32::try_from(N_STAGES - 1).expect("stage index fits u32");
        for (c_inf, takes_the_whole_amount) in [(CHEAP_SLACK_COST, true), (DEAR_SLACK_COST, false)]
        {
            let results = simulate_with_terminal_water_value(
                InflowNonNegativityMethod::TruncationWithPenalty,
                c_inf,
            );
            assert_terminal_water_value_binds(&results, &format!("c_inf {c_inf}"));
            for r in records(&results) {
                let z = r.incremental_inflow_m3s;
                assert!(
                    r.inflow_nonnegativity_slack_m3s <= truncation_amount(z) + tol(z),
                    "c_inf {c_inf}, stage {} hydro {}: slack {} exceeds max(−z, 0) for z = {z}",
                    r.stage_id,
                    r.hydro_id,
                    r.inflow_nonnegativity_slack_m3s
                );
            }
            let negative: Vec<_> = records(&results)
                .filter(|r| r.stage_id == terminal && r.incremental_inflow_m3s < -1.0)
                .collect();
            assert!(
                !negative.is_empty(),
                "c_inf {c_inf}: vacuity guard: no terminal record with z < −1"
            );
            for r in negative {
                let z = r.incremental_inflow_m3s;
                let expected = if takes_the_whole_amount {
                    truncation_amount(z)
                } else {
                    0.0
                };
                assert!(
                    (r.inflow_nonnegativity_slack_m3s - expected).abs() <= tol(z),
                    "c_inf {c_inf}, hydro {}: terminal slack {} must be {expected} for z = {z}",
                    r.hydro_id,
                    r.inflow_nonnegativity_slack_m3s
                );
            }
        }
    }

    /// Under `truncation_with_penalty` a non-negative draw takes no correction even
    /// when the slack is cheaper than stored water; the unbounded `penalty` slack
    /// does take water there.
    #[test]
    fn truncation_with_penalty_slack_is_zero_for_non_negative_inflow() {
        let results = simulate_with_terminal_water_value(
            InflowNonNegativityMethod::TruncationWithPenalty,
            CHEAP_SLACK_COST,
        );
        assert_terminal_water_value_binds(&results, "truncation_with_penalty");
        let non_negative: Vec<_> = records(&results)
            .filter(|r| r.incremental_inflow_m3s >= 0.0)
            .collect();
        assert!(
            !non_negative.is_empty(),
            "vacuity guard: no record with z ≥ 0"
        );
        for r in non_negative {
            assert!(
                r.inflow_nonnegativity_slack_m3s <= tol(r.incremental_inflow_m3s),
                "stage {} hydro {}: slack {} must be 0 for z = {}",
                r.stage_id,
                r.hydro_id,
                r.inflow_nonnegativity_slack_m3s,
                r.incremental_inflow_m3s
            );
        }

        let penalty = simulate_with_terminal_water_value(
            InflowNonNegativityMethod::Penalty,
            CHEAP_SLACK_COST,
        );
        assert!(
            records(&penalty)
                .any(|r| r.incremental_inflow_m3s >= 0.0 && r.inflow_nonnegativity_slack_m3s > 1.0),
            "non-degeneracy twin: the penalty slack must take water on some record with z ≥ 0"
        );
    }

    /// With σ = 0 the PAR value is deterministic and `truncation` fixes the
    /// correction at `−r`: `−μ` for PAR(0), and `−(μ(1 − ψ) + ψ·â)` for an AR(1)
    /// model with a zero residual ratio and incoming lag `â`.
    #[test]
    fn truncation_handles_zero_sigma() {
        let mean_m3s = -20.0;
        let par0 = FixtureSpec {
            mean_m3s,
            std_m3s: 0.0,
            ..FixtureSpec::default()
        };
        let results = simulate_untrained(&par0, InflowNonNegativityMethod::Truncation);
        assert!(
            records(&results).next().is_some(),
            "PAR(0): no simulated record"
        );
        for r in records(&results) {
            assert!(
                (r.inflow_nonnegativity_slack_m3s + mean_m3s).abs() <= tol(mean_m3s),
                "PAR(0) stage {} hydro {}: slack {} must be {}",
                r.stage_id,
                r.hydro_id,
                r.inflow_nonnegativity_slack_m3s,
                -mean_m3s
            );
        }

        let psi = 0.5;
        let incoming_lag = -10.0;
        let ar1 = FixtureSpec {
            mean_m3s,
            std_m3s: 10.0,
            ar_coefficients: vec![psi],
            residual_std_ratio: 0.0,
            initial_lag_m3s: incoming_lag,
            ..FixtureSpec::default()
        };
        let fx = build_fixture_from(&ar1, InflowNonNegativityMethod::Truncation);
        let par = fx.stochastic.par();
        for h in 0..fx.state.storage.len() {
            assert_eq!(par.psi_slice(0, h), &[psi], "AR(1): ψ = ψ* with one season");
            assert_eq!(
                par.sigma(0, h),
                0.0,
                "AR(1): σ = std · residual_std_ratio = 0"
            );
        }
        let expected = -(mean_m3s * (1.0 - psi) + psi * incoming_lag);
        let results = simulate_fixture(&fx, &empty_fcf(&fx)).expect("simulate must succeed");
        let stage0: Vec<_> = records(&results).filter(|r| r.stage_id == 0).collect();
        assert!(!stage0.is_empty(), "AR(1): no stage-0 record");
        for r in stage0 {
            assert!(
                (r.inflow_nonnegativity_slack_m3s - expected).abs() <= tol(expected),
                "AR(1) hydro {}: stage-0 slack {} must be {expected}",
                r.hydro_id,
                r.inflow_nonnegativity_slack_m3s
            );
        }
    }
}

mod raw_lag_continuation {
    use novomodelo_io::config::{
        InflowNonNegativityMethod as CfgInflowMethod, SimulationSelection,
    };
    use novomodelo_sddp::setup::NodePos;
    use novomodelo_sddp::test_support::node_opening_noise;

    use super::common::in_code_studies::{
        ChronologicalNoiseSpec, chronological_noise_study, negative_draw_par1_spec,
    };
    use super::common::{build_setup_in_code, run_simulation};

    fn tol(x: f64) -> f64 {
        1e-9 * x.abs().max(1.0)
    }

    /// Under `truncation` stage 0's raw PAR value `r₁ = b₀ + σ₀·η₀` (zero lag seed),
    /// negative, is stage 1's incoming lag, and stage 1's z is
    /// `b₁ + ψ₁·r₁ + σ₁·η₁` with the correction at `max(−z, 0)`; one opening per
    /// stage makes η known.
    #[test]
    fn truncation_keeps_the_raw_par_value_as_the_next_lag() {
        let spec = ChronologicalNoiseSpec {
            branching_factor: 1,
            ..negative_draw_par1_spec(CfgInflowMethod::Truncation)
        };
        let (system, mut config) = chronological_noise_study(&spec);
        config.simulation.enabled = true;
        config.simulation.selection = Some(SimulationSelection::Sampled { num_scenarios: 1 });
        let hydro_ids: Vec<i32> = system.hydros().iter().map(|h| h.id.0).collect();
        let mut setup = build_setup_in_code(system, &config);
        let results = run_simulation(&mut setup, 1);
        let par = setup.inputs.stochastic.par();
        let eta0 = node_opening_noise(&setup, NodePos(0), 0);
        let eta1 = node_opening_noise(&setup, NodePos(1), 0);
        let scenario = results.first().expect("one simulated scenario");
        let stage = |t: u32| {
            scenario
                .stages
                .iter()
                .find(|s| s.stage_id == t)
                .unwrap_or_else(|| panic!("stage {t} is simulated"))
        };

        let mut any_negative = false;
        for (h, &id) in hydro_ids.iter().enumerate() {
            let r1 = par.deterministic_base(0, h) + par.sigma(0, h) * eta0[h];
            eprintln!("hydro {id}: r1 = {r1} (eta0 = {})", eta0[h]);
            any_negative |= r1 < -1.0;

            let lag0 = stage(1)
                .inflow_lags
                .iter()
                .find(|l| l.hydro_id == id && l.lag_index == 0)
                .unwrap_or_else(|| panic!("hydro {id}: stage 1 records its incoming lag 0"));
            assert!(
                (lag0.inflow_m3s - r1).abs() <= tol(r1),
                "hydro {id}: stage 1's incoming lag {} must be the raw r1 = {r1}",
                lag0.inflow_m3s
            );

            let psi = par.psi_slice(1, h)[0];
            let z2 = par.deterministic_base(1, h) + psi * r1 + par.sigma(1, h) * eta1[h];
            for row in stage(1).hydros.iter().filter(|row| row.hydro_id == id) {
                assert!(
                    (row.incremental_inflow_m3s - z2).abs() <= tol(z2),
                    "hydro {id}: stage 1 z {} must be b + psi*r1 + sigma*eta = {z2}",
                    row.incremental_inflow_m3s
                );
                assert!(
                    (row.inflow_nonnegativity_slack_m3s - (-z2).max(0.0)).abs() <= tol(z2),
                    "hydro {id}: stage 1 slack {} must be max(-z, 0) for z = {z2}",
                    row.inflow_nonnegativity_slack_m3s
                );
            }
        }
        assert!(
            any_negative,
            "precondition: seed 42's single stage-0 opening gives no hydro r1 < -1"
        );
    }
}

mod raw_lag_simulation {
    use novomodelo_io::config::{
        InflowNonNegativityMethod as CfgInflowMethod, SimulationSelection,
    };

    use super::common::in_code_studies::{chronological_noise_study, negative_draw_par1_spec};
    use super::common::{build_setup_in_code, run_simulation};

    /// Under `truncation`, every simulated record's correction is `max(−z, 0)` of its
    /// own raw z, including stage-1 records whose incoming lag is negative.
    #[test]
    fn truncation_simulated_slack_matches_the_negated_inflow_at_raw_lags() {
        let (system, mut config) =
            chronological_noise_study(&negative_draw_par1_spec(CfgInflowMethod::Truncation));
        config.simulation.enabled = true;
        config.simulation.selection = Some(SimulationSelection::Sampled { num_scenarios: 20 });
        let mut setup = build_setup_in_code(system, &config);
        let results = run_simulation(&mut setup, 3);

        let mut negative_lag_with_slack = 0usize;
        for stage in results.iter().flat_map(|scenario| &scenario.stages) {
            for r in &stage.hydros {
                let z = r.incremental_inflow_m3s;
                assert!(
                    (r.inflow_nonnegativity_slack_m3s - (-z).max(0.0)).abs()
                        <= 1e-9 * z.abs().max(1.0),
                    "stage {} hydro {}: slack {} must be max(-z, 0) for z = {z}",
                    stage.stage_id,
                    r.hydro_id,
                    r.inflow_nonnegativity_slack_m3s
                );
                let negative_incoming_lag = stage
                    .inflow_lags
                    .iter()
                    .any(|l| l.hydro_id == r.hydro_id && l.lag_index == 0 && l.inflow_m3s < 0.0);
                negative_lag_with_slack += usize::from(
                    stage.stage_id == 1
                        && negative_incoming_lag
                        && r.inflow_nonnegativity_slack_m3s > 1.0,
                );
            }
        }
        assert!(
            negative_lag_with_slack > 0,
            "guard: no stage-1 record with a negative incoming lag and a slack above 1"
        );
    }
}

mod water_balance_closure {
    use std::collections::BTreeMap;
    use std::path::Path;

    use arrow::array::{Array, Float64Array, Int32Array};
    use arrow::record_batch::RecordBatch;
    use novomodelo_core::BlockMode;
    use novomodelo_io::config::{
        InflowNonNegativityMethod as CfgInflowMethod, PhaseSolverProfileConfig, SimulationSelection,
    };
    use novomodelo_io::output::simulation_writer::{ScenarioWritePayload, SimulationParquetWriter};
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

    use super::common::in_code_studies::{
        CHRONOLOGICAL_NOISE_BLOCK_HOURS, ChronologicalNoiseSpec, chronological_noise_hydro_models,
        chronological_noise_study, negative_draw_par1_spec, negative_draw_water_value_spec,
    };
    use super::common::{build_setup_in_code_with_models, run_simulation};

    /// hm³ per m³/s per hour.
    const M3S_TO_HM3: f64 = 3_600.0 / 1_000_000.0;
    const SIMULATION_SCENARIOS: u32 = 20;
    const TRAINING_ITERATIONS: usize = 3;
    /// With a unit productivity, a 400 MW load and 600 hm³ of initial storage the
    /// water is scarce, so its value exceeds `c_inf/0.0036` and the priced
    /// corrections are used; `none` stays feasible on the study's openings even
    /// after a full stage-0 release.
    const PRODUCTIVITY_MW_PER_M3S: f64 = 1.0;
    const LOAD_MW: f64 = 400.0;
    const INITIAL_STORAGE_HM3: f64 = 600.0;
    /// Under `penalty` the 600 hm³ run prices stage-0 water at exactly the penalty
    /// slack's own price (stage 1 can buy unbounded slack at `c_inf`), so taking the
    /// correction at stage 0 or at stage 1 is a tie; only the warm-started
    /// simulation takes it at stage 1, which the chronological vacuity guard needs.
    /// With 200 hm³ stage 0 is short of water itself.
    const PENALTY_SHORT_INITIAL_STORAGE_HM3: f64 = 200.0;
    /// Withdrawal \[m³/s\] of the water-value cases, on hydro id 1's stage-0 row.
    const WITHDRAWAL_M3S: f64 = 2.0;
    /// Share of the initial storage each hydro must release on the water-value study's loaded stage.
    const MIN_RELEASED_SHARE: f64 = 0.5;
    /// Warm start on and off, spelled out because the `HiGHS` simulation default is
    /// cold. The CLP backend refuses `simulation.solver.use_warm_start` at setup and
    /// always reuses an offered basis, so it runs its default only.
    #[cfg(feature = "highs")]
    const WARM_START_SETTINGS: &[Option<bool>] = &[Some(true), Some(false)];
    #[cfg(not(feature = "highs"))]
    const WARM_START_SETTINGS: &[Option<bool>] = &[None];

    struct HydroRow {
        stage: i32,
        block: i32,
        hydro: i32,
        scenario: i32,
        release: f64,
        z: f64,
        slack: f64,
        evaporation: f64,
        withdrawal_pos: f64,
        withdrawal_neg: f64,
        storage_initial: f64,
        storage_final: f64,
    }

    fn f64_col<'a>(batch: &'a RecordBatch, name: &str) -> &'a Float64Array {
        batch
            .column_by_name(name)
            .unwrap_or_else(|| panic!("simulation/hydros has column {name}"))
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap_or_else(|| panic!("{name} is Float64"))
    }

    fn i32_col<'a>(batch: &'a RecordBatch, name: &str) -> &'a Int32Array {
        batch
            .column_by_name(name)
            .unwrap_or_else(|| panic!("simulation/hydros has column {name}"))
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap_or_else(|| panic!("{name} is Int32"))
    }

    fn read_hydro_rows(dir: &Path, rows: &mut Vec<HydroRow>) {
        for entry in std::fs::read_dir(dir).expect("readable output directory") {
            let path = entry.expect("directory entry").path();
            if path.is_dir() {
                read_hydro_rows(&path, rows);
                continue;
            }
            if path.extension().is_none_or(|e| e != "parquet") {
                continue;
            }
            let file = std::fs::File::open(&path).expect("parquet file opens");
            let reader = ParquetRecordBatchReaderBuilder::try_new(file)
                .expect("parquet reader")
                .build()
                .expect("parquet batches");
            for batch in reader {
                let batch = batch.expect("parquet batch");
                let evaporation = f64_col(&batch, "evaporation_m3s");
                let block = i32_col(&batch, "block_id");
                for i in 0..batch.num_rows() {
                    rows.push(HydroRow {
                        stage: i32_col(&batch, "stage_id").value(i),
                        block: if block.is_null(i) { -1 } else { block.value(i) },
                        hydro: i32_col(&batch, "hydro_id").value(i),
                        scenario: i32_col(&batch, "scenario_id").value(i),
                        release: f64_col(&batch, "turbined_m3s").value(i)
                            + f64_col(&batch, "spillage_m3s").value(i),
                        z: f64_col(&batch, "incremental_inflow_m3s").value(i),
                        slack: f64_col(&batch, "inflow_nonnegativity_slack_m3s").value(i),
                        evaporation: if evaporation.is_null(i) {
                            0.0
                        } else {
                            evaporation.value(i)
                        },
                        withdrawal_pos: f64_col(&batch, "water_withdrawal_violation_pos_m3s")
                            .value(i),
                        withdrawal_neg: f64_col(&batch, "water_withdrawal_violation_neg_m3s")
                            .value(i),
                        storage_initial: f64_col(&batch, "storage_initial_hm3").value(i),
                        storage_final: f64_col(&batch, "storage_final_hm3").value(i),
                    });
                }
            }
        }
    }

    fn simulate_and_read_back(
        spec: &ChronologicalNoiseSpec,
        use_warm_start: Option<bool>,
    ) -> Vec<HydroRow> {
        let (system, mut config) = chronological_noise_study(spec);
        assert!(
            system.hydros().iter().all(|h| h.filling.is_none()),
            "precondition: no hydro short-circuits a filling reservoir into a checked plant"
        );
        config.simulation.enabled = true;
        config.simulation.selection = Some(SimulationSelection::Sampled {
            num_scenarios: SIMULATION_SCENARIOS,
        });
        if let Some(warm) = use_warm_start {
            let profile: PhaseSolverProfileConfig =
                serde_json::from_value(serde_json::json!({ "use_warm_start": warm }))
                    .expect("solver profile deserializes");
            config.simulation.solver = Some(profile);
        }
        let (writer_system, _) = chronological_noise_study(spec);
        let hydro_models = chronological_noise_hydro_models(spec, &system);
        let mut setup = build_setup_in_code_with_models(system, &config, hydro_models);
        let results = run_simulation(&mut setup, TRAINING_ITERATIONS);
        let tmp = tempfile::tempdir().expect("tempdir");
        let mut writer = SimulationParquetWriter::new(tmp.path(), &writer_system).expect("writer");
        for scenario in results {
            writer
                .write_scenario(ScenarioWritePayload::from(scenario))
                .expect("write_scenario");
        }
        let mut rows = Vec::new();
        read_hydro_rows(&tmp.path().join("simulation/hydros"), &mut rows);
        rows
    }

    /// Per (block mode, residual) of one run, from the written outputs only: a
    /// parallel stage closes `Δv = ζ·(z + slack − e − (W − w⁻ + w⁺)) − Σ_k τ_k·(q_k + s_k)`
    /// once per (scenario, stage, hydro); a chronological stage closes the same with
    /// its own `τ_k` and storage pair per block.
    fn residuals(spec: &ChronologicalNoiseSpec, rows: &[HydroRow]) -> Vec<(BlockMode, f64, f64)> {
        let tau: Vec<f64> = CHRONOLOGICAL_NOISE_BLOCK_HOURS
            .iter()
            .map(|h| h * M3S_TO_HM3)
            .collect();
        let zeta: f64 = tau.iter().sum();
        let withdrawal = |stage: i32, hydro: i32| {
            if stage == 0 && hydro == 1 {
                spec.water_withdrawal_m3s
            } else {
                0.0
            }
        };
        let mut out = Vec::new();
        let mut parallel: BTreeMap<(i32, i32, i32), Vec<&HydroRow>> = BTreeMap::new();
        for row in rows {
            let mode = spec.block_modes[usize::try_from(row.stage).expect("stage id")];
            match mode {
                BlockMode::Parallel => parallel
                    .entry((row.scenario, row.stage, row.hydro))
                    .or_default()
                    .push(row),
                BlockMode::Chronological => {
                    let tau_k = tau[usize::try_from(row.block).expect("chronological block id")];
                    let net = row.z + row.slack
                        - row.evaporation
                        - (withdrawal(row.stage, row.hydro) - row.withdrawal_neg
                            + row.withdrawal_pos)
                        - row.release;
                    let res = row.storage_final - row.storage_initial - tau_k * net;
                    let scale = 1.0_f64.max(row.storage_initial).max(tau_k * row.z.abs());
                    out.push((mode, res, scale));
                }
            }
        }
        for ((_, stage, hydro), block_rows) in parallel {
            let first = block_rows[0];
            let released: f64 = block_rows
                .iter()
                .map(|r| tau[usize::try_from(r.block).expect("parallel block id")] * r.release)
                .sum();
            let net = first.z + first.slack
                - first.evaporation
                - (withdrawal(stage, hydro) - first.withdrawal_neg + first.withdrawal_pos);
            let res = first.storage_final - first.storage_initial - (zeta * net - released);
            let scale = 1.0_f64.max(first.storage_initial).max(zeta * first.z.abs());
            out.push((BlockMode::Parallel, res, scale));
        }
        out
    }

    /// Every simulated water balance closes from the written `simulation/hydros`
    /// rows, under every method and with warm start on and, on `HiGHS`, off, at
    /// one tolerance.
    #[test]
    fn simulated_water_balance_closes_from_outputs() {
        let mut table: Vec<String> = Vec::new();
        let mut failures: Vec<String> = Vec::new();
        let mut negative_draw: BTreeMap<(String, String), bool> = BTreeMap::new();
        let mut uses_correction: BTreeMap<(String, String), bool> = BTreeMap::new();
        for (method, initial_storage_hm3) in [
            (CfgInflowMethod::None, INITIAL_STORAGE_HM3),
            (CfgInflowMethod::Truncation, INITIAL_STORAGE_HM3),
            (CfgInflowMethod::TruncationWithPenalty, INITIAL_STORAGE_HM3),
            (CfgInflowMethod::Penalty, INITIAL_STORAGE_HM3),
            (CfgInflowMethod::Penalty, PENALTY_SHORT_INITIAL_STORAGE_HM3),
        ] {
            for &warm in WARM_START_SETTINGS {
                let warm_label = warm.map_or("default".to_string(), |w| w.to_string());
                let spec = ChronologicalNoiseSpec {
                    inflow_nonnegativity_cost: 1.0,
                    load_mw: [LOAD_MW; 2],
                    hydro_productivity_mw_per_m3s: PRODUCTIVITY_MW_PER_M3S,
                    initial_storage_hm3,
                    ..negative_draw_par1_spec(method)
                };
                let rows = simulate_and_read_back(&spec, warm);
                for mode in [BlockMode::Parallel, BlockMode::Chronological] {
                    let stages: Vec<i32> = (0..2)
                        .filter(|&t| spec.block_modes[t] == mode)
                        .map(|t| i32::try_from(t).expect("stage id"))
                        .collect();
                    let mode_rows = || rows.iter().filter(|r| stages.contains(&r.stage));
                    let key = (format!("{method:?}"), format!("{mode:?}"));
                    *negative_draw.entry(key.clone()).or_default() |=
                        mode_rows().any(|r| r.z < 0.0);
                    *uses_correction.entry(key).or_default() |= mode_rows().any(|r| r.slack > 0.0);
                }
                let mut max_by_mode: BTreeMap<String, f64> = BTreeMap::new();
                for (mode, res, scale) in residuals(&spec, &rows) {
                    let key = format!("{mode:?}");
                    let entry = max_by_mode.entry(key.clone()).or_insert(0.0);
                    *entry = entry.max(res.abs());
                    if res.abs() > 1e-6 * scale {
                        failures.push(format!(
                            "{method:?} v0={initial_storage_hm3} {key} warm_start={warm:?}: residual {res:e} hm3 > 1e-6*{scale}"
                        ));
                    }
                }
                for (mode, max) in max_by_mode {
                    table.push(format!(
                        "{method:?} (v0 {initial_storage_hm3}) | {mode} | {warm_label} | {max:.3e}"
                    ));
                }
            }
        }
        eprintln!("max |residual| (hm3) per (method | block mode | warm start):");
        for line in &table {
            eprintln!("  {line}");
        }
        for ((method, mode), any) in &negative_draw {
            assert!(*any, "{method} {mode}: vacuity guard: no row with z < 0");
        }
        for ((method, mode), any) in &uses_correction {
            if method != "None" {
                assert!(
                    *any,
                    "{method} {mode}: vacuity guard: no row uses the correction"
                );
            }
        }
        assert!(
            failures.is_empty(),
            "water balance does not close:\n{}",
            failures.join("\n")
        );
    }

    /// On the water-value study each hydro turbines its stored water for the stage-1 load and
    /// hydro id 1 withholds its stage-0 withdrawal, so the balance closes with a non-zero
    /// release and a withdrawal violation, for both block-mode assignments and both
    /// truncation methods.
    #[test]
    fn simulated_water_balance_closes_with_a_nonzero_release() {
        let scenarios = usize::try_from(SIMULATION_SCENARIOS).expect("scenario count fits usize");
        for block_modes in [
            [BlockMode::Parallel, BlockMode::Chronological],
            [BlockMode::Chronological, BlockMode::Parallel],
        ] {
            for method in [
                CfgInflowMethod::Truncation,
                CfgInflowMethod::TruncationWithPenalty,
            ] {
                let case = format!("water value {method:?} {block_modes:?}");
                let spec = ChronologicalNoiseSpec {
                    block_modes,
                    water_withdrawal_m3s: WITHDRAWAL_M3S,
                    ..negative_draw_water_value_spec(method)
                };
                let rows = simulate_and_read_back(&spec, None);
                for (mode, res, scale) in residuals(&spec, &rows) {
                    assert!(
                        res.abs() <= 1e-6 * scale,
                        "{case} {mode:?}: residual {res:e} hm3 > 1e-6*{scale}"
                    );
                }
                assert!(
                    rows.iter().any(|r| r.z < 0.0) && rows.iter().any(|r| r.slack > 0.0),
                    "{case}: vacuity guard: no negative draw or no correction used"
                );

                let mut released_hm3: BTreeMap<(i32, i32), f64> = BTreeMap::new();
                let mut withheld_withdrawals = 0;
                for row in &rows {
                    let stage = usize::try_from(row.stage).expect("stage id");
                    if spec.load_mw[stage] > 0.0 {
                        let block = usize::try_from(row.block).expect("block id");
                        *released_hm3.entry((row.scenario, row.hydro)).or_default() +=
                            CHRONOLOGICAL_NOISE_BLOCK_HOURS[block] * M3S_TO_HM3 * row.release;
                    }
                    if row.stage == 0
                        && row.hydro == 1
                        && row.withdrawal_neg > 0.5 * spec.water_withdrawal_m3s
                    {
                        withheld_withdrawals += 1;
                    }
                }
                assert_eq!(
                    released_hm3.len(),
                    scenarios * 2,
                    "{case}: (scenario, hydro) releases checked"
                );
                for ((scenario, hydro), volume_hm3) in released_hm3 {
                    assert!(
                        volume_hm3 > MIN_RELEASED_SHARE * spec.initial_storage_hm3,
                        "{case} scenario {scenario} hydro {hydro}: released {volume_hm3} hm3 \
                         on the loaded stage"
                    );
                }
                assert_eq!(
                    withheld_withdrawals,
                    scenarios * CHRONOLOGICAL_NOISE_BLOCK_HOURS.len(),
                    "{case}: stage-0 rows of hydro id 1 that withhold the withdrawal"
                );
            }
        }
    }
}

mod truncation_order_invariance {
    use std::collections::BTreeMap;

    use novomodelo_io::config::{
        InflowNonNegativityMethod as CfgInflowMethod, SimulationSelection,
    };
    use novomodelo_sddp::setup::{NodePos, StageIdx};
    use novomodelo_sddp::test_support::{inflow_slack_bounds_for_probe, node_opening_noise};
    use novomodelo_solver::ActiveSolver;

    use super::common::in_code_studies::{
        ChronologicalNoiseSpec, chronological_noise_study, negative_draw_par1_spec,
    };
    use super::common::{StubComm, build_setup_in_code, run_simulation, stored_trial_states};

    /// Per (scenario, stage, block, hydro id): slack, z and incoming lag-0 bits;
    /// per (stage, stored trial state, opening, hydro id): the probe's slack bound bits.
    struct Fingerprint {
        records: BTreeMap<(u32, u32, Option<u32>, i32), [u64; 3]>,
        bounds: BTreeMap<(usize, usize, usize, i32), [u64; 2]>,
        max_bound: f64,
        min_lag: f64,
        max_lstar_gap: f64,
    }

    fn run(reverse_declaration_order: bool) -> Fingerprint {
        let spec = ChronologicalNoiseSpec {
            reverse_declaration_order,
            ..negative_draw_par1_spec(CfgInflowMethod::Truncation)
        };
        let (system, mut config) = chronological_noise_study(&spec);
        config.simulation.enabled = true;
        config.simulation.selection = Some(SimulationSelection::Sampled { num_scenarios: 20 });
        let mut hydro_ids: Vec<i32> = system.hydros().iter().map(|h| h.id.0).collect();
        hydro_ids.sort_unstable();

        let mut setup = build_setup_in_code(system, &config);
        setup.set_export_states(true);
        let mut solver = ActiveSolver::new().expect("ActiveSolver::new");
        let outcome = setup
            .train(&mut solver, &StubComm, 1, ActiveSolver::new, None, None)
            .expect("training");
        assert!(
            outcome.error.is_none(),
            "training error: {:?}",
            outcome.error
        );

        let mut pool = setup
            .create_workspace_pool(&StubComm, 1, ActiveSolver::new)
            .expect("workspace pool");
        let ws = &mut pool.workspaces[0];
        let training_ctx = setup.training_ctx();
        let mut bounds = BTreeMap::new();
        let (mut max_bound, mut max_lstar_gap) = (0.0_f64, 0.0_f64);
        let n_stages = setup.inputs.node_graph.nodes.len();
        for stage in 1..n_stages {
            let n_openings = setup.inputs.node_graph.nodes[NodePos(stage)].openings.len;
            for (k, x) in stored_trial_states(&setup, NodePos(stage - 1))
                .iter()
                .enumerate()
            {
                for o in 0..n_openings {
                    let noise = node_opening_noise(&setup, NodePos(stage), o);
                    let (lower, upper) = inflow_slack_bounds_for_probe(
                        ws,
                        &training_ctx,
                        StageIdx(stage),
                        x,
                        &noise,
                    );
                    for (h, &id) in hydro_ids.iter().enumerate() {
                        bounds.insert((stage, k, o, id), [lower[h].to_bits(), upper[h].to_bits()]);
                        max_bound = max_bound.max(upper[h]);
                    }
                    max_lstar_gap = max_lstar_gap.max((upper[0] - upper[1]).abs());
                }
            }
        }

        let (system, _) = chronological_noise_study(&spec);
        let mut simulation_setup = build_setup_in_code(system, &config);
        let results = run_simulation(&mut simulation_setup, 1);
        let mut records = BTreeMap::new();
        let mut min_lag = f64::INFINITY;
        for scenario in &results {
            for stage in &scenario.stages {
                for r in &stage.hydros {
                    let lag = stage
                        .inflow_lags
                        .iter()
                        .find(|l| l.hydro_id == r.hydro_id && l.lag_index == 0)
                        .map_or(0.0, |l| l.inflow_m3s);
                    min_lag = min_lag.min(lag);
                    records.insert(
                        (scenario.scenario_id, stage.stage_id, r.block_id, r.hydro_id),
                        [
                            r.inflow_nonnegativity_slack_m3s.to_bits(),
                            r.incremental_inflow_m3s.to_bits(),
                            lag.to_bits(),
                        ],
                    );
                }
            }
        }
        Fingerprint {
            records,
            bounds,
            max_bound,
            min_lag,
            max_lstar_gap,
        }
    }

    /// Under `truncation`, the per-solve correction, the raw z and the incoming
    /// lags of every simulated record, and the slack bounds at every archived trial
    /// state, are bit-identical with the entity declaration order reversed, keyed by
    /// hydro id; ψ and σ differ between the hydros, so a positional mix-up moves them.
    #[test]
    fn truncation_lag_sequence_is_order_invariant() {
        let base = run(false);
        let repeat = run(false);
        let reversed = run(true);

        assert!(base.max_bound > 0.0, "guard: no positive slack bound");
        assert!(base.min_lag < 0.0, "guard: no negative incoming lag");
        assert!(
            base.max_lstar_gap > 1.0,
            "guard: the two hydros' l* never differ by more than 1"
        );
        assert!(
            !base.records.is_empty() && !base.bounds.is_empty(),
            "guard: nothing compared"
        );

        assert!(
            base.records == repeat.records,
            "simulation records differ between two fresh runs"
        );
        assert!(
            base.bounds == repeat.bounds,
            "slack bounds differ between two fresh runs"
        );
        assert!(
            base.records == reversed.records,
            "simulation records differ under declaration-order reversal"
        );
        assert!(
            base.bounds == reversed.bounds,
            "slack bounds differ under declaration-order reversal"
        );
    }
}
