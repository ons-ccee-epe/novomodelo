//! Closed-form cut coefficients of the compact LP on fixtures small enough to derive by hand.
//!
//! No fixture is trained: its resident pool is empty or holds one cut the test injects, so each
//! expected coefficient follows from the fixture's constants and the solve's own duals. On every
//! in-code fixture the hydros empty their reservoirs into a marginal thermal, so a stored hm³ is
//! worth `λ(ρ) = −(c·ρ − c_q)/(M·K)` in cost-scaled units, with `c_q` the turbined cost, and the
//! realized inflow enters its water balance at `κ = hours·M` hm³ per m³/s.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use chrono::NaiveDate;
use novomodelo_core::entities::hydro::{HydroGenerationModel, HydroPenalties};
use novomodelo_core::scenario::{InflowModel, LoadModel};
use novomodelo_core::temporal::{
    Block, BlockMode, NoiseMethod, ScenarioSourceConfig, SeasonCycleType, SeasonDefinition,
    SeasonMap, Stage, StageRiskConfig, StageStateConfig,
};
use novomodelo_core::{
    BoundsCountsSpec, BoundsDefaults, BusStagePenalties, ConstraintExpression, ContractBlockBounds,
    DeficitSegment, EntityId, GenericConstraint, HorizonGraph, Hydro, HydroBlockBounds,
    HydroStageBounds, HydroStorage, InitialConditions, LineBlockBounds, LineStagePenalties,
    LinearTerm, NcsStagePenalties, PenaltiesCountsSpec, PenaltiesDefaults, PenaltyBasis,
    PumpingBlockBounds, ResolvedBounds, ResolvedGenericConstraintBounds, ResolvedPenalties,
    SlackConfig, System, SystemBuilder, ThermalBlockBounds, ThermalStageBounds, VariableRef,
};
use novomodelo_io::Config;
use novomodelo_io::config::{
    EstimationConfig, ExportsConfig, InflowNonNegativityConfig,
    InflowNonNegativityMethod as CfgInflowMethod, ModelingConfig, ParallelismConfig, PolicyConfig,
    RowSelectionConfig, SimulationConfig as IoSimulationConfig, StoppingMode, StoppingRuleConfig,
    TrainingConfig, TrainingSelection, TrainingSolverConfig, UpperBoundEvaluationConfig,
};
use novomodelo_sddp::hydro_models::{
    EvaporationModel, EvaporationModelSet, LinearizedEvaporation, PrepareHydroModelsResult,
    ProductionModelSet, ResolvedProductionModel,
};
use novomodelo_sddp::indexer::{CutSlot, CutStateProjection, HydroSys, StateDim};
use novomodelo_sddp::setup::{NodePos, StageIdx};
use novomodelo_sddp::test_support::template_structure::state_lag_dims;
use novomodelo_sddp::test_support::{
    ProbeCut, ProbeCuts, ProbeOpening, empty_row_batch, inflow_slack_bounds_for_probe,
    opening_cut_for_probe, oracle_initial_state,
};
use novomodelo_sddp::{DEFAULT_COST_SCALE_FACTOR, StudySetup, build_cut_row_batch_into};
use novomodelo_solver::{ActiveSolver, SolverInterface};

use crate::common::builders::{
    BusSpec, HydroSpec, StageSpec, ThermalSpec, make_bus, make_hydro, make_stage, make_thermal,
};
use crate::common::in_code_studies::{
    CHRONOLOGICAL_NOISE_BLOCK_HOURS, CHRONOLOGICAL_NOISE_RELEASE_COST,
    chronological_noise_hydro_models, chronological_noise_study, negative_draw_water_value_spec,
};
use crate::common::oracle_tolerance::{UnscaledTolerance, assert_first_try, rounding_factor};
use crate::common::{StubComm, build_setup_in_code_with_models, fresh_system_and_setup_with};

const M3S_TO_HM3: f64 = 3_600.0 / 1_000_000.0;
const BUS_ID: EntityId = EntityId(10);
const THERMAL_ID: EntityId = EntityId(20);
const SOLE_HYDRO_ID: EntityId = EntityId(1);
const UPSTREAM_ID: EntityId = EntityId(1);
const TARGET_ID: EntityId = EntityId(2);

const THERMAL_COST_PER_MWH: f64 = 100.0;
const THERMAL_MAX_MW: f64 = 1_000.0;
const DEFICIT_COST_PER_MWH: f64 = 500.0;
const MAX_STORAGE_HM3: f64 = 1_000.0;
const MAX_TURBINED_M3S: f64 = 1_000.0;
const MAX_GENERATION_MW: f64 = 1_000.0;
const SPILLAGE_COST: f64 = 0.01;
const EVAPORATION_VIOLATION_COST: f64 = 5_000.0;
const INFLOW_MEAN_M3S: f64 = 100.0;
const INFLOW_STD_M3S: f64 = 8.0;
const MONTH_HOURS: f64 = 744.0;
const FORTNIGHT_HOURS: f64 = 336.0;
const LOAD_MW: f64 = 300.0;
const INITIAL_STORAGE_HM3: f64 = 200.0;
const PRODUCTIVITY: f64 = 0.5;

const TERMINAL_PSI: f64 = 0.3;
const TERMINAL_TWIN_PSI: f64 = 0.6;

const PAR2_PHI: [f64; 2] = [0.5, 0.25];
/// `β_inj(h, 0)` and `β_inj(h, 1)`, in cost-scaled units per m³/s.
const INJECTED_LAG_SLOPES: [f64; 2] = [-0.01, -0.004];
const EVAPORATION: LinearizedEvaporation = LinearizedEvaporation {
    intercept_m3s: 1.0,
    volume_slope_m3s_per_hm3: 0.01,
};
const EVAPORATION_REFERENCE_HM3: f64 = 100.0;
const INJECTED_INTERCEPT: f64 = 10.0;

const UPSTREAM_PHI: f64 = 0.4;
const TARGET_PHI: f64 = 0.3;
const UPSTREAM_INITIAL_STORAGE_HM3: f64 = 50.0;
const PREFILLING_LOAD_MW: f64 = 400.0;
/// The upstream hydro enters at stage 2, so it is `PreFilling` at stages 0 and 1.
const UPSTREAM_ENTRY_STAGE: i32 = 2;

const RANGE_STAGE: usize = 1;
const RANGE_CONSTRAINT_ID: i32 = 30;
/// The range row's coefficient on `z_h`, which is `G` of the row's inflow coupling.
const RANGE_INFLOW_COEFFICIENT: f64 = 1.0;
/// Relative distance within which an unscaled template bound is read as a fixture bound.
const BOUND_MATCH_TOLERANCE: f64 = 1e-12;

/// The thermal cost [`chronological_noise_study`] writes inline.
const NEGATIVE_DRAW_THERMAL_COST_PER_MWH: f64 = 100.0;
/// Per-hydro ẑ targets of the negative-draw study's stage 1, each below zero.
const NEGATIVE_DRAW_TARGETS_M3S: [f64; 2] = [-15.0, -10.0];
/// Per-hydro ẑ targets of the negative-draw study's stage 1, each above zero.
const POSITIVE_DRAW_TARGETS_M3S: [f64; 2] = [20.0, 15.0];

const WEEKLY_PAR_DECK: &str = "examples/deterministic/d29-weekly-par-noise-sharing";
/// The deck's first stage with a producer pool whose week does not close its month.
const WEEKLY_PAR_NON_FINALIZING_STAGE: usize = 1;
/// The deck's last week, which closes its month.
const WEEKLY_PAR_FINALIZING_STAGE: usize = 3;

const POWER_FLOOR: f64 = 1e-9;
const SEPARATION_FLOOR: f64 = 1e-6;

struct HydroCase {
    id: EntityId,
    name: &'static str,
    downstream: Option<EntityId>,
    entry_stage_id: Option<i32>,
    initial_storage_hm3: f64,
    productivity: f64,
    ar_coefficients: Vec<f64>,
    evaporation: Option<LinearizedEvaporation>,
}

impl HydroCase {
    fn sole(ar_coefficients: Vec<f64>) -> Self {
        Self {
            id: SOLE_HYDRO_ID,
            name: "H",
            downstream: None,
            entry_stage_id: None,
            initial_storage_hm3: INITIAL_STORAGE_HM3,
            productivity: PRODUCTIVITY,
            ar_coefficients,
            evaporation: None,
        }
    }
}

/// A soft generic row `lower ≤ z_h ≤ upper` on the sole hydro's inflow at one stage, with its
/// slack priced per hour.
#[derive(Clone, Copy)]
struct InflowRange {
    stage: usize,
    lower_m3s: f64,
    upper_m3s: f64,
    penalty_per_hour: f64,
}

/// One parallel block per monthly stage; the hydros are declared in ascending id order, so
/// their declaration order is the system order.
struct StudyCase {
    name: &'static str,
    n_stages: usize,
    block_hours: f64,
    load_mw: f64,
    hydros: Vec<HydroCase>,
    /// The stage whose state leaves the inflow lags out of the cut, so the pool before it is
    /// storage-only.
    storage_only_stage: Option<usize>,
    inflow_range: Option<InflowRange>,
}

fn stage_date(index: usize) -> NaiveDate {
    let month = u32::try_from(index + 1).expect("a fixture stage index fits a month");
    NaiveDate::from_ymd_opt(2024, month, 1).expect("fixture stages start on a 2024 month")
}

fn monthly_season_map() -> SeasonMap {
    SeasonMap {
        cycle_type: SeasonCycleType::Monthly,
        seasons: (0..12)
            .map(|id| SeasonDefinition {
                id,
                label: format!("Month{}", id + 1),
                month_start: u32::try_from(id + 1).expect("a month fits u32"),
                day_start: None,
                month_end: None,
                day_end: None,
            })
            .collect(),
    }
}

fn fixture_penalties() -> HydroPenalties {
    HydroPenalties {
        spillage_cost: SPILLAGE_COST,
        diversion_cost: 0.0,
        turbined_cost: 0.0,
        storage_violation_below_cost: 0.0,
        filling_target_violation_cost: 0.0,
        turbined_violation_below_cost: 0.0,
        outflow_violation_below_cost: 0.0,
        outflow_violation_above_cost: 0.0,
        generation_violation_below_cost: 0.0,
        evaporation_violation_cost: EVAPORATION_VIOLATION_COST,
        water_withdrawal_violation_cost: 0.0,
        water_withdrawal_violation_pos_cost: 0.0,
        water_withdrawal_violation_neg_cost: 0.0,
        evaporation_violation_pos_cost: EVAPORATION_VIOLATION_COST,
        evaporation_violation_neg_cost: EVAPORATION_VIOLATION_COST,
        inflow_nonnegativity_cost: 0.0,
    }
}

fn fixture_hydro(hydro: &HydroCase) -> Hydro {
    make_hydro(
        hydro.id,
        HydroSpec {
            name: hydro.name.to_string(),
            operational_start_date: stage_date(0),
            bus_id: BUS_ID,
            downstream_id: hydro.downstream,
            entry_stage_id: hydro.entry_stage_id,
            min_storage_hm3: 0.0,
            max_storage_hm3: MAX_STORAGE_HM3,
            max_turbined_m3s: MAX_TURBINED_M3S,
            max_generation_mw: MAX_GENERATION_MW,
            generation_model: HydroGenerationModel::ConstantProductivity,
            penalties: fixture_penalties(),
            ..Default::default()
        },
    )
}

fn fixture_stages(case: &StudyCase) -> Vec<Stage> {
    (0..case.n_stages)
        .map(|i| {
            make_stage(
                i,
                StageSpec {
                    start_date: stage_date(i),
                    end_date: stage_date(i + 1),
                    season_id: Some(i),
                    blocks: vec![Block {
                        index: 0,
                        name: "B0".to_string(),
                        duration_hours: case.block_hours,
                    }],
                    block_mode: BlockMode::Parallel,
                    state_config: StageStateConfig {
                        storage: true,
                        inflow_lags: case.storage_only_stage != Some(i),
                    },
                    risk_config: StageRiskConfig::Expectation,
                    scenario_config: ScenarioSourceConfig {
                        branching_factor: 1,
                        noise_method: NoiseMethod::Saa,
                    },
                },
            )
        })
        .collect()
}

fn stage_id(index: usize) -> i32 {
    i32::try_from(index).expect("a fixture stage index fits i32")
}

fn fixture_bounds(case: &StudyCase) -> ResolvedBounds {
    ResolvedBounds::new(
        &BoundsCountsSpec {
            n_hydros: case.hydros.len(),
            n_thermals: 1,
            n_lines: 0,
            n_pumping: 0,
            n_contracts: 0,
            n_stages: case.n_stages,
            k_max: 0,
        },
        &BoundsDefaults {
            hydro: HydroStageBounds {
                min_storage_hm3: 0.0,
                max_storage_hm3: MAX_STORAGE_HM3,
                filling_min_rate_m3s: 0.0,
                water_withdrawal_m3s: 0.0,
            },
            hydro_block: HydroBlockBounds {
                max_turbined_m3s: MAX_TURBINED_M3S,
                max_generation_mw: MAX_GENERATION_MW,
                ..Default::default()
            },
            thermal: ThermalStageBounds {
                cost_per_mwh: THERMAL_COST_PER_MWH,
            },
            thermal_block: ThermalBlockBounds {
                min_generation_mw: 0.0,
                max_generation_mw: THERMAL_MAX_MW,
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

fn fixture_penalty_table(case: &StudyCase) -> ResolvedPenalties {
    ResolvedPenalties::new(
        &PenaltiesCountsSpec {
            n_hydros: case.hydros.len(),
            n_buses: 1,
            n_lines: 0,
            n_ncs: 0,
            n_stages: case.n_stages,
        },
        &PenaltiesDefaults {
            hydro: fixture_penalties(),
            bus: BusStagePenalties { excess_cost: 0.0 },
            line: LineStagePenalties { exchange_cost: 0.0 },
            ncs: NcsStagePenalties {
                curtailment_cost: 0.0,
            },
        },
    )
}

fn build_system(case: &StudyCase) -> System {
    let bus = make_bus(
        BUS_ID,
        BusSpec {
            name: "B".to_string(),
            operational_start_date: stage_date(0),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: DEFICIT_COST_PER_MWH,
            }],
            excess_cost: 0.0,
        },
    );
    let thermal = make_thermal(
        THERMAL_ID,
        ThermalSpec {
            name: "T".to_string(),
            operational_start_date: stage_date(0),
            bus_id: BUS_ID,
            cost_per_mwh: THERMAL_COST_PER_MWH,
            min_generation_mw: 0.0,
            max_generation_mw: THERMAL_MAX_MW,
            anticipated_config: None,
            ..Default::default()
        },
    );
    let inflow_models = case
        .hydros
        .iter()
        .flat_map(|hydro| {
            (0..case.n_stages).map(move |i| InflowModel {
                hydro_id: hydro.id,
                stage_id: stage_id(i),
                mean_m3s: INFLOW_MEAN_M3S,
                std_m3s: INFLOW_STD_M3S,
                ar_coefficients: hydro.ar_coefficients.clone(),
                residual_std_ratio: 1.0,
                annual: None,
            })
        })
        .collect();
    let load_models = (0..case.n_stages)
        .map(|i| LoadModel {
            bus_id: BUS_ID,
            stage_id: stage_id(i),
            mean_mw: case.load_mw,
            std_mw: 0.0,
        })
        .collect();
    let initial_conditions = InitialConditions {
        storage: case
            .hydros
            .iter()
            .map(|hydro| HydroStorage {
                hydro_id: hydro.id,
                value_hm3: hydro.initial_storage_hm3,
            })
            .collect(),
        filling_storage: vec![],
        past_anticipated_commitments: vec![],
        recent_observations: vec![],
        past_defluences: vec![],
    };
    let builder = SystemBuilder::new()
        .buses(vec![bus])
        .thermals(vec![thermal])
        .hydros(case.hydros.iter().map(fixture_hydro).collect())
        .stages(fixture_stages(case))
        .inflow_models(inflow_models)
        .load_models(load_models)
        .bounds(fixture_bounds(case))
        .penalties(fixture_penalty_table(case))
        .initial_conditions(initial_conditions)
        .policy_graph(HorizonGraph {
            season_map: Some(monthly_season_map()),
            ..HorizonGraph::default()
        });
    let builder = match case.inflow_range {
        Some(range) => {
            let (constraint, bounds) = inflow_range_constraint(range);
            builder
                .generic_constraints(vec![constraint])
                .resolved_generic_bounds(bounds)
        }
        None => builder,
    };
    builder
        .build()
        .unwrap_or_else(|e| panic!("{}: the system must build: {e:?}", case.name))
}

fn inflow_range_constraint(
    range: InflowRange,
) -> (GenericConstraint, ResolvedGenericConstraintBounds) {
    let constraint = GenericConstraint {
        id: EntityId(RANGE_CONSTRAINT_ID),
        name: "inflow_range".to_string(),
        description: None,
        expression: ConstraintExpression {
            terms: vec![LinearTerm::literal(
                RANGE_INFLOW_COEFFICIENT,
                VariableRef::HydroInflow {
                    hydro_id: SOLE_HYDRO_ID,
                    block_id: None,
                },
            )],
        },
        slack: Some(SlackConfig {
            penalty: range.penalty_per_hour,
            penalty_basis: PenaltyBasis::PerHour,
        }),
        bound_lower: None,
        bound_upper: None,
    };
    let id_map: HashMap<i32, usize> = [(RANGE_CONSTRAINT_ID, 0)].into_iter().collect();
    let rows = [(
        RANGE_CONSTRAINT_ID,
        stage_id(range.stage),
        None::<i32>,
        Some(range.lower_m3s),
        Some(range.upper_m3s),
    )];
    let bounds = ResolvedGenericConstraintBounds::new(&id_map, rows.into_iter());
    (constraint, bounds)
}

fn fixture_config() -> Config {
    Config {
        schema: None,
        modeling: ModelingConfig {
            inflow_non_negativity: InflowNonNegativityConfig {
                method: CfgInflowMethod::None,
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
        simulation: IoSimulationConfig::default(),
        exports: ExportsConfig::default(),
        estimation: EstimationConfig::default(),
    }
}

fn fixture_hydro_models(case: &StudyCase, system: &System) -> PrepareHydroModelsResult {
    let mut models = PrepareHydroModelsResult::default_from_system(system);
    models.production = ProductionModelSet::new(
        case.hydros
            .iter()
            .map(|hydro| {
                vec![
                    ResolvedProductionModel::ConstantProductivity {
                        productivity: hydro.productivity,
                    };
                    case.n_stages
                ]
            })
            .collect(),
        system.hydros(),
        case.n_stages,
    );
    models.evaporation = EvaporationModelSet::new(
        case.hydros
            .iter()
            .map(|hydro| {
                hydro
                    .evaporation
                    .map_or(EvaporationModel::None, |coefficients| {
                        EvaporationModel::Linearized {
                            coefficients: vec![coefficients; case.n_stages],
                            reference_volumes_hm3: vec![EVAPORATION_REFERENCE_HM3; case.n_stages],
                        }
                    })
            })
            .collect(),
    );
    models
}

/// The untrained compact setup of `case` and κ of its stage `stage`, in hm³ per m³/s.
fn compact_study(case: &StudyCase, stage: usize) -> (StudySetup, f64) {
    let system = build_system(case);
    let stage_hours: f64 = system.stages()[stage]
        .blocks
        .iter()
        .map(|block| block.duration_hours)
        .sum();
    let models = fixture_hydro_models(case, &system);
    let setup = build_setup_in_code_with_models(system, &fixture_config(), models);
    (setup, stage_hours * M3S_TO_HM3)
}

/// Appends to `pool` one cut of intercept [`INJECTED_INTERCEPT`] whose coefficient at each
/// `(slot, β)` is β and zero elsewhere, and returns the pool's active slots.
fn inject_cut(setup: &mut StudySetup, pool: usize, slopes: &[(usize, f64)]) -> Vec<usize> {
    let mut coefficients = vec![0.0; setup.fcf.pools[pool].state_dimension];
    for &(slot, beta) in slopes {
        coefficients[slot] = beta;
    }
    let node_id = setup.inputs.node_graph.node_ids[NodePos(pool)];
    setup
        .fcf
        .add_cut(node_id, pool, 1, 0, INJECTED_INTERCEPT, &coefficients);
    setup
        .fcf
        .active_cuts(pool)
        .map(|(slot, _, _)| slot)
        .collect()
}

/// The cut opening 0 of `stage` yields for pool `stage − 1`, solved on `stage`'s compact template
/// with the rows of its pool's `pool_slots`; `label` names the solve in a failure.
fn compact_cut(
    setup: &StudySetup,
    stage: usize,
    x_hat: &[f64],
    noise: &[f64],
    pool_slots: &[usize],
    label: &str,
) -> ProbeCut {
    let ctx = setup.stage_ctx();
    let training_ctx = setup.training_ctx();
    let template = &setup.inputs.stage_data.stage_templates.templates[stage];
    let mut cut_rows = empty_row_batch();
    build_cut_row_batch_into(
        &mut cut_rows,
        &setup.fcf,
        stage,
        setup.stage_state(),
        &training_ctx.cut_state_layouts[stage],
        &template.col_scale,
    );
    assert_eq!(
        cut_rows.num_rows,
        pool_slots.len(),
        "{label}: stage {stage}: one cut row per pool slot"
    );
    let mut workspaces = setup
        .create_workspace_pool(&StubComm, 1, ActiveSolver::new)
        .expect("the workspace pool must build");
    let ws = &mut workspaces.workspaces[0];
    ws.solver.load_model(template);
    ws.solver.add_rows(&cut_rows);

    let mut cut = ProbeCut::new(&training_ctx);
    let before = ws.solver.statistics();
    opening_cut_for_probe(
        ws,
        &ctx,
        &training_ctx,
        &ProbeOpening {
            stage: StageIdx(stage),
            pinned_state: x_hat,
            raw_noise: noise,
            cuts: ProbeCuts {
                pool: &setup.fcf.pools[stage],
                pool_id: stage,
                slots: pool_slots,
            },
            extracted_pool_id: stage - 1,
            inflow_slack_bounds: None,
        },
        &mut cut,
    )
    .unwrap_or_else(|e| panic!("{label}: stage {stage}: the compact opening must solve: {e:?}"));
    let after = ws.solver.statistics();
    assert_first_try(
        &format!("{label}: stage {stage}: compact opening"),
        &before,
        &after,
    );
    cut
}

fn storage_slot(projection: &CutStateProjection, h: usize) -> usize {
    (0..projection.n_slots())
        .find(|&slot| projection.global_state_index(CutSlot::new(slot)) == StateDim::new(h))
        .unwrap_or_else(|| panic!("hydro {h}'s storage has a cut slot"))
}

fn lag_slot(projection: &CutStateProjection, n_hydros: usize, h: usize, l: usize) -> usize {
    projection.lag_point().full().start + l * n_hydros + h
}

/// The global state dimension of hydro `h`'s lag `l`.
fn lag_dim(setup: &StudySetup, h: usize, l: usize) -> usize {
    let state = setup.training_ctx().state;
    state_lag_dims(state).start + l * state.hydro_count + h
}

fn incoming_storage_col_scale(setup: &StudySetup, stage: usize, h: usize) -> f64 {
    let col = setup
        .stage_state()
        .state_to_lp_incoming_column(StateDim::new(h))
        .get();
    setup.inputs.stage_data.stage_templates.templates[stage].col_scale[col]
}

/// The row scale of the one base row `ẑ_h` enters at `stage`.
fn coupled_row_scale(setup: &StudySetup, stage: usize, h: usize) -> f64 {
    let ctx = setup.stage_ctx();
    let rows: Vec<usize> = ctx.inflow_coupling[stage]
        .hydro_entries(HydroSys::new(h))
        .map(|(row, _)| row)
        .collect();
    assert_eq!(
        rows.len(),
        1,
        "stage {stage}: hydro {h}'s inflow enters one parallel water balance"
    );
    setup.inputs.stage_data.stage_templates.templates[stage].row_scale[rows[0]]
}

/// `Σ_r |G_r|·tol(π_r)` over the base rows `ẑ_h` enters at `stage`: the reported-dual tolerance
/// of `(Gᵀπ)_h`.
fn coupled_dual_tolerance(
    setup: &StudySetup,
    stage: usize,
    h: usize,
    tol: UnscaledTolerance,
) -> f64 {
    let ctx = setup.stage_ctx();
    let row_scale = &setup.inputs.stage_data.stage_templates.templates[stage].row_scale;
    ctx.inflow_coupling[stage]
        .hydro_entries(HydroSys::new(h))
        .map(|(row, coefficient)| coefficient.abs() * tol.row_dual(row_scale[row]))
        .sum()
}

/// `λ(ρ)`: the cost-scaled value of one hm³ turbined at `productivity` in place of the thermal.
fn water_value(productivity: f64) -> f64 {
    -(THERMAL_COST_PER_MWH * productivity) / (M3S_TO_HM3 * DEFAULT_COST_SCALE_FACTOR)
}

fn psi_at(setup: &StudySetup, stage: usize, h: usize) -> Vec<f64> {
    setup
        .training_ctx()
        .stochastic
        .par()
        .psi_slice(stage, h)
        .to_vec()
}

fn opening_noise(setup: &StudySetup, stage: usize) -> Vec<f64> {
    setup
        .training_ctx()
        .stochastic
        .tree_view()
        .opening(stage, 0)
        .to_vec()
}

fn assert_has_power(fixture: &str, quantity: &str, value: f64) {
    assert!(
        value.abs() > POWER_FLOOR,
        "{fixture}: the {quantity} {value:e} is within {POWER_FLOOR:e} of zero, so the closed \
         form could hold vacuously"
    );
}

fn assert_non_degenerate(fixture: &str, quantity: &str, value: f64) {
    assert!(
        value != 0.0 && value.abs() != 1.0,
        "{fixture}: the {quantity} {value} is 0 or ±1, which hides a sign or a scale"
    );
}

/// A derived value, compared within `2γ_n·S + added`; `slot` is `None` for the objective.
struct ClosedForm<'a> {
    fixture: &'a str,
    quantity: &'a str,
    slot: Option<usize>,
    want: f64,
    n: usize,
    abs_sum: f64,
    added: f64,
}

impl ClosedForm<'_> {
    fn assert_solved(&self, got: f64) {
        let bound = 2.0 * rounding_factor(self.n) * self.abs_sum + self.added;
        let gap = (got - self.want).abs();
        let at = self
            .slot
            .map_or_else(String::new, |slot| format!("slot {slot} "));
        assert!(
            gap <= bound,
            "{}: {at}{}: derived {:e}, solved {got:e}, gap {gap:e} > bound {bound:e} (n = {})",
            self.fixture,
            self.quantity,
            self.want,
            self.n
        );
    }
}

fn terminal_par1_case(psi: f64, block_hours: f64) -> StudyCase {
    StudyCase {
        name: "terminal PAR(1)",
        n_stages: 2,
        block_hours,
        load_mw: LOAD_MW,
        hydros: vec![HydroCase::sole(vec![psi])],
        storage_only_stage: None,
        inflow_range: None,
    }
}

fn assert_terminal_stage_lag_price(psi_coefficient: f64, block_hours: f64) {
    const STAGE: usize = 1;
    let case = terminal_par1_case(psi_coefficient, block_hours);
    let fixture = case.name;
    let (setup, kappa) = compact_study(&case, STAGE);
    let x_hat = oracle_initial_state(&setup);
    let training_ctx = setup.training_ctx();
    let noise = opening_noise(&setup, STAGE);
    let cut = compact_cut(&setup, STAGE, &x_hat, &noise, &[], fixture);

    let extracted = &training_ctx.cut_state_layouts[STAGE - 1];
    let storage = storage_slot(extracted, 0);
    let lag = lag_slot(extracted, 1, 0, 0);
    let psi = psi_at(&setup, STAGE, 0)[0];
    assert_has_power(fixture, "storage coefficient", cut.coefficients[storage]);
    assert_non_degenerate(fixture, "lag-1 coefficient ψ", psi);
    assert_non_degenerate(fixture, "inflow coupling κ", kappa);
    assert_non_degenerate(fixture, "product ψ·κ", psi * kappa);

    let tol = UnscaledTolerance::from_active_profile();
    let lambda = water_value(PRODUCTIVITY);
    // n = 5: c·ρ, M, M·K and ÷ derive λ, rc ÷ col_scale solves it; added: v_in's rc tolerance.
    ClosedForm {
        fixture,
        quantity: "storage coefficient",
        slot: Some(storage),
        want: lambda,
        n: 5,
        abs_sum: lambda.abs(),
        added: tol.reduced_cost(incoming_storage_col_scale(&setup, STAGE, 0)),
    }
    .assert_solved(cut.coefficients[storage]);
    // n = 13: λ (4), κ (2), ψ·κ·λ (2); d·row_scale, G·π, ψ·g and G = −κ (2) solve it;
    // added: |ψ|·κ times the water balance's row-dual tolerance.
    ClosedForm {
        fixture,
        quantity: "lag-0 coefficient",
        slot: Some(lag),
        want: psi * kappa * lambda,
        n: 13,
        abs_sum: (psi * kappa * lambda).abs(),
        added: psi.abs() * kappa * tol.row_dual(coupled_row_scale(&setup, STAGE, 0)),
    }
    .assert_solved(cut.coefficients[lag]);
}

/// derived: the hydro empties into a marginal thermal with no future, so the storage coefficient
/// is `λ = −c·ρ/(M·K)` and ẑ, `ψ·â` higher per unit lag, enters the balance at `κ = 744·M`:
/// the lag coefficient is `ψ·κ·λ`. A negated coupling term flips it against the storage sign.
#[test]
fn a_terminal_stage_prices_a_par1_lag_at_psi_kappa_times_the_water_value() {
    assert_terminal_stage_lag_price(TERMINAL_PSI, MONTH_HOURS);
}

/// derived: the same `λ` and `ψ·κ·λ` at ψ = 0.6 and a 336-hour stage, so `κ = 336·M`; a
/// coupling term of the wrong sign or of another stage's κ misses it.
#[test]
fn a_terminal_stage_prices_a_par1_lag_at_psi_kappa_times_the_water_value_on_another_psi_and_stage_length()
 {
    assert_terminal_stage_lag_price(TERMINAL_TWIN_PSI, FORTNIGHT_HOURS);
}

fn injected_cut_case() -> StudyCase {
    StudyCase {
        name: "PAR(2) with one injected cut",
        n_stages: 4,
        block_hours: MONTH_HOURS,
        load_mw: LOAD_MW,
        hydros: vec![HydroCase {
            evaporation: Some(EVAPORATION),
            ..HydroCase::sole(PAR2_PHI.to_vec())
        }],
        storage_only_stage: None,
        inflow_range: None,
    }
}

fn assert_injected_cut_chain_rule(lag_slopes: [f64; 2]) {
    const STAGE: usize = 2;
    let case = injected_cut_case();
    let fixture = case.name;
    let (mut setup, kappa) = compact_study(&case, STAGE);
    let resident_lags = {
        let resident = &setup.training_ctx().cut_state_layouts[STAGE];
        [lag_slot(resident, 1, 0, 0), lag_slot(resident, 1, 0, 1)]
    };
    let pool_slots = inject_cut(
        &mut setup,
        STAGE,
        &[
            (resident_lags[0], lag_slopes[0]),
            (resident_lags[1], lag_slopes[1]),
        ],
    );
    let x_hat = oracle_initial_state(&setup);
    let training_ctx = setup.training_ctx();
    let noise = opening_noise(&setup, STAGE);
    let cut = compact_cut(&setup, STAGE, &x_hat, &noise, &pool_slots, fixture);

    let extracted = &training_ctx.cut_state_layouts[STAGE - 1];
    let storage = storage_slot(extracted, 0);
    let storage_coefficient = cut.coefficients[storage];
    let psi = psi_at(&setup, STAGE, 0);
    let storage_col_scale = incoming_storage_col_scale(&setup, STAGE, 0);
    assert_has_power(fixture, "storage coefficient", storage_coefficient);
    assert_non_degenerate(fixture, "lag-1 coefficient ψ₁", psi[0]);
    assert_non_degenerate(fixture, "lag-2 coefficient ψ₂", psi[1]);
    assert!(
        psi[0] != psi[1],
        "{fixture}: ψ₁ = ψ₂ = {} hides a lag offset",
        psi[0]
    );
    assert_non_degenerate(fixture, "inflow coupling κ", kappa);
    assert!(
        storage_col_scale != 1.0,
        "{fixture}: the incoming storage column scale is 1, which hides its unscaling"
    );
    assert!(
        lag_slopes[0] != 0.0 && lag_slopes[1] != 0.0 && lag_slopes[0] != lag_slopes[1],
        "{fixture}: the injected lag slopes {lag_slopes:?} must be distinct and non-zero"
    );
    let [(slot, y)] = cut.cut_dual_slots[..] else {
        panic!(
            "{fixture}: exactly the injected cut must bind, got {:?}",
            cut.cut_dual_slots
        );
    };
    assert!(
        slot == pool_slots[0] && y > 0.0,
        "{fixture}: the injected cut at slot {} must bind, got y = {y} at slot {slot}",
        pool_slots[0]
    );

    let tol = UnscaledTolerance::from_active_profile();
    let evaporation_slope = EVAPORATION.volume_slope_m3s_per_hm3 / 2.0;
    let retained = 1.0 - kappa * evaporation_slope;
    let pi = storage_coefficient / retained;
    let g = kappa * pi + y * lag_slopes[0];
    let g_abs = (kappa * pi).abs() + (y * lag_slopes[0]).abs();
    let dual_term = kappa
        * (tol.reduced_cost(storage_col_scale) / retained
            + tol.row_dual(coupled_row_scale(&setup, STAGE, 0)));
    // n = 22: E, M, κ, κ·E, 1 − κE, π, κ·π, y·β₀, g, ψ₁·g, y·β₁ and + derive it; rc ÷ col_scale,
    // d·row_scale, G·π, G = −κ (2), y·β₀, +, ψ₁·g, y·β₁ and + solve it; added: ψ₁ times κ times
    // the storage rc tolerance over 1 − κE plus the water balance's row-dual tolerance.
    ClosedForm {
        fixture,
        quantity: "lag-0 coefficient",
        slot: Some(lag_slot(extracted, 1, 0, 0)),
        want: psi[0] * g + y * lag_slopes[1],
        n: 22,
        abs_sum: psi[0].abs() * g_abs + (y * lag_slopes[1]).abs(),
        added: psi[0].abs() * dual_term,
    }
    .assert_solved(cut.coefficients[lag_slot(extracted, 1, 0, 0)]);
    // n = 18: E, M, κ, κ·E, 1 − κE, π, κ·π, y·β₀, g and ψ₂·g derive it; rc ÷ col_scale,
    // d·row_scale, G·π, G = −κ (2), y·β₀, + and ψ₂·g solve it; added: the same tolerance times ψ₂.
    ClosedForm {
        fixture,
        quantity: "lag-1 coefficient",
        slot: Some(lag_slot(extracted, 1, 0, 1)),
        want: psi[1] * g,
        n: 18,
        abs_sum: psi[1].abs() * g_abs,
        added: psi[1].abs() * dual_term,
    }
    .assert_solved(cut.coefficients[lag_slot(extracted, 1, 0, 1)]);
}

/// derived: the injected cut binds at y = 1, so `g = κ·π + y·β_inj(h, 0)` with
/// `π = s/(1 − κ·E)` from the storage coefficient s, and `β(h, 0) = ψ₁·g + y·β_inj(h, 1)`,
/// `β(h, 1) = ψ₂·g`. Dropping w, reading β(h, l) for β(h, l + 1), negating `y·β` in g or
/// multiplying the storage dual by its non-unit column scale each moves a lag coefficient off it.
#[test]
fn an_injected_binding_cut_adds_its_next_lag_coefficient_to_the_producer_cut() {
    assert_injected_cut_chain_rule(INJECTED_LAG_SLOPES);
}

/// derived: the same chain rule with `β_inj(h, 0)` and `β_inj(h, 1)` swapped, so an offset read
/// of the injected lags lands on the other slope.
#[test]
fn an_injected_binding_cut_adds_its_next_lag_coefficient_to_the_producer_cut_with_the_injected_lags_swapped()
 {
    let [first, second] = INJECTED_LAG_SLOPES;
    assert_injected_cut_chain_rule([second, first]);
}

fn distinct_trial_lags_case() -> StudyCase {
    StudyCase {
        name: "PAR(2) with one injected cut at distinct trial lags",
        hydros: vec![HydroCase::sole(PAR2_PHI.to_vec())],
        ..injected_cut_case()
    }
}

/// derived: the injected cut binds, so θ is its row's right-hand side `α + β₀·ẑ + β₁·â₀` with
/// `ẑ = μ + φ₁(â₀ − μ) + φ₂(â₁ − μ) + σ·η` at the trial lags `â = (120, 80)`, and the hydro empties
/// into the marginal thermal, so `Q = c·(L·T − ρ·(v + κ·ẑ)/M)/K + θ`. A cut row whose shifted
/// slot reads `â₁` for `â₀` moves Q by `β₁·(â₁ − â₀)`.
#[test]
fn an_injected_binding_cut_is_evaluated_at_the_trial_lag_one_step_back() {
    const STAGE: usize = 2;
    const TRIAL_LAGS_M3S: [f64; 2] = [120.0, 80.0];
    let case = distinct_trial_lags_case();
    let fixture = case.name;
    let (mut setup, kappa) = compact_study(&case, STAGE);
    let resident_lags = {
        let resident = &setup.training_ctx().cut_state_layouts[STAGE];
        [lag_slot(resident, 1, 0, 0), lag_slot(resident, 1, 0, 1)]
    };
    let pool_slots = inject_cut(
        &mut setup,
        STAGE,
        &[
            (resident_lags[0], INJECTED_LAG_SLOPES[0]),
            (resident_lags[1], INJECTED_LAG_SLOPES[1]),
        ],
    );
    let mut x_hat = oracle_initial_state(&setup);
    for (l, lag) in TRIAL_LAGS_M3S.into_iter().enumerate() {
        x_hat[lag_dim(&setup, 0, l)] = lag;
    }
    let noise = opening_noise(&setup, STAGE);
    let cut = compact_cut(&setup, STAGE, &x_hat, &noise, &pool_slots, fixture);

    let [near, far] = TRIAL_LAGS_M3S;
    let [beta_now, beta_shifted] = INJECTED_LAG_SLOPES;
    assert_non_degenerate(fixture, "trial lag â₀", near);
    assert_non_degenerate(fixture, "trial lag â₁", far);
    assert!(
        beta_now != 0.0 && beta_shifted != 0.0,
        "{fixture}: the injected lag slopes {INJECTED_LAG_SLOPES:?} must be non-zero"
    );
    assert!(
        (beta_shifted * (far - near)).abs() > SEPARATION_FLOOR,
        "{fixture}: β₁·(â₁ − â₀) is within {SEPARATION_FLOOR:e} of zero, which hides a lag offset"
    );
    let [(slot, y)] = cut.cut_dual_slots[..] else {
        panic!(
            "{fixture}: exactly the injected cut must bind, got {:?}",
            cut.cut_dual_slots
        );
    };
    assert!(
        slot == pool_slots[0] && y > 0.0,
        "{fixture}: the injected cut at slot {} must bind, got y = {y} at slot {slot}",
        pool_slots[0]
    );

    let z_hat = INFLOW_MEAN_M3S
        + PAR2_PHI[0] * (near - INFLOW_MEAN_M3S)
        + PAR2_PHI[1] * (far - INFLOW_MEAN_M3S)
        + INFLOW_STD_M3S * noise[0];
    let theta = INJECTED_INTERCEPT + beta_now * z_hat + beta_shifted * near;
    let turbined_mwh = PRODUCTIVITY * (INITIAL_STORAGE_HM3 + kappa * z_hat) / M3S_TO_HM3;
    let z_abs = INFLOW_MEAN_M3S
        + PAR2_PHI[0] * (near - INFLOW_MEAN_M3S).abs()
        + PAR2_PHI[1] * (far - INFLOW_MEAN_M3S).abs()
        + INFLOW_STD_M3S * noise[0].abs();
    let turbined_abs = PRODUCTIVITY * (INITIAL_STORAGE_HM3 + kappa * z_abs) / M3S_TO_HM3;
    let theta_abs = INJECTED_INTERCEPT + beta_now.abs() * z_abs + beta_shifted.abs() * near;
    // n = 45: ẑ (8), κ (2), the turbined energy (4), the thermal cost (3), θ (4), their sum and
    // ÷ K derive it; the solver's sums of cost × primal over the stage's columns solve it; no
    // reported dual enters.
    ClosedForm {
        fixture,
        quantity: "objective",
        slot: None,
        want: THERMAL_COST_PER_MWH * (LOAD_MW * MONTH_HOURS - turbined_mwh)
            / DEFAULT_COST_SCALE_FACTOR
            + theta,
        n: 45,
        abs_sum: THERMAL_COST_PER_MWH * (LOAD_MW * MONTH_HOURS + turbined_abs)
            / DEFAULT_COST_SCALE_FACTOR
            + theta_abs,
        added: 0.0,
    }
    .assert_solved(cut.objective);
}

fn prefilling_case(upstream_productivity: f64, target_productivity: f64) -> StudyCase {
    StudyCase {
        name: "PreFilling upstream to target",
        n_stages: 3,
        block_hours: MONTH_HOURS,
        load_mw: PREFILLING_LOAD_MW,
        hydros: vec![
            HydroCase {
                id: UPSTREAM_ID,
                name: "U",
                downstream: Some(TARGET_ID),
                entry_stage_id: Some(UPSTREAM_ENTRY_STAGE),
                initial_storage_hm3: UPSTREAM_INITIAL_STORAGE_HM3,
                productivity: upstream_productivity,
                ar_coefficients: vec![UPSTREAM_PHI],
                evaporation: None,
            },
            HydroCase {
                id: TARGET_ID,
                name: "T",
                productivity: target_productivity,
                ..HydroCase::sole(vec![TARGET_PHI])
            },
        ],
        storage_only_stage: None,
        inflow_range: None,
    }
}

fn assert_prefilling_lag_price(upstream_productivity: f64, target_productivity: f64) {
    const STAGE: usize = 1;
    const UPSTREAM: usize = 0;
    const TARGET: usize = 1;
    let case = prefilling_case(upstream_productivity, target_productivity);
    let fixture = case.name;
    let (mut setup, kappa) = compact_study(&case, STAGE);
    let upstream_storage = storage_slot(&setup.training_ctx().cut_state_layouts[STAGE], UPSTREAM);
    let upstream_value = water_value(upstream_productivity);
    let pool_slots = inject_cut(&mut setup, STAGE, &[(upstream_storage, upstream_value)]);
    let x_hat = oracle_initial_state(&setup);
    let training_ctx = setup.training_ctx();
    let noise = opening_noise(&setup, STAGE);
    let cut = compact_cut(&setup, STAGE, &x_hat, &noise, &pool_slots, fixture);

    let extracted = &training_ctx.cut_state_layouts[STAGE - 1];
    let solved_upstream = cut.coefficients[storage_slot(extracted, UPSTREAM)];
    let target_storage = storage_slot(extracted, TARGET);
    let solved_target = cut.coefficients[target_storage];
    let psi = psi_at(&setup, STAGE, UPSTREAM)[0];
    assert_has_power(fixture, "upstream storage coefficient", solved_upstream);
    assert_has_power(fixture, "target storage coefficient", solved_target);
    assert_non_degenerate(fixture, "upstream lag-1 coefficient ψ", psi);
    assert_non_degenerate(fixture, "inflow coupling κ", kappa);
    let separation = (solved_upstream - solved_target).abs();
    assert!(
        separation > SEPARATION_FLOOR * solved_upstream.abs().max(solved_target.abs()),
        "{fixture}: the upstream and target water values {solved_upstream:e} and \
         {solved_target:e} must differ by more than {SEPARATION_FLOOR:e} relative"
    );
    let target_value = water_value(target_productivity);
    assert!(
        (solved_upstream < solved_target) == (upstream_value < target_value),
        "{fixture}: the water values {solved_upstream:e} (upstream) and {solved_target:e} \
         (target) must order as the productivities give"
    );

    let tol = UnscaledTolerance::from_active_profile();
    // n = 5: c·ρ, M, M·K and ÷ derive λ_T, rc ÷ col_scale solves it; added: v_in's rc tolerance.
    ClosedForm {
        fixture,
        quantity: "target storage coefficient",
        slot: Some(target_storage),
        want: target_value,
        n: 5,
        abs_sum: target_value.abs(),
        added: tol.reduced_cost(incoming_storage_col_scale(&setup, STAGE, TARGET)),
    }
    .assert_solved(solved_target);
    // n = 13: λ_T (4), κ (2), ψ·κ·λ_T (2); d·row_scale, G·π, ψ·g and G = −κ (2) solve it;
    // added: |ψ|·κ times the coupled row's row-dual tolerance.
    let upstream_lag = lag_slot(extracted, case.hydros.len(), UPSTREAM, 0);
    ClosedForm {
        fixture,
        quantity: "upstream lag-0 coefficient",
        slot: Some(upstream_lag),
        want: psi * kappa * target_value,
        n: 13,
        abs_sum: (psi * kappa * target_value).abs(),
        added: psi.abs() * kappa * tol.row_dual(coupled_row_scale(&setup, STAGE, UPSTREAM)),
    }
    .assert_solved(cut.coefficients[upstream_lag]);
}

/// derived: the dormant upstream's inflow fills the target's balance, so the target empties at
/// `λ_T = −c·ρ_T/(M·K)` while the injected cut holds the upstream at `λ_U`: the upstream lag
/// coefficient is `ψ_U·κ·λ_T`. Coupling it on the upstream's own row prices it at `ψ_U·κ·λ_U`.
#[test]
fn a_prefilling_upstream_inflow_is_priced_at_the_target_water_value() {
    assert_prefilling_lag_price(0.8, PRODUCTIVITY);
}

/// derived: the same `ψ_U·κ·λ_T` with the productivities swapped, so `λ_U > λ_T` and the
/// upstream's own water value lies on the other side of the target's.
#[test]
fn a_prefilling_upstream_inflow_is_priced_at_the_target_water_value_with_the_productivities_swapped()
 {
    assert_prefilling_lag_price(PRODUCTIVITY, 0.8);
}

/// One side of an [`InflowRange`] that `ẑ` leaves, so the row's slack covers the violation.
#[derive(Clone, Copy)]
enum RangeSide {
    Lower,
    Upper,
}

impl RangeSide {
    fn name(self) -> &'static str {
        match self {
            Self::Lower => "lower side",
            Self::Upper => "upper side",
        }
    }

    /// The sign of `∂Q/∂bound` at the binding bound: a higher lower bound costs more slack, a
    /// higher upper bound less.
    fn dual_sign(self) -> f64 {
        match self {
            Self::Lower => 1.0,
            Self::Upper => -1.0,
        }
    }

    fn bound(self, range: InflowRange) -> f64 {
        match self {
            Self::Lower => range.lower_m3s,
            Self::Upper => range.upper_m3s,
        }
    }

    fn violation(self, range: InflowRange, z_hat: f64) -> f64 {
        match self {
            Self::Lower => range.lower_m3s - z_hat,
            Self::Upper => z_hat - range.upper_m3s,
        }
    }
}

fn range_case(range: InflowRange) -> StudyCase {
    StudyCase {
        name: "terminal PAR(1) with a range inflow row",
        inflow_range: Some(range),
        ..terminal_par1_case(TERMINAL_PSI, MONTH_HOURS)
    }
}

/// The template row whose unscaled bounds are the fixture's `[L, U]`.
fn range_row(setup: &StudySetup, stage: usize, range: InflowRange) -> usize {
    let template = &setup.inputs.stage_data.stage_templates.templates[stage];
    let matches = |scaled: f64, row_scale: f64, want: f64| {
        (scaled / row_scale - want).abs() <= BOUND_MATCH_TOLERANCE * want.abs().max(1.0)
    };
    let rows: Vec<usize> = (0..template.num_rows)
        .filter(|&row| {
            let row_scale = template.row_scale[row];
            matches(template.row_lower[row], row_scale, range.lower_m3s)
                && matches(template.row_upper[row], row_scale, range.upper_m3s)
        })
        .collect();
    let [row] = rows[..] else {
        panic!(
            "stage {stage}: exactly one row must carry the range [{}, {}], got {rows:?}",
            range.lower_m3s, range.upper_m3s
        );
    };
    row
}

fn assert_range_row_lag_price(range: InflowRange, solves: [(RangeSide, f64); 2]) {
    let case = range_case(range);
    let fixture = case.name;
    let (setup, kappa) = compact_study(&case, RANGE_STAGE);
    let training_ctx = setup.training_ctx();
    let noise = opening_noise(&setup, RANGE_STAGE);
    let par = training_ctx.stochastic.par();
    let base = par.deterministic_base(RANGE_STAGE, 0);
    let innovation = par.sigma(RANGE_STAGE, 0) * noise[0];
    let psi = psi_at(&setup, RANGE_STAGE, 0)[0];
    let row = range_row(&setup, RANGE_STAGE, range);
    let row_scale = setup.inputs.stage_data.stage_templates.templates[RANGE_STAGE].row_scale[row];
    let outcomes: Vec<(RangeSide, f64, f64, ProbeCut)> = solves
        .iter()
        .map(|&(side, z_target)| {
            let lag_value = (z_target - base - innovation) / psi;
            let z_hat = base + psi * lag_value + innovation;
            assert!(
                side.violation(range, z_hat) > 0.0,
                "{fixture}: {}: ẑ = {z_hat} must leave [{}, {}] on that side",
                side.name(),
                range.lower_m3s,
                range.upper_m3s
            );
            let mut x_hat = oracle_initial_state(&setup);
            x_hat[lag_dim(&setup, 0, 0)] = lag_value;
            let label = format!("{fixture}: {}", side.name());
            let cut = compact_cut(&setup, RANGE_STAGE, &x_hat, &noise, &[], &label);
            (side, lag_value, z_hat, cut)
        })
        .collect();

    assert_non_degenerate(fixture, "lag-1 coefficient ψ", psi);
    assert_non_degenerate(fixture, "inflow coupling κ", kappa);
    for (side, _, _, cut) in &outcomes {
        let generic_dual = cut.dual[row] * row_scale;
        assert!(
            side.dual_sign() * generic_dual > POWER_FLOOR,
            "{fixture}: {}: the range row's dual {generic_dual:e} must exceed {POWER_FLOOR:e} \
             in magnitude with the sign {:+} of the bound on that side",
            side.name(),
            side.dual_sign()
        );
    }

    let tol = UnscaledTolerance::from_active_profile();
    let lambda = water_value(PRODUCTIVITY);
    let slack_price = range.penalty_per_hour * MONTH_HOURS / DEFAULT_COST_SCALE_FACTOR;
    let lag = lag_slot(&training_ctx.cut_state_layouts[RANGE_STAGE - 1], 1, 0, 0);
    let lag_tolerance = psi.abs() * coupled_dual_tolerance(&setup, RANGE_STAGE, 0, tol);
    for (side, lag_value, z_hat, cut) in &outcomes {
        let label = format!("{fixture}: {}", side.name());
        let generic_dual = side.dual_sign() * slack_price;
        // n = 20: λ (4), κ (2), κ·λ, p = penalty·hours/K (2), G·π_gen, −, ψ·(…) derive it;
        // d·row_scale and G·π on both coupled rows, their sum and ψ·g solve it; added: |ψ| times
        // both coupled rows' |G|·row-dual tolerance.
        ClosedForm {
            fixture: &label,
            quantity: "lag-0 coefficient",
            slot: Some(lag),
            want: psi * (kappa * lambda - RANGE_INFLOW_COEFFICIENT * generic_dual),
            n: 20,
            abs_sum: psi.abs() * ((kappa * lambda).abs() + slack_price),
            added: lag_tolerance,
        }
        .assert_solved(cut.coefficients[lag]);
        let turbined_mwh = PRODUCTIVITY * (INITIAL_STORAGE_HM3 + kappa * z_hat) / M3S_TO_HM3;
        let violation_cost = range.penalty_per_hour * MONTH_HOURS * side.violation(range, *z_hat);
        let z_abs = base.abs() + (psi * lag_value).abs() + innovation.abs();
        let turbined_abs = PRODUCTIVITY * (INITIAL_STORAGE_HM3 + kappa * z_abs) / M3S_TO_HM3;
        let violation_abs =
            range.penalty_per_hour * MONTH_HOURS * (side.bound(range).abs() + z_abs);
        // n = 40: ẑ (4), κ (2), the turbined energy (4), the thermal cost (3), the violation
        // cost (3), their sum and ÷ K derive it; the solver's sums of cost × primal over the
        // stage's columns solve it; no reported dual enters.
        ClosedForm {
            fixture: &label,
            quantity: "objective",
            slot: None,
            want: (THERMAL_COST_PER_MWH * (LOAD_MW * MONTH_HOURS - turbined_mwh) + violation_cost)
                / DEFAULT_COST_SCALE_FACTOR,
            n: 40,
            abs_sum: (THERMAL_COST_PER_MWH * (LOAD_MW * MONTH_HOURS + turbined_abs)
                + violation_abs)
                / DEFAULT_COST_SCALE_FACTOR,
            added: 0.0,
        }
        .assert_solved(cut.objective);
    }
}

/// derived: `ẑ` is pinned below the range and then above it, the slack covers the violation at
/// `p = penalty·hours/K`, so the row's dual is `+p` then `−p` and the lag coefficient is
/// `ψ·(κ·λ − G·π_gen)` with `G = 1`, while `Q` adds `penalty·hours·(L − ẑ)` then
/// `penalty·hours·(ẑ − U)` to the thermal cost. A row that misses the realized inflow loses the
/// dual, and a shift of the lower bounds alone breaks the upper-side solve.
#[test]
fn a_range_inflow_row_prices_the_lag_on_both_of_its_sides() {
    assert_range_row_lag_price(
        InflowRange {
            stage: RANGE_STAGE,
            lower_m3s: -10.0,
            upper_m3s: 30.0,
            penalty_per_hour: 30.0,
        },
        [(RangeSide::Lower, -30.0), (RangeSide::Upper, 60.0)],
    );
}

/// derived: the same two closed forms on a narrower range with a higher penalty, solved above the
/// range first, so a dual priced at another penalty misses both and the upper side is not
/// reached through the lower one.
#[test]
fn a_range_inflow_row_prices_the_lag_on_both_of_its_sides_on_a_narrower_range() {
    assert_range_row_lag_price(
        InflowRange {
            stage: RANGE_STAGE,
            lower_m3s: 0.0,
            upper_m3s: 20.0,
            penalty_per_hour: 80.0,
        },
        [(RangeSide::Upper, 50.0), (RangeSide::Lower, -25.0)],
    );
}

fn assert_storage_only_projection(keeps_lag_state: bool) {
    const STAGE: usize = 1;
    let case = StudyCase {
        name: if keeps_lag_state {
            "terminal PAR(1) keeping the lag state"
        } else {
            "terminal PAR(1) with a storage-only producer pool"
        },
        storage_only_stage: (!keeps_lag_state).then_some(STAGE),
        ..terminal_par1_case(TERMINAL_PSI, MONTH_HOURS)
    };
    let fixture = case.name;
    let n_hydros = case.hydros.len();
    let (setup, kappa) = compact_study(&case, STAGE);
    let x_hat = oracle_initial_state(&setup);
    let training_ctx = setup.training_ctx();
    let noise = opening_noise(&setup, STAGE);
    let cut = compact_cut(&setup, STAGE, &x_hat, &noise, &[], fixture);

    let extracted = &training_ctx.cut_state_layouts[STAGE - 1];
    let lag_depth = training_ctx.state.max_par_order;
    let want_slots = if keeps_lag_state {
        n_hydros * (1 + lag_depth)
    } else {
        n_hydros
    };
    assert_eq!(
        (extracted.n_slots(), cut.coefficients.len()),
        (want_slots, want_slots),
        "{fixture}: the producer pool projects {want_slots} slots and its cut has one \
         coefficient per slot"
    );
    if !keeps_lag_state {
        for slot in 0..extracted.n_slots() {
            let dim = extracted.global_state_index(CutSlot::new(slot)).get();
            assert!(
                dim < n_hydros,
                "{fixture}: slot {slot} projects state dimension {dim}, which is not a storage"
            );
        }
    }
    assert_has_power(fixture, "coupled dual", cut.coupled_duals[0]);

    let tol = UnscaledTolerance::from_active_profile();
    let lambda = water_value(PRODUCTIVITY);
    let storage = storage_slot(extracted, 0);
    // n = 5: c·ρ, M, M·K and ÷ derive λ, rc ÷ col_scale solves it; added: v_in's rc tolerance.
    ClosedForm {
        fixture,
        quantity: "storage coefficient",
        slot: Some(storage),
        want: lambda,
        n: 5,
        abs_sum: lambda.abs(),
        added: tol.reduced_cost(incoming_storage_col_scale(&setup, STAGE, 0)),
    }
    .assert_solved(cut.coefficients[storage]);
    if keeps_lag_state {
        let psi = psi_at(&setup, STAGE, 0)[0];
        let lag = lag_slot(extracted, n_hydros, 0, 0);
        assert_non_degenerate(fixture, "lag-1 coefficient ψ", psi);
        // n = 13: λ (4), κ (2), ψ·κ·λ (2); d·row_scale, G·π, ψ·g and G = −κ (2) solve it;
        // added: |ψ|·κ times the water balance's row-dual tolerance.
        ClosedForm {
            fixture,
            quantity: "lag-0 coefficient",
            slot: Some(lag),
            want: psi * kappa * lambda,
            n: 13,
            abs_sum: (psi * kappa * lambda).abs(),
            added: psi.abs() * kappa * tol.row_dual(coupled_row_scale(&setup, STAGE, 0)),
        }
        .assert_solved(cut.coefficients[lag]);
    }
}

/// derived: the solved stage keeps the lags out of its producer pool, so the cut holds the one
/// storage coefficient `λ = −c·ρ/(M·K)`; the coupled dual is non-zero, so a lag coefficient
/// written onto the pool's leading slots moves the storage coefficient off `λ`.
#[test]
fn a_storage_only_cut_carries_no_lag_coefficient() {
    assert_storage_only_projection(false);
}

/// derived: with the lag state kept the same solve yields the storage coefficient `λ` and the lag
/// coefficient `ψ·κ·λ`, so the coupled dual that the storage-only cut drops is a non-zero lag
/// price.
#[test]
fn a_cut_that_keeps_the_lag_state_carries_its_lag_coefficient() {
    assert_storage_only_projection(true);
}

/// The sign of every hydro's pinned `ẑ` in the truncated negative-draw study's stage 1.
#[derive(Clone, Copy)]
enum Draw {
    Negative,
    Positive,
}

impl Draw {
    fn fixture(self) -> &'static str {
        match self {
            Self::Negative => "truncated negative draw",
            Self::Positive => "truncated positive draw",
        }
    }

    fn targets(self) -> [f64; 2] {
        match self {
            Self::Negative => NEGATIVE_DRAW_TARGETS_M3S,
            Self::Positive => POSITIVE_DRAW_TARGETS_M3S,
        }
    }
}

fn assert_truncated_draw_lag_price(draw: Draw) {
    const STAGE: usize = 1;
    let fixture = draw.fixture();
    let spec = negative_draw_water_value_spec(CfgInflowMethod::Truncation);
    let (system, config) = chronological_noise_study(&spec);
    let models = chronological_noise_hydro_models(&spec, &system);
    let setup = build_setup_in_code_with_models(system, &config, models);
    let training_ctx = setup.training_ctx();
    let n_hydros = training_ctx.state.hydro_count;
    let noise = opening_noise(&setup, STAGE);
    let par = training_ctx.stochastic.par();
    let mut x_hat = oracle_initial_state(&setup);
    let mut z_hat = vec![0.0; n_hydros];
    for (h, (z, z_target)) in z_hat.iter_mut().zip(draw.targets()).enumerate() {
        let base = par.deterministic_base(STAGE, h);
        let innovation = par.sigma(STAGE, h) * noise[h];
        let psi = psi_at(&setup, STAGE, h)[0];
        let lag_value = (z_target - base - innovation) / psi;
        x_hat[lag_dim(&setup, h, 0)] = lag_value;
        *z = base + psi * lag_value + innovation;
    }
    let mut workspaces = setup
        .create_workspace_pool(&StubComm, 1, ActiveSolver::new)
        .expect("the workspace pool must build");
    let (slack_lower, _) = inflow_slack_bounds_for_probe(
        &mut workspaces.workspaces[0],
        &training_ctx,
        StageIdx(STAGE),
        &x_hat,
        &noise,
    );
    let cut = compact_cut(&setup, STAGE, &x_hat, &noise, &[], fixture);

    let template = &setup.inputs.stage_data.stage_templates.templates[STAGE];
    let slack_columns = setup.stage_ctx().geometry_per_stage[STAGE]
        .inflow_slack
        .clone();
    for h in 0..n_hydros {
        let slack_col = slack_columns.start + h;
        let slack_rc = cut.reduced_costs[slack_col] / template.col_scale[slack_col];
        match draw {
            Draw::Negative => {
                assert!(
                    z_hat[h] < 0.0,
                    "{fixture}: hydro {h}'s ẑ = {} must be negative",
                    z_hat[h]
                );
                assert!(
                    slack_lower[h] > 0.0,
                    "{fixture}: hydro {h}'s inflow-slack lower bound {} must be positive",
                    slack_lower[h]
                );
                assert_has_power(fixture, &format!("hydro {h}'s inflow-slack rc"), slack_rc);
            }
            Draw::Positive => {
                assert!(
                    z_hat[h] > 0.0,
                    "{fixture}: hydro {h}'s ẑ = {} must be positive",
                    z_hat[h]
                );
                assert!(
                    slack_lower[h] == 0.0,
                    "{fixture}: hydro {h}'s inflow-slack lower bound {} must be zero",
                    slack_lower[h]
                );
            }
        }
        assert_non_degenerate(
            fixture,
            &format!("hydro {h}'s lag-1 coefficient ψ"),
            psi_at(&setup, STAGE, h)[0],
        );
    }

    let tol = UnscaledTolerance::from_active_profile();
    let lambda = -(NEGATIVE_DRAW_THERMAL_COST_PER_MWH * spec.hydro_productivity_mw_per_m3s
        - CHRONOLOGICAL_NOISE_RELEASE_COST)
        / (M3S_TO_HM3 * DEFAULT_COST_SCALE_FACTOR);
    let kappa: f64 = CHRONOLOGICAL_NOISE_BLOCK_HOURS
        .iter()
        .map(|hours| hours * M3S_TO_HM3)
        .sum();
    assert_non_degenerate(fixture, "inflow coupling κ", kappa);
    let extracted = &training_ctx.cut_state_layouts[STAGE - 1];
    for h in 0..n_hydros {
        let psi = psi_at(&setup, STAGE, h)[0];
        let lag = lag_slot(extracted, n_hydros, h, 0);
        // n = 17: λ (4), κ over the two blocks (3), ψ·κ·λ (2) derive it; d·row_scale and G·π on
        // both block rows, their sum, ψ·g and G = −τ_k (2) solve it; added: |ψ| times both block
        // rows' |G|·row-dual tolerance.
        ClosedForm {
            fixture,
            quantity: &format!("hydro {h}'s lag-0 coefficient"),
            slot: Some(lag),
            want: psi * kappa * lambda,
            n: 17,
            abs_sum: (psi * kappa * lambda).abs(),
            added: psi.abs() * coupled_dual_tolerance(&setup, STAGE, h, tol),
        }
        .assert_solved(cut.coefficients[lag]);
    }
}

/// derived: every hydro turbines all it holds into the marginal thermal, so each block's balance
/// prices water at `λ = −(c·ρ − c_q)/(M·K)` and ẑ enters the two block rows at `κ = 744·M` in all:
/// the lag coefficient is `ψ_h·κ·λ`. The pinned `ẑ_h` is negative, so the slack is fixed at
/// `l* = −ẑ_h > 0` with a non-zero reduced cost, which a chain rule that differentiates `l*` adds.
#[test]
fn a_truncated_negative_draw_prices_the_lag_without_the_truncation_bound() {
    assert_truncated_draw_lag_price(Draw::Negative);
}

/// derived: the same `ψ_h·κ·λ` at positive draws, where `l* = 0` and the slack's bound carries no
/// term to differentiate.
#[test]
fn a_truncated_positive_draw_prices_the_lag_at_the_same_water_value() {
    assert_truncated_draw_lag_price(Draw::Positive);
}

fn weekly_par_deck() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(WEEKLY_PAR_DECK)
}

/// `β_inj(h, l)`: distinct and non-zero over the resident pool's lag slots.
fn weekly_injected_slope(n_hydros: usize, h: usize, l: usize) -> f64 {
    let position = u32::try_from(l * n_hydros + h + 1).expect("a lag position fits u32");
    INJECTED_LAG_SLOPES[0] / f64::from(position)
}

fn assert_weekly_lag_alias(stage: usize, finalizes: bool) {
    let fixture = if finalizes {
        "weekly PAR at the week that closes its month"
    } else {
        "weekly PAR within its month"
    };
    let (_, mut setup) = fresh_system_and_setup_with(&weekly_par_deck(), |_| {});
    assert_eq!(
        setup.stage_ctx().stage_lag(StageIdx(stage)).finalize_period,
        finalizes,
        "{fixture}: stage {stage}'s transition must finalize its period exactly when it closes \
         its month"
    );
    let (n_hydros, lag_depth) = {
        let state_space = setup.training_ctx().state;
        (state_space.hydro_count, state_space.max_par_order)
    };
    let slopes: Vec<(usize, f64)> = {
        let resident = &setup.training_ctx().cut_state_layouts[stage];
        assert_eq!(
            resident.lag_point().full().len(),
            n_hydros * lag_depth,
            "{fixture}: stage {stage}'s resident pool must project every lag slot"
        );
        (0..lag_depth)
            .flat_map(|l| (0..n_hydros).map(move |h| (h, l)))
            .map(|(h, l)| {
                (
                    lag_slot(resident, n_hydros, h, l),
                    weekly_injected_slope(n_hydros, h, l),
                )
            })
            .collect()
    };
    let pool_slots = inject_cut(&mut setup, stage, &slopes);
    let training_ctx = setup.training_ctx();
    let x_hat = training_ctx.initial_state.to_vec();
    let noise = opening_noise(&setup, stage);
    let cut = compact_cut(&setup, stage, &x_hat, &noise, &pool_slots, fixture);

    let [(slot, y)] = cut.cut_dual_slots[..] else {
        panic!(
            "{fixture}: exactly the injected cut must bind, got {:?}",
            cut.cut_dual_slots
        );
    };
    assert!(
        slot == pool_slots[0],
        "{fixture}: the injected cut at slot {} must bind, got slot {slot}",
        pool_slots[0]
    );
    assert_has_power(fixture, "injected cut's dual y", y);
    let extracted = &training_ctx.cut_state_layouts[stage - 1];
    for h in 0..n_hydros {
        let coupled = cut.coupled_duals[h];
        let psi = psi_at(&setup, stage, h);
        let g = -coupled + y * weekly_injected_slope(n_hydros, h, 0);
        let g_abs = coupled.abs() + (y * weekly_injected_slope(n_hydros, h, 0)).abs();
        assert_non_degenerate(fixture, &format!("hydro {h}'s lag-1 coefficient ψ"), psi[0]);
        assert_has_power(fixture, &format!("hydro {h}'s lag-0 sensitivity g"), g);
        for l in 0..lag_depth {
            let psi_l = psi.get(l).copied().unwrap_or(0.0);
            let next = if l + 1 < lag_depth {
                y * weekly_injected_slope(n_hydros, h, l + 1)
            } else {
                0.0
            };
            let lag = lag_slot(extracted, n_hydros, h, l);
            // n = 5: y·β_inj(h, 0), −c_h + y·β_inj(h, 0), ψ·g, y·β_inj(h, l + 1) and the last +
            // on either side; no tolerance is added, since c_h and y are the solve's own.
            ClosedForm {
                fixture,
                quantity: &format!("hydro {h}'s lag-{l} coefficient"),
                slot: Some(lag),
                want: psi_l * g + next,
                n: 5,
                abs_sum: psi_l.abs() * g_abs + next.abs(),
                added: 0.0,
            }
            .assert_solved(cut.coefficients[lag]);
        }
    }
}

/// derived: a week inside its month evaluates the lag at the same point as a closing week, so
/// with the injected cut binding at y the lag coefficient is `ψ_{h,l}·(−c_h + y·β_inj(h, 0)) +
/// y·β_inj(h, l + 1)`, with `β_inj(h, L) = 0`, from the solve's own `c_h` and y. A chain rule that
/// reads the resident `(h, l)` slot within the month adds `y·β_inj(h, l)` to it.
#[test]
fn a_non_finalizing_stage_keeps_the_lag_alias() {
    assert_weekly_lag_alias(WEEKLY_PAR_NON_FINALIZING_STAGE, false);
}

/// derived: the same alias formula at the week that closes the month.
#[test]
fn a_finalizing_stage_keeps_the_lag_alias() {
    assert_weekly_lag_alias(WEEKLY_PAR_FINALIZING_STAGE, true);
}
