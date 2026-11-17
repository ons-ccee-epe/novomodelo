//! Closed-form cut coefficients of the compact LP on fixtures small enough to derive by hand.
//!
//! No fixture is trained: its resident pool is empty or holds one cut the test injects, so each
//! expected coefficient follows from the fixture's constants and the solve's own duals. On every
//! in-code fixture the hydros empty their reservoirs into a marginal thermal, so a stored hm³ is
//! worth `λ(ρ) = −(c·ρ − c_q)/(M·K)` in cost-scaled units, with `c_q` the turbined cost, and the
//! realized inflow enters its water balance at `κ = hours·M` hm³ per m³/s.

use std::collections::{BTreeSet, HashMap};
use std::path::{Path, PathBuf};
use std::sync::mpsc;

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
    SlackConfig, System, SystemBuilder, TailraceFlow, TailraceFlowInfluence, ThermalBlockBounds,
    TrainingEvent, VariableRef,
};
use novomodelo_io::Config;
use novomodelo_io::config::{
    BackwardScheduler, EstimationConfig, ExportsConfig, InflowNonNegativityConfig,
    InflowNonNegativityMethod as CfgInflowMethod, ModelingConfig, ParallelismConfig, PolicyConfig,
    RowSelectionConfig, SimulationConfig as IoSimulationConfig, StoppingMode, StoppingRuleConfig,
    TrainingConfig, TrainingSelection, TrainingSolverConfig, UpperBoundEvaluationConfig,
};
use novomodelo_sddp::hydro_models::{
    EvaporationCoefficientSign, EvaporationModel, EvaporationModelSet, FphaPlane,
    LinearizedEvaporation, PrepareHydroModelsResult, ProductionModelSet, ResolvedProductionModel,
};
use novomodelo_sddp::indexer::{
    BlockIdx, BusSys, CutSlot, CutStateProjection, FphaCellLocal, HydroSys, LagPointMode, StateDim,
};
use novomodelo_sddp::setup::{NodePos, StageIdx};
use novomodelo_sddp::test_support::bound_recording::{
    AppendedCutRowBound, CutRowPushLog, CutRowPushRecord, CutRowPushRecorder,
};
use novomodelo_sddp::test_support::decks::committed_decks;
use novomodelo_sddp::test_support::template_structure::state_lag_dims;
use novomodelo_sddp::test_support::{
    ProbeCut, ProbeCuts, ProbeOpening, empty_row_batch, inflow_slack_bounds_for_probe,
    opening_cut_for_probe, oracle_initial_state,
};
use novomodelo_sddp::{
    DEFAULT_COST_SCALE_FACTOR, SolverProfiles, StudySetup, build_cut_row_batch_into,
};
use novomodelo_solver::{ActiveSolver, SolverInterface};

use crate::common::builders::{
    BusSpec, HydroSpec, StageSpec, ThermalSpec, make_bus, make_hydro, make_stage, make_thermal,
};
use crate::common::expanded_solve::{OracleLagPoint, hydro_noise_slot, realized_inflow};
use crate::common::in_code_studies::{
    CHRONOLOGICAL_NOISE_BLOCK_HOURS, CHRONOLOGICAL_NOISE_RELEASE_COST, SAME_PATH_LAG_SEED_M3S,
    WEEKS_THEN_MONTH_TERMINAL_OPENINGS, chronological_noise_hydro_models,
    chronological_noise_study, negative_draw_water_value_spec, same_path_partial_sum_study,
    weeks_then_month_study,
};
use crate::common::oracle_tolerance::{
    UnscaledTolerance, assert_first_try, rhs_tolerance, rounding_factor, value_tolerance,
};
use crate::common::stored_cut_oracles::{
    PartialSums, apply_oracle_config, pushed_cut_rhs_at_visit, replay_partial_sums,
};
use crate::common::{
    StubComm, build_setup_in_code_with_models, fresh_system_and_setup_with,
    inject_synthetic_boundary_cut, lift_to_global_state,
};

const M3S_TO_HM3: f64 = 3_600.0 / 1_000_000.0;
const BUS_ID: EntityId = EntityId(10);
const THERMAL_ID: EntityId = EntityId(20);
const SOLE_HYDRO_ID: EntityId = EntityId(1);
const UPSTREAM_ID: EntityId = EntityId(1);
const TARGET_ID: EntityId = EntityId(2);
const INFLUENCING_ID: EntityId = EntityId(2);
/// The system index of [`INFLUENCING_ID`] in [`lateral_fpha_case`].
const INFLUENCING: usize = 1;

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
    coefficient_sign: EvaporationCoefficientSign::Positive,
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

const FPHA_STAGE: usize = 1;
const FPHA_PLANE: FphaPlane = FphaPlane {
    intercept: 0.0,
    gamma_v: 0.05,
    gamma_q: 0.5,
    gamma_s: 0.0,
};
/// `σ_c` of the fixture's one cell, which holds the whole plant.
const FPHA_CELL_SHARE: f64 = 1.0;
/// Far above the thermal cost, so the thermal, not σ, covers the load the plane leaves.
const IDLE_FPHA_VIOLATION_COST: f64 = 10.0 * DEFICIT_COST_PER_MWH;
/// Below the thermal cost, so σ, not the thermal, covers the load the plane leaves.
const ACTIVE_FPHA_VIOLATION_COST: f64 = 40.0;
/// Above the thermal and deficit costs and below the floor's penalty.
const FLOOR_FPHA_VIOLATION_COST: f64 = 800.0;
const GENERATION_FLOOR_CONSTRAINT_ID: i32 = 31;
/// The floor row's coefficient on the FPHA generation `g`.
const GENERATION_FLOOR_COEFFICIENT: f64 = 1.0;
/// Above the plane's output at the initial state, so σ or the floor's slack covers the rest.
const GENERATION_FLOOR: GenerationFloor = GenerationFloor {
    lower_mw: 250.0,
    penalty_per_hour: 2_000.0,
};
/// How many `ClosedForm` bounds the head term must exceed, so a closed form without it fails.
const HEAD_TERM_MARGIN: f64 = 100.0;
/// [`FPHA_PLANE`] with a tailrace slope, so a lateral inflow lowers the plane.
const LATERAL_FPHA_PLANE: FphaPlane = FphaPlane {
    gamma_s: -0.1,
    ..FPHA_PLANE
};
/// `f_k`: the weight of the influencing hydro's incremental inflow in the tailrace flow.
const LATERAL_FACTOR: f64 = 0.8;

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

const CLOSING_MONTH_FIXTURE: &str = "tests/fixtures/weekly_then_closing_month";
/// The fixture's last week of January, which closes its month.
const CLOSING_MONTH_FIXTURE_CLOSING_STAGE: usize = 3;
const MULTI_RESOLUTION_DECK: &str = "examples/deterministic/d30-multi-resolution-monthly-quarterly";
/// The deck's last month, whose lags rebuild from the downstream ring.
const MULTI_RESOLUTION_REBUILD_STAGE: usize = 6;

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

/// A soft generic row `G·g ≥ lower` on the sole hydro's FPHA generation at every stage, with its
/// slack priced per hour.
#[derive(Clone, Copy)]
struct GenerationFloor {
    lower_mw: f64,
    penalty_per_hour: f64,
}

/// The case's first hydro produces through `plane` at every stage, its slack σ priced at
/// `violation_cost`; `lateral_inflow` adds another hydro's incremental inflow, at a factor, to its
/// tailrace flow.
#[derive(Clone, Copy)]
struct FphaCase {
    plane: FphaPlane,
    violation_cost: f64,
    generation_floor: Option<GenerationFloor>,
    lateral_inflow: Option<(EntityId, f64)>,
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
    fpha: Option<FphaCase>,
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
        fpha_violation_cost: 0.0,
        evaporation_violation_cost: EVAPORATION_VIOLATION_COST,
        water_withdrawal_violation_cost: 0.0,
        water_withdrawal_violation_pos_cost: 0.0,
        water_withdrawal_violation_neg_cost: 0.0,
        evaporation_violation_pos_cost: EVAPORATION_VIOLATION_COST,
        evaporation_violation_neg_cost: EVAPORATION_VIOLATION_COST,
        inflow_nonnegativity_cost: 0.0,
    }
}

fn case_penalties(case: &StudyCase) -> HydroPenalties {
    match case.fpha {
        Some(fpha) => HydroPenalties {
            fpha_violation_cost: fpha.violation_cost,
            ..fixture_penalties()
        },
        None => fixture_penalties(),
    }
}

/// The tailrace flow of the hydro at `position` in `case`: the FPHA hydro's carries the case's
/// lateral inflow.
fn case_tailrace_flow(case: &StudyCase, position: usize) -> TailraceFlow {
    match case.fpha.and_then(|fpha| fpha.lateral_inflow) {
        Some((hydro_id, factor)) if position == 0 => TailraceFlow {
            incremental_inflow_influences: vec![TailraceFlowInfluence { hydro_id, factor }],
            ..TailraceFlow::default()
        },
        _ => TailraceFlow::default(),
    }
}

fn fixture_hydro(
    hydro: &HydroCase,
    penalties: HydroPenalties,
    tailrace_flow: TailraceFlow,
) -> Hydro {
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
            penalties,
            tailrace_flow,
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
            thermal_block: ThermalBlockBounds {
                min_generation_mw: 0.0,
                max_generation_mw: THERMAL_MAX_MW,
                cost_per_mwh: THERMAL_COST_PER_MWH,
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
            hydro: case_penalties(case),
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
        .hydros(
            case.hydros
                .iter()
                .enumerate()
                .map(|(position, hydro)| {
                    fixture_hydro(
                        hydro,
                        case_penalties(case),
                        case_tailrace_flow(case, position),
                    )
                })
                .collect(),
        )
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
    let (constraints, bounds) = fixture_generic_constraints(case);
    let builder = if constraints.is_empty() {
        builder
    } else {
        builder
            .generic_constraints(constraints)
            .resolved_generic_bounds(bounds)
    };
    builder
        .build()
        .unwrap_or_else(|e| panic!("{}: the system must build: {e:?}", case.name))
}

/// `(constraint id, stage id, block id, lower, upper)`, one resolved generic bound row.
type GenericBoundRow = (i32, i32, Option<i32>, Option<f64>, Option<f64>);

fn fixture_generic_constraints(
    case: &StudyCase,
) -> (Vec<GenericConstraint>, ResolvedGenericConstraintBounds) {
    let generic = case
        .inflow_range
        .map(inflow_range_constraint)
        .into_iter()
        .chain(
            case.fpha
                .and_then(|fpha| fpha.generation_floor)
                .map(|floor| generation_floor_constraint(floor, case.n_stages)),
        );
    let (constraints, rows): (Vec<GenericConstraint>, Vec<Vec<GenericBoundRow>>) = generic.unzip();
    let id_map: HashMap<i32, usize> = constraints
        .iter()
        .enumerate()
        .map(|(position, constraint)| (constraint.id.0, position))
        .collect();
    let bounds = ResolvedGenericConstraintBounds::new(&id_map, rows.into_iter().flatten());
    (constraints, bounds)
}

fn generation_floor_constraint(
    floor: GenerationFloor,
    n_stages: usize,
) -> (GenericConstraint, Vec<GenericBoundRow>) {
    let constraint = GenericConstraint {
        id: EntityId(GENERATION_FLOOR_CONSTRAINT_ID),
        name: "generation_floor".to_string(),
        description: None,
        expression: ConstraintExpression {
            terms: vec![LinearTerm::literal(
                GENERATION_FLOOR_COEFFICIENT,
                VariableRef::HydroGeneration {
                    hydro_id: SOLE_HYDRO_ID,
                    block_id: None,
                    bus_id: None,
                },
            )],
        },
        slack: Some(SlackConfig {
            penalty: floor.penalty_per_hour,
            penalty_basis: PenaltyBasis::PerHour,
        }),
        bound_lower: None,
        bound_upper: None,
    };
    let rows = (0..n_stages)
        .map(|stage| {
            (
                GENERATION_FLOOR_CONSTRAINT_ID,
                stage_id(stage),
                None,
                Some(floor.lower_mw),
                None,
            )
        })
        .collect();
    (constraint, rows)
}

fn inflow_range_constraint(range: InflowRange) -> (GenericConstraint, Vec<GenericBoundRow>) {
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
    let rows = vec![(
        RANGE_CONSTRAINT_ID,
        stage_id(range.stage),
        None,
        Some(range.lower_m3s),
        Some(range.upper_m3s),
    )];
    (constraint, rows)
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
            .enumerate()
            .map(|(position, hydro)| {
                let model = match case.fpha {
                    Some(fpha) if position == 0 => ResolvedProductionModel::Fpha {
                        planes: vec![fpha.plane],
                    },
                    _ => ResolvedProductionModel::ConstantProductivity {
                        productivity: hydro.productivity,
                    },
                };
                vec![model; case.n_stages]
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

/// A derived value, compared within `2γ_n·S + added`; `slot` is `None` for a quantity at no cut
/// slot.
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
    fn bound(&self) -> f64 {
        2.0 * rounding_factor(self.n) * self.abs_sum + self.added
    }

    fn assert_solved(&self, got: f64) {
        let bound = self.bound();
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
        fpha: None,
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
        fpha: None,
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
        fpha: None,
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

/// The template row whose unscaled bounds are the fixture's `[lower, upper]`; an infinite side
/// matches only itself.
fn row_with_bounds(setup: &StudySetup, stage: usize, lower: f64, upper: f64) -> usize {
    let template = &setup.inputs.stage_data.stage_templates.templates[stage];
    let matches = |scaled: f64, row_scale: f64, want: f64| {
        if want.is_infinite() {
            scaled == want
        } else {
            (scaled / row_scale - want).abs() <= BOUND_MATCH_TOLERANCE * want.abs().max(1.0)
        }
    };
    let rows: Vec<usize> = (0..template.num_rows)
        .filter(|&row| {
            let row_scale = template.row_scale[row];
            matches(template.row_lower[row], row_scale, lower)
                && matches(template.row_upper[row], row_scale, upper)
        })
        .collect();
    let [row] = rows[..] else {
        panic!(
            "stage {stage}: exactly one row must carry the bounds [{lower}, {upper}], got {rows:?}"
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
    let row = row_with_bounds(&setup, RANGE_STAGE, range.lower_m3s, range.upper_m3s);
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
            resident.lag_point().mode(),
            if finalizes {
                LagPointMode::Closing
            } else {
                LagPointMode::Carry
            },
            "{fixture}: stage {stage}'s pool must take the point its period position gives"
        );
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
    let closing_share = {
        let stage_ctx = setup.stage_ctx();
        let weight = |t: usize| stage_ctx.stage_lag(StageIdx(t)).accumulate_weight;
        let incoming = (0..stage).fold(0.0, |sum, t| sum + weight(t));
        weight(stage) / (incoming + weight(stage))
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
    let cut_term_scale = if finalizes { closing_share } else { 0.0 };
    for h in 0..n_hydros {
        let coupled = cut.coupled_duals[h];
        let psi = psi_at(&setup, stage, h);
        let injected = |l: usize| {
            if l < lag_depth {
                y * weekly_injected_slope(n_hydros, h, l)
            } else {
                0.0
            }
        };
        let g = -coupled + cut_term_scale * injected(0);
        let g_alias = -coupled + injected(0);
        let g_abs = coupled.abs() + (cut_term_scale * injected(0)).abs();
        assert_non_degenerate(fixture, &format!("hydro {h}'s lag-1 coefficient ψ"), psi[0]);
        if finalizes {
            assert_has_power(fixture, &format!("hydro {h}'s lag-0 sensitivity g"), g);
        } else {
            assert_has_power(
                fixture,
                &format!("hydro {h}'s injected lag-0 term"),
                injected(0),
            );
        }
        let mut separation = f64::NEG_INFINITY;
        for l in 0..lag_depth {
            let psi_l = psi.get(l).copied().unwrap_or(0.0);
            let carried = if finalizes {
                injected(l + 1)
            } else {
                injected(l)
            };
            let want = psi_l * g + carried;
            let alias = psi_l * g_alias + injected(l + 1);
            let abs_sum = psi_l.abs() * g_abs + carried.abs();
            let lag = lag_slot(extracted, n_hydros, h, l);
            // n = 5: the cut term of g, g itself, ψ·g, the carried term and the last + on either
            // side; no tolerance is added, since c_h and y are the solve's own.
            ClosedForm {
                fixture,
                quantity: &format!("hydro {h}'s lag-{l} coefficient"),
                slot: Some(lag),
                want,
                n: 5,
                abs_sum,
                added: 0.0,
            }
            .assert_solved(cut.coefficients[lag]);
            separation = separation.max((alias - want).abs() - 2.0 * rounding_factor(5) * abs_sum);
        }
        assert!(
            separation > 0.0,
            "{fixture}: hydro {h}'s lag coefficients under the shifted alias are within the \
             tolerance of the derived ones, so the closed form could hold vacuously"
        );
    }
}

/// derived: week 1 of the deck leaves January open, so its pool evaluates the cut at the lags
/// it is handed, `Φ = â`, and the cut leaves `g_h = −(Gᵀπ)_h`. With the injected cut binding at y
/// the lag coefficient is `−ψ_{h,l}·(Gᵀπ)_h + y·β_inj(h, l)`, with `β_inj(h, l)` over every slot.
/// The shifted alias reads `β_inj(h, l + 1)` and adds `ψ_{h,l}·y·β_inj(h, 0)` to g.
#[test]
fn a_non_finalizing_stage_evaluates_cuts_at_the_carried_lags() {
    assert_weekly_lag_alias(WEEKLY_PAR_NON_FINALIZING_STAGE, false);
}

/// derived: week 3 closes January over the weight W of the weeks before it and its own w, so
/// `Φ(h, 0) = (s + w·ẑ_h)/(W + w)` and `Φ(h, l) = â(h, l − 1)`. The cut term of `g_h` moves with
/// ẑ at `c = w/(W + w)`, so the lag coefficient is `ψ_{h,l}·(−(Gᵀπ)_h + c·y·β_inj(h, 0)) +
/// y·β_inj(h, l + 1)`, with `β_inj(h, L) = 0`. The shifted alias takes `c = 1`.
#[test]
fn a_finalizing_stage_evaluates_cuts_at_the_closing_period_mean() {
    assert_weekly_lag_alias(WEEKLY_PAR_FINALIZING_STAGE, true);
}

fn leaves_the_shifted_point(dir: &Path) -> bool {
    let (_, setup) = fresh_system_and_setup_with(dir, |_| {});
    setup
        .training_ctx()
        .cut_state_layouts
        .iter()
        .filter(|layout| layout.has_lag_slots())
        .any(|layout| layout.lag_point().mode() != LagPointMode::Shifted)
}

/// derived: weeks 0 to 2 of the fixture leave the January period open, so their pools take the
/// carried lag point, and week 3 closes it over the weight of the three weeks before it plus its
/// own: `inv = 1/(W + w)` and `c = w·inv`.
#[test]
fn lag_point_modes_of_the_weekly_closing_month_fixture_follow_its_calendar() {
    let (_, setup) = fresh_system_and_setup_with(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join(CLOSING_MONTH_FIXTURE),
        |_| {},
    );
    let stage_ctx = setup.stage_ctx();
    let weight = |t: usize| stage_ctx.stage_lag(StageIdx(t)).accumulate_weight;
    let layouts = setup.training_ctx().cut_state_layouts;

    for (t, layout) in layouts
        .iter()
        .enumerate()
        .take(CLOSING_MONTH_FIXTURE_CLOSING_STAGE)
        .skip(1)
    {
        assert_eq!(layout.lag_point().mode(), LagPointMode::Carry, "stage {t}");
    }
    let closing = layouts[CLOSING_MONTH_FIXTURE_CLOSING_STAGE].lag_point();
    assert_eq!(closing.mode(), LagPointMode::Closing);
    let incoming = (0..CLOSING_MONTH_FIXTURE_CLOSING_STAGE).fold(0.0, |sum, t| sum + weight(t));
    let own = weight(CLOSING_MONTH_FIXTURE_CLOSING_STAGE);
    let inverse = 1.0 / (incoming + own);
    assert_eq!(closing.incoming_weight()[0], incoming);
    assert_eq!(closing.closing_inverse_weight()[0], inverse);
    assert_eq!(closing.closing_share()[0], own * inverse);
}

/// derived: the month-to-quarter stage rebuilds its lags from the downstream ring, which
/// the shifted point already evaluates, so its pool is shifted. The primary weight survives the
/// rebuild, so the quarter after it closes a period that already holds the last month's weight
/// and each hydro's share of it is 1/2.
#[test]
fn lag_point_mode_of_the_quarter_after_a_rebuild_is_closing_at_one_half() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(MULTI_RESOLUTION_DECK);
    let (_, setup) = fresh_system_and_setup_with(&dir, |_| {});
    let layouts = setup.training_ctx().cut_state_layouts;

    assert_eq!(
        layouts[MULTI_RESOLUTION_REBUILD_STAGE].lag_point().mode(),
        LagPointMode::Shifted
    );
    let quarter = layouts[MULTI_RESOLUTION_REBUILD_STAGE + 1].lag_point();
    assert_eq!(quarter.mode(), LagPointMode::Closing);
    assert!(!quarter.closing_share().is_empty());
    for (h, &share) in quarter.closing_share().iter().enumerate() {
        assert_eq!(share, 0.5, "hydro {h}");
    }
}

/// A study whose stages all close their own lag period evaluates every cut at the shifted
/// point, so only the studies that mix lag periods leave it.
#[test]
fn lag_point_modes_of_uniform_committed_decks_are_shifted() {
    let mixed: BTreeSet<String> = committed_decks()
        .into_iter()
        .filter(|deck| leaves_the_shifted_point(&deck.dir))
        .map(|deck| deck.key)
        .collect();
    let expected: BTreeSet<String> = [
        WEEKLY_PAR_DECK,
        MULTI_RESOLUTION_DECK,
        "crates/novomodelo-sddp/tests/fixtures/weekly_then_closing_month",
    ]
    .map(String::from)
    .into();
    assert_eq!(mixed, expected);
}

struct TrainedRun {
    setup: StudySetup,
    partial_sums: PartialSums,
    bounds: Vec<(u64, f64, f64)>,
}

/// Trains `setup` with every pool storing its trial states, after replaying the partial sums its
/// forward visits read, and collects each iteration's lower and upper bound.
fn train_exporting_states(mut setup: StudySetup) -> TrainedRun {
    setup.set_export_states(true);
    let partial_sums = replay_partial_sums(&setup);
    let mut solver = ActiveSolver::new().expect("the training solver must build");
    let (sender, events) = mpsc::channel();
    let outcome = setup
        .train(
            &mut solver,
            &StubComm,
            1,
            ActiveSolver::new,
            Some(sender),
            None,
        )
        .expect("training must return Ok");
    assert!(
        outcome.error.is_none(),
        "training must not error: {:?}",
        outcome.error
    );
    let bounds = events
        .try_iter()
        .filter_map(|event| match event {
            TrainingEvent::IterationSummary {
                iteration,
                lower_bound,
                upper_bound,
                ..
            } => Some((iteration, lower_bound, upper_bound)),
            _ => None,
        })
        .collect();
    TrainedRun {
        setup,
        partial_sums,
        bounds,
    }
}

fn train_same_path_deck() -> TrainedRun {
    let (system, mut config, hydro_models) = same_path_partial_sum_study();
    apply_oracle_config(&mut config);
    train_exporting_states(build_setup_in_code_with_models(
        system,
        &config,
        hydro_models,
    ))
}

/// What one forward visit of a stage saw: the state it was pinned at, the state it left and the
/// lag accumulator it read.
struct Visit {
    incoming: Vec<f64>,
    outgoing: Vec<f64>,
    partial_sum: Vec<f64>,
}

fn stored_state(setup: &StudySetup, pool_id: usize, slot: usize) -> Vec<f64> {
    let n_state = setup.training_ctx().state.n_state;
    let layout = &setup.training_ctx().cut_state_layouts[pool_id];
    assert_eq!(
        layout.n_slots(),
        n_state,
        "pool {pool_id} must project every state dimension for its row to be a whole state"
    );
    let row = setup.fcf.pools[pool_id]
        .trial_state(slot)
        .unwrap_or_else(|| panic!("pool {pool_id} must store the trial state of slot {slot}"));
    lift_to_global_state(layout, row, &vec![0.0; n_state])
}

/// The forward visit that generated the cut of `pool_id` at `slot`: the visit of stage `pool_id`
/// whose outgoing state the cut was computed at, in the iteration and trial that wrote the cut.
fn generating_visit(run: &TrainedRun, pool_id: usize, slot: usize) -> Visit {
    let setup = &run.setup;
    let origin = setup.fcf.pools[pool_id].metadata(slot);
    let incoming = match pool_id.checked_sub(1) {
        None => setup.training_ctx().initial_state.to_vec(),
        Some(previous) => {
            let pool = &setup.fcf.pools[previous];
            let sibling = (0..pool.populated())
                .find(|&candidate| {
                    let other = pool.metadata(candidate);
                    other.iteration_generated == origin.iteration_generated
                        && other.forward_pass_index == origin.forward_pass_index
                })
                .unwrap_or_else(|| panic!("pool {previous} must hold the visit of slot {slot}"));
            stored_state(setup, previous, sibling)
        }
    };
    Visit {
        incoming,
        outgoing: stored_state(setup, pool_id, slot),
        partial_sum: run
            .partial_sums
            .of_cut(&setup.fcf.pools[pool_id], slot, pool_id),
    }
}

/// `p(â)` of hydro 0 at `stage`: the PAR conditional mean at the lags of `incoming`.
fn conditional_mean(setup: &StudySetup, stage: usize, incoming: &[f64]) -> f64 {
    let par = setup.training_ctx().stochastic.par();
    let lags: f64 = par
        .psi_slice(stage, 0)
        .iter()
        .enumerate()
        .map(|(l, psi)| psi * incoming[lag_dim(setup, 0, l)])
        .sum();
    par.deterministic_base(stage, 0) + lags
}

/// `Σ_l β(0, l)·x̂(0, l)` over the lag slots of the cut of `pool_id` at `slot`, with `x̂` the
/// visit's outgoing lags, and the same sum over the lags `lags` replaces them with.
fn lag_terms(setup: &StudySetup, pool_id: usize, slot: usize, lags: impl Fn(usize) -> f64) -> f64 {
    let pool = &setup.fcf.pools[pool_id];
    let layout = &setup.training_ctx().cut_state_layouts[pool_id];
    let n_lags = setup.training_ctx().state.max_par_order;
    (0..n_lags).fold(0.0, |sum, l| {
        sum + pool.coefficient_row(slot)[lag_slot(layout, 1, 0, l)] * lags(l)
    })
}

/// Checks that the RHS the production prep pushes for each cut of `pool_id` at the visit that
/// generated it is `α + Σ β·x̂` over the lags the visit left, with `z_hat` the visit's realized
/// inflow; returns the smallest distance between that value and the one at `ẑ` in lag 0.
fn assert_cut_rhs_at_its_own_trial_point(
    run: &TrainedRun,
    fixture: &str,
    pool_id: usize,
    z_hat: impl Fn(&Visit, usize) -> f64,
    noise: impl Fn(&Visit, usize) -> Vec<f64>,
) -> f64 {
    let setup = &run.setup;
    let pool = &setup.fcf.pools[pool_id];
    let mut separation = f64::INFINITY;
    for slot in 0..pool.populated() {
        let visit = generating_visit(run, pool_id, slot);
        let pushed = pushed_cut_rhs_at_visit(
            setup,
            pool_id,
            slot,
            &visit.incoming,
            &visit.partial_sum,
            &noise(&visit, slot),
        );
        let alpha = pool.intercept(slot);
        let outgoing = |l: usize| visit.outgoing[lag_dim(setup, 0, l)];
        let want = alpha + lag_terms(setup, pool_id, slot, outgoing);
        let z = z_hat(&visit, slot);
        let aliased = alpha
            + lag_terms(
                setup,
                pool_id,
                slot,
                |l| {
                    if l == 0 { z } else { outgoing(l) }
                },
            );
        let beta_0 = pool.coefficient_row(slot)
            [lag_slot(&setup.training_ctx().cut_state_layouts[pool_id], 1, 0, 0)];
        let tolerance = rhs_tolerance(alpha.abs() + beta_0.abs() * (outgoing(0).abs() + z.abs()));
        assert_has_power(
            fixture,
            &format!("pool {pool_id} slot {slot}'s lag-0 coefficient"),
            beta_0,
        );
        assert!(
            (pushed - want).abs() <= tolerance,
            "{fixture}: pool {pool_id} slot {slot}: the pushed RHS {pushed} is not α + β·x̂ = \
             {want} at the cut's own trial point (tolerance {tolerance:e}); the shifted alias \
             reads {aliased}"
        );
        separation = separation.min((aliased - want).abs() - 2.0 * tolerance);
    }
    separation
}

/// derived: five stages of one PAR(1) hydro, `ψ = 1/2`, one opening each, over January's weeks
/// 0 to 2, its last ten days (the stage that closes it) and February, with the lag seeded at 200
/// m³/s above every January mean. A 400 MW load over a 200 MW plant and a 100 `$/MWh` thermal
/// gives water a constant value, so a cut generated at a trial state is the successor value
/// there. The partial sum is the same on every iteration, so each pool's cuts are exact at its
/// generating visit: pools 0 to 2 carry the seeded lag, `x̂_lag = 200`, and pool 3 closes January
/// at `x̂_lag = Σ_t w_t·ẑ_t/Σ_t w_t` over the four stages' hours `w_t`, with
/// `ẑ_t = b_t + ψ·200 + σ_t·η_t`. Each cut's RHS at its generating visit is `α + β_lag·x̂_lag`;
/// the shifted alias gives `α + β_lag·ẑ`, which is off by `β_lag·(ẑ − x̂_lag)`, a distance each
/// iteration's `η_t` makes non-zero here.
#[test]
fn stored_cuts_are_tight_at_their_own_trial_point_on_a_same_path_partial_sum_deck() {
    const FIXTURE: &str = "same-path partial-sum deck";
    const JANUARY_STAGE_HOURS: [f64; 4] = [168.0, 168.0, 168.0, 240.0];
    let run = train_same_path_deck();
    let setup = &run.setup;
    let state = setup.training_ctx().state;
    assert_eq!(
        (state.hydro_count, state.max_par_order),
        (1, 1),
        "{FIXTURE}: one hydro with one lag"
    );
    let z_at = |stage: usize, incoming: &[f64]| {
        let lags: Vec<f64> = (0..state.max_par_order)
            .map(|l| incoming[lag_dim(setup, 0, l)])
            .collect();
        realized_inflow(setup, stage, 0, &lags)
    };
    let window = [0_usize, 1, 2, 3];
    let mut pools_checked = 0;
    for pool_id in window {
        let pool = &setup.fcf.pools[pool_id];
        assert_eq!(
            pool.populated(),
            run.bounds.len(),
            "{FIXTURE}: pool {pool_id} holds one cut per iteration"
        );
        let carried = SAME_PATH_LAG_SEED_M3S;
        let january_mean = {
            let seed = setup.training_ctx().initial_state.to_vec();
            let total: f64 = JANUARY_STAGE_HOURS.iter().sum();
            window
                .iter()
                .zip(JANUARY_STAGE_HOURS)
                .map(|(&t, hours)| hours / total * z_at(t, &seed))
                .sum::<f64>()
        };
        for slot in 0..pool.populated() {
            let visit = generating_visit(&run, pool_id, slot);
            let want_lag = if pool_id < 3 { carried } else { january_mean };
            let got = visit.outgoing[lag_dim(setup, 0, 0)];
            assert!(
                (got - want_lag).abs() <= rhs_tolerance(want_lag.abs()),
                "{FIXTURE}: pool {pool_id} slot {slot}: the visit leaves lag {got}, derived \
                 {want_lag}"
            );
        }
        let separation = assert_cut_rhs_at_its_own_trial_point(
            &run,
            FIXTURE,
            pool_id,
            |visit, _| z_at(pool_id, &visit.incoming),
            |_, _| opening_noise(setup, pool_id),
        );
        assert!(
            separation > 0.0,
            "{FIXTURE}: pool {pool_id}: the alias RHS is within the tolerance of α + β·x̂, so \
             the tightness check could hold vacuously"
        );
        pools_checked += 1;
    }
    assert_eq!(pools_checked, 4, "{FIXTURE}: every pool is checked");
}

/// derived: on the deck above every iteration's cuts are exact at the generating visit, so a
/// stage-0 value built from them is the cost of a path the single forward pass can take: the
/// lower bound never exceeds the forward cost. A cut evaluated at the shifted alias is off by
/// `β_lag·(ẑ − x̂_lag)`, which with a lag seeded above every mean overstates the value of the
/// future and lifts the lower bound past the path's cost.
#[test]
fn lower_bound_stays_below_the_upper_bound_on_a_same_path_partial_sum_deck() {
    let run = train_same_path_deck();
    assert!(
        run.bounds.len() > 1,
        "the run must iterate for the bound to move"
    );
    for &(iteration, lower, upper) in &run.bounds {
        assert!(
            lower <= upper + value_tolerance(lower, upper),
            "iteration {iteration}: the lower bound {lower} exceeds the upper bound {upper}"
        );
    }
}

const WEEKS_THEN_MONTH_TERMINAL_STAGE: usize = 5;
const WEEKS_THEN_MONTH_FORWARD_PASSES: u32 = 6;
const WEEKS_THEN_MONTH_BOUNDARY_INTERCEPT: f64 = 1.0e4;
const WEEKS_THEN_MONTH_BOUNDARY_LAG_SLOPE: f64 = -0.75;

/// The terminal pool's boundary row as the worker solves of stage 5 saw it in the one iteration
/// of a run: the forward trials first, then every trial's backward openings.
struct TerminalBoundaryPushes {
    forward: Vec<f64>,
    backward: Vec<f64>,
}

/// The weeks-then-month study with a one-cut boundary on its first lag installed, trained for one
/// iteration of `WEEKS_THEN_MONTH_FORWARD_PASSES` trials; the worker solves that carry the
/// boundary row alone are the terminal stage's, since no interior pool holds a cut yet.
fn train_weeks_then_month_deck(
    backward_scheduler: BackwardScheduler,
) -> (StudySetup, TerminalBoundaryPushes) {
    let (system, mut config, hydro_models) = weeks_then_month_study();
    config.training.selection = Some(TrainingSelection::Sampled {
        forward_passes: WEEKS_THEN_MONTH_FORWARD_PASSES,
    });
    let mut setup = build_setup_in_code_with_models(system, &config, hydro_models);

    let lag = lag_dim(&setup, 0, 0);
    let coefficients: Vec<f64> = (0..setup.fcf.state_dimension)
        .map(|dim| {
            if dim == lag {
                WEEKS_THEN_MONTH_BOUNDARY_LAG_SLOPE
            } else {
                0.0
            }
        })
        .collect();
    let boundary_dir = tempfile::tempdir().expect("tempdir");
    inject_synthetic_boundary_cut(
        &mut setup,
        boundary_dir.path(),
        WEEKS_THEN_MONTH_BOUNDARY_INTERCEPT,
        &coefficients,
    );

    let theta = setup.training_ctx().state.theta;
    let worker_log = CutRowPushLog::new();
    let lower_bound_log = CutRowPushLog::new();
    let recording = |log: &CutRowPushLog| {
        ActiveSolver::new().map(|solver| {
            CutRowPushRecorder::new(solver, theta, log.clone(), AppendedCutRowBound::Patched)
        })
    };
    let mut solver = recording(&lower_bound_log).expect("the lower-bound solver must build");
    let outcome = setup
        .train_with_solver_profiles(
            &mut solver,
            &StubComm,
            1,
            || recording(&worker_log),
            SolverProfiles {
                backward_scheduler,
                ..SolverProfiles::default()
            },
        )
        .expect("training must return Ok");
    assert!(
        outcome.error.is_none(),
        "training must not error: {:?}",
        outcome.error
    );

    let at_terminal_stage =
        |record: &&CutRowPushRecord| record.frozen_rows == 1 && record.appended_rows == 0;
    let pushes: Vec<f64> = worker_log
        .records()
        .iter()
        .filter(at_terminal_stage)
        .map(|record| record.cut_row_rhs[0])
        .collect();
    let trials = WEEKS_THEN_MONTH_FORWARD_PASSES as usize;
    assert_eq!(
        pushes.len(),
        trials * (1 + WEEKS_THEN_MONTH_TERMINAL_OPENINGS),
        "one forward solve per trial and one backward solve per trial and terminal opening"
    );
    let (forward, backward) = pushes.split_at(trials);
    let pushes = TerminalBoundaryPushes {
        forward: forward.to_vec(),
        backward: backward.to_vec(),
    };
    (setup, pushes)
}

/// derived: five January-season weeks and the rest of February, one PAR(1) hydro, `ψ = 1/2`, one
/// opening at every stage but the terminal one (three), the lag seeded at 200 m³/s. The fifth week
/// runs 29 January to 5 February: its 72 h in January's 744 h finalize January, and its 96 h
/// open February's period, whose 696 h the terminal stage closes with its own 600. The terminal
/// pool is therefore `Closing` with `W = 96/696` and `w = 600/696`, and every path of the study
/// reads the same partial sum `s = W·ẑ_4`, so a cut there is exact: a boundary row `θ ≥ α + β·x`
/// on the first lag is pushed at `α + β·Φ` with `Φ = (96·ẑ_4 + 600·ẑ_5)/696`, `ẑ_5` that
/// opening's realized inflow at the lag `(168·(ẑ_0 + ẑ_1 + ẑ_2 + ẑ_3) + 72·ẑ_4)/744` January
/// leaves, and `ẑ_t = b_t + ψ·200 + σ_t·η_t` before it. The shifted alias pushes `α + β·ẑ_5`, off
/// by `β·(96/696)·(ẑ_5 − ẑ_4)`, and a terminal-stage solve is pushed at one opening's `Φ`, so the
/// openings' pushes differ from each other.
#[test]
fn terminal_boundary_value_on_a_weeks_then_month_calendar_matches_the_closed_form() {
    const FIXTURE: &str = "weeks-then-month deck";
    const WEEK_HOURS: f64 = 168.0;
    const JANUARY_HOURS: f64 = 744.0;
    const STRADDLE_JANUARY_HOURS: f64 = 72.0;
    const STRADDLE_FEBRUARY_HOURS: f64 = 96.0;
    const FEBRUARY_HOURS: f64 = 696.0;
    const TERMINAL_HOURS: f64 = 600.0;

    for backward_scheduler in [
        BackwardScheduler::ByScenario {},
        BackwardScheduler::ByNode { block_size: None },
    ] {
        let label = format!("{FIXTURE} ({backward_scheduler:?})");
        let (setup, pushes) = train_weeks_then_month_deck(backward_scheduler);
        let terminal = WEEKS_THEN_MONTH_TERMINAL_STAGE;
        let state = setup.training_ctx().state;
        assert_eq!(
            (state.hydro_count, state.max_par_order),
            (1, 1),
            "{label}: one hydro with one lag"
        );

        let (own_weight, period_weight) = match OracleLagPoint::of(&setup, terminal) {
            OracleLagPoint::Closing {
                own_weight,
                period_weight,
            } => (own_weight, period_weight[0]),
            other => panic!("{label}: stage {terminal} must close its period, got {other:?}"),
        };
        let share = TERMINAL_HOURS / FEBRUARY_HOURS;
        assert!(
            (own_weight - share).abs() <= f64::EPSILON
                && (period_weight - 1.0).abs() <= f64::EPSILON,
            "{label}: the terminal stage closes February at w = {own_weight} over \
             {period_weight}, derived {share} over 1"
        );

        let z_at = |stage: usize, opening: usize, lag: f64| {
            realized_inflow(&setup, stage, opening, &[lag])
        };
        let seed = SAME_PATH_LAG_SEED_M3S;
        let weeks: Vec<f64> = (0..4).map(|stage| z_at(stage, 0, seed)).collect();
        let straddle = z_at(4, 0, seed);
        let january_mean = (WEEK_HOURS * weeks.iter().sum::<f64>()
            + STRADDLE_JANUARY_HOURS * straddle)
            / JANUARY_HOURS;

        let pool = &setup.fcf.pools[terminal];
        assert_eq!(pool.populated(), 1, "{label}: the boundary is one cut");
        let alpha = pool.intercept(0);
        let beta = pool.coefficient_row(0)
            [lag_slot(&setup.training_ctx().cut_state_layouts[terminal], 1, 0, 0)];
        assert_has_power(&label, "boundary lag slope", beta);
        assert_non_degenerate(&label, "boundary lag slope", beta);

        let mut closed = Vec::new();
        let mut alias = Vec::new();
        let mut tolerance = 0.0_f64;
        for opening in 0..WEEKS_THEN_MONTH_TERMINAL_OPENINGS {
            let z = z_at(terminal, opening, january_mean);
            let phi = (STRADDLE_FEBRUARY_HOURS * straddle + TERMINAL_HOURS * z) / FEBRUARY_HOURS;
            closed.push(alpha + beta * phi);
            alias.push(alpha + beta * z);
            tolerance = tolerance.max(rhs_tolerance(
                alpha.abs() + beta.abs() * (straddle.abs() + z.abs()),
            ));
        }
        for (opening, (&want, &aliased)) in closed.iter().zip(&alias).enumerate() {
            assert!(
                (aliased - want).abs() > 2.0 * tolerance,
                "{label}: opening {opening}'s alias push {aliased} is within the tolerance of \
                 the closed form {want}, so the check could hold vacuously"
            );
        }
        for (i, &first) in closed.iter().enumerate() {
            for (j, &second) in closed.iter().enumerate().skip(i + 1) {
                assert!(
                    (first - second).abs() > 2.0 * tolerance,
                    "{label}: openings {i} and {j} push the same bound, so the terminal \
                     openings could not be told apart"
                );
            }
        }

        let opening_of = |site: &str, trial: usize, pushed: f64| {
            let matches: Vec<usize> = (0..closed.len())
                .filter(|&opening| (pushed - closed[opening]).abs() <= tolerance)
                .collect();
            let [opening] = matches[..] else {
                panic!(
                    "{label}: the {site} solve of trial {trial} pushes {pushed}, which is not \
                     α + β·Φ of exactly one opening {closed:?} (tolerance {tolerance:e}); the \
                     shifted alias pushes {alias:?}"
                );
            };
            opening
        };
        let forward: BTreeSet<usize> = pushes
            .forward
            .iter()
            .enumerate()
            .map(|(trial, &pushed)| opening_of("forward", trial, pushed))
            .collect();
        assert!(
            forward.len() >= 2,
            "{label}: the forward trials drew one terminal opening, so Φ never varied"
        );
        let mut backward_openings = vec![0_usize; closed.len()];
        for (index, &pushed) in pushes.backward.iter().enumerate() {
            backward_openings[opening_of("backward", index, pushed)] += 1;
        }
        assert_eq!(
            backward_openings,
            vec![WEEKS_THEN_MONTH_FORWARD_PASSES as usize; closed.len()],
            "{label}: every trial's backward pass solves each terminal opening once"
        );
    }
}

fn multi_resolution_deck() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(MULTI_RESOLUTION_DECK)
}

/// derived: stage 6 rebuilds its lags from the downstream ring, and the primary weight survives
/// the rebuild, so stage 7 closes a period that already holds the weight `W` of one half: its
/// lag 0 is `Φ(0, 0) = s/(W + w) + c·ẑ` with `c = w/(W + w) = 1/2`, and `Φ(0, l) = â(0, l − 1)`
/// for `l ≥ 1`. The RHS each stage-7 cut is pushed at its generating visit is `α + β·x̂` over the
/// lags the kernel wrote there. The forward noise is drawn out of the tree, so the visit's ẑ is
/// read back from the lag the kernel wrote, `ẑ = (x̂_lag0 − s/(W + w))/c`, and its noise from
/// `ẑ = p(â) + σ·η`; the shifted alias evaluates at `ẑ` itself.
#[test]
fn d30_stage_seven_cuts_evaluate_at_the_closing_point_of_one_half() {
    const FIXTURE: &str = "d30";
    const STAGE: usize = MULTI_RESOLUTION_REBUILD_STAGE + 1;
    let (_, setup) = fresh_system_and_setup_with(&multi_resolution_deck(), apply_oracle_config);
    let (own_weight, period_weight) = match OracleLagPoint::of(&setup, STAGE) {
        OracleLagPoint::Closing {
            own_weight,
            period_weight,
        } => (own_weight, period_weight[0]),
        other => panic!("{FIXTURE}: stage {STAGE} must close its period, got {other:?}"),
    };
    let closing_share = own_weight / period_weight;
    assert_eq!(
        closing_share, 0.5,
        "{FIXTURE}: stage {STAGE} closes at one half"
    );
    let run = train_exporting_states(setup);
    let setup = &run.setup;
    let noise_slot = hydro_noise_slot(setup);
    let sigma = setup.training_ctx().stochastic.par().sigma(STAGE, 0);
    assert_non_degenerate(FIXTURE, "σ of stage 7", sigma);
    let z_of = |visit: &Visit| {
        let carried = visit.partial_sum[0] / period_weight;
        (visit.outgoing[lag_dim(setup, 0, 0)] - carried) / closing_share
    };
    let separation = assert_cut_rhs_at_its_own_trial_point(
        &run,
        FIXTURE,
        STAGE,
        |visit, _| z_of(visit),
        |visit, _| {
            let mut noise = opening_noise(setup, STAGE);
            noise[noise_slot] =
                (z_of(visit) - conditional_mean(setup, STAGE, &visit.incoming)) / sigma;
            noise
        },
    );
    assert!(
        separation > 0.0,
        "{FIXTURE}: the alias RHS is within the tolerance of α + β·x̂, so the closing-point check \
         could hold vacuously"
    );
}

/// derived: the hydro empties into the resource priced `price` (the thermal, or σ when σ is
/// basic), so one more hm³ of `V_in` adds `γ_q/κ` m³/s of turbining and `σ_c·γᵥ/2` MW of head
/// through the plane, each displacing `price`.
fn fpha_storage_value(price: f64, plane: &FphaPlane, share: f64, hours: f64) -> f64 {
    -(price * plane.gamma_q) / (M3S_TO_HM3 * DEFAULT_COST_SCALE_FACTOR)
        - share * price * hours * plane.gamma_v / (2.0 * DEFAULT_COST_SCALE_FACTOR)
}

/// [`terminal_par1_case`] with the sole hydro on [`FPHA_PLANE`], σ priced at `violation_cost`.
fn fpha_case(
    name: &'static str,
    violation_cost: f64,
    generation_floor: Option<GenerationFloor>,
) -> StudyCase {
    StudyCase {
        name,
        fpha: Some(FphaCase {
            plane: FPHA_PLANE,
            violation_cost,
            generation_floor,
            lateral_inflow: None,
        }),
        ..terminal_par1_case(TERMINAL_PSI, MONTH_HOURS)
    }
}

/// The compact setup of `case`, the cut opening 0 of [`FPHA_STAGE`] yields at the initial state
/// with an empty pool, and that solve's unscaled σ in MW.
fn solve_fpha_case(case: &StudyCase) -> (StudySetup, ProbeCut, f64) {
    let (setup, _) = compact_study(case, FPHA_STAGE);
    let x_hat = oracle_initial_state(&setup);
    let noise = opening_noise(&setup, FPHA_STAGE);
    let cut = compact_cut(&setup, FPHA_STAGE, &x_hat, &noise, &[], case.name);
    let col = setup.stage_ctx().geometry_per_stage[FPHA_STAGE]
        .fpha_slack_col(FphaCellLocal::new(0), BlockIdx::new(0));
    let col_scale = setup.inputs.stage_data.stage_templates.templates[FPHA_STAGE].col_scale[col];
    let sigma_mw = cut.primal[col] * col_scale;
    (setup, cut, sigma_mw)
}

fn assert_fpha_storage_coefficient(
    case: &StudyCase,
    setup: &StudySetup,
    cut: &ProbeCut,
    price: f64,
) {
    let fixture = case.name;
    let storage = storage_slot(&setup.training_ctx().cut_state_layouts[FPHA_STAGE - 1], 0);
    let hours = case.block_hours;
    let want = fpha_storage_value(price, &FPHA_PLANE, FPHA_CELL_SHARE, hours);
    let head =
        FPHA_CELL_SHARE * price * hours * FPHA_PLANE.gamma_v / (2.0 * DEFAULT_COST_SCALE_FACTOR);
    let tol = UnscaledTolerance::from_active_profile();
    // n = 10: price·γ_q, M, M·K and ÷ (4), σ_c·price, ·h, ·γᵥ and ÷ (4, 2·K is exact) and their
    // difference derive it, rc ÷ col_scale solves it; both terms are negative, so |want| is their
    // absolute sum; added: v_in's rc tolerance.
    let closed_form = ClosedForm {
        fixture,
        quantity: "storage coefficient",
        slot: Some(storage),
        want,
        n: 10,
        abs_sum: want.abs(),
        added: tol.reduced_cost(incoming_storage_col_scale(setup, FPHA_STAGE, 0)),
    };
    assert!(
        head > HEAD_TERM_MARGIN * closed_form.bound(),
        "{fixture}: the head term {head:e} is within {HEAD_TERM_MARGIN} bounds {:e} of the closed \
         form, so a closed form without it could pass",
        closed_form.bound()
    );
    closed_form.assert_solved(cut.coefficients[storage]);
}

/// derived: σ is priced far above the thermal, so the thermal covers the load the plane leaves,
/// σ is 0, the plane dual is `−c·h/K` and the storage coefficient is
/// `fpha_storage_value(c, ..)`: the water value at `γ_q` plus the head term `σ_c·c·h·γᵥ/(2K)`,
/// which a plane without `V_in` drops.
#[test]
fn an_fpha_storage_coefficient_prices_the_head_at_the_thermal_cost_when_the_slack_is_idle() {
    let case = fpha_case(
        "terminal PAR(1) on an FPHA plane, slack idle",
        IDLE_FPHA_VIOLATION_COST,
        None,
    );
    let (setup, cut, sigma_mw) = solve_fpha_case(&case);
    assert!(
        sigma_mw <= POWER_FLOOR,
        "{}: the FPHA slack {sigma_mw:e} MW must stay within {POWER_FLOOR:e} of zero at the price \
         {IDLE_FPHA_VIOLATION_COST}",
        case.name
    );
    assert_fpha_storage_coefficient(&case, &setup, &cut, THERMAL_COST_PER_MWH);
}

/// derived: σ is priced below the thermal, so σ covers the load the plane leaves, the plane dual
/// is `−p·h/K` and the storage coefficient is `fpha_storage_value(p, ..)`; a plane dual read at
/// the thermal cost misses it by the head and turbining terms at `c − p`.
#[test]
fn an_fpha_storage_coefficient_prices_the_head_at_the_slack_price_when_the_slack_is_active() {
    let case = fpha_case(
        "terminal PAR(1) on an FPHA plane, slack active",
        ACTIVE_FPHA_VIOLATION_COST,
        None,
    );
    let hours = case.block_hours;
    let idle = fpha_storage_value(THERMAL_COST_PER_MWH, &FPHA_PLANE, FPHA_CELL_SHARE, hours);
    let active = fpha_storage_value(
        ACTIVE_FPHA_VIOLATION_COST,
        &FPHA_PLANE,
        FPHA_CELL_SHARE,
        hours,
    );
    assert!(
        (active - idle).abs() > SEPARATION_FLOOR * active.abs().max(idle.abs()),
        "{}: the idle and active storage values {idle:e} and {active:e} must differ by more than \
         {SEPARATION_FLOOR:e} relative",
        case.name
    );
    let (setup, cut, sigma_mw) = solve_fpha_case(&case);
    assert_has_power(case.name, "FPHA slack", sigma_mw);
    assert_fpha_storage_coefficient(&case, &setup, &cut, ACTIVE_FPHA_VIOLATION_COST);
}

/// derived: the floor holds `g` above the plane's output, so σ is basic at `p`, above the deficit
/// cost plus the generation-floor penalty, while the thermal stays marginal. `g` is basic, so its
/// zero reduced cost gives `|π_plane| = |π_load| + G·|π_floor|` with `|π_load| = c·h/K`, and the
/// storage coefficient is `fpha_storage_value(p, ..)`.
#[test]
fn a_generation_floor_activates_the_fpha_slack_above_the_two_term_sum() {
    let two_term_sum = DEFICIT_COST_PER_MWH + fixture_penalties().generation_violation_below_cost;
    assert!(
        FLOOR_FPHA_VIOLATION_COST > two_term_sum,
        "the FPHA price {FLOOR_FPHA_VIOLATION_COST} must exceed the deficit cost plus the \
         generation-floor penalty, {two_term_sum}"
    );
    let case = fpha_case(
        "terminal PAR(1) on an FPHA plane with a generation floor",
        FLOOR_FPHA_VIOLATION_COST,
        Some(GENERATION_FLOOR),
    );
    let fixture = case.name;
    let (setup, cut, sigma_mw) = solve_fpha_case(&case);
    assert_has_power(fixture, "FPHA slack", sigma_mw);

    let geometry = &setup.stage_ctx().geometry_per_stage[FPHA_STAGE];
    assert_eq!(
        geometry.fpha.len(),
        1,
        "{fixture}: one cell, one block and one plane give one plane row"
    );
    let plane_row = geometry.fpha.start;
    let load_row = geometry.load_balance_row(BusSys::new(0), BlockIdx::new(0));
    let floor_row = row_with_bounds(&setup, FPHA_STAGE, GENERATION_FLOOR.lower_mw, f64::INFINITY);
    let row_scale = &setup.inputs.stage_data.stage_templates.templates[FPHA_STAGE].row_scale;
    let unscaled_dual = |row: usize| cut.dual[row] * row_scale[row];
    let (plane, load, floor) = (
        unscaled_dual(plane_row),
        unscaled_dual(load_row),
        unscaled_dual(floor_row),
    );
    assert_has_power(fixture, "generation-floor dual", floor);

    let tol = UnscaledTolerance::from_active_profile();
    // n = 3: c·h and ÷ K derive it, d·row_scale solves it; added: the load row's row-dual
    // tolerance.
    let load_price = THERMAL_COST_PER_MWH * case.block_hours / DEFAULT_COST_SCALE_FACTOR;
    ClosedForm {
        fixture,
        quantity: "load-balance dual magnitude",
        slot: None,
        want: load_price,
        n: 3,
        abs_sum: load_price,
        added: tol.row_dual(row_scale[load_row]),
    }
    .assert_solved(load.abs());
    // n = 5: three d·row_scale, G·|π_floor| and the sum; added: the three rows' row-dual
    // tolerances, the floor's times G.
    let through_g = load.abs() + GENERATION_FLOOR_COEFFICIENT * floor.abs();
    ClosedForm {
        fixture,
        quantity: "plane dual magnitude",
        slot: None,
        want: through_g,
        n: 5,
        abs_sum: through_g,
        added: tol.row_dual(row_scale[plane_row])
            + tol.row_dual(row_scale[load_row])
            + GENERATION_FLOOR_COEFFICIENT * tol.row_dual(row_scale[floor_row]),
    }
    .assert_solved(plane.abs());

    assert_fpha_storage_coefficient(&case, &setup, &cut, FLOOR_FPHA_VIOLATION_COST);
}

/// The lateral part of the influencing hydro's lag-0 coefficient: one more m³/s of lag raises
/// `z_k` by `ψ_k`, which lowers the plane by `σ_c·(−γ_s)·f_k·ψ_k` MW, each MW displacing `price`
/// over `hours`.
fn lateral_lag_value(
    price: f64,
    plane: &FphaPlane,
    share: f64,
    factor: f64,
    psi: f64,
    hours: f64,
) -> f64 {
    -(price * hours * share * plane.gamma_s * factor * psi) / DEFAULT_COST_SCALE_FACTOR
}

/// Two independent terminal PAR(1) hydros: the first on [`LATERAL_FPHA_PLANE`], σ priced at
/// `violation_cost`, whose tailrace flow takes the second's incremental inflow at `factor`; the
/// second at constant productivity.
fn lateral_fpha_case(name: &'static str, violation_cost: f64, factor: f64) -> StudyCase {
    StudyCase {
        name,
        hydros: vec![
            HydroCase::sole(vec![TERMINAL_PSI]),
            HydroCase {
                id: INFLUENCING_ID,
                name: "K",
                ..HydroCase::sole(vec![TERMINAL_TWIN_PSI])
            },
        ],
        fpha: Some(FphaCase {
            plane: LATERAL_FPHA_PLANE,
            violation_cost,
            generation_floor: None,
            lateral_inflow: Some((INFLUENCING_ID, factor)),
        }),
        ..terminal_par1_case(TERMINAL_PSI, MONTH_HOURS)
    }
}

/// Asserts the influencing hydro's lag-0 coefficient of `declared` exceeds that of `zeroed`, the
/// same solve at a zero factor, by `lateral_lag_value(price, ..)`; both solves share a basis, so
/// the water-balance term cancels in the difference.
fn assert_lateral_lag_coefficient(
    declared: (&StudyCase, &StudySetup, &ProbeCut),
    zeroed: (&StudyCase, &StudySetup, &ProbeCut),
    price: f64,
) {
    let (case, setup, cut) = declared;
    let fixture = case.name;
    let extracted = &setup.training_ctx().cut_state_layouts[FPHA_STAGE - 1];
    let lag = lag_slot(extracted, case.hydros.len(), INFLUENCING, 0);
    let psi = psi_at(setup, FPHA_STAGE, INFLUENCING)[0];
    assert_non_degenerate(fixture, "influencing lag-1 coefficient ψ_k", psi);
    assert_non_degenerate(fixture, "lateral factor f_k", LATERAL_FACTOR);
    let hours = case.block_hours;
    let want = lateral_lag_value(
        price,
        &LATERAL_FPHA_PLANE,
        FPHA_CELL_SHARE,
        LATERAL_FACTOR,
        psi,
        hours,
    );
    let water_term = psi * hours * price * PRODUCTIVITY / DEFAULT_COST_SCALE_FACTOR;
    let tol = UnscaledTolerance::from_active_profile();
    // n = 22: price·h, ·σ_c, ·γ_s, ·f_k, ·ψ and ÷ K derive it (6); the declared solve's
    // d·row_scale and G·π on both coupled rows, their sum, ψ·g, G = −κ (2) and
    // G = σ_c·(−γ_s)·f_k (2) give 10, the zeroed solve's single coupled row gives 5, and their
    // difference 1; the water term `ψ·κ·λ_k` enters both solves before it cancels, so it counts
    // twice in abs_sum; added: |ψ| times both solves' coupled rows' |G|·row-dual tolerance.
    let closed_form = ClosedForm {
        fixture,
        quantity: "influencing lag-0 coefficient, declared minus zero factor",
        slot: Some(lag),
        want,
        n: 22,
        abs_sum: want.abs() + 2.0 * water_term.abs(),
        added: psi.abs()
            * (coupled_dual_tolerance(setup, FPHA_STAGE, INFLUENCING, tol)
                + coupled_dual_tolerance(zeroed.1, FPHA_STAGE, INFLUENCING, tol)),
    };
    assert!(
        want > HEAD_TERM_MARGIN * closed_form.bound(),
        "{fixture}: the lateral term {want:e} is within {HEAD_TERM_MARGIN} bounds {:e} of zero, \
         so a cut without it could pass",
        closed_form.bound()
    );
    let zeroed_lag = lag_slot(
        &zeroed.1.training_ctx().cut_state_layouts[FPHA_STAGE - 1],
        zeroed.0.hydros.len(),
        INFLUENCING,
        0,
    );
    assert_eq!(lag, zeroed_lag, "{fixture}: both solves share the lag slot");
    closed_form.assert_solved(cut.coefficients[lag] - zeroed.2.coefficients[lag]);
}

/// derived: σ is priced far above the thermal, so in both solves the thermal covers the load the
/// plane leaves and the plane dual is `−c·h/K`. `z_k = … + ψ_k·lag` enters the plane row at
/// `σ_c·(−γ_s)·f_k`, so the lag coefficient gains `−c·h·σ_c·γ_s·f_k·ψ_k/K`, positive for
/// `γ_s < 0`: a higher tailrace costs thermal energy, against the negative water term `ψ_k·κ·λ_k`
/// that both solves share.
#[test]
fn a_lateral_inflow_prices_the_influencing_lag_through_the_fpha_plane_when_the_slack_is_idle() {
    let declared = lateral_fpha_case(
        "lateral inflow on an FPHA plane, slack idle",
        IDLE_FPHA_VIOLATION_COST,
        LATERAL_FACTOR,
    );
    let zeroed = lateral_fpha_case(
        "lateral inflow on an FPHA plane at a zero factor, slack idle",
        IDLE_FPHA_VIOLATION_COST,
        0.0,
    );
    let (declared_setup, declared_cut, declared_sigma) = solve_fpha_case(&declared);
    let (zeroed_setup, zeroed_cut, zeroed_sigma) = solve_fpha_case(&zeroed);
    for (case, sigma_mw) in [(&declared, declared_sigma), (&zeroed, zeroed_sigma)] {
        assert!(
            sigma_mw <= POWER_FLOOR,
            "{}: the FPHA slack {sigma_mw:e} MW must stay within {POWER_FLOOR:e} of zero at the \
             price {IDLE_FPHA_VIOLATION_COST}",
            case.name
        );
    }
    assert_lateral_lag_coefficient(
        (&declared, &declared_setup, &declared_cut),
        (&zeroed, &zeroed_setup, &zeroed_cut),
        THERMAL_COST_PER_MWH,
    );
}

/// derived: σ is priced below the thermal, so in both solves σ covers the load the plane leaves
/// and the plane dual is `−p·h/K`: the lag coefficient gains `−p·h·σ_c·γ_s·f_k·ψ_k/K`, positive
/// for `γ_s < 0`. A plane dual read at the thermal cost misses it by the same term at `c − p`.
#[test]
fn a_lateral_inflow_prices_the_influencing_lag_through_the_fpha_plane_when_the_slack_is_active() {
    let declared = lateral_fpha_case(
        "lateral inflow on an FPHA plane, slack active",
        ACTIVE_FPHA_VIOLATION_COST,
        LATERAL_FACTOR,
    );
    let zeroed = lateral_fpha_case(
        "lateral inflow on an FPHA plane at a zero factor, slack active",
        ACTIVE_FPHA_VIOLATION_COST,
        0.0,
    );
    let (declared_setup, declared_cut, declared_sigma) = solve_fpha_case(&declared);
    let (zeroed_setup, zeroed_cut, zeroed_sigma) = solve_fpha_case(&zeroed);
    assert_has_power(declared.name, "FPHA slack", declared_sigma);
    assert_has_power(zeroed.name, "FPHA slack", zeroed_sigma);

    let psi = psi_at(&declared_setup, FPHA_STAGE, INFLUENCING)[0];
    let lateral = |price| {
        lateral_lag_value(
            price,
            &LATERAL_FPHA_PLANE,
            FPHA_CELL_SHARE,
            LATERAL_FACTOR,
            psi,
            declared.block_hours,
        )
    };
    let (idle, active) = (
        lateral(THERMAL_COST_PER_MWH),
        lateral(ACTIVE_FPHA_VIOLATION_COST),
    );
    assert!(
        (active - idle).abs() > SEPARATION_FLOOR * active.abs().max(idle.abs()),
        "{}: the idle and active lateral values {idle:e} and {active:e} must differ by more than \
         {SEPARATION_FLOOR:e} relative",
        declared.name
    );
    assert_lateral_lag_coefficient(
        (&declared, &declared_setup, &declared_cut),
        (&zeroed, &zeroed_setup, &zeroed_cut),
        ACTIVE_FPHA_VIOLATION_COST,
    );
}
