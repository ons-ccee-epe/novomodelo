//! [`StageSolvePrep`]'s pin and opening phases reach the solver with the same
//! calls the direct transforms produce, in the same order.

use std::collections::BTreeMap;

use chrono::NaiveDate;
use novomodelo_core::entities::hydro::{Hydro, HydroGenerationModel, HydroPenalties};
use novomodelo_core::entities::non_controllable::NonControllableSource;
use novomodelo_core::scenario::{
    CorrelationEntity, CorrelationGroup, CorrelationModel, CorrelationProfile, InflowModel,
    LoadModel, NcsModel, SamplingScheme,
};
use novomodelo_core::temporal::{
    Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig, StageStateConfig,
};
use novomodelo_core::{Bus, DeficitSegment, EntityId, SystemBuilder};
use novomodelo_solver::{
    Basis, RowBatch, SolutionView, SolverError, SolverInterface, SolverStatistics, StageTemplate,
};
use novomodelo_stochastic::context::{ClassSchemes, OpeningTreeInputs, build_stochastic_context};
use novomodelo_stochastic::{
    StochasticContext, evaluate_par_conditional_mean_batch, evaluate_par_noise_term_batch,
};

use super::{
    BracketSource, LoadedCutRows, NoiseTermSource, SingleSolvePrepParams, StageSolvePrep,
    StateSource,
};
use crate::{
    context::{StageContext, TrainingContext},
    cut::{
        CutPool,
        rhs::cut_rhs_into,
        rhs_reference::{cut_bracket, cut_noise_term},
    },
    dcs::DcsParams,
    horizon_mode::HorizonMode,
    inflow_method::InflowNonNegativityMethod,
    lead_time::AnticipatedResolution,
    lp::builder::{InflowCoupling, PatchBuffer, StageGeometry, StateBox},
    lp::indexer::{BlockGrid, BlockRowFamily, CutStateProjection, StateSpace, StudyDimensions},
    noise::{
        build_dense_ncs_col_indices, gather_dense_ncs_bounds, realize_inflow, transform_load_noise,
        transform_ncs_noise,
    },
    setup::{
        NodeId,
        node_graph::{NodeGraph, StageIdx},
    },
    solve::StageOpeningSolver,
    test_support::{
        StageContextFixture, all_enabled_cut_state_layouts, bits, chain_node_graph,
        equipment_free_geometry, geometry_with_load_balance, inflow_coupling_with_entries,
        state_layout, study_dims, transit_bucket_only_template,
    },
    workspace::{NoisePreallocation, ScratchBuffers, WorkspaceSizing},
};

/// Single-hydro, single-stage [`StochasticContext`] with a real PAR(0) inflow
/// model, so the inflow transform runs as it does at the forward site rather than
/// vacuously. `load` additionally declares a stochastic load-bus noise model on the
/// same bus, when present.
fn make_stochastic_context(load: Option<LoadModel>) -> StochasticContext {
    stochastic_context(load, true)
}

/// [`make_stochastic_context`], or with `with_par` false, the same system with no stage and no
/// inflow model, so `par().n_stages()` is 0.
fn stochastic_context(load: Option<LoadModel>, with_par: bool) -> StochasticContext {
    let bus = Bus {
        id: EntityId(0),
        name: "B0".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        deficit_segments: vec![DeficitSegment {
            depth_mw: None,
            cost_per_mwh: 1000.0,
        }],
        excess_cost: 0.0,
    };
    let mut hydro = Hydro {
        unit_groups: Vec::new(),
        id: EntityId(1),
        name: "H1".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        downstream_id: None,
        travel_time_hours: None,
        entry_stage_id: None,
        exit_stage_id: None,
        min_storage_hm3: 0.0,
        max_storage_hm3: 100.0,
        min_outflow_m3s: 0.0,
        max_outflow_m3s: None,
        generation_model: HydroGenerationModel::ConstantProductivity,
        min_turbined_m3s: 0.0,
        max_turbined_m3s: 100.0,
        specific_productivity_mw_per_m3s_per_m: None,
        min_generation_mw: 0.0,
        max_generation_mw: 100.0,
        tailrace: None,
        hydraulic_losses: None,
        efficiency: None,
        evaporation_coefficients_mm: None,
        evaporation_reference_volumes_hm3: None,
        diversion: None,
        filling: None,
        penalties: HydroPenalties {
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
            inflow_nonnegativity_cost: 1000.0,
        },
    };
    hydro.declare_mirror_unit_group(EntityId(0));

    let stage = Stage {
        index: 0,
        id: 0,
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
            inflow_lags: false,
        },
        risk_config: StageRiskConfig::Expectation,
        scenario_config: ScenarioSourceConfig {
            branching_factor: 1,
            noise_method: NoiseMethod::Saa,
        },
    };

    let inflow = InflowModel {
        hydro_id: EntityId(1),
        stage_id: 0,
        mean_m3s: 100.0,
        std_m3s: 30.0,
        ar_coefficients: vec![],
        residual_std_ratio: 1.0,
        annual: None,
    };

    let mut profiles = BTreeMap::new();
    profiles.insert(
        "default".to_string(),
        CorrelationProfile {
            groups: vec![CorrelationGroup {
                name: "g1".to_string(),
                entities: vec![CorrelationEntity {
                    entity_type: "inflow".to_string(),
                    id: EntityId(1),
                }],
                matrix: vec![vec![1.0]],
            }],
        },
    );
    let correlation = CorrelationModel {
        method: "spectral".to_string(),
        profiles,
        schedule: vec![],
    };

    let mut builder = SystemBuilder::new()
        .buses(vec![bus])
        .hydros(vec![hydro])
        .stages(if with_par { vec![stage] } else { Vec::new() })
        .inflow_models(if with_par { vec![inflow] } else { Vec::new() })
        .correlation(correlation);
    if let Some(load_model) = load {
        builder = builder.load_models(vec![load_model]);
    }
    let system = builder.build().unwrap();

    build_stochastic_context(
        &system,
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

/// N=1, L=0 template: columns `[storage(0), storage_in(1), theta(2)]`, one
/// water-balance row (`storage_in` has the row-0 nonzero), unscaled.
fn minimal_forward_template() -> StageTemplate {
    StageTemplate {
        num_cols: 3,
        num_rows: 1,
        num_nz: 1,
        col_starts: vec![0_i32, 0, 1, 1],
        row_indices: vec![0_i32],
        values: vec![1.0],
        col_lower: vec![0.0, 0.0, 0.0],
        col_upper: vec![f64::INFINITY, f64::INFINITY, f64::INFINITY],
        objective: vec![0.0, 0.0, 1.0],
        row_lower: vec![0.0],
        row_upper: vec![0.0],
        n_state: 1,
        col_scale: Vec::new(),
        row_scale: Vec::new(),
    }
}

/// Single hydro, single stochastic load bus, 2 rows (water balance + load balance).
fn single_hydro_load_template() -> StageTemplate {
    StageTemplate {
        num_rows: 2,
        row_lower: vec![0.0, 0.0],
        row_upper: vec![0.0, 0.0],
        ..minimal_forward_template()
    }
}

/// Every dimension unbounded — the pin-time box-membership assert is vacuous.
fn unbounded_state_box(n_state: usize) -> StateBox {
    StateBox {
        lower: vec![f64::NEG_INFINITY; n_state],
        upper: vec![f64::INFINITY; n_state],
    }
}

fn minimal_sizing() -> WorkspaceSizing {
    WorkspaceSizing {
        max_openings: 1,
        initial_pool_capacity: 1,
        max_local_fwd: 1,
        noise: NoisePreallocation::StochasticDim,
    }
}

/// The single owner of the [`TrainingContext`] literal these tests share.
struct PrepContext {
    node_graph: NodeGraph,
    horizon: HorizonMode,
    cut_state_layouts: Vec<CutStateProjection>,
}

impl PrepContext {
    fn new(state: &StateSpace, stochastic: &StochasticContext) -> Self {
        Self {
            node_graph: chain_node_graph(stochastic),
            horizon: HorizonMode::Finite { num_stages: 1 },
            cut_state_layouts: all_enabled_cut_state_layouts(state, 1),
        }
    }

    fn ctx<'a>(
        &'a self,
        state: &'a StateSpace,
        stochastic: &'a StochasticContext,
        study_dims: &'a StudyDimensions,
        inflow_method: &'a InflowNonNegativityMethod,
        stages: &'a [Stage],
    ) -> TrainingContext<'a> {
        TrainingContext {
            node_graph: &self.node_graph,
            horizon: &self.horizon,
            state,
            cut_state_layouts: &self.cut_state_layouts,
            study_dims,
            inflow_method,
            stochastic,
            initial_state: &[],
            inflow_scheme: SamplingScheme::InSample,
            load_scheme: SamplingScheme::InSample,
            ncs_scheme: SamplingScheme::InSample,
            stages,
            historical_library: None,
            external_inflow_library: None,
            external_load_library: None,
            external_ncs_library: None,
            lag_accum_seed: &[],
            lag_weight_seed: &[],
            dcs: None,
        }
    }
}

/// Records every `set_col_bounds`/`set_row_bounds`/`set_row_lower_bounds_range`
/// call verbatim, and their interleaving in `calls` (`'c'` per column call, `'r'`
/// per row call, `'R'` per range call); `solve` is never exercised by the
/// solve-preparation pipeline.
#[derive(Default)]
struct RecordingSolver {
    col_bounds_calls: Vec<(Vec<usize>, Vec<f64>, Vec<f64>)>,
    row_bounds_calls: Vec<(Vec<usize>, Vec<f64>, Vec<f64>)>,
    row_range_calls: Vec<(usize, Vec<f64>)>,
    calls: Vec<char>,
}

impl SolverInterface for RecordingSolver {
    type Profile = novomodelo_solver::ActiveProfile;

    fn apply_profile(&mut self, _profile: &Self::Profile) {}

    fn solver_name_version(&self) -> String {
        "RecordingSolver 0.0.0".to_string()
    }

    fn load_model(&mut self, _template: &StageTemplate) {}

    fn add_rows(&mut self, _rows: &RowBatch) {}

    fn set_row_bounds(&mut self, indices: &[usize], lower: &[f64], upper: &[f64]) {
        self.calls.push('r');
        self.row_bounds_calls
            .push((indices.to_vec(), lower.to_vec(), upper.to_vec()));
    }

    fn set_row_lower_bounds_range(&mut self, first_row: usize, lower: &[f64]) {
        self.calls.push('R');
        self.row_range_calls.push((first_row, lower.to_vec()));
    }

    fn set_col_bounds(&mut self, indices: &[usize], lower: &[f64], upper: &[f64]) {
        self.calls.push('c');
        self.col_bounds_calls
            .push((indices.to_vec(), lower.to_vec(), upper.to_vec()));
    }

    fn solve(&mut self, _basis: Option<&Basis>) -> Result<SolutionView<'_>, SolverError> {
        unreachable!("solve() is not exercised by the solve-preparation pipeline")
    }

    fn get_basis(&mut self, out: &mut Basis) {
        crate::test_support::fill_consistent_basis(out);
    }

    fn statistics(&self) -> SolverStatistics {
        SolverStatistics::default()
    }

    fn statistics_into(&self, out: &mut SolverStatistics) {
        out.copy_from(&SolverStatistics::default());
    }

    fn name(&self) -> &'static str {
        "Recording"
    }
}

/// The pin phase then the opening phase on `pinned_state` and `raw_noise` over a
/// loaded LP with no cut rows.
fn prepare_without_cut_rows(
    solver: &mut RecordingSolver,
    patch_buf: &mut PatchBuffer,
    scratch: &mut ScratchBuffers,
    ctx: &StageContext<'_>,
    training_ctx: &TrainingContext<'_>,
    pinned_state: &[f64],
    raw_noise: &[f64],
) {
    let projection = &training_ctx.cut_state_layouts[0];
    let pool = CutPool::new(1, projection.n_slots(), 1, 0);
    StageSolvePrep::prepare_single(
        solver,
        patch_buf,
        scratch,
        ctx,
        training_ctx,
        StageIdx(0),
        &SingleSolvePrepParams {
            state_source: StateSource(pinned_state),
            raw_noise,
            pool: &pool,
            projection,
            opening_solver: &StageOpeningSolver::Frozen,
        },
        std::iter::empty(),
    );
}

/// One bus, one NCS entity, availability factor `mean=0.5, std=0.1` — the D15
/// NCS fixture shape (`lb_evaluate_stage_0_patches_ncs_bounds_per_opening` in
/// `training/lower_bound.rs`), adapted to drive [`StageSolvePrep::prepare_single`].
fn make_ncs_stochastic_context() -> (StochasticContext, Stage) {
    let ncs_entity_id = EntityId(10);
    let bus = Bus {
        id: EntityId(0),
        name: "B0".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        deficit_segments: vec![DeficitSegment {
            depth_mw: None,
            cost_per_mwh: 1000.0,
        }],
        excess_cost: 0.0,
    };
    let ncs_source = NonControllableSource {
        id: ncs_entity_id,
        name: "W1".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        bus_id: EntityId(0),
        entry_stage_id: None,
        exit_stage_id: None,
        max_generation_mw: 100.0,
        allow_curtailment: true,
        curtailment_cost: 0.0,
    };
    let stage = Stage {
        index: 0,
        id: 0,
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
            storage: false,
            inflow_lags: false,
        },
        risk_config: StageRiskConfig::Expectation,
        scenario_config: ScenarioSourceConfig {
            branching_factor: 1,
            noise_method: NoiseMethod::Saa,
        },
    };
    let ncs_model = NcsModel {
        ncs_id: ncs_entity_id,
        stage_id: 0,
        mean: 0.5,
        std: 0.1,
    };
    let mut profiles = BTreeMap::new();
    profiles.insert(
        "default".to_string(),
        CorrelationProfile {
            groups: vec![CorrelationGroup {
                name: "ncs_group".to_string(),
                entities: vec![CorrelationEntity {
                    entity_type: "ncs".to_string(),
                    id: ncs_entity_id,
                }],
                matrix: vec![vec![1.0]],
            }],
        },
    );
    let correlation = CorrelationModel {
        method: "spectral".to_string(),
        profiles,
        schedule: vec![],
    };
    let system = SystemBuilder::new()
        .buses(vec![bus])
        .non_controllable_sources(vec![ncs_source])
        .stages(vec![stage.clone()])
        .ncs_models(vec![ncs_model])
        .correlation(correlation)
        .build()
        .unwrap();
    let stoch = build_stochastic_context(
        &system,
        42,
        None,
        &[],
        &[],
        OpeningTreeInputs::default(),
        ClassSchemes {
            inflow: None,
            load: None,
            ncs: Some(SamplingScheme::InSample),
        },
    )
    .unwrap();
    assert_eq!(stoch.n_stochastic_ncs(), 1);
    (stoch, stage)
}

/// [`StageSolvePrep`]'s NCS-patch wiring must reach the solver with the same
/// `set_col_bounds` call the direct sequence produces (`transform_ncs_noise` →
/// `build_dense_ncs_col_indices` → `gather_dense_ncs_bounds` → `set_col_bounds`).
// Rationale: the inline System/StochasticContext fixture and the reference vs.
// owner comparison are one coherent scenario; splitting them into helpers would
// scatter the setup the assertions depend on and obscure the test.
#[allow(clippy::too_many_lines)]
#[test]
fn the_prep_wires_the_ncs_patch_matching_the_direct_sequence() {
    let (stoch, ncs_stage) = make_ncs_stochastic_context();
    let state = state_layout(0, 0);
    let templates = vec![StageTemplate {
        num_cols: 1,
        num_rows: 0,
        num_nz: 0,
        col_starts: vec![0_i32, 0],
        row_indices: vec![],
        values: vec![],
        col_lower: vec![0.0],
        col_upper: vec![100.0],
        objective: vec![0.0],
        row_lower: vec![],
        row_upper: vec![],
        n_state: 0,
        col_scale: Vec::new(),
        row_scale: Vec::new(),
    }];
    let ncs_max_gen = vec![100.0_f64];
    let ncs_allow_curtailment = vec![true];
    let ncs_stochastic_dense_col = vec![0_usize];
    let ncs_stochastic_windows: Vec<(Option<i32>, Option<i32>)> = vec![(None, None)];
    let state_boxes = vec![unbounded_state_box(state.n_state)];
    let couplings = vec![inflow_coupling_with_entries(0, &[])];
    let geometry = vec![StageGeometry {
        ncs_generation: 0..1,
        n_blks: 1,
        ..equipment_free_geometry(&[1]).remove(0)
    }];
    let fixture = StageContextFixture::new(&templates, &state_boxes, &geometry)
        .without_inflow_entries(1)
        .inflow_coupling(&couplings)
        .ncs_stochastic_dense_col(&ncs_stochastic_dense_col)
        .ncs_stochastic_windows(&ncs_stochastic_windows)
        .ncs_max_gen(&ncs_max_gen)
        .ncs_allow_curtailment(&ncs_allow_curtailment);
    let ctx = fixture.ctx();
    let prep = PrepContext::new(&state, &stoch);
    let study_dims = StudyDimensions::default();
    let stages = vec![ncs_stage];
    let training_ctx = prep.ctx(
        &state,
        &stoch,
        &study_dims,
        &InflowNonNegativityMethod::None,
        &stages,
    );

    let raw_noise = vec![0.37_f64];
    let sizing = minimal_sizing();

    // ---- reference: transform_ncs_noise -> build indices -> gather -> set,
    // called directly, independent of StageSolvePrep's own wiring ----
    let mut reference_scratch = ScratchBuffers::new(&training_ctx, &ctx, sizing);
    let mut reference_solver = RecordingSolver::default();
    transform_ncs_noise(
        &raw_noise,
        &stoch,
        StageIdx(0),
        1,
        &ncs_max_gen,
        &ncs_allow_curtailment,
        &mut reference_scratch.ncs_col_lower_buf,
        &mut reference_scratch.ncs_col_upper_buf,
    );
    build_dense_ncs_col_indices(
        &ncs_stochastic_dense_col,
        &geometry[0],
        &mut reference_scratch.ncs_col_indices_buf,
    );
    gather_dense_ncs_bounds(
        &ncs_stochastic_windows,
        0,
        1,
        &reference_scratch.ncs_col_lower_buf,
        &reference_scratch.ncs_col_upper_buf,
        &mut reference_scratch.ncs_col_lower_active_buf,
        &mut reference_scratch.ncs_col_upper_active_buf,
    );
    reference_solver.set_col_bounds(
        &reference_scratch.ncs_col_indices_buf,
        &reference_scratch.ncs_col_lower_active_buf,
        &reference_scratch.ncs_col_upper_active_buf,
    );

    // ---- owner: StageSolvePrep's NCS-patch wiring ----
    let mut owner_scratch = ScratchBuffers::new(&training_ctx, &ctx, sizing);
    let mut owner_solver = RecordingSolver::default();
    let mut owner_patch_buf = PatchBuffer::new(&state, &[], &[], &couplings);
    prepare_without_cut_rows(
        &mut owner_solver,
        &mut owner_patch_buf,
        &mut owner_scratch,
        &ctx,
        &training_ctx,
        &[],
        &raw_noise,
    );

    assert_eq!(
        owner_solver.col_bounds_calls.last(),
        reference_solver.col_bounds_calls.last(),
        "the prep's NCS patch must match the direct sequence"
    );
    let (indices, lower, upper) = owner_solver
        .col_bounds_calls
        .last()
        .expect("the NCS patch must issue at least one set_col_bounds call");
    assert_eq!(indices, &[0_usize]);
    assert_eq!(lower, &[0.0]);
    let expected_upper = 100.0_f64 * (0.5 + 0.1 * 0.37_f64).clamp(0.0, 1.0);
    assert!((upper[0] - expected_upper).abs() < 1e-9);
}

/// The state pin reaches the solver first, then the row bounds, then the NCS
/// column bounds, then the cut-row range.
#[test]
fn the_prep_pins_the_state_columns_before_any_row_bound() {
    let (stoch, ncs_stage) = make_ncs_stochastic_context();
    let state = state_layout(0, 0);
    let templates = vec![StageTemplate {
        num_cols: 1,
        num_rows: 0,
        num_nz: 0,
        col_starts: vec![0_i32, 0],
        row_indices: vec![],
        values: vec![],
        col_lower: vec![0.0],
        col_upper: vec![100.0],
        objective: vec![0.0],
        row_lower: vec![],
        row_upper: vec![],
        n_state: 0,
        col_scale: Vec::new(),
        row_scale: Vec::new(),
    }];
    let ncs_max_gen = vec![100.0_f64];
    let ncs_allow_curtailment = vec![true];
    let ncs_stochastic_dense_col = vec![0_usize];
    let ncs_stochastic_windows: Vec<(Option<i32>, Option<i32>)> = vec![(None, None)];
    let state_boxes = vec![unbounded_state_box(state.n_state)];
    let couplings = vec![inflow_coupling_with_entries(0, &[])];
    let geometry = vec![StageGeometry {
        ncs_generation: 0..1,
        n_blks: 1,
        ..equipment_free_geometry(&[1]).remove(0)
    }];
    let fixture = StageContextFixture::new(&templates, &state_boxes, &geometry)
        .without_inflow_entries(1)
        .inflow_coupling(&couplings)
        .ncs_stochastic_dense_col(&ncs_stochastic_dense_col)
        .ncs_stochastic_windows(&ncs_stochastic_windows)
        .ncs_max_gen(&ncs_max_gen)
        .ncs_allow_curtailment(&ncs_allow_curtailment);
    let ctx = fixture.ctx();
    let prep = PrepContext::new(&state, &stoch);
    let study_dims = StudyDimensions::default();
    let stages = vec![ncs_stage];
    let training_ctx = prep.ctx(
        &state,
        &stoch,
        &study_dims,
        &InflowNonNegativityMethod::None,
        &stages,
    );

    let raw_noise = vec![0.37_f64];
    let mut scratch = ScratchBuffers::new(&training_ctx, &ctx, minimal_sizing());
    let mut solver = RecordingSolver::default();
    let mut patch_buf = PatchBuffer::new(&state, &[], &[], &couplings);
    prepare_without_cut_rows(
        &mut solver,
        &mut patch_buf,
        &mut scratch,
        &ctx,
        &training_ctx,
        &[],
        &raw_noise,
    );

    assert_eq!(solver.calls, ['c', 'r', 'c', 'R']);
    assert_eq!(
        solver.col_bounds_calls[0].0.len(),
        patch_buf.state_col_patch_count(),
        "the first column call is the state pin"
    );
}

/// The prep patches the load-balance row and shifts the inflow-coupled row from
/// the raw noise, each matching a direct reference call of its own transform.
#[test]
fn the_prep_patches_the_load_row_and_shifts_the_coupled_row_from_the_raw_noise() {
    let state = state_layout(1, 0);
    let stochastic = make_stochastic_context(Some(LoadModel {
        bus_id: EntityId(0),
        stage_id: 0,
        mean_mw: 300.0,
        std_mw: 50.0,
    }));
    let templates = vec![single_hydro_load_template()];
    let state_boxes = vec![unbounded_state_box(state.n_state)];
    let geometry_per_stage = vec![geometry_with_load_balance(1, 1, 1)];
    let load_bus_indices = [0_usize];
    let couplings = vec![inflow_coupling_with_entries(1, &[(0, 0)])];
    let fixture = StageContextFixture::new(&templates, &state_boxes, &geometry_per_stage)
        .without_inflow_entries(1)
        .inflow_coupling(&couplings)
        .load_bus_indices(&load_bus_indices);
    let ctx = fixture.ctx();
    let prep = PrepContext::new(&state, &stochastic);
    let study_dims = study_dims();
    let training_ctx = prep.ctx(
        &state,
        &stochastic,
        &study_dims,
        &InflowNonNegativityMethod::None,
        &[],
    );

    let current_state = vec![42.0_f64];
    // [hydro eta | load eta].
    let raw_noise = vec![0.3_f64, 0.4_f64];
    let sizing = minimal_sizing();
    let mut scratch = ScratchBuffers::new(&training_ctx, &ctx, sizing);

    let mut solver = RecordingSolver::default();
    let mut patch_buf =
        PatchBuffer::new(&state, &load_bus_indices, &geometry_per_stage, &couplings);
    prepare_without_cut_rows(
        &mut solver,
        &mut patch_buf,
        &mut scratch,
        &ctx,
        &training_ctx,
        &current_state,
        &raw_noise,
    );

    assert_eq!(solver.col_bounds_calls.len(), 1, "the state pin runs");

    let mut reference_scratch = ScratchBuffers::new(&training_ctx, &ctx, sizing);
    transform_load_noise(
        &raw_noise,
        &stochastic,
        StageIdx(0),
        1,
        &mut reference_scratch.load_rhs_buf,
    );
    let mut reference_patch_buf =
        PatchBuffer::new(&state, &load_bus_indices, &geometry_per_stage, &couplings);
    let grid = BlockGrid::new(1, training_ctx.study_dims.max_deficit_segments);
    let load_rows = BlockRowFamily::per_block(1..2);
    reference_patch_buf.fill_load_patches(
        load_rows,
        grid,
        &reference_scratch.load_rhs_buf,
        &[0],
        &[],
    );
    evaluate_par_conditional_mean_batch(
        stochastic.par(),
        0,
        &current_state[state.state_dim_lag_range()],
        &mut reference_scratch.par_mean_buf,
    );
    realize_inflow(
        &raw_noise,
        StageIdx(0),
        &training_ctx,
        &mut reference_scratch,
        None,
    );
    assert!(
        (reference_scratch.realized_inflow_buf[0] - 109.0).abs() < 1e-12,
        "PAR(0) with mean 100 and std 30 at eta 0.3 gives 109, got {}",
        reference_scratch.realized_inflow_buf[0]
    );
    reference_patch_buf.fill_inflow_shift_patches(
        &couplings[0],
        &reference_scratch.realized_inflow_buf,
        &[],
    );

    let (indices, lower, upper) = solver
        .row_bounds_calls
        .last()
        .expect("the load and inflow patches must issue one set_row_bounds call");
    assert_eq!(solver.row_bounds_calls.len(), 1);
    assert_eq!(indices, &[1_usize, 0_usize]);
    assert_eq!(
        (lower[0], upper[0]),
        (reference_patch_buf.lower[0], reference_patch_buf.upper[0]),
        "the load-balance row must match the direct transform_load_noise + fill_load_patches reference"
    );
    assert_eq!(
        (lower[1], upper[1]),
        (reference_patch_buf.lower[1], reference_patch_buf.upper[1]),
        "the coupled row must match the direct realize_inflow + fill_inflow_shift_patches reference"
    );
    assert_eq!(
        (lower[1], upper[1]),
        (109.0, 109.0),
        "the coupled row is shifted by the realized inflow"
    );
}

/// The prep patches the inflow-slack bounds after the other patches
/// only under the truncation methods: PAR(0) with mean 100 and std 30 at η = -5
/// gives `r = -50`, so `TruncationWithPenalty`'s cap is 50 and `Truncation` fixes
/// the slack at `[50, 50]`; `Penalty` gets no slack patch.
#[test]
fn the_prep_patches_the_inflow_slack_bounds_only_under_the_truncation_methods() {
    let state = state_layout(1, 0);
    let stochastic = make_stochastic_context(None);
    let templates = vec![minimal_forward_template()];
    let state_boxes = vec![unbounded_state_box(state.n_state)];
    let couplings = vec![inflow_coupling_with_entries(1, &[(0, 0)])];
    let geometry = vec![StageGeometry {
        inflow_slack: 4..5,
        ..equipment_free_geometry(&[1]).remove(0)
    }];
    let fixture = StageContextFixture::new(&templates, &state_boxes, &geometry)
        .without_inflow_entries(1)
        .inflow_coupling(&couplings);
    let ctx = fixture.ctx();
    let prep = PrepContext::new(&state, &stochastic);
    let study_dims = study_dims();
    let current_state = vec![42.0_f64];
    let raw_noise = vec![-5.0_f64];

    for (method, expected_calls, expected_last) in [
        (
            InflowNonNegativityMethod::TruncationWithPenalty,
            2,
            Some((vec![4_usize], vec![0.0_f64], vec![50.0_f64])),
        ),
        (
            InflowNonNegativityMethod::Truncation,
            2,
            Some((vec![4], vec![50.0], vec![50.0])),
        ),
        (InflowNonNegativityMethod::Penalty, 1, None),
    ] {
        let training_ctx = prep.ctx(&state, &stochastic, &study_dims, &method, &[]);
        let mut scratch = ScratchBuffers::new(&training_ctx, &ctx, minimal_sizing());
        let mut solver = RecordingSolver::default();
        let mut patch_buf = PatchBuffer::new(&state, &[], &geometry, &couplings);
        prepare_without_cut_rows(
            &mut solver,
            &mut patch_buf,
            &mut scratch,
            &ctx,
            &training_ctx,
            &current_state,
            &raw_noise,
        );
        assert_eq!(
            solver.col_bounds_calls.len(),
            expected_calls,
            "{method:?}: set_col_bounds call count"
        );
        if let Some(expected) = expected_last {
            assert_eq!(
                solver.col_bounds_calls.last(),
                Some(&expected),
                "{method:?}: the last set_col_bounds is the inflow-slack patch"
            );
        }
    }
}

const POOL_CAPACITY: usize = 3;

/// Compact layout `[storage, lag (0, 0), lag (0, 1)]` over a PAR(0) model with
/// p = 100 and σ = 30, a template whose every column scales by 2 and whose one
/// structural row (scale 4) the hydro's inflow shifts, an inflow-slack column
/// after θ, and a pool of `cuts`.
struct CompactCase {
    state: StateSpace,
    stochastic: StochasticContext,
    templates: Vec<StageTemplate>,
    state_boxes: Vec<StateBox>,
    geometry: Vec<StageGeometry>,
    couplings: Vec<InflowCoupling>,
    study_dims: StudyDimensions,
    prep: PrepContext,
    pool: CutPool,
}

impl CompactCase {
    fn new(cuts: &[(f64, [f64; 3])]) -> Self {
        Self::with_stochastic(make_stochastic_context(None), cuts)
    }

    fn with_stochastic(stochastic: StochasticContext, cuts: &[(f64, [f64; 3])]) -> Self {
        let state = StateSpace::new(
            1,
            2,
            Vec::new(),
            Vec::new(),
            AnticipatedResolution::default(),
            &[2],
        );
        let num_cols = state.theta + 2;
        let templates = vec![StageTemplate {
            num_rows: 1,
            row_lower: vec![0.0],
            row_upper: vec![0.0],
            col_scale: vec![2.0; num_cols],
            row_scale: vec![4.0],
            ..transit_bucket_only_template(num_cols, state.n_state)
        }];
        let geometry = vec![StageGeometry {
            inflow_slack: state.theta + 1..state.theta + 2,
            ..equipment_free_geometry(&[1]).remove(0)
        }];
        let state_boxes = vec![unbounded_state_box(state.n_state)];
        let prep = PrepContext::new(&state, &stochastic);
        let projection = &prep.cut_state_layouts[0];
        assert_eq!(projection.lag_point().lag0(), 1..2);
        assert_eq!(projection.lag_point().shifted(), 2..3);
        let mut pool = CutPool::new(POOL_CAPACITY, projection.n_slots(), 1, 0);
        for (k, (intercept, row)) in (0_u64..).zip(cuts) {
            pool.add_cut(NodeId(0), k, 0, *intercept, row);
        }
        Self {
            state,
            stochastic,
            templates,
            state_boxes,
            geometry,
            couplings: vec![inflow_coupling_with_entries(1, &[(0, 0)])],
            study_dims: study_dims(),
            prep,
            pool,
        }
    }

    fn fixture(&self) -> StageContextFixture<'_> {
        StageContextFixture::new(&self.templates, &self.state_boxes, &self.geometry)
            .without_inflow_entries(1)
            .inflow_coupling(&self.couplings)
    }

    fn scratch(ctx: &StageContext<'_>, training_ctx: &TrainingContext<'_>) -> ScratchBuffers {
        let sizing = WorkspaceSizing {
            initial_pool_capacity: POOL_CAPACITY,
            ..minimal_sizing()
        };
        ScratchBuffers::new(training_ctx, ctx, sizing)
    }

    fn pin_and_bracket(
        &self,
        trial_state: &[f64],
        slots: &[usize],
        source: BracketSource,
    ) -> (RecordingSolver, PatchBuffer, ScratchBuffers) {
        self.run_folded(
            InflowNonNegativityMethod::None,
            trial_state,
            slots,
            source,
            None,
        )
    }

    /// The pin phase on [`TRIAL_STATE`], then, with a recorder cleared of the pin's
    /// calls, the opening phase on `raw_noise`.
    fn patch_opening(
        &self,
        method: InflowNonNegativityMethod,
        slots: &[usize],
        raw_noise: &[f64],
    ) -> (RecordingSolver, PatchBuffer, ScratchBuffers) {
        self.run_folded(
            method,
            &TRIAL_STATE,
            slots,
            BracketSource::Computed,
            Some(raw_noise),
        )
    }

    fn run_folded(
        &self,
        method: InflowNonNegativityMethod,
        trial_state: &[f64],
        slots: &[usize],
        source: BracketSource,
        raw_noise: Option<&[f64]>,
    ) -> (RecordingSolver, PatchBuffer, ScratchBuffers) {
        let (solver, patch_buf, scratch, _) = self.run_openings(
            method,
            trial_state,
            slots,
            source,
            raw_noise.as_slice(),
            NoiseTermSource::OnTheFly,
            |_| {},
        );
        (solver, patch_buf, scratch)
    }

    /// Runs `prep` on a fresh recorder, patch buffer and scratch; returns the recorder and scratch.
    fn record(
        &self,
        method: InflowNonNegativityMethod,
        prep: impl FnOnce(
            &mut RecordingSolver,
            &mut PatchBuffer,
            &mut ScratchBuffers,
            &StageContext<'_>,
            &TrainingContext<'_>,
        ),
    ) -> (RecordingSolver, ScratchBuffers) {
        let fixture = self.fixture();
        let ctx = fixture.ctx();
        let training_ctx = self.prep.ctx(
            &self.state,
            &self.stochastic,
            &self.study_dims,
            &method,
            &[],
        );
        let mut scratch = Self::scratch(&ctx, &training_ctx);
        let mut solver = RecordingSolver::default();
        let mut patch_buf = PatchBuffer::new(&self.state, &[], &self.geometry, &self.couplings);
        prep(
            &mut solver,
            &mut patch_buf,
            &mut scratch,
            &ctx,
            &training_ctx,
        );
        (solver, scratch)
    }

    /// One pin phase on `trial_state`, then one opening phase per entry of `raw_noises`, each with
    /// `noise` and a recorder cleared of the earlier calls. `prepare` edits the fresh scratch before
    /// the pin.
    /// Returns the recorder and scratch as the last opening left them, and the cut-row RHS after
    /// each opening.
    fn run_openings(
        &self,
        method: InflowNonNegativityMethod,
        trial_state: &[f64],
        slots: &[usize],
        source: BracketSource,
        raw_noises: &[&[f64]],
        noise: NoiseTermSource<'_>,
        prepare: impl FnOnce(&mut ScratchBuffers),
    ) -> (RecordingSolver, PatchBuffer, ScratchBuffers, Vec<Vec<f64>>) {
        let fixture = self.fixture();
        let ctx = fixture.ctx();
        let training_ctx = self.prep.ctx(
            &self.state,
            &self.stochastic,
            &self.study_dims,
            &method,
            &[],
        );
        let mut scratch = Self::scratch(&ctx, &training_ctx);
        prepare(&mut scratch);
        let mut solver = RecordingSolver::default();
        let mut patch_buf = PatchBuffer::new(&self.state, &[], &self.geometry, &self.couplings);
        let cut_rows = LoadedCutRows {
            pool: &self.pool,
            projection: &self.prep.cut_state_layouts[0],
            slots,
            first_row: ctx.template(StageIdx(0)).num_rows,
        };
        StageSolvePrep::pin_and_bracket(
            &mut solver,
            &mut patch_buf,
            &mut scratch,
            &ctx,
            &training_ctx,
            StageIdx(0),
            StateSource(trial_state),
            &cut_rows,
            source,
        );
        let mut rhs_after_each = Vec::new();
        for raw_noise in raw_noises {
            solver = RecordingSolver::default();
            StageSolvePrep::patch_opening(
                &mut solver,
                &mut patch_buf,
                &mut scratch,
                &ctx,
                &training_ctx,
                StageIdx(0),
                raw_noise,
                &cut_rows,
                noise,
                &StageOpeningSolver::Frozen,
            );
            rhs_after_each.push(scratch.cut_rhs_buf.clone());
        }
        (solver, patch_buf, scratch, rhs_after_each)
    }
}

const TRIAL_STATE: [f64; 3] = [42.0, 5.0, 7.0];

#[test]
fn pin_folded_pins_the_columned_regions_and_skips_the_compact_lags() {
    let case = CompactCase::new(&[(1.0, [4.0, 2.0, 3.0])]);

    let (solver, patch_buf, _) = case.pin_and_bracket(&TRIAL_STATE, &[0], BracketSource::Computed);

    assert_eq!(solver.calls, ['c']);
    assert_eq!(
        solver.col_bounds_calls,
        [(vec![case.state.storage_in.start], vec![21.0], vec![21.0])]
    );
    assert_eq!(patch_buf.active_state_col_patch_count(), 1);
}

#[test]
fn pin_folded_brackets_each_cut_row_from_the_raw_trial_state() {
    let case = CompactCase::new(&[(1.0, [4.0, 2.0, 3.0])]);

    let (_, _, scratch) = case.pin_and_bracket(&TRIAL_STATE, &[0], BracketSource::Computed);

    let mut expected_mean = vec![0.0; 1];
    evaluate_par_conditional_mean_batch(
        case.stochastic.par(),
        0,
        &TRIAL_STATE[1..3],
        &mut expected_mean,
    );
    assert_eq!(bits(&scratch.par_mean_buf), bits(&expected_mean));
    assert_eq!(scratch.cut_bracket_buf.len(), 1);
    assert_eq!(scratch.cut_bracket_buf[0].to_bits(), 216.0_f64.to_bits());
}

#[test]
fn pin_folded_reads_the_cut_rows_in_slot_list_order() {
    let case = CompactCase::new(&[
        (1.0, [4.0, 2.0, 3.0]),
        (10.0, [1.0, 1.5, 2.0]),
        (-3.0, [0.5, 3.0, 1.0]),
    ]);
    let slots = [2, 0];

    let (_, _, scratch) = case.pin_and_bracket(&TRIAL_STATE, &slots, BracketSource::Computed);

    assert_eq!(scratch.cut_bracket_buf.len(), slots.len());
    for (k, &slot) in slots.iter().enumerate() {
        let expected = cut_bracket(
            case.pool.intercept(slot),
            case.pool.coefficient_row(slot),
            case.prep.cut_state_layouts[0].lag_point(),
            &TRIAL_STATE[1..3],
            &scratch.par_mean_buf,
        );
        assert_eq!(scratch.cut_bracket_buf[k].to_bits(), expected.to_bits());
    }
    assert_ne!(
        scratch.cut_bracket_buf[0].to_bits(),
        scratch.cut_bracket_buf[1].to_bits()
    );
}

const SLAB_SLOTS: [usize; 3] = [0, 1, 2];

fn slab_case() -> CompactCase {
    CompactCase::new(&[
        (1.0, [4.0, 2.0, 3.0]),
        (10.0, [1.0, 1.5, 2.0]),
        (-3.0, [0.5, 3.0, 1.0]),
    ])
}

fn differ_from_the_intercepts(case: &CompactCase, brackets: &[f64]) -> bool {
    SLAB_SLOTS
        .iter()
        .zip(brackets)
        .all(|(&slot, b)| b.to_bits() != case.pool.intercept(slot).to_bits())
}

fn computed_brackets(case: &CompactCase) -> Vec<f64> {
    let (_, _, scratch) = case.pin_and_bracket(&TRIAL_STATE, &SLAB_SLOTS, BracketSource::Computed);
    scratch.cut_bracket_buf
}

fn slab_reference(case: &CompactCase) -> Vec<f64> {
    let reference = computed_brackets(case);
    assert!(
        differ_from_the_intercepts(case, &reference),
        "the reference brackets {reference:?} must not equal the bare intercepts"
    );
    reference
}

#[test]
fn pin_folded_slab_reference_brackets_carry_the_trial_state() {
    let case = slab_case();

    let reference = slab_reference(&case);

    assert_eq!(bits(&reference), bits(&[216.0, 170.0, 302.0]));
}

#[test]
fn pin_folded_slab_power_check_fails_without_cut_coefficients() {
    let case = CompactCase::new(&[
        (1.0, [0.0, 0.0, 0.0]),
        (10.0, [0.0, 0.0, 0.0]),
        (-3.0, [0.0, 0.0, 0.0]),
    ]);

    let reference = computed_brackets(&case);

    assert_eq!(bits(&reference), bits(&[1.0, 10.0, -3.0]));
    assert!(!differ_from_the_intercepts(&case, &reference));
}

#[test]
fn pin_folded_slab_hit_equals_the_computed_brackets_and_does_not_count() {
    let case = slab_case();
    let reference = slab_reference(&case);

    let (_, _, scratch) = case.pin_and_bracket(
        &TRIAL_STATE,
        &SLAB_SLOTS,
        BracketSource::Slab(Some(&reference[..1])),
    );

    assert_eq!(bits(&scratch.cut_bracket_buf), bits(&reference));
    assert_eq!(scratch.slab_misses, 0);
}

#[test]
fn pin_folded_slab_reads_the_prefix_and_computes_the_delta() {
    let case = slab_case();
    let reference = slab_reference(&case);
    let perturbed: Vec<f64> = reference.iter().map(|b| b + 1000.0).collect();

    let (_, _, scratch) = case.pin_and_bracket(
        &TRIAL_STATE,
        &SLAB_SLOTS,
        BracketSource::Slab(Some(&perturbed[..1])),
    );

    assert_eq!(
        bits(&scratch.cut_bracket_buf),
        bits(&[perturbed[0], reference[1], reference[2]])
    );
    assert_eq!(scratch.slab_misses, 0);
}

#[test]
fn pin_folded_slab_full_prefix_computes_nothing() {
    let case = slab_case();
    let reference = slab_reference(&case);
    let perturbed: Vec<f64> = reference.iter().map(|b| b + 1000.0).collect();

    let (_, _, scratch) = case.pin_and_bracket(
        &TRIAL_STATE,
        &SLAB_SLOTS,
        BracketSource::Slab(Some(&reference)),
    );
    assert_eq!(bits(&scratch.cut_bracket_buf), bits(&reference));

    let (_, _, scratch) = case.pin_and_bracket(
        &TRIAL_STATE,
        &SLAB_SLOTS,
        BracketSource::Slab(Some(&perturbed)),
    );
    assert_eq!(bits(&scratch.cut_bracket_buf), bits(&perturbed));
    assert_eq!(scratch.slab_misses, 0);
}

#[test]
fn pin_folded_slab_miss_equals_the_computed_brackets_and_counts() {
    let case = slab_case();
    let reference = slab_reference(&case);

    let (_, _, scratch) =
        case.pin_and_bracket(&TRIAL_STATE, &SLAB_SLOTS, BracketSource::Slab(None));

    assert_eq!(bits(&scratch.cut_bracket_buf), bits(&reference));
    assert_eq!(scratch.slab_misses, 1);
}

#[test]
fn open_folded_pushes_the_shifted_rows_then_the_column_bounds_then_the_cut_range() {
    let case = CompactCase::new(&[(1.0, [4.0, 2.0, 3.0])]);

    let (solver, _, _) = case.patch_opening(InflowNonNegativityMethod::Truncation, &[0], &[0.5]);

    assert_eq!(solver.calls, ['r', 'c', 'R']);
    assert_eq!(
        solver.row_bounds_calls,
        [(vec![0], vec![460.0], vec![460.0])]
    );
    assert_eq!(
        solver.col_bounds_calls,
        [(vec![case.state.theta + 1], vec![0.0], vec![0.0])]
    );
    assert_eq!(solver.row_range_calls, [(1, vec![246.0])]);
}

#[test]
fn open_folded_writes_cut_rhs_as_bracket_plus_noise_term() {
    let one_cut = CompactCase::new(&[(1.0, [4.0, 2.0, 3.0])]);

    let (_, _, scratch) = one_cut.patch_opening(InflowNonNegativityMethod::None, &[0], &[0.5]);

    assert_eq!(
        bits(&scratch.cut_rhs_buf),
        bits(&[246.0]),
        "cut_rhs_buf is {:?}, expected [246.0]",
        scratch.cut_rhs_buf
    );

    let case = CompactCase::new(&[
        (1.0, [4.0, 2.0, 3.0]),
        (10.0, [1.0, 1.5, 2.0]),
        (-3.0, [0.5, 3.0, 1.0]),
    ]);
    let slots = [2, 0];

    let (solver, _, scratch) = case.patch_opening(InflowNonNegativityMethod::None, &slots, &[0.5]);

    let mut noise_term = vec![0.0; 1];
    evaluate_par_noise_term_batch(case.stochastic.par(), 0, &[0.5], &mut noise_term);
    let mut expected = vec![0.0; slots.len()];
    cut_rhs_into(
        &case.pool,
        &slots,
        case.prep.cut_state_layouts[0].lag_point(),
        &noise_term,
        &scratch.cut_bracket_buf,
        &mut expected,
    );
    assert_eq!(bits(&scratch.cut_rhs_buf), bits(&expected));
    assert_eq!(solver.row_range_calls.len(), 1);
    assert_eq!(solver.row_range_calls[0].0, case.templates[0].num_rows);
    assert_eq!(bits(&solver.row_range_calls[0].1), bits(&expected));
}

const STAGE_NOISE_CUTS: [(f64, [f64; 3]); 3] = [
    (1.0, [4.0, 2.0, 3.0]),
    (10.0, [1.0, 1.5, 2.0]),
    (-3.0, [0.5, 3.0, 1.0]),
];
const STAGE_NOISE_SLOTS: [usize; 2] = [2, 0];

/// The stage-wide σ·η of the one hydro at `eta` (0.0 without a PAR model) and the W of each slot.
fn stage_noise_values(case: &CompactCase, slots: &[usize], eta: f64) -> (Vec<f64>, Vec<f64>) {
    let par = case.stochastic.par();
    let mut noise_term = vec![0.0; 1];
    if par.n_stages() > 0 {
        evaluate_par_noise_term_batch(par, 0, &[eta], &mut noise_term);
    }
    let lag_point = case.prep.cut_state_layouts[0].lag_point();
    let cut_noise = slots
        .iter()
        .map(|&slot| cut_noise_term(case.pool.coefficient_row(slot), lag_point, &noise_term))
        .collect();
    (noise_term, cut_noise)
}

/// Whether `Stage` and `OnTheFly` could differ at `eta`: σ·η and every W are nonzero and the
/// realized inflow is negative, so the truncation bound is active.
fn stage_noise_has_power(case: &CompactCase, slots: &[usize], eta: f64) -> bool {
    let (noise_term, cut_noise) = stage_noise_values(case, slots, eta);
    let (_, _, scratch, _) = case.run_openings(
        InflowNonNegativityMethod::Truncation,
        &TRIAL_STATE,
        slots,
        BracketSource::Computed,
        &[&[eta]],
        NoiseTermSource::OnTheFly,
        |_| {},
    );
    noise_term.iter().all(|&w| w != 0.0)
        && cut_noise.iter().all(|&w| w != 0.0)
        && scratch.realized_inflow_buf.iter().all(|&z| z < 0.0)
}

fn bounds_call_bits(
    calls: &[(Vec<usize>, Vec<f64>, Vec<f64>)],
) -> Vec<(Vec<usize>, Vec<u64>, Vec<u64>)> {
    calls
        .iter()
        .map(|(indices, lower, upper)| (indices.clone(), bits(lower), bits(upper)))
        .collect()
}

fn assert_stage_noise_matches_on_the_fly(
    case: &CompactCase,
    slots: &[usize],
    method: InflowNonNegativityMethod,
    eta: f64,
) {
    let (noise_term, cut_noise) = stage_noise_values(case, slots, eta);
    let (on_the_fly_solver, _, on_the_fly, _) = case.run_openings(
        method,
        &TRIAL_STATE,
        slots,
        BracketSource::Computed,
        &[&[eta]],
        NoiseTermSource::OnTheFly,
        |_| {},
    );
    let (stage_solver, _, stage, _) = case.run_openings(
        method,
        &TRIAL_STATE,
        slots,
        BracketSource::Computed,
        &[&[eta]],
        NoiseTermSource::Stage {
            noise_term: &noise_term,
            cut_noise: &cut_noise,
        },
        |scratch| scratch.par_noise_term_buf.fill(f64::NAN),
    );

    let label = format!("{method:?}, eta {eta}");
    assert_eq!(
        bits(&stage.realized_inflow_buf),
        bits(&on_the_fly.realized_inflow_buf),
        "{label}: realized inflow"
    );
    assert_eq!(
        bits(&stage.inflow_slack_lower_buf),
        bits(&on_the_fly.inflow_slack_lower_buf),
        "{label}: slack lower"
    );
    assert_eq!(
        bits(&stage.inflow_slack_upper_buf),
        bits(&on_the_fly.inflow_slack_upper_buf),
        "{label}: slack upper"
    );
    assert_eq!(
        stage_solver.calls, on_the_fly_solver.calls,
        "{label}: calls"
    );
    assert_eq!(
        bounds_call_bits(&stage_solver.row_bounds_calls),
        bounds_call_bits(&on_the_fly_solver.row_bounds_calls),
        "{label}: row bounds"
    );
    assert_eq!(
        bounds_call_bits(&stage_solver.col_bounds_calls),
        bounds_call_bits(&on_the_fly_solver.col_bounds_calls),
        "{label}: column bounds"
    );
    assert_eq!(stage_solver.row_range_calls.len(), 1, "{label}: cut range");
    let (first_row, rhs) = &stage_solver.row_range_calls[0];
    let (on_the_fly_first_row, on_the_fly_rhs) = &on_the_fly_solver.row_range_calls[0];
    assert_eq!(first_row, on_the_fly_first_row, "{label}: cut range start");
    assert_eq!(bits(rhs), bits(on_the_fly_rhs), "{label}: cut range");
    assert_eq!(
        bits(&stage.par_noise_term_buf),
        [f64::NAN.to_bits()],
        "{label}: the stage source leaves the scratch noise term unwritten"
    );
}

#[test]
fn open_folded_stage_noise_matches_on_the_fly_bit_for_bit() {
    let case = CompactCase::new(&STAGE_NOISE_CUTS);

    for (eta, truncation_active) in [(0.5, false), (-4.0, true)] {
        assert_eq!(
            stage_noise_has_power(&case, &STAGE_NOISE_SLOTS, eta),
            truncation_active,
            "eta {eta}"
        );
        assert_stage_noise_matches_on_the_fly(
            &case,
            &STAGE_NOISE_SLOTS,
            InflowNonNegativityMethod::Truncation,
            eta,
        );
    }
}

#[test]
fn open_folded_stage_noise_without_a_par_model_matches_on_the_fly() {
    let case = CompactCase::with_stochastic(stochastic_context(None, false), &STAGE_NOISE_CUTS);
    assert_eq!(case.stochastic.par().n_stages(), 0);

    assert_stage_noise_matches_on_the_fly(
        &case,
        &STAGE_NOISE_SLOTS,
        InflowNonNegativityMethod::Truncation,
        0.5,
    );
}

#[test]
fn open_folded_stage_noise_power_check_fails_at_zero_noise() {
    let case = CompactCase::new(&STAGE_NOISE_CUTS);

    assert!(stage_noise_has_power(&case, &STAGE_NOISE_SLOTS, -4.0));
    assert!(!stage_noise_has_power(&case, &STAGE_NOISE_SLOTS, 0.0));
}

#[test]
fn one_pin_serves_every_opening_after_it() {
    let case = CompactCase::new(&[(1.0, [4.0, 2.0, 3.0]), (10.0, [1.0, 1.5, 2.0])]);
    let slots = [1, 0];
    let etas = [-0.5, 1.5];
    let lag_point = case.prep.cut_state_layouts[0].lag_point();

    let (_, _, pinned) = case.pin_and_bracket(&TRIAL_STATE, &slots, BracketSource::Computed);
    let (_, _, scratch, rhs) = case.run_openings(
        InflowNonNegativityMethod::None,
        &TRIAL_STATE,
        &slots,
        BracketSource::Computed,
        &[&[etas[0]], &[etas[1]]],
        NoiseTermSource::OnTheFly,
        |_| {},
    );

    assert_eq!(
        bits(&scratch.cut_bracket_buf),
        bits(&pinned.cut_bracket_buf),
        "B moved after an opening"
    );
    let noise_terms: Vec<Vec<f64>> = etas
        .iter()
        .map(|&eta| {
            let mut term = vec![0.0; 1];
            evaluate_par_noise_term_batch(case.stochastic.par(), 0, &[eta], &mut term);
            term
        })
        .collect();
    let w: Vec<Vec<f64>> = noise_terms
        .iter()
        .map(|term| {
            slots
                .iter()
                .map(|&slot| cut_noise_term(case.pool.coefficient_row(slot), lag_point, term))
                .collect()
        })
        .collect();
    for (k, rhs_k) in rhs.iter().enumerate() {
        let expected: Vec<f64> = pinned
            .cut_bracket_buf
            .iter()
            .zip(&w[k])
            .map(|(b, w)| b + w)
            .collect();
        assert_eq!(bits(rhs_k), bits(&expected), "opening {k}: RHS is B + W");
    }
    assert_ne!(
        bits(&w[0]),
        bits(&w[1]),
        "the two etas must give different W"
    );
    assert_ne!(
        bits(&rhs[0]),
        bits(&rhs[1]),
        "the two etas must give different RHS"
    );
}

#[test]
fn without_a_par_model_p_w_and_the_realized_inflow_are_positive_zero() {
    let case = CompactCase::with_stochastic(
        stochastic_context(None, false),
        &[(1.0, [4.0, 2.0, 3.0]), (-3.0, [0.5, -3.0, 1.0])],
    );
    assert_eq!(case.stochastic.par().n_stages(), 0);
    let slots = [0, 1];

    let (_, _, scratch, rhs) = case.run_openings(
        InflowNonNegativityMethod::None,
        &TRIAL_STATE,
        &slots,
        BracketSource::Computed,
        &[&[0.5]],
        NoiseTermSource::OnTheFly,
        |scratch| {
            scratch.par_mean_buf.fill(f64::NAN);
            scratch.par_noise_term_buf.fill(f64::NAN);
            scratch.realized_inflow_buf.fill(f64::NAN);
        },
    );

    let positive_zero = 0.0_f64.to_bits();
    assert_eq!(bits(&scratch.par_mean_buf), [positive_zero], "p");
    assert_eq!(
        bits(&scratch.par_noise_term_buf),
        [positive_zero],
        "sigma * eta"
    );
    assert_eq!(bits(&scratch.realized_inflow_buf), [positive_zero], "z-hat");
    let lag_point = case.prep.cut_state_layouts[0].lag_point();
    for (k, &slot) in slots.iter().enumerate() {
        let w = cut_noise_term(
            case.pool.coefficient_row(slot),
            lag_point,
            &scratch.par_noise_term_buf,
        );
        assert_eq!(w.to_bits(), positive_zero, "cut {slot}: W");
        assert_eq!(
            rhs[0][k].to_bits(),
            scratch.cut_bracket_buf[k].to_bits(),
            "cut {slot}: RHS is B"
        );
    }
}

#[test]
fn open_folded_shifts_the_coupled_rows_by_the_realized_inflow() {
    let case = CompactCase::new(&[(1.0, [4.0, 2.0, 3.0])]);

    let (solver, _, scratch) = case.patch_opening(InflowNonNegativityMethod::None, &[0], &[0.5]);

    assert_eq!(bits(&scratch.realized_inflow_buf), bits(&[115.0]));
    assert_eq!(solver.calls, ['r', 'R']);
    let mut reference = PatchBuffer::new(&case.state, &[], &case.geometry, &case.couplings);
    reference.fill_inflow_shift_patches(&case.couplings[0], &[115.0], &case.templates[0].row_scale);
    let pc = reference.forward_patch_count();
    assert_eq!(
        solver.row_bounds_calls,
        [(
            reference.indices[..pc].to_vec(),
            reference.lower[..pc].to_vec(),
            reference.upper[..pc].to_vec()
        )]
    );
}

#[test]
fn slack_bounds_follow_the_method_and_write_positive_zero_for_a_non_negative_inflow() {
    let case = CompactCase::new(&[(1.0, [4.0, 2.0, 3.0])]);
    let positive_zero = 0.0_f64.to_bits();

    for method in [
        InflowNonNegativityMethod::Truncation,
        InflowNonNegativityMethod::TruncationWithPenalty,
    ] {
        let (_, _, negative) = case.patch_opening(method, &[], &[-5.0]);
        let (lower, upper) = match method {
            InflowNonNegativityMethod::Truncation => ([50.0], [50.0]),
            _ => ([0.0], [50.0]),
        };
        assert_eq!(negative.inflow_slack_lower_buf, lower, "{method:?}: lower");
        assert_eq!(negative.inflow_slack_upper_buf, upper, "{method:?}: upper");

        let (_, _, positive) = case.patch_opening(method, &[], &[0.5]);
        assert_eq!(
            bits(&positive.inflow_slack_lower_buf),
            [positive_zero],
            "{method:?}: lower"
        );
        assert_eq!(
            bits(&positive.inflow_slack_upper_buf),
            [positive_zero],
            "{method:?}: upper"
        );
    }
}

#[test]
fn open_folded_with_no_cut_rows_pushes_an_empty_range() {
    let case = CompactCase::new(&[(1.0, [4.0, 2.0, 3.0])]);

    let (solver, _, scratch) = case.patch_opening(InflowNonNegativityMethod::None, &[], &[0.5]);

    assert_eq!(solver.calls.last(), Some(&'R'));
    assert_eq!(solver.row_range_calls, [(1, vec![])]);
    assert!(scratch.cut_rhs_buf.is_empty());
}

fn range_call_bits(solver: &RecordingSolver) -> Vec<(usize, Vec<u64>)> {
    solver
        .row_range_calls
        .iter()
        .map(|(first_row, rhs)| (*first_row, bits(rhs)))
        .collect()
}

#[test]
fn prepare_folded_single_makes_the_pin_then_open_calls_bit_for_bit() {
    let case = CompactCase::new(&STAGE_NOISE_CUTS);
    let method = InflowNonNegativityMethod::Truncation;
    let raw_noise = [0.5];
    let projection = &case.prep.cut_state_layouts[0];

    let (single, single_scratch) =
        case.record(method, |solver, patch_buf, scratch, ctx, training_ctx| {
            StageSolvePrep::prepare_single(
                solver,
                patch_buf,
                scratch,
                ctx,
                training_ctx,
                StageIdx(0),
                &SingleSolvePrepParams {
                    state_source: StateSource(&TRIAL_STATE),
                    raw_noise: &raw_noise,
                    pool: &case.pool,
                    projection,
                    opening_solver: &StageOpeningSolver::Frozen,
                },
                STAGE_NOISE_SLOTS.iter().copied(),
            );
        });
    let (two_step, _) = case.record(method, |solver, patch_buf, scratch, ctx, training_ctx| {
        let rows = LoadedCutRows {
            pool: &case.pool,
            projection,
            slots: &STAGE_NOISE_SLOTS,
            first_row: ctx.template(StageIdx(0)).num_rows,
        };
        StageSolvePrep::pin_and_bracket(
            solver,
            patch_buf,
            scratch,
            ctx,
            training_ctx,
            StageIdx(0),
            StateSource(&TRIAL_STATE),
            &rows,
            BracketSource::Computed,
        );
        StageSolvePrep::patch_opening(
            solver,
            patch_buf,
            scratch,
            ctx,
            training_ctx,
            StageIdx(0),
            &raw_noise,
            &rows,
            NoiseTermSource::OnTheFly,
            &StageOpeningSolver::Frozen,
        );
    });

    assert_eq!(single.calls, ['c', 'r', 'c', 'R']);
    assert!(single.row_range_calls[0].1.iter().any(|&rhs| rhs != 0.0));
    assert_eq!(single.calls, two_step.calls);
    assert_eq!(
        bounds_call_bits(&single.col_bounds_calls),
        bounds_call_bits(&two_step.col_bounds_calls)
    );
    assert_eq!(
        bounds_call_bits(&single.row_bounds_calls),
        bounds_call_bits(&two_step.row_bounds_calls)
    );
    assert_eq!(range_call_bits(&single), range_call_bits(&two_step));
    assert_eq!(single_scratch.cut_slots_buf, STAGE_NOISE_SLOTS);
}

#[test]
fn the_lazy_opening_solver_leaves_the_cut_range_to_the_lazy_solve() {
    let case = CompactCase::new(&STAGE_NOISE_CUTS);
    let raw_noise = [0.5];
    let open_with = |opening_solver: &StageOpeningSolver| {
        case.record(
            InflowNonNegativityMethod::None,
            |solver, patch_buf, scratch, ctx, training_ctx| {
                let rows = LoadedCutRows {
                    pool: &case.pool,
                    projection: &case.prep.cut_state_layouts[0],
                    slots: &STAGE_NOISE_SLOTS,
                    first_row: ctx.template(StageIdx(0)).num_rows,
                };
                StageSolvePrep::pin_and_bracket(
                    solver,
                    patch_buf,
                    scratch,
                    ctx,
                    training_ctx,
                    StageIdx(0),
                    StateSource(&TRIAL_STATE),
                    &rows,
                    BracketSource::Computed,
                );
                StageSolvePrep::patch_opening(
                    solver,
                    patch_buf,
                    scratch,
                    ctx,
                    training_ctx,
                    StageIdx(0),
                    &raw_noise,
                    &rows,
                    NoiseTermSource::OnTheFly,
                    opening_solver,
                );
            },
        )
    };

    let (frozen, _) = open_with(&StageOpeningSolver::Frozen);
    let (lazy, lazy_scratch) = open_with(&StageOpeningSolver::Lazy(DcsParams {
        candidate_recency: None,
        seed_window: 2,
        max_added_per_round: 10,
        violation_tolerance: 1e-9,
        start_iteration: 1,
        max_inner_iterations: 50,
    }));

    assert_eq!(frozen.calls, ['c', 'r', 'R']);
    let [(_, pushed)] = frozen.row_range_calls.as_slice() else {
        unreachable!("the frozen run pushes one cut range");
    };
    assert!(
        pushed.iter().any(|&rhs| rhs != 0.0),
        "the pushed range is all zero"
    );
    assert_eq!(lazy.calls, ['c', 'r']);
    assert!(lazy.row_range_calls.is_empty());
    assert_eq!(bits(&lazy_scratch.cut_rhs_buf), bits(pushed));
}
