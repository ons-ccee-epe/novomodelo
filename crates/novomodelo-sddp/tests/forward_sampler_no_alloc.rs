//! Pins the out-of-sample forward draw to zero heap allocation, once its
//! per-iteration noise tables are built and its buffers have warmed up, for
//! the Sobol, Halton and Latin-hypercube point methods and for a correlation
//! group wider than `apply_groups_for_class`'s stack-allocated fast path.
//!
//! It also pins one folded stage solve and its cut, single-threaded after
//! warm-up, to zero Rust heap allocation. Allocations inside the linked C++
//! solver are not counted.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp
)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::collections::BTreeMap;

use novomodelo_core::{
    DeficitSegment, EntityId, SystemBuilder,
    entities::hydro::{Hydro, HydroPenalties},
    scenario::{
        CorrelationEntity, CorrelationGroup, CorrelationModel, CorrelationProfile, InflowModel,
        SamplingScheme,
    },
    temporal::{Block, NoiseMethod, ScenarioSourceConfig, Stage},
};
use novomodelo_stochastic::{
    ForwardNoiseTables, ForwardSamplerConfig, SampleRequest, build_forward_sampler,
    context::{ClassSchemes, OpeningTreeInputs, StochasticContext, build_stochastic_context},
};

mod common;
use common::builders::{BusSpec, HydroSpec, StageSpec, make_bus, make_hydro, make_stage};

struct CountingAllocator;

thread_local! {
    // Per thread: the test harness allocates on its own thread while a test runs.
    static ALLOC_COUNT: Cell<usize> = const { Cell::new(0) };
}

// SAFETY: every method is a verbatim forward to `System`, which upholds
// `GlobalAlloc`'s contract on its own; this wrapper only adds a counter
// increment around the call.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        count_alloc();
        // SAFETY: `layout` is forwarded unchanged to `System::alloc`.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr`/`layout` are forwarded unchanged to `System::dealloc`.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        count_alloc();
        // SAFETY: `ptr`/`layout`/`new_size` are forwarded unchanged to
        // `System::realloc`.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

fn count_alloc() {
    ALLOC_COUNT.with(|count| count.set(count.get() + 1));
}

fn alloc_count() -> usize {
    ALLOC_COUNT.with(Cell::get)
}

fn reset_alloc_count() {
    ALLOC_COUNT.with(|count| count.set(0));
}

fn make_bus_spec(id: i32) -> BusSpec {
    BusSpec {
        name: format!("Bus{id}"),
        deficit_segments: vec![DeficitSegment {
            depth_mw: None,
            cost_per_mwh: 1000.0,
        }],
        ..Default::default()
    }
}

fn make_hydro_spec(id: i32) -> HydroSpec {
    HydroSpec {
        name: format!("H{id}"),
        max_storage_hm3: 100.0,
        max_turbined_m3s: 100.0,
        max_generation_mw: 100.0,
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
        ..Default::default()
    }
}

fn make_stage_spec(bf: usize, method: NoiseMethod) -> StageSpec {
    StageSpec {
        season_id: Some(0),
        blocks: vec![Block {
            index: 0,
            name: "SINGLE".to_string(),
            duration_hours: 744.0,
        }],
        scenario_config: ScenarioSourceConfig {
            branching_factor: bf,
            noise_method: method,
        },
        ..Default::default()
    }
}

fn make_inflow_model(hydro_id: i32, stage_id: i32) -> InflowModel {
    InflowModel {
        hydro_id: EntityId(hydro_id),
        stage_id,
        mean_m3s: 100.0,
        std_m3s: 30.0,
        ar_coefficients: vec![],
        residual_std_ratio: 1.0,
        annual: None,
    }
}

fn correlated_correlation(ids: &[i32], rho: f64) -> CorrelationModel {
    let n = ids.len();
    let matrix: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..n).map(|j| if i == j { 1.0 } else { rho }).collect())
        .collect();
    let mut profiles = BTreeMap::new();
    profiles.insert(
        "default".to_string(),
        CorrelationProfile {
            groups: vec![CorrelationGroup {
                name: "g1".to_string(),
                entities: ids
                    .iter()
                    .map(|&id| CorrelationEntity {
                        entity_type: "inflow".to_string(),
                        id: EntityId(id),
                    })
                    .collect(),
                matrix,
            }],
        },
    );
    CorrelationModel {
        method: "spectral".to_string(),
        profiles,
        schedule: vec![],
    }
}

fn build_test_system(
    methods: [NoiseMethod; 3],
    correlation: CorrelationModel,
) -> novomodelo_core::System {
    let hydros: Vec<Hydro> = (1..=70)
        .map(|id| make_hydro(EntityId(id), make_hydro_spec(id)))
        .collect();
    let stages = vec![
        make_stage(0, make_stage_spec(5, methods[0])),
        make_stage(1, make_stage_spec(5, methods[1])),
        make_stage(2, make_stage_spec(5, methods[2])),
    ];
    let inflow_models: Vec<InflowModel> = (1..=70)
        .flat_map(|hydro_id| (0..3).map(move |stage_id| make_inflow_model(hydro_id, stage_id)))
        .collect();
    SystemBuilder::new()
        .buses(vec![make_bus(EntityId(0), make_bus_spec(0))])
        .hydros(hydros)
        .stages(stages)
        .inflow_models(inflow_models)
        .correlation(correlation)
        .build()
        .unwrap()
}

fn build_test_ctx(
    system: &novomodelo_core::System,
    forward_seed: Option<u64>,
) -> StochasticContext {
    build_stochastic_context(
        system,
        42,
        forward_seed,
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

fn stages_from_system(system: &novomodelo_core::System) -> Vec<Stage> {
    system
        .stages()
        .iter()
        .filter(|s| s.id >= 0)
        .cloned()
        .collect()
}

fn make_sampler_config<'a>(
    scheme: SamplingScheme,
    ctx: &'a StochasticContext,
    stages: &'a [Stage],
) -> ForwardSamplerConfig<'a> {
    ForwardSamplerConfig {
        class_schemes: ClassSchemes {
            inflow: Some(scheme),
            load: Some(scheme),
            ncs: Some(scheme),
        },
        ctx,
        forward_seed: ctx.forward_seed(),
        stages,
        historical_library: None,
        external_inflow_library: None,
        external_load_library: None,
        external_ncs_library: None,
    }
}

#[test]
fn out_of_sample_forward_draw_allocates_nothing() {
    let entity_ids: Vec<i32> = (1..=70).collect();
    let system = build_test_system(
        [
            NoiseMethod::QmcSobol,
            NoiseMethod::QmcHalton,
            NoiseMethod::Lhs,
        ],
        correlated_correlation(&entity_ids, 0.4),
    );
    let ctx = build_test_ctx(&system, Some(99));
    let stages = stages_from_system(&system);
    let sampler = build_forward_sampler(make_sampler_config(
        SamplingScheme::OutOfSample,
        &ctx,
        &stages,
    ))
    .unwrap();
    let dim = ctx.dim();
    assert_eq!(
        dim, 70,
        "fixture must supply exactly 70 correlated entities to exercise \
         apply_groups_for_class's scratch (non-stack) path"
    );

    let total_scenarios: u32 = 8;
    let mut tables = ForwardNoiseTables::default();
    sampler
        .rebuild_noise_tables(0, total_scenarios, &[], &mut tables)
        .expect("70-dim Sobol stays within the Sobol dimension cap");

    let mut noise_buf = vec![0.0f64; dim];
    let mut corr_scratch = vec![0.0f64; 2 * dim];

    let mut sample_stage = |stage_idx: usize, scenario: u32| {
        let stage_u32 = u32::try_from(stage_idx).unwrap();
        sampler
            .sample(SampleRequest {
                iteration: 0,
                scenario,
                stage: stage_u32,
                stage_idx,
                noise_buf: &mut noise_buf,
                corr_scratch: &mut corr_scratch,
                total_scenarios,
                noise_group_id: stage_u32,
                node_opening_offset: 0,
                node_opening_len: 0,
                pinned_scenario: None,
                tables: &tables,
            })
            .unwrap();
    };

    // Warm up every stage once so any first-draw buffer growth happens before
    // the counter resets; a warm-up allocation is one-time, not per draw.
    for stage_idx in 0..3_usize {
        sample_stage(stage_idx, 0);
    }

    reset_alloc_count();

    for stage_idx in 0..3_usize {
        for scenario in 0..total_scenarios {
            sample_stage(stage_idx, scenario);
        }
    }

    let observed = alloc_count();
    assert_eq!(
        observed, 0,
        "out-of-sample forward draw allocated {observed} times across the measured draws"
    );
}

mod folded_solve {
    use novomodelo_sddp::cut::chain_rule::{
        AggregateScratch, ChainRuleInputs, ChildOpenings, OpeningDuals, aggregate_first_cut_into,
    };
    use novomodelo_sddp::indexer::ChainRuleSlots;
    use novomodelo_sddp::setup::{NodePos, StageIdx};
    use novomodelo_sddp::test_support::{
        FoldedProbeCut, FoldedProbeCuts, FoldedProbeOpening, bits, cut_rows_for_slots_into,
        empty_row_batch, folded_opening_cut_for_probe, node_opening_noise,
    };
    use novomodelo_sddp::{SddpError, StudySetup};
    use novomodelo_solver::{ActiveSolver, SolverInterface};

    use super::{alloc_count, reset_alloc_count};
    use crate::common::expanded_solve::{CutRowSet, stored_incoming_states};
    use crate::common::in_code_studies::{FphaLagOptions, fpha_lag_study_with};
    use crate::common::stored_cut_oracles::apply_oracle_config;
    use crate::common::{
        Emission, StubComm, build_setup_in_code_with_models_as, train_storing_trial_states,
    };

    const STAGE: usize = 1;
    const RESIDENT_POOL: usize = 1;
    const EXTRACTED_POOL: usize = 0;
    const OPENINGS: usize = 3;

    fn trained_lag_study() -> StudySetup {
        let (system, mut config, hydro_models) = fpha_lag_study_with(FphaLagOptions {
            openings: OPENINGS,
            ..FphaLagOptions::default()
        });
        apply_oracle_config(&mut config);
        let mut setup =
            build_setup_in_code_with_models_as(system, &config, hydro_models, Emission::Compact);
        train_storing_trial_states("folded no-alloc lag study", &mut setup);
        setup
    }

    #[test]
    fn a_folded_solve_and_its_cut_allocate_nothing_after_warm_up() {
        let setup = trained_lag_study();
        let ctx = setup.stage_ctx();
        let training_ctx = setup.training_ctx();
        let layouts = &training_ctx.cut_state_layouts;
        let pool = &setup.fcf.pools[RESIDENT_POOL];
        let template = &setup.inputs.stage_data.stage_templates.templates[STAGE];

        let slots = CutRowSet::active(&setup.fcf, RESIDENT_POOL, EXTRACTED_POOL).slots;
        let mut cut_rows = empty_row_batch();
        cut_rows_for_slots_into(
            &mut cut_rows,
            pool,
            &slots,
            setup.stage_state(),
            &layouts[RESIDENT_POOL],
            &template.col_scale,
        );
        let mut workspaces = setup
            .create_workspace_pool(&StubComm, 1, ActiveSolver::new)
            .expect("the workspace pool must build");
        let ws = &mut workspaces.workspaces[0];
        ws.solver.load_model(template);
        ws.solver.add_rows(&cut_rows);

        let stored =
            stored_incoming_states(&setup, STAGE).expect("pool 0 projects every state dimension");
        assert!(
            stored.len() >= 2,
            "pool 0 stores {} trial states, two are needed",
            stored.len()
        );
        let states: [&[f64]; 2] = [&stored[0], &stored[1]];
        let noise: [Vec<f64>; OPENINGS] =
            std::array::from_fn(|w| node_opening_noise(&setup, NodePos(STAGE), w));
        let mut outs: [FoldedProbeCut; OPENINGS] =
            std::array::from_fn(|_| FoldedProbeCut::new(&training_ctx));

        let chain_slots = ChainRuleSlots::compose(
            training_ctx.state,
            &layouts[RESIDENT_POOL],
            &layouts[EXTRACTED_POOL],
        );
        let mut scratch = AggregateScratch::new(
            training_ctx.state.hydro_count,
            training_ctx.state.max_par_order,
        );
        scratch.reserve(pool.capacity, OPENINGS * slots.len());
        let mut coefficients = vec![0.0; layouts[EXTRACTED_POOL].n_slots()];
        let weights = [1.0 / 3.0; OPENINGS];

        let mut round =
            |outs: &mut [FoldedProbeCut; OPENINGS], x_hat: &[f64]| -> Result<f64, SddpError> {
                for (out, raw_noise) in outs.iter_mut().zip(&noise) {
                    folded_opening_cut_for_probe(
                        ws,
                        &ctx,
                        &training_ctx,
                        &FoldedProbeOpening {
                            stage: StageIdx(STAGE),
                            pinned_state: x_hat,
                            raw_noise,
                            cuts: FoldedProbeCuts {
                                pool,
                                pool_id: RESIDENT_POOL,
                                slots: &slots,
                            },
                            extracted_pool_id: EXTRACTED_POOL,
                        },
                        out,
                    )?;
                }
                let duals: [OpeningDuals<'_>; OPENINGS] = std::array::from_fn(|w| OpeningDuals {
                    objective: outs[w].objective,
                    state_duals: &outs[w].state_duals,
                    coupled_duals: &outs[w].coupled_duals,
                    cut_duals: &outs[w].cut_dual_slots,
                });
                let children = [ChildOpenings {
                    inputs: ChainRuleInputs {
                        slots: &chain_slots,
                        resident_pool: pool,
                        extracted: &layouts[EXTRACTED_POOL],
                        par: training_ctx.stochastic.par(),
                        stage: STAGE,
                    },
                    openings: &duals,
                }];
                Ok(aggregate_first_cut_into(
                    &children,
                    &weights,
                    x_hat,
                    &mut scratch,
                    &mut coefficients,
                ))
            };

        let mut warm_up = Vec::new();
        let mut cut_dual_seen = false;
        for x_hat in states {
            warm_up.push(round(&mut outs, x_hat));
            cut_dual_seen |= outs
                .iter()
                .flat_map(|out| &out.cut_dual_slots)
                .any(|&(_, y)| y != 0.0);
        }

        assert!(
            !slots.is_empty(),
            "stage {STAGE}'s LP has no cut rows, so the cut-row passes do not run"
        );
        assert!(
            layouts[RESIDENT_POOL].has_lag_slots(),
            "pool {RESIDENT_POOL} has no lag slots, so the cut-row duals are not extracted"
        );
        assert!(
            cut_dual_seen,
            "no warm-up opening has a non-zero cut-row dual, so the chain rule's cut-dual sum does not run"
        );
        assert_ne!(
            bits(states[0]),
            bits(states[1]),
            "the two pinned states are bit-identical"
        );
        for a in 0..OPENINGS {
            for b in a + 1..OPENINGS {
                assert_ne!(
                    bits(&noise[a]),
                    bits(&noise[b]),
                    "openings {a} and {b} have bit-identical noise"
                );
            }
        }
        assert!(
            warm_up.iter().all(Result::is_ok),
            "a warm-up round failed: {warm_up:?}"
        );

        reset_alloc_count();
        let measured = [
            round(&mut outs, states[0]),
            round(&mut outs, states[1]),
            round(&mut outs, states[0]),
        ];
        let observed = alloc_count();

        assert!(
            measured.iter().all(Result::is_ok),
            "a measured round failed: {measured:?}"
        );
        assert_eq!(
            observed, 0,
            "a folded solve round allocated {observed} times across the measured rounds \
             (state 0, state 1, state 0 again)"
        );
    }
}
