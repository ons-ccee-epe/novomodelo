use std::collections::HashMap;
use std::num::NonZeroU32;

use novomodelo_io::Config;
use novomodelo_io::config::SelectionMethod;
use novomodelo_sddp::dcs::build_initial_resident_set;
use novomodelo_sddp::indexer::{CutSlot, HydroSys};
use novomodelo_sddp::test_support::expander::{ExpandedStudy, expand_stage_templates};
use novomodelo_sddp::test_support::solve_sites::{CutRowShape, SolveSiteKind};
use novomodelo_sddp::{FutureCostFunction, StudySetup};
use novomodelo_solver::LpSolution;

use crate::common::expanded_solve::{
    CutRowSet, ExpandedSolve, assemble_expanded_solve, check_lifted_compact_solution, lag_dims,
    record_expanded_evidence, rounding_factor, scale_at, solve_from_mapped_basis,
    stored_incoming_states,
};
use crate::common::in_code_studies::{FphaLagOptions, fpha_lag_study_with};
use crate::common::oracle_tolerance::UnscaledTolerance;
use crate::common::power_guards::{Guard, PowerEvidence};
use crate::common::stored_cut_oracles::{apply_oracle_config, apply_oracle_config_with_selection};
use crate::common::{
    build_setup_in_code_with_models, is_chain, positive, train_storing_trial_states,
};

const OPENINGS: usize = 3;

const SCALE_GUARDS: [Guard; 5] = [
    Guard::BindingLagCut,
    Guard::NonUnitZScale,
    Guard::NonUnitLagScale,
    Guard::RescaledCoupledRowWithDual,
    Guard::NonUnitCompactCoupledRowScaleWithDual,
];

const LOWER_BOUND_MAP_UNMET_GUARDS: &[(Guard, &str)] = &[(
    Guard::RescaledCoupledRowWithDual,
    "stage 0 couples only parallel-stage water-balance rows whose compact and expanded row \
     scales are bit-identical, and the fixture's only rescaled coupled row is at stage 1",
)];

const STORAGE_ONLY_RESIDENT_UNMET_GUARDS: &[(Guard, &str)] = &[(
    Guard::BindingLagCut,
    "the resident pool has no lag slot, so no cut row carries a lag coefficient",
)];

struct BetaGap {
    at: String,
    slot: usize,
    gap: f64,
    bound: f64,
}

struct KktRun {
    evidence: PowerEvidence,
    solves: usize,
    slots_checked: usize,
    binding_cut_rows_with_dual: usize,
    nonzero_betas: usize,
    lifted_rows: usize,
    lifted_cut_rows: usize,
    lifted_binding_rows_with_dual: usize,
    lifted_slack_rows: usize,
    worst: Option<BetaGap>,
}

impl KktRun {
    fn new(compact: &StudySetup) -> Self {
        let state = compact.stage_state();
        Self {
            evidence: PowerEvidence::new(state.hydro_count, state.max_par_order),
            solves: 0,
            slots_checked: 0,
            binding_cut_rows_with_dual: 0,
            nonzero_betas: 0,
            lifted_rows: 0,
            lifted_cut_rows: 0,
            lifted_binding_rows_with_dual: 0,
            lifted_slack_rows: 0,
            worst: None,
        }
    }

    fn summary(&self) -> String {
        let worst = self
            .worst
            .as_ref()
            .map(|w| {
                format!(
                    "{:e} of bound {:e} ({}, slot {})",
                    w.gap, w.bound, w.at, w.slot
                )
            })
            .unwrap_or_default();
        format!(
            "{} solves, {} slots, {} binding cut rows with a dual, {} nonzero betas, \
             worst beta gap {worst}; lifted rows {} (cut rows {}, binding with dual {}, slack {})",
            self.solves,
            self.slots_checked,
            self.binding_cut_rows_with_dual,
            self.nonzero_betas,
            self.lifted_rows,
            self.lifted_cut_rows,
            self.lifted_binding_rows_with_dual,
            self.lifted_slack_rows
        )
    }

    /// Checks every opening at every state of `states` over the cut rows `rows` of `stage`.
    fn check_stage(
        &mut self,
        label: &str,
        compact: &StudySetup,
        expanded: &ExpandedStudy,
        fcf: &FutureCostFunction,
        rows: &CutRowSet,
        stage: usize,
        states: &[Vec<f64>],
    ) {
        let training_ctx = compact.training_ctx();
        let tree = training_ctx.stochastic.tree_view();
        let lag_dims = lag_dims(&expanded.layout);
        for opening in 0..tree.n_openings(stage) {
            let raw_noise = tree.opening(stage, opening);
            for (index, x_hat) in states.iter().enumerate() {
                let at = format!("{label} stage {stage} opening {opening} state {index}");
                let solve =
                    assemble_expanded_solve(compact, expanded, fcf, rows, stage, x_hat, raw_noise);
                let warm = solve_from_mapped_basis(&at, expanded, &solve);
                self.check_solve(
                    &at, compact, expanded, stage, x_hat, &solve, &warm, &lag_dims,
                );
            }
        }
    }

    /// Checks every slot of the extracted projection, then the lifted solution, and records the
    /// solve's evidence.
    fn check_solve(
        &mut self,
        at: &str,
        compact: &StudySetup,
        expanded: &ExpandedStudy,
        stage: usize,
        x_hat: &[f64],
        solve: &ExpandedSolve,
        warm: &LpSolution,
        lag_dims: &[Option<(usize, usize)>],
    ) {
        self.check_betas(at, compact, expanded, stage, solve, warm, lag_dims);

        let report =
            check_lifted_compact_solution(at, compact, expanded, stage, solve, x_hat, warm);
        self.lifted_rows += report.rows_checked;
        self.lifted_cut_rows += report.cut_rows_checked;
        self.lifted_binding_rows_with_dual += report.binding_rows_with_dual;
        self.lifted_slack_rows += report.slack_rows;

        record_expanded_evidence(
            &mut self.evidence,
            compact,
            expanded,
            stage,
            solve,
            &warm.dual,
        );
        let compact_template = &compact.inputs.stage_data.stage_templates.templates[stage];
        let compact_cut = &solve.compact_cut;
        for &r in compact.stage_ctx().inflow_coupling[stage].touched_rows() {
            let r = usize::try_from(r).expect("row fits usize");
            let compact_scale = scale_at(&compact_template.row_scale, r);
            self.evidence.record_coupled_row(
                compact_scale * compact_cut.dual[r],
                compact_scale,
                scale_at(&expanded.scaled[stage].row_scale, expanded.layout.row(r)),
            );
        }
        let dims = expanded.layout.state();
        let width = dims.hydro_count * dims.max_par_order;
        let cut_duals = &compact_cut.dual[compact_template.num_rows..];
        for (k, &y) in cut_duals.iter().enumerate() {
            self.evidence
                .record_cut(y, &solve.lag_coefficients[k * width..(k + 1) * width]);
        }
        self.binding_cut_rows_with_dual += cut_duals.iter().filter(|&&y| y != 0.0).count();
        self.solves += 1;
    }

    fn check_betas(
        &mut self,
        at: &str,
        compact: &StudySetup,
        expanded: &ExpandedStudy,
        stage: usize,
        solve: &ExpandedSolve,
        warm: &LpSolution,
        lag_dims: &[Option<(usize, usize)>],
    ) {
        let layout = &expanded.layout;
        let training_ctx = compact.training_ctx();
        let extracted = &training_ctx.cut_state_layouts[solve.rows.extracted_pool];
        let space = compact.stage_state();
        let (n, depth) = (space.hydro_count, space.max_par_order);
        let width = n * depth;
        let par = training_ctx.stochastic.par();
        let coupling = &compact.stage_ctx().inflow_coupling[stage];
        let compact_template = &compact.inputs.stage_data.stage_templates.templates[stage];
        let compact_cut = &solve.compact_cut;
        let tolerance = UnscaledTolerance::from_active_profile();
        let mut compact_col = vec![None; extracted.n_slots()];
        for (slot, col) in extracted.incoming_pairs() {
            compact_col[slot.get()] = Some(col.get());
        }
        let cut_duals: Vec<(usize, f64)> = compact_cut
            .cut_dual_slots
            .iter()
            .map(|&(slot, y)| {
                let k = solve
                    .rows
                    .slots
                    .iter()
                    .position(|&s| s == slot)
                    .expect("a cut dual belongs to a cut row");
                (k, y)
            })
            .collect();

        for (j, &compact_column) in compact_col.iter().enumerate() {
            let d = extracted.global_state_index(CutSlot::new(j));
            let col = layout.incoming_state_col(d);
            let col_scale = solve.template.col_scale[col];
            let expanded_cost = warm.reduced_costs[col] / col_scale;
            let beta = compact_cut.coefficients[j];

            // n counts the roundings of the longest chain through the kernel's value of a lag
            // slot: the row-scale and coupling products and `m_h` adds of (Gᵀπ)_h, `K` adds of
            // the cut-dual terms into g, the product ψ·g and the final add of w. A slot with a
            // compact incoming column copies one reduced cost divided by its column scale, so
            // n = 1. The factor 2 covers the same quantity read from the expanded solve.
            let (propagated, magnitude, order, what) = if let Some((h, l)) = lag_dims[d.get()] {
                let psi = par.psi_slice(stage, h).get(l).copied().unwrap_or(0.0);
                let row_scale = |r: usize| scale_at(&compact_template.row_scale, r);
                let mut g_tolerance = 0.0;
                let mut g_magnitude = 0.0;
                let mut m_h = 0;
                for (r, coefficient) in coupling.hydro_entries(HydroSys::new(h)) {
                    g_tolerance += coefficient.abs() * tolerance.row_dual(row_scale(r));
                    g_magnitude += (coefficient * (row_scale(r) * compact_cut.dual[r])).abs();
                    m_h += 1;
                }
                let y_tolerance = tolerance.row_dual(1.0);
                let mut w_tolerance = 0.0;
                let mut w_magnitude = 0.0;
                for &(k, y) in &cut_duals {
                    let lags = &solve.lag_coefficients[k * width..(k + 1) * width];
                    g_tolerance += lags[h].abs() * y_tolerance;
                    g_magnitude += (y * lags[h]).abs();
                    if l + 1 < depth {
                        let shifted = lags[(l + 1) * n + h];
                        w_tolerance += shifted.abs() * y_tolerance;
                        w_magnitude += (y * shifted).abs();
                    }
                }
                (
                    psi.abs() * g_tolerance + w_tolerance,
                    psi.abs() * g_magnitude + w_magnitude,
                    m_h + cut_duals.len() + 4,
                    format!("lag slot (hydro {h}, lag {l})"),
                )
            } else {
                let c = compact_column.expect("a slot without a lag has a compact column");
                assert_eq!(
                    layout.col(c),
                    col,
                    "{at} stage {stage} slot {j}: compact column {c} vs expanded column"
                );
                (
                    tolerance.reduced_cost(scale_at(&compact_template.col_scale, c)),
                    beta.abs(),
                    1,
                    format!("compact column {c}"),
                )
            };
            let expanded_tolerance = tolerance.reduced_cost(col_scale);
            let rounding = 2.0 * rounding_factor(order) * magnitude;
            let bound = expanded_tolerance + propagated + rounding;
            let gap = (beta - expanded_cost).abs();
            assert!(
                gap <= bound,
                "{at} stage {stage} slot {j} ({what}): fold beta {beta:e} vs expanded reduced \
                 cost {expanded_cost:e}, gap {gap:e} > bound {bound:e} (expanded tolerance \
                 {expanded_tolerance:e} + propagated compact tolerance {propagated:e} + \
                 rounding {rounding:e}, n = {order}, magnitude {magnitude:e})"
            );

            self.slots_checked += 1;
            self.nonzero_betas += usize::from(beta != 0.0);
            let worse = self
                .worst
                .as_ref()
                .is_none_or(|worst| gap * worst.bound > worst.gap * bound);
            if worse {
                self.worst = Some(BetaGap {
                    at: at.to_owned(),
                    slot: j,
                    gap,
                    bound,
                });
            }
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum Fixture {
    SharedProjections,
    StorageOnlyResident,
    LagOnlyResident,
}

impl Fixture {
    fn options(self) -> FphaLagOptions {
        match self {
            Fixture::SharedProjections => FphaLagOptions {
                openings: OPENINGS,
                ..FphaLagOptions::default()
            },
            Fixture::StorageOnlyResident => FphaLagOptions {
                openings: OPENINGS,
                last_stage_inflow_lags: false,
                ..FphaLagOptions::default()
            },
            Fixture::LagOnlyResident => FphaLagOptions {
                openings: OPENINGS,
                last_stage_storage: false,
                ..FphaLagOptions::default()
            },
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum Selection {
    Off,
    Level1,
    Dynamic,
}

impl Selection {
    fn apply(self, config: &mut Config) {
        match self {
            Selection::Off => apply_oracle_config(config),
            Selection::Level1 => apply_oracle_config_with_selection(
                config,
                SelectionMethod::Level1 {
                    check_frequency: positive(1),
                    lookback_iterations: None,
                },
            ),
            Selection::Dynamic => apply_oracle_config_with_selection(
                config,
                SelectionMethod::Dynamic {
                    start_iteration: NonZeroU32::MIN,
                    candidate_recency: None,
                    seed_window: 2,
                    max_added_per_round: NonZeroU32::MIN,
                    violation_tolerance: None,
                },
            ),
        }
    }
}

#[derive(Clone, Copy)]
enum StageRule {
    EveryStageWithActiveCut,
    Root,
    Stage(usize),
}

struct KktConfig {
    label: &'static str,
    fixture: Fixture,
    selection: Selection,
    stages: StageRule,
    rows: CutRowShape,
    unmet: &'static [(Guard, &'static str)],
}

fn kkt_config(shape: CutRowShape) -> KktConfig {
    match shape {
        CutRowShape::Frozen => KktConfig {
            label: "frozen rows",
            fixture: Fixture::SharedProjections,
            selection: Selection::Level1,
            stages: StageRule::EveryStageWithActiveCut,
            rows: shape,
            unmet: &[],
        },
        CutRowShape::FrozenPlusDelta => KktConfig {
            label: "frozen rows plus delta rows",
            fixture: Fixture::SharedProjections,
            selection: Selection::Level1,
            stages: StageRule::EveryStageWithActiveCut,
            rows: shape,
            unmet: &[],
        },
        CutRowShape::LowerBoundMap => KktConfig {
            label: "lower-bound row map",
            fixture: Fixture::SharedProjections,
            selection: Selection::Level1,
            stages: StageRule::Root,
            rows: shape,
            unmet: LOWER_BOUND_MAP_UNMET_GUARDS,
        },
        CutRowShape::DynamicResidentSet => KktConfig {
            label: "dynamic resident set",
            fixture: Fixture::SharedProjections,
            selection: Selection::Dynamic,
            stages: StageRule::EveryStageWithActiveCut,
            rows: shape,
            unmet: &[],
        },
    }
}

fn differing_projection_configs() -> [KktConfig; 2] {
    [
        KktConfig {
            label: "storage-only resident pool, frozen rows plus delta rows",
            fixture: Fixture::StorageOnlyResident,
            selection: Selection::Off,
            stages: StageRule::Stage(1),
            rows: CutRowShape::FrozenPlusDelta,
            unmet: STORAGE_ONLY_RESIDENT_UNMET_GUARDS,
        },
        KktConfig {
            label: "lag-only resident pool, frozen rows plus delta rows",
            fixture: Fixture::LagOnlyResident,
            selection: Selection::Off,
            stages: StageRule::Stage(1),
            rows: CutRowShape::FrozenPlusDelta,
            unmet: &[],
        },
    ]
}

struct Trained {
    compact: StudySetup,
    expanded: ExpandedStudy,
}

fn train(config: &KktConfig) -> Trained {
    let options = config.fixture.options();
    let (system, mut study_config, hydro_models) = fpha_lag_study_with(options);
    config.selection.apply(&mut study_config);
    let mut compact = build_setup_in_code_with_models(system, &study_config, hydro_models);
    let (system, _, _) = fpha_lag_study_with(options);
    assert!(
        is_chain(&compact),
        "{}: the study must be a chain, so pool ids are stage indices",
        config.label
    );
    train_storing_trial_states(config.label, &mut compact);
    let expanded = expand_stage_templates(&system, &compact);
    Trained { compact, expanded }
}

fn cut_rows(
    label: &str,
    shape: CutRowShape,
    fcf: &mut FutureCostFunction,
    stage: usize,
) -> CutRowSet {
    match shape {
        CutRowShape::Frozen => {
            let mut rows = CutRowSet::active(fcf, stage, stage - 1);
            rows.frozen = rows.slots.len();
            assert!(
                rows.frozen >= 1,
                "{label} stage {stage}: pool {stage} holds no active cut to freeze"
            );
            eprintln!(
                "{label} stage {stage}: frozen {} of {}",
                rows.frozen,
                rows.slots.len()
            );
            rows
        }
        CutRowShape::FrozenPlusDelta => {
            let mut rows = CutRowSet::active(fcf, stage, stage - 1);
            rows.frozen = rows.slots.len() / 2;
            assert!(
                rows.frozen >= 1 && rows.slots.len() - rows.frozen >= 1,
                "{label} stage {stage}: {} active cuts split into {} frozen and {} delta rows",
                rows.slots.len(),
                rows.frozen,
                rows.slots.len() - rows.frozen
            );
            eprintln!(
                "{label} stage {stage}: frozen {} of {} active cuts",
                rows.frozen,
                rows.slots.len()
            );
            rows
        }
        CutRowShape::LowerBoundMap => {
            let pool = stage;
            let first_active = fcf
                .active_cuts(pool)
                .map(|(slot, _, _)| slot)
                .next()
                .unwrap_or_else(|| panic!("{label}: the root pool holds no active cut"));
            fcf.pools[pool].set_active(
                u32::try_from(first_active).expect("a cut slot fits u32"),
                false,
            );
            let slots: Vec<usize> = (0..fcf.pools[pool].populated()).collect();
            assert!(
                slots.iter().any(|&slot| !fcf.pools[pool].is_active(slot)),
                "{label}: every listed root slot is active"
            );
            eprintln!(
                "{label} stage {stage}: deactivated root slot {first_active}, {} slots listed",
                slots.len()
            );
            CutRowSet {
                resident_pool: pool,
                extracted_pool: pool,
                slots,
                frozen: 0,
            }
        }
        CutRowShape::DynamicResidentSet => {
            let pool = &fcf.pools[stage];
            let active = fcf.active_cuts(stage).count();
            let last_iteration = (0..pool.populated())
                .map(|slot| pool.metadata(slot).iteration_generated)
                .max()
                .unwrap_or_else(|| panic!("{label} stage {stage}: pool {stage} holds no cut"));
            let mut resident = Vec::new();
            let window = (0..=u32::try_from(last_iteration).expect("an iteration fits u32"))
                .find(|&window| {
                    build_initial_resident_set(pool, last_iteration, window, &mut resident);
                    !resident.is_empty() && resident.len() < active
                })
                .unwrap_or_else(|| {
                    panic!(
                        "{label} stage {stage}: no seed window up to the last iteration \
                         {last_iteration} gives a non-empty strict subset of the {active} \
                         active cuts"
                    )
                });
            eprintln!(
                "{label} stage {stage}: seed_window {window} at iteration {last_iteration} \
                 seeds {} of {active} active cuts",
                resident.len()
            );
            CutRowSet {
                resident_pool: stage,
                extracted_pool: stage - 1,
                slots: resident.iter().map(|&slot| slot as usize).collect(),
                frozen: 0,
            }
        }
    }
}

fn run_config(config: &KktConfig, trained: &Trained) -> KktRun {
    let label = config.label;
    let compact = &trained.compact;
    let training_ctx = compact.training_ctx();
    let layouts = training_ctx.cut_state_layouts;
    let mut fcf = compact.fcf.clone();
    let stored = |stage: usize| {
        stored_incoming_states(compact, stage).unwrap_or_else(|| {
            panic!(
                "{label} stage {stage}: pool {} drops a state dimension",
                stage - 1
            )
        })
    };
    let points: Vec<(usize, Vec<Vec<f64>>)> = match config.stages {
        StageRule::EveryStageWithActiveCut => (1..fcf.pools.len())
            .filter(|&stage| fcf.active_cuts(stage).next().is_some())
            .map(|stage| (stage, stored(stage)))
            .collect(),
        StageRule::Root => vec![(0, vec![training_ctx.initial_state.to_vec()])],
        StageRule::Stage(stage) => vec![(stage, stored(stage))],
    };
    assert!(!points.is_empty(), "{label}: no stage has an active cut");

    let mut run = KktRun::new(compact);
    for (stage, states) in points {
        let rows = cut_rows(label, config.rows, &mut fcf, stage);
        let (resident, extracted) = (&layouts[rows.resident_pool], &layouts[rows.extracted_pool]);
        eprintln!(
            "{label} stage {stage}: resident pool {} holds {} slots, extracted pool {} {}",
            rows.resident_pool,
            resident.n_slots(),
            rows.extracted_pool,
            extracted.n_slots()
        );
        match config.fixture {
            Fixture::SharedProjections => assert_eq!(
                resident.n_slots(),
                extracted.n_slots(),
                "{label} stage {stage}: the projections must agree in slot count"
            ),
            Fixture::StorageOnlyResident | Fixture::LagOnlyResident => assert_ne!(
                resident.n_slots(),
                extracted.n_slots(),
                "{label} stage {stage}: the projections must differ in slot count"
            ),
        }
        run.check_stage(
            label,
            compact,
            &trained.expanded,
            &fcf,
            &rows,
            stage,
            &states,
        );
    }
    run
}

fn assert_guards(label: &str, evidence: &PowerEvidence, unmet: &[(Guard, &str)]) {
    let reached: Vec<Guard> = SCALE_GUARDS
        .into_iter()
        .filter(|guard| unmet.iter().all(|(listed, _)| listed != guard))
        .collect();
    evidence.assert_holds(label, &reached);
    for (guard, reason) in unmet {
        assert!(
            !evidence.holds(*guard),
            "{label}: {guard:?} now holds, so assert it instead of listing it (listed because: {reason})"
        );
    }
}

#[test]
fn the_fold_meets_the_expanded_kkt_conditions_at_every_cut_row_shape() {
    let mut shapes: Vec<CutRowShape> = Vec::new();
    for kind in SolveSiteKind::ALL {
        let shape = kind.cut_row_shape();
        if !shapes.contains(&shape) {
            shapes.push(shape);
        }
    }
    let mut configs: Vec<KktConfig> = shapes.into_iter().map(kkt_config).collect();
    configs.extend(differing_projection_configs());

    let mut trainings: HashMap<(Fixture, Selection), Trained> = HashMap::new();
    for config in &configs {
        let label = config.label;
        let trained = trainings
            .entry((config.fixture, config.selection))
            .or_insert_with(|| train(config));
        let run = run_config(config, trained);

        eprintln!("{label}: {}", run.summary());
        assert!(run.solves > 0, "{label}: no solve was checked");
        assert!(
            run.binding_cut_rows_with_dual > 0,
            "{label}: no binding cut row has a nonzero dual"
        );
        assert!(
            run.nonzero_betas > 0,
            "{label}: no extracted slot has a nonzero beta"
        );
        let measured_unmet: Vec<Guard> = SCALE_GUARDS
            .into_iter()
            .filter(|&guard| !run.evidence.holds(guard))
            .collect();
        eprintln!("{label}: unmet guards {measured_unmet:?}");
        assert_guards(label, &run.evidence, config.unmet);
    }
}
