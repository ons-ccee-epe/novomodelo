//! Cost semantics of the anticipated-decision column when its delivery
//! target lands post-study: the decision column books the discounted,
//! delivery-anchored fuel cost (`cost_per_mwh[thermal, dest] *
//! post_study_hours[dest] * post_study_cumulative_discount[dest]`, unscaled)
//! and is bounded by `post_study_stages.json`'s own `[min_mw, max_mw]`
//! interval ALONE — the sole post-horizon bound surface, no intersection
//! with a second declaration. The terminal boundary FCF prices the carried
//! state fuel-exclusively, so this booking does not double-count.
//!
//! The fixture is deliberately minimal: zero hydros, one bus, one anticipated
//! thermal declared `LeadTime` over two study stages. Its OWN in-study
//! delivery (stage 1) decides at stage 0 (`lead_stages >= 1`, so the ring
//! sizes non-degenerately) and the post-study target decides at stage 1 —
//! two distinct decider stages, no fan-out, isolating the post-study
//! decision column's bound/cost from every other column family.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::doc_markdown,
    clippy::too_many_lines
)]

use chrono::NaiveDate;
use cobre_core::entities::thermal::AnticipatedConfig;
use cobre_core::{
    BoundsCountsSpec, BoundsDefaults, ContractBlockBounds, EntityId, HydroBlockBounds,
    HydroStageBounds, LineBlockBounds, PostStudyStage, PostStudyStages, PostStudyThermalBound,
    PumpingBlockBounds, ResolvedBounds, System, SystemBuilder, ThermalBlockBounds,
    ThermalStageBounds,
};
use cobre_solver::{ActiveSolver, SolverInterface};

mod common;
use common::build_setup_in_code;
use common::builders::{BusSpec, StageSpec, ThermalSpec, make_bus, make_stage, make_thermal};

const BUS_ID: EntityId = EntityId(1);
const THERMAL_ID: EntityId = EntityId(2);
/// `LeadTime` delta (hours), ~41.7 days. The thermal's own stage-1 delivery
/// decides at stage 0 (a genuine one-stage-ahead decision — `lead_stages ==
/// 1`, avoiding the degenerate `k_max == 0` ring) and the post-study target
/// decides at stage 1 — two distinct decider stages, no fan-out.
const DELTA_HOURS: f64 = 1000.0;
/// The decider stage for the post-study target and the decision-column
/// lookups below — the LAST study stage.
const DECIDER_STAGE: usize = 1;
/// Fuel cost `$/MWh` at the resolved post-study cell.
const COST_PER_MWH: f64 = 37.5;
/// Post-study stage duration \[h\] (`H`); the study carries a `0.0` discount
/// rate, so the post-study cumulative discount factor (`D`) is exactly `1.0`
/// and the analytical objective is `COST_PER_MWH * POST_STUDY_HOURS`.
const POST_STUDY_HOURS: f64 = 720.0;

fn study_start() -> NaiveDate {
    NaiveDate::from_ymd_opt(2024, 1, 1).expect("valid date")
}

/// Two stages matching their real calendar span exactly (required for
/// `StageCalendar` coverage): stage 0 is January (744h), stage 1 is February
/// (696h in a common year, but 2024 is a leap year — 30 days declared here to
/// keep `duration_hours` an exact calendar match).
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

/// The one declared post-study stage: `[2024-04-01, 2024-05-01)` — 30 days,
/// `POST_STUDY_HOURS` (720h) — the post-study target `m` (`DECIDER_STAGE + 1`
/// modular reach) resolves into, so `min_k`/`max_k` alone bound and cost the
/// decision column (`post_study_stages.json`'s table is the sole post-horizon
/// bound surface).
fn post_study_stages(min_k: f64, max_k: f64) -> PostStudyStages {
    PostStudyStages {
        stages: vec![PostStudyStage {
            start_date: NaiveDate::from_ymd_opt(2024, 4, 1).expect("valid date"),
            duration_hours: POST_STUDY_HOURS,
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

/// `with_commitment` toggles the declared `post_study_stages` destination;
/// `(min_k, max_k)` are read only when `with_commitment` is `true`.
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
        .penalties(penalties());
    if with_commitment {
        builder = builder.post_study_stages(Some(post_study_stages(min_k, max_k)));
    }
    builder.build().expect("fixture System must build")
}

fn config() -> cobre_io::config::Config {
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
            // Unscaled: the analytical assertions compare `template.objective`
            // directly against `C * H * D` with no COST_SCALE_FACTOR back-out.
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
        simulation: SimulationConfig::default(),
        exports: ExportsConfig::default(),
        estimation: EstimationConfig::default(),
    }
}

/// The decider stage's anticipated-decision column, resolved through the
/// same `StageGeometry::anticipated_decision` the LP builder itself derives
/// — never a hand-computed offset that could drift from the real layout.
/// The range is dense (one column per anticipated plant at EVERY stage, not
/// only where a genuine decision fires), so — unlike the retired
/// `commitment_decision` range — its length alone cannot confirm "the
/// fixture's plant decides at this stage"; callers needing that confirm it
/// separately (a nonzero objective, e.g.).
fn decision_col(setup: &cobre_sddp::StudySetup, decider_stage: usize) -> usize {
    setup.stage_ctx().geometry_per_stage[decider_stage]
        .anticipated_decision
        .start
}

// -- Analytical costing + post-study-table bound (LP structural, no solve) --

mod analytical_costing_and_bounds {
    use super::*;

    #[test]
    fn decision_column_objective_equals_cost_times_hours_times_discount() {
        let system = build_system(true, 20.0, 80.0);
        let setup = build_setup_in_code(system, &config());

        let col = decision_col(&setup, DECIDER_STAGE);
        let template = &setup.stage_ctx().templates[DECIDER_STAGE];

        let expected = COST_PER_MWH * POST_STUDY_HOURS * 1.0;
        assert!(
            (template.objective[col] - expected).abs() < 1e-9,
            "unscaled objective must equal C*H*D: expected {expected}, got {}",
            template.objective[col]
        );
    }

    /// The decision column's bound equals `post_study_stages.json`'s own
    /// `[min_mw, max_mw]` interval ALONE — the sole post-horizon bound
    /// surface, no intersection with a second declaration.
    #[test]
    fn decision_column_bound_equals_the_post_study_capability_alone() {
        let system = build_system(true, 20.0, 80.0);
        let setup = build_setup_in_code(system, &config());

        let col = decision_col(&setup, DECIDER_STAGE);
        let template = &setup.stage_ctx().templates[DECIDER_STAGE];

        assert_eq!(template.col_lower[col], 20.0, "col_lower must equal min_k");
        assert_eq!(template.col_upper[col], 80.0, "col_upper must equal max_k");
    }
}

// -- Pinned commitment: min_k == max_k, solved end-to-end --

mod pinned_decision {
    use super::*;

    #[test]
    fn pinned_commitment_within_capability_fixes_the_decision_and_its_contribution() {
        let pinned = 30.0;
        let system = build_system(true, pinned, pinned);
        let setup = build_setup_in_code(system, &config());

        let col = decision_col(&setup, DECIDER_STAGE);
        let template = &setup.stage_ctx().templates[DECIDER_STAGE];

        assert_eq!(template.col_lower[col], pinned);
        assert_eq!(template.col_upper[col], pinned);

        let mut solver = ActiveSolver::new().expect("ActiveSolver::new: must succeed");
        solver.load_model(template);
        let view = solver.solve(None).expect("stage LP must solve");

        assert!(
            (view.primal[col] - pinned).abs() < 1e-6,
            "solved primal must equal the pinned commitment: expected {pinned}, got {}",
            view.primal[col]
        );

        let expected_contribution = pinned * COST_PER_MWH * POST_STUDY_HOURS;
        let actual_contribution = view.primal[col] * template.objective[col];
        assert!(
            (actual_contribution - expected_contribution).abs() < 1e-6,
            "objective contribution must equal min_c*C*H*D: expected \
             {expected_contribution}, got {actual_contribution}"
        );
    }
}

// -- Inert without a post-horizon commitment (byte-identity regression guard) --

mod byte_identity_without_commitment {
    use super::*;

    /// Dropping the declared `post_study_stages` destination is carried by
    /// the SAME in-study anticipated ring every study already has
    /// (`commit_out.len() == n_anticipated * k_max`, no separate lane block
    /// ever appended). The two fixtures differ only in whether the ring's
    /// DECIDER_STAGE column is ACTIVE: with no post-study destination
    /// declared, the plant has no genuine decision at DECIDER_STAGE at all
    /// (dormant `[0, 0]`, zero objective); with one declared, that same
    /// column books the real post-study fuel cost.
    #[test]
    fn no_post_study_destination_resolves_zero_commitment_windows() {
        let with_setup = build_setup_in_code(build_system(true, 20.0, 80.0), &config());
        let without_setup = build_setup_in_code(build_system(false, 0.0, 0.0), &config());

        for (label, setup) in [("with", &with_setup), ("without", &without_setup)] {
            let state = setup.stage_state();
            assert_eq!(
                state.commit_out.len(),
                state.n_anticipated * state.k_max,
                "{label}: commit_out must be exactly the in-study ring's width, never a \
                 separate lane block"
            );
        }

        let without_col = decision_col(&without_setup, DECIDER_STAGE);
        assert_eq!(
            without_setup.stage_ctx().templates[DECIDER_STAGE].objective[without_col],
            0.0,
            "without a declared post-study destination, the plant has no genuine decision at \
             DECIDER_STAGE — the column stays dormant"
        );

        let with_col = decision_col(&with_setup, DECIDER_STAGE);
        assert!(
            with_setup.stage_ctx().templates[DECIDER_STAGE].objective[with_col].abs() > 1e-9,
            "with a declared post-study destination, the plant's post-study delivery decides \
             at DECIDER_STAGE — the column books the real fuel cost"
        );
    }
}
