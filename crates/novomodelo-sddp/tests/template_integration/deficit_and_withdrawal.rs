//! `deficit_and_withdrawal` section tests.

use super::*;
use novomodelo_sddp::indexer::{BlockIdx, Boundary, BusSys, HydroCell, HydroSys};

#[test]
#[allow(clippy::too_many_lines)]
fn test_multi_segment_deficit_column_count() {
    use chrono::NaiveDate;
    use novomodelo_core::scenario::LoadModel;
    use novomodelo_core::temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, StageRiskConfig, StageStateConfig,
    };

    let bus0 = make_bus(
        EntityId(1),
        BusSpec {
            name: "Bus0".to_string(),
            deficit_segments: vec![
                DeficitSegment {
                    depth_mw: Some(10.0),
                    cost_per_mwh: 100.0,
                },
                DeficitSegment {
                    depth_mw: Some(20.0),
                    cost_per_mwh: 200.0,
                },
                DeficitSegment {
                    depth_mw: None,
                    cost_per_mwh: 5000.0,
                },
            ],
            ..Default::default()
        },
    );
    let bus1 = make_bus(
        EntityId(2),
        BusSpec {
            name: "Bus1".to_string(),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 1000.0,
            }],
            ..Default::default()
        },
    );

    let load_models = vec![
        LoadModel {
            bus_id: EntityId(1),
            stage_id: 0,
            mean_mw: 0.0,
            std_mw: 0.0,
        },
        LoadModel {
            bus_id: EntityId(2),
            stage_id: 0,
            mean_mw: 0.0,
            std_mw: 0.0,
        },
    ];

    let stage = make_stage(
        0,
        StageSpec {
            start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
            season_id: None,
            blocks: vec![
                Block {
                    index: 0,
                    name: "B0".to_string(),
                    duration_hours: 360.0,
                },
                Block {
                    index: 1,
                    name: "B1".to_string(),
                    duration_hours: 360.0,
                },
            ],
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
            ..Default::default()
        },
    );

    let bounds = ResolvedBounds::new(
        &BoundsCountsSpec {
            n_hydros: 0,
            n_thermals: 0,
            n_lines: 0,
            n_pumping: 0,
            n_contracts: 0,
            n_stages: 1,
            k_max: 0,
        },
        &BoundsDefaults {
            hydro: default_hydro_bounds(),
            hydro_block: default_hydro_block_bounds(),
            thermal_block: ThermalBlockBounds {
                min_generation_mw: 0.0,
                max_generation_mw: 0.0,
                cost_per_mwh: 0.0,
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
    let penalties = ResolvedPenalties::new(
        &PenaltiesCountsSpec {
            n_hydros: 0,
            n_buses: 2,
            n_lines: 0,
            n_ncs: 0,
            n_stages: 1,
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

    let system = SystemBuilder::new()
        .buses(vec![bus0, bus1])
        .stages(vec![stage])
        .load_models(load_models)
        .bounds(bounds)
        .penalties(penalties)
        .build()
        .expect("2-bus 2-block system: valid");

    let result = build_stage_templates_resolving_layout(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("build ok");

    let t = &result.templates[0];

    // N=0, L=0 → theta=0, decision_start=1
    // No thermals, no lines → col_deficit_start = 1
    // B=2, S_max=3, K=2 → deficit region = 2*3*2 = 12 columns → col_excess_start = 13
    // excess region = B*K = 2*2 = 4 → num_cols = 13 + 4 = 17
    let col_deficit_start = 1_usize;
    let max_deficit_segments = 3_usize;
    let n_blks = 2_usize;
    let n_buses = 2_usize;
    let deficit_region = n_buses * max_deficit_segments * n_blks;
    assert_eq!(
        deficit_region, 12,
        "deficit region must be B*S_max*K = 2*3*2 = 12"
    );
    let col_excess_start = col_deficit_start + deficit_region;
    let excess_region = n_buses * n_blks;
    let expected_num_cols = col_excess_start + excess_region;
    assert_eq!(
        t.num_cols, expected_num_cols,
        "num_cols must include expanded deficit region"
    );
}

#[test]
fn test_multi_segment_deficit_bounds_and_objective() {
    let bus = make_bus(
        EntityId(1),
        BusSpec {
            name: "Bus0".to_string(),
            deficit_segments: vec![
                DeficitSegment {
                    depth_mw: Some(10.0),
                    cost_per_mwh: 500.0,
                },
                DeficitSegment {
                    depth_mw: None,
                    cost_per_mwh: 5000.0,
                },
            ],
            ..Default::default()
        },
    );

    let block_hours = 730.0_f64;
    let system = multi_segment_system(vec![bus], block_hours);

    let result = build_stage_templates_resolving_layout(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("build ok");

    let t = &result.templates[0];

    // N=0 → theta=0, decision_start=1, no thermals/lines
    // col_deficit_start = 1
    // B=1, S_max=2, K=1 → deficit region = 1*2*1 = 2
    // seg0 col = 1 + 0*2*1 + 0*1 + 0 = 1
    // seg1 col = 1 + 0*2*1 + 1*1 + 0 = 2
    let col_seg0 = 1_usize;
    let col_seg1 = 2_usize;

    assert_eq!(
        t.col_upper[col_seg0], 10.0,
        "segment 0 upper bound must equal depth_mw = 10.0"
    );
    assert!(
        t.col_upper[col_seg1].is_infinite() && t.col_upper[col_seg1] > 0.0,
        "segment 1 upper bound must be +infinity (unbounded final segment)"
    );
    assert!(
        (t.objective[col_seg0] - 500.0 * block_hours / COST_SCALE_FACTOR).abs() < 1e-12,
        "segment 0 objective must be cost * block_hours / COST_SCALE_FACTOR = {} but got {}",
        500.0 * block_hours / COST_SCALE_FACTOR,
        t.objective[col_seg0]
    );
    assert!(
        (t.objective[col_seg1] - 5000.0 * block_hours / COST_SCALE_FACTOR).abs() < 1e-12,
        "segment 1 objective must be cost * block_hours / COST_SCALE_FACTOR = {} but got {}",
        5000.0 * block_hours / COST_SCALE_FACTOR,
        t.objective[col_seg1]
    );
}

#[test]
fn spare_deficit_slots_are_pinned_to_zero() {
    let bus_two_segments = make_bus(
        EntityId(1),
        BusSpec {
            name: "Bus1".to_string(),
            deficit_segments: vec![
                DeficitSegment {
                    depth_mw: Some(10.0),
                    cost_per_mwh: 500.0,
                },
                DeficitSegment {
                    depth_mw: None,
                    cost_per_mwh: 5000.0,
                },
            ],
            ..Default::default()
        },
    );
    let bus_one_segment = make_bus(
        EntityId(2),
        BusSpec {
            name: "Bus2".to_string(),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 1000.0,
            }],
            ..Default::default()
        },
    );

    let block_hours = 730.0_f64;
    let system = multi_segment_system(vec![bus_two_segments, bus_one_segment], block_hours);
    let t = &build_templates_for(&system)[0];

    // S=2, K=1, no thermals/lines: col = 1 + b*S*K + seg*K + blk.
    let (b0_seg0, b0_seg1) = (1_usize, 2_usize);
    let (b1_seg0, b1_spare) = (3_usize, 4_usize);

    assert_eq!(t.col_lower[b1_spare], 0.0, "spare slot lower bound");
    assert_eq!(t.col_upper[b1_spare], 0.0, "spare slot upper bound");
    assert_eq!(t.objective[b1_spare], 0.0, "spare slot objective");

    assert_eq!(t.col_upper[b0_seg0], 10.0);
    assert!(t.col_upper[b0_seg1].is_infinite() && t.col_upper[b0_seg1] > 0.0);
    assert!((t.objective[b0_seg0] - 500.0 * block_hours / COST_SCALE_FACTOR).abs() < 1e-12);
    assert!((t.objective[b0_seg1] - 5000.0 * block_hours / COST_SCALE_FACTOR).abs() < 1e-12);

    assert!(t.col_upper[b1_seg0].is_infinite() && t.col_upper[b1_seg0] > 0.0);
    assert!((t.objective[b1_seg0] - 1000.0 * block_hours / COST_SCALE_FACTOR).abs() < 1e-12);
}

const OVERRIDE_STAGE_BLOCK_HOURS: [&[f64]; 2] = [&[730.0], &[200.0, 472.0]];

fn override_buses() -> Vec<Bus> {
    let bus = |id, depth_mw, cost_per_mwh, final_cost_per_mwh| {
        make_bus(
            EntityId(id),
            BusSpec {
                name: format!("Bus{id}"),
                deficit_segments: vec![
                    DeficitSegment {
                        depth_mw: Some(depth_mw),
                        cost_per_mwh,
                    },
                    DeficitSegment {
                        depth_mw: None,
                        cost_per_mwh: final_cost_per_mwh,
                    },
                ],
                ..Default::default()
            },
        )
    };
    vec![bus(1, 10.0, 500.0, 5000.0), bus(2, 20.0, 800.0, 3000.0)]
}

/// Baseline and overridden systems over [`override_buses`]: a block override at
/// (stage 1, block 0, bus 0, segment 0) and a stage-wide override at
/// (stage 0, bus 1, segment 0).
fn baseline_and_overridden_deficit_systems() -> (novomodelo_core::System, novomodelo_core::System) {
    let buses = override_buses();
    let mut table = ResolvedDeficitSegments::seeded(&buses, 2, 2);
    *table.block_segment_mut(0, 1, 0, 0).expect("cell") = Some(DeficitSegment {
        depth_mw: Some(3.0),
        cost_per_mwh: 500.0,
    });
    *table.stage_segment_mut(1, 0, 0).expect("cell") = DeficitSegment {
        depth_mw: Some(7.0),
        cost_per_mwh: 900.0,
    };
    let system = |deficit_table| {
        deficit_test_system(
            buses.clone(),
            &OVERRIDE_STAGE_BLOCK_HOURS,
            vec![],
            novomodelo_core::ResolvedGenericConstraintBounds::empty(),
            deficit_table,
        )
    };
    (system(None), system(Some(table)))
}

fn differing_columns(base: &[f64], overridden: &[f64]) -> Vec<usize> {
    (0..base.len())
        .filter(|&c| base[c].to_bits() != overridden[c].to_bits())
        .collect()
}

#[test]
fn deficit_block_override_reaches_only_its_stage_and_block() {
    let (baseline, overridden) = baseline_and_overridden_deficit_systems();
    let base = build_templates_for(&baseline);
    let over = build_templates_for(&overridden);

    // S = 2: stage 0 (K = 1) bus 1 segment 0 is column 3; stage 1 (K = 2)
    // bus 0 segment 0 is column 1 at block 0 and column 2 at block 1.
    assert_eq!(
        differing_columns(&base[0].col_upper, &over[0].col_upper),
        [3]
    );
    assert_eq!(
        differing_columns(&base[0].objective, &over[0].objective),
        [3]
    );
    assert_eq!(
        differing_columns(&base[1].col_upper, &over[1].col_upper),
        [1]
    );
    assert_eq!(
        differing_columns(&base[1].objective, &over[1].objective),
        Vec::<usize>::new(),
        "the block cell keeps the base cost"
    );

    assert_eq!(over[0].col_upper[3], 7.0);
    assert!((over[0].objective[3] - 900.0 * 730.0 / COST_SCALE_FACTOR).abs() < 1e-12);
    assert_eq!(over[1].col_upper[1], 3.0);
    assert_eq!(
        over[1].col_upper[2], 10.0,
        "block 1 reads the stage cell: only block 0 is overridden"
    );
}

#[test]
fn deficit_override_leaves_lp_structure_unchanged() {
    let (baseline, overridden) = baseline_and_overridden_deficit_systems();
    let base = build_templates_for(&baseline);
    let over = build_templates_for(&overridden);

    assert_eq!(base.len(), over.len());
    for (stage, (b, o)) in base.iter().zip(&over).enumerate() {
        assert_eq!(b.num_cols, o.num_cols, "stage {stage} num_cols");
        assert_eq!(b.num_rows, o.num_rows, "stage {stage} num_rows");
        assert_eq!(b.num_nz, o.num_nz, "stage {stage} num_nz");
        assert_eq!(b.col_starts, o.col_starts, "stage {stage} col_starts");
        assert_eq!(b.row_indices, o.row_indices, "stage {stage} row_indices");
        assert_eq!(b.values, o.values, "stage {stage} values");
        assert_eq!(b.col_lower, o.col_lower, "stage {stage} col_lower");
        assert_eq!(b.row_lower, o.row_lower, "stage {stage} row_lower");
        assert_eq!(b.row_upper, o.row_upper, "stage {stage} row_upper");
    }
}

#[test]
fn test_single_segment_backward_compat() {
    let cost = 1000.0_f64;
    let block_hours = 744.0_f64;

    let bus = make_bus(
        EntityId(1),
        BusSpec {
            name: "Bus0".to_string(),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: cost,
            }],
            ..Default::default()
        },
    );

    let system = multi_segment_system(vec![bus], block_hours);

    let result = build_stage_templates_resolving_layout(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("build ok");

    let t = &result.templates[0];

    // N=0 → theta=0, decision_start=1, col_deficit_start=1
    // B=1, S_max=1, K=1 → 1 deficit column at index 1
    let col_def = 1_usize;

    assert!(
        t.col_upper[col_def].is_infinite() && t.col_upper[col_def] > 0.0,
        "single segment must be unbounded (None depth_mw)"
    );
    assert!(
        (t.objective[col_def] - cost * block_hours / COST_SCALE_FACTOR).abs() < 1e-12,
        "single-segment objective must be cost * block_hours / COST_SCALE_FACTOR"
    );

    // Excess column immediately follows deficit (S_max=1, B=1, K=1 → excess at col 2)
    let col_exc = 2_usize;
    assert!(
        t.col_upper[col_exc].is_infinite() && t.col_upper[col_exc] > 0.0,
        "excess column must be unbounded"
    );
}

#[test]
fn test_multi_segment_deficit_load_balance_coefficients() {
    let bus = make_bus(
        EntityId(1),
        BusSpec {
            name: "Bus0".to_string(),
            deficit_segments: vec![
                DeficitSegment {
                    depth_mw: Some(10.0),
                    cost_per_mwh: 500.0,
                },
                DeficitSegment {
                    depth_mw: None,
                    cost_per_mwh: 5000.0,
                },
            ],
            ..Default::default()
        },
    );

    let system = multi_segment_system(vec![bus], 730.0);

    let result = build_stage_templates_resolving_layout(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("build ok");

    let t = &result.templates[0];

    // N=0, no thermals/lines → col_deficit_start = 1
    // B=1, S_max=2, K=1 → seg0 at col 1, seg1 at col 2
    let col_seg0 = 1_usize;
    let col_seg1 = 2_usize;

    // N=0 hydros → row_load_balance_start = 0; bus 0, block 0 → row 0
    let load_balance_row = 0_usize;

    let entries_seg0 = entries_for_col(t, col_seg0);
    assert_eq!(
        entries_seg0.len(),
        1,
        "deficit segment 0 column must have exactly 1 CSC entry (load-balance row), got {entries_seg0:?}"
    );
    assert_eq!(
        entries_seg0[0].0, load_balance_row,
        "deficit segment 0 entry must be at the load-balance row {load_balance_row}, got row {}",
        entries_seg0[0].0
    );
    assert!(
        (entries_seg0[0].1 - 1.0).abs() < 1e-12,
        "deficit segment 0 load-balance coefficient must be +1.0, got {}",
        entries_seg0[0].1
    );

    let entries_seg1 = entries_for_col(t, col_seg1);
    assert_eq!(
        entries_seg1.len(),
        1,
        "deficit segment 1 column must have exactly 1 CSC entry (load-balance row), got {entries_seg1:?}"
    );
    assert_eq!(
        entries_seg1[0].0, load_balance_row,
        "deficit segment 1 entry must be at the load-balance row {load_balance_row}, got row {}",
        entries_seg1[0].0
    );
    assert!(
        (entries_seg1[0].1 - 1.0).abs() < 1e-12,
        "deficit segment 1 load-balance coefficient must be +1.0, got {}",
        entries_seg1[0].1
    );
}

/// Water balance RHS = -(ζ * `water_withdrawal_m3s`); the realized inflow
/// enters through the per-solve row-bound patch rather than a template RHS term. For
/// withdrawal=10 the RHS is `-(744 * 3600/1_000_000 * 10) = -2.6784`.
#[test]
fn withdrawal_rhs_subtracted_from_water_balance() {
    let withdrawal = 10.0_f64;
    let system = one_hydro_system_with_withdrawal(1, 0, withdrawal, 0.0);
    let result = build_stage_templates_resolving_layout(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("build ok");

    let t = &result.templates[0];
    let row_water = 0_usize; // the water-balance rows start at row 0
    let total_hours = 744.0_f64;
    let zeta = total_hours * 3_600.0 / 1_000_000.0;
    let expected_rhs = -(zeta * withdrawal);
    assert!(
        (t.row_lower[row_water] - expected_rhs).abs() < 1e-12,
        "water balance row_lower: expected {expected_rhs}, got {}",
        t.row_lower[row_water]
    );
    assert!(
        (t.row_upper[row_water] - expected_rhs).abs() < 1e-12,
        "water balance row_upper: expected {expected_rhs}, got {}",
        t.row_upper[row_water]
    );
}

/// `PrecomputedPar::default()` has `n_stages`=0, so `deterministic_base`=0 cannot
/// be set to a non-zero value here; this test only confirms withdrawal=0 leaves
/// the RHS identical to the no-withdrawal case.
#[test]
fn withdrawal_zero_leaves_rhs_unchanged_from_base() {
    let system_zero = one_hydro_system_with_withdrawal(1, 0, 0.0, 0.0);
    let system_base = one_hydro_system(1, 0);

    let result_zero = build_stage_templates_resolving_layout(
        &system_zero,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system_zero),
        &default_evaporation(&system_zero),
        &ResolvedParameters::default(),
    )
    .expect("zero-withdrawal build ok");

    let result_base = build_stage_templates_resolving_layout(
        &system_base,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system_base),
        &default_evaporation(&system_base),
        &ResolvedParameters::default(),
    )
    .expect("base build ok");

    let t_zero = &result_zero.templates[0];
    let t_base = &result_base.templates[0];

    // N=1, L=0: the water-balance rows start at row 0.
    let row_water = 0_usize;
    assert!(
        (t_zero.row_lower[row_water] - t_base.row_lower[row_water]).abs() < 1e-15,
        "zero-withdrawal row_lower must equal base: {} vs {}",
        t_zero.row_lower[row_water],
        t_base.row_lower[row_water]
    );
    assert!(
        (t_zero.row_upper[row_water] - t_base.row_upper[row_water]).abs() < 1e-15,
        "zero-withdrawal row_upper must equal base: {} vs {}",
        t_zero.row_upper[row_water],
        t_base.row_upper[row_water]
    );
}

#[test]
fn withdrawal_slack_matrix_entry_coefficient_is_minus_zeta() {
    let system = one_hydro_system_with_withdrawal(1, 0, 5.0, 1000.0);
    let result = build_stage_templates_resolving_layout(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("build ok");

    let t = &result.templates[0];
    let col_neg = result.geometry_per_stage[0].withdrawal_slack_neg_col(HydroSys::new(0));
    let row_water = 0_usize; // water balance for hydro 0

    let total_hours = 744.0_f64;
    let zeta = total_hours * 3_600.0 / 1_000_000.0;

    let coeff = csc_entry(t, col_neg, row_water).unwrap_or_else(|| {
        panic!(
            "withdrawal slack neg column {col_neg} has no entry at water balance row {row_water}"
        )
    });
    assert!(
        (coeff - (-zeta)).abs() < 1e-12,
        "withdrawal slack neg coefficient: expected {}, got {coeff}",
        -zeta
    );

    let all_entries = entries_for_col(t, col_neg);
    assert_eq!(
        all_entries.len(),
        1,
        "withdrawal slack neg column must have exactly 1 CSC entry, got {all_entries:?}"
    );
}

#[test]
fn withdrawal_slack_objective_equals_cost_times_hours() {
    let violation_cost = 1_000.0_f64;
    let total_hours = 744.0_f64;
    let system = one_hydro_system_with_withdrawal(1, 0, 5.0, violation_cost);
    let result = build_stage_templates_resolving_layout(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("build ok");

    let t = &result.templates[0];
    let col_w = result.geometry_per_stage[0].withdrawal_slack_neg_col(HydroSys::new(0));
    let expected_obj = violation_cost * total_hours / COST_SCALE_FACTOR;
    assert!(
        (t.objective[col_w] - expected_obj).abs() < 1e-12,
        "withdrawal slack objective: expected {expected_obj}, got {}",
        t.objective[col_w]
    );
}

#[test]
fn withdrawal_slack_objective_zero_when_cost_is_zero() {
    let system = one_hydro_system_with_withdrawal(1, 0, 0.0, 0.0);
    let result = build_stage_templates_resolving_layout(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("build ok");

    let t = &result.templates[0];
    let col_w = result.geometry_per_stage[0].withdrawal_slack_neg_col(HydroSys::new(0));
    assert!(
        t.objective[col_w].abs() < 1e-15,
        "withdrawal slack neg objective must be 0 when cost=0, got {}",
        t.objective[col_w]
    );
}

#[test]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::too_many_lines
)]
fn two_hydro_withdrawal_slack_entries_per_hydro() {
    use chrono::NaiveDate;
    use novomodelo_core::entities::hydro::{HydroGenerationModel, HydroPenalties};
    use novomodelo_core::scenario::{InflowModel, LoadModel};
    use novomodelo_core::temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, StageRiskConfig, StageStateConfig,
    };

    let bus = make_bus(
        EntityId(1),
        BusSpec {
            name: "B1".to_string(),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 500.0,
            }],
            ..Default::default()
        },
    );

    let stages = vec![make_stage(
        0,
        StageSpec {
            start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
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
    )];

    let inflow_models = vec![
        InflowModel {
            hydro_id: EntityId(2),
            stage_id: 0,
            mean_m3s: 80.0,
            std_m3s: 20.0,
            ar_coefficients: vec![],
            residual_std_ratio: 1.0,
            annual: None,
        },
        InflowModel {
            hydro_id: EntityId(3),
            stage_id: 0,
            mean_m3s: 50.0,
            std_m3s: 10.0,
            ar_coefficients: vec![],
            residual_std_ratio: 1.0,
            annual: None,
        },
    ];

    let load_models = vec![LoadModel {
        bus_id: EntityId(1),
        stage_id: 0,
        mean_mw: 100.0,
        std_mw: 0.0,
    }];

    let hydro_bounds_default = HydroStageBounds {
        min_storage_hm3: 0.0,
        max_storage_hm3: 200.0,
        filling_min_rate_m3s: 0.0,
        water_withdrawal_m3s: 0.0,
    };
    let hydro_bounds_default_block = HydroBlockBounds {
        max_turbined_m3s: 100.0,
        max_generation_mw: 250.0,
        ..Default::default()
    };
    let mut bounds = ResolvedBounds::new(
        &BoundsCountsSpec {
            n_hydros: 2,
            n_thermals: 0,
            n_lines: 0,
            n_pumping: 0,
            n_contracts: 0,
            n_stages: 1,
            k_max: 0,
        },
        &BoundsDefaults {
            hydro: hydro_bounds_default,
            hydro_block: hydro_bounds_default_block,
            thermal_block: ThermalBlockBounds {
                min_generation_mw: 0.0,
                max_generation_mw: 0.0,
                cost_per_mwh: 0.0,
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
    bounds.hydro_bounds_mut(0, 0).water_withdrawal_m3s = 8.0;
    bounds.hydro_bounds_mut(1, 0).water_withdrawal_m3s = 12.0;

    let hydro_penalties_default = HydroPenalties {
        spillage_cost: 0.01,
        diversion_cost: 0.0,
        turbined_cost: 0.0,
        storage_violation_below_cost: 0.0,
        filling_target_violation_cost: 0.0,
        turbined_violation_below_cost: 0.0,
        outflow_violation_below_cost: 0.0,
        outflow_violation_above_cost: 0.0,
        generation_violation_below_cost: 0.0,
        fpha_violation_cost: 0.0,
        evaporation_violation_cost: 0.0,
        water_withdrawal_violation_cost: 2_000.0,
        water_withdrawal_violation_pos_cost: 2_000.0,
        water_withdrawal_violation_neg_cost: 2_000.0,
        evaporation_violation_pos_cost: 0.0,
        evaporation_violation_neg_cost: 0.0,
        inflow_nonnegativity_cost: 1000.0,
    };
    let penalties = ResolvedPenalties::new(
        &PenaltiesCountsSpec {
            n_hydros: 2,
            n_buses: 1,
            n_lines: 0,
            n_ncs: 0,
            n_stages: 1,
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

    let system = SystemBuilder::new()
        .buses(vec![bus])
        .hydros(vec![
            make_hydro(
                EntityId(2),
                HydroSpec {
                    name: "H2".to_string(),
                    bus_id: EntityId(1),
                    max_storage_hm3: 200.0,
                    generation_model: HydroGenerationModel::ConstantProductivity,
                    max_turbined_m3s: 100.0,
                    max_generation_mw: 250.0,
                    penalties: HydroPenalties {
                        spillage_cost: 0.01,
                        diversion_cost: 0.0,
                        turbined_cost: 0.0,
                        storage_violation_below_cost: 0.0,
                        filling_target_violation_cost: 0.0,
                        turbined_violation_below_cost: 0.0,
                        outflow_violation_below_cost: 0.0,
                        outflow_violation_above_cost: 0.0,
                        generation_violation_below_cost: 0.0,
                        fpha_violation_cost: 0.0,
                        evaporation_violation_cost: 0.0,
                        water_withdrawal_violation_cost: 2_000.0,
                        water_withdrawal_violation_pos_cost: 2_000.0,
                        water_withdrawal_violation_neg_cost: 2_000.0,
                        evaporation_violation_pos_cost: 0.0,
                        evaporation_violation_neg_cost: 0.0,
                        inflow_nonnegativity_cost: 1000.0,
                    },
                    ..Default::default()
                },
            ),
            make_hydro(
                EntityId(3),
                HydroSpec {
                    name: "H3".to_string(),
                    bus_id: EntityId(1),
                    max_storage_hm3: 200.0,
                    generation_model: HydroGenerationModel::ConstantProductivity,
                    max_turbined_m3s: 100.0,
                    max_generation_mw: 250.0,
                    penalties: HydroPenalties {
                        spillage_cost: 0.01,
                        diversion_cost: 0.0,
                        turbined_cost: 0.0,
                        storage_violation_below_cost: 0.0,
                        filling_target_violation_cost: 0.0,
                        turbined_violation_below_cost: 0.0,
                        outflow_violation_below_cost: 0.0,
                        outflow_violation_above_cost: 0.0,
                        generation_violation_below_cost: 0.0,
                        fpha_violation_cost: 0.0,
                        evaporation_violation_cost: 0.0,
                        water_withdrawal_violation_cost: 2_000.0,
                        water_withdrawal_violation_pos_cost: 2_000.0,
                        water_withdrawal_violation_neg_cost: 2_000.0,
                        evaporation_violation_pos_cost: 0.0,
                        evaporation_violation_neg_cost: 0.0,
                        inflow_nonnegativity_cost: 1000.0,
                    },
                    ..Default::default()
                },
            ),
        ])
        .stages(stages)
        .inflow_models(inflow_models)
        .load_models(load_models)
        .bounds(bounds)
        .penalties(penalties)
        .build()
        .expect("two_hydro_with_withdrawal: valid");

    let result = build_stage_templates_resolving_layout(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("build ok");

    let t = &result.templates[0];
    let g = &result.geometry_per_stage[0];
    let col_w0 = g.withdrawal_slack_neg_col(HydroSys::new(0));
    let col_w1 = g.withdrawal_slack_neg_col(HydroSys::new(1));

    let row_w0 = 0_usize; // water balance for hydro 0
    let row_w1 = 1_usize; // water balance for hydro 1

    let total_hours = 744.0_f64;
    let zeta = total_hours * 3_600.0 / 1_000_000.0;

    let coeff_w0 = csc_entry(t, col_w0, row_w0).unwrap_or_else(|| {
        panic!("withdrawal slack col {col_w0} has no entry at water balance row {row_w0}")
    });
    assert!(
        (coeff_w0 - (-zeta)).abs() < 1e-12,
        "hydro-0 withdrawal slack coeff: expected {}, got {coeff_w0}",
        -zeta
    );

    let coeff_w1 = csc_entry(t, col_w1, row_w1).unwrap_or_else(|| {
        panic!("withdrawal slack col {col_w1} has no entry at water balance row {row_w1}")
    });
    assert!(
        (coeff_w1 - (-zeta)).abs() < 1e-12,
        "hydro-1 withdrawal slack coeff: expected {}, got {coeff_w1}",
        -zeta
    );

    assert!(
        csc_entry(t, col_w0, row_w1).is_none(),
        "hydro-0 withdrawal slack must not appear in hydro-1 water balance row"
    );
    assert!(
        csc_entry(t, col_w1, row_w0).is_none(),
        "hydro-1 withdrawal slack must not appear in hydro-0 water balance row"
    );
}

#[test]
#[allow(clippy::too_many_lines)]
fn three_hydro_num_cols_includes_three_withdrawal_slacks() {
    use chrono::NaiveDate;
    use novomodelo_core::entities::hydro::{HydroGenerationModel, HydroPenalties};
    use novomodelo_core::scenario::{InflowModel, LoadModel};
    use novomodelo_core::temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, StageRiskConfig, StageStateConfig,
    };

    let bus = make_bus(
        EntityId(1),
        BusSpec {
            name: "B1".to_string(),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 500.0,
            }],
            ..Default::default()
        },
    );

    let stages = vec![make_stage(
        0,
        StageSpec {
            start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
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
    )];

    let inflow_models: Vec<InflowModel> = [1, 2, 3]
        .iter()
        .map(|&hid| InflowModel {
            hydro_id: EntityId(hid),
            stage_id: 0,
            mean_m3s: 50.0,
            std_m3s: 10.0,
            ar_coefficients: vec![],
            residual_std_ratio: 1.0,
            annual: None,
        })
        .collect();

    let load_models = vec![LoadModel {
        bus_id: EntityId(1),
        stage_id: 0,
        mean_mw: 100.0,
        std_mw: 0.0,
    }];

    let bounds = ResolvedBounds::new(
        &BoundsCountsSpec {
            n_hydros: 3,
            n_thermals: 0,
            n_lines: 0,
            n_pumping: 0,
            n_contracts: 0,
            n_stages: 1,
            k_max: 0,
        },
        &BoundsDefaults {
            hydro: HydroStageBounds {
                min_storage_hm3: 0.0,
                max_storage_hm3: 200.0,
                filling_min_rate_m3s: 0.0,
                water_withdrawal_m3s: 5.0,
            },
            hydro_block: HydroBlockBounds {
                max_turbined_m3s: 100.0,
                max_generation_mw: 250.0,
                ..Default::default()
            },
            thermal_block: ThermalBlockBounds {
                min_generation_mw: 0.0,
                max_generation_mw: 0.0,
                cost_per_mwh: 0.0,
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

    let penalties = ResolvedPenalties::new(
        &PenaltiesCountsSpec {
            n_hydros: 3,
            n_buses: 1,
            n_lines: 0,
            n_ncs: 0,
            n_stages: 1,
        },
        &PenaltiesDefaults {
            hydro: HydroPenalties {
                spillage_cost: 0.01,
                diversion_cost: 0.0,
                turbined_cost: 0.0,
                storage_violation_below_cost: 0.0,
                filling_target_violation_cost: 0.0,
                turbined_violation_below_cost: 0.0,
                outflow_violation_below_cost: 0.0,
                outflow_violation_above_cost: 0.0,
                generation_violation_below_cost: 0.0,
                fpha_violation_cost: 0.0,
                evaporation_violation_cost: 0.0,
                water_withdrawal_violation_cost: 1_000.0,
                water_withdrawal_violation_pos_cost: 1_000.0,
                water_withdrawal_violation_neg_cost: 1_000.0,
                evaporation_violation_pos_cost: 0.0,
                evaporation_violation_neg_cost: 0.0,
                inflow_nonnegativity_cost: 1000.0,
            },
            bus: BusStagePenalties { excess_cost: 0.0 },
            line: LineStagePenalties { exchange_cost: 0.0 },
            ncs: NcsStagePenalties {
                curtailment_cost: 0.0,
            },
        },
    );

    let system = SystemBuilder::new()
        .buses(vec![bus])
        .hydros(vec![
            make_hydro(
                EntityId(1),
                HydroSpec {
                    name: "H1".to_string(),
                    bus_id: EntityId(1),
                    max_storage_hm3: 200.0,
                    generation_model: HydroGenerationModel::ConstantProductivity,
                    max_turbined_m3s: 100.0,
                    max_generation_mw: 250.0,
                    penalties: HydroPenalties {
                        spillage_cost: 0.01,
                        diversion_cost: 0.0,
                        turbined_cost: 0.0,
                        storage_violation_below_cost: 0.0,
                        filling_target_violation_cost: 0.0,
                        turbined_violation_below_cost: 0.0,
                        outflow_violation_below_cost: 0.0,
                        outflow_violation_above_cost: 0.0,
                        generation_violation_below_cost: 0.0,
                        fpha_violation_cost: 0.0,
                        evaporation_violation_cost: 0.0,
                        water_withdrawal_violation_cost: 1_000.0,
                        water_withdrawal_violation_pos_cost: 1_000.0,
                        water_withdrawal_violation_neg_cost: 1_000.0,
                        evaporation_violation_pos_cost: 0.0,
                        evaporation_violation_neg_cost: 0.0,
                        inflow_nonnegativity_cost: 1000.0,
                    },
                    ..Default::default()
                },
            ),
            make_hydro(
                EntityId(2),
                HydroSpec {
                    name: "H2".to_string(),
                    bus_id: EntityId(1),
                    max_storage_hm3: 200.0,
                    generation_model: HydroGenerationModel::ConstantProductivity,
                    max_turbined_m3s: 100.0,
                    max_generation_mw: 250.0,
                    penalties: HydroPenalties {
                        spillage_cost: 0.01,
                        diversion_cost: 0.0,
                        turbined_cost: 0.0,
                        storage_violation_below_cost: 0.0,
                        filling_target_violation_cost: 0.0,
                        turbined_violation_below_cost: 0.0,
                        outflow_violation_below_cost: 0.0,
                        outflow_violation_above_cost: 0.0,
                        generation_violation_below_cost: 0.0,
                        fpha_violation_cost: 0.0,
                        evaporation_violation_cost: 0.0,
                        water_withdrawal_violation_cost: 1_000.0,
                        water_withdrawal_violation_pos_cost: 1_000.0,
                        water_withdrawal_violation_neg_cost: 1_000.0,
                        evaporation_violation_pos_cost: 0.0,
                        evaporation_violation_neg_cost: 0.0,
                        inflow_nonnegativity_cost: 1000.0,
                    },
                    ..Default::default()
                },
            ),
            make_hydro(
                EntityId(3),
                HydroSpec {
                    name: "H3".to_string(),
                    bus_id: EntityId(1),
                    max_storage_hm3: 200.0,
                    generation_model: HydroGenerationModel::ConstantProductivity,
                    max_turbined_m3s: 100.0,
                    max_generation_mw: 250.0,
                    penalties: HydroPenalties {
                        spillage_cost: 0.01,
                        diversion_cost: 0.0,
                        turbined_cost: 0.0,
                        storage_violation_below_cost: 0.0,
                        filling_target_violation_cost: 0.0,
                        turbined_violation_below_cost: 0.0,
                        outflow_violation_below_cost: 0.0,
                        outflow_violation_above_cost: 0.0,
                        generation_violation_below_cost: 0.0,
                        fpha_violation_cost: 0.0,
                        evaporation_violation_cost: 0.0,
                        water_withdrawal_violation_cost: 1_000.0,
                        water_withdrawal_violation_pos_cost: 1_000.0,
                        water_withdrawal_violation_neg_cost: 1_000.0,
                        evaporation_violation_pos_cost: 0.0,
                        evaporation_violation_neg_cost: 0.0,
                        inflow_nonnegativity_cost: 1000.0,
                    },
                    ..Default::default()
                },
            ),
        ])
        .stages(stages)
        .inflow_models(inflow_models)
        .load_models(load_models)
        .bounds(bounds)
        .penalties(penalties)
        .build()
        .expect("three_hydro_system: valid");

    let result = build_stage_templates_resolving_layout(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("build ok");

    let t = &result.templates[0];

    // N=3, L=0, 1 block, no thermals/lines/evap/inflow-penalty:
    // state+aux: 3 storage + 3 storage_in + 1 theta = 7
    // turbine: 3, spillage: 3, diversion: 3
    // deficit: 1, excess: 1
    // inflow slack: 0 (no penalty config)
    // evap: 0
    // withdrawal slacks: 3 neg + 3 pos = 6
    // operational violation slacks: 4*3 = 12
    // Total = 7 + 3 + 3 + 3 + 1 + 1 + 6 + 12 = 36
    let expected_cols = 36_usize;
    assert_eq!(
        t.num_cols, expected_cols,
        "3-hydro system: num_cols should be {expected_cols}, got {}",
        t.num_cols
    );

    let g = &result.geometry_per_stage[0];
    for h in 0..system.hydros().len() {
        let col_neg = g.withdrawal_slack_neg_col(HydroSys::new(h));
        let col_pos = g.withdrawal_slack_pos_col(HydroSys::new(h));
        assert!(
            (t.col_upper[col_neg] - 5.0).abs() < 1e-15,
            "withdrawal slack neg column for hydro {h} must be capped at |T| = 5.0, got {}",
            t.col_upper[col_neg]
        );
        assert_eq!(
            t.col_upper[col_pos],
            f64::INFINITY,
            "withdrawal slack pos column for hydro {h} should be unbounded above (T > 0)"
        );
    }
}

/// Hydro `D` (id 2, canonical position 0) is the target. Hydro `P` (id 3, position 1) is
/// `PreFilling` at stage 0 and routes onto `D`. `D` prices its withdrawal slacks at 5000
/// and `P` at 7000, so the price of a transferred withdrawal shows which plant's slacks
/// carry it.
fn prefilling_pair_system(
    prefilling_withdrawal_m3s: f64,
    target_withdrawal_m3s: f64,
) -> novomodelo_core::System {
    use chrono::NaiveDate;
    use novomodelo_core::entities::hydro::HydroGenerationModel;
    use novomodelo_core::scenario::LoadModel;
    use novomodelo_core::temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, StageRiskConfig, StageStateConfig,
    };

    const N_STAGES: usize = 2;
    const HYDRO_IDS: [i32; 2] = [2, 3];

    let withdrawal_penalties = |cost: f64| HydroPenalties {
        spillage_cost: 0.01,
        diversion_cost: 0.0,
        turbined_cost: 0.0,
        storage_violation_below_cost: 0.0,
        filling_target_violation_cost: 0.0,
        turbined_violation_below_cost: 0.0,
        outflow_violation_below_cost: 0.0,
        outflow_violation_above_cost: 0.0,
        generation_violation_below_cost: 0.0,
        fpha_violation_cost: 0.0,
        evaporation_violation_cost: 0.0,
        water_withdrawal_violation_cost: cost,
        water_withdrawal_violation_pos_cost: cost,
        water_withdrawal_violation_neg_cost: cost,
        evaporation_violation_pos_cost: 0.0,
        evaporation_violation_neg_cost: 0.0,
        inflow_nonnegativity_cost: 1000.0,
    };
    let target_penalties = withdrawal_penalties(5_000.0);
    let prefilling_penalties = withdrawal_penalties(7_000.0);

    let bus = make_bus(
        EntityId(1),
        BusSpec {
            name: "B1".to_string(),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 500.0,
            }],
            ..Default::default()
        },
    );

    let hydros = HYDRO_IDS
        .iter()
        .map(|&id| {
            let is_target = id == HYDRO_IDS[0];
            make_hydro(
                EntityId(id),
                HydroSpec {
                    name: format!("H{id}"),
                    bus_id: EntityId(1),
                    downstream_id: (!is_target).then_some(EntityId(HYDRO_IDS[0])),
                    entry_stage_id: (!is_target).then_some(1),
                    max_storage_hm3: 200.0,
                    generation_model: HydroGenerationModel::ConstantProductivity,
                    max_turbined_m3s: 100.0,
                    max_generation_mw: 250.0,
                    penalties: if is_target {
                        target_penalties
                    } else {
                        prefilling_penalties
                    },
                    ..Default::default()
                },
            )
        })
        .collect::<Vec<_>>();

    let stages = (0..N_STAGES)
        .map(|i| {
            let month = i as u32 + 1;
            make_stage(
                i,
                StageSpec {
                    start_date: NaiveDate::from_ymd_opt(2024, month, 1).unwrap(),
                    end_date: NaiveDate::from_ymd_opt(2024, month + 1, 1).unwrap(),
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
                },
            )
        })
        .collect::<Vec<_>>();

    let inflow_models = HYDRO_IDS
        .iter()
        .flat_map(|&id| {
            (0..N_STAGES).map(move |i| InflowModel {
                hydro_id: EntityId(id),
                stage_id: i as i32,
                mean_m3s: 50.0,
                std_m3s: 10.0,
                ar_coefficients: vec![],
                residual_std_ratio: 1.0,
                annual: None,
            })
        })
        .collect::<Vec<_>>();

    let load_models = (0..N_STAGES)
        .map(|i| LoadModel {
            bus_id: EntityId(1),
            stage_id: i as i32,
            mean_mw: 100.0,
            std_mw: 0.0,
        })
        .collect::<Vec<_>>();

    let mut bounds = ResolvedBounds::new(
        &BoundsCountsSpec {
            n_hydros: HYDRO_IDS.len(),
            n_thermals: 0,
            n_lines: 0,
            n_pumping: 0,
            n_contracts: 0,
            n_stages: N_STAGES,
            k_max: 0,
        },
        &BoundsDefaults {
            hydro: HydroStageBounds {
                min_storage_hm3: 0.0,
                max_storage_hm3: 200.0,
                filling_min_rate_m3s: 0.0,
                water_withdrawal_m3s: 0.0,
            },
            hydro_block: HydroBlockBounds {
                max_turbined_m3s: 100.0,
                max_generation_mw: 250.0,
                ..Default::default()
            },
            thermal_block: ThermalBlockBounds {
                min_generation_mw: 0.0,
                max_generation_mw: 0.0,
                cost_per_mwh: 0.0,
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
    let mut penalties = ResolvedPenalties::new(
        &PenaltiesCountsSpec {
            n_hydros: HYDRO_IDS.len(),
            n_buses: 1,
            n_lines: 0,
            n_ncs: 0,
            n_stages: N_STAGES,
        },
        &PenaltiesDefaults {
            hydro: target_penalties,
            bus: BusStagePenalties { excess_cost: 0.0 },
            line: LineStagePenalties { exchange_cost: 0.0 },
            ncs: NcsStagePenalties {
                curtailment_cost: 0.0,
            },
        },
    );
    for stage in 0..N_STAGES {
        bounds.hydro_bounds_mut(0, stage).water_withdrawal_m3s = target_withdrawal_m3s;
        bounds.hydro_bounds_mut(1, stage).water_withdrawal_m3s = prefilling_withdrawal_m3s;
        *penalties.hydro_penalties_mut(1, stage) = prefilling_penalties;
    }

    SystemBuilder::new()
        .buses(vec![bus])
        .hydros(hydros)
        .stages(stages)
        .inflow_models(inflow_models)
        .load_models(load_models)
        .bounds(bounds)
        .penalties(penalties)
        .build()
        .expect("prefilling_pair_system: valid")
}

/// derived: `neg ∈ [0, T]` and `pos ∈ [0, ∞)` for `T > 0`; `neg ∈ [0, 0]` and
/// `pos ∈ [0, -T]` for `T < 0`; both `[0, 0]` at `T = 0`.
#[test]
fn withdrawal_slack_bounds_follow_effective_withdrawal_sign() {
    for (t, neg_upper, pos_upper) in [
        (10.0, 10.0, f64::INFINITY),
        (-5.0, 0.0, 5.0),
        (0.0, 0.0, 0.0),
    ] {
        let system = one_hydro_system_with_withdrawal(1, 0, t, 5_000.0);
        let result = build_stage_templates_resolving_layout(
            &system,
            no_penalty_config(),
            &PrecomputedPar::default(),
            &PrecomputedNormal::default(),
            &default_production(&system),
            &default_evaporation(&system),
            &ResolvedParameters::default(),
        )
        .expect("build ok");

        let template = &result.templates[0];
        let geometry = &result.geometry_per_stage[0];
        let col_neg = geometry.withdrawal_slack_neg_col(HydroSys::new(0));
        let col_pos = geometry.withdrawal_slack_pos_col(HydroSys::new(0));
        assert_eq!(template.col_lower[col_neg], 0.0, "T = {t}: neg lower");
        assert_eq!(template.col_upper[col_neg], neg_upper, "T = {t}: neg upper");
        assert_eq!(template.col_lower[col_pos], 0.0, "T = {t}: pos lower");
        assert_eq!(template.col_upper[col_pos], pos_upper, "T = {t}: pos upper");
    }
}

/// derived: `T_eff[D] = T[D] + T[P]` and `T_eff[P] = 0`, so `D` carries the bounds of the
/// sign table at `T_eff` and `P`'s own slacks are `[0, 0]`; a transferred withdrawal is
/// priced at `D`'s cost, `5000 * 744 / COST_SCALE_FACTOR`, never `P`'s.
#[test]
fn prefilling_withdrawal_moves_slack_bound_to_target() {
    let build = |prefilling_withdrawal_m3s: f64, target_withdrawal_m3s: f64| {
        let system = prefilling_pair_system(prefilling_withdrawal_m3s, target_withdrawal_m3s);
        build_stage_templates_resolving_layout(
            &system,
            no_penalty_config(),
            &PrecomputedPar::default(),
            &PrecomputedNormal::default(),
            &default_production(&system),
            &default_evaporation(&system),
            &ResolvedParameters::default(),
        )
        .expect("build ok")
    };
    let baseline = build(0.0, 0.0);

    let to_surplus = build(0.2, 0.07);
    let template = &to_surplus.templates[0];
    let geometry = &to_surplus.geometry_per_stage[0];
    let (d, p) = (HydroSys::new(0), HydroSys::new(1));
    assert_eq!(
        template.col_upper[geometry.withdrawal_slack_neg_col(d)].to_bits(),
        (0.07_f64 + 0.2).to_bits()
    );
    assert_eq!(
        template.col_upper[geometry.withdrawal_slack_pos_col(d)],
        f64::INFINITY
    );
    assert_eq!(
        template.col_upper[geometry.withdrawal_slack_neg_col(p)].to_bits(),
        0.0_f64.to_bits()
    );
    assert_eq!(
        template.col_upper[geometry.withdrawal_slack_pos_col(p)].to_bits(),
        0.0_f64.to_bits()
    );
    let target_cost = 5_000.0 * 744.0 / COST_SCALE_FACTOR;
    assert!(
        (template.objective[geometry.withdrawal_slack_neg_col(d)] - target_cost).abs() < 1e-12,
        "D's neg objective must be priced at D's own cost"
    );
    let prefilling_cost = 7_000.0 * 744.0 / COST_SCALE_FACTOR;
    assert!(
        (template.objective[geometry.withdrawal_slack_neg_col(p)] - prefilling_cost).abs() < 1e-12,
        "P's own neg objective must stay priced at P's cost"
    );
    assert_eq!(template.num_cols, baseline.templates[0].num_cols);

    let to_return = build(0.2, -0.5);
    let template = &to_return.templates[0];
    let geometry = &to_return.geometry_per_stage[0];
    assert_eq!(
        template.col_upper[geometry.withdrawal_slack_neg_col(d)].to_bits(),
        0.0_f64.to_bits()
    );
    assert_eq!(
        template.col_upper[geometry.withdrawal_slack_pos_col(d)].to_bits(),
        (-(-0.5_f64 + 0.2)).to_bits()
    );
    assert_eq!(template.num_cols, baseline.templates[0].num_cols);
}

const ZETA_744H: f64 = 744.0 * 3_600.0 / 1_000_000.0;

fn empty_storage_stage0(
    system: &novomodelo_core::System,
    production: &ProductionModelSet,
) -> (
    novomodelo_sddp::StageTemplates,
    novomodelo_solver::ActiveSolver,
) {
    use novomodelo_solver::SolverInterface;

    let result = build_stage_templates_resolving_layout(
        system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        production,
        &default_evaporation(system),
        &ResolvedParameters::default(),
    )
    .expect("build ok");
    let n_h = system.hydros().len();
    let state = novomodelo_sddp::test_support::state_layout(n_h, 0);
    let grid = result.geometry_per_stage[0].storage_boundary_grid();
    let incoming: Vec<usize> = (0..n_h)
        .map(|h| grid.col(&state, HydroSys::new(h), Boundary::Incoming))
        .collect();
    let mut solver = load_template_with_no_cuts(&result.templates[0]);
    solver.set_col_bounds(&incoming, &vec![0.0; n_h], &vec![0.0; n_h]);
    (result, solver)
}

/// derived: with `v_in = 0`, `v_out ≥ 0` and no inflow, `D`'s water row forces
/// `neg − pos − s − q − v_out/ζ = T_eff = 0.07 + 0.2`, and `neg ≤ T_eff` leaves the rest at 0.
#[test]
fn prefilling_withdrawal_closes_through_target_slack() {
    use novomodelo_solver::SolverInterface;

    let (feeder_t, target_t) = (0.2, 0.07);
    let system = prefilling_pair_system(feeder_t, target_t);
    let (result, mut solver) = empty_storage_stage0(&system, &production_set(&[1.0, 1.0], 2));
    let g = &result.geometry_per_stage[0];
    let (d, p) = (HydroSys::new(0), HydroSys::new(1));
    let view = solver
        .solve(None)
        .expect("the feeder's withdrawal must close through D's neg slack");

    let neg_d = view.primal[g.withdrawal_slack_neg_col(d)];
    assert!(
        (neg_d - (target_t + feeder_t)).abs() < 1e-9,
        "neg_D must carry T_eff, got {neg_d}"
    );
    for (name, col) in [
        ("pos_D", g.withdrawal_slack_pos_col(d)),
        ("s_D", g.spillage_col(d, BlockIdx::new(0))),
        ("q_D", g.turbine_col(HydroCell::new(0), BlockIdx::new(0))),
    ] {
        assert!(
            view.primal[col] <= 1e-9,
            "{name} must be 0, got {}",
            view.primal[col]
        );
    }
    for (name, col) in [
        ("neg_P", g.withdrawal_slack_neg_col(p)),
        ("pos_P", g.withdrawal_slack_pos_col(p)),
    ] {
        assert!(
            view.primal[col].abs() < 1e-12,
            "{name} must be 0, got {}",
            view.primal[col]
        );
    }
}

/// derived: twin with no feeder withdrawal, `T_eff = 0.07 + 0.0`, so `neg_D` covers only `D`'s own.
#[test]
fn prefilling_withdrawal_closes_target_only_without_feeder_withdrawal() {
    use novomodelo_solver::SolverInterface;

    let target_t = 0.07;
    let system = prefilling_pair_system(0.0, target_t);
    let (result, mut solver) = empty_storage_stage0(&system, &production_set(&[1.0, 1.0], 2));
    let g = &result.geometry_per_stage[0];
    let view = solver
        .solve(None)
        .expect("D's own withdrawal must close through its neg slack");

    let neg_d = view.primal[g.withdrawal_slack_neg_col(HydroSys::new(0))];
    assert!(
        (neg_d - target_t).abs() < 1e-9,
        "neg_D must carry D's own T only, got {neg_d}"
    );
}

/// derived: twin with 1 m³/s injected into `D`'s row, `q + s + pos − neg + v_out/ζ = 1 − T_eff`;
/// turbining saves 500 $/MWh at `ρ = 1`, above the 0.01 spillage cost and below the 5000
/// slack price, so `q_D = 1 − (0.07 + 0.2)` and both slacks stay at 0.
#[test]
fn prefilling_withdrawal_closes_through_turbine_with_inflow() {
    use novomodelo_solver::SolverInterface;

    let (feeder_t, target_t, inflow) = (0.2, 0.07, 1.0);
    let system = prefilling_pair_system(feeder_t, target_t);
    let (result, mut solver) = empty_storage_stage0(&system, &production_set(&[1.0, 1.0], 2));
    let g = &result.geometry_per_stage[0];
    let d = HydroSys::new(0);
    let row = g.water_balance_row(d, BlockIdx::new(0));
    let rhs = result.templates[0].row_lower[row] + ZETA_744H * inflow;
    solver.set_row_bounds(&[row], &[rhs], &[rhs]);
    let view = solver
        .solve(None)
        .expect("D must turbine the injected inflow net of T_eff");

    for (name, col) in [
        ("neg_D", g.withdrawal_slack_neg_col(d)),
        ("pos_D", g.withdrawal_slack_pos_col(d)),
    ] {
        assert!(
            view.primal[col] <= 1e-9,
            "{name} must be 0, got {}",
            view.primal[col]
        );
    }
    let q_d = view.primal[g.turbine_col(HydroCell::new(0), BlockIdx::new(0))];
    assert!(
        (q_d - (inflow - (target_t + feeder_t))).abs() < 1e-9,
        "q_D must turbine the inflow net of T_eff, got {q_d}"
    );
}

/// derived: `T = −5` pins `neg ∈ [0, 0]`, so the row `v_out/ζ + q + s + pos = −T` with
/// `v_in = 0` turbines `q = 5` at 500 $/MWh and the deficit is `100 − ρ·q`. Power guard:
/// `q` and the deficit are basic, so `ζ·y_w = −ρ·500·744/K` and
/// `rc_neg = 744·c/K + ζ·y_w = (744/K)·(c − 500·ρ) = −0.371256 < 0`: the `[0, 0]` bound binds.
#[test]
fn negative_withdrawal_cannot_create_water() {
    use novomodelo_solver::SolverInterface;

    let (withdrawal_m3s, slack_price, rho, load_mw, deficit_price) = (-5.0, 1.0, 1.0, 100.0, 500.0);
    let system = one_hydro_system_with_withdrawal(1, 0, withdrawal_m3s, slack_price);
    let (result, mut solver) = empty_storage_stage0(&system, &production_set(&[rho], 1));
    let template = &result.templates[0];
    let geometry = &result.geometry_per_stage[0];
    let hydro = HydroSys::new(0);
    let col_neg = geometry.withdrawal_slack_neg_col(hydro);
    let col_deficit = geometry.deficit_col(BusSys::new(0), 0, BlockIdx::new(0), 1);
    let water_row = geometry.water_balance_row(hydro, BlockIdx::new(0));
    let view = solver
        .solve(None)
        .expect("the returned water must close the balance without neg");

    let neg = view.primal[col_neg];
    assert!(neg.abs() < 1e-12, "neg must be 0, got {neg}");
    let pos = view.primal[geometry.withdrawal_slack_pos_col(hydro)];
    assert!(pos <= 1e-9, "pos must be 0, got {pos}");
    let turbined = view.primal[geometry.turbine_col(HydroCell::new(0), BlockIdx::new(0))];
    assert!(
        (turbined - (-withdrawal_m3s)).abs() < 1e-9,
        "q must turbine only the returned water, got {turbined}"
    );
    let deficit = view.primal[col_deficit];
    assert!(
        (deficit - (load_mw - rho * (-withdrawal_m3s))).abs() < 1e-9,
        "deficit must be the load net of the returned water's generation, got {deficit}"
    );

    let rc_neg = view.reduced_costs[col_neg];
    let y_w = view.dual[water_row];
    let kkt = template.objective[col_neg] - (-ZETA_744H) * y_w;
    assert!(
        (rc_neg - kkt).abs() < 1e-9,
        "KKT: rc_neg = {rc_neg}, c_neg + ζ·y_w = {kkt}"
    );
    assert!(
        rc_neg < 0.0,
        "neg's [0, 0] bound must bind, rc_neg = {rc_neg}"
    );
    let closed_form = (744.0 / COST_SCALE_FACTOR) * (slack_price - deficit_price * rho);
    assert!(
        (rc_neg - closed_form).abs() <= 1e-9 * closed_form.abs(),
        "rc_neg = {rc_neg}, closed form = {closed_form}, deficit objective = {}",
        template.objective[col_deficit]
    );
}
