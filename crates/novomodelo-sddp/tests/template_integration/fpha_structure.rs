//! `fpha_structure` section tests.

use super::*;
use novomodelo_core::TailraceFlowInfluence;
use novomodelo_sddp::indexer::{BlockIdx, FphaCellLocal, HydroCell, HydroSys};
use novomodelo_solver::{ActiveSolver, RowBatch, SolverInterface};

fn empty_cut_batch() -> RowBatch {
    RowBatch {
        num_rows: 0,
        row_starts: vec![0_i32],
        col_indices: vec![],
        values: vec![],
        row_lower: vec![],
        row_upper: vec![],
    }
}

#[test]
fn fpha_ac1_dimensions_one_fpha_hydro_five_planes() {
    let (system, production) = one_fpha_hydro_system(5);

    let fpha_result = build_stage_templates_resolving_layout(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &production,
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("FPHA system ok");

    let const_result = build_stage_templates_resolving_layout(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("constant productivity ok");

    let fpha_tmpl = &fpha_result.templates[0];
    let const_tmpl = &const_result.templates[0];

    assert_eq!(
        fpha_tmpl.num_cols,
        const_tmpl.num_cols + 2,
        "FPHA adds 1 generation and 1 slack column (1 cell * 1 block)"
    );
    assert_eq!(
        fpha_tmpl.num_rows,
        const_tmpl.num_rows + 5,
        "FPHA adds exactly 5 constraint rows (5 planes * 1 block)"
    );
}

#[test]
fn fpha_plant_without_capacity_builds_no_fpha_structure() {
    const N_PLANES: usize = 5;
    /// derived: 1 cell * 1 block for each of the generation and plane-slack columns.
    const FPHA_COLUMNS: usize = 2;
    /// derived: `N_PLANES` planes * 1 block.
    const FPHA_ROWS: usize = N_PLANES;

    let build = |max_turbined_m3s: f64, max_generation_mw: f64, label: &str| {
        let (system, production) = one_fpha_hydro_system_with_block_bounds(
            N_PLANES,
            HydroBlockBounds {
                max_turbined_m3s,
                max_generation_mw,
                ..Default::default()
            },
        );
        build_stage_templates_resolving_layout(
            &system,
            no_penalty_config(),
            &PrecomputedPar::default(),
            &PrecomputedNormal::default(),
            &production,
            &default_evaporation(&system),
            &ResolvedParameters::default(),
        )
        .unwrap_or_else(|err| panic!("{label}: template build failed: {err:?}"))
    };
    let turbine_bounds = |result: &novomodelo_sddp::StageTemplates| {
        let col = result.geometry_per_stage[0].turbine_col(HydroCell::new(0), BlockIdx::new(0));
        let tmpl = &result.templates[0];
        (tmpl.col_lower[col], tmpl.col_upper[col])
    };

    let twin = build(150.0, 300.0, "FPHA twin");
    assert_eq!(
        twin.geometry_per_stage[0].fpha_hydro_indices,
        [HydroSys::new(0)],
        "the twin keeps the plant in FPHA, or the comparison below is Dormant against Dormant"
    );
    assert_eq!(
        turbine_bounds(&twin).1,
        150.0,
        "the twin's turbine column is open"
    );

    for (max_turbined_m3s, max_generation_mw, label) in [
        (150.0, 0.0, "generation limit 0"),
        (0.0, 300.0, "turbined limit 0"),
        (0.0, 0.0, "both limits 0"),
    ] {
        let result = build(max_turbined_m3s, max_generation_mw, label);
        let geometry = &result.geometry_per_stage[0];

        assert_eq!(
            result.templates[0].num_cols + FPHA_COLUMNS,
            twin.templates[0].num_cols,
            "{label}: no generation and no plane-slack column"
        );
        assert_eq!(
            result.templates[0].num_rows + FPHA_ROWS,
            twin.templates[0].num_rows,
            "{label}: no plane rows"
        );
        assert!(
            geometry.fpha_hydro_indices.is_empty(),
            "{label}: the plant is not an FPHA plant at this stage"
        );
        assert!(geometry.generation.is_empty(), "{label}: generation");
        assert!(geometry.fpha.is_empty(), "{label}: plane rows");
        assert!(geometry.fpha_slack.is_empty(), "{label}: plane slack");
        assert_eq!(
            turbine_bounds(&result),
            (0.0, 0.0),
            "{label}: the turbine column is frozen"
        );
    }
}

#[test]
fn fpha_slack_column_structure_one_cell_five_planes() {
    let n_planes = 5;
    let (system, production) = one_fpha_hydro_system(n_planes);
    let result = build_stage_templates_resolving_layout(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &production,
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("FPHA system ok");

    let tmpl = &result.templates[0];
    let geometry = &result.geometry_per_stage[0];
    assert_eq!(geometry.fpha_slack.len(), 1, "1 cell * 1 block");
    let col = geometry.fpha_slack_col(FphaCellLocal::new(0), BlockIdx::new(0));

    assert_eq!(tmpl.col_lower[col], 0.0);
    assert_eq!(tmpl.col_upper[col], f64::INFINITY);

    let price = system.penalties().hydro_penalties(0, 0).fpha_violation_cost;
    let hours = 744.0_f64;
    assert_eq!(
        tmpl.objective[col].to_bits(),
        (price * hours / COST_SCALE_FACTOR).to_bits(),
        "price times block hours, divided by the cost scale"
    );

    let entries = entries_for_col(tmpl, col);
    let expected_rows: Vec<usize> = (geometry.fpha.start..geometry.fpha.start + n_planes).collect();
    let mut rows: Vec<usize> = entries.iter().map(|&(row, _)| row).collect();
    rows.sort_unstable();
    assert_eq!(
        rows, expected_rows,
        "one entry in each plane row, none elsewhere"
    );
    assert!(
        entries.iter().all(|&(_, value)| value == -1.0),
        "every plane row carries -1 on the slack: {entries:?}"
    );
}

#[test]
fn fpha_slack_absent_without_fpha() {
    let system = one_hydro_system(1, 2);
    let result = build_stage_templates_resolving_layout(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(&system),
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("constant productivity ok");

    assert!(result.geometry_per_stage[0].fpha_slack.is_empty());
}

#[test]
fn fpha_ac2_generation_column_entries() {
    let n_planes = 5;
    let (system, production) = one_fpha_hydro_system(n_planes);

    let result = build_stage_templates_resolving_layout(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &production,
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("FPHA system ok");

    let tmpl = &result.templates[0];

    // N=1, L=0 → n_state = 1, decision_start = 3, col_generation_start = 8.
    // (turbine[3..4], spillage[4..5], diversion[5..6], deficit[6..7], excess[7..8], generation[8])
    let col_g = 8_usize;

    // row_fpha_start = 2 (= N water balance + B*1 load balance)
    let row_fpha_start = 2_usize;

    for p in 0..n_planes {
        let row = row_fpha_start + p;
        let coeff = csc_entry(tmpl, col_g, row)
            .unwrap_or_else(|| panic!("generation column missing entry in FPHA row {row}"));
        assert!(
            (coeff - 1.0).abs() < 1e-12,
            "generation col FPHA row {row}: expected +1.0, got {coeff}"
        );
    }

    let row_lb = 1_usize; // load balance row
    let lb_coeff = csc_entry(tmpl, col_g, row_lb)
        .unwrap_or_else(|| panic!("generation column missing entry in load balance row {row_lb}"));
    assert!(
        (lb_coeff - 1.0).abs() < 1e-12,
        "generation col load balance row: expected +1.0, got {lb_coeff}"
    );
}

/// The `v_in` column carries `-gamma_v/2` in every FPHA row; gamma_v = 0.5 → -0.25.
#[test]
fn fpha_ac3_v_in_column_entries() {
    let n_planes = 5;
    let (system, production) = one_fpha_hydro_system(n_planes);

    let result = build_stage_templates_resolving_layout(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &production,
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("FPHA system ok");

    let tmpl = &result.templates[0];

    let col_v_in = 1_usize; // N=1, L=0: storage_in follows storage_out
    let row_fpha_start = 2_usize; // N water balance + B*K load balance
    let expected = -0.5_f64 / 2.0;

    for p in 0..n_planes {
        let row = row_fpha_start + p;
        let coeff = csc_entry(tmpl, col_v_in, row)
            .unwrap_or_else(|| panic!("v_in column missing entry in FPHA row {row}"));
        assert!(
            (coeff - expected).abs() < 1e-12,
            "v_in col FPHA row {row}: expected {expected}, got {coeff}"
        );
    }
}

/// The outgoing-storage column `v` carries `-gamma_v/2` in every FPHA row; -0.25 here.
#[test]
fn fpha_ac4_v_out_column_entries() {
    let n_planes = 5;
    let (system, production) = one_fpha_hydro_system(n_planes);

    let result = build_stage_templates_resolving_layout(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &production,
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("FPHA system ok");

    let tmpl = &result.templates[0];

    let col_v = 0_usize; // outgoing storage is column 0 for N=1
    let row_fpha_start = 2_usize; // N water balance + B*K load balance
    let expected = -0.5_f64 / 2.0;

    for p in 0..n_planes {
        let row = row_fpha_start + p;
        let coeff = csc_entry(tmpl, col_v, row)
            .unwrap_or_else(|| panic!("v column missing entry in FPHA row {row}"));
        assert!(
            (coeff - expected).abs() < 1e-12,
            "v col FPHA row {row}: expected {expected}, got {coeff}"
        );
    }
}

#[test]
fn fpha_ac5_mixed_system_load_balance_uses_generation_col() {
    let (system, production) = four_hydro_mixed_system();

    let result = build_stage_templates_resolving_layout(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &production,
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("mixed FPHA/constant system ok");

    let tmpl = &result.templates[0];

    // Base (constant) num_cols = 9 + 4*1*3 (turbine+spillage+diversion per hydro/block)
    // + 1*1*2 = 23; + 2 FPHA generation = 25; + 2*N=8 withdrawal = 33;
    // + 4*N=16 op-violation slacks = 49; + 2 FPHA slack = 51.
    // num_rows (no state-fixing rows) = N water balance(4) + load balance(1)
    // = 5; + 2*3 FPHA = 11; + 4*N=16 op-violation = 27.
    assert_eq!(
        tmpl.num_cols, 51,
        "4-hydro mixed system: num_cols should be 51 (includes diversion, 2*4 withdrawal, operational slacks and FPHA slacks)"
    );
    assert_eq!(
        tmpl.num_rows, 27,
        "4-hydro mixed system: num_rows should be 27 (no state-fixing rows)"
    );

    let row_lb = 4_usize; // N water balance(4) + bus_blk_idx(0)

    // FPHA hydro h_idx=2 (local_idx=0): generation col 23 (= 19 + 4 diversion cols).
    let col_g_fpha0 = 23_usize;
    let g0_lb_coeff = csc_entry(tmpl, col_g_fpha0, row_lb).unwrap_or_else(|| {
        panic!("FPHA hydro 0 generation column missing entry in load balance row {row_lb}")
    });
    assert!(
        (g0_lb_coeff - 1.0).abs() < 1e-12,
        "FPHA hydro 0 load balance: expected +1.0, got {g0_lb_coeff}"
    );

    let col_g_fpha1 = 24_usize; // FPHA hydro h_idx=3 (local_idx=1)
    let g1_lb_coeff = csc_entry(tmpl, col_g_fpha1, row_lb).unwrap_or_else(|| {
        panic!("FPHA hydro 1 generation column missing entry in load balance row {row_lb}")
    });
    assert!(
        (g1_lb_coeff - 1.0).abs() < 1e-12,
        "FPHA hydro 1 load balance: expected +1.0, got {g1_lb_coeff}"
    );

    let col_turb_const = 9_usize; // constant hydro h_idx=0: col_turbine_start
    let turb_lb_coeff = csc_entry(tmpl, col_turb_const, row_lb);
    assert!(
        turb_lb_coeff.is_some(),
        "constant hydro 0 turbine col must appear in load balance row"
    );
    // Constant hydro enters the power-balance matrix with rho (the productivity
    // scalar), NOT rho * block_hours — block_hours scales cost objectives only.
    // Hydro 0 has productivity 2.5 MW/(m³/s), so the coefficient is 2.5.
    let expected_rho_coeff = 2.5_f64;
    assert!(
        (turb_lb_coeff.unwrap() - expected_rho_coeff).abs() < 1e-12,
        "constant hydro 0 turbine: expected rho = {expected_rho_coeff}, got {turb_lb_coeff:?}"
    );
}

#[test]
fn fpha_solve_one_hydro_optimal() {
    let (system, production) = fpha_solve_system();
    let result = build_stage_templates_resolving_layout(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &production,
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("FPHA template build must succeed");

    let template = &result.templates[0];
    let mut solver = ActiveSolver::new().expect("ActiveSolver::new must succeed");
    solver.load_model(template);
    solver.add_rows(&empty_cut_batch());

    let v_in = 100.0_f64;
    solver.set_col_bounds(&[1], &[v_in], &[v_in]);

    let view = solver
        .solve(None)
        .expect("FPHA LP must be feasible and optimal");

    let col_g = 8_usize;
    let generation = view.primal[col_g];
    assert!(
        generation > 0.0,
        "FPHA generation must be strictly positive, got {generation}"
    );

    let col_slack =
        result.geometry_per_stage[0].fpha_slack_col(FphaCellLocal::new(0), BlockIdx::new(0));
    let slack = view.primal[col_slack];
    assert!(
        slack.abs() < 1e-9,
        "the FPHA plane slack must be zero while the planes are met, got {slack}"
    );
}

/// After solving, every plane holds within 1e-6:
/// `g <= intercept + gamma_v * v_avg + gamma_q * q + gamma_s * s`,
/// `v_avg = (v + v_in) / 2`.
#[test]
fn fpha_solve_hyperplane_constraints_hold() {
    let (system, production) = fpha_solve_system();

    // Extract planes before moving production into build_stage_templates_resolving_layout.
    let planes = match production.model(0, 0) {
        ResolvedProductionModel::Fpha { planes, .. } => planes.clone(),
        ResolvedProductionModel::ConstantProductivity { .. } => {
            panic!("expected Fpha model")
        }
    };

    let result = build_stage_templates_resolving_layout(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &production,
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("FPHA template build must succeed");

    let template = &result.templates[0];
    let mut solver = ActiveSolver::new().expect("ActiveSolver::new must succeed");
    solver.load_model(template);
    solver.add_rows(&empty_cut_batch());

    let v_in = 100.0_f64;
    solver.set_col_bounds(&[1], &[v_in], &[v_in]);

    let view = solver.solve(None).expect("FPHA LP must solve to optimal");
    let primal = view.primal;

    let col_v = 0_usize;
    let col_v_in = 1_usize;
    let col_q = 3_usize;
    let col_s = 4_usize;
    let col_g = 8_usize;

    let g = primal[col_g];
    let v = primal[col_v];
    let v_in_sol = primal[col_v_in];
    let q = primal[col_q];
    let s = primal[col_s];
    let v_avg = f64::midpoint(v, v_in_sol);

    for (p_idx, plane) in planes.iter().enumerate() {
        let rhs = plane.intercept + plane.gamma_v * v_avg + plane.gamma_q * q + plane.gamma_s * s;
        assert!(
            g <= rhs + 1e-6,
            "FPHA plane {p_idx}: g={g} must be <= rhs={rhs} \
             (intercept={intercept}, gamma_v={gamma_v}, v_avg={v_avg}, \
              gamma_q={gamma_q}, q={q}, gamma_s={gamma_s}, s={s})",
            intercept = plane.intercept,
            gamma_v = plane.gamma_v,
            gamma_q = plane.gamma_q,
            gamma_s = plane.gamma_s,
        );
    }
}

/// The reduced cost of the pinned `storage_in` column
/// differs between FPHA and constant productivity. The `-gamma_v/2` FPHA entries
/// on the `v_in` column propagate through the simplex dual to that reduced cost.
///
/// The tight planes (`intercept=0, gamma_v=0.5, gamma_q=1.0, gamma_s=0.0`) keep
/// the FPHA capacity below the 200 MW load so the constraint binds at the optimum
/// — necessary for a non-zero dual. With K=1, `zeta`≈2.678 and `v_in`=100 hm³:
/// - FPHA row: `g <= 0.25*(v + v_in) + q`
/// - Water balance: `v = v_in + inflow*zeta - (q+s)*zeta`
/// - `q_max = v_in/zeta + inflow ≈ 37.3 + 80 = 117.3 m³/s` (at `v`=0)
/// - `g_max ≈ 0.25*v_in + q_max ≈ 25 + 117.3 = 142.3 MW < 200 MW`
///
/// So the optimizer covers the shortfall with costly deficit and extra `v_in`
/// lowers cost → FPHA dual < 0. Constant productivity gives `rho`=0
/// (`default_from_system`), generation is `v_in`-independent → dual = 0.
#[test]
fn fpha_solve_incoming_storage_reduced_cost_differs_from_constant() {
    let (system, _) = one_fpha_hydro_system(1);

    let tight_planes = vec![FphaPlane {
        intercept: 0.0,
        gamma_v: 0.5,
        gamma_q: 1.0,
        gamma_s: 0.0,
    }];
    let fpha_production = ProductionModelSet::new(
        vec![vec![ResolvedProductionModel::Fpha {
            planes: tight_planes,
        }]],
        system.hydros(),
        1,
    );

    let fpha_result = build_stage_templates_resolving_layout(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &fpha_production,
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("FPHA template build must succeed");

    let const_production = default_production(&system);
    let const_result = build_stage_templates_resolving_layout(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &const_production,
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("constant productivity template build must succeed");

    let solve_and_get_storage_dual = |template: &novomodelo_solver::StageTemplate| -> f64 {
        let mut solver = ActiveSolver::new().expect("ActiveSolver::new must succeed");
        solver.load_model(template);
        solver.add_rows(&empty_cut_batch());
        // Storage is pinned via column bounds: col 0 = storage_out, 1 = storage_in.
        let col_storage_in = 1_usize;
        let v_in = 100.0_f64;
        solver.set_col_bounds(&[col_storage_in], &[v_in], &[v_in]);
        let view = solver.solve(None).expect("LP must solve to optimal");
        view.reduced_costs[col_storage_in]
    };

    let fpha_dual = solve_and_get_storage_dual(&fpha_result.templates[0]);
    let const_dual = solve_and_get_storage_dual(&const_result.templates[0]);

    assert!(
        const_dual.abs() < 1e-6,
        "constant-productivity dual must be ~0, got {const_dual}"
    );

    assert!(
        fpha_dual.abs() > 1e-6,
        "FPHA incoming-storage reduced cost must be non-zero (FPHA v_in contribution \
         must be present), got {fpha_dual}"
    );

    assert_ne!(
        (fpha_dual * 1e6).round(),
        (const_dual * 1e6).round(),
        "incoming-storage reduced cost must differ between FPHA ({fpha_dual}) and \
         constant-productivity ({const_dual})"
    );
}

#[test]
fn fpha_solve_mixed_system_optimal() {
    let (system, production) = four_hydro_mixed_system();

    let result = build_stage_templates_resolving_layout(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &production,
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("mixed FPHA/constant system template build must succeed");

    let template = &result.templates[0];
    let mut solver = ActiveSolver::new().expect("ActiveSolver::new must succeed");
    solver.load_model(template);
    solver.add_rows(&empty_cut_batch());

    solver.set_col_bounds(
        &[4, 5, 6, 7],
        &[100.0, 100.0, 100.0, 100.0],
        &[100.0, 100.0, 100.0, 100.0],
    );

    let view = solver
        .solve(None)
        .expect("mixed FPHA LP must be feasible and optimal");

    assert!(
        view.objective.is_finite(),
        "objective must be finite, got {}",
        view.objective
    );

    let col_g0 = 23_usize; // FPHA generation variables for hydros 2 and 3
    let col_g1 = 24_usize;
    assert!(
        view.primal[col_g0] >= 0.0,
        "FPHA hydro 0 generation must be non-negative, got {}",
        view.primal[col_g0]
    );
    assert!(
        view.primal[col_g1] >= 0.0,
        "FPHA hydro 1 generation must be non-negative, got {}",
        view.primal[col_g1]
    );
}

/// derived: the mixed fixture's plane slope `γ_s`; each plant has one cell, so `σ_c = 1`.
const LATERAL_GAMMA_S: f64 = 0.1;

/// derived: planes per FPHA plant in the mixed fixture, one block.
const LATERAL_PLANES: usize = 3;

fn tailrace_flow(own: f64, outflow: &[(i32, f64)], incremental: &[(i32, f64)]) -> TailraceFlow {
    let influences = |list: &[(i32, f64)]| {
        list.iter()
            .map(|&(id, factor)| TailraceFlowInfluence {
                hydro_id: EntityId(id),
                factor,
            })
            .collect()
    };
    TailraceFlow {
        own_outflow_factor: own,
        outflow_influences: influences(outflow),
        incremental_inflow_influences: influences(incremental),
    }
}

fn mixed_templates(
    flow_102: TailraceFlow,
    method: InflowNonNegativityMethod,
) -> novomodelo_sddp::StageTemplates {
    let (system, production) = four_hydro_mixed_system_with(flow_102);
    build_stage_templates_resolving_layout(
        &system,
        method,
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &production,
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("mixed FPHA/constant system with a tailrace declaration builds")
}

/// Hydro 102's plane rows: the first FPHA plant of the mixed fixture.
fn plane_rows_of_102(
    geometry: &novomodelo_sddp::lp::builder::StageGeometry,
) -> std::ops::Range<usize> {
    geometry.fpha.start..geometry.fpha.start + LATERAL_PLANES
}

fn assert_entry(got: Option<f64>, want: f64, what: &str) {
    let got = got.unwrap_or_else(|| panic!("{what}: no entry, expected {want}"));
    assert!(
        (got - want).abs() < 1e-12,
        "{what}: expected {want}, got {got}"
    );
}

fn coupling_entries(result: &novomodelo_sddp::StageTemplates, hydro: usize) -> Vec<(usize, f64)> {
    result.inflow_coupling_per_stage[0]
        .hydro_entries(HydroSys::new(hydro))
        .collect()
}

/// Hydro 102 declares own factor 1.5, outflow of 100 at 0.8 and incremental of 101 at 0.5.
#[test]
fn fpha_lateral_entries_carry_the_scaled_plane_slope_on_every_influencing_column() {
    let result = mixed_templates(
        tailrace_flow(1.5, &[(100, 0.8)], &[(101, 0.5)]),
        no_penalty_config(),
    );
    let tmpl = &result.templates[0];
    let geometry = &result.geometry_per_stage[0];
    let blk = BlockIdx::new(0);
    let own_spillage = geometry.spillage_col(HydroSys::new(2), blk);
    let influenced = [
        ("100 turbine", geometry.turbine_col(HydroCell::new(0), blk)),
        ("100 spillage", geometry.spillage_col(HydroSys::new(0), blk)),
    ];

    for row in plane_rows_of_102(geometry) {
        assert_entry(
            csc_entry(tmpl, own_spillage, row),
            -LATERAL_GAMMA_S * 1.5,
            &format!("102 spillage, row {row}"),
        );
        for (name, col) in influenced {
            assert_entry(
                csc_entry(tmpl, col, row),
                -LATERAL_GAMMA_S * 0.8,
                &format!("{name}, row {row}"),
            );
        }
    }
    let plane_rows = plane_rows_of_102(geometry);
    for row in plane_rows.end..plane_rows.end + LATERAL_PLANES {
        for (name, col) in influenced {
            assert_eq!(
                csc_entry(tmpl, col, row),
                None,
                "{name} must carry nothing on 103's plane row {row}"
            );
        }
    }
}

#[test]
fn fpha_lateral_inflow_records_the_coupling_and_the_truncation_slack() {
    let result = mixed_templates(
        tailrace_flow(1.5, &[(100, 0.8)], &[(101, 0.5)]),
        InflowNonNegativityMethod::Truncation,
    );
    let tmpl = &result.templates[0];
    let geometry = &result.geometry_per_stage[0];
    let recorded = coupling_entries(&result, 1);
    let slack = geometry.inflow_slack_col(HydroSys::new(1));

    for row in plane_rows_of_102(geometry) {
        let hits: Vec<f64> = recorded
            .iter()
            .filter(|&&(r, _)| r == row)
            .map(|&(_, c)| c)
            .collect();
        assert_eq!(hits.len(), 1, "one record of hydro 101 on plane row {row}");
        assert_entry(
            Some(hits[0]),
            -LATERAL_GAMMA_S * 0.5,
            &format!("101 record, row {row}"),
        );
        assert_entry(
            csc_entry(tmpl, slack, row),
            -LATERAL_GAMMA_S * 0.5,
            &format!("101 inflow slack, row {row}"),
        );
    }
}

#[test]
fn fpha_lateral_inflow_skips_the_slack_outside_truncation() {
    for method in [
        InflowNonNegativityMethod::Penalty,
        InflowNonNegativityMethod::None,
    ] {
        let result = mixed_templates(tailrace_flow(1.5, &[(100, 0.8)], &[(101, 0.5)]), method);
        let tmpl = &result.templates[0];
        let geometry = &result.geometry_per_stage[0];
        let recorded = coupling_entries(&result, 1);

        for row in plane_rows_of_102(geometry) {
            let record = recorded.iter().find(|&&(r, _)| r == row).copied();
            assert_entry(
                record.map(|(_, c)| c),
                -LATERAL_GAMMA_S * 0.5,
                &format!("{method:?}: 101 record, row {row}"),
            );
            if !geometry.inflow_slack.is_empty() {
                let slack = geometry.inflow_slack_col(HydroSys::new(1));
                assert_eq!(
                    csc_entry(tmpl, slack, row),
                    None,
                    "{method:?}: 101 inflow slack must carry nothing on plane row {row}"
                );
            }
        }
    }
}

/// Hydro 102 declares its own incremental inflow, own factor 1.0 and no outflow influence.
#[test]
fn fpha_lateral_self_inflow_couples_the_plants_own_incremental() {
    let declared = || tailrace_flow(1.0, &[], &[(102, 0.5)]);

    for method in [
        InflowNonNegativityMethod::Truncation,
        InflowNonNegativityMethod::Penalty,
    ] {
        let result = mixed_templates(declared(), method);
        let undeclared = mixed_templates(TailraceFlow::default(), method);
        let tmpl = &result.templates[0];
        let geometry = &result.geometry_per_stage[0];
        let plane_rows = plane_rows_of_102(geometry);
        let slack = geometry.inflow_slack_col(HydroSys::new(2));
        let recorded = coupling_entries(&result, 2);

        for row in plane_rows.clone() {
            let hits: Vec<f64> = recorded
                .iter()
                .filter(|&&(r, _)| r == row)
                .map(|&(_, c)| c)
                .collect();
            assert_eq!(hits.len(), 1, "{method:?}: one record on plane row {row}");
            assert_entry(
                Some(hits[0]),
                -LATERAL_GAMMA_S * 0.5,
                &format!("{method:?}: 102 record, row {row}"),
            );
            let slack_entry = csc_entry(tmpl, slack, row);
            if method == InflowNonNegativityMethod::Truncation {
                assert_entry(
                    slack_entry,
                    -LATERAL_GAMMA_S * 0.5,
                    &format!("{method:?}: 102 inflow slack, row {row}"),
                );
            } else {
                assert_eq!(
                    slack_entry, None,
                    "{method:?}: 102 inflow slack must carry nothing on plane row {row}"
                );
            }
        }

        let outside_plane_rows: Vec<(usize, u64)> = recorded
            .iter()
            .filter(|&&(r, _)| !plane_rows.contains(&r))
            .map(|&(r, c)| (r, c.to_bits()))
            .collect();
        let undeclared_entries: Vec<(usize, u64)> = coupling_entries(&undeclared, 2)
            .iter()
            .map(|&(r, c)| (r, c.to_bits()))
            .collect();
        assert_eq!(
            outside_plane_rows, undeclared_entries,
            "{method:?}: 102's entries off its plane rows are the undeclared system's"
        );
    }
}

#[test]
fn fpha_zero_factor_declaration_builds_the_undeclared_template() {
    let method = InflowNonNegativityMethod::Truncation;
    let zeroed = mixed_templates(tailrace_flow(1.0, &[(100, 0.0)], &[(101, 0.0)]), method);
    let undeclared = mixed_templates(TailraceFlow::default(), method);
    let (zeroed_tmpl, undeclared_tmpl) = (&zeroed.templates[0], &undeclared.templates[0]);

    assert_eq!(zeroed_tmpl.col_starts, undeclared_tmpl.col_starts);
    assert_eq!(zeroed_tmpl.row_indices, undeclared_tmpl.row_indices);
    let bits = |values: &[f64]| values.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&zeroed_tmpl.values), bits(&undeclared_tmpl.values));
    assert_eq!(
        bits(&zeroed_tmpl.row_lower),
        bits(&undeclared_tmpl.row_lower)
    );
    assert_eq!(
        bits(&zeroed_tmpl.row_upper),
        bits(&undeclared_tmpl.row_upper)
    );
    for hydro in 0..4 {
        let entry_bits = |result: &novomodelo_sddp::StageTemplates| {
            coupling_entries(result, hydro)
                .iter()
                .map(|&(r, c)| (r, c.to_bits()))
                .collect::<Vec<_>>()
        };
        assert_eq!(
            entry_bits(&zeroed),
            entry_bits(&undeclared),
            "hydro {hydro}: inflow-coupling entries"
        );
    }
}
