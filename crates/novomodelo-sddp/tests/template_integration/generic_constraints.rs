//! `generic_constraints` section tests.

use super::*;

use novomodelo_sddp::test_support::assert_all_templates_byte_identical;

/// Parse `json` (a `generic_constraints.json` body) through the real
/// `novomodelo_io::constraints::parse_generic_constraints` path — the same loader the
/// CLI uses — into the flat `Vec<GenericConstraint>`. `name_to_id` is empty (these
/// fixtures carry no `@param` coefficients) and the line topology is empty (no
/// `line_exchange(source_bus=…, target_bus=…)` form).
fn parse_generic_from_str(json: &str) -> Vec<novomodelo_core::GenericConstraint> {
    parse_generic_from_str_with_params(json, &std::collections::HashMap::new())
}

/// Discharge the desugaring invariant on one twin pair: a sugared study and its
/// hand-flattened equivalent must desugar to the same flat form AND build a
/// byte-identical LP. Asserts both the parsed `Vec<GenericConstraint>` (flat-form
/// identity — implies echo identity, the echo being a pure function of the flat
/// terms) and, stage by stage, the full `StageTemplate` (coefficients, bounds,
/// and slacks — not just row/col counts).
fn assert_lp_byte_identical(
    sugared_json: &str,
    flat_json: &str,
    n_blks: usize,
    bounds: &novomodelo_core::ResolvedGenericConstraintBounds,
) {
    let sugared = parse_generic_from_str(sugared_json);
    let flat = parse_generic_from_str(flat_json);
    assert_eq!(
        sugared, flat,
        "desugared flat form must equal the hand-flattened twin"
    );

    let sugared_tpl = build_templates_for(&one_bus_system_n_blks_with_generic(
        n_blks,
        sugared,
        bounds.clone(),
    ));
    let flat_tpl = build_templates_for(&one_bus_system_n_blks_with_generic(
        n_blks,
        flat,
        bounds.clone(),
    ));
    assert_all_templates_byte_identical(&sugared_tpl, &flat_tpl, "sugared vs hand-flattened");
}

/// Byte-identity twin: a single-column named-expression case (the `d13`/`d54`
/// anchor shape, `thermal_generation(0)`) declared as `@fnese` and its
/// hand-flattened twin desugar identically and build the same LP.
#[test]
fn desugared_named_expression_lp_matches_hand_flattened_twin() {
    use novomodelo_core::ResolvedGenericConstraintBounds;
    use std::collections::HashMap;

    let sugared = r#"{
      "expressions": [ { "name": "fnese", "expression": "thermal_generation(0)" } ],
      "constraints": [
        { "id": 1, "name": "cap", "expression": "@fnese", "slack": { "enabled": false } }
      ]
    }"#;
    let flat = r#"{
      "constraints": [
        { "id": 1, "name": "cap", "expression": "thermal_generation(0)", "slack": { "enabled": false } }
      ]
    }"#;

    let id_map: HashMap<i32, usize> = [(1_i32, 0)].into_iter().collect();
    let rows = vec![(1_i32, 0_i32, None::<i32>, None, Some(10.0_f64))];
    let bounds = ResolvedGenericConstraintBounds::new(&id_map, rows.into_iter());

    assert_lp_byte_identical(sugared, flat, 1, &bounds);
}

/// Declaration-order invariance: three named expressions, each a single term on the
/// SAME LP column (`bus_excess` of the one bus) with DISTINCT literal coefficients
/// (1, 2, 4). The expression engine merges the three references into one
/// `bus_excess(1)` term per load, and with `n_blks = 1` it lands on a single
/// generic-constraint row. Every one of the `3! = 6` permutations of the
/// declaration-and-reference order must build a byte-identical `StageTemplates`.
#[test]
fn declaration_order_permutation_is_invariant_in_lp() {
    use novomodelo_core::ResolvedGenericConstraintBounds;
    use std::collections::HashMap;

    let defs = [
        ("e1", "1 * bus_excess(1)"),
        ("e2", "2 * bus_excess(1)"),
        ("e3", "4 * bus_excess(1)"),
    ];
    let perms: [[usize; 3]; 6] = [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ];

    let id_map: HashMap<i32, usize> = [(1_i32, 0)].into_iter().collect();
    let rows = vec![(1_i32, 0_i32, None::<i32>, None, Some(500.0_f64))];
    let bounds = ResolvedGenericConstraintBounds::new(&id_map, rows.into_iter());

    let digest = |perm: &[usize; 3]| -> Vec<novomodelo_solver::StageTemplate> {
        let expressions: Vec<serde_json::Value> = perm
            .iter()
            .map(|&i| serde_json::json!({ "name": defs[i].0, "expression": defs[i].1 }))
            .collect();
        let reference = perm
            .iter()
            .map(|&i| format!("@{}", defs[i].0))
            .collect::<Vec<_>>()
            .join(" + ");
        let file = serde_json::json!({
            "expressions": expressions,
            "constraints": [
                { "id": 1, "name": "c", "expression": reference, "slack": { "enabled": false } }
            ]
        });
        let parsed = parse_generic_from_str(&file.to_string());
        let system = one_bus_system_n_blks_with_generic(1, parsed, bounds.clone());
        build_templates_for(&system)
    };

    let first = digest(&perms[0]);
    for perm in &perms[1..] {
        let templates = digest(perm);
        assert_all_templates_byte_identical(
            &templates,
            &first,
            &format!("declaration-order permutation {perm:?}"),
        );
    }
}

#[test]
fn generic_constraints_zero_does_not_change_layout() {
    let system = one_bus_system_n_blks(1);
    let templates = build_templates_for(&system);
    let t = &templates[0];
    assert!(t.num_cols > 0);
    assert!(t.num_rows > 0);
    // A second build must be bit-for-bit identical (determinism).
    let templates2 = build_templates_for(&system);
    assert_eq!(
        templates2[0].num_cols, t.num_cols,
        "second build must not change num_cols"
    );
    assert_eq!(
        templates2[0].num_rows, t.num_rows,
        "second build must not change num_rows"
    );
}

#[test]
fn generic_constraint_no_slack_block_id_none_3_blocks_collapses() {
    use novomodelo_core::ResolvedGenericConstraintBounds;
    use std::collections::HashMap;

    let n_blks = 3_usize;
    let baseline_system = one_bus_system_n_blks(n_blks);
    let baseline_rows = build_templates_for(&baseline_system)[0].num_rows;
    let baseline_cols = build_templates_for(&baseline_system)[0].num_cols;

    let id_map: HashMap<i32, usize> = [(10_i32, 0)].into_iter().collect();
    let rows = vec![(10_i32, 0_i32, None::<i32>, None, Some(500.0_f64))];
    let generic_bounds = ResolvedGenericConstraintBounds::new(&id_map, rows.into_iter());

    let constraint = make_constraint(10, false);
    let system = one_bus_system_n_blks_with_generic(n_blks, vec![constraint], generic_bounds);
    let t_rows = build_templates_for(&system)[0].num_rows;
    let t_cols = build_templates_for(&system)[0].num_cols;

    assert_eq!(
        t_rows,
        baseline_rows + 1,
        "num_rows must increase by 1 (block-independent block_id=None collapses to a single row)"
    );
    assert_eq!(
        t_cols, baseline_cols,
        "num_cols must be unchanged (no slack columns)"
    );
}

#[test]
fn generic_constraint_no_slack_block_id_none_3_blocks_block_level_per_block() {
    use novomodelo_core::ResolvedGenericConstraintBounds;
    use std::collections::HashMap;

    let n_blks = 3_usize;
    let baseline_system = one_bus_system_n_blks(n_blks);
    let baseline_rows = build_templates_for(&baseline_system)[0].num_rows;
    let baseline_cols = build_templates_for(&baseline_system)[0].num_cols;

    let id_map: HashMap<i32, usize> = [(11_i32, 0)].into_iter().collect();
    let rows = vec![(11_i32, 0_i32, None::<i32>, None, Some(500.0_f64))];
    let generic_bounds = ResolvedGenericConstraintBounds::new(&id_map, rows.into_iter());

    let constraint = make_constraint_with_expr(11, false, block_level_excess_expr(1));
    let system = one_bus_system_n_blks_with_generic(n_blks, vec![constraint], generic_bounds);
    let t_rows = build_templates_for(&system)[0].num_rows;
    let t_cols = build_templates_for(&system)[0].num_cols;

    assert_eq!(
        t_rows,
        baseline_rows + n_blks,
        "num_rows must increase by n_blks={n_blks} (block-level expression keeps one row per block)"
    );
    assert_eq!(
        t_cols, baseline_cols,
        "num_cols must be unchanged (no slack columns)"
    );
}

#[test]
fn generic_constraint_le_slack_enabled_2_blocks_collapses() {
    use novomodelo_core::ResolvedGenericConstraintBounds;
    use std::collections::HashMap;

    let n_blks = 2_usize;
    let baseline_system = one_bus_system_n_blks(n_blks);
    let baseline_rows = build_templates_for(&baseline_system)[0].num_rows;
    let baseline_cols = build_templates_for(&baseline_system)[0].num_cols;

    let id_map: HashMap<i32, usize> = [(20_i32, 0)].into_iter().collect();
    let rows = vec![(20_i32, 0_i32, None::<i32>, None, Some(300.0_f64))];
    let generic_bounds = ResolvedGenericConstraintBounds::new(&id_map, rows.into_iter());

    let constraint = make_constraint(20, true);
    let system = one_bus_system_n_blks_with_generic(n_blks, vec![constraint], generic_bounds);
    let t_rows = build_templates_for(&system)[0].num_rows;
    let t_cols = build_templates_for(&system)[0].num_cols;

    assert_eq!(
        t_rows,
        baseline_rows + 1,
        "num_rows must increase by 1 (collapsed stage-level row)"
    );
    assert_eq!(
        t_cols,
        baseline_cols + 1,
        "num_cols must increase by 1 (one `<=` slack on the single collapsed row)"
    );
}

#[test]
fn generic_constraint_le_slack_enabled_2_blocks_block_level_per_block() {
    use novomodelo_core::ResolvedGenericConstraintBounds;
    use std::collections::HashMap;

    let n_blks = 2_usize;
    let baseline_system = one_bus_system_n_blks(n_blks);
    let baseline_rows = build_templates_for(&baseline_system)[0].num_rows;
    let baseline_cols = build_templates_for(&baseline_system)[0].num_cols;

    let id_map: HashMap<i32, usize> = [(21_i32, 0)].into_iter().collect();
    let rows = vec![(21_i32, 0_i32, None::<i32>, None, Some(300.0_f64))];
    let generic_bounds = ResolvedGenericConstraintBounds::new(&id_map, rows.into_iter());

    let constraint = make_constraint_with_expr(21, true, block_level_excess_expr(1));
    let system = one_bus_system_n_blks_with_generic(n_blks, vec![constraint], generic_bounds);
    let t_rows = build_templates_for(&system)[0].num_rows;
    let t_cols = build_templates_for(&system)[0].num_cols;

    assert_eq!(
        t_rows,
        baseline_rows + 2,
        "num_rows must increase by 2 (block-level keeps one row per block)"
    );
    assert_eq!(
        t_cols,
        baseline_cols + 2,
        "num_cols must increase by 2 (one slack per per-block row for `<=`)"
    );
}

#[test]
fn generic_constraint_degenerate_band_two_slacks_collapses() {
    use novomodelo_core::ResolvedGenericConstraintBounds;
    use std::collections::HashMap;

    let n_blks = 2_usize;
    let baseline_system = one_bus_system_n_blks(n_blks);
    let baseline_rows = build_templates_for(&baseline_system)[0].num_rows;
    let baseline_cols = build_templates_for(&baseline_system)[0].num_cols;

    let id_map: HashMap<i32, usize> = [(30_i32, 0)].into_iter().collect();
    let rows = vec![(30_i32, 0_i32, None::<i32>, Some(100.0_f64), Some(100.0_f64))];
    let generic_bounds = ResolvedGenericConstraintBounds::new(&id_map, rows.into_iter());

    let constraint = make_constraint(30, true);
    let system = one_bus_system_n_blks_with_generic(n_blks, vec![constraint], generic_bounds);
    let t_rows = build_templates_for(&system)[0].num_rows;
    let t_cols = build_templates_for(&system)[0].num_cols;

    assert_eq!(
        t_rows,
        baseline_rows + 1,
        "num_rows must increase by 1 (collapsed stage-level row)"
    );
    assert_eq!(
        t_cols,
        baseline_cols + 2,
        "num_cols must increase by 2 (plus+minus slacks on the single collapsed row)"
    );
}

#[test]
fn generic_constraint_two_sided_two_slacks_collapses() {
    use novomodelo_core::ResolvedGenericConstraintBounds;
    use std::collections::HashMap;

    let n_blks = 2_usize;
    let baseline_system = one_bus_system_n_blks(n_blks);
    let baseline_rows = build_templates_for(&baseline_system)[0].num_rows;
    let baseline_cols = build_templates_for(&baseline_system)[0].num_cols;

    let id_map: HashMap<i32, usize> = [(70_i32, 0)].into_iter().collect();
    let rows = vec![(70_i32, 0_i32, None::<i32>, Some(100.0_f64), Some(200.0_f64))];
    let generic_bounds = ResolvedGenericConstraintBounds::new(&id_map, rows.into_iter());

    let constraint = make_constraint(70, true);
    let system = one_bus_system_n_blks_with_generic(n_blks, vec![constraint], generic_bounds);
    let t_rows = build_templates_for(&system)[0].num_rows;
    let t_cols = build_templates_for(&system)[0].num_cols;

    assert_eq!(
        t_rows,
        baseline_rows + 1,
        "num_rows must increase by 1 (collapsed stage-level row)"
    );
    assert_eq!(
        t_cols,
        baseline_cols + 2,
        "num_cols must increase by 2 (plus+minus slacks on the single collapsed row)"
    );
}

#[test]
fn generic_constraint_degenerate_band_two_slacks_block_level_per_block() {
    use novomodelo_core::ResolvedGenericConstraintBounds;
    use std::collections::HashMap;

    let n_blks = 2_usize;
    let baseline_system = one_bus_system_n_blks(n_blks);
    let baseline_rows = build_templates_for(&baseline_system)[0].num_rows;
    let baseline_cols = build_templates_for(&baseline_system)[0].num_cols;

    let id_map: HashMap<i32, usize> = [(31_i32, 0)].into_iter().collect();
    let rows = vec![(31_i32, 0_i32, None::<i32>, Some(100.0_f64), Some(100.0_f64))];
    let generic_bounds = ResolvedGenericConstraintBounds::new(&id_map, rows.into_iter());

    let constraint = make_constraint_with_expr(31, true, block_level_excess_expr(1));
    let system = one_bus_system_n_blks_with_generic(n_blks, vec![constraint], generic_bounds);
    let t_rows = build_templates_for(&system)[0].num_rows;
    let t_cols = build_templates_for(&system)[0].num_cols;

    assert_eq!(
        t_rows,
        baseline_rows + 2,
        "num_rows must increase by 2 (block-level keeps one row per block)"
    );
    assert_eq!(
        t_cols,
        baseline_cols + 4,
        "num_cols must increase by 4 (two slacks per per-block row for `==`)"
    );
}

#[test]
fn generic_constraint_two_sided_two_slacks_block_level_per_block() {
    use novomodelo_core::ResolvedGenericConstraintBounds;
    use std::collections::HashMap;

    let n_blks = 2_usize;
    let baseline_system = one_bus_system_n_blks(n_blks);
    let baseline_rows = build_templates_for(&baseline_system)[0].num_rows;
    let baseline_cols = build_templates_for(&baseline_system)[0].num_cols;

    let id_map: HashMap<i32, usize> = [(71_i32, 0)].into_iter().collect();
    let rows = vec![(71_i32, 0_i32, None::<i32>, Some(100.0_f64), Some(200.0_f64))];
    let generic_bounds = ResolvedGenericConstraintBounds::new(&id_map, rows.into_iter());

    let constraint = make_constraint_with_expr(71, true, block_level_excess_expr(1));
    let system = one_bus_system_n_blks_with_generic(n_blks, vec![constraint], generic_bounds);
    let t_rows = build_templates_for(&system)[0].num_rows;
    let t_cols = build_templates_for(&system)[0].num_cols;

    assert_eq!(
        t_rows,
        baseline_rows + 2,
        "num_rows must increase by 2 (block-level keeps one row per block)"
    );
    assert_eq!(
        t_cols,
        baseline_cols + 4,
        "num_cols must increase by 4 (two slacks per per-block row for range)"
    );
}

#[test]
fn generic_constraint_collapsed_slack_priced_by_total_stage_hours() {
    use novomodelo_core::ResolvedGenericConstraintBounds;
    use std::collections::HashMap;

    let n_blks = 3_usize;
    let block_hours = 720.0_f64; // one_bus_system_n_blks uses 720 h per block
    let penalty = 5000.0_f64; // make_constraint sets penalty = 5000 when slack enabled
    let total_stage_hours = block_hours * (n_blks as f64);

    let baseline_system = one_bus_system_n_blks(n_blks);
    let baseline_cols = build_templates_for(&baseline_system)[0].num_cols;

    let id_map: HashMap<i32, usize> = [(60_i32, 0)].into_iter().collect();
    let rows = vec![(60_i32, 0_i32, None::<i32>, None, Some(500.0_f64))];
    let generic_bounds = ResolvedGenericConstraintBounds::new(&id_map, rows.into_iter());

    let constraint = make_constraint(60, true);
    let system = one_bus_system_n_blks_with_generic(n_blks, vec![constraint], generic_bounds);
    let t = &build_templates_for(&system)[0];

    assert_eq!(
        t.num_cols,
        baseline_cols + 1,
        "collapsed `<=` row must add exactly one slack column"
    );

    let slack_col = baseline_cols; // new slack appended after the baseline cols
    let generic_row = t.num_rows - 1; // the single collapsed generic row is last

    let expected_obj = penalty * total_stage_hours / COST_SCALE_FACTOR;
    let per_block_sum = penalty * block_hours * (n_blks as f64) / COST_SCALE_FACTOR;
    assert!(
        (expected_obj - per_block_sum).abs() < 1e-9,
        "penalty conservation identity must hold: {expected_obj} vs {per_block_sum}"
    );
    assert!(
        (t.objective[slack_col] - expected_obj).abs() < 1e-9,
        "collapsed slack objective must be penalty × total_stage_hours / scale = {expected_obj}, \
         got {}",
        t.objective[slack_col]
    );

    let entries = csc_entries_at(t, slack_col, generic_row);
    let total: f64 = entries.iter().sum();
    assert!(
        (total - (-1.0)).abs() < f64::EPSILON,
        "collapsed `<=` slack must have -1.0 at the generic row, got {total}"
    );
}

#[test]
fn generic_constraint_specific_block_id_generates_one_row() {
    use novomodelo_core::ResolvedGenericConstraintBounds;
    use std::collections::HashMap;

    let n_blks = 3_usize;
    let baseline_system = one_bus_system_n_blks(n_blks);
    let baseline_rows = build_templates_for(&baseline_system)[0].num_rows;
    let baseline_cols = build_templates_for(&baseline_system)[0].num_cols;

    let id_map: HashMap<i32, usize> = [(40_i32, 0)].into_iter().collect();
    let rows = vec![(40_i32, 0_i32, Some(1_i32), None, Some(200.0_f64))];
    let generic_bounds = ResolvedGenericConstraintBounds::new(&id_map, rows.into_iter());

    let constraint = make_constraint(40, false);
    let system = one_bus_system_n_blks_with_generic(n_blks, vec![constraint], generic_bounds);
    let t_rows = build_templates_for(&system)[0].num_rows;
    let t_cols = build_templates_for(&system)[0].num_cols;

    assert_eq!(
        t_rows,
        baseline_rows + 1,
        "num_rows must increase by exactly 1 (only the specified block)"
    );
    assert_eq!(
        t_cols, baseline_cols,
        "num_cols must be unchanged (no slack columns)"
    );
}

#[test]
fn generic_constraint_inactive_does_not_contribute_rows() {
    use novomodelo_core::ResolvedGenericConstraintBounds;
    use std::collections::HashMap;

    let n_blks = 2_usize;
    let baseline_system = one_bus_system_n_blks(n_blks);
    let baseline_rows = build_templates_for(&baseline_system)[0].num_rows;

    // Constraint 51 has no bounds row → inactive.
    let id_map: HashMap<i32, usize> = [(50_i32, 0), (51_i32, 1)].into_iter().collect();
    let rows = vec![(50_i32, 0_i32, None::<i32>, None, Some(400.0_f64))];
    let generic_bounds = ResolvedGenericConstraintBounds::new(&id_map, rows.into_iter());

    let c_active = make_constraint(50, false);
    let c_inactive = make_constraint(51, false);
    let system =
        one_bus_system_n_blks_with_generic(n_blks, vec![c_active, c_inactive], generic_bounds);
    let t_rows = build_templates_for(&system)[0].num_rows;

    // The active constraint's trivial expression collapses to one row (without the
    // collapse it would be baseline_rows + n_blks); the test's point is that the
    // inactive constraint stays absent under either representation.
    assert_eq!(
        t_rows,
        baseline_rows + 1,
        "only the active constraint must contribute rows (collapsed to one)"
    );
}

#[test]
fn generic_constraint_thermal_le_row_bounds_and_csc_entry() {
    use novomodelo_core::ResolvedGenericConstraintBounds;
    use novomodelo_core::{ConstraintExpression, GenericConstraint, LinearTerm, VariableRef};
    use std::collections::HashMap;

    let thermal_entity_id = EntityId(2);

    let constraint = GenericConstraint {
        id: EntityId(10),
        name: "gc_thermal_le".to_string(),
        description: None,
        expression: ConstraintExpression {
            terms: vec![LinearTerm::literal(
                1.0,
                VariableRef::ThermalGeneration {
                    thermal_id: thermal_entity_id,
                    block_id: None,
                },
            )],
        },
        slack: None,
        bound_lower: None,
        bound_upper: None,
    };

    let id_map: HashMap<i32, usize> = [(10_i32, 0)].into_iter().collect();
    let rows = vec![(10_i32, 0_i32, None::<i32>, None, Some(50.0_f64))];
    let generic_bounds = ResolvedGenericConstraintBounds::new(&id_map, rows.into_iter());

    let system = one_bus_one_thermal_system(thermal_entity_id, vec![constraint], generic_bounds);
    let t = &build_templates_for(&system)[0];

    // Layout for N=0, T=1, B=1, K=1: theta=0, thermal col 0/block 0 = col 1,
    // load_balance row 0 → generic row at row 1.
    let thermal_col = 1_usize;
    let generic_row = 1_usize;

    assert!(
        t.row_lower[generic_row].is_infinite() && t.row_lower[generic_row] < 0.0,
        "row_lower must be -INF for an upper-only row, got {}",
        t.row_lower[generic_row]
    );
    assert!(
        (t.row_upper[generic_row] - 50.0).abs() < f64::EPSILON,
        "row_upper must be 50.0, got {}",
        t.row_upper[generic_row]
    );

    let entries = csc_entries_at(t, thermal_col, generic_row);
    assert!(
        !entries.is_empty(),
        "no CSC entry found at (col={thermal_col}, row={generic_row})"
    );
    let total: f64 = entries.iter().sum();
    assert!(
        (total - 1.0).abs() < f64::EPSILON,
        "expected +1.0 total at thermal col / generic row, got {total}"
    );
}

#[test]
fn generic_constraint_thermal_le_slack_column_and_csc_entry() {
    use novomodelo_core::ResolvedGenericConstraintBounds;
    use novomodelo_core::{
        ConstraintExpression, GenericConstraint, LinearTerm, PenaltyBasis, SlackConfig, VariableRef,
    };
    use std::collections::HashMap;

    let thermal_entity_id = EntityId(2);
    let block_hours = 744.0_f64;
    let penalty = 5000.0_f64;

    let constraint = GenericConstraint {
        id: EntityId(20),
        name: "gc_thermal_le_slack".to_string(),
        description: None,
        expression: ConstraintExpression {
            terms: vec![LinearTerm::literal(
                1.0,
                VariableRef::ThermalGeneration {
                    thermal_id: thermal_entity_id,
                    block_id: None,
                },
            )],
        },
        slack: Some(SlackConfig {
            penalty,
            penalty_basis: PenaltyBasis::PerHour,
        }),
        bound_lower: None,
        bound_upper: None,
    };

    let id_map: HashMap<i32, usize> = [(20_i32, 0)].into_iter().collect();
    let rows = vec![(20_i32, 0_i32, None::<i32>, None, Some(50.0_f64))];
    let generic_bounds = ResolvedGenericConstraintBounds::new(&id_map, rows.into_iter());

    let system = one_bus_one_thermal_system(thermal_entity_id, vec![constraint], generic_bounds);
    let t = &build_templates_for(&system)[0];

    // Layout for N=0, T=1, B=1, K=1, 1 slack col:
    //   withdrawal_slack_start = col_evap_start + 0 evap cols
    //   = col_generation_start + 0 generation cols = inflow_slack_end
    //   = excess_end = 1(theta)+1(thermal)+1(deficit)+1(excess) = 4
    //   col_generic_slack_start = withdrawal_slack_start + n_h(=0) = 4
    //   slack_plus_col = 4
    let slack_col = 4_usize;
    let generic_row = 1_usize;

    assert!(
        t.col_lower[slack_col].abs() < f64::EPSILON,
        "slack col_lower must be 0.0, got {}",
        t.col_lower[slack_col]
    );
    assert!(
        t.col_upper[slack_col].is_infinite() && t.col_upper[slack_col] > 0.0,
        "slack col_upper must be +INF, got {}",
        t.col_upper[slack_col]
    );

    let expected_obj = penalty * block_hours / COST_SCALE_FACTOR;
    assert!(
        (t.objective[slack_col] - expected_obj).abs() < 1e-12,
        "slack objective must be {expected_obj}, got {}",
        t.objective[slack_col]
    );

    let entries = csc_entries_at(t, slack_col, generic_row);
    assert!(
        !entries.is_empty(),
        "no CSC entry found at (col={slack_col}, row={generic_row})"
    );
    let total: f64 = entries.iter().sum();
    assert!(
        (total - (-1.0)).abs() < f64::EPSILON,
        "expected -1.0 at slack col / generic row for an upper-only row, got {total}"
    );
}

#[test]
fn generic_constraint_thermal_ge_row_bounds() {
    use novomodelo_core::ResolvedGenericConstraintBounds;
    use novomodelo_core::{ConstraintExpression, GenericConstraint, LinearTerm, VariableRef};
    use std::collections::HashMap;

    let thermal_entity_id = EntityId(2);

    let constraint = GenericConstraint {
        id: EntityId(30),
        name: "gc_thermal_ge".to_string(),
        description: None,
        expression: ConstraintExpression {
            terms: vec![LinearTerm::literal(
                1.0,
                VariableRef::ThermalGeneration {
                    thermal_id: thermal_entity_id,
                    block_id: None,
                },
            )],
        },
        slack: None,
        bound_lower: None,
        bound_upper: None,
    };

    let id_map: HashMap<i32, usize> = [(30_i32, 0)].into_iter().collect();
    let rows = vec![(30_i32, 0_i32, None::<i32>, Some(10.0_f64), None)];
    let generic_bounds = ResolvedGenericConstraintBounds::new(&id_map, rows.into_iter());

    let system = one_bus_one_thermal_system(thermal_entity_id, vec![constraint], generic_bounds);
    let t = &build_templates_for(&system)[0];

    let generic_row = 1_usize;
    assert!(
        (t.row_lower[generic_row] - 10.0).abs() < f64::EPSILON,
        "row_lower must be 10.0 for a lower-only row, got {}",
        t.row_lower[generic_row]
    );
    assert!(
        t.row_upper[generic_row].is_infinite() && t.row_upper[generic_row] > 0.0,
        "row_upper must be +INF for a lower-only row, got {}",
        t.row_upper[generic_row]
    );
}

/// A lower-only row (`bound_lower` present, `bound_upper` absent) on a
/// slack-enabled constraint allocates exactly ONE slack column with `+1.0` at
/// the generic row — the mirror image of the upper-only `-1.0` case above.
#[test]
fn generic_constraint_thermal_ge_slack_column_and_csc_entry() {
    use novomodelo_core::ResolvedGenericConstraintBounds;
    use novomodelo_core::{
        ConstraintExpression, GenericConstraint, LinearTerm, PenaltyBasis, SlackConfig, VariableRef,
    };
    use std::collections::HashMap;

    let thermal_entity_id = EntityId(2);
    let block_hours = 744.0_f64;
    let penalty = 5000.0_f64;

    let constraint = GenericConstraint {
        id: EntityId(35),
        name: "gc_thermal_ge_slack".to_string(),
        description: None,
        expression: ConstraintExpression {
            terms: vec![LinearTerm::literal(
                1.0,
                VariableRef::ThermalGeneration {
                    thermal_id: thermal_entity_id,
                    block_id: None,
                },
            )],
        },
        slack: Some(SlackConfig {
            penalty,
            penalty_basis: PenaltyBasis::PerHour,
        }),
        bound_lower: None,
        bound_upper: None,
    };

    let id_map: HashMap<i32, usize> = [(35_i32, 0)].into_iter().collect();
    let rows = vec![(35_i32, 0_i32, None::<i32>, Some(10.0_f64), None)];
    let generic_bounds = ResolvedGenericConstraintBounds::new(&id_map, rows.into_iter());

    let system = one_bus_one_thermal_system(thermal_entity_id, vec![constraint], generic_bounds);
    let t = &build_templates_for(&system)[0];

    // Same layout as the upper-only slack test: theta=0, thermal=1, deficit=1,
    // excess=1 → col_generic_slack_start = 4, slack_plus_col = 4.
    let slack_col = 4_usize;
    let generic_row = 1_usize;

    assert_eq!(
        t.row_lower[generic_row], 10.0,
        "row_lower must be 10.0 for a lower-only row"
    );
    assert!(
        t.row_upper[generic_row].is_infinite() && t.row_upper[generic_row] > 0.0,
        "row_upper must be +INF for a lower-only row"
    );

    let expected_obj = penalty * block_hours / COST_SCALE_FACTOR;
    assert!(
        (t.objective[slack_col] - expected_obj).abs() < 1e-12,
        "slack objective must be {expected_obj}, got {}",
        t.objective[slack_col]
    );

    let entries = csc_entries_at(t, slack_col, generic_row);
    assert!(
        !entries.is_empty(),
        "no CSC entry found at (col={slack_col}, row={generic_row})"
    );
    let total: f64 = entries.iter().sum();
    assert!(
        (total - 1.0).abs() < f64::EPSILON,
        "expected +1.0 at slack col / generic row for a lower-only row, got {total}"
    );
}

#[test]
fn generic_constraint_two_sided_thermal_row_bounds_and_csc_entry() {
    use novomodelo_core::ResolvedGenericConstraintBounds;
    use novomodelo_core::{ConstraintExpression, GenericConstraint, LinearTerm, VariableRef};
    use std::collections::HashMap;

    let thermal_entity_id = EntityId(2);

    let constraint = GenericConstraint {
        id: EntityId(72),
        name: "gc_thermal_range".to_string(),
        description: None,
        expression: ConstraintExpression {
            terms: vec![LinearTerm::literal(
                1.0,
                VariableRef::ThermalGeneration {
                    thermal_id: thermal_entity_id,
                    block_id: None,
                },
            )],
        },
        slack: None,
        bound_lower: None,
        bound_upper: None,
    };

    let id_map: HashMap<i32, usize> = [(72_i32, 0)].into_iter().collect();
    let rows = vec![(72_i32, 0_i32, None::<i32>, Some(5.0_f64), Some(20.0_f64))];
    let generic_bounds = ResolvedGenericConstraintBounds::new(&id_map, rows.into_iter());

    let system = one_bus_one_thermal_system(thermal_entity_id, vec![constraint], generic_bounds);
    let t = &build_templates_for(&system)[0];

    // Layout for N=0, T=1, B=1, K=1 (same as generic_constraint_thermal_le_row_bounds_and_csc_entry).
    let thermal_col = 1_usize;
    let generic_row = 1_usize;

    assert!(
        (t.row_lower[generic_row] - 5.0).abs() < f64::EPSILON,
        "row_lower must be 5.0 (bound_lower) for a two-sided row, got {}",
        t.row_lower[generic_row]
    );
    assert!(
        (t.row_upper[generic_row] - 20.0).abs() < f64::EPSILON,
        "row_upper must be 20.0 (bound_upper) for a two-sided row, got {}",
        t.row_upper[generic_row]
    );

    let entries = csc_entries_at(t, thermal_col, generic_row);
    assert!(
        !entries.is_empty(),
        "no CSC entry found at (col={thermal_col}, row={generic_row})"
    );
    let total: f64 = entries.iter().sum();
    assert!(
        (total - 1.0).abs() < f64::EPSILON,
        "expected +1.0 total at thermal col / generic row, got {total}"
    );
}

#[test]
fn generic_constraint_degenerate_band_thermal_two_slacks() {
    use novomodelo_core::ResolvedGenericConstraintBounds;
    use novomodelo_core::{ConstraintExpression, GenericConstraint, PenaltyBasis, SlackConfig};
    use std::collections::HashMap;

    let thermal_entity_id = EntityId(2);
    let penalty = 5000.0_f64;
    let block_hours = 744.0_f64;

    let constraint = GenericConstraint {
        id: EntityId(40),
        name: "gc_thermal_eq_slack".to_string(),
        description: None,
        expression: ConstraintExpression { terms: vec![] },
        slack: Some(SlackConfig {
            penalty,
            penalty_basis: PenaltyBasis::PerHour,
        }),
        bound_lower: None,
        bound_upper: None,
    };

    let id_map: HashMap<i32, usize> = [(40_i32, 0)].into_iter().collect();
    let rows = vec![(40_i32, 0_i32, None::<i32>, Some(80.0_f64), Some(80.0_f64))];
    let generic_bounds = ResolvedGenericConstraintBounds::new(&id_map, rows.into_iter());

    let system = one_bus_one_thermal_system(thermal_entity_id, vec![constraint], generic_bounds);
    let t = &build_templates_for(&system)[0];

    let slack_plus_col = 4_usize;
    let slack_minus_col = 5_usize;
    let generic_row = 1_usize;

    assert!(
        (t.row_lower[generic_row] - 80.0).abs() < f64::EPSILON,
        "row_lower must be 80.0 for a degenerate band, got {}",
        t.row_lower[generic_row]
    );
    assert!(
        (t.row_upper[generic_row] - 80.0).abs() < f64::EPSILON,
        "row_upper must be 80.0 for a degenerate band, got {}",
        t.row_upper[generic_row]
    );

    // Two slack columns: num_cols baseline is 4 (theta=0, thermal=1, deficit=1, excess=1,
    // withdrawal_slack=0), with 2 slacks → num_cols = 6.
    assert_eq!(t.num_cols, 6, "num_cols must be 6 with 2 slack columns");

    assert!(
        t.col_upper[slack_plus_col].is_infinite() && t.col_upper[slack_plus_col] > 0.0,
        "plus slack col_upper must be +INF"
    );
    let expected_obj = penalty * block_hours / COST_SCALE_FACTOR;
    assert!(
        (t.objective[slack_plus_col] - expected_obj).abs() < 1e-12,
        "plus slack objective must be {expected_obj}, got {}",
        t.objective[slack_plus_col]
    );
    let plus_entries = csc_entries_at(t, slack_plus_col, generic_row);
    assert!(
        !plus_entries.is_empty(),
        "no CSC entry at plus slack col / generic row"
    );
    let plus_total: f64 = plus_entries.iter().sum();
    assert!(
        (plus_total - 1.0).abs() < f64::EPSILON,
        "plus slack CSC must be +1.0 for a degenerate band, got {plus_total}"
    );

    assert!(
        t.col_upper[slack_minus_col].is_infinite() && t.col_upper[slack_minus_col] > 0.0,
        "minus slack col_upper must be +INF"
    );
    assert!(
        (t.objective[slack_minus_col] - expected_obj).abs() < 1e-12,
        "minus slack objective must be {expected_obj}"
    );
    let minus_entries = csc_entries_at(t, slack_minus_col, generic_row);
    assert!(
        !minus_entries.is_empty(),
        "no CSC entry at minus slack col / generic row"
    );
    let minus_total: f64 = minus_entries.iter().sum();
    assert!(
        (minus_total - (-1.0)).abs() < f64::EPSILON,
        "minus slack CSC must be -1.0 for a degenerate band, got {minus_total}"
    );
}

#[test]
fn generic_constraint_two_sided_thermal_two_slacks() {
    use novomodelo_core::ResolvedGenericConstraintBounds;
    use novomodelo_core::{ConstraintExpression, GenericConstraint, PenaltyBasis, SlackConfig};
    use std::collections::HashMap;

    let thermal_entity_id = EntityId(2);
    let penalty = 5000.0_f64;
    let block_hours = 744.0_f64;

    let constraint = GenericConstraint {
        id: EntityId(73),
        name: "gc_thermal_range_slack".to_string(),
        description: None,
        expression: ConstraintExpression { terms: vec![] },
        slack: Some(SlackConfig {
            penalty,
            penalty_basis: PenaltyBasis::PerHour,
        }),
        bound_lower: None,
        bound_upper: None,
    };

    let id_map: HashMap<i32, usize> = [(73_i32, 0)].into_iter().collect();
    let rows = vec![(73_i32, 0_i32, None::<i32>, Some(30.0_f64), Some(80.0_f64))];
    let generic_bounds = ResolvedGenericConstraintBounds::new(&id_map, rows.into_iter());

    let system = one_bus_one_thermal_system(thermal_entity_id, vec![constraint], generic_bounds);
    let t = &build_templates_for(&system)[0];

    let slack_plus_col = 4_usize;
    let slack_minus_col = 5_usize;
    let generic_row = 1_usize;

    assert!(
        (t.row_lower[generic_row] - 30.0).abs() < f64::EPSILON,
        "row_lower must be 30.0 for a two-sided row, got {}",
        t.row_lower[generic_row]
    );
    assert!(
        (t.row_upper[generic_row] - 80.0).abs() < f64::EPSILON,
        "row_upper must be 80.0 for a two-sided row, got {}",
        t.row_upper[generic_row]
    );

    // Two slack columns: num_cols baseline is 4 (theta=0, thermal=1, deficit=1, excess=1,
    // withdrawal_slack=0), with 2 slacks → num_cols = 6.
    assert_eq!(t.num_cols, 6, "num_cols must be 6 with 2 slack columns");

    assert!(
        t.col_upper[slack_plus_col].is_infinite() && t.col_upper[slack_plus_col] > 0.0,
        "plus slack col_upper must be +INF"
    );
    let expected_obj = penalty * block_hours / COST_SCALE_FACTOR;
    assert!(
        (t.objective[slack_plus_col] - expected_obj).abs() < 1e-12,
        "plus slack objective must be {expected_obj}, got {}",
        t.objective[slack_plus_col]
    );
    let plus_entries = csc_entries_at(t, slack_plus_col, generic_row);
    assert!(
        !plus_entries.is_empty(),
        "no CSC entry at plus slack col / generic row"
    );
    let plus_total: f64 = plus_entries.iter().sum();
    assert!(
        (plus_total - 1.0).abs() < f64::EPSILON,
        "plus slack CSC must be +1.0 for a two-sided row, got {plus_total}"
    );

    assert!(
        t.col_upper[slack_minus_col].is_infinite() && t.col_upper[slack_minus_col] > 0.0,
        "minus slack col_upper must be +INF"
    );
    assert!(
        (t.objective[slack_minus_col] - expected_obj).abs() < 1e-12,
        "minus slack objective must be {expected_obj}"
    );
    let minus_entries = csc_entries_at(t, slack_minus_col, generic_row);
    assert!(
        !minus_entries.is_empty(),
        "no CSC entry at minus slack col / generic row"
    );
    let minus_total: f64 = minus_entries.iter().sum();
    assert!(
        (minus_total - (-1.0)).abs() < f64::EPSILON,
        "minus slack CSC must be -1.0 for a two-sided row, got {minus_total}"
    );
}

/// A degenerate two-sided band (`bound_lower == bound_upper`) is still
/// two-sided by the null-pattern (both endpoints present), so it must get an
/// equality row (`row_lower == row_upper`) AND the full two-slack allocation
/// — the shape derivation must not special-case equal endpoints down to a
/// one-sided row.
#[test]
fn generic_constraint_degenerate_band_gets_equality_row_and_two_slacks() {
    use novomodelo_core::ResolvedGenericConstraintBounds;
    use novomodelo_core::{ConstraintExpression, GenericConstraint, PenaltyBasis, SlackConfig};
    use std::collections::HashMap;

    let thermal_entity_id = EntityId(2);
    let penalty = 5000.0_f64;
    let id = 74;

    let constraint = GenericConstraint {
        id: EntityId(id),
        name: format!("gc_{id}"),
        description: None,
        expression: ConstraintExpression { terms: vec![] },
        slack: Some(SlackConfig {
            penalty,
            penalty_basis: PenaltyBasis::PerHour,
        }),
        bound_lower: None,
        bound_upper: None,
    };
    let id_map: HashMap<i32, usize> = [(id, 0_usize)].into_iter().collect();
    let rows = vec![(id, 0_i32, None::<i32>, Some(10.0_f64), Some(10.0_f64))];
    let generic_bounds = ResolvedGenericConstraintBounds::new(&id_map, rows.into_iter());
    let system = one_bus_one_thermal_system(thermal_entity_id, vec![constraint], generic_bounds);

    let t = &build_templates_for(&system)[0];
    let generic_row = 1_usize;

    assert_eq!(
        t.row_lower[generic_row], 10.0,
        "degenerate band row_lower must equal the shared endpoint value"
    );
    assert_eq!(
        t.row_upper[generic_row], 10.0,
        "degenerate band row_upper must equal the shared endpoint value"
    );
    assert_eq!(
        t.num_cols, 6,
        "degenerate two-sided band must still allocate BOTH slack columns"
    );
}

#[test]
#[allow(clippy::cast_possible_wrap)]
fn generic_constraint_two_hydros_sum_csc_entries() {
    use chrono::NaiveDate;
    use novomodelo_core::ResolvedGenericConstraintBounds;
    use novomodelo_core::entities::hydro::{HydroGenerationModel, HydroPenalties};
    use novomodelo_core::scenario::{InflowModel, LoadModel};
    use novomodelo_core::temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, StageRiskConfig, StageStateConfig,
    };
    use novomodelo_core::{ConstraintExpression, GenericConstraint, LinearTerm, VariableRef};
    use std::collections::HashMap;

    let h1_id = EntityId(5);
    let h2_id = EntityId(10);
    let prod_h1 = 2.5_f64;
    let prod_h2 = 3.0_f64;

    let hydros = vec![
        make_hydro(
            h1_id,
            HydroSpec {
                name: format!("H{}", h1_id.0),
                operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                bus_id: EntityId(1),
                downstream_id: None,
                entry_stage_id: None,
                exit_stage_id: None,
                min_storage_hm3: 0.0,
                max_storage_hm3: 200.0,
                min_outflow_m3s: 0.0,
                max_outflow_m3s: None,
                generation_model: HydroGenerationModel::ConstantProductivity,
                min_turbined_m3s: 0.0,
                max_turbined_m3s: 100.0,
                specific_productivity_mw_per_m3s_per_m: None,
                min_generation_mw: 0.0,
                max_generation_mw: 250.0,
                tailrace: None,
                hydraulic_losses: None,
                efficiency: None,
                evaporation_coefficients_mm: None,
                evaporation_reference_volumes_hm3: None,
                diversion: None,
                filling: None,
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
                    evaporation_violation_cost: 0.0,
                    water_withdrawal_violation_cost: 0.0,
                    water_withdrawal_violation_pos_cost: 0.0,
                    water_withdrawal_violation_neg_cost: 0.0,
                    evaporation_violation_pos_cost: 0.0,
                    evaporation_violation_neg_cost: 0.0,
                    inflow_nonnegativity_cost: 1000.0,
                },
                ..Default::default()
            },
        ),
        make_hydro(
            h2_id,
            HydroSpec {
                name: format!("H{}", h2_id.0),
                operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                bus_id: EntityId(1),
                downstream_id: None,
                entry_stage_id: None,
                exit_stage_id: None,
                min_storage_hm3: 0.0,
                max_storage_hm3: 200.0,
                min_outflow_m3s: 0.0,
                max_outflow_m3s: None,
                generation_model: HydroGenerationModel::ConstantProductivity,
                min_turbined_m3s: 0.0,
                max_turbined_m3s: 100.0,
                specific_productivity_mw_per_m3s_per_m: None,
                min_generation_mw: 0.0,
                max_generation_mw: 250.0,
                tailrace: None,
                hydraulic_losses: None,
                efficiency: None,
                evaporation_coefficients_mm: None,
                evaporation_reference_volumes_hm3: None,
                diversion: None,
                filling: None,
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
                    evaporation_violation_cost: 0.0,
                    water_withdrawal_violation_cost: 0.0,
                    water_withdrawal_violation_pos_cost: 0.0,
                    water_withdrawal_violation_neg_cost: 0.0,
                    evaporation_violation_pos_cost: 0.0,
                    evaporation_violation_neg_cost: 0.0,
                    inflow_nonnegativity_cost: 1000.0,
                },
                ..Default::default()
            },
        ),
    ];

    let bus = make_bus(
        EntityId(1),
        BusSpec {
            name: "B1".to_string(),
            operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 500.0,
            }],
            excess_cost: 0.0,
            ..Default::default()
        },
    );

    let stage = make_stage(
        0,
        StageSpec {
            start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
            season_id: None,
            blocks: vec![Block {
                index: 0,
                name: "BLK0".to_string(),
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
    );

    let inflow_models = vec![
        InflowModel {
            hydro_id: h1_id,
            stage_id: 0,
            mean_m3s: 50.0,
            std_m3s: 0.0,
            ar_coefficients: vec![],
            residual_std_ratio: 0.0,
            annual: None,
        },
        InflowModel {
            hydro_id: h2_id,
            stage_id: 0,
            mean_m3s: 50.0,
            std_m3s: 0.0,
            ar_coefficients: vec![],
            residual_std_ratio: 0.0,
            annual: None,
        },
    ];

    let load_models = vec![LoadModel {
        bus_id: EntityId(1),
        stage_id: 0,
        mean_mw: 100.0,
        std_mw: 0.0,
    }];

    // Generic constraint: hydro_generation(H1) + hydro_generation(H2) <= 200
    let constraint = GenericConstraint {
        id: EntityId(100),
        name: "gc_sum_gen".to_string(),
        description: None,
        expression: ConstraintExpression {
            terms: vec![
                LinearTerm::literal(
                    1.0,
                    VariableRef::HydroGeneration {
                        hydro_id: h1_id,
                        block_id: None,
                        bus_id: None,
                    },
                ),
                LinearTerm::literal(
                    1.0,
                    VariableRef::HydroGeneration {
                        hydro_id: h2_id,
                        block_id: None,
                        bus_id: None,
                    },
                ),
            ],
        },
        slack: None,
        bound_lower: None,
        bound_upper: None,
    };

    let id_map: HashMap<i32, usize> = [(100_i32, 0)].into_iter().collect();
    let rows = vec![(100_i32, 0_i32, None::<i32>, None, Some(200.0_f64))];
    let generic_bounds = ResolvedGenericConstraintBounds::new(&id_map, rows.into_iter());

    let resolved_bounds = ResolvedBounds::new(
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
            hydro: default_hydro_bounds(),
            hydro_block: default_hydro_block_bounds(),
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
    let penalties = ResolvedPenalties::new(
        &PenaltiesCountsSpec {
            n_hydros: 2,
            n_buses: 1,
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
        .buses(vec![bus])
        .hydros(hydros)
        .stages(vec![stage])
        .inflow_models(inflow_models)
        .load_models(load_models)
        .bounds(resolved_bounds)
        .penalties(penalties)
        .generic_constraints(vec![constraint])
        .resolved_generic_bounds(generic_bounds)
        .build()
        .expect("two_hydro_system: valid");

    // Productivities are supplied explicitly (prod_h1=2.5 at index 0, prod_h2=3.0
    // at index 1) so the generic-row LP coefficients equal the productivities.
    let pm = production_set(&[prod_h1, prod_h2], 1);
    let t = &build_stage_templates_resolving_layout(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &pm,
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("two_hydro_system: valid")
    .templates[0];

    // For constant-productivity hydros, HydroGeneration maps to the turbine column
    // with multiplier = productivity; the generic constraint row is the last row.
    let generic_row = t.num_rows - 1;

    // N=2: storage=[0..2], z_inflow=[2..4], storage_in=[4..6], theta=6, turbine_start=7.
    let turbine_h1_col = 7_usize;
    let turbine_h2_col = 8_usize;

    let entries_h1 = csc_entries_at(t, turbine_h1_col, generic_row);
    assert!(
        !entries_h1.is_empty(),
        "no CSC entry found for H1 turbine at generic constraint row"
    );
    let total_h1: f64 = entries_h1.iter().sum();
    assert!(
        (total_h1 - prod_h1).abs() < f64::EPSILON,
        "expected coefficient {prod_h1} for H1, got {total_h1}"
    );

    let entries_h2 = csc_entries_at(t, turbine_h2_col, generic_row);
    assert!(
        !entries_h2.is_empty(),
        "no CSC entry found for H2 turbine at generic constraint row"
    );
    let total_h2: f64 = entries_h2.iter().sum();
    assert!(
        (total_h2 - prod_h2).abs() < f64::EPSILON,
        "expected coefficient {prod_h2} for H2, got {total_h2}"
    );
}

/// Build a single-hydro, one-bus, `n_blks`-block, one-stage system under
/// `block_mode` with `storage` state on, optionally carrying `constraint` +
/// `bounds`. Column layout (N=1, no thermal/FPHA/evap): storage=[0,1),
/// z_inflow=[1,2), storage_in=[2,3), theta=3, control region from col 4.
#[allow(clippy::cast_possible_wrap)]
fn one_hydro_system(
    n_blks: usize,
    block_mode: novomodelo_core::BlockMode,
    constraint: Option<novomodelo_core::GenericConstraint>,
    bounds: novomodelo_core::ResolvedGenericConstraintBounds,
) -> novomodelo_core::System {
    use chrono::NaiveDate;
    use novomodelo_core::HydroGenerationModel;
    use novomodelo_core::scenario::{InflowModel, LoadModel};
    use novomodelo_core::temporal::{
        Block, NoiseMethod, ScenarioSourceConfig, StageRiskConfig, StageStateConfig,
    };

    let hydro_id = EntityId(5);

    let hydro = make_hydro(
        hydro_id,
        HydroSpec {
            name: "H1".to_string(),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            bus_id: EntityId(1),
            min_storage_hm3: 0.0,
            max_storage_hm3: 200.0,
            max_turbined_m3s: 100.0,
            generation_model: HydroGenerationModel::ConstantProductivity,
            max_generation_mw: 250.0,
            ..Default::default()
        },
    );

    let bus = make_bus(
        EntityId(1),
        BusSpec {
            name: "B1".to_string(),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 500.0,
            }],
            excess_cost: 0.0,
            ..Default::default()
        },
    );

    let blocks: Vec<Block> = (0..n_blks)
        .map(|i| Block {
            index: i,
            name: format!("BLK{i}"),
            duration_hours: 720.0,
        })
        .collect();

    let stage = make_stage(
        0,
        StageSpec {
            start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
            season_id: None,
            blocks,
            block_mode,
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
    );

    let inflow_models = vec![InflowModel {
        hydro_id,
        stage_id: 0,
        mean_m3s: 50.0,
        std_m3s: 0.0,
        ar_coefficients: vec![],
        residual_std_ratio: 0.0,
        annual: None,
    }];

    let load_models = vec![LoadModel {
        bus_id: EntityId(1),
        stage_id: 0,
        mean_mw: 100.0,
        std_mw: 0.0,
    }];

    let resolved_bounds = ResolvedBounds::new(
        &BoundsCountsSpec {
            n_hydros: 1,
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
    let penalties = ResolvedPenalties::new(
        &PenaltiesCountsSpec {
            n_hydros: 1,
            n_buses: 1,
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

    let mut builder = SystemBuilder::new()
        .buses(vec![bus])
        .hydros(vec![hydro])
        .stages(vec![stage])
        .inflow_models(inflow_models)
        .load_models(load_models)
        .bounds(resolved_bounds)
        .penalties(penalties);
    if let Some(c) = constraint {
        builder = builder
            .generic_constraints(vec![c])
            .resolved_generic_bounds(bounds);
    }
    builder.build().expect("one_hydro_system: valid")
}

#[test]
#[allow(clippy::cast_possible_wrap)]
fn generic_constraint_chronological_stage_net_storage_one_row() {
    use novomodelo_core::ResolvedGenericConstraintBounds;
    use novomodelo_core::{
        BlockMode, ConstraintExpression, GenericConstraint, LinearTerm, VariableRef,
    };
    use std::collections::HashMap;

    let n_blks = 3_usize;
    let hydro_id = EntityId(5);

    let baseline = one_hydro_system(
        n_blks,
        BlockMode::Chronological,
        None,
        ResolvedGenericConstraintBounds::new(&HashMap::new(), std::iter::empty()),
    );
    let baseline_rows = build_templates_for(&baseline)[0].num_rows;

    let constraint = GenericConstraint {
        id: EntityId(100),
        name: "gc_ramp".to_string(),
        description: None,
        expression: ConstraintExpression {
            terms: vec![
                LinearTerm::literal(
                    1.0,
                    VariableRef::HydroStorageFinal {
                        hydro_id,
                        block_id: None,
                    },
                ),
                LinearTerm::literal(
                    -1.0,
                    VariableRef::HydroStorageInitial {
                        hydro_id,
                        block_id: None,
                    },
                ),
            ],
        },
        slack: None,
        bound_lower: None,
        bound_upper: None,
    };

    let id_map: HashMap<i32, usize> = [(100_i32, 0)].into_iter().collect();
    let rows = vec![(100_i32, 0_i32, None::<i32>, None, Some(50.0_f64))];
    let generic_bounds = ResolvedGenericConstraintBounds::new(&id_map, rows.into_iter());

    let system = one_hydro_system(
        n_blks,
        BlockMode::Chronological,
        Some(constraint),
        generic_bounds,
    );
    let t = &build_templates_for(&system)[0];

    assert_eq!(
        t.num_rows,
        baseline_rows + 1,
        "stage net storage (block_id=None) collapses to exactly one row"
    );

    // S⁰ = storage_in.start + 0 = 2; Sᴷ = storage.start + 0 = 0.
    let generic_row = baseline_rows;
    let final_total: f64 = csc_entries_at(t, 0, generic_row).iter().sum();
    assert!(
        (final_total - 1.0).abs() < f64::EPSILON,
        "expected +1.0 on stage-final Sᴷ (col 0), got {final_total}"
    );
    let initial_total: f64 = csc_entries_at(t, 2, generic_row).iter().sum();
    assert!(
        (initial_total + 1.0).abs() < f64::EPSILON,
        "expected -1.0 on stage-initial S⁰ (col 2), got {initial_total}"
    );
}

#[test]
#[allow(clippy::cast_possible_wrap)]
fn generic_constraint_chronological_specific_block_ramp_one_row() {
    use novomodelo_core::ResolvedGenericConstraintBounds;
    use novomodelo_core::{
        BlockMode, ConstraintExpression, GenericConstraint, LinearTerm, VariableRef,
    };
    use std::collections::HashMap;

    let n_blks = 3_usize;
    let hydro_id = EntityId(5);
    let target_block = 1_usize;

    let baseline = one_hydro_system(
        n_blks,
        BlockMode::Chronological,
        None,
        ResolvedGenericConstraintBounds::new(&HashMap::new(), std::iter::empty()),
    );
    let baseline_rows = build_templates_for(&baseline)[0].num_rows;

    let constraint = GenericConstraint {
        id: EntityId(100),
        name: "gc_ramp_b1".to_string(),
        description: None,
        expression: ConstraintExpression {
            terms: vec![
                LinearTerm::literal(
                    1.0,
                    VariableRef::HydroStorageFinal {
                        hydro_id,
                        block_id: Some(target_block),
                    },
                ),
                LinearTerm::literal(
                    -1.0,
                    VariableRef::HydroStorageInitial {
                        hydro_id,
                        block_id: Some(target_block),
                    },
                ),
            ],
        },
        slack: None,
        bound_lower: None,
        bound_upper: None,
    };

    let id_map: HashMap<i32, usize> = [(100_i32, 0)].into_iter().collect();
    let rows = vec![(
        100_i32,
        0_i32,
        Some(target_block as i32),
        None,
        Some(50.0_f64),
    )];
    let generic_bounds = ResolvedGenericConstraintBounds::new(&id_map, rows.into_iter());

    let system = one_hydro_system(
        n_blks,
        BlockMode::Chronological,
        Some(constraint),
        generic_bounds,
    );
    let t = &build_templates_for(&system)[0];

    assert_eq!(
        t.num_rows,
        baseline_rows + 1,
        "block_id = Some(b) ramp must produce exactly one row"
    );

    // block 1 boundaries: initial k=1 = S¹ = 4, final k=2 = S² = 5.
    let generic_row = baseline_rows;
    let final_total: f64 = csc_entries_at(t, 5, generic_row).iter().sum();
    assert!(
        (final_total - 1.0).abs() < f64::EPSILON,
        "expected +1.0 on S² (col 5), got {final_total}"
    );
    let initial_total: f64 = csc_entries_at(t, 4, generic_row).iter().sum();
    assert!(
        (initial_total + 1.0).abs() < f64::EPSILON,
        "expected -1.0 on S¹ (col 4), got {initial_total}"
    );
}

/// Objective coefficient of the slack on a soft `S ≤ 150 hm³` storage row, on a
/// parallel stage of `n_blks` 720-hour blocks.
fn storage_slack_objective(n_blks: usize, penalty_basis: novomodelo_core::PenaltyBasis) -> f64 {
    use novomodelo_core::ResolvedGenericConstraintBounds;
    use novomodelo_core::{
        BlockMode, ConstraintExpression, GenericConstraint, LinearTerm, SlackConfig, VariableRef,
    };
    use std::collections::HashMap;

    let baseline = one_hydro_system(
        n_blks,
        BlockMode::Parallel,
        None,
        ResolvedGenericConstraintBounds::new(&HashMap::new(), std::iter::empty()),
    );
    let baseline_tpl = &build_templates_for(&baseline)[0];

    let constraint = GenericConstraint {
        id: EntityId(100),
        name: "gc_storage_cap".to_string(),
        description: None,
        expression: ConstraintExpression {
            terms: vec![LinearTerm::literal(
                1.0,
                VariableRef::HydroStorage {
                    hydro_id: EntityId(5),
                },
            )],
        },
        slack: Some(SlackConfig {
            penalty: 40.0,
            penalty_basis,
        }),
        bound_lower: None,
        bound_upper: None,
    };
    let id_map: HashMap<i32, usize> = [(100_i32, 0)].into_iter().collect();
    let rows = vec![(100_i32, 0_i32, None::<i32>, None, Some(150.0_f64))];
    let generic_bounds = ResolvedGenericConstraintBounds::new(&id_map, rows.into_iter());

    let system = one_hydro_system(
        n_blks,
        BlockMode::Parallel,
        Some(constraint),
        generic_bounds,
    );
    let t = &build_templates_for(&system)[0];

    assert_eq!(t.num_cols, baseline_tpl.num_cols + 1);
    assert_eq!(t.num_rows, baseline_tpl.num_rows + 1);
    let slack_col = baseline_tpl.num_cols;
    let generic_row = baseline_tpl.num_rows;
    let entry: f64 = csc_entries_at(t, slack_col, generic_row).iter().sum();
    assert!(
        (entry + 1.0).abs() < f64::EPSILON,
        "`<=` slack must carry -1.0 on the storage row, got {entry}"
    );
    t.objective[slack_col]
}

/// derived: a `per_unit` slack costs `penalty / COST_SCALE_FACTOR` per hm³ short on
/// any stage length; its `per_hour` twin costs `penalty × stage_hours / COST_SCALE_FACTOR`.
#[test]
#[allow(clippy::cast_precision_loss)]
fn per_unit_storage_slack_price_ignores_stage_hours() {
    use novomodelo_core::PenaltyBasis;

    let penalty = 40.0_f64;
    for n_blks in [1_usize, 3] {
        let stage_hours = 720.0 * n_blks as f64;

        let per_unit = storage_slack_objective(n_blks, PenaltyBasis::PerUnit);
        let expected_per_unit = penalty / COST_SCALE_FACTOR;
        assert!(
            (per_unit - expected_per_unit).abs() < 1e-12,
            "{n_blks} blocks: per_unit slack must cost {expected_per_unit}, got {per_unit}"
        );

        let per_hour = storage_slack_objective(n_blks, PenaltyBasis::PerHour);
        let expected_per_hour = penalty * stage_hours / COST_SCALE_FACTOR;
        assert!(
            (per_hour - expected_per_hour).abs() < 1e-9,
            "{n_blks} blocks: per_hour slack must cost {expected_per_hour}, got {per_hour}"
        );
        assert!(
            (per_hour / per_unit - stage_hours).abs() < 1e-9,
            "the per_hour twin must cost {stage_hours}× the per_unit slack"
        );
    }
}

// ── Parameter (stage, block) axis + symbolic RHS bound byte-identity twins ──
//
// These twins prove the resolved parameter axis and `@name`-bounded constraints
// desugar to the unchanged LP core. Unlike the named-expression twins above, the
// sugared and hand-flattened forms are NOT equal as `Vec<GenericConstraint>` (a
// parameter coefficient / symbolic bound vs literals) and need different resolved
// parameters per side, so `assert_lp_byte_identical` (which asserts flat-form
// identity under one shared parameter table) does not apply — the identity lives
// only at the built `StageTemplates`, compared field-by-field (bitwise) via
// `test_support::assert_all_templates_byte_identical` below.

/// Resolve `params` over a single stage with `n_blks` blocks through the real
/// resolver (the path `scalar_parameters_declaration_order.rs` exercises).
/// `cost_scale_factor` is the default so a twin built here divides its objective
/// column identically to one built with [`ResolvedParameters::default`].
fn resolved_params_single_stage(
    n_blks: usize,
    params: &[novomodelo_core::ScalarParameter],
) -> ResolvedParameters {
    use novomodelo_core::StageId;
    use novomodelo_sddp::build_resolved_parameters;
    use novomodelo_sddp::energy_conversion::{
        EnergyConversionSet, HydroEnergyProductivityOverride,
    };

    let n_stages = 1_usize;
    let ec = EnergyConversionSet::new(vec![], vec![], &[], n_stages);
    let overrides = HydroEnergyProductivityOverride::default();
    let hydros: Vec<novomodelo_core::Hydro> = Vec::new();
    build_resolved_parameters(
        params,
        &ec,
        &overrides,
        &hydros,
        &[0_i32],
        &[StageId(0)],
        &[n_blks],
        COST_SCALE_FACTOR,
    )
    .expect("resolved parameters build")
}

/// Build the stage templates for `system` under `resolved` — the `build_templates_for`
/// path with a caller-supplied parameter table instead of the default empty one.
fn build_templates_with_params(
    system: &novomodelo_core::System,
    resolved: &ResolvedParameters,
) -> Vec<novomodelo_solver::StageTemplate> {
    build_stage_templates_resolving_layout(
        system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &default_production(system),
        &default_evaporation(system),
        resolved,
    )
    .expect("build templates with params")
    .templates
}

/// Discharge the desugaring invariant on one twin whose sugared and hand-flattened
/// forms differ in their flat representation: build each study under its own
/// resolved parameters and assert every stage's `StageTemplate` is byte-identical.
#[allow(clippy::too_many_arguments)]
fn assert_desugaring_twin_builds_identical_lp(
    n_blks: usize,
    sugared_constraints: Vec<novomodelo_core::GenericConstraint>,
    sugared_bounds: novomodelo_core::ResolvedGenericConstraintBounds,
    sugared_params: &ResolvedParameters,
    flat_constraints: Vec<novomodelo_core::GenericConstraint>,
    flat_bounds: novomodelo_core::ResolvedGenericConstraintBounds,
    flat_params: &ResolvedParameters,
) {
    let sugared_tpl = build_templates_with_params(
        &one_bus_system_n_blks_with_generic(n_blks, sugared_constraints, sugared_bounds),
        sugared_params,
    );
    let flat_tpl = build_templates_with_params(
        &one_bus_system_n_blks_with_generic(n_blks, flat_constraints, flat_bounds),
        flat_params,
    );
    assert_all_templates_byte_identical(&sugared_tpl, &flat_tpl, "sugared vs hand-flattened");
}

/// A `PerStageBlock` parameter used as a coefficient resolves to its own block's
/// value on each per-block row. The sugared study places two terms — each a
/// distinct block-varying parameter — on the SAME excess column; the two terms
/// merge into one entry per block row, so the twin checks the summed per-block
/// value. The hand-flattened twin writes the resolved per-block literals
/// directly, one block-confined constraint per block, and must build the
/// identical LP.
#[test]
fn per_stage_block_coefficient_twin_matches_hand_flattened_literals() {
    use novomodelo_core::{
        ConstraintExpression, GenericConstraint, LinearTerm, ParameterKind,
        ResolvedGenericConstraintBounds, ScalarExpr, ScalarParameter, VariableRef,
    };
    use std::collections::HashMap;

    let n_blks = 2_usize;
    let upper = 400.0_f64;
    let excess = |block_id: Option<usize>| VariableRef::BusExcess {
        bus_id: EntityId(1),
        block_id,
    };

    // Sugared: one block-varying constraint, two per-block parameter coefficients
    // on the same excess column. Block 0 resolves to {2.0, 3.0}; block 1 to {5.0,
    // 7.0}. Parameter ids ascend with their per-block values so the parameter-keyed
    // and literal-keyed canonical orders agree.
    let mut sugared_expr = ConstraintExpression {
        terms: vec![
            LinearTerm {
                coefficient: ScalarExpr::param(EntityId(200)),
                variable: excess(None),
            },
            LinearTerm {
                coefficient: ScalarExpr::param(EntityId(201)),
                variable: excess(None),
            },
        ],
    };
    sugared_expr.canonicalize();
    let sugared_constraint = GenericConstraint {
        id: EntityId(300),
        name: "gc_block_coef".to_string(),
        description: None,
        expression: sugared_expr,
        slack: None,
        bound_lower: None,
        bound_upper: None,
    };
    let sugared_id_map: HashMap<i32, usize> = [(300_i32, 0)].into_iter().collect();
    let sugared_rows = vec![(300_i32, 0_i32, None::<i32>, None, Some(upper))];
    let sugared_bounds =
        ResolvedGenericConstraintBounds::new(&sugared_id_map, sugared_rows.into_iter());
    let sugared_params = resolved_params_single_stage(
        n_blks,
        &[
            ScalarParameter {
                id: EntityId(200),
                name: "coef_a".to_string(),
                kind: ParameterKind::PerStageBlock {
                    values: vec![(0, 0, 2.0), (0, 1, 5.0)],
                },
            },
            ScalarParameter {
                id: EntityId(201),
                name: "coef_b".to_string(),
                kind: ParameterKind::PerStageBlock {
                    values: vec![(0, 0, 3.0), (0, 1, 7.0)],
                },
            },
        ],
    );

    // Hand-flattened: one block-confined constraint per block, each carrying the
    // resolved per-block literals directly. Block-specific terms keep every entry
    // in its own block's column, matching the sugared per-block rows.
    let flat_constraint = |id: i32, block: usize, a: f64, b: f64| {
        let mut expr = ConstraintExpression {
            terms: vec![
                LinearTerm::literal(a, excess(Some(block))),
                LinearTerm::literal(b, excess(Some(block))),
            ],
        };
        expr.canonicalize();
        GenericConstraint {
            id: EntityId(id),
            name: format!("gc_lit_{id}"),
            description: None,
            expression: expr,
            slack: None,
            bound_lower: None,
            bound_upper: None,
        }
    };
    let flat_constraints = vec![
        flat_constraint(100, 0, 2.0, 3.0),
        flat_constraint(101, 1, 5.0, 7.0),
    ];
    let flat_id_map: HashMap<i32, usize> = [(100_i32, 0), (101_i32, 1)].into_iter().collect();
    let flat_rows = vec![
        (100_i32, 0_i32, Some(0_i32), None, Some(upper)),
        (101_i32, 0_i32, Some(1_i32), None, Some(upper)),
    ];
    let flat_bounds = ResolvedGenericConstraintBounds::new(&flat_id_map, flat_rows.into_iter());

    assert_desugaring_twin_builds_identical_lp(
        n_blks,
        vec![sugared_constraint],
        sugared_bounds,
        &sugared_params,
        flat_constraints,
        flat_bounds,
        &ResolvedParameters::default(),
    );
}

/// A symbolic upper bound (`bound_upper` naming a `PerStageBlock` parameter,
/// with the numeric `bound_upper` left null in the activation rows) resolves per
/// `(stage, block)` and builds the identical LP to a study whose rows carry the
/// resolved literal `bound_upper` per block and no reference. The block-varying
/// reference keeps the rows per-block on both sides.
#[test]
fn symbolic_upper_bound_ref_twin_matches_literal_per_block_bounds() {
    use novomodelo_core::{
        ConstraintExpression, GenericConstraint, LinearTerm, ParameterKind,
        ResolvedGenericConstraintBounds, ScalarExpr, ScalarParameter, VariableRef,
    };
    use std::collections::HashMap;

    let n_blks = 2_usize;
    let cap_block0 = 120.0_f64;
    let cap_block1 = 340.0_f64;
    let expr = || ConstraintExpression {
        terms: vec![LinearTerm::literal(
            1.0,
            VariableRef::BusExcess {
                bus_id: EntityId(1),
                block_id: None,
            },
        )],
    };

    // Sugared: the reference lives on the constraint; the activation row's numeric
    // upper endpoint stays null, so the parameter's (stage, block) axis supplies it.
    let sugared_constraint = GenericConstraint {
        id: EntityId(300),
        name: "gc_sym_cap".to_string(),
        description: None,
        expression: expr(),
        slack: None,
        bound_lower: None,
        bound_upper: Some(ScalarExpr::param(EntityId(200))),
    };
    let sugared_id_map: HashMap<i32, usize> = [(300_i32, 0)].into_iter().collect();
    let sugared_rows = vec![(300_i32, 0_i32, None::<i32>, None, None)];
    let sugared_bounds =
        ResolvedGenericConstraintBounds::new(&sugared_id_map, sugared_rows.into_iter());
    let sugared_params = resolved_params_single_stage(
        n_blks,
        &[ScalarParameter {
            id: EntityId(200),
            name: "cap".to_string(),
            kind: ParameterKind::PerStageBlock {
                values: vec![(0, 0, cap_block0), (0, 1, cap_block1)],
            },
        }],
    );

    // Hand-flattened: no reference, resolved literal upper per (stage, block).
    let flat_constraint = GenericConstraint {
        id: EntityId(300),
        name: "gc_sym_cap".to_string(),
        description: None,
        expression: expr(),
        slack: None,
        bound_lower: None,
        bound_upper: None,
    };
    let flat_id_map: HashMap<i32, usize> = [(300_i32, 0)].into_iter().collect();
    let flat_rows = vec![
        (300_i32, 0_i32, Some(0_i32), None, Some(cap_block0)),
        (300_i32, 0_i32, Some(1_i32), None, Some(cap_block1)),
    ];
    let flat_bounds = ResolvedGenericConstraintBounds::new(&flat_id_map, flat_rows.into_iter());

    assert_desugaring_twin_builds_identical_lp(
        n_blks,
        vec![sugared_constraint],
        sugared_bounds,
        &sugared_params,
        vec![flat_constraint],
        flat_bounds,
        &ResolvedParameters::default(),
    );
}

/// A block-independent expression whose symbolic lower bound (`bound_lower`)
/// names a stage-level (broadcast) parameter collapses to a single stage-level row,
/// and the collapse path resolves the reference to the parameter's one-per-stage
/// value — byte-identical to the same study written with a literal `bound_lower`
/// and no reference.
#[test]
fn symbolic_lower_bound_ref_stage_level_collapse_twin() {
    use novomodelo_core::{
        ConstraintExpression, GenericConstraint, ParameterKind, ResolvedGenericConstraintBounds,
        ScalarExpr, ScalarParameter,
    };
    use std::collections::HashMap;

    let n_blks = 3_usize;
    let floor = 275.0_f64;
    let empty_expr = || ConstraintExpression { terms: vec![] };

    // Sugared: block-independent (empty) expression, stage-level floor reference,
    // numeric lower endpoint null in the activation row.
    let sugared_constraint = GenericConstraint {
        id: EntityId(300),
        name: "gc_sym_floor".to_string(),
        description: None,
        expression: empty_expr(),
        slack: None,
        bound_lower: Some(ScalarExpr::param(EntityId(200))),
        bound_upper: None,
    };
    let sugared_id_map: HashMap<i32, usize> = [(300_i32, 0)].into_iter().collect();
    let sugared_rows = vec![(300_i32, 0_i32, None::<i32>, None, None)];
    let sugared_bounds =
        ResolvedGenericConstraintBounds::new(&sugared_id_map, sugared_rows.into_iter());
    let sugared_params = resolved_params_single_stage(
        n_blks,
        &[ScalarParameter {
            id: EntityId(200),
            name: "floor".to_string(),
            kind: ParameterKind::Constant { value: floor },
        }],
    );

    // Hand-flattened: no reference, literal lower endpoint on the same collapsing row.
    let flat_constraint = GenericConstraint {
        id: EntityId(300),
        name: "gc_sym_floor".to_string(),
        description: None,
        expression: empty_expr(),
        slack: None,
        bound_lower: None,
        bound_upper: None,
    };
    let flat_id_map: HashMap<i32, usize> = [(300_i32, 0)].into_iter().collect();
    let flat_rows = vec![(300_i32, 0_i32, None::<i32>, Some(floor), None)];
    let flat_bounds = ResolvedGenericConstraintBounds::new(&flat_id_map, flat_rows.into_iter());

    assert_desugaring_twin_builds_identical_lp(
        n_blks,
        vec![sugared_constraint],
        sugared_bounds,
        &sugared_params,
        vec![flat_constraint],
        flat_bounds,
        &ResolvedParameters::default(),
    );
}

/// Parse `json` through the real loader with `name_to_id` supplying `@name`
/// parameter resolution — the path a study with genuine `@param` coefficients
/// exercises, unlike [`parse_generic_from_str`]'s empty table.
fn parse_generic_from_str_with_params(
    json: &str,
    name_to_id: &std::collections::HashMap<String, novomodelo_core::EntityId>,
) -> Vec<novomodelo_core::GenericConstraint> {
    use std::io::Write;

    let mut file = tempfile::NamedTempFile::new().expect("create tempfile");
    file.write_all(json.as_bytes()).expect("write json fixture");
    novomodelo_io::constraints::parse_generic_constraints(
        file.path(),
        name_to_id,
        &novomodelo_io::constraints::LineBusPairIndex::default(),
    )
    .expect("parse generic constraints")
}

/// Parse `json` through the real scalar-parameter loader, so an expression body
/// reaches the resolver exactly as an authored `generic_parameters.json` does.
fn parse_scalar_parameters_from_str(json: &str) -> Vec<novomodelo_core::ScalarParameter> {
    use std::io::Write;

    let mut file = tempfile::NamedTempFile::new().expect("create tempfile");
    file.write_all(json.as_bytes()).expect("write json fixture");
    novomodelo_io::parse_scalar_parameters_json(file.path()).expect("parse scalar parameters")
}

/// A decision-variable RHS (`thermal_generation(0) <= 0.5 * hydro_generation(1) +
/// 12`) normalizes onto the LHS with the RHS variable sign-flipped and the
/// constant remainder assigned to the upper endpoint — byte-identical to writing
/// the already-normalized inequality by hand.
#[test]
fn decision_variable_rhs_normalizes_to_hand_flattened_form() {
    use novomodelo_core::ResolvedGenericConstraintBounds;
    use std::collections::HashMap;

    let sugared = r#"{
      "constraints": [
        { "id": 1, "name": "cap", "expression": "thermal_generation(0) <= 0.5 * hydro_generation(1) + 12", "slack": { "enabled": false } }
      ]
    }"#;
    let flat = r#"{
      "constraints": [
        { "id": 1, "name": "cap", "expression": "thermal_generation(0) - 0.5 * hydro_generation(1) <= 12", "slack": { "enabled": false } }
      ]
    }"#;

    let id_map: HashMap<i32, usize> = [(1_i32, 0)].into_iter().collect();
    let rows = vec![(1_i32, 0_i32, None::<i32>, None, None)];
    let bounds = ResolvedGenericConstraintBounds::new(&id_map, rows.into_iter());

    assert_lp_byte_identical(sugared, flat, 1, &bounds);
}

/// A same-variable, same-coefficient-kind repeat on one side (`2 *
/// hydro_generation(0) - hydro_generation(0)`) merges to its summed effective
/// coefficient before the bound fold, rather than reaching the LP as two
/// separate columns.
#[test]
fn same_variable_repeat_merges_to_summed_coefficient() {
    use novomodelo_core::ResolvedGenericConstraintBounds;
    use std::collections::HashMap;

    let sugared = r#"{
      "constraints": [
        { "id": 2, "name": "merge", "expression": "2 * hydro_generation(0) - hydro_generation(0) <= 5", "slack": { "enabled": false } }
      ]
    }"#;
    let flat = r#"{
      "constraints": [
        { "id": 2, "name": "merge", "expression": "hydro_generation(0) <= 5", "slack": { "enabled": false } }
      ]
    }"#;

    let parsed = parse_generic_from_str(sugared);
    assert_eq!(parsed[0].expression.terms.len(), 1);
    assert_eq!(
        parsed[0].expression.terms[0].coefficient.as_literal(),
        Some(1.0)
    );

    let id_map: HashMap<i32, usize> = [(2_i32, 0)].into_iter().collect();
    let rows = vec![(2_i32, 0_i32, None::<i32>, None, None)];
    let bounds = ResolvedGenericConstraintBounds::new(&id_map, rows.into_iter());

    assert_lp_byte_identical(sugared, flat, 1, &bounds);
}

/// A full cancellation (`hydro_generation(0) - hydro_generation(0)`) merges to a
/// net-zero column and drops it, leaving a term-less LHS whose row still carries
/// the bound — the same shape a modeller declaring only the bound, with no
/// expression terms at all, would produce.
#[test]
fn full_cancellation_drops_net_zero_column() {
    use novomodelo_core::{
        ConstraintExpression, GenericConstraint, ResolvedGenericConstraintBounds,
    };
    use std::collections::HashMap;

    let sugared = r#"{
      "constraints": [
        { "id": 3, "name": "cancel", "expression": "hydro_generation(0) - hydro_generation(0) <= 5", "slack": { "enabled": false } }
      ]
    }"#;
    let sugared_constraints = parse_generic_from_str(sugared);
    assert!(sugared_constraints[0].expression.terms.is_empty());

    let sugared_id_map: HashMap<i32, usize> = [(3_i32, 0)].into_iter().collect();
    let sugared_rows = vec![(3_i32, 0_i32, None::<i32>, None, None)];
    let sugared_bounds =
        ResolvedGenericConstraintBounds::new(&sugared_id_map, sugared_rows.into_iter());

    let flat_constraint = GenericConstraint {
        id: EntityId(3),
        name: "cancel".to_string(),
        description: None,
        expression: ConstraintExpression { terms: vec![] },
        slack: None,
        bound_lower: None,
        bound_upper: None,
    };
    let flat_id_map: HashMap<i32, usize> = [(3_i32, 0)].into_iter().collect();
    let flat_rows = vec![(3_i32, 0_i32, None::<i32>, None, Some(5.0_f64))];
    let flat_bounds = ResolvedGenericConstraintBounds::new(&flat_id_map, flat_rows.into_iter());

    assert_desugaring_twin_builds_identical_lp(
        1,
        sugared_constraints,
        sugared_bounds,
        &ResolvedParameters::default(),
        vec![flat_constraint],
        flat_bounds,
        &ResolvedParameters::default(),
    );
}

/// A coefficient-scaled paren group (`2 * (hydro_generation(0) -
/// hydro_generation(1))`) distributes to the same per-variable coefficients as
/// the hand-flattened `2 * hydro_generation(0) - 2 * hydro_generation(1)`, and
/// both must build the identical LP.
#[test]
fn paren_distribution_normalizes_to_hand_flattened_terms() {
    use novomodelo_core::ResolvedGenericConstraintBounds;
    use std::collections::HashMap;

    let sugared = r#"{
      "constraints": [
        { "id": 4, "name": "paren", "expression": "2 * (hydro_generation(0) - hydro_generation(1)) <= 10", "slack": { "enabled": false } }
      ]
    }"#;
    let flat = r#"{
      "constraints": [
        { "id": 4, "name": "paren", "expression": "2 * hydro_generation(0) - 2 * hydro_generation(1) <= 10", "slack": { "enabled": false } }
      ]
    }"#;

    let id_map: HashMap<i32, usize> = [(4_i32, 0)].into_iter().collect();
    let rows = vec![(4_i32, 0_i32, None::<i32>, None, None)];
    let bounds = ResolvedGenericConstraintBounds::new(&id_map, rows.into_iter());

    assert_desugaring_twin_builds_identical_lp(
        1,
        parse_generic_from_str(sugared),
        bounds.clone(),
        &ResolvedParameters::default(),
        parse_generic_from_str(flat),
        bounds,
        &ResolvedParameters::default(),
    );
}

/// A parquet-base upper endpoint (`100`) folds with an inline affine remainder
/// established from the LHS constant (`+ 5`, negated to `-5`) to the same
/// effective bound as the pre-folded literal `95` — `fold_endpoint`'s `base +
/// remainder` arm, not the base or the remainder alone.
#[test]
fn parquet_base_folds_with_inline_affine_remainder() {
    use novomodelo_core::{
        ConstraintExpression, GenericConstraint, LinearTerm, ResolvedGenericConstraintBounds,
        VariableRef,
    };
    use std::collections::HashMap;

    let sugared = r#"{
      "constraints": [
        { "id": 5, "name": "fold", "expression": "thermal_generation(0) + 5 <= 0", "slack": { "enabled": false } }
      ]
    }"#;
    let sugared_constraints = parse_generic_from_str(sugared);
    let sugared_id_map: HashMap<i32, usize> = [(5_i32, 0)].into_iter().collect();
    let sugared_rows = vec![(5_i32, 0_i32, None::<i32>, None, Some(100.0_f64))];
    let sugared_bounds =
        ResolvedGenericConstraintBounds::new(&sugared_id_map, sugared_rows.into_iter());

    let flat_constraint = GenericConstraint {
        id: EntityId(5),
        name: "fold".to_string(),
        description: None,
        expression: ConstraintExpression {
            terms: vec![LinearTerm::literal(
                1.0,
                VariableRef::ThermalGeneration {
                    thermal_id: EntityId(0),
                    block_id: None,
                },
            )],
        },
        slack: None,
        bound_lower: None,
        bound_upper: None,
    };
    let flat_id_map: HashMap<i32, usize> = [(5_i32, 0)].into_iter().collect();
    let flat_rows = vec![(5_i32, 0_i32, None::<i32>, None, Some(95.0_f64))];
    let flat_bounds = ResolvedGenericConstraintBounds::new(&flat_id_map, flat_rows.into_iter());

    assert_desugaring_twin_builds_identical_lp(
        1,
        sugared_constraints,
        sugared_bounds,
        &ResolvedParameters::default(),
        vec![flat_constraint],
        flat_bounds,
        &ResolvedParameters::default(),
    );
}

/// A parameter-coefficient RHS (`@a * bus_excess(1) <= -@b * bus_excess(1) +
/// 400`) lowers to one `bus_excess(1)` term whose coefficient program reads
/// both `@a` and `@b`. The sugared template holds a single value on that column
/// and matches the hand-flattened twin.
///
/// derived: 2.0 + 6.0
///
/// Guarded mutation: changing the hand-flattened side's second literal away
/// from `@b`'s resolved value (`6.0`, independent of the first term's `2.0`)
/// changes the merged entry and fails the digest comparison — this twin is
/// sensitive to both terms' values, not merely their count.
#[test]
fn parameter_rhs_terms_on_one_column_merge_to_one_entry() {
    use novomodelo_core::{
        ConstraintExpression, GenericConstraint, LinearTerm, ParameterKind,
        ResolvedGenericConstraintBounds, ScalarParameter, VariableRef,
    };
    use std::collections::HashMap;

    let excess = VariableRef::BusExcess {
        bus_id: EntityId(1),
        block_id: None,
    };

    let sugared_json = r#"{
      "constraints": [
        { "id": 6, "name": "param_rhs", "expression": "@a * bus_excess(1) <= -@b * bus_excess(1) + 400", "slack": { "enabled": false } }
      ]
    }"#;
    let name_to_id: HashMap<String, EntityId> = [
        ("a".to_string(), EntityId(220)),
        ("b".to_string(), EntityId(221)),
    ]
    .into_iter()
    .collect();
    let sugared_constraints = parse_generic_from_str_with_params(sugared_json, &name_to_id);
    let sugared_id_map: HashMap<i32, usize> = [(6_i32, 0)].into_iter().collect();
    let sugared_rows = vec![(6_i32, 0_i32, None::<i32>, None, None)];
    let sugared_bounds =
        ResolvedGenericConstraintBounds::new(&sugared_id_map, sugared_rows.into_iter());
    let sugared_params = resolved_params_single_stage(
        1,
        &[
            ScalarParameter {
                id: EntityId(220),
                name: "a".to_string(),
                kind: ParameterKind::Constant { value: 2.0 },
            },
            ScalarParameter {
                id: EntityId(221),
                name: "b".to_string(),
                kind: ParameterKind::Constant { value: 6.0 },
            },
        ],
    );

    let mut flat_expr = ConstraintExpression {
        terms: vec![
            LinearTerm::literal(2.0, excess),
            LinearTerm::literal(6.0, excess),
        ],
    };
    flat_expr.canonicalize();
    let flat_constraint = GenericConstraint {
        id: EntityId(6),
        name: "param_rhs".to_string(),
        description: None,
        expression: flat_expr,
        slack: None,
        bound_lower: None,
        bound_upper: None,
    };
    let flat_id_map: HashMap<i32, usize> = [(6_i32, 0)].into_iter().collect();
    let flat_rows = vec![(6_i32, 0_i32, None::<i32>, None, Some(400.0_f64))];
    let flat_bounds = ResolvedGenericConstraintBounds::new(&flat_id_map, flat_rows.into_iter());

    // Layout (N=0, T=0, B=1, K=1): theta=0, deficit=1, excess=2; load_balance row 0,
    // generic row 1.
    let (excess_col, generic_row) = (2_usize, 1_usize);
    let sugared_tpl = build_templates_with_params(
        &one_bus_system_n_blks_with_generic(1, sugared_constraints.clone(), sugared_bounds.clone()),
        &sugared_params,
    );
    assert_eq!(
        csc_entries_at(&sugared_tpl[0], excess_col, generic_row),
        vec![8.0],
        "the two terms on bus_excess(1) must merge into one entry"
    );

    assert_desugaring_twin_builds_identical_lp(
        1,
        sugared_constraints,
        sugared_bounds,
        &sugared_params,
        vec![flat_constraint],
        flat_bounds,
        &ResolvedParameters::default(),
    );
}

/// Two studies whose constraint declaration order differs (id 21 before id 20,
/// or after) and whose id-20 RHS terms are authored in a different commutative
/// order build byte-identical flat forms and `StageTemplates` — the id-ascending
/// constraint sort and the RHS fold are both pure functions of content, never
/// authoring order.
#[test]
fn declaration_order_and_rhs_term_order_are_invariant_in_lp() {
    use novomodelo_core::ResolvedGenericConstraintBounds;
    use std::collections::HashMap;

    let doc_a = r#"{
      "constraints": [
        { "id": 20, "name": "c20", "expression": "thermal_generation(0) <= 0.5 * hydro_generation(1) + 3 * hydro_generation(2) + 12", "slack": { "enabled": false } },
        { "id": 21, "name": "c21", "expression": "bus_excess(1) <= 20", "slack": { "enabled": false } }
      ]
    }"#;
    let doc_b = r#"{
      "constraints": [
        { "id": 21, "name": "c21", "expression": "bus_excess(1) <= 20", "slack": { "enabled": false } },
        { "id": 20, "name": "c20", "expression": "thermal_generation(0) <= 12 + 3 * hydro_generation(2) + 0.5 * hydro_generation(1)", "slack": { "enabled": false } }
      ]
    }"#;

    let parsed_a = parse_generic_from_str(doc_a);
    let parsed_b = parse_generic_from_str(doc_b);
    assert_eq!(
        parsed_a, parsed_b,
        "declaration order and commutative RHS term order must not affect the flat form"
    );

    let id_map: HashMap<i32, usize> = [(20_i32, 0), (21_i32, 1)].into_iter().collect();
    let rows = vec![
        (20_i32, 0_i32, None::<i32>, None, None),
        (21_i32, 0_i32, None::<i32>, None, None),
    ];
    let bounds_a = ResolvedGenericConstraintBounds::new(&id_map, rows.clone().into_iter());
    let bounds_b = ResolvedGenericConstraintBounds::new(&id_map, rows.into_iter());

    let tpl_a = build_templates_for(&one_bus_system_n_blks_with_generic(1, parsed_a, bounds_a));
    let tpl_b = build_templates_for(&one_bus_system_n_blks_with_generic(1, parsed_b, bounds_b));
    assert_all_templates_byte_identical(&tpl_a, &tpl_b, "declaration order and commutative RHS");
}

/// An operator-free expression repeating one variable (`bus_excess(1) +
/// bus_excess(1)`) keeps both terms through the loader, so the builder alone
/// merges them: the template holds one entry on the excess column and loads and
/// solves on the active backend.
///
/// derived: 1.0 + 1.0
#[test]
fn operator_free_repeated_variable_loads_one_entry_and_solves() {
    use novomodelo_core::ResolvedGenericConstraintBounds;
    use novomodelo_solver::{ActiveSolver, SolverInterface};
    use std::collections::HashMap;

    let json = r#"{
      "constraints": [
        { "id": 8, "name": "repeat", "expression": "bus_excess(1) + bus_excess(1)", "slack": { "enabled": false } }
      ]
    }"#;
    let constraints = parse_generic_from_str(json);
    let id_map: HashMap<i32, usize> = [(8_i32, 0)].into_iter().collect();
    let rows = vec![(8_i32, 0_i32, None::<i32>, None, Some(400.0_f64))];
    let bounds = ResolvedGenericConstraintBounds::new(&id_map, rows.into_iter());

    // Layout (N=0, T=0, B=1, K=1): theta=0, deficit=1, excess=2; load_balance row 0,
    // generic row 1.
    let (excess_col, generic_row) = (2_usize, 1_usize);
    let templates =
        build_templates_for(&one_bus_system_n_blks_with_generic(1, constraints, bounds));
    assert_eq!(
        csc_entries_at(&templates[0], excess_col, generic_row),
        vec![2.0],
        "the two terms on bus_excess(1) must merge into one entry"
    );

    let mut solver = ActiveSolver::new().expect("ActiveSolver::new must succeed");
    solver.load_model(&templates[0]);
    solver
        .solve(None)
        .expect("merged generic row must load and solve");
}

/// A coefficient parameter resolving to exactly zero on one block stores no entry
/// on that block's row, while the block where it resolves nonzero keeps its value.
///
/// derived: block 0 coefficient 0.0 stores nothing; block 1 coefficient 3.0
#[test]
fn block_varying_zero_parameter_coefficient_has_no_entry() {
    use novomodelo_core::{ParameterKind, ResolvedGenericConstraintBounds, ScalarParameter};
    use std::collections::HashMap;

    let json = r#"{
      "constraints": [
        { "id": 9, "name": "zero_coef", "expression": "@a * bus_excess(1) <= 400", "slack": { "enabled": false } }
      ]
    }"#;
    let name_to_id: HashMap<String, EntityId> =
        [("a".to_string(), EntityId(230))].into_iter().collect();
    let constraints = parse_generic_from_str_with_params(json, &name_to_id);
    let id_map: HashMap<i32, usize> = [(9_i32, 0)].into_iter().collect();
    let rows = vec![(9_i32, 0_i32, None::<i32>, None, None)];
    let bounds = ResolvedGenericConstraintBounds::new(&id_map, rows.into_iter());
    let params = resolved_params_single_stage(
        2,
        &[ScalarParameter {
            id: EntityId(230),
            name: "a".to_string(),
            kind: ParameterKind::PerStageBlock {
                values: vec![(0, 0, 0.0), (0, 1, 3.0)],
            },
        }],
    );

    // Layout (N=0, T=0, B=1, K=2): theta=0, deficit=[1,3), excess=[3,5); load_balance
    // rows 0..2, generic rows 2 (block 0) and 3 (block 1).
    let templates = build_templates_with_params(
        &one_bus_system_n_blks_with_generic(2, constraints, bounds),
        &params,
    );
    assert_eq!(
        csc_entries_at(&templates[0], 3, 2),
        Vec::<f64>::new(),
        "block 0: a coefficient resolving to exactly zero stores no entry"
    );
    assert_eq!(
        csc_entries_at(&templates[0], 4, 3),
        vec![3.0],
        "block 1: the nonzero coefficient is stored"
    );
}

/// A summed parameter used as a coefficient builds the same LP as the inline sum. The body
/// is written in the inline form's association, because floating-point addition is not
/// associative.
#[test]
fn expression_parameter_sum_coefficient_matches_inline_sum_in_the_same_association() {
    use novomodelo_core::ResolvedGenericConstraintBounds;
    use std::collections::HashMap;

    let scalar_params = parse_scalar_parameters_from_str(
        r#"{
      "scalar_parameters": [
        { "id": 230, "name": "a", "kind": "constant", "value": 0.1 },
        { "id": 231, "name": "b", "kind": "constant", "value": 0.2 },
        { "id": 232, "name": "c", "kind": "constant", "value": 0.3 },
        { "id": 233, "name": "e", "kind": "expression", "expression": "@a + @b + @c" }
      ]
    }"#,
    );
    let name_to_id: HashMap<String, EntityId> = scalar_params
        .iter()
        .map(|param| (param.name.clone(), param.id))
        .collect();
    let params = resolved_params_single_stage(1, &scalar_params);

    let inline_constraints = parse_generic_from_str_with_params(
        r#"{
      "constraints": [
        { "id": 7, "name": "inline_sum", "expression": "(@a + @b + @c) * bus_excess(1) <= 400", "slack": { "enabled": false } }
      ]
    }"#,
        &name_to_id,
    );
    let expression_constraints = parse_generic_from_str_with_params(
        r#"{
      "constraints": [
        { "id": 7, "name": "inline_sum", "expression": "@e * bus_excess(1) <= 400", "slack": { "enabled": false } }
      ]
    }"#,
        &name_to_id,
    );
    let id_map: HashMap<i32, usize> = [(7_i32, 0)].into_iter().collect();
    let rows = vec![(7_i32, 0_i32, None::<i32>, None, None)];
    let inline_bounds = ResolvedGenericConstraintBounds::new(&id_map, rows.clone().into_iter());
    let expression_bounds = ResolvedGenericConstraintBounds::new(&id_map, rows.into_iter());

    let (excess_col, generic_row) = (2_usize, 1_usize);
    let expression_tpl = build_templates_with_params(
        &one_bus_system_n_blks_with_generic(
            1,
            expression_constraints.clone(),
            expression_bounds.clone(),
        ),
        &params,
    );
    let entries = csc_entries_at(&expression_tpl[0], excess_col, generic_row);
    assert_eq!(
        entries.len(),
        1,
        "sum-coefficient twin: the expression row must hold one entry on bus_excess(1)"
    );
    assert_eq!(
        entries[0].to_bits(),
        ((0.1_f64 + 0.2) + 0.3).to_bits(),
        "sum-coefficient twin: the entry must be the sum in the inline association"
    );

    assert_desugaring_twin_builds_identical_lp(
        1,
        inline_constraints,
        inline_bounds,
        &params,
        expression_constraints,
        expression_bounds,
        &params,
    );
}

/// A parameter built from a block-varying leaf resolves per block, and a coefficient that
/// reads it builds the same LP as the inline product.
#[test]
fn block_varying_expression_parameter_coefficient_matches_inline_product() {
    use novomodelo_core::ResolvedGenericConstraintBounds;
    use std::collections::HashMap;

    let scalar_params = parse_scalar_parameters_from_str(
        r#"{
      "scalar_parameters": [
        { "id": 240, "name": "k", "kind": "per_stage_block", "block_values": [[0, 0, 1.5], [0, 1, 2.5]] },
        { "id": 241, "name": "e", "kind": "expression", "expression": "@k * 2" }
      ]
    }"#,
    );
    let name_to_id: HashMap<String, EntityId> = scalar_params
        .iter()
        .map(|param| (param.name.clone(), param.id))
        .collect();
    let params = resolved_params_single_stage(2, &scalar_params);
    assert!(
        params.is_block_varying(EntityId(241)),
        "block-varying twin: an expression over a block-varying leaf must be block-varying"
    );
    assert_eq!(
        params.get(EntityId(241), 0, 0).to_bits(),
        3.0_f64.to_bits(),
        "block-varying twin: block 0 must resolve to 1.5 * 2"
    );
    assert_eq!(
        params.get(EntityId(241), 0, 1).to_bits(),
        5.0_f64.to_bits(),
        "block-varying twin: block 1 must resolve to 2.5 * 2"
    );

    let inline_constraints = parse_generic_from_str_with_params(
        r#"{
      "constraints": [
        { "id": 8, "name": "block_product", "expression": "@k * 2 * bus_excess(1) <= 400", "slack": { "enabled": false } }
      ]
    }"#,
        &name_to_id,
    );
    let expression_constraints = parse_generic_from_str_with_params(
        r#"{
      "constraints": [
        { "id": 8, "name": "block_product", "expression": "@e * bus_excess(1) <= 400", "slack": { "enabled": false } }
      ]
    }"#,
        &name_to_id,
    );
    let id_map: HashMap<i32, usize> = [(8_i32, 0)].into_iter().collect();
    let rows = vec![(8_i32, 0_i32, None::<i32>, None, None)];
    let inline_bounds = ResolvedGenericConstraintBounds::new(&id_map, rows.clone().into_iter());
    let expression_bounds = ResolvedGenericConstraintBounds::new(&id_map, rows.into_iter());

    assert_desugaring_twin_builds_identical_lp(
        2,
        inline_constraints,
        inline_bounds,
        &params,
        expression_constraints,
        expression_bounds,
        &params,
    );
}

/// A parameter that sums two constants used as a bound builds the same LP as the inline sum
/// on the right-hand side.
#[test]
fn expression_parameter_bound_matches_inline_bound_sum() {
    use novomodelo_core::ResolvedGenericConstraintBounds;
    use std::collections::HashMap;

    let scalar_params = parse_scalar_parameters_from_str(
        r#"{
      "scalar_parameters": [
        { "id": 250, "name": "a", "kind": "constant", "value": 0.1 },
        { "id": 251, "name": "b", "kind": "constant", "value": 0.2 },
        { "id": 252, "name": "e", "kind": "expression", "expression": "@a + @b" }
      ]
    }"#,
    );
    let name_to_id: HashMap<String, EntityId> = scalar_params
        .iter()
        .map(|param| (param.name.clone(), param.id))
        .collect();
    let params = resolved_params_single_stage(1, &scalar_params);

    let inline_constraints = parse_generic_from_str_with_params(
        r#"{
      "constraints": [
        { "id": 9, "name": "bound_sum", "expression": "bus_excess(1) <= @a + @b", "slack": { "enabled": false } }
      ]
    }"#,
        &name_to_id,
    );
    let expression_constraints = parse_generic_from_str_with_params(
        r#"{
      "constraints": [
        { "id": 9, "name": "bound_sum", "expression": "bus_excess(1) <= @e", "slack": { "enabled": false } }
      ]
    }"#,
        &name_to_id,
    );
    let id_map: HashMap<i32, usize> = [(9_i32, 0)].into_iter().collect();
    let rows = vec![(9_i32, 0_i32, None::<i32>, None, None)];
    let inline_bounds = ResolvedGenericConstraintBounds::new(&id_map, rows.clone().into_iter());
    let expression_bounds = ResolvedGenericConstraintBounds::new(&id_map, rows.into_iter());

    let generic_row = 1_usize;
    let expression_tpl = build_templates_with_params(
        &one_bus_system_n_blks_with_generic(
            1,
            expression_constraints.clone(),
            expression_bounds.clone(),
        ),
        &params,
    );
    assert_eq!(
        expression_tpl[0].row_upper[generic_row].to_bits(),
        (0.1_f64 + 0.2).to_bits(),
        "bound twin: the row's upper bound must be the sum in the inline association"
    );

    assert_desugaring_twin_builds_identical_lp(
        1,
        inline_constraints,
        inline_bounds,
        &params,
        expression_constraints,
        expression_bounds,
        &params,
    );
}
