//! CLP dual-sign convention probe — sign of `cobre_clp_get_row_price` for an
//! equality row.
//!
//! The canonical dual-sign convention is the `HiGHS` one: row duals feed
//! downstream cut construction without negation. CLP reports
//! `row_price[0] == -100.0` for the equality-row fixture, the SAME sign as
//! `HiGHS`; therefore `ClpSolver::solve` MUST not negate
//! `cobre_clp_get_row_price`.

#![cfg(all(feature = "clp", feature = "test-support"))]
#![cfg_attr(
    test,
    allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::float_cmp,
        clippy::print_stdout
    )
)]

use cobre_solver::test_support::{
    CLP_STATUS_OPTIMAL, cobre_clp_create, cobre_clp_destroy, cobre_clp_dual,
    cobre_clp_get_row_price, cobre_clp_load_problem, cobre_clp_objective_value,
    cobre_clp_set_log_level, cobre_clp_status,
};

#[test]
fn clp_sign_convention_row_equality() {
    // SAFETY: `cobre_clp_create` takes no arguments and returns a freshly
    // allocated, caller-owned CLP model pointer (or null on allocation
    // failure). We check for null immediately below.
    let model = unsafe { cobre_clp_create() };
    assert!(!model.is_null(), "cobre_clp_create() returned null");

    // SAFETY: `model` is the non-null handle just created above.
    unsafe { cobre_clp_set_log_level(model, 0) };

    let col_starts: [i32; 4] = [0, 2, 2, 3];
    let row_indices: [i32; 3] = [0, 1, 1];
    let values: [f64; 3] = [1.0, 2.0, 1.0];
    let col_lower: [f64; 3] = [0.0, 0.0, 0.0];
    // Forward f64::INFINITY directly — the C wrapper owns the inf → DBL_MAX
    // translation; do NOT pre-translate.
    let col_upper: [f64; 3] = [10.0, f64::INFINITY, 8.0];
    let objective: [f64; 3] = [0.0, 1.0, 50.0];
    let row_lower: [f64; 2] = [6.0, 14.0];
    let row_upper: [f64; 2] = [6.0, 14.0];

    // SAFETY: `model` is the non-null pointer from `cobre_clp_create`. All array
    // pointers are valid for the dimensions passed: `col_starts` has
    // `num_cols + 1 = 4` entries; `col_lower`/`col_upper`/`objective` have
    // `num_cols = 3` entries; `row_indices`/`values` have
    // `col_starts[3] = 3` nonzeros; `row_lower`/`row_upper` have
    // `num_rows = 2` entries. All pointers outlive the call.
    unsafe {
        cobre_clp_load_problem(
            model,
            3, // num_cols
            2, // num_rows
            col_starts.as_ptr(),
            row_indices.as_ptr(),
            values.as_ptr(),
            col_lower.as_ptr(),
            col_upper.as_ptr(),
            objective.as_ptr(),
            row_lower.as_ptr(),
            row_upper.as_ptr(),
        );
    }

    // SAFETY: `model` is a valid CLP model with a loaded problem.
    // `if_values_pass = 0` requests no values pass. The return value is the
    // CLP solve status (0 = optimal).
    let solve_status = unsafe { cobre_clp_dual(model, 0) };
    assert_eq!(
        solve_status, CLP_STATUS_OPTIMAL,
        "cobre_clp_dual() returned solve status {solve_status}"
    );

    // SAFETY: `model` is a valid CLP model that has just been solved.
    let status = unsafe { cobre_clp_status(model) };
    assert_eq!(
        status, CLP_STATUS_OPTIMAL,
        "expected Optimal status, got {status}"
    );

    // SAFETY: `model` is a valid solved CLP model.
    let objective_value = unsafe { cobre_clp_objective_value(model) };

    // SAFETY: `model` is a valid solved CLP model. `cobre_clp_get_row_price`
    // returns a pointer into CLP-owned memory of length `num_rows = 2`, valid
    // until the next solve. We copy the 2 elements into an owned Vec and never
    // dereference the pointer afterwards.
    let row_price: Vec<f64> = unsafe {
        let ptr = cobre_clp_get_row_price(model);
        assert!(!ptr.is_null(), "cobre_clp_get_row_price() returned null");
        std::slice::from_raw_parts(ptr, 2).to_vec()
    };

    println!("\n=== CLP dual-sign probe (equality-row fixture) ===");
    println!("  objective       = {objective_value}");
    println!("  row_price        = {row_price:?}");
    println!(
        "  row_price[0] sign = {}",
        if row_price[0] > 0.0 { "+" } else { "-" }
    );
    println!("  HiGHS canonical  : dual[0] == -100.0, objective == 100.0");

    assert!(
        (objective_value - 100.0).abs() < 1e-6,
        "objective {objective_value} is not within 1e-6 of HiGHS baseline 100.0"
    );

    assert!(
        (row_price[0].abs() - 100.0).abs() < 1e-6,
        "|row_price[0]| = {} is not within 1e-6 of 100.0",
        row_price[0].abs()
    );

    assert!(
        (row_price[0] - (-100.0)).abs() < 1e-6,
        "row_price[0] = {} is not within 1e-6 of the observed -100.0",
        row_price[0]
    );

    // SAFETY: `model` is a valid CLP model created by `cobre_clp_create` and not
    // yet destroyed; all CLP-owned values have been read and copied. This
    // transfers ownership back to CLP for deallocation. `model` is not used
    // after this call.
    unsafe { cobre_clp_destroy(model) };
}
