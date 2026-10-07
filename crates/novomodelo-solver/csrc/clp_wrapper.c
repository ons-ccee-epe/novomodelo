/* Thin C wrapper around the CLP (COIN-OR Linear Programming) C API for use by
 * cobre-solver FFI bindings.
 *
 * Each function is a direct call-through to the corresponding Clp_* function
 * with no additional logic, except cobre_clp_load_problem which translates
 * IEEE ±infinity bounds to the CLP infinity sentinel and fixes the objective
 * sense to minimize.
 *
 * The CLP C API models the opaque handle as `Clp_Simplex`, which is
 * `typedef void Clp_Simplex;` in the default (non-CLP_EXTERN_C) build, so the
 * `void*` handle used across the FFI boundary is ABI-compatible. The CLP
 * getters take a non-const `Clp_Simplex*`, so the read-only `const void*`
 * handles are cast to `void*` before forwarding; the underlying calls do not
 * mutate the model.
 */

#include "clp_wrapper.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>

#include <Clp_C_Interface.h>

/* Compile-time guard: cobre-solver assumes CoinBigIndex is 32-bit so that the
 * i32 col_starts array on the Rust side is ABI-compatible with the
 * `const CoinBigIndex*` parameter of Clp_loadProblem. CoinBigIndex is `int`
 * when the superbuild is configured with COIN_BIG_INDEX == 0 (the default);
 * if it were widened to long/long long, the CSC offsets would be
 * misinterpreted. Fail the build early instead.
 * MSVC does not support C11 _Static_assert in C mode; use a negative-sized
 * array trick that works on all compilers. */
typedef char cobre_clp_bigindex_size_check_[(sizeof(CoinBigIndex) == sizeof(int32_t)) ? 1 : -1];

/* CLP uses COIN_DBL_MAX as its infinity sentinel, not IEEE infinity. Passing a
 * raw IEEE ±infinity to Clp_loadProblem corrupts the model. COIN_DBL_MAX is
 * defined in CoinFinite.hpp as std::numeric_limits<double>::max(), which is
 * exactly C's DBL_MAX (<float.h>). CoinFinite.hpp is a C++ header and must not
 * be included from this C translation unit, so we use DBL_MAX directly — it is
 * value-identical to COIN_DBL_MAX. */
#define COBRE_CLP_INFINITY DBL_MAX

/* Translate a bound array of length `n` from IEEE infinity to the CLP
 * sentinel: +inf -> +DBL_MAX (== +COIN_DBL_MAX), -inf -> -DBL_MAX, finite
 * values pass through unchanged. `out` must point to `n` writable doubles. */
static void cobre_clp_translate_bounds(const double* in, double* out, int32_t n) {
    for (int32_t i = 0; i < n; ++i) {
        const double v = in[i];
        if (isinf(v)) {
            out[i] = (v > 0.0) ? COBRE_CLP_INFINITY : -COBRE_CLP_INFINITY;
        } else {
            out[i] = v;
        }
    }
}

/* =========================================================================
 * Lifecycle
 * ========================================================================= */

void* cobre_clp_create(void) {
    return Clp_newModel();
}

void cobre_clp_destroy(void* model) {
    Clp_deleteModel((Clp_Simplex*)model);
}

void cobre_clp_set_log_level(void* model, int32_t value) {
    Clp_setLogLevel((Clp_Simplex*)model, (int)value);
}

/* =========================================================================
 * Model Loading
 * ========================================================================= */

void cobre_clp_load_problem(
    void*           model,
    int32_t         num_cols,
    int32_t         num_rows,
    const int32_t*  col_starts,
    const int32_t*  row_indices,
    const double*   values,
    const double*   col_lower,
    const double*   col_upper,
    const double*   objective,
    const double*   row_lower,
    const double*   row_upper
) {
    /* Bound arrays may contain IEEE ±infinity. Translate into malloc'd scratch
     * so the caller's arrays are left untouched, then free after the load.
     * (Objective and matrix values never carry infinity sentinels and are
     * forwarded directly.)
     *
     * Allocate only when the dimension is positive. malloc(0) is allowed to
     * return NULL on a conforming C runtime, which would otherwise trip the
     * allocation-failure path for a legitimate zero-row (or zero-column)
     * problem. When a dimension is zero the scratch pointer stays NULL and that
     * NULL is forwarded to Clp_loadProblem, which accepts NULL bound arrays for
     * an empty dimension. */
    double* col_lower_t = num_cols > 0
        ? (double*)malloc((size_t)num_cols * sizeof(double)) : NULL;
    double* col_upper_t = num_cols > 0
        ? (double*)malloc((size_t)num_cols * sizeof(double)) : NULL;
    double* row_lower_t = num_rows > 0
        ? (double*)malloc((size_t)num_rows * sizeof(double)) : NULL;
    double* row_upper_t = num_rows > 0
        ? (double*)malloc((size_t)num_rows * sizeof(double)) : NULL;

    if ((num_cols > 0 && (col_lower_t == NULL || col_upper_t == NULL))
        || (num_rows > 0 && (row_lower_t == NULL || row_upper_t == NULL))) {
        /* Genuine allocation failure: free whatever succeeded and bail without
         * touching the model. The Rust layer treats an unsolved model
         * (status != optimal) as an error on the subsequent solve. */
        free(col_lower_t);
        free(col_upper_t);
        free(row_lower_t);
        free(row_upper_t);
        return;
    }

    /* translate_bounds is a no-op when n == 0, leaving the NULL scratch alone. */
    cobre_clp_translate_bounds(col_lower, col_lower_t, num_cols);
    cobre_clp_translate_bounds(col_upper, col_upper_t, num_cols);
    cobre_clp_translate_bounds(row_lower, row_lower_t, num_rows);
    cobre_clp_translate_bounds(row_upper, row_upper_t, num_rows);

    Clp_loadProblem(
        (Clp_Simplex*)model,
        (int)num_cols,
        (int)num_rows,
        (const CoinBigIndex*)col_starts,
        (const int*)row_indices,
        values,
        col_lower_t,
        col_upper_t,
        objective,
        row_lower_t,
        row_upper_t
    );

    /* Fix the objective sense to minimize (1 = minimize, -1 = maximize). */
    Clp_setOptimizationDirection((Clp_Simplex*)model, 1.0);

    free(col_lower_t);
    free(col_upper_t);
    free(row_lower_t);
    free(row_upper_t);
}

/* =========================================================================
 * Incremental Mutation
 * ========================================================================= */

void cobre_clp_add_rows(
    void*           model,
    int32_t         number,
    const double*   row_lower,
    const double*   row_upper,
    const int32_t*  row_starts,
    const int32_t*  columns,
    const double*   elements
) {
    /* Nothing to append: leave the model untouched (matches the Rust-side
     * empty-batch guard, where number == 0 returns before reaching the FFI). */
    if (number <= 0) {
        return;
    }

    /* Translate the appended row bounds into malloc'd scratch so the caller's
     * arrays are left untouched, then free after the append. The matrix data
     * (row_starts / columns / elements) never carries infinity sentinels and is
     * forwarded directly. */
    double* row_lower_t = (double*)malloc((size_t)number * sizeof(double));
    double* row_upper_t = (double*)malloc((size_t)number * sizeof(double));
    if (row_lower_t == NULL || row_upper_t == NULL) {
        /* Genuine allocation failure: free whatever succeeded and bail without
         * touching the model. The Rust layer treats an unsolved model
         * (status != optimal) as an error on the subsequent solve. */
        free(row_lower_t);
        free(row_upper_t);
        return;
    }

    cobre_clp_translate_bounds(row_lower, row_lower_t, number);
    cobre_clp_translate_bounds(row_upper, row_upper_t, number);

    Clp_addRows(
        (Clp_Simplex*)model,
        (int)number,
        row_lower_t,
        row_upper_t,
        (const CoinBigIndex*)row_starts,
        (const int*)columns,
        elements
    );

    free(row_lower_t);
    free(row_upper_t);
}

/* Shared helper for the four full-length bound-change wrappers: translate the
 * caller's `n`-length bound array into malloc'd scratch (so IEEE infinities
 * become ±COIN_DBL_MAX) and hand the scratch to `chg`. `n` is the model's
 * current row or column count, queried by the caller. A non-positive `n` (no
 * rows/columns) is a no-op; an allocation failure bails without touching the
 * model, matching cobre_clp_load_problem's failure handling. */
static void cobre_clp_chg_bounds(
    void* model,
    const double* bounds,
    int n,
    void (*chg)(Clp_Simplex*, const double*)
) {
    if (n <= 0) {
        return;
    }
    double* scratch = (double*)malloc((size_t)n * sizeof(double));
    if (scratch == NULL) {
        return;
    }
    cobre_clp_translate_bounds(bounds, scratch, (int32_t)n);
    chg((Clp_Simplex*)model, scratch);
    free(scratch);
}

void cobre_clp_chg_row_lower(void* model, const double* row_lower) {
    cobre_clp_chg_bounds(
        model, row_lower, Clp_getNumRows((Clp_Simplex*)model), Clp_chgRowLower);
}

void cobre_clp_chg_row_upper(void* model, const double* row_upper) {
    cobre_clp_chg_bounds(
        model, row_upper, Clp_getNumRows((Clp_Simplex*)model), Clp_chgRowUpper);
}

void cobre_clp_chg_column_lower(void* model, const double* column_lower) {
    cobre_clp_chg_bounds(
        model, column_lower, Clp_getNumCols((Clp_Simplex*)model), Clp_chgColumnLower);
}

void cobre_clp_chg_column_upper(void* model, const double* column_upper) {
    cobre_clp_chg_bounds(
        model, column_upper, Clp_getNumCols((Clp_Simplex*)model), Clp_chgColumnUpper);
}

/* =========================================================================
 * Solving
 * ========================================================================= */

int32_t cobre_clp_dual(void* model, int32_t if_values_pass) {
    return (int32_t)Clp_dual((Clp_Simplex*)model, (int)if_values_pass);
}

int32_t cobre_clp_primal(void* model, int32_t if_values_pass) {
    return (int32_t)Clp_primal((Clp_Simplex*)model, (int)if_values_pass);
}

/* =========================================================================
 * Solution Extraction
 * ========================================================================= */

double cobre_clp_objective_value(const void* model) {
    return Clp_objectiveValue((Clp_Simplex*)model);
}

int32_t cobre_clp_status(const void* model) {
    return (int32_t)Clp_status((Clp_Simplex*)model);
}

int32_t cobre_clp_number_iterations(const void* model) {
    return (int32_t)Clp_numberIterations((Clp_Simplex*)model);
}

const double* cobre_clp_get_col_solution(const void* model) {
    return Clp_getColSolution((Clp_Simplex*)model);
}

const double* cobre_clp_get_row_price(const void* model) {
    return Clp_getRowPrice((Clp_Simplex*)model);
}

const double* cobre_clp_get_reduced_cost(const void* model) {
    return Clp_getReducedCost((Clp_Simplex*)model);
}

/* =========================================================================
 * Configuration
 * ========================================================================= */

void cobre_clp_set_perturbation(void* model, int32_t value) {
    Clp_setPerturbation((Clp_Simplex*)model, (int)value);
}

void cobre_clp_scaling(void* model, int32_t mode) {
    Clp_scaling((Clp_Simplex*)model, (int)mode);
}

void cobre_clp_set_primal_tolerance(void* model, double value) {
    Clp_setPrimalTolerance((Clp_Simplex*)model, value);
}

void cobre_clp_set_dual_tolerance(void* model, double value) {
    Clp_setDualTolerance((Clp_Simplex*)model, value);
}

void cobre_clp_set_maximum_iterations(void* model, int32_t value) {
    Clp_setMaximumIterations((Clp_Simplex*)model, (int)value);
}

/* =========================================================================
 * Basis Management
 * ========================================================================= */

int32_t cobre_clp_get_column_status(const void* model, int32_t sequence) {
    return (int32_t)Clp_getColumnStatus((Clp_Simplex*)model, (int)sequence);
}

void cobre_clp_set_column_status(void* model, int32_t sequence, int32_t value) {
    Clp_setColumnStatus((Clp_Simplex*)model, (int)sequence, (int)value);
}

int32_t cobre_clp_get_row_status(const void* model, int32_t sequence) {
    return (int32_t)Clp_getRowStatus((Clp_Simplex*)model, (int)sequence);
}

void cobre_clp_set_row_status(void* model, int32_t sequence, int32_t value) {
    Clp_setRowStatus((Clp_Simplex*)model, (int)sequence, (int)value);
}

/* =========================================================================
 * Version query (no instance required)
 * ========================================================================= */

int32_t cobre_clp_version_major(void) {
    return (int32_t)Clp_VersionMajor();
}

int32_t cobre_clp_version_minor(void) {
    return (int32_t)Clp_VersionMinor();
}

int32_t cobre_clp_version_release(void) {
    return (int32_t)Clp_VersionRelease();
}
