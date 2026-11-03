/* C++ shims for CLP operations that only the C++ ClpSimplex class exposes:
 * dual-steepest-edge pricing, factorization frequency, and the row excess of a
 * primal solution against the model's rows.
 *
 * This file is compiled as C++17 and exposes functions with C linkage. It
 * exists as a separate translation unit because the main wrapper
 * (clp_wrapper.c) is compiled as plain C and reaches CLP only through
 * <Clp_C_Interface.h>, which does NOT expose setFactorizationFrequency or the
 * dual-row pivot setter. Those live solely on the C++ ClpSimplex class.
 *
 * Handle layout (critical): the opaque handle threaded through the FFI is the
 * pointer novomodelo_clp_create() returns from Clp_newModel(). That is NOT a
 * ClpSimplex* — it is a `Clp_Simplex*` WRAPPER struct. With CLP_EXTERN_C
 * defined, Coin_C_defines.h declares Clp_Simplex as a concrete
 * `{ ClpSimplex* model_; CMessageHandler* handler_; }`, and Clp_newModel()
 * does `model->model_ = new ClpSimplex()` (Clp_C_Interface.cpp). So the real
 * C++ ClpSimplex object lives at `wrapper->model_`. Every Clp_* C-API call in
 * clp_wrapper.c reaches it the same way (the C API dereferences ->model_
 * internally). This shim MUST do the same: cast the void* to Clp_Simplex* and
 * take ->model_. Casting the wrapper directly to ClpSimplex* (an earlier bug)
 * reinterprets the wrapper's two leading pointers (model_/handler_) as
 * ClpSimplex members and dispatches every method on garbage.
 *
 * We define CLP_EXTERN_C before including the CLP headers so the Clp_Simplex
 * struct is the concrete two-pointer form (not `typedef void`); that is what
 * makes ->model_ accessible. CLP_EXTERN_C only affects that typedef and gives
 * the C API declarations C linkage; this shim calls no Clp_* C function, only
 * ClpSimplex class methods, so it has no other effect here.
 */

#include "clp_wrapper.h"

// With CLP_EXTERN_C defined, <Coin_C_defines.h> (pulled in by
// <Clp_C_Interface.h>) declares Clp_Simplex as a concrete struct whose first
// member is `ClpSimplex *model_;`. That struct references the C++ classes
// ClpSimplex and ClpSolve by name, so their class headers MUST be included
// FIRST — otherwise Coin_C_defines.h sees undeclared types and the struct
// silently degrades. Order is therefore: define the macro, include the C++
// class headers, then the C interface header.
#define CLP_EXTERN_C
#include <ClpDualRowSteepest.hpp>
#include <ClpSimplex.hpp>
#include <ClpSolve.hpp>

#include <Clp_C_Interface.h>

#include <algorithm>
#include <cmath>
#include <cstddef>

extern "C" {

void novomodelo_clp_set_dual_row_steepest(void* model, int32_t mode) {
    // SAFETY: `model` is a `Clp_Simplex*` wrapper from novomodelo_clp_create()
    // (Clp_newModel()); the real C++ ClpSimplex is `wrapper->model_`. We cast
    // the void* to the concrete wrapper struct and read ->model_, matching how
    // every Clp_* C-API call reaches the model.
    ClpSimplex* simplex = static_cast<Clp_Simplex*>(model)->model_;
    ClpDualRowSteepest steepest(mode);
    simplex->setDualRowPivotAlgorithm(steepest);
}

void novomodelo_clp_set_factorization_frequency(void* model, int32_t value) {
    // SAFETY: `model` is a `Clp_Simplex*` wrapper from novomodelo_clp_create(); the
    // real ClpSimplex is `wrapper->model_`. See the file header invariant.
    ClpSimplex* simplex = static_cast<Clp_Simplex*>(model)->model_;
    simplex->setFactorizationFrequency(value);
}

double novomodelo_clp_max_row_violation(
    void*         model,
    const double* col_value,
    double*       activity,
    double*       magnitude,
    int32_t       num_row,
    double        primal_tolerance
) {
    // SAFETY: `model` is a `Clp_Simplex*` wrapper from novomodelo_clp_create(); the
    // real ClpSimplex is `wrapper->model_`. See the file header invariant.
    const ClpSimplex* simplex = static_cast<Clp_Simplex*>(model)->model_;
    if (num_row <= 0 || num_row != simplex->numberRows()) {
        return 0.0;
    }
    const CoinPackedMatrix* a = simplex->matrix();
    if (a == nullptr) {
        return 0.0;
    }

    const std::size_t m = static_cast<std::size_t>(num_row);
    std::fill(activity, activity + m, 0.0);
    std::fill(magnitude, magnitude + m, 0.0);

    const CoinBigIndex* start = a->getVectorStarts();
    const int* length = a->getVectorLengths();
    const int* index = a->getIndices();
    const double* value = a->getElements();
    const int major_dim = a->getMajorDim();
    if (a->isColOrdered()) {
        for (int j = 0; j < major_dim; ++j) {
            const double x = col_value[j];
            for (CoinBigIndex k = start[j]; k < start[j] + length[j]; ++k) {
                const double term = value[k] * x;
                const std::size_t i = static_cast<std::size_t>(index[k]);
                activity[i] += term;
                magnitude[i] += std::fabs(term);
            }
        }
    } else {
        for (int i = 0; i < major_dim; ++i) {
            double act = 0.0;
            double mag = 0.0;
            for (CoinBigIndex k = start[i]; k < start[i] + length[i]; ++k) {
                const double term = value[k] * col_value[index[k]];
                act += term;
                mag += std::fabs(term);
            }
            activity[i] = act;
            magnitude[i] = mag;
        }
    }

    const double* row_lower = simplex->rowLower();
    const double* row_upper = simplex->rowUpper();
    double rho = 0.0;
    for (std::size_t i = 0; i < m; ++i) {
        const double v = std::max(row_lower[i] - activity[i], activity[i] - row_upper[i]);
        if (v > primal_tolerance) {
            rho = std::max(rho, (v - primal_tolerance) / std::max(1.0, magnitude[i]));
        }
    }
    return rho;
}

} /* extern "C" */
