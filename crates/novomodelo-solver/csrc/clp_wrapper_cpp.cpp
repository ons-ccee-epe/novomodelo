/* C++ shim for the CLP class-only knobs (dual-steepest-edge pricing,
 * factorization frequency).
 *
 * This file is compiled as C++17 and exposes two functions with C linkage. It
 * exists as a separate translation unit because the main wrapper
 * (clp_wrapper.c) is compiled as plain C and reaches CLP only through
 * <Clp_C_Interface.h>, which does NOT expose setFactorizationFrequency or the
 * dual-row pivot setter. Those live solely on the C++ ClpSimplex class.
 *
 * Handle layout (critical): the opaque handle threaded through the FFI is the
 * pointer cobre_clp_create() returns from Clp_newModel(). That is NOT a
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

extern "C" {

void cobre_clp_set_dual_row_steepest(void* model, int32_t mode) {
    // SAFETY: `model` is a `Clp_Simplex*` wrapper from cobre_clp_create()
    // (Clp_newModel()); the real C++ ClpSimplex is `wrapper->model_`. We cast
    // the void* to the concrete wrapper struct and read ->model_, matching how
    // every Clp_* C-API call reaches the model.
    ClpSimplex* simplex = static_cast<Clp_Simplex*>(model)->model_;
    ClpDualRowSteepest steepest(mode);
    simplex->setDualRowPivotAlgorithm(steepest);
}

void cobre_clp_set_factorization_frequency(void* model, int32_t value) {
    // SAFETY: `model` is a `Clp_Simplex*` wrapper from cobre_clp_create(); the
    // real ClpSimplex is `wrapper->model_`. See the file header invariant.
    ClpSimplex* simplex = static_cast<Clp_Simplex*>(model)->model_;
    simplex->setFactorizationFrequency(value);
}

} /* extern "C" */
