/* C++ shims for HiGHS operations the C API cannot express.
 *
 * This file is compiled as C++17 and exposes functions with C linkage. It
 * exists as a separate translation unit because the main wrapper
 * (highs_wrapper.c) is compiled as plain C and cannot use C++ types such as
 * HighsBasis or HighsLp directly.
 *
 * novomodelo_highs_set_basis_non_alien constructs a HighsBasis with alien = false before calling
 * Highs::setBasis(const HighsBasis&).  This bypasses the throwaway LU
 * factorisation that the HiGHS C API (Highs_setBasis) triggers when alien is
 * true.  Callers must guarantee that the total number of kBasic entries equals
 * num_row; the function returns kHighsStatusError if isBasisConsistent fails
 * inside HiGHS.
 */

#include "highs_wrapper.h"
#include <Highs.h>

#include <algorithm>
#include <cmath>

extern "C" {

int32_t novomodelo_highs_set_basis_non_alien(
    void*           highs,
    const int32_t*  col_status,
    const int32_t*  row_status
) {
    Highs* h = reinterpret_cast<Highs*>(highs);

    const HighsInt num_col = h->getNumCol();
    const HighsInt num_row = h->getNumRow();

    HighsBasis basis;
    basis.alien = false;
    basis.valid = true;
    basis.col_status.resize(static_cast<std::size_t>(num_col));
    basis.row_status.resize(static_cast<std::size_t>(num_row));

    for (HighsInt i = 0; i < num_col; ++i) {
        basis.col_status[static_cast<std::size_t>(i)] =
            static_cast<HighsBasisStatus>(col_status[i]);
    }

    for (HighsInt i = 0; i < num_row; ++i) {
        basis.row_status[static_cast<std::size_t>(i)] =
            static_cast<HighsBasisStatus>(row_status[i]);
    }

    return static_cast<int32_t>(h->setBasis(basis));
}

double novomodelo_highs_max_row_violation(
    void*         highs,
    const double* col_value,
    double*       activity,
    double*       magnitude,
    int32_t       num_row,
    double        primal_tolerance
) {
    const Highs* h = reinterpret_cast<const Highs*>(highs);
    const HighsLp& lp = h->getLp();
    if (num_row <= 0 || static_cast<HighsInt>(num_row) != lp.num_row_) {
        return 0.0;
    }

    const std::size_t m = static_cast<std::size_t>(num_row);
    std::fill(activity, activity + m, 0.0);
    std::fill(magnitude, magnitude + m, 0.0);

    const HighsSparseMatrix& a = lp.a_matrix_;
    if (a.isColwise()) {
        for (HighsInt j = 0; j < lp.num_col_; ++j) {
            const double x = col_value[j];
            for (HighsInt k = a.start_[j]; k < a.start_[j + 1]; ++k) {
                const double term = a.value_[k] * x;
                const std::size_t i = static_cast<std::size_t>(a.index_[k]);
                activity[i] += term;
                magnitude[i] += std::fabs(term);
            }
        }
    } else {
        for (HighsInt i = 0; i < lp.num_row_; ++i) {
            double act = 0.0;
            double mag = 0.0;
            for (HighsInt k = a.start_[i]; k < a.start_[i + 1]; ++k) {
                const double term = a.value_[k] * col_value[a.index_[k]];
                act += term;
                mag += std::fabs(term);
            }
            activity[i] = act;
            magnitude[i] = mag;
        }
    }

    double rho = 0.0;
    for (std::size_t i = 0; i < m; ++i) {
        const double v = std::max(
            lp.row_lower_[i] - activity[i], activity[i] - lp.row_upper_[i]);
        if (v > primal_tolerance) {
            rho = std::max(
                rho, (v - primal_tolerance) / std::max(1.0, magnitude[i]));
        }
    }
    return rho;
}

double novomodelo_highs_primal_tolerance_in_effect(const void* highs) {
    /* HiGHS accepts a solution in user-bound-scaled units and multiplies the
     * returned column values by 2^(-user_bound_scale) when unscaling, so its
     * allowance in returned units is the tolerance scaled the same way. */
    const HighsOptions& options =
        reinterpret_cast<const Highs*>(highs)->getOptions();
    return std::ldexp(
        options.primal_feasibility_tolerance,
        static_cast<int>(-options.user_bound_scale));
}

} /* extern "C" */
