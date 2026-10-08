//! Coefficient-sign and `α_FPHA > 0` validation of the fitted plane set.

use super::error::FphaFittingError;
use super::hull_fit::RawPlane;

/// Validate a fitted plane set and its `α_FPHA` correction factor — the physical
/// sign contract `γ_v ≥ 0`, `γ_q ≥ 0`, `γ_s ≤ 0` (each within a `1e-10` rounding
/// tolerance), plus a non-empty set and `α_FPHA > 0`.
///
/// # Errors
///
/// | Condition | Error variant |
/// |-----------|---------------|
/// | `planes` is empty | [`FphaFittingError::NoHyperplanesProduced`] |
/// | `alpha <= 0` | [`FphaFittingError::NonPositiveAlpha`] |
/// | `gamma_v < -1e-10` for any plane | [`FphaFittingError::InvalidCoefficient`] |
/// | `gamma_q < -1e-10` for any plane | [`FphaFittingError::InvalidCoefficient`] |
/// | `gamma_s > 1e-10` for any plane | [`FphaFittingError::InvalidCoefficient`] |
pub(crate) fn validate_fitted_planes(
    planes: &[RawPlane],
    alpha: f64,
    hydro_name: &str,
) -> Result<(), FphaFittingError> {
    if planes.is_empty() {
        return Err(FphaFittingError::NoHyperplanesProduced {
            hydro_name: hydro_name.to_owned(),
        });
    }

    if alpha <= 0.0 {
        return Err(FphaFittingError::NonPositiveAlpha {
            hydro_name: hydro_name.to_owned(),
            alpha,
        });
    }

    for (idx, plane) in planes.iter().enumerate() {
        if plane.gamma_v < -1e-10 {
            return Err(FphaFittingError::InvalidCoefficient {
                hydro_name: hydro_name.to_owned(),
                plane_index: idx,
                detail: format!(
                    "gamma_v={:.6e} must be >= 0 (more storage should increase production)",
                    plane.gamma_v
                ),
            });
        }
        if plane.gamma_q < -1e-10 {
            return Err(FphaFittingError::InvalidCoefficient {
                hydro_name: hydro_name.to_owned(),
                plane_index: idx,
                detail: format!(
                    "gamma_q={:.6e} must be >= 0 (turbined flow should produce power)",
                    plane.gamma_q
                ),
            });
        }
        if plane.gamma_s > 1e-10 {
            return Err(FphaFittingError::InvalidCoefficient {
                hydro_name: hydro_name.to_owned(),
                plane_index: idx,
                detail: format!(
                    "gamma_s={:.6e} must be <= 0 (spillage should not increase production)",
                    plane.gamma_s
                ),
            });
        }
    }

    Ok(())
}
