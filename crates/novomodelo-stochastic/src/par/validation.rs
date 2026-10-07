//! Validation of PAR model parameters.
//!
//! [`validate_par_parameters`] returns [`StochasticError::InvalidParParameters`]
//! when a model requires nonzero variance to normalize its AR coefficients.
//!
//! [`StochasticError::InvalidParParameters`]: crate::StochasticError::InvalidParParameters

use cobre_core::InflowModel;

use crate::StochasticError;

// ---------------------------------------------------------------------------
// validate_par_parameters
// ---------------------------------------------------------------------------

/// Validates that every model with `ar_order() > 0` has `std_m3s > 0` — zero
/// std cannot normalize the AR coefficients.
///
/// # Errors
///
/// Returns [`StochasticError::InvalidParParameters`] when an [`InflowModel`]
/// has `ar_order() > 0` but `std_m3s == 0.0`.
///
/// # Examples
///
/// ```
/// use cobre_core::{EntityId, scenario::InflowModel};
/// use cobre_stochastic::par::validation::validate_par_parameters;
///
/// let valid = InflowModel {
///     hydro_id: EntityId(1),
///     stage_id: 0,
///     mean_m3s: 150.0,
///     std_m3s: 30.0,
///     ar_coefficients: vec![0.3],
///     residual_std_ratio: 0.954,
///     annual: None,
/// };
/// assert!(validate_par_parameters(&[valid]).is_ok());
///
/// // Invalid: zero std with nonzero AR order.
/// let bad = InflowModel {
///     hydro_id: EntityId(2),
///     stage_id: 1,
///     mean_m3s: 100.0,
///     std_m3s: 0.0,
///     ar_coefficients: vec![0.3],
///     residual_std_ratio: 0.954,
///     annual: None,
/// };
/// let result = validate_par_parameters(&[bad]);
/// assert!(result.is_err());
/// ```
pub fn validate_par_parameters(inflow_models: &[InflowModel]) -> Result<(), StochasticError> {
    for model in inflow_models {
        if model.ar_order() > 0 && model.std_m3s == 0.0 {
            return Err(StochasticError::InvalidParParameters {
                hydro_id: model.hydro_id.0,
                stage_id: model.stage_id,
                reason: format!(
                    "zero standard deviation with ar_order={}: \
                     AR model requires nonzero variance to normalize coefficients",
                    model.ar_order()
                ),
            });
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use cobre_core::InflowModel;

    use super::validate_par_parameters;
    use crate::StochasticError;
    use crate::test_support::InflowModelSpec;

    fn make_model(
        hydro_id: i32,
        stage_id: i32,
        std_m3s: f64,
        ar_coefficients: Vec<f64>,
        residual_std_ratio: f64,
    ) -> InflowModel {
        crate::test_support::make_inflow_model(InflowModelSpec {
            hydro_id,
            stage_id,
            std_m3s,
            ar_coefficients,
            residual_std_ratio,
            ..Default::default()
        })
    }

    fn make_model_with_annual(
        hydro_id: i32,
        stage_id: i32,
        std_m3s: f64,
        ar_coefficients: Vec<f64>,
        residual_std_ratio: f64,
    ) -> InflowModel {
        use cobre_core::scenario::AnnualComponent;
        crate::test_support::make_inflow_model(InflowModelSpec {
            hydro_id,
            stage_id,
            std_m3s,
            ar_coefficients,
            residual_std_ratio,
            annual: Some(AnnualComponent {
                coefficient: 0.15,
                mean_m3s: 90.0,
                std_m3s: 12.0,
            }),
            ..Default::default()
        })
    }

    #[test]
    fn empty_input_is_valid() {
        assert!(validate_par_parameters(&[]).is_ok());
    }

    #[test]
    fn ar_order_zero_with_positive_std_is_valid() {
        let model = make_model(1, 5, 30.0, vec![], 1.0);
        assert!(validate_par_parameters(&[model]).is_ok());
    }

    #[test]
    fn ar_order_zero_with_zero_std_is_valid() {
        let model = make_model(3, 0, 0.0, vec![], 1.0);
        let result = validate_par_parameters(&[model]);
        assert!(result.is_ok());
    }

    #[test]
    fn zero_std_with_nonzero_ar_order_returns_error() {
        let model = make_model(1, 1, 0.0, vec![0.3], 0.954);
        let result = validate_par_parameters(&[model]);

        assert!(result.is_err());
        match result.unwrap_err() {
            StochasticError::InvalidParParameters {
                hydro_id,
                stage_id,
                reason,
            } => {
                assert_eq!(hydro_id, 1);
                assert_eq!(stage_id, 1);
                assert!(reason.contains("zero standard deviation"));
            }
            other => panic!("expected InvalidParParameters, got {other:?}"),
        }
    }

    #[test]
    fn ar_order_positive_with_positive_std_is_valid() {
        let model = make_model(1, 0, 30.0, vec![0.3], 0.954);
        assert!(validate_par_parameters(&[model]).is_ok());
    }

    #[test]
    fn first_fatal_error_stops_iteration() {
        let bad = make_model(1, 0, 0.0, vec![0.3], 0.954);
        let valid_model = make_model(2, 1, 30.0, vec![0.3], 0.954);
        let result = validate_par_parameters(&[bad, valid_model]);

        assert!(result.is_err());
        match result.unwrap_err() {
            StochasticError::InvalidParParameters { hydro_id, .. } => {
                assert_eq!(hydro_id, 1);
            }
            other => panic!("expected InvalidParParameters, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Tests for models with annual: Some(_) (PAR(p)-A extension)
    // -----------------------------------------------------------------------

    #[test]
    fn validate_with_annual_some_is_valid() {
        let model = make_model_with_annual(1, 0, 30.0, vec![0.3], 0.954);
        assert!(validate_par_parameters(&[model]).is_ok());
    }

    #[test]
    fn validate_with_annual_some_zero_std_errors() {
        let model = make_model_with_annual(5, 7, 0.0, vec![0.3], 0.954);
        let result = validate_par_parameters(&[model]);
        assert!(result.is_err());
        match result.unwrap_err() {
            StochasticError::InvalidParParameters {
                hydro_id,
                stage_id,
                reason,
            } => {
                assert_eq!(hydro_id, 5);
                assert_eq!(stage_id, 7);
                assert!(
                    reason.contains("zero standard deviation"),
                    "reason must mention zero standard deviation, got: {reason}"
                );
            }
            other => panic!("expected InvalidParParameters, got {other:?}"),
        }
    }
}
