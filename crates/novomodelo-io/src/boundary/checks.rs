//! Per-record checks shared by the current-format and legacy boundary parsers.

use std::collections::HashSet;
use std::path::Path;

use chrono::NaiveDate;

use crate::LoadError;
use crate::windowed_history::WindowedRecord;

/// Rejects two windows sharing an `(entity, start_date)`; a repeated start on
/// one entity is a duplicate, distinct starts are not.
///
/// # Errors
///
/// [`LoadError::SchemaError`] naming the first duplicate in input order.
pub(crate) fn reject_duplicate_window_starts(
    windows: &[WindowedRecord],
    surface: &str,
    entity_label: &str,
    path: &Path,
) -> Result<(), LoadError> {
    let mut seen: HashSet<(i32, NaiveDate)> = HashSet::new();
    for (i, window) in windows.iter().enumerate() {
        if !seen.insert((window.entity_id, window.start_date)) {
            return Err(LoadError::SchemaError {
                path: path.to_path_buf(),
                field: format!("{surface}[{i}].start_date"),
                message: format!(
                    "duplicate ({entity_label} {}, start_date {}) in {surface}",
                    window.entity_id, window.start_date
                ),
            });
        }
    }
    Ok(())
}

/// Rejects a negative window value; finiteness is enforced by
/// [`crate::windowed_history::validate_windowed_records`].
///
/// # Errors
///
/// [`LoadError::SchemaError`] naming the first negative value in input order.
pub(crate) fn reject_negative_window_values(
    windows: &[WindowedRecord],
    surface: &str,
    value_label: &str,
    path: &Path,
) -> Result<(), LoadError> {
    for (i, window) in windows.iter().enumerate() {
        if window.value < 0.0 {
            return Err(LoadError::SchemaError {
                path: path.to_path_buf(),
                field: format!("{surface}[{i}].{value_label}"),
                message: format!(
                    "{surface}[{i}].{value_label} must be >= 0.0, got {}",
                    window.value
                ),
            });
        }
    }
    Ok(())
}

/// Rejects a bound row whose values are not finite or whose `min_mw` exceeds
/// `max_mw`. `field_prefix` is the row's field path, `cell` names the row in
/// the `min_mw`/`max_mw` message.
///
/// # Errors
///
/// [`LoadError::SchemaError`] naming the first non-finite value, in
/// `cost_per_mwh`, `min_mw`, `max_mw` order, else the `min_mw`/`max_mw` order
/// violation.
pub(crate) fn check_bound_values(
    cost_per_mwh: f64,
    min_mw: f64,
    max_mw: f64,
    field_prefix: &str,
    cell: &str,
    path: &Path,
) -> Result<(), LoadError> {
    for (value, name) in [
        (cost_per_mwh, "cost_per_mwh"),
        (min_mw, "min_mw"),
        (max_mw, "max_mw"),
    ] {
        if !value.is_finite() {
            return Err(LoadError::SchemaError {
                path: path.to_path_buf(),
                field: format!("{field_prefix}.{name}"),
                message: format!("{name} must be a finite number, got {value}"),
            });
        }
    }
    if min_mw > max_mw {
        return Err(LoadError::SchemaError {
            path: path.to_path_buf(),
            field: format!("{field_prefix}.max_mw"),
            message: format!("max_mw ({max_mw}) must be >= min_mw ({min_mw}) for {cell}"),
        });
    }
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::{
        check_bound_values, reject_duplicate_window_starts, reject_negative_window_values,
    };
    use crate::LoadError;
    use crate::test_support::date;
    use crate::windowed_history::WindowedRecord;
    use std::path::Path;

    fn window(
        entity_id: i32,
        start: (i32, u32, u32),
        end: (i32, u32, u32),
        value: f64,
    ) -> WindowedRecord {
        WindowedRecord {
            entity_id,
            start_date: date(start.0, start.1, start.2),
            end_date: date(end.0, end.1, end.2),
            value,
        }
    }

    fn field_and_message(result: Result<(), LoadError>) -> (String, String) {
        match result.unwrap_err() {
            LoadError::SchemaError { field, message, .. } => (field, message),
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    #[test]
    fn test_duplicate_window_start_rejected_with_exact_field_and_message() {
        let windows = [
            window(7, (2026, 1, 1), (2026, 2, 1), 1.0),
            window(8, (2026, 1, 1), (2026, 2, 1), 1.0),
            window(7, (2026, 1, 1), (2026, 3, 1), 2.0),
        ];
        let (field, message) = field_and_message(reject_duplicate_window_starts(
            &windows,
            "surface",
            "thermal_id",
            Path::new("x"),
        ));
        assert_eq!(field, "surface[2].start_date");
        assert_eq!(
            message,
            "duplicate (thermal_id 7, start_date 2026-01-01) in surface"
        );
    }

    #[test]
    fn test_adjacent_windows_accepted_by_duplicate_start_check() {
        let windows = [
            window(7, (2026, 1, 1), (2026, 2, 1), 1.0),
            window(7, (2026, 2, 1), (2026, 3, 1), 2.0),
        ];
        reject_duplicate_window_starts(&windows, "surface", "thermal_id", Path::new("x")).unwrap();
    }

    #[test]
    fn test_negative_window_value_rejected_with_exact_field_and_message() {
        let windows = [
            window(1, (2026, 4, 1), (2026, 4, 4), 0.0),
            window(1, (2026, 4, 4), (2026, 4, 8), -2.5),
        ];
        let (field, message) = field_and_message(reject_negative_window_values(
            &windows,
            "surface",
            "value_m3s",
            Path::new("x"),
        ));
        assert_eq!(field, "surface[1].value_m3s");
        assert_eq!(message, "surface[1].value_m3s must be >= 0.0, got -2.5");
    }

    #[test]
    fn test_non_negative_window_values_accepted() {
        let windows = [
            window(1, (2026, 4, 1), (2026, 4, 4), 0.0),
            window(1, (2026, 4, 4), (2026, 4, 8), 3.0),
        ];
        reject_negative_window_values(&windows, "surface", "value_m3s", Path::new("x")).unwrap();
    }

    #[test]
    fn test_non_finite_bound_value_rejected_with_exact_field_and_message() {
        let (field, message) = field_and_message(check_bound_values(
            1.0,
            f64::NAN,
            5.0,
            "rows[3]",
            "cell",
            Path::new("x"),
        ));
        assert_eq!(field, "rows[3].min_mw");
        assert_eq!(message, "min_mw must be a finite number, got NaN");
    }

    #[test]
    fn test_min_above_max_rejected_with_exact_field_and_message() {
        let (field, message) = field_and_message(check_bound_values(
            1.0,
            10.0,
            5.0,
            "rows[3]",
            "thermal_id 1, post_study_stage_index 0",
            Path::new("x"),
        ));
        assert_eq!(field, "rows[3].max_mw");
        assert_eq!(
            message,
            "max_mw (5) must be >= min_mw (10) for thermal_id 1, post_study_stage_index 0"
        );
    }

    #[test]
    fn test_equal_min_and_max_accepted() {
        check_bound_values(1.0, 5.0, 5.0, "rows[0]", "cell", Path::new("x")).unwrap();
    }
}
