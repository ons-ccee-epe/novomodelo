//! Assemble the run-level fixed-delivery echo from the study's declared
//! post-horizon (class-4) anticipated commitment windows.
//!
//! One row per declared window, verbatim, in the accessor's canonical order —
//! the same chokepoint both the CLI and Python write paths assemble through, so
//! record-content parity is structural rather than duplicated.

use cobre_core::{AnticipatedCommitmentHistory, System};
use cobre_io::FixedDeliveryRow;

use crate::StudySetup;

/// Build the run-level fixed-delivery rows for `setup`/`system`, in the
/// canonical order [`StudySetup::build_terminal_fixed_post_horizon_windows`]
/// returns.
///
/// `system` is passed explicitly because [`StudySetup`] does not own it.
#[must_use]
pub fn build_fixed_delivery_rows(setup: &StudySetup, system: &System) -> Vec<FixedDeliveryRow> {
    fixed_delivery_rows_from_windows(&setup.build_terminal_fixed_post_horizon_windows(system))
}

/// Map each window verbatim to a [`FixedDeliveryRow`], preserving input order —
/// decoupled from [`StudySetup`] so the assembly is unit-testable without a full
/// setup.
fn fixed_delivery_rows_from_windows(
    windows: &[AnticipatedCommitmentHistory],
) -> Vec<FixedDeliveryRow> {
    windows
        .iter()
        .map(|w| FixedDeliveryRow {
            thermal_id: w.thermal_id.0,
            start_date: w.start_date,
            end_date: w.end_date,
            value_mw: w.value_mw,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::NaiveDate;
    use cobre_core::EntityId;

    fn window(
        thermal_id: i32,
        start_date: NaiveDate,
        end_date: NaiveDate,
        value_mw: f64,
    ) -> AnticipatedCommitmentHistory {
        AnticipatedCommitmentHistory {
            thermal_id: EntityId(thermal_id),
            start_date,
            end_date,
            value_mw,
        }
    }

    fn date(y: i32, m: u32, d: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(y, m, d).expect("valid date")
    }

    #[test]
    fn fixed_delivery_maps_multi_plant_windows_preserving_order() {
        let windows = vec![
            window(3, date(2030, 1, 1), date(2030, 1, 31), 120.5),
            window(3, date(2030, 2, 1), date(2030, 2, 28), 110.0),
            window(7, date(2030, 1, 1), date(2030, 1, 31), 88.0),
        ];

        let rows = fixed_delivery_rows_from_windows(&windows);

        assert_eq!(rows.len(), windows.len());
        for (row, w) in rows.iter().zip(windows.iter()) {
            assert_eq!(row.thermal_id, w.thermal_id.0);
            assert_eq!(row.start_date, w.start_date);
            assert_eq!(row.end_date, w.end_date);
            assert_eq!(row.value_mw, w.value_mw);
        }
    }

    #[test]
    fn fixed_delivery_echoes_a_zero_value_commissioning_inactive_window() {
        let windows = vec![window(5, date(2031, 1, 1), date(2031, 12, 31), 0.0)];

        let rows = fixed_delivery_rows_from_windows(&windows);

        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].value_mw, 0.0);
    }

    #[test]
    fn fixed_delivery_empty_windows_yields_empty_rows() {
        let rows = fixed_delivery_rows_from_windows(&[]);

        assert!(rows.is_empty());
    }
}
