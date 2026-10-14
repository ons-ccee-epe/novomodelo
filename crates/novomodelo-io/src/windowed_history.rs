//! Shared windowed-history record and Layer-1 validation.
//!
//! `recent_observations`, `past_defluences`, and `inflow_history` are the
//! same on-disk shape: `{hydro_id, [start_date, end_date), value_m3s}`, with
//! `end_date` exclusive. [`WindowedRecord`] is the post-parse view each
//! surface maps its rows into before calling [`validate_windowed_records`];
//! it introduces no on-disk shape of its own.

use chrono::NaiveDate;
use std::collections::BTreeMap;
use std::path::Path;

use crate::LoadError;

/// A windowed `[start_date, end_date)` observation over one physical
/// quantity, keyed by `entity_id`. The shared post-parse view for Layer-1
/// validation.
#[derive(Debug, Clone, Copy)]
pub(crate) struct WindowedRecord {
    pub(crate) entity_id: i32,
    pub(crate) start_date: NaiveDate,
    pub(crate) end_date: NaiveDate,
    pub(crate) value: f64,
}

/// Parse an ISO-8601 (`YYYY-MM-DD`) date. `field` is the already-formatted
/// `<surface>[<index>].<field_name>` path, used verbatim in the returned
/// error's `field` and as the message prefix.
pub(crate) fn parse_iso_date(field: &str, raw: &str, path: &Path) -> Result<NaiveDate, LoadError> {
    NaiveDate::parse_from_str(raw, "%Y-%m-%d").map_err(|_| LoadError::SchemaError {
        path: path.to_path_buf(),
        field: field.to_string(),
        message: format!("{field} '{raw}' is not a valid ISO 8601 date (expected YYYY-MM-DD)"),
    })
}

/// Validate `end_date > start_date`, per-entity no-overlap/contiguity
/// (`start == prev_end` accepted), and value finiteness over `records`, using
/// `(entity_id, start_date)` as the canonical order for the overlap sweep —
/// the shared Layer-1 check for every windowed temporal surface.
/// `surface_label` (e.g. `"recent_observations"`) names the caller in field
/// paths and messages, `entity_label` (e.g. `"hydro_id"`) names the id kind
/// in overlap messages, and `value_label` (e.g. `"value_m3s"`) names the
/// value field in finiteness field paths and messages — so per-surface text
/// stays byte-identical to each surface's own convention.
///
/// `records` may be supplied in any declaration order — the canonical sort is
/// internal to this function, so the validation outcome (and, on rejection,
/// which entity is named) does not depend on input order.
pub(crate) fn validate_windowed_records(
    records: &[WindowedRecord],
    surface_label: &str,
    entity_label: &str,
    value_label: &str,
    path: &Path,
) -> Result<(), LoadError> {
    for (i, r) in records.iter().enumerate() {
        if r.end_date <= r.start_date {
            return Err(LoadError::SchemaError {
                path: path.to_path_buf(),
                field: format!("{surface_label}[{i}].end_date"),
                message: format!(
                    "{surface_label}[{i}]: end_date must be after start_date for \
                     {entity_label} {} (start_date={}, end_date={})",
                    r.entity_id, r.start_date, r.end_date
                ),
            });
        }
        if !r.value.is_finite() {
            return Err(LoadError::SchemaError {
                path: path.to_path_buf(),
                field: format!("{surface_label}[{i}].{value_label}"),
                message: format!(
                    "{surface_label}[{i}].{value_label} must be a finite number, got {}",
                    r.value
                ),
            });
        }
    }

    let mut by_entity: BTreeMap<i32, Vec<usize>> = BTreeMap::new();
    for (i, r) in records.iter().enumerate() {
        by_entity.entry(r.entity_id).or_default().push(i);
    }

    for (entity_id, mut indices) in by_entity {
        indices.sort_by_key(|&i| records[i].start_date);

        for window in indices.windows(2) {
            let (i_prev, i_curr) = (window[0], window[1]);
            let prev_end = records[i_prev].end_date;
            let curr_start = records[i_curr].start_date;

            if curr_start < prev_end {
                return Err(LoadError::SchemaError {
                    path: path.to_path_buf(),
                    field: format!("{surface_label}[{i_curr}].start_date"),
                    message: format!(
                        "{surface_label}: overlapping windows for {entity_label} {entity_id}: \
                         entry [{i_prev}] ends on {prev_end} but entry [{i_curr}] starts on \
                         {curr_start}"
                    ),
                });
            }
        }
    }

    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;

    fn record(
        entity_id: i32,
        y1: i32,
        m1: u32,
        d1: u32,
        y2: i32,
        m2: u32,
        d2: u32,
        value: f64,
    ) -> WindowedRecord {
        WindowedRecord {
            entity_id,
            start_date: NaiveDate::from_ymd_opt(y1, m1, d1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(y2, m2, d2).unwrap(),
            value,
        }
    }

    #[test]
    fn test_end_date_before_start_date_rejected() {
        let records = vec![record(1, 2026, 4, 5, 2026, 4, 1, 100.0)];
        let err =
            validate_windowed_records(&records, "surface", "hydro_id", "value_m3s", Path::new("x"))
                .unwrap_err();
        match err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(field.contains("end_date"));
                assert!(message.contains("end_date must be after start_date"));
                assert!(
                    message.contains("hydro_id 1"),
                    "message should name the entity, got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    #[test]
    fn test_end_date_equals_start_date_rejected() {
        let records = vec![record(1, 2026, 4, 1, 2026, 4, 1, 100.0)];
        let err =
            validate_windowed_records(&records, "surface", "hydro_id", "value_m3s", Path::new("x"))
                .unwrap_err();
        match err {
            LoadError::SchemaError { message, .. } => {
                assert!(message.contains("end_date must be after start_date"));
                assert!(
                    message.contains("hydro_id 1"),
                    "message should name the entity, got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    #[test]
    fn test_non_finite_value_rejected() {
        let records = vec![record(1, 2026, 4, 1, 2026, 4, 4, f64::NAN)];
        let err =
            validate_windowed_records(&records, "surface", "hydro_id", "value_m3s", Path::new("x"))
                .unwrap_err();
        match err {
            LoadError::SchemaError { field, .. } => assert!(field.contains("value_m3s")),
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// The commitment surface (`past_anticipated_commitments`) passes its own
    /// `thermal_id`/`value_mw` labels, not the hydro surfaces' `hydro_id`/
    /// `value_m3s` — both the finiteness field path and the overlap message
    /// use the caller-supplied labels verbatim.
    #[test]
    fn test_commitment_surface_labels_used_verbatim() {
        let non_finite = vec![record(9, 2026, 1, 1, 2026, 2, 1, f64::NAN)];
        let err = validate_windowed_records(
            &non_finite,
            "past_anticipated_commitments",
            "thermal_id",
            "value_mw",
            Path::new("x"),
        )
        .unwrap_err();
        match err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(field.contains("value_mw"), "got: {field}");
                assert!(!field.contains("value_m3s"), "got: {field}");
                assert!(message.contains("value_mw"), "got: {message}");
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }

        let overlapping = vec![
            record(7, 2026, 1, 1, 2026, 2, 15, 120.0),
            record(7, 2026, 2, 1, 2026, 3, 1, 130.0),
        ];
        let err = validate_windowed_records(
            &overlapping,
            "past_anticipated_commitments",
            "thermal_id",
            "value_mw",
            Path::new("x"),
        )
        .unwrap_err();
        match err {
            LoadError::SchemaError { message, .. } => {
                assert!(message.contains("thermal_id 7"), "got: {message}");
                assert!(!message.contains("hydro_id"), "got: {message}");
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    #[test]
    fn test_adjacent_windows_accepted() {
        let records = vec![
            record(1, 2026, 4, 1, 2026, 4, 4, 100.0),
            record(1, 2026, 4, 4, 2026, 4, 11, 110.0),
        ];
        assert!(
            validate_windowed_records(&records, "surface", "hydro_id", "value_m3s", Path::new("x"))
                .is_ok()
        );
    }

    /// Two-entity fixture: entity 1's windows overlap, entity 2's are
    /// adjacent (valid). Only the overlapping entity is rejected, and it is
    /// named in the error.
    #[test]
    fn test_two_entity_fixture_rejects_only_overlapping_entity() {
        let records = vec![
            record(1, 2026, 4, 1, 2026, 4, 5, 100.0),
            record(1, 2026, 4, 3, 2026, 4, 10, 110.0),
            record(2, 2026, 4, 1, 2026, 4, 4, 200.0),
            record(2, 2026, 4, 4, 2026, 4, 11, 210.0),
        ];
        let err =
            validate_windowed_records(&records, "surface", "hydro_id", "value_m3s", Path::new("x"))
                .unwrap_err();
        match err {
            LoadError::SchemaError { message, .. } => {
                assert!(message.contains("overlapping windows"));
                assert!(
                    message.contains("hydro_id 1"),
                    "should name entity 1, got: {message}"
                );
                assert!(!message.contains("hydro_id 2"));
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// Declaration order must not change which entity an overlap is
    /// attributed to — the overlap sweep runs in `(entity_id, start_date)`
    /// canonical order regardless of input order.
    #[test]
    fn test_declaration_order_invariance_for_overlap_detection() {
        let forward = vec![
            record(1, 2026, 4, 1, 2026, 4, 5, 100.0),
            record(1, 2026, 4, 3, 2026, 4, 10, 110.0),
            record(2, 2026, 4, 1, 2026, 4, 4, 200.0),
        ];
        let mut reversed = forward.clone();
        reversed.reverse();
        let interleaved = vec![forward[2], forward[0], forward[1]];

        for records in [forward, reversed, interleaved] {
            let err = validate_windowed_records(
                &records,
                "surface",
                "hydro_id",
                "value_m3s",
                Path::new("x"),
            )
            .unwrap_err();
            match err {
                LoadError::SchemaError { message, .. } => {
                    assert!(message.contains("hydro_id 1"), "got: {message}");
                }
                other => panic!("expected SchemaError, got: {other:?}"),
            }
        }
    }

    mod declaration_order_invariance_proptests {
        use super::super::*;
        use proptest::prelude::*;
        use std::path::Path;

        proptest! {
            /// For any set of per-entity contiguous (non-overlapping) windows,
            /// shuffling the declaration order (via a decorate-sort-undecorate
            /// permutation) must not change the validation outcome: both the
            /// canonical and the shuffled order validate `Ok`.
            #[test]
            fn shuffled_valid_records_still_validate(
                n_entities in 1_i32..=4,
                windows_per_entity in 1_usize..=4,
                shuffle_keys in prop::collection::vec(0_u32..1000, 1..=16),
            ) {
                let mut canonical = Vec::new();
                for entity_id in 0..n_entities {
                    let mut cursor = NaiveDate::from_ymd_opt(2026, 1, 1).unwrap();
                    for _ in 0..windows_per_entity {
                        let end = cursor + chrono::TimeDelta::days(5);
                        canonical.push(WindowedRecord {
                            entity_id,
                            start_date: cursor,
                            end_date: end,
                            value: 100.0,
                        });
                        cursor = end;
                    }
                }

                let mut shuffled: Vec<(u32, WindowedRecord)> = canonical
                    .iter()
                    .copied()
                    .enumerate()
                    .map(|(i, r)| (shuffle_keys[i % shuffle_keys.len()], r))
                    .collect();
                shuffled.sort_by_key(|&(k, _)| k);
                let shuffled: Vec<WindowedRecord> = shuffled.into_iter().map(|(_, r)| r).collect();

                prop_assert!(validate_windowed_records(&canonical, "surface", "hydro_id", "value_m3s", Path::new("x")).is_ok());
                prop_assert!(validate_windowed_records(&shuffled, "surface", "hydro_id", "value_m3s", Path::new("x")).is_ok());
            }
        }
    }
}
