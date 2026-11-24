//! Parsing for `boundary_conditions.json` — the values a study needs at the
//! edges of its horizon, keyed by plant.
//!
//! [`parse_boundary_conditions_file`] reads the file alone and returns a
//! [`BoundaryConditionsFile`] with plants sorted by id, windows by `start_date`
//! and bound rows by `stage_id`. Resolving plant ids against `hydros.json` and
//! `thermals.json`, and `stage_id` against `stages.json`, is left to the step
//! that assembles the file into core types.
//!
//! ## JSON structure
//!
//! ```json
//! {
//!   "$schema": "https://raw.githubusercontent.com/ons-ccee-epe/novomodelo/refs/heads/main/schemas/boundary_conditions.schema.json",
//!   "hydros": [
//!     {
//!       "hydro_id": 0,
//!       "storage_hm3": 80.0,
//!       "recent_inflows": [
//!         { "start_date": "2025-12-15", "end_date": "2026-01-01", "value_m3s": 12.0 }
//!       ],
//!       "past_defluences": [
//!         { "start_date": "2025-11-01", "end_date": "2026-01-01", "value_m3s": 20.0 }
//!       ]
//!     }
//!   ],
//!   "thermals": [
//!     {
//!       "thermal_id": 86,
//!       "commitments": [
//!         { "start_date": "2026-01-01", "end_date": "2026-02-01", "value_mw": 0.0 }
//!       ],
//!       "post_study_bounds": [
//!         { "stage_id": 121, "cost_per_mwh": 210.0, "min_mw": 0.0, "max_mw": 350.0 }
//!       ]
//!     }
//!   ]
//! }
//! ```
//!
//! ## Validation
//!
//! 1. No `hydro_id` repeats within `hydros` and no `thermal_id` within
//!    `thermals`.
//! 2. Every `storage_hm3` is non-negative.
//! 3. Every window parses as an ISO 8601 `[start_date, end_date)` range with
//!    `end_date > start_date` and a finite value, and a plant's windows of one
//!    list do not overlap (adjacent ranges where `start == prev_end` are
//!    accepted).
//! 4. Every `past_defluences` value is non-negative; a negative
//!    `recent_inflows` value is accepted (the quantity is incremental inflow).
//! 5. No two `commitments` of a plant share a `start_date`.
//! 6. Every `post_study_bounds` row has finite values with `min_mw <= max_mw`,
//!    and no `stage_id` repeats within a plant.

use std::collections::HashSet;
use std::path::Path;

use chrono::NaiveDate;
use serde::Deserialize;

use crate::LoadError;
use crate::boundary::checks::{
    check_bound_values, reject_duplicate_window_starts, reject_negative_window_values,
};
use crate::windowed_history::{WindowedRecord, parse_iso_date, validate_windowed_records};

// ── Intermediate serde types ──────────────────────────────────────────────────

/// Root object of `boundary_conditions.json`.
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct RawBoundaryConditionsFile {
    /// JSON schema URI — informational, not validated.
    #[serde(rename = "$schema")]
    _schema: Option<String>,

    /// Boundary values per hydro plant. A plant with none may be left out.
    /// Optional; defaults to empty.
    #[serde(default)]
    hydros: Vec<RawHydroBoundary>,

    /// Boundary values per thermal plant. A plant with none may be left out.
    /// Optional; defaults to empty.
    #[serde(default)]
    thermals: Vec<RawThermalBoundary>,
}

/// Boundary values of one hydro plant.
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawHydroBoundary {
    /// Hydro plant identifier. Must be unique within `hydros`.
    hydro_id: i32,

    /// Reservoir volume at the study start (hm³). Must be >= 0.0. Optional.
    storage_hm3: Option<f64>,

    /// Observed inflow for partial periods before the study start. Windows
    /// must not overlap; adjacent windows (start == previous end) are
    /// accepted. Optional; defaults to empty.
    #[serde(default)]
    recent_inflows: Vec<RawWindowM3s>,

    /// Past release windows of the plant, on the pre-study calendar. Windows
    /// must not overlap; adjacent windows (start == previous end) are
    /// accepted. Optional; defaults to empty.
    #[serde(default)]
    past_defluences: Vec<RawWindowM3s>,
}

/// Boundary values of one thermal plant.
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawThermalBoundary {
    /// Thermal plant identifier. Must be unique within `thermals`.
    thermal_id: i32,

    /// Committed MW windows decided before the study, covering the leading
    /// in-study delivery stages or the post-horizon deliveries. Windows must
    /// not overlap or share a `start_date`; adjacent windows (start ==
    /// previous end) are accepted. Optional; defaults to empty.
    #[serde(default)]
    commitments: Vec<RawWindowMw>,

    /// Cost and generation bounds per post-study stage the study decides. A
    /// `stage_id` may appear once per plant. Optional; defaults to empty.
    #[serde(default)]
    post_study_bounds: Vec<RawPostStudyBound>,
}

/// A `[start_date, end_date)` window carrying a flow rate in m³/s.
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawWindowM3s {
    /// Start of the window (inclusive), as an ISO 8601 date (YYYY-MM-DD).
    start_date: String,
    /// End of the window (exclusive), as an ISO 8601 date (YYYY-MM-DD). Must be
    /// after `start_date`.
    end_date: String,
    /// Average rate over the window (m³/s). Must be finite.
    value_m3s: f64,
}

/// A `[start_date, end_date)` window carrying a power rate in MW.
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawWindowMw {
    /// Start of the window (inclusive), as an ISO 8601 date (YYYY-MM-DD).
    start_date: String,
    /// End of the window (exclusive), as an ISO 8601 date (YYYY-MM-DD). Must be
    /// after `start_date`.
    end_date: String,
    /// Committed rate held constant over the window (MW). Must be finite.
    value_mw: f64,
}

/// Cost and generation bounds for one post-study stage of a thermal plant.
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawPostStudyBound {
    /// Identifier of a post-study thermal stage declared in `stages.json`.
    stage_id: i32,
    /// Fuel cost (`$/MWh`) at this stage. Must be finite.
    cost_per_mwh: f64,
    /// Lower bound of the delivered MW rate at this stage. Must be finite.
    min_mw: f64,
    /// Upper bound of the delivered MW rate at this stage. Must be finite and
    /// `>= min_mw`.
    max_mw: f64,
}

// ── Parsed types ──────────────────────────────────────────────────────────────

/// A validated `boundary_conditions.json`, sorted by plant id.
#[derive(Debug, PartialEq)]
pub(crate) struct BoundaryConditionsFile {
    pub(crate) hydros: Vec<HydroBoundary>,
    pub(crate) thermals: Vec<ThermalBoundary>,
}

/// Boundary values of one hydro plant; windows sorted by `start_date`.
#[derive(Debug, PartialEq)]
pub(crate) struct HydroBoundary {
    pub(crate) hydro_id: i32,
    pub(crate) storage_hm3: Option<f64>,
    pub(crate) recent_inflows: Vec<Window>,
    pub(crate) past_defluences: Vec<Window>,
}

/// Boundary values of one thermal plant; `commitments` sorted by `start_date`
/// and `post_study_bounds` by `stage_id`.
#[derive(Debug, PartialEq)]
pub(crate) struct ThermalBoundary {
    pub(crate) thermal_id: i32,
    pub(crate) commitments: Vec<Window>,
    pub(crate) post_study_bounds: Vec<PostStudyBound>,
}

/// A `[start_date, end_date)` window and its rate (m³/s or MW by list).
#[derive(Debug, PartialEq)]
pub(crate) struct Window {
    pub(crate) start_date: NaiveDate,
    pub(crate) end_date: NaiveDate,
    pub(crate) value: f64,
}

/// Cost and generation bounds for one post-study stage.
#[derive(Debug, PartialEq)]
pub(crate) struct PostStudyBound {
    pub(crate) stage_id: i32,
    pub(crate) cost_per_mwh: f64,
    pub(crate) min_mw: f64,
    pub(crate) max_mw: f64,
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Load and validate `boundary_conditions.json` from `path`.
///
/// # Errors
///
/// | Condition                                                      | Error variant              |
/// | -------------------------------------------------------------- | -------------------------- |
/// | File not found / read failure                                  | [`LoadError::IoError`]     |
/// | Invalid JSON, unknown key, missing required field              | [`LoadError::ParseError`]  |
/// | Repeated `hydro_id`, `thermal_id` or `stage_id`                | [`LoadError::SchemaError`] |
/// | Negative `storage_hm3` or `past_defluences` value              | [`LoadError::SchemaError`] |
/// | Invalid date, `end_date` not after `start_date`, overlap       | [`LoadError::SchemaError`] |
/// | Non-finite window or bound value, `min_mw > max_mw`            | [`LoadError::SchemaError`] |
/// | Two `commitments` of a plant sharing a `start_date`            | [`LoadError::SchemaError`] |
pub(crate) fn parse_boundary_conditions_file(
    path: &Path,
) -> Result<BoundaryConditionsFile, LoadError> {
    let raw_text = std::fs::read_to_string(path).map_err(|e| LoadError::io(path, e))?;

    let raw: RawBoundaryConditionsFile =
        serde_json::from_str(&raw_text).map_err(|e| LoadError::parse(path, e.to_string()))?;

    reject_duplicate_ids(
        raw.hydros.iter().map(|hydro| hydro.hydro_id),
        "hydros",
        "hydro_id",
        path,
    )?;
    reject_duplicate_ids(
        raw.thermals.iter().map(|thermal| thermal.thermal_id),
        "thermals",
        "thermal_id",
        path,
    )?;

    let mut hydros = raw
        .hydros
        .iter()
        .enumerate()
        .map(|(k, hydro)| convert_hydro(k, hydro, path))
        .collect::<Result<Vec<_>, _>>()?;
    let mut thermals = raw
        .thermals
        .iter()
        .enumerate()
        .map(|(k, thermal)| convert_thermal(k, thermal, path))
        .collect::<Result<Vec<_>, _>>()?;
    hydros.sort_by_key(|hydro| hydro.hydro_id);
    thermals.sort_by_key(|thermal| thermal.thermal_id);

    Ok(BoundaryConditionsFile { hydros, thermals })
}

// ── Validation and conversion ─────────────────────────────────────────────────

fn convert_hydro(
    k: usize,
    raw: &RawHydroBoundary,
    path: &Path,
) -> Result<HydroBoundary, LoadError> {
    if let Some(storage) = raw.storage_hm3.filter(|storage| *storage < 0.0) {
        return Err(LoadError::SchemaError {
            path: path.to_path_buf(),
            field: format!("hydros[{k}].storage_hm3"),
            message: format!("storage_hm3 must be >= 0.0, got {storage}"),
        });
    }

    let recent_surface = format!("hydros[{k}].recent_inflows");
    let recent_inflows = parse_windows(
        raw.hydro_id,
        raw.recent_inflows
            .iter()
            .map(|w| (w.start_date.as_str(), w.end_date.as_str(), w.value_m3s)),
        &recent_surface,
        path,
    )?;
    validate_windowed_records(
        &recent_inflows,
        &recent_surface,
        "hydro_id",
        "value_m3s",
        path,
    )?;

    let defluence_surface = format!("hydros[{k}].past_defluences");
    let past_defluences = parse_windows(
        raw.hydro_id,
        raw.past_defluences
            .iter()
            .map(|w| (w.start_date.as_str(), w.end_date.as_str(), w.value_m3s)),
        &defluence_surface,
        path,
    )?;
    validate_windowed_records(
        &past_defluences,
        &defluence_surface,
        "hydro_id",
        "value_m3s",
        path,
    )?;
    reject_negative_window_values(&past_defluences, &defluence_surface, "value_m3s", path)?;

    Ok(HydroBoundary {
        hydro_id: raw.hydro_id,
        storage_hm3: raw.storage_hm3,
        recent_inflows: into_sorted_windows(recent_inflows),
        past_defluences: into_sorted_windows(past_defluences),
    })
}

fn convert_thermal(
    k: usize,
    raw: &RawThermalBoundary,
    path: &Path,
) -> Result<ThermalBoundary, LoadError> {
    let commitments_surface = format!("thermals[{k}].commitments");
    let commitments = parse_windows(
        raw.thermal_id,
        raw.commitments
            .iter()
            .map(|w| (w.start_date.as_str(), w.end_date.as_str(), w.value_mw)),
        &commitments_surface,
        path,
    )?;
    reject_duplicate_window_starts(&commitments, &commitments_surface, "thermal_id", path)?;
    validate_windowed_records(
        &commitments,
        &commitments_surface,
        "thermal_id",
        "value_mw",
        path,
    )?;

    let bounds_surface = format!("thermals[{k}].post_study_bounds");
    for (i, bound) in raw.post_study_bounds.iter().enumerate() {
        check_bound_values(
            bound.cost_per_mwh,
            bound.min_mw,
            bound.max_mw,
            &format!("{bounds_surface}[{i}]"),
            &format!("thermal_id {}, stage_id {}", raw.thermal_id, bound.stage_id),
            path,
        )?;
    }
    reject_duplicate_ids(
        raw.post_study_bounds.iter().map(|bound| bound.stage_id),
        &bounds_surface,
        "stage_id",
        path,
    )?;
    let mut post_study_bounds: Vec<PostStudyBound> = raw
        .post_study_bounds
        .iter()
        .map(|bound| PostStudyBound {
            stage_id: bound.stage_id,
            cost_per_mwh: bound.cost_per_mwh,
            min_mw: bound.min_mw,
            max_mw: bound.max_mw,
        })
        .collect();
    post_study_bounds.sort_by_key(|bound| bound.stage_id);

    Ok(ThermalBoundary {
        thermal_id: raw.thermal_id,
        commitments: into_sorted_windows(commitments),
        post_study_bounds,
    })
}

/// Parse the dates of `rows` (`start_date`, `end_date`, value) into the shared
/// windowed view; `surface` is the declared field path of the list.
fn parse_windows<'a>(
    entity_id: i32,
    rows: impl Iterator<Item = (&'a str, &'a str, f64)>,
    surface: &str,
    path: &Path,
) -> Result<Vec<WindowedRecord>, LoadError> {
    rows.enumerate()
        .map(|(i, (start_date, end_date, value))| {
            Ok(WindowedRecord {
                entity_id,
                start_date: parse_iso_date(
                    &format!("{surface}[{i}].start_date"),
                    start_date,
                    path,
                )?,
                end_date: parse_iso_date(&format!("{surface}[{i}].end_date"), end_date, path)?,
                value,
            })
        })
        .collect()
}

fn into_sorted_windows(mut records: Vec<WindowedRecord>) -> Vec<Window> {
    records.sort_by_key(|record| record.start_date);
    records
        .into_iter()
        .map(|record| Window {
            start_date: record.start_date,
            end_date: record.end_date,
            value: record.value,
        })
        .collect()
}

fn reject_duplicate_ids(
    ids: impl Iterator<Item = i32>,
    list: &str,
    id_label: &str,
    path: &Path,
) -> Result<(), LoadError> {
    let mut seen = HashSet::new();
    for (i, id) in ids.enumerate() {
        if !seen.insert(id) {
            return Err(LoadError::SchemaError {
                path: path.to_path_buf(),
                field: format!("{list}[{i}].{id_label}"),
                message: format!("duplicate {id_label} {id} in {list}"),
            });
        }
    }
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use std::path::Path;

    use super::{
        BoundaryConditionsFile, HydroBoundary, PostStudyBound, ThermalBoundary, Window,
        parse_boundary_conditions_file,
    };
    use crate::LoadError;
    use crate::test_support::{date, write_json};

    fn parse(json: &str) -> Result<BoundaryConditionsFile, LoadError> {
        let file = write_json(json);
        parse_boundary_conditions_file(file.path())
    }

    const SPEC_EXAMPLE: &str = r#"{
        "$schema": "https://example.invalid/boundary_conditions.schema.json",
        "hydros": [{
            "hydro_id": 0,
            "storage_hm3": 80.0,
            "recent_inflows": [
                { "start_date": "2025-12-15", "end_date": "2026-01-01", "value_m3s": 12.0 }
            ],
            "past_defluences": [
                { "start_date": "2025-11-01", "end_date": "2026-01-01", "value_m3s": 20.0 }
            ]
        }],
        "thermals": [{
            "thermal_id": 86,
            "commitments": [
                { "start_date": "2026-01-01", "end_date": "2026-02-01", "value_mw": 0.0 },
                { "start_date": "2026-11-01", "end_date": "2026-11-07", "value_mw": 120.0 }
            ],
            "post_study_bounds": [
                { "stage_id": 121, "cost_per_mwh": 210.0, "min_mw": 0.0, "max_mw": 350.0 }
            ]
        }]
    }"#;

    #[test]
    fn parses_spec_example() {
        let file = parse(SPEC_EXAMPLE).unwrap();
        assert_eq!(
            file,
            BoundaryConditionsFile {
                hydros: vec![HydroBoundary {
                    hydro_id: 0,
                    storage_hm3: Some(80.0),
                    recent_inflows: vec![Window {
                        start_date: date(2025, 12, 15),
                        end_date: date(2026, 1, 1),
                        value: 12.0,
                    }],
                    past_defluences: vec![Window {
                        start_date: date(2025, 11, 1),
                        end_date: date(2026, 1, 1),
                        value: 20.0,
                    }],
                }],
                thermals: vec![ThermalBoundary {
                    thermal_id: 86,
                    commitments: vec![
                        Window {
                            start_date: date(2026, 1, 1),
                            end_date: date(2026, 2, 1),
                            value: 0.0,
                        },
                        Window {
                            start_date: date(2026, 11, 1),
                            end_date: date(2026, 11, 7),
                            value: 120.0,
                        },
                    ],
                    post_study_bounds: vec![PostStudyBound {
                        stage_id: 121,
                        cost_per_mwh: 210.0,
                        min_mw: 0.0,
                        max_mw: 350.0,
                    }],
                }],
            }
        );
    }

    #[test]
    fn empty_object_parses_to_empty_file() {
        let file = parse("{}").unwrap();
        assert!(file.hydros.is_empty());
        assert!(file.thermals.is_empty());
    }

    #[test]
    fn plant_without_optional_keys_parses() {
        let file =
            parse(r#"{"hydros": [{"hydro_id": 4}], "thermals": [{"thermal_id": 9}]}"#).unwrap();
        assert_eq!(
            file.hydros,
            [HydroBoundary {
                hydro_id: 4,
                storage_hm3: None,
                recent_inflows: vec![],
                past_defluences: vec![],
            }]
        );
        assert_eq!(
            file.thermals,
            [ThermalBoundary {
                thermal_id: 9,
                commitments: vec![],
                post_study_bounds: vec![],
            }]
        );
    }

    #[test]
    fn negative_recent_inflow_is_accepted() {
        let file = parse(
            r#"{"hydros": [{"hydro_id": 0, "recent_inflows": [
                { "start_date": "2026-01-01", "end_date": "2026-01-04", "value_m3s": -2.5 }
            ]}]}"#,
        )
        .unwrap();
        assert!((file.hydros[0].recent_inflows[0].value - -2.5).abs() < f64::EPSILON);
    }

    #[test]
    fn output_is_sorted_whatever_the_declaration_order() {
        let forward = r#"{
            "hydros": [
                { "hydro_id": 3, "recent_inflows": [
                    { "start_date": "2026-01-01", "end_date": "2026-01-04", "value_m3s": 1.0 },
                    { "start_date": "2026-01-04", "end_date": "2026-01-08", "value_m3s": 2.0 }
                ]},
                { "hydro_id": 5 }
            ],
            "thermals": [
                { "thermal_id": 2,
                  "commitments": [
                    { "start_date": "2026-01-01", "end_date": "2026-02-01", "value_mw": 1.0 },
                    { "start_date": "2026-02-01", "end_date": "2026-03-01", "value_mw": 2.0 }
                  ],
                  "post_study_bounds": [
                    { "stage_id": 10, "cost_per_mwh": 1.0, "min_mw": 0.0, "max_mw": 5.0 },
                    { "stage_id": 11, "cost_per_mwh": 2.0, "min_mw": 0.0, "max_mw": 6.0 }
                  ] },
                { "thermal_id": 9 }
            ]
        }"#;
        let reversed = r#"{
            "hydros": [
                { "hydro_id": 5 },
                { "hydro_id": 3, "recent_inflows": [
                    { "start_date": "2026-01-04", "end_date": "2026-01-08", "value_m3s": 2.0 },
                    { "start_date": "2026-01-01", "end_date": "2026-01-04", "value_m3s": 1.0 }
                ]}
            ],
            "thermals": [
                { "thermal_id": 9 },
                { "thermal_id": 2,
                  "commitments": [
                    { "start_date": "2026-02-01", "end_date": "2026-03-01", "value_mw": 2.0 },
                    { "start_date": "2026-01-01", "end_date": "2026-02-01", "value_mw": 1.0 }
                  ],
                  "post_study_bounds": [
                    { "stage_id": 11, "cost_per_mwh": 2.0, "min_mw": 0.0, "max_mw": 6.0 },
                    { "stage_id": 10, "cost_per_mwh": 1.0, "min_mw": 0.0, "max_mw": 5.0 }
                  ] }
            ]
        }"#;
        let sorted = parse(forward).unwrap();
        assert_eq!(sorted, parse(reversed).unwrap());
        assert_eq!(
            sorted.hydros.iter().map(|h| h.hydro_id).collect::<Vec<_>>(),
            [3, 5]
        );
        assert_eq!(
            sorted
                .thermals
                .iter()
                .map(|t| t.thermal_id)
                .collect::<Vec<_>>(),
            [2, 9]
        );
        assert_eq!(
            sorted.hydros[0].recent_inflows[0].start_date,
            date(2026, 1, 1)
        );
        assert_eq!(
            sorted.thermals[0].commitments[0].start_date,
            date(2026, 1, 1)
        );
        assert_eq!(sorted.thermals[0].post_study_bounds[0].stage_id, 10);
    }

    #[test]
    fn missing_bound_field_is_a_parse_error_naming_it() {
        let err = parse(
            r#"{"thermals": [{"thermal_id": 1, "post_study_bounds": [
                { "stage_id": 121, "cost_per_mwh": 210.0, "min_mw": 0.0 }
            ]}]}"#,
        )
        .unwrap_err();
        match err {
            LoadError::ParseError { message, .. } => {
                assert!(message.contains("max_mw"), "got: {message}");
            }
            other => panic!("expected ParseError, got: {other:?}"),
        }
    }

    #[test]
    fn unreadable_file_is_an_io_error() {
        let err =
            parse_boundary_conditions_file(Path::new("/nonexistent/dir/boundary_conditions.json"))
                .unwrap_err();
        assert_eq!(err.kind(), "IoError");
    }

    /// `(case, file, error kind, detail)`: `detail` is the exact `field` of a
    /// `SchemaError` and a message substring of a `ParseError`.
    const REJECTIONS: &[(&str, &str, &str, &str)] = &[
        (
            "duplicate hydro_id",
            r#"{"hydros": [{"hydro_id": 5}, {"hydro_id": 3}, {"hydro_id": 5}]}"#,
            "SchemaError",
            "hydros[2].hydro_id",
        ),
        (
            "duplicate thermal_id",
            r#"{"thermals": [{"thermal_id": 9}, {"thermal_id": 9}]}"#,
            "SchemaError",
            "thermals[1].thermal_id",
        ),
        (
            "negative storage",
            r#"{"hydros": [{"hydro_id": 5}, {"hydro_id": 3, "storage_hm3": -1.0}]}"#,
            "SchemaError",
            "hydros[1].storage_hm3",
        ),
        (
            "recent inflow with a bad date",
            r#"{"hydros": [{"hydro_id": 0, "recent_inflows": [
                { "start_date": "2026-13-01", "end_date": "2026-02-01", "value_m3s": 1.0 }
            ]}]}"#,
            "SchemaError",
            "hydros[0].recent_inflows[0].start_date",
        ),
        (
            "recent inflow ending before it starts",
            r#"{"hydros": [{"hydro_id": 0, "recent_inflows": [
                { "start_date": "2026-01-01", "end_date": "2026-01-04", "value_m3s": 1.0 },
                { "start_date": "2026-02-01", "end_date": "2026-01-15", "value_m3s": 1.0 }
            ]}]}"#,
            "SchemaError",
            "hydros[0].recent_inflows[1].end_date",
        ),
        (
            "recent inflows overlapping",
            r#"{"hydros": [{"hydro_id": 0, "recent_inflows": [
                { "start_date": "2026-01-01", "end_date": "2026-01-10", "value_m3s": 1.0 },
                { "start_date": "2026-01-05", "end_date": "2026-01-15", "value_m3s": 1.0 }
            ]}]}"#,
            "SchemaError",
            "hydros[0].recent_inflows[1].start_date",
        ),
        (
            "past defluences overlapping on the second hydro",
            r#"{"hydros": [{"hydro_id": 9}, {"hydro_id": 0, "past_defluences": [
                { "start_date": "2026-01-01", "end_date": "2026-01-10", "value_m3s": 1.0 },
                { "start_date": "2026-01-05", "end_date": "2026-01-15", "value_m3s": 1.0 }
            ]}]}"#,
            "SchemaError",
            "hydros[1].past_defluences[1].start_date",
        ),
        (
            "negative past defluence",
            r#"{"hydros": [{"hydro_id": 0, "past_defluences": [
                { "start_date": "2026-01-01", "end_date": "2026-01-04", "value_m3s": -1.0 }
            ]}]}"#,
            "SchemaError",
            "hydros[0].past_defluences[0].value_m3s",
        ),
        (
            "commitment with a bad date",
            r#"{"thermals": [{"thermal_id": 1, "commitments": [
                { "start_date": "2026-01-01", "end_date": "soon", "value_mw": 1.0 }
            ]}]}"#,
            "SchemaError",
            "thermals[0].commitments[0].end_date",
        ),
        (
            "commitment ending on its start",
            r#"{"thermals": [{"thermal_id": 1, "commitments": [
                { "start_date": "2026-01-01", "end_date": "2026-01-01", "value_mw": 1.0 }
            ]}]}"#,
            "SchemaError",
            "thermals[0].commitments[0].end_date",
        ),
        (
            "commitments overlapping",
            r#"{"thermals": [{"thermal_id": 1, "commitments": [
                { "start_date": "2026-01-01", "end_date": "2026-02-01", "value_mw": 1.0 },
                { "start_date": "2026-01-15", "end_date": "2026-03-01", "value_mw": 1.0 }
            ]}]}"#,
            "SchemaError",
            "thermals[0].commitments[1].start_date",
        ),
        (
            "commitments sharing a start",
            r#"{"thermals": [{"thermal_id": 1, "commitments": [
                { "start_date": "2026-01-01", "end_date": "2026-02-01", "value_mw": 1.0 },
                { "start_date": "2026-01-01", "end_date": "2026-03-01", "value_mw": 1.0 }
            ]}]}"#,
            "SchemaError",
            "thermals[0].commitments[1].start_date",
        ),
        (
            "bound with min above max",
            r#"{"thermals": [{"thermal_id": 1, "post_study_bounds": [
                { "stage_id": 121, "cost_per_mwh": 1.0, "min_mw": 9.0, "max_mw": 3.0 }
            ]}]}"#,
            "SchemaError",
            "thermals[0].post_study_bounds[0].max_mw",
        ),
        (
            "repeated stage_id on the second thermal",
            r#"{"thermals": [{"thermal_id": 9}, {"thermal_id": 1, "post_study_bounds": [
                { "stage_id": 121, "cost_per_mwh": 1.0, "min_mw": 0.0, "max_mw": 3.0 },
                { "stage_id": 122, "cost_per_mwh": 1.0, "min_mw": 0.0, "max_mw": 3.0 },
                { "stage_id": 121, "cost_per_mwh": 2.0, "min_mw": 0.0, "max_mw": 4.0 }
            ]}]}"#,
            "SchemaError",
            "thermals[1].post_study_bounds[2].stage_id",
        ),
        (
            "unknown top-level key",
            r#"{"plants": []}"#,
            "ParseError",
            "`plants`",
        ),
        (
            "unknown hydro key",
            r#"{"hydros": [{"hydro_id": 0, "bus_id": 1}]}"#,
            "ParseError",
            "`bus_id`",
        ),
        (
            "unknown thermal key",
            r#"{"thermals": [{"thermal_id": 0, "bus_id": 1}]}"#,
            "ParseError",
            "`bus_id`",
        ),
        (
            "unknown recent inflow key",
            r#"{"hydros": [{"hydro_id": 0, "recent_inflows": [
                { "start_date": "2026-01-01", "end_date": "2026-01-04",
                  "value_m3s": 1.0, "note": "x" }
            ]}]}"#,
            "ParseError",
            "`note`",
        ),
        (
            "unknown past defluence key",
            r#"{"hydros": [{"hydro_id": 0, "past_defluences": [
                { "start_date": "2026-01-01", "end_date": "2026-01-04",
                  "value_m3s": 1.0, "note": "x" }
            ]}]}"#,
            "ParseError",
            "`note`",
        ),
        (
            "unknown commitment key",
            r#"{"thermals": [{"thermal_id": 0, "commitments": [
                { "start_date": "2026-01-01", "end_date": "2026-01-04",
                  "value_mw": 1.0, "note": "x" }
            ]}]}"#,
            "ParseError",
            "`note`",
        ),
        (
            "unknown bound key",
            r#"{"thermals": [{"thermal_id": 0, "post_study_bounds": [
                { "stage_id": 1, "cost_per_mwh": 1.0, "min_mw": 0.0, "max_mw": 1.0,
                  "note": "x" }
            ]}]}"#,
            "ParseError",
            "`note`",
        ),
        (
            "hydro window carrying value_mw",
            r#"{"hydros": [{"hydro_id": 0, "recent_inflows": [
                { "start_date": "2026-01-01", "end_date": "2026-01-04", "value_mw": 1.0 }
            ]}]}"#,
            "ParseError",
            "`value_mw`",
        ),
        (
            "hydro without hydro_id",
            r#"{"hydros": [{"storage_hm3": 1.0}]}"#,
            "ParseError",
            "`hydro_id`",
        ),
    ];

    #[test]
    fn boundary_file_rejections() {
        for &(case, json, kind, detail) in REJECTIONS {
            let err = parse(json).unwrap_err();
            assert_eq!(err.kind(), kind, "{case}: {err}");
            match err {
                LoadError::SchemaError { field, .. } => assert_eq!(field, detail, "{case}"),
                LoadError::ParseError { message, .. } => {
                    assert!(message.contains(detail), "{case}: {message}");
                }
                other => panic!("{case}: unexpected {other:?}"),
            }
        }
    }
}
