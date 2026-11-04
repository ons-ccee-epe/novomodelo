//! Parsing for `post_study_stages.json` — the post-horizon boundary calendar and
//! per-`(thermal, post-study stage)` cost/bounds table.
//!
//! [`parse_post_study_file`] reads `post_study_stages.json` from the case
//! directory root, classifies its [`PostStudyFormat`], and returns a
//! [`PostStudyStages`] whose `thermal_stages` and `water_bucket_stages` are ordered
//! ascending by `start_date` and whose `thermal_bounds` are in
//! `(thermal_id, post_study_stage_index)` order.
//!
//! ## JSON structure
//!
//! ```json
//! {
//!   "$schema": "https://raw.githubusercontent.com/ons-ccee-epe/novomodelo/refs/heads/main/schemas/post_study_stages.schema.json",
//!   "thermal_stages": [
//!     { "start_date": "2026-11-01", "duration_hours": 720.0 },
//!     { "start_date": "2026-12-01", "duration_hours": 744.0 }
//!   ],
//!   "thermal_bounds": [
//!     { "thermal_id": 86, "post_study_stage_index": 0, "cost_per_mwh": 210.0, "min_mw": 0.0, "max_mw": 350.0 }
//!   ],
//!   "water_bucket_stages": [
//!     { "start_date": "2026-11-01", "duration_hours": 720.0 }
//!   ]
//! }
//! ```
//!
//! `stages` is the deprecated spelling of `thermal_stages`; a file using it and
//! neither new key is [`PostStudyFormat::Legacy`], and one using it together with
//! a new key is [`PostStudyFormat::Mixed`].
//!
//! ## Validation
//!
//! The reader enforces the per-file invariants; the cross-file invariants
//! (date-contiguity, first `start_date` equals the study horizon end, a
//! `PostStudyThermalBound` for every post-study stage an anticipated lead
//! reaches, and no post-study delivery decided at a pre-study stage) are
//! enforced by the `novomodelo-io` semantic validator, which has the study
//! calendar this reader does not.
//!
//! 1. Every `start_date` parses as ISO 8601 (`YYYY-MM-DD`); no two stages share a
//!    `start_date`.
//! 2. Every `duration_hours` is finite and `> 0.0`.
//! 3. Every `cost_per_mwh`, `min_mw`, and `max_mw` is finite, and `min_mw <= max_mw`.
//! 4. Every `post_study_stage_index` is `< thermal_stages.len()` (`stages.len()` in the
//!    deprecated format).
//! 5. No two thermal-bound rows share a `(thermal_id, post_study_stage_index)`.

use std::collections::HashSet;
use std::path::Path;

use novomodelo_core::{
    EntityId, PostStudyStage, PostStudyStages, PostStudyThermalBound, WaterBucketStages,
};
#[cfg(feature = "schema")]
use schemars::Schema;
use serde::Deserialize;
#[cfg(feature = "schema")]
use serde_json::{Value, json};

use crate::LoadError;
use crate::windowed_history::parse_iso_date;

/// Format of a deck's `post_study_stages.json`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PostStudyFormat {
    Absent,
    Legacy,
    /// Each flag records key presence; an empty list counts as present.
    Current {
        thermal_stages: bool,
        thermal_bounds: bool,
        water_bucket_stages: bool,
    },
    Mixed,
}

// ── Intermediate serde types ──────────────────────────────────────────────────

/// Root object of `post_study_stages.json`.
///
/// A file uses either the deprecated `stages` key or the `thermal_stages` and
/// `water_bucket_stages` keys, never both. `scripts/migrate/post_study_stages.py`
/// rewrites the former into the latter.
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[cfg_attr(feature = "schema", schemars(transform = post_study_file_schema))]
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct RawPostStudyStagesFile {
    /// JSON schema URI — informational, not validated.
    #[serde(rename = "$schema")]
    _schema: Option<String>,

    /// Deprecated: replaced by `thermal_stages`. Cannot be combined with
    /// `thermal_stages` or `water_bucket_stages`.
    stages: Option<Vec<RawPostStudyStage>>,

    /// Ordered post-horizon thermal calendar stages; they date thermal
    /// deliveries and `thermal_bounds`.
    thermal_stages: Option<Vec<RawPostStudyStage>>,

    /// Ordered post-horizon water-bucket calendar stages; they date water-bucket
    /// slots.
    water_bucket_stages: Option<Vec<RawPostStudyStage>>,

    /// Per-`(thermal, post-study stage)` cost and generation bounds. Required
    /// with the deprecated `stages` key.
    thermal_bounds: Option<Vec<RawPostStudyThermalBound>>,
}

#[cfg(feature = "schema")]
fn post_study_file_schema(schema: &mut Schema) {
    schema.insert(
        "oneOf".to_owned(),
        json!([
            {
                "required": ["stages", "thermal_bounds"],
                "not": { "anyOf": [
                    { "required": ["thermal_stages"] },
                    { "required": ["water_bucket_stages"] },
                ] },
            },
            {
                "not": { "anyOf": [{ "required": ["stages"] }] },
                "dependentRequired": {
                    "thermal_stages": ["thermal_bounds"],
                    "thermal_bounds": ["thermal_stages"],
                },
            },
        ]),
    );
    if let Some(stages) = schema
        .pointer_mut("/properties/stages")
        .and_then(Value::as_object_mut)
    {
        stages.insert("deprecated".to_owned(), Value::Bool(true));
    }
}

/// One post-horizon calendar stage: a `[start_date, start_date + duration_hours)`
/// segment. `end_date` is not declared; it is derived downstream.
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawPostStudyStage {
    /// Stage start date (inclusive), as an ISO 8601 date (YYYY-MM-DD).
    start_date: String,
    /// Stage duration (h). Must be finite and `> 0.0`.
    duration_hours: f64,
}

/// Cost and generation bounds for one `(thermal, post-study stage)` cell.
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawPostStudyThermalBound {
    /// Thermal plant identifier. Must reference an anticipated thermal.
    thermal_id: i32,
    /// Index into `thermal_stages` (`stages` in the deprecated format). Must be
    /// less than that list's length.
    post_study_stage_index: usize,
    /// Fuel cost (`$/MWh`) at this cell. Must be finite.
    cost_per_mwh: f64,
    /// Lower bound of the delivered MW rate at this cell. Must be finite.
    min_mw: f64,
    /// Upper bound of the delivered MW rate at this cell. Must be finite and
    /// `>= min_mw`.
    max_mw: f64,
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Load, classify, and validate `post_study_stages.json` from `path`.
///
/// Returns `(Mixed, None)` for a file that declares `stages` together with
/// `thermal_stages` or `water_bucket_stages`, and `(Current, None)` for a current file that
/// declares only one of `thermal_stages` and `thermal_bounds`; the semantic layer reports both.
///
/// # Errors
///
/// | Condition                                                        | Error variant              |
/// | ---------------------------------------------------------------- | -------------------------- |
/// | File not found / read failure                                    | [`LoadError::IoError`]     |
/// | Invalid JSON, unknown key, legacy file without `thermal_bounds`  | [`LoadError::ParseError`]  |
/// | Invalid `start_date` / duplicate `start_date`                    | [`LoadError::SchemaError`] |
/// | Non-finite or non-positive `duration_hours`                      | [`LoadError::SchemaError`] |
/// | Non-finite cost/bound, `min_mw > max_mw`                         | [`LoadError::SchemaError`] |
/// | `post_study_stage_index` out of range                            | [`LoadError::SchemaError`] |
/// | Duplicate `(thermal_id, post_study_stage_index)`                 | [`LoadError::SchemaError`] |
pub(crate) fn parse_post_study_file(
    path: &Path,
) -> Result<(PostStudyFormat, Option<PostStudyStages>), LoadError> {
    let raw_text = std::fs::read_to_string(path).map_err(|e| LoadError::io(path, e))?;

    let raw: RawPostStudyStagesFile =
        serde_json::from_str(&raw_text).map_err(|e| LoadError::parse(path, e.to_string()))?;

    if let Some(legacy_stages) = &raw.stages {
        if raw.thermal_stages.is_some() || raw.water_bucket_stages.is_some() {
            return Ok((PostStudyFormat::Mixed, None));
        }
        let legacy_bounds = raw
            .thermal_bounds
            .as_deref()
            .ok_or_else(|| LoadError::parse(path, "missing field `thermal_bounds`"))?;
        let thermal_stages = convert_stages(legacy_stages, "stages", path)?;
        let thermal_bounds = convert_thermal_bounds(legacy_bounds, thermal_stages.len(), path)?;
        return Ok((
            PostStudyFormat::Legacy,
            Some(PostStudyStages {
                thermal_stages,
                thermal_bounds,
                water_bucket_stages: WaterBucketStages::FromThermalStages,
            }),
        ));
    }

    let format = PostStudyFormat::Current {
        thermal_stages: raw.thermal_stages.is_some(),
        thermal_bounds: raw.thermal_bounds.is_some(),
        water_bucket_stages: raw.water_bucket_stages.is_some(),
    };
    if raw.thermal_stages.is_some() != raw.thermal_bounds.is_some() {
        return Ok((format, None));
    }
    let thermal_stages = convert_stages(
        raw.thermal_stages.as_deref().unwrap_or_default(),
        "thermal_stages",
        path,
    )?;
    let thermal_bounds = convert_thermal_bounds(
        raw.thermal_bounds.as_deref().unwrap_or_default(),
        thermal_stages.len(),
        path,
    )?;
    let water_bucket_stages = convert_stages(
        raw.water_bucket_stages.as_deref().unwrap_or_default(),
        "water_bucket_stages",
        path,
    )?;
    Ok((
        format,
        Some(PostStudyStages {
            thermal_stages,
            thermal_bounds,
            water_bucket_stages: WaterBucketStages::Declared(water_bucket_stages),
        }),
    ))
}

// ── Conversion + validation ─────────────────────────────────────────────────────

/// Parse and validate the stages declared under `key`, returning them sorted
/// ascending by `start_date`.
fn convert_stages(
    entries: &[RawPostStudyStage],
    key: &str,
    path: &Path,
) -> Result<Vec<PostStudyStage>, LoadError> {
    let mut stages: Vec<PostStudyStage> = Vec::with_capacity(entries.len());
    let mut seen: HashSet<chrono::NaiveDate> = HashSet::new();
    for (i, entry) in entries.iter().enumerate() {
        let start_date =
            parse_iso_date(&format!("{key}[{i}].start_date"), &entry.start_date, path)?;
        if !entry.duration_hours.is_finite() || entry.duration_hours <= 0.0 {
            return Err(LoadError::SchemaError {
                path: path.to_path_buf(),
                field: format!("{key}[{i}].duration_hours"),
                message: format!(
                    "duration_hours must be a finite number > 0.0, got {}",
                    entry.duration_hours
                ),
            });
        }
        if !seen.insert(start_date) {
            return Err(LoadError::SchemaError {
                path: path.to_path_buf(),
                field: format!("{key}[{i}].start_date"),
                message: format!("duplicate start_date {start_date} in {key}"),
            });
        }
        stages.push(PostStudyStage {
            start_date,
            duration_hours: entry.duration_hours,
        });
    }
    stages.sort_by_key(|s| s.start_date);
    Ok(stages)
}

/// Parse and validate `thermal_bounds`, returning them sorted by
/// `(thermal_id, post_study_stage_index)`.
fn convert_thermal_bounds(
    entries: &[RawPostStudyThermalBound],
    num_stages: usize,
    path: &Path,
) -> Result<Vec<PostStudyThermalBound>, LoadError> {
    let mut bounds: Vec<PostStudyThermalBound> = Vec::with_capacity(entries.len());
    let mut seen: HashSet<(i32, usize)> = HashSet::new();
    for (i, entry) in entries.iter().enumerate() {
        for (value, name) in [
            (entry.cost_per_mwh, "cost_per_mwh"),
            (entry.min_mw, "min_mw"),
            (entry.max_mw, "max_mw"),
        ] {
            if !value.is_finite() {
                return Err(LoadError::SchemaError {
                    path: path.to_path_buf(),
                    field: format!("thermal_bounds[{i}].{name}"),
                    message: format!("{name} must be a finite number, got {value}"),
                });
            }
        }
        if entry.min_mw > entry.max_mw {
            return Err(LoadError::SchemaError {
                path: path.to_path_buf(),
                field: format!("thermal_bounds[{i}].max_mw"),
                message: format!(
                    "max_mw ({}) must be >= min_mw ({}) for thermal_id {}, \
                     post_study_stage_index {}",
                    entry.max_mw, entry.min_mw, entry.thermal_id, entry.post_study_stage_index
                ),
            });
        }
        if entry.post_study_stage_index >= num_stages {
            return Err(LoadError::SchemaError {
                path: path.to_path_buf(),
                field: format!("thermal_bounds[{i}].post_study_stage_index"),
                message: format!(
                    "post_study_stage_index {} is out of range for {num_stages} post-study \
                     stage(s)",
                    entry.post_study_stage_index
                ),
            });
        }
        if !seen.insert((entry.thermal_id, entry.post_study_stage_index)) {
            return Err(LoadError::SchemaError {
                path: path.to_path_buf(),
                field: format!("thermal_bounds[{i}]"),
                message: format!(
                    "duplicate (thermal_id {}, post_study_stage_index {}) in thermal_bounds",
                    entry.thermal_id, entry.post_study_stage_index
                ),
            });
        }
        bounds.push(PostStudyThermalBound {
            thermal_id: EntityId(entry.thermal_id),
            post_study_stage_index: entry.post_study_stage_index,
            cost_per_mwh: entry.cost_per_mwh,
            min_mw: entry.min_mw,
            max_mw: entry.max_mw,
        });
    }
    bounds.sort_by_key(|b| (b.thermal_id.0, b.post_study_stage_index));
    Ok(bounds)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::test_support::write_json;

    const VALID_JSON: &str = r#"{
      "$schema": "https://raw.githubusercontent.com/ons-ccee-epe/novomodelo/refs/heads/main/schemas/post_study_stages.schema.json",
      "stages": [
        { "start_date": "2026-11-01", "duration_hours": 720.0 },
        { "start_date": "2026-12-01", "duration_hours": 744.0 }
      ],
      "thermal_bounds": [
        { "thermal_id": 86, "post_study_stage_index": 1, "cost_per_mwh": 220.0, "min_mw": 0.0, "max_mw": 300.0 },
        { "thermal_id": 86, "post_study_stage_index": 0, "cost_per_mwh": 210.0, "min_mw": 0.0, "max_mw": 350.0 }
      ]
    }"#;

    fn parse(json: &str) -> (PostStudyFormat, Option<PostStudyStages>) {
        parse_post_study_file(write_json(json).path()).unwrap()
    }

    fn parse_err(json: &str) -> LoadError {
        parse_post_study_file(write_json(json).path()).unwrap_err()
    }

    #[test]
    fn test_parse_valid_sorts_both_collections() {
        let (format, ps) = parse(VALID_JSON);
        let ps = ps.unwrap();
        assert_eq!(format, PostStudyFormat::Legacy);
        assert_eq!(ps.thermal_stages.len(), 2);
        assert_eq!(
            ps.thermal_stages[0].start_date,
            chrono::NaiveDate::from_ymd_opt(2026, 11, 1).unwrap()
        );
        // thermal_bounds sorted by (thermal_id, post_study_stage_index).
        assert_eq!(ps.thermal_bounds[0].post_study_stage_index, 0);
        assert_eq!(ps.thermal_bounds[1].post_study_stage_index, 1);
        assert_eq!(ps.water_bucket_stages, WaterBucketStages::FromThermalStages);
    }

    #[test]
    fn test_declaration_order_invariance() {
        let reversed = r#"{
          "stages": [
            { "start_date": "2026-12-01", "duration_hours": 744.0 },
            { "start_date": "2026-11-01", "duration_hours": 720.0 }
          ],
          "thermal_bounds": [
            { "thermal_id": 86, "post_study_stage_index": 1, "cost_per_mwh": 220.0, "min_mw": 0.0, "max_mw": 300.0 },
            { "thermal_id": 86, "post_study_stage_index": 0, "cost_per_mwh": 210.0, "min_mw": 0.0, "max_mw": 350.0 }
          ]
        }"#;
        assert_eq!(parse(VALID_JSON), parse(reversed));
    }

    #[test]
    fn test_non_positive_duration_rejected() {
        let json = r#"{ "stages": [{ "start_date": "2026-11-01", "duration_hours": 0.0 }], "thermal_bounds": [] }"#;
        match parse_err(json) {
            LoadError::SchemaError { field, .. } => assert!(field.contains("duration_hours")),
            other => panic!("expected SchemaError, got {other:?}"),
        }
    }

    #[test]
    fn test_inverted_bounds_rejected() {
        let json = r#"{
          "stages": [{ "start_date": "2026-11-01", "duration_hours": 720.0 }],
          "thermal_bounds": [{ "thermal_id": 1, "post_study_stage_index": 0, "cost_per_mwh": 1.0, "min_mw": 10.0, "max_mw": 5.0 }]
        }"#;
        match parse_err(json) {
            LoadError::SchemaError { message, .. } => assert!(message.contains("max_mw")),
            other => panic!("expected SchemaError, got {other:?}"),
        }
    }

    #[test]
    fn test_stage_index_out_of_range_rejected() {
        let json = r#"{
          "stages": [{ "start_date": "2026-11-01", "duration_hours": 720.0 }],
          "thermal_bounds": [{ "thermal_id": 1, "post_study_stage_index": 3, "cost_per_mwh": 1.0, "min_mw": 0.0, "max_mw": 5.0 }]
        }"#;
        match parse_err(json) {
            LoadError::SchemaError { field, .. } => {
                assert!(field.contains("post_study_stage_index"));
            }
            other => panic!("expected SchemaError, got {other:?}"),
        }
    }

    #[test]
    fn test_duplicate_cell_rejected() {
        let json = r#"{
          "stages": [{ "start_date": "2026-11-01", "duration_hours": 720.0 }],
          "thermal_bounds": [
            { "thermal_id": 1, "post_study_stage_index": 0, "cost_per_mwh": 1.0, "min_mw": 0.0, "max_mw": 5.0 },
            { "thermal_id": 1, "post_study_stage_index": 0, "cost_per_mwh": 2.0, "min_mw": 0.0, "max_mw": 6.0 }
          ]
        }"#;
        match parse_err(json) {
            LoadError::SchemaError { message, .. } => assert!(message.contains("duplicate")),
            other => panic!("expected SchemaError, got {other:?}"),
        }
    }

    #[test]
    fn test_unknown_field_rejected() {
        let json = r#"{ "stages": [], "thermal_bounds": [], "extra": 1 }"#;
        assert!(matches!(parse_err(json), LoadError::ParseError { .. }));
    }

    #[test]
    fn test_legacy_without_thermal_bounds_rejected() {
        match parse_err(r#"{ "stages": [] }"#) {
            LoadError::ParseError { message, .. } => {
                assert!(message.contains("missing field `thermal_bounds`"));
            }
            other => panic!("expected ParseError, got {other:?}"),
        }
    }

    #[test]
    fn test_current_full_declares_water_bucket_stages() {
        let json = r#"{
          "thermal_stages": [{ "start_date": "2026-11-01", "duration_hours": 720.0 }],
          "thermal_bounds": [{ "thermal_id": 1, "post_study_stage_index": 0, "cost_per_mwh": 1.0, "min_mw": 0.0, "max_mw": 5.0 }],
          "water_bucket_stages": [
            { "start_date": "2026-12-01", "duration_hours": 744.0 },
            { "start_date": "2026-11-01", "duration_hours": 720.0 }
          ]
        }"#;
        let (format, ps) = parse(json);
        let ps = ps.unwrap();
        assert_eq!(
            format,
            PostStudyFormat::Current {
                thermal_stages: true,
                thermal_bounds: true,
                water_bucket_stages: true,
            }
        );
        assert_eq!(ps.thermal_stages.len(), 1);
        assert_eq!(ps.thermal_bounds.len(), 1);
        let WaterBucketStages::Declared(water) = ps.water_bucket_stages else {
            panic!("expected Declared");
        };
        assert_eq!(
            water.iter().map(|s| s.duration_hours).collect::<Vec<_>>(),
            [720.0, 744.0]
        );
    }

    #[test]
    fn test_current_water_only_leaves_thermal_side_empty() {
        let json = r#"{ "water_bucket_stages": [{ "start_date": "2026-11-01", "duration_hours": 720.0 }] }"#;
        let (format, ps) = parse(json);
        let ps = ps.unwrap();
        assert_eq!(
            format,
            PostStudyFormat::Current {
                thermal_stages: false,
                thermal_bounds: false,
                water_bucket_stages: true,
            }
        );
        assert!(ps.thermal_stages.is_empty());
        assert!(ps.thermal_bounds.is_empty());
        assert!(matches!(
            ps.water_bucket_stages,
            WaterBucketStages::Declared(w) if w.len() == 1
        ));
    }

    #[test]
    fn test_empty_object_is_current_with_everything_absent() {
        let (format, ps) = parse("{}");
        assert_eq!(
            format,
            PostStudyFormat::Current {
                thermal_stages: false,
                thermal_bounds: false,
                water_bucket_stages: false,
            }
        );
        assert_eq!(
            ps.unwrap(),
            PostStudyStages {
                thermal_stages: vec![],
                thermal_bounds: vec![],
                water_bucket_stages: WaterBucketStages::Declared(vec![]),
            }
        );
    }

    #[test]
    fn test_empty_lists_count_as_present() {
        let json = r#"{ "thermal_stages": [], "thermal_bounds": [], "water_bucket_stages": [] }"#;
        let (format, ps) = parse(json);
        assert_eq!(
            format,
            PostStudyFormat::Current {
                thermal_stages: true,
                thermal_bounds: true,
                water_bucket_stages: true,
            }
        );
        assert!(ps.is_some());
    }

    #[test]
    fn test_stages_with_a_new_key_is_mixed_without_carrier() {
        for json in [
            r#"{ "stages": [], "thermal_bounds": [], "thermal_stages": [] }"#,
            r#"{ "stages": [], "thermal_bounds": [], "water_bucket_stages": [] }"#,
            r#"{ "stages": [], "thermal_stages": [], "water_bucket_stages": [] }"#,
        ] {
            assert_eq!(parse(json), (PostStudyFormat::Mixed, None), "{json}");
        }
    }

    #[test]
    fn test_incomplete_thermal_pair_is_current_without_carrier() {
        for (json, thermal_stages, thermal_bounds) in [
            (
                r#"{ "thermal_stages": [{ "start_date": "2026-11-01", "duration_hours": -1.0 }] }"#,
                true,
                false,
            ),
            (
                r#"{ "thermal_bounds": [{ "thermal_id": 1, "post_study_stage_index": 0, "cost_per_mwh": 1.0, "min_mw": 0.0, "max_mw": 5.0 }] }"#,
                false,
                true,
            ),
        ] {
            assert_eq!(
                parse(json),
                (
                    PostStudyFormat::Current {
                        thermal_stages,
                        thermal_bounds,
                        water_bucket_stages: false,
                    },
                    None
                ),
                "{json}"
            );
        }
    }

    #[test]
    fn test_diagnostics_name_the_key_the_user_wrote() {
        let thermal = r#"{
          "thermal_stages": [{ "start_date": "2026-11-01", "duration_hours": -1.0 }],
          "thermal_bounds": []
        }"#;
        match parse_err(thermal) {
            LoadError::SchemaError { field, .. } => {
                assert_eq!(field, "thermal_stages[0].duration_hours");
            }
            other => panic!("expected SchemaError, got {other:?}"),
        }
        let water = r#"{ "water_bucket_stages": [
          { "start_date": "2026-11-01", "duration_hours": 720.0 },
          { "start_date": "2026-11-01", "duration_hours": 744.0 }
        ] }"#;
        match parse_err(water) {
            LoadError::SchemaError { field, message, .. } => {
                assert_eq!(field, "water_bucket_stages[1].start_date");
                assert!(message.contains("in water_bucket_stages"));
            }
            other => panic!("expected SchemaError, got {other:?}"),
        }
    }
}

#[cfg(all(test, feature = "schema"))]
mod schema_tests {
    use schemars::schema_for;
    use serde_json::json;

    use super::RawPostStudyStagesFile;

    #[test]
    fn post_study_schema_encodes_both_formats() {
        let schema = schema_for!(RawPostStudyStagesFile).to_value();

        let branches = &schema["oneOf"];
        assert_eq!(branches.as_array().map(Vec::len), Some(2));
        assert_eq!(branches[0]["required"], json!(["stages", "thermal_bounds"]));
        assert_eq!(
            branches[0]["not"],
            json!({ "anyOf": [
                { "required": ["thermal_stages"] },
                { "required": ["water_bucket_stages"] },
            ] })
        );
        assert_eq!(
            branches[1]["not"],
            json!({ "anyOf": [{ "required": ["stages"] }] })
        );
        assert_eq!(
            branches[1]["dependentRequired"],
            json!({
                "thermal_stages": ["thermal_bounds"],
                "thermal_bounds": ["thermal_stages"],
            })
        );

        assert_eq!(schema["properties"]["stages"]["deprecated"], json!(true));

        let stage_item = json!({ "$ref": "#/$defs/RawPostStudyStage" });
        for key in ["stages", "thermal_stages", "water_bucket_stages"] {
            assert_eq!(schema["properties"][key]["items"], stage_item, "{key}");
        }
        assert!(schema["$defs"]["RawPostStudyStage"].is_object());
    }
}
