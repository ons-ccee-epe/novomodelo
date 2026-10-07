//! Parsing for `scenarios/noise_openings.parquet` — user-supplied noise
//! realisations for the opening scenario tree.
//!
//! ## Parquet schema
//!
//! | Column           | Type    | Required | Description                                  |
//! | ---------------- | ------- | -------- | -------------------------------------------- |
//! | `stage_id`       | INT32   | Yes      | Declared study-stage id (not a 0-based index)|
//! | `opening_index`  | UINT32  | Yes      | Opening index within the stage (0-based)     |
//! | `entity_index`   | UINT32  | Yes      | Entity index within the noise vector (0-based)|
//! | `value`          | DOUBLE  | Yes      | Noise realisation value                      |
//!
//! Rows are sorted by `(stage_id, opening_index, entity_index)` ascending to match
//! the stage-major, row-major layout required by [`OpeningTree::from_parts`].

use std::path::PathBuf;

use cobre_stochastic::OpeningTree;
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use crate::LoadError;
use crate::StageIdResolver;
use crate::parquet_helpers::{
    extract_required_float64, extract_required_int32, extract_required_uint32,
    open_record_batch_reader,
};

/// A single row from `scenarios/noise_openings.parquet`.
///
/// # Examples
///
/// ```
/// use cobre_io::scenarios::NoiseOpeningRow;
///
/// let row = NoiseOpeningRow {
///     stage_id: 0,
///     opening_index: 1,
///     entity_index: 2,
///     value: -0.5,
/// };
/// assert_eq!(row.stage_id, 0);
/// assert_eq!(row.opening_index, 1);
/// assert_eq!(row.entity_index, 2);
/// assert!((row.value - (-0.5)).abs() < 1e-15);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct NoiseOpeningRow {
    /// Declared study-stage id this row applies to (not a 0-based index).
    pub stage_id: i32,
    /// Opening index within the stage (0-based).
    pub opening_index: u32,
    /// Entity index within the noise vector (0-based).
    pub entity_index: u32,
    /// Noise realisation value.
    pub value: f64,
}

/// Parse `scenarios/noise_openings.parquet` and return rows sorted by
/// `(stage_id, opening_index, entity_index)` ascending. Cross-dimensional
/// validation is deferred to [`validate_noise_openings`].
///
/// # Errors
///
/// | Condition                                     | Error variant              |
/// |---------------------------------------------- |--------------------------- |
/// | File not found or permission denied           | [`LoadError::IoError`]     |
/// | Malformed Parquet (corrupt header, etc.)      | [`LoadError::ParseError`]  |
/// | Required column missing or wrong type         | [`LoadError::SchemaError`] |
///
/// # Examples
///
/// ```no_run
/// use cobre_io::scenarios::parse_noise_openings;
/// use std::path::Path;
///
/// let rows = parse_noise_openings(Path::new("scenarios/noise_openings.parquet"))
///     .expect("valid noise openings file");
/// println!("loaded {} noise opening rows", rows.len());
/// ```
pub fn parse_noise_openings(path: &Path) -> Result<Vec<NoiseOpeningRow>, LoadError> {
    let reader = open_record_batch_reader(path)?;

    let mut rows: Vec<NoiseOpeningRow> = Vec::new();

    for batch_result in reader {
        let batch = batch_result.map_err(|e| LoadError::parse(path, e.to_string()))?;

        let stage_id_col = extract_required_int32(&batch, "stage_id", path)?;
        let opening_index_col = extract_required_uint32(&batch, "opening_index", path)?;
        let entity_index_col = extract_required_uint32(&batch, "entity_index", path)?;
        let value_col = extract_required_float64(&batch, "value", path)?;

        let n = batch.num_rows();
        rows.reserve(n);

        for i in 0..n {
            rows.push(NoiseOpeningRow {
                stage_id: stage_id_col.value(i),
                opening_index: opening_index_col.value(i),
                entity_index: entity_index_col.value(i),
                value: value_col.value(i),
            });
        }
    }

    rows.sort_by(|a, b| {
        a.stage_id
            .cmp(&b.stage_id)
            .then_with(|| a.opening_index.cmp(&b.opening_index))
            .then_with(|| a.entity_index.cmp(&b.entity_index))
    });

    Ok(rows)
}

/// Validate parsed noise opening rows against expected system dimensions,
/// keying every stage check on the resolved study index (`resolver`) so a deck
/// numbered `1..T` validates identically to its `0..T-1` equivalent.
///
/// Assumes `rows` is already sorted by `(stage_id, opening_index, entity_index)`
/// as produced by [`parse_noise_openings`]. `expected_openings_per_stage` is the
/// declared opening count per study stage, indexed by resolved study index: each
/// stage's loaded distinct-opening count must equal its declared count (the cross
/// check against the declaration), and those indices must form `0..count`.
///
/// # Errors
///
/// | Condition                                                            | Error variant              |
/// |----------------------------------------------------------------------|----------------------------|
/// | A row `stage_id` resolves to no declared study stage                 | [`LoadError::SchemaError`] |
/// | Distinct entity count != `expected_dim`                              | [`LoadError::SchemaError`] |
/// | Distinct resolved stage count != `expected_openings_per_stage.len()` | [`LoadError::SchemaError`] |
/// | A stage's loaded opening count != its declared count                 | [`LoadError::SchemaError`] |
/// | A stage's opening indices are not `0..count`                         | [`LoadError::SchemaError`] |
///
/// Returns [`LoadError::SchemaError`] if any opening count exceeds `u32::MAX`.
///
/// # Examples
///
/// ```
/// use cobre_io::scenarios::{NoiseOpeningRow, validate_noise_openings};
/// use cobre_io::StageIdResolver;
///
/// // 2 stages, 3 openings each, dim=2 → 12 rows
/// let rows: Vec<NoiseOpeningRow> = (0..2_i32)
///     .flat_map(|s| (0..3_u32).flat_map(move |o| (0..2_u32).map(move |e| NoiseOpeningRow {
///         stage_id: s, opening_index: o, entity_index: e, value: 0.0,
///     })))
///     .collect();
///
/// let resolver = StageIdResolver::from_study_stage_ids(&[0, 1]);
/// validate_noise_openings(&rows, 2, &[3, 3], &resolver).expect("valid dimensions");
/// ```
pub fn validate_noise_openings(
    rows: &[NoiseOpeningRow],
    expected_dim: usize,
    expected_openings_per_stage: &[usize],
    resolver: &StageIdResolver,
) -> Result<(), LoadError> {
    for (i, row) in rows.iter().enumerate() {
        if resolver.resolve(row.stage_id).is_none() {
            return Err(resolver.unresolved_stage_id_error(
                "scenarios/noise_openings.parquet",
                format!("noise_openings[{i}].stage_id"),
                row.stage_id,
            ));
        }
    }

    let distinct_entities: BTreeSet<u32> = rows.iter().map(|r| r.entity_index).collect();
    let actual_dim = distinct_entities.len();
    if actual_dim != expected_dim {
        return Err(LoadError::SchemaError {
            path: PathBuf::from("scenarios/noise_openings.parquet"),
            field: "entity_index".to_string(),
            message: format!(
                "dimension mismatch: expected {expected_dim} entities, found {actual_dim}"
            ),
        });
    }

    let distinct_stages: BTreeSet<usize> = rows
        .iter()
        .filter_map(|r| resolver.resolve(r.stage_id))
        .collect();
    let actual_stages = distinct_stages.len();
    let expected_stages = expected_openings_per_stage.len();
    if actual_stages != expected_stages {
        return Err(LoadError::SchemaError {
            path: PathBuf::from("scenarios/noise_openings.parquet"),
            field: "stage_id".to_string(),
            message: format!(
                "stage count mismatch: expected {expected_stages} stages, found {actual_stages}"
            ),
        });
    }

    let mut openings_by_index: BTreeMap<usize, BTreeSet<u32>> = BTreeMap::new();
    for row in rows {
        if let Some(idx) = resolver.resolve(row.stage_id) {
            openings_by_index
                .entry(idx)
                .or_default()
                .insert(row.opening_index);
        }
    }

    for (&idx, opening_set) in &openings_by_index {
        let count = opening_set.len();
        let declared = expected_openings_per_stage.get(idx).copied().unwrap_or(0);
        if count != declared {
            let stage_id = resolver.id_at(idx).unwrap_or_default();
            return Err(LoadError::SchemaError {
                path: PathBuf::from("scenarios/noise_openings.parquet"),
                field: "opening_index".to_string(),
                message: format!(
                    "opening count mismatch for stage {stage_id}: declared {declared} openings, \
                     found {count}"
                ),
            });
        }
        let expected_max = u32::try_from(count).map_err(|_| LoadError::SchemaError {
            path: PathBuf::from("scenarios/noise_openings.parquet"),
            field: String::new(),
            message: format!("opening count {count} exceeds u32::MAX"),
        })?;
        let expected_set: BTreeSet<u32> = (0..expected_max).collect();
        if *opening_set != expected_set {
            let stage_id = resolver.id_at(idx).unwrap_or_default();
            return Err(LoadError::SchemaError {
                path: PathBuf::from("scenarios/noise_openings.parquet"),
                field: "opening_index".to_string(),
                message: format!("missing opening indices for stage {stage_id}"),
            });
        }
    }

    Ok(())
}

/// Assemble an [`OpeningTree`] from validated, sorted noise opening rows, laid
/// out by resolved study index (`resolver`) rather than by sorted-`stage_id`
/// position, so a deck numbered `1..T` assembles the same tree as its `0..T-1`
/// equivalent.
///
/// `rows` must be sorted by `(stage_id, opening_index, entity_index)` ascending —
/// the layout produced by [`parse_noise_openings`] — and must have already passed
/// [`validate_noise_openings`] (in particular every `stage_id` resolves). The
/// sort order matches the stage-major, row-major memory layout required by
/// [`OpeningTree::from_parts`].
///
/// `dim` is the number of entities per opening vector (the noise dimension).
///
/// # Panics
///
/// Panics if `rows.len()` is not consistent with the implied
/// `sum(openings_per_stage) * dim` (delegated to [`OpeningTree::from_parts`]).
///
/// # Examples
///
/// ```
/// use cobre_io::scenarios::{NoiseOpeningRow, assemble_opening_tree};
/// use cobre_io::StageIdResolver;
///
/// // 2 stages, 3 openings each, dim=2 → 12 rows
/// let rows: Vec<NoiseOpeningRow> = (0..2_i32)
///     .flat_map(|s| (0..3_u32).flat_map(move |o| (0..2_u32).map(move |e| NoiseOpeningRow {
///         stage_id: s, opening_index: o, entity_index: e, value: f64::from(s * 6 + o as i32 * 2 + e as i32),
///     })))
///     .collect();
///
/// let resolver = StageIdResolver::from_study_stage_ids(&[0, 1]);
/// let tree = assemble_opening_tree(rows, 2, &resolver);
/// assert_eq!(tree.n_stages(), 2);
/// assert_eq!(tree.n_openings(0), 3);
/// assert_eq!(tree.dim(), 2);
/// ```
#[must_use]
pub fn assemble_opening_tree(
    rows: Vec<NoiseOpeningRow>,
    dim: usize,
    resolver: &StageIdResolver,
) -> OpeningTree {
    let mut openings_per_stage: Vec<usize> = Vec::new();
    let mut current_index: Option<usize> = None;
    let mut current_opening_count: usize = 0;
    let mut last_opening: Option<u32> = None;

    for row in &rows {
        let study_index = resolver.resolve(row.stage_id);
        if current_index != study_index {
            if current_index.is_some() {
                openings_per_stage.push(current_opening_count);
            }
            current_index = study_index;
            current_opening_count = 1;
            last_opening = Some(row.opening_index);
        } else if Some(row.opening_index) != last_opening {
            current_opening_count += 1;
            last_opening = Some(row.opening_index);
        }
    }
    if current_index.is_some() {
        openings_per_stage.push(current_opening_count);
    }

    let data: Vec<f64> = rows.into_iter().map(|r| r.value).collect();
    OpeningTree::from_parts(data, openings_per_stage, dim)
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(
    clippy::doc_markdown,
    clippy::expect_used,
    clippy::panic,
    clippy::too_many_lines,
    clippy::unwrap_used
)]
mod tests {
    use super::*;
    use crate::test_support::write_parquet;
    use arrow::array::{Float64Array, Int32Array, UInt32Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use std::sync::Arc;

    // ── Helpers ───────────────────────────────────────────────────────────────

    fn schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("stage_id", DataType::Int32, false),
            Field::new("opening_index", DataType::UInt32, false),
            Field::new("entity_index", DataType::UInt32, false),
            Field::new("value", DataType::Float64, false),
        ]))
    }

    fn make_batch(
        stage_ids: &[i32],
        opening_indices: &[u32],
        entity_indices: &[u32],
        values: &[f64],
    ) -> RecordBatch {
        RecordBatch::try_new(
            schema(),
            vec![
                Arc::new(Int32Array::from(stage_ids.to_vec())),
                Arc::new(UInt32Array::from(opening_indices.to_vec())),
                Arc::new(UInt32Array::from(entity_indices.to_vec())),
                Arc::new(Float64Array::from(values.to_vec())),
            ],
        )
        .expect("valid batch")
    }

    /// Build a complete, sorted row set for `n_stages` stages each with
    /// `openings` openings and `dim` entities. Values are sequential floats.
    fn make_rows(n_stages: usize, openings: usize, dim: usize) -> Vec<NoiseOpeningRow> {
        let mut rows = Vec::new();
        let mut v = 0.0_f64;
        for s in 0..n_stages {
            for o in 0..openings {
                for e in 0..dim {
                    rows.push(NoiseOpeningRow {
                        stage_id: i32::try_from(s).unwrap(),
                        opening_index: u32::try_from(o).unwrap(),
                        entity_index: u32::try_from(e).unwrap(),
                        value: v,
                    });
                    v += 1.0;
                }
            }
        }
        rows
    }

    // ── parse_valid_file_returns_sorted_rows ──────────────────────────────────

    #[test]
    fn parse_valid_file_returns_sorted_rows() {
        let batch = make_batch(
            &[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            &[2, 2, 1, 1, 0, 0, 2, 2, 1, 1, 0, 0],
            &[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            &[11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0],
        );
        let tmp = write_parquet(&batch);
        let rows = parse_noise_openings(tmp.path()).unwrap();

        assert_eq!(rows.len(), 12, "expected 12 rows");

        for w in rows.windows(2) {
            let a = &w[0];
            let b = &w[1];
            let cmp = a
                .stage_id
                .cmp(&b.stage_id)
                .then_with(|| a.opening_index.cmp(&b.opening_index))
                .then_with(|| a.entity_index.cmp(&b.entity_index));
            assert!(
                cmp != std::cmp::Ordering::Greater,
                "rows not sorted: {a:?} > {b:?}"
            );
        }

        assert_eq!(rows[0].stage_id, 0);
        assert_eq!(rows[0].opening_index, 0);
        assert_eq!(rows[0].entity_index, 0);
        assert!((rows[0].value - 0.0).abs() < 1e-15);
    }

    // ── parse_missing_column_returns_schema_error ─────────────────────────────

    #[test]
    fn parse_missing_column_returns_schema_error() {
        let schema_no_value = Arc::new(Schema::new(vec![
            Field::new("stage_id", DataType::Int32, false),
            Field::new("opening_index", DataType::UInt32, false),
            Field::new("entity_index", DataType::UInt32, false),
        ]));
        let batch = RecordBatch::try_new(
            schema_no_value,
            vec![
                Arc::new(Int32Array::from(vec![0_i32])),
                Arc::new(UInt32Array::from(vec![0_u32])),
                Arc::new(UInt32Array::from(vec![0_u32])),
            ],
        )
        .unwrap();
        let tmp = write_parquet(&batch);
        let err = parse_noise_openings(tmp.path()).unwrap_err();

        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(
                    field.contains("value"),
                    "field should contain 'value', got: {field}"
                );
                assert!(
                    message.contains("missing required column"),
                    "message should mention missing column, got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    // ── validate_correct_dimensions_returns_ok ────────────────────────────────

    #[test]
    fn validate_correct_dimensions_returns_ok() {
        let rows = make_rows(2, 3, 2);
        let resolver = StageIdResolver::from_study_stage_ids(&[0, 1]);
        validate_noise_openings(&rows, 2, &[3, 3], &resolver).unwrap();
    }

    // ── validate_dimension_mismatch_returns_error ─────────────────────────────

    #[test]
    fn validate_dimension_mismatch_returns_error() {
        let rows = make_rows(2, 3, 3);
        let resolver = StageIdResolver::from_study_stage_ids(&[0, 1]);
        let err = validate_noise_openings(&rows, 2, &[3, 3], &resolver).unwrap_err();

        match &err {
            LoadError::SchemaError { message, .. } => {
                assert!(
                    message.contains("dimension mismatch"),
                    "message should contain 'dimension mismatch', got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    // ── validate_stage_count_mismatch_returns_error ───────────────────────────

    #[test]
    fn validate_stage_count_mismatch_returns_error() {
        let rows = make_rows(2, 3, 2);
        // Study declares 3 stages, but the deck only covers 2.
        let resolver = StageIdResolver::from_study_stage_ids(&[0, 1, 2]);
        let err = validate_noise_openings(&rows, 2, &[3, 3, 3], &resolver).unwrap_err();

        match &err {
            LoadError::SchemaError { message, .. } => {
                assert!(
                    message.contains("stage count mismatch"),
                    "message should contain 'stage count mismatch', got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    // ── validate_missing_openings_returns_error ───────────────────────────────

    #[test]
    fn validate_missing_openings_returns_error() {
        // openings 0 and 2 only (index 1 missing), dim=2 → 4 rows.
        let rows: Vec<NoiseOpeningRow> = [0u32, 2u32]
            .iter()
            .flat_map(|&o| {
                [0u32, 1u32].iter().map(move |&e| NoiseOpeningRow {
                    stage_id: 0,
                    opening_index: o,
                    entity_index: e,
                    value: 0.0,
                })
            })
            .collect();

        let resolver = StageIdResolver::from_study_stage_ids(&[0]);
        // Declared count matches the loaded distinct-opening count (2), so the
        // count cross-check passes and the non-contiguous set trips contiguity.
        let err = validate_noise_openings(&rows, 2, &[2], &resolver).unwrap_err();

        match &err {
            LoadError::SchemaError { message, .. } => {
                assert!(
                    message.contains("missing opening indices"),
                    "message should contain 'missing opening indices', got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    // ── validate_count_mismatch_vs_declared_returns_error ─────────────────────

    /// A stage whose loaded distinct-opening count differs from its declared
    /// `num_openings` is rejected naming the stage, the declared count, and the
    /// found count — the I8 cross-check against the declaration (not a
    /// self-derived count).
    #[test]
    fn validate_count_mismatch_vs_declared_returns_error() {
        // 2 stages, 3 openings each in the file; stage 1 declares only 2.
        let rows = make_rows(2, 3, 2);
        let resolver = StageIdResolver::from_study_stage_ids(&[0, 1]);
        let err = validate_noise_openings(&rows, 2, &[3, 2], &resolver).unwrap_err();

        match &err {
            LoadError::SchemaError { message, .. } => {
                assert!(
                    message.contains("opening count mismatch")
                        && message.contains("stage 1")
                        && message.contains("declared 2")
                        && message.contains("found 3"),
                    "message should name stage/declared/found, got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    // ── assemble_produces_correct_opening_tree ────────────────────────────────

    #[test]
    fn assemble_produces_correct_opening_tree() {
        let rows = make_rows(2, 3, 2);
        let expected: Vec<f64> = rows.iter().map(|r| r.value).collect();

        let resolver = StageIdResolver::from_study_stage_ids(&[0, 1]);
        let tree = assemble_opening_tree(rows, 2, &resolver);

        assert_eq!(tree.n_stages(), 2);
        assert_eq!(tree.n_openings(0), 3);
        assert_eq!(tree.n_openings(1), 3);
        assert_eq!(tree.dim(), 2);

        assert_eq!(tree.data(), expected.as_slice());

        // make_rows emits sequential stage-major values:
        // Stage 0, opening 0: values 0.0, 1.0
        assert_eq!(tree.opening(0, 0), &[0.0_f64, 1.0]);
        // Stage 0, opening 2: values 4.0, 5.0
        assert_eq!(tree.opening(0, 2), &[4.0_f64, 5.0]);
        // Stage 1, opening 0: values 6.0, 7.0
        assert_eq!(tree.opening(1, 0), &[6.0_f64, 7.0]);
        // Stage 1, opening 2: values 10.0, 11.0
        assert_eq!(tree.opening(1, 2), &[10.0_f64, 11.0]);
    }

    // ── noise_openings_rekey_matches_zero_based_equivalent ────────────────────

    /// A `1..T` deck and its `0..T-1` equivalent (identical opening/entity/value
    /// content, stage labels shifted by one) assemble a byte-identical
    /// `OpeningTree` once each is keyed through its own resolver.
    #[test]
    fn noise_openings_rekey_matches_zero_based_equivalent() {
        let zero_rows = make_rows(4, 3, 2);
        let zero_resolver = StageIdResolver::from_study_stage_ids(&[0, 1, 2, 3]);
        let zero_tree = assemble_opening_tree(zero_rows, 2, &zero_resolver);

        let mut one_rows = make_rows(4, 3, 2);
        for row in &mut one_rows {
            row.stage_id += 1;
        }
        let one_resolver = StageIdResolver::from_study_stage_ids(&[1, 2, 3, 4]);
        let one_tree = assemble_opening_tree(one_rows, 2, &one_resolver);

        assert_eq!(one_tree.n_stages(), zero_tree.n_stages());
        for s in 0..zero_tree.n_stages() {
            assert_eq!(one_tree.n_openings(s), zero_tree.n_openings(s));
        }
        assert_eq!(one_tree.dim(), zero_tree.dim());
        assert_eq!(one_tree.data(), zero_tree.data());
    }

    // ── noise_openings_unresolved_stage_id_rejected ───────────────────────────

    /// A row whose `stage_id` matches no declared study stage fails validation
    /// with the shared error naming the file, the offending value, and the
    /// declared study-stage-id set.
    #[test]
    fn noise_openings_unresolved_stage_id_rejected() {
        let mut rows = make_rows(2, 3, 2);
        rows.push(NoiseOpeningRow {
            stage_id: 5,
            opening_index: 0,
            entity_index: 0,
            value: 0.0,
        });
        rows.push(NoiseOpeningRow {
            stage_id: 5,
            opening_index: 0,
            entity_index: 1,
            value: 0.0,
        });
        rows.sort_by(|a, b| {
            a.stage_id
                .cmp(&b.stage_id)
                .then_with(|| a.opening_index.cmp(&b.opening_index))
                .then_with(|| a.entity_index.cmp(&b.entity_index))
        });

        let resolver = StageIdResolver::from_study_stage_ids(&[0, 1]);
        let err = validate_noise_openings(&rows, 2, &[3, 3], &resolver).unwrap_err();

        match &err {
            LoadError::SchemaError { path, message, .. } => {
                assert!(
                    path.to_string_lossy()
                        .contains("scenarios/noise_openings.parquet"),
                    "path names the file, got: {path:?}"
                );
                assert!(
                    message.contains('5'),
                    "message names the offending value, got: {message}"
                );
                assert!(
                    message.contains("[0, 1]"),
                    "message names the declared set, got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }
}
