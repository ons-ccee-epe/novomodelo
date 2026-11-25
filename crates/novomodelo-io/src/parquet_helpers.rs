//! Centralises the shared input-Parquet opening, column-set check and typed
//! column-downcast logic used by every Parquet parser in `novomodelo-io`.

use arrow::array::{
    Array, ArrayRef, Date32Array, Float64Array, Int32Array, StringArray, UInt32Array,
};
use arrow::datatypes::DataType;
use arrow::record_batch::RecordBatch;
use parquet::arrow::arrow_reader::{ParquetRecordBatchReader, ParquetRecordBatchReaderBuilder};
use std::fs::File;
use std::path::Path;

use crate::LoadError;

/// Compression codecs an input Parquet file may use; an uncompressed file is
/// also read.
pub const INPUT_PARQUET_CODECS: [&str; 4] = ["zstd", "snappy", "gzip", "lz4"];

/// The columns one input Parquet file may carry: [`open_parquet_input`] refuses
/// any other, and [`InputBatch`] reads only these.
pub(crate) struct ParquetColumns {
    /// Every column the parser reads, required or optional.
    pub(crate) known: &'static [&'static str],
    /// Columns an earlier layout carried, each with the instruction its refusal
    /// gives in place of a closest-name hint.
    pub(crate) retired: &'static [(&'static str, &'static str)],
}

/// Single owner of `novomodelo-io`'s parquet open/build error mappings and of
/// the unknown-column refusal, which runs on the file schema before any row is
/// read.
pub(crate) fn open_parquet_input<'p>(
    path: &'p Path,
    columns: &'static ParquetColumns,
) -> Result<ParquetInput<'p>, LoadError> {
    let file = File::open(path).map_err(|e| LoadError::io(path, e))?;

    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| LoadError::parse(path, e.to_string()))?;

    if let Some(field) = builder
        .schema()
        .fields()
        .iter()
        .find(|field| !columns.known.contains(&field.name().as_str()))
    {
        return Err(unknown_column_error(field.name(), columns, path));
    }

    let reader = builder
        .build()
        .map_err(|e| LoadError::parse(path, e.to_string()))?;
    Ok(ParquetInput {
        reader,
        path,
        columns,
    })
}

/// The record batches of an input Parquet file whose columns passed the check.
pub(crate) struct ParquetInput<'p> {
    reader: ParquetRecordBatchReader,
    path: &'p Path,
    columns: &'static ParquetColumns,
}

impl<'p> Iterator for ParquetInput<'p> {
    type Item = Result<InputBatch<'p>, LoadError>;

    fn next(&mut self) -> Option<Self::Item> {
        let batch = self.reader.next()?;
        Some(
            batch
                .map(|batch| InputBatch {
                    batch,
                    path: self.path,
                    columns: self.columns,
                })
                .map_err(|e| LoadError::parse(self.path, e.to_string())),
        )
    }
}

/// One record batch of an input Parquet file, read by column name.
pub(crate) struct InputBatch<'p> {
    batch: RecordBatch,
    path: &'p Path,
    columns: &'static ParquetColumns,
}

impl InputBatch<'_> {
    pub(crate) fn num_rows(&self) -> usize {
        self.batch.num_rows()
    }

    pub(crate) fn has_column(&self, name: &str) -> bool {
        self.column(name).is_some()
    }

    pub(crate) fn required_int32(&self, name: &str) -> Result<&Int32Array, LoadError> {
        self.required(name, "Int32")
    }

    pub(crate) fn required_uint32(&self, name: &str) -> Result<&UInt32Array, LoadError> {
        self.required(name, "UInt32")
    }

    pub(crate) fn required_float64(&self, name: &str) -> Result<&Float64Array, LoadError> {
        self.required(name, "Float64")
    }

    pub(crate) fn required_date32(&self, name: &str) -> Result<&Date32Array, LoadError> {
        self.required(name, "Date32")
    }

    pub(crate) fn required_string(&self, name: &str) -> Result<&StringArray, LoadError> {
        self.required(name, "Utf8")
    }

    pub(crate) fn optional_int32(&self, name: &str) -> Result<Option<&Int32Array>, LoadError> {
        self.optional(name, "Int32")
    }

    pub(crate) fn optional_float64(&self, name: &str) -> Result<Option<&Float64Array>, LoadError> {
        self.optional(name, "Float64")
    }

    fn column(&self, name: &str) -> Option<&ArrayRef> {
        debug_assert!(
            self.columns.known.contains(&name),
            "column \"{name}\" is read but missing from its file's ParquetColumns"
        );
        self.batch.column_by_name(name)
    }

    fn required<T: Array + 'static>(&self, name: &str, expected: &str) -> Result<&T, LoadError> {
        let col = self
            .column(name)
            .ok_or_else(|| missing_column_error(name, self.path))?;
        downcast(col, name, self.path, expected)
    }

    fn optional<T: Array + 'static>(
        &self,
        name: &str,
        expected: &str,
    ) -> Result<Option<&T>, LoadError> {
        self.column(name)
            .map(|col| downcast(col, name, self.path, expected))
            .transpose()
    }
}

fn downcast<'a, T: Array + 'static>(
    col: &'a ArrayRef,
    name: &str,
    path: &Path,
    expected: &str,
) -> Result<&'a T, LoadError> {
    col.as_any()
        .downcast_ref::<T>()
        .ok_or_else(|| wrong_type_error(name, path, col.data_type(), expected))
}

fn unknown_column_error(name: &str, columns: &ParquetColumns, path: &Path) -> LoadError {
    let message = if let Some((_, instruction)) =
        columns.retired.iter().find(|(retired, _)| *retired == name)
    {
        format!("column \"{name}\" is no longer read; {instruction}")
    } else if let Some(closest) = closest_column(name, columns.known) {
        format!("unknown column \"{name}\"; did you mean \"{closest}\"?")
    } else {
        format!(
            "unknown column \"{name}\"; the columns this file may carry are {}",
            columns.known.join(", ")
        )
    };
    LoadError::SchemaError {
        path: path.to_path_buf(),
        field: name.to_string(),
        message,
    }
}

/// The known column nearest to `name` ignoring ASCII case, if its edit distance
/// is at most a third of the longer name; ties go to the first declared.
fn closest_column(name: &str, known: &[&'static str]) -> Option<&'static str> {
    let name = name.to_ascii_lowercase();
    known
        .iter()
        .map(|candidate| {
            let distance = edit_distance(&name, &candidate.to_ascii_lowercase());
            (distance, *candidate)
        })
        .filter(|(distance, candidate)| {
            *distance <= (name.chars().count().max(candidate.chars().count()) / 3).max(1)
        })
        .min_by_key(|(distance, _)| *distance)
        .map(|(_, candidate)| candidate)
}

/// Levenshtein distance, in characters.
fn edit_distance(a: &str, b: &str) -> usize {
    let b: Vec<char> = b.chars().collect();
    let mut previous: Vec<usize> = (0..=b.len()).collect();
    let mut current = vec![0; b.len() + 1];
    for (i, ca) in a.chars().enumerate() {
        current[0] = i + 1;
        for (j, cb) in b.iter().enumerate() {
            let substitution = previous[j] + usize::from(ca != *cb);
            current[j + 1] = substitution.min(previous[j + 1] + 1).min(current[j] + 1);
        }
        std::mem::swap(&mut previous, &mut current);
    }
    previous[b.len()]
}

fn missing_column_error(name: &str, path: &Path) -> LoadError {
    LoadError::SchemaError {
        path: path.to_path_buf(),
        field: name.to_string(),
        message: format!("missing required column \"{name}\""),
    }
}

fn wrong_type_error(name: &str, path: &Path, actual: &DataType, expected: &str) -> LoadError {
    LoadError::SchemaError {
        path: path.to_path_buf(),
        field: name.to_string(),
        message: format!("column \"{name}\" has type {actual} but {expected} is required"),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::test_support::{write_json, write_parquet, write_parquet_batches};
    use arrow::datatypes::{Field, Schema};
    use parquet::arrow::ArrowWriter;
    use parquet::basic::{Compression, GzipLevel, ZstdLevel};
    use parquet::file::properties::WriterProperties;
    use std::sync::Arc;
    use tempfile::NamedTempFile;

    static TEST_COLUMNS: ParquetColumns = ParquetColumns {
        known: &[
            "hydro_id",
            "stage_id",
            "mean_m3s",
            "count",
            "start_date",
            "source",
            "max_turbined_m3s",
        ],
        retired: &[("residual_std_ratio", "remove it")],
    };

    fn make_batch(fields: Vec<Field>, columns: Vec<ArrayRef>) -> RecordBatch {
        RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).unwrap()
    }

    fn input_batch(batch: RecordBatch) -> InputBatch<'static> {
        InputBatch {
            batch,
            path: Path::new("test.parquet"),
            columns: &TEST_COLUMNS,
        }
    }

    /// One single-row column of each type the accessors read.
    fn column_of(data_type: &DataType) -> ArrayRef {
        match data_type {
            DataType::Int32 => Arc::new(Int32Array::from(vec![7])),
            DataType::UInt32 => Arc::new(UInt32Array::from(vec![7_u32])),
            DataType::Float64 => Arc::new(Float64Array::from(vec![7.5])),
            DataType::Date32 => Arc::new(Date32Array::from(vec![7])),
            DataType::Utf8 => Arc::new(StringArray::from(vec!["seven"])),
            other => panic!("no fixture for {other}"),
        }
    }

    fn batch_with(name: &str, data_type: &DataType) -> InputBatch<'static> {
        input_batch(make_batch(
            vec![Field::new(name, data_type.clone(), false)],
            vec![column_of(data_type)],
        ))
    }

    fn schema_error(err: LoadError) -> (String, String) {
        match err {
            LoadError::SchemaError {
                path,
                field,
                message,
            } => {
                assert_eq!(path, Path::new("test.parquet"));
                (field, message)
            }
            other => panic!("expected SchemaError, got {other:?}"),
        }
    }

    /// Each accessor with its column type and the type its refusal names.
    type Accessor = fn(&InputBatch<'static>, &str) -> Result<Option<usize>, LoadError>;
    const ACCESSORS: &[(&str, DataType, &str, Accessor)] = &[
        ("required_int32", DataType::Int32, "Int32", |b, n| {
            b.required_int32(n).map(|a| Some(a.len()))
        }),
        ("required_uint32", DataType::UInt32, "UInt32", |b, n| {
            b.required_uint32(n).map(|a| Some(a.len()))
        }),
        ("required_float64", DataType::Float64, "Float64", |b, n| {
            b.required_float64(n).map(|a| Some(a.len()))
        }),
        ("required_date32", DataType::Date32, "Date32", |b, n| {
            b.required_date32(n).map(|a| Some(a.len()))
        }),
        ("required_string", DataType::Utf8, "Utf8", |b, n| {
            b.required_string(n).map(|a| Some(a.len()))
        }),
        ("optional_int32", DataType::Int32, "Int32", |b, n| {
            b.optional_int32(n).map(|a| a.map(Array::len))
        }),
        ("optional_float64", DataType::Float64, "Float64", |b, n| {
            b.optional_float64(n).map(|a| a.map(Array::len))
        }),
    ];

    #[test]
    fn accessors_read_a_present_column() {
        for (accessor, data_type, _, read) in ACCESSORS {
            let batch = batch_with("hydro_id", data_type);
            assert_eq!(read(&batch, "hydro_id").unwrap(), Some(1), "{accessor}");
        }
    }

    #[test]
    fn accessors_refuse_a_column_of_another_type() {
        for (accessor, data_type, expected, read) in ACCESSORS {
            let other = if *data_type == DataType::Utf8 {
                DataType::Int32
            } else {
                DataType::Utf8
            };
            let batch = batch_with("hydro_id", &other);
            let (field, message) = schema_error(read(&batch, "hydro_id").unwrap_err());
            assert_eq!(field, "hydro_id", "{accessor}");
            assert!(
                message.starts_with("column \"hydro_id\" has type "),
                "{accessor}: {message}"
            );
            assert!(
                message.ends_with(&format!(" but {expected} is required")),
                "{accessor}: {message}"
            );
        }
    }

    #[test]
    fn a_missing_column_is_refused_when_required_and_absent_when_optional() {
        for (accessor, data_type, _, read) in ACCESSORS {
            let batch = batch_with("stage_id", data_type);
            let result = read(&batch, "hydro_id");
            if accessor.starts_with("required") {
                let (field, message) = schema_error(result.unwrap_err());
                assert_eq!(field, "hydro_id", "{accessor}");
                assert_eq!(
                    message, "missing required column \"hydro_id\"",
                    "{accessor}"
                );
            } else {
                assert_eq!(result.unwrap(), None, "{accessor}");
            }
        }
    }

    #[test]
    fn has_column_reports_whether_the_batch_carries_a_declared_column() {
        let batch = batch_with("stage_id", &DataType::Int32);
        assert!(batch.has_column("stage_id"));
        assert!(!batch.has_column("hydro_id"));
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "column \"undeclared\" is read but missing")]
    fn reading_an_undeclared_column_panics() {
        let batch = batch_with("undeclared", &DataType::Int32);
        let _ = batch.required_int32("undeclared");
    }

    #[test]
    fn open_parquet_input_reads_every_batch() {
        let b1 = make_batch(
            vec![Field::new("hydro_id", DataType::Int32, false)],
            vec![Arc::new(Int32Array::from(vec![1, 2, 3]))],
        );
        let b2 = make_batch(
            vec![Field::new("hydro_id", DataType::Int32, false)],
            vec![Arc::new(Int32Array::from(vec![4, 5]))],
        );
        let total_rows = b1.num_rows() + b2.num_rows();
        let tmp = write_parquet_batches(&[b1, b2]);
        let mut rows = 0;
        for batch in open_parquet_input(tmp.path(), &TEST_COLUMNS).unwrap() {
            rows += batch.unwrap().num_rows();
        }
        assert_eq!(rows, total_rows);
    }

    #[test]
    fn open_parquet_input_errors_on_missing_file() {
        let path = Path::new("/nonexistent-dir/does-not-exist.parquet");
        let Err(LoadError::IoError { path: err_path, .. }) =
            open_parquet_input(path, &TEST_COLUMNS)
        else {
            panic!("expected IoError");
        };
        assert_eq!(err_path, path);
    }

    #[test]
    fn open_parquet_input_errors_on_non_parquet_file() {
        let tmp = write_json("{}");
        let Err(LoadError::ParseError { path: err_path, .. }) =
            open_parquet_input(tmp.path(), &TEST_COLUMNS)
        else {
            panic!("expected ParseError");
        };
        assert_eq!(err_path, tmp.path());
    }

    /// A file with `hydro_id` and `extra`, opened with [`TEST_COLUMNS`].
    fn open_with_extra(extra: &str) -> (String, String) {
        let batch = make_batch(
            vec![
                Field::new("hydro_id", DataType::Int32, false),
                Field::new(extra, DataType::Float64, true),
            ],
            vec![
                Arc::new(Int32Array::from(Vec::<i32>::new())),
                Arc::new(Float64Array::from(Vec::<f64>::new())),
            ],
        );
        let tmp = write_parquet(&batch);
        let Err(LoadError::SchemaError {
            path,
            field,
            message,
        }) = open_parquet_input(tmp.path(), &TEST_COLUMNS)
        else {
            panic!("a file with column {extra} must be refused at open");
        };
        assert_eq!(path, tmp.path());
        (field, message)
    }

    #[test]
    fn an_unknown_column_is_refused_with_the_closest_known_column() {
        let (field, message) = open_with_extra("max_turbined_m3");
        assert_eq!(field, "max_turbined_m3");
        assert_eq!(
            message,
            "unknown column \"max_turbined_m3\"; did you mean \"max_turbined_m3s\"?"
        );
    }

    #[test]
    fn the_closest_column_ignores_ascii_case() {
        let (_, message) = open_with_extra("Mean_M3S");
        assert_eq!(
            message,
            "unknown column \"Mean_M3S\"; did you mean \"mean_m3s\"?"
        );
    }

    #[test]
    fn an_unknown_column_far_from_every_known_one_lists_them() {
        let (field, message) = open_with_extra("spill");
        assert_eq!(field, "spill");
        assert_eq!(
            message,
            "unknown column \"spill\"; the columns this file may carry are hydro_id, \
             stage_id, mean_m3s, count, start_date, source, max_turbined_m3s"
        );
    }

    #[test]
    fn a_retired_column_is_refused_with_its_instruction() {
        let (field, message) = open_with_extra("residual_std_ratio");
        assert_eq!(field, "residual_std_ratio");
        assert_eq!(
            message,
            "column \"residual_std_ratio\" is no longer read; remove it"
        );
    }

    #[test]
    fn closest_column_takes_the_nearest_within_a_third_of_the_name() {
        let known = &["stage_id", "start_date", "max_storage_hm3"];
        assert_eq!(closest_column("stage", known), None);
        assert_eq!(closest_column("stageid", known), Some("stage_id"));
        assert_eq!(
            closest_column("max_storage", known),
            Some("max_storage_hm3")
        );
        assert_eq!(closest_column("x", &["y"]), Some("y"));
        assert_eq!(closest_column("xyz", &["abc"]), None);
    }

    #[test]
    fn edit_distance_counts_insertions_deletions_and_substitutions() {
        assert_eq!(edit_distance("", ""), 0);
        assert_eq!(edit_distance("abc", ""), 3);
        assert_eq!(edit_distance("", "abc"), 3);
        assert_eq!(edit_distance("kitten", "sitting"), 3);
        assert_eq!(edit_distance("hydro_id", "hydro_id"), 0);
        assert_eq!(edit_distance("m³", "m3"), 1);
    }

    fn parquet_compressions(codec: &str) -> Vec<Compression> {
        match codec {
            "zstd" => vec![Compression::ZSTD(ZstdLevel::default())],
            "snappy" => vec![Compression::SNAPPY],
            "gzip" => vec![Compression::GZIP(GzipLevel::default())],
            "lz4" => vec![Compression::LZ4, Compression::LZ4_RAW],
            other => panic!("INPUT_PARQUET_CODECS lists {other}, which no fixture writes"),
        }
    }

    fn codec_fixture_batch() -> RecordBatch {
        let rows = 500_i32;
        make_batch(
            vec![
                Field::new("hydro_id", DataType::Int32, false),
                Field::new("mean_m3s", DataType::Float64, true),
                Field::new("start_date", DataType::Date32, false),
                Field::new("source", DataType::Utf8, false),
            ],
            vec![
                Arc::new(Int32Array::from_iter_values((0..rows).map(|row| row % 7))),
                Arc::new(
                    (0..rows)
                        .map(|row| (row % 5 != 0).then(|| f64::from(row) * 1.25))
                        .collect::<Float64Array>(),
                ),
                Arc::new(Date32Array::from_iter_values(0..rows)),
                Arc::new(StringArray::from_iter_values(
                    (0..rows).map(|row| format!("plant-{}", row % 3)),
                )),
            ],
        )
    }

    fn write_parquet_compressed(batch: &RecordBatch, compression: Compression) -> NamedTempFile {
        let tmp = NamedTempFile::new().unwrap();
        let properties = WriterProperties::builder()
            .set_compression(compression)
            .build();
        let mut writer =
            ArrowWriter::try_new(tmp.reopen().unwrap(), batch.schema(), Some(properties)).unwrap();
        writer.write(batch).unwrap();
        writer.close().unwrap();
        tmp
    }

    fn file_compressions(path: &Path) -> Vec<Compression> {
        let builder = ParquetRecordBatchReaderBuilder::try_new(File::open(path).unwrap()).unwrap();
        let mut compressions = Vec::new();
        for row_group in builder.metadata().row_groups() {
            for column in row_group.columns() {
                compressions.push(column.compression());
            }
        }
        compressions
    }

    fn read_input_batches(path: &Path) -> Vec<RecordBatch> {
        open_parquet_input(path, &TEST_COLUMNS)
            .unwrap()
            .map(|batch| batch.unwrap().batch)
            .collect()
    }

    #[test]
    fn every_listed_input_codec_reads_the_rows_of_the_uncompressed_file() {
        let batch = codec_fixture_batch();
        let uncompressed = write_parquet_compressed(&batch, Compression::UNCOMPRESSED);
        let expected = read_input_batches(uncompressed.path());
        assert_eq!(expected, vec![batch.clone()]);

        for codec in INPUT_PARQUET_CODECS {
            for compression in parquet_compressions(codec) {
                let file = write_parquet_compressed(&batch, compression);
                assert!(
                    file_compressions(file.path())
                        .iter()
                        .all(|written| *written == compression),
                    "the {codec} fixture must be written as {compression:?}"
                );
                assert_eq!(
                    read_input_batches(file.path()),
                    expected,
                    "{compression:?} must read the rows of the uncompressed file"
                );
            }
        }
    }
}
