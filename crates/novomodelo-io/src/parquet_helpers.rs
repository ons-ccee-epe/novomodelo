//! Centralises the shared Parquet reader-opening and typed column-downcast
//! logic used by every Parquet parser in `cobre-io`.

use arrow::array::{Array, Date32Array, Float64Array, Int32Array, UInt32Array};
use arrow::datatypes::DataType;
use arrow::record_batch::RecordBatch;
use parquet::arrow::arrow_reader::{ParquetRecordBatchReader, ParquetRecordBatchReaderBuilder};
use std::fs::File;
use std::path::Path;

use crate::LoadError;

/// Single owner of `cobre-io`'s parquet open/build error mappings.
pub(crate) fn open_record_batch_reader(path: &Path) -> Result<ParquetRecordBatchReader, LoadError> {
    let file = File::open(path).map_err(|e| LoadError::io(path, e))?;

    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| LoadError::parse(path, e.to_string()))?;

    builder
        .build()
        .map_err(|e| LoadError::parse(path, e.to_string()))
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

pub(crate) fn extract_required_int32<'a>(
    batch: &'a RecordBatch,
    name: &str,
    path: &Path,
) -> Result<&'a Int32Array, LoadError> {
    let col = batch
        .column_by_name(name)
        .ok_or_else(|| missing_column_error(name, path))?;
    col.as_any()
        .downcast_ref::<Int32Array>()
        .ok_or_else(|| wrong_type_error(name, path, col.data_type(), "Int32"))
}

pub(crate) fn extract_required_float64<'a>(
    batch: &'a RecordBatch,
    name: &str,
    path: &Path,
) -> Result<&'a Float64Array, LoadError> {
    let col = batch
        .column_by_name(name)
        .ok_or_else(|| missing_column_error(name, path))?;
    col.as_any()
        .downcast_ref::<Float64Array>()
        .ok_or_else(|| wrong_type_error(name, path, col.data_type(), "Float64"))
}

pub(crate) fn extract_optional_int32<'a>(
    batch: &'a RecordBatch,
    name: &str,
    path: &Path,
) -> Result<Option<&'a Int32Array>, LoadError> {
    let Some(col) = batch.column_by_name(name) else {
        return Ok(None);
    };
    let arr = col
        .as_any()
        .downcast_ref::<Int32Array>()
        .ok_or_else(|| wrong_type_error(name, path, col.data_type(), "Int32"))?;
    Ok(Some(arr))
}

pub(crate) fn extract_optional_float64<'a>(
    batch: &'a RecordBatch,
    name: &str,
    path: &Path,
) -> Result<Option<&'a Float64Array>, LoadError> {
    let Some(col) = batch.column_by_name(name) else {
        return Ok(None);
    };
    let arr = col
        .as_any()
        .downcast_ref::<Float64Array>()
        .ok_or_else(|| wrong_type_error(name, path, col.data_type(), "Float64"))?;
    Ok(Some(arr))
}

pub(crate) fn extract_required_uint32<'a>(
    batch: &'a RecordBatch,
    name: &str,
    path: &Path,
) -> Result<&'a UInt32Array, LoadError> {
    let col = batch
        .column_by_name(name)
        .ok_or_else(|| missing_column_error(name, path))?;
    col.as_any()
        .downcast_ref::<UInt32Array>()
        .ok_or_else(|| wrong_type_error(name, path, col.data_type(), "UInt32"))
}

pub(crate) fn extract_required_date32<'a>(
    batch: &'a RecordBatch,
    name: &str,
    path: &Path,
) -> Result<&'a Date32Array, LoadError> {
    let col = batch
        .column_by_name(name)
        .ok_or_else(|| missing_column_error(name, path))?;
    col.as_any()
        .downcast_ref::<Date32Array>()
        .ok_or_else(|| wrong_type_error(name, path, col.data_type(), "Date32"))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::test_support::{write_json, write_parquet_batches};
    use arrow::array::ArrayRef;
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    fn make_batch(fields: Vec<Field>, columns: Vec<ArrayRef>) -> RecordBatch {
        RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).unwrap()
    }

    #[test]
    fn extract_required_int32_reads_column() {
        let batch = make_batch(
            vec![Field::new("hydro_id", DataType::Int32, false)],
            vec![Arc::new(Int32Array::from(vec![1, 2, 3]))],
        );
        let result = extract_required_int32(&batch, "hydro_id", Path::new("test.parquet")).unwrap();
        assert_eq!(result.value(0), 1);
        assert_eq!(result.value(1), 2);
        assert_eq!(result.value(2), 3);
    }

    #[test]
    fn extract_required_int32_errors_on_missing_column() {
        let batch = make_batch(
            vec![Field::new("other", DataType::Int32, false)],
            vec![Arc::new(Int32Array::from(vec![0]))],
        );
        let path = Path::new("test.parquet");
        let err = extract_required_int32(&batch, "hydro_id", path).unwrap_err();
        let LoadError::SchemaError {
            path: err_path,
            field,
            message,
        } = err
        else {
            panic!("expected SchemaError, got a different variant");
        };
        assert_eq!(field, "hydro_id");
        assert_eq!(err_path, path);
        assert_eq!(message, "missing required column \"hydro_id\"");
    }

    #[test]
    fn extract_required_int32_errors_on_wrong_type() {
        let batch = make_batch(
            vec![Field::new("hydro_id", DataType::Float64, false)],
            vec![Arc::new(Float64Array::from(vec![1.0]))],
        );
        let path = Path::new("test.parquet");
        let err = extract_required_int32(&batch, "hydro_id", path).unwrap_err();
        let LoadError::SchemaError {
            path: err_path,
            field,
            message,
        } = err
        else {
            panic!("expected SchemaError, got a different variant");
        };
        assert_eq!(field, "hydro_id");
        assert_eq!(err_path, path);
        assert!(message.starts_with("column \"hydro_id\" has type "));
        assert!(message.ends_with(" but Int32 is required"));
    }

    #[test]
    fn extract_required_float64_reads_column() {
        let batch = make_batch(
            vec![Field::new("mean_m3s", DataType::Float64, false)],
            vec![Arc::new(Float64Array::from(vec![1.5, 2.5]))],
        );
        let result =
            extract_required_float64(&batch, "mean_m3s", Path::new("test.parquet")).unwrap();
        assert!((result.value(0) - 1.5).abs() < f64::EPSILON);
        assert!((result.value(1) - 2.5).abs() < f64::EPSILON);
    }

    #[test]
    fn extract_required_float64_errors_on_missing_column() {
        let batch = make_batch(
            vec![Field::new("other", DataType::Int32, false)],
            vec![Arc::new(Int32Array::from(vec![0]))],
        );
        let path = Path::new("test.parquet");
        let err = extract_required_float64(&batch, "mean_m3s", path).unwrap_err();
        let LoadError::SchemaError {
            path: err_path,
            field,
            message,
        } = err
        else {
            panic!("expected SchemaError, got a different variant");
        };
        assert_eq!(field, "mean_m3s");
        assert_eq!(err_path, path);
        assert_eq!(message, "missing required column \"mean_m3s\"");
    }

    #[test]
    fn extract_required_float64_errors_on_wrong_type() {
        let batch = make_batch(
            vec![Field::new("mean_m3s", DataType::Int32, false)],
            vec![Arc::new(Int32Array::from(vec![1]))],
        );
        let path = Path::new("test.parquet");
        let err = extract_required_float64(&batch, "mean_m3s", path).unwrap_err();
        let LoadError::SchemaError {
            path: err_path,
            field,
            message,
        } = err
        else {
            panic!("expected SchemaError, got a different variant");
        };
        assert_eq!(field, "mean_m3s");
        assert_eq!(err_path, path);
        assert!(message.starts_with("column \"mean_m3s\" has type "));
        assert!(message.ends_with(" but Float64 is required"));
    }

    #[test]
    fn extract_optional_int32_reads_column() {
        let batch = make_batch(
            vec![Field::new("stage_id", DataType::Int32, false)],
            vec![Arc::new(Int32Array::from(vec![5, 6]))],
        );
        let result = extract_optional_int32(&batch, "stage_id", Path::new("test.parquet"))
            .unwrap()
            .unwrap();
        assert_eq!(result.value(0), 5);
        assert_eq!(result.value(1), 6);
    }

    #[test]
    fn extract_optional_int32_returns_none_on_missing_column() {
        let batch = make_batch(
            vec![Field::new("other", DataType::Int32, false)],
            vec![Arc::new(Int32Array::from(vec![0]))],
        );
        let result = extract_optional_int32(&batch, "stage_id", Path::new("test.parquet")).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn extract_optional_int32_errors_on_wrong_type() {
        let batch = make_batch(
            vec![Field::new("stage_id", DataType::Float64, false)],
            vec![Arc::new(Float64Array::from(vec![1.0]))],
        );
        let path = Path::new("test.parquet");
        let err = extract_optional_int32(&batch, "stage_id", path).unwrap_err();
        let LoadError::SchemaError {
            path: err_path,
            field,
            message,
        } = err
        else {
            panic!("expected SchemaError, got a different variant");
        };
        assert_eq!(field, "stage_id");
        assert_eq!(err_path, path);
        assert!(message.starts_with("column \"stage_id\" has type "));
        assert!(message.ends_with(" but Int32 is required"));
    }

    #[test]
    fn extract_optional_float64_reads_column() {
        let batch = make_batch(
            vec![Field::new("std_m3s", DataType::Float64, false)],
            vec![Arc::new(Float64Array::from(vec![0.1, 0.2]))],
        );
        let result = extract_optional_float64(&batch, "std_m3s", Path::new("test.parquet"))
            .unwrap()
            .unwrap();
        assert!((result.value(0) - 0.1).abs() < f64::EPSILON);
        assert!((result.value(1) - 0.2).abs() < f64::EPSILON);
    }

    #[test]
    fn extract_optional_float64_returns_none_on_missing_column() {
        let batch = make_batch(
            vec![Field::new("other", DataType::Int32, false)],
            vec![Arc::new(Int32Array::from(vec![0]))],
        );
        let result =
            extract_optional_float64(&batch, "std_m3s", Path::new("test.parquet")).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn extract_optional_float64_errors_on_wrong_type() {
        let batch = make_batch(
            vec![Field::new("std_m3s", DataType::Int32, false)],
            vec![Arc::new(Int32Array::from(vec![1]))],
        );
        let path = Path::new("test.parquet");
        let err = extract_optional_float64(&batch, "std_m3s", path).unwrap_err();
        let LoadError::SchemaError {
            path: err_path,
            field,
            message,
        } = err
        else {
            panic!("expected SchemaError, got a different variant");
        };
        assert_eq!(field, "std_m3s");
        assert_eq!(err_path, path);
        assert!(message.starts_with("column \"std_m3s\" has type "));
        assert!(message.ends_with(" but Float64 is required"));
    }

    #[test]
    fn extract_required_uint32_reads_column() {
        let batch = make_batch(
            vec![Field::new("unit_count", DataType::UInt32, false)],
            vec![Arc::new(UInt32Array::from(vec![7, 9]))],
        );
        let result =
            extract_required_uint32(&batch, "unit_count", Path::new("test.parquet")).unwrap();
        assert_eq!(result.value(0), 7);
        assert_eq!(result.value(1), 9);
    }

    #[test]
    fn extract_required_uint32_errors_on_missing_column() {
        let batch = make_batch(
            vec![Field::new("other", DataType::Int32, false)],
            vec![Arc::new(Int32Array::from(vec![0]))],
        );
        let path = Path::new("test.parquet");
        let err = extract_required_uint32(&batch, "unit_count", path).unwrap_err();
        let LoadError::SchemaError {
            path: err_path,
            field,
            message,
        } = err
        else {
            panic!("expected SchemaError, got a different variant");
        };
        assert_eq!(field, "unit_count");
        assert_eq!(err_path, path);
        assert_eq!(message, "missing required column \"unit_count\"");
    }

    #[test]
    fn extract_required_uint32_errors_on_wrong_type() {
        let batch = make_batch(
            vec![Field::new("unit_count", DataType::Int32, false)],
            vec![Arc::new(Int32Array::from(vec![1]))],
        );
        let path = Path::new("test.parquet");
        let err = extract_required_uint32(&batch, "unit_count", path).unwrap_err();
        let LoadError::SchemaError {
            path: err_path,
            field,
            message,
        } = err
        else {
            panic!("expected SchemaError, got a different variant");
        };
        assert_eq!(field, "unit_count");
        assert_eq!(err_path, path);
        assert!(message.starts_with("column \"unit_count\" has type "));
        assert!(message.ends_with(" but UInt32 is required"));
    }

    #[test]
    fn extract_required_date32_reads_column() {
        let batch = make_batch(
            vec![Field::new("start_date", DataType::Date32, false)],
            vec![Arc::new(Date32Array::from(vec![19000, 19001]))],
        );
        let result =
            extract_required_date32(&batch, "start_date", Path::new("test.parquet")).unwrap();
        assert_eq!(result.value(0), 19000);
        assert_eq!(result.value(1), 19001);
    }

    #[test]
    fn extract_required_date32_errors_on_missing_column() {
        let batch = make_batch(
            vec![Field::new("other", DataType::Int32, false)],
            vec![Arc::new(Int32Array::from(vec![0]))],
        );
        let path = Path::new("test.parquet");
        let err = extract_required_date32(&batch, "start_date", path).unwrap_err();
        let LoadError::SchemaError {
            path: err_path,
            field,
            message,
        } = err
        else {
            panic!("expected SchemaError, got a different variant");
        };
        assert_eq!(field, "start_date");
        assert_eq!(err_path, path);
        assert_eq!(message, "missing required column \"start_date\"");
    }

    #[test]
    fn extract_required_date32_errors_on_wrong_type() {
        let batch = make_batch(
            vec![Field::new("start_date", DataType::Int32, false)],
            vec![Arc::new(Int32Array::from(vec![1]))],
        );
        let path = Path::new("test.parquet");
        let err = extract_required_date32(&batch, "start_date", path).unwrap_err();
        let LoadError::SchemaError {
            path: err_path,
            field,
            message,
        } = err
        else {
            panic!("expected SchemaError, got a different variant");
        };
        assert_eq!(field, "start_date");
        assert_eq!(err_path, path);
        assert!(message.starts_with("column \"start_date\" has type "));
        assert!(message.ends_with(" but Date32 is required"));
    }

    #[test]
    fn open_record_batch_reader_reads_every_batch() {
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
        let reader = open_record_batch_reader(tmp.path()).unwrap();
        let mut rows = 0;
        for batch_result in reader {
            rows += batch_result.unwrap().num_rows();
        }
        assert_eq!(rows, total_rows);
    }

    #[test]
    fn open_record_batch_reader_errors_on_missing_file() {
        let path = Path::new("/nonexistent-dir/does-not-exist.parquet");
        let err = open_record_batch_reader(path).unwrap_err();
        let LoadError::IoError { path: err_path, .. } = err else {
            panic!("expected IoError, got a different variant");
        };
        assert_eq!(err_path, path);
    }

    #[test]
    fn open_record_batch_reader_errors_on_non_parquet_file() {
        let tmp = write_json("{}");
        let err = open_record_batch_reader(tmp.path()).unwrap_err();
        let LoadError::ParseError { path: err_path, .. } = err else {
            panic!("expected ParseError, got a different variant");
        };
        assert_eq!(err_path, tmp.path());
    }
}
