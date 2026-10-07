//! JSON Schema export helper for the `cobre.schema` sub-module.
//!
//! [`export`] delegates to the same `cobre-io` writer as the `cobre schema
//! export` CLI, but returns the count of files written rather than printing a
//! confirmation line.

use std::path::PathBuf;

use cobre_io::schema::SchemaExportError;
use pyo3::exceptions::{PyOSError, PyValueError};
use pyo3::prelude::*;

/// Generate JSON Schema files for all case-directory input types and write them
/// to `output_dir` (a `str` or `pathlib.Path`), creating it if needed.
///
/// Existing schema files are overwritten without prompting. Returns the number
/// of files written.
///
/// # Raises
///
/// * `ValueError` — schema generation failed.
/// * `OSError` — the output directory could not be created or a file write
///   failed.
///
/// # Examples
///
/// ```python
/// import cobre.schema
/// count = cobre.schema.export("schemas")
/// print(f"wrote {count} schema files")
/// ```
#[allow(clippy::needless_pass_by_value)]
#[pyfunction]
#[pyo3(signature = (output_dir=PathBuf::from(".")))]
pub fn export(output_dir: PathBuf) -> PyResult<usize> {
    cobre_io::schema::export_schemas(&output_dir).map_err(|err| match err {
        SchemaExportError::Generation(e) => {
            PyValueError::new_err(format!("schema generation failed: {e}"))
        }
        SchemaExportError::Serialization { filename, source } => PyValueError::new_err(format!(
            "serialization error for schema {filename}: {source}"
        )),
        SchemaExportError::Io { path, source } => {
            PyOSError::new_err(format!("{}: {source}", path.display()))
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_export_writes_all_schemas_as_valid_json() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let path = dir.path().to_path_buf();

        let count = export(path.clone()).expect("export must succeed");

        let entries: Vec<_> = std::fs::read_dir(&path)
            .expect("read temp dir")
            .map(|e| e.expect("dir entry").path())
            .collect();

        assert_eq!(count, entries.len());
        assert!(count > 0, "at least one schema must be written");

        for entry in &entries {
            let content = std::fs::read_to_string(entry).expect("read schema file");
            let parsed: serde_json::Value =
                serde_json::from_str(&content).expect("schema file must be valid JSON");
            assert!(parsed.is_object(), "each schema must be a JSON object");
        }
    }

    #[test]
    fn test_export_creates_missing_directory() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let nested = dir.path().join("nested").join("schemas");

        let count = export(nested.clone()).expect("export must create nested dirs");

        assert!(nested.is_dir(), "nested output directory must be created");
        assert!(count > 0);
    }
}
