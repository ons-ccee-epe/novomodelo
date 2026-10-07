//! JSON writer for the LP scaling report.
//!
//! The scaling report is a one-time diagnostic artifact produced after
//! template build and column/row scaling. It captures the coefficient
//! ranges before and after scaling for every stage.

use std::path::Path;

use super::atomic::{ensure_parent_dir, write_json_atomic};
use super::error::OutputError;

use serde::Serialize;

/// Write a scaling report as pretty-printed JSON, atomically.
///
/// Generic over `Serialize` so the report struct stays in the calling algorithm
/// crate, keeping this crate algorithm-agnostic.
///
/// # Errors
///
/// Returns [`OutputError::IoError`] on filesystem failures, or
/// [`OutputError::SerializationError`] if JSON serialization fails.
pub fn write_scaling_report(path: &Path, report: &impl Serialize) -> Result<(), OutputError> {
    ensure_parent_dir(path)?;

    write_json_atomic(path, report, "scaling_report")
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[derive(Serialize)]
    struct MockReport {
        cost_scale_factor: f64,
        num_stages: usize,
    }

    #[test]
    fn write_and_read_back_json() {
        let dir = TempDir::new().expect("temp dir");
        let path = dir.path().join("training/scaling_report.json");

        let report = MockReport {
            cost_scale_factor: 1000.0,
            num_stages: 3,
        };

        write_scaling_report(&path, &report).expect("write should succeed");

        let content = std::fs::read_to_string(&path).expect("read");
        assert!(content.contains("\"cost_scale_factor\": 1000.0"));
        assert!(content.contains("\"num_stages\": 3"));
    }

    #[test]
    fn tmp_file_is_cleaned_up() {
        let dir = TempDir::new().expect("temp dir");
        let path = dir.path().join("report.json");

        let report = MockReport {
            cost_scale_factor: 1.0,
            num_stages: 1,
        };

        write_scaling_report(&path, &report).expect("write should succeed");

        let tmp_path = path.with_extension("json.tmp");
        assert!(
            !tmp_path.exists(),
            "tmp file should be removed after rename"
        );
        assert!(path.exists(), "final file should exist");
    }
}
