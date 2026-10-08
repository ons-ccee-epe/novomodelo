//! JSON writer for the model provenance report.
//!
//! The provenance report is a one-time diagnostic artifact produced after
//! stochastic preprocessing. It captures which data sources were used for
//! each role — seasonal statistics, AR coefficients, correlation, and the
//! opening scenario tree.

use std::path::Path;

use super::atomic::{ensure_parent_dir, write_json_atomic};
use super::error::OutputError;

use serde::Serialize;

/// Write a model provenance report as pretty-printed JSON, atomically.
///
/// Generic over `Serialize` so the report struct stays in the calling algorithm
/// crate, keeping this crate algorithm-agnostic.
///
/// # Errors
///
/// Returns [`OutputError::IoError`] on filesystem failures, or
/// [`OutputError::SerializationError`] if JSON serialization fails.
pub fn write_provenance_report(path: &Path, report: &impl Serialize) -> Result<(), OutputError> {
    ensure_parent_dir(path)?;

    write_json_atomic(path, report, "model_provenance")
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;
    use serde::{Deserialize, Serialize};
    use tempfile::TempDir;

    #[derive(Serialize)]
    struct MockProvenanceReport {
        estimation_path: String,
        seasonal_stats_source: String,
        ar_coefficients_source: String,
        correlation_source: String,
        opening_tree_source: String,
        n_hydros: usize,
    }

    fn make_mock_report() -> MockProvenanceReport {
        MockProvenanceReport {
            estimation_path: "full_estimation".to_string(),
            seasonal_stats_source: "estimated".to_string(),
            ar_coefficients_source: "estimated".to_string(),
            correlation_source: "estimated".to_string(),
            opening_tree_source: "estimated".to_string(),
            n_hydros: 3,
        }
    }

    #[test]
    fn write_and_read_back_json() {
        let dir = TempDir::new().expect("temp dir");
        let path = dir.path().join("training/model_provenance.json");

        let report = make_mock_report();

        write_provenance_report(&path, &report).expect("write should succeed");

        let content = std::fs::read_to_string(&path).expect("read");
        assert!(content.contains("\"full_estimation\""));
        assert!(content.contains("\"n_hydros\": 3"));
    }

    #[test]
    fn round_trip_all_fields() {
        let dir = TempDir::new().expect("temp dir");
        let path = dir.path().join("model_provenance.json");

        let report = MockProvenanceReport {
            estimation_path: "deterministic".to_string(),
            seasonal_stats_source: "n/a".to_string(),
            ar_coefficients_source: "n/a".to_string(),
            correlation_source: "n/a".to_string(),
            opening_tree_source: "n/a".to_string(),
            n_hydros: 0,
        };

        write_provenance_report(&path, &report).expect("write should succeed");

        let content = std::fs::read_to_string(&path).expect("read");
        let value: serde_json::Value =
            serde_json::from_str(&content).expect("valid JSON after round-trip");

        assert_eq!(value["estimation_path"], "deterministic");
        assert_eq!(value["seasonal_stats_source"], "n/a");
        assert_eq!(value["ar_coefficients_source"], "n/a");
        assert_eq!(value["correlation_source"], "n/a");
        assert_eq!(value["opening_tree_source"], "n/a");
        assert_eq!(value["n_hydros"], 0);
    }

    #[test]
    fn tmp_file_is_cleaned_up() {
        let dir = TempDir::new().expect("temp dir");
        let path = dir.path().join("model_provenance.json");

        let report = make_mock_report();

        write_provenance_report(&path, &report).expect("write should succeed");

        let tmp_path = path.with_extension("json.tmp");
        assert!(
            !tmp_path.exists(),
            "tmp file should be removed after rename"
        );
        assert!(path.exists(), "final file should exist");
    }

    /// Nested mock mirroring the cross-model report shape, defined locally so
    /// the round-trip test never depends on an algorithm crate (genericity rule).
    #[derive(Debug, Serialize, Deserialize, PartialEq)]
    struct MockReport {
        inflow: MockSection,
        hydro_production: MockSection,
    }

    #[derive(Debug, Serialize, Deserialize, PartialEq)]
    struct MockSection {
        source: String,
        count: usize,
    }

    fn make_nested_mock() -> MockReport {
        MockReport {
            inflow: MockSection {
                source: "estimated".to_string(),
                count: 5,
            },
            hydro_production: MockSection {
                source: "precomputed".to_string(),
                count: 12,
            },
        }
    }

    #[test]
    fn write_provenance_report_round_trips_nested_mock() {
        let dir = TempDir::new().expect("temp dir");
        let path = dir.path().join("training/model_provenance.json");

        let report = make_nested_mock();
        write_provenance_report(&path, &report).expect("write should succeed");

        let content = std::fs::read_to_string(&path).expect("read");
        let decoded: MockReport =
            serde_json::from_str(&content).expect("valid JSON after round-trip");
        assert_eq!(decoded, report);
    }
}
