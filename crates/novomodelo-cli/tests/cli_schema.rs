//! Integration tests for `cobre schema export`.
//!
//! Verifies that the subcommand writes `.schema.json` files to the specified
//! output directory, creates directories on demand, handles the default
//! (current-directory) case, and exits non-zero when a path is unwritable.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use assert_cmd::prelude::*;
use predicates::prelude::*;
use std::collections::BTreeSet;
use std::path::Path;
use std::process::Command;
use tempfile::TempDir;

fn cobre() -> Command {
    Command::new(assert_cmd::cargo::cargo_bin!("cobre"))
}

#[test]
fn test_schema_export_writes_files() {
    let tmp = TempDir::new().unwrap();
    let output_dir = tmp.path();

    cobre()
        .args([
            "schema",
            "export",
            "--output-dir",
            &output_dir.to_string_lossy(),
        ])
        .assert()
        .success()
        .stderr(predicate::str::contains("Exported"))
        .stderr(predicate::str::contains("schema files to"));

    let schema_files: Vec<_> = std::fs::read_dir(output_dir)
        .unwrap()
        .filter_map(std::result::Result::ok)
        .filter(|e| e.file_name().to_string_lossy().ends_with(".schema.json"))
        .collect();

    // Compare emitted vs committed `schemas/` basenames as sets derived
    // at test time — never a hard-coded count — so a dropped/added/renamed schema fails.
    let emitted: BTreeSet<String> = schema_files
        .iter()
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();

    let committed_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../schemas");
    let committed: BTreeSet<String> = std::fs::read_dir(&committed_dir)
        .expect("read schemas")
        .filter_map(std::result::Result::ok)
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n.ends_with(".schema.json"))
        .collect();

    assert_eq!(
        emitted,
        committed,
        "emitted schema set must equal committed schemas set; \
         emitted-only: {:?}; committed-only: {:?}",
        emitted.difference(&committed).collect::<Vec<_>>(),
        committed.difference(&emitted).collect::<Vec<_>>(),
    );

    let config_path = output_dir.join("config.schema.json");
    assert!(config_path.exists(), "config.schema.json must exist");

    let content = std::fs::read_to_string(&config_path).unwrap();
    let value: serde_json::Value =
        serde_json::from_str(&content).expect("config.schema.json must be valid JSON");

    assert!(
        value.is_object(),
        "config.schema.json root must be a JSON object"
    );
    assert!(
        value.get("$schema").is_some(),
        "config.schema.json must contain a '$schema' key"
    );
    let schema_url = value["$schema"].as_str().unwrap_or("");
    assert!(
        schema_url.contains("json-schema.org"),
        "config.schema.json '$schema' must point to the JSON Schema draft URL, got: {schema_url}"
    );
    assert!(
        value.get("properties").is_some(),
        "config.schema.json must contain a 'properties' key"
    );

    let props = value["properties"].as_object().unwrap();
    assert!(
        props.contains_key("training"),
        "config schema 'properties' must contain 'training'"
    );
    assert!(
        props.contains_key("simulation"),
        "config schema 'properties' must contain 'simulation'"
    );
}

#[test]
fn test_schema_export_default_dir() {
    let tmp = TempDir::new().unwrap();

    cobre()
        .args(["schema", "export"])
        .current_dir(tmp.path())
        .assert()
        .success()
        .stderr(predicate::str::contains("Exported"));

    let schema_files: Vec<_> = std::fs::read_dir(tmp.path())
        .unwrap()
        .filter_map(std::result::Result::ok)
        .filter(|e| e.file_name().to_string_lossy().ends_with(".schema.json"))
        .collect();

    assert!(
        !schema_files.is_empty(),
        "at least one .schema.json file must be written to the current directory"
    );
}

#[test]
fn test_schema_export_creates_dir() {
    let tmp = TempDir::new().unwrap();
    let new_dir = tmp.path().join("schemas");

    assert!(!new_dir.exists(), "precondition: directory must not exist");

    cobre()
        .args([
            "schema",
            "export",
            "--output-dir",
            &new_dir.to_string_lossy(),
        ])
        .assert()
        .success()
        .stderr(predicate::str::contains("Exported"));

    assert!(new_dir.exists(), "output directory must be created");

    let schema_files: Vec<_> = std::fs::read_dir(&new_dir)
        .unwrap()
        .filter_map(std::result::Result::ok)
        .filter(|e| e.file_name().to_string_lossy().ends_with(".schema.json"))
        .collect();

    assert!(
        !schema_files.is_empty(),
        "at least one .schema.json file must exist in the created directory"
    );
}

/// The root `/nonexistent` is absent on any standard filesystem, so the parent
/// directories cannot be created and the export exits non-zero.
#[test]
fn test_schema_export_unwritable_dir_exits_nonzero() {
    cobre()
        .args([
            "schema",
            "export",
            "--output-dir",
            "/nonexistent/x/y/schemas",
        ])
        .assert()
        .failure();
}
