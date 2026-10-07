//! Integration test: CLI training metadata carries the per-phase setup timings.
//!
//! The `setup` section is informational and never enters any parity hash, so its
//! presence does not perturb the hashed Parquet output set.

#![allow(clippy::expect_used, clippy::panic)]

use std::path::PathBuf;
use std::process::Command;

use assert_cmd::prelude::*;
use tempfile::TempDir;

fn cobre() -> Command {
    Command::new(assert_cmd::cargo::cargo_bin!("cobre"))
}

fn d01_case_dir() -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let root = manifest
        .parent()
        .and_then(|p| p.parent())
        .expect("repo root must be two levels above CARGO_MANIFEST_DIR");
    root.join("examples/deterministic/d01-thermal-dispatch")
}

#[test]
fn training_metadata_carries_well_formed_setup_timings() {
    let case = d01_case_dir();
    assert!(
        case.join("config.json").is_file(),
        "D01 fixture must exist at {}",
        case.display()
    );

    // Temp output dir so the committed `output/` tree under the fixture is never disturbed.
    let out = TempDir::new().expect("create temp output dir");

    cobre()
        .args([
            "run",
            case.to_str().expect("D01 path is valid UTF-8"),
            "--output",
            out.path().to_str().expect("temp path is valid UTF-8"),
            "--quiet",
        ])
        .assert()
        .success();

    let metadata_path = out.path().join("training/metadata.json");
    assert!(
        metadata_path.is_file(),
        "training/metadata.json must exist at {}",
        metadata_path.display()
    );

    let contents = std::fs::read_to_string(&metadata_path).expect("read training/metadata.json");
    let json: serde_json::Value =
        serde_json::from_str(&contents).expect("training/metadata.json must be valid JSON");

    let setup = json
        .get("setup")
        .expect("training metadata must carry a `setup` object on rank 0");
    assert!(
        setup.is_object(),
        "the `setup` field must be a JSON object, got {setup}"
    );

    for field in [
        "load_seconds",
        "stochastic_fit_seconds",
        "production_fit_seconds",
        "evaporation_fit_seconds",
        "broadcast_seconds",
    ] {
        let value = setup
            .get(field)
            .unwrap_or_else(|| panic!("setup object must contain `{field}`"));
        let seconds = value
            .as_f64()
            .unwrap_or_else(|| panic!("setup.{field} must be a JSON number, got {value}"));
        assert!(
            seconds.is_finite(),
            "setup.{field} must be finite, got {seconds}"
        );
        assert!(
            seconds >= 0.0,
            "setup.{field} must be non-negative, got {seconds}"
        );
    }
}
