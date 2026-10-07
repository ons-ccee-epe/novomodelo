//! Integration test: CLI training-metadata names the active solver backend.
//!
//! The asserted `solver` field tracks the built backend (`highs` by default,
//! `clp` under `--no-default-features --features clp`). Combined with the
//! identical struct-literal wiring in `cobre-python/src/run.rs` and the
//! source-level parity gate (`check_python_parity.py`), this establishes
//! CLI↔Python output-metadata parity per backend without requiring a built CLP
//! Python wheel.

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
fn training_metadata_solver_matches_active_backend() {
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

    let solver = json["solver"]
        .as_str()
        .expect("training metadata must carry a string `solver` field");
    assert_eq!(
        solver,
        cobre_solver::active_solver_metadata_id(),
        "training metadata `solver` must name the active backend ({})",
        cobre_solver::active_solver_metadata_id()
    );

    let solver_version = json["solver_version"]
        .as_str()
        .expect("training metadata must carry a non-null string `solver_version` field");
    assert!(
        !solver_version.is_empty(),
        "training metadata `solver_version` must be non-empty"
    );
}
