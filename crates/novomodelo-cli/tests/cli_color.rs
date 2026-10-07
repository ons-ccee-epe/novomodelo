//! Integration tests for the `--color` global flag.

#![allow(clippy::unwrap_used)]

use assert_cmd::prelude::*;
use predicates::prelude::*;
use tempfile::TempDir;

mod common;

const CONFIG_JSON: &str = r#"{
    "training": {
        "selection": { "method": "sampled", "forward_passes": 1 },
        "stopping_rules": [
            { "type": "iteration_limit", "limit": 2 }
        ],
        "scenario_source": { "inflow": { "scheme": "in_sample" }, "seed": 42 }
    }
}"#;

const STAGES_JSON: &str = r#"{
    "policy_graph": {
        "type": "finite_horizon",
        "annual_discount_rate": 0.06,
        "transitions": []
    },
    "stages": [
        {
            "id": 0,
            "start_date": "2024-01-01",
            "end_date": "2024-02-01",
            "blocks": [{ "id": 0, "name": "FLAT", "hours": 744.0 }],
            "num_openings": 2
        },
        {
            "id": 1,
            "start_date": "2024-02-01",
            "end_date": "2024-03-01",
            "blocks": [{ "id": 0, "name": "FLAT", "hours": 672.0 }],
            "num_openings": 2
        }
    ]
}"#;

const INITIAL_CONDITIONS_JSON: &str = r#"{ "storage": [], "filling_storage": [] }"#;
const BUSES_JSON: &str =
    r#"{ "buses": [{ "id": 1, "name": "BUS_1", "operational_start_date": "2024-01-01" }] }"#;
const LINES_JSON: &str = r#"{ "lines": [] }"#;
const HYDROS_JSON: &str = r#"{ "hydros": [] }"#;
const THERMALS_JSON: &str = r#"{ "thermals": [] }"#;

fn make_valid_case(dir: &TempDir) {
    let root = dir.path();
    common::write_file(root, "config.json", CONFIG_JSON);
    common::write_file(root, "penalties.json", common::PENALTIES_JSON);
    common::write_file(root, "stages.json", STAGES_JSON);
    common::write_file(root, "initial_conditions.json", INITIAL_CONDITIONS_JSON);
    common::write_file(root, "system/buses.json", BUSES_JSON);
    common::write_file(root, "system/lines.json", LINES_JSON);
    common::write_file(root, "system/hydros.json", HYDROS_JSON);
    common::write_file(root, "system/thermals.json", THERMALS_JSON);
}

#[test]
fn color_always_flag_forces_ansi_in_banner() {
    let dir = TempDir::new().unwrap();
    make_valid_case(&dir);
    let out = TempDir::new().unwrap();

    common::cobre()
        .args([
            "run",
            "--color",
            "always",
            dir.path().to_str().unwrap(),
            "--output",
            out.path().to_str().unwrap(),
        ])
        .assert()
        .success()
        // Auto-detection would disable color on a piped stderr subprocess; the
        // 256-color orange busbar escape is present only when color is forced on.
        .stderr(predicate::str::contains("\x1b[38;5;172m"));
}

/// `--quiet` suppresses the `indicatif` progress bar, whose cursor-movement
/// sequences (`\x1b[1A`, `\x1b[2K`) are structural, not color-related, and would
/// otherwise survive `--color never` and defeat the no-ANSI assertion.
#[test]
fn color_never_flag_suppresses_ansi_in_banner() {
    let dir = TempDir::new().unwrap();
    make_valid_case(&dir);
    let out = TempDir::new().unwrap();

    common::cobre()
        .args([
            "run",
            "--color",
            "never",
            "--quiet",
            dir.path().to_str().unwrap(),
            "--output",
            out.path().to_str().unwrap(),
        ])
        .assert()
        .success()
        .stderr(predicate::str::contains("\x1b[").not());
}

#[test]
fn color_always_global_flag_before_subcommand_is_accepted() {
    let dir = TempDir::new().unwrap();
    make_valid_case(&dir);
    let out = TempDir::new().unwrap();

    common::cobre()
        .args([
            "--color",
            "always",
            "run",
            dir.path().to_str().unwrap(),
            "--output",
            out.path().to_str().unwrap(),
        ])
        .assert()
        .success()
        .stderr(predicate::str::contains("\x1b[38;5;172m"));
}
