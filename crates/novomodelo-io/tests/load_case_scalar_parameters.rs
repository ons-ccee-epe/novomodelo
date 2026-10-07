//! Integration tests for scalar-parameter cross-validation in [`load_case`].

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod helpers;

use cobre_io::{LoadError, load_case};
use tempfile::TempDir;

/// References `hydro_id` 99, which the minimal case (zero hydros) cannot satisfy.
const PARAMS_DANGLING_HYDRO: &str = r#"{
    "scalar_parameters": [
        {
            "id": 1,
            "name": "rho_eq_missing",
            "kind": "computed",
            "computed_spec": {
                "tag": "equivalent_productivity",
                "hydro_id": 99
            }
        }
    ]
}"#;

/// `per_stage` vector of length 2 against the minimal case's single study stage.
const PARAMS_PER_STAGE_LENGTH_MISMATCH: &str = r#"{
    "scalar_parameters": [
        {
            "id": 1,
            "name": "wrong_length",
            "kind": "per_stage",
            "values": [[0, 1.0], [1, 2.0]]
        }
    ]
}"#;

const PARAMS_DUPLICATE_ID: &str = r#"{
    "scalar_parameters": [
        { "id": 1, "name": "a", "kind": "constant", "value": 1.0 },
        { "id": 1, "name": "b", "kind": "constant", "value": 2.0 }
    ]
}"#;

#[test]
fn load_case_rejects_scalar_parameter_with_missing_hydro_reference() {
    let dir = TempDir::new().unwrap();
    helpers::make_minimal_case(&dir);
    helpers::write_file(
        dir.path(),
        "constraints/generic_parameters.json",
        PARAMS_DANGLING_HYDRO,
    );

    let err = load_case(dir.path()).expect_err("dangling hydro_id must be rejected");
    match err {
        LoadError::ConstraintError { description } => {
            assert!(
                description.contains("rho_eq_missing"),
                "error must name the offending parameter; got: {description}"
            );
            assert!(
                description.contains("99"),
                "error must mention the dangling hydro id 99; got: {description}"
            );
        }
        other => panic!("expected ConstraintError, got: {other:?}"),
    }
}

#[test]
fn load_case_rejects_per_stage_length_mismatch() {
    let dir = TempDir::new().unwrap();
    helpers::make_minimal_case(&dir);
    helpers::write_file(
        dir.path(),
        "constraints/generic_parameters.json",
        PARAMS_PER_STAGE_LENGTH_MISMATCH,
    );

    let err = load_case(dir.path()).expect_err("per_stage length mismatch must be rejected");
    match err {
        LoadError::ConstraintError { description } => {
            assert!(
                description.contains("wrong_length"),
                "error must name the offending parameter; got: {description}"
            );
            assert!(
                description.contains("n_stages")
                    || description.contains("expected 1")
                    || description.contains("expected 0"),
                "error must mention the expected length; got: {description}"
            );
        }
        other => panic!("expected ConstraintError, got: {other:?}"),
    }
}

#[test]
fn load_case_rejects_duplicate_parameter_id() {
    let dir = TempDir::new().unwrap();
    helpers::make_minimal_case(&dir);
    helpers::write_file(
        dir.path(),
        "constraints/generic_parameters.json",
        PARAMS_DUPLICATE_ID,
    );

    let err = load_case(dir.path()).expect_err("duplicate parameter id must be rejected");
    match err {
        LoadError::ConstraintError { description } => {
            assert!(
                description.contains("duplicate"),
                "error must mention duplicate; got: {description}"
            );
        }
        other => panic!("expected ConstraintError, got: {other:?}"),
    }
}

#[test]
fn load_case_accepts_valid_scalar_parameters() {
    let dir = TempDir::new().unwrap();
    helpers::make_minimal_case(&dir);
    helpers::write_file(
        dir.path(),
        "constraints/generic_parameters.json",
        r#"{
            "scalar_parameters": [
                { "id": 1, "name": "alpha", "kind": "constant", "value": 0.5 },
                { "id": 2, "name": "beta", "kind": "per_stage", "values": [[0, 1.0]] }
            ]
        }"#,
    );

    load_case(dir.path()).expect("valid scalar parameters must load");
}
