//! Integration tests for the `novomodelo init` subcommand.

#![allow(clippy::unwrap_used)]

use std::fs;

use assert_cmd::prelude::*;
use predicates::prelude::*;
use std::process::Command;
use tempfile::TempDir;

fn novomodelo() -> Command {
    Command::new(assert_cmd::cargo::cargo_bin!("novomodelo"))
}

#[test]
fn test_init_list_shows_1dtoy() {
    novomodelo()
        .args(["init", "--list"])
        .assert()
        .success()
        .stdout(predicate::str::contains("1dtoy"));
}

#[test]
fn test_init_1dtoy_creates_valid_case() {
    let dir = TempDir::new().unwrap();
    let dir_str = dir.path().to_str().unwrap();

    novomodelo()
        .args(["init", "--template", "1dtoy", dir_str])
        .assert()
        .success();

    novomodelo().args(["validate", dir_str]).assert().success();
}

#[test]
fn test_init_unknown_template_fails() {
    let dir = TempDir::new().unwrap();
    let dir_str = dir.path().to_str().unwrap();

    novomodelo()
        .args(["init", "--template", "bogus", dir_str])
        .assert()
        .failure()
        .stderr(predicate::str::contains("Available"));
}

#[test]
fn test_init_no_args_fails() {
    novomodelo().args(["init"]).assert().failure();
}

#[test]
fn test_init_existing_non_empty_dir_fails() {
    let dir = TempDir::new().unwrap();
    fs::write(dir.path().join("dummy.txt"), "x").unwrap();

    novomodelo()
        .args(["init", "--template", "1dtoy", dir.path().to_str().unwrap()])
        .assert()
        .failure()
        .code(2)
        .stderr(predicate::str::contains("force"));
}

#[test]
fn test_init_force_overwrites() {
    let dir = TempDir::new().unwrap();
    fs::write(dir.path().join("dummy.txt"), "x").unwrap();

    novomodelo()
        .args([
            "init",
            "--template",
            "1dtoy",
            dir.path().to_str().unwrap(),
            "--force",
        ])
        .assert()
        .success();
}
