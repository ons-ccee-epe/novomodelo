//! Smoke tests for the `novomodelo` binary using `assert_cmd`.

use assert_cmd::prelude::*;
use predicates::prelude::*;
use std::process::Command;

fn novomodelo() -> Command {
    // `cargo_bin!` honors custom build directories via CARGO_BIN_EXE_novomodelo, unlike
    // the deprecated `Command::cargo_bin`.
    Command::new(assert_cmd::cargo::cargo_bin!("novomodelo"))
}

#[test]
fn help_exits_0_and_lists_subcommands() {
    novomodelo()
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("init"))
        .stdout(predicate::str::contains("run"))
        .stdout(predicate::str::contains("validate"))
        .stdout(predicate::str::contains("schema"))
        .stdout(predicate::str::contains("version"));
}

#[test]
fn run_help_exits_0_and_lists_flags() {
    novomodelo()
        .args(["run", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("--output"))
        .stdout(predicate::str::contains("--quiet"))
        .stdout(predicate::str::contains("--threads"))
        .stdout(predicate::str::contains("CASE_DIR"));
}

/// Exit 2 here is a clap validation error (the `.range(1..)` parser), not I/O —
/// the case path is never touched.
#[test]
fn run_threads_zero_exits_with_clap_error() {
    novomodelo()
        .args(["run", "--threads", "0", "/some/path"])
        .assert()
        .failure()
        .code(2);
}

/// A positive `--threads` passes clap; the failure is the missing path (I/O),
/// proving execution proceeded past argument parsing.
#[test]
fn run_threads_positive_is_accepted_by_clap() {
    novomodelo()
        .args(["run", "--threads", "2", "/nonexistent/path"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("I/O error"));
}

#[test]
fn version_exits_0_and_contains_version_string() {
    let version = env!("CARGO_PKG_VERSION");
    novomodelo()
        .arg("version")
        .assert()
        .success()
        .stdout(predicate::str::contains(version))
        .stdout(predicate::str::contains(
            novomodelo_solver::active_solver_name(),
        ));
}

#[test]
fn version_exits_0_and_stdout_contains_novomodelo_prefix() {
    novomodelo()
        .arg("version")
        .assert()
        .success()
        .stdout(predicate::str::contains("novomodelo "));
}

#[test]
fn version_stdout_contains_active_solver() {
    let expected = format!("solver: {}", novomodelo_solver::active_solver_name());
    novomodelo()
        .arg("version")
        .assert()
        .success()
        .stdout(predicate::str::contains(expected));
}

#[test]
fn run_nonexistent_path_exits_2_with_io_error() {
    novomodelo()
        .args(["run", "/nonexistent/path"])
        .assert()
        .failure()
        .code(2)
        .stderr(predicate::str::contains("I/O error"));
}

#[test]
fn validate_nonexistent_path_exits_2() {
    novomodelo()
        .args(["validate", "/nonexistent/path"])
        .assert()
        .failure()
        .code(2);
}

#[test]
fn unknown_subcommand_exits_nonzero() {
    novomodelo().arg("unknown-subcommand").assert().failure();
}
