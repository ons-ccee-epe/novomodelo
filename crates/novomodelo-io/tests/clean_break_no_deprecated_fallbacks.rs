#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, missing_docs)]

use std::path::Path;
use std::process::Command;

/// The 0.14 clean break retired every plan-introduced deprecated-with-fallback
/// spelling with no whitelist: the aliased-parquet-column extractor mechanism
/// (`extract_required_int32_aliased` / `extract_required_float64_aliased` /
/// their shared `resolve_required_column` resolver) and any `#[serde(alias =
/// "…")]` kept for a retired config spelling. This gate fails the moment either
/// reappears anywhere under `crates/cobre-io/src`.
///
/// Complements, and does not replace, the per-site behaviour tests that already
/// pin the individual removals: `retired_scheduler_spellings_are_deserialize_error`
/// (`src/config/training.rs`), `test_num_scenarios_removed_field_rejected`
/// (`src/stages.rs`), and the `domination_count` `FlatBuffers` conformance check
/// (`tests/flatbuffers_schema_conformance.rs`).
fn cobre_io_src(workspace_root: &Path) -> std::path::PathBuf {
    workspace_root.join("crates").join("cobre-io").join("src")
}

fn workspace_root() -> std::path::PathBuf {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    Path::new(manifest_dir).join("..").join("..")
}

#[test]
fn no_aliased_parquet_column_extractor_remains() {
    let root = workspace_root();
    let src = cobre_io_src(&root);

    let output = Command::new("grep")
        .args(["-rnE", r"extract_[A-Za-z0-9_]*_aliased"])
        .arg(&src)
        .current_dir(&root)
        .output()
        .expect("no_aliased_parquet_column_extractor_remains: failed to execute grep");

    assert_eq!(
        output.status.code(),
        Some(1),
        "an aliased parquet-column extractor (preferred-then-legacy column \
         resolution) reappeared under crates/cobre-io/src — the 0.14 clean break \
         removed this mechanism with no whitelist:\n{}",
        String::from_utf8_lossy(&output.stdout)
    );
}

#[test]
fn no_serde_alias_for_retired_spelling() {
    let root = workspace_root();
    let src = cobre_io_src(&root);

    let output = Command::new("grep")
        .args(["-rnF", "#[serde(alias"])
        .arg(&src)
        .current_dir(&root)
        .output()
        .expect("no_serde_alias_for_retired_spelling: failed to execute grep");

    assert_eq!(
        output.status.code(),
        Some(1),
        "a #[serde(alias = \"…\")] reappeared under crates/cobre-io/src — the 0.14 \
         clean break accepts exactly one spelling per config key, with no \
         deprecated-alias whitelist:\n{}",
        String::from_utf8_lossy(&output.stdout)
    );
}
