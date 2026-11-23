#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, missing_docs)]

use std::path::Path;
use std::process::Command;

/// `novomodelo-io` keeps no deprecated-with-fallback spelling, with no
/// whitelist: no aliased-parquet-column extractor (`extract_*_aliased`) and no
/// `#[serde(alias = "…")]`. This gate fails if either appears anywhere under
/// `crates/novomodelo-io/src`.
///
/// The per-site behaviour tests pin the individual refusals: `trial_point_and_opening_block_scheduler_spellings_are_deserialize_errors`
/// and `a_removed_selection_method_names_level1` (`src/config/training.rs`),
/// `test_num_scenarios_field_rejected_and_num_openings_accepted`
/// (`src/stages.rs`), and the `domination_count` `FlatBuffers` conformance check
/// (`tests/flatbuffers_schema_conformance.rs`).
fn novomodelo_io_src(workspace_root: &Path) -> std::path::PathBuf {
    workspace_root
        .join("crates")
        .join("novomodelo-io")
        .join("src")
}

fn workspace_root() -> std::path::PathBuf {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    Path::new(manifest_dir).join("..").join("..")
}

#[test]
fn no_aliased_parquet_column_extractor_under_io_sources() {
    let root = workspace_root();
    let src = novomodelo_io_src(&root);

    let output = Command::new("grep")
        .args(["-rnE", r"extract_[A-Za-z0-9_]*_aliased"])
        .arg(&src)
        .current_dir(&root)
        .output()
        .expect("no_aliased_parquet_column_extractor_under_io_sources: failed to execute grep");

    assert_eq!(
        output.status.code(),
        Some(1),
        "an aliased parquet-column extractor (preferred-then-fallback column \
         resolution) appeared under crates/novomodelo-io/src — every parquet column \
         is read under exactly one name:\n{}",
        String::from_utf8_lossy(&output.stdout)
    );
}

#[test]
fn no_serde_alias_under_io_sources() {
    let root = workspace_root();
    let src = novomodelo_io_src(&root);

    let output = Command::new("grep")
        .args(["-rnF", "#[serde(alias"])
        .arg(&src)
        .current_dir(&root)
        .output()
        .expect("no_serde_alias_under_io_sources: failed to execute grep");

    assert_eq!(
        output.status.code(),
        Some(1),
        "a #[serde(alias = \"…\")] appeared under crates/novomodelo-io/src — every config \
         key accepts exactly one spelling, with no alias:\n{}",
        String::from_utf8_lossy(&output.stdout)
    );
}
