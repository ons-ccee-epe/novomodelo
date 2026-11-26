#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, missing_docs)]

use std::path::Path;
use std::process::Command;

#[test]
fn infrastructure_genericity_no_sddp_references() {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let workspace_root = Path::new(manifest_dir).join("..").join("..");
    let src_path = Path::new(manifest_dir).join("src");

    let output = Command::new("grep")
        .args(["-riE", "sddp"])
        .arg(&src_path)
        .current_dir(&workspace_root)
        .output()
        .expect("infrastructure_genericity: failed to execute grep");

    assert_eq!(
        output.status.code(),
        Some(1),
        "grep found algorithm-specific references in novomodelo-io/src/:\n{}",
        String::from_utf8_lossy(&output.stdout)
    );
}

/// Assemble a grep-pattern fragment from chars so this gate does not embed the
/// forbidden idiom verbatim (the project grep-gate obfuscation convention).
fn token(chars: &[char]) -> String {
    chars.iter().collect()
}

/// The `stage_id -> study-index` inverse map has exactly two sanctioned owners:
/// `StageIdResolver` (`novomodelo-io/src/stage_resolve.rs`) and novomodelo-core's
/// `build_stage_index` (`novomodelo-core/src/system/`, upstream of novomodelo-io and
/// allow-listed). This gate fails if an inline construction of that map
/// reappears in the consolidated downstream paths, so a future edit routes
/// through the resolver instead of deriving a third copy that can drift.
#[test]
fn no_inline_stage_id_index_map_outside_resolver() {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let workspace_root = Path::new(manifest_dir).join("..").join("..");

    let tuple_idiom = token(&[
        '\\', '(', 's', '\\', '.', 'i', 'd', ',', ' ', '?', 'i', 'd', 'x', '\\', ')',
    ]);
    let map_decl = token(&[
        's', 't', 'a', 'g', 'e', '_', 'i', 'd', '_', 't', 'o', '_', 'i', 'd', 'x', ':', ' ', '?',
        '(', 'H', 'a', 's', 'h', 'M', 'a', 'p', '|', 'B', 'T', 'r', 'e', 'e', 'M', 'a', 'p', ')',
    ]);
    let pattern = format!("{tuple_idiom}|{map_decl}");

    let output = Command::new("grep")
        .args(["-rInE", &pattern])
        .arg("crates/novomodelo-io/src/pipeline.rs")
        .arg("crates/novomodelo-io/src/resolution")
        .arg("crates/novomodelo-sddp/src/lp/builder")
        .current_dir(&workspace_root)
        .output()
        .expect("stage_id index-map gate: failed to execute grep");

    assert_eq!(
        output.status.code(),
        Some(1),
        "an inline stage-id to study-index inverse-map idiom reappeared outside the \
         sanctioned owners (StageIdResolver / novomodelo-core build_stage_index):\n{}",
        String::from_utf8_lossy(&output.stdout)
    );
}
