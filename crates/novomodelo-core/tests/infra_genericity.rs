//! Runs `scripts/ci/check-infra-genericity.sh` and asserts it exits successfully.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::Path;
use std::process::Command;

#[test]
fn infra_genericity_gate() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest_dir
        .parent()
        .expect("cobre-core has a parent directory")
        .parent()
        .expect("crates/ has a parent directory");

    let script = workspace_root.join("scripts/ci/check-infra-genericity.sh");

    assert!(
        script.exists(),
        "Gate script not found at {}: run from the workspace root or ensure the script exists",
        script.display()
    );

    let output = Command::new("bash")
        .arg(&script)
        .current_dir(workspace_root)
        .output()
        .expect("failed to execute check-infra-genericity.sh");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    assert!(
        output.status.success(),
        "Infra genericity gate FAILED.\n\
         --- stdout ---\n{stdout}\n\
         --- stderr ---\n{stderr}\n\
         Fix the flagged violations before committing."
    );
}
