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
        "grep found algorithm-specific references in cobre-io/src/:\n{}",
        String::from_utf8_lossy(&output.stdout)
    );
}

/// Assemble a grep-pattern fragment from chars so this gate does not embed the
/// forbidden idiom verbatim (the project grep-gate obfuscation convention).
fn token(chars: &[char]) -> String {
    chars.iter().collect()
}

/// The `stage_id -> study-index` inverse map has exactly two sanctioned owners:
/// `StageIdResolver` (`cobre-io/src/stage_resolve.rs`) and cobre-core's
/// `build_stage_index` (`cobre-core/src/system/`, upstream of cobre-io and
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
        .arg("crates/cobre-io/src/pipeline.rs")
        .arg("crates/cobre-io/src/resolution")
        .arg("crates/cobre-sddp/src/lp/builder")
        .current_dir(&workspace_root)
        .output()
        .expect("stage_id index-map gate: failed to execute grep");

    assert_eq!(
        output.status.code(),
        Some(1),
        "an inline stage-id to study-index inverse-map idiom reappeared outside the \
         sanctioned owners (StageIdResolver / cobre-core build_stage_index):\n{}",
        String::from_utf8_lossy(&output.stdout)
    );
}

/// The historical-input-standardization reshape retired the positional
/// per-plant MW-vector field on the anticipated-commitment record in favor of
/// the windowed `{thermal_id, start_date, end_date, value_mw}` record. Built
/// via [`token`] so this gate's own source does not itself match the
/// word-boundary pattern it greps for; the new singular field never
/// false-positives because the retired name is plural.
#[test]
fn historical_input_no_positional_values_mw_field() {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let workspace_root = Path::new(manifest_dir).join("..").join("..");

    let retired_field = token(&['v', 'a', 'l', 'u', 'e', 's', '_', 'm', 'w']);
    let pattern = format!(r"\b{retired_field}\b");

    let output = Command::new("grep")
        .args(["-rn", "--include=*.rs", &pattern, "crates/"])
        .current_dir(&workspace_root)
        .output()
        .expect("historical_input_no_positional_values_mw_field: failed to execute grep");

    assert_eq!(
        output.status.code(),
        Some(1),
        "the retired positional MW-vector field reappeared:\n{}",
        String::from_utf8_lossy(&output.stdout)
    );
}

/// The positional-length check on the old MW-vector field was retired; the
/// windowed reshape replaced it with calendar-derived coverage
/// (`check_commitment_coverage`). Built via [`token`] for the same
/// self-match reason as the sibling gates in this file.
#[test]
fn historical_input_no_required_anticipated_commitment_count() {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let workspace_root = Path::new(manifest_dir).join("..").join("..");

    let retired_check = token(&[
        'r', 'e', 'q', 'u', 'i', 'r', 'e', 'd', '_', 'a', 'n', 't', 'i', 'c', 'i', 'p', 'a', 't',
        'e', 'd', '_', 'c', 'o', 'm', 'm', 'i', 't', 'm', 'e', 'n', 't', '_', 'c', 'o', 'u', 'n',
        't',
    ]);
    let pattern = format!(r"\b{retired_check}\b");

    let output = Command::new("grep")
        .args(["-rnE", "--include=*.rs", &pattern, "crates/"])
        .current_dir(&workspace_root)
        .output()
        .expect(
            "historical_input_no_required_anticipated_commitment_count: failed to execute grep",
        );

    assert_eq!(
        output.status.code(),
        Some(1),
        "the retired positional-length commitment check reappeared:\n{}",
        String::from_utf8_lossy(&output.stdout)
    );
}

/// The three per-surface date-window validators (`RawAnticipatedCommitmentHistory`
/// recent-observations pair, past-defluences pair, and the inflow-history-row
/// overlap check) were unified onto the single shared windowed-record validator
/// in `cobre-io`. This gate fails if any of their retired names resurface.
/// Each name is built via [`token`] for the same self-match reason as the
/// sibling gates in this file.
#[test]
fn historical_input_no_retired_per_surface_date_window_validators() {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let workspace_root = Path::new(manifest_dir).join("..").join("..");

    let retired_validators = [
        token(&[
            'v', 'a', 'l', 'i', 'd', 'a', 't', 'e', '_', 'r', 'e', 'c', 'e', 'n', 't', '_', 'o',
            'b', 's', 'e', 'r', 'v', 'a', 't', 'i', 'o', 'n', 's', '_', 'd', 'a', 't', 'e', 's',
        ]),
        token(&[
            'v', 'a', 'l', 'i', 'd', 'a', 't', 'e', '_', 'r', 'e', 'c', 'e', 'n', 't', '_', 'o',
            'b', 's', 'e', 'r', 'v', 'a', 't', 'i', 'o', 'n', 's', '_', 'n', 'o', '_', 'o', 'v',
            'e', 'r', 'l', 'a', 'p',
        ]),
        token(&[
            'v', 'a', 'l', 'i', 'd', 'a', 't', 'e', '_', 'p', 'a', 's', 't', '_', 'd', 'e', 'f',
            'l', 'u', 'e', 'n', 'c', 'e', 's', '_', 'd', 'a', 't', 'e', 's',
        ]),
        token(&[
            'v', 'a', 'l', 'i', 'd', 'a', 't', 'e', '_', 'p', 'a', 's', 't', '_', 'd', 'e', 'f',
            'l', 'u', 'e', 'n', 'c', 'e', 's', '_', 'n', 'o', '_', 'o', 'v', 'e', 'r', 'l', 'a',
            'p',
        ]),
        token(&[
            'v', 'a', 'l', 'i', 'd', 'a', 't', 'e', '_', 'w', 'i', 'n', 'd', 'o', 'w', 's', '_',
            'n', 'o', '_', 'o', 'v', 'e', 'r', 'l', 'a', 'p',
        ]),
    ];

    let mut hits = Vec::new();
    for validator in &retired_validators {
        let pattern = format!(r"\b{validator}\b");
        let output = Command::new("grep")
            .args(["-rnE", "--include=*.rs", &pattern, "crates/"])
            .current_dir(&workspace_root)
            .output()
            .unwrap_or_else(|e| {
                panic!(
                    "historical_input_no_retired_per_surface_date_window_validators: \
                     failed to execute grep for {validator}: {e}"
                )
            });
        if output.status.code() != Some(1) {
            hits.push(format!(
                "{validator}:\n{}",
                String::from_utf8_lossy(&output.stdout)
            ));
        }
    }

    assert!(
        hits.is_empty(),
        "a retired per-surface date-window validator reappeared:\n{}",
        hits.join("\n")
    );
}

/// The separate defluence-seeding resolver module named by the two tokens
/// below was retired; defluence-bucket seeding now routes through the shared
/// windowed-record resolver. Two sanctioned survivors carry the same two
/// tokens and each escapes the `\b<token>\b` pattern on a different side: the
/// LP-side splice callsite (`splice_transit_bucket_seed`) has a LEADING
/// `transit_` continuation, so there is no boundary right before the first
/// token; the test-fixture helpers (`bucket_seed_*`) have a TRAILING `_`
/// continuation, so there is no boundary right after the second token.
#[test]
fn historical_input_no_bucket_seed_resolver_module() {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let workspace_root = Path::new(manifest_dir).join("..").join("..");

    let retired_module = token(&['b', 'u', 'c', 'k', 'e', 't', '_', 's', 'e', 'e', 'd']);
    let pattern = format!(r"\b{retired_module}\b");

    let output = Command::new("grep")
        .args([
            "-rnE",
            "--include=*.rs",
            &pattern,
            "crates/cobre-sddp/src/setup/",
        ])
        .current_dir(&workspace_root)
        .output()
        .expect("historical_input_no_bucket_seed_resolver_module: failed to execute grep");

    assert_eq!(
        output.status.code(),
        Some(1),
        "the retired defluence-bucket seeding resolver module/symbol reappeared under \
         crates/cobre-sddp/src/setup/:\n{}",
        String::from_utf8_lossy(&output.stdout)
    );
}
