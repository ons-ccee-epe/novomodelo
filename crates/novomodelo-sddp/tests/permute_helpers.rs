//! Unit tests of the declaration-order permutation primitive
//! ([`common::permute::permute_case`]), relocated out of the shared `tests/common/`
//! aggregator so `mod common;` contributes no test to its consumers.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::path::Path;

mod common;
use common::permute::permute_case;

fn fixture_dir() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/nonzero_stage_fpha_override")
}

fn read_json(path: &Path) -> serde_json::Value {
    serde_json::from_str(&std::fs::read_to_string(path).expect("read")).expect("parse")
}

/// Seeds scanned by [`permute_case_reorders_a_whitelisted_registry_and_seeds_differ`].
/// A 2-element registry has only two possible orderings, so a single fixed
/// pair of seeds asserting they differ is a coin flip; scanning this range
/// deterministically (seed is a pure function input post-sort, see
/// `copy_dir_permuted`) observes both orderings instead of guessing one pair.
const REORDER_PROBE_SEEDS: std::ops::Range<u64> = 0..16;

/// Seeds must be able to produce different orderings of at least one
/// whitelisted registry — otherwise the "seeded" primitive would be
/// indistinguishable from a fixed permutation.
#[test]
fn permute_case_reorders_a_whitelisted_registry_and_seeds_differ() {
    let base_dir = fixture_dir();
    let base_hydros = read_json(&base_dir.join("system/hydros.json"))["hydros"].clone();
    let mut sorted_base = base_hydros.as_array().unwrap().clone();
    sorted_base.sort_by_key(ToString::to_string);

    let mut distinct_orderings = std::collections::HashSet::new();
    let mut any_reordered = false;

    for seed in REORDER_PROBE_SEEDS {
        let permuted = permute_case(&base_dir, seed);
        let hydros = read_json(&permuted.path().join("system/hydros.json"))["hydros"].clone();

        let mut sorted = hydros.as_array().unwrap().clone();
        sorted.sort_by_key(ToString::to_string);
        assert_eq!(
            sorted_base, sorted,
            "seed {seed}: shuffle must not add/drop/mutate records"
        );

        any_reordered |= hydros != base_hydros;
        distinct_orderings.insert(hydros.to_string());
    }

    assert!(
        any_reordered,
        "at least one seed in {REORDER_PROBE_SEEDS:?} must reorder the hydros registry \
         (base={base_hydros:?})"
    );
    assert!(
        distinct_orderings.len() >= 2,
        "seeds in {REORDER_PROBE_SEEDS:?} must be able to produce different orderings \
         of the hydros registry, got {distinct_orderings:?}"
    );
}

#[test]
fn permute_case_copies_non_whitelisted_files_byte_for_byte_and_skips_output() {
    let base_dir = fixture_dir();
    let permuted = permute_case(&base_dir, 7);

    for rel in ["config.json", "penalties.json"] {
        let expected = std::fs::read(base_dir.join(rel)).expect("read base");
        let actual = std::fs::read(permuted.path().join(rel)).expect("read permuted");
        assert_eq!(actual, expected, "{rel} must be byte-identical");
    }

    assert!(
        !permuted.path().join("output").exists(),
        "the permuted copy must never contain an output/ subtree"
    );
}

#[test]
#[should_panic(expected = "unclassified array")]
fn permute_case_panics_on_unclassified_array() {
    let tmp = tempfile::tempdir().expect("tempdir");
    std::fs::write(
        tmp.path().join("mystery.json"),
        r#"{"mystery_registry": [{"id": 1}]}"#,
    )
    .expect("write fixture");

    let _ = permute_case(tmp.path(), 42);
}
