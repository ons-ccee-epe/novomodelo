//! LP-fact and parity-hash bit-identity gate: every stage-LP template fact group
//! ([`template_fact_groups`]) and the `HiGHS` parity hash of a committed deck must
//! match `fixtures/bit_identity_lp.json`.
//!
//! Baseline groups are frozen; a group the current run adds is reported, never
//! failed, so a later phase can add a group without recapturing the baseline. A
//! one-time capture on the baseline commit writes the file
//! (`NOVOMODELO_BIT_IDENTITY_CAPTURE=1` with `bit_identity_lp_full_sweep`).
//!
//! The migrated leg runs the committed decks as they are: every fact group and
//! parity hash equals the baseline unless `expected_deltas.json` lists it
//! under the deck's `lp_facts` (a listed name must differ), and the inputs equal
//! the baseline too, except the migrated post-study files of a deck that has a
//! legacy-input overlay. The legacy leg rebuilds each such deck from its overlay
//! and holds it to the same baseline, input hashes included.

#![cfg(feature = "highs")]
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use novomodelo_io::config::{SimulationSelection, StoppingRuleConfig};
use novomodelo_io::{Config, validate_case};
use novomodelo_sddp::test_support::decks::committed_decks;
use novomodelo_sddp::test_support::legacy_inputs::{flatten_key, materialize, overlay_dir};
use novomodelo_sddp::test_support::template_fact_groups;
use novomodelo_solver::highs::HighsSolver;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use tempfile::TempDir;

use common::copy_toy_case;
use common::fresh_setup_with;
use common::parity_hash::compute_case_hash_at_dir;

const BASELINE_COMMIT: &str = "55a2181e";
const CAPTURE_ENV: &str = "NOVOMODELO_BIT_IDENTITY_CAPTURE";
const CAP_ITERATIONS: u32 = 10;
const CAP_SCENARIOS: u32 = 10;

const D55_KEY: &str = "examples/deterministic/d55-post-study-anticipated-lanes";
const BOUNDARY_KEY: &str = "crates/novomodelo-sddp/tests/gate_fixtures/dual_calendar_boundary";

const PARITY_HASH: &str = "parity_hash";

const POST_STUDY_FILE: &str = "post_study_stages.json";
const MIGRATED_POST_STUDY_FILES: [&str; 2] = [POST_STUDY_FILE, "source/post_study_stages.json"];

const PRODUCT_SOURCES: [&str; 8] = [
    "crates/novomodelo-core/src",
    "crates/novomodelo-io/src",
    "crates/novomodelo-sddp/src",
    "crates/novomodelo-cli/src",
    "crates/novomodelo-stochastic/src",
    "crates/novomodelo-solver/src",
    "crates/novomodelo-comm/src",
    "crates/novomodelo-python/src",
];

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct Cap {
    iterations: u32,
    scenarios: u32,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct Entry {
    input_sha256: BTreeMap<String, String>,
    lp_facts: BTreeMap<String, String>,
    parity_hash: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct Baseline {
    generated_on: String,
    cap: Cap,
    decks: BTreeMap<String, Entry>,
}

#[derive(Debug, Deserialize)]
struct ExpectedList {
    decks: BTreeMap<String, ExpectedDeck>,
}

#[derive(Debug, Deserialize)]
struct ExpectedDeck {
    #[serde(default)]
    lp_facts: Vec<String>,
}

#[derive(Debug, Default, PartialEq, Eq)]
struct GroupDiff {
    changed: Vec<String>,
    missing: Vec<String>,
    additive: Vec<String>,
}

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn baseline_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/bit_identity_lp.json")
}

fn expected_deltas_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../novomodelo-cli/tests/fixtures/bit_identity/expected_deltas.json")
}

fn expected_lp_facts(deck: &str) -> Vec<String> {
    let path = expected_deltas_path();
    let bytes = fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let mut list: ExpectedList = serde_json::from_slice(&bytes).unwrap();
    list.decks
        .remove(deck)
        .map_or_else(Vec::new, |deck| deck.lp_facts)
}

fn deck(key: &str) -> (String, PathBuf) {
    (key.to_string(), repo_root().join(key))
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .fold(String::with_capacity(64), |mut hex, byte| {
            let _ = write!(hex, "{byte:02x}");
            hex
        })
}

fn list_files(root: &Path) -> Vec<String> {
    fn walk(root: &Path, dir: &Path, files: &mut Vec<String>) {
        for entry in fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                walk(root, &path, files);
            } else {
                let components: Vec<_> = path
                    .strip_prefix(root)
                    .unwrap()
                    .components()
                    .map(|c| c.as_os_str().to_str().unwrap())
                    .collect();
                files.push(components.join("/"));
            }
        }
    }
    let mut files = Vec::new();
    walk(root, root, &mut files);
    files
}

fn cap_config(config: &mut Config) {
    if let Some(rules) = config.training.stopping_rules.as_mut() {
        for rule in rules {
            if let StoppingRuleConfig::IterationLimit { limit } = rule {
                *limit = (*limit).min(CAP_ITERATIONS);
            }
        }
    }
    if let Some(SimulationSelection::Sampled { num_scenarios }) =
        config.simulation.selection.as_mut()
    {
        *num_scenarios = (*num_scenarios).min(CAP_SCENARIOS);
    }
}

fn cap_config_file(case: &Path) {
    fn cap(value: &mut Value, limit: u32) {
        if let Some(n) = value.as_u64() {
            *value = n.min(u64::from(limit)).into();
        }
    }
    let path = case.join("config.json");
    let mut config: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
    if let Some(rules) = config
        .pointer_mut("/training/stopping_rules")
        .and_then(Value::as_array_mut)
    {
        for rule in rules.iter_mut().filter(|r| r["type"] == "iteration_limit") {
            cap(&mut rule["limit"], CAP_ITERATIONS);
        }
    }
    if let Some(n) = config.pointer_mut("/simulation/selection/num_scenarios") {
        cap(n, CAP_SCENARIOS);
    }
    fs::write(&path, serde_json::to_vec_pretty(&config).unwrap()).unwrap();
}

fn parity_hash(dir: &Path) -> String {
    let work = TempDir::new().unwrap();
    let case = work.path().join("case");
    copy_toy_case(dir, &case);
    cap_config_file(&case);
    compute_case_hash_at_dir::<HighsSolver, _>(&case, HighsSolver::new)
}

fn input_hashes(dir: &Path) -> BTreeMap<String, String> {
    list_files(dir)
        .into_iter()
        .map(|rel| {
            let digest = sha256_hex(&fs::read(dir.join(&rel)).unwrap());
            (rel, digest)
        })
        .collect()
}

fn entry_from_inputs(key: &str, dir: &Path, input_sha256: BTreeMap<String, String>) -> Entry {
    let lp_facts = template_fact_groups(&fresh_setup_with(dir, cap_config))
        .into_iter()
        .map(|(group, bytes)| (group.to_string(), sha256_hex(&bytes)))
        .collect();
    Entry {
        input_sha256,
        lp_facts,
        parity_hash: (key != BOUNDARY_KEY).then(|| parity_hash(dir)),
    }
}

fn entry_for(key: &str, dir: &Path) -> Entry {
    entry_from_inputs(key, dir, input_hashes(dir))
}

fn deprecation_warnings(case: &Path) -> usize {
    let (_, report) = validate_case(case).unwrap();
    report
        .warnings
        .iter()
        .filter(|warning| warning.message.contains("is deprecated"))
        .count()
}

fn legacy_entry_for(
    key: &str,
    overlay: &Path,
    baseline_inputs: &BTreeMap<String, String>,
) -> Result<(Entry, usize), Vec<String>> {
    let staging = TempDir::new().unwrap();
    let legacy = staging.path().join("legacy");
    copy_toy_case(&repo_root().join(key), &legacy);
    materialize(overlay, &legacy).expect("the legacy overlay must materialize");
    let input_sha256 = input_hashes(&legacy);
    let drift = diff_inputs(key, baseline_inputs, &input_sha256);
    if !drift.is_empty() {
        return Err(drift);
    }
    let deprecations = deprecation_warnings(&legacy);
    Ok((entry_from_inputs(key, &legacy, input_sha256), deprecations))
}

fn sweep(decks: Vec<(String, PathBuf)>) -> BTreeMap<String, Entry> {
    decks
        .into_iter()
        .map(|(key, dir)| {
            let entry = entry_for(&key, &dir);
            (key, entry)
        })
        .collect()
}

fn compare_lp_facts(
    baseline: &BTreeMap<String, String>,
    current: &BTreeMap<String, String>,
) -> GroupDiff {
    let mut diff = GroupDiff::default();
    for (group, hash) in baseline {
        match current.get(group) {
            Some(current_hash) if current_hash == hash => {}
            Some(_) => diff.changed.push(group.clone()),
            None => diff.missing.push(group.clone()),
        }
    }
    diff.additive = current
        .keys()
        .filter(|group| !baseline.contains_key(*group))
        .cloned()
        .collect();
    diff
}

fn diff_inputs(
    deck: &str,
    baseline: &BTreeMap<String, String>,
    actual: &BTreeMap<String, String>,
) -> Vec<String> {
    let paths: BTreeSet<&String> = baseline.keys().chain(actual.keys()).collect();
    paths
        .into_iter()
        .filter(|path| baseline.get(*path) != actual.get(*path))
        .map(|path| {
            let show = |hash: Option<&String>| hash.map_or("<missing>", String::as_str).to_string();
            format!(
                "input changed: {deck}: {path}: baseline {} actual {}",
                show(baseline.get(path)),
                show(actual.get(path))
            )
        })
        .collect()
}

fn compare(deck: &str, baseline: &Entry, actual: &Entry) -> Vec<String> {
    let inputs = diff_inputs(deck, &baseline.input_sha256, &actual.input_sha256);
    if inputs.is_empty() {
        compare_outputs(deck, baseline, actual, &[])
    } else {
        inputs
    }
}

fn compare_migrated(deck: &str, baseline: &Entry, actual: &Entry) -> Vec<String> {
    if !overlay_dir(deck).is_dir() {
        return compare(deck, baseline, actual);
    }
    let unmigrated = |entry: &Entry| -> BTreeMap<String, String> {
        entry
            .input_sha256
            .iter()
            .filter(|(path, _)| !MIGRATED_POST_STUDY_FILES.contains(&path.as_str()))
            .map(|(path, hash)| (path.clone(), hash.clone()))
            .collect()
    };
    let mut diffs = diff_inputs(deck, &unmigrated(baseline), &unmigrated(actual));
    if baseline.input_sha256.get(POST_STUDY_FILE) == actual.input_sha256.get(POST_STUDY_FILE) {
        diffs.push(format!(
            "input not migrated: {deck}: {POST_STUDY_FILE} equals the baseline"
        ));
    }
    if diffs.is_empty() {
        compare_outputs(deck, baseline, actual, &expected_lp_facts(deck))
    } else {
        diffs
    }
}

fn compare_outputs(deck: &str, baseline: &Entry, actual: &Entry, listed: &[String]) -> Vec<String> {
    let groups = compare_lp_facts(&baseline.lp_facts, &actual.lp_facts);
    for group in &groups.additive {
        eprintln!("additive group: {deck}: {group}");
    }
    let show = |facts: &BTreeMap<String, String>, group: &String| {
        facts
            .get(group)
            .map_or("<missing>", String::as_str)
            .to_string()
    };
    let is_listed = |name: &str| listed.iter().any(|item| item == name);
    let mut diffs: Vec<String> = groups
        .changed
        .iter()
        .chain(&groups.missing)
        .filter(|group| !is_listed(group))
        .map(|group| {
            format!(
                "{deck}: {group}: baseline {} actual {}",
                show(&baseline.lp_facts, group),
                show(&actual.lp_facts, group)
            )
        })
        .collect();
    let parity_differs = baseline.parity_hash != actual.parity_hash;
    if parity_differs && !is_listed(PARITY_HASH) {
        diffs.push(format!(
            "{deck}: {PARITY_HASH}: baseline {:?} actual {:?}",
            baseline.parity_hash, actual.parity_hash
        ));
    }
    let differing: BTreeSet<&str> = groups
        .changed
        .iter()
        .chain(&groups.missing)
        .map(String::as_str)
        .chain(parity_differs.then_some(PARITY_HASH))
        .collect();
    diffs.extend(
        listed
            .iter()
            .filter(|item| !differing.contains(item.as_str()))
            .map(|item| format!("missing expected delta: {deck}: {item} equals the baseline")),
    );
    diffs
}

fn load_baseline() -> Baseline {
    let path = baseline_path();
    let bytes = fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    serde_json::from_slice(&bytes).unwrap()
}

fn assert_matches_baseline(
    baseline: &Baseline,
    swept: &BTreeMap<String, Entry>,
    compare: fn(&str, &Entry, &Entry) -> Vec<String>,
) {
    assert_eq!(baseline.generated_on, BASELINE_COMMIT);
    assert_eq!(
        baseline.cap,
        Cap {
            iterations: CAP_ITERATIONS,
            scenarios: CAP_SCENARIOS
        }
    );
    let mut diffs = Vec::new();
    for (key, actual) in swept {
        match baseline.decks.get(key) {
            Some(expected) => diffs.extend(compare(key, expected, actual)),
            None => diffs.push(format!("{key}: missing from the baseline")),
        }
    }
    assert!(
        diffs.is_empty(),
        "LP-fact bit-identity violations:\n{}",
        diffs.join("\n")
    );
}

fn assert_legacy_leg_matches_baseline(keys: &[String]) {
    let baseline = load_baseline();
    let mut swept = BTreeMap::new();
    let mut warning_diffs = Vec::new();
    for key in keys {
        let expected = baseline
            .decks
            .get(key)
            .unwrap_or_else(|| panic!("{key}: missing from the baseline"));
        let (entry, deprecations) =
            legacy_entry_for(key, &overlay_dir(key), &expected.input_sha256)
                .unwrap_or_else(|drift| panic!("{}", drift.join("\n")));
        if deprecations != 1 {
            warning_diffs.push(format!(
                "{key}: validate_case reported {deprecations} `is deprecated` warnings, expected 1"
            ));
        }
        swept.insert(key.clone(), entry);
    }
    assert_matches_baseline(&baseline, &swept, compare);
    assert!(
        warning_diffs.is_empty(),
        "legacy deprecation warning violations:\n{}",
        warning_diffs.join("\n")
    );
}

fn assert_capturable(path: &Path) {
    assert!(
        !path.exists(),
        "{} already exists; the baseline is captured once and never recaptured",
        path.display()
    );
    let status = Command::new("git")
        .current_dir(repo_root())
        .args(["diff", "--quiet", BASELINE_COMMIT, "--"])
        .args(PRODUCT_SOURCES)
        .status()
        .unwrap();
    assert!(
        status.success(),
        "capture refused: product sources differ from {BASELINE_COMMIT}"
    );
}

fn write_baseline(path: &Path, swept: BTreeMap<String, Entry>) {
    let baseline = Baseline {
        generated_on: BASELINE_COMMIT.to_string(),
        cap: Cap {
            iterations: CAP_ITERATIONS,
            scenarios: CAP_SCENARIOS,
        },
        decks: swept,
    };
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    let mut json = serde_json::to_string_pretty(&baseline).unwrap();
    json.push('\n');
    fs::write(path, json).unwrap();
}

#[test]
fn bit_identity_lp_default_subset() {
    let swept = sweep(vec![deck(D55_KEY), deck(BOUNDARY_KEY)]);
    assert_eq!(swept.len(), 2);
    assert_matches_baseline(&load_baseline(), &swept, compare_migrated);
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "full committed-deck LP-fact sweep"
)]
fn bit_identity_lp_full_sweep() {
    let capture = std::env::var(CAPTURE_ENV).is_ok_and(|v| v == "1");
    if capture {
        assert_capturable(&baseline_path());
    }
    let decks: Vec<_> = committed_decks()
        .into_iter()
        .map(|d| (d.key, d.dir))
        .chain([deck(BOUNDARY_KEY)])
        .collect();
    let swept = sweep(decks);
    if capture {
        write_baseline(&baseline_path(), swept);
        return;
    }
    let baseline = load_baseline();
    assert_eq!(
        baseline.decks.keys().collect::<Vec<_>>(),
        swept.keys().collect::<Vec<_>>(),
        "the baseline holds one entry per committed deck plus the boundary fixture"
    );
    assert_matches_baseline(&baseline, &swept, compare_migrated);
}

#[test]
fn bit_identity_lp_legacy_leg_default_subset() {
    assert_legacy_leg_matches_baseline(&[D55_KEY.to_string(), BOUNDARY_KEY.to_string()]);
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "full legacy-input LP-fact sweep"
)]
fn bit_identity_lp_legacy_leg_full_sweep() {
    let keys: Vec<String> = committed_decks()
        .into_iter()
        .map(|d| d.key)
        .chain([BOUNDARY_KEY.to_string()])
        .filter(|key| overlay_dir(key).is_dir())
        .collect();
    let overlay_root = overlay_dir(D55_KEY).parent().unwrap().to_path_buf();
    let overlays: BTreeSet<String> = fs::read_dir(overlay_root)
        .unwrap()
        .map(|entry| entry.unwrap().file_name().into_string().unwrap())
        .collect();
    let flattened: BTreeSet<String> = keys.iter().map(|key| flatten_key(key)).collect();
    assert_eq!(
        overlays, flattened,
        "every legacy overlay belongs to a swept deck"
    );
    assert_legacy_leg_matches_baseline(&keys);
}

#[cfg(test)]
mod tests {
    use super::*;

    const GROUP: &str = "time_value.delivery_stage_ids";

    fn facts(groups: &[(&str, &str)]) -> BTreeMap<String, String> {
        groups
            .iter()
            .map(|(group, hash)| ((*group).to_string(), (*hash).to_string()))
            .collect()
    }

    fn entry(lp_facts: BTreeMap<String, String>) -> Entry {
        Entry {
            input_sha256: facts(&[("config.json", "aa")]),
            lp_facts,
            parity_hash: Some("cc".to_string()),
        }
    }

    #[test]
    fn compare_names_the_deck_and_group_of_an_altered_hash() {
        let baseline = entry(facts(&[(GROUP, "00"), ("other", "11")]));
        let actual = entry(facts(&[(GROUP, "01"), ("other", "11")]));

        let diffs = compare(D55_KEY, &baseline, &actual);
        assert_eq!(diffs.len(), 1);
        assert!(diffs[0].contains(D55_KEY) && diffs[0].contains(GROUP));
        assert!(diffs[0].contains("baseline 00") && diffs[0].contains("actual 01"));
    }

    #[test]
    fn compare_fails_on_missing_group_and_only_reports_additive_group() {
        let baseline = facts(&[("kept", "11"), ("dropped", "22"), ("changed", "33")]);
        let current = facts(&[("kept", "11"), ("changed", "34"), ("added", "44")]);
        let diff = compare_lp_facts(&baseline, &current);
        assert_eq!(diff.missing, ["dropped"]);
        assert_eq!(diff.changed, ["changed"]);
        assert_eq!(diff.additive, ["added"]);

        let grown = facts(&[
            ("kept", "11"),
            ("dropped", "22"),
            ("changed", "33"),
            ("added", "44"),
        ]);
        let diff = compare_lp_facts(&baseline, &grown);
        assert!(diff.changed.is_empty() && diff.missing.is_empty());
        assert_eq!(diff.additive, ["added"]);
        assert!(compare(D55_KEY, &entry(baseline), &entry(grown)).is_empty());
    }

    #[test]
    fn compare_flags_a_differing_parity_hash() {
        let baseline = entry(facts(&[(GROUP, "00")]));
        let mut actual = baseline.clone();
        actual.parity_hash = Some("dd".to_string());

        let diffs = compare(D55_KEY, &baseline, &actual);
        assert_eq!(diffs.len(), 1);
        assert!(diffs[0].contains(D55_KEY) && diffs[0].contains("parity_hash"));
    }

    #[test]
    fn compare_labels_an_altered_input_hash_as_an_input_change() {
        let baseline = entry(facts(&[(GROUP, "00")]));
        let mut actual = baseline.clone();
        actual
            .input_sha256
            .insert("config.json".to_string(), "bb".to_string());
        actual.lp_facts.insert(GROUP.to_string(), "01".to_string());

        let diffs = compare(D55_KEY, &baseline, &actual);
        assert_eq!(diffs.len(), 1);
        assert!(diffs[0].starts_with("input changed:"));
    }

    #[test]
    fn legacy_leg_labels_a_doctored_overlay_file_as_an_input_change() {
        let doctored = TempDir::new().unwrap();
        copy_toy_case(&overlay_dir(D55_KEY), doctored.path());
        let file = doctored.path().join("post_study_stages.json");
        let mut bytes = fs::read(&file).unwrap();
        bytes.push(b'\n');
        fs::write(&file, bytes).unwrap();

        let baseline = load_baseline();
        let drift = legacy_entry_for(
            D55_KEY,
            doctored.path(),
            &baseline.decks[D55_KEY].input_sha256,
        )
        .unwrap_err();
        assert_eq!(drift.len(), 1);
        assert!(drift[0].starts_with("input changed:"));
        assert!(drift[0].contains("post_study_stages.json"));
    }

    fn names(items: &[&str]) -> Vec<String> {
        items.iter().map(|item| (*item).to_string()).collect()
    }

    #[test]
    fn a_listed_group_must_differ_and_an_unlisted_one_must_equal() {
        let baseline = entry(facts(&[("a", "11"), ("b", "22")]));
        let mut actual = baseline.clone();
        actual.lp_facts.insert("b".to_string(), "23".to_string());

        assert!(compare_outputs(D55_KEY, &baseline, &actual, &names(&["b"])).is_empty());

        let removed = compare_outputs(D55_KEY, &baseline, &actual, &[]);
        assert_eq!(removed.len(), 1);
        assert!(removed[0].contains(D55_KEY) && removed[0].contains("b: baseline 22 actual 23"));

        let spurious = compare_outputs(D55_KEY, &baseline, &actual, &names(&["a", "b"]));
        assert_eq!(
            spurious,
            [format!(
                "missing expected delta: {D55_KEY}: a equals the baseline"
            )]
        );
    }

    #[test]
    fn a_listed_group_the_run_drops_counts_as_differing() {
        let baseline = entry(facts(&[("a", "11"), ("b", "22")]));
        let mut actual = baseline.clone();
        actual.lp_facts.remove("b");

        assert!(compare_outputs(D55_KEY, &baseline, &actual, &names(&["b"])).is_empty());
        assert_eq!(compare_outputs(D55_KEY, &baseline, &actual, &[]).len(), 1);
    }

    #[test]
    fn the_parity_hash_is_listed_like_a_group() {
        let baseline = entry(facts(&[("a", "11")]));
        let mut actual = baseline.clone();
        actual.parity_hash = Some("dd".to_string());

        assert!(compare_outputs(D55_KEY, &baseline, &actual, &names(&[PARITY_HASH])).is_empty());
        let removed = compare_outputs(D55_KEY, &baseline, &actual, &[]);
        assert_eq!(removed.len(), 1);
        assert!(removed[0].contains(PARITY_HASH));

        let spurious = compare_outputs(D55_KEY, &baseline, &baseline, &names(&[PARITY_HASH]));
        assert_eq!(
            spurious,
            [format!(
                "missing expected delta: {D55_KEY}: {PARITY_HASH} equals the baseline"
            )]
        );
    }

    #[test]
    fn a_listed_group_may_not_hide_an_unlisted_one() {
        let baseline = entry(facts(&[("a", "11"), ("b", "22")]));
        let mut actual = baseline.clone();
        actual.lp_facts.insert("a".to_string(), "12".to_string());
        actual.lp_facts.insert("b".to_string(), "23".to_string());

        let diffs = compare_outputs(D55_KEY, &baseline, &actual, &names(&["b"]));
        assert_eq!(diffs.len(), 1);
        assert!(diffs[0].contains("a: baseline 11 actual 12"));
    }

    #[test]
    fn every_listed_group_of_a_migrated_deck_is_a_baseline_group_or_the_parity_hash() {
        let baseline = load_baseline();
        let path = expected_deltas_path();
        let list: ExpectedList = serde_json::from_slice(&fs::read(path).unwrap()).unwrap();
        for (deck, expected) in list.decks {
            for name in expected.lp_facts {
                assert!(
                    name == PARITY_HASH || baseline.decks[&deck].lp_facts.contains_key(&name),
                    "{deck}: {name} is not an LP-fact group of the baseline"
                );
            }
        }
    }

    #[test]
    #[should_panic(expected = "bit_identity_lp.json already exists")]
    fn capture_refuses_to_overwrite_an_existing_baseline() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("bit_identity_lp.json");
        fs::write(&path, "{}").unwrap();
        assert_capturable(&path);
    }
}
