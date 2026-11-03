//! LP-fact and parity-hash bit-identity gate: every stage-LP template fact group
//! ([`template_fact_groups`]) and the `HiGHS` parity hash of a committed deck must
//! match `fixtures/bit_identity_lp.json`.
//!
//! Baseline groups are frozen; a group the current run adds is reported, never
//! failed, so a later phase can add a group without recapturing the baseline. A
//! one-time capture on the baseline commit writes the file
//! (`NOVOMODELO_BIT_IDENTITY_CAPTURE=1` with `bit_identity_lp_full_sweep`).

#![cfg(feature = "highs")]
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use novomodelo_io::Config;
use novomodelo_io::config::{SimulationSelection, StoppingRuleConfig};
use novomodelo_sddp::test_support::decks::committed_decks;
use novomodelo_sddp::test_support::template_fact_groups;
use novomodelo_solver::highs::HighsSolver;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use tempfile::TempDir;

use common::copy_toy_case;
use common::fresh_setup_with;
use common::parity_hash::compute_case_hash_at_dir;

const BASELINE_COMMIT: &str = "93996013";
const CAPTURE_ENV: &str = "NOVOMODELO_BIT_IDENTITY_CAPTURE";
const CAP_ITERATIONS: u32 = 10;
const CAP_SCENARIOS: u32 = 10;

const D55_KEY: &str = "examples/deterministic/d55-post-study-anticipated-lanes";
const BOUNDARY_KEY: &str = "crates/novomodelo-sddp/tests/gate_fixtures/dual_calendar_boundary";

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

fn entry_for(key: &str, dir: &Path) -> Entry {
    let input_sha256 = list_files(dir)
        .into_iter()
        .map(|rel| {
            let digest = sha256_hex(&fs::read(dir.join(&rel)).unwrap());
            (rel, digest)
        })
        .collect();
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
    if !inputs.is_empty() {
        return inputs;
    }
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
    let mut diffs: Vec<String> = groups
        .changed
        .iter()
        .chain(&groups.missing)
        .map(|group| {
            format!(
                "{deck}: {group}: baseline {} actual {}",
                show(&baseline.lp_facts, group),
                show(&actual.lp_facts, group)
            )
        })
        .collect();
    if baseline.parity_hash != actual.parity_hash {
        diffs.push(format!(
            "{deck}: parity_hash: baseline {:?} actual {:?}",
            baseline.parity_hash, actual.parity_hash
        ));
    }
    diffs
}

fn load_baseline() -> Baseline {
    let path = baseline_path();
    let bytes = fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    serde_json::from_slice(&bytes).unwrap()
}

fn assert_matches_baseline(baseline: &Baseline, swept: &BTreeMap<String, Entry>) {
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
    assert_matches_baseline(&load_baseline(), &swept);
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
    assert_matches_baseline(&baseline, &swept);
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
    #[should_panic(expected = "bit_identity_lp.json already exists")]
    fn capture_refuses_to_overwrite_an_existing_baseline() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("bit_identity_lp.json");
        fs::write(&path, "{}").unwrap();
        assert_capturable(&path);
    }
}
