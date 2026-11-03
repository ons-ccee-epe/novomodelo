//! Output-file bit-identity gate: every file a `novomodelo run` writes for a
//! committed deck must hash to the baseline in `fixtures/bit_identity/highs.json`.
//!
//! Only wall-clock, hostname and software-version fields are canonicalized; every
//! other byte is result-bearing. A one-time capture on the baseline commit writes
//! the file (`NOVOMODELO_BIT_IDENTITY_CAPTURE=1` with `bit_identity_full_sweep`).

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use arrow::array::{Array, AsArray, PrimitiveArray};
use arrow::datatypes::{
    ArrowPrimitiveType, DataType, Date32Type, Float64Type, Int8Type, Int32Type, Int64Type,
    UInt32Type, UInt64Type,
};
use novomodelo_sddp::test_support::decks::committed_decks;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use tempfile::TempDir;

mod common;
use common::{copy_dir_recursive, novomodelo, prepare_dual_calendar_boundary_case};

const BASELINE_COMMIT: &str = "93996013";
const CAPTURE_ENV: &str = "NOVOMODELO_BIT_IDENTITY_CAPTURE";
const CAP_ITERATIONS: u64 = 10;
const CAP_SCENARIOS: u64 = 10;

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

const RUN_IDENTITY_KEYS: [&str; 5] = [
    "started_at",
    "completed_at",
    "duration_seconds",
    "hostname",
    "software_version",
];

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct Cap {
    iterations: u64,
    scenarios: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct Entry {
    status: String,
    exit_code: i32,
    input_sha256: BTreeMap<String, String>,
    files: BTreeMap<String, String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct Baseline {
    generated_on: String,
    cap: Cap,
    decks: BTreeMap<String, Entry>,
}

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn baseline_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/bit_identity/highs.json")
}

fn deck(key: &str) -> (String, PathBuf) {
    (key.to_string(), repo_root().join(key))
}

fn hex(digest: impl AsRef<[u8]>) -> String {
    digest
        .as_ref()
        .iter()
        .fold(String::with_capacity(64), |mut hex, byte| {
            let _ = write!(hex, "{byte:02x}");
            hex
        })
}

fn sha256_hex(bytes: &[u8]) -> String {
    hex(Sha256::digest(bytes))
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

fn drop_keys(object: Option<&mut Value>, drop: impl Fn(&str) -> bool) {
    if let Some(map) = object.and_then(Value::as_object_mut) {
        map.retain(|key, _| !drop(key));
    }
}

fn sorted(value: Value) -> Value {
    match value {
        Value::Object(map) => {
            let mut entries: Vec<_> = map.into_iter().collect();
            entries.sort_by(|a, b| a.0.cmp(&b.0));
            Value::Object(entries.into_iter().map(|(k, v)| (k, sorted(v))).collect())
        }
        Value::Array(items) => Value::Array(items.into_iter().map(sorted).collect()),
        other => other,
    }
}

fn canonical_json(relpath: &str, bytes: &[u8]) -> Vec<u8> {
    let mut value: Value = serde_json::from_slice(bytes).unwrap();
    match relpath {
        "training/metadata.json" | "simulation/metadata.json" => {
            drop_keys(Some(&mut value), |key| RUN_IDENTITY_KEYS.contains(&key));
            if relpath == "training/metadata.json" {
                for section in ["setup", "solve_stats"] {
                    drop_keys(value.get_mut(section), |key| key.ends_with("_seconds"));
                }
            } else {
                drop_keys(value.get_mut("solve_stats"), |key| key == "solve_seconds");
            }
            if let Some(hosts) = value
                .pointer_mut("/distribution/hosts")
                .and_then(Value::as_array_mut)
            {
                for host in hosts {
                    drop_keys(Some(host), |key| key == "hostname");
                }
            }
        }
        "training/dictionaries/codes.json" => {
            drop_keys(Some(&mut value), |key| key == "generated_at");
        }
        other => unreachable!("{other} is not canonicalized as JSON"),
    }
    serde_json::to_vec(&sorted(value)).unwrap()
}

fn absorb_primitive<T: ArrowPrimitiveType, const N: usize>(
    hasher: &mut Sha256,
    array: &PrimitiveArray<T>,
    bytes: impl Fn(T::Native) -> [u8; N],
) {
    for value in array {
        match value {
            Some(v) => {
                hasher.update([1]);
                hasher.update(bytes(v));
            }
            None => hasher.update([0]),
        }
    }
}

fn absorb_column(hasher: &mut Sha256, name: &str, array: &dyn Array) {
    match array.data_type() {
        DataType::Float64 => absorb_primitive(hasher, array.as_primitive::<Float64Type>(), |v| {
            v.to_bits().to_le_bytes()
        }),
        DataType::Int32 => absorb_primitive(hasher, array.as_primitive::<Int32Type>(), |v| {
            v.to_le_bytes()
        }),
        DataType::Int64 => absorb_primitive(hasher, array.as_primitive::<Int64Type>(), |v| {
            v.to_le_bytes()
        }),
        DataType::UInt32 => absorb_primitive(hasher, array.as_primitive::<UInt32Type>(), |v| {
            v.to_le_bytes()
        }),
        DataType::UInt64 => absorb_primitive(hasher, array.as_primitive::<UInt64Type>(), |v| {
            v.to_le_bytes()
        }),
        DataType::Int8 => absorb_primitive(hasher, array.as_primitive::<Int8Type>(), |v| {
            v.to_le_bytes()
        }),
        DataType::Date32 => absorb_primitive(hasher, array.as_primitive::<Date32Type>(), |v| {
            v.to_le_bytes()
        }),
        DataType::Boolean => {
            for value in array.as_boolean() {
                match value {
                    Some(v) => hasher.update([1, u8::from(v)]),
                    None => hasher.update([0]),
                }
            }
        }
        DataType::Utf8 => {
            for value in array.as_string::<i32>() {
                match value {
                    Some(s) => {
                        hasher.update([1]);
                        hasher.update((s.len() as u64).to_le_bytes());
                        hasher.update(s.as_bytes());
                    }
                    None => hasher.update([0]),
                }
            }
        }
        other => panic!("parquet column {name} has the unhandled type {other}"),
    }
}

fn canonical_parquet_digest(path: &Path) -> String {
    let builder = ParquetRecordBatchReaderBuilder::try_new(fs::File::open(path).unwrap()).unwrap();
    let schema = builder.schema().clone();
    let kept: Vec<usize> = (0..schema.fields().len())
        .filter(|&i| !schema.field(i).name().ends_with("_ms"))
        .collect();
    let mut columns: Vec<Sha256> = kept.iter().map(|_| Sha256::new()).collect();
    let mut rows = 0u64;
    for batch in builder.build().unwrap() {
        let batch = batch.unwrap();
        rows += batch.num_rows() as u64;
        for (hasher, &i) in columns.iter_mut().zip(&kept) {
            absorb_column(hasher, schema.field(i).name(), batch.column(i));
        }
    }
    let mut file = Sha256::new();
    file.update(rows.to_le_bytes());
    for (hasher, &i) in columns.into_iter().zip(&kept) {
        let field = schema.field(i);
        for text in [field.name().clone(), field.data_type().to_string()] {
            file.update((text.len() as u64).to_le_bytes());
            file.update(text.as_bytes());
        }
        file.update(hasher.finalize());
    }
    hex(file.finalize())
}

fn canonical_manifest(bytes: &[u8]) -> Vec<u8> {
    let mut manifest = novomodelo_io::deserialize_checkpoint_manifest(bytes).unwrap();
    manifest.created_at.clear();
    manifest.software_version.clear();
    novomodelo_io::serialize_checkpoint_manifest(&manifest)
}

fn file_digest(relpath: &str, path: &Path) -> String {
    if relpath.ends_with(".parquet") {
        return canonical_parquet_digest(path);
    }
    let bytes = fs::read(path).unwrap();
    match relpath {
        "training/metadata.json"
        | "simulation/metadata.json"
        | "training/dictionaries/codes.json" => sha256_hex(&canonical_json(relpath, &bytes)),
        "policy/manifest.bin" => sha256_hex(&canonical_manifest(&bytes)),
        _ => sha256_hex(&bytes),
    }
}

fn cap_config(case: &Path) {
    fn cap(value: &mut Value, limit: u64) {
        if let Some(n) = value.as_u64() {
            *value = n.min(limit).into();
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

fn run_deck(key: &str, dir: &Path) -> (TempDir, i32) {
    let work = TempDir::new().unwrap();
    let case = work.path().join("case");
    if key == BOUNDARY_KEY {
        prepare_dual_calendar_boundary_case(&case);
    } else {
        copy_dir_recursive(dir, &case);
    }
    cap_config(&case);
    let run = novomodelo()
        .arg("run")
        .arg(&case)
        .arg("--output")
        .arg(work.path().join("out"))
        .args(["--threads", "1", "--comm-backend", "local", "--quiet"])
        .output()
        .unwrap();
    let exit_code = run
        .status
        .code()
        .expect("the run must exit, not die on a signal");
    (work, exit_code)
}

fn entry_for(key: &str, dir: &Path) -> Entry {
    let input_sha256 = list_files(dir)
        .into_iter()
        .map(|rel| {
            let digest = sha256_hex(&fs::read(dir.join(&rel)).unwrap());
            (rel, digest)
        })
        .collect();
    let (work, exit_code) = run_deck(key, dir);
    let out = work.path().join("out");
    let files = if out.exists() {
        list_files(&out)
            .into_iter()
            .map(|rel| {
                let digest = file_digest(&rel, &out.join(&rel));
                (rel, digest)
            })
            .collect()
    } else {
        BTreeMap::new()
    };
    Entry {
        status: if exit_code == 0 { "ok" } else { "rejected" }.to_string(),
        exit_code,
        input_sha256,
        files,
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

fn diff_hashes(
    prefix: &str,
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
                "{prefix}{deck}: {path}: baseline {} actual {}",
                show(baseline.get(path)),
                show(actual.get(path))
            )
        })
        .collect()
}

fn compare(deck: &str, baseline: &Entry, actual: &Entry) -> Vec<String> {
    let inputs = diff_hashes(
        "input changed: ",
        deck,
        &baseline.input_sha256,
        &actual.input_sha256,
    );
    if !inputs.is_empty() {
        return inputs;
    }
    let mut diffs = Vec::new();
    if (&baseline.status, baseline.exit_code) != (&actual.status, actual.exit_code) {
        diffs.push(format!(
            "{deck}: baseline {} (exit {}) actual {} (exit {})",
            baseline.status, baseline.exit_code, actual.status, actual.exit_code
        ));
    }
    diffs.extend(diff_hashes("", deck, &baseline.files, &actual.files));
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
        "output bit-identity violations:\n{}",
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
fn bit_identity_default_subset() {
    let swept = sweep(vec![deck(D55_KEY), deck(BOUNDARY_KEY)]);
    assert_eq!(swept.len(), 2);
    assert_matches_baseline(&load_baseline(), &swept);
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "full committed-deck output sweep"
)]
fn bit_identity_full_sweep() {
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
    use std::sync::Arc;

    use arrow::array::{Float64Array, Int64Array, RecordBatch};
    use arrow::datatypes::{Field, Schema};
    use parquet::arrow::ArrowWriter;

    use super::*;

    const CONVERGENCE: &str = "training/convergence.parquet";

    fn d55_baseline_entry() -> Entry {
        load_baseline().decks.remove(D55_KEY).unwrap()
    }

    #[test]
    fn canonical_json_ignores_run_identity_and_timing() {
        let metadata = |host: &str, version: &str, started: &str, seconds: f64| {
            serde_json::json!({
                "hostname": host,
                "software_version": version,
                "started_at": started,
                "completed_at": started,
                "duration_seconds": seconds,
                "status": "complete",
                "setup": { "load_seconds": seconds, "other": 1 },
                "solve_stats": { "forward_solve_seconds": seconds, "total_lp_solves": 240 },
                "distribution": { "hosts": [{ "hostname": host, "ranks": [0] }] },
            })
            .to_string()
        };
        let a = metadata("host-a", "0.19.1", "2026-01-01T00:00:00Z", 0.5);
        let b = metadata("host-b", "0.20.0", "2026-02-02T00:00:00Z", 9.5);
        assert_eq!(
            canonical_json("training/metadata.json", a.as_bytes()),
            canonical_json("training/metadata.json", b.as_bytes())
        );

        let changed = a.replace("complete", "failed");
        assert_ne!(
            canonical_json("training/metadata.json", a.as_bytes()),
            canonical_json("training/metadata.json", changed.as_bytes())
        );
    }

    #[test]
    fn canonical_json_sorts_keys() {
        assert_eq!(
            canonical_json(
                "training/dictionaries/codes.json",
                br#"{"b":1,"generated_at":"x","a":{"d":1,"c":2}}"#
            ),
            br#"{"a":{"c":2,"d":1},"b":1}"#
        );
    }

    #[test]
    fn canonical_parquet_digest_ignores_ms_columns_and_nothing_else() {
        let dir = TempDir::new().unwrap();
        let write = |name: &str, value: f64, solve_ms: i64| {
            let schema = Arc::new(Schema::new(vec![
                Field::new("value", DataType::Float64, false),
                Field::new("solve_time_ms", DataType::Int64, false),
            ]));
            let batch = RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(Float64Array::from(vec![value, 2.0])),
                    Arc::new(Int64Array::from(vec![solve_ms, solve_ms])),
                ],
            )
            .unwrap();
            let path = dir.path().join(name);
            let mut writer =
                ArrowWriter::try_new(fs::File::create(&path).unwrap(), schema, None).unwrap();
            writer.write(&batch).unwrap();
            writer.close().unwrap();
            canonical_parquet_digest(&path)
        };
        assert_eq!(write("a.parquet", 1.0, 5), write("b.parquet", 1.0, 900));
        assert_ne!(write("c.parquet", 1.0, 5), write("d.parquet", 1.5, 5));
    }

    #[test]
    fn canonical_manifest_ignores_created_at_and_software_version() {
        let (work, exit_code) = run_deck(D55_KEY, &repo_root().join(D55_KEY));
        assert_eq!(exit_code, 0);
        let original = fs::read(work.path().join("out/policy/manifest.bin")).unwrap();
        let mut manifest = novomodelo_io::deserialize_checkpoint_manifest(&original).unwrap();
        manifest.created_at = "1970-01-01T00:00:00Z".to_string();
        manifest.software_version = "0.0.1".to_string();
        let restamped = novomodelo_io::serialize_checkpoint_manifest(&manifest);
        assert_ne!(original, restamped);
        assert_eq!(
            canonical_manifest(&original),
            canonical_manifest(&restamped)
        );
    }

    #[test]
    fn compare_names_the_deck_and_file_of_an_altered_output_hash() {
        let baseline = d55_baseline_entry();
        let mut actual = baseline.clone();
        let hash = actual.files.get_mut(CONVERGENCE).unwrap();
        let first = if hash.starts_with('0') { "1" } else { "0" };
        hash.replace_range(..1, first);

        let diffs = compare(D55_KEY, &baseline, &actual);
        assert_eq!(diffs.len(), 1);
        assert!(diffs[0].contains(D55_KEY) && diffs[0].contains(CONVERGENCE));
    }

    #[test]
    fn compare_flags_missing_and_extra_files_both_ways() {
        let baseline = d55_baseline_entry();
        let mut actual = baseline.clone();
        actual.files.remove(CONVERGENCE);
        actual
            .files
            .insert("training/unexpected.bin".to_string(), "0".repeat(64));

        let diffs = compare(D55_KEY, &baseline, &actual).join("\n");
        assert!(diffs.contains(&format!("{CONVERGENCE}: baseline")));
        assert!(diffs.contains("actual <missing>"));
        assert!(diffs.contains("training/unexpected.bin: baseline <missing>"));
    }

    #[test]
    fn compare_labels_an_altered_input_hash_as_an_input_change() {
        let baseline = d55_baseline_entry();
        let mut actual = baseline.clone();
        let hash = actual.input_sha256.values_mut().next().unwrap();
        let first = if hash.starts_with('0') { "1" } else { "0" };
        hash.replace_range(..1, first);
        actual.files.insert(CONVERGENCE.to_string(), "0".repeat(64));

        let diffs = compare(D55_KEY, &baseline, &actual);
        assert_eq!(diffs.len(), 1);
        assert!(diffs[0].starts_with("input changed:"));
    }

    #[test]
    #[should_panic(expected = "highs.json already exists")]
    fn capture_refuses_to_overwrite_an_existing_baseline() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("highs.json");
        fs::write(&path, "{}").unwrap();
        assert_capturable(&path);
    }
}
