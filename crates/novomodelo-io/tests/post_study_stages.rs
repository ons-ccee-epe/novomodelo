//! Integration tests for the standalone `post_study_stages.json` boundary input,
//! the sole post-horizon surface: the `System` flow (present → `Some`, absent →
//! `None`, additive), the semantic rejection paths (contiguity, first-start,
//! Rule 1 missing bound cell, the fixed post-horizon coverage rules (V2/V3/V5),
//! lead-exceeds-horizon), and declaration-order invariance — all through the
//! public `cobre_io::load_case` pipeline. Post-study deliveries are driven by
//! an anticipated thermal whose `LeadTime` reaches the post-study calendar, not
//! by a declared window.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::path::Path;

use cobre_io::{load_case, validate_case};
use tempfile::TempDir;

mod helpers;
use helpers::{make_minimal_case, write_file};

/// Write a `system/thermals.json` declaring thermal 86 as an anticipated
/// `LeadTime` plant on bus 1.
fn write_anticipated_thermal_86(root: &Path, lead_time_hours: f64) {
    write_file(
        root,
        "system/thermals.json",
        &format!(
            r#"{{
          "thermals": [
            {{
                "id": 86,
                "name": "T_ANT",
                "operational_start_date": "2024-01-01",
                "bus_id": 1,
                "cost_per_mwh": 10.0,
                "generation": {{ "min_mw": 0.0, "max_mw": 300.0 }},
                "anticipated_config": {{ "lead_time_hours": {lead_time_hours} }}
            }}
          ]
        }}"#
        ),
    );
}

/// Write an `initial_conditions.json` whose single `past_anticipated_commitments`
/// window tiles the leading study stage (id 0, `[2024-01-01, 2024-02-01)`) at
/// 0 MW — the bijection every anticipated thermal needs.
fn write_ic_with_past_commitment(root: &Path) {
    write_file(
        root,
        "initial_conditions.json",
        r#"{
          "storage": [],
          "filling_storage": [],
          "past_anticipated_commitments": [
            { "thermal_id": 86, "start_date": "2024-01-01", "end_date": "2024-02-01", "value_mw": 0.0 }
          ]
        }"#,
    );
}

/// Write `system/thermals.json` declaring thermal 86 as an anticipated
/// `LeadTime` plant on bus 1 that exits (decommissions) at continued stage id
/// `exit_stage_id` — used to place every post-study stage outside the plant's
/// commissioning window.
fn write_anticipated_thermal_86_with_exit(root: &Path, lead_time_hours: f64, exit_stage_id: i32) {
    write_file(
        root,
        "system/thermals.json",
        &format!(
            r#"{{
          "thermals": [
            {{
                "id": 86,
                "name": "T_ANT",
                "operational_start_date": "2024-01-01",
                "bus_id": 1,
                "cost_per_mwh": 10.0,
                "exit_stage_id": {exit_stage_id},
                "generation": {{ "min_mw": 0.0, "max_mw": 300.0 }},
                "anticipated_config": {{ "lead_time_hours": {lead_time_hours} }}
            }}
          ]
        }}"#
        ),
    );
}

/// Overwrite `stages.json` with a 4-stage finite-horizon calendar, each stage
/// 720h (30 days), starting 2024-01-01: stage 0 `[2024-01-01, 2024-01-31)`
/// through stage 3 `[2024-03-31, 2024-04-30)`. The post-study calendar in the
/// fixtures built on top of this starts exactly at the resulting horizon end,
/// 2024-04-30.
fn write_four_stage_study(root: &Path) {
    write_file(
        root,
        "stages.json",
        r#"{
          "policy_graph": {
            "type": "finite_horizon",
            "annual_discount_rate": 0.06,
            "transitions": [
              { "source_id": 0, "target_id": 1, "probability": 1.0 },
              { "source_id": 1, "target_id": 2, "probability": 1.0 },
              { "source_id": 2, "target_id": 3, "probability": 1.0 }
            ]
          },
          "stages": [
            { "id": 0, "start_date": "2024-01-01", "end_date": "2024-01-31", "blocks": [{ "id": 0, "name": "FLAT", "hours": 720.0 }], "num_openings": 10 },
            { "id": 1, "start_date": "2024-01-31", "end_date": "2024-03-01", "blocks": [{ "id": 0, "name": "FLAT", "hours": 720.0 }], "num_openings": 10 },
            { "id": 2, "start_date": "2024-03-01", "end_date": "2024-03-31", "blocks": [{ "id": 0, "name": "FLAT", "hours": 720.0 }], "num_openings": 10 },
            { "id": 3, "start_date": "2024-03-31", "end_date": "2024-04-30", "blocks": [{ "id": 0, "name": "FLAT", "hours": 720.0 }], "num_openings": 10 }
          ]
        }"#,
    );
}

/// Write `initial_conditions.json` for thermal 86 with one
/// `past_anticipated_commitments` window per `(start_date, end_date,
/// value_mw)` triple in `windows`.
fn write_ic_with_commitments(root: &Path, windows: &[(&str, &str, f64)]) {
    let entries: Vec<String> = windows
        .iter()
        .map(|(start, end, value_mw)| {
            format!(
                r#"{{ "thermal_id": 86, "start_date": "{start}", "end_date": "{end}", "value_mw": {value_mw} }}"#
            )
        })
        .collect();
    write_file(
        root,
        "initial_conditions.json",
        &format!(
            r#"{{
              "storage": [],
              "filling_storage": [],
              "past_anticipated_commitments": [{}]
            }}"#,
            entries.join(",\n")
        ),
    );
}

/// Write `post_study_stages.json`: 7 contiguous 720h stages starting at the
/// study horizon end (2024-04-30, matching [`write_four_stage_study`]), plus
/// one `thermal_bounds` entry against thermal 86 per `(post_study_stage_index,
/// min_mw, max_mw)` triple in `bounds`.
fn write_seven_stage_post_study(root: &Path, bounds: &[(usize, f64, f64)]) {
    const STAGE_STARTS: [&str; 7] = [
        "2024-04-30",
        "2024-05-30",
        "2024-06-29",
        "2024-07-29",
        "2024-08-28",
        "2024-09-27",
        "2024-10-27",
    ];
    let stages: Vec<String> = STAGE_STARTS
        .iter()
        .map(|start| format!(r#"{{ "start_date": "{start}", "duration_hours": 720.0 }}"#))
        .collect();
    let bound_entries: Vec<String> = bounds
        .iter()
        .map(|(j, min_mw, max_mw)| {
            format!(
                r#"{{ "thermal_id": 86, "post_study_stage_index": {j}, "cost_per_mwh": 100.0, "min_mw": {min_mw}, "max_mw": {max_mw} }}"#
            )
        })
        .collect();
    write_file(
        root,
        "post_study_stages.json",
        &format!(
            r#"{{ "stages": [{}], "thermal_bounds": [{}] }}"#,
            stages.join(",\n"),
            bound_entries.join(",\n")
        ),
    );
}

// ── present → System.post_study_stages Some, both collections sorted ──────

#[test]
fn test_present_carries_sorted_post_study_stages_onto_system() {
    let dir = TempDir::new().unwrap();
    make_minimal_case(&dir);
    // Stages and bounds declared out of canonical order; the reader sorts both.
    write_file(
        dir.path(),
        "post_study_stages.json",
        r#"{
          "stages": [
            { "start_date": "2024-03-01", "duration_hours": 744.0 },
            { "start_date": "2024-02-01", "duration_hours": 696.0 }
          ],
          "thermal_bounds": [
            { "thermal_id": 86, "post_study_stage_index": 1, "cost_per_mwh": 220.0, "min_mw": 0.0, "max_mw": 300.0 },
            { "thermal_id": 86, "post_study_stage_index": 0, "cost_per_mwh": 210.0, "min_mw": 0.0, "max_mw": 350.0 }
          ]
        }"#,
    );

    let system = load_case(dir.path()).unwrap_or_else(|e| panic!("expected Ok(System), got: {e}"));
    let ps = system
        .post_study_stages()
        .expect("post_study_stages must be Some when the file is present");

    assert_eq!(ps.stages.len(), 2);
    assert_eq!(
        ps.stages[0].start_date,
        chrono::NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
        "stages must be ascending by start_date"
    );
    assert_eq!(ps.thermal_bounds[0].post_study_stage_index, 0);
    assert_eq!(ps.thermal_bounds[1].post_study_stage_index, 1);
}

// ── absent → None, System is otherwise unchanged (additive/inert) ─────────

#[test]
fn test_absent_yields_none_and_is_additive() {
    let dir = TempDir::new().unwrap();
    make_minimal_case(&dir);

    let system = load_case(dir.path()).unwrap_or_else(|e| panic!("expected Ok(System), got: {e}"));
    assert!(
        system.post_study_stages().is_none(),
        "absent post_study_stages.json must leave System.post_study_stages None"
    );
    // The rest of the System is unaffected — the input is inert when absent.
    assert_eq!(system.n_buses(), 1);
    assert_eq!(system.n_stages(), 1);
}

// ── first post-study start != study end → Err naming the mismatch ─────────

#[test]
fn test_first_start_not_study_end_rejected() {
    let dir = TempDir::new().unwrap();
    make_minimal_case(&dir);
    write_file(
        dir.path(),
        "post_study_stages.json",
        r#"{
          "stages": [ { "start_date": "2024-03-01", "duration_hours": 744.0 } ],
          "thermal_bounds": []
        }"#,
    );

    let msg = load_case(dir.path()).unwrap_err().to_string();
    assert!(
        msg.contains("study horizon end") && msg.contains("2024-02-01"),
        "error should name the first-start / study-end mismatch, got: {msg}"
    );
}

// ── non-contiguous post-study stages → Err ────────────────────────────────

#[test]
fn test_non_contiguous_stages_rejected() {
    let dir = TempDir::new().unwrap();
    make_minimal_case(&dir);
    // stage 0 ends 2024-03-01 (Feb has 29 days in 2024), but stage 1 starts
    // 2024-04-01 — a one-month gap.
    write_file(
        dir.path(),
        "post_study_stages.json",
        r#"{
          "stages": [
            { "start_date": "2024-02-01", "duration_hours": 696.0 },
            { "start_date": "2024-04-01", "duration_hours": 720.0 }
          ],
          "thermal_bounds": []
        }"#,
    );

    let msg = load_case(dir.path()).unwrap_err().to_string();
    assert!(
        msg.contains("not date-contiguous"),
        "error should name the gap, got: {msg}"
    );
}

// ── anticipated lead exceeds the horizon with no post_study → Err ─────────

#[test]
fn test_lead_exceeds_horizon_without_post_study_rejected() {
    let dir = TempDir::new().unwrap();
    make_minimal_case(&dir);
    // LeadTime 1440h (60 days) exceeds the single 744h study stage; with no
    // post_study_stages.json declared, the plant can never deliver.
    write_anticipated_thermal_86(dir.path(), 1440.0);
    write_ic_with_past_commitment(dir.path());
    // No post_study_stages.json written.

    let msg = load_case(dir.path()).unwrap_err().to_string();
    assert!(
        msg.contains("can never deliver within the study horizon") && msg.contains("Thermal 86"),
        "error should name the lead-exceeds-horizon rejection, got: {msg}"
    );
}

// ── lead reaches a post-study stage with no bound cell → Err (Rule 1) ──────

#[test]
fn test_missing_post_study_bound_cell_rejected() {
    let dir = TempDir::new().unwrap();
    make_minimal_case(&dir);
    // LeadTime 744h: decided at study stage 0, delivered into post-study stage 0.
    write_anticipated_thermal_86(dir.path(), 744.0);
    write_ic_with_past_commitment(dir.path());
    // Post-study stage 0 is reached, but no bound cell is declared for it.
    write_file(
        dir.path(),
        "post_study_stages.json",
        r#"{
          "stages": [ { "start_date": "2024-02-01", "duration_hours": 696.0 } ],
          "thermal_bounds": []
        }"#,
    );

    let msg = load_case(dir.path()).unwrap_err().to_string();
    assert!(
        msg.contains("no thermal_bounds entry")
            && msg.contains("thermal_id 86")
            && msg.contains("post_study_stage_index 0"),
        "error should name the missing (thermal, stage) bound cell, got: {msg}"
    );
}

// ── a pre-study-decided post-study delivery loads once tiled (V2) ────────

#[test]
fn test_pre_study_decided_post_study_delivery_loads_when_tiled() {
    let dir = TempDir::new().unwrap();
    make_minimal_case(&dir);
    // LeadTime 1440h: post-study stage 0 (Feb) is decided pre-study
    // (fixed_post_study), while post-study stage 1 (Mar) is decided in-study
    // (carried). A commitment window tiling Feb makes the case load cleanly.
    write_anticipated_thermal_86(dir.path(), 1440.0);
    write_ic_with_commitments(
        dir.path(),
        &[
            ("2024-01-01", "2024-02-01", 0.0),
            ("2024-02-01", "2024-03-01", 0.0),
        ],
    );
    write_file(
        dir.path(),
        "post_study_stages.json",
        r#"{
          "stages": [
            { "start_date": "2024-02-01", "duration_hours": 696.0 },
            { "start_date": "2024-03-01", "duration_hours": 744.0 }
          ],
          "thermal_bounds": [
            { "thermal_id": 86, "post_study_stage_index": 1, "cost_per_mwh": 220.0, "min_mw": 0.0, "max_mw": 350.0 }
          ]
        }"#,
    );

    let system = load_case(dir.path()).unwrap_or_else(|e| {
        panic!("a tiled pre-study-decided post-study delivery must load, got: {e}")
    });
    assert!(system.post_study_stages().is_some());
}

// ── A valid reached delivery loads cleanly ────────────────────────────────────

#[test]
fn test_valid_covered_delivery_loads() {
    let dir = TempDir::new().unwrap();
    make_minimal_case(&dir);
    write_anticipated_thermal_86(dir.path(), 744.0);
    write_ic_with_past_commitment(dir.path());
    write_file(
        dir.path(),
        "post_study_stages.json",
        r#"{
          "stages": [ { "start_date": "2024-02-01", "duration_hours": 696.0 } ],
          "thermal_bounds": [
            { "thermal_id": 86, "post_study_stage_index": 0, "cost_per_mwh": 210.0, "min_mw": 0.0, "max_mw": 350.0 }
          ]
        }"#,
    );

    let system = load_case(dir.path())
        .unwrap_or_else(|e| panic!("a reached, bounded delivery must load, got: {e}"));
    assert!(system.post_study_stages().is_some());
}

// ── declaration-order invariance through System ───────────────────────────

#[test]
fn test_declaration_order_invariance_through_system() {
    let forward = r#"{
      "stages": [
        { "start_date": "2024-02-01", "duration_hours": 696.0 },
        { "start_date": "2024-03-01", "duration_hours": 744.0 }
      ],
      "thermal_bounds": [
        { "thermal_id": 86, "post_study_stage_index": 0, "cost_per_mwh": 210.0, "min_mw": 0.0, "max_mw": 350.0 },
        { "thermal_id": 86, "post_study_stage_index": 1, "cost_per_mwh": 220.0, "min_mw": 0.0, "max_mw": 300.0 }
      ]
    }"#;
    let reversed = r#"{
      "stages": [
        { "start_date": "2024-03-01", "duration_hours": 744.0 },
        { "start_date": "2024-02-01", "duration_hours": 696.0 }
      ],
      "thermal_bounds": [
        { "thermal_id": 86, "post_study_stage_index": 1, "cost_per_mwh": 220.0, "min_mw": 0.0, "max_mw": 300.0 },
        { "thermal_id": 86, "post_study_stage_index": 0, "cost_per_mwh": 210.0, "min_mw": 0.0, "max_mw": 350.0 }
      ]
    }"#;

    let dir1 = TempDir::new().unwrap();
    make_minimal_case(&dir1);
    write_file(dir1.path(), "post_study_stages.json", forward);

    let dir2 = TempDir::new().unwrap();
    make_minimal_case(&dir2);
    write_file(dir2.path(), "post_study_stages.json", reversed);

    let s1 = load_case(dir1.path()).unwrap();
    let s2 = load_case(dir2.path()).unwrap();
    assert_eq!(
        s1.post_study_stages(),
        s2.post_study_stages(),
        "post_study_stages must be identical regardless of declaration order"
    );
}

// ── horizon-straddle guard: reject a straddle, accept a split pair ───────

/// A single `past_anticipated_commitments` window whose span crosses the
/// study horizon end (2024-02-01, [`make_minimal_case`]'s single 744h stage)
/// is rejected: the window cannot deliver both in-study and post-study
/// coverage, and the rejection names the plant and the offending window's
/// dates.
#[test]
fn test_straddling_commitment_window_rejected() {
    let dir = TempDir::new().unwrap();
    make_minimal_case(&dir);
    write_anticipated_thermal_86(dir.path(), 744.0);
    write_ic_with_commitments(dir.path(), &[("2024-01-15", "2024-02-15", 0.0)]);

    let msg = load_case(dir.path()).unwrap_err().to_string();
    assert!(
        msg.contains("straddles the study horizon"),
        "expected the horizon-straddle rejection, got: {msg}"
    );
    assert!(
        msg.contains("Thermal 86") && msg.contains("2024-01-15") && msg.contains("2024-02-15"),
        "expected the message to name the plant and the offending window's dates, got: {msg}"
    );
}

/// The same overall span as
/// [`test_straddling_commitment_window_rejected`], but declared as two
/// windows split exactly at the horizon end (2024-02-01): one ending there
/// (purely in-study) and one starting there (purely post-study). Both
/// boundary-exact windows are accepted by the straddle guard, and — tiling
/// the leading study stage and the pre-study-decided post-study stage — the
/// deck loads cleanly end to end. Pins the guard's off-by-one boundary,
/// independent of `test_pre_study_decided_post_study_delivery_loads_when_tiled`'s
/// V2-tiling rationale.
#[test]
fn test_horizon_split_commitment_window_pair_loads_cleanly() {
    let dir = TempDir::new().unwrap();
    make_minimal_case(&dir);
    write_anticipated_thermal_86(dir.path(), 1440.0);
    write_ic_with_commitments(
        dir.path(),
        &[
            ("2024-01-01", "2024-02-01", 0.0),
            ("2024-02-01", "2024-03-01", 0.0),
        ],
    );
    write_file(
        dir.path(),
        "post_study_stages.json",
        r#"{
          "stages": [
            { "start_date": "2024-02-01", "duration_hours": 696.0 },
            { "start_date": "2024-03-01", "duration_hours": 744.0 }
          ],
          "thermal_bounds": [
            { "thermal_id": 86, "post_study_stage_index": 1, "cost_per_mwh": 220.0, "min_mw": 0.0, "max_mw": 350.0 }
          ]
        }"#,
    );

    let system = load_case(dir.path()).unwrap_or_else(|e| {
        panic!("a horizon-split commitment window pair must load cleanly, got: {e}")
    });
    assert!(system.post_study_stages().is_some());
}

// ── V2: fixed post-horizon windows must tile the class-4 stages ──────────

/// A `LeadTime` of 5040h over the 4×720h study calendar (2880h total) decides
/// the whole study horizon plus post-study stages 0-2 pre-study (the class-4
/// set); stages 3-6 are decided in-study (carried). Windows tile all four
/// study stages and all three class-4 post-study stages, and every carried
/// stage has its bound cell: a fully-tiled class-4 set loads cleanly.
#[test]
fn test_fixed_post_horizon_windows_tiling_class_four_stages_loads() {
    let dir = TempDir::new().unwrap();
    make_minimal_case(&dir);
    write_four_stage_study(dir.path());
    write_anticipated_thermal_86(dir.path(), 5040.0);
    write_ic_with_commitments(
        dir.path(),
        &[
            ("2024-01-01", "2024-01-31", 0.0),
            ("2024-01-31", "2024-03-01", 0.0),
            ("2024-03-01", "2024-03-31", 0.0),
            ("2024-03-31", "2024-04-30", 0.0),
            ("2024-04-30", "2024-05-30", 0.0),
            ("2024-05-30", "2024-06-29", 0.0),
            ("2024-06-29", "2024-07-29", 0.0),
        ],
    );
    write_seven_stage_post_study(
        dir.path(),
        &[
            (3, 0.0, 300.0),
            (4, 0.0, 300.0),
            (5, 0.0, 300.0),
            (6, 0.0, 300.0),
        ],
    );

    let system = load_case(dir.path())
        .unwrap_or_else(|e| panic!("a fully-tiled class-4 set must load cleanly, got: {e}"));
    assert!(system.post_study_stages().is_some());
}

/// The same class-4 set as above, but the window tiling post-study stage 2 is
/// omitted. The tiling rejection names index 2, tells the user to declare the
/// commitment with a committed 0 MW counting as explicit, and never advises
/// shortening the lead.
#[test]
fn test_untiled_fixed_post_horizon_stage_rejected() {
    let dir = TempDir::new().unwrap();
    make_minimal_case(&dir);
    write_four_stage_study(dir.path());
    write_anticipated_thermal_86(dir.path(), 5040.0);
    write_ic_with_commitments(
        dir.path(),
        &[
            ("2024-01-01", "2024-01-31", 0.0),
            ("2024-01-31", "2024-03-01", 0.0),
            ("2024-03-01", "2024-03-31", 0.0),
            ("2024-03-31", "2024-04-30", 0.0),
            ("2024-04-30", "2024-05-30", 0.0),
            ("2024-05-30", "2024-06-29", 0.0),
            // post-study stage 2 is left untiled.
        ],
    );
    write_seven_stage_post_study(
        dir.path(),
        &[
            (3, 0.0, 300.0),
            (4, 0.0, 300.0),
            (5, 0.0, 300.0),
            (6, 0.0, 300.0),
        ],
    );

    let msg = load_case(dir.path()).unwrap_err().to_string();
    let v2_line = msg
        .lines()
        .find(|l| l.contains("do not tile post-study"))
        .unwrap_or_else(|| panic!("expected a fixed post-horizon tiling rejection, got: {msg}"));
    assert!(
        v2_line.contains('2'),
        "message should name post-study stage index 2, got: {v2_line}"
    );
    assert!(
        v2_line.contains("committed 0 MW is explicit"),
        "message should say a committed 0 MW is explicit, got: {v2_line}"
    );
    assert!(
        !v2_line.contains("Shorten"),
        "the fixed post-horizon tiling rejection must not advise shortening \
         the lead, got: {v2_line}"
    );
}

/// A plant whose every reached post-study delivery is decided in-study
/// (class-4 empty) needs no post-horizon window at all — the leading study
/// window is enough. Mirrors `test_valid_covered_delivery_loads`'s fixture
/// deliberately: this test's subject is that V2 stays silent when it has
/// nothing to require, not the Rule 1 pairing that test already covers.
#[test]
fn test_empty_class_four_set_requires_no_post_horizon_window() {
    let dir = TempDir::new().unwrap();
    make_minimal_case(&dir);
    write_anticipated_thermal_86(dir.path(), 744.0);
    write_ic_with_past_commitment(dir.path());
    write_file(
        dir.path(),
        "post_study_stages.json",
        r#"{
          "stages": [ { "start_date": "2024-02-01", "duration_hours": 696.0 } ],
          "thermal_bounds": [
            { "thermal_id": 86, "post_study_stage_index": 0, "cost_per_mwh": 210.0, "min_mw": 0.0, "max_mw": 350.0 }
          ]
        }"#,
    );

    let system = load_case(dir.path()).unwrap_or_else(|e| {
        panic!(
            "an empty class-4 set must not raise the fixed post-horizon tiling rejection, got: {e}"
        )
    });
    assert!(system.post_study_stages().is_some());
}

/// An `exit_stage_id` of 1 decommissions the plant exactly at the study
/// horizon end (continued stage id 1, the first post-study stage), so all
/// three declared post-study stages are commissioning-inactive for this
/// plant and none land in the class-4 set. No post-horizon window is
/// declared and the case loads cleanly.
#[test]
fn test_commissioning_inactive_post_study_stage_requires_no_window() {
    let dir = TempDir::new().unwrap();
    make_minimal_case(&dir);
    write_anticipated_thermal_86_with_exit(dir.path(), 744.0, 1);
    write_ic_with_past_commitment(dir.path());
    write_file(
        dir.path(),
        "post_study_stages.json",
        r#"{
          "stages": [
            { "start_date": "2024-02-01", "duration_hours": 696.0 },
            { "start_date": "2024-03-01", "duration_hours": 744.0 },
            { "start_date": "2024-04-01", "duration_hours": 720.0 }
          ],
          "thermal_bounds": []
        }"#,
    );

    let system = load_case(dir.path()).unwrap_or_else(|e| {
        panic!("a commissioning-inactive post-study stage must not require a window, got: {e}")
    });
    assert!(system.post_study_stages().is_some());
}

/// Rule 1's missing-bound-cell rejection (post-study stage 3, carried) and
/// V2's tiling rejection (post-study stage 2, class-4) fire independently on
/// the same deck — neither short-circuits the other.
#[test]
fn test_missing_bound_cell_and_untiled_fixed_window_both_reported() {
    let dir = TempDir::new().unwrap();
    make_minimal_case(&dir);
    write_four_stage_study(dir.path());
    write_anticipated_thermal_86(dir.path(), 5040.0);
    write_ic_with_commitments(
        dir.path(),
        &[
            ("2024-01-01", "2024-01-31", 0.0),
            ("2024-01-31", "2024-03-01", 0.0),
            ("2024-03-01", "2024-03-31", 0.0),
            ("2024-03-31", "2024-04-30", 0.0),
            ("2024-04-30", "2024-05-30", 0.0),
            ("2024-05-30", "2024-06-29", 0.0),
            // post-study stage 2 (class-4) is left untiled.
        ],
    );
    // Carried post-study stage 3's bound cell is omitted; stages 4-6 keep theirs.
    write_seven_stage_post_study(
        dir.path(),
        &[(4, 0.0, 300.0), (5, 0.0, 300.0), (6, 0.0, 300.0)],
    );

    let msg = load_case(dir.path()).unwrap_err().to_string();
    assert!(
        msg.contains("no thermal_bounds entry") && msg.contains("post_study_stage_index 3"),
        "expected the missing bound-cell rejection for stage 3, got: {msg}"
    );
    assert!(
        msg.contains("do not tile post-study stage index(es) [2]"),
        "expected the fixed post-horizon tiling rejection for stage 2, got: {msg}"
    );
}

// ── V3: no window may cover a carried or beyond-reach post-study stage ────

/// A window covering post-study stage 3 (carried — decided in-study), on top
/// of a fully-tiled class-4 set and complete bound cells, raises the
/// carried-stage V3 rejection naming index 3 and saying the study itself
/// decides the delivery.
#[test]
fn test_window_on_a_study_decided_post_study_stage_rejected() {
    let dir = TempDir::new().unwrap();
    make_minimal_case(&dir);
    write_four_stage_study(dir.path());
    write_anticipated_thermal_86(dir.path(), 5040.0);
    write_ic_with_commitments(
        dir.path(),
        &[
            ("2024-01-01", "2024-01-31", 0.0),
            ("2024-01-31", "2024-03-01", 0.0),
            ("2024-03-01", "2024-03-31", 0.0),
            ("2024-03-31", "2024-04-30", 0.0),
            ("2024-04-30", "2024-05-30", 0.0),
            ("2024-05-30", "2024-06-29", 0.0),
            ("2024-06-29", "2024-07-29", 0.0),
            // Extra window covering carried post-study stage 3.
            ("2024-07-29", "2024-08-28", 0.0),
        ],
    );
    write_seven_stage_post_study(
        dir.path(),
        &[
            (3, 0.0, 300.0),
            (4, 0.0, 300.0),
            (5, 0.0, 300.0),
            (6, 0.0, 300.0),
        ],
    );

    let msg = load_case(dir.path()).unwrap_err().to_string();
    let v3_line = msg
        .lines()
        .find(|l| l.contains("which the study itself decides"))
        .unwrap_or_else(|| panic!("expected a carried-stage V3 rejection, got: {msg}"));
    assert!(
        v3_line.contains("index(es) [3]"),
        "message should name post-study stage index 3, got: {v3_line}"
    );
}

/// A window covering post-study stage 6 (beyond the plant's decision reach)
/// raises a rejection distinct from the carried-stage one, naming index 6.
#[test]
fn test_window_beyond_the_decision_reach_rejected() {
    let dir = TempDir::new().unwrap();
    make_minimal_case(&dir);
    write_four_stage_study(dir.path());
    // LeadTime 1440h decides post-study stages 0-1 in-study (carried) and
    // stages 2-6 at a post-study stage itself, past the plant's reach.
    write_anticipated_thermal_86(dir.path(), 1440.0);
    write_ic_with_commitments(
        dir.path(),
        &[
            ("2024-01-01", "2024-01-31", 0.0),
            ("2024-01-31", "2024-03-01", 0.0),
            // Erroneous window covering beyond-reach post-study stage 6.
            ("2024-10-27", "2024-11-26", 0.0),
        ],
    );
    write_seven_stage_post_study(dir.path(), &[(0, 0.0, 300.0), (1, 0.0, 300.0)]);

    let msg = load_case(dir.path()).unwrap_err().to_string();
    let v3_line = msg
        .lines()
        .find(|l| l.contains("past the plant's decision reach"))
        .unwrap_or_else(|| panic!("expected a beyond-reach V3 rejection, got: {msg}"));
    assert!(
        v3_line.contains("index(es) [6]"),
        "message should name post-study stage index 6, got: {v3_line}"
    );
    assert!(
        !v3_line.contains("which the study itself decides"),
        "the beyond-reach message must be distinct from the carried-stage one, got: {v3_line}"
    );
}

/// Coverage confined to the class-4 (`fixed_post_study`) set — V2's own
/// passing fixture — loads cleanly: no window covers the carried set, so V3
/// never fires either.
#[test]
fn test_window_confined_to_the_fixed_set_accepted() {
    let dir = TempDir::new().unwrap();
    make_minimal_case(&dir);
    write_four_stage_study(dir.path());
    write_anticipated_thermal_86(dir.path(), 5040.0);
    write_ic_with_commitments(
        dir.path(),
        &[
            ("2024-01-01", "2024-01-31", 0.0),
            ("2024-01-31", "2024-03-01", 0.0),
            ("2024-03-01", "2024-03-31", 0.0),
            ("2024-03-31", "2024-04-30", 0.0),
            ("2024-04-30", "2024-05-30", 0.0),
            ("2024-05-30", "2024-06-29", 0.0),
            ("2024-06-29", "2024-07-29", 0.0),
        ],
    );
    write_seven_stage_post_study(
        dir.path(),
        &[
            (3, 0.0, 300.0),
            (4, 0.0, 300.0),
            (5, 0.0, 300.0),
            (6, 0.0, 300.0),
        ],
    );

    let system = load_case(dir.path()).unwrap_or_else(|e| {
        panic!("coverage confined to the fixed set must load cleanly, got: {e}")
    });
    assert!(system.post_study_stages().is_some());
}

/// A window covering a commissioning-inactive post-study stage — an
/// `exit_stage_id` of 1 decommissions the plant right at the post-study
/// horizon, so every declared post-study stage is commissioning-inactive for
/// it — loads cleanly; a declared window there is inert, not contradictory.
#[test]
fn test_window_on_a_commissioning_inactive_post_study_stage_accepted() {
    let dir = TempDir::new().unwrap();
    make_minimal_case(&dir);
    write_anticipated_thermal_86_with_exit(dir.path(), 744.0, 1);
    write_ic_with_commitments(
        dir.path(),
        &[
            ("2024-01-01", "2024-02-01", 0.0),
            // Window on commissioning-inactive post-study stage 0.
            ("2024-02-01", "2024-03-01", 0.0),
        ],
    );
    write_file(
        dir.path(),
        "post_study_stages.json",
        r#"{
          "stages": [
            { "start_date": "2024-02-01", "duration_hours": 696.0 },
            { "start_date": "2024-03-01", "duration_hours": 744.0 },
            { "start_date": "2024-04-01", "duration_hours": 720.0 }
          ],
          "thermal_bounds": []
        }"#,
    );

    let system = load_case(dir.path()).unwrap_or_else(|e| {
        panic!(
            "a window on a commissioning-inactive post-study stage must not \
             raise a V3 rejection, got: {e}"
        )
    });
    assert!(system.post_study_stages().is_some());
}

/// A deck with a carried-stage window (V3) and an untiled class-4 stage (V2)
/// reports both — neither short-circuits the other.
#[test]
fn test_v2_and_v3_failures_both_reported() {
    let dir = TempDir::new().unwrap();
    make_minimal_case(&dir);
    write_four_stage_study(dir.path());
    write_anticipated_thermal_86(dir.path(), 5040.0);
    write_ic_with_commitments(
        dir.path(),
        &[
            ("2024-01-01", "2024-01-31", 0.0),
            ("2024-01-31", "2024-03-01", 0.0),
            ("2024-03-01", "2024-03-31", 0.0),
            ("2024-03-31", "2024-04-30", 0.0),
            ("2024-04-30", "2024-05-30", 0.0),
            ("2024-05-30", "2024-06-29", 0.0),
            // Post-study stage 2 (class-4) is left untiled -- V2.
            // Extra window covers carried post-study stage 3 -- V3.
            ("2024-07-29", "2024-08-28", 0.0),
        ],
    );
    write_seven_stage_post_study(
        dir.path(),
        &[
            (3, 0.0, 300.0),
            (4, 0.0, 300.0),
            (5, 0.0, 300.0),
            (6, 0.0, 300.0),
        ],
    );

    let msg = load_case(dir.path()).unwrap_err().to_string();
    assert!(
        msg.lines().any(|l| l.contains("do not tile post-study")),
        "expected a V2 tiling rejection, got: {msg}"
    );
    assert!(
        msg.lines()
            .any(|l| l.contains("which the study itself decides") && l.contains("index(es) [3]")),
        "expected a V3 carried-stage rejection naming index 3, got: {msg}"
    );
}

// ── V4: the committed-value envelope applies to post-horizon windows too ──

/// V2's tiling rejection (post-study stage 2 left untiled) and V4's envelope
/// rejection (the window tiling post-study stage 0 carries a value outside
/// thermal 86's `[0, 300]` bounds) fire independently on the same deck —
/// neither short-circuits the other.
#[test]
fn test_untiled_fixed_window_and_out_of_envelope_value_both_reported() {
    let dir = TempDir::new().unwrap();
    make_minimal_case(&dir);
    write_four_stage_study(dir.path());
    write_anticipated_thermal_86(dir.path(), 5040.0);
    write_ic_with_commitments(
        dir.path(),
        &[
            ("2024-01-01", "2024-01-31", 0.0),
            ("2024-01-31", "2024-03-01", 0.0),
            ("2024-03-01", "2024-03-31", 0.0),
            ("2024-03-31", "2024-04-30", 0.0),
            // post-study stage 0 (class-4) is tiled with an out-of-envelope value.
            ("2024-04-30", "2024-05-30", 500.0),
            ("2024-05-30", "2024-06-29", 0.0),
            // post-study stage 2 (class-4) is left untiled.
        ],
    );
    write_seven_stage_post_study(
        dir.path(),
        &[
            (3, 0.0, 300.0),
            (4, 0.0, 300.0),
            (5, 0.0, 300.0),
            (6, 0.0, 300.0),
        ],
    );

    let msg = load_case(dir.path()).unwrap_err().to_string();
    assert!(
        msg.lines().any(|l| l.contains("do not tile post-study")),
        "expected a V2 tiling rejection, got: {msg}"
    );
    assert!(
        msg.lines()
            .any(|l| l.contains("outside the plant's generation bounds") && l.contains("500")),
        "expected a V4 envelope rejection naming the out-of-bounds value, got: {msg}"
    );
}

// ── V5: a non-zero fixed value must lie inside the commissioning window ──

/// A unique substring of V5's rejection message, absent from every other
/// `check_post_study_stages` rule's text — used to isolate V5's own line(s)
/// when another rule co-fires on the same deck.
const V5_MARKER: &str = "is not in service for this stage";

/// An `exit_stage_id` of 1 decommissions the plant exactly at the study
/// horizon end, so all three declared post-study stages are
/// commissioning-inactive for it (`fixed_post_study`/`carried`/`beyond_reach`
/// all stay empty — no Rule 1/V2/V3 co-firing). A single post-horizon
/// window of `value_mw = 120.0` spans all three, so V5 rejects each index.
#[test]
fn test_nonzero_fixed_commitment_outside_commissioning_window_rejected() {
    let dir = TempDir::new().unwrap();
    make_minimal_case(&dir);
    write_anticipated_thermal_86_with_exit(dir.path(), 744.0, 1);
    write_ic_with_commitments(
        dir.path(),
        &[
            ("2024-01-01", "2024-02-01", 0.0),
            ("2024-02-01", "2024-05-01", 120.0),
        ],
    );
    write_file(
        dir.path(),
        "post_study_stages.json",
        r#"{
          "stages": [
            { "start_date": "2024-02-01", "duration_hours": 696.0 },
            { "start_date": "2024-03-01", "duration_hours": 744.0 },
            { "start_date": "2024-04-01", "duration_hours": 720.0 }
          ],
          "thermal_bounds": []
        }"#,
    );

    let msg = load_case(dir.path()).unwrap_err().to_string();
    let v5_lines: Vec<&str> = msg.lines().filter(|l| l.contains(V5_MARKER)).collect();
    assert_eq!(
        v5_lines.len(),
        3,
        "expected one V5 violation per post-study stage index, got: {msg}"
    );
    for j in 0..3 {
        assert!(
            v5_lines
                .iter()
                .any(|l| l.contains(&format!("post-study stage index {j}"))),
            "expected a V5 violation naming post-study stage index {j}, got: {msg}"
        );
    }
    for line in &v5_lines {
        assert!(
            line.contains("120") && line.contains("commissioning window"),
            "expected the violation to name the value 120 and the commissioning window, got: {line}"
        );
    }
}

/// The same fixture as
/// [`test_nonzero_fixed_commitment_outside_commissioning_window_rejected`],
/// but the post-horizon window's `value_mw` is `0.0`: an explicit zero
/// declaration over an inactive stage is legitimate, so the case loads
/// cleanly.
#[test]
fn test_zero_fixed_commitment_outside_commissioning_window_accepted() {
    let dir = TempDir::new().unwrap();
    make_minimal_case(&dir);
    write_anticipated_thermal_86_with_exit(dir.path(), 744.0, 1);
    write_ic_with_commitments(
        dir.path(),
        &[
            ("2024-01-01", "2024-02-01", 0.0),
            ("2024-02-01", "2024-05-01", 0.0),
        ],
    );
    write_file(
        dir.path(),
        "post_study_stages.json",
        r#"{
          "stages": [
            { "start_date": "2024-02-01", "duration_hours": 696.0 },
            { "start_date": "2024-03-01", "duration_hours": 744.0 },
            { "start_date": "2024-04-01", "duration_hours": 720.0 }
          ],
          "thermal_bounds": []
        }"#,
    );

    let system = load_case(dir.path()).unwrap_or_else(|e| {
        panic!(
            "a zero-valued window over a commissioning-inactive post-study stage \
             must not be rejected, got: {e}"
        )
    });
    assert!(system.post_study_stages().is_some());
}

/// A windowless plant (`entry_stage_id` and `exit_stage_id` both absent) is
/// always commissioning-active, so its `commissioning_inactive` set is
/// structurally empty and V5 never fires, regardless of the value declared.
/// `LeadTime` 1440h decides post-study stage 0 pre-study (`fixed_post_study`);
/// the nonzero window fully tiling that stage keeps V2 silent, so the deck
/// loads cleanly.
#[test]
fn test_windowless_plant_fixed_commitment_accepted() {
    let dir = TempDir::new().unwrap();
    make_minimal_case(&dir);
    write_anticipated_thermal_86(dir.path(), 1440.0);
    write_ic_with_commitments(
        dir.path(),
        &[
            ("2024-01-01", "2024-02-01", 0.0),
            ("2024-02-01", "2024-03-01", 120.0),
        ],
    );
    write_file(
        dir.path(),
        "post_study_stages.json",
        r#"{
          "stages": [
            { "start_date": "2024-02-01", "duration_hours": 696.0 },
            { "start_date": "2024-03-01", "duration_hours": 744.0 }
          ],
          "thermal_bounds": [
            { "thermal_id": 86, "post_study_stage_index": 1, "cost_per_mwh": 220.0, "min_mw": 0.0, "max_mw": 350.0 }
          ]
        }"#,
    );

    let system = load_case(dir.path())
        .unwrap_or_else(|e| panic!("a windowless plant's fixed commitment must load, got: {e}"));
    assert!(system.post_study_stages().is_some());
}

/// A plant whose commissioning window (`exit_stage_id = 100`) excludes none
/// of its reached stages keeps every class-4 (`fixed_post_study`) index
/// commissioning-active, so a non-zero window confined to that class never
/// raises V5; the fully-tiled class-4 set keeps V2 silent too, so the deck
/// loads cleanly.
#[test]
fn test_fixed_commitment_inside_commissioning_window_accepted() {
    let dir = TempDir::new().unwrap();
    make_minimal_case(&dir);
    write_four_stage_study(dir.path());
    write_anticipated_thermal_86_with_exit(dir.path(), 5040.0, 100);
    write_ic_with_commitments(
        dir.path(),
        &[
            ("2024-01-01", "2024-01-31", 0.0),
            ("2024-01-31", "2024-03-01", 0.0),
            ("2024-03-01", "2024-03-31", 0.0),
            ("2024-03-31", "2024-04-30", 0.0),
            ("2024-04-30", "2024-05-30", 0.0),
            ("2024-05-30", "2024-06-29", 150.0),
            ("2024-06-29", "2024-07-29", 0.0),
        ],
    );
    write_seven_stage_post_study(
        dir.path(),
        &[
            (3, 0.0, 300.0),
            (4, 0.0, 300.0),
            (5, 0.0, 300.0),
            (6, 0.0, 300.0),
        ],
    );

    let system = load_case(dir.path()).unwrap_or_else(|e| {
        panic!("a non-zero window confined to commissioning-active stages must load, got: {e}")
    });
    assert!(system.post_study_stages().is_some());
}

/// V5's own message must never claim LP infeasibility: a fixed post-horizon
/// commitment has no LP column to make infeasible, unlike the in-study seed
/// rule it mirrors.
#[test]
fn test_fixed_commitment_window_message_makes_no_infeasibility_claim() {
    let dir = TempDir::new().unwrap();
    make_minimal_case(&dir);
    write_anticipated_thermal_86_with_exit(dir.path(), 744.0, 1);
    write_ic_with_commitments(
        dir.path(),
        &[
            ("2024-01-01", "2024-02-01", 0.0),
            ("2024-02-01", "2024-03-01", 120.0),
        ],
    );
    write_file(
        dir.path(),
        "post_study_stages.json",
        r#"{
          "stages": [ { "start_date": "2024-02-01", "duration_hours": 696.0 } ],
          "thermal_bounds": []
        }"#,
    );

    let msg = load_case(dir.path()).unwrap_err().to_string();
    let v5_line = msg
        .lines()
        .find(|l| l.contains(V5_MARKER))
        .unwrap_or_else(|| {
            panic!("expected a commissioning-window fixed-value rejection, got: {msg}")
        });
    assert!(
        !v5_line.to_lowercase().contains("infeasible"),
        "the fixed post-horizon rejection must not claim LP infeasibility \
         (there is no LP column post-horizon), got: {v5_line}"
    );
    assert!(
        v5_line.contains("Thermal 86")
            && v5_line.contains("120")
            && v5_line.contains("commissioning window"),
        "expected the message to name the plant, value, and commissioning window, got: {v5_line}"
    );
}

// ── the fixed-window validation matrix composes on one deck ──────────────

/// The reference-shaped deck a faithful post-study calendar produces: four
/// study stages, a post-study calendar whose leading three stages are the
/// plant's class-4 (pre-study-decided) set — tiled by
/// `past_anticipated_commitments`, including an explicit `0 MW` window — and
/// whose remaining four stages are study-decided (class-3), each with its
/// `PostStudyThermalBound` cell. This is the shape the whole matrix exists to
/// make loadable; it must report zero `BusinessRuleViolation` diagnostics.
#[test]
fn test_reference_shaped_fixed_post_horizon_deck_loads() {
    let dir = TempDir::new().unwrap();
    make_minimal_case(&dir);
    write_four_stage_study(dir.path());
    write_anticipated_thermal_86(dir.path(), 5040.0);
    write_ic_with_commitments(
        dir.path(),
        &[
            ("2024-01-01", "2024-01-31", 0.0),
            ("2024-01-31", "2024-03-01", 0.0),
            ("2024-03-01", "2024-03-31", 0.0),
            ("2024-03-31", "2024-04-30", 0.0),
            ("2024-04-30", "2024-05-30", 0.0),
            ("2024-05-30", "2024-06-29", 0.0),
            ("2024-06-29", "2024-07-29", 0.0),
        ],
    );
    write_seven_stage_post_study(
        dir.path(),
        &[
            (3, 0.0, 300.0),
            (4, 0.0, 300.0),
            (5, 0.0, 300.0),
            (6, 0.0, 300.0),
        ],
    );

    let (system, report) = validate_case(dir.path())
        .unwrap_or_else(|e| panic!("the reference-shaped deck must load, got: {e}"));
    assert_eq!(
        report.error_count, 0,
        "expected zero BusinessRuleViolation diagnostics, got: {:?}",
        report.errors
    );

    let mut bound_indices: Vec<usize> = system
        .post_study_stages()
        .expect("post_study_stages must be Some")
        .thermal_bounds
        .iter()
        .map(|b| b.post_study_stage_index)
        .collect();
    bound_indices.sort_unstable();
    assert_eq!(
        bound_indices,
        vec![3, 4, 5, 6],
        "the required bound cells must be exactly the study-decided (class-3) \
         post-study stages"
    );
}

/// One deck violating V2 (an untiled class-4 stage), V3 (a window covering a
/// study-decided stage), V4 (an out-of-envelope value), and V5 (a non-zero
/// fixed value outside the commissioning window) at the same time. An
/// `exit_stage_id` of 9 splits the plant's post-study reach into an active
/// prefix (indices 0-4, unaffected) and an inactive suffix (indices 5-6,
/// where a non-zero window triggers V5), so all four rules fire on disjoint
/// stage indices and none can short-circuit another.
#[test]
fn test_v2_v3_v4_and_v5_violations_are_all_reported() {
    let dir = TempDir::new().unwrap();
    make_minimal_case(&dir);
    write_four_stage_study(dir.path());
    write_anticipated_thermal_86_with_exit(dir.path(), 5040.0, 9);
    write_ic_with_commitments(
        dir.path(),
        &[
            ("2024-01-01", "2024-01-31", 0.0),
            ("2024-01-31", "2024-03-01", 0.0),
            ("2024-03-01", "2024-03-31", 0.0),
            ("2024-03-31", "2024-04-30", 0.0),
            // post-study stage 0 (class-4): an out-of-envelope value -- V4.
            ("2024-04-30", "2024-05-30", 500.0),
            ("2024-05-30", "2024-06-29", 0.0),
            // post-study stage 2 (class-4) is left untiled -- V2.
            // post-study stage 3 (class-3): a window wrongly covers it -- V3.
            ("2024-07-29", "2024-08-28", 0.0),
            // post-study stage 5 (outside the commissioning window): non-zero -- V5.
            ("2024-09-27", "2024-10-27", 120.0),
        ],
    );
    write_seven_stage_post_study(dir.path(), &[(3, 0.0, 300.0), (4, 0.0, 300.0)]);

    let msg = load_case(dir.path()).unwrap_err().to_string();
    let lines: Vec<&str> = msg.lines().collect();
    assert!(
        lines.len() >= 4,
        "expected at least four distinct violations, got: {msg}"
    );
    assert!(
        lines
            .iter()
            .any(|l| l.contains("do not tile post-study") && l.contains("[2]")),
        "expected a V2 tiling rejection naming index 2, got: {msg}"
    );
    assert!(
        lines
            .iter()
            .any(|l| l.contains("which the study itself decides") && l.contains("[3]")),
        "expected a V3 study-decided-stage rejection naming index 3, got: {msg}"
    );
    assert!(
        lines
            .iter()
            .any(|l| l.contains("outside the plant's generation bounds") && l.contains("500")),
        "expected a V4 envelope rejection naming the value 500, got: {msg}"
    );
    assert!(
        lines.iter().any(|l| l.contains(V5_MARKER)
            && l.contains("post-study stage index 5")
            && l.contains("120")),
        "expected a V5 commissioning-window rejection naming index 5 and value 120, got: {msg}"
    );
}

// ── the retired no-carrier advice never resurfaces ────────────────────────

/// Every diagnostic collected from the reference-shaped deck (this file's
/// `test_reference_shaped_fixed_post_horizon_deck_loads` fixture) and from
/// the tiled pre-study-decided fixture (`test_pre_study_decided_post_study_delivery_loads_when_tiled`)
/// must never mention the retired no-carrier advice to shorten the lead — a
/// substring-absence check over every collected diagnostic, so a future
/// reintroduction fails loudly instead of silently regressing the matrix.
#[test]
fn test_retired_no_carrier_advice_is_absent_from_every_diagnostic() {
    const RETIRED_ADVICE_FRAGMENTS: [&str; 2] = ["no carrier", "Shorten the lead"];

    let mut diagnostics: Vec<String> = Vec::new();

    let dir1 = TempDir::new().unwrap();
    make_minimal_case(&dir1);
    write_four_stage_study(dir1.path());
    write_anticipated_thermal_86(dir1.path(), 5040.0);
    write_ic_with_commitments(
        dir1.path(),
        &[
            ("2024-01-01", "2024-01-31", 0.0),
            ("2024-01-31", "2024-03-01", 0.0),
            ("2024-03-01", "2024-03-31", 0.0),
            ("2024-03-31", "2024-04-30", 0.0),
            ("2024-04-30", "2024-05-30", 0.0),
            ("2024-05-30", "2024-06-29", 0.0),
            ("2024-06-29", "2024-07-29", 0.0),
        ],
    );
    write_seven_stage_post_study(
        dir1.path(),
        &[
            (3, 0.0, 300.0),
            (4, 0.0, 300.0),
            (5, 0.0, 300.0),
            (6, 0.0, 300.0),
        ],
    );
    let (_, report1) = validate_case(dir1.path())
        .unwrap_or_else(|e| panic!("the reference-shaped deck must load, got: {e}"));
    diagnostics.extend(report1.errors.into_iter().map(|entry| entry.message));
    diagnostics.extend(report1.warnings.into_iter().map(|entry| entry.message));

    let dir2 = TempDir::new().unwrap();
    make_minimal_case(&dir2);
    write_anticipated_thermal_86(dir2.path(), 1440.0);
    write_ic_with_commitments(
        dir2.path(),
        &[
            ("2024-01-01", "2024-02-01", 0.0),
            ("2024-02-01", "2024-03-01", 0.0),
        ],
    );
    write_file(
        dir2.path(),
        "post_study_stages.json",
        r#"{
          "stages": [
            { "start_date": "2024-02-01", "duration_hours": 696.0 },
            { "start_date": "2024-03-01", "duration_hours": 744.0 }
          ],
          "thermal_bounds": [
            { "thermal_id": 86, "post_study_stage_index": 1, "cost_per_mwh": 220.0, "min_mw": 0.0, "max_mw": 350.0 }
          ]
        }"#,
    );
    let (_, report2) = validate_case(dir2.path())
        .unwrap_or_else(|e| panic!("the tiled pre-study-decided fixture must load, got: {e}"));
    diagnostics.extend(report2.errors.into_iter().map(|entry| entry.message));
    diagnostics.extend(report2.warnings.into_iter().map(|entry| entry.message));

    for fragment in RETIRED_ADVICE_FRAGMENTS {
        assert!(
            !diagnostics.iter().any(|m| m.contains(fragment)),
            "the retired no-carrier advice fragment {fragment:?} must never appear in any \
             collected diagnostic, got: {diagnostics:?}"
        );
    }
}

// ── declaration-order invariance: commitment windows ──────────────────────

/// The reference-shaped deck loads to an identical `System` regardless of
/// the order its `past_anticipated_commitments` windows are declared in —
/// mirroring `test_declaration_order_invariance_through_system` for the
/// commitment-window surface.
#[test]
fn test_commitment_window_declaration_order_invariance_through_system() {
    let windows: [(&str, &str, f64); 7] = [
        ("2024-01-01", "2024-01-31", 0.0),
        ("2024-01-31", "2024-03-01", 0.0),
        ("2024-03-01", "2024-03-31", 0.0),
        ("2024-03-31", "2024-04-30", 0.0),
        ("2024-04-30", "2024-05-30", 0.0),
        ("2024-05-30", "2024-06-29", 0.0),
        ("2024-06-29", "2024-07-29", 0.0),
    ];
    let mut reversed_windows = windows;
    reversed_windows.reverse();

    let bounds: [(usize, f64, f64); 4] = [
        (3, 0.0, 300.0),
        (4, 0.0, 300.0),
        (5, 0.0, 300.0),
        (6, 0.0, 300.0),
    ];

    let dir1 = TempDir::new().unwrap();
    make_minimal_case(&dir1);
    write_four_stage_study(dir1.path());
    write_anticipated_thermal_86(dir1.path(), 5040.0);
    write_ic_with_commitments(dir1.path(), &windows);
    write_seven_stage_post_study(dir1.path(), &bounds);

    let dir2 = TempDir::new().unwrap();
    make_minimal_case(&dir2);
    write_four_stage_study(dir2.path());
    write_anticipated_thermal_86(dir2.path(), 5040.0);
    write_ic_with_commitments(dir2.path(), &reversed_windows);
    write_seven_stage_post_study(dir2.path(), &bounds);

    let s1 = load_case(dir1.path())
        .unwrap_or_else(|e| panic!("the canonical-order deck must load, got: {e}"));
    let s2 = load_case(dir2.path())
        .unwrap_or_else(|e| panic!("the reverse-order deck must load, got: {e}"));
    assert_eq!(
        s1, s2,
        "the loaded System must be identical regardless of commitment-window \
         declaration order"
    );
}
