//! `in_transit` arrival dates equal the policy manifest's `HydroTransitBucket`
//! slot dates row by row, and hand-computed literals pin the shared dating
//! functions the differential check cannot see into.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::collections::BTreeMap;
use std::path::Path;

use novomodelo_io::Config;
use novomodelo_io::StateFamily;
use novomodelo_io::output::policy::{ENTITY_SLOT_DATE_SENTINEL, EntitySlot};
use novomodelo_sddp::simulation::types::SimulationTransitBucketResult;
use novomodelo_sddp::test_support::stage_entity_manifests;
use novomodelo_sddp::test_support::unmigrated_inputs::{Generation, materialize_twin};
use tempfile::TempDir;

use common::permute::permute_case;
use common::{copy_toy_case, fresh_system_and_setup_with, run_simulation};

const TRAVEL_TIME_ARC: &str = "crates/novomodelo-sddp/tests/fixtures/travel_time_arc";

type Dates = (Option<i32>, Option<i32>);
type Manifests = Vec<(i32, Vec<EntitySlot>)>;
type Rows = Vec<(u32, SimulationTransitBucketResult)>;

fn copy_fixture(relative: &str) -> TempDir {
    let dir = TempDir::new().expect("tempdir");
    copy_toy_case(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join(relative),
        dir.path(),
    );
    dir
}

fn declared_calendar_case() -> TempDir {
    copy_fixture("tests/unswept_fixtures/dual_calendar_boundary")
}

fn drop_boundary(config: &mut Config) {
    config.policy.boundary = None;
}

fn dates(row: &SimulationTransitBucketResult) -> Dates {
    (row.arrival_start_date, row.arrival_end_date)
}

fn slot_dates(slot: &EntitySlot) -> Dates {
    let dated = |date: i32| (date != ENTITY_SLOT_DATE_SENTINEL).then_some(date);
    (dated(slot.interval_start), dated(slot.interval_end))
}

fn run_case(dir: &Path, mutate: impl FnOnce(&mut Config)) -> (Manifests, Rows) {
    let (system, mut setup) = fresh_system_and_setup_with(dir, mutate);
    let manifests = stage_entity_manifests(&setup, &system);
    let terminal = setup.build_terminal_entity_manifest(&system);
    assert_eq!(
        serde_json::to_value(&manifests.last().expect("one manifest per cut pool").1)
            .expect("serialize helper manifest"),
        serde_json::to_value(&terminal).expect("serialize terminal manifest"),
        "the helper's last pool must equal the terminal manifest",
    );
    let rows = run_simulation(&mut setup, 1)
        .into_iter()
        .flat_map(|scenario| {
            let id = scenario.scenario_id;
            scenario
                .stages
                .into_iter()
                .flat_map(|stage| stage.transit_buckets)
                .map(move |row| (id, row))
        })
        .collect();
    (manifests, rows)
}

fn assert_rows_equal_manifest_slots(manifests: &Manifests, rows: &Rows) {
    let (mut matched, mut skipped) = (0_usize, 0_usize);
    for (scenario, row) in rows {
        let stage_id = i32::try_from(row.stage_id).expect("stage id fits i32");
        let slot = manifests
            .iter()
            .filter(|(id, _)| *id == stage_id)
            .flat_map(|(_, manifest)| manifest)
            .find(|slot| {
                slot.family() == Some(StateFamily::HydroTransitBucket)
                    && slot.entity_id == row.hydro_id
                    && slot.subindex == row.lag
            });
        let Some(slot) = slot else {
            skipped += 1;
            continue;
        };
        matched += 1;
        assert_eq!(
            dates(row),
            slot_dates(slot),
            "scenario {scenario} stage {} hydro {} lag {}: row dates vs manifest slot",
            row.stage_id,
            row.hydro_id,
            row.lag,
        );
    }
    assert!(
        matched > 0,
        "no row matched a manifest bucket slot ({skipped} rows skipped)"
    );
}

fn assert_dates_at(rows: &Rows, stage_id: u32, lag: u32, expected: Dates) {
    let at: Vec<_> = rows
        .iter()
        .filter(|(_, row)| row.stage_id == stage_id && row.lag == lag)
        .collect();
    assert!(!at.is_empty(), "no row at stage {stage_id} lag {lag}");
    for (scenario, row) in at {
        assert_eq!(
            dates(row),
            expected,
            "scenario {scenario} stage {stage_id} lag {lag}"
        );
    }
}

fn dates_by_row_key(rows: &Rows) -> BTreeMap<(u32, i32, u32), Dates> {
    let mut map = BTreeMap::new();
    for (scenario, row) in rows {
        let key = (row.stage_id, row.hydro_id, row.lag);
        if let Some(previous) = map.insert(key, dates(row)) {
            assert_eq!(
                previous,
                dates(row),
                "scenario {scenario}: dates of {key:?} differ across scenarios"
            );
        }
    }
    map
}

#[test]
fn arrival_dates_equal_manifest_bucket_slots_on_every_row() {
    let dir = declared_calendar_case();
    let (manifests, rows) = run_case(dir.path(), drop_boundary);
    assert_rows_equal_manifest_slots(&manifests, &rows);
}

#[test]
fn arrival_dates_null_past_padded_water_calendar() {
    let dir = copy_fixture("tests/fixtures/travel_time_arc");
    materialize_twin(Generation::PostStudyStagesKey, TRAVEL_TIME_ARC, dir.path())
        .expect("padded-calendar twin");
    let (manifests, rows) = run_case(dir.path(), |_| {});

    let max_lag = rows.iter().map(|(_, row)| row.lag).max().expect("rows");
    assert_eq!(max_lag, 2, "observed maximum lag");
    assert_rows_equal_manifest_slots(&manifests, &rows);
    assert!(
        rows.iter().any(|(_, row)| dates(row) == (None, None)),
        "no row past the padded water calendar"
    );
    assert_dates_at(&rows, 0, 1, (Some(20_240_201), Some(20_240_301)));
    for (scenario, row) in &rows {
        if row.stage_id + row.lag >= 3 {
            assert_eq!(
                dates(row),
                (None, None),
                "scenario {scenario} stage {} lag {} arrives past the study calendar",
                row.stage_id,
                row.lag,
            );
        }
    }
}

#[test]
fn arrival_dates_pin_hand_computed_intervals() {
    let dir = declared_calendar_case();
    let (_, rows) = run_case(dir.path(), drop_boundary);
    assert_dates_at(&rows, 0, 1, (Some(20_240_201), Some(20_240_301)));
    assert_dates_at(&rows, 0, 2, (Some(20_240_301), Some(20_240_401)));
    assert_dates_at(&rows, 5, 2, (Some(20_240_801), Some(20_240_901)));
}

#[test]
fn arrival_dates_invariant_to_declaration_order() {
    let dir = declared_calendar_case();
    let declared = dates_by_row_key(&run_case(dir.path(), drop_boundary).1);
    let permuted_dir = permute_case(dir.path(), 7);
    let permuted = dates_by_row_key(&run_case(permuted_dir.path(), drop_boundary).1);
    assert!(!declared.is_empty(), "no in_transit rows to compare");
    assert_eq!(declared, permuted);
}
