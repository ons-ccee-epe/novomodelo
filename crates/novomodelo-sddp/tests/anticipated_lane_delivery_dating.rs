//! `anticipated_lanes` delivery dates equal the policy manifest's
//! `AnticipatedThermalState` ring-slot dates row by row, and a post-study stage
//! starting mid-month pins the exact-date columns the month anchor cannot carry.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::collections::BTreeSet;
use std::path::Path;

use novomodelo_io::StateFamily;
use novomodelo_io::config::SimulationSelection;
use novomodelo_io::output::policy::{ENTITY_SLOT_DATE_SENTINEL, EntitySlot};
use novomodelo_sddp::simulation::types::SimulationAnticipatedLaneResult;
use novomodelo_sddp::test_support::stage_entity_manifests;
use serde_json::json;
use tempfile::TempDir;

use common::{copy_toy_case, edit_json, fresh_system_and_setup_with, run_simulation};

const D55: &str = "../../examples/deterministic/d55-post-study-anticipated-lanes";

type Manifests = Vec<(i32, Vec<EntitySlot>)>;
type Rows = Vec<(u32, SimulationAnticipatedLaneResult)>;

fn d55_copy() -> TempDir {
    let dir = TempDir::new().expect("tempdir");
    copy_toy_case(&Path::new(env!("CARGO_MANIFEST_DIR")).join(D55), dir.path());
    dir
}

fn run_lanes(dir: &Path) -> (Manifests, Rows) {
    let (system, mut setup) = fresh_system_and_setup_with(dir, |config| {
        config.simulation.enabled = true;
        config.simulation.selection = Some(SimulationSelection::Sampled { num_scenarios: 1 });
    });
    let manifests = stage_entity_manifests(&setup, &system);
    let rows = run_simulation(&mut setup, 1)
        .into_iter()
        .flat_map(|scenario| {
            let id = scenario.scenario_id;
            scenario
                .stages
                .into_iter()
                .flat_map(|stage| stage.anticipated_lanes)
                .map(move |row| (id, row))
        })
        .collect();
    (manifests, rows)
}

fn ring_slot_dates(manifests: &Manifests, stage_id: i32) -> BTreeSet<(i32, i32, i32)> {
    manifests
        .iter()
        .filter(|(id, _)| *id == stage_id)
        .flat_map(|(_, manifest)| manifest)
        .filter(|slot| {
            slot.family() == Some(StateFamily::AnticipatedThermalState)
                && slot.interval_start != ENTITY_SLOT_DATE_SENTINEL
        })
        .map(|slot| (slot.entity_id, slot.interval_start, slot.interval_end))
        .collect()
}

#[test]
fn delivery_dates_match_manifest_ring_slots() {
    let dir = d55_copy();
    let (manifests, rows) = run_lanes(dir.path());
    assert!(!rows.is_empty(), "no anticipated_lanes rows");
    for (scenario, row) in &rows {
        let stage_id = i32::try_from(row.stage_id).expect("stage id fits i32");
        let slots = ring_slot_dates(&manifests, stage_id);
        let dates = (
            row.thermal_id,
            row.delivery_start_date,
            row.delivery_end_date,
        );
        assert!(
            slots.contains(&dates),
            "scenario {scenario} stage {stage_id}: lane dates {dates:?} not among the \
             manifest's dated ring slots {slots:?}",
        );
    }
}

#[test]
fn delivery_dates_pin_mid_month_stage() {
    let dir = d55_copy();
    edit_json(&dir.path().join("stages.json"), |stages| {
        stages["stages"][5]["end_date"] = json!("2024-06-16");
        stages["post_study_thermal_stages"] =
            json!([{"id": 6, "start_date": "2024-06-16", "end_date": "2024-07-01"}]);
    });
    let (_, rows) = run_lanes(dir.path());

    assert!(!rows.is_empty(), "no anticipated_lanes rows");
    for (scenario, row) in &rows {
        assert_eq!(
            (
                row.delivery_date,
                row.delivery_start_date,
                row.delivery_end_date
            ),
            (20_240_601, 20_240_616, 20_240_701),
            "scenario {scenario} stage {} thermal {}",
            row.stage_id,
            row.thermal_id,
        );
    }
}

#[test]
fn delivery_date_keeps_month_anchor() {
    let dir = d55_copy();
    let (_, rows) = run_lanes(dir.path());
    assert!(!rows.is_empty(), "no anticipated_lanes rows");
    for (scenario, row) in &rows {
        assert_eq!(
            (
                row.delivery_date,
                row.delivery_start_date,
                row.delivery_end_date
            ),
            (20_240_701, 20_240_701, 20_240_801),
            "scenario {scenario} stage {} thermal {}",
            row.stage_id,
            row.thermal_id,
        );
    }
}
