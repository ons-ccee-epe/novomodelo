//! A current-format deck that changes only `water_bucket_stages` moves the water
//! facts (bucket slots, the state dimension they size) and leaves every thermal
//! fact (`time_value.*`, the anticipated ring slots) unchanged.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::fs;
use std::path::Path;

use chrono::{Days, NaiveDate};
use novomodelo_core::System;
use novomodelo_io::StateFamily;
use novomodelo_io::output::policy::{ENTITY_SLOT_DATE_SENTINEL, EntitySlot, encode_slot_date};
use novomodelo_sddp::StudySetup;
use novomodelo_sddp::test_support::{state_space, template_fact_groups};
use serde_json::json;
use tempfile::TempDir;

use common::{copy_toy_case, fresh_system_and_setup_with};

struct WaterStage {
    start: NaiveDate,
    days: u32,
}

impl WaterStage {
    fn json(&self) -> serde_json::Value {
        json!({ "start_date": self.start.to_string(), "duration_hours": f64::from(self.days) * 24.0 })
    }

    fn interval(&self) -> (i32, i32) {
        (
            encode_slot_date(self.start),
            encode_slot_date(self.start + Days::new(u64::from(self.days))),
        )
    }
}

fn date(year: i32, month: u32, day: u32) -> NaiveDate {
    NaiveDate::from_ymd_opt(year, month, day).expect("valid date")
}

fn monthly_water_stages() -> Vec<WaterStage> {
    vec![
        WaterStage {
            start: date(2024, 7, 1),
            days: 31,
        },
        WaterStage {
            start: date(2024, 8, 1),
            days: 31,
        },
    ]
}

fn weekly_water_stages() -> Vec<WaterStage> {
    (0..9)
        .map(|week| WaterStage {
            start: date(2024, 7, 1) + Days::new(7 * week),
            days: 7,
        })
        .collect()
}

fn case_with_water_bucket_stages(water: &[WaterStage]) -> TempDir {
    let fixture =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/gate_fixtures/dual_calendar_boundary");
    let dir = TempDir::new().expect("tempdir");
    copy_toy_case(&fixture, dir.path());
    fs::remove_dir_all(dir.path().join("source")).expect("remove source deck");

    let thermal_bound = |post_study_stage_index: u32| {
        json!({
            "thermal_id": 1,
            "post_study_stage_index": post_study_stage_index,
            "cost_per_mwh": 5.0,
            "min_mw": 0.0,
            "max_mw": 150.0
        })
    };
    let post_study = json!({
        "thermal_stages": [
            { "start_date": "2024-07-01", "duration_hours": 744.0 },
            { "start_date": "2024-08-01", "duration_hours": 744.0 },
            { "start_date": "2024-09-01", "duration_hours": 720.0 }
        ],
        "thermal_bounds": [thermal_bound(0), thermal_bound(1), thermal_bound(2)],
        "water_bucket_stages": water.iter().map(WaterStage::json).collect::<Vec<_>>()
    });
    fs::write(
        dir.path().join("post_study_stages.json"),
        serde_json::to_vec_pretty(&post_study).expect("serialize post_study_stages"),
    )
    .expect("write post_study_stages.json");
    dir
}

fn monthly_and_weekly_setups() -> [(System, StudySetup); 2] {
    [monthly_water_stages(), weekly_water_stages()].map(|water| {
        let dir = case_with_water_bucket_stages(&water);
        fresh_system_and_setup_with(dir.path(), |_| {})
    })
}

fn slot_intervals(manifest: &[EntitySlot], family: StateFamily) -> Vec<(i32, i32)> {
    manifest
        .iter()
        .filter(|slot| slot.family() == Some(family))
        .map(|slot| (slot.interval_start, slot.interval_end))
        .collect()
}

fn is_dated((start, end): (i32, i32)) -> bool {
    start != ENTITY_SLOT_DATE_SENTINEL && end != ENTITY_SLOT_DATE_SENTINEL
}

#[test]
fn changing_only_water_bucket_stages_moves_buckets_and_keeps_thermal_facts() {
    let [(_, monthly), (_, weekly)] = monthly_and_weekly_setups();
    let monthly_facts = template_fact_groups(&monthly);
    let weekly_facts = template_fact_groups(&weekly);

    assert!(
        monthly_facts.keys().eq(weekly_facts.keys()),
        "fact group names"
    );
    let moved_thermal: Vec<_> = monthly_facts
        .iter()
        .filter(|(group, bytes)| {
            group.starts_with("time_value.") && weekly_facts[*group] != **bytes
        })
        .map(|(group, _)| *group)
        .collect();
    assert!(
        moved_thermal.is_empty(),
        "water calendar moved thermal facts: {moved_thermal:?}"
    );

    let (monthly_state, weekly_state) = (state_space(&monthly), state_space(&weekly));
    assert_eq!(monthly_state.n_anticipated, weekly_state.n_anticipated);
    assert_eq!(monthly_state.k_max, weekly_state.k_max);

    assert!(
        monthly_facts
            .iter()
            .any(|(group, bytes)| !group.starts_with("time_value.")
                && weekly_facts[*group] != **bytes),
        "no water fact moved with the water calendar"
    );
}

#[test]
fn public_manifest_dates_bucket_slots_from_water_bucket_stages() {
    let [(monthly_system, monthly), (weekly_system, weekly)] = monthly_and_weekly_setups();
    let monthly_manifest = monthly.build_terminal_entity_manifest(&monthly_system);
    let weekly_manifest = weekly.build_terminal_entity_manifest(&weekly_system);

    let monthly_ring = slot_intervals(&monthly_manifest, StateFamily::AnticipatedThermalState);
    let weekly_ring = slot_intervals(&weekly_manifest, StateFamily::AnticipatedThermalState);
    assert!(
        monthly_ring.iter().copied().any(is_dated),
        "no dated ring slot to compare"
    );
    assert_eq!(
        monthly_ring, weekly_ring,
        "ring slots follow the thermal calendar"
    );

    let monthly_buckets = slot_intervals(&monthly_manifest, StateFamily::HydroTransitBucket);
    let weekly_buckets = slot_intervals(&weekly_manifest, StateFamily::HydroTransitBucket);
    assert_ne!(
        monthly_buckets, weekly_buckets,
        "bucket slots follow the water calendar"
    );

    let weekly_intervals: Vec<_> = weekly_water_stages()
        .iter()
        .map(WaterStage::interval)
        .collect();
    let dated: Vec<_> = weekly_buckets
        .iter()
        .copied()
        .filter(|&i| is_dated(i))
        .collect();
    assert!(!dated.is_empty(), "no dated bucket slot to compare");
    for interval in dated {
        assert!(
            weekly_intervals.contains(&interval),
            "bucket interval {interval:?} is not one of the water stages {weekly_intervals:?}"
        );
    }
}
