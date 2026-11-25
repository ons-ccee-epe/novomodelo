//! The `post_study` provenance section lists the calendars a deck declares in
//! `stages.json` or in the deprecated `post_study_stages.json`, reports the
//! padded water calendar of a deck with a travel-time arc and no post-study
//! input, and is absent otherwise. Hand-pinned dates, hours and the padding
//! descriptor catch a shifted date the builder's own arithmetic cannot.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::path::{Path, PathBuf};

use novomodelo_sddp::test_support::unmigrated_inputs::{Generation, materialize_twin};
use novomodelo_sddp::test_support::ymd;
use novomodelo_sddp::{
    PaddedArcProvenance, PostStudyCalendarOrigin, PostStudyCalendarProvenance, PostStudyProvenance,
    PostStudyStageProvenance, WaterCalendarPaddingProvenance, build_post_study_provenance,
};
use tempfile::TempDir;

use common::copy_toy_case;
use common::permute::permute_case;

const TRAVEL_TIME_ARC: &str = "crates/novomodelo-sddp/tests/fixtures/travel_time_arc";

fn deck(relative: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(relative)
}

fn section(dir: &Path) -> Option<PostStudyProvenance> {
    let system = novomodelo_io::load_case(dir)
        .unwrap_or_else(|e| panic!("load_case({}) failed: {e}", dir.display()));
    build_post_study_provenance(&system)
}

fn padded_twin() -> TempDir {
    let dir = TempDir::new().expect("tempdir");
    copy_toy_case(&deck("tests/fixtures/travel_time_arc"), dir.path());
    materialize_twin(Generation::PostStudyStagesKey, TRAVEL_TIME_ARC, dir.path())
        .expect("padded-calendar twin of travel_time_arc");
    dir
}

fn row(start: (i32, u32, u32), end: (i32, u32, u32), hours: f64) -> PostStudyStageProvenance {
    PostStudyStageProvenance {
        start_date: ymd(start.0, start.1, start.2),
        end_date: ymd(end.0, end.1, end.2),
        hours,
    }
}

fn declared(stages: Vec<PostStudyStageProvenance>) -> PostStudyCalendarProvenance {
    PostStudyCalendarProvenance {
        origin: PostStudyCalendarOrigin::Declared,
        stages,
        padding: None,
    }
}

fn padded_water_calendar() -> PostStudyCalendarProvenance {
    PostStudyCalendarProvenance {
        origin: PostStudyCalendarOrigin::PaddedThermal,
        stages: vec![],
        padding: Some(WaterCalendarPaddingProvenance {
            trailing_stage_hours: 730.0,
            padded_stage_count: 2,
            deepest_arc: PaddedArcProvenance {
                upstream_hydro_id: 0,
                downstream_hydro_id: 1,
                travel_time_hours: 1460.0,
            },
        }),
    }
}

#[test]
fn post_study_lists_declared_calendars_from_stages_json() {
    let section = section(&deck("tests/unswept_fixtures/dual_calendar_boundary"))
        .expect("dual_calendar_boundary declares post-study calendars");

    let calendar = || {
        declared(vec![
            row((2024, 7, 1), (2024, 8, 1), 744.0),
            row((2024, 8, 1), (2024, 9, 1), 744.0),
            row((2024, 9, 1), (2024, 10, 1), 720.0),
        ])
    };
    assert_eq!(
        section,
        PostStudyProvenance {
            thermal: Some(calendar()),
            water: calendar(),
        }
    );
}

#[test]
fn post_study_lists_declared_calendars_from_deprecated_file() {
    let section = section(&deck("tests/fixtures/travel_time_arc"))
        .expect("travel_time_arc declares a post-study water calendar");

    assert_eq!(
        section,
        PostStudyProvenance {
            thermal: Some(declared(vec![])),
            water: declared(vec![
                row((2024, 4, 1), (2024, 5, 1), 730.0),
                row((2024, 5, 1), (2024, 5, 31), 730.0),
            ]),
        }
    );
}

#[test]
fn post_study_reports_padded_water_calendar_on_legacy_twin() {
    let dir = padded_twin();
    let section = section(dir.path()).expect("a travel-time arc pads the water calendar");

    assert_eq!(
        section,
        PostStudyProvenance {
            thermal: None,
            water: padded_water_calendar(),
        }
    );
    assert_eq!(
        serde_json::to_value(&section).expect("serialize post_study section"),
        serde_json::json!({
            "water": {
                "origin": "padded_thermal",
                "stages": [],
                "padding": {
                    "trailing_stage_hours": 730.0,
                    "padded_stage_count": 2,
                    "deepest_arc": {
                        "upstream_hydro_id": 0,
                        "downstream_hydro_id": 1,
                        "travel_time_hours": 1460.0,
                    },
                },
            },
        })
    );
}

#[test]
fn post_study_absent_without_post_study_input_or_arc() {
    assert_eq!(
        section(&deck("../../examples/deterministic/d01-thermal-dispatch")),
        None
    );
}

#[test]
fn post_study_invariant_to_declaration_order() {
    let declared_dir = deck("tests/unswept_fixtures/dual_calendar_boundary");
    let twin = padded_twin();

    for (name, dir) in [
        ("dual_calendar_boundary", declared_dir.as_path()),
        ("padded twin of travel_time_arc", twin.path()),
    ] {
        let unpermuted = section(dir).unwrap_or_else(|| panic!("{name} has a post_study section"));
        let permuted = permute_case(dir, 7);
        assert_eq!(
            section(permuted.path()),
            Some(unpermuted),
            "{name}: section changed under declaration-order permutation"
        );
    }
}
