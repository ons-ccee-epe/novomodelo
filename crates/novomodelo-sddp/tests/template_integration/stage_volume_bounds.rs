//! Stage-volume (hm³) bound column tests.

use std::cmp::Reverse;
use std::collections::HashMap;
use std::path::Path;

use arrow::array::{Array, Int32Array};
use novomodelo_core::{
    BlockBoundsCountsSpec, BlockMode, EntityId, HydroBlockBounds, HydroPenalties, PumpingStation,
    ResolvedBlockBounds, ResolvedGenericConstraintBounds, ResolvedStageVolumeBounds,
    StageVolumeBoundsCountsSpec, volume_cost_from_flow_cost,
};
use novomodelo_sddp::StudySetup;
use novomodelo_sddp::test_support::{assert_all_templates_byte_identical, template_fact_groups};
use novomodelo_solver::StageTemplate;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use tempfile::TempDir;

use super::common::parquet_fixtures::{
    Hm3Columns, HydroBoundsRow, PumpingBoundsRow, write_hydro_bounds, write_pumping_bounds,
};
use super::common::permute::permute_case;
use super::common::{copy_toy_case, fresh_setup_with};
use super::generic_constraints::{
    GenericBoundsRow, HydroStudy, PUMP_MAX, VOLUME_BLOCK_HOURS, WATER_ROW_5, WATER_ROW_10,
    diversion_pumping_hydro_study, flow_columns, hydro_study_system, parse_generic_from_str,
    row_entries, two_block_volume_terms, volume_tau,
};
use super::{
    COST_SCALE_FACTOR, build_templates_for, csc_entries_at, default_hydro_block_bounds,
    default_hydro_penalties,
};

fn d32_copy() -> TempDir {
    let deck = TempDir::new().expect("tempdir");
    copy_toy_case(
        &Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../examples/deterministic/d32-reversible-plant"),
        deck.path(),
    );
    deck
}

/// `d32-reversible-plant` with a stage-0 bounds row for each hydro and for the
/// pumping station, every stage-volume cell null.
fn d32_with_null_bounds(hm3: Hm3Columns) -> TempDir {
    let deck = d32_copy();
    let constraints = deck.path().join("constraints");
    std::fs::create_dir(&constraints).expect("create constraints dir");
    let hydro_rows = [0, 1].map(|hydro_id| HydroBoundsRow {
        hydro_id,
        stage_id: 0,
        min_outflow_hm3: None,
        max_outflow_hm3: None,
        min_diversion_hm3: None,
        max_diversion_hm3: None,
    });
    write_hydro_bounds(&constraints.join("hydro_bounds.parquet"), &hydro_rows, hm3);
    let pumping_rows = [PumpingBoundsRow {
        pumping_station_id: 0,
        stage_id: 0,
        min_hm3: None,
        max_hm3: None,
    }];
    write_pumping_bounds(
        &constraints.join("pumping_bounds.parquet"),
        &pumping_rows,
        hm3,
    );
    deck
}

#[test]
fn null_hm3_columns_build_byte_identical_templates() {
    let omitted = d32_with_null_bounds(Hm3Columns::Omit);
    let written = d32_with_null_bounds(Hm3Columns::Write);

    let omitted_setup = fresh_setup_with(omitted.path(), |_| {});
    let written_setup = fresh_setup_with(written.path(), |_| {});

    assert_all_templates_byte_identical(
        &omitted_setup.inputs.stage_data.stage_templates.templates,
        &written_setup.inputs.stage_data.stage_templates.templates,
        "null hm3 columns vs omitted",
    );
}

const BLOCK_MODES: [BlockMode; 2] = [BlockMode::Parallel, BlockMode::Chronological];

fn bits(entries: &[(usize, f64)]) -> Vec<(usize, u64)> {
    entries.iter().map(|&(col, v)| (col, v.to_bits())).collect()
}

/// `study` built with no stage-volume overlay and with `stage_volume`, as
/// `(baseline, declared)`.
fn baseline_and_declared(
    study: impl Fn() -> HydroStudy,
    stage_volume: ResolvedStageVolumeBounds,
) -> (StageTemplate, StageTemplate) {
    let build = |stage_volume| {
        build_templates_for(&hydro_study_system(HydroStudy {
            stage_volume,
            ..study()
        }))
        .swap_remove(0)
    };
    (
        build(ResolvedStageVolumeBounds::empty()),
        build(stage_volume),
    )
}

/// `study` with and without an outflow-volume declaration on its first hydro at stage 0,
/// as `(baseline, declared)`.
fn outflow_volume_templates_of(
    study: impl Fn() -> HydroStudy,
    min_outflow_hm3: Option<f64>,
    max_outflow_hm3: Option<f64>,
) -> (StageTemplate, StageTemplate) {
    let mut stage_volume = stage_zero_overlay();
    let cell = stage_volume.hydro_mut(0, 0).expect("hydro 0 is in range");
    cell.min_outflow_hm3 = min_outflow_hm3;
    cell.max_outflow_hm3 = max_outflow_hm3;

    baseline_and_declared(study, stage_volume)
}

fn outflow_volume_templates(
    block_mode: BlockMode,
    min_outflow_hm3: Option<f64>,
    max_outflow_hm3: Option<f64>,
) -> (StageTemplate, StageTemplate) {
    outflow_volume_templates_of(
        || HydroStudy {
            block_mode,
            ..diversion_pumping_hydro_study()
        },
        min_outflow_hm3,
        max_outflow_hm3,
    )
}

#[test]
fn native_outflow_volume_row_carries_tau_on_turbine_and_spillage() {
    for block_mode in BLOCK_MODES {
        let (baseline, t) = outflow_volume_templates(block_mode, Some(3.0), Some(50.0));
        let row = baseline.num_rows;
        let (below_slack, above_slack) = (baseline.num_cols, baseline.num_cols + 1);
        let cols = flow_columns(2, 2, block_mode);

        assert_eq!(t.num_rows, row + 1, "{block_mode:?}: one row");
        assert_eq!(
            t.num_cols,
            baseline.num_cols + 2,
            "{block_mode:?}: two slacks"
        );

        let expected: Vec<(usize, f64)> = two_block_volume_terms(cols.turbine)
            .into_iter()
            .chain(two_block_volume_terms(cols.spillage))
            .collect();
        let found: Vec<(usize, f64)> = row_entries(&t, row)
            .into_iter()
            .filter(|&(col, _)| col < baseline.num_cols)
            .collect();
        assert_eq!(bits(&found), bits(&expected), "{block_mode:?}: terms");
        for k in 0..2 {
            assert!(
                csc_entries_at(&t, cols.diversion + k, row).is_empty(),
                "{block_mode:?}: diversion block {k} is not outflow"
            );
        }

        assert_eq!(csc_entries_at(&t, below_slack, row), vec![1.0]);
        assert_eq!(csc_entries_at(&t, above_slack, row), vec![-1.0]);
        assert_eq!(t.row_lower[row].to_bits(), 3.0_f64.to_bits());
        assert_eq!(t.row_upper[row].to_bits(), 50.0_f64.to_bits());
    }
}

#[test]
fn native_outflow_volume_absent_side_is_infinite() {
    for block_mode in BLOCK_MODES {
        let (baseline, min_only) = outflow_volume_templates(block_mode, Some(3.0), None);
        let row = baseline.num_rows;
        assert_eq!(min_only.row_lower[row].to_bits(), 3.0_f64.to_bits());
        assert_eq!(min_only.row_upper[row], f64::INFINITY, "{block_mode:?}");

        let (_, max_only) = outflow_volume_templates(block_mode, None, Some(50.0));
        assert_eq!(max_only.row_lower[row], f64::NEG_INFINITY, "{block_mode:?}");
        assert_eq!(max_only.row_upper[row].to_bits(), 50.0_f64.to_bits());
    }
}

const BELOW_FLOW_COST: f64 = 36.0;
const ABOVE_VOLUME_COST: f64 = 11.0;
const OTHER_BLOCK_HOURS: [f64; 2] = [300.0, 444.0];

fn priced_slack_study(block_mode: BlockMode, block_hours: [f64; 2]) -> HydroStudy {
    HydroStudy {
        block_mode,
        block_hours: block_hours.to_vec(),
        hydro_penalties: HydroPenalties {
            outflow_volume_violation_below_cost: volume_cost_from_flow_cost(BELOW_FLOW_COST),
            outflow_volume_violation_above_cost: ABOVE_VOLUME_COST,
            ..default_hydro_penalties()
        },
        ..diversion_pumping_hydro_study()
    }
}

#[test]
fn outflow_volume_slack_prices_are_the_resolved_volume_costs() {
    for block_mode in BLOCK_MODES {
        let (baseline, t) = outflow_volume_templates_of(
            || priced_slack_study(block_mode, VOLUME_BLOCK_HOURS),
            Some(3.0),
            Some(50.0),
        );
        let (below, above) = (baseline.num_cols, baseline.num_cols + 1);

        assert_eq!(
            t.objective[below].to_bits(),
            (volume_cost_from_flow_cost(BELOW_FLOW_COST) / COST_SCALE_FACTOR).to_bits(),
            "{block_mode:?}: below"
        );
        assert_eq!(
            t.objective[above].to_bits(),
            (ABOVE_VOLUME_COST / COST_SCALE_FACTOR).to_bits(),
            "{block_mode:?}: above"
        );
        for col in [below, above] {
            assert_eq!(t.col_lower[col], 0.0, "{block_mode:?}: col {col}");
            assert_eq!(t.col_upper[col], f64::INFINITY, "{block_mode:?}: col {col}");
        }
    }
}

#[test]
fn outflow_volume_slack_prices_ignore_stage_hours() {
    for block_mode in BLOCK_MODES {
        let slack_objectives = |block_hours| {
            let (baseline, t) = outflow_volume_templates_of(
                || priced_slack_study(block_mode, block_hours),
                Some(3.0),
                Some(50.0),
            );
            [baseline.num_cols, baseline.num_cols + 1].map(|col| t.objective[col])
        };
        let short = slack_objectives(VOLUME_BLOCK_HOURS);
        let long = slack_objectives(OTHER_BLOCK_HOURS);

        for (side, (short, long)) in ["below", "above"]
            .into_iter()
            .zip(short.into_iter().zip(long))
        {
            assert_ne!(short, 0.0, "{block_mode:?}: {side} is priced");
            assert_eq!(short.to_bits(), long.to_bits(), "{block_mode:?}: {side}");
        }
    }
}

#[test]
fn outflow_volume_absent_side_slack_is_pinned() {
    for block_mode in BLOCK_MODES {
        let study = || priced_slack_study(block_mode, VOLUME_BLOCK_HOURS);
        let (baseline, min_only) = outflow_volume_templates_of(study, Some(3.0), None);
        let (_, max_only) = outflow_volume_templates_of(study, None, Some(50.0));
        let (below, above) = (baseline.num_cols, baseline.num_cols + 1);

        let bounds = |t: &StageTemplate, col: usize| (t.col_lower[col], t.col_upper[col]);
        assert_eq!(
            bounds(&min_only, below),
            (0.0, f64::INFINITY),
            "{block_mode:?}"
        );
        assert_eq!(bounds(&min_only, above), (0.0, 0.0), "{block_mode:?}");
        assert_eq!(bounds(&max_only, below), (0.0, 0.0), "{block_mode:?}");
        assert_eq!(
            bounds(&max_only, above),
            (0.0, f64::INFINITY),
            "{block_mode:?}"
        );
    }
}

/// `diversion_pumping_hydro_study()` with and without a diversion-volume declaration on
/// the hydro at `hydro_position` at stage 0, as `(baseline, declared)`.
fn diversion_volume_templates(
    block_mode: BlockMode,
    hydro_position: usize,
    min_diversion_hm3: Option<f64>,
    max_diversion_hm3: Option<f64>,
) -> (StageTemplate, StageTemplate) {
    let mut stage_volume = stage_zero_overlay();
    let cell = stage_volume
        .hydro_mut(hydro_position, 0)
        .expect("hydro position is in range");
    cell.min_diversion_hm3 = min_diversion_hm3;
    cell.max_diversion_hm3 = max_diversion_hm3;

    baseline_and_declared(
        || HydroStudy {
            block_mode,
            ..diversion_pumping_hydro_study()
        },
        stage_volume,
    )
}

#[test]
fn native_diversion_volume_row_is_hard() {
    for block_mode in BLOCK_MODES {
        let (baseline, t) = diversion_volume_templates(block_mode, 0, Some(1.0), Some(20.0));
        let row = baseline.num_rows;

        assert_eq!(t.num_rows, row + 1, "{block_mode:?}: one row");
        assert_eq!(t.num_cols, baseline.num_cols, "{block_mode:?}: no slack");
        assert_eq!(
            bits(&row_entries(&t, row)),
            bits(&two_block_volume_terms(
                flow_columns(2, 2, block_mode).diversion
            )),
            "{block_mode:?}: terms"
        );
        assert_eq!(t.row_lower[row].to_bits(), 1.0_f64.to_bits());
        assert_eq!(t.row_upper[row].to_bits(), 20.0_f64.to_bits());
    }
}

#[test]
fn native_diversion_volume_of_a_non_diverter_reads_its_pinned_column() {
    for block_mode in BLOCK_MODES {
        let (baseline, t) = diversion_volume_templates(block_mode, 1, None, Some(5.0));
        let row = baseline.num_rows;
        let expected = two_block_volume_terms(flow_columns(2, 2, block_mode).diversion + 2);

        assert_eq!(
            bits(&row_entries(&t, row)),
            bits(&expected),
            "{block_mode:?}: terms"
        );
        for (k, &(col, _)) in expected.iter().enumerate() {
            assert_eq!(t.col_lower[col], 0.0, "{block_mode:?}: block {k}: lower");
            assert_eq!(t.col_upper[col], 0.0, "{block_mode:?}: block {k}: upper");
        }
    }
}

#[test]
fn native_diversion_volume_absent_side_is_infinite() {
    for block_mode in BLOCK_MODES {
        let (baseline, min_only) = diversion_volume_templates(block_mode, 0, Some(1.0), None);
        let row = baseline.num_rows;
        assert_eq!(min_only.row_lower[row].to_bits(), 1.0_f64.to_bits());
        assert_eq!(min_only.row_upper[row], f64::INFINITY, "{block_mode:?}");

        let (_, max_only) = diversion_volume_templates(block_mode, 0, None, Some(20.0));
        assert_eq!(max_only.row_lower[row], f64::NEG_INFINITY, "{block_mode:?}");
        assert_eq!(max_only.row_upper[row].to_bits(), 20.0_f64.to_bits());
    }
}

/// `study` with and without a pumping-volume declaration on the station at
/// `station_position` at stage 0, as `(baseline, declared)`.
fn pumping_volume_templates_of(
    study: impl Fn() -> HydroStudy,
    station_position: usize,
    min_hm3: Option<f64>,
    max_hm3: Option<f64>,
) -> (StageTemplate, StageTemplate) {
    let mut stage_volume = ResolvedStageVolumeBounds::new(&StageVolumeBoundsCountsSpec {
        n_hydros: 2,
        n_pumping: study().pumping_stations.len(),
        n_stages: 1,
    });
    let cell = stage_volume
        .pumping_mut(station_position, 0)
        .expect("station position is in range");
    cell.min_hm3 = min_hm3;
    cell.max_hm3 = max_hm3;

    baseline_and_declared(study, stage_volume)
}

fn pumping_volume_templates(
    block_mode: BlockMode,
    min_hm3: Option<f64>,
    max_hm3: Option<f64>,
) -> (StageTemplate, StageTemplate) {
    pumping_volume_templates_of(
        || HydroStudy {
            block_mode,
            ..diversion_pumping_hydro_study()
        },
        0,
        min_hm3,
        max_hm3,
    )
}

/// Station 30 (source hydro 10, destination hydro 5) beside station 31, which pumps the
/// other way.
fn two_pump_study(block_mode: BlockMode) -> HydroStudy {
    let mut study = HydroStudy {
        block_mode,
        ..diversion_pumping_hydro_study()
    };
    let reverse = PumpingStation {
        id: EntityId(31),
        name: "P31".to_string(),
        source_hydro_id: EntityId(5),
        destination_hydro_id: EntityId(10),
        ..study.pumping_stations[0].clone()
    };
    study.pumping_stations.push(reverse);
    study
}

fn dormant_pump_study(block_mode: BlockMode) -> HydroStudy {
    let mut study = HydroStudy {
        block_mode,
        ..diversion_pumping_hydro_study()
    };
    study.pumping_stations[0].entry_stage_id = Some(1);
    study
}

/// derived: water-balance rows follow the z-inflow rows, so `WATER_ROW_5` and `WATER_ROW_10`
/// are a parallel stage's row of hydro 5 and of hydro 10; a chronological stage has one row
/// per hydro and block, hydro-major from `WATER_ROW_5`.
fn water_row(block_mode: BlockMode, parallel_row: usize, block: usize) -> usize {
    match block_mode {
        BlockMode::Parallel => parallel_row,
        BlockMode::Chronological => {
            WATER_ROW_5 + (parallel_row - WATER_ROW_5) * VOLUME_BLOCK_HOURS.len() + block
        }
    }
}

/// derived: a pumping-volume row holds the station's pump column of each block at that
/// block's `volume_tau`, and the pump moves water from its source plant's row of the block
/// (`+tau`) to its destination's (`-tau`). Returns the row's columns.
fn assert_pump_volume_row(
    t: &StageTemplate,
    row: usize,
    block_mode: BlockMode,
    source_row: usize,
    destination_row: usize,
) -> Vec<usize> {
    let entries = row_entries(t, row);
    let taus = VOLUME_BLOCK_HOURS.map(volume_tau);
    assert_eq!(
        entries
            .iter()
            .map(|&(_, coefficient)| coefficient.to_bits())
            .collect::<Vec<_>>(),
        taus.map(f64::to_bits),
        "{block_mode:?}: terms"
    );
    assert!(entries[0].0 < entries[1].0, "{block_mode:?}: ascending");
    for (k, (&(col, _), tau)) in entries.iter().zip(taus).enumerate() {
        assert_eq!(
            csc_entries_at(t, col, water_row(block_mode, source_row, k)),
            vec![tau],
            "{block_mode:?}: block {k}: source row"
        );
        assert_eq!(
            csc_entries_at(t, col, water_row(block_mode, destination_row, k)),
            vec![-tau],
            "{block_mode:?}: block {k}: destination row"
        );
    }
    entries.into_iter().map(|(col, _)| col).collect()
}

#[test]
fn native_pumping_volume_row_is_hard() {
    for block_mode in BLOCK_MODES {
        let (baseline, t) = pumping_volume_templates(block_mode, Some(2.0), Some(30.0));
        let row = baseline.num_rows;

        assert_eq!(t.num_rows, row + 1, "{block_mode:?}: one row");
        assert_eq!(t.num_cols, baseline.num_cols, "{block_mode:?}: no slack");
        for col in assert_pump_volume_row(&t, row, block_mode, WATER_ROW_10, WATER_ROW_5) {
            assert_eq!(t.col_upper[col], PUMP_MAX, "{block_mode:?}: col {col}");
        }
        assert_eq!(t.row_lower[row].to_bits(), 2.0_f64.to_bits());
        assert_eq!(t.row_upper[row].to_bits(), 30.0_f64.to_bits());

        let (baseline, t) =
            pumping_volume_templates_of(|| two_pump_study(block_mode), 1, Some(2.0), Some(30.0));
        let row = baseline.num_rows;

        assert_eq!(t.num_rows, row + 1, "{block_mode:?}: two stations: one row");
        assert_eq!(
            t.num_cols, baseline.num_cols,
            "{block_mode:?}: two stations: no slack"
        );
        for col in assert_pump_volume_row(&t, row, block_mode, WATER_ROW_5, WATER_ROW_10) {
            assert_eq!(
                t.col_upper[col], PUMP_MAX,
                "{block_mode:?}: two stations: col {col}"
            );
        }
    }
}

#[test]
fn native_pumping_volume_of_a_dormant_station_reads_its_pinned_columns() {
    for block_mode in BLOCK_MODES {
        let (baseline, t) =
            pumping_volume_templates_of(|| dormant_pump_study(block_mode), 0, None, Some(5.0));
        let row = baseline.num_rows;

        assert_eq!(t.num_rows, row + 1, "{block_mode:?}: one row");
        for col in assert_pump_volume_row(&t, row, block_mode, WATER_ROW_10, WATER_ROW_5) {
            assert_eq!(t.col_lower[col], 0.0, "{block_mode:?}: col {col}: lower");
            assert_eq!(t.col_upper[col], 0.0, "{block_mode:?}: col {col}: upper");
        }
    }
}

#[test]
fn native_pumping_volume_absent_side_is_infinite() {
    for block_mode in BLOCK_MODES {
        let (baseline, min_only) = pumping_volume_templates(block_mode, Some(2.0), None);
        let row = baseline.num_rows;
        assert_eq!(min_only.row_lower[row].to_bits(), 2.0_f64.to_bits());
        assert_eq!(min_only.row_upper[row], f64::INFINITY, "{block_mode:?}");

        let (_, max_only) = pumping_volume_templates(block_mode, None, Some(30.0));
        assert_eq!(max_only.row_lower[row], f64::NEG_INFINITY, "{block_mode:?}");
        assert_eq!(max_only.row_upper[row].to_bits(), 30.0_f64.to_bits());
    }
}

/// derived: a `per_unit` slack, native or generic, costs this over `COST_SCALE_FACTOR`,
/// with no hours factor.
const PARITY_SLACK_PENALTY: f64 = 40.0;

fn stage_zero_overlay() -> ResolvedStageVolumeBounds {
    ResolvedStageVolumeBounds::new(&StageVolumeBoundsCountsSpec {
        n_hydros: 2,
        n_pumping: 1,
        n_stages: 1,
    })
}

/// `diversion_pumping_hydro_study()` under `block_mode` as `(baseline, native, generic)`:
/// no declaration, `stage_volume` alone, and the constraint with its bounds rows alone.
fn three_templates(
    block_mode: BlockMode,
    stage_volume: ResolvedStageVolumeBounds,
    constraint_json: &str,
    bounds_rows: Vec<GenericBoundsRow>,
) -> (StageTemplate, StageTemplate, StageTemplate) {
    let study = || HydroStudy {
        block_mode,
        hydro_penalties: HydroPenalties {
            outflow_volume_violation_below_cost: PARITY_SLACK_PENALTY,
            outflow_volume_violation_above_cost: PARITY_SLACK_PENALTY,
            ..default_hydro_penalties()
        },
        ..diversion_pumping_hydro_study()
    };
    let constraints = parse_generic_from_str(constraint_json);
    let id_map: HashMap<i32, usize> = constraints
        .iter()
        .enumerate()
        .map(|(position, constraint)| (constraint.id.0, position))
        .collect();
    let generic_bounds = ResolvedGenericConstraintBounds::new(&id_map, bounds_rows.into_iter());

    let (baseline, native) = baseline_and_declared(study, stage_volume);
    let generic = build_templates_for(&hydro_study_system(HydroStudy {
        constraints,
        generic_bounds,
        ..study()
    }))
    .swap_remove(0);
    (baseline, native, generic)
}

#[test]
fn native_outflow_volume_matches_the_generic_constraint() {
    let cases = [
        ("two-sided", Some(3.0), Some(50.0)),
        ("min-only", Some(3.0), None),
    ];
    for block_mode in BLOCK_MODES {
        for (case, min_hm3, max_hm3) in cases {
            let label = format!("outflow {block_mode:?} {case}");
            let mut overlay = stage_zero_overlay();
            let cell = overlay.hydro_mut(0, 0).expect("hydro 0 is in range");
            cell.min_outflow_hm3 = min_hm3;
            cell.max_outflow_hm3 = max_hm3;
            let json = format!(
                r#"{{"constraints":[{{"id":100,"name":"out","expression":"hydro_outflow_volume(5)","slack":{{"enabled":true,"penalty":{PARITY_SLACK_PENALTY:?},"penalty_basis":"per_unit"}}}}]}}"#
            );
            let (baseline, native, generic) = three_templates(
                block_mode,
                overlay,
                &json,
                vec![(100, 0, None, min_hm3, max_hm3)],
            );
            let row = baseline.num_rows;

            assert_eq!(native.num_rows, row + 1, "{label}: native: one row");
            assert_eq!(generic.num_rows, row + 1, "{label}: generic: one row");

            let structural = |t: &StageTemplate| -> Vec<(usize, f64)> {
                row_entries(t, row)
                    .into_iter()
                    .filter(|&(col, _)| col < baseline.num_cols)
                    .collect()
            };
            let cols = flow_columns(2, 2, block_mode);
            let expected: Vec<(usize, f64)> = two_block_volume_terms(cols.turbine)
                .into_iter()
                .chain(two_block_volume_terms(cols.spillage))
                .collect();
            assert_eq!(
                bits(&structural(&native)),
                bits(&structural(&generic)),
                "{label}: native vs generic terms"
            );
            assert_eq!(
                bits(&structural(&native)),
                bits(&expected),
                "{label}: native terms"
            );

            let bounds = |t: &StageTemplate| [t.row_lower[row], t.row_upper[row]].map(f64::to_bits);
            assert_eq!(bounds(&native), bounds(&generic), "{label}: row bounds");

            let below = baseline.num_cols;
            let sides = [(below, 1.0), (below + 1, -1.0)];
            let active = &sides[..if max_hm3.is_some() { 2 } else { 1 }];
            assert_eq!(
                generic.num_cols,
                baseline.num_cols + active.len(),
                "{label}: generic carries the active slacks only"
            );
            let price = (PARITY_SLACK_PENALTY / COST_SCALE_FACTOR).to_bits();
            for &(col, entry) in active {
                assert_eq!(
                    csc_entries_at(&native, col, row),
                    vec![entry],
                    "{label}: native slack {col}"
                );
                assert_eq!(
                    csc_entries_at(&generic, col, row),
                    vec![entry],
                    "{label}: generic slack {col}"
                );
                assert_eq!(
                    native.objective[col].to_bits(),
                    generic.objective[col].to_bits(),
                    "{label}: slack {col} price, native vs generic"
                );
                assert_eq!(
                    native.objective[col].to_bits(),
                    price,
                    "{label}: slack {col} price"
                );
            }
        }
    }
}

#[test]
fn native_diversion_volume_matches_the_generic_constraint() {
    for block_mode in BLOCK_MODES {
        let label = format!("diversion {block_mode:?}");
        let mut overlay = stage_zero_overlay();
        let cell = overlay.hydro_mut(0, 0).expect("hydro 0 is in range");
        cell.min_diversion_hm3 = Some(1.0);
        cell.max_diversion_hm3 = Some(20.0);
        let json = r#"{"constraints":[{"id":200,"name":"div","expression":"hydro_diversion_volume(5)","slack":{"enabled":false}}]}"#;
        let (baseline, native, generic) = three_templates(
            block_mode,
            overlay,
            json,
            vec![(200, 0, None, Some(1.0), Some(20.0))],
        );
        let row = baseline.num_rows;

        assert_eq!(native.num_rows, row + 1, "{label}: native: one row");
        assert_eq!(generic.num_rows, row + 1, "{label}: generic: one row");
        assert_eq!(
            native.num_cols, baseline.num_cols,
            "{label}: native: no slack"
        );
        assert_eq!(
            generic.num_cols, baseline.num_cols,
            "{label}: generic: no slack"
        );

        assert_eq!(
            bits(&row_entries(&native, row)),
            bits(&row_entries(&generic, row)),
            "{label}: native vs generic terms"
        );
        assert_eq!(
            bits(&row_entries(&native, row)),
            bits(&two_block_volume_terms(
                flow_columns(2, 2, block_mode).diversion
            )),
            "{label}: native terms"
        );

        let bounds = |t: &StageTemplate| [t.row_lower[row], t.row_upper[row]].map(f64::to_bits);
        assert_eq!(bounds(&native), bounds(&generic), "{label}: row bounds");
    }
}

#[test]
fn native_pumping_volume_matches_the_generic_constraint() {
    for block_mode in BLOCK_MODES {
        let label = format!("pumping {block_mode:?}");
        let mut overlay = stage_zero_overlay();
        let cell = overlay.pumping_mut(0, 0).expect("station 0 is in range");
        cell.min_hm3 = Some(2.0);
        cell.max_hm3 = Some(30.0);
        let json = r#"{"constraints":[{"id":300,"name":"vol","expression":"pumping_volume(30)","slack":{"enabled":false}}]}"#;
        let (baseline, native, generic) = three_templates(
            block_mode,
            overlay,
            json,
            vec![(300, 0, None, Some(2.0), Some(30.0))],
        );
        let row = baseline.num_rows;

        assert_eq!(native.num_rows, row + 1, "{label}: native: one row");
        assert_eq!(generic.num_rows, row + 1, "{label}: generic: one row");
        assert_eq!(
            native.num_cols, baseline.num_cols,
            "{label}: native: no slack"
        );
        assert_eq!(
            generic.num_cols, baseline.num_cols,
            "{label}: generic: no slack"
        );

        assert_eq!(
            bits(&row_entries(&native, row)),
            bits(&row_entries(&generic, row)),
            "{label}: native vs generic terms"
        );
        assert_pump_volume_row(&native, row, block_mode, WATER_ROW_10, WATER_ROW_5);

        let bounds = |t: &StageTemplate| [t.row_lower[row], t.row_upper[row]].map(f64::to_bits);
        assert_eq!(bounds(&native), bounds(&generic), "{label}: row bounds");
    }
}

const COEXISTENCE_MIN_OUTFLOW_M3S: f64 = 2.0;
const COEXISTENCE_MAX_OUTFLOW_M3S: f64 = 500.0;
const COEXISTENCE_BLOCK_MAX_OUTFLOW_M3S: f64 = 400.0;
const OUTFLOW_HM3: (f64, f64) = (3.0, 50.0);
const DIVERSION_HM3: (f64, f64) = (1.0, 20.0);
const PUMPING_HM3: (f64, f64) = (2.0, 30.0);

const COEXISTENCE_CONSTRAINTS_JSON: &str = r#"{"constraints":[{"id":100,"name":"out","expression":"hydro_outflow_volume(5)","slack":{"enabled":false}},{"id":200,"name":"div","expression":"hydro_diversion_volume(5)","slack":{"enabled":false}},{"id":300,"name":"vol","expression":"pumping_volume(30)","slack":{"enabled":false}}]}"#;

fn coexistence_study(block_mode: BlockMode) -> HydroStudy {
    let constraints = parse_generic_from_str(COEXISTENCE_CONSTRAINTS_JSON);
    let id_map: HashMap<i32, usize> = constraints
        .iter()
        .enumerate()
        .map(|(position, constraint)| (constraint.id.0, position))
        .collect();
    let bounds_rows: Vec<GenericBoundsRow> =
        [(100, OUTFLOW_HM3), (200, DIVERSION_HM3), (300, PUMPING_HM3)]
            .into_iter()
            .map(|(id, (min_hm3, max_hm3))| (id, 0, None, Some(min_hm3), Some(max_hm3)))
            .collect();

    let mut block_bounds = ResolvedBlockBounds::new(&BlockBoundsCountsSpec {
        n_hydros: 2,
        n_thermals: 0,
        n_lines: 0,
        n_pumping: 1,
        n_contracts: 0,
        n_stages: 1,
        max_blocks: VOLUME_BLOCK_HOURS.len(),
    });
    block_bounds
        .hydro_override_mut(0, 0, 1)
        .expect("hydro 0, stage 0, block 1 is in range")
        .max_outflow_m3s = Some(COEXISTENCE_BLOCK_MAX_OUTFLOW_M3S);

    HydroStudy {
        block_mode,
        hydro_block: HydroBlockBounds {
            min_outflow_m3s: COEXISTENCE_MIN_OUTFLOW_M3S,
            max_outflow_m3s: Some(COEXISTENCE_MAX_OUTFLOW_M3S),
            ..default_hydro_block_bounds()
        },
        block_bounds,
        constraints,
        generic_bounds: ResolvedGenericConstraintBounds::new(&id_map, bounds_rows.into_iter()),
        ..diversion_pumping_hydro_study()
    }
}

fn coexistence_overlay() -> ResolvedStageVolumeBounds {
    let mut overlay = stage_zero_overlay();
    let hydro = overlay.hydro_mut(0, 0).expect("hydro 0 is in range");
    (hydro.min_outflow_hm3, hydro.max_outflow_hm3) = (Some(OUTFLOW_HM3.0), Some(OUTFLOW_HM3.1));
    (hydro.min_diversion_hm3, hydro.max_diversion_hm3) =
        (Some(DIVERSION_HM3.0), Some(DIVERSION_HM3.1));
    let station = overlay.pumping_mut(0, 0).expect("station 0 is in range");
    (station.min_hm3, station.max_hm3) = (Some(PUMPING_HM3.0), Some(PUMPING_HM3.1));
    overlay
}

/// derived: the native rows follow the generic ones, one per family, and the outflow row
/// adds its two slack columns; m³/s outflow rows are one per hydro and block, and the
/// override replaces hydro 5's block-1 maximum.
#[test]
fn stage_volume_rows_coexist_with_every_other_bound() {
    let families = [
        ("outflow", OUTFLOW_HM3),
        ("diversion", DIVERSION_HM3),
        ("pumping", PUMPING_HM3),
    ];
    let (n_hydros, n_blocks) = (2, VOLUME_BLOCK_HOURS.len());
    for block_mode in BLOCK_MODES {
        let (without, with) =
            baseline_and_declared(|| coexistence_study(block_mode), coexistence_overlay());

        assert_eq!(
            with.num_rows,
            without.num_rows + families.len(),
            "{block_mode:?}: rows"
        );
        assert_eq!(with.num_cols, without.num_cols + 2, "{block_mode:?}: cols");

        for row in 0..without.num_rows {
            for (field, got, expected) in [
                ("lower", with.row_lower[row], without.row_lower[row]),
                ("upper", with.row_upper[row], without.row_upper[row]),
            ] {
                assert_eq!(
                    got.to_bits(),
                    expected.to_bits(),
                    "{block_mode:?}: row {row}: {field}"
                );
            }
            assert_eq!(
                bits(&row_entries(&with, row)),
                bits(&row_entries(&without, row)),
                "{block_mode:?}: row {row}: entries"
            );
        }
        for col in 0..without.num_cols {
            for (field, got, expected) in [
                ("lower", with.col_lower[col], without.col_lower[col]),
                ("upper", with.col_upper[col], without.col_upper[col]),
                ("objective", with.objective[col], without.objective[col]),
            ] {
                assert_eq!(
                    got.to_bits(),
                    expected.to_bits(),
                    "{block_mode:?}: col {col}: {field}"
                );
            }
        }

        let rows_bounded_by = |lower: f64, upper: f64| {
            (0..without.num_rows)
                .filter(|&row| {
                    with.row_lower[row].to_bits() == lower.to_bits()
                        && with.row_upper[row].to_bits() == upper.to_bits()
                })
                .count()
        };
        assert_eq!(
            rows_bounded_by(COEXISTENCE_MIN_OUTFLOW_M3S, f64::INFINITY),
            n_hydros * n_blocks,
            "{block_mode:?}: m3/s minimum-outflow rows"
        );
        assert_eq!(
            rows_bounded_by(f64::NEG_INFINITY, COEXISTENCE_MAX_OUTFLOW_M3S),
            n_hydros * n_blocks - 1,
            "{block_mode:?}: m3/s maximum-outflow rows"
        );
        assert_eq!(
            rows_bounded_by(f64::NEG_INFINITY, COEXISTENCE_BLOCK_MAX_OUTFLOW_M3S),
            1,
            "{block_mode:?}: block-override maximum-outflow row"
        );

        let structural = |row: usize| -> Vec<(usize, u64)> {
            bits(
                &row_entries(&with, row)
                    .into_iter()
                    .filter(|&(col, _)| col < without.num_cols)
                    .collect::<Vec<_>>(),
            )
        };
        let first_generic = without.num_rows - families.len();
        let cols = flow_columns(n_hydros, n_blocks, block_mode);
        let outflow_terms: Vec<(usize, f64)> = two_block_volume_terms(cols.turbine)
            .into_iter()
            .chain(two_block_volume_terms(cols.spillage))
            .collect();
        assert_eq!(
            structural(first_generic),
            bits(&outflow_terms),
            "{block_mode:?}: generic outflow terms"
        );
        assert_eq!(
            structural(first_generic + 1),
            bits(&two_block_volume_terms(cols.diversion)),
            "{block_mode:?}: generic diversion terms"
        );
        assert_pump_volume_row(
            &with,
            first_generic + 2,
            block_mode,
            WATER_ROW_10,
            WATER_ROW_5,
        );

        for (k, (family, (min_hm3, max_hm3))) in families.into_iter().enumerate() {
            let (generic, native) = (first_generic + k, without.num_rows + k);
            assert_eq!(
                structural(native),
                structural(generic),
                "{block_mode:?}: {family}: native vs generic terms"
            );
            for (kind, row) in [("generic", generic), ("native", native)] {
                assert_eq!(
                    [with.row_lower[row], with.row_upper[row]].map(f64::to_bits),
                    [min_hm3, max_hm3].map(f64::to_bits),
                    "{block_mode:?}: {family}: {kind} row {row}: bounds"
                );
            }
        }
    }
}

fn read_json(path: &Path) -> serde_json::Value {
    serde_json::from_str(&std::fs::read_to_string(path).expect("read json")).expect("parse json")
}

fn entity_id(entity: &serde_json::Value) -> i64 {
    entity["id"].as_i64().expect("entity id")
}

fn edit_registry(
    deck: &Path,
    file: &str,
    key: &str,
    edit: impl FnOnce(&mut Vec<serde_json::Value>),
) {
    let path = deck.join(file);
    let mut document = read_json(&path);
    edit(document[key].as_array_mut().expect("registry array"));
    std::fs::write(&path, document.to_string()).expect("write registry");
}

fn registry_ids(deck: &Path, file: &str, key: &str) -> Vec<i64> {
    read_json(&deck.join(file))[key]
        .as_array()
        .expect("registry array")
        .iter()
        .map(entity_id)
        .collect()
}

/// `(entity id, stage id)` of each row of a bounds parquet, in file order.
fn bound_row_keys(parquet: &Path, id_column: &str) -> Vec<(i32, i32)> {
    let reader = ParquetRecordBatchReaderBuilder::try_new(
        std::fs::File::open(parquet).expect("open bounds parquet"),
    )
    .expect("parquet footer")
    .build()
    .expect("parquet reader");
    reader
        .flat_map(|batch| {
            let batch = batch.expect("decode batch");
            let column = |name: &str| {
                batch
                    .column_by_name(name)
                    .and_then(|array| array.as_any().downcast_ref::<Int32Array>())
                    .expect("int32 key column")
                    .values()
                    .to_vec()
            };
            column(id_column)
                .into_iter()
                .zip(column("stage_id"))
                .collect::<Vec<_>>()
        })
        .collect()
}

/// d32 with a second station that pumps the other way.
fn d32_with_two_stations() -> TempDir {
    let deck = d32_copy();
    edit_registry(
        deck.path(),
        "system/pumping_stations.json",
        "pumping_stations",
        |stations| {
            let mut second = stations[0].clone();
            second["id"] = serde_json::json!(1);
            second["name"] = serde_json::json!("Reversible Plant H0->H1");
            second["source_hydro_id"] = serde_json::json!(0);
            second["destination_hydro_id"] = serde_json::json!(1);
            stations.push(second);
        },
    );
    deck
}

/// Outflow bounds on both hydros and pumping bounds on both stations at stages 0
/// and 1, rows ascending by `(id, stage)` unless `reverse_rows`. The stations'
/// upper bounds differ, so a swap of their rows is observable.
fn write_declared_bounds(deck: &Path, reverse_rows: bool) {
    let stages = [0, 1];
    let mut hydro_rows: Vec<HydroBoundsRow> = [0, 1]
        .into_iter()
        .flat_map(|hydro_id| {
            stages.map(|stage_id| HydroBoundsRow {
                hydro_id,
                stage_id,
                min_outflow_hm3: Some(1.0),
                max_outflow_hm3: Some(1000.0),
                min_diversion_hm3: None,
                max_diversion_hm3: None,
            })
        })
        .collect();
    let mut pumping_rows: Vec<PumpingBoundsRow> = [(0, 20.0), (1, 19.0)]
        .into_iter()
        .flat_map(|(pumping_station_id, max_hm3)| {
            stages.map(|stage_id| PumpingBoundsRow {
                pumping_station_id,
                stage_id,
                min_hm3: Some(6.0),
                max_hm3: Some(max_hm3),
            })
        })
        .collect();
    if reverse_rows {
        hydro_rows.reverse();
        pumping_rows.reverse();
    }
    let constraints = deck.join("constraints");
    std::fs::create_dir_all(&constraints).expect("create constraints dir");
    write_hydro_bounds(
        &constraints.join("hydro_bounds.parquet"),
        &hydro_rows,
        Hm3Columns::Write,
    );
    write_pumping_bounds(
        &constraints.join("pumping_bounds.parquet"),
        &pumping_rows,
        Hm3Columns::Write,
    );
}

fn templates_of(setup: &StudySetup) -> &[StageTemplate] {
    &setup.inputs.stage_data.stage_templates.templates
}

fn fact_group_mismatches(a: &StudySetup, b: &StudySetup) -> Vec<&'static str> {
    let (a, b) = (template_fact_groups(a), template_fact_groups(b));
    assert!(
        a.keys().eq(b.keys()),
        "fact-group key sets differ: {:?} vs {:?}",
        a.keys().collect::<Vec<_>>(),
        b.keys().collect::<Vec<_>>(),
    );
    a.iter()
        .filter(|&(group, bytes)| b.get(group) != Some(bytes))
        .map(|(&group, _)| group)
        .collect()
}

#[test]
fn stage_volume_deck_is_invariant_to_declaration_order() {
    const SEED: u64 = 0x5EED_C0BE_5EED_C0BE;
    const REGISTRIES: [(&str, &str); 2] = [
        ("system/hydros.json", "hydros"),
        ("system/pumping_stations.json", "pumping_stations"),
    ];
    const BOUNDS: [(&str, &str); 2] = [
        ("constraints/hydro_bounds.parquet", "hydro_id"),
        ("constraints/pumping_bounds.parquet", "pumping_station_id"),
    ];

    let baseline_deck = d32_with_two_stations();
    let deck_a = d32_with_two_stations();
    write_declared_bounds(deck_a.path(), false);

    let deck_b = permute_case(deck_a.path(), SEED);
    for (file, key) in REGISTRIES {
        edit_registry(deck_b.path(), file, key, |entities| {
            entities.sort_by_key(|entity| Reverse(entity_id(entity)));
        });
    }
    write_declared_bounds(deck_b.path(), true);

    for (file, key) in REGISTRIES {
        assert_eq!(
            registry_ids(deck_b.path(), file, key),
            [1, 0],
            "deck B {key}"
        );
    }
    for (file, id_column) in BOUNDS {
        let keys_a = bound_row_keys(&deck_a.path().join(file), id_column);
        let keys_b = bound_row_keys(&deck_b.path().join(file), id_column);
        assert_eq!(
            keys_b.first(),
            keys_a.last(),
            "{file}: deck B starts at deck A's last row"
        );
        assert_ne!(
            keys_b.first(),
            keys_a.first(),
            "{file}: deck B is reordered"
        );
    }

    let baseline = fresh_setup_with(baseline_deck.path(), |_| {});
    let setup_a = fresh_setup_with(deck_a.path(), |_| {});
    let setup_b = fresh_setup_with(deck_b.path(), |_| {});

    let (declared, plain) = (templates_of(&setup_a), templates_of(&baseline));
    let (declared_stages, outflow_rows, pumping_rows) = (2, 2, 2);
    assert_eq!(
        declared.len(),
        declared_stages + 1,
        "d32 has one more stage"
    );
    for (stage, (t, base)) in declared.iter().zip(plain).enumerate() {
        let (rows, cols) = if stage < declared_stages {
            (outflow_rows + pumping_rows, 2 * outflow_rows)
        } else {
            (0, 0)
        };
        assert_eq!(t.num_rows, base.num_rows + rows, "stage {stage}: rows");
        assert_eq!(t.num_cols, base.num_cols + cols, "stage {stage}: cols");
    }

    // A loaded deck's template holds pre-scaled row bounds: bound * row_scale.
    let stage_zero = &declared[0];
    let last = stage_zero.num_rows - 1;
    for (row, bound, label) in [
        (last - 1, 20.0_f64, "station 0"),
        (last, 19.0_f64, "station 1"),
    ] {
        assert_eq!(
            stage_zero.row_upper[row].to_bits(),
            (bound * stage_zero.row_scale[row]).to_bits(),
            "stage 0: {label}'s pumping row; station 1's follows station 0's"
        );
    }

    let mismatches = fact_group_mismatches(&setup_a, &setup_b);
    assert!(
        mismatches.is_empty(),
        "deck B differs from deck A in: {mismatches:?}"
    );
}

fn declared_study_overlay() -> ResolvedStageVolumeBounds {
    let mut overlay = ResolvedStageVolumeBounds::new(&StageVolumeBoundsCountsSpec {
        n_hydros: 2,
        n_pumping: 2,
        n_stages: 1,
    });
    let hydro = overlay.hydro_mut(0, 0).expect("hydro 0 is in range");
    hydro.min_outflow_hm3 = Some(3.0);
    hydro.max_outflow_hm3 = Some(50.0);
    hydro.min_diversion_hm3 = Some(1.0);
    hydro.max_diversion_hm3 = Some(20.0);
    for (position, (min_hm3, max_hm3)) in [(2.0, 30.0), (3.0, 25.0)].into_iter().enumerate() {
        let station = overlay
            .pumping_mut(position, 0)
            .expect("station position is in range");
        station.min_hm3 = Some(min_hm3);
        station.max_hm3 = Some(max_hm3);
    }
    overlay
}

#[test]
fn stage_volume_study_is_invariant_to_declaration_order() {
    for block_mode in BLOCK_MODES {
        let declared_order = || two_pump_study(block_mode);
        let reversed_order = || {
            let mut study = two_pump_study(block_mode);
            study.hydros.reverse();
            study.pumping_stations.reverse();
            study
        };
        let (baseline, declared) = baseline_and_declared(declared_order, declared_study_overlay());
        let (_, reversed) = baseline_and_declared(reversed_order, declared_study_overlay());

        let (outflow_rows, diversion_rows, pumping_rows) = (1, 1, 2);
        assert_eq!(
            declared.num_rows,
            baseline.num_rows + outflow_rows + diversion_rows + pumping_rows,
            "{block_mode:?}: rows"
        );
        assert_eq!(
            declared.num_cols,
            baseline.num_cols + 2 * outflow_rows,
            "{block_mode:?}: cols"
        );
        assert_all_templates_byte_identical(
            std::slice::from_ref(&reversed),
            std::slice::from_ref(&declared),
            &format!("{block_mode:?}: reversed vs declared"),
        );
    }
}
