//! Regression suite: a resolved override declared for one entity ID must read
//! back for that SAME entity once `SystemBuilder::build` re-sorts entities by
//! `(operational_start_date, id)`, even when a resolver indexed its output
//! table by parse-order position (id order) instead.
//!
//! Drives the real `cobre_io::load_case` pipeline end to end (structural,
//! schema, referential, dimensional, semantic validation, the `resolve_*`
//! functions, then `SystemBuilder::build`) rather than hand-assembling
//! internals, so any misalignment found here is the one production hits.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::too_many_lines,
    clippy::doc_markdown
)]

mod helpers;

use std::sync::Arc;

use arrow::array::{Float64Array, Int32Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use cobre_core::EntityId;
use cobre_io::{LoadError, load_case};
use parquet::arrow::ArrowWriter;
use tempfile::TempDir;

/// Two hydros whose `operational_start_date` order is the REVERSE of their id
/// order (hydro 1 is dated 2030, hydro 2 is dated 2020), plus a
/// `constraints/hydro_bounds.parquet` override declared only for hydro 1 at
/// stage 1: a stage-wide `max_turbined_m3s=12345.0` and a per-block (block 1)
/// `min_turbined_m3s=777.0`. `hydros.rs::parse_hydros` sorts its output by id
/// (so parse order is `[1, 2]`); `SystemBuilder::build`'s `sort_canonical`
/// re-sorts by `(operational_start_date, id)` (so canonical order is `[2, 1]`).
fn make_id_date_misaligned_case(dir: &TempDir) {
    let root = dir.path();

    helpers::write_file(root, "config.json", helpers::VALID_CONFIG_JSON);
    helpers::write_file(root, "penalties.json", helpers::VALID_PENALTIES_JSON);
    helpers::write_file(
        root,
        "initial_conditions.json",
        helpers::VALID_INITIAL_CONDITIONS_JSON,
    );

    helpers::write_file(
        root,
        "stages.json",
        r#"{
    "policy_graph": {
        "type": "finite_horizon",
        "annual_discount_rate": 0.06,
        "transitions": [
            { "source_id": 0, "target_id": 1, "probability": 1.0 }
        ]
    },
    "stages": [
        {
            "id": 0,
            "start_date": "2024-01-01",
            "end_date": "2024-02-01",
            "blocks": [{ "id": 0, "name": "B0", "hours": 744.0 }],
            "num_openings": 1
        },
        {
            "id": 1,
            "start_date": "2024-02-01",
            "end_date": "2024-03-01",
            "blocks": [
                { "id": 0, "name": "B0", "hours": 336.0 },
                { "id": 1, "name": "B1", "hours": 336.0 }
            ],
            "num_openings": 1
        }
    ]
}"#,
    );

    helpers::write_file(
        root,
        "system/buses.json",
        r#"{ "buses": [{ "id": 1, "name": "BUS_1", "operational_start_date": "2024-01-01" }] }"#,
    );
    helpers::write_file(root, "system/lines.json", r#"{ "lines": [] }"#);
    helpers::write_file(root, "system/thermals.json", r#"{ "thermals": [] }"#);

    // Hydro 1's base max_turbined_m3s (20000.0) comfortably exceeds the
    // 12345.0 override so `check_bound_raises_declared_capacity` accepts it.
    // Hydro 2's base bounds (300.0 / 0.0) are far from both override values
    // (12345.0 / 777.0), so a leaked override is unmistakable in the assert.
    helpers::write_file(
        root,
        "system/hydros.json",
        r#"{
    "hydros": [
        {
            "id": 1,
            "name": "HYDRO_1_LATE",
            "operational_start_date": "2030-01-01",
            "downstream_id": null,
            "reservoir": { "min_storage_hm3": 0.0, "max_storage_hm3": 1000.0 },
            "outflow": { "min_outflow_m3s": 0.0, "max_outflow_m3s": null },
            "generation": {
                "model": "constant_productivity",
                "min_turbined_m3s": 0.0,
                "max_turbined_m3s": 20000.0,
                "min_generation_mw": 0.0,
                "max_generation_mw": 20000.0
            },
            "unit_groups": [
                {
                    "id": 0,
                    "name": "HYDRO_1_LATE",
                    "bus_id": 1,
                    "min_generation_mw": 0.0,
                    "max_generation_mw": 20000.0,
                    "min_turbined_m3s": 0.0,
                    "max_turbined_m3s": 20000.0
                }
            ]
        },
        {
            "id": 2,
            "name": "HYDRO_2_EARLY",
            "operational_start_date": "2020-01-01",
            "downstream_id": null,
            "reservoir": { "min_storage_hm3": 0.0, "max_storage_hm3": 500.0 },
            "outflow": { "min_outflow_m3s": 0.0, "max_outflow_m3s": null },
            "generation": {
                "model": "constant_productivity",
                "min_turbined_m3s": 0.0,
                "max_turbined_m3s": 300.0,
                "min_generation_mw": 0.0,
                "max_generation_mw": 300.0
            },
            "unit_groups": [
                {
                    "id": 0,
                    "name": "HYDRO_2_EARLY",
                    "bus_id": 1,
                    "min_generation_mw": 0.0,
                    "max_generation_mw": 300.0,
                    "min_turbined_m3s": 0.0,
                    "max_turbined_m3s": 300.0
                }
            ]
        }
    ]
}"#,
    );

    helpers::write_file(
        root,
        "system/hydro_production_models.json",
        r#"{
    "production_models": [
        {
            "hydro_id": 1,
            "selection_mode": "stage_ranges",
            "stage_ranges": [
                {
                    "start_stage_id": 0,
                    "end_stage_id": null,
                    "model": "constant_productivity",
                    "productivity_mw_per_m3s": 0.9
                }
            ]
        },
        {
            "hydro_id": 2,
            "selection_mode": "stage_ranges",
            "stage_ranges": [
                {
                    "start_stage_id": 0,
                    "end_stage_id": null,
                    "model": "constant_productivity",
                    "productivity_mw_per_m3s": 0.9
                }
            ]
        }
    ]
}"#,
    );

    std::fs::create_dir_all(root.join("constraints")).unwrap();
    let schema = Arc::new(Schema::new(vec![
        Field::new("hydro_id", DataType::Int32, false),
        Field::new("stage_id", DataType::Int32, false),
        Field::new("max_turbined_m3s", DataType::Float64, true),
        Field::new("min_turbined_m3s", DataType::Float64, true),
        Field::new("block_id", DataType::Int32, true),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![1, 1])),
            Arc::new(Int32Array::from(vec![1, 1])),
            Arc::new(Float64Array::from(vec![Some(12345.0), None])),
            Arc::new(Float64Array::from(vec![None, Some(777.0)])),
            Arc::new(Int32Array::from(vec![None, Some(1)])),
        ],
    )
    .unwrap();
    let file = std::fs::File::create(root.join("constraints/hydro_bounds.parquet")).unwrap();
    let mut writer = ArrowWriter::try_new(file, batch.schema(), None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

#[test]
fn hydro_bound_override_follows_declared_id_through_canonical_resort() {
    let dir = TempDir::new().unwrap();
    make_id_date_misaligned_case(&dir);

    let system = load_case(dir.path()).unwrap_or_else(|e| panic!("expected Ok(System): {e}"));

    let hydros = system.hydros();
    assert_eq!(hydros.len(), 2, "expected exactly 2 hydros");

    let pos_h1 = hydros
        .iter()
        .position(|h| h.id == EntityId::from(1))
        .expect("hydro 1 present in System::hydros()");
    let pos_h2 = hydros
        .iter()
        .position(|h| h.id == EntityId::from(2))
        .expect("hydro 2 present in System::hydros()");

    // Precondition: the misalignment opportunity must actually be present --
    // canonical (date, id) order must differ from parse (id) order.
    assert_eq!(
        pos_h2, 0,
        "hydro 2 (earlier operational_start_date) must sort first in canonical order"
    );
    assert_eq!(
        pos_h1, 1,
        "hydro 1 (later operational_start_date) must sort second in canonical \
         order, while its parse-order/id-order position was 0"
    );

    let stage1_idx = 1;
    let block1_idx = 1;

    let bounds = system.bounds();
    let h1_resolved = bounds.hydro_bounds_at_block(pos_h1, stage1_idx, block1_idx);
    let h2_resolved = bounds.hydro_bounds_at_block(pos_h2, stage1_idx, block1_idx);

    assert!(
        (h1_resolved.max_turbined_m3s - 12345.0).abs() < f64::EPSILON,
        "hydro 1 (declared the override, canonical position {pos_h1}) should \
         read back its own stage-wide override max_turbined_m3s=12345.0, got {}",
        h1_resolved.max_turbined_m3s
    );
    assert!(
        (h1_resolved.min_turbined_m3s - 777.0).abs() < f64::EPSILON,
        "hydro 1 (declared the override, canonical position {pos_h1}) should \
         read back its own per-block override min_turbined_m3s=777.0 at block \
         1, got {}",
        h1_resolved.min_turbined_m3s
    );

    assert!(
        (h2_resolved.max_turbined_m3s - 300.0).abs() < f64::EPSILON,
        "hydro 2 (declared NO override, canonical position {pos_h2}) should \
         read back its own base max_turbined_m3s=300.0, got {} (12345.0 would \
         mean hydro 1's override leaked onto hydro 2's canonical slot)",
        h2_resolved.max_turbined_m3s
    );
    assert!(
        (h2_resolved.min_turbined_m3s - 0.0).abs() < f64::EPSILON,
        "hydro 2 (declared NO per-block override, canonical position \
         {pos_h2}) should read back its own base min_turbined_m3s=0.0, got {} \
         (777.0 would mean hydro 1's override leaked onto hydro 2's canonical \
         slot)",
        h2_resolved.min_turbined_m3s
    );
}

// ─── The same defect, pinned per canonical family ──────────────────────────
//
// `resolve_bounds`, `resolve_penalties`, `resolve_ncs_bounds` all index their
// output by position in a `data.<family>` slice exactly like `resolve_bounds`
// indexed `data.hydros` above; each of the other six canonical families gets
// its own fixture and its own override family so a per-family regression is
// pinned independently of the hydro case.

/// Single-stage, single-block `stages.json` shared by the per-family and
/// key-duality fixtures below, none of which need more than one stage to
/// exercise the alignment.
const SINGLE_STAGE_STAGES_JSON: &str = r#"{
    "policy_graph": {
        "type": "finite_horizon",
        "annual_discount_rate": 0.06,
        "transitions": []
    },
    "stages": [
        {
            "id": 0,
            "start_date": "2024-01-01",
            "end_date": "2024-02-01",
            "blocks": [{ "id": 0, "name": "B0", "hours": 744.0 }],
            "num_openings": 1
        }
    ]
}"#;

/// Two buses whose `operational_start_date` order is the REVERSE of their id
/// order (bus 1 dated 2030, bus 2 dated 2020), plus a
/// `constraints/penalty_overrides_bus.parquet` override declared only for bus
/// 1 at stage 0 (`excess_cost=99999.0`). Exercises `resolve_penalties`'s bus
/// axis, for the `buses` family.
fn make_bus_id_date_misaligned_case(dir: &TempDir) {
    let root = dir.path();

    helpers::write_file(root, "config.json", helpers::VALID_CONFIG_JSON);
    helpers::write_file(root, "penalties.json", helpers::VALID_PENALTIES_JSON);
    helpers::write_file(
        root,
        "initial_conditions.json",
        helpers::VALID_INITIAL_CONDITIONS_JSON,
    );
    helpers::write_file(root, "stages.json", SINGLE_STAGE_STAGES_JSON);

    helpers::write_file(
        root,
        "system/buses.json",
        r#"{
    "buses": [
        { "id": 1, "name": "BUS_1_LATE", "operational_start_date": "2030-01-01" },
        { "id": 2, "name": "BUS_2_EARLY", "operational_start_date": "2020-01-01" }
    ]
}"#,
    );
    helpers::write_file(root, "system/lines.json", r#"{ "lines": [] }"#);
    helpers::write_file(root, "system/hydros.json", r#"{ "hydros": [] }"#);
    helpers::write_file(root, "system/thermals.json", r#"{ "thermals": [] }"#);

    std::fs::create_dir_all(root.join("constraints")).unwrap();
    let schema = Arc::new(Schema::new(vec![
        Field::new("bus_id", DataType::Int32, false),
        Field::new("stage_id", DataType::Int32, false),
        Field::new("excess_cost", DataType::Float64, true),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![1])),
            Arc::new(Int32Array::from(vec![0])),
            Arc::new(Float64Array::from(vec![Some(99999.0)])),
        ],
    )
    .unwrap();
    let file =
        std::fs::File::create(root.join("constraints/penalty_overrides_bus.parquet")).unwrap();
    let mut writer = ArrowWriter::try_new(file, batch.schema(), None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

/// A `resolve_penalties` bus-axis override follows its declared bus through
/// the canonical resort exactly like the hydro override case above.
#[test]
fn bus_penalty_override_follows_declared_id_through_canonical_resort() {
    let dir = TempDir::new().unwrap();
    make_bus_id_date_misaligned_case(&dir);

    let system = load_case(dir.path()).unwrap_or_else(|e| panic!("expected Ok(System): {e}"));

    let buses = system.buses();
    assert_eq!(buses.len(), 2, "expected exactly 2 buses");

    let pos_b1 = buses
        .iter()
        .position(|b| b.id == EntityId::from(1))
        .expect("bus 1 present in System::buses()");
    let pos_b2 = buses
        .iter()
        .position(|b| b.id == EntityId::from(2))
        .expect("bus 2 present in System::buses()");

    assert_eq!(
        pos_b2, 0,
        "bus 2 (earlier operational_start_date) must sort first in canonical order"
    );
    assert_eq!(
        pos_b1, 1,
        "bus 1 (later operational_start_date) must sort second in canonical order"
    );

    let penalties = system.penalties();
    let b1 = penalties.bus_penalties(pos_b1, 0);
    let b2 = penalties.bus_penalties(pos_b2, 0);

    assert!(
        (b1.excess_cost - 99999.0).abs() < f64::EPSILON,
        "bus 1 (declared the override, canonical position {pos_b1}) should read \
         back its own override excess_cost=99999.0, got {}",
        b1.excess_cost
    );
    assert!(
        (b2.excess_cost - 100.0).abs() < f64::EPSILON,
        "bus 2 (declared NO override, canonical position {pos_b2}) should read \
         back its base excess_cost=100.0, got {} (99999.0 would mean bus 1's \
         override leaked onto bus 2's canonical slot)",
        b2.excess_cost
    );
}

/// Two transmission lines whose `operational_start_date` order is the REVERSE
/// of their id order (line 1 dated 2030, line 2 dated 2020), plus a
/// `constraints/line_bounds.parquet` override declared only for line 1 at
/// stage 0 (`direct_mw=99999.0`). Exercises `resolve_bounds`'s line axis — the
/// `lines` family.
fn make_line_id_date_misaligned_case(dir: &TempDir) {
    let root = dir.path();

    helpers::write_file(root, "config.json", helpers::VALID_CONFIG_JSON);
    helpers::write_file(root, "penalties.json", helpers::VALID_PENALTIES_JSON);
    helpers::write_file(
        root,
        "initial_conditions.json",
        helpers::VALID_INITIAL_CONDITIONS_JSON,
    );
    helpers::write_file(root, "stages.json", SINGLE_STAGE_STAGES_JSON);

    helpers::write_file(
        root,
        "system/buses.json",
        r#"{
    "buses": [
        { "id": 1, "name": "BUS_1", "operational_start_date": "2024-01-01" },
        { "id": 2, "name": "BUS_2", "operational_start_date": "2024-01-01" }
    ]
}"#,
    );
    helpers::write_file(
        root,
        "system/lines.json",
        r#"{
    "lines": [
        {
            "id": 1,
            "name": "LINE_1_LATE",
            "operational_start_date": "2030-01-01",
            "source_bus_id": 1,
            "target_bus_id": 2,
            "capacity": { "direct_mw": 20000.0, "reverse_mw": 15000.0 }
        },
        {
            "id": 2,
            "name": "LINE_2_EARLY",
            "operational_start_date": "2020-01-01",
            "source_bus_id": 1,
            "target_bus_id": 2,
            "capacity": { "direct_mw": 300.0, "reverse_mw": 250.0 }
        }
    ]
}"#,
    );
    helpers::write_file(root, "system/hydros.json", r#"{ "hydros": [] }"#);
    helpers::write_file(root, "system/thermals.json", r#"{ "thermals": [] }"#);

    std::fs::create_dir_all(root.join("constraints")).unwrap();
    let schema = Arc::new(Schema::new(vec![
        Field::new("line_id", DataType::Int32, false),
        Field::new("stage_id", DataType::Int32, false),
        Field::new("direct_mw", DataType::Float64, true),
        Field::new("reverse_mw", DataType::Float64, true),
        Field::new("block_id", DataType::Int32, true),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![1])),
            Arc::new(Int32Array::from(vec![0])),
            Arc::new(Float64Array::from(vec![Some(99999.0)])),
            Arc::new(Float64Array::from(vec![None])),
            Arc::new(Int32Array::from(vec![None])),
        ],
    )
    .unwrap();
    let file = std::fs::File::create(root.join("constraints/line_bounds.parquet")).unwrap();
    let mut writer = ArrowWriter::try_new(file, batch.schema(), None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

/// A `resolve_bounds` line-axis override follows its declared line through
/// the canonical resort exactly like the hydro override case above.
#[test]
fn line_bound_override_follows_declared_id_through_canonical_resort() {
    let dir = TempDir::new().unwrap();
    make_line_id_date_misaligned_case(&dir);

    let system = load_case(dir.path()).unwrap_or_else(|e| panic!("expected Ok(System): {e}"));

    let lines = system.lines();
    assert_eq!(lines.len(), 2, "expected exactly 2 lines");

    let pos_l1 = lines
        .iter()
        .position(|l| l.id == EntityId::from(1))
        .expect("line 1 present in System::lines()");
    let pos_l2 = lines
        .iter()
        .position(|l| l.id == EntityId::from(2))
        .expect("line 2 present in System::lines()");

    assert_eq!(
        pos_l2, 0,
        "line 2 (earlier operational_start_date) must sort first in canonical order"
    );
    assert_eq!(
        pos_l1, 1,
        "line 1 (later operational_start_date) must sort second in canonical order"
    );

    let bounds = system.bounds();
    let l1 = bounds.line_block_base(pos_l1, 0);
    let l2 = bounds.line_block_base(pos_l2, 0);

    assert!(
        (l1.direct_mw - 99999.0).abs() < f64::EPSILON,
        "line 1 (declared the override, canonical position {pos_l1}) should \
         read back its own override direct_mw=99999.0, got {}",
        l1.direct_mw
    );
    assert!(
        (l2.direct_mw - 300.0).abs() < f64::EPSILON,
        "line 2 (declared NO override, canonical position {pos_l2}) should \
         read back its base direct_mw=300.0, got {} (99999.0 would mean line \
         1's override leaked onto line 2's canonical slot)",
        l2.direct_mw
    );
}

/// Two thermal plants whose `operational_start_date` order is the REVERSE of
/// their id order (thermal 1 dated 2030, thermal 2 dated 2020), plus a
/// `constraints/thermal_bounds.parquet` override declared only for thermal 1
/// at stage 0 (`max_generation_mw=99999.0`). Exercises `resolve_bounds`'s
/// thermal axis, for the `thermals` family.
fn make_thermal_id_date_misaligned_case(dir: &TempDir) {
    let root = dir.path();

    helpers::write_file(root, "config.json", helpers::VALID_CONFIG_JSON);
    helpers::write_file(root, "penalties.json", helpers::VALID_PENALTIES_JSON);
    helpers::write_file(
        root,
        "initial_conditions.json",
        helpers::VALID_INITIAL_CONDITIONS_JSON,
    );
    helpers::write_file(root, "stages.json", SINGLE_STAGE_STAGES_JSON);

    helpers::write_file(
        root,
        "system/buses.json",
        r#"{ "buses": [{ "id": 1, "name": "BUS_1", "operational_start_date": "2024-01-01" }] }"#,
    );
    helpers::write_file(root, "system/lines.json", r#"{ "lines": [] }"#);
    helpers::write_file(root, "system/hydros.json", r#"{ "hydros": [] }"#);
    helpers::write_file(
        root,
        "system/thermals.json",
        r#"{
    "thermals": [
        {
            "id": 1,
            "name": "THERMAL_1_LATE",
            "operational_start_date": "2030-01-01",
            "bus_id": 1,
            "cost_per_mwh": 80.0,
            "generation": { "min_mw": 0.0, "max_mw": 20000.0 }
        },
        {
            "id": 2,
            "name": "THERMAL_2_EARLY",
            "operational_start_date": "2020-01-01",
            "bus_id": 1,
            "cost_per_mwh": 80.0,
            "generation": { "min_mw": 0.0, "max_mw": 300.0 }
        }
    ]
}"#,
    );

    std::fs::create_dir_all(root.join("constraints")).unwrap();
    let schema = Arc::new(Schema::new(vec![
        Field::new("thermal_id", DataType::Int32, false),
        Field::new("stage_id", DataType::Int32, false),
        Field::new("min_generation_mw", DataType::Float64, true),
        Field::new("max_generation_mw", DataType::Float64, true),
        Field::new("cost_per_mwh", DataType::Float64, true),
        Field::new("block_id", DataType::Int32, true),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![1])),
            Arc::new(Int32Array::from(vec![0])),
            Arc::new(Float64Array::from(vec![None])),
            Arc::new(Float64Array::from(vec![Some(99999.0)])),
            Arc::new(Float64Array::from(vec![None])),
            Arc::new(Int32Array::from(vec![None])),
        ],
    )
    .unwrap();
    let file = std::fs::File::create(root.join("constraints/thermal_bounds.parquet")).unwrap();
    let mut writer = ArrowWriter::try_new(file, batch.schema(), None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

/// A `resolve_bounds` thermal-axis override follows its declared thermal
/// through the canonical resort exactly like the hydro override case above.
#[test]
fn thermal_bound_override_follows_declared_id_through_canonical_resort() {
    let dir = TempDir::new().unwrap();
    make_thermal_id_date_misaligned_case(&dir);

    let system = load_case(dir.path()).unwrap_or_else(|e| panic!("expected Ok(System): {e}"));

    let thermals = system.thermals();
    assert_eq!(thermals.len(), 2, "expected exactly 2 thermals");

    let pos_t1 = thermals
        .iter()
        .position(|t| t.id == EntityId::from(1))
        .expect("thermal 1 present in System::thermals()");
    let pos_t2 = thermals
        .iter()
        .position(|t| t.id == EntityId::from(2))
        .expect("thermal 2 present in System::thermals()");

    assert_eq!(
        pos_t2, 0,
        "thermal 2 (earlier operational_start_date) must sort first in canonical order"
    );
    assert_eq!(
        pos_t1, 1,
        "thermal 1 (later operational_start_date) must sort second in canonical order"
    );

    let bounds = system.bounds();
    let t1 = bounds.thermal_block_base(pos_t1, 0);
    let t2 = bounds.thermal_block_base(pos_t2, 0);

    assert!(
        (t1.max_generation_mw - 99999.0).abs() < f64::EPSILON,
        "thermal 1 (declared the override, canonical position {pos_t1}) should \
         read back its own override max_generation_mw=99999.0, got {}",
        t1.max_generation_mw
    );
    assert!(
        (t2.max_generation_mw - 300.0).abs() < f64::EPSILON,
        "thermal 2 (declared NO override, canonical position {pos_t2}) should \
         read back its base max_generation_mw=300.0, got {} (99999.0 would \
         mean thermal 1's override leaked onto thermal 2's canonical slot)",
        t2.max_generation_mw
    );
}

/// Two hydros with valid, distinct reservoirs used only as the fixed
/// source/destination endpoints for the pumping-station and contract
/// fixtures below — neither hydro's own id/date order is under test here.
const TWO_ANCHOR_HYDROS_JSON: &str = r#"{
    "hydros": [
        {
            "id": 10,
            "name": "HYDRO_SRC",
            "operational_start_date": "2024-01-01",
            "downstream_id": null,
            "reservoir": { "min_storage_hm3": 0.0, "max_storage_hm3": 100.0 },
            "outflow": { "min_outflow_m3s": 0.0, "max_outflow_m3s": null },
            "generation": {
                "model": "constant_productivity",
                "min_turbined_m3s": 0.0,
                "max_turbined_m3s": 50.0,
                "min_generation_mw": 0.0,
                "max_generation_mw": 50.0
            },
            "unit_groups": [
                {
                    "id": 0,
                    "name": "HYDRO_SRC",
                    "bus_id": 1,
                    "min_generation_mw": 0.0,
                    "max_generation_mw": 50.0,
                    "min_turbined_m3s": 0.0,
                    "max_turbined_m3s": 50.0
                }
            ]
        },
        {
            "id": 20,
            "name": "HYDRO_DST",
            "operational_start_date": "2024-01-01",
            "downstream_id": null,
            "reservoir": { "min_storage_hm3": 0.0, "max_storage_hm3": 100.0 },
            "outflow": { "min_outflow_m3s": 0.0, "max_outflow_m3s": null },
            "generation": {
                "model": "constant_productivity",
                "min_turbined_m3s": 0.0,
                "max_turbined_m3s": 50.0,
                "min_generation_mw": 0.0,
                "max_generation_mw": 50.0
            },
            "unit_groups": [
                {
                    "id": 0,
                    "name": "HYDRO_DST",
                    "bus_id": 1,
                    "min_generation_mw": 0.0,
                    "max_generation_mw": 50.0,
                    "min_turbined_m3s": 0.0,
                    "max_turbined_m3s": 50.0
                }
            ]
        }
    ]
}"#;

/// `hydro_production_models.json` covering both [`TWO_ANCHOR_HYDROS_JSON`] plants.
const TWO_ANCHOR_HYDROS_PRODUCTION_MODELS_JSON: &str = r#"{
    "production_models": [
        {
            "hydro_id": 10,
            "selection_mode": "stage_ranges",
            "stage_ranges": [
                { "start_stage_id": 0, "end_stage_id": null, "model": "constant_productivity", "productivity_mw_per_m3s": 0.9 }
            ]
        },
        {
            "hydro_id": 20,
            "selection_mode": "stage_ranges",
            "stage_ranges": [
                { "start_stage_id": 0, "end_stage_id": null, "model": "constant_productivity", "productivity_mw_per_m3s": 0.9 }
            ]
        }
    ]
}"#;

/// Two pumping stations whose `operational_start_date` order is the REVERSE
/// of their id order (station 1 dated 2030, station 2 dated 2020), both
/// moving water from hydro 10 to hydro 20, plus a
/// `constraints/pumping_bounds.parquet` override declared only for station 1
/// at stage 0 (`max_m3s=99999.0`). Exercises `resolve_bounds`'s pumping axis —
/// the `pumping_stations` family.
fn make_pumping_id_date_misaligned_case(dir: &TempDir) {
    let root = dir.path();

    helpers::write_file(root, "config.json", helpers::VALID_CONFIG_JSON);
    helpers::write_file(root, "penalties.json", helpers::VALID_PENALTIES_JSON);
    helpers::write_file(
        root,
        "initial_conditions.json",
        helpers::VALID_INITIAL_CONDITIONS_JSON,
    );
    helpers::write_file(root, "stages.json", SINGLE_STAGE_STAGES_JSON);

    helpers::write_file(
        root,
        "system/buses.json",
        r#"{ "buses": [{ "id": 1, "name": "BUS_1", "operational_start_date": "2024-01-01" }] }"#,
    );
    helpers::write_file(root, "system/lines.json", r#"{ "lines": [] }"#);
    helpers::write_file(root, "system/thermals.json", r#"{ "thermals": [] }"#);
    helpers::write_file(root, "system/hydros.json", TWO_ANCHOR_HYDROS_JSON);
    helpers::write_file(
        root,
        "system/hydro_production_models.json",
        TWO_ANCHOR_HYDROS_PRODUCTION_MODELS_JSON,
    );

    helpers::write_file(
        root,
        "system/pumping_stations.json",
        r#"{
    "pumping_stations": [
        {
            "id": 1,
            "name": "PUMP_1_LATE",
            "operational_start_date": "2030-01-01",
            "bus_id": 1,
            "source_hydro_id": 10,
            "destination_hydro_id": 20,
            "consumption_mw_per_m3s": 0.5,
            "flow": { "min_m3s": 0.0, "max_m3s": 20000.0 }
        },
        {
            "id": 2,
            "name": "PUMP_2_EARLY",
            "operational_start_date": "2020-01-01",
            "bus_id": 1,
            "source_hydro_id": 10,
            "destination_hydro_id": 20,
            "consumption_mw_per_m3s": 0.5,
            "flow": { "min_m3s": 0.0, "max_m3s": 300.0 }
        }
    ]
}"#,
    );

    std::fs::create_dir_all(root.join("constraints")).unwrap();
    let schema = Arc::new(Schema::new(vec![
        Field::new("pumping_station_id", DataType::Int32, false),
        Field::new("stage_id", DataType::Int32, false),
        Field::new("min_m3s", DataType::Float64, true),
        Field::new("max_m3s", DataType::Float64, true),
        Field::new("block_id", DataType::Int32, true),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![1])),
            Arc::new(Int32Array::from(vec![0])),
            Arc::new(Float64Array::from(vec![None])),
            Arc::new(Float64Array::from(vec![Some(99999.0)])),
            Arc::new(Int32Array::from(vec![None])),
        ],
    )
    .unwrap();
    let file = std::fs::File::create(root.join("constraints/pumping_bounds.parquet")).unwrap();
    let mut writer = ArrowWriter::try_new(file, batch.schema(), None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

/// A `resolve_bounds` pumping-axis override follows its declared station
/// through the canonical resort exactly like the hydro override case above.
#[test]
fn pumping_bound_override_follows_declared_id_through_canonical_resort() {
    let dir = TempDir::new().unwrap();
    make_pumping_id_date_misaligned_case(&dir);

    let system = load_case(dir.path()).unwrap_or_else(|e| panic!("expected Ok(System): {e}"));

    let stations = system.pumping_stations();
    assert_eq!(stations.len(), 2, "expected exactly 2 pumping stations");

    let pos_p1 = stations
        .iter()
        .position(|p| p.id == EntityId::from(1))
        .expect("station 1 present in System::pumping_stations()");
    let pos_p2 = stations
        .iter()
        .position(|p| p.id == EntityId::from(2))
        .expect("station 2 present in System::pumping_stations()");

    assert_eq!(
        pos_p2, 0,
        "station 2 (earlier operational_start_date) must sort first in canonical order"
    );
    assert_eq!(
        pos_p1, 1,
        "station 1 (later operational_start_date) must sort second in canonical order"
    );

    let bounds = system.bounds();
    let p1 = bounds.pumping_block_base(pos_p1, 0);
    let p2 = bounds.pumping_block_base(pos_p2, 0);

    assert!(
        (p1.max_flow_m3s - 99999.0).abs() < f64::EPSILON,
        "station 1 (declared the override, canonical position {pos_p1}) should \
         read back its own override max_m3s=99999.0, got {}",
        p1.max_flow_m3s
    );
    assert!(
        (p2.max_flow_m3s - 300.0).abs() < f64::EPSILON,
        "station 2 (declared NO override, canonical position {pos_p2}) should \
         read back its base max_m3s=300.0, got {} (99999.0 would mean station \
         1's override leaked onto station 2's canonical slot)",
        p2.max_flow_m3s
    );
}

/// Two energy contracts whose `operational_start_date` order is the REVERSE
/// of their id order (contract 1 dated 2030, contract 2 dated 2020), plus a
/// `constraints/contract_bounds.parquet` override declared only for contract 1
/// at stage 0 (`max_mw=99999.0`). Exercises `resolve_bounds`'s contract axis —
/// the `contracts` family.
fn make_contract_id_date_misaligned_case(dir: &TempDir) {
    let root = dir.path();

    helpers::write_file(root, "config.json", helpers::VALID_CONFIG_JSON);
    helpers::write_file(root, "penalties.json", helpers::VALID_PENALTIES_JSON);
    helpers::write_file(
        root,
        "initial_conditions.json",
        helpers::VALID_INITIAL_CONDITIONS_JSON,
    );
    helpers::write_file(root, "stages.json", SINGLE_STAGE_STAGES_JSON);

    helpers::write_file(
        root,
        "system/buses.json",
        r#"{ "buses": [{ "id": 1, "name": "BUS_1", "operational_start_date": "2024-01-01" }] }"#,
    );
    helpers::write_file(root, "system/lines.json", r#"{ "lines": [] }"#);
    helpers::write_file(root, "system/hydros.json", r#"{ "hydros": [] }"#);
    helpers::write_file(root, "system/thermals.json", r#"{ "thermals": [] }"#);

    helpers::write_file(
        root,
        "system/energy_contracts.json",
        r#"{
    "contracts": [
        {
            "id": 1,
            "name": "CONTRACT_1_LATE",
            "operational_start_date": "2030-01-01",
            "bus_id": 1,
            "type": "import",
            "price_per_mwh": 50.0,
            "limits": { "min_mw": 0.0, "max_mw": 20000.0 }
        },
        {
            "id": 2,
            "name": "CONTRACT_2_EARLY",
            "operational_start_date": "2020-01-01",
            "bus_id": 1,
            "type": "import",
            "price_per_mwh": 50.0,
            "limits": { "min_mw": 0.0, "max_mw": 300.0 }
        }
    ]
}"#,
    );

    std::fs::create_dir_all(root.join("constraints")).unwrap();
    let schema = Arc::new(Schema::new(vec![
        Field::new("contract_id", DataType::Int32, false),
        Field::new("stage_id", DataType::Int32, false),
        Field::new("min_mw", DataType::Float64, true),
        Field::new("max_mw", DataType::Float64, true),
        Field::new("price_per_mwh", DataType::Float64, true),
        Field::new("block_id", DataType::Int32, true),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![1])),
            Arc::new(Int32Array::from(vec![0])),
            Arc::new(Float64Array::from(vec![None])),
            Arc::new(Float64Array::from(vec![Some(99999.0)])),
            Arc::new(Float64Array::from(vec![None])),
            Arc::new(Int32Array::from(vec![None])),
        ],
    )
    .unwrap();
    let file = std::fs::File::create(root.join("constraints/contract_bounds.parquet")).unwrap();
    let mut writer = ArrowWriter::try_new(file, batch.schema(), None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

/// A `resolve_bounds` contract-axis override follows its declared contract
/// through the canonical resort exactly like the hydro override case above.
#[test]
fn contract_bound_override_follows_declared_id_through_canonical_resort() {
    let dir = TempDir::new().unwrap();
    make_contract_id_date_misaligned_case(&dir);

    let system = load_case(dir.path()).unwrap_or_else(|e| panic!("expected Ok(System): {e}"));

    let contracts = system.contracts();
    assert_eq!(contracts.len(), 2, "expected exactly 2 contracts");

    let pos_c1 = contracts
        .iter()
        .position(|c| c.id == EntityId::from(1))
        .expect("contract 1 present in System::contracts()");
    let pos_c2 = contracts
        .iter()
        .position(|c| c.id == EntityId::from(2))
        .expect("contract 2 present in System::contracts()");

    assert_eq!(
        pos_c2, 0,
        "contract 2 (earlier operational_start_date) must sort first in canonical order"
    );
    assert_eq!(
        pos_c1, 1,
        "contract 1 (later operational_start_date) must sort second in canonical order"
    );

    let bounds = system.bounds();
    let c1 = bounds.contract_block_base(pos_c1, 0);
    let c2 = bounds.contract_block_base(pos_c2, 0);

    assert!(
        (c1.max_mw - 99999.0).abs() < f64::EPSILON,
        "contract 1 (declared the override, canonical position {pos_c1}) should \
         read back its own override max_mw=99999.0, got {}",
        c1.max_mw
    );
    assert!(
        (c2.max_mw - 300.0).abs() < f64::EPSILON,
        "contract 2 (declared NO override, canonical position {pos_c2}) should \
         read back its base max_mw=300.0, got {} (99999.0 would mean contract \
         1's override leaked onto contract 2's canonical slot)",
        c2.max_mw
    );
}

/// Two non-controllable sources whose `operational_start_date` order is the
/// REVERSE of their id order (source 1 dated 2030, source 2 dated 2020), plus
/// a `constraints/ncs_bounds.parquet` override declared only for source 1 at
/// stage 0 (`available_generation_mw=99999.0`). Exercises
/// `resolve_ncs_bounds`, for the `non_controllable_sources` family.
fn make_ncs_id_date_misaligned_case(dir: &TempDir) {
    let root = dir.path();

    helpers::write_file(root, "config.json", helpers::VALID_CONFIG_JSON);
    helpers::write_file(root, "penalties.json", helpers::VALID_PENALTIES_JSON);
    helpers::write_file(
        root,
        "initial_conditions.json",
        helpers::VALID_INITIAL_CONDITIONS_JSON,
    );
    helpers::write_file(root, "stages.json", SINGLE_STAGE_STAGES_JSON);

    helpers::write_file(
        root,
        "system/buses.json",
        r#"{ "buses": [{ "id": 1, "name": "BUS_1", "operational_start_date": "2024-01-01" }] }"#,
    );
    helpers::write_file(root, "system/lines.json", r#"{ "lines": [] }"#);
    helpers::write_file(root, "system/hydros.json", r#"{ "hydros": [] }"#);
    helpers::write_file(root, "system/thermals.json", r#"{ "thermals": [] }"#);

    helpers::write_file(
        root,
        "system/non_controllable_sources.json",
        r#"{
    "non_controllable_sources": [
        {
            "id": 1,
            "name": "NCS_1_LATE",
            "operational_start_date": "2030-01-01",
            "bus_id": 1,
            "max_generation_mw": 20000.0
        },
        {
            "id": 2,
            "name": "NCS_2_EARLY",
            "operational_start_date": "2020-01-01",
            "bus_id": 1,
            "max_generation_mw": 300.0
        }
    ]
}"#,
    );

    std::fs::create_dir_all(root.join("constraints")).unwrap();
    let schema = Arc::new(Schema::new(vec![
        Field::new("ncs_id", DataType::Int32, false),
        Field::new("stage_id", DataType::Int32, false),
        Field::new("available_generation_mw", DataType::Float64, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![1])),
            Arc::new(Int32Array::from(vec![0])),
            Arc::new(Float64Array::from(vec![99999.0])),
        ],
    )
    .unwrap();
    let file = std::fs::File::create(root.join("constraints/ncs_bounds.parquet")).unwrap();
    let mut writer = ArrowWriter::try_new(file, batch.schema(), None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

/// A `resolve_ncs_bounds` override follows its declared source through the
/// canonical resort exactly like the hydro override case above.
#[test]
fn ncs_bound_override_follows_declared_id_through_canonical_resort() {
    let dir = TempDir::new().unwrap();
    make_ncs_id_date_misaligned_case(&dir);

    let system = load_case(dir.path()).unwrap_or_else(|e| panic!("expected Ok(System): {e}"));

    let ncs = system.non_controllable_sources();
    assert_eq!(ncs.len(), 2, "expected exactly 2 non-controllable sources");

    let pos_n1 = ncs
        .iter()
        .position(|n| n.id == EntityId::from(1))
        .expect("source 1 present in System::non_controllable_sources()");
    let pos_n2 = ncs
        .iter()
        .position(|n| n.id == EntityId::from(2))
        .expect("source 2 present in System::non_controllable_sources()");

    assert_eq!(
        pos_n2, 0,
        "source 2 (earlier operational_start_date) must sort first in canonical order"
    );
    assert_eq!(
        pos_n1, 1,
        "source 1 (later operational_start_date) must sort second in canonical order"
    );

    let ncs_bounds = system.resolved_ncs_bounds();
    let n1 = ncs_bounds.available_generation(pos_n1, 0);
    let n2 = ncs_bounds.available_generation(pos_n2, 0);

    assert!(
        (n1 - 99999.0).abs() < f64::EPSILON,
        "source 1 (declared the override, canonical position {pos_n1}) should \
         read back its own override available_generation_mw=99999.0, got {n1}"
    );
    assert!(
        (n2 - 300.0).abs() < f64::EPSILON,
        "source 2 (declared NO override, canonical position {pos_n2}) should \
         read back its base available_generation_mw=300.0, got {n2} (99999.0 \
         would mean source 1's override leaked onto source 2's canonical slot)"
    );
}

/// Two hydros whose `operational_start_date` order is the REVERSE of their id
/// order (hydro 1 dated 2030, hydro 2 dated 2020), each declaring exactly one
/// hydro unit group (declaration is mandatory), plus a
/// `constraints/hydro_unit_group_bounds.parquet` override declared only for
/// hydro 1's group at stage 0 (`max_turbined_m3s=12345.0`). Exercises
/// `resolve_hydro_unit_group_bounds` -- the group axis, for the
/// `hydro_unit_groups` family.
fn make_hydro_unit_group_id_date_misaligned_case(dir: &TempDir) {
    let root = dir.path();

    helpers::write_file(root, "config.json", helpers::VALID_CONFIG_JSON);
    helpers::write_file(root, "penalties.json", helpers::VALID_PENALTIES_JSON);
    helpers::write_file(
        root,
        "initial_conditions.json",
        helpers::VALID_INITIAL_CONDITIONS_JSON,
    );
    helpers::write_file(root, "stages.json", SINGLE_STAGE_STAGES_JSON);

    helpers::write_file(
        root,
        "system/buses.json",
        r#"{ "buses": [{ "id": 1, "name": "BUS_1", "operational_start_date": "2024-01-01" }] }"#,
    );
    helpers::write_file(root, "system/lines.json", r#"{ "lines": [] }"#);
    helpers::write_file(root, "system/thermals.json", r#"{ "thermals": [] }"#);

    helpers::write_file(
        root,
        "system/hydros.json",
        r#"{
    "hydros": [
        {
            "id": 1,
            "name": "HYDRO_1_LATE",
            "operational_start_date": "2030-01-01",
            "downstream_id": null,
            "reservoir": { "min_storage_hm3": 0.0, "max_storage_hm3": 1000.0 },
            "outflow": { "min_outflow_m3s": 0.0, "max_outflow_m3s": null },
            "generation": {
                "model": "constant_productivity",
                "min_turbined_m3s": 0.0,
                "max_turbined_m3s": 20000.0,
                "min_generation_mw": 0.0,
                "max_generation_mw": 20000.0
            },
            "unit_groups": [
                {
                    "id": 0,
                    "name": "HYDRO_1_LATE",
                    "bus_id": 1,
                    "min_generation_mw": 0.0,
                    "max_generation_mw": 20000.0,
                    "min_turbined_m3s": 0.0,
                    "max_turbined_m3s": 20000.0
                }
            ]
        },
        {
            "id": 2,
            "name": "HYDRO_2_EARLY",
            "operational_start_date": "2020-01-01",
            "downstream_id": null,
            "reservoir": { "min_storage_hm3": 0.0, "max_storage_hm3": 500.0 },
            "outflow": { "min_outflow_m3s": 0.0, "max_outflow_m3s": null },
            "generation": {
                "model": "constant_productivity",
                "min_turbined_m3s": 0.0,
                "max_turbined_m3s": 300.0,
                "min_generation_mw": 0.0,
                "max_generation_mw": 300.0
            },
            "unit_groups": [
                {
                    "id": 0,
                    "name": "HYDRO_2_EARLY",
                    "bus_id": 1,
                    "min_generation_mw": 0.0,
                    "max_generation_mw": 300.0,
                    "min_turbined_m3s": 0.0,
                    "max_turbined_m3s": 300.0
                }
            ]
        }
    ]
}"#,
    );

    helpers::write_file(
        root,
        "system/hydro_production_models.json",
        r#"{
    "production_models": [
        {
            "hydro_id": 1,
            "selection_mode": "stage_ranges",
            "stage_ranges": [
                { "start_stage_id": 0, "end_stage_id": null, "model": "constant_productivity", "productivity_mw_per_m3s": 0.9 }
            ]
        },
        {
            "hydro_id": 2,
            "selection_mode": "stage_ranges",
            "stage_ranges": [
                { "start_stage_id": 0, "end_stage_id": null, "model": "constant_productivity", "productivity_mw_per_m3s": 0.9 }
            ]
        }
    ]
}"#,
    );

    std::fs::create_dir_all(root.join("constraints")).unwrap();
    let schema = Arc::new(Schema::new(vec![
        Field::new("hydro_id", DataType::Int32, false),
        Field::new("hydro_unit_group_id", DataType::Int32, false),
        Field::new("stage_id", DataType::Int32, false),
        Field::new("max_turbined_m3s", DataType::Float64, true),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![1])),
            Arc::new(Int32Array::from(vec![0])),
            Arc::new(Int32Array::from(vec![0])),
            Arc::new(Float64Array::from(vec![Some(12345.0)])),
        ],
    )
    .unwrap();
    let file =
        std::fs::File::create(root.join("constraints/hydro_unit_group_bounds.parquet")).unwrap();
    let mut writer = ArrowWriter::try_new(file, batch.schema(), None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

/// A `resolve_hydro_unit_group_bounds` override follows its declared hydro's
/// group through the canonical resort exactly like the hydro-plant-axis case
/// above. `resolve_hydro_unit_group_bounds` is an eighth resolver sharing the
/// identical `entity_idx`-from-parse-order hazard as the seven family tests
/// above, but addresses the `hydro.unit_groups` axis rather than a
/// plant-level column, so none of those seven exercise it.
#[test]
fn hydro_unit_group_bound_override_follows_declared_id_through_canonical_resort() {
    let dir = TempDir::new().unwrap();
    make_hydro_unit_group_id_date_misaligned_case(&dir);

    let system = load_case(dir.path()).unwrap_or_else(|e| panic!("expected Ok(System): {e}"));

    let hydros = system.hydros();
    assert_eq!(hydros.len(), 2, "expected exactly 2 hydros");

    let pos_h1 = hydros
        .iter()
        .position(|h| h.id == EntityId::from(1))
        .expect("hydro 1 present in System::hydros()");
    let pos_h2 = hydros
        .iter()
        .position(|h| h.id == EntityId::from(2))
        .expect("hydro 2 present in System::hydros()");

    // Precondition: the misalignment opportunity must actually be present --
    // canonical (date, id) order must differ from parse (id) order.
    assert_eq!(
        pos_h2, 0,
        "hydro 2 (earlier operational_start_date) must sort first in canonical order"
    );
    assert_eq!(
        pos_h1, 1,
        "hydro 1 (later operational_start_date) must sort second in canonical \
         order, while its parse-order/id-order position was 0"
    );

    let group_pos = 0;
    let stage0_idx = 0;
    let block0_idx = 0;

    let group_overlay = system.bounds().group_overlay();
    let h1_resolved = group_overlay.override_at_block(pos_h1, group_pos, stage0_idx, block0_idx);
    let h2_resolved = group_overlay.override_at_block(pos_h2, group_pos, stage0_idx, block0_idx);

    let h1_max = h1_resolved
        .max_turbined_m3s
        .expect("hydro 1 (declared the override) should read back Some(max_turbined_m3s)");
    assert!(
        (h1_max - 12345.0).abs() < f64::EPSILON,
        "hydro 1 (declared the override, canonical position {pos_h1}) should \
         read back its own group override max_turbined_m3s=12345.0, got {h1_max}"
    );
    assert!(
        h2_resolved.max_turbined_m3s.is_none(),
        "hydro 2 (declared NO override, canonical position {pos_h2}) should \
         read back no group override, got {:?} (Some(12345.0) would mean \
         hydro 1's override leaked onto hydro 2's canonical slot)",
        h2_resolved.max_turbined_m3s
    );
}

// ─── The pipeline's presort key must stay identical to the builder's ──────
//
// The fix deliberately leaves TWO places computing canonical order:
// `SystemBuilder::build`'s `sort_canonical` and `pipeline::sort_into_canonical_order`.
// `System::hydros()`'s final order is always `sort_canonical`'s — it is the
// last sort applied — so it cannot witness which key the pipeline used. What
// CAN witness it is exactly the per-family technique above: a resolver's `entity_idx` is
// assigned from the pipeline's presort order, so an override's readback
// position reveals whether that presort agreed with `sort_canonical`. Three
// hydros, no two of which share a relative order under `(date, id)` and under
// either named wrong key (`(id, date)` or `id` alone — both coincide with
// parse order because id is already unique), make that witness unambiguous.

/// Three hydros whose relative order disagrees between `(date, id)` (the
/// correct canonical key: `H_B(2010) < H_C(2015) < H_A(2020)`) and `(id,
/// date)` / `id`-only (both coincide with id order: `H_A(1) < H_B(2) <
/// H_C(3)`, since id already uniquely orders them). A
/// `constraints/hydro_bounds.parquet` override on H_B alone
/// (`max_turbined_m3s=99999.0`) reads back correctly only when the pipeline's
/// presort key matches `(date, id)`.
fn make_three_hydro_key_duality_case(dir: &TempDir) {
    let root = dir.path();

    helpers::write_file(root, "config.json", helpers::VALID_CONFIG_JSON);
    helpers::write_file(root, "penalties.json", helpers::VALID_PENALTIES_JSON);
    helpers::write_file(
        root,
        "initial_conditions.json",
        helpers::VALID_INITIAL_CONDITIONS_JSON,
    );
    helpers::write_file(root, "stages.json", SINGLE_STAGE_STAGES_JSON);

    helpers::write_file(
        root,
        "system/buses.json",
        r#"{ "buses": [{ "id": 1, "name": "BUS_1", "operational_start_date": "2024-01-01" }] }"#,
    );
    helpers::write_file(root, "system/lines.json", r#"{ "lines": [] }"#);
    helpers::write_file(root, "system/thermals.json", r#"{ "thermals": [] }"#);

    helpers::write_file(
        root,
        "system/hydros.json",
        r#"{
    "hydros": [
        {
            "id": 1,
            "name": "HYDRO_A",
            "operational_start_date": "2020-06-15",
            "downstream_id": null,
            "reservoir": { "min_storage_hm3": 0.0, "max_storage_hm3": 100.0 },
            "outflow": { "min_outflow_m3s": 0.0, "max_outflow_m3s": null },
            "generation": {
                "model": "constant_productivity",
                "min_turbined_m3s": 0.0,
                "max_turbined_m3s": 111.0,
                "min_generation_mw": 0.0,
                "max_generation_mw": 111.0
            },
            "unit_groups": [
                {
                    "id": 0,
                    "name": "HYDRO_A",
                    "bus_id": 1,
                    "min_generation_mw": 0.0,
                    "max_generation_mw": 111.0,
                    "min_turbined_m3s": 0.0,
                    "max_turbined_m3s": 111.0
                }
            ]
        },
        {
            "id": 2,
            "name": "HYDRO_B",
            "operational_start_date": "2010-01-01",
            "downstream_id": null,
            "reservoir": { "min_storage_hm3": 0.0, "max_storage_hm3": 100.0 },
            "outflow": { "min_outflow_m3s": 0.0, "max_outflow_m3s": null },
            "generation": {
                "model": "constant_productivity",
                "min_turbined_m3s": 0.0,
                "max_turbined_m3s": 100000.0,
                "min_generation_mw": 0.0,
                "max_generation_mw": 100000.0
            },
            "unit_groups": [
                {
                    "id": 0,
                    "name": "HYDRO_B",
                    "bus_id": 1,
                    "min_generation_mw": 0.0,
                    "max_generation_mw": 100000.0,
                    "min_turbined_m3s": 0.0,
                    "max_turbined_m3s": 100000.0
                }
            ]
        },
        {
            "id": 3,
            "name": "HYDRO_C",
            "operational_start_date": "2015-03-01",
            "downstream_id": null,
            "reservoir": { "min_storage_hm3": 0.0, "max_storage_hm3": 100.0 },
            "outflow": { "min_outflow_m3s": 0.0, "max_outflow_m3s": null },
            "generation": {
                "model": "constant_productivity",
                "min_turbined_m3s": 0.0,
                "max_turbined_m3s": 555.0,
                "min_generation_mw": 0.0,
                "max_generation_mw": 555.0
            },
            "unit_groups": [
                {
                    "id": 0,
                    "name": "HYDRO_C",
                    "bus_id": 1,
                    "min_generation_mw": 0.0,
                    "max_generation_mw": 555.0,
                    "min_turbined_m3s": 0.0,
                    "max_turbined_m3s": 555.0
                }
            ]
        }
    ]
}"#,
    );

    helpers::write_file(
        root,
        "system/hydro_production_models.json",
        r#"{
    "production_models": [
        {
            "hydro_id": 1,
            "selection_mode": "stage_ranges",
            "stage_ranges": [
                { "start_stage_id": 0, "end_stage_id": null, "model": "constant_productivity", "productivity_mw_per_m3s": 0.9 }
            ]
        },
        {
            "hydro_id": 2,
            "selection_mode": "stage_ranges",
            "stage_ranges": [
                { "start_stage_id": 0, "end_stage_id": null, "model": "constant_productivity", "productivity_mw_per_m3s": 0.9 }
            ]
        },
        {
            "hydro_id": 3,
            "selection_mode": "stage_ranges",
            "stage_ranges": [
                { "start_stage_id": 0, "end_stage_id": null, "model": "constant_productivity", "productivity_mw_per_m3s": 0.9 }
            ]
        }
    ]
}"#,
    );

    std::fs::create_dir_all(root.join("constraints")).unwrap();
    let schema = Arc::new(Schema::new(vec![
        Field::new("hydro_id", DataType::Int32, false),
        Field::new("stage_id", DataType::Int32, false),
        Field::new("max_turbined_m3s", DataType::Float64, true),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![2])),
            Arc::new(Int32Array::from(vec![0])),
            Arc::new(Float64Array::from(vec![Some(99999.0)])),
        ],
    )
    .unwrap();
    let file = std::fs::File::create(root.join("constraints/hydro_bounds.parquet")).unwrap();
    let mut writer = ArrowWriter::try_new(file, batch.schema(), None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

/// Pins that the pipeline's pre-resolution sort key is the SAME `(date, id)`
/// key `SystemBuilder::build` uses — not merely that alignment happens to
/// hold on the 2-entity fixture above. A future edit to
/// `sort_canonical`'s key (or to `sort_into_canonical_order`'s) that silently
/// diverges the two would misroute H_B's override exactly as this test would
/// catch.
#[test]
fn hydro_bound_pipeline_presort_key_matches_builder_canonical_key() {
    let dir = TempDir::new().unwrap();
    make_three_hydro_key_duality_case(&dir);

    let system = load_case(dir.path()).unwrap_or_else(|e| panic!("expected Ok(System): {e}"));

    let hydros = system.hydros();
    assert_eq!(hydros.len(), 3, "expected exactly 3 hydros");

    let pos_a = hydros
        .iter()
        .position(|h| h.id == EntityId::from(1))
        .expect("HYDRO_A present in System::hydros()");
    let pos_b = hydros
        .iter()
        .position(|h| h.id == EntityId::from(2))
        .expect("HYDRO_B present in System::hydros()");
    let pos_c = hydros
        .iter()
        .position(|h| h.id == EntityId::from(3))
        .expect("HYDRO_C present in System::hydros()");

    // Precondition: canonical (date, id) order is [B, C, A] -- the OPPOSITE of
    // both id-order and (id, date)-order [A, B, C], so a wrong-key mutation
    // is observable, not accidentally masked.
    assert_eq!(pos_b, 0, "HYDRO_B (earliest date) must sort first");
    assert_eq!(pos_c, 1, "HYDRO_C (middle date) must sort second");
    assert_eq!(pos_a, 2, "HYDRO_A (latest date) must sort third");

    let bounds = system.bounds();
    let a = bounds.hydro_block_base(pos_a, 0);
    let b = bounds.hydro_block_base(pos_b, 0);
    let c = bounds.hydro_block_base(pos_c, 0);

    assert!(
        (b.max_turbined_m3s - 99999.0).abs() < f64::EPSILON,
        "HYDRO_B (declared the override, canonical position {pos_b}) should \
         read back its own override max_turbined_m3s=99999.0, got {}. A \
         pipeline presort keyed by (id, date) or by id alone would place \
         HYDRO_B at entity_idx=1 during resolution (parse/id order [A, B, C]) \
         while HYDRO_B's canonical position is 0, misrouting this override \
         onto HYDRO_C's slot instead.",
        b.max_turbined_m3s
    );
    assert!(
        (c.max_turbined_m3s - 555.0).abs() < f64::EPSILON,
        "HYDRO_C (declared NO override, canonical position {pos_c}) should \
         read back its base max_turbined_m3s=555.0, got {}",
        c.max_turbined_m3s
    );
    assert!(
        (a.max_turbined_m3s - 111.0).abs() < f64::EPSILON,
        "HYDRO_A (declared NO override, canonical position {pos_a}) should \
         read back its base max_turbined_m3s=111.0, got {}",
        a.max_turbined_m3s
    );
}

// ─── Stage-wide vs. per-block spillage override resolution ────────────────

/// A single hydro on a three-block stage with a `constraints/hydro_bounds.parquet`
/// stage-wide `max_spillage_m3s=12.0` override and a `block_id=1` override
/// tightening it to `4.0` — `hydro_bounds_at_block` must apply the per-block
/// override only to its own block, falling back to the stage-wide value on the
/// other two.
fn make_hydro_spillage_stage_and_block_override_case(dir: &TempDir) {
    let root = dir.path();

    helpers::write_file(root, "config.json", helpers::VALID_CONFIG_JSON);
    helpers::write_file(root, "penalties.json", helpers::VALID_PENALTIES_JSON);
    helpers::write_file(
        root,
        "initial_conditions.json",
        helpers::VALID_INITIAL_CONDITIONS_JSON,
    );

    helpers::write_file(
        root,
        "stages.json",
        r#"{
    "policy_graph": {
        "type": "finite_horizon",
        "annual_discount_rate": 0.06,
        "transitions": []
    },
    "stages": [
        {
            "id": 0,
            "start_date": "2024-01-01",
            "end_date": "2024-02-01",
            "blocks": [
                { "id": 0, "name": "B0", "hours": 248.0 },
                { "id": 1, "name": "B1", "hours": 248.0 },
                { "id": 2, "name": "B2", "hours": 248.0 }
            ],
            "num_openings": 1
        }
    ]
}"#,
    );

    helpers::write_file(
        root,
        "system/buses.json",
        r#"{ "buses": [{ "id": 1, "name": "BUS_1", "operational_start_date": "2024-01-01" }] }"#,
    );
    helpers::write_file(root, "system/lines.json", r#"{ "lines": [] }"#);
    helpers::write_file(root, "system/thermals.json", r#"{ "thermals": [] }"#);

    helpers::write_file(
        root,
        "system/hydros.json",
        r#"{
    "hydros": [
        {
            "id": 1,
            "name": "HYDRO_1",
            "operational_start_date": "2024-01-01",
            "downstream_id": null,
            "reservoir": { "min_storage_hm3": 0.0, "max_storage_hm3": 1000.0 },
            "outflow": { "min_outflow_m3s": 0.0, "max_outflow_m3s": null },
            "generation": {
                "model": "constant_productivity",
                "min_turbined_m3s": 0.0,
                "max_turbined_m3s": 200.0,
                "min_generation_mw": 0.0,
                "max_generation_mw": 200.0
            },
            "unit_groups": [
                {
                    "id": 0,
                    "name": "HYDRO_1",
                    "bus_id": 1,
                    "min_generation_mw": 0.0,
                    "max_generation_mw": 200.0,
                    "min_turbined_m3s": 0.0,
                    "max_turbined_m3s": 200.0
                }
            ]
        }
    ]
}"#,
    );

    helpers::write_file(
        root,
        "system/hydro_production_models.json",
        r#"{
    "production_models": [
        {
            "hydro_id": 1,
            "selection_mode": "stage_ranges",
            "stage_ranges": [
                { "start_stage_id": 0, "end_stage_id": null, "model": "constant_productivity", "productivity_mw_per_m3s": 0.9 }
            ]
        }
    ]
}"#,
    );

    std::fs::create_dir_all(root.join("constraints")).unwrap();
    let schema = Arc::new(Schema::new(vec![
        Field::new("hydro_id", DataType::Int32, false),
        Field::new("stage_id", DataType::Int32, false),
        Field::new("max_spillage_m3s", DataType::Float64, true),
        Field::new("block_id", DataType::Int32, true),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![1, 1])),
            Arc::new(Int32Array::from(vec![0, 0])),
            Arc::new(Float64Array::from(vec![Some(12.0), Some(4.0)])),
            Arc::new(Int32Array::from(vec![None, Some(1)])),
        ],
    )
    .unwrap();
    let file = std::fs::File::create(root.join("constraints/hydro_bounds.parquet")).unwrap();
    let mut writer = ArrowWriter::try_new(file, batch.schema(), None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

/// A stage-wide `max_spillage_m3s` override and a `block_id=1` override on
/// the same stage resolve per block — block 1 reads back its own tighter value,
/// blocks 0 and 2 fall through to the stage-wide value.
#[test]
fn hydro_spillage_override_resolves_stage_wide_and_per_block() {
    let dir = TempDir::new().unwrap();
    make_hydro_spillage_stage_and_block_override_case(&dir);

    let system = load_case(dir.path()).unwrap_or_else(|e| panic!("expected Ok(System): {e}"));

    let hydros = system.hydros();
    assert_eq!(hydros.len(), 1, "expected exactly 1 hydro");
    let pos = hydros
        .iter()
        .position(|h| h.id == EntityId::from(1))
        .expect("hydro 1 present in System::hydros()");

    let bounds = system.bounds();
    let stage_idx = 0;

    let block0 = bounds.hydro_bounds_at_block(pos, stage_idx, 0);
    let block1 = bounds.hydro_bounds_at_block(pos, stage_idx, 1);
    let block2 = bounds.hydro_bounds_at_block(pos, stage_idx, 2);

    assert!(
        matches!(block0.max_spillage_m3s, Some(v) if (v - 12.0).abs() < f64::EPSILON),
        "block 0 has no per-block override, so it reads back the stage-wide \
         max_spillage_m3s=12.0, got {:?}",
        block0.max_spillage_m3s
    );
    assert!(
        matches!(block1.max_spillage_m3s, Some(v) if (v - 4.0).abs() < f64::EPSILON),
        "block 1's own override (4.0) takes precedence over the stage-wide \
         value (12.0), got {:?}",
        block1.max_spillage_m3s
    );
    assert!(
        matches!(block2.max_spillage_m3s, Some(v) if (v - 12.0).abs() < f64::EPSILON),
        "block 2 has no per-block override, so it reads back the stage-wide \
         max_spillage_m3s=12.0, got {:?}",
        block2.max_spillage_m3s
    );
}

// ─── The sort runs after validation, not before ───────────────────────────

/// Two hydros, dated the REVERSE of their id order (hydro 1 dated 2030, hydro
/// 2 dated 2020) exactly like the fixture above, but both referencing a
/// non-existent Bus 999 instead of carrying a `hydro_bounds` override. Pins
/// validation error ORDER rather than resolved-value alignment.
fn make_hydro_referential_error_case(dir: &TempDir) {
    let root = dir.path();

    helpers::write_file(root, "config.json", helpers::VALID_CONFIG_JSON);
    helpers::write_file(root, "penalties.json", helpers::VALID_PENALTIES_JSON);
    helpers::write_file(
        root,
        "initial_conditions.json",
        helpers::VALID_INITIAL_CONDITIONS_JSON,
    );
    helpers::write_file(root, "stages.json", SINGLE_STAGE_STAGES_JSON);

    helpers::write_file(
        root,
        "system/buses.json",
        r#"{ "buses": [{ "id": 1, "name": "BUS_1", "operational_start_date": "2024-01-01" }] }"#,
    );
    helpers::write_file(root, "system/lines.json", r#"{ "lines": [] }"#);
    helpers::write_file(root, "system/thermals.json", r#"{ "thermals": [] }"#);

    helpers::write_file(
        root,
        "system/hydros.json",
        r#"{
    "hydros": [
        {
            "id": 1,
            "name": "HYDRO_1_LATE",
            "operational_start_date": "2030-01-01",
            "downstream_id": null,
            "reservoir": { "min_storage_hm3": 0.0, "max_storage_hm3": 1000.0 },
            "outflow": { "min_outflow_m3s": 0.0, "max_outflow_m3s": null },
            "generation": {
                "model": "constant_productivity",
                "min_turbined_m3s": 0.0,
                "max_turbined_m3s": 20000.0,
                "min_generation_mw": 0.0,
                "max_generation_mw": 20000.0
            },
            "unit_groups": [
                {
                    "id": 0,
                    "name": "HYDRO_1_LATE",
                    "bus_id": 999,
                    "min_generation_mw": 0.0,
                    "max_generation_mw": 20000.0,
                    "min_turbined_m3s": 0.0,
                    "max_turbined_m3s": 20000.0
                }
            ]
        },
        {
            "id": 2,
            "name": "HYDRO_2_EARLY",
            "operational_start_date": "2020-01-01",
            "downstream_id": null,
            "reservoir": { "min_storage_hm3": 0.0, "max_storage_hm3": 500.0 },
            "outflow": { "min_outflow_m3s": 0.0, "max_outflow_m3s": null },
            "generation": {
                "model": "constant_productivity",
                "min_turbined_m3s": 0.0,
                "max_turbined_m3s": 300.0,
                "min_generation_mw": 0.0,
                "max_generation_mw": 300.0
            },
            "unit_groups": [
                {
                    "id": 0,
                    "name": "HYDRO_2_EARLY",
                    "bus_id": 999,
                    "min_generation_mw": 0.0,
                    "max_generation_mw": 300.0,
                    "min_turbined_m3s": 0.0,
                    "max_turbined_m3s": 300.0
                }
            ]
        }
    ]
}"#,
    );

    helpers::write_file(
        root,
        "system/hydro_production_models.json",
        r#"{
    "production_models": [
        {
            "hydro_id": 1,
            "selection_mode": "stage_ranges",
            "stage_ranges": [
                { "start_stage_id": 0, "end_stage_id": null, "model": "constant_productivity", "productivity_mw_per_m3s": 0.9 }
            ]
        },
        {
            "hydro_id": 2,
            "selection_mode": "stage_ranges",
            "stage_ranges": [
                { "start_stage_id": 0, "end_stage_id": null, "model": "constant_productivity", "productivity_mw_per_m3s": 0.9 }
            ]
        }
    ]
}"#,
    );
}

/// The canonical resort must run AFTER validation, so validation's error
/// order stays on parse (id) order. `check_hydro_references` iterates
/// `data.hydros` and pushes one error per hydro in that iteration order; with
/// both hydros invalid, "Hydro 1" must precede "Hydro 2" in the combined
/// description regardless of when the canonical resort runs. Sorting before
/// validation would flip this to canonical (date) order, where hydro 2
/// (dated 2020) sorts before hydro 1 (dated 2030).
#[test]
fn referential_error_order_survives_the_post_validation_sort() {
    let dir = TempDir::new().unwrap();
    make_hydro_referential_error_case(&dir);

    let err = load_case(dir.path()).expect_err("both hydros reference a non-existent bus");
    let LoadError::ConstraintError { description } = err else {
        panic!("expected LoadError::ConstraintError, got {err:?}");
    };

    let pos_h1 = description
        .find("Hydro 1 unit group 0 references non-existent Bus 999")
        .expect("description must mention Hydro 1's bad bus reference");
    let pos_h2 = description
        .find("Hydro 2 unit group 0 references non-existent Bus 999")
        .expect("description must mention Hydro 2's bad bus reference");

    assert!(
        pos_h1 < pos_h2,
        "Hydro 1's error must precede Hydro 2's, matching validation's parse/id \
         iteration order -- not canonical (date) order, where hydro 2 (dated \
         2020) would sort before hydro 1 (dated 2030); got description: {description}"
    );
}
