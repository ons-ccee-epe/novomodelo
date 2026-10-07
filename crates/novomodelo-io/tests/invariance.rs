//! Declaration-order invariance tests for `cobre_io::load_case`.
//!
//! `load_case` must produce bit-for-bit identical [`System`] values regardless
//! of entity declaration order: collections are stored in canonical
//! (operational_start_date, name) order, so a reversed input array must yield an
//! equal result.
#![allow(
    clippy::unwrap_used,
    clippy::panic,
    clippy::too_many_lines,
    clippy::doc_markdown
)]

mod helpers;

use cobre_core::EntityId;
use cobre_io::load_case;
use tempfile::TempDir;

// ── make_shuffled_multi_entity_case ───────────────────────────────────────────

/// Same logical data as [`helpers::make_multi_entity_case`], with entity arrays
/// reversed. Only buses and stages have >1 entity, so only those are
/// meaningfully reordered; lines, hydros, and thermals are single-entity no-ops.
fn make_shuffled_multi_entity_case(dir: &TempDir) {
    let root = dir.path();

    helpers::write_file(root, "config.json", helpers::VALID_CONFIG_JSON);
    helpers::write_file(root, "penalties.json", helpers::VALID_PENALTIES_JSON);

    // Stages deliberately reversed (id=1 before id=0) — do not "fix" to canonical order.
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
            "id": 1,
            "start_date": "2024-02-01",
            "end_date": "2024-03-01",
            "blocks": [{ "id": 0, "name": "FLAT", "hours": 672.0 }],
            "num_openings": 10
        },
        {
            "id": 0,
            "start_date": "2024-01-01",
            "end_date": "2024-02-01",
            "blocks": [{ "id": 0, "name": "FLAT", "hours": 744.0 }],
            "num_openings": 10
        }
    ]
}"#,
    );

    helpers::write_file(
        root,
        "initial_conditions.json",
        helpers::VALID_INITIAL_CONDITIONS_JSON,
    );

    // Buses deliberately reversed (id=2 first) — do not "fix" to canonical order.
    helpers::write_file(
        root,
        "system/buses.json",
        r#"{
    "buses": [
        { "id": 2, "name": "BUS_S", "operational_start_date": "2024-01-01" },
        { "id": 1, "name": "BUS_SE", "operational_start_date": "2024-01-01" }
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
            "name": "SE-S",
            "operational_start_date": "2024-01-01",
            "source_bus_id": 1,
            "target_bus_id": 2,
            "capacity": { "direct_mw": 2000.0, "reverse_mw": 1500.0 }
        }
    ]
}"#,
    );

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
        "system/thermals.json",
        r#"{
    "thermals": [
        {
            "id": 1,
            "name": "THERMAL_1",
            "operational_start_date": "2024-01-01",
            "bus_id": 2,
            "cost_per_mwh": 80.0,
            "generation": { "min_mw": 0.0, "max_mw": 300.0 }
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
        }
    ]
}"#,
    );
}

// ── test_bus_ordering_invariance ──────────────────────────────────────────────

#[test]
fn test_bus_ordering_invariance() {
    let canonical_dir = TempDir::new().unwrap();
    helpers::make_multi_entity_case(&canonical_dir);

    let shuffled_dir = TempDir::new().unwrap();
    make_shuffled_multi_entity_case(&shuffled_dir);

    let canonical = load_case(canonical_dir.path())
        .unwrap_or_else(|e| panic!("canonical case should load successfully, got: {e}"));

    let shuffled = load_case(shuffled_dir.path())
        .unwrap_or_else(|e| panic!("shuffled case should load successfully, got: {e}"));

    assert_eq!(
        canonical.n_buses(),
        shuffled.n_buses(),
        "bus count must match between canonical and shuffled cases"
    );

    let canonical_bus1 = canonical
        .bus(EntityId(1))
        .unwrap_or_else(|| panic!("bus id=1 must exist in canonical case"));
    let shuffled_bus1 = shuffled
        .bus(EntityId(1))
        .unwrap_or_else(|| panic!("bus id=1 must exist in shuffled case"));

    assert_eq!(
        canonical_bus1.name, shuffled_bus1.name,
        "bus id=1 name must be identical in canonical and shuffled cases"
    );

    let canonical_bus2 = canonical
        .bus(EntityId(2))
        .unwrap_or_else(|| panic!("bus id=2 must exist in canonical case"));
    let shuffled_bus2 = shuffled
        .bus(EntityId(2))
        .unwrap_or_else(|| panic!("bus id=2 must exist in shuffled case"));

    assert_eq!(
        canonical_bus2.name, shuffled_bus2.name,
        "bus id=2 name must be identical in canonical and shuffled cases"
    );

    // `==` skips the HashMap indices and compares the sorted entity vecs.
    assert_eq!(
        canonical, shuffled,
        "Systems from canonical and shuffled bus orderings must be equal"
    );
}

// ── test_stage_ordering_invariance ────────────────────────────────────────────

#[test]
fn test_stage_ordering_invariance() {
    let canonical_dir = TempDir::new().unwrap();
    helpers::make_multi_entity_case(&canonical_dir);

    let shuffled_dir = TempDir::new().unwrap();
    make_shuffled_multi_entity_case(&shuffled_dir);

    let canonical = load_case(canonical_dir.path())
        .unwrap_or_else(|e| panic!("canonical case should load successfully, got: {e}"));

    let shuffled = load_case(shuffled_dir.path())
        .unwrap_or_else(|e| panic!("shuffled case should load successfully, got: {e}"));

    assert_eq!(
        canonical.n_stages(),
        shuffled.n_stages(),
        "stage count must match between canonical and shuffled cases"
    );

    let canonical_first = canonical
        .stages()
        .first()
        .unwrap_or_else(|| panic!("canonical case must have at least one stage"));
    let shuffled_first = shuffled
        .stages()
        .first()
        .unwrap_or_else(|| panic!("shuffled case must have at least one stage"));

    assert_eq!(
        canonical_first.id, 0,
        "canonical case: stages must be sorted so stages[0].id == 0"
    );
    assert_eq!(
        shuffled_first.id, 0,
        "shuffled case: stages must be sorted so stages[0].id == 0 even when declared in reversed order"
    );

    assert_eq!(
        canonical_first.id, shuffled_first.id,
        "first stage id must be the same in both canonical and shuffled cases"
    );
}

// ── test_full_case_ordering_invariance ────────────────────────────────────────

#[test]
fn test_full_case_ordering_invariance() {
    let canonical_dir = TempDir::new().unwrap();
    helpers::make_multi_entity_case(&canonical_dir);

    let shuffled_dir = TempDir::new().unwrap();
    make_shuffled_multi_entity_case(&shuffled_dir);

    let canonical = load_case(canonical_dir.path())
        .unwrap_or_else(|e| panic!("canonical case should load successfully, got: {e}"));

    let shuffled = load_case(shuffled_dir.path())
        .unwrap_or_else(|e| panic!("shuffled case should load successfully, got: {e}"));

    assert_eq!(
        canonical, shuffled,
        "Systems built from canonical and fully-shuffled input must be structurally equal"
    );

    let canonical_bus1 = canonical
        .bus(EntityId(1))
        .unwrap_or_else(|| panic!("bus id=1 must exist in canonical case"));
    let shuffled_bus1 = shuffled
        .bus(EntityId(1))
        .unwrap_or_else(|| panic!("bus id=1 must exist in shuffled case"));

    assert_eq!(
        canonical_bus1.name, shuffled_bus1.name,
        "bus(EntityId(1)).name must be identical regardless of declaration order"
    );

    assert_eq!(
        canonical.n_hydros(),
        shuffled.n_hydros(),
        "hydro count must match between canonical and shuffled cases"
    );
    assert_eq!(
        canonical.n_stages(),
        shuffled.n_stages(),
        "stage count must match between canonical and shuffled cases"
    );
}
