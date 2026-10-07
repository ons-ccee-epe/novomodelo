//! Fixture builders that construct case directories in a [`TempDir`].
#![allow(dead_code, unused_imports)]
// Rationale: each of the six binaries below calls only a subset of the two
// case builders and the re-exported corpus names; both lints fire per test
// binary, not crate-wide.

use tempfile::TempDir;

pub use cobre_io::test_support::{
    VALID_BUSES_JSON, VALID_CONFIG_JSON, VALID_HYDROS_JSON, VALID_INITIAL_CONDITIONS_JSON,
    VALID_LINES_JSON, VALID_PENALTIES_JSON, VALID_STAGES_JSON, VALID_THERMALS_JSON,
    make_minimal_case, write_file,
};

// ── make_multi_entity_case ────────────────────────────────────────────────────

/// Populate `dir` with a richer 8-file case: 2 buses, 1 hydro (bus 1), 1 thermal
/// (bus 2), 1 line (bus 1→2), 2 study stages with a transition.
pub fn make_multi_entity_case(dir: &TempDir) {
    let root = dir.path();

    write_file(root, "config.json", VALID_CONFIG_JSON);
    write_file(root, "penalties.json", VALID_PENALTIES_JSON);

    write_file(
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
            "blocks": [{ "id": 0, "name": "FLAT", "hours": 744.0 }],
            "num_openings": 10
        },
        {
            "id": 1,
            "start_date": "2024-02-01",
            "end_date": "2024-03-01",
            "blocks": [{ "id": 0, "name": "FLAT", "hours": 672.0 }],
            "num_openings": 10
        }
    ]
}"#,
    );

    write_file(
        root,
        "initial_conditions.json",
        VALID_INITIAL_CONDITIONS_JSON,
    );

    write_file(
        root,
        "system/buses.json",
        r#"{
    "buses": [
        { "id": 1, "name": "BUS_SE", "operational_start_date": "2024-01-01" },
        { "id": 2, "name": "BUS_S", "operational_start_date": "2024-01-01" }
    ]
}"#,
    );

    write_file(
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

    write_file(
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

    write_file(
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

    write_file(
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

// ── make_referential_violation_case ───────────────────────────────────────────

/// `make_multi_entity_case` with the hydro's unit group `bus_id` set to a
/// non-existent 999, so `load_case` must reject it for referential integrity.
pub fn make_referential_violation_case(dir: &TempDir) {
    make_multi_entity_case(dir);

    write_file(
        dir.path(),
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
                    "bus_id": 999,
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
}
