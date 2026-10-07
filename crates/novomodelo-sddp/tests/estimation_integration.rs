//! Integration tests for the estimation pipeline.
//!
//! Exercises [`cobre_sddp::estimate_from_history`] end-to-end with a real
//! temporary case directory, a synthetic `inflow_history.parquet`, and minimal
//! supporting files. Stages span the full history period so every observation
//! date falls within a stage's `[start_date, end_date)` range, which
//! seasonal-statistics estimation requires.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::too_many_lines
)]
// `..Default::default()` in the make_* Spec calls is the intentional future-field
// seam from `common::builders` — a no-op today, not dead code.
#![allow(clippy::needless_update)]

use std::path::Path;
use std::sync::Arc;

use arrow::array::{Date32Array, Float64Array, Int32Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use chrono::NaiveDate;
use cobre_core::{
    DeficitSegment, EntityId, SystemBuilder,
    entities::hydro::{HydroGenerationModel, HydroPenalties},
    temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
        StageStateConfig,
    },
};
use cobre_io::Config;
use cobre_sddp::{EstimationPath, estimate_from_history};
use parquet::arrow::ArrowWriter;
use tempfile::TempDir;

mod common;
use common::builders::{BusSpec, HydroSpec, StageSpec, make_bus, make_hydro, make_stage};

// ── Constants ─────────────────────────────────────────────────────────────────

const N_SEASONS: usize = 12;

/// History years — enough for stable PAR(2) estimation.
const N_YEARS: usize = 15;

const START_YEAR: i32 = 2000;

const HYDRO_ID: i32 = 1;

const BUS_ID: i32 = 10;

// ── Parquet helpers ───────────────────────────────────────────────────────────

fn inflow_history_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("hydro_id", DataType::Int32, false),
        Field::new("start_date", DataType::Date32, false),
        Field::new("end_date", DataType::Date32, false),
        Field::new("value_m3s", DataType::Float64, false),
    ]))
}

/// Convert a `NaiveDate` to a Date32 integer (days since the Unix epoch).
fn date_to_date32(date: NaiveDate) -> i32 {
    let epoch = NaiveDate::from_ymd_opt(1970, 1, 1).unwrap();
    i32::try_from((date - epoch).num_days()).expect("date in Date32 range")
}

/// Write a Parquet file of synthetic inflow history: `N_YEARS * N_SEASONS`
/// monthly observations for one hydro, each a full-coverage `[1st,
/// 1st-of-next-month)` window so it counts as a complete occurrence under any
/// real `SeasonMap` (the exact-coverage gate in `cobre-io`'s estimation
/// reader) as well as falling inside the stage's own window. Values use a
/// seasonal sine-wave pattern to give non-trivial autocorrelation structure
/// for PAR(p) fitting.
fn write_inflow_history(path: &Path) {
    write_inflow_history_n_years(path, N_YEARS);
}

/// [`write_inflow_history`], parameterized over the number of years so a caller
/// can grow the per-season observation count past a fixed minimum-sample
/// threshold in the correlation-estimation path.
fn write_inflow_history_n_years(path: &Path, n_years: usize) {
    let schema = inflow_history_schema();

    let mut hydro_ids: Vec<i32> = Vec::new();
    let mut start_dates: Vec<i32> = Vec::new();
    let mut end_dates: Vec<i32> = Vec::new();
    let mut values: Vec<f64> = Vec::new();

    for year in 0..n_years {
        for month in 0..N_SEASONS {
            let cal_year = START_YEAR + year as i32;
            let cal_month = (month as u32) + 1;
            let month_start = NaiveDate::from_ymd_opt(cal_year, cal_month, 1).unwrap();
            let month_end = if cal_month == 12 {
                NaiveDate::from_ymd_opt(cal_year + 1, 1, 1).unwrap()
            } else {
                NaiveDate::from_ymd_opt(cal_year, cal_month + 1, 1).unwrap()
            };

            // Deterministic perturbation so values differ across years.
            let phase = std::f64::consts::TAU * (month as f64) / (N_SEASONS as f64);
            let noise_seed = (year * N_SEASONS + month) as f64;
            let noise = (noise_seed * std::f64::consts::PI).sin() * 30.0;
            let value = 500.0 + 200.0 * phase.sin() + noise;

            hydro_ids.push(HYDRO_ID);
            start_dates.push(date_to_date32(month_start));
            end_dates.push(date_to_date32(month_end));
            values.push(value.max(1.0));
        }
    }

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(hydro_ids)),
            Arc::new(Date32Array::from(start_dates)),
            Arc::new(Date32Array::from(end_dates)),
            Arc::new(Float64Array::from(values)),
        ],
    )
    .expect("valid batch");

    let file = std::fs::File::create(path).expect("create parquet file");
    let mut writer = ArrowWriter::try_new(file, schema, None).expect("ArrowWriter");
    writer.write(&batch).expect("write batch");
    writer.close().expect("close writer");
}

// ── Case directory helpers ────────────────────────────────────────────────────

fn config_json(order_selection: &str, max_order: u32) -> String {
    format!(
        r#"{{
            "training": {{ "tree_seed": 42 }},
            "simulation": {{ "enabled": false, "io_channel_capacity": 16 }},
            "modeling": {{}},
            "policy": {{}},
            "exports": {{}},
            "estimation": {{
                "order_selection": "{order_selection}",
                "max_order": {max_order},
                "min_observations_per_season": 2
            }}
        }}"#
    )
}

/// Create the minimal directory skeleton required by `validate_structure`. The
/// stub files have content `{}` — only their existence matters for the manifest;
/// only `config.json` carries real estimation settings.
fn create_minimal_case_skeleton(case_dir: &Path, order_selection: &str, max_order: u32) {
    std::fs::create_dir_all(case_dir.join("system")).unwrap();
    std::fs::create_dir_all(case_dir.join("scenarios")).unwrap();

    std::fs::write(
        case_dir.join("config.json"),
        config_json(order_selection, max_order),
    )
    .unwrap();

    for name in &[
        "penalties.json",
        "stages.json",
        "initial_conditions.json",
        "system/buses.json",
        "system/lines.json",
        "system/hydros.json",
        "system/thermals.json",
    ] {
        std::fs::write(case_dir.join(name), b"{}").unwrap();
    }
}

// ── System builder helpers ────────────────────────────────────────────────────

/// Build a `System` with one hydro and `N_YEARS × N_SEASONS` monthly study stages
/// spanning the full history period. Each stage carries `season_id = month_index`
/// (0 = January, …, 11 = December) so every observation in `inflow_history.parquet`
/// falls within a stage's `[start_date, end_date)` window — mirroring what
/// `load_case` produces from per-stage `season_id` fields.
fn build_system_with_one_hydro() -> cobre_core::System {
    let bus = make_bus(
        EntityId::from(BUS_ID),
        BusSpec {
            name: "B1".to_string(),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            deficit_segments: vec![DeficitSegment {
                depth_mw: Some(f64::INFINITY),
                cost_per_mwh: 3000.0,
            }],
            excess_cost: 0.0,
            ..Default::default()
        },
    );

    let hydro = make_hydro(
        EntityId::from(HYDRO_ID),
        HydroSpec {
            name: "H1".to_string(),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            bus_id: EntityId::from(BUS_ID),
            downstream_id: None,
            entry_stage_id: None,
            exit_stage_id: None,
            min_storage_hm3: 0.0,
            max_storage_hm3: 5000.0,
            min_outflow_m3s: 0.0,
            max_outflow_m3s: None,
            generation_model: HydroGenerationModel::ConstantProductivity,
            min_turbined_m3s: 0.0,
            max_turbined_m3s: 1000.0,
            specific_productivity_mw_per_m3s_per_m: None,
            min_generation_mw: 0.0,
            max_generation_mw: 900.0,
            tailrace: None,
            hydraulic_losses: None,
            efficiency: None,
            evaporation_coefficients_mm: None,
            evaporation_reference_volumes_hm3: None,
            diversion: None,
            filling: None,
            penalties: HydroPenalties {
                spillage_cost: 0.0,
                diversion_cost: 0.0,
                turbined_cost: 0.0,
                storage_violation_below_cost: 1000.0,
                filling_target_violation_cost: 0.0,
                turbined_violation_below_cost: 0.0,
                outflow_violation_below_cost: 0.0,
                outflow_violation_above_cost: 0.0,
                generation_violation_below_cost: 0.0,
                evaporation_violation_cost: 0.0,
                water_withdrawal_violation_cost: 0.0,
                water_withdrawal_violation_pos_cost: 0.0,
                water_withdrawal_violation_neg_cost: 0.0,
                evaporation_violation_pos_cost: 0.0,
                evaporation_violation_neg_cost: 0.0,
                inflow_nonnegativity_cost: 1000.0,
            },
            ..Default::default()
        },
    );

    let mut stages: Vec<Stage> = Vec::with_capacity(N_YEARS * N_SEASONS);
    let mut stage_index: usize = 0;

    for year in 0..N_YEARS {
        let cal_year = START_YEAR + year as i32;
        for month in 0..N_SEASONS {
            let cal_month = (month as u32) + 1;

            let start_date = NaiveDate::from_ymd_opt(cal_year, cal_month, 1).unwrap();
            let (end_year, end_month) = if cal_month == 12 {
                (cal_year + 1, 1u32)
            } else {
                (cal_year, cal_month + 1)
            };
            let end_date = NaiveDate::from_ymd_opt(end_year, end_month, 1).unwrap();

            stages.push(make_stage(
                stage_index,
                StageSpec {
                    start_date,
                    end_date,
                    season_id: Some(month),
                    blocks: vec![Block {
                        index: 0,
                        name: format!("M{month:02}"),
                        duration_hours: 720.0,
                    }],
                    block_mode: BlockMode::Parallel,
                    state_config: StageStateConfig {
                        storage: true,
                        inflow_lags: false,
                    },
                    risk_config: StageRiskConfig::Expectation,
                    scenario_config: ScenarioSourceConfig {
                        branching_factor: 1,
                        noise_method: NoiseMethod::Saa,
                    },
                    ..Default::default()
                },
            ));
            stage_index += 1;
        }
    }

    SystemBuilder::new()
        .buses(vec![bus])
        .hydros(vec![hydro])
        .stages(stages)
        .build()
        .expect("valid system")
}

fn parse_config(case_dir: &Path) -> Config {
    let content = std::fs::read_to_string(case_dir.join("config.json")).unwrap();
    serde_json::from_str(&content).expect("valid config JSON")
}

// ── Tests ─────────────────────────────────────────────────────────────────────

/// PACF estimation with `max_order = 2` produces one `InflowModel` per
/// (hydro, stage) pair with `ar_order() <= 2` and finite, positive `mean_m3s` /
/// `std_m3s`. With no `inflow_seasonal_stats.parquet` /
/// `inflow_ar_coefficients.parquet` present, the from-history estimation path runs.
#[test]
fn test_estimate_from_history_fixed_order() {
    let dir = TempDir::new().unwrap();
    let case_dir = dir.path();

    create_minimal_case_skeleton(case_dir, "pacf", 2);
    write_inflow_history(&case_dir.join("scenarios/inflow_history.parquet"));

    let system = build_system_with_one_hydro();
    let config = parse_config(case_dir);

    let (updated, _report, _path) = estimate_from_history(system, case_dir, &config)
        .expect("estimation should succeed with 15 years of monthly data");

    // Per-season estimates are expanded to every stage sharing the same season_id,
    // so 1 hydro × 180 stages = 180 models.
    let models = updated.inflow_models();
    assert_eq!(
        models.len(),
        180,
        "expected 180 inflow models (1 hydro × 180 stages), got {}",
        models.len()
    );

    for m in models {
        assert_eq!(m.hydro_id, EntityId::from(HYDRO_ID));

        assert!(
            m.mean_m3s.is_finite() && m.mean_m3s > 0.0,
            "mean_m3s should be positive and finite, got {} for stage {}",
            m.mean_m3s,
            m.stage_id
        );
        assert!(
            m.std_m3s.is_finite() && m.std_m3s >= 0.0,
            "std_m3s should be non-negative and finite, got {} for stage {}",
            m.std_m3s,
            m.stage_id
        );

        // Contribution validation may reduce a model below order 2 if it detects
        // explosive behaviour, so the bound is `<=`, not `==`.
        assert!(
            m.ar_order() <= 2,
            "fixed-order=2 should produce ar_order()<=2, got {} for stage {}",
            m.ar_order(),
            m.stage_id
        );
    }
}

/// AIC order selection produces `InflowModel`s whose `ar_order()` is in
/// `[0, max_order]` with finite, positive `mean_m3s` / `std_m3s`.
///
/// The test does NOT assert AIC always selects below `max_order` — whether it does
/// depends on the synthetic data's autocorrelation structure; the invariant is only
/// `ar_order() <= max_order`.
#[test]
fn test_estimate_from_history_aic_order() {
    const MAX_ORDER: u32 = 3;

    let dir = TempDir::new().unwrap();
    let case_dir = dir.path();

    create_minimal_case_skeleton(case_dir, "pacf", MAX_ORDER);
    write_inflow_history(&case_dir.join("scenarios/inflow_history.parquet"));

    let system = build_system_with_one_hydro();
    let config = parse_config(case_dir);

    let (updated, _report, _path) = estimate_from_history(system, case_dir, &config)
        .expect("AIC estimation should succeed with 15 years of monthly data");

    let models = updated.inflow_models();
    assert_eq!(
        models.len(),
        180,
        "expected 180 inflow models (1 hydro × 180 stages), got {}",
        models.len()
    );

    for m in models {
        assert_eq!(m.hydro_id, EntityId::from(HYDRO_ID));

        assert!(
            m.mean_m3s.is_finite() && m.mean_m3s > 0.0,
            "mean_m3s should be positive and finite, got {} for stage {}",
            m.mean_m3s,
            m.stage_id
        );
        assert!(
            m.std_m3s.is_finite() && m.std_m3s >= 0.0,
            "std_m3s should be non-negative and finite, got {} for stage {}",
            m.std_m3s,
            m.stage_id
        );

        assert!(
            m.ar_order() <= MAX_ORDER as usize,
            "AIC order {} exceeds max_order {} for stage {}",
            m.ar_order(),
            MAX_ORDER,
            m.stage_id
        );
    }
}

// ── PAR(1) round-trip tests ───────────────────────────────────────────────────

const N_OBS_PER_SEASON: usize = 200;

const N_SEASONS_RT: usize = 2;

/// True AR(1) coefficient for the round-trip tests.
const TRUE_PHI: f64 = 0.6;

/// True mean for the round-trip tests (m³/s).
const TRUE_MEAN: f64 = 300.0;

/// True standard deviation for the round-trip tests (m³/s).
const TRUE_STD: f64 = 60.0;

/// AR(1) coefficient SE ≈ 1/sqrt(N) ≈ 0.07; two standard errors ≈ 0.14 < 0.15.
const PHI_TOLERANCE: f64 = 0.15;

/// Relative tolerance for mean and std estimates (10% of true value).
const STAT_RELATIVE_TOLERANCE: f64 = 0.10;

/// Generate an interleaved PAR(1) time series for two seasons, identical mean,
/// std, and phi for both.
///
/// The generative model is the cross-season form the estimation path fits,
/// where `x_{t-1}` is the IMMEDIATELY preceding
/// observation (from the previous season):
///
/// `(x_t,s - mu) / sigma = phi * (x_{t-1,s-1} - mu) / sigma + sqrt(1 - phi^2) * eps_t`
///
/// Returns `2 * n_per_season` values alternating between the two seasons
/// (`[s0_0, s1_0, s0_1, s1_1, ...]`) — the chronological ordering the lag
/// lookup requires.
fn generate_par1_interleaved(
    mean: f64,
    std: f64,
    phi: f64,
    n_per_season: usize,
    seed: u64,
) -> Vec<f64> {
    use rand::{SeedableRng, rngs::StdRng};
    use rand_distr::{Distribution, Normal};

    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0_f64, 1.0_f64).expect("valid normal distribution");

    let residual_std = std * (1.0 - phi * phi).sqrt();
    let total = 2 * n_per_season;
    let mut values = Vec::with_capacity(total);

    let mut prev = mean;

    for _ in 0..total {
        let eps = normal.sample(&mut rng);
        let x = mean + phi * (prev - mean) + residual_std * eps;
        let x_clamped = x.max(1.0);
        values.push(x_clamped);
        prev = x_clamped;
    }

    values
}

/// Write `inflow_history.parquet` for `n_hydros` hydros with 2 seasons (January
/// season 0, February season 1, dated the 15th to fall within their stage windows).
///
/// The Jan2000, Feb2000, Jan2001, ... chronological ordering ensures the lag-1
/// lookup (which looks `lag` positions back in the sorted-by-date observations)
/// finds the cross-season predecessor PAR(1) needs.
fn write_par1_inflow_history(path: &Path, n_hydros: usize) {
    let schema = inflow_history_schema();

    let mut hydro_ids: Vec<i32> = Vec::new();
    let mut start_dates: Vec<i32> = Vec::new();
    let mut end_dates: Vec<i32> = Vec::new();
    let mut values: Vec<f64> = Vec::new();

    for h in 0..n_hydros {
        let hydro_id = (h + 1) as i32;

        // A different seed per hydro makes the hydros statistically independent.
        let series =
            generate_par1_interleaved(TRUE_MEAN, TRUE_STD, TRUE_PHI, N_OBS_PER_SEASON, h as u64);

        // Even-indexed observations -> January, odd -> February.
        for (i, v) in series.into_iter().enumerate() {
            let year = 2000 + (i / 2) as i32;
            let month = if i % 2 == 0 { 1u32 } else { 2u32 };
            let date = chrono::NaiveDate::from_ymd_opt(year, month, 15).unwrap();
            hydro_ids.push(hydro_id);
            start_dates.push(date_to_date32(date));
            end_dates.push(date_to_date32(date.succ_opt().expect("valid next date")));
            values.push(v);
        }
    }

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(hydro_ids)),
            Arc::new(Date32Array::from(start_dates)),
            Arc::new(Date32Array::from(end_dates)),
            Arc::new(Float64Array::from(values)),
        ],
    )
    .expect("valid batch");

    let file = std::fs::File::create(path).expect("create parquet file");
    let mut writer = ArrowWriter::try_new(file, schema, None).expect("ArrowWriter");
    writer.write(&batch).expect("write batch");
    writer.close().expect("close writer");
}

/// Build a `System` with `n_hydros` hydros and `N_OBS_PER_SEASON` years of stages,
/// each year a January stage (`season_id` 0) and a February stage (`season_id` 1),
/// so every PAR(1) observation (dated the 15th) falls within a stage's
/// `[1st, 1st-of-next)` window.
fn build_system_for_par1(n_hydros: usize) -> cobre_core::System {
    use cobre_core::{
        DeficitSegment, EntityId, SystemBuilder,
        entities::hydro::{Hydro, HydroGenerationModel, HydroPenalties},
        temporal::{
            Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
            StageStateConfig,
        },
    };

    let bus = make_bus(
        EntityId::from(BUS_ID),
        BusSpec {
            name: "B1".to_string(),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            deficit_segments: vec![DeficitSegment {
                depth_mw: Some(f64::INFINITY),
                cost_per_mwh: 3000.0,
            }],
            excess_cost: 0.0,
            ..Default::default()
        },
    );

    let hydros: Vec<Hydro> = (0..n_hydros)
        .map(|h| {
            make_hydro(
                EntityId::from((h + 1) as i32),
                HydroSpec {
                    name: format!("H{}", h + 1),
                    operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                    bus_id: EntityId::from(BUS_ID),
                    downstream_id: None,
                    entry_stage_id: None,
                    exit_stage_id: None,
                    min_storage_hm3: 0.0,
                    max_storage_hm3: 5000.0,
                    min_outflow_m3s: 0.0,
                    max_outflow_m3s: None,
                    generation_model: HydroGenerationModel::ConstantProductivity,
                    min_turbined_m3s: 0.0,
                    max_turbined_m3s: 1000.0,
                    specific_productivity_mw_per_m3s_per_m: None,
                    min_generation_mw: 0.0,
                    max_generation_mw: 900.0,
                    tailrace: None,
                    hydraulic_losses: None,
                    efficiency: None,
                    evaporation_coefficients_mm: None,
                    evaporation_reference_volumes_hm3: None,
                    diversion: None,
                    filling: None,
                    penalties: HydroPenalties {
                        spillage_cost: 0.0,
                        diversion_cost: 0.0,
                        turbined_cost: 0.0,
                        storage_violation_below_cost: 1000.0,
                        filling_target_violation_cost: 0.0,
                        turbined_violation_below_cost: 0.0,
                        outflow_violation_below_cost: 0.0,
                        outflow_violation_above_cost: 0.0,
                        generation_violation_below_cost: 0.0,
                        evaporation_violation_cost: 0.0,
                        water_withdrawal_violation_cost: 0.0,
                        water_withdrawal_violation_pos_cost: 0.0,
                        water_withdrawal_violation_neg_cost: 0.0,
                        evaporation_violation_pos_cost: 0.0,
                        evaporation_violation_neg_cost: 0.0,
                        inflow_nonnegativity_cost: 1000.0,
                    },
                    ..Default::default()
                },
            )
        })
        .collect();

    let mut stages: Vec<Stage> = Vec::with_capacity(N_OBS_PER_SEASON * N_SEASONS_RT);
    let mut stage_index: usize = 0;
    for year in 0..N_OBS_PER_SEASON {
        let cal_year = 2000 + year as i32;

        stages.push(make_stage(
            stage_index,
            StageSpec {
                start_date: chrono::NaiveDate::from_ymd_opt(cal_year, 1, 1).unwrap(),
                end_date: chrono::NaiveDate::from_ymd_opt(cal_year, 2, 1).unwrap(),
                season_id: Some(0),
                blocks: vec![Block {
                    index: 0,
                    name: "JAN".to_string(),
                    duration_hours: 744.0,
                }],
                block_mode: BlockMode::Parallel,
                state_config: StageStateConfig {
                    storage: true,
                    inflow_lags: false,
                },
                risk_config: StageRiskConfig::Expectation,
                scenario_config: ScenarioSourceConfig {
                    branching_factor: 1,
                    noise_method: NoiseMethod::Saa,
                },
                ..Default::default()
            },
        ));
        stage_index += 1;

        stages.push(make_stage(
            stage_index,
            StageSpec {
                start_date: chrono::NaiveDate::from_ymd_opt(cal_year, 2, 1).unwrap(),
                end_date: chrono::NaiveDate::from_ymd_opt(cal_year, 3, 1).unwrap(),
                season_id: Some(1),
                blocks: vec![Block {
                    index: 0,
                    name: "FEB".to_string(),
                    duration_hours: 672.0,
                }],
                block_mode: BlockMode::Parallel,
                state_config: StageStateConfig {
                    storage: true,
                    inflow_lags: false,
                },
                risk_config: StageRiskConfig::Expectation,
                scenario_config: ScenarioSourceConfig {
                    branching_factor: 1,
                    noise_method: NoiseMethod::Saa,
                },
                ..Default::default()
            },
        ));
        stage_index += 1;
    }

    SystemBuilder::new()
        .buses(vec![bus])
        .hydros(hydros)
        .stages(stages)
        .build()
        .expect("valid system for PAR(1) round-trip")
}

/// PAR(1) round-trip accuracy with 1 hydro and 2 seasons: synthetic `phi = 0.6`
/// observations through the full estimation pipeline must recover phi within
/// `PHI_TOLERANCE` and mean/std within `STAT_RELATIVE_TOLERANCE`.
#[test]
fn test_estimation_round_trip_par1() {
    let dir = TempDir::new().unwrap();
    let case_dir = dir.path();

    create_minimal_case_skeleton(case_dir, "pacf", 1);
    write_par1_inflow_history(&case_dir.join("scenarios/inflow_history.parquet"), 1);

    let system = build_system_for_par1(1);
    let config = parse_config(case_dir);

    let (updated, _report, _path) = estimate_from_history(system, case_dir, &config)
        .expect("PAR(1) estimation with N=200 per season should succeed");

    let n_stages_rt = N_OBS_PER_SEASON * N_SEASONS_RT;
    let models = updated.inflow_models();
    assert_eq!(
        models.len(),
        n_stages_rt,
        "expected {} inflow models (1 hydro × {} stages), got {}",
        n_stages_rt,
        n_stages_rt,
        models.len()
    );

    for m in models {
        assert_eq!(m.hydro_id, EntityId::from(1), "hydro_id should be 1");

        assert!(
            m.mean_m3s.is_finite() && m.mean_m3s > 0.0,
            "mean_m3s should be positive and finite, got {} for stage {}",
            m.mean_m3s,
            m.stage_id
        );
        assert!(
            m.std_m3s.is_finite() && m.std_m3s >= 0.0,
            "std_m3s should be non-negative and finite, got {} for stage {}",
            m.std_m3s,
            m.stage_id
        );

        assert_eq!(
            m.ar_order(),
            1,
            "fixed-order=1 should give ar_order()==1 for stage {}",
            m.stage_id
        );
        let phi_est = m.ar_coefficients[0];
        assert!(
            (phi_est - TRUE_PHI).abs() < PHI_TOLERANCE,
            "estimated phi ({phi_est:.4}) is not within {PHI_TOLERANCE} of true phi ({TRUE_PHI}) \
             for stage {}",
            m.stage_id
        );

        let mean_err = (m.mean_m3s - TRUE_MEAN).abs() / TRUE_MEAN;
        assert!(
            mean_err < STAT_RELATIVE_TOLERANCE,
            "mean_m3s ({:.2}) deviates more than {:.0}% from true mean ({TRUE_MEAN}) \
             for stage {}",
            m.mean_m3s,
            STAT_RELATIVE_TOLERANCE * 100.0,
            m.stage_id
        );

        let std_err = (m.std_m3s - TRUE_STD).abs() / TRUE_STD;
        assert!(
            std_err < STAT_RELATIVE_TOLERANCE,
            "std_m3s ({:.2}) deviates more than {:.0}% from true std ({TRUE_STD}) \
             for stage {}",
            m.std_m3s,
            STAT_RELATIVE_TOLERANCE * 100.0,
            m.stage_id
        );
    }
}

// ── PartialEstimation integration tests ──────────────────────────────────────

/// User-supplied seasonal stats, chosen far from anything the estimator would
/// produce from the synthetic PAR(1) data (`TRUE_MEAN` / `TRUE_STD`) so that
/// "stats preserved unchanged" is unambiguous.
const USER_MEAN: f64 = 12_345.6;
const USER_STD: f64 = 1_111.1;

/// Write `inflow_seasonal_stats.parquet` with `USER_MEAN` / `USER_STD` for one
/// hydro across all stages in `build_system_for_par1(1)`.
fn write_user_inflow_seasonal_stats(case_dir: &Path) {
    use cobre_core::EntityId;
    use cobre_io::output::write_inflow_seasonal_stats;
    use cobre_io::scenarios::InflowSeasonalStatsRow;

    let n_stages = N_OBS_PER_SEASON * N_SEASONS_RT;

    let rows: Vec<InflowSeasonalStatsRow> = (0..n_stages)
        .map(|stage_id| InflowSeasonalStatsRow {
            hydro_id: EntityId::from(1_i32),
            stage_id: stage_id as i32,
            mean_m3s: USER_MEAN,
            std_m3s: USER_STD,
        })
        .collect();

    write_inflow_seasonal_stats(
        &case_dir.join("scenarios/inflow_seasonal_stats.parquet"),
        &rows,
    )
    .expect("write_inflow_seasonal_stats must succeed");
}

/// `build_system_for_par1(1)` with pre-loaded inflow models carrying `USER_MEAN` /
/// `USER_STD` and empty `ar_coefficients` — the state after `load_case` has parsed
/// `inflow_seasonal_stats.parquet`, before estimation.
fn build_system_for_par1_with_user_stats() -> cobre_core::System {
    use cobre_core::scenario::InflowModel;

    let base = build_system_for_par1(1);
    let stages = base.stages();

    let inflow_models: Vec<InflowModel> = stages
        .iter()
        .map(|s| InflowModel {
            hydro_id: EntityId::from(1_i32),
            stage_id: s.id,
            mean_m3s: USER_MEAN,
            std_m3s: USER_STD,
            ar_coefficients: vec![],
            residual_std_ratio: 1.0,
            annual: None,
        })
        .collect();

    let stages_owned = stages.to_vec();
    let hydros_owned = base.hydros().to_vec();
    let buses_owned = (0..base.n_buses())
        .map(|_| {
            make_bus(
                EntityId::from(BUS_ID),
                BusSpec {
                    name: "B1".to_string(),
                    operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                    deficit_segments: vec![DeficitSegment {
                        depth_mw: Some(f64::INFINITY),
                        cost_per_mwh: 3000.0,
                    }],
                    excess_cost: 0.0,
                    ..Default::default()
                },
            )
        })
        .take(1)
        .collect::<Vec<_>>();

    SystemBuilder::new()
        .buses(buses_owned)
        .hydros(hydros_owned)
        .stages(stages_owned)
        .inflow_models(inflow_models)
        .build()
        .expect("valid system for PAR(1) partial estimation")
}

/// `PartialEstimation` end-to-end: with user seasonal stats present but no
/// `inflow_ar_coefficients.parquet`, the user `mean_m3s` / `std_m3s` survive
/// bitwise-unchanged while AR coefficients are estimated from history.
#[test]
fn test_partial_estimation_end_to_end() {
    let dir = TempDir::new().unwrap();
    let case_dir = dir.path();

    create_minimal_case_skeleton(case_dir, "pacf", 2);
    write_par1_inflow_history(&case_dir.join("scenarios/inflow_history.parquet"), 1);
    write_user_inflow_seasonal_stats(case_dir);
    let system = build_system_for_par1_with_user_stats();
    let config = parse_config(case_dir);

    let (updated, report, path) =
        estimate_from_history(system, case_dir, &config).expect("partial estimation must succeed");

    assert_eq!(
        path,
        EstimationPath::PartialEstimation,
        "expected PartialEstimation path, got {path:?}"
    );

    let report = report.expect("PartialEstimation must return Some(EstimationReport)");
    assert_eq!(report.method, "PACF", "estimation method must be PACF");
    assert_eq!(
        report.entries.len(),
        1,
        "report must contain exactly 1 entry (one hydro)"
    );

    let models = updated.inflow_models();
    assert!(!models.is_empty(), "must produce at least one inflow model");

    for m in models {
        // Bitwise (to_bits) comparison: partial estimation must not round or
        // transform the user stats.
        assert_eq!(
            m.mean_m3s.to_bits(),
            USER_MEAN.to_bits(),
            "mean_m3s must equal USER_MEAN ({USER_MEAN}) for stage {}; got {}",
            m.stage_id,
            m.mean_m3s
        );
        assert_eq!(
            m.std_m3s.to_bits(),
            USER_STD.to_bits(),
            "std_m3s must equal USER_STD ({USER_STD}) for stage {}; got {}",
            m.stage_id,
            m.std_m3s
        );

        assert!(
            !m.ar_coefficients.is_empty(),
            "ar_coefficients must be non-empty for stage {} (estimated from history)",
            m.stage_id
        );
    }
}

/// PAR(1) round-trip with 2 hydros: `test_estimation_round_trip_par1` extended to
/// two statistically independent hydros, asserting the model count scales and each
/// hydro recovers phi within `PHI_TOLERANCE`.
#[test]
fn test_estimation_round_trip_two_hydros() {
    const N_HYDROS: usize = 2;
    let n_stages_rt = N_OBS_PER_SEASON * N_SEASONS_RT;
    let expected_models = N_HYDROS * n_stages_rt;

    let dir = TempDir::new().unwrap();
    let case_dir = dir.path();

    create_minimal_case_skeleton(case_dir, "pacf", 1);
    write_par1_inflow_history(&case_dir.join("scenarios/inflow_history.parquet"), N_HYDROS);

    let system = build_system_for_par1(N_HYDROS);
    let config = parse_config(case_dir);

    let (updated, _report, _path) = estimate_from_history(system, case_dir, &config)
        .expect("PAR(1) estimation with 2 hydros should succeed");

    let models = updated.inflow_models();
    assert_eq!(
        models.len(),
        expected_models,
        "expected {} inflow models ({N_HYDROS} hydros × {n_stages_rt} stages), got {}",
        expected_models,
        models.len()
    );

    for m in models {
        assert!(
            m.mean_m3s.is_finite() && m.mean_m3s > 0.0,
            "mean_m3s should be positive and finite, got {} for hydro {} stage {}",
            m.mean_m3s,
            m.hydro_id.0,
            m.stage_id
        );
        assert!(
            m.std_m3s.is_finite() && m.std_m3s > 0.0,
            "std_m3s should be positive and finite, got {} for hydro {} stage {}",
            m.std_m3s,
            m.hydro_id.0,
            m.stage_id
        );

        assert_eq!(
            m.ar_order(),
            1,
            "fixed-order=1 should give ar_order()==1 for hydro {} stage {}",
            m.hydro_id.0,
            m.stage_id
        );
        let phi_est = m.ar_coefficients[0];
        assert!(
            (phi_est - TRUE_PHI).abs() < PHI_TOLERANCE,
            "estimated phi ({phi_est:.4}) is not within {PHI_TOLERANCE} of true phi ({TRUE_PHI}) \
             for hydro {} stage {}",
            m.hydro_id.0,
            m.stage_id
        );
    }
}

// ── UserArHistoryStats (pre-study synthesis) integration tests ────────────────

/// Calendar year the partial-year study's stages fall in; well inside
/// `write_inflow_history`'s `[START_YEAR, START_YEAR + N_YEARS)` range so the
/// pre-study lag window (into the prior calendar months) has observations on
/// both sides.
const PARTIAL_STUDY_YEAR: i32 = 2005;

/// Build a 12-season monthly `SeasonMap` (season id `m` maps to calendar month
/// `m + 1`), required for `synthesize_prestudy_stages`'s out-of-window fallback.
fn monthly_season_map() -> cobre_core::SeasonMap {
    use cobre_core::{SeasonCycleType, SeasonDefinition, SeasonMap};
    let seasons = (0..N_SEASONS)
        .map(|m| SeasonDefinition {
            id: m,
            label: format!("M{m}"),
            month_start: u32::try_from(m + 1).unwrap(),
            day_start: None,
            month_end: None,
            day_end: None,
        })
        .collect();
    SeasonMap {
        cycle_type: SeasonCycleType::Monthly,
        seasons,
    }
}

/// Build a `System` with one hydro and `n` monthly study stages spanning
/// seasons `[first_season, first_season + n)` of `PARTIAL_STUDY_YEAR`, carrying
/// a 12-season `SeasonMap` on `policy_graph` — the fixture
/// `run_user_ar_estimation`'s pre-study synthesis needs to resolve out-of-window
/// lag seasons.
fn build_season_mapped_system(first_season: usize, n: usize) -> cobre_core::System {
    use cobre_core::{HorizonGraph, PolicyGraphType};

    let bus = make_bus(
        EntityId::from(BUS_ID),
        BusSpec {
            name: "B1".to_string(),
            operational_start_date: NaiveDate::from_ymd_opt(PARTIAL_STUDY_YEAR, 1, 1).unwrap(),
            deficit_segments: vec![DeficitSegment {
                depth_mw: Some(f64::INFINITY),
                cost_per_mwh: 3000.0,
            }],
            excess_cost: 0.0,
            ..Default::default()
        },
    );

    let hydro = make_hydro(
        EntityId::from(HYDRO_ID),
        HydroSpec {
            name: "H1".to_string(),
            operational_start_date: NaiveDate::from_ymd_opt(PARTIAL_STUDY_YEAR, 1, 1).unwrap(),
            bus_id: EntityId::from(BUS_ID),
            downstream_id: None,
            entry_stage_id: None,
            exit_stage_id: None,
            min_storage_hm3: 0.0,
            max_storage_hm3: 5000.0,
            min_outflow_m3s: 0.0,
            max_outflow_m3s: None,
            generation_model: HydroGenerationModel::ConstantProductivity,
            min_turbined_m3s: 0.0,
            max_turbined_m3s: 1000.0,
            specific_productivity_mw_per_m3s_per_m: None,
            min_generation_mw: 0.0,
            max_generation_mw: 900.0,
            tailrace: None,
            hydraulic_losses: None,
            efficiency: None,
            evaporation_coefficients_mm: None,
            evaporation_reference_volumes_hm3: None,
            diversion: None,
            filling: None,
            penalties: HydroPenalties {
                spillage_cost: 0.0,
                diversion_cost: 0.0,
                turbined_cost: 0.0,
                storage_violation_below_cost: 1000.0,
                filling_target_violation_cost: 0.0,
                turbined_violation_below_cost: 0.0,
                outflow_violation_below_cost: 0.0,
                outflow_violation_above_cost: 0.0,
                generation_violation_below_cost: 0.0,
                evaporation_violation_cost: 0.0,
                water_withdrawal_violation_cost: 0.0,
                water_withdrawal_violation_pos_cost: 0.0,
                water_withdrawal_violation_neg_cost: 0.0,
                evaporation_violation_pos_cost: 0.0,
                evaporation_violation_neg_cost: 0.0,
                inflow_nonnegativity_cost: 1000.0,
            },
            ..Default::default()
        },
    );

    let stages: Vec<Stage> = (0..n)
        .map(|k| {
            let season = first_season + k;
            let month = u32::try_from(season % N_SEASONS + 1).unwrap();
            let cal_year = PARTIAL_STUDY_YEAR + i32::try_from(season / N_SEASONS).unwrap();
            let (end_year, end_month) = if month == 12 {
                (cal_year + 1, 1u32)
            } else {
                (cal_year, month + 1)
            };
            make_stage(
                k,
                StageSpec {
                    start_date: NaiveDate::from_ymd_opt(cal_year, month, 1).unwrap(),
                    end_date: NaiveDate::from_ymd_opt(end_year, end_month, 1).unwrap(),
                    season_id: Some(season % N_SEASONS),
                    blocks: vec![Block {
                        index: 0,
                        name: "SINGLE".to_string(),
                        duration_hours: 720.0,
                    }],
                    block_mode: BlockMode::Parallel,
                    state_config: StageStateConfig {
                        storage: true,
                        inflow_lags: false,
                    },
                    risk_config: StageRiskConfig::Expectation,
                    scenario_config: ScenarioSourceConfig {
                        branching_factor: 1,
                        noise_method: NoiseMethod::Saa,
                    },
                },
            )
        })
        .collect();

    let policy_graph = HorizonGraph {
        stage_discount_rate_overrides: std::collections::BTreeMap::new(),
        graph_type: PolicyGraphType::FiniteHorizon,
        annual_discount_rate: 0.0,
        transitions: Vec::new(),
        nodes: Vec::new(),
        season_map: Some(monthly_season_map()),
    };

    SystemBuilder::new()
        .buses(vec![bus])
        .hydros(vec![hydro])
        .stages(stages)
        .policy_graph(policy_graph)
        .build()
        .expect("valid season-mapped system")
}

/// Write `inflow_ar_coefficients.parquet` with a fixed PAR(`max_order`) model for
/// `HYDRO_ID` across stages `0..n_stages`, driving the R=1 manifest flag for the
/// `UserArHistoryStats` path (H=1, S=0, R=1).
fn write_user_inflow_ar_coefficients(case_dir: &Path, n_stages: i32, max_order: i32) {
    use cobre_io::output::write_inflow_ar_coefficients;
    use cobre_io::scenarios::InflowArCoefficientRow;

    let mut rows = Vec::with_capacity(usize::try_from(n_stages * max_order).unwrap());
    for stage_id in 0..n_stages {
        for lag in 1..=max_order {
            rows.push(InflowArCoefficientRow {
                hydro_id: EntityId::from(HYDRO_ID),
                stage_id,
                lag,
                coefficient: 0.1 * f64::from(lag),
            });
        }
    }

    write_inflow_ar_coefficients(
        &case_dir.join("scenarios/inflow_ar_coefficients.parquet"),
        &rows,
    )
    .expect("write_inflow_ar_coefficients must succeed");
}

/// `UserArHistoryStats`, partial-year study (Sep–Dec, seasons 8–11): the first
/// study stage's lag-1 season (7 = August) has no study stage of its own, so
/// `run_user_ar_estimation` must synthesize a pre-study stage (id −1) and carry
/// a history-derived, non-zero `mean_m3s` there, proving the
/// `synthesize_prestudy_stages` call is wired into the estimation path.
///
/// An in-memory-edit discrimination control (temporarily reverting
/// `run_user_ar_estimation` to its pre-fix body, confirming this assertion then
/// fails with `mean_at_lag1 == 0.0`, then restoring the file byte-identical) is
/// performed out-of-band during implementation, not as an automated test path.
#[test]
fn user_ar_prestudy_synthesizes_history_derived_lag_stats() {
    let dir = TempDir::new().unwrap();
    let case_dir = dir.path();

    create_minimal_case_skeleton(case_dir, "pacf", 2);
    write_inflow_history(&case_dir.join("scenarios/inflow_history.parquet"));
    write_user_inflow_ar_coefficients(case_dir, 4, 2);

    let system = build_season_mapped_system(8, 4);
    let config = parse_config(case_dir);

    let (updated, report, path) = estimate_from_history(system, case_dir, &config)
        .expect("UserArHistoryStats estimation must succeed");

    assert_eq!(
        path,
        EstimationPath::UserArHistoryStats,
        "expected UserArHistoryStats path, got {path:?}"
    );
    let report = report.expect("UserArHistoryStats must return Some(EstimationReport)");
    assert_eq!(
        report.method, "user_provided",
        "method must be user_provided"
    );

    // Stage id -1 is the synthesized pre-study stage for season 7 (August), the
    // first study stage's lag-1 season. A missing entry (pre-fix: no synthesis)
    // maps to 0.0 via map_or, matching PrecomputedPar::build's own zero fallback.
    let mean_at_lag1 = updated
        .inflow_models()
        .iter()
        .find(|m| m.hydro_id == EntityId::from(HYDRO_ID) && m.stage_id == -1)
        .map_or(0.0, |m| m.mean_m3s);

    assert!(
        mean_at_lag1.abs() > f64::EPSILON,
        "pre-study stage -1 (season 7 / August) must carry a non-zero \
         history-derived mean_m3s, got {mean_at_lag1}; a zero here means \
         run_user_ar_estimation is not synthesizing pre-study stages before \
         estimating seasonal stats"
    );
}

/// `UserArHistoryStats`, full-year study (all 12 seasons covered): pre-study
/// synthesis is a no-op (`synthesize_prestudy_stages` returns empty when every
/// season already has a study stage), so no negative-`stage_id` `InflowModel`
/// entries appear — the fix leaves the full-year path unchanged.
#[test]
fn user_ar_prestudy_full_year_study_synthesizes_nothing() {
    let dir = TempDir::new().unwrap();
    let case_dir = dir.path();

    create_minimal_case_skeleton(case_dir, "pacf", 2);
    write_inflow_history(&case_dir.join("scenarios/inflow_history.parquet"));
    write_user_inflow_ar_coefficients(case_dir, 12, 2);

    let system = build_season_mapped_system(0, 12);
    let config = parse_config(case_dir);

    let (updated, _report, path) = estimate_from_history(system, case_dir, &config)
        .expect("full-year UserArHistoryStats estimation must succeed");

    assert_eq!(
        path,
        EstimationPath::UserArHistoryStats,
        "expected UserArHistoryStats path, got {path:?}"
    );

    let models = updated.inflow_models();
    assert_eq!(
        models.len(),
        12,
        "full-year study must produce exactly 12 models (one per season), got {}",
        models.len()
    );
    assert!(
        models.iter().all(|m| m.stage_id >= 0),
        "full-year study must synthesize no pre-study stages: found negative \
         stage_id(s) {:?}",
        models
            .iter()
            .filter(|m| m.stage_id < 0)
            .map(|m| m.stage_id)
            .collect::<Vec<_>>()
    );
}

/// `UserArHistoryStats`, partial-year study, correlation estimated (no
/// `scenarios/correlation.json`): September (season 8, the study's first
/// stage) needs both its AR lags — August and July, seasons 7 and 6 — neither
/// of which has a study stage of its own; only the pre-study-synthesized
/// stages carry them. Passing the study-only stage set (rather than the
/// pre-study-extended set already used for seasonal-stats estimation) to the
/// correlation call drops every September residual, since the lag lookup
/// misses both seasons. A history long enough to cross the correlation
/// module's fixed 30-residual-per-season minimum turns that into an
/// observable difference: a `season_08` profile appears once the fix is
/// applied and never appears under the study-only regression.
///
/// An in-memory-edit discrimination control (temporarily reverting
/// `run_user_ar_estimation`'s correlation call to pass `stages` instead of
/// `extended`, confirming this assertion then fails, then restoring the file
/// byte-identical) is performed out-of-band during implementation, not as an
/// automated test path.
#[test]
fn user_ar_prestudy_correlation_uses_extended_stages() {
    const HISTORY_YEARS: usize = 32;

    let dir = TempDir::new().unwrap();
    let case_dir = dir.path();

    create_minimal_case_skeleton(case_dir, "pacf", 2);
    write_inflow_history_n_years(
        &case_dir.join("scenarios/inflow_history.parquet"),
        HISTORY_YEARS,
    );
    write_user_inflow_ar_coefficients(case_dir, 4, 2);

    let system = build_season_mapped_system(8, 4);
    let config = parse_config(case_dir);

    let (updated, _report, path) = estimate_from_history(system, case_dir, &config)
        .expect("UserArHistoryStats estimation must succeed");

    assert_eq!(
        path,
        EstimationPath::UserArHistoryStats,
        "expected UserArHistoryStats path, got {path:?}"
    );

    let correlation = updated.correlation();
    assert!(
        correlation.profiles.contains_key("season_08"),
        "expected a season_08 (September) correlation profile once {HISTORY_YEARS} \
         years of history cross the correlation module's minimum-sample threshold; \
         a missing profile means correlation estimation dropped every September \
         residual because its AR lags (August, July) require seasonal stats that \
         only the pre-study-extended stage set carries, got profiles {:?}",
        correlation.profiles.keys().collect::<Vec<_>>()
    );
}
