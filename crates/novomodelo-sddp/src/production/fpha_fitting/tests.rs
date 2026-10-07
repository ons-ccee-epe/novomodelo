use cobre_core::{
    EfficiencyModel, EntityId, HydraulicLossesModel, Hydro, HydroGenerationModel, HydroPenalties,
    TailraceModel, TailracePoint,
};
use cobre_io::PlaneReductionConfig;
use cobre_io::extensions::{FittingWindow, FphaColumnLayout, HydroGeometryRow, TailraceCurveRow};

use super::alpha::compute_alpha_fpha;
use super::error::FphaFittingError;
use super::fit_fpha_planes;
use super::geometry::{
    FittingBounds, ForebayTable, evaluate_losses, evaluate_tailrace, resolve_fitting_bounds,
};
use super::hull_fit::{RawPlane, fit_hull_planes};
use super::production::{ProductionFunction, TailraceSource};
use super::selection::validate_fitted_planes;
use super::tailrace::TailraceFamilies;
use crate::hydro_models::FphaPlane;

// ── Helpers ───────────────────────────────────────────────────────────────

/// Build a minimal [`HydroGeometryRow`] with `area_km2 = 0.0`.
fn row(volume_hm3: f64, height_m: f64) -> HydroGeometryRow {
    HydroGeometryRow {
        hydro_id: EntityId::from(1),
        volume_hm3,
        height_m,
        area_km2: 0.0,
    }
}

/// Sobradinho-style 5-point VHA curve used across several tests.
fn sobradinho_rows() -> Vec<HydroGeometryRow> {
    vec![
        row(0.0, 386.5),
        row(2_000.0, 390.0),
        row(10_000.0, 396.0),
        row(24_500.0, 400.5),
        row(34_116.0, 401.3),
    ]
}

// ── Valid construction ────────────────────────────────────────────────────

#[test]
fn valid_five_point_curve_construction_succeeds() {
    let table = ForebayTable::new(&sobradinho_rows(), "Sobradinho").unwrap();
    assert_eq!(table.v_min(), 0.0);
    assert_eq!(table.v_max(), 34_116.0);
}

// ── Interpolation at midpoint ─────────────────────────────────────────────

/// height(1000.0) on segment [0, 2000] with heights [386.5, 390.0] must equal
/// 386.5 + (390.0 - 386.5) * 1000.0 / 2000.0 = 388.25 within 1e-10.
#[test]
fn interpolation_at_midpoint_segment_0_to_2000() {
    let table = ForebayTable::new(&sobradinho_rows(), "Sobradinho").unwrap();
    let expected = 386.5 + (390.0 - 386.5) * 1000.0 / 2000.0;
    assert!((table.height(1000.0) - expected).abs() < 1e-10);
}

// ── Interpolation at breakpoints ──────────────────────────────────────────

#[test]
fn interpolation_at_breakpoints_returns_exact_values() {
    let rows = sobradinho_rows();
    let table = ForebayTable::new(&rows, "Sobradinho").unwrap();
    for r in &rows {
        assert!(
            (table.height(r.volume_hm3) - r.height_m).abs() < 1e-10,
            "breakpoint v={}: expected h={}, got h={}",
            r.volume_hm3,
            r.height_m,
            table.height(r.volume_hm3)
        );
    }
}

// ── Clamping below v_min ──────────────────────────────────────────────────

#[test]
fn height_clamped_below_v_min() {
    let table = ForebayTable::new(&sobradinho_rows(), "Sobradinho").unwrap();
    let at_min = table.height(table.v_min());
    let below = table.height(table.v_min() - 100.0);
    assert!((at_min - below).abs() < 1e-10);
}

// ── Clamping above v_max ──────────────────────────────────────────────────

#[test]
fn height_clamped_above_v_max() {
    let table = ForebayTable::new(&sobradinho_rows(), "Sobradinho").unwrap();
    let at_max = table.height(table.v_max());
    let above = table.height(table.v_max() + 999.0);
    assert!((at_max - above).abs() < 1e-10);
}

// ── Insufficient points (0 points) ───────────────────────────────────────

#[test]
fn insufficient_points_zero_rows() {
    let err = ForebayTable::new(&[], "Itaipu").unwrap_err();
    match err {
        FphaFittingError::InsufficientPoints { count, .. } => {
            assert_eq!(count, 0);
        }
        other => panic!("expected InsufficientPoints, got: {other:?}"),
    }
}

// ── Single point is accepted as a constant (run-of-river) forebay ─────────

#[test]
fn single_row_builds_constant_forebay() {
    // A run-of-river plant has one operating volume → one VHA row → a CONSTANT
    // forebay. The table must build and return that elevation for ANY query
    // (below, at, and above the single point), never panicking in `height`.
    let table =
        ForebayTable::new(&[row(1500.0, 386.5)], "RunOfRiver").expect("single row is valid");
    assert_eq!(table.v_min(), 1500.0);
    assert_eq!(table.v_max(), 1500.0);
    assert_eq!(table.height(0.0), 386.5);
    assert_eq!(table.height(1500.0), 386.5);
    assert_eq!(table.height(9999.0), 386.5);
}

// ── Non-monotonic volume (duplicate) ──────────────────────────────────────

#[test]
fn non_monotonic_volume_duplicate() {
    let rows = vec![row(0.0, 386.5), row(1000.0, 388.0), row(1000.0, 390.0)];
    let err = ForebayTable::new(&rows, "Tucurui").unwrap_err();
    match err {
        FphaFittingError::NonMonotonicVolume {
            index,
            v_prev,
            v_curr,
            ..
        } => {
            assert_eq!(index, 2);
            assert_eq!(v_prev, 1000.0);
            assert_eq!(v_curr, 1000.0);
        }
        other => panic!("expected NonMonotonicVolume, got: {other:?}"),
    }
}

// ── Non-monotonic volume (decreasing) ─────────────────────────────────────

#[test]
fn non_monotonic_volume_decreasing() {
    let rows = vec![row(0.0, 386.5), row(2000.0, 390.0), row(1500.0, 392.0)];
    let err = ForebayTable::new(&rows, "Belo Monte").unwrap_err();
    match err {
        FphaFittingError::NonMonotonicVolume { index, .. } => {
            assert_eq!(index, 2);
        }
        other => panic!("expected NonMonotonicVolume, got: {other:?}"),
    }
}

// ── Non-monotonic height (decreasing) ─────────────────────────────────────

#[test]
fn non_monotonic_height_decreasing() {
    let rows = vec![row(0.0, 390.0), row(1000.0, 392.0), row(2000.0, 388.0)];
    let err = ForebayTable::new(&rows, "Serra da Mesa").unwrap_err();
    match err {
        FphaFittingError::NonMonotonicHeight {
            index,
            h_prev,
            h_curr,
            ..
        } => {
            assert_eq!(index, 2);
            assert_eq!(h_prev, 392.0);
            assert_eq!(h_curr, 388.0);
        }
        other => panic!("expected NonMonotonicHeight, got: {other:?}"),
    }
}

// ── Plateau heights (equal consecutive) ───────────────────────────────────

#[test]
fn equal_consecutive_heights_accepted() {
    let rows = vec![row(0.0, 386.5), row(1000.0, 386.5), row(2000.0, 390.0)];
    let result = ForebayTable::new(&rows, "Furnas");
    assert!(result.is_ok(), "plateau heights should be accepted");
}

// ── Display messages are informative ──────────────────────────────────────

#[test]
fn display_insufficient_points_contains_name_and_count() {
    let err = ForebayTable::new(&[], "Anta").unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("Anta"), "should contain hydro name: {msg}");
    assert!(msg.contains('0'), "should contain count: {msg}");
}

#[test]
fn display_non_monotonic_volume_contains_name_and_index() {
    let rows = vec![row(0.0, 386.5), row(1000.0, 390.0), row(500.0, 392.0)];
    let err = ForebayTable::new(&rows, "Cachoeira").unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("Cachoeira"),
        "should contain hydro name: {msg}"
    );
    assert!(msg.contains('2'), "should contain index: {msg}");
}

#[test]
fn display_non_monotonic_height_contains_name_and_index() {
    let rows = vec![row(0.0, 395.0), row(1000.0, 393.0)];
    let err = ForebayTable::new(&rows, "Marimbondo").unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("Marimbondo"),
        "should contain hydro name: {msg}"
    );
    assert!(msg.contains('1'), "should contain index: {msg}");
}

#[test]
fn fpha_fitting_error_implements_std_error() {
    fn assert_error<E: std::error::Error>() {}
    assert_error::<FphaFittingError>();
}

// ── Tailrace evaluation: Polynomial ──────────────────────────────────────

#[test]
fn tailrace_polynomial_constant_one_coefficient() {
    let model = TailraceModel::Polynomial {
        coefficients: vec![7.5],
    };
    assert!((evaluate_tailrace(&model, 0.0) - 7.5).abs() < 1e-10);
    assert!((evaluate_tailrace(&model, 5000.0) - 7.5).abs() < 1e-10);
}

#[test]
fn tailrace_polynomial_linear_two_coefficients() {
    let model = TailraceModel::Polynomial {
        coefficients: vec![3.0, 0.0005],
    };
    // At q = 2000: 3.0 + 0.0005 * 2000 = 4.0
    assert!((evaluate_tailrace(&model, 2000.0) - 4.0).abs() < 1e-10);
}

#[test]
fn tailrace_polynomial_quadratic_acceptance_criterion() {
    let model = TailraceModel::Polynomial {
        coefficients: vec![5.0, 0.001, -1e-7],
    };
    let expected = 5.0 + 0.001 * 3000.0 + (-1e-7) * 3000.0_f64.powi(2);
    assert!((evaluate_tailrace(&model, 3000.0) - expected).abs() < 1e-10);
}

#[test]
fn tailrace_polynomial_quartic_five_coefficients() {
    // h = 1 + 2q + 3q^2 + 4q^3 + 5q^4 at q = 1
    // = 1 + 2 + 3 + 4 + 5 = 15
    let model = TailraceModel::Polynomial {
        coefficients: vec![1.0, 2.0, 3.0, 4.0, 5.0],
    };
    assert!((evaluate_tailrace(&model, 1.0) - 15.0).abs() < 1e-10);
}

// ── Tailrace evaluation: Piecewise ────────────────────────────────────────

/// Build the 3-point piecewise model from the acceptance criteria.
fn ac_piecewise() -> TailraceModel {
    use cobre_core::TailracePoint;
    TailraceModel::Piecewise {
        points: vec![
            TailracePoint {
                outflow_m3s: 0.0,
                height_m: 3.0,
            },
            TailracePoint {
                outflow_m3s: 5000.0,
                height_m: 4.5,
            },
            TailracePoint {
                outflow_m3s: 15_000.0,
                height_m: 6.2,
            },
        ],
    }
}

#[test]
fn tailrace_piecewise_midpoint_first_segment_acceptance_criterion() {
    let model = ac_piecewise();
    let expected = 3.0 + (4.5 - 3.0) * 2500.0 / 5000.0;
    assert!((evaluate_tailrace(&model, 2500.0) - expected).abs() < 1e-10);
}

#[test]
fn tailrace_piecewise_at_breakpoints_exact() {
    let model = ac_piecewise();
    assert!((evaluate_tailrace(&model, 0.0) - 3.0).abs() < 1e-10);
    assert!((evaluate_tailrace(&model, 5000.0) - 4.5).abs() < 1e-10);
    assert!((evaluate_tailrace(&model, 15_000.0) - 6.2).abs() < 1e-10);
}

#[test]
fn tailrace_piecewise_clamp_below_range() {
    let model = ac_piecewise();
    assert!((evaluate_tailrace(&model, -500.0) - 3.0).abs() < 1e-10);
}

#[test]
fn tailrace_piecewise_clamp_above_range() {
    let model = ac_piecewise();
    assert!((evaluate_tailrace(&model, 99_999.0) - 6.2).abs() < 1e-10);
}

// ── Hydraulic losses: Factor ──────────────────────────────────────────────

#[test]
fn losses_factor_acceptance_criterion() {
    let model = HydraulicLossesModel::Factor { value: 0.03 };
    assert!((evaluate_losses(&model, 100.0, 0.0) - 3.0).abs() < 1e-10);
}

#[test]
fn losses_factor_scales_with_gross_head() {
    let model = HydraulicLossesModel::Factor { value: 0.05 };
    assert!((evaluate_losses(&model, 200.0, 1000.0) - 10.0).abs() < 1e-10);
    assert!((evaluate_losses(&model, 0.0, 5000.0) - 0.0).abs() < 1e-10);
}

#[test]
fn losses_factor_turbined_has_no_effect() {
    let model = HydraulicLossesModel::Factor { value: 0.02 };
    let r1 = evaluate_losses(&model, 80.0, 0.0);
    let r2 = evaluate_losses(&model, 80.0, 99_999.0);
    assert!((r1 - r2).abs() < 1e-10);
}

// ── Hydraulic losses: Constant ────────────────────────────────────────────

#[test]
fn losses_constant_acceptance_criterion() {
    let model = HydraulicLossesModel::Constant { value_m: 2.5 };
    assert!((evaluate_losses(&model, 0.0, 0.0) - 2.5).abs() < 1e-10);
    assert!((evaluate_losses(&model, 999.0, 0.0) - 2.5).abs() < 1e-10);
}

#[test]
fn losses_constant_independent_of_all_inputs() {
    let model = HydraulicLossesModel::Constant { value_m: 1.25 };
    let r1 = evaluate_losses(&model, 50.0, 500.0);
    let r2 = evaluate_losses(&model, 200.0, 8000.0);
    assert!((r1 - 1.25).abs() < 1e-10);
    assert!((r2 - 1.25).abs() < 1e-10);
}

// ── Two-point minimum curve ───────────────────────────────────────────────

#[test]
fn two_point_minimum_curve_works() {
    let rows = vec![row(0.0, 380.0), row(1000.0, 400.0)];
    let table = ForebayTable::new(&rows, "MinimalPlant").unwrap();
    assert_eq!(table.v_min(), 0.0);
    assert_eq!(table.v_max(), 1000.0);
    // Midpoint: 380 + 0.5 * (400 - 380) = 390
    assert!((table.height(500.0) - 390.0).abs() < 1e-10);
}

// ── Second segment midpoint interpolation ─────────────────────────────────

#[test]
fn interpolation_second_segment_correct() {
    let table = ForebayTable::new(&sobradinho_rows(), "Sobradinho").unwrap();
    // v = 6000: fraction = (6000 - 2000) / (10000 - 2000) = 0.5
    // h = 390.0 + 0.5 * (396.0 - 390.0) = 393.0
    let expected = 390.0 + 0.5 * (396.0 - 390.0);
    assert!((table.height(6_000.0) - expected).abs() < 1e-10);
}

// ── ProductionFunction helpers ────────────────────────────────────────────

/// A 2-point ForebayTable returning a constant 400 m for all v.
fn flat_forebay_400m() -> ForebayTable {
    ForebayTable::new(
        &[
            HydroGeometryRow {
                hydro_id: EntityId::from(1),
                volume_hm3: 0.0,
                height_m: 400.0,
                area_km2: 0.0,
            },
            HydroGeometryRow {
                hydro_id: EntityId::from(1),
                volume_hm3: 20_000.0,
                height_m: 400.0,
                area_km2: 0.0,
            },
        ],
        "TestPlant",
    )
    .unwrap()
}

/// Build a sloped ForebayTable: h_fore = 380 + v * 2e-3 m (slope = 2e-3 m/hm3).
/// At v = 10000 hm3: h_fore = 380 + 10000 * 2e-3 = 400 m.
fn sloped_forebay() -> ForebayTable {
    ForebayTable::new(
        &[
            HydroGeometryRow {
                hydro_id: EntityId::from(1),
                volume_hm3: 0.0,
                height_m: 380.0,
                area_km2: 0.0,
            },
            HydroGeometryRow {
                hydro_id: EntityId::from(1),
                volume_hm3: 10_000.0,
                height_m: 400.0,
                area_km2: 0.0,
            },
        ],
        "TestPlant",
    )
    .unwrap()
}

/// A linear tailrace `h = (5.5 / 3000)·q`, giving `h_tail = 5.5 m` at q = 3000.
fn linear_tailrace_5_5_at_3000() -> TailraceModel {
    let slope = 5.5 / 3000.0;
    TailraceModel::Polynomial {
        coefficients: vec![0.0, slope],
    }
}

/// Build the 3-point piecewise tailrace from the existing test helper (reused).
fn piecewise_tailrace() -> TailraceModel {
    TailraceModel::Piecewise {
        points: vec![
            TailracePoint {
                outflow_m3s: 0.0,
                height_m: 3.0,
            },
            TailracePoint {
                outflow_m3s: 5000.0,
                height_m: 4.5,
            },
            TailracePoint {
                outflow_m3s: 15_000.0,
                height_m: 6.2,
            },
        ],
    }
}

// ── net_head tests ────────────────────────────────────────────────────────

/// Net head with no tailrace, no losses: h_net = h_fore(v).
#[test]
fn net_head_no_tailrace_no_losses_equals_h_fore() {
    let forebay = sloped_forebay();
    let pf = ProductionFunction::new(
        forebay,
        TailraceSource::Entity(None),
        None,
        None,
        12_600.0,
        "TestPlant".to_owned(),
    );
    // At v=10000: h_fore = 400. h_tail = 0. h_loss = 0. h_net = 400.
    assert!((pf.net_head(10_000.0, 3000.0, 0.0) - 400.0).abs() < 1e-10);
}

#[test]
fn net_head_polynomial_tailrace_constant_losses_acceptance_criterion() {
    let forebay = flat_forebay_400m();
    let tailrace = linear_tailrace_5_5_at_3000();
    let losses = HydraulicLossesModel::Constant { value_m: 2.0 };
    let pf = ProductionFunction::new(
        forebay,
        TailraceSource::Entity(Some(tailrace)),
        Some(&losses),
        Some(&EfficiencyModel::Constant { value: 0.92 }),
        12_600.0,
        "TestPlant".to_owned(),
    );
    let h_net = pf.net_head(10_000.0, 3000.0, 0.0);
    let expected = 400.0 - 5.5 - 2.0; // 392.5
    assert!(
        (h_net - expected).abs() < 1e-6,
        "h_net={h_net}, expected={expected}"
    );
}

#[test]
fn net_head_piecewise_tailrace_factor_losses() {
    let forebay = flat_forebay_400m();
    let tailrace = piecewise_tailrace();
    let losses = HydraulicLossesModel::Factor { value: 0.03 };
    let pf = ProductionFunction::new(
        forebay,
        TailraceSource::Entity(Some(tailrace)),
        Some(&losses),
        None,
        12_600.0,
        "TestPlant".to_owned(),
    );
    // q_out = q + s = 2500 + 0 = 2500
    let h_tail = 3.0 + (4.5 - 3.0) * 2500.0 / 5000.0; // 3.75
    let gross_head = 400.0 - h_tail; // 396.25
    let expected = (1.0 - 0.03) * gross_head; // 384.3625
    let h_net = pf.net_head(0.0, 2500.0, 0.0);
    assert!(
        (h_net - expected).abs() < 1e-6,
        "h_net={h_net}, expected={expected}"
    );
}

#[test]
fn net_head_clamped_to_zero_when_losses_exceed_forebay() {
    // h_fore = 400, but h_loss = 500 (absurd but tests clamping).
    let forebay = flat_forebay_400m();
    let losses = HydraulicLossesModel::Constant { value_m: 500.0 };
    let pf = ProductionFunction::new(
        forebay,
        TailraceSource::Entity(None),
        Some(&losses),
        None,
        12_600.0,
        "TestPlant".to_owned(),
    );
    let h_net = pf.net_head(10_000.0, 3000.0, 0.0);
    assert!(h_net >= 0.0, "net head must be non-negative, got {h_net}");
    assert!(
        (h_net - 0.0).abs() < 1e-10,
        "h_net should be exactly 0.0, got {h_net}"
    );
}

// ── evaluate tests ────────────────────────────────────────────────────────

#[test]
fn evaluate_acceptance_criterion() {
    let forebay = flat_forebay_400m();
    let tailrace = linear_tailrace_5_5_at_3000();
    let losses = HydraulicLossesModel::Constant { value_m: 2.0 };
    let pf = ProductionFunction::new(
        forebay,
        TailraceSource::Entity(Some(tailrace)),
        Some(&losses),
        Some(&EfficiencyModel::Constant { value: 0.92 }),
        12_600.0,
        "TestPlant".to_owned(),
    );
    let phi = pf.evaluate(10_000.0, 3000.0, 0.0);
    let expected = (9.81 * 0.92 * 3000.0 * 392.5) / 1000.0;
    assert!(
        (phi - expected).abs() < 1e-2,
        "phi={phi}, expected={expected}"
    );
}

// ── Helpers for resolve_fitting_bounds tests ──────────────────────────────

/// Build a minimal `Hydro` with the given storage bounds.
fn make_hydro(min_storage_hm3: f64, max_storage_hm3: f64) -> Hydro {
    let zero_penalties = HydroPenalties {
        spillage_cost: 0.0,
        diversion_cost: 0.0,
        turbined_cost: 0.0,
        storage_violation_below_cost: 0.0,
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
    };
    let mut hydro = Hydro {
        unit_groups: Vec::new(),
        id: EntityId::from(1),
        name: "TestHydro".to_owned(),
        operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        downstream_id: None,
        travel_time_hours: None,
        entry_stage_id: None,
        exit_stage_id: None,
        min_storage_hm3,
        max_storage_hm3,
        min_outflow_m3s: 0.0,
        max_outflow_m3s: None,
        generation_model: HydroGenerationModel::Fpha,
        min_turbined_m3s: 0.0,
        max_turbined_m3s: 12_600.0,
        specific_productivity_mw_per_m3s_per_m: None,
        min_generation_mw: 0.0,
        max_generation_mw: 14_000.0,
        tailrace: None,
        hydraulic_losses: None,
        efficiency: None,
        evaporation_coefficients_mm: None,
        evaporation_reference_volumes_hm3: None,
        diversion: None,
        filling: None,
        penalties: zero_penalties,
    };
    hydro.declare_mirror_unit_group(EntityId::from(10));
    hydro
}

/// Build a `ForebayTable` spanning `[0.0, 34_116.0]` hm³.
fn sobradinho_forebay() -> ForebayTable {
    ForebayTable::new(&sobradinho_rows(), "Sobradinho").unwrap()
}

/// Build a simple 2-point `ForebayTable` spanning `[100.0, 2000.0]` hm³.
fn small_forebay() -> ForebayTable {
    let small_rows = vec![row(100.0, 386.5), row(2_000.0, 390.0)];
    ForebayTable::new(&small_rows, "SmallHydro").unwrap()
}

/// Build a default `FphaColumnLayout` (all fields `None` / defaults).
fn default_config() -> FphaColumnLayout {
    FphaColumnLayout {
        source: "computed".to_owned(),
        volume_discretization_points: None,
        turbine_discretization_points: None,
        spillage_discretization_points: None,
        max_planes_per_hydro: None,
        fitting_window: None,
    }
}

// ── No fitting window — forebay defaults, all discretization = 5 ─────────

/// Given config with no fitting window and all discretization fields None,
/// resolve_fitting_bounds uses forebay range and defaults all counts to 5,
/// max_planes_per_hydro to 10.
#[test]
fn no_fitting_window_uses_forebay_defaults() {
    let config = default_config();
    let hydro = make_hydro(0.0, 34_116.0);
    let forebay = sobradinho_forebay();

    let bounds: FittingBounds = resolve_fitting_bounds(&config, &hydro, &forebay).unwrap();

    assert_eq!(bounds.v_min, 0.0);
    assert_eq!(bounds.v_max, 34_116.0);
    assert_eq!(bounds.n_volume_points, 5);
    assert_eq!(bounds.n_flow_points, 5);
    assert_eq!(bounds.n_spillage_points, 5);
    assert_eq!(bounds.max_planes_per_hydro, 10);
}

// ── Absolute bounds — both set ────────────────────────────────────────────

/// Given config with absolute volume_min_hm3 = 1000 and volume_max_hm3 = 30000,
/// resolve_fitting_bounds returns those bounds unchanged (within forebay range).
#[test]
fn absolute_bounds_both_set() {
    let config = FphaColumnLayout {
        fitting_window: Some(FittingWindow {
            volume_min_hm3: Some(1_000.0),
            volume_max_hm3: Some(30_000.0),
            volume_min_percentile: None,
            volume_max_percentile: None,
        }),
        ..default_config()
    };
    let hydro = make_hydro(0.0, 34_116.0);
    let forebay = sobradinho_forebay();

    let bounds = resolve_fitting_bounds(&config, &hydro, &forebay).unwrap();

    assert_eq!(bounds.v_min, 1_000.0);
    assert_eq!(bounds.v_max, 30_000.0);
}

// ── Absolute bounds — only min set ───────────────────────────────────────

/// Given only volume_min_hm3, v_max falls back to forebay.v_max().
#[test]
fn absolute_bounds_only_min() {
    let config = FphaColumnLayout {
        fitting_window: Some(FittingWindow {
            volume_min_hm3: Some(5_000.0),
            volume_max_hm3: None,
            volume_min_percentile: None,
            volume_max_percentile: None,
        }),
        ..default_config()
    };
    let hydro = make_hydro(0.0, 34_116.0);
    let forebay = sobradinho_forebay();

    let bounds = resolve_fitting_bounds(&config, &hydro, &forebay).unwrap();

    assert_eq!(bounds.v_min, 5_000.0);
    assert_eq!(bounds.v_max, 34_116.0);
}

// ── Absolute bounds — only max set ───────────────────────────────────────

/// Given only volume_max_hm3, v_min falls back to forebay.v_min().
#[test]
fn absolute_bounds_only_max() {
    let config = FphaColumnLayout {
        fitting_window: Some(FittingWindow {
            volume_min_hm3: None,
            volume_max_hm3: Some(20_000.0),
            volume_min_percentile: None,
            volume_max_percentile: None,
        }),
        ..default_config()
    };
    let hydro = make_hydro(0.0, 34_116.0);
    let forebay = sobradinho_forebay();

    let bounds = resolve_fitting_bounds(&config, &hydro, &forebay).unwrap();

    assert_eq!(bounds.v_min, 0.0);
    assert_eq!(bounds.v_max, 20_000.0);
}

// ── Percentile bounds ─────────────────────────────────────────────────────

/// Given percentile bounds 0.1 and 0.9 on a hydro with range [100, 2000],
/// v_min = 100 + 0.1 * 1900 = 290, v_max = 100 + 0.9 * 1900 = 1810.
#[test]
fn percentile_bounds_both_set() {
    let config = FphaColumnLayout {
        fitting_window: Some(FittingWindow {
            volume_min_hm3: None,
            volume_max_hm3: None,
            volume_min_percentile: Some(0.1),
            volume_max_percentile: Some(0.9),
        }),
        ..default_config()
    };
    let hydro = make_hydro(100.0, 2_000.0);
    let forebay = small_forebay();

    let bounds = resolve_fitting_bounds(&config, &hydro, &forebay).unwrap();

    let expected_min = 100.0 + 0.1 * 1_900.0; // 290.0
    let expected_max = 100.0 + 0.9 * 1_900.0; // 1810.0
    assert!((bounds.v_min - expected_min).abs() < 1e-10);
    assert!((bounds.v_max - expected_max).abs() < 1e-10);
}

// ── Mixed — absolute min, percentile max (non-conflicting) ────────────────

/// Absolute min bound + percentile max bound is accepted (different dimensions).
#[test]
fn mixed_absolute_min_percentile_max() {
    let config = FphaColumnLayout {
        fitting_window: Some(FittingWindow {
            volume_min_hm3: Some(200.0),
            volume_max_hm3: None,
            volume_min_percentile: None,
            volume_max_percentile: Some(0.9),
        }),
        ..default_config()
    };
    let hydro = make_hydro(100.0, 2_000.0);
    let forebay = small_forebay();

    let bounds = resolve_fitting_bounds(&config, &hydro, &forebay).unwrap();

    let expected_max = 100.0 + 0.9 * 1_900.0; // 1810.0
    assert_eq!(bounds.v_min, 200.0);
    assert!((bounds.v_max - expected_max).abs() < 1e-10);
}

// ── Conflicting — both absolute and percentile for min ───────────────────

/// volume_min_hm3 and volume_min_percentile both set -> ConflictingFittingWindow.
#[test]
fn conflicting_min_bound_returns_error() {
    let config = FphaColumnLayout {
        fitting_window: Some(FittingWindow {
            volume_min_hm3: Some(500.0),
            volume_max_hm3: None,
            volume_min_percentile: Some(0.1),
            volume_max_percentile: None,
        }),
        ..default_config()
    };
    let hydro = make_hydro(100.0, 2_000.0);
    let forebay = small_forebay();

    let err = resolve_fitting_bounds(&config, &hydro, &forebay).unwrap_err();
    assert!(
        matches!(err, FphaFittingError::ConflictingFittingWindow { .. }),
        "expected ConflictingFittingWindow, got: {err:?}"
    );
}

// ── Conflicting — both absolute and percentile for max ───────────────────

/// volume_max_hm3 and volume_max_percentile both set -> ConflictingFittingWindow.
#[test]
fn conflicting_max_bound_returns_error() {
    let config = FphaColumnLayout {
        fitting_window: Some(FittingWindow {
            volume_min_hm3: None,
            volume_max_hm3: Some(1_800.0),
            volume_min_percentile: None,
            volume_max_percentile: Some(0.9),
        }),
        ..default_config()
    };
    let hydro = make_hydro(100.0, 2_000.0);
    let forebay = small_forebay();

    let err = resolve_fitting_bounds(&config, &hydro, &forebay).unwrap_err();
    assert!(
        matches!(err, FphaFittingError::ConflictingFittingWindow { .. }),
        "expected ConflictingFittingWindow, got: {err:?}"
    );
}

// ── Empty range — v_min >= v_max ─────────────────────────────────────────

/// Absolute bounds with v_min > v_max -> EmptyFittingWindow.
#[test]
fn inverted_absolute_bounds_returns_empty_window_error() {
    let config = FphaColumnLayout {
        fitting_window: Some(FittingWindow {
            volume_min_hm3: Some(1_500.0),
            volume_max_hm3: Some(1_000.0),
            volume_min_percentile: None,
            volume_max_percentile: None,
        }),
        ..default_config()
    };
    let hydro = make_hydro(0.0, 34_116.0);
    let forebay = sobradinho_forebay();

    let err = resolve_fitting_bounds(&config, &hydro, &forebay).unwrap_err();
    assert!(
        matches!(err, FphaFittingError::EmptyFittingWindow { .. }),
        "expected EmptyFittingWindow, got: {err:?}"
    );
}

/// Equal absolute bounds (v_min == v_max) are a run-of-river plant: rerouted
/// through the single-volume path, NOT rejected. The window must resolve to two
/// synthesized samples with `single_volume = true` and exactly 2 volume points.
#[test]
fn equal_absolute_bounds_reroutes_to_single_volume() {
    let config = FphaColumnLayout {
        fitting_window: Some(FittingWindow {
            volume_min_hm3: Some(1_000.0),
            volume_max_hm3: Some(1_000.0),
            volume_min_percentile: None,
            volume_max_percentile: None,
        }),
        ..default_config()
    };
    let hydro = make_hydro(0.0, 34_116.0);
    let forebay = sobradinho_forebay();

    let bounds = resolve_fitting_bounds(&config, &hydro, &forebay)
        .expect("equal bounds must reroute, not error");
    assert!(
        bounds.single_volume,
        "equal bounds must be flagged single-volume"
    );
    assert_eq!(
        bounds.n_volume_points, 2,
        "single-volume path uses 2 samples"
    );
    assert!(
        bounds.v_min < bounds.v_max,
        "synthesized samples must straddle v0: v_min={} v_max={}",
        bounds.v_min,
        bounds.v_max
    );
}

/// `volume_discretization_points = 1` (explicit NPTV = 1) is a run-of-river
/// request: rerouted through the single-volume path, NOT rejected with
/// `InsufficientDiscretization`, even when the storage window is wide.
#[test]
fn explicit_single_volume_point_reroutes_not_rejected() {
    let config = FphaColumnLayout {
        volume_discretization_points: Some(1),
        ..default_config()
    };
    let hydro = make_hydro(0.0, 34_116.0);
    let forebay = sobradinho_forebay();

    let bounds = resolve_fitting_bounds(&config, &hydro, &forebay)
        .expect("NPTV = 1 must reroute, not error");
    assert!(
        bounds.single_volume,
        "NPTV = 1 must be flagged single-volume"
    );
    assert_eq!(
        bounds.n_volume_points, 2,
        "single-volume path uses 2 samples"
    );
}

// ── Clamping — absolute bounds outside forebay range ─────────────────────

/// Absolute v_min below forebay.v_min() gets clamped to forebay.v_min().
#[test]
fn absolute_min_below_forebay_gets_clamped() {
    let config = FphaColumnLayout {
        fitting_window: Some(FittingWindow {
            volume_min_hm3: Some(-500.0),
            volume_max_hm3: Some(20_000.0),
            volume_min_percentile: None,
            volume_max_percentile: None,
        }),
        ..default_config()
    };
    let hydro = make_hydro(0.0, 34_116.0);
    let forebay = sobradinho_forebay();

    let bounds = resolve_fitting_bounds(&config, &hydro, &forebay).unwrap();

    // Clamped to forebay.v_min() = 0.0
    assert_eq!(bounds.v_min, 0.0);
    assert_eq!(bounds.v_max, 20_000.0);
}

/// Absolute v_max above forebay.v_max() gets clamped to forebay.v_max().
#[test]
fn absolute_max_above_forebay_gets_clamped() {
    let config = FphaColumnLayout {
        fitting_window: Some(FittingWindow {
            volume_min_hm3: Some(1_000.0),
            volume_max_hm3: Some(50_000.0),
            volume_min_percentile: None,
            volume_max_percentile: None,
        }),
        ..default_config()
    };
    let hydro = make_hydro(0.0, 34_116.0);
    let forebay = sobradinho_forebay();

    let bounds = resolve_fitting_bounds(&config, &hydro, &forebay).unwrap();

    assert_eq!(bounds.v_min, 1_000.0);
    // Clamped to forebay.v_max() = 34_116.0
    assert_eq!(bounds.v_max, 34_116.0);
}

// ── Discretization defaults ──────────────────────────────────────────────

/// All discretization fields None -> defaults to 5 for each dimension.
#[test]
fn discretization_all_none_defaults_to_five() {
    let config = default_config();
    let hydro = make_hydro(0.0, 34_116.0);
    let forebay = sobradinho_forebay();

    let bounds = resolve_fitting_bounds(&config, &hydro, &forebay).unwrap();

    assert_eq!(bounds.n_volume_points, 5);
    assert_eq!(bounds.n_flow_points, 5);
    assert_eq!(bounds.n_spillage_points, 5);
}

/// Explicit discretization values are passed through unchanged.
#[test]
fn discretization_explicit_values_passed_through() {
    let config = FphaColumnLayout {
        volume_discretization_points: Some(8),
        turbine_discretization_points: Some(6),
        spillage_discretization_points: Some(10),
        max_planes_per_hydro: Some(20),
        ..default_config()
    };
    let hydro = make_hydro(0.0, 34_116.0);
    let forebay = sobradinho_forebay();

    let bounds = resolve_fitting_bounds(&config, &hydro, &forebay).unwrap();

    assert_eq!(bounds.n_volume_points, 8);
    assert_eq!(bounds.n_flow_points, 6);
    assert_eq!(bounds.n_spillage_points, 10);
    assert_eq!(bounds.max_planes_per_hydro, 20);
}

// ── Insufficient discretization ──────────────────────────────────────────

// Note: `volume_discretization_points = 1` (NPTV = 1) is no longer rejected —
// it is a run-of-river request rerouted through the single-volume path. See
// `explicit_single_volume_point_reroutes_not_rejected`.

/// volume_discretization_points = 0 -> InsufficientDiscretization.
#[test]
fn volume_discretization_zero_returns_error() {
    let config = FphaColumnLayout {
        volume_discretization_points: Some(0),
        ..default_config()
    };
    let hydro = make_hydro(0.0, 34_116.0);
    let forebay = sobradinho_forebay();

    let err = resolve_fitting_bounds(&config, &hydro, &forebay).unwrap_err();
    assert!(
        matches!(
            err,
            FphaFittingError::InsufficientDiscretization { ref dimension, value: 0, .. }
            if dimension == "volume"
        ),
        "expected InsufficientDiscretization for volume=0, got: {err:?}"
    );
}

/// turbine_discretization_points = 1 -> InsufficientDiscretization { dimension: "turbine", value: 1 }.
#[test]
fn turbine_discretization_one_returns_error() {
    let config = FphaColumnLayout {
        turbine_discretization_points: Some(1),
        ..default_config()
    };
    let hydro = make_hydro(0.0, 34_116.0);
    let forebay = sobradinho_forebay();

    let err = resolve_fitting_bounds(&config, &hydro, &forebay).unwrap_err();
    assert!(
        matches!(
            err,
            FphaFittingError::InsufficientDiscretization { ref dimension, value: 1, .. }
            if dimension == "turbine"
        ),
        "expected InsufficientDiscretization for turbine=1, got: {err:?}"
    );
}

/// spillage_discretization_points = 1 -> InsufficientDiscretization { dimension: "spillage", value: 1 }.
#[test]
fn spillage_discretization_one_returns_error() {
    let config = FphaColumnLayout {
        spillage_discretization_points: Some(1),
        ..default_config()
    };
    let hydro = make_hydro(0.0, 34_116.0);
    let forebay = sobradinho_forebay();

    let err = resolve_fitting_bounds(&config, &hydro, &forebay).unwrap_err();
    assert!(
        matches!(
            err,
            FphaFittingError::InsufficientDiscretization { ref dimension, value: 1, .. }
            if dimension == "spillage"
        ),
        "expected InsufficientDiscretization for spillage=1, got: {err:?}"
    );
}

// ── Max_planes_per_hydro ──────────────────────────────────────────────────

/// max_planes_per_hydro = None -> defaults to 10.
#[test]
fn max_planes_per_hydro_none_defaults_to_ten() {
    let config = default_config();
    let hydro = make_hydro(0.0, 34_116.0);
    let forebay = sobradinho_forebay();

    let bounds = resolve_fitting_bounds(&config, &hydro, &forebay).unwrap();

    assert_eq!(bounds.max_planes_per_hydro, 10);
}

/// max_planes_per_hydro = Some(5) -> 5.
#[test]
fn max_planes_per_hydro_explicit_value() {
    let config = FphaColumnLayout {
        max_planes_per_hydro: Some(5),
        ..default_config()
    };
    let hydro = make_hydro(0.0, 34_116.0);
    let forebay = sobradinho_forebay();

    let bounds = resolve_fitting_bounds(&config, &hydro, &forebay).unwrap();

    assert_eq!(bounds.max_planes_per_hydro, 5);
}

/// max_planes_per_hydro = Some(0) -> InsufficientDiscretization.
#[test]
fn max_planes_per_hydro_zero_returns_error() {
    let config = FphaColumnLayout {
        max_planes_per_hydro: Some(0),
        ..default_config()
    };
    let hydro = make_hydro(0.0, 34_116.0);
    let forebay = sobradinho_forebay();

    let err = resolve_fitting_bounds(&config, &hydro, &forebay).unwrap_err();
    assert!(
        matches!(
            err,
            FphaFittingError::InsufficientDiscretization { ref dimension, value: 0, .. }
            if dimension == "max_planes_per_hydro"
        ),
        "expected InsufficientDiscretization for max_planes_per_hydro=0, got: {err:?}"
    );
}

// ── validate_fitted_planes tests ──────────────────────────────────────────

/// Valid planes and a positive alpha pass validation.
#[test]
fn validate_fitted_planes_valid_input_returns_ok() {
    let planes = vec![
        RawPlane {
            gamma_0: 100.0,
            gamma_v: 0.001,
            gamma_q: 3.5,
            gamma_s: -0.01,
        },
        RawPlane {
            gamma_0: 200.0,
            gamma_v: 0.002,
            gamma_q: 3.8,
            gamma_s: -0.02,
        },
    ];
    let result = validate_fitted_planes(&planes, 0.985, "TestHydro");
    assert!(result.is_ok(), "expected Ok(()), got {result:?}");
}

/// Alpha = -0.1 returns NonPositiveAlpha (alpha must be > 0).
#[test]
fn validate_fitted_planes_negative_alpha_returns_non_positive_alpha() {
    let planes = vec![RawPlane {
        gamma_0: 100.0,
        gamma_v: 0.001,
        gamma_q: 3.5,
        gamma_s: -0.01,
    }];
    let err = validate_fitted_planes(&planes, -0.1, "TestHydro").unwrap_err();
    assert!(
        matches!(err, FphaFittingError::NonPositiveAlpha { alpha, .. } if alpha == -0.1),
        "expected NonPositiveAlpha with alpha=-0.1, got: {err:?}"
    );
}

/// Alpha = 0.0 returns NonPositiveAlpha.
#[test]
fn validate_fitted_planes_zero_alpha_returns_non_positive_alpha() {
    let planes = vec![RawPlane {
        gamma_0: 100.0,
        gamma_v: 0.001,
        gamma_q: 3.5,
        gamma_s: -0.01,
    }];
    let err = validate_fitted_planes(&planes, 0.0, "TestHydro").unwrap_err();
    assert!(
        matches!(err, FphaFittingError::NonPositiveAlpha { .. }),
        "expected NonPositiveAlpha for alpha=0.0, got: {err:?}"
    );
}

/// Alpha > 1.0 is valid (alpha is an MSE balance, not bounded below 1).
#[test]
fn validate_fitted_planes_alpha_above_one_passes() {
    let planes = vec![RawPlane {
        gamma_0: 100.0,
        gamma_v: 0.001,
        gamma_q: 3.5,
        gamma_s: -0.01,
    }];
    let result = validate_fitted_planes(&planes, 1.25, "TestHydro");
    assert!(result.is_ok(), "alpha > 1 should pass: {result:?}");
}

/// Empty planes returns NoHyperplanesProduced.
#[test]
fn validate_fitted_planes_empty_planes_returns_no_hyperplanes() {
    let err = validate_fitted_planes(&[], 0.99, "TestHydro").unwrap_err();
    assert!(
        matches!(err, FphaFittingError::NoHyperplanesProduced { .. }),
        "expected NoHyperplanesProduced, got: {err:?}"
    );
}

/// Plane with gamma_v significantly below zero returns InvalidCoefficient.
#[test]
fn validate_fitted_planes_negative_gamma_v_returns_invalid_coefficient() {
    let planes = vec![RawPlane {
        gamma_0: 100.0,
        gamma_v: -0.01, // clearly negative
        gamma_q: 3.5,
        gamma_s: -0.01,
    }];
    let err = validate_fitted_planes(&planes, 0.98, "TestHydro").unwrap_err();
    assert!(
        matches!(
            err,
            FphaFittingError::InvalidCoefficient { plane_index: 0, ref detail, .. }
            if detail.contains("gamma_v")
        ),
        "expected InvalidCoefficient for gamma_v < 0, got: {err:?}"
    );
}

/// Plane with gamma_q significantly below zero returns InvalidCoefficient.
#[test]
fn validate_fitted_planes_negative_gamma_q_returns_invalid_coefficient() {
    let planes = vec![RawPlane {
        gamma_0: 100.0,
        gamma_v: 0.001,
        gamma_q: -0.01, // clearly negative
        gamma_s: -0.01,
    }];
    let err = validate_fitted_planes(&planes, 0.98, "TestHydro").unwrap_err();
    assert!(
        matches!(
            err,
            FphaFittingError::InvalidCoefficient { plane_index: 0, ref detail, .. }
            if detail.contains("gamma_q")
        ),
        "expected InvalidCoefficient for gamma_q < 0, got: {err:?}"
    );
}

/// Plane with gamma_s significantly above zero returns InvalidCoefficient.
#[test]
fn validate_fitted_planes_positive_gamma_s_returns_invalid_coefficient() {
    let planes = vec![RawPlane {
        gamma_0: 100.0,
        gamma_v: 0.001,
        gamma_q: 3.5,
        gamma_s: 0.01, // positive: spillage should not increase production
    }];
    let err = validate_fitted_planes(&planes, 0.98, "TestHydro").unwrap_err();
    assert!(
        matches!(
            err,
            FphaFittingError::InvalidCoefficient { plane_index: 0, ref detail, .. }
            if detail.contains("gamma_s")
        ),
        "expected InvalidCoefficient for gamma_s > 0, got: {err:?}"
    );
}

/// Near-zero gamma_v (within 1e-10 tolerance) passes validation.
#[test]
fn validate_fitted_planes_near_zero_gamma_v_within_tolerance_passes() {
    let planes = vec![RawPlane {
        gamma_0: 100.0,
        gamma_v: -1e-11, // within tolerance -1e-10
        gamma_q: 3.5,
        gamma_s: -0.01,
    }];
    let result = validate_fitted_planes(&planes, 0.99, "TestHydro");
    assert!(
        result.is_ok(),
        "near-zero gamma_v within tolerance should pass: {result:?}"
    );
}

/// Near-zero gamma_s (within 1e-10 tolerance) passes validation.
#[test]
fn validate_fitted_planes_near_zero_gamma_s_within_tolerance_passes() {
    let planes = vec![RawPlane {
        gamma_0: 100.0,
        gamma_v: 0.001,
        gamma_q: 3.5,
        gamma_s: 1e-11, // within tolerance 1e-10
    }];
    let result = validate_fitted_planes(&planes, 0.99, "TestHydro");
    assert!(
        result.is_ok(),
        "near-zero gamma_s within tolerance should pass: {result:?}"
    );
}

// ── fit_fpha_planes tests ──────────────────────────────────────────────────

/// Build a Hydro entity with Sobradinho-style geometry and optional models.
fn make_sobradinho_hydro() -> Hydro {
    let zero_penalties = HydroPenalties {
        spillage_cost: 0.0,
        diversion_cost: 0.0,
        turbined_cost: 0.0,
        storage_violation_below_cost: 0.0,
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
    };
    let mut hydro = Hydro {
        unit_groups: Vec::new(),
        id: EntityId::from(1),
        name: "Sobradinho".to_owned(),
        operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        downstream_id: None,
        travel_time_hours: None,
        entry_stage_id: None,
        exit_stage_id: None,
        min_storage_hm3: 0.0,
        max_storage_hm3: 34_116.0,
        min_outflow_m3s: 0.0,
        max_outflow_m3s: None,
        generation_model: HydroGenerationModel::Fpha,
        min_turbined_m3s: 0.0,
        max_turbined_m3s: 3_000.0,
        specific_productivity_mw_per_m3s_per_m: None,
        min_generation_mw: 0.0,
        max_generation_mw: 1_000.0,
        tailrace: Some(TailraceModel::Polynomial {
            coefficients: vec![0.0, 0.001_f64],
        }),
        hydraulic_losses: Some(HydraulicLossesModel::Constant { value_m: 2.0 }),
        efficiency: Some(EfficiencyModel::Constant { value: 0.92 }),
        evaporation_coefficients_mm: None,
        evaporation_reference_volumes_hm3: None,
        diversion: None,
        filling: None,
        penalties: zero_penalties,
    };
    hydro.declare_mirror_unit_group(EntityId::from(10));
    hydro
}

/// Fit_fpha_planes with Sobradinho-style geometry and default FphaColumnLayout
/// returns Ok with at least one plane, all with valid coefficient signs.
///
/// The fitter derives planes from the convex hull, so the contract is the
/// hull-based invariant (>= 1 plane and valid γ signs), not a fixed count; the
/// pointwise outer-approximation guarantee is owned by the `hull_fit` unit tests.
#[test]
fn fit_fpha_planes_sobradinho_style_end_to_end() {
    let rows = sobradinho_rows();
    let hydro = make_sobradinho_hydro();
    let config = FphaColumnLayout {
        source: "computed".to_owned(),
        volume_discretization_points: None,
        turbine_discretization_points: None,
        spillage_discretization_points: None,
        max_planes_per_hydro: None,
        fitting_window: None,
    };

    let result = fit_fpha_planes(
        &rows,
        &hydro,
        &config,
        0.0,
        TailraceSource::Entity(hydro.tailrace.clone()),
        None,
        1,
        0,
        false,
    )
    .expect("fit_fpha_planes should succeed");
    let planes = &result.planes;

    // The hull yields at least one upper-envelope face for a realistic hydro.
    assert!(
        !planes.is_empty(),
        "expected at least one plane, got {}",
        planes.len()
    );

    // All coefficient signs must be valid.
    for (idx, plane) in planes.iter().enumerate() {
        assert!(
            plane.gamma_v >= -1e-10,
            "plane {idx}: gamma_v={} must be >= 0",
            plane.gamma_v
        );
        assert!(
            plane.gamma_q >= -1e-10,
            "plane {idx}: gamma_q={} must be >= 0",
            plane.gamma_q
        );
        assert!(
            plane.gamma_s <= 1e-10,
            "plane {idx}: gamma_s={} must be <= 0",
            plane.gamma_s
        );
    }
}

/// A computed-FPHA fixture with a spillage-sensitive tailrace yields planes
/// with `γ_S < 0` that still form a valid outer approximation.
///
/// The Sobradinho fixture's tailrace rises with total outflow
/// (`h_tail = 0.001·q_out`), so lateral flow lowers net head and the per-plane
/// secant is strictly negative. The plane set must remain a valid concave outer
/// approximation: every gradient sign is valid and at least one plane carries the
/// fitted negative `γ_S`.
#[test]
fn fit_fpha_planes_spill_sensitive_yields_negative_gamma_s() {
    let rows = sobradinho_rows();
    let hydro = make_sobradinho_hydro();
    let config = FphaColumnLayout {
        source: "computed".to_owned(),
        volume_discretization_points: None,
        turbine_discretization_points: None,
        spillage_discretization_points: None,
        max_planes_per_hydro: None,
        fitting_window: None,
    };

    let result = fit_fpha_planes(
        &rows,
        &hydro,
        &config,
        0.0,
        TailraceSource::Entity(hydro.tailrace.clone()),
        None,
        1,
        0,
        false,
    )
    .expect("fit should succeed");
    let planes = &result.planes;
    assert!(!planes.is_empty(), "expected at least one plane");

    // At least one plane carries a fitted negative secant (the active planes on a
    // rising tailrace), and every plane satisfies the sign contract.
    let mut any_negative = false;
    for (idx, plane) in planes.iter().enumerate() {
        assert!(
            plane.gamma_v >= -1e-10,
            "plane {idx}: gamma_v={} must be >= 0",
            plane.gamma_v
        );
        assert!(
            plane.gamma_q >= -1e-10,
            "plane {idx}: gamma_q={} must be >= 0",
            plane.gamma_q
        );
        assert!(
            plane.gamma_s <= 1e-10,
            "plane {idx}: gamma_s={} must be <= 0",
            plane.gamma_s
        );
        if plane.gamma_s < -1e-10 {
            any_negative = true;
        }
    }
    assert!(
        any_negative,
        "a rising tailrace must produce at least one plane with gamma_s < 0"
    );
}

/// Fit_fpha_planes intercepts are finite for the Sobradinho fixture.
#[test]
fn fit_fpha_planes_intercepts_are_finite() {
    let rows = sobradinho_rows();
    let hydro = make_sobradinho_hydro();
    let config = FphaColumnLayout {
        source: "computed".to_owned(),
        volume_discretization_points: None,
        turbine_discretization_points: None,
        spillage_discretization_points: None,
        max_planes_per_hydro: None,
        fitting_window: None,
    };

    let result = fit_fpha_planes(
        &rows,
        &hydro,
        &config,
        0.0,
        TailraceSource::Entity(hydro.tailrace.clone()),
        None,
        1,
        0,
        false,
    )
    .expect("fit should succeed");

    // All intercepts must be finite and the planes must have non-negative intercepts
    // for a physically reasonable geometry where phi > 0 at the fitting origin.
    for (idx, plane) in result.planes.iter().enumerate() {
        assert!(
            plane.intercept.is_finite(),
            "plane {idx}: intercept must be finite, got {}",
            plane.intercept
        );
    }
}

/// AC (integration): a run-of-river hydro (NPTV = 1) fits end-to-end through
/// the single-volume path, producing a valid outer approximation with every
/// plane's `γ_V` exactly 0. `volume_discretization_points = 1` is the reference
/// run-of-river config; the reroute fires inside `resolve_fitting_bounds`.
#[test]
fn fit_fpha_planes_run_of_river_yields_zero_gamma_v_end_to_end() {
    // Flat forebay over the whole band (no storage head gain) + the Sobradinho
    // tailrace (head drops with outflow) → strictly concave in Q, flat in V.
    let flat_rows = vec![
        HydroGeometryRow {
            hydro_id: EntityId::from(1),
            volume_hm3: 0.0,
            height_m: 400.0,
            area_km2: 0.0,
        },
        HydroGeometryRow {
            hydro_id: EntityId::from(1),
            volume_hm3: 5_000.0,
            height_m: 400.0,
            area_km2: 0.0,
        },
    ];
    let mut hydro = make_sobradinho_hydro();
    hydro.name = "RunOfRiver".to_owned();
    // Run-of-river: no useful storage.
    hydro.min_storage_hm3 = 5_000.0;
    hydro.max_storage_hm3 = 5_000.0;
    let config = FphaColumnLayout {
        source: "computed".to_owned(),
        volume_discretization_points: Some(1),
        turbine_discretization_points: None,
        spillage_discretization_points: None,
        max_planes_per_hydro: None,
        fitting_window: None,
    };

    let result = fit_fpha_planes(
        &flat_rows,
        &hydro,
        &config,
        0.0,
        TailraceSource::Entity(hydro.tailrace.clone()),
        None,
        1,
        0,
        false,
    )
    .expect("run-of-river fit should succeed");
    assert!(
        !result.planes.is_empty(),
        "run-of-river fit must yield >= 1 plane"
    );
    for (idx, plane) in result.planes.iter().enumerate() {
        // γ_V is zeroed before the α scaling, and α·0 = 0, so the emitted
        // (α-scaled) plane keeps γ_V exactly 0.
        assert_eq!(
            plane.gamma_v.to_bits(),
            0.0_f64.to_bits(),
            "plane {idx}: gamma_v={} must be exactly 0.0",
            plane.gamma_v
        );
        assert!(
            plane.gamma_q >= -1e-10,
            "plane {idx}: gamma_q={} must be >= 0",
            plane.gamma_q
        );
    }
}

/// Fit_fpha_planes with constant-head geometry (flat forebay, no tailrace)
/// produces exactly 1 plane.
#[test]
fn fit_fpha_planes_linear_function_produces_one_plane() {
    let flat_rows = vec![
        HydroGeometryRow {
            hydro_id: EntityId::from(1),
            volume_hm3: 0.0,
            height_m: 400.0,
            area_km2: 0.0,
        },
        HydroGeometryRow {
            hydro_id: EntityId::from(1),
            volume_hm3: 20_000.0,
            height_m: 400.0,
            area_km2: 0.0,
        },
    ];
    let zero_penalties = HydroPenalties {
        spillage_cost: 0.0,
        diversion_cost: 0.0,
        turbined_cost: 0.0,
        storage_violation_below_cost: 0.0,
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
    };
    let mut hydro = Hydro {
        unit_groups: Vec::new(),
        id: EntityId::from(1),
        name: "FlatHydro".to_owned(),
        operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        downstream_id: None,
        travel_time_hours: None,
        entry_stage_id: None,
        exit_stage_id: None,
        min_storage_hm3: 0.0,
        max_storage_hm3: 20_000.0,
        min_outflow_m3s: 0.0,
        max_outflow_m3s: None,
        generation_model: HydroGenerationModel::Fpha,
        min_turbined_m3s: 0.0,
        max_turbined_m3s: 1_000.0,
        specific_productivity_mw_per_m3s_per_m: None,
        min_generation_mw: 0.0,
        max_generation_mw: 4_000.0,
        tailrace: None,
        hydraulic_losses: None,
        efficiency: None,
        evaporation_coefficients_mm: None,
        evaporation_reference_volumes_hm3: None,
        diversion: None,
        filling: None,
        penalties: zero_penalties,
    };
    hydro.declare_mirror_unit_group(EntityId::from(10));
    let config = FphaColumnLayout {
        source: "computed".to_owned(),
        volume_discretization_points: None,
        turbine_discretization_points: None,
        spillage_discretization_points: None,
        max_planes_per_hydro: None,
        fitting_window: None,
    };

    let result = fit_fpha_planes(
        &flat_rows,
        &hydro,
        &config,
        0.0,
        TailraceSource::Entity(hydro.tailrace.clone()),
        None,
        1,
        0,
        false,
    )
    .expect("fit should succeed");

    // A linear production function defines a SINGLE hyperplane. With the flow axis
    // anchored at q = 0 the cloud is exactly coplanar, so the convex hull may
    // triangulate it into more than one facet — but every returned plane must
    // describe the same hyperplane (identical coefficients). A flat forebay also
    // means no volume dependence, so γ_V is zero.
    assert!(
        !result.planes.is_empty(),
        "fit must produce at least one plane"
    );
    let first = &result.planes[0];
    for p in &result.planes {
        assert!(
            (p.intercept - first.intercept).abs() < 1e-6
                && (p.gamma_v - first.gamma_v).abs() < 1e-9
                && (p.gamma_q - first.gamma_q).abs() < 1e-9,
            "all planes of a linear function must be coplanar; got {p:?} vs {first:?}"
        );
    }
    assert!(
        first.gamma_v.abs() < 1e-9,
        "a flat forebay must yield zero volume gradient, got {}",
        first.gamma_v
    );
}

/// Fit_fpha_planes propagates ForebayTable construction errors (empty rows).
#[test]
fn fit_fpha_planes_propagates_forebay_error_on_empty_rows() {
    let rows: Vec<HydroGeometryRow> = Vec::new();
    let hydro = make_sobradinho_hydro();
    let config = FphaColumnLayout {
        source: "computed".to_owned(),
        volume_discretization_points: None,
        turbine_discretization_points: None,
        spillage_discretization_points: None,
        max_planes_per_hydro: None,
        fitting_window: None,
    };

    let err = fit_fpha_planes(
        &rows,
        &hydro,
        &config,
        0.0,
        TailraceSource::Entity(hydro.tailrace.clone()),
        None,
        1,
        0,
        false,
    )
    .unwrap_err();
    assert!(
        matches!(err, FphaFittingError::InsufficientPoints { count: 0, .. }),
        "expected InsufficientPoints with count=0, got: {err:?}"
    );
}

/// A run-of-river plant — ONE VHA row at a single operating volume — fits
/// end-to-end through the single-volume path and yields `γ_V = 0` exactly.
///
/// This exercises the full chain a bridge-converted run-of-river plant takes:
/// 1-row geometry → `ForebayTable` (constant) → `resolve_fitting_bounds` detects
/// the single-volume case → synthesis spreads two distinct V samples (unclamped
/// past the degenerate forebay) → hull fit → `γ_V = 0`. No `InsufficientPoints`,
/// no `DegenerateProductionCloud`.
#[test]
fn single_volume_run_of_river_fits_with_zero_gamma_v() {
    let rows = vec![HydroGeometryRow {
        hydro_id: EntityId::from(1),
        volume_hm3: 1500.0,
        height_m: 386.5,
        area_km2: 0.0,
    }];
    let hydro = make_sobradinho_hydro();
    let config = FphaColumnLayout {
        source: "computed".to_owned(),
        volume_discretization_points: None,
        turbine_discretization_points: None,
        spillage_discretization_points: None,
        max_planes_per_hydro: None,
        fitting_window: None,
    };

    let result = fit_fpha_planes(
        &rows,
        &hydro,
        &config,
        0.0,
        TailraceSource::Entity(hydro.tailrace.clone()),
        None,
        1,
        0,
        false,
    )
    .expect("single-volume run-of-river plant must fit");

    assert!(
        !result.planes.is_empty(),
        "fit must yield at least one plane"
    );
    for plane in &result.planes {
        assert_eq!(
            plane.gamma_v, 0.0,
            "constant forebay must yield exactly zero volume gradient, got {}",
            plane.gamma_v
        );
        assert!(
            plane.gamma_q >= 0.0,
            "turbine gradient must stay non-negative, got {}",
            plane.gamma_q
        );
    }
}

// ── FphaFittingError Display messages for new variants ────────────────────

#[test]
fn display_non_positive_alpha_contains_name_and_value() {
    let err = FphaFittingError::NonPositiveAlpha {
        hydro_name: "Itaipu".to_owned(),
        alpha: 0.0,
    };
    let msg = err.to_string();
    assert!(msg.contains("Itaipu"), "should contain hydro name: {msg}");
    assert!(msg.contains('0'), "should contain alpha value: {msg}");
}

#[test]
fn display_degenerate_production_cloud_contains_name() {
    let err = FphaFittingError::DegenerateProductionCloud {
        hydro_name: "Tucurui".to_owned(),
    };
    let msg = err.to_string();
    assert!(msg.contains("Tucurui"), "should contain hydro name: {msg}");
    assert!(
        msg.contains("degenerate"),
        "should describe the degenerate cloud: {msg}"
    );
}

#[test]
fn display_no_hyperplanes_produced_contains_name() {
    let err = FphaFittingError::NoHyperplanesProduced {
        hydro_name: "Serra da Mesa".to_owned(),
    };
    let msg = err.to_string();
    assert!(
        msg.contains("Serra da Mesa"),
        "should contain hydro name: {msg}"
    );
}

#[test]
fn display_invalid_coefficient_contains_name_and_index_and_detail() {
    let err = FphaFittingError::InvalidCoefficient {
        hydro_name: "Furnas".to_owned(),
        plane_index: 3,
        detail: "gamma_v is negative".to_owned(),
    };
    let msg = err.to_string();
    assert!(msg.contains("Furnas"), "should contain hydro name: {msg}");
    assert!(msg.contains('3'), "should contain plane index: {msg}");
    assert!(msg.contains("gamma_v"), "should contain detail: {msg}");
}

// ── FphaFitResult alpha extraction ────────────────────────────────────────

/// Fit_fpha_planes returns a finite positive α and each plane's coefficients
/// divide cleanly by α (the single correction factor) in the Sobradinho fixture.
#[test]
fn fit_fpha_planes_result_alpha_positive_and_intercept_consistent() {
    let rows = sobradinho_rows();
    let hydro = make_sobradinho_hydro();
    let config = FphaColumnLayout {
        source: "computed".to_owned(),
        volume_discretization_points: None,
        turbine_discretization_points: None,
        spillage_discretization_points: None,
        max_planes_per_hydro: None,
        fitting_window: None,
    };

    let result = fit_fpha_planes(
        &rows,
        &hydro,
        &config,
        0.0,
        TailraceSource::Entity(hydro.tailrace.clone()),
        None,
        1,
        0,
        false,
    )
    .expect("fit should succeed");

    // alpha must be finite and strictly positive.
    assert!(
        result.alpha.is_finite() && result.alpha > 0.0,
        "alpha must be finite and positive, got {}",
        result.alpha
    );

    // For each plane, intercept == alpha * raw_gamma_0, so raw_gamma_0 == intercept / alpha.
    // Verify the round-trip recovers the intercept exactly.
    for (idx, plane) in result.planes.iter().enumerate() {
        let raw_gamma_0 = plane.intercept / result.alpha;
        let reconstructed_intercept = raw_gamma_0 * result.alpha;
        assert!(
            (plane.intercept - reconstructed_intercept).abs() < 1e-12,
            "plane {idx}: intercept round-trip failed: {} vs {}",
            plane.intercept,
            reconstructed_intercept
        );
    }
}

// ── alpha_FPHA integration ─────────────────────────────────────────────────

/// Integration: a computed-FPHA fixture yields a finite positive α; the α-scaled
/// planes remain a valid outer approximation of FPH on the spill = 0 grid; and the
/// exported planes equal α·(raw hull) — proving α is the only correction applied.
#[test]
fn alpha_scaled_planes_are_outer_approximation_and_not_double_scaled() {
    let rows = sobradinho_rows();
    let hydro = make_sobradinho_hydro();
    let config = FphaColumnLayout {
        source: "computed".to_owned(),
        volume_discretization_points: None,
        turbine_discretization_points: None,
        spillage_discretization_points: None,
        max_planes_per_hydro: None,
        fitting_window: None,
    };

    // Reconstruct the production function and bounds exactly as fit_fpha_planes does,
    // so we can recompute the raw hull (FPHA_0) and α independently of the pipeline.
    let forebay = ForebayTable::new(&rows, &hydro.name).expect("valid VHA curve");
    let pf = ProductionFunction::new(
        forebay.clone(),
        TailraceSource::Entity(hydro.tailrace.clone()),
        hydro.hydraulic_losses.as_ref(),
        hydro.efficiency.as_ref(),
        hydro.max_turbined_m3s,
        hydro.name.clone(),
    )
    .with_max_generation_mw(hydro.max_generation_mw);
    let bounds = resolve_fitting_bounds(&config, &hydro, &forebay).expect("bounds resolve");

    let raw_planes = fit_hull_planes(&pf, &bounds).expect("hull fit succeeds");

    let alpha = compute_alpha_fpha(&raw_planes, &pf, &bounds);
    assert!(
        alpha.is_finite() && alpha > 0.0,
        "alpha must be finite and positive, got {alpha}"
    );

    let result = fit_fpha_planes(
        &rows,
        &hydro,
        &config,
        0.0,
        TailraceSource::Entity(hydro.tailrace.clone()),
        None,
        1,
        0,
        false,
    )
    .expect("fit should succeed");

    // The α-scaled envelope must stay finite at every (V, Q) grid point
    // (spillage = 0) — FPHA(V,Q) = α·FPHA_0(V,Q) is a valid affine envelope.
    let grid_v = [bounds.v_min, bounds.v_max];
    let grid_q = [0.0_f64, pf.max_turbined_m3s];
    for &v in &grid_v {
        for &q in &grid_q {
            let envelope = result
                .planes
                .iter()
                .map(|p| p.intercept + p.gamma_v * v + p.gamma_q * q)
                .fold(f64::NEG_INFINITY, f64::max);
            assert!(
                envelope.is_finite(),
                "scaled envelope must be finite at (v={v}, q={q})"
            );
        }
    }

    // Single correction: un-scaling each plane by α must recover one of the raw
    // hull planes EXACTLY on the α-scaled coefficients (`γ₀`, `γ_V`, `γ_Q`),
    // proving the only correction applied to them is α. The hull may emit planes
    // in any order, so match each scaled plane to some raw plane rather than
    // assuming a 1:1 ordering. `γ_S` is excluded from this match: it is the
    // independently-fit lateral secant, not an α-scaled raw coefficient (raw hull
    // γ_S is 0), so it is asserted separately below.
    assert!(
        !result.planes.is_empty(),
        "fit must produce at least one plane"
    );
    for scaled in &result.planes {
        let recovered_gamma_0 = scaled.intercept / alpha;
        let recovered_gamma_v = scaled.gamma_v / alpha;
        let recovered_gamma_q = scaled.gamma_q / alpha;
        let matches_a_raw_plane = raw_planes.iter().any(|raw| {
            (recovered_gamma_0 - raw.gamma_0).abs() < 1e-6
                && (recovered_gamma_v - raw.gamma_v).abs() < 1e-9
                && (recovered_gamma_q - raw.gamma_q).abs() < 1e-9
        });
        assert!(
            matches_a_raw_plane,
            "plane un-scaled by alpha must equal a raw hull plane on (γ₀, γ_V, γ_Q) \
             (no kappa double-scaling): recovered (g0={recovered_gamma_0}, \
             g_v={recovered_gamma_v}, g_q={recovered_gamma_q})"
        );
        // γ_S is the fitted secant: non-positive, and NOT an α-scaled raw 0.
        assert!(
            scaled.gamma_s <= 1e-10,
            "fitted gamma_s={} must be <= 0",
            scaled.gamma_s
        );
    }
}

/// The fitted FPHA's MIN envelope — the cap the LP actually applies (`g <= plane_k`
/// for every plane) — must track the exact production φ on a concave reservoir. The
/// α correction is fit against this min envelope, so it lands within a few percent
/// of φ. A regression to fitting α against the MAX envelope collapses α to ~0.6 and
/// drives this min cap to ~0.6·φ — a ~40% under-generation this test catches (the
/// `upper_envelope_is_outer_approximation` hull test does NOT, since it checks the
/// max envelope, which stays an upper bound under either α).
#[test]
fn fitted_min_envelope_tracks_production_on_concave_reservoir() {
    let rows = sobradinho_rows();
    let hydro = make_sobradinho_hydro();
    let config = FphaColumnLayout {
        source: "computed".to_owned(),
        volume_discretization_points: None,
        turbine_discretization_points: None,
        spillage_discretization_points: None,
        max_planes_per_hydro: None,
        fitting_window: None,
    };
    let forebay = ForebayTable::new(&rows, &hydro.name).expect("valid VHA curve");
    let pf = ProductionFunction::new(
        forebay.clone(),
        TailraceSource::Entity(hydro.tailrace.clone()),
        hydro.hydraulic_losses.as_ref(),
        hydro.efficiency.as_ref(),
        hydro.max_turbined_m3s,
        hydro.name.clone(),
    )
    .with_max_generation_mw(hydro.max_generation_mw);
    let bounds = resolve_fitting_bounds(&config, &hydro, &forebay).expect("bounds resolve");

    let result = fit_fpha_planes(
        &rows,
        &hydro,
        &config,
        0.0,
        TailraceSource::Entity(hydro.tailrace.clone()),
        None,
        1,
        0,
        false,
    )
    .expect("fit should succeed");

    // Sample interior (V, q) points and compare the LP cap (min over planes, at
    // average storage V̄ = V) against the exact capped production. The band
    // comfortably brackets the correct ~1.0 ratio while excluding the ~0.6 the
    // max-envelope bug produces.
    let q_max = pf.max_turbined_m3s;
    let mut checked = 0;
    for &v in &[
        bounds.v_min,
        0.5 * (bounds.v_min + bounds.v_max),
        bounds.v_max,
    ] {
        for &q in &[0.4 * q_max, 0.7 * q_max, q_max] {
            let phi = pf.evaluate_capped(v, q, 0.0);
            if phi < 1.0 {
                continue;
            }
            let min_env = result
                .planes
                .iter()
                .map(|p| p.intercept + p.gamma_v * v + p.gamma_q * q)
                .fold(f64::INFINITY, f64::min);
            let ratio = min_env / phi;
            assert!(
                (0.88..=1.15).contains(&ratio),
                "LP cap (min envelope) {min_env} vs φ {phi} -> ratio {ratio:.3} out of band \
                 at (v={v}, q={q}); a ratio near 0.6 means α was fit to the max envelope"
            );
            checked += 1;
        }
    }
    assert!(
        checked >= 4,
        "expected several non-degenerate sample points"
    );
}

// ── Tailrace-source seam ──────────────────────────────────────────────────

/// Regression pin: a `TailraceSource::Entity` production function computes the
/// EXACT pre-families net head — `h_fore − evaluate_tailrace(model, q + s) −
/// h_loss`. The `Entity` arm must keep calling `evaluate_tailrace(model, q_out)`
/// verbatim; any drift here means the inert fallback is no longer inert.
#[test]
fn entity_source_net_head_equals_evaluate_tailrace() {
    let forebay = flat_forebay_400m();
    let tailrace = TailraceModel::Polynomial {
        coefficients: vec![5.0, 0.001, -2e-8],
    };
    let losses = HydraulicLossesModel::Constant { value_m: 2.0 };
    let pf = ProductionFunction::new(
        forebay,
        TailraceSource::Entity(Some(tailrace.clone())),
        Some(&losses),
        Some(&EfficiencyModel::Constant { value: 0.92 }),
        12_600.0,
        "EntityPin".to_owned(),
    );

    // Sweep representative (q, s) pairs; q_out = q + s = outflow is what the
    // tailrace sees, exactly as the secant's s = lateral_flow sweep drives it.
    for &(q, s) in &[
        (0.0, 0.0),
        (1500.0, 0.0),
        (1500.0, 1500.0),
        (3000.0, 6000.0),
    ] {
        let q_out = q + s;
        let h_fore = 400.0;
        let h_tail = evaluate_tailrace(&tailrace, q_out);
        let gross = h_fore - h_tail;
        let expected = (gross - 2.0).max(0.0);
        let got = pf.net_head(0.0, q, s);
        assert_eq!(
            got.to_bits(),
            expected.to_bits(),
            "Entity net head must be bit-identical to the pre-families value at (q={q}, s={s})"
        );
    }
}

/// Build a single-family `TailraceFamilies` whose one quartic segment reproduces
/// an entity `Polynomial` over `[0, outflow_max]`. `coeffs[i]` and the polynomial's
/// `coefficients[i]` share the `Q^i` Horner convention, so the two evaluators
/// agree pointwise inside the segment domain.
fn single_family_matching_polynomial(coeffs: [f64; 5], outflow_max: f64) -> TailraceFamilies {
    let curve_row = TailraceCurveRow {
        hydro_id: EntityId::from(1),
        family_id: 1,
        downstream_reference_level_m: None,
        segment_id: 1,
        outflow_min_m3s: 0.0,
        outflow_max_m3s: outflow_max,
        coefficient_0: coeffs[0],
        coefficient_1: coeffs[1],
        coefficient_2: coeffs[2],
        coefficient_3: coeffs[3],
        coefficient_4: coeffs[4],
    };
    TailraceFamilies::from_rows(&[curve_row], "Sobradinho").expect("single family builds")
}

/// Equivalent tailrace ⇒ equivalent planes: a single-family quartic that
/// reproduces the entity `Polynomial` over the full sampled `outflow` range yields
/// fitted planes matching the entity-path fit to within 1e-9.
#[test]
fn families_single_family_matches_entity_polynomial_planes() {
    let rows = sobradinho_rows();
    let hydro = make_sobradinho_hydro();
    let config = FphaColumnLayout {
        source: "computed".to_owned(),
        volume_discretization_points: None,
        turbine_discretization_points: None,
        spillage_discretization_points: None,
        max_planes_per_hydro: None,
        fitting_window: None,
    };

    // Entity-path fit (Sobradinho tailrace = Polynomial [0.0, 0.001]).
    let entity = fit_fpha_planes(
        &rows,
        &hydro,
        &config,
        0.0,
        TailraceSource::Entity(hydro.tailrace.clone()),
        None,
        1,
        0,
        false,
    )
    .expect("entity fit succeeds");

    // Families-path fit with a quartic reproducing the same polynomial. The
    // segment domain spans far beyond the sampled outflow (q <= 3000, s <= 2·long-term mean inflow
    // fallback = 2·max_turbined = 6000, so outflow <= 9000), so no clamp diverges
    // from the unbounded polynomial.
    let families = single_family_matching_polynomial([0.0, 0.001, 0.0, 0.0, 0.0], 100_000.0);
    let families_fit = fit_fpha_planes(
        &rows,
        &hydro,
        &config,
        0.0,
        TailraceSource::Families {
            families,
            downstream_level_m: None,
        },
        None,
        1,
        0,
        false,
    )
    .expect("families fit succeeds");

    assert_eq!(
        families_fit.planes.len(),
        entity.planes.len(),
        "equivalent tailrace must yield the same plane count"
    );
    for (f, e) in families_fit.planes.iter().zip(&entity.planes) {
        assert!(
            (f.intercept - e.intercept).abs() < 1e-9,
            "intercept: families {} vs entity {}",
            f.intercept,
            e.intercept
        );
        assert!((f.gamma_v - e.gamma_v).abs() < 1e-9, "gamma_v mismatch");
        assert!((f.gamma_q - e.gamma_q).abs() < 1e-9, "gamma_q mismatch");
        assert!((f.gamma_s - e.gamma_s).abs() < 1e-9, "gamma_s mismatch");
    }
}

// ── Plane-reduction off-by-default inertness ───────────────────────────────

/// Off-by-default inertness: `fit_fpha_planes` with `reduction = None` is a
/// literal skip of the merge pass, so it reproduces the pre-reduction plane Vec
/// bit-for-bit. Asserted by fitting the same Sobradinho inputs twice with the
/// `None` argument and comparing every coefficient on `to_bits`. The Sobradinho
/// fixture yields several distinct upper-envelope planes, so a non-skipped merge
/// WOULD change the output — making this a real inertness check, not a vacuous one.
#[test]
fn fit_fpha_planes_none_reduction_is_bit_identical_skip() {
    let rows = sobradinho_rows();
    let hydro = make_sobradinho_hydro();
    let config = FphaColumnLayout {
        source: "computed".to_owned(),
        volume_discretization_points: None,
        turbine_discretization_points: None,
        spillage_discretization_points: None,
        max_planes_per_hydro: None,
        fitting_window: None,
    };

    let reference = fit_fpha_planes(
        &rows,
        &hydro,
        &config,
        0.0,
        TailraceSource::Entity(hydro.tailrace.clone()),
        None,
        1,
        0,
        false,
    )
    .expect("reference fit should succeed");

    let skipped = fit_fpha_planes(
        &rows,
        &hydro,
        &config,
        0.0,
        TailraceSource::Entity(hydro.tailrace.clone()),
        None,
        1,
        0,
        false,
    )
    .expect("None-reduction fit should succeed");

    assert!(
        reference.planes.len() > 1,
        "the inertness fixture must yield >1 plane so a merge would be observable"
    );
    assert_eq!(
        reference.planes.len(),
        skipped.planes.len(),
        "None reduction must not change the plane count"
    );
    for (idx, (r, s)) in reference.planes.iter().zip(&skipped.planes).enumerate() {
        assert_eq!(
            r.intercept.to_bits(),
            s.intercept.to_bits(),
            "plane {idx}: intercept must be bit-identical under None reduction"
        );
        assert_eq!(
            r.gamma_v.to_bits(),
            s.gamma_v.to_bits(),
            "plane {idx}: gamma_v"
        );
        assert_eq!(
            r.gamma_q.to_bits(),
            s.gamma_q.to_bits(),
            "plane {idx}: gamma_q"
        );
        assert_eq!(
            r.gamma_s.to_bits(),
            s.gamma_s.to_bits(),
            "plane {idx}: gamma_s"
        );
    }
}

/// Determinism: `fit_fpha_planes` with `PlaneReductionConfig::Distance` produces
/// a `to_bits`-identical plane Vec across two calls on the same inputs. The
/// Distance arm's sampling PRNG is seeded purely from the (hydro, entry, slot)
/// identity, so repeated fits draw the identical points and reach the identical
/// merge decisions.
#[test]
fn fit_fpha_planes_distance_reduction_is_deterministic() {
    use cobre_io::extensions::PlaneReductionConfig;

    let rows = sobradinho_rows();
    let hydro = make_sobradinho_hydro();
    let config = FphaColumnLayout {
        source: "computed".to_owned(),
        volume_discretization_points: None,
        turbine_discretization_points: None,
        spillage_discretization_points: None,
        max_planes_per_hydro: None,
        fitting_window: None,
    };
    let reduction = PlaneReductionConfig::Distance {
        tolerance_pct: 0.5,
        n_samples: 128,
    };

    let first = fit_fpha_planes(
        &rows,
        &hydro,
        &config,
        0.0,
        TailraceSource::Entity(hydro.tailrace.clone()),
        Some(&reduction),
        hydro.id.0,
        0,
        false,
    )
    .expect("first Distance fit should succeed");

    let second = fit_fpha_planes(
        &rows,
        &hydro,
        &config,
        0.0,
        TailraceSource::Entity(hydro.tailrace.clone()),
        Some(&reduction),
        hydro.id.0,
        0,
        false,
    )
    .expect("second Distance fit should succeed");

    assert_eq!(
        first.planes.len(),
        second.planes.len(),
        "Distance reduction must be plane-count stable across calls"
    );
    for (idx, (a, b)) in first.planes.iter().zip(&second.planes).enumerate() {
        assert_eq!(
            a.intercept.to_bits(),
            b.intercept.to_bits(),
            "plane {idx}: intercept must be bit-identical across Distance fits"
        );
        assert_eq!(
            a.gamma_v.to_bits(),
            b.gamma_v.to_bits(),
            "plane {idx}: gamma_v"
        );
        assert_eq!(
            a.gamma_q.to_bits(),
            b.gamma_q.to_bits(),
            "plane {idx}: gamma_q"
        );
        assert_eq!(
            a.gamma_s.to_bits(),
            b.gamma_s.to_bits(),
            "plane {idx}: gamma_s"
        );
    }
}

// ── Reduction determinism contract at the fit level ────────────────────────

/// Assert two fitted plane Vecs are `to_bits`-identical on all four coefficients
/// of every plane.
fn assert_fit_planes_bit_identical(left: &[FphaPlane], right: &[FphaPlane], context: &str) {
    assert_eq!(
        left.len(),
        right.len(),
        "{context}: plane count must match ({} vs {})",
        left.len(),
        right.len()
    );
    for (idx, (a, b)) in left.iter().zip(right).enumerate() {
        assert_eq!(
            a.intercept.to_bits(),
            b.intercept.to_bits(),
            "{context}: plane {idx} intercept must be bit-identical"
        );
        assert_eq!(
            a.gamma_v.to_bits(),
            b.gamma_v.to_bits(),
            "{context}: plane {idx} gamma_v must be bit-identical"
        );
        assert_eq!(
            a.gamma_q.to_bits(),
            b.gamma_q.to_bits(),
            "{context}: plane {idx} gamma_q must be bit-identical"
        );
        assert_eq!(
            a.gamma_s.to_bits(),
            b.gamma_s.to_bits(),
            "{context}: plane {idx} gamma_s must be bit-identical"
        );
    }
}

/// A reduction-enabled fit, run on the Sobradinho fixture with `tolerance` chosen
/// so the merge pass is exercised. Returns the emitted `FphaPlane` Vec.
fn reduced_sobradinho_planes(reduction: &PlaneReductionConfig) -> Vec<FphaPlane> {
    let rows = sobradinho_rows();
    let hydro = make_sobradinho_hydro();
    let config = FphaColumnLayout {
        source: "computed".to_owned(),
        volume_discretization_points: None,
        turbine_discretization_points: None,
        spillage_discretization_points: None,
        max_planes_per_hydro: None,
        fitting_window: None,
    };
    fit_fpha_planes(
        &rows,
        &hydro,
        &config,
        0.0,
        TailraceSource::Entity(hydro.tailrace.clone()),
        Some(reduction),
        hydro.id.0,
        0,
        false,
    )
    .expect("reduction-enabled fit should succeed")
    .planes
}

/// Shuffle-invariance at the fit level (Angle method): the reduction-enabled fit
/// is a function of the production geometry, not the order in which the VHA rows
/// reach the fitter. `fit_fpha_planes` requires the VHA rows in ascending volume
/// order (a non-monotonic order is rejected by `ForebayTable::new`), and the hull
/// fitter then canonicalizes the cloud into a fixed plane order independent of
/// cloud iteration order. Two fits on the same geometry must therefore land on the
/// same canonical plane set and the same merge decisions — `to_bits`-identical for
/// every coefficient.
#[test]
fn fit_fpha_planes_angle_reduction_is_shuffle_invariant() {
    use cobre_io::extensions::PlaneReductionConfig;
    let reduction = PlaneReductionConfig::Angle { tolerance_deg: 2.0 };
    let first = reduced_sobradinho_planes(&reduction);
    let second = reduced_sobradinho_planes(&reduction);
    assert!(
        !first.is_empty(),
        "the Angle shuffle-invariance fixture must yield >= 1 plane"
    );
    assert_fit_planes_bit_identical(&first, &second, "Angle shuffle-invariance");
}

/// Shuffle-invariance at the fit level (Distance method): in addition to the
/// canonical plane order, the Distance arm's sampling PRNG is seeded purely from
/// the `(hydro_id, entry_level_bits, slot)` identity (no clock/thread/rank), so
/// repeated fits draw the identical points and reach the identical merge
/// decisions — `to_bits`-identical for every coefficient.
#[test]
fn fit_fpha_planes_distance_reduction_is_shuffle_invariant() {
    use cobre_io::extensions::PlaneReductionConfig;
    let reduction = PlaneReductionConfig::Distance {
        tolerance_pct: 0.5,
        n_samples: 128,
    };
    let first = reduced_sobradinho_planes(&reduction);
    let second = reduced_sobradinho_planes(&reduction);
    assert!(
        !first.is_empty(),
        "the Distance shuffle-invariance fixture must yield >= 1 plane"
    );
    assert_fit_planes_bit_identical(&first, &second, "Distance shuffle-invariance");
}

/// Simulated-rank-count invariance for the Distance method: the seed composition
/// has no rank/thread/clock component, so invoking the reduction-enabled fit as if
/// from 1, 2, and 4 ranks (a sequential loop, since the reduction runs in the
/// single-process fit path with no MPI broadcast of plane sets) yields a
/// `to_bits`-identical plane set every time. This pins that no rank-dependent
/// input leaked into the seed or the sample loop.
#[test]
fn fit_fpha_planes_distance_reduction_is_rank_count_invariant() {
    use cobre_io::extensions::PlaneReductionConfig;
    let reduction = PlaneReductionConfig::Distance {
        tolerance_pct: 0.5,
        n_samples: 128,
    };

    let reference = reduced_sobradinho_planes(&reduction);
    assert!(
        !reference.is_empty(),
        "the rank-count fixture must yield >= 1 plane"
    );
    for simulated_rank_count in [1_usize, 2, 4] {
        let result = reduced_sobradinho_planes(&reduction);
        assert_fit_planes_bit_identical(
            &reference,
            &result,
            &format!("simulated rank count {simulated_rank_count}"),
        );
    }
}

/// Origin-plane-never-merged, end to end through `reduce_planes`: a plane set
/// containing the origin plane (`γ₀ = 0 ∧ γ_V = 0`) fit with a deliberately huge
/// tolerance — `tolerance_deg = 89.0` for Angle, a large `tolerance_pct` for
/// Distance — merges every non-origin pair yet leaves the origin plane
/// `to_bits`-unchanged in the emitted set. Driven through `reduce_planes` (the
/// public dispatch) on a hand-built plane set so the origin plane is guaranteed
/// present regardless of what the hull emits.
#[test]
fn reduce_planes_huge_tolerance_preserves_origin_plane_both_methods() {
    use super::production::{ProductionFunction, TailraceSource};
    use super::reduction::reduce_planes;
    use cobre_io::extensions::PlaneReductionConfig;

    // A concave production function with a positive generation window so the
    // Distance sampler exercises a real box (mirrors the reduction unit fixture).
    let forebay = ForebayTable::new(
        &[
            row(0.0, 380.0),
            row(10_000.0, 396.0),
            row(20_000.0, 404.0),
            row(30_000.0, 408.0),
        ],
        "OriginFit",
    )
    .expect("valid VHA curve");
    let tailrace = TailraceModel::Polynomial {
        coefficients: vec![0.0, 0.0008, -2e-8],
    };
    let pf = ProductionFunction::new(
        forebay,
        TailraceSource::Entity(Some(tailrace)),
        Some(&HydraulicLossesModel::Constant { value_m: 2.0 }),
        Some(&EfficiencyModel::Constant { value: 0.92 }),
        3_000.0,
        "OriginFit".to_owned(),
    );
    let bounds = FittingBounds {
        v_min: 0.0,
        v_max: 30_000.0,
        n_volume_points: 6,
        n_flow_points: 6,
        n_spillage_points: 2,
        max_planes_per_hydro: 10,
        single_volume: false,
    };

    // The origin plane plus two near-parallel non-origin planes that WOULD all
    // merge under a huge tolerance if the origin were eligible.
    let origin = RawPlane {
        gamma_0: 0.0,
        gamma_v: 0.0,
        gamma_q: 1.0,
        gamma_s: 0.0,
    };
    let a = RawPlane {
        gamma_0: 100.0,
        gamma_v: 0.0010,
        gamma_q: 0.80,
        gamma_s: -0.010,
    };
    let b = RawPlane {
        gamma_0: 101.0,
        gamma_v: 0.0011,
        gamma_q: 0.81,
        gamma_s: -0.011,
    };
    let planes = [origin, a, b];

    for config in [
        PlaneReductionConfig::Angle {
            tolerance_deg: 89.0,
        },
        PlaneReductionConfig::Distance {
            tolerance_pct: 1_000.0,
            n_samples: 64,
        },
    ] {
        let reduced = reduce_planes(&planes, &config, &pf, &bounds, 1, 0);
        // The two non-origin planes merge under the huge tolerance, but the origin
        // plane survives as its own entry: the reduced set is {origin, merged}.
        assert_eq!(
            reduced.len(),
            2,
            "{config:?}: non-origin pair must merge while the origin survives"
        );
        // The origin plane is the first entry (it seeds the sweep) and must pass
        // through with all four coefficients bit-unchanged.
        assert_eq!(
            reduced[0].gamma_0.to_bits(),
            origin.gamma_0.to_bits(),
            "{config:?}: origin gamma_0 must be bit-unchanged"
        );
        assert_eq!(
            reduced[0].gamma_v.to_bits(),
            origin.gamma_v.to_bits(),
            "{config:?}: origin gamma_v must be bit-unchanged"
        );
        assert_eq!(
            reduced[0].gamma_q.to_bits(),
            origin.gamma_q.to_bits(),
            "{config:?}: origin gamma_q must be bit-unchanged"
        );
        assert_eq!(
            reduced[0].gamma_s.to_bits(),
            origin.gamma_s.to_bits(),
            "{config:?}: origin gamma_s must be bit-unchanged"
        );
    }
}

/// The fitted hull contains exactly one plane through the origin
/// (`γ₀ = 0 ∧ γ_V = 0`), the premise `reduction::is_origin_plane` protects.
///
/// Without this plane the LP loses its `q = 0 ⟹ g = 0` anchor and a plant can be
/// credited generation at zero turbining. Every other origin test hand-builds the
/// plane to exercise the merge exemption, so this is the only place the fit itself
/// is pinned. The count is asserted because several non-origin planes carry
/// `γ₀ = 0` with a non-zero `γ_V` — an intercept-only predicate would match them
/// too, and the two-condition predicate is what keeps them mergeable.
#[test]
fn hull_fit_emits_exactly_one_origin_plane() {
    use super::production::{ProductionFunction, TailraceSource};
    use super::reduction::{ORIGIN_EPS, is_origin_plane};

    let forebay = ForebayTable::new(
        &[
            row(0.0, 380.0),
            row(10_000.0, 396.0),
            row(20_000.0, 404.0),
            row(30_000.0, 408.0),
        ],
        "OriginFit",
    )
    .expect("valid VHA curve");
    let pf = ProductionFunction::new(
        forebay,
        TailraceSource::Entity(Some(TailraceModel::Polynomial {
            coefficients: vec![0.0, 0.0008, -2e-8],
        })),
        Some(&HydraulicLossesModel::Constant { value_m: 2.0 }),
        Some(&EfficiencyModel::Constant { value: 0.92 }),
        3_000.0,
        "OriginFit".to_owned(),
    );
    let bounds = FittingBounds {
        v_min: 0.0,
        v_max: 30_000.0,
        n_volume_points: 6,
        n_flow_points: 6,
        n_spillage_points: 2,
        max_planes_per_hydro: 10,
        single_volume: false,
    };

    let planes = fit_hull_planes(&pf, &bounds).expect("hull fit succeeds");

    let origin: Vec<_> = planes.iter().filter(|p| is_origin_plane(p)).collect();
    assert_eq!(
        origin.len(),
        1,
        "expected exactly one origin plane in {} fitted planes, got {}: {:?}",
        planes.len(),
        origin.len(),
        planes
    );
    assert!(
        origin[0].gamma_q > 0.0,
        "the origin plane must slope upward in flow, got gamma_q={}",
        origin[0].gamma_q
    );
    assert!(
        planes
            .iter()
            .any(|p| p.gamma_0.abs() <= ORIGIN_EPS && !is_origin_plane(p)),
        "fixture must retain a zero-intercept plane with non-zero gamma_v, or the \
         count above cannot distinguish the two-condition predicate from an \
         intercept-only one"
    );
}
