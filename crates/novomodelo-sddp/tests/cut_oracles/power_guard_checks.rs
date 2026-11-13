use std::panic::{AssertUnwindSafe, catch_unwind};

use crate::common::power_guards::{DualFamily, Guard, PowerEvidence};

const ALL_GUARDS: [Guard; 10] = [
    Guard::BindingLagCut,
    Guard::BindingShiftedLagCut,
    Guard::NonUnitZScale,
    Guard::NonUnitLagScale,
    Guard::RescaledCoupledRowWithDual,
    Guard::NonUnitCompactCoupledRowScaleWithDual,
    Guard::FamilyDual(DualFamily::Fpha),
    Guard::FamilyDual(DualFamily::Evaporation),
    Guard::FamilyDual(DualFamily::Generic),
    Guard::CouplingDualEveryHydro,
];

fn failure_message(evidence: &PowerEvidence, guards: &[Guard]) -> Option<String> {
    let payload = catch_unwind(AssertUnwindSafe(|| {
        evidence.assert_holds("demo gate", guards);
    }))
    .err()?;
    Some(
        *payload
            .downcast::<String>()
            .expect("an assert message is a String"),
    )
}

#[test]
fn complete_evidence_meets_every_guard() {
    let mut evidence = PowerEvidence::new(2, 2);
    evidence.record_cut(1.5, &[3.0, 0.0, 0.0, 0.0]);
    evidence.record_cut(0.5, &[0.0, 0.0, 2.0, 0.0]);
    evidence.record_expanded_scales(&[0.5, 1.0], &[1.0, 2.0]);
    evidence.record_coupled_row(0.7, 2.0, 4.0);
    for family in [
        DualFamily::Fpha,
        DualFamily::Evaporation,
        DualFamily::Generic,
    ] {
        evidence.record_family_dual(family, 1.0);
    }
    evidence.record_coupling_dual(0, 1.0);
    evidence.record_coupling_dual(1, -2.0);

    evidence.assert_holds("demo gate", &ALL_GUARDS);
}

#[test]
fn a_negative_zero_dual_is_not_binding() {
    let coefficients = [1.0, 0.0, 0.0, 0.0];
    let mut evidence = PowerEvidence::new(2, 2);
    evidence.record_cut(-0.0, &coefficients);
    evidence.record_cut(1.0, &[-0.0; 4]);
    let message = failure_message(&evidence, &[Guard::BindingLagCut])
        .expect("neither a -0.0 dual nor a -0.0 coefficient is binding");
    assert!(message.contains("[BindingLagCut]"), "{message}");

    evidence.record_cut(1.0, &coefficients);
    evidence.assert_holds("demo gate", &[Guard::BindingLagCut]);
}

#[test]
fn a_lag_zero_only_cut_fails_the_shifted_guard() {
    let mut evidence = PowerEvidence::new(2, 2);
    evidence.record_cut(1.0, &[2.0, 0.0, 0.0, 0.0]);
    evidence.record_cut(1.0, &[0.0, 2.0, 0.0, 0.0]);
    evidence.assert_holds("demo gate", &[Guard::BindingLagCut]);
    let message = failure_message(&evidence, &[Guard::BindingShiftedLagCut])
        .expect("a lag-0 coefficient is not shifted");
    assert!(message.contains("[BindingShiftedLagCut]"), "{message}");

    evidence.record_cut(1.0, &[0.0, 0.0, 2.0, 0.0]);
    evidence.assert_holds("demo gate", &[Guard::BindingShiftedLagCut]);
}

#[test]
fn every_unmet_guard_is_listed_in_one_panic() {
    let evidence = PowerEvidence::new(2, 2);
    evidence.assert_holds("demo gate", &[]);
    let message = failure_message(
        &evidence,
        &[
            Guard::NonUnitZScale,
            Guard::FamilyDual(DualFamily::Evaporation),
            Guard::CouplingDualEveryHydro,
        ],
    )
    .expect("empty evidence meets no guard");
    assert_eq!(
        message,
        "demo gate: the fixture lacks power: unmet guards \
         [NonUnitZScale, FamilyDual(Evaporation), CouplingDualEveryHydro]"
    );
}

#[test]
fn a_zero_coupling_dual_on_one_hydro_fails() {
    let mut evidence = PowerEvidence::new(3, 1);
    evidence.record_coupling_dual(0, 1.0);
    evidence.record_coupling_dual(1, -0.0);
    evidence.record_coupling_dual(2, 3.0);
    let message = failure_message(&evidence, &[Guard::CouplingDualEveryHydro])
        .expect("hydro 1 has no nonzero coupling dual");
    assert!(message.contains("[CouplingDualEveryHydro]"), "{message}");

    evidence.record_coupling_dual(1, 0.5);
    evidence.assert_holds("demo gate", &[Guard::CouplingDualEveryHydro]);
}

#[test]
fn unit_scales_fail_the_scale_guards() {
    let mut evidence = PowerEvidence::new(2, 2);
    evidence.record_expanded_scales(&[1.0, 1.0], &[1.0, 1.0]);
    evidence.record_coupled_row(2.0, 1.0, 1.0);
    evidence.record_coupled_row(0.0, 2.0, 4.0);
    let message = failure_message(
        &evidence,
        &[
            Guard::NonUnitZScale,
            Guard::NonUnitLagScale,
            Guard::RescaledCoupledRowWithDual,
            Guard::NonUnitCompactCoupledRowScaleWithDual,
        ],
    )
    .expect("unit scales and a zero-dual row are not evidence");
    assert!(
        message.ends_with(
            "[NonUnitZScale, NonUnitLagScale, RescaledCoupledRowWithDual, \
             NonUnitCompactCoupledRowScaleWithDual]"
        ),
        "{message}"
    );
}

#[test]
fn a_unit_compact_scale_leaves_the_compact_scale_guard_unmet() {
    let mut evidence = PowerEvidence::new(2, 2);
    evidence.record_coupled_row(1.0, 1.0, 4.0);
    assert!(evidence.holds(Guard::RescaledCoupledRowWithDual));
    assert!(!evidence.holds(Guard::NonUnitCompactCoupledRowScaleWithDual));

    let mut evidence = PowerEvidence::new(2, 2);
    evidence.record_coupled_row(1.0, 4.0, 4.0);
    assert!(evidence.holds(Guard::NonUnitCompactCoupledRowScaleWithDual));
    assert!(!evidence.holds(Guard::RescaledCoupledRowWithDual));
}
