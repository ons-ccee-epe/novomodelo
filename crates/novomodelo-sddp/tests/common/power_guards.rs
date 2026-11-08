//! Evidence a fold gate records while it runs, and the guards it asserts at the end so that it
//! cannot pass vacuously: a comparison of a compact LP against its expanded counterpart has power
//! only if the fixture exercises what the fold changes. The caller solves the LP and passes plain
//! numbers, so nothing here touches a setup, a solver or the expander. "Binding" means `!= 0.0`,
//! so a `-0.0` never counts as evidence, and scales compare by bit pattern.

/// A constraint family whose duals a gate may require to be nonzero.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DualFamily {
    /// Piecewise-linear production-function rows.
    Fpha,
    /// Evaporation rows.
    Evaporation,
    /// User-defined generic constraint rows.
    Generic,
}

/// A condition on the recorded evidence; "recorded" means recorded since [`PowerEvidence::new`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Guard {
    /// Some recorded cut has a nonzero dual and a nonzero lag coefficient.
    BindingLagCut,
    /// [`Guard::BindingLagCut`] holds at a lag of at least 1.
    BindingShiftedLagCut,
    /// Some recorded z scale differs from 1.0 by bits.
    NonUnitZScale,
    /// Some recorded lag scale differs from 1.0 by bits.
    NonUnitLagScale,
    /// Some recorded coupled row has a nonzero dual and compact and expanded scales that differ by bits.
    RescaledCoupledRowWithDual,
    /// Some recorded coupled row has a nonzero dual and an expanded scale that differs from 1.0 by bits.
    NonUnitCoupledRowScaleWithDual,
    /// Some recorded dual of the family is nonzero.
    FamilyDual(DualFamily),
    /// Every hydro has some recorded nonzero coupling dual.
    CouplingDualEveryHydro,
}

/// Flags and per-hydro booleans only, so recording allocates nothing after [`PowerEvidence::new`].
// Rationale (struct_excessive_bools): one independent flag per `Guard` variant; a bitset would
// need a variant-to-bit map that the flags make unnecessary.
#[allow(clippy::struct_excessive_bools)]
pub struct PowerEvidence {
    n_hydros: usize,
    lag_depth: usize,
    binding_lag_cut: bool,
    binding_shifted_lag_cut: bool,
    non_unit_z_scale: bool,
    non_unit_lag_scale: bool,
    rescaled_coupled_row_with_dual: bool,
    non_unit_coupled_row_scale_with_dual: bool,
    family_dual: [bool; 3],
    coupling_dual: Vec<bool>,
}

fn is_non_unit(scale: f64) -> bool {
    scale.to_bits() != 1.0_f64.to_bits()
}

impl PowerEvidence {
    /// Empty evidence for a study with `n_hydros` hydros and `lag_depth` inflow lags.
    #[must_use]
    pub fn new(n_hydros: usize, lag_depth: usize) -> Self {
        Self {
            n_hydros,
            lag_depth,
            binding_lag_cut: false,
            binding_shifted_lag_cut: false,
            non_unit_z_scale: false,
            non_unit_lag_scale: false,
            rescaled_coupled_row_with_dual: false,
            non_unit_coupled_row_scale_with_dual: false,
            family_dual: [false; 3],
            coupling_dual: vec![false; n_hydros],
        }
    }

    /// Records a cut's row dual and its lag coefficients in lag-major order: offset `o` is hydro
    /// `o % n_hydros` at lag `o / n_hydros`.
    pub fn record_cut(&mut self, dual: f64, lag_coefficients: &[f64]) {
        assert_eq!(
            lag_coefficients.len(),
            self.n_hydros * self.lag_depth,
            "record_cut: lag coefficients must number n_hydros * lag_depth"
        );
        if dual != 0.0 {
            for (offset, &coefficient) in lag_coefficients.iter().enumerate() {
                if coefficient != 0.0 {
                    self.binding_lag_cut = true;
                    self.binding_shifted_lag_cut |= offset >= self.n_hydros;
                }
            }
        }
    }

    /// Records the expanded z and lag column scales.
    pub fn record_expanded_scales(&mut self, z_scales: &[f64], lag_scales: &[f64]) {
        self.non_unit_z_scale |= z_scales.iter().copied().any(is_non_unit);
        self.non_unit_lag_scale |= lag_scales.iter().copied().any(is_non_unit);
    }

    /// Records a coupled row's dual and its compact and expanded scales.
    pub fn record_coupled_row(&mut self, dual: f64, compact_scale: f64, expanded_scale: f64) {
        if dual != 0.0 {
            self.rescaled_coupled_row_with_dual |=
                compact_scale.to_bits() != expanded_scale.to_bits();
            self.non_unit_coupled_row_scale_with_dual |= is_non_unit(expanded_scale);
        }
    }

    /// Records one dual of `family`.
    pub fn record_family_dual(&mut self, family: DualFamily, dual: f64) {
        self.family_dual[family as usize] |= dual != 0.0;
    }

    /// Records the coupling dual `(Gᵀπ)_h` of `hydro`.
    pub fn record_coupling_dual(&mut self, hydro: usize, gt_pi: f64) {
        assert!(
            hydro < self.n_hydros,
            "record_coupling_dual: expected hydro < {}, got {hydro}",
            self.n_hydros
        );
        self.coupling_dual[hydro] |= gt_pi != 0.0;
    }

    fn holds(&self, guard: Guard) -> bool {
        match guard {
            Guard::BindingLagCut => self.binding_lag_cut,
            Guard::BindingShiftedLagCut => self.binding_shifted_lag_cut,
            Guard::NonUnitZScale => self.non_unit_z_scale,
            Guard::NonUnitLagScale => self.non_unit_lag_scale,
            Guard::RescaledCoupledRowWithDual => self.rescaled_coupled_row_with_dual,
            Guard::NonUnitCoupledRowScaleWithDual => self.non_unit_coupled_row_scale_with_dual,
            Guard::FamilyDual(family) => self.family_dual[family as usize],
            Guard::CouplingDualEveryHydro => self.coupling_dual.iter().all(|&seen| seen),
        }
    }

    /// Fails with one message listing every unmet guard of `guards`, in the given order.
    pub fn assert_holds(&self, label: &str, guards: &[Guard]) {
        let unmet: Vec<Guard> = guards
            .iter()
            .copied()
            .filter(|&guard| !self.holds(guard))
            .collect();
        assert!(
            unmet.is_empty(),
            "{label}: the fixture lacks power: unmet guards {unmet:?}"
        );
    }
}
