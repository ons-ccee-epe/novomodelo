//! Pre-resolved per-(entity, stage) penalty containers for O(1) solver lookup.
//!
//! Tables store their per-stage structs in a flat `Vec<T>` indexed
//! `data[entity_idx * n_stages + stage_idx]`. Populated by `cobre-io` after the
//! three-tier penalty cascade is applied; never modified after construction.

use crate::HydroPenalties;

/// Bus penalty values for a given (bus, stage) pair.
///
/// Only `excess_cost`: deficit segments are **not** stage-varying (Penalty System
/// spec SS3), so the piecewise deficit structure is not duplicated per stage.
///
/// # Examples
///
/// ```
/// use cobre_core::resolved::BusStagePenalties;
///
/// let p = BusStagePenalties { excess_cost: 0.01 };
/// let q = p;
/// assert!((q.excess_cost - 0.01).abs() < f64::EPSILON);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BusStagePenalties {
    /// Excess generation absorption cost \[$/`MWh`\].
    pub excess_cost: f64,
}

/// Line penalty values for a given (line, stage) pair.
///
/// # Examples
///
/// ```
/// use cobre_core::resolved::LineStagePenalties;
///
/// let p = LineStagePenalties { exchange_cost: 0.5 };
/// let q = p;
/// assert!((q.exchange_cost - 0.5).abs() < f64::EPSILON);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LineStagePenalties {
    /// Flow regularization cost \[$/`MWh`\].
    pub exchange_cost: f64,
}

/// Non-controllable source penalty values for a given (source, stage) pair.
///
/// # Examples
///
/// ```
/// use cobre_core::resolved::NcsStagePenalties;
///
/// let p = NcsStagePenalties { curtailment_cost: 10.0 };
/// let q = p;
/// assert!((q.curtailment_cost - 10.0).abs() < f64::EPSILON);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct NcsStagePenalties {
    /// Curtailment regularization cost \[$/`MWh`\].
    pub curtailment_cost: f64,
}

// ─── Pre-resolved containers ──────────────────────────────────────────────────

/// Pre-resolved penalty table for all entities across all stages.
///
/// # Examples
///
/// ```
/// use cobre_core::HydroPenalties;
/// use cobre_core::resolved::{
///     BusStagePenalties, LineStagePenalties, NcsStagePenalties, PenaltiesCountsSpec,
///     PenaltiesDefaults, ResolvedPenalties,
/// };
///
/// let hydro_default = HydroPenalties {
///     spillage_cost: 0.01,
///     diversion_cost: 0.02,
///     turbined_cost: 0.03,
///     storage_violation_below_cost: 1000.0,
///     filling_target_violation_cost: 5000.0,
///     turbined_violation_below_cost: 500.0,
///     outflow_violation_below_cost: 500.0,
///     outflow_violation_above_cost: 500.0,
///     generation_violation_below_cost: 500.0,
///     evaporation_violation_cost: 500.0,
///     water_withdrawal_violation_cost: 500.0,
///     water_withdrawal_violation_pos_cost: 500.0,
///     water_withdrawal_violation_neg_cost: 500.0,
///     evaporation_violation_pos_cost: 500.0,
///     evaporation_violation_neg_cost: 500.0,
///     inflow_nonnegativity_cost: 1000.0,
/// };
/// let bus_default = BusStagePenalties { excess_cost: 100.0 };
/// let line_default = LineStagePenalties { exchange_cost: 5.0 };
/// let ncs_default = NcsStagePenalties { curtailment_cost: 50.0 };
///
/// let table = ResolvedPenalties::new(
///     &PenaltiesCountsSpec { n_hydros: 3, n_buses: 2, n_lines: 1, n_ncs: 4, n_stages: 5 },
///     &PenaltiesDefaults { hydro: hydro_default, bus: bus_default, line: line_default, ncs: ncs_default },
/// );
///
/// // Hydro 1, stage 2 returns the default penalties.
/// let p = table.hydro_penalties(1, 2);
/// assert!((p.spillage_cost - 0.01).abs() < f64::EPSILON);
/// ```
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ResolvedPenalties {
    /// Stride for every entity table below: `data[entity_idx * n_stages + stage_idx]`.
    n_stages: usize,
    hydro: Vec<HydroPenalties>,
    bus: Vec<BusStagePenalties>,
    line: Vec<LineStagePenalties>,
    ncs: Vec<NcsStagePenalties>,
}

/// Entity counts for constructing a [`ResolvedPenalties`] table.
#[derive(Debug, Clone)]
pub struct PenaltiesCountsSpec {
    /// Number of hydro plants.
    pub n_hydros: usize,
    /// Number of buses.
    pub n_buses: usize,
    /// Number of transmission lines.
    pub n_lines: usize,
    /// Number of non-controllable sources.
    pub n_ncs: usize,
    /// Number of time stages.
    pub n_stages: usize,
}

/// Default per-stage penalty values for each entity type.
#[derive(Debug, Clone)]
pub struct PenaltiesDefaults {
    /// Default hydro penalties for all (hydro, stage) cells.
    pub hydro: HydroPenalties,
    /// Default bus penalties for all (bus, stage) cells.
    pub bus: BusStagePenalties,
    /// Default line penalties for all (line, stage) cells.
    pub line: LineStagePenalties,
    /// Default NCS penalties for all (ncs, stage) cells.
    pub ncs: NcsStagePenalties,
}

impl ResolvedPenalties {
    /// Return an empty penalty table with zero entities and zero stages.
    ///
    /// The default value in [`System`](crate::System) before penalty resolution.
    ///
    /// # Examples
    ///
    /// ```
    /// use cobre_core::ResolvedPenalties;
    ///
    /// let empty = ResolvedPenalties::empty();
    /// assert_eq!(empty.n_stages(), 0);
    /// ```
    #[must_use]
    pub fn empty() -> Self {
        Self {
            n_stages: 0,
            hydro: Vec::new(),
            bus: Vec::new(),
            line: Vec::new(),
            ncs: Vec::new(),
        }
    }

    /// Allocate a new resolved-penalties table filled with the given defaults.
    ///
    /// `counts.n_stages` must be `> 0`. Entity counts may be `0`.
    #[must_use]
    pub fn new(counts: &PenaltiesCountsSpec, defaults: &PenaltiesDefaults) -> Self {
        Self {
            n_stages: counts.n_stages,
            hydro: vec![defaults.hydro; counts.n_hydros * counts.n_stages],
            bus: vec![defaults.bus; counts.n_buses * counts.n_stages],
            line: vec![defaults.line; counts.n_lines * counts.n_stages],
            ncs: vec![defaults.ncs; counts.n_ncs * counts.n_stages],
        }
    }

    /// Return the resolved penalties for a hydro plant at a specific stage.
    #[inline]
    #[must_use]
    pub fn hydro_penalties(&self, hydro_index: usize, stage_index: usize) -> HydroPenalties {
        self.hydro[hydro_index * self.n_stages + stage_index]
    }

    /// Return the resolved penalties for a bus at a specific stage.
    #[inline]
    #[must_use]
    pub fn bus_penalties(&self, bus_index: usize, stage_index: usize) -> BusStagePenalties {
        self.bus[bus_index * self.n_stages + stage_index]
    }

    /// Return the resolved penalties for a line at a specific stage.
    #[inline]
    #[must_use]
    pub fn line_penalties(&self, line_index: usize, stage_index: usize) -> LineStagePenalties {
        self.line[line_index * self.n_stages + stage_index]
    }

    /// Return the resolved penalties for a non-controllable source at a specific stage.
    #[inline]
    #[must_use]
    pub fn ncs_penalties(&self, ncs_index: usize, stage_index: usize) -> NcsStagePenalties {
        self.ncs[ncs_index * self.n_stages + stage_index]
    }

    /// Return a mutable reference to the hydro penalty cell for in-place update.
    #[inline]
    pub fn hydro_penalties_mut(
        &mut self,
        hydro_index: usize,
        stage_index: usize,
    ) -> &mut HydroPenalties {
        &mut self.hydro[hydro_index * self.n_stages + stage_index]
    }

    /// Return a mutable reference to the bus penalty cell for in-place update.
    #[inline]
    pub fn bus_penalties_mut(
        &mut self,
        bus_index: usize,
        stage_index: usize,
    ) -> &mut BusStagePenalties {
        &mut self.bus[bus_index * self.n_stages + stage_index]
    }

    /// Return a mutable reference to the line penalty cell for in-place update.
    #[inline]
    pub fn line_penalties_mut(
        &mut self,
        line_index: usize,
        stage_index: usize,
    ) -> &mut LineStagePenalties {
        &mut self.line[line_index * self.n_stages + stage_index]
    }

    /// Return a mutable reference to the NCS penalty cell for in-place update.
    #[inline]
    pub fn ncs_penalties_mut(
        &mut self,
        ncs_index: usize,
        stage_index: usize,
    ) -> &mut NcsStagePenalties {
        &mut self.ncs[ncs_index * self.n_stages + stage_index]
    }

    /// Return the number of stages in this table.
    #[inline]
    #[must_use]
    pub fn n_stages(&self) -> usize {
        self.n_stages
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_hydro_penalties() -> HydroPenalties {
        HydroPenalties {
            spillage_cost: 0.01,
            diversion_cost: 0.02,
            turbined_cost: 0.03,
            storage_violation_below_cost: 1000.0,
            filling_target_violation_cost: 5000.0,
            turbined_violation_below_cost: 500.0,
            outflow_violation_below_cost: 400.0,
            outflow_violation_above_cost: 300.0,
            generation_violation_below_cost: 200.0,
            evaporation_violation_cost: 150.0,
            water_withdrawal_violation_cost: 100.0,
            water_withdrawal_violation_pos_cost: 100.0,
            water_withdrawal_violation_neg_cost: 100.0,
            evaporation_violation_pos_cost: 150.0,
            evaporation_violation_neg_cost: 150.0,
            inflow_nonnegativity_cost: 1000.0,
        }
    }

    #[test]
    fn test_hydro_stage_penalties_copy() {
        let p = make_hydro_penalties();
        let q = p;
        let r = p;
        assert_eq!(q, r);
        assert!((q.spillage_cost - p.spillage_cost).abs() < f64::EPSILON);
    }

    #[test]
    fn test_all_penalty_structs_are_copy() {
        let bp = BusStagePenalties { excess_cost: 1.0 };
        let lp = LineStagePenalties { exchange_cost: 2.0 };
        let np = NcsStagePenalties {
            curtailment_cost: 3.0,
        };

        assert_eq!(bp, bp);
        assert_eq!(lp, lp);
        assert_eq!(np, np);
        let bp2 = bp;
        let lp2 = lp;
        let np2 = np;
        assert!((bp2.excess_cost - 1.0).abs() < f64::EPSILON);
        assert!((lp2.exchange_cost - 2.0).abs() < f64::EPSILON);
        assert!((np2.curtailment_cost - 3.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_resolved_penalties_construction() {
        let hp = make_hydro_penalties();
        let bp = BusStagePenalties { excess_cost: 100.0 };
        let lp = LineStagePenalties { exchange_cost: 5.0 };
        let np = NcsStagePenalties {
            curtailment_cost: 50.0,
        };

        let table = ResolvedPenalties::new(
            &PenaltiesCountsSpec {
                n_hydros: 2,
                n_buses: 1,
                n_lines: 1,
                n_ncs: 1,
                n_stages: 3,
            },
            &PenaltiesDefaults {
                hydro: hp,
                bus: bp,
                line: lp,
                ncs: np,
            },
        );

        for hydro_idx in 0..2 {
            for stage_idx in 0..3 {
                let p = table.hydro_penalties(hydro_idx, stage_idx);
                assert!((p.spillage_cost - 0.01).abs() < f64::EPSILON);
                assert!((p.storage_violation_below_cost - 1000.0).abs() < f64::EPSILON);
            }
        }

        assert!((table.bus_penalties(0, 0).excess_cost - 100.0).abs() < f64::EPSILON);
        assert!((table.line_penalties(0, 1).exchange_cost - 5.0).abs() < f64::EPSILON);
        assert!((table.ncs_penalties(0, 2).curtailment_cost - 50.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_resolved_penalties_indexed_access() {
        let hp = make_hydro_penalties();
        let bp = BusStagePenalties { excess_cost: 10.0 };
        let lp = LineStagePenalties { exchange_cost: 1.0 };
        let np = NcsStagePenalties {
            curtailment_cost: 5.0,
        };

        let table = ResolvedPenalties::new(
            &PenaltiesCountsSpec {
                n_hydros: 3,
                n_buses: 0,
                n_lines: 0,
                n_ncs: 0,
                n_stages: 5,
            },
            &PenaltiesDefaults {
                hydro: hp,
                bus: bp,
                line: lp,
                ncs: np,
            },
        );
        assert_eq!(table.n_stages(), 5);

        let p = table.hydro_penalties(1, 3);
        assert!((p.diversion_cost - 0.02).abs() < f64::EPSILON);
        assert!((p.filling_target_violation_cost - 5000.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_resolved_penalties_mutable_update() {
        let hp = make_hydro_penalties();
        let bp = BusStagePenalties { excess_cost: 10.0 };
        let lp = LineStagePenalties { exchange_cost: 1.0 };
        let np = NcsStagePenalties {
            curtailment_cost: 5.0,
        };

        let mut table = ResolvedPenalties::new(
            &PenaltiesCountsSpec {
                n_hydros: 2,
                n_buses: 2,
                n_lines: 1,
                n_ncs: 1,
                n_stages: 3,
            },
            &PenaltiesDefaults {
                hydro: hp,
                bus: bp,
                line: lp,
                ncs: np,
            },
        );

        table.hydro_penalties_mut(0, 1).spillage_cost = 99.0;

        assert!((table.hydro_penalties(0, 1).spillage_cost - 99.0).abs() < f64::EPSILON);
        assert!((table.hydro_penalties(0, 0).spillage_cost - 0.01).abs() < f64::EPSILON);
        assert!((table.hydro_penalties(1, 1).spillage_cost - 0.01).abs() < f64::EPSILON);

        table.bus_penalties_mut(1, 2).excess_cost = 999.0;
        assert!((table.bus_penalties(1, 2).excess_cost - 999.0).abs() < f64::EPSILON);
        assert!((table.bus_penalties(0, 2).excess_cost - 10.0).abs() < f64::EPSILON);
    }

    #[test]
    #[cfg(feature = "serde")]
    fn test_resolved_penalties_serde_roundtrip() {
        let hp = make_hydro_penalties();
        let bp = BusStagePenalties { excess_cost: 100.0 };
        let lp = LineStagePenalties { exchange_cost: 5.0 };
        let np = NcsStagePenalties {
            curtailment_cost: 50.0,
        };

        let original = ResolvedPenalties::new(
            &PenaltiesCountsSpec {
                n_hydros: 2,
                n_buses: 1,
                n_lines: 1,
                n_ncs: 1,
                n_stages: 3,
            },
            &PenaltiesDefaults {
                hydro: hp,
                bus: bp,
                line: lp,
                ncs: np,
            },
        );
        let json = serde_json::to_string(&original).expect("serialize");
        let restored: ResolvedPenalties = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(original, restored);
    }
}
