//! Pre-resolved per-`(entity, stage)` stage-volume bound overlay. Every value
//! is a total over the whole stage in hm³, not a flow rate, and `None` means no
//! bound on that column. An empty overlay ([`ResolvedStageVolumeBounds::empty`])
//! is the state for every study without stage-volume bound columns; every
//! reader then returns the all-`None` default. Populated by `novomodelo-io`;
//! never modified after construction.
//!
//! The columns are stage-level only: the overlay has no block axis.

/// Stage-total volume bounds for one hydro plant at one stage \[hm³\]; `None`
/// is no bound.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct HydroStageVolumeBounds {
    /// Minimum outflow volume, stage total \[hm³\]; `None` is no bound.
    pub min_outflow_hm3: Option<f64>,
    /// Maximum outflow volume, stage total \[hm³\]; `None` is no bound.
    pub max_outflow_hm3: Option<f64>,
    /// Minimum diverted volume, stage total \[hm³\]; `None` is no bound.
    pub min_diversion_hm3: Option<f64>,
    /// Maximum diverted volume, stage total \[hm³\]; `None` is no bound.
    pub max_diversion_hm3: Option<f64>,
}

/// Stage-total volume bounds for one pumping station at one stage \[hm³\];
/// `None` is no bound.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PumpingStageVolumeBounds {
    /// Minimum pumped volume, stage total \[hm³\]; `None` is no bound.
    pub min_hm3: Option<f64>,
    /// Maximum pumped volume, stage total \[hm³\]; `None` is no bound.
    pub max_hm3: Option<f64>,
}

/// Entity counts for constructing a [`ResolvedStageVolumeBounds`] table.
#[derive(Debug, Clone)]
pub struct StageVolumeBoundsCountsSpec {
    /// Number of hydro plants.
    pub n_hydros: usize,
    /// Number of pumping stations.
    pub n_pumping: usize,
    /// Number of time stages.
    pub n_stages: usize,
}

/// Pre-resolved stage-total volume bound table for hydro plants and pumping
/// stations, across all stages \[hm³\]; `None` is no bound.
///
/// Each family `Vec` is indexed `entity_idx * n_stages + stage_idx` and is
/// independently lazy: a family stays empty until a row resolves to a cell in
/// that family (via its own `<family>_mut`), so a study with volume bounds on
/// hydros only never allocates the pumping family. An empty table
/// ([`ResolvedStageVolumeBounds::empty`]) makes every reader return the family
/// default and every writer return `None`.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ResolvedStageVolumeBounds {
    n_stages: usize,
    n_hydros: usize,
    n_pumping: usize,
    hydro: Vec<HydroStageVolumeBounds>,
    pumping: Vec<PumpingStageVolumeBounds>,
}

impl Default for ResolvedStageVolumeBounds {
    fn default() -> Self {
        Self::empty()
    }
}

impl ResolvedStageVolumeBounds {
    /// Create an empty table; every reader returns the family default and
    /// every writer returns `None`.
    ///
    /// # Examples
    ///
    /// ```
    /// use novomodelo_core::resolved::ResolvedStageVolumeBounds;
    ///
    /// let t = ResolvedStageVolumeBounds::empty();
    /// assert!(t.is_empty());
    /// assert_eq!(t.hydro(5, 3).max_outflow_hm3, None);
    /// ```
    #[must_use]
    pub fn empty() -> Self {
        Self {
            n_stages: 0,
            n_hydros: 0,
            n_pumping: 0,
            hydro: Vec::new(),
            pumping: Vec::new(),
        }
    }

    /// Store the table's dimensions; both family `Vec`s start empty, and
    /// `hydro_mut` / `pumping_mut` grow their own family to full size on its
    /// first successful write.
    #[must_use]
    pub fn new(counts: &StageVolumeBoundsCountsSpec) -> Self {
        Self {
            n_stages: counts.n_stages,
            n_hydros: counts.n_hydros,
            n_pumping: counts.n_pumping,
            hydro: Vec::new(),
            pumping: Vec::new(),
        }
    }

    fn flat_index(&self, n_entities: usize, entity_idx: usize, stage_idx: usize) -> Option<usize> {
        if entity_idx >= n_entities || stage_idx >= self.n_stages {
            return None;
        }
        Some(entity_idx * self.n_stages + stage_idx)
    }

    /// Look up the hydro volume bounds at `(hydro_idx, stage_idx)`; the
    /// all-`None` default when the family is empty or an index is out of range.
    #[inline]
    #[must_use]
    pub fn hydro(&self, hydro_idx: usize, stage_idx: usize) -> HydroStageVolumeBounds {
        self.flat_index(self.n_hydros, hydro_idx, stage_idx)
            .and_then(|idx| self.hydro.get(idx))
            .copied()
            .unwrap_or_default()
    }

    /// Return a mutable handle to the hydro cell, growing the hydro family to
    /// full size on its first call, or `None` when an index is out of range.
    #[inline]
    pub fn hydro_mut(
        &mut self,
        hydro_idx: usize,
        stage_idx: usize,
    ) -> Option<&mut HydroStageVolumeBounds> {
        let idx = self.flat_index(self.n_hydros, hydro_idx, stage_idx)?;
        if self.hydro.is_empty() {
            self.hydro = vec![HydroStageVolumeBounds::default(); self.n_hydros * self.n_stages];
        }
        self.hydro.get_mut(idx)
    }

    /// Look up the pumping volume bounds at `(station_idx, stage_idx)`; the
    /// all-`None` default when the family is empty or an index is out of range.
    #[inline]
    #[must_use]
    pub fn pumping(&self, station_idx: usize, stage_idx: usize) -> PumpingStageVolumeBounds {
        self.flat_index(self.n_pumping, station_idx, stage_idx)
            .and_then(|idx| self.pumping.get(idx))
            .copied()
            .unwrap_or_default()
    }

    /// Return a mutable handle to the pumping cell, growing the pumping family
    /// to full size on its first call, or `None` when an index is out of range.
    #[inline]
    pub fn pumping_mut(
        &mut self,
        station_idx: usize,
        stage_idx: usize,
    ) -> Option<&mut PumpingStageVolumeBounds> {
        let idx = self.flat_index(self.n_pumping, station_idx, stage_idx)?;
        if self.pumping.is_empty() {
            self.pumping =
                vec![PumpingStageVolumeBounds::default(); self.n_pumping * self.n_stages];
        }
        self.pumping.get_mut(idx)
    }

    /// Returns `true` while both family tables are empty.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.hydro.is_empty() && self.pumping.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::{
        HydroStageVolumeBounds, PumpingStageVolumeBounds, ResolvedStageVolumeBounds,
        StageVolumeBoundsCountsSpec,
    };

    fn table() -> ResolvedStageVolumeBounds {
        ResolvedStageVolumeBounds::new(&StageVolumeBoundsCountsSpec {
            n_hydros: 2,
            n_pumping: 1,
            n_stages: 3,
        })
    }

    #[test]
    fn empty_overlay_reads_no_bound() {
        for t in [ResolvedStageVolumeBounds::empty(), table()] {
            assert_eq!(t.hydro(3, 7), HydroStageVolumeBounds::default());
            assert_eq!(t.pumping(3, 7), PumpingStageVolumeBounds::default());
            assert!(t.is_empty());
        }
        assert_eq!(
            ResolvedStageVolumeBounds::default(),
            ResolvedStageVolumeBounds::empty()
        );
    }

    #[test]
    fn first_write_grows_only_its_family() {
        let mut t = table();

        t.hydro_mut(1, 2)
            .expect("in-range hydro cell")
            .min_diversion_hm3 = Some(4.0);

        assert_eq!(t.hydro(1, 2).min_diversion_hm3, Some(4.0));
        assert_eq!(t.hydro(1, 1), HydroStageVolumeBounds::default());
        assert_eq!(t.hydro(0, 2), HydroStageVolumeBounds::default());
        assert_eq!(t.pumping(0, 0), PumpingStageVolumeBounds::default());
        assert!(t.pumping.is_empty());
        assert!(!t.is_empty());

        t.pumping_mut(0, 2).expect("in-range pumping cell").max_hm3 = Some(9.0);

        assert_eq!(t.pumping(0, 2).max_hm3, Some(9.0));
        assert_eq!(t.pumping(0, 1), PumpingStageVolumeBounds::default());
        assert_eq!(t.hydro(1, 2).min_diversion_hm3, Some(4.0));
    }

    #[test]
    fn out_of_range_write_returns_none() {
        let mut t = table();

        assert!(t.hydro_mut(2, 0).is_none());
        assert!(t.hydro_mut(0, 3).is_none());
        assert!(t.hydro_mut(usize::MAX, 0).is_none());
        assert!(t.pumping_mut(1, 0).is_none());
        assert!(t.pumping_mut(0, 3).is_none());
        assert!(t.is_empty());

        let mut empty = ResolvedStageVolumeBounds::empty();
        assert!(empty.hydro_mut(0, 0).is_none());
        assert!(empty.pumping_mut(0, 0).is_none());
        assert!(empty.is_empty());
    }
}
