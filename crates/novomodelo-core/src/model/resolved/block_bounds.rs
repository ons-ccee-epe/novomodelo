//! Pre-resolved per-block bound override overlay: layer 1 of the bound
//! precedence law. A bound row's optional `block_id` selects one block of a
//! stage rather than the whole stage; this overlay carries only the
//! per-family columns that a `block_id` may target. A `None` field in any
//! per-family override struct means no override for that column. An empty
//! overlay ([`ResolvedBlockBounds::empty`]) makes every per-block lookup fall
//! back to exactly the stage-wide cell it would otherwise override. Populated
//! by `cobre-io`; never modified after construction.
//!
//! Each `<Family>BlockOverride` struct is a full field-for-field mirror of its
//! `<Family>BlockBounds` counterpart rather than an `Option<<Family>BlockBounds>`
//! because optionality is per-**column**, not per-row: a layer-1 row may set
//! one column and fall through to the base on the rest, which a single
//! `Option` around the whole struct cannot express.

/// Per-block override for a hydro plant's block-eligible bounds.
///
/// Carries only the block-eligible hydro columns — every field on
/// [`HydroBlockBounds`](super::HydroBlockBounds). Deliberately excludes
/// `min_storage_hm3`, `max_storage_hm3`, `filling_min_rate_m3s`, and
/// `water_withdrawal_m3s` — those are stage-level `HydroStageBounds` columns;
/// a `block_id` on them is an error, not a silent skip.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct HydroBlockOverride {
    /// Minimum turbined flow override \[m³/s\].
    pub min_turbined_m3s: Option<f64>,
    /// Maximum turbined flow override \[m³/s\].
    pub max_turbined_m3s: Option<f64>,
    /// Environmental flow requirement override \[m³/s\].
    pub min_outflow_m3s: Option<f64>,
    /// Flood-control limit override \[m³/s\].
    pub max_outflow_m3s: Option<f64>,
    /// Minimum generation override \[MW\].
    pub min_generation_mw: Option<f64>,
    /// Maximum generation override \[MW\].
    pub max_generation_mw: Option<f64>,
    /// Minimum diversion flow override \[m³/s\].
    pub min_diversion_m3s: Option<f64>,
    /// Maximum diversion flow override \[m³/s\].
    pub max_diversion_m3s: Option<f64>,
    /// Minimum spillage flow override \[m³/s\].
    pub min_spillage_m3s: Option<f64>,
    /// Maximum spillage flow override \[m³/s\].
    pub max_spillage_m3s: Option<f64>,
}

/// Per-block override for a thermal unit's block-eligible bounds.
///
/// Deliberately has no `cost_per_mwh` field: per-block thermal cost is out of
/// scope, asymmetric with [`ContractBlockOverride::price_per_mwh`], which is
/// block-eligible.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ThermalBlockOverride {
    /// Minimum stable generation override \[MW\].
    pub min_generation_mw: Option<f64>,
    /// Maximum generation capacity override \[MW\].
    pub max_generation_mw: Option<f64>,
}

/// Per-block override for a transmission line's block-eligible bounds.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LineBlockOverride {
    /// Maximum direct flow capacity override \[MW\].
    pub direct_mw: Option<f64>,
    /// Maximum reverse flow capacity override \[MW\].
    pub reverse_mw: Option<f64>,
}

/// Per-block override for an energy contract's block-eligible bounds.
///
/// `price_per_mwh` IS block-eligible — deliberately asymmetric with
/// [`ThermalBlockOverride`]'s excluded `cost_per_mwh`.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ContractBlockOverride {
    /// Minimum contract usage override \[MW\].
    pub min_mw: Option<f64>,
    /// Maximum contract usage override \[MW\].
    pub max_mw: Option<f64>,
    /// Contract price override \[$/`MWh`\].
    pub price_per_mwh: Option<f64>,
}

/// Per-block override for a pumping station's block-eligible bounds.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PumpingBlockOverride {
    /// Minimum pumped flow override \[m³/s\].
    pub min_flow_m3s: Option<f64>,
    /// Maximum pumped flow override \[m³/s\].
    pub max_flow_m3s: Option<f64>,
}

// ─── Pre-resolved container ───────────────────────────────────────────────────

/// Entity counts for constructing a [`ResolvedBlockBounds`] table.
#[derive(Debug, Clone)]
pub struct BlockBoundsCountsSpec {
    /// Number of hydro plants.
    pub n_hydros: usize,
    /// Number of thermal units.
    pub n_thermals: usize,
    /// Number of transmission lines.
    pub n_lines: usize,
    /// Number of pumping stations.
    pub n_pumping: usize,
    /// Number of energy contracts.
    pub n_contracts: usize,
    /// Number of time stages.
    pub n_stages: usize,
    /// Maximum `stage.blocks.len()` across study stages.
    pub max_blocks: usize,
}

/// Pre-resolved per-block bound override table for every block-eligible
/// entity family, across all stages and blocks.
///
/// Every family `Vec` is indexed
/// `(entity_idx * n_stages + stage_idx) * max_blocks + block_idx`, and is
/// independently lazy: a family stays empty until a row actually resolves to
/// a cell in that family (via its own `<family>_override_mut`), so a study
/// with a block row in only one family never allocates the other four. An
/// empty table ([`ResolvedBlockBounds::empty`]) is the state for every study
/// with no `block_id` bound row at all; every reader then returns the family
/// default and every writer returns `None`.
///
/// # Examples
///
/// ```
/// use cobre_core::resolved::ResolvedBlockBounds;
///
/// let empty = ResolvedBlockBounds::empty();
/// assert!(empty.is_empty());
/// assert_eq!(empty.thermal_override(3, 7, 2).max_generation_mw, None);
/// ```
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ResolvedBlockBounds {
    n_stages: usize,
    max_blocks: usize,
    n_hydros: usize,
    n_thermals: usize,
    n_lines: usize,
    n_pumping: usize,
    n_contracts: usize,
    hydro: Vec<HydroBlockOverride>,
    thermal: Vec<ThermalBlockOverride>,
    line: Vec<LineBlockOverride>,
    pumping: Vec<PumpingBlockOverride>,
    contract: Vec<ContractBlockOverride>,
}

impl Default for ResolvedBlockBounds {
    fn default() -> Self {
        Self::empty()
    }
}

impl ResolvedBlockBounds {
    /// Create an empty per-block override table; every reader returns the
    /// family default and every writer returns `None`.
    ///
    /// # Examples
    ///
    /// ```
    /// use cobre_core::resolved::ResolvedBlockBounds;
    ///
    /// let t = ResolvedBlockBounds::empty();
    /// assert!(t.is_empty());
    /// assert_eq!(t.hydro_override(5, 3, 2).min_turbined_m3s, None);
    /// ```
    #[must_use]
    pub fn empty() -> Self {
        Self {
            n_stages: 0,
            max_blocks: 0,
            n_hydros: 0,
            n_thermals: 0,
            n_lines: 0,
            n_pumping: 0,
            n_contracts: 0,
            hydro: Vec::new(),
            thermal: Vec::new(),
            line: Vec::new(),
            pumping: Vec::new(),
            contract: Vec::new(),
        }
    }

    /// Store the table's dimensions; every family `Vec` starts empty —
    /// `<family>_override_mut` grows its own family to full size on its first
    /// successful write, so a family with zero applied rows stays empty (and
    /// [`is_empty`](Self::is_empty) stays `true` while every family does).
    #[must_use]
    pub fn new(counts: &BlockBoundsCountsSpec) -> Self {
        Self {
            n_stages: counts.n_stages,
            max_blocks: counts.max_blocks,
            n_hydros: counts.n_hydros,
            n_thermals: counts.n_thermals,
            n_lines: counts.n_lines,
            n_pumping: counts.n_pumping,
            n_contracts: counts.n_contracts,
            hydro: Vec::new(),
            thermal: Vec::new(),
            line: Vec::new(),
            pumping: Vec::new(),
            contract: Vec::new(),
        }
    }

    fn flat_index(&self, entity_idx: usize, stage_idx: usize, block_idx: usize) -> Option<usize> {
        if stage_idx >= self.n_stages || block_idx >= self.max_blocks {
            return None;
        }
        Some((entity_idx * self.n_stages + stage_idx) * self.max_blocks + block_idx)
    }

    /// Look up the hydro per-block override at `(hydro_idx, stage_idx, block_idx)`.
    /// Returns [`HydroBlockOverride::default`] (all `None`) when the table is
    /// empty or any index is out of range.
    #[inline]
    #[must_use]
    pub fn hydro_override(
        &self,
        hydro_idx: usize,
        stage_idx: usize,
        block_idx: usize,
    ) -> HydroBlockOverride {
        self.flat_index(hydro_idx, stage_idx, block_idx)
            .and_then(|idx| self.hydro.get(idx))
            .copied()
            .unwrap_or_default()
    }

    /// Return a mutable handle to the hydro per-block override cell, growing
    /// this family to full size on its first call, or `None` when the table
    /// is empty or any index is out of range.
    #[inline]
    pub fn hydro_override_mut(
        &mut self,
        hydro_idx: usize,
        stage_idx: usize,
        block_idx: usize,
    ) -> Option<&mut HydroBlockOverride> {
        let idx = self.flat_index(hydro_idx, stage_idx, block_idx)?;
        if self.hydro.is_empty() {
            self.hydro = vec![
                HydroBlockOverride::default();
                self.n_hydros * self.n_stages * self.max_blocks
            ];
        }
        self.hydro.get_mut(idx)
    }

    /// Look up the thermal per-block override at `(thermal_idx, stage_idx, block_idx)`.
    /// Returns [`ThermalBlockOverride::default`] (all `None`) when the table
    /// is empty or any index is out of range.
    #[inline]
    #[must_use]
    pub fn thermal_override(
        &self,
        thermal_idx: usize,
        stage_idx: usize,
        block_idx: usize,
    ) -> ThermalBlockOverride {
        self.flat_index(thermal_idx, stage_idx, block_idx)
            .and_then(|idx| self.thermal.get(idx))
            .copied()
            .unwrap_or_default()
    }

    /// Return a mutable handle to the thermal per-block override cell,
    /// growing this family to full size on its first call, or `None` when
    /// the table is empty or any index is out of range.
    #[inline]
    pub fn thermal_override_mut(
        &mut self,
        thermal_idx: usize,
        stage_idx: usize,
        block_idx: usize,
    ) -> Option<&mut ThermalBlockOverride> {
        let idx = self.flat_index(thermal_idx, stage_idx, block_idx)?;
        if self.thermal.is_empty() {
            self.thermal = vec![
                ThermalBlockOverride::default();
                self.n_thermals * self.n_stages * self.max_blocks
            ];
        }
        self.thermal.get_mut(idx)
    }

    /// Look up the line per-block override at `(line_idx, stage_idx, block_idx)`.
    /// Returns [`LineBlockOverride::default`] (all `None`) when the table is
    /// empty or any index is out of range.
    #[inline]
    #[must_use]
    pub fn line_override(
        &self,
        line_idx: usize,
        stage_idx: usize,
        block_idx: usize,
    ) -> LineBlockOverride {
        self.flat_index(line_idx, stage_idx, block_idx)
            .and_then(|idx| self.line.get(idx))
            .copied()
            .unwrap_or_default()
    }

    /// Return a mutable handle to the line per-block override cell, growing
    /// this family to full size on its first call, or `None` when the table
    /// is empty or any index is out of range.
    #[inline]
    pub fn line_override_mut(
        &mut self,
        line_idx: usize,
        stage_idx: usize,
        block_idx: usize,
    ) -> Option<&mut LineBlockOverride> {
        let idx = self.flat_index(line_idx, stage_idx, block_idx)?;
        if self.line.is_empty() {
            self.line =
                vec![LineBlockOverride::default(); self.n_lines * self.n_stages * self.max_blocks];
        }
        self.line.get_mut(idx)
    }

    /// Look up the pumping per-block override at `(pumping_idx, stage_idx, block_idx)`.
    /// Returns [`PumpingBlockOverride::default`] (all `None`) when the table
    /// is empty or any index is out of range.
    #[inline]
    #[must_use]
    pub fn pumping_override(
        &self,
        pumping_idx: usize,
        stage_idx: usize,
        block_idx: usize,
    ) -> PumpingBlockOverride {
        self.flat_index(pumping_idx, stage_idx, block_idx)
            .and_then(|idx| self.pumping.get(idx))
            .copied()
            .unwrap_or_default()
    }

    /// Return a mutable handle to the pumping per-block override cell,
    /// growing this family to full size on its first call, or `None` when
    /// the table is empty or any index is out of range.
    #[inline]
    pub fn pumping_override_mut(
        &mut self,
        pumping_idx: usize,
        stage_idx: usize,
        block_idx: usize,
    ) -> Option<&mut PumpingBlockOverride> {
        let idx = self.flat_index(pumping_idx, stage_idx, block_idx)?;
        if self.pumping.is_empty() {
            self.pumping = vec![
                PumpingBlockOverride::default();
                self.n_pumping * self.n_stages * self.max_blocks
            ];
        }
        self.pumping.get_mut(idx)
    }

    /// Look up the contract per-block override at `(contract_idx, stage_idx, block_idx)`.
    /// Returns [`ContractBlockOverride::default`] (all `None`) when the table
    /// is empty or any index is out of range.
    #[inline]
    #[must_use]
    pub fn contract_override(
        &self,
        contract_idx: usize,
        stage_idx: usize,
        block_idx: usize,
    ) -> ContractBlockOverride {
        self.flat_index(contract_idx, stage_idx, block_idx)
            .and_then(|idx| self.contract.get(idx))
            .copied()
            .unwrap_or_default()
    }

    /// Return a mutable handle to the contract per-block override cell,
    /// growing this family to full size on its first call, or `None` when
    /// the table is empty or any index is out of range.
    #[inline]
    pub fn contract_override_mut(
        &mut self,
        contract_idx: usize,
        stage_idx: usize,
        block_idx: usize,
    ) -> Option<&mut ContractBlockOverride> {
        let idx = self.flat_index(contract_idx, stage_idx, block_idx)?;
        if self.contract.is_empty() {
            self.contract = vec![
                ContractBlockOverride::default();
                self.n_contracts * self.n_stages * self.max_blocks
            ];
        }
        self.contract.get_mut(idx)
    }

    /// Returns `true` when every family table is empty.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.hydro.is_empty()
            && self.thermal.is_empty()
            && self.line.is_empty()
            && self.pumping.is_empty()
            && self.contract.is_empty()
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::{
        BlockBoundsCountsSpec, ContractBlockOverride, HydroBlockOverride, LineBlockOverride,
        PumpingBlockOverride, ResolvedBlockBounds, ThermalBlockOverride,
    };

    #[test]
    fn test_empty_block_bounds_returns_all_none_and_never_panics() {
        let table = ResolvedBlockBounds::empty();
        let result = table.thermal_override(3, 7, 2);
        assert_eq!(result, ThermalBlockOverride::default());
        assert_eq!(result.min_generation_mw, None);
        assert_eq!(result.max_generation_mw, None);
        assert!(table.is_empty());
    }

    #[test]
    fn test_block_override_write_is_visible_at_its_own_triple_only() {
        let mut table = ResolvedBlockBounds::new(&BlockBoundsCountsSpec {
            n_hydros: 0,
            n_thermals: 2,
            n_lines: 0,
            n_pumping: 0,
            n_contracts: 0,
            n_stages: 3,
            max_blocks: 3,
        });

        table
            .thermal_override_mut(1, 2, 0)
            .unwrap()
            .max_generation_mw = Some(100.0);

        assert_eq!(
            table.thermal_override(1, 2, 0).max_generation_mw,
            Some(100.0)
        );
        assert_eq!(table.thermal_override(1, 2, 1).max_generation_mw, None);
        assert_eq!(table.thermal_override(0, 2, 0).max_generation_mw, None);
        assert_eq!(table.thermal_override(1, 0, 0).max_generation_mw, None);

        // Cross-axis aliasing guard: with `n_stages == max_blocks` (as here), the
        // unparenthesized `entity_idx * n_stages + stage_idx * max_blocks +
        // block_idx` computes the identical cell for (0, 1, 0) and (1, 0, 0) —
        // this write must not be visible at the transposed triple.
        table
            .thermal_override_mut(0, 1, 0)
            .unwrap()
            .max_generation_mw = Some(55.0);
        assert_eq!(
            table.thermal_override(0, 1, 0).max_generation_mw,
            Some(55.0)
        );
        assert_eq!(table.thermal_override(1, 0, 0).max_generation_mw, None);
    }

    #[test]
    fn test_out_of_range_block_override_read_returns_default() {
        let mut table = ResolvedBlockBounds::new(&BlockBoundsCountsSpec {
            n_hydros: 2,
            n_thermals: 0,
            n_lines: 0,
            n_pumping: 0,
            n_contracts: 0,
            n_stages: 3,
            max_blocks: 3,
        });
        table.hydro_override_mut(0, 0, 0).unwrap().max_turbined_m3s = Some(42.0);

        assert_eq!(
            table.hydro_override(99, 0, 0),
            HydroBlockOverride::default()
        );
        assert_eq!(
            table.hydro_override(0, 0, 99),
            HydroBlockOverride::default()
        );
        assert!(table.hydro_override_mut(99, 0, 0).is_none());
        assert!(table.hydro_override_mut(0, 0, 99).is_none());
    }

    #[test]
    fn test_block_override_structs_pin_the_cost_price_asymmetry() {
        let hydro = HydroBlockOverride {
            min_turbined_m3s: Some(1.0),
            max_turbined_m3s: Some(2.0),
            min_outflow_m3s: Some(3.0),
            max_outflow_m3s: Some(4.0),
            min_generation_mw: Some(5.0),
            max_generation_mw: Some(6.0),
            min_diversion_m3s: Some(7.0),
            max_diversion_m3s: Some(8.0),
            min_spillage_m3s: Some(9.0),
            max_spillage_m3s: Some(10.0),
        };
        assert_eq!(hydro.min_turbined_m3s, Some(1.0));
        assert_eq!(hydro.min_diversion_m3s, Some(7.0));
        assert_eq!(hydro.max_diversion_m3s, Some(8.0));
        assert_eq!(hydro.min_spillage_m3s, Some(9.0));
        assert_eq!(hydro.max_spillage_m3s, Some(10.0));

        let thermal = ThermalBlockOverride {
            min_generation_mw: Some(1.0),
            max_generation_mw: Some(2.0),
        };
        assert_eq!(thermal.max_generation_mw, Some(2.0));

        let contract = ContractBlockOverride {
            min_mw: Some(1.0),
            max_mw: Some(2.0),
            price_per_mwh: Some(3.0),
        };
        assert_eq!(contract.price_per_mwh, Some(3.0));
    }

    #[test]
    fn test_round_trip_writes_a_distinct_value_per_family() {
        let mut table = ResolvedBlockBounds::new(&BlockBoundsCountsSpec {
            n_hydros: 1,
            n_thermals: 1,
            n_lines: 1,
            n_pumping: 1,
            n_contracts: 1,
            n_stages: 2,
            max_blocks: 2,
        });

        table.hydro_override_mut(0, 0, 0).unwrap().max_generation_mw = Some(11.0);
        table
            .thermal_override_mut(0, 0, 0)
            .unwrap()
            .max_generation_mw = Some(22.0);
        table.line_override_mut(0, 0, 0).unwrap().direct_mw = Some(33.0);
        table.pumping_override_mut(0, 0, 0).unwrap().max_flow_m3s = Some(44.0);
        table.contract_override_mut(0, 0, 0).unwrap().price_per_mwh = Some(55.0);

        assert_eq!(table.hydro_override(0, 0, 0).max_generation_mw, Some(11.0));
        assert_eq!(
            table.thermal_override(0, 0, 0).max_generation_mw,
            Some(22.0)
        );
        assert_eq!(table.line_override(0, 0, 0).direct_mw, Some(33.0));
        assert_eq!(table.pumping_override(0, 0, 0).max_flow_m3s, Some(44.0));
        assert_eq!(table.contract_override(0, 0, 0).price_per_mwh, Some(55.0));

        let l = LineBlockOverride::default();
        assert_eq!(l.reverse_mw, None);
        let p = PumpingBlockOverride::default();
        assert_eq!(p.min_flow_m3s, None);
    }
}
