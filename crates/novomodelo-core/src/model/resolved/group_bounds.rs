//! Pre-resolved per-`(hydro unit group, stage, block)` override overlay: layers
//! 1 and 2 of the bound-precedence law on the group axis. A row's optional
//! `block_id` selects one block of a stage rather than the whole stage; layer
//! 3 (the group's own declared value) stays on [`HydroUnitGroup`](crate::HydroUnitGroup)
//! and is applied by the consumer, never copied into this table. A `None`
//! field in [`HydroUnitGroupOverride`] means no override for that column.
//! Populated by `cobre-io`; never modified after construction.
//!
//! The group axis is ragged (group counts vary per plant), so it is addressed
//! as `(hydro_idx, group_pos)` — `group_pos` being the position within the
//! owning plant's `unit_groups` slice — and stored via a CSR offset vector
//! rather than a dense `max_groups_per_plant` stride.

/// Per-(stage or block) override for a hydro unit group's four declared bounds.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct HydroUnitGroupOverride {
    /// Minimum turbined flow override \[m³/s\].
    pub min_turbined_m3s: Option<f64>,
    /// Maximum turbined flow override \[m³/s\].
    pub max_turbined_m3s: Option<f64>,
    /// Minimum generation override \[MW\].
    pub min_generation_mw: Option<f64>,
    /// Maximum generation override \[MW\].
    pub max_generation_mw: Option<f64>,
}

// ─── Pre-resolved container ───────────────────────────────────────────────────

/// Group counts for constructing a [`ResolvedHydroUnitGroupBounds`] table.
#[derive(Debug, Clone)]
pub struct HydroUnitGroupBoundsCountsSpec<'a> {
    /// Number of unit groups declared by each plant, plant-index-ordered.
    pub groups_per_plant: &'a [usize],
    /// Number of time stages.
    pub n_stages: usize,
    /// Maximum `stage.blocks.len()` across study stages.
    pub max_blocks: usize,
}

/// Pre-resolved per-`(hydro unit group, stage, block)` override table.
///
/// `stage` and `block` are independently lazy: each stays empty until a row
/// actually resolves to a cell in that layer, so a study with no group-bound
/// rows — or only unresolvable ones — leaves both empty regardless of how many
/// groups `hydros` declares. [`is_empty`](Self::is_empty) reports exactly this.
///
/// # Examples
///
/// ```
/// use cobre_core::resolved::ResolvedHydroUnitGroupBounds;
///
/// let empty = ResolvedHydroUnitGroupBounds::empty();
/// assert!(empty.is_empty());
/// assert_eq!(
///     empty.override_at_block(0, 0, 0, 0),
///     Default::default()
/// );
/// ```
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ResolvedHydroUnitGroupBounds {
    n_stages: usize,
    max_blocks: usize,
    /// CSR offsets into the group axis, length `groups_per_plant.len() + 1`;
    /// plant `h`'s groups occupy slots `plant_group_start[h]..plant_group_start[h + 1]`.
    plant_group_start: Vec<usize>,
    stage: Vec<HydroUnitGroupOverride>,
    block: Vec<HydroUnitGroupOverride>,
}

impl Default for ResolvedHydroUnitGroupBounds {
    fn default() -> Self {
        Self::empty()
    }
}

impl ResolvedHydroUnitGroupBounds {
    /// Create an empty override table; every reader returns the column default
    /// and every writer returns `None`.
    #[must_use]
    pub fn empty() -> Self {
        Self {
            n_stages: 0,
            max_blocks: 0,
            plant_group_start: Vec::new(),
            stage: Vec::new(),
            block: Vec::new(),
        }
    }

    /// Allocate the group axis for a new override table. `stage` and `block`
    /// start empty — [`stage_override_mut`](Self::stage_override_mut) and
    /// [`block_override_mut`](Self::block_override_mut) grow each to full size
    /// on its own first successful write, so a table with zero applied rows
    /// stays [`is_empty`](Self::is_empty).
    #[must_use]
    pub fn new(spec: &HydroUnitGroupBoundsCountsSpec<'_>) -> Self {
        let mut plant_group_start = Vec::with_capacity(spec.groups_per_plant.len() + 1);
        plant_group_start.push(0);
        let mut total = 0usize;
        for &n in spec.groups_per_plant {
            total += n;
            plant_group_start.push(total);
        }
        Self {
            n_stages: spec.n_stages,
            max_blocks: spec.max_blocks,
            plant_group_start,
            stage: Vec::new(),
            block: Vec::new(),
        }
    }

    fn total_groups(&self) -> usize {
        self.plant_group_start.last().copied().unwrap_or(0)
    }

    /// Resolve `(hydro_idx, group_pos)` to its CSR slot, or `None` when
    /// `hydro_idx` is out of range or `group_pos` is not one of that plant's
    /// declared groups — never aliasing a sibling plant's slot.
    fn group_slot(&self, hydro_idx: usize, group_pos: usize) -> Option<usize> {
        let &start = self.plant_group_start.get(hydro_idx)?;
        let &end = self.plant_group_start.get(hydro_idx + 1)?;
        let slot = start + group_pos;
        (slot < end).then_some(slot)
    }

    fn stage_flat_index(
        &self,
        hydro_idx: usize,
        group_pos: usize,
        stage_idx: usize,
    ) -> Option<usize> {
        let slot = self.group_slot(hydro_idx, group_pos)?;
        (stage_idx < self.n_stages).then_some(slot * self.n_stages + stage_idx)
    }

    fn block_flat_index(
        &self,
        hydro_idx: usize,
        group_pos: usize,
        stage_idx: usize,
        block_idx: usize,
    ) -> Option<usize> {
        let stage_flat = self.stage_flat_index(hydro_idx, group_pos, stage_idx)?;
        (block_idx < self.max_blocks).then_some(stage_flat * self.max_blocks + block_idx)
    }

    /// Look up the stage-wide override at `(hydro_idx, group_pos, stage_idx)`.
    /// Returns [`HydroUnitGroupOverride::default`] (all `None`) when the layer
    /// is empty or any index is out of range.
    #[inline]
    #[must_use]
    pub fn stage_override(
        &self,
        hydro_idx: usize,
        group_pos: usize,
        stage_idx: usize,
    ) -> HydroUnitGroupOverride {
        self.stage_flat_index(hydro_idx, group_pos, stage_idx)
            .and_then(|idx| self.stage.get(idx))
            .copied()
            .unwrap_or_default()
    }

    /// Return a mutable handle to the stage-wide override cell, growing the
    /// layer to full size on its first call, or `None` when any index is out
    /// of range.
    #[inline]
    pub fn stage_override_mut(
        &mut self,
        hydro_idx: usize,
        group_pos: usize,
        stage_idx: usize,
    ) -> Option<&mut HydroUnitGroupOverride> {
        let idx = self.stage_flat_index(hydro_idx, group_pos, stage_idx)?;
        if self.stage.is_empty() {
            self.stage =
                vec![HydroUnitGroupOverride::default(); self.total_groups() * self.n_stages];
        }
        self.stage.get_mut(idx)
    }

    /// Look up the per-block override at `(hydro_idx, group_pos, stage_idx, block_idx)`.
    /// Returns [`HydroUnitGroupOverride::default`] (all `None`) when the layer
    /// is empty or any index is out of range.
    #[inline]
    #[must_use]
    pub fn block_override(
        &self,
        hydro_idx: usize,
        group_pos: usize,
        stage_idx: usize,
        block_idx: usize,
    ) -> HydroUnitGroupOverride {
        self.block_flat_index(hydro_idx, group_pos, stage_idx, block_idx)
            .and_then(|idx| self.block.get(idx))
            .copied()
            .unwrap_or_default()
    }

    /// Return a mutable handle to the per-block override cell, growing the
    /// layer to full size on its first call, or `None` when any index is out
    /// of range.
    #[inline]
    pub fn block_override_mut(
        &mut self,
        hydro_idx: usize,
        group_pos: usize,
        stage_idx: usize,
        block_idx: usize,
    ) -> Option<&mut HydroUnitGroupOverride> {
        let idx = self.block_flat_index(hydro_idx, group_pos, stage_idx, block_idx)?;
        if self.block.is_empty() {
            self.block = vec![
                HydroUnitGroupOverride::default();
                self.total_groups() * self.n_stages * self.max_blocks
            ];
        }
        self.block.get_mut(idx)
    }

    /// Return the resolved override at `(hydro_idx, group_pos, stage_idx, block_idx)`,
    /// applying the bound-precedence law column-independently:
    /// `block.<col>.or(stage.<col>)`. A block row touching only one column
    /// never erases a stage-wide value on another column.
    #[inline]
    #[must_use]
    pub fn override_at_block(
        &self,
        hydro_idx: usize,
        group_pos: usize,
        stage_idx: usize,
        block_idx: usize,
    ) -> HydroUnitGroupOverride {
        let stage = self.stage_override(hydro_idx, group_pos, stage_idx);
        let block = self.block_override(hydro_idx, group_pos, stage_idx, block_idx);
        HydroUnitGroupOverride {
            min_turbined_m3s: block.min_turbined_m3s.or(stage.min_turbined_m3s),
            max_turbined_m3s: block.max_turbined_m3s.or(stage.max_turbined_m3s),
            min_generation_mw: block.min_generation_mw.or(stage.min_generation_mw),
            max_generation_mw: block.max_generation_mw.or(stage.max_generation_mw),
        }
    }

    /// Returns `true` when both layers are empty — no row has ever resolved
    /// to a written cell.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.stage.is_empty() && self.block.is_empty()
    }
}
