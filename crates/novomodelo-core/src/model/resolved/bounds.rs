//! Pre-resolved per-(entity, stage) bound containers for O(1) solver lookup.
//!
//! Each block-eligible family exposes up to three accessor kinds:
//!
//! - **Stage bounds** (`<family>_bounds`) — the columns with no block
//!   dimension ([`HydroStageBounds`], [`ThermalStageBounds`]). Lines, pumping
//!   stations, and contracts have no stage-only columns, so they have no
//!   `<family>_bounds` accessor.
//! - **Block bounds** (`<family>_bounds_at_block`) — the value that applies at
//!   a named block: the block-base cell with the per-block override overlay
//!   applied on top. With an empty overlay this is bit-identical to
//!   `<family>_block_base` for every `block_index`; the empty-overlay path is
//!   never special-cased.
//! - **Base** (`<family>_block_base`) — the block-eligible columns at
//!   `(entity, stage)` granularity, ignoring the overlay. Reserved for four
//!   sanctioned caller categories: the dictionary report path
//!   (`write_bounds_parquet`'s null-`block_id` base row, all five families),
//!   the anticipated-commitment decision column
//!   (`fill_anticipated_columns`, thermal only), the admissible state-box
//!   machinery (the box builder and the initial-seed canonicalization, thermal
//!   only), and the load-time over-commitment validator
//!   (`check_committed_value_bounds`, thermal only, reading the delivery
//!   stage's commitment bound). Any other caller is a design question, not an
//!   implementation detail.
//!
//! Most entity tables share a uniform flat entity/stage layout; [`ResolvedBounds`]
//! documents the exact stride for each family, including thermal's extended one.
//! Populated by `cobre-io` after base bounds are overlaid with stage-specific
//! overrides; never modified after construction.

use super::{ResolvedBlockBounds, ResolvedHydroUnitGroupBounds};

/// Stage-level hydro bounds for a given (hydro, stage) pair — the four
/// stage-boundary-stock columns; [`HydroBlockBounds`] holds the
/// block-eligible columns instead.
///
/// Resolved from `hydros.json` overlaid with optional per-stage overrides from
/// `constraints/hydro_bounds.parquet`. Rows mirror the spec SS11 hydro bounds table.
///
/// # Examples
///
/// ```
/// use cobre_core::resolved::HydroStageBounds;
///
/// let b = HydroStageBounds {
///     min_storage_hm3: 10.0,
///     max_storage_hm3: 200.0,
///     filling_min_rate_m3s: 0.0,
///     water_withdrawal_m3s: 0.0,
/// };
/// assert!((b.min_storage_hm3 - 10.0).abs() < f64::EPSILON);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct HydroStageBounds {
    /// Operative per-stage storage floor \[hm³\] (dead volume, per-stage modif) — not the entity's physical minimum. Soft lower bound; slack `storage_violation_below`.
    pub min_storage_hm3: f64,
    /// Operative per-stage storage ceiling \[hm³\] (flood control, RHV, per-stage modif) — not the entity's physical capacity. Hard upper bound.
    pub max_storage_hm3: f64,
    /// Minimum dead-volume filling rate \[m³/s\], anchoring a per-stage minimum
    /// target-storage trajectory on `min_storage_hm3`. Not an inflow and not a cap. Default `0.0`.
    pub filling_min_rate_m3s: f64,
    /// Water withdrawal per stage \[m³/s\]. Positive = removed; negative = added. Default `0.0`.
    pub water_withdrawal_m3s: f64,
}

/// Block-eligible hydro flow/generation bounds for a given (hydro, stage)
/// pair — the capacity half of the hydro split; [`HydroStageBounds`] holds
/// the stage-boundary-stock half.
///
/// # Examples
///
/// ```
/// use cobre_core::resolved::HydroBlockBounds;
///
/// let b = HydroBlockBounds {
///     min_turbined_m3s: 0.0,
///     max_turbined_m3s: 500.0,
///     min_outflow_m3s: 5.0,
///     max_outflow_m3s: None,
///     min_generation_mw: 0.0,
///     max_generation_mw: 100.0,
///     min_diversion_m3s: None,
///     max_diversion_m3s: None,
///     min_spillage_m3s: None,
///     max_spillage_m3s: None,
/// };
/// let c = b; // Copy
/// assert!((c.max_turbined_m3s - 500.0).abs() < f64::EPSILON);
/// assert!(c.max_outflow_m3s.is_none());
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct HydroBlockBounds {
    /// Minimum turbined flow \[m³/s\]. Soft lower bound; slack `turbined_violation_below`.
    pub min_turbined_m3s: f64,
    /// Maximum turbined flow \[m³/s\]. Hard upper bound.
    pub max_turbined_m3s: f64,
    /// Environmental flow requirement \[m³/s\]. Soft lower bound; slack `outflow_violation_below`.
    pub min_outflow_m3s: f64,
    /// Flood-control limit \[m³/s\]. Soft upper bound; slack `outflow_violation_above`. `None` = unbounded.
    pub max_outflow_m3s: Option<f64>,
    /// Minimum generation \[MW\]. Soft lower bound; slack `generation_violation_below`.
    pub min_generation_mw: f64,
    /// Maximum generation \[MW\]. Hard upper bound.
    pub max_generation_mw: f64,
    /// Minimum diversion flow \[m³/s\]. `None` = unbounded below.
    pub min_diversion_m3s: Option<f64>,
    /// Maximum diversion flow \[m³/s\]. Hard upper bound. `None` = no diversion channel.
    pub max_diversion_m3s: Option<f64>,
    /// Minimum spillage flow \[m³/s\]. `None` = unbounded below.
    pub min_spillage_m3s: Option<f64>,
    /// Maximum spillage flow \[m³/s\]. `None` = unbounded above.
    pub max_spillage_m3s: Option<f64>,
}

/// Neutral test scaffold — `0.0` minima/maxima, `None` optionals — never a
/// bound source; production sites construct `HydroBlockBounds` exhaustively.
impl Default for HydroBlockBounds {
    fn default() -> Self {
        Self {
            min_turbined_m3s: 0.0,
            max_turbined_m3s: 0.0,
            min_outflow_m3s: 0.0,
            max_outflow_m3s: None,
            min_generation_mw: 0.0,
            max_generation_mw: 0.0,
            min_diversion_m3s: None,
            max_diversion_m3s: None,
            min_spillage_m3s: None,
            max_spillage_m3s: None,
        }
    }
}

/// Dense per-(hydro, stage) cell pairing [`HydroStageBounds`] and
/// [`HydroBlockBounds`] — one stride and one bounds check per hydro
/// lookup; splitting into two parallel `Vec`s could desync their lengths.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
struct HydroCell {
    stage: HydroStageBounds,
    block: HydroBlockBounds,
}

/// Block-eligible thermal generation bounds for a given (thermal, stage)
/// pair — the capacity half of the thermal split; [`ThermalStageBounds`]
/// holds the stage-level cost half.
///
/// # Examples
///
/// ```
/// use cobre_core::resolved::ThermalBlockBounds;
///
/// let b = ThermalBlockBounds { min_generation_mw: 50.0, max_generation_mw: 400.0 };
/// let c = b; // Copy
/// assert!((c.max_generation_mw - 400.0).abs() < f64::EPSILON);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ThermalBlockBounds {
    /// Minimum stable generation \[MW\]. Hard lower bound.
    pub min_generation_mw: f64,
    /// Maximum generation capacity \[MW\]. Hard upper bound.
    pub max_generation_mw: f64,
}

/// Thermal cost for a given (thermal, stage) pair; block-eligible generation
/// capacity lives in [`ThermalBlockBounds`] instead — per-block thermal cost
/// is out of scope (spec §6).
///
/// Resolved from `thermals.json` overlaid with `constraints/thermal_bounds.parquet`.
///
/// # Examples
///
/// ```
/// use cobre_core::resolved::ThermalStageBounds;
///
/// let b = ThermalStageBounds { cost_per_mwh: 120.0 };
/// let c = b; // Copy
/// assert!((c.cost_per_mwh - 120.0).abs() < f64::EPSILON);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ThermalStageBounds {
    /// Dispatch cost override (`$/MWh`). Resolved from `Thermal.cost_per_mwh` with optional
    /// per-stage override from `constraints/thermal_bounds.parquet` (null `block_id` rows only).
    pub cost_per_mwh: f64,
}

/// Dense per-(thermal, stage) cell pairing [`ThermalStageBounds`] and
/// [`ThermalBlockBounds`] — one stride and one bounds check per thermal
/// lookup; splitting into two parallel `Vec`s could desync their lengths.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
struct ThermalCell {
    stage: ThermalStageBounds,
    block: ThermalBlockBounds,
}

/// Block-eligible transmission line bounds for a given (line, stage) pair.
///
/// Lines have no stage-only bound column — every line column is block-eligible —
/// so this is line's only per-(line, stage) bound type, stored directly with no
/// stage half to pair against. Resolved from `lines.json` overlaid with
/// `constraints/line_bounds.parquet`. A per-block capacity override
/// ([`LineBlockOverride`](crate::resolved::LineBlockOverride)) wins over this
/// stage-wide value at LP construction time.
///
/// # Examples
///
/// ```
/// use cobre_core::resolved::LineBlockBounds;
///
/// let b = LineBlockBounds { direct_mw: 1000.0, reverse_mw: 800.0 };
/// let c = b; // Copy
/// assert!((c.direct_mw - 1000.0).abs() < f64::EPSILON);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LineBlockBounds {
    /// Maximum direct flow capacity \[MW\]. Hard upper bound.
    pub direct_mw: f64,
    /// Maximum reverse flow capacity \[MW\]. Hard upper bound.
    pub reverse_mw: f64,
}

/// Block-eligible pumping station bounds for a given (pumping, stage) pair.
///
/// Pumping stations have no stage-only bound column — every pumping column is
/// block-eligible — so this is pumping's only per-(pumping, stage) bound type,
/// stored directly with no stage half to pair against. Resolved from
/// `pumping_stations.json` overlaid with `constraints/pumping_bounds.parquet`.
///
/// # Examples
///
/// ```
/// use cobre_core::resolved::PumpingBlockBounds;
///
/// let b = PumpingBlockBounds { min_flow_m3s: 0.0, max_flow_m3s: 50.0 };
/// let c = b; // Copy
/// assert!((c.max_flow_m3s - 50.0).abs() < f64::EPSILON);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PumpingBlockBounds {
    /// Minimum pumped flow \[m³/s\]. Hard lower bound.
    pub min_flow_m3s: f64,
    /// Maximum pumped flow \[m³/s\]. Hard upper bound.
    pub max_flow_m3s: f64,
}

/// Block-eligible energy contract bounds for a given (contract, stage) pair.
///
/// Contracts have no stage-only bound column — every contract column,
/// including `price_per_mwh`, is block-eligible — so this is contract's only
/// per-(contract, stage) bound type, stored directly with no stage half to
/// pair against. Resolved from `energy_contracts.json` overlaid with
/// `constraints/contract_bounds.parquet`. `price_per_mwh` being block-eligible
/// is deliberately asymmetric with [`ThermalStageBounds`]'s stage-only
/// `cost_per_mwh` (spec §7 decision 6 against §6) — do not symmetrize the two.
///
/// # Examples
///
/// ```
/// use cobre_core::resolved::ContractBlockBounds;
///
/// let b = ContractBlockBounds { min_mw: 0.0, max_mw: 200.0, price_per_mwh: 80.0 };
/// let c = b; // Copy
/// assert!((c.max_mw - 200.0).abs() < f64::EPSILON);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ContractBlockBounds {
    /// Minimum contract usage \[MW\]. Hard lower bound.
    pub min_mw: f64,
    /// Maximum contract usage \[MW\]. Hard upper bound.
    pub max_mw: f64,
    /// Effective contract price \[$/`MWh`\]. May differ from base when a block override
    /// supplies a per-block price.
    pub price_per_mwh: f64,
}

// ─── Pre-resolved containers ──────────────────────────────────────────────────

/// Pre-resolved bound table for all entities across all stages.
///
/// Most tables index `data[entity_idx * n_stages + stage_idx]`, computed by the
/// private `cell_index` helper; the `thermal` table uses an extended
/// `n_stages + k_max` stride computed by `thermal_cell_index` — see
/// [`thermal_stage_axis_len`](Self::thermal_stage_axis_len).
///
/// # Examples
///
/// ```
/// use cobre_core::resolved::{
///     BoundsCountsSpec, BoundsDefaults, ContractBlockBounds, HydroBlockBounds, HydroStageBounds,
///     LineBlockBounds, PumpingBlockBounds, ResolvedBounds, ThermalBlockBounds,
///     ThermalStageBounds,
/// };
///
/// let hydro_default = HydroStageBounds {
///     min_storage_hm3: 0.0, max_storage_hm3: 100.0,
///     filling_min_rate_m3s: 0.0, water_withdrawal_m3s: 0.0,
/// };
/// let hydro_block_default = HydroBlockBounds {
///     min_turbined_m3s: 0.0, max_turbined_m3s: 50.0,
///     min_outflow_m3s: 0.0, max_outflow_m3s: None,
///     min_generation_mw: 0.0, max_generation_mw: 30.0,
///     min_diversion_m3s: None, max_diversion_m3s: None,
///     min_spillage_m3s: None, max_spillage_m3s: None,
/// };
/// let thermal_default = ThermalStageBounds { cost_per_mwh: 50.0 };
/// let thermal_block_default = ThermalBlockBounds { min_generation_mw: 0.0, max_generation_mw: 100.0 };
/// let line_default = LineBlockBounds { direct_mw: 500.0, reverse_mw: 500.0 };
/// let pumping_default = PumpingBlockBounds { min_flow_m3s: 0.0, max_flow_m3s: 20.0 };
/// let contract_default = ContractBlockBounds { min_mw: 0.0, max_mw: 50.0, price_per_mwh: 80.0 };
///
/// let table = ResolvedBounds::new(
///     &BoundsCountsSpec { n_hydros: 2, n_thermals: 1, n_lines: 1, n_pumping: 1, n_contracts: 1, n_stages: 3, k_max: 0 },
///     &BoundsDefaults {
///         hydro: hydro_default, hydro_block: hydro_block_default,
///         thermal: thermal_default, thermal_block: thermal_block_default,
///         line_block: line_default, pumping_block: pumping_default, contract_block: contract_default,
///     },
/// );
///
/// let b = table.hydro_bounds(0, 2);
/// assert!((b.max_storage_hm3 - 100.0).abs() < f64::EPSILON);
/// ```
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(try_from = "ResolvedBoundsWire"))]
pub struct ResolvedBounds {
    /// Stride for every entity table except `thermal` — see the layout note above.
    n_stages: usize,
    /// Stride for the `thermal` Vec; equals `n_stages + k_max`. Required on the
    /// wire and never defaulted: a missing or zero stride (with `thermal`
    /// non-empty) is rejected by [`ResolvedBoundsWire`]'s `TryFrom`, because
    /// defaulting to `0` would alias every thermal to thermal 0's stage block and
    /// silently return wrong bounds.
    thermal_stage_axis_len: usize,
    hydro: Vec<HydroCell>,
    /// Indexed `[thermal_idx * thermal_stage_axis_len + stage_idx]`. The stage axis
    /// is extended by `k_max` cells per thermal: `[0, n_stages)` is the study
    /// horizon, `[n_stages, n_stages + k_max)` the padding for delivery-stage
    /// lookups by anticipated-decision columns.
    thermal: Vec<ThermalCell>,
    line: Vec<LineBlockBounds>,
    pumping: Vec<PumpingBlockBounds>,
    contract: Vec<ContractBlockBounds>,
    block: ResolvedBlockBounds,
    group: ResolvedHydroUnitGroupBounds,
}

/// Deserialization shadow for [`ResolvedBounds`].
///
/// Has no `serde(default)` on `thermal_stage_axis_len`, so a missing field is
/// rejected rather than aliasing every thermal to thermal 0; the `TryFrom` below
/// also rejects a present-but-zero stride with a non-empty thermal table.
#[cfg(feature = "serde")]
#[derive(serde::Deserialize)]
struct ResolvedBoundsWire {
    n_stages: usize,
    thermal_stage_axis_len: usize,
    hydro: Vec<HydroCell>,
    thermal: Vec<ThermalCell>,
    line: Vec<LineBlockBounds>,
    pumping: Vec<PumpingBlockBounds>,
    contract: Vec<ContractBlockBounds>,
    #[serde(default)]
    block: ResolvedBlockBounds,
    #[serde(default)]
    group: ResolvedHydroUnitGroupBounds,
}

#[cfg(feature = "serde")]
impl TryFrom<ResolvedBoundsWire> for ResolvedBounds {
    type Error = String;

    fn try_from(wire: ResolvedBoundsWire) -> Result<Self, Self::Error> {
        if !wire.thermal.is_empty() && wire.thermal_stage_axis_len == 0 {
            return Err(
                "thermal_stage_axis_len must be > 0 when the thermal table is non-empty; \
                 a zero stride aliases every thermal to thermal 0"
                    .to_string(),
            );
        }
        Ok(Self {
            n_stages: wire.n_stages,
            thermal_stage_axis_len: wire.thermal_stage_axis_len,
            hydro: wire.hydro,
            thermal: wire.thermal,
            line: wire.line,
            pumping: wire.pumping,
            contract: wire.contract,
            block: wire.block,
            group: wire.group,
        })
    }
}

/// Entity counts for constructing a [`ResolvedBounds`] table.
#[derive(Debug, Clone)]
pub struct BoundsCountsSpec {
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
    /// Maximum lead-stages `K_max` across anticipated thermals; the thermal
    /// Vec stage axis is sized `n_stages + k_max`. Zero means no padding.
    pub k_max: usize,
}

/// Default per-stage bound values for each entity type.
#[derive(Debug, Clone)]
pub struct BoundsDefaults {
    /// Default stage-level hydro bounds for all (hydro, stage) cells.
    pub hydro: HydroStageBounds,
    /// Default block-eligible hydro bounds for all (hydro, stage) cells.
    pub hydro_block: HydroBlockBounds,
    /// Default thermal cost for all (thermal, stage) cells.
    pub thermal: ThermalStageBounds,
    /// Default thermal generation capacity for all (thermal, stage) cells.
    pub thermal_block: ThermalBlockBounds,
    /// Default line bounds for all (line, stage) cells.
    pub line_block: LineBlockBounds,
    /// Default pumping bounds for all (pumping, stage) cells.
    pub pumping_block: PumpingBlockBounds,
    /// Default contract bounds for all (contract, stage) cells.
    pub contract_block: ContractBlockBounds,
}

impl ResolvedBounds {
    /// Return an empty bounds table with zero entities and zero stages.
    ///
    /// The default value in [`System`](crate::System) before bound resolution.
    ///
    /// # Examples
    ///
    /// ```
    /// use cobre_core::ResolvedBounds;
    ///
    /// let empty = ResolvedBounds::empty();
    /// assert_eq!(empty.n_stages(), 0);
    /// ```
    #[must_use]
    pub fn empty() -> Self {
        Self {
            n_stages: 0,
            thermal_stage_axis_len: 0,
            hydro: Vec::new(),
            thermal: Vec::new(),
            line: Vec::new(),
            pumping: Vec::new(),
            contract: Vec::new(),
            block: ResolvedBlockBounds::empty(),
            group: ResolvedHydroUnitGroupBounds::empty(),
        }
    }

    /// Allocate a new resolved-bounds table filled with the given defaults.
    ///
    /// `counts.n_stages` must be `> 0`. Entity counts may be `0`.
    #[must_use]
    pub fn new(counts: &BoundsCountsSpec, defaults: &BoundsDefaults) -> Self {
        debug_assert!(
            counts.n_stages > 0,
            "ResolvedBounds::new: n_stages must be > 0 (got 0)"
        );
        let thermal_axis = counts.n_stages + counts.k_max;
        let thermal_cell = ThermalCell {
            stage: defaults.thermal,
            block: defaults.thermal_block,
        };
        let hydro_cell = HydroCell {
            stage: defaults.hydro,
            block: defaults.hydro_block,
        };
        Self {
            n_stages: counts.n_stages,
            thermal_stage_axis_len: thermal_axis,
            hydro: vec![hydro_cell; counts.n_hydros * counts.n_stages],
            thermal: vec![thermal_cell; counts.n_thermals * thermal_axis],
            line: vec![defaults.line_block; counts.n_lines * counts.n_stages],
            pumping: vec![defaults.pumping_block; counts.n_pumping * counts.n_stages],
            contract: vec![defaults.contract_block; counts.n_contracts * counts.n_stages],
            block: ResolvedBlockBounds::empty(),
            group: ResolvedHydroUnitGroupBounds::empty(),
        }
    }

    /// Return the flat index for `(entity_index, stage_index)` in the uniform
    /// `n_stages`-strided tables (`hydro`, `line`, `pumping`, `contract`);
    /// `thermal` uses its own extended stride — see
    /// [`thermal_cell_index`](Self::thermal_cell_index).
    #[inline]
    fn cell_index(&self, entity_index: usize, stage_index: usize) -> usize {
        debug_assert!(
            stage_index < self.n_stages,
            "stage_index out of bounds: {stage_index} >= n_stages {}",
            self.n_stages
        );
        entity_index * self.n_stages + stage_index
    }

    /// Return the resolved stage-level bounds for a hydro plant at a specific stage.
    ///
    /// Returns a reference rather than a copy to avoid copying the struct on hot paths.
    ///
    /// [`HydroStageBounds`] carries the four stage-boundary-stock columns only;
    /// the block-eligible columns live on [`HydroBlockBounds`], read through
    /// [`hydro_bounds_at_block`](Self::hydro_bounds_at_block) or
    /// [`hydro_block_base`](Self::hydro_block_base):
    ///
    /// ```compile_fail
    /// use cobre_core::ResolvedBounds;
    ///
    /// let bounds = ResolvedBounds::empty();
    /// let _ = bounds.hydro_bounds(0, 0).max_turbined_m3s;
    /// ```
    ///
    /// The sibling below differs only in the accessor — it reads the same
    /// turbined capacity through [`hydro_bounds_at_block`](Self::hydro_bounds_at_block)
    /// and compiles:
    ///
    /// ```
    /// use cobre_core::resolved::{
    ///     BoundsCountsSpec, BoundsDefaults, ContractBlockBounds, HydroBlockBounds, HydroStageBounds,
    ///     LineBlockBounds, PumpingBlockBounds, ResolvedBounds, ThermalBlockBounds,
    ///     ThermalStageBounds,
    /// };
    ///
    /// let bounds = ResolvedBounds::new(
    ///     &BoundsCountsSpec {
    ///         n_hydros: 1, n_thermals: 0, n_lines: 0, n_pumping: 0, n_contracts: 0,
    ///         n_stages: 1, k_max: 0,
    ///     },
    ///     &BoundsDefaults {
    ///         hydro: HydroStageBounds {
    ///             min_storage_hm3: 0.0, max_storage_hm3: 0.0,
    ///             filling_min_rate_m3s: 0.0, water_withdrawal_m3s: 0.0,
    ///         },
    ///         hydro_block: HydroBlockBounds {
    ///             min_turbined_m3s: 0.0, max_turbined_m3s: 500.0,
    ///             min_outflow_m3s: 0.0, max_outflow_m3s: None,
    ///             min_generation_mw: 0.0, max_generation_mw: 0.0,
    ///             min_diversion_m3s: None, max_diversion_m3s: None,
    ///             min_spillage_m3s: None, max_spillage_m3s: None,
    ///         },
    ///         thermal: ThermalStageBounds { cost_per_mwh: 0.0 },
    ///         thermal_block: ThermalBlockBounds { min_generation_mw: 0.0, max_generation_mw: 0.0 },
    ///         line_block: LineBlockBounds { direct_mw: 0.0, reverse_mw: 0.0 },
    ///         pumping_block: PumpingBlockBounds { min_flow_m3s: 0.0, max_flow_m3s: 0.0 },
    ///         contract_block: ContractBlockBounds { min_mw: 0.0, max_mw: 0.0, price_per_mwh: 0.0 },
    ///     },
    /// );
    ///
    /// let block = bounds.hydro_bounds_at_block(0, 0, 0);
    /// assert!((block.max_turbined_m3s - 500.0).abs() < f64::EPSILON);
    /// ```
    #[inline]
    #[must_use]
    pub fn hydro_bounds(&self, hydro_index: usize, stage_index: usize) -> &HydroStageBounds {
        &self.hydro[self.cell_index(hydro_index, stage_index)].stage
    }

    /// Return the flat `self.thermal` index for `(thermal_index, stage_index)`,
    /// asserting the stride invariant every thermal accessor relies on.
    #[inline]
    fn thermal_cell_index(&self, thermal_index: usize, stage_index: usize) -> usize {
        debug_assert!(
            self.thermal.is_empty() || self.thermal_stage_axis_len > 0,
            "thermal_stage_axis_len must be > 0 when the thermal table is non-empty"
        );
        thermal_index * self.thermal_stage_axis_len + stage_index
    }

    /// Return the resolved cost bounds for a thermal unit at a specific stage.
    ///
    /// `stage_index` is valid in `[0, thermal_stage_axis_len())`; indices
    /// `>= n_stages()` access the padded delivery-stage region.
    ///
    /// [`ThermalStageBounds`] carries `cost_per_mwh` only; the block-eligible
    /// generation capacity lives on [`ThermalBlockBounds`], read through
    /// [`thermal_bounds_at_block`](Self::thermal_bounds_at_block) or
    /// [`thermal_block_base`](Self::thermal_block_base):
    ///
    /// ```compile_fail
    /// use cobre_core::ResolvedBounds;
    ///
    /// let bounds = ResolvedBounds::empty();
    /// let _ = bounds.thermal_bounds(0, 0).max_generation_mw;
    /// ```
    ///
    /// The sibling below differs only in the accessor — it reads the same
    /// capacity pair through [`thermal_bounds_at_block`](Self::thermal_bounds_at_block)
    /// and compiles:
    ///
    /// ```
    /// use cobre_core::resolved::{
    ///     BoundsCountsSpec, BoundsDefaults, ContractBlockBounds, HydroBlockBounds, HydroStageBounds,
    ///     LineBlockBounds, PumpingBlockBounds, ResolvedBounds, ThermalBlockBounds,
    ///     ThermalStageBounds,
    /// };
    ///
    /// let hydro_default = HydroStageBounds {
    ///     min_storage_hm3: 0.0, max_storage_hm3: 0.0,
    ///     filling_min_rate_m3s: 0.0, water_withdrawal_m3s: 0.0,
    /// };
    /// let hydro_block_default = HydroBlockBounds {
    ///     min_turbined_m3s: 0.0, max_turbined_m3s: 0.0,
    ///     min_outflow_m3s: 0.0, max_outflow_m3s: None,
    ///     min_generation_mw: 0.0, max_generation_mw: 0.0,
    ///     min_diversion_m3s: None, max_diversion_m3s: None,
    ///     min_spillage_m3s: None, max_spillage_m3s: None,
    /// };
    ///
    /// let bounds = ResolvedBounds::new(
    ///     &BoundsCountsSpec {
    ///         n_hydros: 0, n_thermals: 1, n_lines: 0, n_pumping: 0, n_contracts: 0,
    ///         n_stages: 1, k_max: 0,
    ///     },
    ///     &BoundsDefaults {
    ///         hydro: hydro_default,
    ///         hydro_block: hydro_block_default,
    ///         thermal: ThermalStageBounds { cost_per_mwh: 50.0 },
    ///         thermal_block: ThermalBlockBounds { min_generation_mw: 0.0, max_generation_mw: 400.0 },
    ///         line_block: LineBlockBounds { direct_mw: 0.0, reverse_mw: 0.0 },
    ///         pumping_block: PumpingBlockBounds { min_flow_m3s: 0.0, max_flow_m3s: 0.0 },
    ///         contract_block: ContractBlockBounds { min_mw: 0.0, max_mw: 0.0, price_per_mwh: 0.0 },
    ///     },
    /// );
    ///
    /// let block = bounds.thermal_bounds_at_block(0, 0, 0);
    /// assert!((block.max_generation_mw - 400.0).abs() < f64::EPSILON);
    /// ```
    #[inline]
    #[must_use]
    pub fn thermal_bounds(&self, thermal_index: usize, stage_index: usize) -> ThermalStageBounds {
        self.thermal[self.thermal_cell_index(thermal_index, stage_index)].stage
    }

    /// Return a mutable reference to the hydro stage-level bounds cell for
    /// in-place update.
    #[inline]
    pub fn hydro_bounds_mut(
        &mut self,
        hydro_index: usize,
        stage_index: usize,
    ) -> &mut HydroStageBounds {
        let idx = self.cell_index(hydro_index, stage_index);
        &mut self.hydro[idx].stage
    }

    /// Return a mutable reference to the hydro block-base cell for in-place
    /// update; see [`hydro_block_base`](Self::hydro_block_base) for the
    /// overlay-ignoring read this writes through to.
    #[inline]
    pub fn hydro_block_base_mut(
        &mut self,
        hydro_index: usize,
        stage_index: usize,
    ) -> &mut HydroBlockBounds {
        let idx = self.cell_index(hydro_index, stage_index);
        &mut self.hydro[idx].block
    }

    /// Return a mutable reference to the thermal cost cell for in-place update.
    ///
    /// `stage_index` is valid in `[0, thermal_stage_axis_len())`; indices
    /// `>= n_stages()` write into the padded delivery-stage region.
    #[inline]
    pub fn thermal_bounds_mut(
        &mut self,
        thermal_index: usize,
        stage_index: usize,
    ) -> &mut ThermalStageBounds {
        let idx = self.thermal_cell_index(thermal_index, stage_index);
        &mut self.thermal[idx].stage
    }

    /// Return a mutable reference to the thermal block-base cell for in-place
    /// update; see [`thermal_block_base`](Self::thermal_block_base) for the
    /// overlay-ignoring read this writes through to.
    ///
    /// `stage_index` is valid in `[0, thermal_stage_axis_len())`; indices
    /// `>= n_stages()` write into the padded delivery-stage region.
    #[inline]
    pub fn thermal_block_base_mut(
        &mut self,
        thermal_index: usize,
        stage_index: usize,
    ) -> &mut ThermalBlockBounds {
        let idx = self.thermal_cell_index(thermal_index, stage_index);
        &mut self.thermal[idx].block
    }

    /// Return a mutable reference to the line bounds cell for in-place update.
    #[inline]
    pub fn line_bounds_mut(
        &mut self,
        line_index: usize,
        stage_index: usize,
    ) -> &mut LineBlockBounds {
        let idx = self.cell_index(line_index, stage_index);
        &mut self.line[idx]
    }

    /// Return a mutable reference to the pumping bounds cell for in-place update.
    #[inline]
    pub fn pumping_bounds_mut(
        &mut self,
        pumping_index: usize,
        stage_index: usize,
    ) -> &mut PumpingBlockBounds {
        let idx = self.cell_index(pumping_index, stage_index);
        &mut self.pumping[idx]
    }

    /// Return a mutable reference to the contract bounds cell for in-place update.
    #[inline]
    pub fn contract_bounds_mut(
        &mut self,
        contract_index: usize,
        stage_index: usize,
    ) -> &mut ContractBlockBounds {
        let idx = self.cell_index(contract_index, stage_index);
        &mut self.contract[idx]
    }

    /// Install the per-block override overlay (bound-precedence layer 1).
    ///
    /// The overlay stays [`ResolvedBlockBounds::empty`] — every `*_bounds_at_block`
    /// call falls through to the stage cell — until this is called.
    pub fn set_block_overlay(&mut self, block: ResolvedBlockBounds) {
        self.block = block;
    }

    /// Return the installed per-block override overlay.
    #[inline]
    #[must_use]
    pub fn block_overlay(&self) -> &ResolvedBlockBounds {
        &self.block
    }

    /// Return a mutable handle to the per-block override overlay.
    #[inline]
    pub fn block_overlay_mut(&mut self) -> &mut ResolvedBlockBounds {
        &mut self.block
    }

    /// Install the hydro unit group override overlay (bound-precedence layers
    /// 1 and 2 on the group axis).
    ///
    /// The overlay stays [`ResolvedHydroUnitGroupBounds::empty`] until this is
    /// called; no reader here consults it — the group axis has no consumer on
    /// [`ResolvedBounds`] itself.
    pub fn set_group_overlay(&mut self, group: ResolvedHydroUnitGroupBounds) {
        self.group = group;
    }

    /// Return the installed hydro unit group override overlay.
    #[inline]
    #[must_use]
    pub fn group_overlay(&self) -> &ResolvedHydroUnitGroupBounds {
        &self.group
    }

    /// Return a mutable handle to the hydro unit group override overlay.
    #[inline]
    pub fn group_overlay_mut(&mut self) -> &mut ResolvedHydroUnitGroupBounds {
        &mut self.group
    }

    /// Return the resolved hydro block-eligible bounds for
    /// `(hydro_index, stage_index, block_index)`, applying the block overlay
    /// over the block-base cell from [`hydro_block_base`](Self::hydro_block_base):
    /// each block-eligible column takes the overlay's value when `Some`,
    /// otherwise falls through to the block-base cell. With an empty overlay
    /// this returns a value bit-identical to `hydro_block_base` for every
    /// `block_index` — never special-case the empty-overlay path.
    #[inline]
    #[must_use]
    pub fn hydro_bounds_at_block(
        &self,
        hydro_index: usize,
        stage_index: usize,
        block_index: usize,
    ) -> HydroBlockBounds {
        let cell = self.hydro[self.cell_index(hydro_index, stage_index)].block;
        let over = self
            .block
            .hydro_override(hydro_index, stage_index, block_index);
        HydroBlockBounds {
            min_turbined_m3s: over.min_turbined_m3s.unwrap_or(cell.min_turbined_m3s),
            max_turbined_m3s: over.max_turbined_m3s.unwrap_or(cell.max_turbined_m3s),
            min_outflow_m3s: over.min_outflow_m3s.unwrap_or(cell.min_outflow_m3s),
            max_outflow_m3s: over.max_outflow_m3s.or(cell.max_outflow_m3s),
            min_generation_mw: over.min_generation_mw.unwrap_or(cell.min_generation_mw),
            max_generation_mw: over.max_generation_mw.unwrap_or(cell.max_generation_mw),
            min_diversion_m3s: over.min_diversion_m3s.or(cell.min_diversion_m3s),
            max_diversion_m3s: over.max_diversion_m3s.or(cell.max_diversion_m3s),
            min_spillage_m3s: over.min_spillage_m3s.or(cell.min_spillage_m3s),
            max_spillage_m3s: over.max_spillage_m3s.or(cell.max_spillage_m3s),
        }
    }

    /// Return the resolved thermal generation bounds for a specific block; see
    /// [`hydro_bounds_at_block`](Self::hydro_bounds_at_block) for the overlay
    /// contract. Thermal cost has no overlay column and no per-block reader —
    /// read it stage-level via [`thermal_bounds`](Self::thermal_bounds).
    #[inline]
    #[must_use]
    pub fn thermal_bounds_at_block(
        &self,
        thermal_index: usize,
        stage_index: usize,
        block_index: usize,
    ) -> ThermalBlockBounds {
        let cell = self.thermal[self.thermal_cell_index(thermal_index, stage_index)].block;
        let over = self
            .block
            .thermal_override(thermal_index, stage_index, block_index);
        ThermalBlockBounds {
            min_generation_mw: over.min_generation_mw.unwrap_or(cell.min_generation_mw),
            max_generation_mw: over.max_generation_mw.unwrap_or(cell.max_generation_mw),
        }
    }

    /// Return the resolved line bounds for a specific block; see
    /// [`hydro_bounds_at_block`](Self::hydro_bounds_at_block) for the overlay contract.
    ///
    /// Lines have no stage-level bound column — there is no `line_bounds`
    /// accessor to call. Read capacity through this accessor or
    /// [`line_block_base`](Self::line_block_base):
    ///
    /// ```compile_fail
    /// use cobre_core::ResolvedBounds;
    ///
    /// let bounds = ResolvedBounds::empty();
    /// let _ = bounds.line_bounds(0, 0).direct_mw;
    /// ```
    ///
    /// The sibling below differs only in the accessor — it reads the same line
    /// capacity through this method and compiles:
    ///
    /// ```
    /// use cobre_core::resolved::{
    ///     BoundsCountsSpec, BoundsDefaults, ContractBlockBounds, HydroBlockBounds, HydroStageBounds,
    ///     LineBlockBounds, PumpingBlockBounds, ResolvedBounds, ThermalBlockBounds,
    ///     ThermalStageBounds,
    /// };
    ///
    /// let hydro_default = HydroStageBounds {
    ///     min_storage_hm3: 0.0, max_storage_hm3: 0.0,
    ///     filling_min_rate_m3s: 0.0, water_withdrawal_m3s: 0.0,
    /// };
    /// let hydro_block_default = HydroBlockBounds {
    ///     min_turbined_m3s: 0.0, max_turbined_m3s: 0.0,
    ///     min_outflow_m3s: 0.0, max_outflow_m3s: None,
    ///     min_generation_mw: 0.0, max_generation_mw: 0.0,
    ///     min_diversion_m3s: None, max_diversion_m3s: None,
    ///     min_spillage_m3s: None, max_spillage_m3s: None,
    /// };
    ///
    /// let bounds = ResolvedBounds::new(
    ///     &BoundsCountsSpec {
    ///         n_hydros: 0, n_thermals: 0, n_lines: 1, n_pumping: 0, n_contracts: 0,
    ///         n_stages: 1, k_max: 0,
    ///     },
    ///     &BoundsDefaults {
    ///         hydro: hydro_default,
    ///         hydro_block: hydro_block_default,
    ///         thermal: ThermalStageBounds { cost_per_mwh: 0.0 },
    ///         thermal_block: ThermalBlockBounds { min_generation_mw: 0.0, max_generation_mw: 0.0 },
    ///         line_block: LineBlockBounds { direct_mw: 1000.0, reverse_mw: 800.0 },
    ///         pumping_block: PumpingBlockBounds { min_flow_m3s: 0.0, max_flow_m3s: 0.0 },
    ///         contract_block: ContractBlockBounds { min_mw: 0.0, max_mw: 0.0, price_per_mwh: 0.0 },
    ///     },
    /// );
    ///
    /// let block = bounds.line_bounds_at_block(0, 0, 0);
    /// assert!((block.direct_mw - 1000.0).abs() < f64::EPSILON);
    /// ```
    #[inline]
    #[must_use]
    pub fn line_bounds_at_block(
        &self,
        line_index: usize,
        stage_index: usize,
        block_index: usize,
    ) -> LineBlockBounds {
        let cell = self.line[self.cell_index(line_index, stage_index)];
        let over = self
            .block
            .line_override(line_index, stage_index, block_index);
        LineBlockBounds {
            direct_mw: over.direct_mw.unwrap_or(cell.direct_mw),
            reverse_mw: over.reverse_mw.unwrap_or(cell.reverse_mw),
        }
    }

    /// Return the resolved pumping bounds for a specific block; see
    /// [`hydro_bounds_at_block`](Self::hydro_bounds_at_block) for the overlay contract.
    ///
    /// Pumping stations have no stage-level bound column — there is no
    /// `pumping_bounds` accessor to call. Read flow limits through this
    /// accessor or [`pumping_block_base`](Self::pumping_block_base):
    ///
    /// ```compile_fail
    /// use cobre_core::ResolvedBounds;
    ///
    /// let bounds = ResolvedBounds::empty();
    /// let _ = bounds.pumping_bounds(0, 0).max_flow_m3s;
    /// ```
    ///
    /// The sibling below differs only in the accessor — it reads the same
    /// pumping flow limit through this method and compiles:
    ///
    /// ```
    /// use cobre_core::resolved::{
    ///     BoundsCountsSpec, BoundsDefaults, ContractBlockBounds, HydroBlockBounds, HydroStageBounds,
    ///     LineBlockBounds, PumpingBlockBounds, ResolvedBounds, ThermalBlockBounds,
    ///     ThermalStageBounds,
    /// };
    ///
    /// let hydro_default = HydroStageBounds {
    ///     min_storage_hm3: 0.0, max_storage_hm3: 0.0,
    ///     filling_min_rate_m3s: 0.0, water_withdrawal_m3s: 0.0,
    /// };
    /// let hydro_block_default = HydroBlockBounds {
    ///     min_turbined_m3s: 0.0, max_turbined_m3s: 0.0,
    ///     min_outflow_m3s: 0.0, max_outflow_m3s: None,
    ///     min_generation_mw: 0.0, max_generation_mw: 0.0,
    ///     min_diversion_m3s: None, max_diversion_m3s: None,
    ///     min_spillage_m3s: None, max_spillage_m3s: None,
    /// };
    ///
    /// let bounds = ResolvedBounds::new(
    ///     &BoundsCountsSpec {
    ///         n_hydros: 0, n_thermals: 0, n_lines: 0, n_pumping: 1, n_contracts: 0,
    ///         n_stages: 1, k_max: 0,
    ///     },
    ///     &BoundsDefaults {
    ///         hydro: hydro_default,
    ///         hydro_block: hydro_block_default,
    ///         thermal: ThermalStageBounds { cost_per_mwh: 0.0 },
    ///         thermal_block: ThermalBlockBounds { min_generation_mw: 0.0, max_generation_mw: 0.0 },
    ///         line_block: LineBlockBounds { direct_mw: 0.0, reverse_mw: 0.0 },
    ///         pumping_block: PumpingBlockBounds { min_flow_m3s: 0.0, max_flow_m3s: 50.0 },
    ///         contract_block: ContractBlockBounds { min_mw: 0.0, max_mw: 0.0, price_per_mwh: 0.0 },
    ///     },
    /// );
    ///
    /// let block = bounds.pumping_bounds_at_block(0, 0, 0);
    /// assert!((block.max_flow_m3s - 50.0).abs() < f64::EPSILON);
    /// ```
    #[inline]
    #[must_use]
    pub fn pumping_bounds_at_block(
        &self,
        pumping_index: usize,
        stage_index: usize,
        block_index: usize,
    ) -> PumpingBlockBounds {
        let cell = self.pumping[self.cell_index(pumping_index, stage_index)];
        let over = self
            .block
            .pumping_override(pumping_index, stage_index, block_index);
        PumpingBlockBounds {
            min_flow_m3s: over.min_flow_m3s.unwrap_or(cell.min_flow_m3s),
            max_flow_m3s: over.max_flow_m3s.unwrap_or(cell.max_flow_m3s),
        }
    }

    /// Return the resolved contract bounds for a specific block; see
    /// [`hydro_bounds_at_block`](Self::hydro_bounds_at_block) for the overlay
    /// contract. `price_per_mwh` IS block-eligible, deliberately asymmetric
    /// with [`thermal_bounds_at_block`](Self::thermal_bounds_at_block)'s
    /// `cost_per_mwh`.
    ///
    /// Contracts have no stage-level bound column — there is no
    /// `contract_bounds` accessor to call. Read all three columns through this
    /// accessor or [`contract_block_base`](Self::contract_block_base):
    ///
    /// ```compile_fail
    /// use cobre_core::ResolvedBounds;
    ///
    /// let bounds = ResolvedBounds::empty();
    /// let _ = bounds.contract_bounds(0, 0).price_per_mwh;
    /// ```
    ///
    /// The sibling below differs only in the accessor — it reads the same
    /// contract price through this method and compiles:
    ///
    /// ```
    /// use cobre_core::resolved::{
    ///     BoundsCountsSpec, BoundsDefaults, ContractBlockBounds, HydroBlockBounds, HydroStageBounds,
    ///     LineBlockBounds, PumpingBlockBounds, ResolvedBounds, ThermalBlockBounds,
    ///     ThermalStageBounds,
    /// };
    ///
    /// let hydro_default = HydroStageBounds {
    ///     min_storage_hm3: 0.0, max_storage_hm3: 0.0,
    ///     filling_min_rate_m3s: 0.0, water_withdrawal_m3s: 0.0,
    /// };
    /// let hydro_block_default = HydroBlockBounds {
    ///     min_turbined_m3s: 0.0, max_turbined_m3s: 0.0,
    ///     min_outflow_m3s: 0.0, max_outflow_m3s: None,
    ///     min_generation_mw: 0.0, max_generation_mw: 0.0,
    ///     min_diversion_m3s: None, max_diversion_m3s: None,
    ///     min_spillage_m3s: None, max_spillage_m3s: None,
    /// };
    ///
    /// let bounds = ResolvedBounds::new(
    ///     &BoundsCountsSpec {
    ///         n_hydros: 0, n_thermals: 0, n_lines: 0, n_pumping: 0, n_contracts: 1,
    ///         n_stages: 1, k_max: 0,
    ///     },
    ///     &BoundsDefaults {
    ///         hydro: hydro_default,
    ///         hydro_block: hydro_block_default,
    ///         thermal: ThermalStageBounds { cost_per_mwh: 0.0 },
    ///         thermal_block: ThermalBlockBounds { min_generation_mw: 0.0, max_generation_mw: 0.0 },
    ///         line_block: LineBlockBounds { direct_mw: 0.0, reverse_mw: 0.0 },
    ///         pumping_block: PumpingBlockBounds { min_flow_m3s: 0.0, max_flow_m3s: 0.0 },
    ///         contract_block: ContractBlockBounds { min_mw: 0.0, max_mw: 0.0, price_per_mwh: 80.0 },
    ///     },
    /// );
    ///
    /// let block = bounds.contract_bounds_at_block(0, 0, 0);
    /// assert!((block.price_per_mwh - 80.0).abs() < f64::EPSILON);
    /// ```
    #[inline]
    #[must_use]
    pub fn contract_bounds_at_block(
        &self,
        contract_index: usize,
        stage_index: usize,
        block_index: usize,
    ) -> ContractBlockBounds {
        let cell = self.contract[self.cell_index(contract_index, stage_index)];
        let over = self
            .block
            .contract_override(contract_index, stage_index, block_index);
        ContractBlockBounds {
            min_mw: over.min_mw.unwrap_or(cell.min_mw),
            max_mw: over.max_mw.unwrap_or(cell.max_mw),
            price_per_mwh: over.price_per_mwh.unwrap_or(cell.price_per_mwh),
        }
    }

    /// Return the block-eligible hydro columns at `(hydro_index, stage_index)`,
    /// ignoring the per-block overlay. This is **not** the value that applies at
    /// a block — see [`hydro_bounds_at_block`](Self::hydro_bounds_at_block) for
    /// the block-resolved reader. Only the dictionary path (`write_bounds_parquet`)
    /// calls this one — see the module docs for the two-caller `*_block_base`
    /// contract.
    #[inline]
    #[must_use]
    pub fn hydro_block_base(&self, hydro_index: usize, stage_index: usize) -> HydroBlockBounds {
        self.hydro[self.cell_index(hydro_index, stage_index)].block
    }

    /// Return the block-eligible thermal columns at `(thermal_index,
    /// stage_index)`, ignoring the per-block overlay. This is **not** the value
    /// that applies at a block — see
    /// [`thermal_bounds_at_block`](Self::thermal_bounds_at_block) for the
    /// block-resolved reader. `*_block_base` accessors exist for four
    /// sanctioned caller categories: the dictionary report path
    /// (`write_bounds_parquet`'s null-`block_id` base row), the
    /// anticipated-commitment decision column (`fill_anticipated_columns`,
    /// reading the delivery stage's bounds), the admissible state-box
    /// machinery (the box builder and the initial-seed canonicalization,
    /// reading the delivery stage's commitment bound), and the
    /// load-time over-commitment validator (`check_committed_value_bounds`,
    /// reading the delivery stage's commitment bound); this accessor serves
    /// them. Any other caller is a design question, not an implementation
    /// detail. `stage_index` may land in the padded delivery-stage region.
    #[inline]
    #[must_use]
    pub fn thermal_block_base(
        &self,
        thermal_index: usize,
        stage_index: usize,
    ) -> ThermalBlockBounds {
        self.thermal[self.thermal_cell_index(thermal_index, stage_index)].block
    }

    /// Return the block-eligible line columns at `(line_index, stage_index)`,
    /// ignoring the per-block overlay. This is **not** the value that applies at
    /// a block — see [`line_bounds_at_block`](Self::line_bounds_at_block) for the
    /// block-resolved reader. Only the dictionary path (`write_bounds_parquet`)
    /// calls this one — see the module docs for the two-caller `*_block_base`
    /// contract.
    #[inline]
    #[must_use]
    pub fn line_block_base(&self, line_index: usize, stage_index: usize) -> LineBlockBounds {
        self.line[self.cell_index(line_index, stage_index)]
    }

    /// Return the block-eligible pumping columns at `(pumping_index,
    /// stage_index)`, ignoring the per-block overlay. This is **not** the value
    /// that applies at a block — see
    /// [`pumping_bounds_at_block`](Self::pumping_bounds_at_block) for the
    /// block-resolved reader. Only the dictionary path (`write_bounds_parquet`)
    /// calls this one — see the module docs for the two-caller `*_block_base`
    /// contract.
    #[inline]
    #[must_use]
    pub fn pumping_block_base(
        &self,
        pumping_index: usize,
        stage_index: usize,
    ) -> PumpingBlockBounds {
        self.pumping[self.cell_index(pumping_index, stage_index)]
    }

    /// Return the block-eligible contract columns at `(contract_index,
    /// stage_index)`, ignoring the per-block overlay. This is **not** the value
    /// that applies at a block — see
    /// [`contract_bounds_at_block`](Self::contract_bounds_at_block) for the
    /// block-resolved reader. Only the dictionary path (`write_bounds_parquet`)
    /// calls this one — see the module docs for the two-caller `*_block_base`
    /// contract.
    #[inline]
    #[must_use]
    pub fn contract_block_base(
        &self,
        contract_index: usize,
        stage_index: usize,
    ) -> ContractBlockBounds {
        self.contract[self.cell_index(contract_index, stage_index)]
    }

    /// Return the number of stages in this table.
    #[inline]
    #[must_use]
    pub fn n_stages(&self) -> usize {
        self.n_stages
    }

    /// Return the number of pumping stations.
    ///
    /// Derived from the `pumping` Vec length and `n_stages` rather than a stored
    /// count, since `n_pumping` is never serialized. The `n_stages == 0` guard
    /// avoids divide-by-zero on [`ResolvedBounds::empty`].
    #[inline]
    #[must_use]
    pub fn n_pumping(&self) -> usize {
        if self.n_stages == 0 {
            0
        } else {
            self.pumping.len() / self.n_stages
        }
    }

    /// Return the number of energy contracts.
    ///
    /// Derived from the `contract` Vec length and `n_stages` rather than a stored
    /// count, since `n_contracts` is never serialized. The `n_stages == 0` guard
    /// avoids divide-by-zero on [`ResolvedBounds::empty`].
    #[inline]
    #[must_use]
    pub fn n_contracts(&self) -> usize {
        if self.n_stages == 0 {
            0
        } else {
            debug_assert_eq!(
                self.contract.len() % self.n_stages,
                0,
                "contract Vec length must be a multiple of n_stages"
            );
            self.contract.len() / self.n_stages
        }
    }

    /// Return the stride used to index the thermal Vec; equals `n_stages() + k_max`.
    ///
    /// `k_max` is the maximum lead-stages across anticipated thermals. The thermal
    /// table reserves indices `[n_stages(), thermal_stage_axis_len())` for
    /// delivery-stage lookups by anticipated-decision columns.
    #[inline]
    #[must_use]
    pub fn thermal_stage_axis_len(&self) -> usize {
        self.thermal_stage_axis_len
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::super::BlockBoundsCountsSpec;
    #[cfg(feature = "serde")]
    use super::super::HydroUnitGroupBoundsCountsSpec;
    #[cfg(feature = "serde")]
    use super::ResolvedHydroUnitGroupBounds;
    use super::{
        BoundsCountsSpec, BoundsDefaults, ContractBlockBounds, HydroBlockBounds, HydroStageBounds,
        LineBlockBounds, PumpingBlockBounds, ResolvedBlockBounds, ResolvedBounds,
        ThermalBlockBounds, ThermalStageBounds,
    };
    use crate::test_support::{f64_bits_eq, opt_f64_bits_eq};

    fn make_hydro_bounds() -> HydroStageBounds {
        HydroStageBounds {
            min_storage_hm3: 10.0,
            max_storage_hm3: 200.0,
            filling_min_rate_m3s: 0.0,
            water_withdrawal_m3s: 0.0,
        }
    }

    fn make_hydro_block_bounds() -> HydroBlockBounds {
        HydroBlockBounds {
            max_turbined_m3s: 500.0,
            min_outflow_m3s: 5.0,
            max_generation_mw: 100.0,
            ..Default::default()
        }
    }

    #[test]
    fn test_all_bound_structs_are_copy() {
        let hb = make_hydro_bounds();
        let hydro_block = make_hydro_block_bounds();
        let cost_bounds = ThermalStageBounds { cost_per_mwh: 50.0 };
        let gen_bounds = ThermalBlockBounds {
            min_generation_mw: 0.0,
            max_generation_mw: 100.0,
        };
        let lb = LineBlockBounds {
            direct_mw: 500.0,
            reverse_mw: 500.0,
        };
        let pb = PumpingBlockBounds {
            min_flow_m3s: 0.0,
            max_flow_m3s: 20.0,
        };
        let cb = ContractBlockBounds {
            min_mw: 0.0,
            max_mw: 50.0,
            price_per_mwh: 80.0,
        };

        let hb2 = hb;
        let hydro_block2 = hydro_block;
        let cost_bounds_copy = cost_bounds;
        let gen_bounds_copy = gen_bounds;
        let lb2 = lb;
        let pb2 = pb;
        let cb2 = cb;
        assert_eq!(hb, hb2);
        assert_eq!(hydro_block, hydro_block2);
        assert_eq!(cost_bounds, cost_bounds_copy);
        assert_eq!(gen_bounds, gen_bounds_copy);
        assert_eq!(lb, lb2);
        assert_eq!(pb, pb2);
        assert_eq!(cb, cb2);
    }

    #[test]
    fn test_resolved_bounds_construction() {
        let hb = make_hydro_bounds();
        let hbl = make_hydro_block_bounds();
        let tb = ThermalStageBounds { cost_per_mwh: 0.0 };
        let tbb = ThermalBlockBounds {
            min_generation_mw: 50.0,
            max_generation_mw: 400.0,
        };
        let lb = LineBlockBounds {
            direct_mw: 1000.0,
            reverse_mw: 800.0,
        };
        let pb = PumpingBlockBounds {
            min_flow_m3s: 0.0,
            max_flow_m3s: 20.0,
        };
        let cb = ContractBlockBounds {
            min_mw: 0.0,
            max_mw: 100.0,
            price_per_mwh: 80.0,
        };

        let table = ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: 1,
                n_thermals: 2,
                n_lines: 1,
                n_pumping: 1,
                n_contracts: 1,
                n_stages: 3,
                k_max: 0,
            },
            &BoundsDefaults {
                hydro: hb,
                hydro_block: hbl,
                thermal: tb,
                thermal_block: tbb,
                line_block: lb,
                pumping_block: pb,
                contract_block: cb,
            },
        );

        let b = table.hydro_bounds(0, 2);
        assert!((b.min_storage_hm3 - 10.0).abs() < f64::EPSILON);
        assert!((b.max_storage_hm3 - 200.0).abs() < f64::EPSILON);
        let block_base = table.hydro_block_base(0, 2);
        assert!(block_base.max_outflow_m3s.is_none());
        assert!(block_base.max_diversion_m3s.is_none());

        let t0 = table.thermal_block_base(0, 0);
        let t1 = table.thermal_block_base(1, 2);
        assert!((t0.max_generation_mw - 400.0).abs() < f64::EPSILON);
        assert!((t1.min_generation_mw - 50.0).abs() < f64::EPSILON);

        assert!((table.line_block_base(0, 1).direct_mw - 1000.0).abs() < f64::EPSILON);
        assert!((table.pumping_block_base(0, 0).max_flow_m3s - 20.0).abs() < f64::EPSILON);
        assert!((table.contract_block_base(0, 2).price_per_mwh - 80.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_resolved_bounds_mutable_update() {
        let hb = make_hydro_bounds();
        let hbl = make_hydro_block_bounds();
        let tb = ThermalStageBounds { cost_per_mwh: 0.0 };
        let tbb = ThermalBlockBounds {
            min_generation_mw: 0.0,
            max_generation_mw: 200.0,
        };
        let lb = LineBlockBounds {
            direct_mw: 500.0,
            reverse_mw: 500.0,
        };
        let pb = PumpingBlockBounds {
            min_flow_m3s: 0.0,
            max_flow_m3s: 30.0,
        };
        let cb = ContractBlockBounds {
            min_mw: 0.0,
            max_mw: 50.0,
            price_per_mwh: 60.0,
        };

        let mut table = ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: 2,
                n_thermals: 1,
                n_lines: 1,
                n_pumping: 1,
                n_contracts: 1,
                n_stages: 3,
                k_max: 0,
            },
            &BoundsDefaults {
                hydro: hb,
                hydro_block: hbl,
                thermal: tb,
                thermal_block: tbb,
                line_block: lb,
                pumping_block: pb,
                contract_block: cb,
            },
        );

        let cell = table.hydro_bounds_mut(1, 0);
        cell.min_storage_hm3 = 25.0;
        table.hydro_block_base_mut(1, 0).max_outflow_m3s = Some(1000.0);

        assert!((table.hydro_bounds(1, 0).min_storage_hm3 - 25.0).abs() < f64::EPSILON);
        assert_eq!(table.hydro_block_base(1, 0).max_outflow_m3s, Some(1000.0));
        assert!((table.hydro_bounds(0, 0).min_storage_hm3 - 10.0).abs() < f64::EPSILON);
        assert!(table.hydro_block_base(1, 1).max_outflow_m3s.is_none());

        table.thermal_block_base_mut(0, 2).max_generation_mw = 150.0;
        assert!((table.thermal_block_base(0, 2).max_generation_mw - 150.0).abs() < f64::EPSILON);
        assert!((table.thermal_block_base(0, 0).max_generation_mw - 200.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_thermal_stage_axis_extends_with_k_max() {
        let tbb = ThermalBlockBounds {
            min_generation_mw: 0.0,
            max_generation_mw: 100.0,
        };
        let table = ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: 0,
                n_thermals: 2,
                n_lines: 0,
                n_pumping: 0,
                n_contracts: 0,
                n_stages: 3,
                k_max: 2,
            },
            &BoundsDefaults {
                thermal_block: tbb,
                ..zero_defaults()
            },
        );
        assert_eq!(table.thermal_stage_axis_len(), 5);
        let padded = table.thermal_block_base(1, 4);
        assert!((padded.max_generation_mw - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_thermal_stage_axis_zero_k_max_unchanged() {
        let tbb = ThermalBlockBounds {
            min_generation_mw: 0.0,
            max_generation_mw: 50.0,
        };
        let table = ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: 0,
                n_thermals: 1,
                n_lines: 0,
                n_pumping: 0,
                n_contracts: 0,
                n_stages: 4,
                k_max: 0,
            },
            &BoundsDefaults {
                thermal_block: tbb,
                ..zero_defaults()
            },
        );
        assert_eq!(table.thermal_stage_axis_len(), table.n_stages());
        let last = table.thermal_block_base(0, 3);
        assert!((last.max_generation_mw - 50.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_empty_bounds_has_zero_thermal_axis() {
        let empty = ResolvedBounds::empty();
        assert_eq!(empty.thermal_stage_axis_len(), 0);
        assert_eq!(empty.n_stages(), 0);
    }

    #[test]
    fn test_n_pumping_recovers_station_count() {
        let table = ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: 0,
                n_thermals: 0,
                n_lines: 0,
                n_pumping: 2,
                n_contracts: 0,
                n_stages: 3,
                k_max: 0,
            },
            &zero_defaults(),
        );
        assert_eq!(table.n_pumping(), 2);
    }

    #[test]
    fn test_n_pumping_zero_when_no_stations() {
        let table = make_bounds_for_boundary_tests(4, 0);
        assert_eq!(table.n_pumping(), 0);
    }

    #[test]
    fn test_n_pumping_empty_table_is_zero() {
        assert_eq!(ResolvedBounds::empty().n_pumping(), 0);
    }

    // ─── Thermal-bounds padding boundary tests ───────────────────────────────
    //
    // This module verifies only the uniform `BoundsDefaults.thermal` fill; the
    // per-thermal base-fill semantics are owned by `cobre-io`'s resolution tests,
    // which construct `Thermal` entities.

    /// Sentinel defaults used by the thermal-padding boundary tests. Values are
    /// picked so an off-by-one read returns a value that does not collide with
    /// any plausible production default.
    const T_DEFAULT: ThermalStageBounds = ThermalStageBounds { cost_per_mwh: 7.7 };
    const T_BLOCK_DEFAULT: ThermalBlockBounds = ThermalBlockBounds {
        min_generation_mw: 7.0,
        max_generation_mw: 77.0,
    };

    /// Construct a `ResolvedBounds` with one thermal entity, the given
    /// `n_stages` / `k_max`, and `T_DEFAULT`/`T_BLOCK_DEFAULT` as the thermal
    /// defaults. Other entity types are zero-sized.
    fn make_bounds_for_boundary_tests(n_stages: usize, k_max: usize) -> ResolvedBounds {
        ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: 0,
                n_thermals: 1,
                n_lines: 0,
                n_pumping: 0,
                n_contracts: 0,
                n_stages,
                k_max,
            },
            &BoundsDefaults {
                thermal: T_DEFAULT,
                thermal_block: T_BLOCK_DEFAULT,
                ..zero_defaults()
            },
        )
    }

    /// `T - 1`: writing a distinctive value via `thermal_bounds_mut` at the
    /// last study stage and reading it back via `thermal_bounds` must return
    /// the written value — the padding region must not shadow study stages.
    #[test]
    fn test_thermal_bounds_at_last_study_stage() {
        let mut table = make_bounds_for_boundary_tests(5, 3);
        let written_stage = ThermalStageBounds { cost_per_mwh: 1.1 };
        let written_block = ThermalBlockBounds {
            min_generation_mw: 11.0,
            max_generation_mw: 111.0,
        };
        *table.thermal_bounds_mut(0, 4) = written_stage;
        *table.thermal_block_base_mut(0, 4) = written_block;
        let read = table.thermal_bounds(0, 4);
        let read_block = table.thermal_block_base(0, 4);
        assert!((read_block.min_generation_mw - 11.0).abs() < f64::EPSILON);
        assert!((read_block.max_generation_mw - 111.0).abs() < f64::EPSILON);
        assert!((read.cost_per_mwh - 1.1).abs() < f64::EPSILON);
    }

    /// `T`: the first padded stage must contain the uniform thermal default
    /// after `ResolvedBounds::new` — no spillover from any non-existent prior
    /// override and no zero-initialization regression.
    #[test]
    fn test_thermal_bounds_at_first_padded_stage() {
        let table = make_bounds_for_boundary_tests(5, 3);
        let padded = table.thermal_block_base(0, 5);
        assert!(
            (padded.min_generation_mw - T_BLOCK_DEFAULT.min_generation_mw).abs() < f64::EPSILON
        );
        assert!(
            (padded.max_generation_mw - T_BLOCK_DEFAULT.max_generation_mw).abs() < f64::EPSILON
        );
        assert!(
            (table.thermal_bounds(0, 5).cost_per_mwh - T_DEFAULT.cost_per_mwh).abs() < f64::EPSILON
        );
    }

    /// `T + K_max - 1`: the last padded stage must still return the uniform
    /// thermal default — the padded region is contiguous and uniform.
    #[test]
    fn test_thermal_bounds_at_last_padded_stage() {
        let table = make_bounds_for_boundary_tests(5, 3);
        let padded = table.thermal_block_base(0, 7);
        assert!(
            (padded.min_generation_mw - T_BLOCK_DEFAULT.min_generation_mw).abs() < f64::EPSILON
        );
        assert!(
            (padded.max_generation_mw - T_BLOCK_DEFAULT.max_generation_mw).abs() < f64::EPSILON
        );
        assert!(
            (table.thermal_bounds(0, 7).cost_per_mwh - T_DEFAULT.cost_per_mwh).abs() < f64::EPSILON
        );
    }

    /// `T + K_max`: one past the padding region must panic in debug builds.
    /// Gated by `#[cfg(debug_assertions)]` because release builds may silently
    /// read adjacent memory via `Vec` indexing (see `thermal_bounds` docs).
    #[test]
    #[cfg(debug_assertions)]
    fn test_thermal_bounds_out_of_range_panics_in_debug() {
        let table = make_bounds_for_boundary_tests(5, 3);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = table.thermal_bounds(0, 8);
        }));
        assert!(
            result.is_err(),
            "thermal_bounds(0, 8) must panic in debug builds when n_stages=5, k_max=3"
        );
    }

    /// `n_stages()` returns the *study horizon* length, not the padded axis.
    /// The padded region is internal to the thermal storage; consumers that
    /// iterate the study horizon (forward/backward passes, simulation) must
    /// continue to see `n_stages() == 5`.
    #[test]
    fn test_n_stages_unchanged_with_padding() {
        let table = make_bounds_for_boundary_tests(5, 3);
        assert_eq!(table.n_stages(), 5);
    }

    /// `thermal_stage_axis_len()` returns `n_stages + k_max`. This is the
    /// public accessor anticipated-decision consumers use to validate that
    /// `t + K_i` lookups remain in-range.
    #[test]
    fn test_thermal_stage_axis_len_equals_n_plus_k_max() {
        let table = make_bounds_for_boundary_tests(5, 3);
        assert_eq!(table.thermal_stage_axis_len(), 8);
    }

    /// Asserts `thermal_stage_axis_len() == n_stages + k_max` across a sweep of
    /// `(n_stages, k_max, n_thermals)` configurations.
    mod bounds_padding_invariants {
        use super::{
            BoundsCountsSpec, BoundsDefaults, ResolvedBounds, T_BLOCK_DEFAULT, T_DEFAULT,
            zero_defaults,
        };

        #[test]
        fn axis_len_matches_n_plus_k_max() {
            // n_stages starts at 1: ResolvedBounds::new debug-asserts n_stages > 0,
            // so the 0 case is exercised separately by
            // new_with_zero_n_stages_panics_in_debug.
            let n_stages_grid = [1_usize, 5, 12];
            let k_max_grid = [0_usize, 1, 3, 10];
            let n_thermals_grid = [0_usize, 1, 5];

            let mut count: usize = 0;
            for &n_stages in &n_stages_grid {
                for &k_max in &k_max_grid {
                    for &n_thermals in &n_thermals_grid {
                        let table = ResolvedBounds::new(
                            &BoundsCountsSpec {
                                n_hydros: 0,
                                n_thermals,
                                n_lines: 0,
                                n_pumping: 0,
                                n_contracts: 0,
                                n_stages,
                                k_max,
                            },
                            &BoundsDefaults {
                                thermal: T_DEFAULT,
                                thermal_block: T_BLOCK_DEFAULT,
                                ..zero_defaults()
                            },
                        );
                        assert_eq!(
                            table.thermal_stage_axis_len(),
                            n_stages + k_max,
                            "axis_len mismatch at (n_stages={n_stages}, k_max={k_max}, n_thermals={n_thermals})"
                        );
                        assert_eq!(
                            table.n_stages(),
                            n_stages,
                            "n_stages mismatch at (n_stages={n_stages}, k_max={k_max}, n_thermals={n_thermals})"
                        );
                        count += 1;
                    }
                }
            }
            // Guards against accidental loop truncation if the grids are edited.
            assert!(
                count >= 27,
                "expected at least 27 sweep combinations, got {count}"
            );
        }
    }

    /// Zero-valued defaults for every entity family; a test overrides only the
    /// families it exercises via struct-update syntax.
    fn zero_defaults() -> BoundsDefaults {
        BoundsDefaults {
            hydro: HydroStageBounds {
                min_storage_hm3: 0.0,
                max_storage_hm3: 0.0,
                filling_min_rate_m3s: 0.0,
                water_withdrawal_m3s: 0.0,
            },
            hydro_block: HydroBlockBounds::default(),
            thermal: ThermalStageBounds { cost_per_mwh: 0.0 },
            thermal_block: ThermalBlockBounds {
                min_generation_mw: 0.0,
                max_generation_mw: 0.0,
            },
            line_block: LineBlockBounds {
                direct_mw: 0.0,
                reverse_mw: 0.0,
            },
            pumping_block: PumpingBlockBounds {
                min_flow_m3s: 0.0,
                max_flow_m3s: 0.0,
            },
            contract_block: ContractBlockBounds {
                min_mw: 0.0,
                max_mw: 0.0,
                price_per_mwh: 0.0,
            },
        }
    }

    #[test]
    fn test_hydro_stage_bounds_has_four_fields() {
        let b = HydroStageBounds {
            min_storage_hm3: 1.0,
            max_storage_hm3: 2.0,
            filling_min_rate_m3s: 10.0,
            water_withdrawal_m3s: 11.0,
        };
        assert!((b.min_storage_hm3 - 1.0).abs() < f64::EPSILON);
        assert!((b.max_storage_hm3 - 2.0).abs() < f64::EPSILON);
        assert!((b.filling_min_rate_m3s - 10.0).abs() < f64::EPSILON);
        assert!((b.water_withdrawal_m3s - 11.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_hydro_block_bounds_has_ten_fields() {
        let b = HydroBlockBounds {
            min_turbined_m3s: 3.0,
            max_turbined_m3s: 4.0,
            min_outflow_m3s: 5.0,
            max_outflow_m3s: Some(6.0),
            min_generation_mw: 7.0,
            max_generation_mw: 8.0,
            min_diversion_m3s: Some(9.0),
            max_diversion_m3s: Some(10.0),
            min_spillage_m3s: Some(11.0),
            max_spillage_m3s: Some(12.0),
        };
        assert!((b.min_turbined_m3s - 3.0).abs() < f64::EPSILON);
        assert!((b.max_turbined_m3s - 4.0).abs() < f64::EPSILON);
        assert!((b.min_generation_mw - 7.0).abs() < f64::EPSILON);
        assert!((b.max_generation_mw - 8.0).abs() < f64::EPSILON);
        assert_eq!(b.max_outflow_m3s, Some(6.0));
        assert_eq!(b.min_diversion_m3s, Some(9.0));
        assert_eq!(b.max_diversion_m3s, Some(10.0));
        assert_eq!(b.min_spillage_m3s, Some(11.0));
        assert_eq!(b.max_spillage_m3s, Some(12.0));
    }

    #[test]
    #[cfg(feature = "serde")]
    fn test_resolved_bounds_serde_roundtrip() {
        let hb = make_hydro_bounds();
        let hbl = make_hydro_block_bounds();
        let tb = ThermalStageBounds { cost_per_mwh: 0.0 };
        let tbb = ThermalBlockBounds {
            min_generation_mw: 0.0,
            max_generation_mw: 100.0,
        };
        let lb = LineBlockBounds {
            direct_mw: 500.0,
            reverse_mw: 500.0,
        };
        let pb = PumpingBlockBounds {
            min_flow_m3s: 0.0,
            max_flow_m3s: 20.0,
        };
        let cb = ContractBlockBounds {
            min_mw: 0.0,
            max_mw: 50.0,
            price_per_mwh: 80.0,
        };

        let original = ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: 1,
                n_thermals: 1,
                n_lines: 1,
                n_pumping: 1,
                n_contracts: 1,
                n_stages: 3,
                k_max: 0,
            },
            &BoundsDefaults {
                hydro: hb,
                hydro_block: hbl,
                thermal: tb,
                thermal_block: tbb,
                line_block: lb,
                pumping_block: pb,
                contract_block: cb,
            },
        );
        let json = serde_json::to_string(&original).expect("serialize");
        let restored: ResolvedBounds = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(original, restored);
    }

    /// Roundtrip with a non-zero `k_max`: guards against silent data loss in
    /// the `thermal_stage_axis_len` field. With `serde(default)` on that
    /// field, an absent JSON key would deserialize back to `0`, aliasing all
    /// thermals to thermal 0's cells. This test ensures the field is actually
    /// serialized.
    #[cfg(feature = "serde")]
    #[test]
    fn test_resolved_bounds_serde_roundtrip_with_padding() {
        let hb = make_hydro_bounds();
        let hbl = make_hydro_block_bounds();
        let tb = ThermalStageBounds { cost_per_mwh: 60.0 };
        let tbb = ThermalBlockBounds {
            min_generation_mw: 0.0,
            max_generation_mw: 200.0,
        };
        let lb = LineBlockBounds {
            direct_mw: 50.0,
            reverse_mw: 50.0,
        };
        let pb = PumpingBlockBounds {
            min_flow_m3s: 0.0,
            max_flow_m3s: 20.0,
        };
        let cb = ContractBlockBounds {
            min_mw: 0.0,
            max_mw: 50.0,
            price_per_mwh: 80.0,
        };

        let original = ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: 1,
                n_thermals: 2,
                n_lines: 1,
                n_pumping: 1,
                n_contracts: 1,
                n_stages: 3,
                k_max: 2,
            },
            &BoundsDefaults {
                hydro: hb,
                hydro_block: hbl,
                thermal: tb,
                thermal_block: tbb,
                line_block: lb,
                pumping_block: pb,
                contract_block: cb,
            },
        );
        assert_eq!(original.thermal_stage_axis_len(), 5);
        let json = serde_json::to_string(&original).expect("serialize");
        let restored: ResolvedBounds = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(
            restored.thermal_stage_axis_len(),
            original.thermal_stage_axis_len(),
            "thermal_stage_axis_len must survive serde roundtrip"
        );
        assert_eq!(original, restored);
    }

    /// A JSON payload that omits `thermal_stage_axis_len` while the thermal
    /// table is non-empty must be **rejected**, not silently defaulted to `0`.
    /// A zero stride would alias every thermal to thermal 0's stage block; the
    /// `serde(try_from = "ResolvedBoundsWire")` path errors instead.
    #[cfg(feature = "serde")]
    #[test]
    fn deserialize_missing_thermal_axis_len_with_thermals_is_rejected() {
        let json = r#"{
            "n_stages": 1,
            "hydro": [],
            "thermal": [{"stage": {"cost_per_mwh": 50.0}, "block": {"min_generation_mw": 0.0, "max_generation_mw": 100.0}}],
            "line": [],
            "pumping": [],
            "contract": []
        }"#;
        let result: Result<ResolvedBounds, _> = serde_json::from_str(json);
        assert!(
            result.is_err(),
            "deserializing a non-empty thermal table without thermal_stage_axis_len \
             must error, got Ok"
        );
    }

    /// A present-but-zero `thermal_stage_axis_len` with a non-empty thermal
    /// table is also rejected by the `TryFrom` cross-field check.
    #[cfg(feature = "serde")]
    #[test]
    fn deserialize_zero_thermal_axis_len_with_thermals_is_rejected() {
        let json = r#"{
            "n_stages": 1,
            "thermal_stage_axis_len": 0,
            "hydro": [],
            "thermal": [{"stage": {"cost_per_mwh": 50.0}, "block": {"min_generation_mw": 0.0, "max_generation_mw": 100.0}}],
            "line": [],
            "pumping": [],
            "contract": []
        }"#;
        let result: Result<ResolvedBounds, _> = serde_json::from_str(json);
        assert!(
            result.is_err(),
            "deserializing a non-empty thermal table with thermal_stage_axis_len=0 \
             must error, got Ok"
        );
    }

    /// `ResolvedBounds::new` documents `n_stages > 0` as a precondition and
    /// enforces it with a `debug_assert!`. Verify the debug-build panic.
    #[test]
    #[cfg(debug_assertions)]
    fn new_with_zero_n_stages_panics_in_debug() {
        let result = std::panic::catch_unwind(|| {
            ResolvedBounds::new(
                &BoundsCountsSpec {
                    n_hydros: 1,
                    n_thermals: 1,
                    n_lines: 1,
                    n_pumping: 1,
                    n_contracts: 1,
                    n_stages: 0,
                    k_max: 0,
                },
                &zero_defaults(),
            )
        });
        assert!(
            result.is_err(),
            "ResolvedBounds::new(n_stages=0) must panic in debug builds"
        );
    }

    // ─── Block overlay tests (bound-precedence layer 1) ─────────────────────

    fn hydro_stage_bounds_bits_eq(a: &HydroStageBounds, b: &HydroStageBounds) -> bool {
        let HydroStageBounds {
            min_storage_hm3: a_min_storage_hm3,
            max_storage_hm3: a_max_storage_hm3,
            filling_min_rate_m3s: a_filling_min_rate_m3s,
            water_withdrawal_m3s: a_water_withdrawal_m3s,
        } = a;
        let HydroStageBounds {
            min_storage_hm3: b_min_storage_hm3,
            max_storage_hm3: b_max_storage_hm3,
            filling_min_rate_m3s: b_filling_min_rate_m3s,
            water_withdrawal_m3s: b_water_withdrawal_m3s,
        } = b;
        f64_bits_eq(*a_min_storage_hm3, *b_min_storage_hm3)
            && f64_bits_eq(*a_max_storage_hm3, *b_max_storage_hm3)
            && f64_bits_eq(*a_filling_min_rate_m3s, *b_filling_min_rate_m3s)
            && f64_bits_eq(*a_water_withdrawal_m3s, *b_water_withdrawal_m3s)
    }

    fn hydro_block_bounds_bits_eq(a: &HydroBlockBounds, b: &HydroBlockBounds) -> bool {
        let HydroBlockBounds {
            min_turbined_m3s: a_min_turbined_m3s,
            max_turbined_m3s: a_max_turbined_m3s,
            min_outflow_m3s: a_min_outflow_m3s,
            max_outflow_m3s: a_max_outflow_m3s,
            min_generation_mw: a_min_generation_mw,
            max_generation_mw: a_max_generation_mw,
            min_diversion_m3s: a_min_diversion_m3s,
            max_diversion_m3s: a_max_diversion_m3s,
            min_spillage_m3s: a_min_spillage_m3s,
            max_spillage_m3s: a_max_spillage_m3s,
        } = a;
        let HydroBlockBounds {
            min_turbined_m3s: b_min_turbined_m3s,
            max_turbined_m3s: b_max_turbined_m3s,
            min_outflow_m3s: b_min_outflow_m3s,
            max_outflow_m3s: b_max_outflow_m3s,
            min_generation_mw: b_min_generation_mw,
            max_generation_mw: b_max_generation_mw,
            min_diversion_m3s: b_min_diversion_m3s,
            max_diversion_m3s: b_max_diversion_m3s,
            min_spillage_m3s: b_min_spillage_m3s,
            max_spillage_m3s: b_max_spillage_m3s,
        } = b;
        f64_bits_eq(*a_min_turbined_m3s, *b_min_turbined_m3s)
            && f64_bits_eq(*a_max_turbined_m3s, *b_max_turbined_m3s)
            && f64_bits_eq(*a_min_outflow_m3s, *b_min_outflow_m3s)
            && opt_f64_bits_eq(*a_max_outflow_m3s, *b_max_outflow_m3s)
            && f64_bits_eq(*a_min_generation_mw, *b_min_generation_mw)
            && f64_bits_eq(*a_max_generation_mw, *b_max_generation_mw)
            && opt_f64_bits_eq(*a_min_diversion_m3s, *b_min_diversion_m3s)
            && opt_f64_bits_eq(*a_max_diversion_m3s, *b_max_diversion_m3s)
            && opt_f64_bits_eq(*a_min_spillage_m3s, *b_min_spillage_m3s)
            && opt_f64_bits_eq(*a_max_spillage_m3s, *b_max_spillage_m3s)
    }

    fn thermal_block_bounds_bits_eq(a: &ThermalBlockBounds, b: &ThermalBlockBounds) -> bool {
        let ThermalBlockBounds {
            min_generation_mw: a_min_generation_mw,
            max_generation_mw: a_max_generation_mw,
        } = a;
        let ThermalBlockBounds {
            min_generation_mw: b_min_generation_mw,
            max_generation_mw: b_max_generation_mw,
        } = b;
        f64_bits_eq(*a_min_generation_mw, *b_min_generation_mw)
            && f64_bits_eq(*a_max_generation_mw, *b_max_generation_mw)
    }

    fn line_bounds_bits_eq(a: &LineBlockBounds, b: &LineBlockBounds) -> bool {
        let LineBlockBounds {
            direct_mw: a_direct_mw,
            reverse_mw: a_reverse_mw,
        } = a;
        let LineBlockBounds {
            direct_mw: b_direct_mw,
            reverse_mw: b_reverse_mw,
        } = b;
        f64_bits_eq(*a_direct_mw, *b_direct_mw) && f64_bits_eq(*a_reverse_mw, *b_reverse_mw)
    }

    fn pumping_bounds_bits_eq(a: &PumpingBlockBounds, b: &PumpingBlockBounds) -> bool {
        let PumpingBlockBounds {
            min_flow_m3s: a_min_flow_m3s,
            max_flow_m3s: a_max_flow_m3s,
        } = a;
        let PumpingBlockBounds {
            min_flow_m3s: b_min_flow_m3s,
            max_flow_m3s: b_max_flow_m3s,
        } = b;
        f64_bits_eq(*a_min_flow_m3s, *b_min_flow_m3s)
            && f64_bits_eq(*a_max_flow_m3s, *b_max_flow_m3s)
    }

    fn contract_bounds_bits_eq(a: &ContractBlockBounds, b: &ContractBlockBounds) -> bool {
        let ContractBlockBounds {
            min_mw: a_min_mw,
            max_mw: a_max_mw,
            price_per_mwh: a_price_per_mwh,
        } = a;
        let ContractBlockBounds {
            min_mw: b_min_mw,
            max_mw: b_max_mw,
            price_per_mwh: b_price_per_mwh,
        } = b;
        f64_bits_eq(*a_min_mw, *b_min_mw)
            && f64_bits_eq(*a_max_mw, *b_max_mw)
            && f64_bits_eq(*a_price_per_mwh, *b_price_per_mwh)
    }

    /// Builds a table with distinct per-(entity, stage) values for every family
    /// so an indexing/stride bug in an `*_bounds_at_block` accessor surfaces as
    /// a bit mismatch rather than a coincidental match.
    #[allow(clippy::cast_precision_loss)] // entity/stage indices stay well within f64's exact-integer range
    fn make_distinct_bounds_table(n_entities: usize, n_stages: usize) -> ResolvedBounds {
        let mut table = ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: n_entities,
                n_thermals: n_entities,
                n_lines: n_entities,
                n_pumping: n_entities,
                n_contracts: n_entities,
                n_stages,
                k_max: 0,
            },
            &zero_defaults(),
        );
        for e in 0..n_entities {
            for s in 0..n_stages {
                let base = (e * 1000 + s) as f64;
                *table.hydro_bounds_mut(e, s) = HydroStageBounds {
                    min_storage_hm3: base + 1.0,
                    max_storage_hm3: base + 2.0,
                    filling_min_rate_m3s: base + 10.0,
                    water_withdrawal_m3s: base + 11.0,
                };
                *table.hydro_block_base_mut(e, s) = HydroBlockBounds {
                    min_turbined_m3s: base + 3.0,
                    max_turbined_m3s: base + 4.0,
                    min_outflow_m3s: base + 5.0,
                    max_outflow_m3s: if (e + s) % 2 == 0 {
                        Some(base + 6.0)
                    } else {
                        None
                    },
                    min_generation_mw: base + 7.0,
                    max_generation_mw: base + 8.0,
                    min_diversion_m3s: if (e + s) % 2 == 0 {
                        Some(base + 9.0)
                    } else {
                        None
                    },
                    max_diversion_m3s: if (e + s) % 2 == 0 {
                        None
                    } else {
                        Some(base + 9.0)
                    },
                    min_spillage_m3s: if (e + s) % 2 == 0 {
                        None
                    } else {
                        Some(base + 12.0)
                    },
                    max_spillage_m3s: if (e + s) % 2 == 0 {
                        Some(base + 13.0)
                    } else {
                        None
                    },
                };
                *table.thermal_bounds_mut(e, s) = ThermalStageBounds {
                    cost_per_mwh: base + 3.0,
                };
                *table.thermal_block_base_mut(e, s) = ThermalBlockBounds {
                    min_generation_mw: base + 1.0,
                    max_generation_mw: base + 2.0,
                };
                *table.line_bounds_mut(e, s) = LineBlockBounds {
                    direct_mw: base + 1.0,
                    reverse_mw: base + 2.0,
                };
                *table.pumping_bounds_mut(e, s) = PumpingBlockBounds {
                    min_flow_m3s: base + 1.0,
                    max_flow_m3s: base + 2.0,
                };
                *table.contract_bounds_mut(e, s) = ContractBlockBounds {
                    min_mw: base + 1.0,
                    max_mw: base + 2.0,
                    price_per_mwh: base + 3.0,
                };
            }
        }
        table
    }

    #[test]
    fn test_empty_overlay_block_accessor_is_bit_identical_to_stage_accessor() {
        let n_entities = 2;
        let n_stages = 3;
        let table = make_distinct_bounds_table(n_entities, n_stages);

        for e in 0..n_entities {
            for s in 0..n_stages {
                let hydro_expected = table.hydro_block_base(e, s);
                let thermal_block_expected = table.thermal_block_base(e, s);
                let line_expected = table.line_block_base(e, s);
                let pumping_expected = table.pumping_block_base(e, s);
                let contract_expected = table.contract_block_base(e, s);
                for b in 0..5 {
                    assert!(
                        hydro_block_bounds_bits_eq(
                            &hydro_expected,
                            &table.hydro_bounds_at_block(e, s, b)
                        ),
                        "hydro mismatch at (e={e}, s={s}, b={b})"
                    );
                    assert!(
                        thermal_block_bounds_bits_eq(
                            &thermal_block_expected,
                            &table.thermal_bounds_at_block(e, s, b)
                        ),
                        "thermal mismatch at (e={e}, s={s}, b={b})"
                    );
                    assert!(
                        line_bounds_bits_eq(&line_expected, &table.line_bounds_at_block(e, s, b)),
                        "line mismatch at (e={e}, s={s}, b={b})"
                    );
                    assert!(
                        pumping_bounds_bits_eq(
                            &pumping_expected,
                            &table.pumping_bounds_at_block(e, s, b)
                        ),
                        "pumping mismatch at (e={e}, s={s}, b={b})"
                    );
                    assert!(
                        contract_bounds_bits_eq(
                            &contract_expected,
                            &table.contract_bounds_at_block(e, s, b)
                        ),
                        "contract mismatch at (e={e}, s={s}, b={b})"
                    );
                }
            }
        }
    }

    #[test]
    fn test_block_override_replaces_only_its_own_column_and_block() {
        let tbb = ThermalBlockBounds {
            min_generation_mw: 10.0,
            max_generation_mw: 50.0,
        };
        let mut table = ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: 0,
                n_thermals: 1,
                n_lines: 0,
                n_pumping: 0,
                n_contracts: 0,
                n_stages: 2,
                k_max: 0,
            },
            &BoundsDefaults {
                thermal_block: tbb,
                ..zero_defaults()
            },
        );

        let mut block = ResolvedBlockBounds::new(&BlockBoundsCountsSpec {
            n_hydros: 0,
            n_thermals: 1,
            n_lines: 0,
            n_pumping: 0,
            n_contracts: 0,
            n_stages: 2,
            max_blocks: 3,
        });
        block
            .thermal_override_mut(0, 1, 2)
            .expect("in-range override cell")
            .max_generation_mw = Some(100.0);
        table.set_block_overlay(block);

        let overridden = table.thermal_bounds_at_block(0, 1, 2);
        assert!((overridden.max_generation_mw - 100.0).abs() < f64::EPSILON);
        assert!((overridden.min_generation_mw - tbb.min_generation_mw).abs() < f64::EPSILON);

        let other_block = table.thermal_bounds_at_block(0, 1, 0);
        let block_cell = table.thermal_block_base(0, 1);
        assert!(
            (other_block.min_generation_mw - block_cell.min_generation_mw).abs() < f64::EPSILON
        );
        assert!(
            (other_block.max_generation_mw - block_cell.max_generation_mw).abs() < f64::EPSILON
        );
    }

    #[test]
    fn test_thermal_block_base_ignores_the_overlay() {
        let tbb = ThermalBlockBounds {
            min_generation_mw: 10.0,
            max_generation_mw: 50.0,
        };
        let mut table = ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: 0,
                n_thermals: 1,
                n_lines: 0,
                n_pumping: 0,
                n_contracts: 0,
                n_stages: 2,
                k_max: 0,
            },
            &BoundsDefaults {
                thermal_block: tbb,
                ..zero_defaults()
            },
        );

        let mut block = ResolvedBlockBounds::new(&BlockBoundsCountsSpec {
            n_hydros: 0,
            n_thermals: 1,
            n_lines: 0,
            n_pumping: 0,
            n_contracts: 0,
            n_stages: 2,
            max_blocks: 3,
        });
        block
            .thermal_override_mut(0, 1, 0)
            .expect("in-range override cell")
            .max_generation_mw = Some(100.0);
        table.set_block_overlay(block);

        let base = table.thermal_block_base(0, 1);
        let at_block_0 = table.thermal_bounds_at_block(0, 1, 0);

        assert_eq!(
            base.max_generation_mw.to_bits(),
            tbb.max_generation_mw.to_bits(),
            "base must ignore the overlay and return the unoverridden stage cell"
        );
        assert_eq!(
            at_block_0.max_generation_mw.to_bits(),
            100.0_f64.to_bits(),
            "at-block must return the overridden value"
        );
        assert_ne!(
            base.max_generation_mw.to_bits(),
            at_block_0.max_generation_mw.to_bits(),
            "base and at-block must diverge once a block override is installed"
        );
    }

    /// Mirrors `test_thermal_block_base_ignores_the_overlay` for line: the
    /// at-block read carries the override, the base read is the unoverridden
    /// stage-wide value, compared under `f64::to_bits`.
    #[test]
    fn test_line_block_base_ignores_the_overlay() {
        let lb = LineBlockBounds {
            direct_mw: 1000.0,
            reverse_mw: 800.0,
        };
        let mut table = ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: 0,
                n_thermals: 0,
                n_lines: 1,
                n_pumping: 0,
                n_contracts: 0,
                n_stages: 2,
                k_max: 0,
            },
            &BoundsDefaults {
                line_block: lb,
                ..zero_defaults()
            },
        );

        let mut block = ResolvedBlockBounds::new(&BlockBoundsCountsSpec {
            n_hydros: 0,
            n_thermals: 0,
            n_lines: 1,
            n_pumping: 0,
            n_contracts: 0,
            n_stages: 2,
            max_blocks: 3,
        });
        block
            .line_override_mut(0, 1, 0)
            .expect("in-range override cell")
            .direct_mw = Some(400.0);
        table.set_block_overlay(block);

        let base = table.line_block_base(0, 1);
        let at_block_0 = table.line_bounds_at_block(0, 1, 0);

        assert_eq!(
            base.direct_mw.to_bits(),
            lb.direct_mw.to_bits(),
            "base must ignore the overlay and return the unoverridden stage cell"
        );
        assert_eq!(
            at_block_0.direct_mw.to_bits(),
            400.0_f64.to_bits(),
            "at-block must return the overridden value"
        );
        assert_ne!(
            base.direct_mw.to_bits(),
            at_block_0.direct_mw.to_bits(),
            "base and at-block must diverge once a block override is installed"
        );
    }

    /// Mirrors `test_thermal_block_base_ignores_the_overlay` for pumping: the
    /// at-block read carries the override, the base read is the unoverridden
    /// stage-wide value, compared under `f64::to_bits`.
    #[test]
    fn test_pumping_block_base_ignores_the_overlay() {
        let pb = PumpingBlockBounds {
            min_flow_m3s: 0.0,
            max_flow_m3s: 50.0,
        };
        let mut table = ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: 0,
                n_thermals: 0,
                n_lines: 0,
                n_pumping: 1,
                n_contracts: 0,
                n_stages: 2,
                k_max: 0,
            },
            &BoundsDefaults {
                pumping_block: pb,
                ..zero_defaults()
            },
        );

        let mut block = ResolvedBlockBounds::new(&BlockBoundsCountsSpec {
            n_hydros: 0,
            n_thermals: 0,
            n_lines: 0,
            n_pumping: 1,
            n_contracts: 0,
            n_stages: 2,
            max_blocks: 3,
        });
        block
            .pumping_override_mut(0, 1, 0)
            .expect("in-range override cell")
            .max_flow_m3s = Some(20.0);
        table.set_block_overlay(block);

        let base = table.pumping_block_base(0, 1);
        let at_block_0 = table.pumping_bounds_at_block(0, 1, 0);

        assert_eq!(
            base.max_flow_m3s.to_bits(),
            pb.max_flow_m3s.to_bits(),
            "base must ignore the overlay and return the unoverridden stage cell"
        );
        assert_eq!(
            at_block_0.max_flow_m3s.to_bits(),
            20.0_f64.to_bits(),
            "at-block must return the overridden value"
        );
        assert_ne!(
            base.max_flow_m3s.to_bits(),
            at_block_0.max_flow_m3s.to_bits(),
            "base and at-block must diverge once a block override is installed"
        );
    }

    /// Mirrors `test_thermal_block_base_ignores_the_overlay` for contract: the
    /// at-block read carries the overridden `price_per_mwh`, the base read is
    /// the unoverridden stage-wide value, compared under `f64::to_bits`.
    #[test]
    fn test_contract_block_base_ignores_the_overlay() {
        let cb = ContractBlockBounds {
            min_mw: 0.0,
            max_mw: 200.0,
            price_per_mwh: 80.0,
        };
        let mut table = ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: 0,
                n_thermals: 0,
                n_lines: 0,
                n_pumping: 0,
                n_contracts: 1,
                n_stages: 2,
                k_max: 0,
            },
            &BoundsDefaults {
                contract_block: cb,
                ..zero_defaults()
            },
        );

        let mut block = ResolvedBlockBounds::new(&BlockBoundsCountsSpec {
            n_hydros: 0,
            n_thermals: 0,
            n_lines: 0,
            n_pumping: 0,
            n_contracts: 1,
            n_stages: 2,
            max_blocks: 3,
        });
        block
            .contract_override_mut(0, 1, 0)
            .expect("in-range override cell")
            .price_per_mwh = Some(120.0);
        table.set_block_overlay(block);

        let base = table.contract_block_base(0, 1);
        let at_block_0 = table.contract_bounds_at_block(0, 1, 0);

        assert_eq!(
            base.price_per_mwh.to_bits(),
            cb.price_per_mwh.to_bits(),
            "base must ignore the overlay and return the unoverridden stage cell"
        );
        assert_eq!(
            at_block_0.price_per_mwh.to_bits(),
            120.0_f64.to_bits(),
            "at-block must return the overridden value"
        );
        assert_ne!(
            base.price_per_mwh.to_bits(),
            at_block_0.price_per_mwh.to_bits(),
            "base and at-block must diverge once a block override is installed"
        );
    }

    /// Mirrors `test_thermal_block_base_ignores_the_overlay` for hydro: the
    /// at-block read carries the overridden `max_turbined_m3s`, the base read
    /// is the unoverridden block-base cell, compared under `f64::to_bits`.
    /// Also pins that installing a block overlay never perturbs the stage
    /// cell `hydro_bounds` reads — the two halves of a [`HydroCell`] are
    /// independent axes.
    #[test]
    fn test_hydro_block_base_ignores_the_overlay() {
        let hbl = HydroBlockBounds {
            max_turbined_m3s: 500.0,
            max_generation_mw: 100.0,
            ..Default::default()
        };
        let mut table = ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: 1,
                n_thermals: 0,
                n_lines: 0,
                n_pumping: 0,
                n_contracts: 0,
                n_stages: 2,
                k_max: 0,
            },
            &BoundsDefaults {
                hydro: make_hydro_bounds(),
                hydro_block: hbl,
                ..zero_defaults()
            },
        );
        let stage_before = *table.hydro_bounds(0, 1);

        let mut block = ResolvedBlockBounds::new(&BlockBoundsCountsSpec {
            n_hydros: 1,
            n_thermals: 0,
            n_lines: 0,
            n_pumping: 0,
            n_contracts: 0,
            n_stages: 2,
            max_blocks: 3,
        });
        block
            .hydro_override_mut(0, 1, 0)
            .expect("in-range override cell")
            .max_turbined_m3s = Some(999.0);
        table.set_block_overlay(block);

        let base = table.hydro_block_base(0, 1);
        let at_block_0 = table.hydro_bounds_at_block(0, 1, 0);
        let at_block_1 = table.hydro_bounds_at_block(0, 1, 1);

        assert_eq!(
            base.max_turbined_m3s.to_bits(),
            hbl.max_turbined_m3s.to_bits(),
            "base must ignore the overlay and return the unoverridden block-base cell"
        );
        assert_eq!(
            at_block_0.max_turbined_m3s.to_bits(),
            999.0_f64.to_bits(),
            "at-block must return the overridden value"
        );
        assert_ne!(
            base.max_turbined_m3s.to_bits(),
            at_block_0.max_turbined_m3s.to_bits(),
            "base and at-block must diverge once a block override is installed"
        );
        assert_eq!(
            at_block_1.max_turbined_m3s.to_bits(),
            base.max_turbined_m3s.to_bits(),
            "a block with no override falls through to the block-base cell"
        );

        let stage_after = *table.hydro_bounds(0, 1);
        assert!(
            hydro_stage_bounds_bits_eq(&stage_before, &stage_after),
            "installing a block overlay must not perturb the stage cell"
        );
    }

    #[test]
    fn test_hydro_stage_bounds_bits_eq_distinguishes_signed_zero_and_equates_nan() {
        let positive_zero = HydroStageBounds {
            min_storage_hm3: 0.0,
            max_storage_hm3: 200.0,
            filling_min_rate_m3s: 0.0,
            water_withdrawal_m3s: 0.0,
        };
        let negative_zero = HydroStageBounds {
            min_storage_hm3: -0.0,
            ..positive_zero
        };
        assert!(
            !hydro_stage_bounds_bits_eq(&positive_zero, &negative_zero),
            "+0.0 and -0.0 carry different bit patterns and must compare unequal"
        );

        let nan_a = HydroStageBounds {
            min_storage_hm3: f64::NAN,
            ..positive_zero
        };
        let nan_b = HydroStageBounds {
            min_storage_hm3: f64::NAN,
            ..positive_zero
        };
        assert!(
            hydro_stage_bounds_bits_eq(&nan_a, &nan_b),
            "two NaN bit patterns must compare equal even though NaN != NaN under =="
        );
    }

    #[test]
    fn test_thermal_block_base_reads_padded_delivery_stage() {
        let mut table = make_bounds_for_boundary_tests(5, 3);
        let written_stage = ThermalStageBounds { cost_per_mwh: 2.1 };
        let written_block = ThermalBlockBounds {
            min_generation_mw: 21.0,
            max_generation_mw: 210.0,
        };
        *table.thermal_bounds_mut(0, 6) = written_stage;
        *table.thermal_block_base_mut(0, 6) = written_block;

        let base = table.thermal_block_base(0, 6);
        assert_eq!(
            base.min_generation_mw.to_bits(),
            written_block.min_generation_mw.to_bits()
        );
        assert_eq!(
            base.max_generation_mw.to_bits(),
            written_block.max_generation_mw.to_bits()
        );
        assert_eq!(
            table.thermal_bounds(0, 6).cost_per_mwh.to_bits(),
            written_stage.cost_per_mwh.to_bits()
        );

        let neighbor = table.thermal_block_base(0, 5);
        assert_eq!(
            neighbor.min_generation_mw.to_bits(),
            T_BLOCK_DEFAULT.min_generation_mw.to_bits(),
            "an untouched padded cell must keep the uniform default, distinguishing \
             a correct stage_index-aware read from one that ignores stage_index"
        );
    }

    #[test]
    fn test_optional_column_override_replaces_and_falls_through() {
        let mut hydro_block_default = make_hydro_block_bounds();
        hydro_block_default.max_outflow_m3s = None;

        let mut table = ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: 1,
                n_thermals: 0,
                n_lines: 0,
                n_pumping: 0,
                n_contracts: 0,
                n_stages: 2,
                k_max: 0,
            },
            &BoundsDefaults {
                hydro_block: hydro_block_default,
                ..zero_defaults()
            },
        );

        let mut block = ResolvedBlockBounds::new(&BlockBoundsCountsSpec {
            n_hydros: 1,
            n_thermals: 0,
            n_lines: 0,
            n_pumping: 0,
            n_contracts: 0,
            n_stages: 2,
            max_blocks: 2,
        });
        block
            .hydro_override_mut(0, 0, 0)
            .expect("in-range override cell")
            .max_outflow_m3s = Some(250.0);
        table.set_block_overlay(block);

        assert_eq!(
            table.hydro_bounds_at_block(0, 0, 0).max_outflow_m3s,
            Some(250.0)
        );
        assert_eq!(table.hydro_bounds_at_block(0, 0, 1).max_outflow_m3s, None);

        // `over.X.or(cell.X)` also type-checks with the operands swapped
        // (`cell.X.or(over.X)`, block-base beating the block override); with at
        // most one side `Some` above, the two are indistinguishable. Pin it
        // with both sides `Some` and distinct.
        table.hydro_block_base_mut(0, 1).max_outflow_m3s = Some(400.0);
        table.hydro_block_base_mut(0, 1).max_diversion_m3s = Some(40.0);
        table
            .block_overlay_mut()
            .hydro_override_mut(0, 1, 0)
            .expect("in-range override cell")
            .max_outflow_m3s = Some(650.0);
        table
            .block_overlay_mut()
            .hydro_override_mut(0, 1, 1)
            .expect("in-range override cell")
            .max_diversion_m3s = Some(80.0);

        assert_eq!(
            table.hydro_bounds_at_block(0, 1, 0).max_outflow_m3s,
            Some(650.0)
        );
        assert_eq!(
            table.hydro_bounds_at_block(0, 1, 1).max_diversion_m3s,
            Some(80.0)
        );
    }

    /// The three widened axes (`min_diversion_m3s`, `min_spillage_m3s`,
    /// `max_spillage_m3s`) merge via `.or(...)`, matching every other
    /// `Option<f64>` block-eligible column: a base of `None` plus an overlay
    /// `Some` on block 1 resolves to that value at block 1 only; block 0,
    /// with no override, stays `None`.
    #[test]
    fn test_widened_axes_or_precedence() {
        let mut table = ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: 1,
                n_thermals: 0,
                n_lines: 0,
                n_pumping: 0,
                n_contracts: 0,
                n_stages: 1,
                k_max: 0,
            },
            &BoundsDefaults {
                hydro_block: HydroBlockBounds::default(),
                ..zero_defaults()
            },
        );

        let mut block = ResolvedBlockBounds::new(&BlockBoundsCountsSpec {
            n_hydros: 1,
            n_thermals: 0,
            n_lines: 0,
            n_pumping: 0,
            n_contracts: 0,
            n_stages: 1,
            max_blocks: 2,
        });
        {
            let over = block
                .hydro_override_mut(0, 0, 1)
                .expect("in-range override cell");
            over.min_diversion_m3s = Some(3.0);
            over.min_spillage_m3s = Some(4.0);
            over.max_spillage_m3s = Some(5.0);
        }
        table.set_block_overlay(block);

        let block_1 = table.hydro_bounds_at_block(0, 0, 1);
        assert_eq!(block_1.min_diversion_m3s, Some(3.0));
        assert_eq!(block_1.min_spillage_m3s, Some(4.0));
        assert_eq!(block_1.max_spillage_m3s, Some(5.0));

        let block_0 = table.hydro_bounds_at_block(0, 0, 0);
        assert_eq!(block_0.min_diversion_m3s, None);
        assert_eq!(block_0.min_spillage_m3s, None);
        assert_eq!(block_0.max_spillage_m3s, None);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_resolved_bounds_wire_round_trip_with_overlay() {
        let hb = make_hydro_bounds();
        let hbl = make_hydro_block_bounds();
        let tb = ThermalStageBounds { cost_per_mwh: 20.0 };
        let tbb = ThermalBlockBounds {
            min_generation_mw: 0.0,
            max_generation_mw: 100.0,
        };
        let lb = LineBlockBounds {
            direct_mw: 500.0,
            reverse_mw: 500.0,
        };
        let pb = PumpingBlockBounds {
            min_flow_m3s: 0.0,
            max_flow_m3s: 20.0,
        };
        let cb = ContractBlockBounds {
            min_mw: 0.0,
            max_mw: 50.0,
            price_per_mwh: 80.0,
        };

        let mut original = ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: 1,
                n_thermals: 1,
                n_lines: 1,
                n_pumping: 1,
                n_contracts: 1,
                n_stages: 2,
                k_max: 0,
            },
            &BoundsDefaults {
                hydro: hb,
                hydro_block: hbl,
                thermal: tb,
                thermal_block: tbb,
                line_block: lb,
                pumping_block: pb,
                contract_block: cb,
            },
        );

        let mut block = ResolvedBlockBounds::new(&BlockBoundsCountsSpec {
            n_hydros: 1,
            n_thermals: 1,
            n_lines: 1,
            n_pumping: 1,
            n_contracts: 1,
            n_stages: 2,
            max_blocks: 2,
        });
        block
            .thermal_override_mut(0, 0, 0)
            .expect("in-range override cell")
            .max_generation_mw = Some(75.0);
        original.set_block_overlay(block);

        let json = serde_json::to_string(&original).expect("serialize json");
        let restored_json: ResolvedBounds = serde_json::from_str(&json).expect("deserialize json");
        assert_eq!(original, restored_json);

        let bytes = postcard::to_allocvec(&original).expect("serialize postcard");
        let restored_postcard: ResolvedBounds =
            postcard::from_bytes(&bytes).expect("deserialize postcard");
        assert_eq!(original, restored_postcard);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_resolved_bounds_wire_absent_overlay_defaults_to_empty() {
        let json = r#"{
            "n_stages": 1,
            "thermal_stage_axis_len": 1,
            "hydro": [],
            "thermal": [],
            "line": [],
            "pumping": [],
            "contract": []
        }"#;
        let restored: ResolvedBounds = serde_json::from_str(json).expect("deserialize");
        assert!(restored.block_overlay().is_empty());
    }

    #[cfg(feature = "serde")]
    #[test]
    fn resolved_bounds_group_overlay_round_trips_and_defaults() {
        let mut original = ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: 1,
                n_thermals: 0,
                n_lines: 0,
                n_pumping: 0,
                n_contracts: 0,
                n_stages: 2,
                k_max: 0,
            },
            &BoundsDefaults {
                hydro: make_hydro_bounds(),
                ..zero_defaults()
            },
        );

        let mut group = ResolvedHydroUnitGroupBounds::new(&HydroUnitGroupBoundsCountsSpec {
            groups_per_plant: &[2],
            n_stages: 2,
            max_blocks: 2,
        });
        group
            .stage_override_mut(0, 1, 1)
            .expect("in-range stage cell")
            .max_turbined_m3s = Some(40.0);
        group
            .block_override_mut(0, 1, 1, 0)
            .expect("in-range block cell")
            .max_turbined_m3s = Some(12.0);
        original.set_group_overlay(group);

        let json = serde_json::to_string(&original).expect("serialize");
        let restored: ResolvedBounds = serde_json::from_str(&json).expect("deserialize");

        for (hydro_idx, group_pos, stage_idx, block_idx) in
            [(0, 0, 0, 0), (0, 1, 1, 0), (0, 1, 1, 1), (0, 1, 0, 0)]
        {
            let original_over = original
                .group_overlay()
                .override_at_block(hydro_idx, group_pos, stage_idx, block_idx);
            let restored_over = restored
                .group_overlay()
                .override_at_block(hydro_idx, group_pos, stage_idx, block_idx);
            assert_eq!(
                original_over.max_turbined_m3s.map(f64::to_bits),
                restored_over.max_turbined_m3s.map(f64::to_bits),
                "mismatch at (h={hydro_idx}, g={group_pos}, t={stage_idx}, b={block_idx})"
            );
        }

        let absent_group_json = r#"{
            "n_stages": 1,
            "thermal_stage_axis_len": 1,
            "hydro": [],
            "thermal": [],
            "line": [],
            "pumping": [],
            "contract": []
        }"#;
        let restored_without_group: ResolvedBounds =
            serde_json::from_str(absent_group_json).expect("deserialize without group field");
        assert!(restored_without_group.group_overlay().is_empty());
    }
}
