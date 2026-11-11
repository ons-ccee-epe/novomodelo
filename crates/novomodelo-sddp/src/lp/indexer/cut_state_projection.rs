//! Storage-scoped projection of the global [`StateSpace`] onto the cut-state
//! dimensions a stage enables, for cut storage, subgradient extraction (incoming
//! columns), and cut rendering (outgoing columns).

use std::ops::Range;

use novomodelo_core::temporal::StageStateConfig;

use super::{CutSlot, InCol, OutCol, REGION_ORDER, StateDim, StateRegion, StateSpace};

/// Cut-slot ranges of the lag point a cut row is evaluated at: Φ(h, 0) =
/// `ẑ_h`, the realized inflow of this stage, and Φ(h, l) = `â_(h, l−1)` for
/// l ≥ 1, the incoming lag one step back. Slots are lag-major: with N hydros,
/// slot `lag0.start + l·N + h` holds (h, l).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CutLagPoint {
    lag0: Range<usize>,
    shifted: Range<usize>,
}

impl CutLagPoint {
    /// Slots of (h, 0), read against p and σ·η.
    #[inline]
    #[must_use]
    pub fn lag0(&self) -> Range<usize> {
        self.lag0.clone()
    }

    /// Slots of (h, l ≥ 1), read against â one lag back.
    #[inline]
    #[must_use]
    pub fn shifted(&self) -> Range<usize> {
        self.shifted.clone()
    }

    /// Slots of every (h, l), lag-major; `0..0` when the lag region is absent.
    #[inline]
    #[must_use]
    pub fn full(&self) -> Range<usize> {
        self.lag0.start..self.shifted.end
    }
}

/// A storage-scoped view of the global state vector exposing only the cut-state
/// dimensions a stage enables, plus travel-time buckets and anticipated state
/// (always included, never gated by [`StageStateConfig`]).
///
/// A projection **of** a [`StateSpace`], not a sibling layout: it delegates all
/// column arithmetic to the global layout and never drives the LP.
///
/// ## Default-identity contract
///
/// When `storage: true, inflow_lags: true` the projection is the identity on
/// dimensions: [`Self::n_slots`] equals [`StateSpace::n_state`], slot `j` projects
/// dimension `j`, [`Self::incoming_pairs`] yields
/// `(CutSlot(j), the global incoming column of j)` for every `j` outside the lag
/// region, and [`Self::render_pairs`] reproduces the global
/// `nonzero_state_indices` render of those dimensions exactly. Holds because the construction walks
/// the state index space in `REGION_ORDER`'s storage → lag → buckets →
/// anticipated order — the single owner [`StateSpace::set_nonzero_mask`] also
/// walks. The forbidden alternative is reordering dimensions (e.g. anticipated
/// before buckets), landing a cut coefficient on the wrong LP column and
/// silently corrupting every existing study.
///
/// ## Incoming vs outgoing vs render
///
/// Incoming ([`Self::incoming_pairs`]) and outgoing
/// ([`Self::outgoing_pairs`]) both span every slot of `[0, n_slots())` that has an
/// LP column, including structurally-zero anticipated-padding slots, because dual
/// extraction and DCS scoring read every enabled dimension; a lag slot has no LP
/// column, so neither yields it and it is never rendered.
/// Render ([`Self::render_pairs`]) is the **nonzero subset** (padding dropped,
/// matching the global `nonzero_state_indices`), because a cut row emits no entry
/// for a structurally-zero coefficient.
#[derive(Debug, Clone)]
pub struct CutStateProjection {
    /// Global state-vector [`StateDim`] per cut slot — the projection's own
    /// global→projected gather index (identity for an all-enabled pool).
    global_state_indices: Vec<StateDim>,

    /// Cut slots that have an LP column, ascending; a lag slot has none.
    columned_slots: Vec<CutSlot>,

    /// LP incoming column per columned slot (extraction hot path); parallel to
    /// [`Self::columned_slots`].
    incoming_columns: Vec<InCol>,

    /// LP outgoing column per columned slot (DCS scoring); parallel to
    /// [`Self::columned_slots`].
    outgoing_columns: Vec<OutCol>,

    /// Cut slot per render pair, nonzero-subset order; parallel to
    /// [`Self::render_columns`].
    render_coeff_indices: Vec<CutSlot>,

    /// LP outgoing column per render pair; parallel to
    /// [`Self::render_coeff_indices`].
    render_columns: Vec<OutCol>,

    /// Slot ranges of the lag region; both `0..0` when the pool holds none.
    lag_point: CutLagPoint,
}

impl CutStateProjection {
    /// Project the global [`StateSpace`] onto the cut-state dimensions
    /// `state_config` enables, with travel-time buckets and anticipated state
    /// always included, walking `REGION_ORDER`'s storage → lag → buckets →
    /// anticipated order (the default-identity contract) via
    /// `StateSpace`'s `state_dim_range`, the sole owner of the boundary
    /// arithmetic.
    ///
    /// The per-slot fan-in cannot swap roles: an outgoing column pushed onto
    /// the incoming-column vector fails to compile.
    ///
    /// ```compile_fail
    /// use novomodelo_sddp::indexer::{InCol, StateDim, StateSpace};
    ///
    /// fn misuse(global: &StateSpace) {
    ///     let outgoing = global.lp_column_for_state(StateDim::new(0));
    ///     let mut incoming_columns: Vec<InCol> = Vec::new();
    ///     incoming_columns.push(outgoing); // fan-in swap: outgoing pushed onto the incoming-column vec
    /// }
    /// ```
    #[must_use]
    pub fn new(global: &StateSpace, state_config: StageStateConfig) -> Self {
        let mut global_state_indices = Vec::new();
        let mut columned_slots = Vec::new();
        let mut incoming_columns = Vec::new();
        let mut outgoing_columns = Vec::new();
        let mut render_coeff_indices = Vec::new();
        let mut render_columns = Vec::new();

        let mut push_dim = |g: StateDim, has_columns: bool| {
            let slot = CutSlot::new(global_state_indices.len());
            global_state_indices.push(g);
            if has_columns {
                let outgoing = global.lp_column_for_state(g);
                columned_slots.push(slot);
                incoming_columns.push(global.state_to_lp_incoming_column(g));
                outgoing_columns.push(outgoing);
                // Drop padding slots, never zero-fill: keeps the default render
                // bit-identical to the global nonzero_state_indices render.
                if global.nonzero_state_indices.binary_search(&g).is_ok() {
                    render_coeff_indices.push(slot);
                    render_columns.push(outgoing);
                }
            }
        };

        for region in REGION_ORDER {
            if region.cut_enabled(state_config) {
                let has_columns = region.has_lp_columns();
                for g in global.state_dim_range(region) {
                    push_dim(StateDim::new(g), has_columns);
                }
            }
        }

        debug_assert!(
            !(state_config.storage && state_config.inflow_lags)
                || global_state_indices.len() == global.n_state,
            "default (all-enabled) projection must reproduce the global n_state \
             ({}); got {}",
            global.n_state,
            global_state_indices.len()
        );

        let lag_dims = global.state_dim_lag_range();
        let lag_point = if StateRegion::Lag.cut_enabled(state_config) && !lag_dims.is_empty() {
            let start = global_state_indices.partition_point(|d| d.get() < lag_dims.start);
            let first_shifted = start + global.hydro_count;
            CutLagPoint {
                lag0: start..first_shifted,
                shifted: first_shifted..start + lag_dims.len(),
            }
        } else {
            CutLagPoint {
                lag0: 0..0,
                shifted: 0..0,
            }
        };

        debug_assert!(
            global_state_indices.windows(2).all(|w| w[0] < w[1]),
            "global_state_indices must ascend strictly: lag_point's partition_point searches it"
        );
        debug_assert!(
            lag_point.full().is_empty()
                || global_state_indices[lag_point.full()]
                    .iter()
                    .map(|d| d.get())
                    .eq(lag_dims),
            "the slots of lag_point.full() must project the lag region, lag-major"
        );

        Self {
            global_state_indices,
            columned_slots,
            incoming_columns,
            outgoing_columns,
            render_coeff_indices,
            render_columns,
            lag_point,
        }
    }

    /// Count of enabled cut-state dimensions:
    /// `(storage ? N : 0) + (inflow_lags ? N*L : 0) + B + A*k_max`.
    #[inline]
    #[must_use]
    pub fn n_slots(&self) -> usize {
        self.global_state_indices.len()
    }

    /// Count of slots that have an LP column: [`Self::n_slots`] less the lag slots.
    #[inline]
    #[must_use]
    pub fn columned_len(&self) -> usize {
        self.columned_slots.len()
    }

    /// Slot ranges of the lag region.
    #[inline]
    #[must_use]
    pub fn lag_point(&self) -> &CutLagPoint {
        &self.lag_point
    }

    /// Whether the projection carries inflow-lag slots, the slots the chain rule writes.
    #[inline]
    #[must_use]
    pub fn has_lag_slots(&self) -> bool {
        !self.lag_point.lag0().is_empty()
    }

    #[inline]
    fn checked_slot(&self, s: CutSlot) -> usize {
        let j = s.get();
        debug_assert!(
            j < self.n_slots(),
            "cut slot {j} out of bounds (n_slots = {})",
            self.n_slots()
        );
        j
    }

    /// Map a cut slot `s ∈ [0, n_slots())` to the global [`StateDim`] it
    /// projects — the gather index for reading a `StateDim`-packed trial-state
    /// vector into the pool's projected slot space.
    ///
    /// Identity for an all-enabled pool (`s == global_state_index(s).get()`); a
    /// reduced pool selects the enabled dimensions, which are NOT a prefix (a
    /// `storage:false` pool begins at the first inflow-lag [`StateDim`]). Index a
    /// `StateDim`-packed trial-state vector through this, never the columns of
    /// [`Self::outgoing_pairs`], which skip the lag slots.
    ///
    /// Neither a [`StateDim`] nor an [`OutCol`] can stand in for the [`CutSlot`]
    /// `s`: state-dimension and cut-slot space diverge under a non-identity
    /// projection.
    ///
    /// ```compile_fail
    /// use novomodelo_core::temporal::StageStateConfig;
    /// use novomodelo_sddp::indexer::{CutStateProjection, StateDim, StateSpace};
    ///
    /// fn misuse(global: &StateSpace) {
    ///     let cut = CutStateProjection::new(
    ///         global,
    ///         StageStateConfig { storage: true, inflow_lags: true },
    ///     );
    ///     let _ = cut.global_state_index(StateDim::new(0)); // StateDim substituted for CutSlot
    /// }
    /// ```
    ///
    /// ```compile_fail
    /// use novomodelo_core::temporal::StageStateConfig;
    /// use novomodelo_sddp::indexer::{CutStateProjection, OutCol, StateSpace};
    ///
    /// fn misuse(global: &StateSpace) {
    ///     let cut = CutStateProjection::new(
    ///         global,
    ///         StageStateConfig { storage: true, inflow_lags: true },
    ///     );
    ///     let _ = cut.global_state_index(OutCol::new(0)); // OutCol substituted for CutSlot
    /// }
    /// ```
    ///
    /// # Panics (debug builds only)
    ///
    /// Panics if `s >= n_slots()`.
    #[inline]
    #[must_use]
    pub fn global_state_index(&self, s: CutSlot) -> StateDim {
        self.global_state_indices[self.checked_slot(s)]
    }

    /// Dot this pool's projected cut `coefficients` (length [`Self::n_slots`])
    /// with the full trial-state vector `x_hat` (length
    /// [`StateSpace::n_state`](super::StateSpace::n_state)), gathering each
    /// slot's value through [`Self::global_state_index`]. Cut-intercept
    /// construction (`Q(x̂) − β·x̂`) is its sole caller.
    ///
    /// A positional `coefficients.iter().zip(x_hat)` is the wrong-but-compiling
    /// alternative: it holds only for the all-enabled identity projection; once
    /// a region is dropped (a `storage:false`/`inflow_lags:false` pool), slot
    /// `j` past the gap projects a global dimension `> j`, so the zip pairs the
    /// coefficient with the wrong dimension's value — a silently over- or
    /// under-stated intercept, hence an invalid bound.
    #[inline]
    #[must_use]
    pub(crate) fn dot_trial_state(&self, coefficients: &[f64], x_hat: &[f64]) -> f64 {
        debug_assert_eq!(
            coefficients.len(),
            self.n_slots(),
            "coefficients length {} != n_slots {}",
            coefficients.len(),
            self.n_slots()
        );
        coefficients
            .iter()
            .enumerate()
            .map(|(j, &c)| c * x_hat[self.global_state_index(CutSlot::new(j)).get()])
            .sum()
    }

    /// Gather the full trial-state vector `x_hat` (length
    /// [`StateSpace::n_state`](super::StateSpace::n_state)) into `out` (length
    /// [`Self::n_slots`]) through [`Self::global_state_index`]: the single owner
    /// of "global state to projected state".
    ///
    /// The positional prefix `x_hat[..n_slots]` is the wrong-but-compiling
    /// alternative: it holds only for the all-enabled identity projection; a
    /// dropped leading region shifts every later slot onto the wrong dimension.
    #[inline]
    pub(crate) fn gather_trial_state(&self, x_hat: &[f64], out: &mut [f64]) {
        debug_assert_eq!(
            out.len(),
            self.n_slots(),
            "out length {} != n_slots {}",
            out.len(),
            self.n_slots()
        );
        for (j, value) in out.iter_mut().enumerate() {
            *value = x_hat[self.global_state_index(CutSlot::new(j)).get()];
        }
    }

    /// Every cut slot that has an LP column with its incoming-state column, in
    /// ascending slot order — storage → lag → buckets → anticipated for the
    /// enabled subset.
    ///
    /// A lag slot has no LP column and is skipped; a caller that must read every
    /// slot checks [`Self::columned_len`] against [`Self::n_slots`] first.
    #[inline]
    #[must_use]
    pub fn incoming_pairs(&self) -> impl ExactSizeIterator<Item = (CutSlot, InCol)> + '_ {
        self.columned_slots
            .iter()
            .copied()
            .zip(self.incoming_columns.iter().copied())
    }

    /// The slots of [`Self::incoming_pairs`], in the same order, with their LP
    /// **outgoing**-state column — the column a slot's coefficient is dotted
    /// against in DCS scoring.
    ///
    /// The per-pool analogue of [`StateSpace::lp_column_for_state`], spanning
    /// every columned slot (padding included); see the struct-level "Incoming vs
    /// outgoing vs render" section for the contrast with [`Self::render_pairs`].
    #[inline]
    #[must_use]
    pub fn outgoing_pairs(&self) -> impl ExactSizeIterator<Item = (CutSlot, OutCol)> + '_ {
        self.columned_slots
            .iter()
            .copied()
            .zip(self.outgoing_columns.iter().copied())
    }

    /// Iterate the cut-row render pairs `(cut_slot, outgoing_lp_column)` for this
    /// pool, in nonzero-subset order (storage → buckets → anticipated, padding
    /// slots dropped).
    ///
    /// `cut_slot` indexes a stored cut's `coefficients` slice (length
    /// [`Self::n_slots`]); `outgoing_lp_column` is where the cut-row builder places
    /// the negated, scaled coefficient. A lag slot has no LP column and is never
    /// rendered. For an all-enabled pool this reproduces the global
    /// `nonzero_state_indices` render of the columned dimensions — same reduced
    /// index, same column.
    #[inline]
    #[must_use]
    pub fn render_pairs(&self) -> impl ExactSizeIterator<Item = (CutSlot, OutCol)> + '_ {
        self.render_coeff_indices
            .iter()
            .copied()
            .zip(self.render_columns.iter().copied())
    }

    /// Count of cut-row render entries (nonzero-subset, padding dropped) — the
    /// per-cut non-zero state-coefficient count a builder emits before the theta
    /// entry.
    #[inline]
    #[must_use]
    pub fn render_len(&self) -> usize {
        self.render_coeff_indices.len()
    }

    /// θ plus the LP columns the cut rows touch: in a basic optimal solution at
    /// most this many cut rows carry a non-zero dual, so it is a reservation size
    /// for zero-free pair records, never a capacity.
    #[inline]
    #[must_use]
    pub fn structural_dual_bound(&self) -> usize {
        1 + self.render_len()
    }
}

/// Lag-region slot ranges of one chain-rule step: the cut rows of the resident
/// pool `P_t` are read, and the new cut's lag coefficients are written into the
/// extracted pool `P_(t−1)`. Each pool maps global state dimensions to its own
/// slots, so the pair view composes the two lookups by dimension, never by
/// position. It also carries the extracted pool's slot count.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChainRuleSlots {
    resident_lag0: Range<usize>,
    resident_shifted: Range<usize>,
    extracted_lag: Range<usize>,
    extracted_slots: usize,
    n_hydros: usize,
    lag_depth: usize,
}

impl ChainRuleSlots {
    /// Compose the pair view of `resident` (`P_t`, read) and `extracted`
    /// (`P_(t−1)`, written) over the one `global` layout both project. A pool
    /// without the lag region contributes an empty range.
    #[must_use]
    pub fn compose(
        global: &StateSpace,
        resident: &CutStateProjection,
        extracted: &CutStateProjection,
    ) -> Self {
        Self {
            resident_lag0: resident.lag_point().lag0(),
            resident_shifted: resident.lag_point().shifted(),
            extracted_lag: extracted.lag_point().full(),
            extracted_slots: extracted.n_slots(),
            n_hydros: global.hydro_count,
            lag_depth: global.max_par_order,
        }
    }

    /// Resident slots of (h, 0); the chain rule's `g_h` reads them.
    #[inline]
    #[must_use]
    pub fn resident_lag0(&self) -> Range<usize> {
        self.resident_lag0.clone()
    }

    /// Resident slots of (h, l ≥ 1); the chain rule's w term reads them.
    #[inline]
    #[must_use]
    pub fn resident_shifted(&self) -> Range<usize> {
        self.resident_shifted.clone()
    }

    /// Extracted slots of every (h, l), lag-major; the new cut's lag
    /// coefficients are written here.
    #[inline]
    #[must_use]
    pub fn extracted_lag(&self) -> Range<usize> {
        self.extracted_lag.clone()
    }

    /// Each run of extracted slots outside [`Self::extracted_lag`], paired with that
    /// run's positions in the lag-free state duals. The head below the lag block
    /// keeps its positions and the tail moves down by the lag width; with no lag
    /// block the head is empty and the tail covers every slot.
    #[inline]
    #[must_use]
    pub fn non_lag(&self) -> [(Range<usize>, Range<usize>); 2] {
        let lag = self.extracted_lag.clone();
        [
            (0..lag.start, 0..lag.start),
            (
                lag.end..self.extracted_slots,
                lag.start..self.extracted_slots - lag.len(),
            ),
        ]
    }

    /// Hydro count N of the shared layout.
    #[inline]
    #[must_use]
    pub fn n_hydros(&self) -> usize {
        self.n_hydros
    }

    /// Lag depth L of the shared layout.
    #[inline]
    #[must_use]
    pub fn lag_depth(&self) -> usize {
        self.lag_depth
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ChainRuleSlots, CutSlot, CutStateProjection, InCol, OutCol, StageStateConfig, StateDim,
    };
    use crate::indexer::{HydroSys, StateRegion, StateSpace};
    use crate::test_support::constant_lead_resolution;

    fn finalized(
        hydro_count: usize,
        max_par_order: usize,
        anticipated_lead_stages: &[usize],
    ) -> StateSpace {
        finalized_with_transit_buckets(
            hydro_count,
            max_par_order,
            Vec::new(),
            anticipated_lead_stages,
        )
    }

    /// Like [`finalized`] but with a declared bucket block.
    fn finalized_with_transit_buckets(
        hydro_count: usize,
        max_par_order: usize,
        transit_bucket_column_order: Vec<(HydroSys, usize)>,
        anticipated_lead_stages: &[usize],
    ) -> StateSpace {
        let lag_counts = vec![max_par_order; hydro_count];
        let n_stages = anticipated_lead_stages.iter().copied().max().unwrap_or(0) + 2;
        let resolution = constant_lead_resolution(anticipated_lead_stages, n_stages);
        StateSpace::new(
            hydro_count,
            max_par_order,
            transit_bucket_column_order,
            anticipated_lead_stages.to_owned(),
            resolution,
            &lag_counts,
        )
    }

    const ALL_ENABLED: StageStateConfig = StageStateConfig {
        storage: true,
        inflow_lags: true,
    };
    const STORAGE_ONLY: StageStateConfig = StageStateConfig {
        storage: true,
        inflow_lags: false,
    };
    const LAGS_ONLY: StageStateConfig = StageStateConfig {
        storage: false,
        inflow_lags: true,
    };

    type SlotColumns = (Vec<Option<InCol>>, Vec<Option<OutCol>>);

    /// Per-slot incoming and outgoing columns, `None` exactly at the lag slots;
    /// asserts both pair views cover every other slot.
    pub(super) fn collect_columns(cut: &CutStateProjection) -> SlotColumns {
        let lag = cut.lag_point().full();
        let mut incoming = vec![None; cut.n_slots()];
        let mut outgoing = vec![None; cut.n_slots()];
        for (slot, col) in cut.incoming_pairs() {
            incoming[slot.get()] = Some(col);
        }
        for (slot, col) in cut.outgoing_pairs() {
            outgoing[slot.get()] = Some(col);
        }
        for (j, (in_col, out_col)) in incoming.iter().zip(&outgoing).enumerate() {
            assert_eq!(in_col.is_none(), lag.contains(&j), "incoming slot {j}");
            assert_eq!(out_col.is_none(), lag.contains(&j), "outgoing slot {j}");
        }
        assert_eq!(incoming.iter().flatten().count(), cut.columned_len());
        (incoming, outgoing)
    }

    /// The global resolvers' columns of state dimension `dim`, `None` for a lag
    /// dimension, which has no LP column.
    pub(super) fn global_columns(
        global: &StateSpace,
        dim: usize,
    ) -> (Option<InCol>, Option<OutCol>) {
        if global.state_dim_lag_range().contains(&dim) {
            return (None, None);
        }
        let dim = StateDim::new(dim);
        (
            Some(global.state_to_lp_incoming_column(dim)),
            Some(global.lp_column_for_state(dim)),
        )
    }

    /// The global nonzero-mask render without the lag dimensions.
    pub(super) fn global_render(global: &StateSpace) -> Vec<(CutSlot, OutCol)> {
        let lag = global.state_dim_lag_range();
        global
            .nonzero_state_indices
            .iter()
            .filter(|g| !lag.contains(&g.get()))
            .map(|&g| (CutSlot::new(g.get()), global.lp_column_for_state(g)))
            .collect()
    }

    #[test]
    fn default_projection_is_identity() {
        let global = finalized(3, 2, &[]);
        let cut = CutStateProjection::new(&global, ALL_ENABLED);

        assert_eq!(global.n_state, 9);
        assert_eq!(cut.n_slots(), global.n_state);
        let (incoming, _) = collect_columns(&cut);
        for (j, &col) in incoming.iter().enumerate() {
            assert_eq!(
                col,
                global_columns(&global, j).0,
                "default projection must match the global resolver at j={j}"
            );
        }
    }

    #[test]
    fn storage_only_projection() {
        let global = finalized(3, 2, &[]);
        let cut = CutStateProjection::new(&global, STORAGE_ONLY);

        assert_eq!(cut.n_slots(), 3);
        let (incoming, _) = collect_columns(&cut);
        for (j, &col) in incoming.iter().enumerate() {
            assert_eq!(
                col,
                Some(InCol::new(global.storage_in.start + j)),
                "storage-only slot {j} must map to storage_in.start + {j}"
            );
        }
    }

    #[test]
    fn global_state_index_is_identity_for_all_enabled() {
        let global = finalized(3, 2, &[]);
        let cut = CutStateProjection::new(&global, ALL_ENABLED);

        assert_eq!(cut.n_slots(), global.n_state);
        for j in 0..global.n_state {
            assert_eq!(cut.global_state_index(CutSlot::new(j)), StateDim::new(j));
        }
    }

    /// `storage:false, inflow_lags:true` selects the lag dimensions, which begin
    /// at `StateDim` `N` — NOT a prefix, and where the outgoing column would remap
    /// off the `StateDim` axis; `global_state_index` selects the raw lag dims.
    #[test]
    fn global_state_index_selects_nonprefix_enabled_dims() {
        let global = finalized(2, 1, &[]);
        let cut = CutStateProjection::new(
            &global,
            StageStateConfig {
                storage: false,
                inflow_lags: true,
            },
        );

        assert_eq!(cut.n_slots(), global.hydro_count * global.max_par_order);
        for i in 0..cut.n_slots() {
            assert_eq!(
                cut.global_state_index(CutSlot::new(i)),
                StateDim::new(global.hydro_count + i),
                "lag slot {i} projects global StateDim N + {i}, not a prefix index"
            );
        }
    }

    /// A reduced projection must dot its coefficients against `x_hat` gathered
    /// through `global_state_index`, never positionally: the cut-intercept
    /// regression. `inflow_lags:false` drops the lag block, so the anticipated
    /// slot projects a global dim past its own slot index; a positional
    /// `zip(x_hat)` would multiply the anticipated coefficient by a lag value.
    #[test]
    fn dot_trial_state_gathers_reduced_projection_not_positional() {
        // N=2 storage, L=3 (6 lag dims), A=1/k_max=1 (1 anticipated): full state
        // is 9 dims — storage [0,2), lag [2,8), anticipated {8}.
        let global = finalized(2, 3, &[1]);
        let cut = CutStateProjection::new(&global, STORAGE_ONLY);

        assert_eq!(cut.n_slots(), 3, "storage(2) + anticipated(1), lag dropped");
        assert_eq!(
            cut.global_state_index(CutSlot::new(2)),
            StateDim::new(8),
            "the anticipated slot projects global dim 8, not its positional index 2"
        );

        let x_hat = vec![
            100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0,
        ];
        let coeffs = vec![1.0, 1.0, 1.0];
        assert_eq!(
            cut.dot_trial_state(&coeffs, &x_hat),
            100.0 + 101.0 + 108.0,
            "gathered dot pairs the anticipated coeff with x_hat[8], not x_hat[2]"
        );
        assert_ne!(
            cut.dot_trial_state(&coeffs, &x_hat),
            100.0 + 101.0 + 102.0,
            "a positional zip (the bug) would pair it with a lag value"
        );
    }

    /// `storage:false` drops the leading block, so the lag slots project global
    /// dimensions 2 and 3, not the positional 0 and 1.
    #[test]
    fn gather_trial_state_follows_the_projection_not_position() {
        let global = finalized(2, 1, &[]);
        let x_hat = [10.0, 11.0, 12.0, 13.0];
        assert_eq!(x_hat.len(), global.n_state);

        let lags_only = CutStateProjection::new(
            &global,
            StageStateConfig {
                storage: false,
                inflow_lags: true,
            },
        );
        let mut gathered = [0.0; 2];
        lags_only.gather_trial_state(&x_hat, &mut gathered);
        assert_eq!(gathered, [12.0, 13.0]);
        assert_ne!(gathered, x_hat[..2], "a positional prefix is the bug");

        let identity = CutStateProjection::new(&global, ALL_ENABLED);
        let mut whole = [0.0; 4];
        identity.gather_trial_state(&x_hat, &mut whole);
        assert_eq!(whole, x_hat);
    }

    /// For an all-enabled (identity) projection `dot_trial_state` reduces to the
    /// positional `zip(x_hat)` bit-for-bit — the byte-neutrality the reduced-case
    /// gather must not disturb.
    #[test]
    fn dot_trial_state_all_enabled_matches_positional_zip() {
        let global = finalized(3, 2, &[]);
        let cut = CutStateProjection::new(&global, ALL_ENABLED);

        let x_hat = vec![1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5];
        assert_eq!(x_hat.len(), global.n_state);
        let coeffs: Vec<f64> = vec![2.0, 3.0, 5.0, 7.0, 11.0, 13.0, 17.0, 19.0, 23.0];
        assert_eq!(coeffs.len(), cut.n_slots());

        let positional: f64 = coeffs.iter().zip(&x_hat).map(|(c, x)| c * x).sum();
        assert_eq!(cut.dot_trial_state(&coeffs, &x_hat), positional);
    }

    /// `inflow_lags` disabled drops the lag dims but keeps anticipated:
    /// `n_slots() = N + A*k_max = 2 + 2 = 4`.
    #[test]
    fn storage_only_with_anticipated_includes_anticipated() {
        let global = finalized(2, 1, &[2]);
        let cut = CutStateProjection::new(&global, STORAGE_ONLY);

        assert_eq!(cut.n_slots(), 4);
        let (incoming, _) = collect_columns(&cut);
        for (j, &col) in incoming.iter().take(2).enumerate() {
            assert_eq!(col, Some(InCol::new(global.storage_in.start + j)));
        }
        for i in 0..2 {
            assert_eq!(
                incoming[2 + i],
                Some(InCol::new(global.commit_in.start + i)),
                "anticipated slot {i} must map to commit_in.start + {i}"
            );
        }
    }

    /// `storage: false`: cut state begins at the first enabled dimension, the
    /// inflow lags (`N*L = 2` slots, no storage or anticipated), none of which has
    /// an LP column.
    #[test]
    fn storage_disabled_begins_at_first_enabled_dimension() {
        let global = finalized(2, 1, &[]);
        let cut = CutStateProjection::new(&global, LAGS_ONLY);

        assert_eq!(cut.n_slots(), global.hydro_count * global.max_par_order);
        assert_eq!(
            cut.global_state_index(CutSlot::new(0)),
            StateDim::new(global.state_dim_lag_range().start),
            "first enabled dimension is the inflow lags"
        );
        assert_eq!(cut.lag_point().full(), 0..cut.n_slots());
        assert_eq!(cut.columned_len(), 0);
        assert_eq!(cut.incoming_pairs().count(), 0);
        assert_eq!(cut.render_len(), 0);
    }

    // ── Outgoing projection / render-pairs tests ──────────────────────────────

    #[test]
    fn default_render_matches_global_nonzero_mask() {
        let global = finalized(3, 2, &[]);
        let cut = CutStateProjection::new(&global, ALL_ENABLED);

        assert_eq!(cut.n_slots(), global.n_state);
        let (_, outgoing) = collect_columns(&cut);
        for (j, &col) in outgoing.iter().enumerate() {
            assert_eq!(
                col,
                global_columns(&global, j).1,
                "default outgoing projection must match the global resolver at j={j}"
            );
        }

        let rendered: Vec<(CutSlot, OutCol)> = cut.render_pairs().collect();
        let expected = global_render(&global);
        assert_eq!(
            rendered, expected,
            "render_pairs must reproduce the global nonzero_state_indices render \
             without the lag dimensions"
        );
        assert_eq!(cut.render_len(), expected.len());
    }

    #[test]
    fn storage_only_render_touches_only_storage_columns() {
        let global = finalized(3, 2, &[]);
        let cut = CutStateProjection::new(&global, STORAGE_ONLY);

        assert_eq!(cut.n_slots(), 3);
        assert_eq!(cut.render_len(), 3);

        let rendered: Vec<(CutSlot, OutCol)> = cut.render_pairs().collect();
        // Storage is identity under the outgoing resolver: state_to_lp_column(j)=j.
        assert_eq!(
            rendered,
            vec![
                (CutSlot::new(0), OutCol::new(0)),
                (CutSlot::new(1), OutCol::new(1)),
                (CutSlot::new(2), OutCol::new(2))
            ]
        );

        for (_, col) in &rendered {
            assert!(
                global.storage.contains(&col.get()),
                "storage-only render must touch only storage columns (got {})",
                col.get()
            );
        }
    }

    // ── Bucket block tests ────────────────────────────────────────────────────

    /// `inflow_lags` disabled, but the bucket block stays included (never gated
    /// by `StageStateConfig`): `n_slots() = N + B + A*k_max = 2 + 2 + 2 = 6`, with
    /// bucket slots between storage and anticipated.
    #[test]
    fn bucket_block_always_included_with_storage_only() {
        let global = finalized_with_transit_buckets(
            2,
            1,
            vec![(HydroSys::new(0), 1), (HydroSys::new(1), 1)],
            &[2],
        );
        let cut = CutStateProjection::new(&global, STORAGE_ONLY);

        assert_eq!(cut.n_slots(), 6);
        let (incoming, _) = collect_columns(&cut);

        for (j, &col) in incoming.iter().take(2).enumerate() {
            assert_eq!(
                col,
                Some(InCol::new(global.storage_in.start + j)),
                "storage slot {j} must map to storage_in.start + {j}"
            );
        }
        for i in 0..2 {
            assert_eq!(
                incoming[2 + i],
                Some(InCol::new(global.transit_buckets_in.start + i)),
                "bucket slot {i} must map to transit_buckets_in.start + {i} despite \
                 inflow_lags disabled"
            );
        }
        for i in 0..2 {
            assert_eq!(
                incoming[4 + i],
                Some(InCol::new(global.commit_in.start + i)),
                "anticipated slot {i} must map to commit_in.start + {i}"
            );
        }
    }

    /// The ordering regression guard for the storage→lag→buckets→anticipated walk:
    /// swapping buckets and anticipated would misassign reduced coefficient
    /// indices even though every `(index, column)` pair still resolves to a valid
    /// column. The lag slots between storage and buckets hold no render pair.
    #[test]
    fn bucket_render_pairs_sit_between_storage_and_anticipated() {
        let global = finalized_with_transit_buckets(
            2,
            1,
            vec![(HydroSys::new(0), 1), (HydroSys::new(1), 1)],
            &[2],
        );
        let cut = CutStateProjection::new(&global, ALL_ENABLED);

        assert_eq!(global.transit_buckets_out, 2..4);
        assert_eq!(cut.n_slots(), global.n_state);

        let rendered: Vec<(CutSlot, OutCol)> = cut.render_pairs().collect();
        assert_eq!(
            rendered,
            global_render(&global),
            "render_pairs must match the global walk order exactly, \
             storage→buckets→anticipated"
        );

        let transit_bucket_positions: Vec<usize> = rendered
            .iter()
            .enumerate()
            .filter(|&(_, &(_, col))| global.transit_buckets_out.contains(&col.get()))
            .map(|(pos, _)| pos)
            .collect();
        assert_eq!(
            transit_bucket_positions,
            vec![2, 3],
            "bucket render pairs must land after the storage block (positions 0..2) \
             and before the anticipated block (positions 4..6)"
        );
    }

    /// `B == 0` regression guard for the always-included bucket block: the
    /// projection reproduces `[0, StateSpace::n_state)` and its render
    /// byte-identically to the pre-bucket walk.
    #[test]
    fn b_zero_projection_matches_pre_transit_bucket_walk() {
        let global = finalized_with_transit_buckets(3, 2, vec![], &[1, 2]);
        let cut = CutStateProjection::new(&global, ALL_ENABLED);

        assert_eq!(global.n_buckets, 0);
        assert_eq!(cut.n_slots(), global.n_state);
        let (incoming, _) = collect_columns(&cut);
        for (j, &col) in incoming.iter().enumerate() {
            assert_eq!(
                col,
                global_columns(&global, j).0,
                "B==0 default projection must match the global resolver at j={j}"
            );
        }

        let rendered: Vec<(CutSlot, OutCol)> = cut.render_pairs().collect();
        assert_eq!(
            rendered,
            global_render(&global),
            "B==0 render must reproduce the global nonzero_state_indices render"
        );
    }

    /// A bucket region deeper than a terminal stage's horizon cap still projects
    /// EVERY bucket dim — the deep-lag slots `horizon_cap_active` freezes `[0, 0]`
    /// at the terminal included. `CutStateProjection::new` keys bucket inclusion on
    /// `StateRegion::Buckets` being cut-enabled and walks the whole
    /// `state_dim_range`, with no entity-type or per-stage gate, so the terminal
    /// bucket-state pricing path (`β·bucket_state`) is already wired at the
    /// projection level: un-masking a deep-lag slot adds no projection code.
    #[test]
    fn every_bucket_dim_projects_including_deep_terminal_lags() {
        // One downstream plant, depth 4: lags 1..=3 sit beyond the terminal cap
        // `n_stages − 1 − t` (which reaches 0 at the terminal, keeping only lag 0),
        // so they are frozen `[0, 0]` in the LP today; the state layout retains
        // them (sized from the global max over every anchor).
        let global = finalized_with_transit_buckets(
            1,
            1,
            vec![
                (HydroSys::new(0), 0),
                (HydroSys::new(0), 1),
                (HydroSys::new(0), 2),
                (HydroSys::new(0), 3),
            ],
            &[],
        );
        let cut = CutStateProjection::new(&global, ALL_ENABLED);

        assert!(
            StateRegion::Buckets.cut_enabled(ALL_ENABLED),
            "buckets are the always-included region the projection walk keys on"
        );

        let buckets = global.state_dim_range(StateRegion::Buckets);
        assert_eq!(buckets.len(), global.n_buckets);
        let (incoming, outgoing) = collect_columns(&cut);

        for g in buckets.clone() {
            let dim = StateDim::new(g);
            let offset = g - buckets.start;

            let projecting_slots: Vec<CutSlot> = (0..cut.n_slots())
                .map(CutSlot::new)
                .filter(|&s| cut.global_state_index(s) == dim)
                .collect();
            assert_eq!(
                projecting_slots.len(),
                1,
                "bucket dim {g} must project to exactly one cut slot"
            );
            let slot = projecting_slots[0];

            assert_eq!(
                incoming[slot.get()],
                Some(InCol::new(global.transit_buckets_in.start + offset)),
                "bucket dim {g}: incoming column is the pinned bucket column (subgradient read site)"
            );
            assert_eq!(
                outgoing[slot.get()],
                Some(OutCol::new(global.transit_buckets_out.start + offset)),
                "bucket dim {g}: outgoing column is the identity bucket column (cut-render site)"
            );

            let rendered: Vec<OutCol> = cut
                .render_pairs()
                .filter(|&(s, _)| s == slot)
                .map(|(_, col)| col)
                .collect();
            assert_eq!(
                rendered.len(),
                1,
                "bucket dim {g} must appear in exactly one render pair"
            );
            assert_eq!(
                rendered[0],
                OutCol::new(global.transit_buckets_out.start + offset)
            );
        }
    }

    // ── Commitment-hold ring tests ──────────────────────────────────────────

    /// The commitment-hold region — the single in-study anticipated ring that
    /// now carries every post-study target directly — joins the projection
    /// completely: `n_slots()` grows by exactly the ring's width over the
    /// pre-ring dimension, and every ring slot is present, mirroring the
    /// always-included bucket contract. No entity-type or per-stage arm gates
    /// this inclusion.
    #[test]
    fn commitment_hold_post_study_target_joins_the_projection() {
        let pre_ring = finalized(2, 1, &[]);
        let with_ring = finalized(2, 1, &[2]);

        let cut_pre = CutStateProjection::new(&pre_ring, ALL_ENABLED);
        let cut_with = CutStateProjection::new(&with_ring, ALL_ENABLED);

        assert_eq!(
            cut_with.n_slots(),
            cut_pre.n_slots() + with_ring.n_anticipated * with_ring.k_max,
            "n_slots must grow by exactly the anticipated-ring width"
        );
        assert_eq!(cut_with.n_slots(), with_ring.n_state);

        let (incoming, _) = collect_columns(&cut_with);
        for (i, j) in with_ring.commit_in.clone().enumerate() {
            assert_eq!(
                incoming[cut_pre.n_slots() + i],
                Some(InCol::new(j)),
                "commitment-hold ring incoming slot {j} must appear in the projection"
            );
        }
    }

    // ── Lag region tests ────────────────────────────────────────────────────

    /// N = 2, L = 2, two buckets and one anticipated plant.
    fn lagged_state_space() -> StateSpace {
        let buckets = vec![(HydroSys::new(0), 1), (HydroSys::new(1), 1)];
        let leads = vec![2];
        StateSpace::new(
            2,
            2,
            buckets,
            leads.clone(),
            constant_lead_resolution(&leads, 4),
            &[2, 2],
        )
    }

    #[test]
    fn projection_keeps_every_slot_and_renders_no_lag_slot() {
        let global = lagged_state_space();
        let cut = CutStateProjection::new(&global, ALL_ENABLED);
        let lag_range = global.state_dim_lag_range();

        assert_eq!(cut.n_slots(), global.n_state);
        for j in 0..global.n_state {
            assert_eq!(cut.global_state_index(CutSlot::new(j)), StateDim::new(j));
        }
        for (slot, col) in cut.incoming_pairs() {
            assert_eq!(
                col,
                global.state_to_lp_incoming_column(cut.global_state_index(slot))
            );
        }
        for (slot, col) in cut.outgoing_pairs() {
            assert_eq!(
                col,
                global.lp_column_for_state(cut.global_state_index(slot))
            );
        }

        for (slot, _) in cut.render_pairs() {
            assert!(
                !lag_range.contains(&cut.global_state_index(slot).get()),
                "lag slot {} must not be rendered",
                slot.get()
            );
        }

        let lag_dims_in_mask = global
            .nonzero_state_indices
            .iter()
            .filter(|g| lag_range.contains(&g.get()))
            .count();
        assert_eq!(lag_dims_in_mask, global.hydro_count * global.max_par_order);
        assert_eq!(
            cut.render_len(),
            global.nonzero_state_indices.len() - lag_dims_in_mask
        );
    }

    #[test]
    fn projection_pairs_skip_exactly_the_lag_slots() {
        let global = lagged_state_space();
        let cut = CutStateProjection::new(&global, ALL_ENABLED);
        let full = cut.lag_point().full();
        assert!(!full.is_empty());

        let columned: Vec<usize> = (0..cut.n_slots()).filter(|j| !full.contains(j)).collect();
        let incoming_slots: Vec<usize> = cut.incoming_pairs().map(|(slot, _)| slot.get()).collect();
        let outgoing_slots: Vec<usize> = cut.outgoing_pairs().map(|(slot, _)| slot.get()).collect();
        assert_eq!(incoming_slots, columned);
        assert_eq!(outgoing_slots, columned);
        assert_eq!(cut.columned_len(), cut.n_slots() - full.len());
        for (slot, col) in cut.incoming_pairs() {
            assert_eq!(
                col,
                global.state_to_lp_incoming_column(cut.global_state_index(slot))
            );
        }
        for (slot, col) in cut.outgoing_pairs() {
            assert_eq!(
                col,
                global.lp_column_for_state(cut.global_state_index(slot))
            );
        }
    }

    #[test]
    fn lag_point_follows_the_lag_region_of_each_projection() {
        let global = lagged_state_space();

        for (config, lag0, shifted, full) in [
            (ALL_ENABLED, 2..4, 4..6, 2..6),
            (STORAGE_ONLY, 0..0, 0..0, 0..0),
            (LAGS_ONLY, 0..2, 2..4, 0..4),
        ] {
            let cut = CutStateProjection::new(&global, config);
            assert_eq!(cut.lag_point().lag0(), lag0, "{config:?}");
            assert_eq!(cut.lag_point().shifted(), shifted, "{config:?}");
            assert_eq!(cut.lag_point().full(), full, "{config:?}");
        }
    }

    #[test]
    fn has_lag_slots_follows_the_lag_point() {
        let global = lagged_state_space();
        assert!(CutStateProjection::new(&global, ALL_ENABLED).has_lag_slots());
        assert!(CutStateProjection::new(&global, LAGS_ONLY).has_lag_slots());
        assert!(!CutStateProjection::new(&global, STORAGE_ONLY).has_lag_slots());
        let no_lags = CutStateProjection::new(&finalized(3, 0, &[]), ALL_ENABLED);
        assert!(!no_lags.has_lag_slots());
    }

    #[test]
    fn structural_dual_bound_counts_theta_and_the_rendered_columns() {
        let global = lagged_state_space();
        let cut = CutStateProjection::new(&global, ALL_ENABLED);
        assert_eq!(cut.structural_dual_bound(), 1 + cut.render_pairs().count());

        let storage_columns_only = StateSpace::new(
            2,
            2,
            Vec::new(),
            Vec::new(),
            constant_lead_resolution(&[], 2),
            &[2, 2],
        );
        let cut = CutStateProjection::new(&storage_columns_only, ALL_ENABLED);
        assert_eq!(cut.structural_dual_bound(), 3);
    }

    #[test]
    fn lag_point_of_a_single_lag_has_no_shifted_slots() {
        let single_lag = CutStateProjection::new(&finalized(3, 1, &[]), ALL_ENABLED);
        assert_eq!(single_lag.lag_point().lag0(), 3..6);
        assert!(single_lag.lag_point().shifted().is_empty());
        assert_eq!(single_lag.lag_point().full(), 3..6);

        for global in [finalized(3, 0, &[]), finalized(0, 2, &[])] {
            let cut = CutStateProjection::new(&global, ALL_ENABLED);
            assert_eq!(cut.lag_point().lag0(), 0..0);
            assert_eq!(cut.lag_point().shifted(), 0..0);
            assert_eq!(cut.lag_point().full(), 0..0);
        }
    }

    fn compose_pair(resident: StageStateConfig, extracted: StageStateConfig) -> ChainRuleSlots {
        let global = lagged_state_space();
        ChainRuleSlots::compose(
            &global,
            &CutStateProjection::new(&global, resident),
            &CutStateProjection::new(&global, extracted),
        )
    }

    #[test]
    fn chain_rule_slots_compose_identical_projections() {
        let slots = compose_pair(ALL_ENABLED, ALL_ENABLED);

        assert!(!slots.resident_shifted().is_empty());
        assert_eq!(slots.resident_lag0(), 2..4);
        assert_eq!(slots.resident_shifted(), 4..6);
        assert_eq!(slots.extracted_lag(), 2..6);
        assert_eq!(slots.n_hydros(), 2);
        assert_eq!(slots.lag_depth(), 2);
    }

    #[test]
    fn chain_rule_slots_are_empty_on_the_side_without_lags() {
        let no_extracted_lags = compose_pair(ALL_ENABLED, STORAGE_ONLY);
        assert_eq!(no_extracted_lags.extracted_lag(), 0..0);
        assert_eq!(no_extracted_lags.resident_lag0(), 2..4);
        assert_eq!(no_extracted_lags.resident_shifted(), 4..6);

        let no_resident_lags = compose_pair(STORAGE_ONLY, ALL_ENABLED);
        assert_eq!(no_resident_lags.resident_lag0(), 0..0);
        assert_eq!(no_resident_lags.resident_shifted(), 0..0);
        assert_eq!(no_resident_lags.extracted_lag(), 2..6);
    }

    #[test]
    fn chain_rule_slots_map_by_dimension_not_position() {
        let slots = compose_pair(ALL_ENABLED, LAGS_ONLY);

        assert_eq!(slots.resident_lag0(), 2..4);
        assert_eq!(slots.extracted_lag(), 0..4);
    }

    #[test]
    fn chain_rule_slots_map_the_non_lag_slots_onto_the_columned_order() {
        let global = lagged_state_space();
        let n = CutStateProjection::new(&global, ALL_ENABLED).n_slots();
        assert!(n > 6);
        assert_eq!(
            compose_pair(ALL_ENABLED, ALL_ENABLED).non_lag(),
            [(0..2, 0..2), (6..n, 2..n - 4)]
        );
        let m = CutStateProjection::new(&global, LAGS_ONLY).n_slots();
        assert_eq!(
            compose_pair(ALL_ENABLED, LAGS_ONLY).non_lag(),
            [(0..0, 0..0), (4..m, 0..m - 4)]
        );
        let k = CutStateProjection::new(&global, STORAGE_ONLY).n_slots();
        assert_eq!(
            compose_pair(ALL_ENABLED, STORAGE_ONLY).non_lag(),
            [(0..0, 0..0), (0..k, 0..k)]
        );

        let p = CutStateProjection::new(&global, ALL_ENABLED);
        let [(head_slots, head_positions), (tail_slots, tail_positions)] =
            ChainRuleSlots::compose(&global, &p, &p).non_lag();
        let mapped_slots: Vec<usize> = head_slots.chain(tail_slots).collect();
        let columned_slots: Vec<usize> = p.incoming_pairs().map(|(slot, _)| slot.get()).collect();
        assert_eq!(mapped_slots, columned_slots);
        let positions: Vec<usize> = head_positions.chain(tail_positions).collect();
        assert_eq!(positions, (0..p.columned_len()).collect::<Vec<_>>());
    }
}

#[cfg(test)]
mod proptests {
    use proptest::prelude::*;
    use proptest::test_runner::RngSeed;

    use super::tests::{collect_columns, global_columns, global_render};
    use super::{CutSlot, CutStateProjection, OutCol, StageStateConfig};
    use crate::indexer::{HydroSys, StateSpace};
    use crate::lead_time::{AnticipatedResolution, DeliveryAxis, LeadTime};

    const ALL_ENABLED: StageStateConfig = StageStateConfig {
        storage: true,
        inflow_lags: true,
    };

    /// Fixed cases/seed so a failing shrink is reproducible run-to-run (the
    /// determinism discipline applied to test generation itself).
    fn fixed_config() -> ProptestConfig {
        ProptestConfig {
            cases: 256,
            rng_seed: RngSeed::Fixed(42),
            ..ProptestConfig::default()
        }
    }

    /// A valid [`StateSpace`] over the small parameter space (`hydro_count`,
    /// `max_par_order`, `n_buckets`, `n_anticipated` each `0..=4` or `0..=3`,
    /// per-plant leads `0..=3`, a delivery axis `n_decision <= n_delivery`
    /// both `<= 7`), with every dependent-length vector sized and bounded to
    /// satisfy `StateSpace::new`'s debug-asserts:
    /// `anticipated_lead_stages.len() == n_anticipated`, and
    /// `effective_lag_count.len() == hydro_count` (each `<= max_par_order`).
    /// `n_buckets` sizes `transit_bucket_column_order` (`StateSpace::new`
    /// derives its own bucket count from that vector's length). `k_max` is
    /// derived — `AnticipatedResolution::resolve`'s own
    /// `ring_size(&anticipated_lead_stages)` — never ranged independently of
    /// the leads: a ring deeper than its leads is covered wherever a
    /// resolution produces one.
    fn state_layout_strategy() -> impl Strategy<Value = StateSpace> {
        (0..=4usize, 0..=3usize, 0..=3usize, 0..=3usize)
            .prop_flat_map(|(hydro_count, max_par_order, n_buckets, n_anticipated)| {
                (
                    Just(hydro_count),
                    Just(max_par_order),
                    Just(n_buckets),
                    Just(n_anticipated),
                    prop::collection::vec(0..=3usize, n_anticipated),
                    0..=5usize,
                    0..=2usize,
                    prop::collection::vec((0..=4usize, 0..=3usize), n_buckets),
                    prop::collection::vec(0..=max_par_order, hydro_count),
                )
            })
            .prop_map(
                |(
                    hydro_count,
                    max_par_order,
                    _n_buckets,
                    _n_anticipated,
                    anticipated_lead_stages,
                    n_decision,
                    extra_delivery,
                    transit_bucket_column_order,
                    effective_lag_count,
                )| {
                    let leads: Vec<LeadTime> = anticipated_lead_stages
                        .iter()
                        .map(|&l| LeadTime::Stages(u32::try_from(l).unwrap_or(u32::MAX)))
                        .collect();
                    let study_stage_hours = vec![720.0; n_decision];
                    let post_study_stage_hours = vec![720.0; extra_delivery];
                    let resolution = AnticipatedResolution::resolve(
                        &leads,
                        DeliveryAxis {
                            study_stage_hours: &study_stage_hours,
                            post_study_stage_hours: &post_study_stage_hours,
                        },
                    );
                    let transit_bucket_column_order: Vec<(HydroSys, usize)> =
                        transit_bucket_column_order
                            .into_iter()
                            .map(|(p, lag)| (HydroSys::new(p), lag))
                            .collect();
                    StateSpace::new(
                        hydro_count,
                        max_par_order,
                        transit_bucket_column_order,
                        anticipated_lead_stages,
                        resolution,
                        &effective_lag_count,
                    )
                },
            )
    }

    /// A [`StateSpace`] paired with an arbitrary [`StageStateConfig`] gate
    /// combination, for the gated-projection agreement property.
    fn state_layout_and_config_strategy() -> impl Strategy<Value = (StateSpace, StageStateConfig)> {
        (state_layout_strategy(), any::<bool>(), any::<bool>()).prop_map(
            |(global, storage, inflow_lags)| {
                (
                    global,
                    StageStateConfig {
                        storage,
                        inflow_lags,
                    },
                )
            },
        )
    }

    proptest! {
        #![proptest_config(fixed_config())]

        /// Generalizes `state_dim_ranges_partition_n_state_contiguously`: the
        /// four `state_dim_*_range` regions partition `[0, n_state)` contiguously
        /// (no gap, no overlap) for every generated shape.
        #[test]
        fn partition(global in state_layout_strategy()) {
            let storage = global.state_dim_storage_range();
            let lag = global.state_dim_lag_range();
            let bucket = global.state_dim_bucket_range();
            let commitment_hold = global.state_dim_commitment_hold_range();

            prop_assert_eq!(storage.start, 0);
            prop_assert_eq!(lag.start, storage.end);
            prop_assert_eq!(bucket.start, lag.end);
            prop_assert_eq!(commitment_hold.start, bucket.end);
            prop_assert_eq!(commitment_hold.end, global.n_state);
        }

        /// Generalizes `default_projection_is_identity` and
        /// `default_render_matches_global_nonzero_mask`: an all-enabled
        /// projection reproduces the global resolvers index-for-index, and
        /// `render_pairs` reproduces the global `nonzero_state_indices` render
        /// exactly, for every generated shape.
        #[test]
        fn default_identity(global in state_layout_strategy()) {
            let cut = CutStateProjection::new(&global, ALL_ENABLED);

            prop_assert_eq!(cut.n_slots(), global.n_state);
            let (incoming, outgoing) = collect_columns(&cut);
            for (j, (&in_col, &out_col)) in incoming.iter().zip(&outgoing).enumerate() {
                prop_assert_eq!((in_col, out_col), global_columns(&global, j));
            }

            let rendered: Vec<(CutSlot, OutCol)> = cut.render_pairs().collect();
            prop_assert_eq!(rendered, global_render(&global));
        }

        /// Generalizes `bucket_render_pairs_sit_between_storage_and_anticipated`
        /// over every gate combination. The expected mapping is built from the
        /// concrete per-region block formulas (the raw range fields, with the
        /// storage → buckets → anticipated order of the columned slots written
        /// out literally; the lag slots have no columns) —
        /// never from `REGION_ORDER` or the resolvers the constructor itself
        /// walks, which would shift both sides of the assertion in lockstep
        /// and turn a walk-order or resolver bug green.
        #[test]
        fn gated_projection_agreement(
            (global, state_config) in state_layout_and_config_strategy()
        ) {
            let cut = CutStateProjection::new(&global, state_config);

            let n = global.hydro_count;
            let mut expected: Vec<(usize, usize)> = Vec::new();
            if state_config.storage {
                for h in 0..n {
                    expected.push((global.storage_in.start + h, global.storage.start + h));
                }
            }
            for b in 0..global.n_buckets {
                expected.push((
                    global.transit_buckets_in.start + b,
                    global.transit_buckets_out.start + b,
                ));
            }
            for o in 0..global.n_anticipated * global.k_max {
                expected.push((global.commit_in.start + o, global.commit_out.start + o));
            }

            let (incoming, outgoing) = collect_columns(&cut);
            let collected: Vec<(usize, usize)> = incoming
                .iter()
                .zip(&outgoing)
                .filter_map(|(in_col, out_col)| Some((in_col.as_ref()?.get(), out_col.as_ref()?.get())))
                .collect();
            prop_assert_eq!(collected, expected);
        }

        #[test]
        fn lag_point_slots_map_to_the_lag_state_dims(
            (global, state_config) in state_layout_and_config_strategy()
        ) {
            let cut = CutStateProjection::new(&global, state_config);
            let lag_point = cut.lag_point();
            let (n, l) = (global.hydro_count, global.max_par_order);

            if state_config.inflow_lags && n * l > 0 {
                let (lag0, shifted) = (lag_point.lag0(), lag_point.shifted());
                prop_assert_eq!(lag0.len(), n);
                prop_assert_eq!(shifted.len(), n * (l - 1));
                prop_assert_eq!(shifted.start, lag0.end);
                for lag in 0..l {
                    for h in 0..n {
                        prop_assert_eq!(
                            cut.global_state_index(CutSlot::new(lag0.start + lag * n + h)),
                            global.lag_state_dim(lag, HydroSys::new(h))
                        );
                    }
                }
            } else {
                prop_assert_eq!(lag_point.lag0(), 0..0);
                prop_assert_eq!(lag_point.shifted(), 0..0);
            }
        }
    }
}
