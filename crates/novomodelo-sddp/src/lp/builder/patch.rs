use super::state_box::StateBox;
use super::{InflowCoupling, StageGeometry};
use crate::lp::indexer::{
    BlockGrid, BlockIdx, BlockRowFamily, HydroSys, REGION_ORDER, StateDim, StateRegion, StateSpace,
};

/// Pre-allocated row-bound and column-bound patch arrays for one SDDP stage LP solve.
///
/// Reused across all iterations. The row-bound region (`indices`/`lower`/`upper`,
/// length `M*B + R`) holds only the load and inflow patches; state
/// fixing (storage, travel-time buckets, anticipated state) is applied
/// exclusively via column bounds and lives in the column-bound region
/// (`col_indices`/`col_lower`/`col_upper`), one slot per state dimension that has
/// an LP column: length `N + n_buckets + A*K`; the lag slice has no LP column.
/// `N` is the hydro count, `M` the stochastic-load-bus count, `B` the max block
/// count, `R` the most rows any stage's inflow coupling touches, `n_buckets` the
/// travel-time bucket count, and `A*K` the anticipated-thermal state count. For
/// equality constraints each entry's lower == upper.
///
/// [`fill_col_state_patches`](Self::fill_col_state_patches) is the single owner
/// of the column-bound region: it iterates
/// [`StateSpace::state_to_lp_incoming_column`] over every state dimension with
/// an LP column, region by region, so storage, buckets, and anticipated
/// state all resolve through one call site rather than parallel hardcoded
/// offsets. Every call writes the whole region, so
/// [`active_state_col_patch_count`](Self::active_state_col_patch_count) is its
/// length once the buffer has been filled.
#[derive(Debug, Clone)]
pub struct PatchBuffer {
    /// Row indices to patch.
    pub indices: Vec<usize>,

    /// New lower bounds for each patched row.
    pub lower: Vec<f64>,

    /// New upper bounds for each patched row.
    pub upper: Vec<f64>,

    /// Column indices to patch in the column-bound region.
    pub col_indices: Vec<usize>,

    /// New lower bounds for each patched column in the column-bound region.
    pub col_lower: Vec<f64>,

    /// New upper bounds for each patched column in the column-bound region.
    pub col_upper: Vec<f64>,

    /// Number of buses with stochastic load noise (M).
    load_bus_count: usize,

    /// Maximum block count across all stages (B).
    max_blocks: usize,

    /// Number of load patches from the most recent [`fill_load_patches`] call
    /// (`load_bus_count * n_blocks`); zero before any call or when
    /// `load_bus_count == 0`.
    ///
    /// [`fill_load_patches`]: PatchBuffer::fill_load_patches
    active_load_patches: usize,

    /// Number of inflow patches from the most recent [`fill_inflow_shift_patches`]
    /// call (the touched-row count).
    ///
    /// [`fill_inflow_shift_patches`]: PatchBuffer::fill_inflow_shift_patches
    active_inflow_patches: usize,

    /// Number of pins from the most recent [`fill_col_state_patches`] call; zero
    /// before any call.
    ///
    /// [`fill_col_state_patches`]: PatchBuffer::fill_col_state_patches
    active_state_col_patches: usize,
}

impl PatchBuffer {
    /// Construct a [`PatchBuffer`] sized to `M*B + R` row patches and one column
    /// patch per state dimension that has an LP column
    /// ([`StateRegion::has_lp_columns`]), zero-initialised, from its owners:
    /// `N`, `L`, `n_buckets`, `A`, and `K` come from `state`; `M` is
    /// `load_bus_indices.len()`; `B` is [`StageGeometry::max_blocks`] over
    /// `geometry_per_stage`; `R` is the most distinct rows any one stage's
    /// coupling touches, and `0` for `&[]`.
    ///
    /// The row region holds load patches at `[0, M*B)` followed by `R` inflow
    /// patches, populated before each LP solve via [`fill_load_patches`] and
    /// [`fill_inflow_shift_patches`]; the column region is populated by
    /// `fill_col_state_patches`. Pass `&[]` for `load_bus_indices` when there is no
    /// stochastic load and `&[]` for `geometry_per_stage` when there is no
    /// stochastic load or the caller has no geometry to size from.
    /// `compact_shift_region_holds_the_most_touched_rows_of_any_stage` (this
    /// module's `tests`) pins `R`.
    ///
    /// # Examples
    ///
    /// A 3-hydro AR(2) system with no stochastic load, no buckets, and no
    /// anticipated thermals has column capacity `N = 3`: the lag slice has no LP
    /// column. Anticipated thermals (`A*K`) and travel-time buckets (`n_buckets`)
    /// widen only the column region, and row capacity depends on the coupling
    /// alone, never on `N` or `L`. `patch_buffer_new_sizes_from_its_owners` (this
    /// module's `tests`) is the runnable form of these cases, and
    /// `fill_col_state_patches_skips_the_lag_region_of_a_compact_layout` of the lag slice.
    ///
    /// [`fill_load_patches`]: PatchBuffer::fill_load_patches
    /// [`fill_inflow_shift_patches`]: PatchBuffer::fill_inflow_shift_patches
    #[must_use]
    pub fn new(
        state: &StateSpace,
        load_bus_indices: &[usize],
        geometry_per_stage: &[StageGeometry],
        inflow_coupling_per_stage: &[InflowCoupling],
    ) -> Self {
        let n_load_buses = load_bus_indices.len();
        let max_blocks = StageGeometry::max_blocks(geometry_per_stage);
        let inflow_patch_capacity = inflow_coupling_per_stage
            .iter()
            .map(|coupling| coupling.touched_rows().len())
            .max()
            .unwrap_or(0);
        let capacity = n_load_buses * max_blocks + inflow_patch_capacity;
        let col_capacity: usize = REGION_ORDER
            .into_iter()
            .filter(|&region| region.has_lp_columns())
            .map(|region| state.state_dim_range(region).len())
            .sum();
        Self {
            indices: vec![0; capacity],
            lower: vec![0.0; capacity],
            upper: vec![0.0; capacity],
            col_indices: vec![0; col_capacity],
            col_lower: vec![0.0; col_capacity],
            col_upper: vec![0.0; col_capacity],
            load_bus_count: n_load_buses,
            max_blocks,
            active_load_patches: 0,
            active_inflow_patches: 0,
            active_state_col_patches: 0,
        }
    }

    /// Fill the equality column-bound patches pinning incoming state: storage,
    /// travel-time buckets, and anticipated state.
    ///
    /// The single owner of the column-bound region: walks [`REGION_ORDER`] and
    /// iterates [`StateSpace::state_to_lp_incoming_column`] over every state
    /// dimension of a region that has LP columns — storage, buckets, and
    /// anticipated state all resolve through this one resolver call rather than
    /// parallel hardcoded offsets. A hardcoded anticipated start that
    /// leaves out `n_buckets` silently drops the shift whenever a
    /// bucket block precedes it, misaligning the anticipated patches and leaving
    /// every bucket incoming column unpinned.
    ///
    /// Pins are written region by region and compacted: the lag region has no LP
    /// column and is skipped, so the pins after it sit at lower slots than their
    /// state indices. The pins fill the column-bound region exactly
    /// ([`Self::active_state_col_patch_count`] equals
    /// [`Self::state_col_patch_count`]).
    ///
    /// Enforces `x == v` by setting `lb = ub = v / col_scale[col]` in the scaled LP
    /// — **divided**, contrast with the row-equality path that **multiplies** by
    /// `row_scale[row]`. `col_scale` may be `&[]` (no scaling); when non-empty it
    /// must be at least `state_layout.commit_in.end` long.
    ///
    /// Bucket incoming columns are pinned to whatever the caller supplies in
    /// `state` at the bucket state-vector indices — pin this **once per
    /// stage-visit** (the trial point / decision-driven state), never per
    /// opening, the same contract [`state_to_lp_incoming_column`] documents for
    /// storage (contrast with NCS availability, which patches per
    /// opening).
    ///
    /// [`state_to_lp_incoming_column`]: StateSpace::state_to_lp_incoming_column
    ///
    /// The pinned incoming state is whatever the caller supplies in `state`: at
    /// every stage but the first, the previous stage's own seam-canonicalized
    /// outgoing state, so it lies inside that PRODUCING stage's own admissible
    /// box; `producer_box` is that box (`None` at the first stage, whose
    /// incoming state is the study's initial condition, validated at its own
    /// source with no producing seam). `Some` asserts every state dimension, a
    /// skipped region's included, lies inside it, catching an entry point that
    /// bypassed the seam.
    ///
    /// # Panics
    ///
    /// Panics in debug builds if `state.len() != state_layout.n_state`, if the
    /// emitted patch count does not equal [`Self::state_col_patch_count`], if any
    /// travel-time bucket pin fails to resolve into
    /// `state_layout.transit_buckets_in`, or if `producer_box` is `Some` and any
    /// state dimension lies outside it.
    pub fn fill_col_state_patches(
        &mut self,
        state_layout: &StateSpace,
        state: &[f64],
        col_scale: &[f64],
        producer_box: Option<&StateBox>,
    ) {
        debug_assert_eq!(
            state.len(),
            state_layout.n_state,
            "state slice length {got} != n_state {expected}",
            got = state.len(),
            expected = state_layout.n_state,
        );

        let mut slot = 0;
        let mut bucket_slots = 0..0;
        for region in REGION_ORDER {
            let first_slot = slot;
            let has_columns = region.has_lp_columns();
            for j in state_layout.state_dim_range(region) {
                let sv = state[j];
                if let Some(state_box) = producer_box {
                    debug_assert!(
                        sv >= state_box.lower[j] && sv <= state_box.upper[j],
                        "pinned incoming state[{j}] = {sv} outside admissible box [{lower}, {upper}]",
                        lower = state_box.lower[j],
                        upper = state_box.upper[j],
                    );
                }
                if has_columns {
                    let col = state_layout
                        .state_to_lp_incoming_column(StateDim::new(j))
                        .get();
                    let scaled = if col_scale.is_empty() {
                        sv
                    } else {
                        sv / col_scale[col]
                    };
                    self.col_indices[slot] = col;
                    self.col_lower[slot] = scaled;
                    self.col_upper[slot] = scaled;
                    slot += 1;
                }
            }
            if matches!(region, StateRegion::Buckets) {
                bucket_slots = first_slot..slot;
            }
        }
        self.active_state_col_patches = slot;

        debug_assert_eq!(
            slot,
            self.state_col_patch_count(),
            "emitted patch count must equal state_col_patch_count()"
        );
        debug_assert!(
            self.col_indices[bucket_slots]
                .iter()
                .all(|col| state_layout.transit_buckets_in.contains(col)),
            "every travel-time bucket state index must resolve to a transit_buckets_in column"
        );
    }

    /// Fill `n_load_buses * n_blocks` load-balance equality patches into
    /// the row buffer at offset `0`, addressed via `load_rows` (the stage's own
    /// load-balance row family). `load_rhs` is bus-major, block-minor matching
    /// `bus_positions` order; values are prescaled by `row_scale[row]` when
    /// `row_scale` is non-empty (pass `&[]` for no scaling).
    ///
    /// `grid` must carry this stage's block count (the value the LP template was
    /// built with), NOT a global grid: a global grid would stride by the wrong block
    /// count at any stage whose count differs.
    ///
    /// # Panics
    ///
    /// Panics in debug builds if:
    /// - `load_rhs.len() != self.load_bus_count * grid.n_blks()`
    /// - `bus_positions.len() != self.load_bus_count`
    /// - `grid.n_blks() > self.max_blocks`
    #[inline]
    pub fn fill_load_patches(
        &mut self,
        load_rows: BlockRowFamily,
        grid: BlockGrid,
        load_rhs: &[f64],
        bus_positions: &[usize],
        row_scale: &[f64],
    ) {
        let n_blocks = grid.n_blks();
        debug_assert_eq!(
            load_rhs.len(),
            self.load_bus_count * n_blocks,
            "load_rhs length {got} != load_bus_count*n_blocks {expected}",
            got = load_rhs.len(),
            expected = self.load_bus_count * n_blocks,
        );
        debug_assert_eq!(
            bus_positions.len(),
            self.load_bus_count,
            "bus_positions length {got} != load_bus_count {expected}",
            got = bus_positions.len(),
            expected = self.load_bus_count,
        );
        debug_assert!(
            n_blocks <= self.max_blocks,
            "n_blocks {n_blocks} exceeds max_blocks {mb}",
            mb = self.max_blocks,
        );

        let mut slot = 0;

        for (i, &bus_pos) in bus_positions.iter().enumerate() {
            for blk in 0..n_blocks {
                let row = load_rows.row(bus_pos, BlockIdx::new(blk), n_blocks);
                // Host-array index `i * n_blks + blk` routed through `grid.flat`
                // (start = 0) to keep one owner of the stride.
                let rhs = load_rhs[grid.flat(0, i, BlockIdx::new(blk))];
                let scaled = if row_scale.is_empty() {
                    rhs
                } else {
                    rhs * row_scale[row]
                };
                self.indices[slot] = row;
                self.lower[slot] = scaled;
                self.upper[slot] = scaled;
                slot += 1;
            }
        }

        self.active_load_patches = self.load_bus_count * n_blocks;
    }

    /// Fill one patch per row `coupling` touches, each bound being its unscaled template
    /// bound minus `Σ_h G[r,h]·z_hat[h]`, then multiplied by `row_scale[r]` (pass `&[]`
    /// for no scaling).
    ///
    /// The shift accumulates from `+0.0` in ascending hydro, then entry, order and is
    /// subtracted from the unscaled bound; a scaled bound is never shifted or divided
    /// back. Must be called after [`fill_load_patches`], whose `active_load_patches`
    /// sets this region's offset.
    ///
    /// # Panics
    ///
    /// Panics in debug builds if the region does not fit the row buffer or if
    /// `z_hat.len() != coupling.n_hydros()`.
    ///
    /// [`fill_load_patches`]: PatchBuffer::fill_load_patches
    pub fn fill_inflow_shift_patches(
        &mut self,
        coupling: &InflowCoupling,
        z_hat: &[f64],
        row_scale: &[f64],
    ) {
        let start = self.active_load_patches;
        let touched_rows = coupling.touched_rows();
        let n_rows = touched_rows.len();
        debug_assert!(
            start + n_rows <= self.indices.len(),
            "inflow shift region [{start}, {end}) exceeds the row buffer length {len}",
            end = start + n_rows,
            len = self.indices.len(),
        );
        debug_assert_eq!(
            z_hat.len(),
            coupling.n_hydros(),
            "z_hat length {got} != n_hydros {expected}",
            got = z_hat.len(),
            expected = coupling.n_hydros(),
        );

        for (p, &row) in touched_rows.iter().enumerate() {
            self.indices[start + p] = row as usize;
            self.lower[start + p] = 0.0;
        }
        for (h, &z) in z_hat[..coupling.n_hydros()].iter().enumerate() {
            for (p, coefficient) in coupling.hydro_shift_entries(HydroSys::new(h)) {
                self.lower[start + p] += coefficient * z;
            }
        }
        let scaled = |bound: f64, row: u32| {
            if row_scale.is_empty() {
                bound
            } else {
                bound * row_scale[row as usize]
            }
        };
        let lower_bounds = coupling.touched_row_lower();
        let upper_bounds = coupling.touched_row_upper();
        for (p, &row) in touched_rows.iter().enumerate() {
            let shift = self.lower[start + p];
            self.lower[start + p] = scaled(lower_bounds[p] - shift, row);
            self.upper[start + p] = scaled(upper_bounds[p] - shift, row);
        }

        self.active_inflow_patches = n_rows;
    }

    /// Active row-patch count (load + inflow, from [`Self::fill_inflow_shift_patches`])
    /// for the full forward-pass slice passed to `set_row_bounds`.
    #[must_use]
    #[inline]
    pub fn forward_patch_count(&self) -> usize {
        self.active_load_patches + self.active_inflow_patches
    }

    /// Column-bound region capacity: one incoming-state pin per state dimension
    /// that has an LP column.
    #[must_use]
    #[inline]
    pub fn state_col_patch_count(&self) -> usize {
        self.col_indices.len()
    }

    /// Pins the most recent [`Self::fill_col_state_patches`] wrote: zero before
    /// any call, the capacity after one.
    #[must_use]
    #[inline]
    pub fn active_state_col_patch_count(&self) -> usize {
        self.active_state_col_patches
    }
}

#[cfg(test)]
#[expect(
    clippy::doc_markdown,
    reason = "test docs name LP symbols that are not code identifiers"
)]
mod tests {
    use super::{InflowCoupling, PatchBuffer, StateBox, StateSpace};
    use crate::lead_time::{AnticipatedResolution, DeliveryAxis, LeadTime};
    use crate::lp::builder::inflow_coupling::InflowCouplingSink;
    use crate::lp::indexer::{BlockGrid, BlockRowFamily, HydroSys, StateDim};
    use crate::test_support::{
        bits, constant_lead_resolution, equipment_free_geometry, inflow_coupling_with_entries,
        state_layout, state_layout_full, state_layout_with_transit_buckets,
    };

    /// Every dimension unbounded — the pin-time box-membership assert is vacuous.
    fn unbounded_state_box(n_state: usize) -> StateBox {
        StateBox {
            lower: vec![f64::NEG_INFINITY; n_state],
            upper: vec![f64::INFINITY; n_state],
        }
    }

    /// Canonical `(plant, lag)` order for a bucket-count-only fixture: content
    /// is irrelevant to capacity, only `.len()` is read as `n_buckets` by
    /// [`crate::lp::indexer::StateSpace::build`].
    fn bucket_order(n_buckets: usize) -> Vec<(HydroSys, usize)> {
        (0..n_buckets).map(|d| (HydroSys::new(0), d)).collect()
    }

    // -------------------------------------------------------------------------
    // Capacity formulas (row + column buffers) across scales
    // -------------------------------------------------------------------------

    /// Row capacity is `n_load_buses*max_blocks + R` and column capacity is
    /// `N + n_buckets + A*K`. All formulas are exercised at zero /
    /// unit-anticipated / bucket / combined / production scales in one table
    /// so each scale stays legible via the tuple-naming failure message.
    #[test]
    fn patch_buffer_capacity_formulas() {
        // (n, l, n_buckets, a, k, expected_row_cap, expected_col_cap); n_load_buses
        // and max_blocks are exercised separately (`new_with_load_allocates_correct_capacity`),
        // so both stay 0 (and hence M*B stays 0) in every row here.
        let cases = [
            (0usize, 0usize, 0usize, 0usize, 0usize, 0usize, 0usize),
            (3, 2, 0, 0, 0, 0, 3),
            (0, 0, 0, 1, 2, 0, 2),
            (0, 0, 3, 0, 0, 0, 3),
            (3, 2, 4, 2, 3, 0, 13),
            (160, 12, 0, 0, 0, 0, 160),
        ];

        for (n, l, n_buckets, a, k, expected_row_cap, expected_col_cap) in cases {
            let state =
                state_layout_with_transit_buckets(n, l, bucket_order(n_buckets), vec![k; a]);
            let buf = PatchBuffer::new(&state, &[], &[], &[]);

            for (label, len) in [
                ("col_indices", buf.col_indices.len()),
                ("col_lower", buf.col_lower.len()),
                ("col_upper", buf.col_upper.len()),
            ] {
                assert_eq!(
                    len, expected_col_cap,
                    "{label} col cap mismatch for (n={n}, l={l}, n_buckets={n_buckets}, a={a}, k={k})"
                );
            }

            for (label, len) in [
                ("indices", buf.indices.len()),
                ("lower", buf.lower.len()),
                ("upper", buf.upper.len()),
            ] {
                assert_eq!(
                    len, expected_row_cap,
                    "{label} row cap mismatch for (n={n}, l={l}, n_buckets={n_buckets}, a={a}, k={k})"
                );
            }
        }
    }

    /// The runnable form of [`PatchBuffer::new`]'s rustdoc examples: built
    /// directly through [`StateSpace::new`] (rather than the
    /// `state_layout_with_transit_buckets` helper above) at production /
    /// no-lag / anticipated / bucket scales. Row capacity follows the coupling
    /// alone; column capacity is `N + n_buckets + A*K`.
    #[test]
    fn patch_buffer_new_sizes_from_its_owners() {
        let coupling = inflow_coupling_with_entries(2, &[(0, 1), (1, 4)]);
        let couplings = std::slice::from_ref(&coupling);

        // 3-hydro AR(2) system, no buckets, no anticipated thermals.
        let state = StateSpace::new(
            3,
            2,
            Vec::new(),
            Vec::new(),
            AnticipatedResolution::default(),
            &[2, 2, 2],
        );
        let buf = PatchBuffer::new(&state, &[], &[], couplings);
        assert_eq!(buf.indices.len(), 2);
        assert_eq!(buf.col_indices.len(), 3);

        // Production scale: N = 160, L = 12.
        let big_state = StateSpace::new(
            160,
            12,
            Vec::new(),
            Vec::new(),
            AnticipatedResolution::default(),
            &[12; 160],
        );
        let big = PatchBuffer::new(&big_state, &[], &[], couplings);
        assert_eq!(big.indices.len(), 2);
        assert_eq!(big.col_indices.len(), 160);

        // No lags (L = 0).
        let no_lag_state = StateSpace::new(
            5,
            0,
            Vec::new(),
            Vec::new(),
            AnticipatedResolution::default(),
            &[0; 5],
        );
        let no_lag = PatchBuffer::new(&no_lag_state, &[], &[], couplings);
        assert_eq!(no_lag.indices.len(), 2);
        assert_eq!(no_lag.col_indices.len(), 5);

        // Anticipated thermals: 1 plant, K=2 widens the column region only.
        let resolution = AnticipatedResolution::resolve(
            &[LeadTime::Stages(2)],
            DeliveryAxis {
                study_stage_hours: &[720.0; 3],
                post_study_stage_hours: &[],
            },
        );
        let ant_state = StateSpace::new(3, 2, Vec::new(), vec![2], resolution, &[2, 2, 2]);
        let ant = PatchBuffer::new(&ant_state, &[], &[], couplings);
        assert_eq!(ant.indices.len(), 2);
        assert_eq!(ant.col_indices.len(), 5);

        // Travel-time buckets: n_buckets=4 widens the column region only.
        let bucket_column_order = vec![
            (HydroSys::new(0), 0),
            (HydroSys::new(1), 0),
            (HydroSys::new(0), 1),
            (HydroSys::new(1), 1),
        ];
        let bucket_state = StateSpace::new(
            3,
            2,
            bucket_column_order,
            Vec::new(),
            AnticipatedResolution::default(),
            &[2, 2, 2],
        );
        let transit_buckets = PatchBuffer::new(&bucket_state, &[], &[], couplings);
        assert_eq!(transit_buckets.col_indices.len(), 7);
        assert_eq!(transit_buckets.indices.len(), 2);
    }

    #[test]
    fn compact_shift_region_holds_the_most_touched_rows_of_any_stage() {
        let couplings = [
            inflow_coupling_with_entries(2, &[(0, 1)]),
            inflow_coupling_with_entries(2, &[(0, 2), (1, 2), (0, 5), (1, 7)]),
            inflow_coupling_with_entries(2, &[(0, 3), (1, 4)]),
        ];
        let geometry = equipment_free_geometry(&[2, 1, 2]);
        let state = compact_state(2);
        let row_lens = |buf: &PatchBuffer| [buf.indices.len(), buf.lower.len(), buf.upper.len()];

        assert_eq!(
            row_lens(&PatchBuffer::new(&state, &[0], &geometry, &couplings)),
            [5; 3]
        );
        assert_eq!(
            row_lens(&PatchBuffer::new(&state, &[0], &geometry, &[])),
            [2; 3]
        );
    }

    fn compact_state(n_hydros: usize) -> StateSpace {
        StateSpace::new(
            n_hydros,
            1,
            Vec::new(),
            Vec::new(),
            AnticipatedResolution::default(),
            &vec![1; n_hydros],
        )
    }

    fn shifted(
        n_hydros: usize,
        coupling: &InflowCoupling,
        z_hat: &[f64],
        row_scale: &[f64],
    ) -> PatchBuffer {
        let mut buf = PatchBuffer::new(
            &compact_state(n_hydros),
            &[],
            &[],
            std::slice::from_ref(coupling),
        );
        buf.fill_inflow_shift_patches(coupling, z_hat, row_scale);
        buf
    }

    #[test]
    fn inflow_shift_accumulates_hydros_in_ascending_order_from_positive_zero() {
        let two_53 = 9_007_199_254_740_992.0_f64;
        let z_hat = [two_53, 1.0, -two_53];
        let mut sink = InflowCouplingSink::new(3);
        for h in 0..3 {
            sink.record(HydroSys::new(h), 0, 1.0);
        }
        let coupling = sink.finish(&[5.0], &[5.0]);

        let buf = shifted(3, &coupling, &z_hat, &[]);

        let reversed_shift = z_hat.iter().rev().fold(0.0, |shift, z| shift + 1.0 * z);
        assert_eq!((5.0 - reversed_shift).to_bits(), 4.0_f64.to_bits());
        assert_eq!(bits(&buf.lower[..1]), bits(&[5.0]));
        assert_eq!(bits(&buf.upper[..1]), bits(&[5.0]));
    }

    #[test]
    fn inflow_shift_moves_both_bounds_and_keeps_infinite_ones() {
        let mut sink = InflowCouplingSink::new(1);
        for row in 0..4 {
            sink.record(HydroSys::new(0), row, 2.0);
        }
        let coupling = sink.finish(
            &[5.0, 1.0, 2.0, f64::NEG_INFINITY],
            &[5.0, 9.0, f64::INFINITY, 7.0],
        );

        let buf = shifted(1, &coupling, &[1.5], &[]);

        assert_eq!(buf.indices[..4], [0, 1, 2, 3]);
        assert_eq!(
            bits(&buf.lower[..4]),
            bits(&[2.0, -2.0, -1.0, f64::NEG_INFINITY])
        );
        assert_eq!(bits(&buf.upper[..4]), bits(&[2.0, 6.0, f64::INFINITY, 4.0]));
    }

    #[test]
    fn inflow_shift_scales_the_shifted_unscaled_bound() {
        let mut sink = InflowCouplingSink::new(1);
        sink.record(HydroSys::new(0), 1, 1.0);
        let coupling = sink.finish(&[5.0, 5.0], &[5.0, 5.0]);

        let buf = shifted(1, &coupling, &[3.0], &[7.0, 4.0]);

        assert_eq!(buf.indices[0], 1);
        assert_eq!(bits(&buf.lower[..1]), bits(&[8.0]));
        assert_eq!(bits(&buf.upper[..1]), bits(&[8.0]));
    }

    #[test]
    fn inflow_shift_region_follows_the_load_patches() {
        let state = compact_state(2);
        let coupling = inflow_coupling_with_entries(2, &[(0, 3), (1, 5)]);
        let mut buf = PatchBuffer::new(
            &state,
            &[0, 1],
            &equipment_free_geometry(&[3]),
            std::slice::from_ref(&coupling),
        );
        let load_rhs = [100.0_f64, 90.0, 80.0, 200.0, 190.0, 180.0];
        buf.fill_load_patches(
            BlockRowFamily::per_block(20..26),
            BlockGrid::new(3, 1),
            &load_rhs,
            &[0, 1],
            &[],
        );

        buf.fill_inflow_shift_patches(&coupling, &[1.0, 2.0], &[]);

        assert_eq!(buf.indices[..6], [20, 21, 22, 23, 24, 25]);
        assert_eq!(buf.indices[6..8], [3, 5]);
        assert_eq!(bits(&buf.lower[6..8]), bits(&[1.0, 2.0]));
        assert_eq!(buf.forward_patch_count(), 8);
    }

    /// `state_col_patch_count` returns N + n_buckets + A*K: the lag state has no
    /// LP column.
    #[test]
    fn state_col_patch_count_excludes_the_lag_region() {
        let buf = PatchBuffer::new(&state_layout_full(3, 2, vec![2]), &[], &[], &[]);
        // N + n_buckets + A*K = 3 + 0 + 1*2 = 5
        assert_eq!(buf.state_col_patch_count(), 5);
    }

    /// `state_col_patch_count` includes `n_buckets` alongside storage/anticipated.
    #[test]
    fn state_col_patch_count_includes_transit_bucket_count() {
        let state = state_layout_with_transit_buckets(3, 2, bucket_order(4), vec![2]);
        let buf = PatchBuffer::new(&state, &[], &[], &[]);
        // N + n_buckets + A*K = 3 + 4 + 1*2 = 9
        assert_eq!(buf.state_col_patch_count(), 9);
    }

    /// `state_col_patch_count` equals `N + n_buckets + A*K`, pinned on a
    /// fixture whose anticipated ring is nonzero.
    #[test]
    fn state_col_patch_count_matches_hydro_bucket_anticipated_formula() {
        let state = state_layout_with_transit_buckets(3, 2, bucket_order(4), vec![3, 3]);
        let buf = PatchBuffer::new(&state, &[], &[], &[]);
        // N + n_buckets + A*K = 3 + 4 + 2*3 = 13
        assert_eq!(buf.state_col_patch_count(), 13);
    }

    /// Column buffer is zero-initialised at construction.
    #[test]
    fn col_buffer_zero_initialised() {
        let buf = PatchBuffer::new(&state_layout(3, 2), &[], &[], &[]);
        assert_eq!(buf.col_indices.len(), 3);
        assert!(
            buf.col_indices.iter().all(|&v| v == 0),
            "col_indices not zero-initialised"
        );
        assert!(
            buf.col_lower.iter().all(|&v| v == 0.0),
            "col_lower not zero-initialised"
        );
        assert!(
            buf.col_upper.iter().all(|&v| v == 0.0),
            "col_upper not zero-initialised"
        );
    }

    /// `forward_patch_count` is zero before any row-patch fill call.
    #[test]
    fn forward_patch_count_zero_before_any_fill() {
        let buf = PatchBuffer::new(&state_layout(3, 2), &[], &[], &[]);
        assert_eq!(buf.forward_patch_count(), 0);
    }

    #[test]
    fn clone_and_debug() {
        let buf = PatchBuffer::new(&state_layout(3, 2), &[], &[], &[]);
        let cloned = buf.clone();
        assert_eq!(cloned.indices.len(), buf.indices.len());

        let s = format!("{buf:?}");
        assert!(s.contains("PatchBuffer"));
    }

    // -------------------------------------------------------------------------
    // Load-balance unit tests
    // -------------------------------------------------------------------------

    /// AC (capacity): a 1-bus load region up to 3 blocks → row capacity
    /// M*B = 3. Also covers 2 load buses, up to 3 blocks → M*B = 6.
    #[test]
    fn new_with_load_allocates_correct_capacity() {
        let geometry = equipment_free_geometry(&[3]);

        let state = state_layout(2, 1);
        let buf = PatchBuffer::new(&state, &[0], &geometry, &[]);
        assert_eq!(buf.indices.len(), 3);
        assert_eq!(buf.lower.len(), 3);
        assert_eq!(buf.upper.len(), 3);

        let state_load = state_layout(3, 2);
        let buf_load = PatchBuffer::new(&state_load, &[0, 1], &geometry, &[]);
        assert_eq!(buf_load.indices.len(), 6);
    }

    /// Load-balance row indices follow `row = load_rows.start() + bus_positions[i] * n_blocks + blk`.
    ///
    /// With `n_load_buses=2, n_blocks=2, bus_positions=[0,1], load_rows` starting at 100:
    /// load patches start at slot 0 so indices[0..4] = [100, 101, 102, 103].
    #[test]
    fn fill_load_patches_correct_indices() {
        // M=2, B=2 → row capacity = M*B = 4
        let state = state_layout(0, 0);
        let geometry = equipment_free_geometry(&[2]);
        let mut buf = PatchBuffer::new(&state, &[0, 1], &geometry, &[]);
        let load_rhs = [300.0_f64, 280.0, 500.0, 450.0];
        let bus_positions = [0_usize, 1];
        let load_rows = BlockRowFamily::per_block(100..104);
        buf.fill_load_patches(
            load_rows,
            BlockGrid::new(2, 1),
            &load_rhs,
            &bus_positions,
            &[],
        );

        assert_eq!(buf.indices[0], 100); // bus 0, blk 0
        assert_eq!(buf.indices[1], 101); // bus 0, blk 1
        assert_eq!(buf.indices[2], 102); // bus 1, blk 0
        assert_eq!(buf.indices[3], 103); // bus 1, blk 1
    }

    /// Load-balance lower and upper bounds equal the corresponding `load_rhs` value.
    #[test]
    fn fill_load_patches_correct_values() {
        let state = state_layout(0, 0);
        let geometry = equipment_free_geometry(&[2]);
        let mut buf = PatchBuffer::new(&state, &[0, 1], &geometry, &[]);
        let load_rhs = [300.0_f64, 280.0, 500.0, 450.0];
        let bus_positions = [0_usize, 1];
        let load_rows = BlockRowFamily::per_block(100..104);
        buf.fill_load_patches(
            load_rows,
            BlockGrid::new(2, 1),
            &load_rhs,
            &bus_positions,
            &[],
        );

        assert_eq!(buf.lower[0], 300.0);
        assert_eq!(buf.upper[0], 300.0);
        assert_eq!(buf.lower[1], 280.0);
        assert_eq!(buf.upper[1], 280.0);
        assert_eq!(buf.lower[2], 500.0);
        assert_eq!(buf.upper[2], 500.0);
        assert_eq!(buf.lower[3], 450.0);
        assert_eq!(buf.upper[3], 450.0);
    }

    /// Every load patch must be an equality constraint: `lower[i] == upper[i]`.
    #[test]
    fn fill_load_patches_equality_constraints() {
        let state = state_layout(3, 2);
        let geometry = equipment_free_geometry(&[3]);
        let mut buf = PatchBuffer::new(&state, &[0, 1], &geometry, &[]);

        let load_rhs = [100.0_f64, 90.0, 80.0, 200.0, 190.0, 180.0];
        let bus_positions = [0_usize, 1];
        let load_rows = BlockRowFamily::per_block(20..26);
        buf.fill_load_patches(
            load_rows,
            BlockGrid::new(3, 1),
            &load_rhs,
            &bus_positions,
            &[],
        );

        let count = buf.forward_patch_count();
        for i in 0..count {
            assert_eq!(
                buf.lower[i],
                buf.upper[i],
                "patch {i}: lower {lo} != upper {up}",
                lo = buf.lower[i],
                up = buf.upper[i],
            );
        }
    }

    /// `forward_patch_count` includes load-balance patches after `fill_load_patches`.
    ///
    /// M=2, n_blocks=3 → forward_patch_count = M*n_blocks = 6 (no inflow shift fill).
    #[test]
    fn forward_patch_count_includes_load() {
        let state = state_layout(3, 2);
        let geometry = equipment_free_geometry(&[3]);
        let mut buf = PatchBuffer::new(&state, &[0, 1], &geometry, &[]);

        let load_rhs = [100.0_f64, 90.0, 80.0, 200.0, 190.0, 180.0];
        let bus_positions = [0_usize, 1];
        let load_rows = BlockRowFamily::per_block(20..26);
        buf.fill_load_patches(
            load_rows,
            BlockGrid::new(3, 1),
            &load_rhs,
            &bus_positions,
            &[],
        );

        assert_eq!(buf.forward_patch_count(), 6); // M*n_blocks=6
    }

    /// Bus positions differ from slot order; each patched row must equal
    /// `start + bus_pos * n_blks + blk`, never `start + slot * n_blks + blk`.
    #[test]
    fn load_patches_address_rows_by_bus_position_not_slot() {
        let state = state_layout(0, 0);
        let geometry = equipment_free_geometry(&[2]);
        let mut buf = PatchBuffer::new(&state, &[0, 1, 2], &geometry, &[]);
        let load_rhs = [0.0_f64; 6];
        let bus_positions = [2_usize, 0, 1];
        let load_rows = BlockRowFamily::per_block(10..16);
        buf.fill_load_patches(
            load_rows,
            BlockGrid::new(2, 1),
            &load_rhs,
            &bus_positions,
            &[],
        );

        for (i, &bus_pos) in bus_positions.iter().enumerate() {
            for blk in 0..2 {
                assert_eq!(buf.indices[i * 2 + blk], 10 + bus_pos * 2 + blk);
            }
        }
    }

    // -------------------------------------------------------------------------
    // fill_col_state_patches unit tests
    // -------------------------------------------------------------------------

    /// Storage col_indices[0..3] = [storage_in.start, +1, +2].
    #[test]
    fn fill_col_state_patches_storage_indices() {
        let state = [10.0_f64, 20.0, 30.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let state_layout = state_layout(3, 2);
        let mut buf = PatchBuffer::new(&state_layout, &[], &[], &[]);
        buf.fill_col_state_patches(
            &state_layout,
            &state,
            &[],
            Some(&unbounded_state_box(state.len())),
        );

        let s = state_layout.storage_in.start;
        assert_eq!(buf.col_indices[0], s);
        assert_eq!(buf.col_indices[1], s + 1);
        assert_eq!(buf.col_indices[2], s + 2);
    }

    /// Storage col_lower[0..3] == col_upper[0..3] == [10.0, 20.0, 30.0].
    #[test]
    fn fill_col_state_patches_storage_values() {
        let state = [10.0_f64, 20.0, 30.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let state_layout = state_layout(3, 2);
        let mut buf = PatchBuffer::new(&state_layout, &[], &[], &[]);
        buf.fill_col_state_patches(
            &state_layout,
            &state,
            &[],
            Some(&unbounded_state_box(state.len())),
        );

        assert_eq!(buf.col_lower[0], 10.0);
        assert_eq!(buf.col_upper[0], 10.0);
        assert_eq!(buf.col_lower[1], 20.0);
        assert_eq!(buf.col_upper[1], 20.0);
        assert_eq!(buf.col_lower[2], 30.0);
        assert_eq!(buf.col_upper[2], 30.0);
    }

    /// Anticipated column-bound patches for N=0, L=0, A=1, K=2.
    ///
    /// col_indices[0..2] = [commit_in.start, +1],
    /// col_lower[0..2] == col_upper[0..2] == [7.0, 11.0] (slot-major / plant-minor).
    #[test]
    fn fill_col_state_patches_anticipated_state() {
        // N=0, A=1, K=2 anticipated-only state layout.
        let state_layout = state_layout_full(0, 0, vec![2]);

        // n_state = 0 + 1*2 = 2; the state-VECTOR anticipated block is
        // `commit_out` (== 0 here), NOT the relocated incoming
        // `commit_in` (== 2 here).
        let ant_state_vec_start = state_layout.commit_out.start;
        let ant_incoming_col_start = state_layout.commit_in.start;
        let mut state = vec![0.0_f64; state_layout.n_state];
        state[ant_state_vec_start] = 7.0;
        state[ant_state_vec_start + 1] = 11.0;

        let mut buf = PatchBuffer::new(&state_layout, &[], &[], &[]);
        buf.fill_col_state_patches(
            &state_layout,
            &state,
            &[],
            Some(&unbounded_state_box(state.len())),
        );

        assert_eq!(buf.col_indices[0], ant_incoming_col_start);
        assert_eq!(buf.col_indices[1], ant_incoming_col_start + 1);
        assert_eq!(buf.col_lower[0], 7.0);
        assert_eq!(buf.col_upper[0], 7.0);
        assert_eq!(buf.col_lower[1], 11.0);
        assert_eq!(buf.col_upper[1], 11.0);
    }

    /// Pin-exactness: with `col_scale = 1.0` at the anticipated ring's incoming
    /// columns (D1's unscale override), `fill_col_state_patches` writes
    /// `col_lower == col_upper == v` — the raw commitment, no division — unlike a
    /// non-unit scale elsewhere in the same array, whose `v / d` round-trip is not
    /// guaranteed bit-exact.
    #[test]
    fn fill_col_state_patches_anticipated_state_unscaled_is_exact() {
        let state_layout = state_layout_full(0, 0, vec![2]);
        let ant_state_vec_start = state_layout.commit_out.start;
        let ant_incoming_col_start = state_layout.commit_in.start;
        let mut state = vec![0.0_f64; state_layout.n_state];
        state[ant_state_vec_start] = 7.0;
        state[ant_state_vec_start + 1] = 11.0;

        let ncols = state_layout.commit_in.end;
        let mut col_scale = vec![3.0_f64; ncols];
        col_scale[ant_incoming_col_start] = 1.0;
        col_scale[ant_incoming_col_start + 1] = 1.0;

        let mut buf = PatchBuffer::new(&state_layout, &[], &[], &[]);
        buf.fill_col_state_patches(
            &state_layout,
            &state,
            &col_scale,
            Some(&unbounded_state_box(state.len())),
        );

        assert_eq!(buf.col_lower[0], 7.0);
        assert_eq!(buf.col_upper[0], 7.0);
        assert_eq!(buf.col_lower[1], 11.0);
        assert_eq!(buf.col_upper[1], 11.0);
    }

    /// Every patch in the active col region has col_lower[i] == col_upper[i].
    #[test]
    fn fill_col_state_patches_equality_constraints() {
        let state = [10.0_f64, 20.0, 30.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let state_layout = state_layout(3, 2);
        let mut buf = PatchBuffer::new(&state_layout, &[], &[], &[]);
        buf.fill_col_state_patches(
            &state_layout,
            &state,
            &[],
            Some(&unbounded_state_box(state.len())),
        );

        let count = buf.state_col_patch_count();
        for i in 0..count {
            assert_eq!(
                buf.col_lower[i],
                buf.col_upper[i],
                "col patch {i}: lower {lo} != upper {up}",
                lo = buf.col_lower[i],
                up = buf.col_upper[i],
            );
        }
    }

    /// col_scale divides: col_lower[h] == state[h] / col_scale[col] for storage.
    ///
    /// With col_scale[storage_in.start + h] = 2.0 for all h, and state = [10, 20, 30, ...],
    /// expected col_lower[0..3] = [5.0, 10.0, 15.0].
    #[test]
    fn fill_col_state_patches_unscaled_with_col_scale() {
        let state_layout = state_layout(3, 2);
        let state = [10.0_f64, 20.0, 30.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut buf = PatchBuffer::new(&state_layout, &[], &[], &[]);

        // Build a col_scale long enough to cover commit_in.end.
        // Fill with 1.0 everywhere, then override the storage_in columns to 2.0.
        let ncols = state_layout.commit_in.end.max(state_layout.storage_in.end);
        let mut col_scale = vec![1.0_f64; ncols];
        let s = state_layout.storage_in.start;
        col_scale[s] = 2.0;
        col_scale[s + 1] = 2.0;
        col_scale[s + 2] = 2.0;

        buf.fill_col_state_patches(
            &state_layout,
            &state,
            &col_scale,
            Some(&unbounded_state_box(state.len())),
        );

        assert_eq!(buf.col_lower[0], 5.0);
        assert_eq!(buf.col_upper[0], 5.0);
        assert_eq!(buf.col_lower[1], 10.0);
        assert_eq!(buf.col_upper[1], 10.0);
        assert_eq!(buf.col_lower[2], 15.0);
        assert_eq!(buf.col_upper[2], 15.0);
    }

    /// When n_anticipated == 0, the anticipated region is empty and state_col_patch_count() == N.
    #[test]
    fn fill_col_state_patches_zero_anticipated_collapses_correctly() {
        let state = [10.0_f64, 20.0, 30.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let state_layout = state_layout(3, 2);
        let mut buf = PatchBuffer::new(&state_layout, &[], &[], &[]);
        buf.fill_col_state_patches(
            &state_layout,
            &state,
            &[],
            Some(&unbounded_state_box(state.len())),
        );

        // N + A*K = 3 + 0 = 3
        assert_eq!(buf.state_col_patch_count(), 3);
        assert_eq!(buf.col_indices.len(), 3);
    }

    /// After fill_col_state_patches, the row buffer (indices/lower/upper) is untouched.
    ///
    /// Catches accidental cross-buffer writes; the row buffer must remain
    /// zero-initialised since no row-equality filler has been called.
    #[test]
    fn row_buffer_unchanged_after_fill_col_state_patches() {
        let state = [10.0_f64, 20.0, 30.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let state_layout = state_layout(3, 2);
        let geometry = equipment_free_geometry(&[2]);
        let mut buf = PatchBuffer::new(&state_layout, &[0], &geometry, &[]);
        assert_eq!(buf.indices.len(), 2);
        buf.fill_col_state_patches(
            &state_layout,
            &state,
            &[],
            Some(&unbounded_state_box(state.len())),
        );

        assert!(
            buf.indices.iter().all(|&v| v == 0),
            "row indices modified by fill_col_state_patches"
        );
        assert!(
            buf.lower.iter().all(|&v| v == 0.0),
            "row lower modified by fill_col_state_patches"
        );
        assert!(
            buf.upper.iter().all(|&v| v == 0.0),
            "row upper modified by fill_col_state_patches"
        );
    }

    // -------------------------------------------------------------------------
    // Travel-time bucket column-bound patches (single-owner refactor)
    // -------------------------------------------------------------------------

    /// Every travel-time bucket incoming column receives a state-col patch,
    /// pinned to the value supplied in `state` — the trial point /
    /// decision-driven state, sourced once per stage-visit before any
    /// per-opening loop (contrast NCS availability, which patches per opening).
    #[test]
    fn fill_col_state_patches_every_transit_bucket_incoming_column_is_pinned() {
        let state_layout = state_layout_with_transit_buckets(
            3,
            2,
            vec![(HydroSys::new(0), 0), (HydroSys::new(0), 1)],
            vec![],
        );
        let bucket_dims = state_layout.state_dim_bucket_range();
        let mut state = vec![0.0_f64; state_layout.n_state];
        state[bucket_dims.start] = 100.0;
        state[bucket_dims.start + 1] = 200.0;

        let mut buf = PatchBuffer::new(&state_layout, &[], &[], &[]);
        buf.fill_col_state_patches(
            &state_layout,
            &state,
            &[],
            Some(&unbounded_state_box(state.len())),
        );

        assert_eq!(
            buf.state_col_patch_count(),
            state_layout.n_state - state_layout.state_dim_lag_range().len()
        );

        for (i, j) in bucket_dims.enumerate() {
            let col = state_layout.transit_buckets_in.start + i;
            let slot = state_layout.transit_buckets_out.start + i;
            assert_eq!(
                buf.col_indices[slot], col,
                "bucket state index {j} must pin transit_buckets_in column {col}"
            );
            assert_eq!(buf.col_lower[slot], state[j]);
            assert_eq!(buf.col_upper[slot], state[j]);
        }
    }

    /// With both buckets (`B=2`) and anticipated thermals (`A=1, K=2`), the
    /// anticipated patches land on the slots shifted by `+B`: the anticipated
    /// slots occupy buffer positions `[N+B, N+B+A*K)`, not the unshifted
    /// `[N, N+A*K)` a hardcoded `anticipated_start = N` would (incorrectly)
    /// target.
    #[test]
    fn fill_col_state_patches_anticipated_lands_at_shifted_offset() {
        let n = 3;
        let l = 2;
        let n_buckets = 2;
        let state_layout = state_layout_with_transit_buckets(
            n,
            l,
            vec![(HydroSys::new(0), 0), (HydroSys::new(0), 1)],
            vec![2],
        );
        let unshifted_anticipated_start = n;
        let shifted_anticipated_start = unshifted_anticipated_start + n_buckets;
        assert_eq!(state_layout.commit_out.start, shifted_anticipated_start);

        let bucket_dims = state_layout.state_dim_bucket_range();
        let commit_dims = state_layout.state_dim_commitment_hold_range();
        let mut state = vec![0.0_f64; state_layout.n_state];
        state[bucket_dims.start] = 100.0;
        state[bucket_dims.start + 1] = 200.0;
        state[commit_dims.start] = 7.0;
        state[commit_dims.start + 1] = 11.0;

        let mut buf = PatchBuffer::new(&state_layout, &[], &[], &[]);
        buf.fill_col_state_patches(
            &state_layout,
            &state,
            &[],
            Some(&unbounded_state_box(state.len())),
        );

        // Buckets occupy the unshifted slots; anticipated occupies the shifted ones.
        assert_eq!(buf.col_lower[unshifted_anticipated_start], 100.0);
        assert_eq!(buf.col_lower[unshifted_anticipated_start + 1], 200.0);
        assert_eq!(buf.col_lower[shifted_anticipated_start], 7.0);
        assert_eq!(buf.col_lower[shifted_anticipated_start + 1], 11.0);
        // The pinned LP column is the incoming `commit_in` range, not the
        // outgoing block the slot index sits in.
        assert_eq!(
            buf.col_indices[shifted_anticipated_start],
            state_layout.commit_in.start
        );
        assert_eq!(
            buf.col_indices[shifted_anticipated_start + 1],
            state_layout.commit_in.start + 1
        );
        assert_eq!(
            buf.state_col_patch_count(),
            state_layout.n_state - state_layout.state_dim_lag_range().len()
        );
    }

    /// A compact layout has no lag columns: the pins are the non-lag dimensions in
    /// ascending order packed from slot 0, so the bucket and anticipated pins sit
    /// `N*L` slots below their state indices.
    #[test]
    fn fill_col_state_patches_skips_the_lag_region_of_a_compact_layout() {
        let lead_stages = vec![1];
        let layout = StateSpace::new(
            2,
            1,
            vec![(HydroSys::new(0), 1), (HydroSys::new(0), 2)],
            lead_stages.clone(),
            constant_lead_resolution(&lead_stages, 3),
            &[1; 2],
        );
        let lag_dims = layout.state_dim_lag_range();
        assert_eq!(lag_dims.len(), 2);
        let state: Vec<f64> = (0..layout.n_state)
            .map(|j| 10.0 + f64::from(u8::try_from(j).unwrap()))
            .collect();

        let mut buf = PatchBuffer::new(&layout, &[], &[], &[]);
        buf.fill_col_state_patches(
            &layout,
            &state,
            &[],
            Some(&unbounded_state_box(layout.n_state)),
        );

        assert_eq!(buf.col_indices.len(), layout.n_state - lag_dims.len());
        assert_eq!(
            buf.active_state_col_patch_count(),
            buf.state_col_patch_count()
        );
        assert_eq!(buf.active_state_col_patch_count(), layout.n_state - 2);
        let columned = (0..layout.n_state).filter(|j| !lag_dims.contains(j));
        for (slot, j) in columned.enumerate() {
            let col = layout.state_to_lp_incoming_column(StateDim::new(j)).get();
            assert_eq!(buf.col_indices[slot], col, "pin {slot} of state dim {j}");
            assert_eq!(buf.col_lower[slot], state[j], "pin {slot} lower");
            assert_eq!(buf.col_upper[slot], state[j], "pin {slot} upper");
        }
    }

    /// `B == 0`: `fill_col_state_patches` output equals the per-family formula
    /// (storage → `storage_in`, anticipated → `commit_in.start + slot*A +
    /// plant`); the lag state has no column.
    #[test]
    #[expect(
        clippy::cast_precision_loss,
        reason = "test counts are small, far inside f64's exact-integer range"
    )]
    fn fill_col_state_patches_b_zero_matches_the_per_family_formula() {
        let n = 3;
        let l = 2;
        let a = 2;
        let k = 3;
        let state_layout = state_layout_full(n, l, vec![k; a]);
        assert_eq!(state_layout.n_buckets, 0);

        let state: Vec<f64> = (0..state_layout.n_state)
            .map(|i| (i as f64).mul_add(1.5, 1.0))
            .collect();
        let scale_len = state_layout.commit_in.end.max(state_layout.storage_in.end);
        let col_scale: Vec<f64> = (0..scale_len)
            .map(|i| (i as f64).mul_add(0.1, 1.0))
            .collect();

        let n_pins = n + a * k;
        let mut expected_indices = vec![0usize; n_pins];
        let mut expected_lower = vec![0.0_f64; n_pins];

        let storage_in_start = state_layout.storage_in.start;
        for h in 0..n {
            let col = storage_in_start + h;
            expected_indices[h] = col;
            expected_lower[h] = state[h] / col_scale[col];
        }
        let anticipated_state_start = n * (1 + l);
        let ant_state_col_start = state_layout.commit_in.start;
        for off in 0..k * a {
            let col = ant_state_col_start + off;
            expected_indices[n + off] = col;
            expected_lower[n + off] = state[anticipated_state_start + off] / col_scale[col];
        }

        let mut buf = PatchBuffer::new(&state_layout, &[], &[], &[]);
        buf.fill_col_state_patches(
            &state_layout,
            &state,
            &col_scale,
            Some(&unbounded_state_box(state.len())),
        );

        assert_eq!(buf.col_indices, expected_indices);
        assert_eq!(buf.col_lower, expected_lower);
        assert_eq!(buf.col_upper, expected_lower);
    }

    /// A `PatchBuffer` constructed with `n_buckets = 0` panics (index out of
    /// bounds) when filled against a bucket-aware `StateSpace` — the sizing
    /// contract between `PatchBuffer::new`'s `n_buckets` and the layout it
    /// patches must match, or the column-bound region has no room for the
    /// bucket slots.
    #[test]
    #[should_panic(expected = "index out of bounds")]
    fn fill_col_state_patches_undersized_buffer_panics() {
        let narrow_state = state_layout(3, 2);
        let mut buf = PatchBuffer::new(&narrow_state, &[], &[], &[]);
        let state_layout = state_layout_with_transit_buckets(
            3,
            2,
            vec![(HydroSys::new(0), 0), (HydroSys::new(0), 1)],
            vec![],
        );
        let state = vec![0.0_f64; state_layout.n_state];
        buf.fill_col_state_patches(
            &state_layout,
            &state,
            &[],
            Some(&unbounded_state_box(state.len())),
        );
    }

    /// A pinned incoming state dimension outside its admissible box panics via
    /// the pin-time debug_assert — read-back is the single canonicalization
    /// point, so an out-of-box pin here means an entry point bypassed it.
    #[test]
    #[should_panic(expected = "outside admissible box")]
    fn fill_col_state_patches_panics_on_out_of_box_pin() {
        let state = [10.0_f64, 20.0, 30.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let state_layout = state_layout(3, 2);
        let mut buf = PatchBuffer::new(&state_layout, &[], &[], &[]);
        let mut state_box = unbounded_state_box(state.len());
        state_box.upper[0] = 5.0;

        buf.fill_col_state_patches(&state_layout, &state, &[], Some(&state_box));
    }
}
