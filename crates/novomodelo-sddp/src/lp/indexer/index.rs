//! Base typed vocabulary for LP addresses and state-vector dimensions.
//!
//! A bare `usize` does not distinguish a state-vector dimension from an LP
//! column, an incoming-state column from an outgoing-state one, or a block
//! index from any of those — any of those mismatches still compiles and
//! silently addresses the wrong cell. [`StateDim`] makes the state-vector
//! dimension its own type; [`InCol`]/[`OutCol`] split the column role the
//! state-path resolvers return, so a render call site handed an incoming
//! column (or vice versa) is a compile error, not
//! a silently wrong cut coefficient — on the paths that consume the typed
//! resolvers (the cut pin/extract/render hot path, simulation extraction, and
//! the policy manifest); a consumer indexing a raw buffer through
//! `StateSpace`'s public ranges sits outside that guarantee until the ranges
//! are sealed; [`CutSlot`] indexes the enabled cut-state
//! subset a projection reindexes into, distinct from [`StateDim`]; [`BlockIdx`]
//! makes the block operand of
//! every [`BlockGrid`](super::BlockGrid) shape method its own type. None of these
//! carries arithmetic: offset formulas stay with the owning value type
//! ([`BlockGrid`](super::BlockGrid),
//! [`StorageBoundaryGrid`](super::StorageBoundaryGrid), `RangeCursor`,
//! `DeliveryRing`) — these types only gate which `usize` crosses which
//! boundary. Every type is `#[repr(transparent)]` and extracts its raw
//! `usize` via `get()` for the solver-FFI seam, which stays `usize`-only per
//! the infra-genericity rule.

/// A state-vector dimension index `j ∈ [0, n_state)` — not an LP address;
/// resolve it through the owning state layout's resolver before indexing an
/// LP buffer.
///
/// A raw `StateDim` cannot stand in for the [`OutCol`] a resolver returns:
///
/// ```compile_fail
/// use cobre_sddp::indexer::{OutCol, StateDim};
///
/// let dim = StateDim::new(0);
/// let _unresolved: OutCol = dim; // StateDim substituted for a resolved OutCol
/// ```
///
/// Orderable, unlike every other newtype in this module: it is the sole
/// sorted-membership key (`nonzero_state_indices.binary_search`), so it alone
/// derives `Ord`/`PartialOrd`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(transparent)]
pub struct StateDim(usize);

impl StateDim {
    /// Wrap a raw state-vector dimension index.
    #[inline]
    #[must_use]
    pub fn new(dim: usize) -> Self {
        Self(dim)
    }

    /// Extract the raw state-vector dimension index.
    #[inline]
    #[must_use]
    pub fn get(self) -> usize {
        self.0
    }
}

/// Index into a stage's enabled cut-state subset — distinct from
/// [`StateDim`], the global state-vector dimension: under a non-identity
/// projection the two diverge.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct CutSlot(usize);

impl CutSlot {
    /// Wrap a raw cut-slot index.
    #[inline]
    #[must_use]
    pub fn new(slot: usize) -> Self {
        Self(slot)
    }

    /// Extract the raw cut-slot index.
    #[inline]
    #[must_use]
    pub fn get(self) -> usize {
        self.0
    }
}

/// An incoming-state LP column — the pin target (`set_col_bounds`) and the
/// dual-extraction read (`reduced_costs[col]`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct InCol(usize);

impl InCol {
    /// Wrap a raw incoming-state LP column index.
    #[inline]
    #[must_use]
    pub fn new(col: usize) -> Self {
        Self(col)
    }

    /// Extract the raw LP column index for the solver-FFI seam.
    #[inline]
    #[must_use]
    pub fn get(self) -> usize {
        self.0
    }
}

/// A block index `blk ∈ 0..n_blks` — the block operand of every
/// [`BlockGrid`](super::BlockGrid) shape method.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct BlockIdx(usize);

impl BlockIdx {
    /// Wrap a raw block index.
    #[inline]
    #[must_use]
    pub fn new(blk: usize) -> Self {
        Self(blk)
    }

    /// Extract the raw block index immediately before a `BlockGrid` offset
    /// formula.
    #[inline]
    #[must_use]
    pub fn get(self) -> usize {
        self.0
    }
}

/// An outgoing-state LP column — the forward-pass/DCS cut-row render target.
///
/// An [`InCol`] cannot stand in for it: the render path (e.g.
/// `cut::row::push_scaled_coefficient`'s `col: OutCol` parameter) rejects the
/// incoming-column role the pin/dual-extraction path uses.
///
/// ```compile_fail
/// use cobre_sddp::indexer::{InCol, OutCol};
///
/// fn render(_outgoing: OutCol) {}
///
/// render(InCol::new(3)); // render path handed an InCol, not the OutCol it requires
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct OutCol(usize);

impl OutCol {
    /// Wrap a raw outgoing-state LP column index.
    #[inline]
    #[must_use]
    pub fn new(col: usize) -> Self {
        Self(col)
    }

    /// Extract the raw LP column index for the solver-FFI seam.
    #[inline]
    #[must_use]
    pub fn get(self) -> usize {
        self.0
    }
}

/// The chronological storage-boundary classification `k ∈ 0..=n_blks`:
/// `Incoming` (`k = 0`) and `Outgoing` (`k = n_blks`) are STATE columns,
/// `Interior(k)` (`k ∈ 1..n_blks`) is a CONTROL column. Matched exhaustively
/// by [`StorageBoundaryGrid::col`](super::StorageBoundaryGrid::col) with no
/// `_` arm, so the outgoing endpoint cannot be silently absorbed by an
/// interior catch-all:
///
/// ```compile_fail
/// use cobre_sddp::indexer::Boundary;
///
/// fn interior_catch_all(b: Boundary) -> usize {
///     match b {
///         Boundary::Incoming => 0,
///         Boundary::Interior(k) => k, // Outgoing omitted, no `_` arm
///     }
/// }
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Boundary {
    /// The incoming-state boundary `k = 0` (`S⁰`).
    Incoming,
    /// An interior control boundary, carrying the raw `k ∈ 1..n_blks`.
    Interior(usize),
    /// The outgoing-state boundary `k = n_blks` (`Sᴷ`).
    Outgoing,
}

impl Boundary {
    /// Classify a dynamic `k ∈ 0..=n_blks` into its [`Boundary`] variant.
    #[inline]
    #[must_use]
    pub fn from_index(k: usize, n_blks: usize) -> Boundary {
        if k == 0 {
            Boundary::Incoming
        } else if k == n_blks {
            Boundary::Outgoing
        } else {
            Boundary::Interior(k)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{BlockIdx, Boundary, CutSlot, InCol, OutCol, StateDim};

    #[test]
    fn boundary_from_index_classifies_endpoints_and_interior() {
        const K: usize = 5;
        assert_eq!(Boundary::from_index(0, K), Boundary::Incoming);
        assert_eq!(Boundary::from_index(K, K), Boundary::Outgoing);
        for k in 1..K {
            assert_eq!(Boundary::from_index(k, K), Boundary::Interior(k));
        }
    }

    #[test]
    fn cut_slot_is_zero_cost_and_round_trips() {
        assert_eq!(std::mem::size_of::<CutSlot>(), std::mem::size_of::<usize>());
        assert_eq!(CutSlot::new(11).get(), 11);
    }

    #[test]
    fn block_idx_is_zero_cost_and_round_trips() {
        assert_eq!(
            std::mem::size_of::<BlockIdx>(),
            std::mem::size_of::<usize>()
        );
        assert_eq!(BlockIdx::new(2).get(), 2);
    }

    #[test]
    fn state_dim_is_zero_cost_and_round_trips() {
        assert_eq!(
            std::mem::size_of::<StateDim>(),
            std::mem::size_of::<usize>()
        );
        assert_eq!(StateDim::new(13).get(), 13);
    }

    #[test]
    fn in_col_is_zero_cost_and_round_trips() {
        assert_eq!(std::mem::size_of::<InCol>(), std::mem::size_of::<usize>());
        assert_eq!(InCol::new(5).get(), 5);
    }

    #[test]
    fn out_col_is_zero_cost_and_round_trips() {
        assert_eq!(std::mem::size_of::<OutCol>(), std::mem::size_of::<usize>());
        assert_eq!(OutCol::new(9).get(), 9);
    }

    /// `InCol`/`OutCol` are distinct types even when constructed from the same
    /// raw value — the compile-time guard the in/out role split exists for
    /// cannot be demonstrated by a runtime assertion, so this test pins only
    /// what runtime code CAN observe: each wraps and returns its own value
    /// independently of the other's existence.
    #[test]
    fn in_col_and_out_col_round_trip_independently() {
        assert_eq!(InCol::new(7).get(), 7);
        assert_eq!(OutCol::new(7).get(), 7);
    }
}
