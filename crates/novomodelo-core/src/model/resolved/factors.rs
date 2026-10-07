//! Pre-resolved per-block factor and NCS-availability lookup tables, consumed on
//! the LP-building hot path. Absent factor entries return the no-scaling identity
//! `1.0`; absent NCS availability returns `0.0`.
//! Populated by `cobre-io`; never modified after construction.

/// Pre-resolved per-block load scaling factors.
///
/// O(1) lookup by `(bus_index, stage_index, block_index)` into dense 3D storage
/// (`n_buses * n_stages * max_blocks`); `1.0` for absent entries (no scaling).
///
/// # Examples
///
/// ```
/// use cobre_core::resolved::ResolvedLoadFactors;
///
/// let empty = ResolvedLoadFactors::empty();
/// assert!((empty.factor(0, 0, 0) - 1.0).abs() < f64::EPSILON);
/// ```
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ResolvedLoadFactors {
    /// Flat 3D array indexed `(bus_idx * n_stages + stage_idx) * max_blocks + block_idx`.
    factors: Vec<f64>,
    n_stages: usize,
    max_blocks: usize,
}

impl ResolvedLoadFactors {
    /// Create an empty load factors table; all lookups return `1.0`.
    ///
    /// The default when no `load_factors.json` exists.
    ///
    /// # Examples
    ///
    /// ```
    /// use cobre_core::resolved::ResolvedLoadFactors;
    ///
    /// let t = ResolvedLoadFactors::empty();
    /// assert!((t.factor(5, 3, 2) - 1.0).abs() < f64::EPSILON);
    /// ```
    #[must_use]
    pub fn empty() -> Self {
        Self {
            factors: Vec::new(),
            n_stages: 0,
            max_blocks: 0,
        }
    }

    /// Create a new load factors table with the given dimensions.
    ///
    /// All entries are initialized to `1.0` (no scaling). Use [`set`] to
    /// populate individual entries.
    ///
    /// [`set`]: Self::set
    #[must_use]
    pub fn new(n_buses: usize, n_stages: usize, max_blocks: usize) -> Self {
        Self {
            factors: vec![1.0; n_buses * n_stages * max_blocks],
            n_stages,
            max_blocks,
        }
    }

    /// Set the load factor for a specific `(bus_idx, stage_idx, block_idx)` triple.
    ///
    /// # Panics
    ///
    /// Panics if any index is out of bounds.
    pub fn set(&mut self, bus_idx: usize, stage_idx: usize, block_idx: usize, value: f64) {
        let idx = (bus_idx * self.n_stages + stage_idx) * self.max_blocks + block_idx;
        self.factors[idx] = value;
    }

    /// Look up the load factor for a `(bus_idx, stage_idx, block_idx)` triple.
    /// Returns `1.0` when the table is empty or the flat index lands past `Vec::len`.
    ///
    /// The `1.0` fallback only holds for indices past `Vec::len`; a per-dimension
    /// overflow that stays within `Vec::len` (e.g. `block_idx >= max_blocks` with a
    /// small `bus_idx`) aliases a neighbouring cell. Callers pass only in-range
    /// dimensions — do not rely on the fallback for arbitrary out-of-range triples.
    #[inline]
    #[must_use]
    pub fn factor(&self, bus_idx: usize, stage_idx: usize, block_idx: usize) -> f64 {
        if self.factors.is_empty() {
            return 1.0;
        }
        let idx = (bus_idx * self.n_stages + stage_idx) * self.max_blocks + block_idx;
        self.factors.get(idx).copied().unwrap_or(1.0)
    }
}

/// Pre-resolved per-stage NCS available generation bounds.
///
/// O(1) lookup of `available_generation_mw` by `(ncs_index, stage_index)` into
/// dense 2D storage (`n_ncs * n_stages`); `0.0` for out-of-bounds access. Each NCS
/// is initialized to its installed capacity (`max_generation_mw`), then
/// stage-varying entries from `constraints/ncs_bounds.parquet` overwrite individual cells.
///
/// # Examples
///
/// ```
/// use cobre_core::resolved::ResolvedNcsBounds;
///
/// let empty = ResolvedNcsBounds::empty();
/// assert!(empty.is_empty());
/// ```
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ResolvedNcsBounds {
    /// Flat 2D array indexed `ncs_idx * n_stages + stage_idx`.
    data: Vec<f64>,
    n_stages: usize,
}

impl ResolvedNcsBounds {
    /// Create an empty NCS bounds table.
    ///
    /// The default when no NCS entities exist or no bounds file is provided.
    ///
    /// # Examples
    ///
    /// ```
    /// use cobre_core::resolved::ResolvedNcsBounds;
    ///
    /// let t = ResolvedNcsBounds::empty();
    /// assert!(t.is_empty());
    /// ```
    #[must_use]
    pub fn empty() -> Self {
        Self {
            data: Vec::new(),
            n_stages: 0,
        }
    }

    /// Create a new NCS bounds table with per-entity defaults.
    ///
    /// All stages for NCS entity `i` are initialized to `default_mw[i]`
    /// (the installed capacity). Use [`set`] to apply stage-varying overrides.
    ///
    /// [`set`]: Self::set
    ///
    /// # Panics
    ///
    /// Panics if `default_mw.len() != n_ncs`.
    #[must_use]
    pub fn new(n_ncs: usize, n_stages: usize, default_mw: &[f64]) -> Self {
        assert!(
            default_mw.len() == n_ncs,
            "default_mw length ({}) must equal n_ncs ({n_ncs})",
            default_mw.len()
        );
        let mut data = vec![0.0; n_ncs * n_stages];
        for (ncs_idx, &mw) in default_mw.iter().enumerate() {
            data[ncs_idx * n_stages..(ncs_idx + 1) * n_stages].fill(mw);
        }
        Self { data, n_stages }
    }

    /// Set the available generation for a specific `(ncs_idx, stage_idx)` pair.
    ///
    /// # Panics
    ///
    /// Panics if any index is out of bounds.
    pub fn set(&mut self, ncs_idx: usize, stage_idx: usize, value: f64) {
        let idx = ncs_idx * self.n_stages + stage_idx;
        self.data[idx] = value;
    }

    /// Look up the available generation (MW) for a `(ncs_idx, stage_idx)` pair.
    ///
    /// Returns `0.0` when the index is out of bounds or the table is empty.
    #[inline]
    #[must_use]
    pub fn available_generation(&self, ncs_idx: usize, stage_idx: usize) -> f64 {
        if self.data.is_empty() {
            return 0.0;
        }
        let idx = ncs_idx * self.n_stages + stage_idx;
        self.data.get(idx).copied().unwrap_or(0.0)
    }

    /// Returns `true` when the table has no data.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

/// Pre-resolved per-block NCS generation scaling factors.
///
/// O(1) lookup by `(ncs_index, stage_index, block_index)` into dense 3D storage
/// (`n_ncs * n_stages * max_blocks`); `1.0` for absent entries (no scaling).
///
/// # Examples
///
/// ```
/// use cobre_core::resolved::ResolvedNcsFactors;
///
/// let empty = ResolvedNcsFactors::empty();
/// assert!((empty.factor(0, 0, 0) - 1.0).abs() < f64::EPSILON);
/// ```
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ResolvedNcsFactors {
    /// Flat 3D array indexed `(ncs_idx * n_stages + stage_idx) * max_blocks + block_idx`.
    factors: Vec<f64>,
    n_stages: usize,
    max_blocks: usize,
}

impl ResolvedNcsFactors {
    /// Create an empty NCS factors table; all lookups return `1.0`.
    ///
    /// The default when no `non_controllable_factors.json` exists.
    ///
    /// # Examples
    ///
    /// ```
    /// use cobre_core::resolved::ResolvedNcsFactors;
    ///
    /// let t = ResolvedNcsFactors::empty();
    /// assert!((t.factor(5, 3, 2) - 1.0).abs() < f64::EPSILON);
    /// ```
    #[must_use]
    pub fn empty() -> Self {
        Self {
            factors: Vec::new(),
            n_stages: 0,
            max_blocks: 0,
        }
    }

    /// Create a new NCS factors table with the given dimensions.
    ///
    /// All entries are initialized to `1.0` (no scaling). Use [`set`] to
    /// populate individual entries.
    ///
    /// [`set`]: Self::set
    #[must_use]
    pub fn new(n_ncs: usize, n_stages: usize, max_blocks: usize) -> Self {
        Self {
            factors: vec![1.0; n_ncs * n_stages * max_blocks],
            n_stages,
            max_blocks,
        }
    }

    /// Set the NCS factor for a specific `(ncs_idx, stage_idx, block_idx)` triple.
    ///
    /// # Panics
    ///
    /// Panics if any index is out of bounds.
    pub fn set(&mut self, ncs_idx: usize, stage_idx: usize, block_idx: usize, value: f64) {
        let idx = (ncs_idx * self.n_stages + stage_idx) * self.max_blocks + block_idx;
        self.factors[idx] = value;
    }

    /// Look up the NCS factor for a `(ncs_idx, stage_idx, block_idx)` triple.
    /// Returns `1.0` when the table is empty or the flat index lands past `Vec::len`;
    /// an in-range per-dimension overflow aliases a neighbouring cell (see
    /// [`ResolvedLoadFactors::factor`]).
    #[inline]
    #[must_use]
    pub fn factor(&self, ncs_idx: usize, stage_idx: usize, block_idx: usize) -> f64 {
        if self.factors.is_empty() {
            return 1.0;
        }
        let idx = (ncs_idx * self.n_stages + stage_idx) * self.max_blocks + block_idx;
        self.factors.get(idx).copied().unwrap_or(1.0)
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ─── ResolvedLoadFactors tests ─────────────────────────────────────────────

    #[test]
    fn test_load_factors_empty_returns_one() {
        let t = ResolvedLoadFactors::empty();
        assert!((t.factor(0, 0, 0) - 1.0).abs() < f64::EPSILON);
        assert!((t.factor(5, 3, 2) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_load_factors_new_default_is_one() {
        let t = ResolvedLoadFactors::new(2, 1, 3);
        for bus in 0..2 {
            for blk in 0..3 {
                assert!(
                    (t.factor(bus, 0, blk) - 1.0).abs() < f64::EPSILON,
                    "expected 1.0 at ({bus}, 0, {blk})"
                );
            }
        }
    }

    #[test]
    fn test_load_factors_set_and_get() {
        let mut t = ResolvedLoadFactors::new(2, 1, 3);
        t.set(0, 0, 0, 0.85);
        t.set(0, 0, 1, 1.15);
        assert!((t.factor(0, 0, 0) - 0.85).abs() < 1e-10);
        assert!((t.factor(0, 0, 1) - 1.15).abs() < 1e-10);
        assert!((t.factor(0, 0, 2) - 1.0).abs() < f64::EPSILON);
        assert!((t.factor(1, 0, 0) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_load_factors_out_of_bounds_returns_one() {
        let t = ResolvedLoadFactors::new(1, 1, 2);
        assert!((t.factor(5, 0, 0) - 1.0).abs() < f64::EPSILON);
        assert!((t.factor(0, 0, 99) - 1.0).abs() < f64::EPSILON);
    }

    // ─── ResolvedNcsBounds tests ──────────────────────────────────────────────

    #[test]
    fn test_ncs_bounds_empty_is_empty() {
        let t = ResolvedNcsBounds::empty();
        assert!(t.is_empty());
        assert!((t.available_generation(0, 0) - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_ncs_bounds_new_uses_defaults() {
        let t = ResolvedNcsBounds::new(2, 3, &[100.0, 200.0]);
        assert!(!t.is_empty());
        assert!((t.available_generation(0, 0) - 100.0).abs() < f64::EPSILON);
        assert!((t.available_generation(0, 2) - 100.0).abs() < f64::EPSILON);
        assert!((t.available_generation(1, 0) - 200.0).abs() < f64::EPSILON);
        assert!((t.available_generation(1, 2) - 200.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_ncs_bounds_set_and_get() {
        let mut t = ResolvedNcsBounds::new(2, 3, &[100.0, 200.0]);
        t.set(0, 1, 50.0);
        assert!((t.available_generation(0, 1) - 50.0).abs() < f64::EPSILON);
        assert!((t.available_generation(0, 0) - 100.0).abs() < f64::EPSILON);
        assert!((t.available_generation(1, 0) - 200.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_ncs_bounds_out_of_bounds_returns_zero() {
        let t = ResolvedNcsBounds::new(1, 1, &[100.0]);
        assert!((t.available_generation(5, 0) - 0.0).abs() < f64::EPSILON);
        assert!((t.available_generation(0, 99) - 0.0).abs() < f64::EPSILON);
    }

    // ─── ResolvedNcsFactors tests ─────────────────────────────────────────────

    #[test]
    fn test_ncs_factors_empty_returns_one() {
        let t = ResolvedNcsFactors::empty();
        assert!((t.factor(0, 0, 0) - 1.0).abs() < f64::EPSILON);
        assert!((t.factor(5, 3, 2) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_ncs_factors_new_default_is_one() {
        let t = ResolvedNcsFactors::new(2, 1, 3);
        for ncs in 0..2 {
            for blk in 0..3 {
                assert!(
                    (t.factor(ncs, 0, blk) - 1.0).abs() < f64::EPSILON,
                    "factor({ncs}, 0, {blk}) should be 1.0"
                );
            }
        }
    }

    #[test]
    fn test_ncs_factors_set_and_get() {
        let mut t = ResolvedNcsFactors::new(2, 1, 3);
        t.set(0, 0, 1, 0.8);
        assert!((t.factor(0, 0, 1) - 0.8).abs() < 1e-10);
        assert!((t.factor(0, 0, 0) - 1.0).abs() < f64::EPSILON);
        assert!((t.factor(1, 0, 0) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_ncs_factors_out_of_bounds_returns_one() {
        let t = ResolvedNcsFactors::new(1, 1, 2);
        assert!((t.factor(5, 0, 0) - 1.0).abs() < f64::EPSILON);
        assert!((t.factor(0, 0, 99) - 1.0).abs() < f64::EPSILON);
    }
}
