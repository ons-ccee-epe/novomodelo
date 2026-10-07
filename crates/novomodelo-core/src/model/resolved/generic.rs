//! Pre-resolved RHS bound table for user-defined generic linear constraints.
//!
//! Unlike the dense entity bound tables in `bounds`, generic-constraint bounds are
//! sparse, so absent `(constraint, stage)` pairs report inactive rather than
//! panicking. Populated by `cobre-io`; never modified after construction.

use std::collections::HashMap;
use std::ops::Range;

/// Pre-resolved RHS bound table for user-defined generic linear constraints.
///
/// Sparse `(constraint_index, stage_id)` index with O(1) lookup. Absent pairs
/// report [`is_active`] `false` and [`bounds_for_stage`] empty — no panic.
///
/// # Examples
///
/// ```
/// use cobre_core::ResolvedGenericConstraintBounds;
///
/// let empty = ResolvedGenericConstraintBounds::empty();
/// assert!(!empty.is_active(0, 0));
/// assert!(empty.bounds_for_stage(0, 0).is_empty());
/// ```
///
/// [`is_active`]: ResolvedGenericConstraintBounds::is_active
/// [`bounds_for_stage`]: ResolvedGenericConstraintBounds::bounds_for_stage
#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedGenericConstraintBounds {
    /// Maps `(constraint_idx, stage_id)` to a contiguous range in `entries`.
    /// `stage_id` is `i32` to match domain-level stage IDs, which may be negative.
    index: HashMap<(usize, i32), Range<usize>>,
    /// Flat entries; each key's entries occupy the range `index` records.
    entries: Vec<GenericConstraintBoundEntry>,
}

/// One resolved `(block_id, bound_lower, bound_upper)` entry for a `(constraint, stage)`
/// key. Shape is derived from which endpoints are present: lower-only, upper-only, or
/// both (a band, degenerate to an equality when the two are equal).
///
/// A named record rather than a tuple: `bound_lower` and `bound_upper` are both
/// `Option<f64>`-shaped, so a bare positional tuple invites a silent field swap at a
/// construction site.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GenericConstraintBoundEntry {
    /// Block index (`None` = all blocks).
    pub block_id: Option<i32>,
    /// Lower RHS bound; `None` when the row is upper-only.
    pub bound_lower: Option<f64>,
    /// Upper RHS bound; `None` when the row is lower-only.
    pub bound_upper: Option<f64>,
}

#[cfg(feature = "serde")]
mod serde_generic_bounds {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    use super::{GenericConstraintBoundEntry, ResolvedGenericConstraintBounds};

    /// Wire format for serde: a list of `(constraint_idx, stage_id, pairs)` groups,
    /// because `HashMap<(usize, i32), Range<usize>>` cannot serialize directly
    /// (composite tuple keys are not strings).
    #[derive(Serialize, Deserialize)]
    struct WireEntry {
        constraint_idx: usize,
        stage_id: i32,
        pairs: Vec<GenericConstraintBoundEntry>,
    }

    impl Serialize for ResolvedGenericConstraintBounds {
        fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            // Sort keys for deterministic output regardless of HashMap iteration order.
            let mut keys: Vec<(usize, i32)> = self.index.keys().copied().collect();
            keys.sort_unstable();

            let wire: Vec<WireEntry> = keys
                .into_iter()
                .map(|(constraint_idx, stage_id)| {
                    let range = self.index[&(constraint_idx, stage_id)].clone();
                    WireEntry {
                        constraint_idx,
                        stage_id,
                        pairs: self.entries[range].to_vec(),
                    }
                })
                .collect();

            wire.serialize(serializer)
        }
    }

    impl<'de> Deserialize<'de> for ResolvedGenericConstraintBounds {
        fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
            let wire = Vec::<WireEntry>::deserialize(deserializer)?;

            let mut index = std::collections::HashMap::new();
            let mut entries = Vec::new();

            for entry in wire {
                let start = entries.len();
                entries.extend_from_slice(&entry.pairs);
                let end = entries.len();
                // Drop empty-pairs groups (no key) — inserting an empty range would
                // make `is_active` report `true` for a stage with no bounds. Mirrors `new`.
                if end > start {
                    index.insert((entry.constraint_idx, entry.stage_id), start..end);
                }
            }

            Ok(ResolvedGenericConstraintBounds { index, entries })
        }
    }
}

impl ResolvedGenericConstraintBounds {
    /// Return an empty table with no constraints and no bounds.
    ///
    /// The default value in [`System`](crate::System) when no generic constraints
    /// are loaded; all queries return `false` / empty slices.
    ///
    /// # Examples
    ///
    /// ```
    /// use cobre_core::ResolvedGenericConstraintBounds;
    ///
    /// let t = ResolvedGenericConstraintBounds::empty();
    /// assert!(!t.is_active(0, 0));
    /// assert!(t.bounds_for_stage(99, 5).is_empty());
    /// ```
    #[must_use]
    pub fn empty() -> Self {
        Self {
            index: HashMap::new(),
            entries: Vec::new(),
        }
    }

    /// Build a resolved table from sorted bound rows.
    ///
    /// `constraint_id_to_idx` maps domain `constraint_id: i32` to positional index;
    /// rows with an absent `constraint_id` are silently skipped (caught upstream by
    /// referential validation). `raw_bounds` must be sorted by `(constraint_id,
    /// stage_id, block_id)` ascending — the grouping into contiguous ranges relies on it.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::HashMap;
    /// use cobre_core::ResolvedGenericConstraintBounds;
    ///
    /// // Two constraints with IDs 10 and 20, mapped to positions 0 and 1.
    /// let id_map: HashMap<i32, usize> = [(10, 0), (20, 1)].into_iter().collect();
    ///
    /// // One bound row: constraint 10 at stage 3, block_id = None, bound_lower = 500.0,
    /// // bound_upper = None.
    /// let rows = vec![(10i32, 3i32, None::<i32>, Some(500.0f64), None::<f64>)];
    ///
    /// let table = ResolvedGenericConstraintBounds::new(
    ///     &id_map,
    ///     rows.iter().map(|(cid, sid, bid, bl, bu)| (*cid, *sid, *bid, *bl, *bu)),
    /// );
    ///
    /// assert!(table.is_active(0, 3));
    /// assert!(!table.is_active(1, 3));
    ///
    /// let slice = table.bounds_for_stage(0, 3);
    /// assert_eq!(slice.len(), 1);
    /// assert_eq!(slice[0].block_id, None);
    /// assert_eq!(slice[0].bound_lower, Some(500.0));
    /// assert_eq!(slice[0].bound_upper, None);
    /// ```
    #[must_use]
    pub fn new<I>(constraint_id_to_idx: &HashMap<i32, usize>, raw_bounds: I) -> Self
    where
        I: Iterator<Item = (i32, i32, Option<i32>, Option<f64>, Option<f64>)>,
    {
        let mut index: HashMap<(usize, i32), Range<usize>> = HashMap::new();
        let mut entries: Vec<GenericConstraintBoundEntry> = Vec::new();

        let mut current_key: Option<(usize, i32)> = None;
        let mut range_start: usize = 0;

        for (constraint_id, stage_id, block_id, bound_lower, bound_upper) in raw_bounds {
            let Some(&constraint_idx) = constraint_id_to_idx.get(&constraint_id) else {
                continue;
            };

            let key = (constraint_idx, stage_id);

            if current_key != Some(key) {
                if let Some(prev_key) = current_key {
                    let range_end = entries.len();
                    if range_end > range_start {
                        index.insert(prev_key, range_start..range_end);
                    }
                }
                range_start = entries.len();
                current_key = Some(key);
            }

            entries.push(GenericConstraintBoundEntry {
                block_id,
                bound_lower,
                bound_upper,
            });
        }

        if let Some(last_key) = current_key {
            let range_end = entries.len();
            if range_end > range_start {
                index.insert(last_key, range_start..range_end);
            }
        }

        Self { index, entries }
    }

    /// Return `true` if at least one bound entry exists for this constraint at the given stage.
    ///
    /// Returns `false` for any unknown `(constraint_idx, stage_id)` pair.
    ///
    /// # Examples
    ///
    /// ```
    /// use cobre_core::ResolvedGenericConstraintBounds;
    ///
    /// let empty = ResolvedGenericConstraintBounds::empty();
    /// assert!(!empty.is_active(0, 0));
    /// ```
    #[inline]
    #[must_use]
    pub fn is_active(&self, constraint_idx: usize, stage_id: i32) -> bool {
        self.index.contains_key(&(constraint_idx, stage_id))
    }

    /// Return the bound entries for a constraint at the given stage.
    ///
    /// Returns an empty slice when no bounds exist for the `(constraint_idx, stage_id)` pair.
    ///
    /// # Examples
    ///
    /// ```
    /// use cobre_core::ResolvedGenericConstraintBounds;
    ///
    /// let empty = ResolvedGenericConstraintBounds::empty();
    /// assert!(empty.bounds_for_stage(0, 0).is_empty());
    /// ```
    #[inline]
    #[must_use]
    pub fn bounds_for_stage(
        &self,
        constraint_idx: usize,
        stage_id: i32,
    ) -> &[GenericConstraintBoundEntry] {
        match self.index.get(&(constraint_idx, stage_id)) {
            Some(range) => &self.entries[range.clone()],
            None => &[],
        }
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// `empty()` returns a table where all queries return false/empty.
    #[test]
    fn test_generic_bounds_empty() {
        let t = ResolvedGenericConstraintBounds::empty();
        assert!(!t.is_active(0, 0));
        assert!(!t.is_active(99, -1));
        assert!(t.bounds_for_stage(0, 0).is_empty());
        assert!(t.bounds_for_stage(99, 5).is_empty());
    }

    /// `new()` with 2 constraints, sparse bounds: constraint 0 at stage 0 is active;
    /// constraint 1 at stage 0 is not active.
    #[test]
    fn test_generic_bounds_sparse_active() {
        let id_map: HashMap<i32, usize> = [(0, 0), (1, 1)].into_iter().collect();

        let rows = vec![(0i32, 0i32, None::<i32>, Some(100.0f64), None::<f64>)];
        let t = ResolvedGenericConstraintBounds::new(&id_map, rows.into_iter());

        assert!(t.is_active(0, 0), "constraint 0 at stage 0 must be active");
        assert!(
            !t.is_active(1, 0),
            "constraint 1 at stage 0 must not be active"
        );
        assert!(
            !t.is_active(0, 1),
            "constraint 0 at stage 1 must not be active"
        );
    }

    /// `bounds_for_stage()` with `block_id=None` returns the correct single-entry slice.
    #[test]
    fn test_generic_bounds_single_block_none() {
        let id_map: HashMap<i32, usize> = [(0, 0)].into_iter().collect();
        let rows = vec![(0i32, 0i32, None::<i32>, Some(100.0f64), None::<f64>)];
        let t = ResolvedGenericConstraintBounds::new(&id_map, rows.into_iter());

        let slice = t.bounds_for_stage(0, 0);
        assert_eq!(slice.len(), 1);
        assert_eq!(
            slice[0],
            GenericConstraintBoundEntry {
                block_id: None,
                bound_lower: Some(100.0),
                bound_upper: None,
            }
        );
    }

    /// Multiple (`block_id`, `bound_lower`) pairs for the same (constraint, stage).
    #[test]
    fn test_generic_bounds_multiple_blocks() {
        let id_map: HashMap<i32, usize> = [(0, 0)].into_iter().collect();
        let rows = vec![
            (0i32, 2i32, None::<i32>, Some(50.0f64), None::<f64>),
            (0i32, 2i32, Some(0i32), Some(60.0f64), None::<f64>),
            (0i32, 2i32, Some(1i32), Some(70.0f64), None::<f64>),
        ];
        let t = ResolvedGenericConstraintBounds::new(&id_map, rows.into_iter());

        assert!(t.is_active(0, 2));
        let slice = t.bounds_for_stage(0, 2);
        assert_eq!(slice.len(), 3);
        assert_eq!(
            slice[0],
            GenericConstraintBoundEntry {
                block_id: None,
                bound_lower: Some(50.0),
                bound_upper: None,
            }
        );
        assert_eq!(
            slice[1],
            GenericConstraintBoundEntry {
                block_id: Some(0),
                bound_lower: Some(60.0),
                bound_upper: None,
            }
        );
        assert_eq!(
            slice[2],
            GenericConstraintBoundEntry {
                block_id: Some(1),
                bound_lower: Some(70.0),
                bound_upper: None,
            }
        );
    }

    /// `bound_upper`, when supplied, round-trips through `new()` into
    /// `bounds_for_stage()` alongside `bound_lower` — the actual widened-pipe contract.
    #[test]
    fn test_generic_bounds_bound_upper_round_trips() {
        let id_map: HashMap<i32, usize> = [(0, 0)].into_iter().collect();
        let rows = vec![
            (0i32, 0i32, None::<i32>, Some(50.0f64), Some(90.0f64)),
            (0i32, 0i32, Some(0i32), Some(60.0f64), None::<f64>),
        ];
        let t = ResolvedGenericConstraintBounds::new(&id_map, rows.into_iter());

        let slice = t.bounds_for_stage(0, 0);
        assert_eq!(slice.len(), 2);
        assert_eq!(
            slice[0],
            GenericConstraintBoundEntry {
                block_id: None,
                bound_lower: Some(50.0),
                bound_upper: Some(90.0),
            }
        );
        assert_eq!(
            slice[1],
            GenericConstraintBoundEntry {
                block_id: Some(0),
                bound_lower: Some(60.0),
                bound_upper: None,
            }
        );
    }

    /// A lower-`None`, upper-only row round-trips through `new()`.
    #[test]
    fn test_generic_bounds_upper_only_round_trips() {
        let id_map: HashMap<i32, usize> = [(0, 0)].into_iter().collect();
        let rows = vec![(0i32, 0i32, None::<i32>, None::<f64>, Some(10.0f64))];
        let t = ResolvedGenericConstraintBounds::new(&id_map, rows.into_iter());

        let slice = t.bounds_for_stage(0, 0);
        assert_eq!(slice.len(), 1);
        assert_eq!(
            slice[0],
            GenericConstraintBoundEntry {
                block_id: None,
                bound_lower: None,
                bound_upper: Some(10.0),
            }
        );
    }

    /// Rows with unknown `constraint_id` are silently skipped.
    #[test]
    fn test_generic_bounds_unknown_constraint_id_skipped() {
        let id_map: HashMap<i32, usize> = [(0, 0)].into_iter().collect();
        let rows = vec![(99i32, 0i32, None::<i32>, Some(1000.0f64), None::<f64>)];
        let t = ResolvedGenericConstraintBounds::new(&id_map, rows.into_iter());

        assert!(!t.is_active(0, 0), "unknown constraint_id must be skipped");
        assert!(t.bounds_for_stage(0, 0).is_empty());
    }

    /// Empty `raw_bounds` produces a table identical to `empty()`.
    #[test]
    fn test_generic_bounds_no_rows() {
        let id_map: HashMap<i32, usize> = [(0, 0), (1, 1)].into_iter().collect();
        let t = ResolvedGenericConstraintBounds::new(&id_map, std::iter::empty());

        assert!(!t.is_active(0, 0));
        assert!(!t.is_active(1, 0));
        assert!(t.bounds_for_stage(0, 0).is_empty());
    }

    /// Bounds for constraint 0 at stages 0 and 1; constraint 1 has no bounds.
    #[test]
    fn test_generic_bounds_two_stages_one_constraint() {
        let id_map: HashMap<i32, usize> = [(0, 0), (1, 1)].into_iter().collect();
        let rows = vec![
            (0i32, 0i32, None::<i32>, Some(100.0f64), None::<f64>),
            (0i32, 1i32, None::<i32>, Some(200.0f64), None::<f64>),
        ];
        let t = ResolvedGenericConstraintBounds::new(&id_map, rows.into_iter());

        assert!(t.is_active(0, 0));
        assert!(t.is_active(0, 1));
        assert!(!t.is_active(1, 0));
        assert!(!t.is_active(1, 1));

        let s0 = t.bounds_for_stage(0, 0);
        assert_eq!(s0.len(), 1);
        assert!((s0[0].bound_lower.expect("lower present") - 100.0).abs() < f64::EPSILON);

        let s1 = t.bounds_for_stage(0, 1);
        assert_eq!(s1.len(), 1);
        assert!((s1[0].bound_lower.expect("lower present") - 200.0).abs() < f64::EPSILON);
    }

    #[test]
    #[cfg(feature = "serde")]
    fn test_generic_bounds_serde_roundtrip() {
        let id_map: HashMap<i32, usize> = [(0, 0), (1, 1)].into_iter().collect();
        let rows = vec![
            (0i32, 0i32, None::<i32>, Some(100.0f64), None::<f64>),
            (0i32, 0i32, Some(1i32), Some(150.0f64), Some(175.0f64)),
            (1i32, 2i32, None::<i32>, Some(300.0f64), None::<f64>),
        ];
        let original = ResolvedGenericConstraintBounds::new(&id_map, rows.into_iter());
        let json = serde_json::to_string(&original).expect("serialize");
        let restored: ResolvedGenericConstraintBounds =
            serde_json::from_str(&json).expect("deserialize");
        assert_eq!(original, restored);
    }
}
