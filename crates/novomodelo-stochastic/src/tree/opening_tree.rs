//! Opening scenario tree data structure.
//!
//! In-memory representation of the pre-generated noise realisations consumed by
//! iterative optimisation algorithms.

use crate::error::StochasticError;

/// Fixed opening tree of pre-generated noise realisations, read-only after
/// construction.
///
/// Noise is stored stage-major in a flat array: all of stage 0's openings, then
/// stage 1's, etc.; each opening is a contiguous block of `dim` f64 values, at
/// `data[stage_offsets[stage] + opening_idx * dim ..][..dim]`. The sentinel
/// `stage_offsets[n_stages] == data.len()` keeps bounds checks exact without
/// special-casing the last stage.
#[derive(Debug)]
pub struct OpeningTree {
    data: Box<[f64]>,
    stage_offsets: Box<[usize]>,
    openings_per_stage: Box<[usize]>,
    n_stages: usize,
    dim: usize,
    /// Per-stage solve-order permutation, stage-major in a flat buffer of
    /// `Σ_s n_openings(s)` `u32` indices. `solve_order[order_offsets[s]..order_offsets[s+1]]`
    /// permutes `0..n_openings(s)`. Initialised to the **identity** so the default
    /// solve order is byte-identical to canonical `0..n_openings(s)` iteration;
    /// [`Self::set_solve_order`] may replace it.
    solve_order: Box<[u32]>,
    /// Stage-start offsets into [`Self::solve_order`], length `n_stages + 1`; the
    /// sentinel `order_offsets[n_stages]` equals `solve_order.len()`.
    order_offsets: Box<[usize]>,
}

impl OpeningTree {
    /// Construct an `OpeningTree` from its raw parts.
    ///
    /// # Panics
    ///
    /// Panics if `data.len()` does not equal
    /// `openings_per_stage.iter().sum::<usize>() * dim`.
    #[must_use]
    pub fn from_parts(data: Vec<f64>, openings_per_stage: Vec<usize>, dim: usize) -> Self {
        let n_stages = openings_per_stage.len();

        let mut stage_offsets = Vec::with_capacity(n_stages + 1);
        let mut offset = 0usize;
        stage_offsets.push(offset);
        for &n_openings in &openings_per_stage {
            offset += n_openings * dim;
            stage_offsets.push(offset);
        }

        assert!(
            data.len() == stage_offsets[n_stages],
            "OpeningTree::from_parts: data.len() ({}) does not match expected total ({} = sum(openings_per_stage) * dim)",
            data.len(),
            stage_offsets[n_stages],
        );

        // `order_offsets` accumulates raw opening counts (not `dim`-scaled).
        let mut order_offsets = Vec::with_capacity(n_stages + 1);
        let mut order_offset = 0usize;
        order_offsets.push(order_offset);
        for &n_openings in &openings_per_stage {
            order_offset += n_openings;
            order_offsets.push(order_offset);
        }
        let mut solve_order = Vec::with_capacity(order_offset);
        for &n_openings in &openings_per_stage {
            for idx in 0..n_openings {
                debug_assert!(
                    u32::try_from(idx).is_ok(),
                    "opening index {idx} overflows u32"
                );
                #[allow(clippy::cast_possible_truncation)]
                solve_order.push(idx as u32);
            }
        }

        Self {
            data: data.into_boxed_slice(),
            stage_offsets: stage_offsets.into_boxed_slice(),
            openings_per_stage: openings_per_stage.into_boxed_slice(),
            n_stages,
            dim,
            solve_order: solve_order.into_boxed_slice(),
            order_offsets: order_offsets.into_boxed_slice(),
        }
    }

    /// Replace the per-stage solve order by sorting each stage's openings by a
    /// caller-supplied scalar key, **descending**.
    ///
    /// `keys[s]` holds one key per canonical opening ω (`keys.len() == n_stages()`,
    /// `keys[s].len() == n_openings(s)`). Ties are broken by **ascending canonical
    /// ω**, so the permutation is a deterministic function of `keys`
    /// alone — independent of input ordering and identical on every process given
    /// the same keys. Only the permutation is stored, not the keys.
    ///
    /// # Errors
    ///
    /// Returns [`StochasticError::InsufficientData`] if `keys.len()` does not equal
    /// [`Self::n_stages`], or if any `keys[s].len()` does not equal
    /// `n_openings(s)` — naming the offending stage and both lengths. The solve
    /// order is left unchanged on error.
    pub fn set_solve_order(&mut self, keys: &[Vec<f64>]) -> Result<(), StochasticError> {
        if keys.len() != self.n_stages {
            return Err(StochasticError::InsufficientData {
                context: format!(
                    "solve-order key table has {} stages but the opening tree has {} stages",
                    keys.len(),
                    self.n_stages,
                ),
            });
        }
        for (s, stage_keys) in keys.iter().enumerate() {
            let n_openings = self.openings_per_stage[s];
            if stage_keys.len() != n_openings {
                return Err(StochasticError::InsufficientData {
                    context: format!(
                        "solve-order key table for stage {s} has {} keys but the stage has \
                         {n_openings} openings",
                        stage_keys.len(),
                    ),
                });
            }
        }

        for (s, stage_keys) in keys.iter().enumerate() {
            let start = self.order_offsets[s];
            let end = self.order_offsets[s + 1];
            let n_openings = self.openings_per_stage[s];
            let order = &mut self.solve_order[start..end];
            // Reset to canonical 0..n before sorting so the result is a pure
            // function of `keys`, independent of any prior installed order.
            for (idx, slot) in order.iter_mut().enumerate() {
                #[allow(clippy::cast_possible_truncation)]
                {
                    *slot = idx as u32;
                }
            }
            // Stable sort + the pre-set identity breaks equal-key ties by
            // ascending canonical ω.
            order.sort_by(|&a, &b| {
                let ka = stage_keys[a as usize];
                let kb = stage_keys[b as usize];
                // total_cmp totally orders all f64 bit patterns (NaN/±0), so the
                // permutation is rank-invariant.
                kb.total_cmp(&ka)
            });
            debug_assert!(
                is_permutation(order, n_openings),
                "solve_order for stage {s} is not a permutation of 0..{n_openings}"
            );
        }
        Ok(())
    }

    /// Return a read-only borrowed view.
    #[must_use]
    pub fn view(&self) -> OpeningTreeView<'_> {
        OpeningTreeView {
            data: &self.data,
            stage_offsets: &self.stage_offsets,
            openings_per_stage: &self.openings_per_stage,
            n_stages: self.n_stages,
            dim: self.dim,
            solve_order: &self.solve_order,
            order_offsets: &self.order_offsets,
        }
    }

    /// Return the noise slice for a specific `(stage, opening_idx)` pair.
    ///
    /// # Panics
    ///
    /// Panics if `stage >= n_stages` or `opening_idx >= n_openings(stage)`.
    #[must_use]
    pub fn opening(&self, stage: usize, opening_idx: usize) -> &[f64] {
        assert!(stage < self.n_stages, "stage {stage} out of bounds");
        assert!(
            opening_idx < self.openings_per_stage[stage],
            "opening_idx {opening_idx} out of bounds for stage {stage}"
        );
        let start = self.stage_offsets[stage] + opening_idx * self.dim;
        &self.data[start..start + self.dim]
    }

    /// Return the number of openings (branching factor) at the given stage.
    ///
    /// # Panics
    ///
    /// Panics if `stage >= n_stages`.
    #[must_use]
    pub fn n_openings(&self, stage: usize) -> usize {
        assert!(stage < self.n_stages, "stage {stage} out of bounds");
        self.openings_per_stage[stage]
    }

    /// Return the per-stage solve-order permutation for `stage`.
    ///
    /// A consumer may iterate openings in this order only while keeping any
    /// per-opening aggregation keyed by canonical ω.
    ///
    /// # Panics
    ///
    /// Panics if `stage >= n_stages`.
    #[must_use]
    pub fn solve_order(&self, stage: usize) -> &[u32] {
        assert!(stage < self.n_stages, "stage {stage} out of bounds");
        &self.solve_order[self.order_offsets[stage]..self.order_offsets[stage + 1]]
    }

    /// Return the number of stages in the tree.
    #[must_use]
    pub fn n_stages(&self) -> usize {
        self.n_stages
    }

    /// Return the number of entities (noise dimension) per opening vector.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Return the total number of f64 elements in the backing array.
    #[must_use]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Return `true` if the backing array is empty.
    ///
    /// Required by Clippy when [`len`](Self::len) is defined.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Return the total size in bytes of the backing array.
    #[must_use]
    pub fn size_bytes(&self) -> usize {
        std::mem::size_of_val(&*self.data)
    }

    /// Return a reference to the flat contiguous backing array.
    #[must_use]
    pub fn data(&self) -> &[f64] {
        &self.data
    }

    /// Return a reference to the per-stage branching factor array.
    #[must_use]
    pub fn openings_per_stage_slice(&self) -> &[usize] {
        &self.openings_per_stage
    }
}

/// Borrowed read-only view over opening tree data.
pub struct OpeningTreeView<'a> {
    data: &'a [f64],
    stage_offsets: &'a [usize],
    openings_per_stage: &'a [usize],
    n_stages: usize,
    dim: usize,
    solve_order: &'a [u32],
    order_offsets: &'a [usize],
}

impl<'a> OpeningTreeView<'a> {
    /// Return the noise slice for a specific `(stage, opening_idx)` pair.
    ///
    /// # Panics
    ///
    /// Panics if `stage >= n_stages` or `opening_idx >= n_openings(stage)`.
    #[must_use]
    pub fn opening(&self, stage: usize, opening_idx: usize) -> &[f64] {
        assert!(stage < self.n_stages, "stage {stage} out of bounds");
        assert!(
            opening_idx < self.openings_per_stage[stage],
            "opening_idx {opening_idx} out of bounds for stage {stage}"
        );
        let start = self.stage_offsets[stage] + opening_idx * self.dim;
        &self.data[start..start + self.dim]
    }

    /// Return the noise slice for a specific `(stage, opening_idx)` pair,
    /// with lifetime tied to the backing data (`'a`).
    ///
    /// Use this when the returned slice must outlive the view reference.
    ///
    /// # Panics
    ///
    /// Panics if `stage >= n_stages` or `opening_idx >= n_openings(stage)`.
    #[must_use]
    pub fn opening_data(&self, stage: usize, opening_idx: usize) -> &'a [f64] {
        assert!(stage < self.n_stages, "stage {stage} out of bounds");
        assert!(
            opening_idx < self.openings_per_stage[stage],
            "opening_idx {opening_idx} out of bounds for stage {stage}"
        );
        let start = self.stage_offsets[stage] + opening_idx * self.dim;
        &self.data[start..start + self.dim]
    }

    /// Return the number of openings (branching factor) at the given stage.
    ///
    /// # Panics
    ///
    /// Panics if `stage >= n_stages`.
    #[must_use]
    pub fn n_openings(&self, stage: usize) -> usize {
        assert!(stage < self.n_stages, "stage {stage} out of bounds");
        self.openings_per_stage[stage]
    }

    /// Return the per-stage solve-order permutation for `stage`.
    ///
    /// Mirrors [`OpeningTree::solve_order`]: a permutation of
    /// `0..n_openings(stage)`, the identity by default. The returned slice borrows
    /// the view (`&self`); see [`Self::solve_order_data`] for a `'a`-bound slice.
    ///
    /// # Panics
    ///
    /// Panics if `stage >= n_stages`.
    #[must_use]
    pub fn solve_order(&self, stage: usize) -> &[u32] {
        assert!(stage < self.n_stages, "stage {stage} out of bounds");
        &self.solve_order[self.order_offsets[stage]..self.order_offsets[stage + 1]]
    }

    /// Return the per-stage solve-order permutation for `stage`, with lifetime
    /// tied to the backing data (`'a`).
    ///
    /// Use this when the returned slice must outlive the view reference (the
    /// solve-order analogue of [`Self::opening_data`]).
    ///
    /// # Panics
    ///
    /// Panics if `stage >= n_stages`.
    #[must_use]
    pub fn solve_order_data(&self, stage: usize) -> &'a [u32] {
        assert!(stage < self.n_stages, "stage {stage} out of bounds");
        &self.solve_order[self.order_offsets[stage]..self.order_offsets[stage + 1]]
    }

    /// Return the number of stages in the tree.
    #[must_use]
    pub fn n_stages(&self) -> usize {
        self.n_stages
    }

    /// Return the number of entities (noise dimension) per opening vector.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Return the total number of f64 elements in the backing slice.
    #[must_use]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Return `true` if the backing slice is empty.
    ///
    /// Required by Clippy when [`len`](Self::len) is defined.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Return the total size in bytes of the backing slice.
    #[must_use]
    pub fn size_bytes(&self) -> usize {
        std::mem::size_of_val(self.data)
    }

    /// Return a reference to the flat contiguous backing slice.
    #[must_use]
    pub fn data(&self) -> &[f64] {
        self.data
    }

    /// Return a reference to the per-stage branching factor slice.
    #[must_use]
    pub fn openings_per_stage_slice(&self) -> &[usize] {
        self.openings_per_stage
    }
}

/// Defined unconditionally, not behind `debug_assertions`: the `debug_assert!` in
/// [`OpeningTree::set_solve_order`] still type-checks its argument in release, so
/// gating this would break the release build.
fn is_permutation(order: &[u32], n: usize) -> bool {
    if order.len() != n {
        return false;
    }
    let mut seen = vec![false; n];
    for &idx in order {
        let idx = idx as usize;
        if idx >= n || seen[idx] {
            return false;
        }
        seen[idx] = true;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::uniform_tree;

    #[test]
    fn opening_stage0_opening0_returns_first_dim_elements() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let tree = OpeningTree::from_parts(data, vec![1, 2], 3);
        assert_eq!(tree.opening(0, 0), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn opening_stage1_returns_correct_slices() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let tree = OpeningTree::from_parts(data, vec![1, 2], 3);
        assert_eq!(tree.opening(1, 0), &[4.0, 5.0, 6.0]);
        assert_eq!(tree.opening(1, 1), &[7.0, 8.0, 9.0]);
    }

    #[test]
    fn n_openings_matches_branching_factors() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let tree = OpeningTree::from_parts(data, vec![1, 2], 3);
        assert_eq!(tree.n_openings(0), 1);
        assert_eq!(tree.n_openings(1), 2);
    }

    #[test]
    fn view_returns_identical_data_to_owned() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let tree = OpeningTree::from_parts(data, vec![1, 2], 3);
        let view = tree.view();
        assert_eq!(view.opening(0, 0), tree.opening(0, 0));
        assert_eq!(view.opening(1, 0), tree.opening(1, 0));
        assert_eq!(view.opening(1, 1), tree.opening(1, 1));
    }

    #[test]
    fn len_and_size_bytes_uniform_branching() {
        let tree = OpeningTree::from_parts(vec![1.0; 30], vec![5, 5, 5], 2);
        assert_eq!(tree.len(), 30);
        assert_eq!(tree.size_bytes(), 240);
    }

    #[test]
    fn uniform_branching_3_stages_5_openings_2_entities() {
        let tree = uniform_tree(3, 5, 2);
        assert_eq!(tree.n_stages(), 3);
        assert_eq!(tree.dim(), 2);
        assert_eq!(tree.len(), 30);

        assert_eq!(tree.opening(0, 0), &[0.0_f64, 1.0]);
        assert_eq!(tree.opening(0, 4), &[8.0_f64, 9.0]);
        assert_eq!(tree.opening(1, 0), &[10.0_f64, 11.0]);
        assert_eq!(tree.opening(2, 4), &[28.0_f64, 29.0]);
    }

    #[test]
    fn variable_branching_access() {
        // total = (3+1+4)*2 = 16 elements
        let data: Vec<f64> = (0_i32..16).map(f64::from).collect();
        let tree = OpeningTree::from_parts(data, vec![3, 1, 4], 2);

        assert_eq!(tree.n_stages(), 3);
        assert_eq!(tree.n_openings(0), 3);
        assert_eq!(tree.n_openings(1), 1);
        assert_eq!(tree.n_openings(2), 4);

        assert_eq!(tree.opening(0, 0), &[0.0_f64, 1.0]);
        assert_eq!(tree.opening(0, 2), &[4.0_f64, 5.0]);
        assert_eq!(tree.opening(1, 0), &[6.0_f64, 7.0]);
        assert_eq!(tree.opening(2, 0), &[8.0_f64, 9.0]);
        assert_eq!(tree.opening(2, 3), &[14.0_f64, 15.0]);
    }

    #[test]
    fn single_stage_single_opening() {
        let tree = OpeningTree::from_parts(vec![42.0, 43.0], vec![1], 2);
        assert_eq!(tree.n_stages(), 1);
        assert_eq!(tree.n_openings(0), 1);
        assert_eq!(tree.opening(0, 0), &[42.0, 43.0]);
        assert!(!tree.is_empty());
    }

    #[test]
    #[should_panic(expected = "data.len()")]
    fn from_parts_panics_on_wrong_data_length() {
        // Expected 6 elements (2 openings * 3 dim), provide 5
        let _tree = OpeningTree::from_parts(vec![1.0; 5], vec![2], 3);
    }

    #[test]
    #[should_panic(expected = "stage 3 out of bounds")]
    fn opening_panics_on_out_of_bounds_stage() {
        let tree = uniform_tree(3, 2, 2);
        let _ = tree.opening(3, 0);
    }

    #[test]
    #[should_panic(expected = "opening_idx 5 out of bounds")]
    fn opening_panics_on_out_of_bounds_opening_idx() {
        let tree = uniform_tree(3, 5, 2);
        let _ = tree.opening(0, 5);
    }

    #[test]
    fn view_accessors_match_owned_for_variable_branching() {
        let data: Vec<f64> = (0_i32..16).map(f64::from).collect();
        let tree = OpeningTree::from_parts(data, vec![3, 1, 4], 2);
        let view = tree.view();

        assert_eq!(view.n_stages(), tree.n_stages());
        assert_eq!(view.dim(), tree.dim());
        assert_eq!(view.len(), tree.len());
        assert_eq!(view.size_bytes(), tree.size_bytes());

        for stage in 0..tree.n_stages() {
            assert_eq!(view.n_openings(stage), tree.n_openings(stage));
            for idx in 0..tree.n_openings(stage) {
                assert_eq!(view.opening(stage, idx), tree.opening(stage, idx));
            }
        }
    }

    #[test]
    fn is_empty_false_for_non_empty_tree() {
        let tree = uniform_tree(2, 3, 4);
        assert!(!tree.is_empty());
    }

    #[test]
    fn size_bytes_is_8_times_len() {
        let tree = uniform_tree(4, 3, 5);
        assert_eq!(tree.size_bytes(), tree.len() * 8);
    }

    #[test]
    fn view_is_empty_matches_owned() {
        let tree = uniform_tree(2, 3, 4);
        let view = tree.view();
        assert_eq!(view.is_empty(), tree.is_empty());
    }

    #[test]
    fn data_returns_full_backing_array() {
        let raw = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tree = OpeningTree::from_parts(raw.clone(), vec![1, 2], 2);
        assert_eq!(tree.data(), raw.as_slice());
    }

    #[test]
    fn openings_per_stage_slice_matches_input() {
        let tree = OpeningTree::from_parts(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 2], 2);
        assert_eq!(tree.openings_per_stage_slice(), &[1_usize, 2]);
    }

    #[test]
    fn view_data_matches_owned_data() {
        let raw = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tree = OpeningTree::from_parts(raw.clone(), vec![1, 2], 2);
        let view = tree.view();
        assert_eq!(view.data(), raw.as_slice());
    }

    // ── solve_order ──────────────────────────────────────────────────────────

    #[test]
    fn solve_order_defaults_to_identity_per_stage() {
        let tree = OpeningTree::from_parts((0_i32..16).map(f64::from).collect(), vec![3, 1, 4], 2);
        assert_eq!(tree.solve_order(0), &[0, 1, 2]);
        assert_eq!(tree.solve_order(1), &[0]);
        assert_eq!(tree.solve_order(2), &[0, 1, 2, 3]);
        let view = tree.view();
        for s in 0..tree.n_stages() {
            assert_eq!(view.solve_order(s), tree.solve_order(s));
            assert_eq!(view.solve_order_data(s), tree.solve_order(s));
        }
    }

    #[test]
    fn solve_order_is_permutation_with_extreme_first_descending() {
        let mut tree = OpeningTree::from_parts((0_i32..14).map(f64::from).collect(), vec![3, 4], 2);
        // Stage 0 keys: ω0=1.0, ω1=3.0, ω2=2.0 → descending order [1, 2, 0].
        // Stage 1 keys: ω0=-1.0, ω1=0.5, ω2=10.0, ω3=4.0 → descending [2, 3, 1, 0].
        let keys = vec![vec![1.0, 3.0, 2.0], vec![-1.0, 0.5, 10.0, 4.0]];
        tree.set_solve_order(&keys).expect("dims aligned");

        assert!(is_permutation(tree.solve_order(0), 3));
        assert!(is_permutation(tree.solve_order(1), 4));
        assert_eq!(tree.solve_order(0), &[1, 2, 0]);
        assert_eq!(tree.solve_order(1), &[2, 3, 1, 0]);
    }

    #[test]
    fn solve_order_ties_broken_by_canonical_omega() {
        let mut tree = OpeningTree::from_parts((0_i32..8).map(f64::from).collect(), vec![4], 2);
        let keys = vec![vec![7.0, 7.0, 7.0, 7.0]];
        tree.set_solve_order(&keys).expect("dims aligned");
        assert_eq!(tree.solve_order(0), &[0, 1, 2, 3]);
    }

    #[test]
    fn solve_order_is_idempotent_function_of_keys() {
        let mut tree = OpeningTree::from_parts((0_i32..6).map(f64::from).collect(), vec![3], 2);
        let keys_a = vec![vec![5.0, 1.0, 3.0]];
        let keys_b = vec![vec![1.0, 5.0, 3.0]];
        tree.set_solve_order(&keys_a).expect("dims aligned");
        assert_eq!(tree.solve_order(0), &[0, 2, 1]);
        tree.set_solve_order(&keys_b).expect("dims aligned");
        assert_eq!(tree.solve_order(0), &[1, 2, 0]);
        tree.set_solve_order(&keys_a).expect("dims aligned");
        assert_eq!(tree.solve_order(0), &[0, 2, 1]);
    }

    #[test]
    fn set_solve_order_hard_errors_on_stage_count_mismatch() {
        let mut tree = OpeningTree::from_parts((0_i32..6).map(f64::from).collect(), vec![1, 2], 2);
        let keys = vec![vec![1.0]];
        let err = tree
            .set_solve_order(&keys)
            .expect_err("stage count mismatch must error");
        let msg = err.to_string();
        assert!(msg.contains('1'), "must name keys stage count 1: {msg}");
        assert!(msg.contains('2'), "must name tree stage count 2: {msg}");
    }

    #[test]
    fn set_solve_order_hard_errors_on_opening_count_mismatch() {
        let mut tree = OpeningTree::from_parts((0_i32..6).map(f64::from).collect(), vec![1, 2], 2);
        // Stage 1 has 2 openings but only 1 key is supplied.
        let keys = vec![vec![1.0], vec![3.0]];
        let err = tree
            .set_solve_order(&keys)
            .expect_err("opening count mismatch must error");
        let msg = err.to_string();
        assert!(
            msg.contains("stage 1"),
            "must name the offending stage 1: {msg}"
        );
        assert!(
            msg.contains('2'),
            "must name the stage's opening count 2: {msg}"
        );
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn solve_order_panics_on_out_of_bounds_stage() {
        let tree = uniform_tree(2, 3, 2);
        let _ = tree.solve_order(2);
    }
}
