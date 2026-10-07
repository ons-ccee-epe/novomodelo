//! `InSample` scenario sampling scheme.
//!
//! Draws a fixed set of scenarios from the opening tree per iteration and reuses
//! them across all stages.

use rand::RngExt;

use crate::noise::{rng::rng_from_seed, seed::derive_forward_seed};
use crate::tree::opening_tree::OpeningTreeView;

/// Deterministically select an opening (index + noise slice) for a given
/// `(stage, iteration, scenario)` context, drawn from the node-Ω sub-range
/// `node_opening_offset..node_opening_offset + node_opening_len` rather than the
/// full stage opening set. A chain-degenerate caller passes
/// `(0, tree.n_openings(stage_idx))`, reproducing today's full-range draw
/// bit-for-bit — never retrofit an inverse-CDF here in its place; the two
/// distributions differ and would break chain byte-parity.
///
/// # Panics
///
/// Panics if `stage_idx >= tree.n_stages()`, if `node_opening_len` is zero, or
/// if `node_opening_offset + node_opening_len > tree.n_openings(stage_idx)`.
#[must_use]
pub fn sample_forward<'tree, 'data>(
    tree: &'tree OpeningTreeView<'data>,
    base_seed: u64,
    iteration: u32,
    scenario: u32,
    stage: u32,
    stage_idx: usize,
    node_opening_offset: usize,
    node_opening_len: usize,
) -> (usize, &'data [f64])
where
    'data: 'tree,
{
    assert!(
        node_opening_offset + node_opening_len <= tree.n_openings(stage_idx),
        "sample_forward: node Ω range {node_opening_offset}..{} exceeds stage {stage_idx}'s \
         {} openings",
        node_opening_offset + node_opening_len,
        tree.n_openings(stage_idx),
    );
    let seed = derive_forward_seed(base_seed, iteration, scenario, stage);
    let mut rng = rng_from_seed(seed);
    let j = node_opening_offset + rng.random_range(0..node_opening_len);
    (j, tree.opening_data(stage_idx, j))
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::sample_forward;
    use crate::test_support::uniform_tree;
    use crate::tree::opening_tree::OpeningTreeView;

    /// `sample_forward` over the FULL stage opening set (node Ω == the whole
    /// stage) — the chain-degenerate range.
    fn full<'tree, 'data>(
        view: &'tree OpeningTreeView<'data>,
        base_seed: u64,
        iteration: u32,
        scenario: u32,
        stage: u32,
        stage_idx: usize,
    ) -> (usize, &'data [f64])
    where
        'data: 'tree,
    {
        sample_forward(
            view,
            base_seed,
            iteration,
            scenario,
            stage,
            stage_idx,
            0,
            view.n_openings(stage_idx),
        )
    }

    #[test]
    fn determinism_same_inputs_same_output() {
        let tree = uniform_tree(3, 5, 2);
        let view = tree.view();

        let (idx_a, slice_a) = full(&view, 42, 0, 0, 0, 0);
        let (idx_b, slice_b) = full(&view, 42, 0, 0, 0, 0);

        assert_eq!(idx_a, idx_b);
        assert_eq!(slice_a, slice_b);
    }

    #[test]
    fn different_scenarios_different_indices() {
        let tree = uniform_tree(1, 5, 2);
        let view = tree.view();

        let differing = (0_u32..20)
            .filter(|&s| {
                let (a, _) = full(&view, 42, 0, s, 0, 0);
                let (b, _) = full(&view, 42, 0, s + 1, 0, 0);
                a != b
            })
            .count();

        assert!(
            differing >= 10,
            "expected at least 10 of 20 consecutive scenario pairs to differ, got {differing}"
        );
    }

    #[test]
    fn all_indices_in_bounds() {
        let tree = uniform_tree(1, 10, 2);
        let view = tree.view();

        for scenario in 0_u32..1000 {
            let (idx, _) = full(&view, 42, 0, scenario, 0, 0);
            assert!(
                idx < 10,
                "index {idx} out of bounds for scenario {scenario}"
            );
        }
    }

    #[test]
    fn returned_slice_matches_tree_opening() {
        let tree = uniform_tree(3, 5, 2);
        let view = tree.view();

        let (idx, slice) = full(&view, 42, 0, 0, 0, 0);
        assert_eq!(slice, tree.opening(0, idx));
    }

    #[test]
    fn different_iterations_different_indices() {
        let tree = uniform_tree(1, 5, 2);
        let view = tree.view();

        let (idx_0, _) = full(&view, 42, 0, 0, 0, 0);
        let (idx_1, _) = full(&view, 42, 1, 0, 0, 0);

        assert_ne!(
            idx_0, idx_1,
            "expected different indices for iteration=0 and iteration=1"
        );
    }

    #[test]
    fn stage_domain_id_affects_seed() {
        // Two stages with the same number of openings so both are valid.
        let tree = uniform_tree(2, 10, 2);
        let view = tree.view();

        let (idx_stage0, _) = full(&view, 42, 0, 0, 0, 0);
        let (idx_stage1, _) = full(&view, 42, 0, 0, 1, 0);

        // Overwhelmingly likely to differ for n=10 openings.
        assert_ne!(
            idx_stage0, idx_stage1,
            "expected different indices for stage=0 and stage=1 domain IDs"
        );
    }

    #[test]
    fn single_opening_always_index_zero() {
        let tree = uniform_tree(1, 1, 3);
        let view = tree.view();

        for scenario in 0_u32..50 {
            let (idx, _) = full(&view, 99, 0, scenario, 0, 0);
            assert_eq!(idx, 0, "single-opening tree must always return index 0");
        }
    }

    #[test]
    fn node_opening_range_confines_the_drawn_index() {
        let tree = uniform_tree(1, 20, 2);
        let view = tree.view();
        let (offset, len) = (12, 5);

        for scenario in 0_u32..200 {
            let (idx, _) = sample_forward(&view, 42, 0, scenario, 0, 0, offset, len);
            assert!(
                (offset..offset + len).contains(&idx),
                "index {idx} escaped the node Ω range {offset}..{}",
                offset + len
            );
        }
    }

    #[test]
    fn node_opening_range_matches_offset_plus_full_range_draw() {
        let tree = uniform_tree(1, 20, 2);
        let view = tree.view();
        let (offset, len) = (8, 6);

        for scenario in 0_u32..50 {
            let (idx_scoped, _) = sample_forward(&view, 7, 3, scenario, 0, 0, offset, len);
            let (idx_zero_based, _) = sample_forward(&view, 7, 3, scenario, 0, 0, 0, len);
            assert_eq!(
                idx_scoped,
                offset + idx_zero_based,
                "scenario {scenario}: node-Ω draw must equal offset + the 0-based draw"
            );
        }
    }

    #[test]
    #[should_panic(expected = "exceeds stage 0's 20 openings")]
    fn node_opening_range_overrun_panics() {
        let tree = uniform_tree(1, 20, 2);
        let view = tree.view();

        // offset + len (15 + 10) exceeds the stage's 20 openings; the range
        // check is a hard `assert!`, so this panics in release builds too.
        let _ = sample_forward(&view, 42, 0, 0, 0, 0, 15, 10);
    }

    #[test]
    fn slice_length_equals_dim() {
        let dim = 7;
        let tree = uniform_tree(2, 5, dim);
        let view = tree.view();

        let (_, slice) = full(&view, 1, 2, 3, 0, 0);
        assert_eq!(slice.len(), dim);
    }

    #[test]
    fn resume_invariant_noise_depends_only_on_iteration_seed() {
        let tree = uniform_tree(2, 10, 3);
        let view = tree.view();
        let base_seed = 42;

        // Continuous run: sample iterations 1..=5, record iteration 5.
        let mut continuous_results = Vec::new();
        for scenario in 0_u32..5 {
            let (idx, slice) = full(&view, base_seed, 5, scenario, 0, 0);
            continuous_results.push((idx, slice.to_vec()));
        }

        // Resumed run: iteration-5 draws depend only on the iteration seed, so a
        // run that skipped 1..=3 must reproduce the continuous iteration 5.
        let mut resumed_results = Vec::new();
        for scenario in 0_u32..5 {
            let (idx, slice) = full(&view, base_seed, 5, scenario, 0, 0);
            resumed_results.push((idx, slice.to_vec()));
        }

        for (i, (cont, resu)) in continuous_results.iter().zip(&resumed_results).enumerate() {
            assert_eq!(
                cont.0, resu.0,
                "scenario {i}: opening index mismatch between continuous and resumed run"
            );
            assert_eq!(
                cont.1, resu.1,
                "scenario {i}: noise slice mismatch between continuous and resumed run"
            );
        }
    }
}
