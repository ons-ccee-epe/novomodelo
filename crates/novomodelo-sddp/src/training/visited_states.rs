//! Visited forward-pass states for dominated cut selection.
//!
//! When [`CutSelectionStrategy::Dominated`](crate::CutSelectionStrategy::Dominated)
//! is active, the training loop archives the trial-point state vectors produced
//! by each forward pass so that the domination test can evaluate every cut at
//! every visited point.
//!
//! The archive holds one [`NodeStates`] per **node**, never per stage: sibling
//! nodes share a stage but not a future, so a cut dominated in one sibling's
//! visited region may be the binding cut in another's — pooling siblings into
//! one shared stage-level bucket would silently drop that cut, with no error
//! beyond a worse policy. On a chain (`nodes[]` absent) node position equals
//! stage index, so this degenerates byte-for-byte to the pre-node-native
//! per-stage archive. Each `NodeStates` stores its state vectors in a single
//! flat `Vec<f64>` for cache-friendly iteration during the domination sweep.

use crate::setup::NodePos;

/// Single-node visited-states buffer.
///
/// Stores forward-pass trial points as a flat contiguous `Vec<f64>`:
/// entry `i * state_dimension .. (i + 1) * state_dimension` holds state `i`.
#[derive(Debug, Clone)]
pub struct NodeStates {
    data: Vec<f64>,
    count: usize,
    state_dimension: usize,
}

impl NodeStates {
    /// Creates a buffer pre-allocated for `capacity_states` state vectors.
    #[must_use]
    pub fn new(state_dimension: usize, capacity_states: usize) -> Self {
        Self {
            data: Vec::with_capacity(capacity_states * state_dimension),
            count: 0,
            state_dimension,
        }
    }

    /// Returns the number of states currently stored.
    #[must_use]
    pub fn count(&self) -> usize {
        self.count
    }

    /// Returns the dimension of each state vector.
    #[must_use]
    pub fn state_dimension(&self) -> usize {
        self.state_dimension
    }

    /// Append `total_fwd` state vectors from `gathered`, a flat slice of length
    /// `total_fwd * state_dimension`.
    ///
    /// # Panics (debug only)
    ///
    /// Panics if `gathered.len() != total_fwd * self.state_dimension`.
    pub fn append(&mut self, gathered: &[f64], total_fwd: usize) {
        debug_assert_eq!(gathered.len(), total_fwd * self.state_dimension);
        self.data.extend_from_slice(gathered);
        self.count += total_fwd;
    }

    /// Return the flat slice of all accumulated states (length `count * state_dimension`).
    #[must_use]
    pub fn states(&self) -> &[f64] {
        &self.data[..self.count * self.state_dimension]
    }

    /// Retain only the most recent `window_states` state vectors; no-op when
    /// `count <= window_states`.
    ///
    /// The O(retained) left-shift from `Vec::drain` is acceptable because
    /// trimming runs at the cut-selection cadence, not per cut or per state.
    pub fn trim_to_window(&mut self, window_states: usize) {
        if self.count <= window_states {
            return;
        }
        let to_remove = self.count - window_states;
        let drain_len = to_remove * self.state_dimension;
        debug_assert!(drain_len <= self.data.len());
        self.data.drain(..drain_len);
        self.count = window_states;
        debug_assert_eq!(self.data.len(), self.count * self.state_dimension);
    }
}

/// Multi-node archive of visited forward-pass states, one [`NodeStates`] per
/// node.
///
/// Allocated when any [`CutSelectionStrategy`](crate::CutSelectionStrategy)
/// variant is enabled (the value-evaluation kernel evaluates every populated cut
/// at every archived state) or when state export is requested via
/// [`EventConfig::export_states`](crate::config::EventConfig::export_states).
#[derive(Debug, Clone)]
pub struct VisitedStatesArchive {
    nodes: Vec<NodeStates>,
    total_forward_passes: usize,
}

impl VisitedStatesArchive {
    /// Per-node pre-allocation cap, bounding upfront virtual-memory reservation
    /// when `max_iterations * total_forward_passes` is large; the `Vec` still
    /// grows past it on demand.
    const MAX_INITIAL_CAPACITY: usize = 4096;

    /// Creates an archive with one [`NodeStates`] per node, each pre-allocated
    /// for `max_iterations * total_forward_passes` state vectors capped at
    /// [`Self::MAX_INITIAL_CAPACITY`].
    #[must_use]
    pub fn new(
        num_nodes: usize,
        state_dimension: usize,
        max_iterations: u64,
        total_forward_passes: usize,
    ) -> Self {
        let total_states = usize::try_from(max_iterations)
            .unwrap_or(usize::MAX)
            .saturating_mul(total_forward_passes);
        let capacity_per_node = total_states.min(Self::MAX_INITIAL_CAPACITY);
        let nodes = (0..num_nodes)
            .map(|_| NodeStates::new(state_dimension, capacity_per_node))
            .collect();
        Self {
            nodes,
            total_forward_passes,
        }
    }

    /// Returns the number of nodes in the archive.
    #[must_use]
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Returns a shared reference to the [`NodeStates`] for `node`.
    ///
    /// # Panics
    ///
    /// Panics if `node >= self.num_nodes()`.
    #[must_use]
    pub fn node(&self, node: NodePos) -> &NodeStates {
        &self.nodes[node.0]
    }

    /// Returns a mutable reference to the [`NodeStates`] for `node`.
    ///
    /// # Panics
    ///
    /// Panics if `node >= self.num_nodes()`.
    pub fn node_mut(&mut self, node: NodePos) -> &mut NodeStates {
        &mut self.nodes[node.0]
    }

    /// Archive one iteration's gathered states for `node`.
    pub fn archive_gathered_states(&mut self, node: NodePos, gathered: &[f64], total_fwd: usize) {
        self.nodes[node.0].append(gathered, total_fwd);
    }

    /// Archive the single incoming `state` a node saw on the enumerated
    /// (node-native) path, appending exactly ONE vector — the distinct-
    /// incoming-state count per node under enumeration is 1, so this never
    /// replicates the state `total_forward_passes` times the way
    /// [`Self::archive_gathered_states`] does for the sampled path.
    pub fn archive_one_state(&mut self, node: NodePos, state: &[f64]) {
        self.nodes[node.0].append(state, 1);
    }

    /// Return the flat state slice for `node`.
    #[must_use]
    pub fn states_for_node(&self, node: NodePos) -> &[f64] {
        self.nodes[node.0].states()
    }

    /// The global state dimension every node's buffer is packed at — the stride
    /// a reader must walk the archive by, so the walk stride equals the write
    /// stride by construction rather than by re-deriving it from another source.
    #[must_use]
    pub fn packing_stride(&self) -> usize {
        self.nodes.first().map_or(0, NodeStates::state_dimension)
    }

    /// Number of states accumulated at a given node.
    #[must_use]
    pub fn count(&self, node: NodePos) -> usize {
        self.nodes[node.0].count()
    }

    /// Trim each node to the most recent `window_iterations` iterations' worth
    /// of forward-pass states (`window_iterations * total_forward_passes`).
    ///
    /// Called once per cut-selection check, so the post-trim (steady-state)
    /// archive stays bounded regardless of training length. Because the trim
    /// runs **after** selection has consumed the archive, the pre-trim peak is
    /// ~`2 * window_iterations * total_forward_passes`; that ~2x is by design —
    /// selection evaluates against the full accumulated archive for quality.
    pub fn trim_to_window(&mut self, window_iterations: u64) {
        let window_states = usize::try_from(window_iterations)
            .unwrap_or(usize::MAX)
            .saturating_mul(self.total_forward_passes);
        for node in &mut self.nodes {
            node.trim_to_window(window_states);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{NodeStates, VisitedStatesArchive};
    use crate::setup::NodePos;

    /// Build a synthetic gathered buffer: `base, base+1, ..., base + total_fwd*state_dim - 1`.
    #[allow(clippy::cast_precision_loss)]
    fn make_gathered(state_dim: usize, total_fwd: usize, base: f64) -> Vec<f64> {
        (0..total_fwd * state_dim)
            .map(|i| base + i as f64)
            .collect()
    }

    #[test]
    fn node_states_new_preallocates() {
        let s = NodeStates::new(4, 100);
        assert!(s.states().is_empty());
        assert_eq!(s.count(), 0);
        assert_eq!(s.state_dimension(), 4);
        assert!(s.data.capacity() >= 400);
    }

    #[test]
    fn node_states_append_single_batch() {
        let mut s = NodeStates::new(2, 10);
        let gathered = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        s.append(&gathered, 3);
        assert_eq!(s.count(), 3);
        assert_eq!(s.states(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn node_states_append_multiple_batches() {
        let mut s = NodeStates::new(2, 10);
        let g1 = make_gathered(2, 3, 0.0);
        s.append(&g1, 3);
        let g2 = make_gathered(2, 2, 100.0);
        s.append(&g2, 2);
        assert_eq!(s.count(), 5);
        assert_eq!(
            s.states(),
            &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 100.0, 101.0, 102.0, 103.0]
        );
    }

    #[test]
    fn node_states_empty_states() {
        let s = NodeStates::new(3, 50);
        assert_eq!(s.states(), &[] as &[f64]);
        assert_eq!(s.count(), 0);
    }

    #[test]
    fn archive_new_creates_correct_nodes() {
        let a = VisitedStatesArchive::new(5, 4, 10, 20);
        assert_eq!(a.num_nodes(), 5);
        for t in 0..5 {
            assert_eq!(a.count(NodePos(t)), 0);
            assert!(a.states_for_node(NodePos(t)).is_empty());
        }
    }

    #[test]
    fn archive_gathered_states_delegates() {
        let mut a = VisitedStatesArchive::new(4, 3, 10, 10);
        let gathered = make_gathered(3, 3, 1.0);
        a.archive_gathered_states(NodePos(2), &gathered, 3);
        assert_eq!(a.count(NodePos(2)), 3);
        assert_eq!(a.count(NodePos(0)), 0);
        assert_eq!(a.count(NodePos(1)), 0);
        assert_eq!(a.count(NodePos(3)), 0);
    }

    // -- One-state-per-node archive (enumerated path) --------------------

    /// On a trunk+fan shape, the enumerated path archives exactly ONE state per
    /// cut-generating node per iteration — `count` reflects the node's distinct
    /// incoming states (1 per iteration), never `total_forward_passes`.
    #[test]
    fn archive_one_state_appends_a_single_vector_per_node() {
        // total_forward_passes = 5, but each enumerated archive call adds one.
        let mut a = VisitedStatesArchive::new(3, 2, 10, 5);
        a.archive_one_state(NodePos(0), &[10.0, 11.0]);
        a.archive_one_state(NodePos(1), &[20.0, 21.0]);

        assert_eq!(
            a.count(NodePos(0)),
            1,
            "one distinct incoming state, not total_fwd"
        );
        assert_eq!(a.count(NodePos(1)), 1);
        assert_eq!(a.count(NodePos(2)), 0, "an unvisited node stays empty");
        assert_eq!(a.states_for_node(NodePos(0)), &[10.0, 11.0]);
    }

    /// Across iterations the enumerated node accumulates one state each — after
    /// three iterations its count is 3 (the iteration count), never a
    /// `total_forward_passes` multiple.
    #[test]
    fn archive_one_state_accumulates_one_per_iteration() {
        let mut a = VisitedStatesArchive::new(2, 2, 10, 8);
        for it in 0..3_i32 {
            let base = f64::from(it) * 100.0;
            a.archive_one_state(NodePos(1), &[base, base + 1.0]);
        }
        assert_eq!(a.count(NodePos(1)), 3);
        assert_eq!(
            a.states_for_node(NodePos(1)),
            &[0.0, 1.0, 100.0, 101.0, 200.0, 201.0]
        );
    }

    #[test]
    fn archive_accumulates_across_iterations() {
        let mut a = VisitedStatesArchive::new(3, 2, 10, 5);
        let g1 = make_gathered(2, 5, 0.0);
        a.archive_gathered_states(NodePos(1), &g1, 5);
        let g2 = make_gathered(2, 5, 100.0);
        a.archive_gathered_states(NodePos(1), &g2, 5);
        assert_eq!(a.count(NodePos(1)), 10);
    }

    #[test]
    fn archive_packing_stride_reports_write_dimension() {
        let a = VisitedStatesArchive::new(3, 7, 10, 5);
        assert_eq!(a.packing_stride(), 7);
    }

    #[test]
    fn archive_states_for_node_returns_flat_slice() {
        let mut a = VisitedStatesArchive::new(3, 2, 10, 10);
        let gathered = vec![10.0, 20.0, 30.0, 40.0];
        a.archive_gathered_states(NodePos(1), &gathered, 2);
        assert_eq!(a.states_for_node(NodePos(1)), &[10.0, 20.0, 30.0, 40.0]);
        assert!(a.states_for_node(NodePos(0)).is_empty());
        assert!(a.states_for_node(NodePos(2)).is_empty());
    }

    // -- Sibling-node independence (AC 2) --------------------------------

    /// A two-sibling fan: node 0 is the parent stage's node, nodes 1 and 2 are
    /// two sibling nodes at the next stage (mirroring `NodeGraph`'s K-fan
    /// shape). Writing sibling A's (node 1) states must never appear in
    /// sibling B's (node 2) archive, or in the parent's — each node's
    /// dominance test must see only its own visited region.
    #[test]
    fn sibling_nodes_at_the_same_stage_archive_independently() {
        let mut a = VisitedStatesArchive::new(3, 2, 10, 5);
        let first_sibling_states = make_gathered(2, 3, 0.0);
        let second_sibling_states = make_gathered(2, 3, 100.0);

        a.archive_gathered_states(NodePos(1), &first_sibling_states, 3);
        a.archive_gathered_states(NodePos(2), &second_sibling_states, 3);

        assert_eq!(
            a.states_for_node(NodePos(1)),
            first_sibling_states.as_slice()
        );
        assert_eq!(
            a.states_for_node(NodePos(2)),
            second_sibling_states.as_slice()
        );
        assert_eq!(a.count(NodePos(1)), 3);
        assert_eq!(a.count(NodePos(2)), 3);
        assert!(
            a.states_for_node(NodePos(0)).is_empty(),
            "the parent node must not receive either sibling's states"
        );
        assert_ne!(
            a.states_for_node(NodePos(1)),
            a.states_for_node(NodePos(2)),
            "sibling A's states must not leak into sibling B's archive"
        );
    }

    // -- NodeStates::trim_to_window ------------------------------------

    // Trimming 100 dim-2 states to window 30 keeps the LAST 30
    // (elements 140..200 of the original flat buffer).
    #[test]
    fn node_states_trim_to_window_drops_oldest_states() {
        let state_dim = 2;
        let total = 100;
        let mut s = NodeStates::new(state_dim, total);
        let gathered = make_gathered(state_dim, total, 0.0);
        s.append(&gathered, total);
        assert_eq!(s.count(), 100);
        assert_eq!(s.states().len(), 200);

        s.trim_to_window(30);

        assert_eq!(s.count(), 30);
        assert_eq!(s.states().len(), 60);
        // Retained slice equals the tail of the original buffer
        // (elements 140..200, i.e. 140.0..199.0).
        let expected: Vec<f64> = (140..200).map(f64::from).collect();
        assert_eq!(s.states(), expected.as_slice());
    }

    #[test]
    fn node_states_trim_to_window_noop_when_count_below_window() {
        let state_dim = 2;
        let total = 20;
        let mut s = NodeStates::new(state_dim, total);
        let gathered = make_gathered(state_dim, total, 0.0);
        s.append(&gathered, total);
        assert_eq!(s.count(), 20);
        let before: Vec<f64> = s.states().to_vec();

        s.trim_to_window(50);

        assert_eq!(s.count(), 20);
        assert_eq!(s.states(), before.as_slice());
    }

    #[test]
    fn node_states_trim_to_window_count_equals_window_is_noop() {
        let state_dim = 3;
        let total = 7;
        let mut s = NodeStates::new(state_dim, total);
        let gathered = make_gathered(state_dim, total, 10.0);
        s.append(&gathered, total);
        let before: Vec<f64> = s.states().to_vec();

        s.trim_to_window(7);

        assert_eq!(s.count(), 7);
        assert_eq!(s.states(), before.as_slice());
    }

    #[test]
    fn node_states_trim_to_window_to_zero_clears_buffer() {
        let state_dim = 2;
        let total = 5;
        let mut s = NodeStates::new(state_dim, total);
        let gathered = make_gathered(state_dim, total, 0.0);
        s.append(&gathered, total);

        s.trim_to_window(0);

        assert_eq!(s.count(), 0);
        assert!(s.states().is_empty());
    }

    // After a trim, the retained window stays at the head and newly appended
    // states sit at the tail.
    #[test]
    fn node_states_trim_then_append_preserves_data() {
        let state_dim = 2;
        let mut s = NodeStates::new(state_dim, 100);
        // Initial: 10 states with base 0.0  (values 0..20).
        s.append(&make_gathered(state_dim, 10, 0.0), 10);
        // Trim down to last 4 states (elements 12..20 of original).
        s.trim_to_window(4);
        assert_eq!(s.count(), 4);
        let retained: Vec<f64> = (12..20).map(f64::from).collect();
        assert_eq!(s.states(), retained.as_slice());

        // Append 3 new states starting at 100.0.
        s.append(&make_gathered(state_dim, 3, 100.0), 3);
        assert_eq!(s.count(), 7);

        // Final layout = retained tail + newly appended states.
        let mut expected = retained;
        expected.extend((0..6).map(|i| 100.0 + f64::from(i)));
        assert_eq!(s.states(), expected.as_slice());
    }

    // -- VisitedStatesArchive::trim_to_window ---------------------------

    // 5 nodes of 100 states each, trimmed with window=3 and total_fwd=10,
    // leaves each node with 30 states.
    #[test]
    fn archive_trim_to_window_trims_each_node() {
        let num_nodes = 5;
        let state_dim = 2;
        let total_fwd = 10;
        let mut a = VisitedStatesArchive::new(num_nodes, state_dim, 10, total_fwd);

        // Fill each node with 100 states (10 iterations * 10 forward passes).
        for t in 0..num_nodes {
            for it in 0..10_i32 {
                let base = f64::from(i32::try_from(t).unwrap()) * 1000.0 + f64::from(it) * 100.0;
                let gathered = make_gathered(state_dim, total_fwd, base);
                a.archive_gathered_states(NodePos(t), &gathered, total_fwd);
            }
            assert_eq!(a.count(NodePos(t)), 100);
        }

        a.trim_to_window(3);

        for t in 0..num_nodes {
            assert!(
                a.count(NodePos(t)) <= 30,
                "node {t} has count {}",
                a.count(NodePos(t))
            );
            assert_eq!(a.count(NodePos(t)), 30);
            assert_eq!(a.states_for_node(NodePos(t)).len(), 30 * state_dim);
        }
    }

    #[test]
    fn archive_trim_to_window_noop_when_within_window() {
        let total_fwd = 10;
        let mut a = VisitedStatesArchive::new(2, 2, 10, total_fwd);
        // 2 iterations * 10 forward passes = 20 states per node.
        for it in 0..2_i32 {
            let gathered = make_gathered(2, total_fwd, f64::from(it) * 100.0);
            a.archive_gathered_states(NodePos(0), &gathered, total_fwd);
            a.archive_gathered_states(NodePos(1), &gathered, total_fwd);
        }
        let before_0: Vec<f64> = a.states_for_node(NodePos(0)).to_vec();
        let before_1: Vec<f64> = a.states_for_node(NodePos(1)).to_vec();

        // window=5 -> window_states = 50 > 20.
        a.trim_to_window(5);

        assert_eq!(a.count(NodePos(0)), 20);
        assert_eq!(a.count(NodePos(1)), 20);
        assert_eq!(a.states_for_node(NodePos(0)), before_0.as_slice());
        assert_eq!(a.states_for_node(NodePos(1)), before_1.as_slice());
    }

    #[test]
    fn archive_trim_to_window_retains_most_recent() {
        let state_dim = 2;
        let total_fwd = 5;
        let mut a = VisitedStatesArchive::new(1, state_dim, 10, total_fwd);

        // 4 iterations, each adding 5 states with a distinct base.
        // iter 0: base 0.0   -> values 0..10
        // iter 1: base 100.0 -> values 100..110
        // iter 2: base 200.0 -> values 200..210
        // iter 3: base 300.0 -> values 300..310
        for it in 0..4_i32 {
            let base = f64::from(it) * 100.0;
            a.archive_gathered_states(
                NodePos(0),
                &make_gathered(state_dim, total_fwd, base),
                total_fwd,
            );
        }
        assert_eq!(a.count(NodePos(0)), 20);

        // Keep last 2 iterations (10 states).
        a.trim_to_window(2);
        assert_eq!(a.count(NodePos(0)), 10);

        // Expected: values from iterations 2 and 3.
        let mut expected: Vec<f64> = (200..210).map(f64::from).collect();
        expected.extend((300..310).map(f64::from));
        assert_eq!(a.states_for_node(NodePos(0)), expected.as_slice());
    }

    #[test]
    fn archive_trim_to_window_zero_clears_all_nodes() {
        let total_fwd = 4;
        let mut a = VisitedStatesArchive::new(3, 2, 10, total_fwd);
        for t in 0..3 {
            a.archive_gathered_states(NodePos(t), &make_gathered(2, total_fwd, 0.0), total_fwd);
        }

        a.trim_to_window(0);

        for t in 0..3 {
            assert_eq!(a.count(NodePos(t)), 0);
            assert!(a.states_for_node(NodePos(t)).is_empty());
        }
    }
}
