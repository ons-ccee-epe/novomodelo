//! Visited trial states, archived for state export.
//!
//! The training loop allocates the archive when [`EventConfig::export_states`]
//! is set, and every visited state of the run stays in it.
//!
//! The archive holds one [`NodeStates`] per **node**, indexed by node position,
//! never by stage. The sampled backward pass gathers a level's states with no
//! per-node split, so every node of a multi-node level receives the level's
//! states; the enumerated pass appends each node's own trial state. On a chain
//! (`nodes[]` absent) node position equals stage index. Each `NodeStates`
//! stores its state vectors in one flat `Vec<f64>`.
//!
//! [`EventConfig::export_states`]: crate::config::EventConfig::export_states

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
}

/// Multi-node archive of visited forward-pass states, one [`NodeStates`] per
/// node.
#[derive(Debug, Clone)]
pub struct VisitedStatesArchive {
    nodes: Vec<NodeStates>,
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
        Self { nodes }
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

    /// Archive the single outgoing `state` of a node on the enumerated
    /// (node-native) path, appending exactly ONE vector — a node's cut is
    /// computed at one state under enumeration, so this never replicates the
    /// state `total_forward_passes` times the way
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

    // -- Sibling-node independence ---------------------------------------

    /// A two-sibling fan: node 0 is the parent stage's node, nodes 1 and 2 are
    /// two sibling nodes at the next stage (mirroring `NodeGraph`'s K-fan
    /// shape). Sibling A's (node 1) states must never appear in sibling B's
    /// (node 2) archive, or in the parent's.
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
}
