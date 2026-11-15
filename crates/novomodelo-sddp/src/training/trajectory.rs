//! Trajectory records produced by the SDDP forward pass.
//!
//! A [`TrajectoryRecord`] holds the stage cost, the visited node and the
//! end-of-stage state of one local trial at one stage. The forward pass stores
//! them in a flat `Vec` indexed as
//!
//! ```text
//! records[m * n_stages + stage]
//! ```
//!
//! where `m` is the local trial. The backward pass reads `state` and `node_id`.
//! A record also holds the B of its node's frozen rows, for the frozen backward.
//!
//! `TrajectoryRecord`s are owned by the forward pass workspace, **not** stored
//! inside the [`FutureCostFunction`].
//!
//! [`FutureCostFunction`]: crate::cut::fcf::FutureCostFunction

use crate::{
    context::TrainingContext, setup::NodeId, training::stage_solve_prep::closes_lag_period,
    workspace::workspace::reserve_to,
};

/// One local trial's stage cost, visited node and end-of-stage state at one stage.
#[derive(Debug, Clone, Default)]
pub struct TrajectoryRecord {
    /// LP objective value at this stage, excluding the future-cost theta variable.
    pub stage_cost: f64,

    /// Declared JSON node id this visit resolved against (`NodeGraph::node_ids`).
    pub node_id: NodeId,

    /// End-of-stage state vector (length `n_state`).
    pub state: Vec<f64>,

    /// The lag accumulator this visit's pin read before the solve (length `n_hydros`); empty
    /// unless a pool closes a lag period. Never persisted or exchanged.
    pub(crate) partial_sum: Vec<f64>,

    pub(crate) frozen_brackets: FrozenBrackets,
}

impl TrajectoryRecord {
    pub(crate) fn allocate(count: usize, training_ctx: &TrainingContext<'_>) -> Vec<Self> {
        let closing = training_ctx
            .cut_state_layouts
            .iter()
            .any(|layout| closes_lag_period(layout.lag_point()));
        let n_partial_sum = if closing {
            training_ctx.state.hydro_count
        } else {
            0
        };
        (0..count)
            .map(|_| Self {
                state: vec![0.0; training_ctx.state.n_state],
                partial_sum: vec![0.0; n_partial_sum],
                ..Self::default()
            })
            .collect()
    }

    /// The partial sum this visit's pin read; empty unless a pool closes a lag period.
    #[cfg(any(test, feature = "test-support"))]
    #[must_use]
    pub fn partial_sum(&self) -> &[f64] {
        &self.partial_sum
    }
}

/// The cut-RHS brackets B of one node's frozen rows in `active_cuts()` order,
/// tagged by the forward visit that wrote them. Never persisted or exchanged;
/// a read at any other `(iteration, node)` or row count misses, and a default
/// slab never hits.
#[derive(Debug, Clone, Default)]
pub(crate) struct FrozenBrackets {
    tag: Option<(u64, NodeId)>,
    values: Vec<f64>,
}

impl FrozenBrackets {
    /// Grows capacity to `rows`; never shrinks it. Runs at freeze, off the hot path.
    pub(crate) fn reserve(&mut self, rows: usize) {
        reserve_to(&mut self.values, rows);
    }

    /// Overwrites the slab with `brackets`; allocation-free within the reserved capacity.
    pub(crate) fn write(&mut self, iteration: u64, node_id: NodeId, brackets: &[f64]) {
        debug_assert!(
            brackets.len() <= self.values.capacity(),
            "a forward write exceeds the capacity reserved at freeze"
        );
        self.values.clear();
        self.values.extend_from_slice(brackets);
        self.tag = Some((iteration, node_id));
    }

    pub(crate) fn read(&self, iteration: u64, node_id: NodeId, rows: usize) -> Option<&[f64]> {
        (self.tag == Some((iteration, node_id)) && self.values.len() == rows)
            .then_some(self.values.as_slice())
    }

    /// The buffer's address and capacity, to tell where it grows.
    #[cfg(test)]
    pub(crate) fn allocation(&self) -> (usize, usize) {
        (self.values.as_ptr().addr(), self.values.capacity())
    }
}

#[cfg(test)]
mod tests {
    use super::{FrozenBrackets, NodeId, TrajectoryRecord};
    use crate::test_support::{TrainingContextFixture, state_layout};

    #[test]
    fn construct_and_access_all_fields() {
        let record = TrajectoryRecord {
            stage_cost: 42.0,
            node_id: NodeId(3),
            state: vec![9.0, 8.0],
            ..TrajectoryRecord::default()
        };

        assert_eq!(record.stage_cost, 42.0);
        assert_eq!(record.node_id, NodeId(3));
        assert_eq!(record.state, vec![9.0, 8.0]);
    }

    #[test]
    fn default_is_an_empty_zero_cost_record() {
        let record = TrajectoryRecord::default();

        assert_eq!(record.stage_cost, 0.0);
        assert_eq!(record.node_id, NodeId(0));
        assert!(record.state.is_empty());
    }

    #[test]
    fn clone_produces_identical_independent_copy() {
        let original = TrajectoryRecord {
            stage_cost: 7.5,
            node_id: NodeId(5),
            state: vec![4.0, 5.0],
            ..TrajectoryRecord::default()
        };

        let mut cloned = original.clone();

        assert_eq!(cloned.stage_cost, original.stage_cost);
        assert_eq!(cloned.node_id, original.node_id);
        assert_eq!(cloned.state, original.state);

        cloned.stage_cost = 0.0;
        cloned.state.push(99.0);
        assert_eq!(original.stage_cost, 7.5);
        assert_eq!(original.state.len(), 2);
    }

    #[test]
    fn debug_format_is_non_empty() {
        let record = TrajectoryRecord {
            stage_cost: 3.0,
            state: vec![4.0],
            ..TrajectoryRecord::default()
        };
        let s = format!("{record:?}");
        assert!(!s.is_empty());
    }

    #[test]
    fn flat_vec_indexing_pattern_works_correctly() {
        let n_stages: usize = 5;
        let scenario: usize = 1;

        let mut records: Vec<TrajectoryRecord> = (0_u8..15_u8)
            .map(|i| TrajectoryRecord {
                stage_cost: f64::from(i),
                ..TrajectoryRecord::default()
            })
            .collect();

        records[scenario * n_stages + 3].stage_cost = 999.0;

        assert_eq!(records[scenario * n_stages + 3].stage_cost, 999.0);
        // Each record's initial stage_cost is its flat index, so the untouched
        // neighbours read scenario=1,stage=2 → 7.0 and scenario=1,stage=4 → 9.0.
        assert_eq!(records[scenario * n_stages + 2].stage_cost, 7.0);
        assert_eq!(records[scenario * n_stages + 4].stage_cost, 9.0);
    }

    #[test]
    fn a_record_holds_one_partial_sum_per_hydro_when_a_pool_is_closing() {
        let fixture = TrainingContextFixture::new(state_layout(2, 1)).closing_lag_points();
        let training_ctx = fixture.training_ctx();

        let records = TrajectoryRecord::allocate(3, &training_ctx);

        assert_eq!(records.len(), 3);
        for record in &records {
            assert_eq!(record.partial_sum, [0.0, 0.0]);
            assert_eq!(record.state.len(), training_ctx.state.n_state);
        }
    }

    #[test]
    fn a_record_holds_no_partial_sum_when_no_pool_is_closing() {
        let fixture = TrainingContextFixture::new(state_layout(2, 1));
        let training_ctx = fixture.training_ctx();

        let records = TrajectoryRecord::allocate(3, &training_ctx);

        assert_eq!(records.len(), 3);
        for record in &records {
            assert_eq!(record.partial_sum.capacity(), 0);
            assert_eq!(record.state.len(), training_ctx.state.n_state);
        }
    }

    #[test]
    fn a_default_slab_misses_at_every_tag() {
        let slab = FrozenBrackets::default();

        assert_eq!(slab.read(0, NodeId(0), 0), None);
        assert_eq!(slab.read(1, NodeId(0), 0), None);
    }

    #[test]
    fn a_written_slab_hits_only_at_its_own_iteration_and_node() {
        let mut slab = FrozenBrackets::default();
        slab.reserve(2);

        slab.write(3, NodeId(7), &[1.5, 2.5]);

        assert_eq!(slab.read(3, NodeId(7), 2), Some(&[1.5, 2.5][..]));
        assert_eq!(slab.read(4, NodeId(7), 2), None);
        assert_eq!(slab.read(3, NodeId(8), 2), None);
    }

    #[test]
    fn reserve_never_shrinks_and_a_write_within_capacity_keeps_the_buffer() {
        let mut slab = FrozenBrackets::default();
        slab.reserve(4);
        let capacity = slab.values.capacity();
        assert!(capacity >= 4);

        slab.reserve(2);
        assert_eq!(slab.values.capacity(), capacity);

        let buffer = slab.values.as_ptr();
        slab.write(1, NodeId(0), &[1.0, 2.0, 3.0, 4.0]);
        slab.write(2, NodeId(0), &[5.0]);
        assert_eq!(slab.values.as_ptr(), buffer);
        assert_eq!(slab.values.capacity(), capacity);

        slab.reserve(capacity + 1);
        assert!(slab.values.capacity() > capacity);
    }
}
