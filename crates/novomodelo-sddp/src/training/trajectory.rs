//! Trajectory records produced by the SDDP forward pass.
//!
//! A [`TrajectoryRecord`] captures the complete LP solution for one scenario
//! at one stage. It is the unit of data that flows from the forward pass to
//! the backward pass: the forward pass produces one record per
//! `(scenario, stage)` pair and stores them in a flat `Vec` indexed as
//!
//! ```text
//! records[scenario * n_stages + stage]
//! ```
//!
//! The backward pass reads the `state` and `dual` fields from each record to
//! generate Benders cuts.
//!
//! `TrajectoryRecord`s are owned by the forward pass workspace, **not** stored
//! inside the [`FutureCostFunction`].
//!
//! [`FutureCostFunction`]: crate::cut::fcf::FutureCostFunction

use crate::setup::NodeId;

/// LP solution for one scenario at one stage, produced by the forward pass.
///
/// The backward pass reads `state` and `dual` to compute cut coefficients; the
/// training loop monitors `stage_cost` for convergence statistics.
#[derive(Debug, Clone)]
pub struct TrajectoryRecord {
    /// Full primal solution vector (all LP variables in column order).
    pub primal: Vec<f64>,

    /// Full dual solution vector (one value per LP constraint).
    pub dual: Vec<f64>,

    /// LP objective value at this stage, excluding the future-cost theta variable.
    pub stage_cost: f64,

    /// Declared JSON node id this visit resolved against (`NodeGraph::node_ids`).
    pub node_id: NodeId,

    /// End-of-stage state vector (length `n_state`), extracted from `primal` by the stage indexer.
    pub state: Vec<f64>,
}

#[cfg(test)]
mod tests {
    use super::{NodeId, TrajectoryRecord};

    #[test]
    fn construct_and_access_all_fields() {
        let record = TrajectoryRecord {
            primal: vec![1.0, 2.0, 3.0],
            dual: vec![0.5, 0.6],
            stage_cost: 42.0,
            node_id: NodeId(3),
            state: vec![9.0, 8.0],
        };

        assert_eq!(record.primal, vec![1.0, 2.0, 3.0]);
        assert_eq!(record.dual, vec![0.5, 0.6]);
        assert_eq!(record.stage_cost, 42.0);
        assert_eq!(record.node_id, NodeId(3));
        assert_eq!(record.state, vec![9.0, 8.0]);
    }

    #[test]
    fn stage_cost_value_is_accessible() {
        let record = TrajectoryRecord {
            primal: vec![],
            dual: vec![],
            stage_cost: 42.0,
            node_id: NodeId(0), // unread here — this test exercises stage_cost only
            state: vec![],
        };
        assert_eq!(record.stage_cost, 42.0);
    }

    #[test]
    fn clone_produces_identical_independent_copy() {
        let original = TrajectoryRecord {
            primal: vec![1.0, 2.0],
            dual: vec![3.0],
            stage_cost: 7.5,
            node_id: NodeId(5),
            state: vec![4.0, 5.0],
        };

        let mut cloned = original.clone();

        assert_eq!(cloned.primal, original.primal);
        assert_eq!(cloned.dual, original.dual);
        assert_eq!(cloned.stage_cost, original.stage_cost);
        assert_eq!(cloned.node_id, original.node_id);
        assert_eq!(cloned.state, original.state);

        cloned.stage_cost = 0.0;
        cloned.primal.push(99.0);
        assert_eq!(original.stage_cost, 7.5);
        assert_eq!(original.primal.len(), 2);
    }

    #[test]
    fn debug_format_is_non_empty() {
        let record = TrajectoryRecord {
            primal: vec![1.0],
            dual: vec![2.0],
            stage_cost: 3.0,
            node_id: NodeId(0), // unread here — this test only checks the Debug output is non-empty
            state: vec![4.0],
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
                primal: vec![],
                dual: vec![],
                stage_cost: f64::from(i),
                node_id: NodeId(0), // unread here — this test only exercises flat-index addressing
                state: vec![],
            })
            .collect();

        records[scenario * n_stages + 3].stage_cost = 999.0;

        assert_eq!(records[scenario * n_stages + 3].stage_cost, 999.0);
        // Each record's initial stage_cost is its flat index, so the untouched
        // neighbours read scenario=1,stage=2 → 7.0 and scenario=1,stage=4 → 9.0.
        assert_eq!(records[scenario * n_stages + 2].stage_cost, 7.0);
        assert_eq!(records[scenario * n_stages + 4].stage_cost, 9.0);
    }
}
