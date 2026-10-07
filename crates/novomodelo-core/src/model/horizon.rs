//! Horizon topology: the graph of stage-to-stage transitions, horizon type,
//! and global discount rate.
//!
//! Relocated out of [`super::temporal`], which keeps the stage/block/season
//! types the transitions and nodes below reference.

use std::collections::BTreeMap;

use crate::{Node, PolicyGraphType, SeasonMap, Transition};

/// Parsed and validated horizon topology: stage transitions, horizon type,
/// and global discount rate.
///
/// The clarity-first topology loaded from `stages.json`; the solver-level
/// `HorizonMode` enum is built from it at initialization.
/// See [Horizon Mode Trait](../architecture/horizon-mode-trait.md).
///
/// Source: `stages.json` `policy_graph`.
/// See [Input Scenarios §1.2](input-scenarios.md).
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct HorizonGraph {
    /// Horizon type: finite (acyclic chain) or cyclic (infinite periodic).
    pub graph_type: PolicyGraphType,

    /// Global annual discount rate; `0.0` disables discounting. Must be `> 0`
    /// for cyclic graphs to converge (validation rule 7).
    /// See [Discount Rate §3](../math/discount-rate.md).
    pub annual_discount_rate: f64,

    /// Stage transitions. Finite horizon: a linear chain or DAG. Cyclic: at least
    /// one back-edge (`source_id >= target_id`).
    pub transitions: Vec<Transition>,

    /// Nodes; empty ⇒ a stage chain (`Transition` endpoints are stage ids),
    /// non-empty ⇒ node-native (endpoints are node ids).
    pub nodes: Vec<Node>,

    /// Per-study-stage annual discount rate override, keyed by `Stage::id`; a stage
    /// absent from the map uses `annual_discount_rate`. The declared home of the
    /// override (`stages[].annual_discount_rate_override`); the chain dialect folds
    /// its departing-edge `Transition::annual_discount_rate_override` in here at load.
    pub stage_discount_rate_overrides: BTreeMap<i32, f64>,

    /// Season definitions; `None` when none are provided or required.
    pub season_map: Option<SeasonMap>,
}

impl Default for HorizonGraph {
    /// A finite-horizon graph with no transitions and no discounting; `cobre-io`
    /// replaces it with the graph loaded from `stages.json`.
    fn default() -> Self {
        Self {
            graph_type: PolicyGraphType::FiniteHorizon,
            annual_discount_rate: 0.0,
            transitions: Vec::new(),
            nodes: Vec::new(),
            stage_discount_rate_overrides: BTreeMap::new(),
            season_map: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_horizon_graph_construction() {
        let transitions = vec![
            Transition {
                source_id: 1,
                target_id: 2,
                probability: 1.0,
                annual_discount_rate_override: None,
            },
            Transition {
                source_id: 2,
                target_id: 3,
                probability: 1.0,
                annual_discount_rate_override: Some(0.08),
            },
            Transition {
                source_id: 3,
                target_id: 4,
                probability: 1.0,
                annual_discount_rate_override: None,
            },
        ];

        let graph = HorizonGraph {
            stage_discount_rate_overrides: std::collections::BTreeMap::new(),
            graph_type: PolicyGraphType::FiniteHorizon,
            annual_discount_rate: 0.06,
            transitions,
            nodes: Vec::new(),
            season_map: None,
        };

        assert_eq!(graph.graph_type, PolicyGraphType::FiniteHorizon);
        assert!((graph.annual_discount_rate - 0.06).abs() < f64::EPSILON);
        assert_eq!(graph.transitions.len(), 3);
        assert_eq!(
            graph.transitions[1].annual_discount_rate_override,
            Some(0.08)
        );
        assert!(graph.season_map.is_none());
        assert!(graph.nodes.is_empty());
    }

    #[test]
    fn test_horizon_graph_carries_nodes() {
        let graph = HorizonGraph {
            stage_discount_rate_overrides: std::collections::BTreeMap::new(),
            graph_type: PolicyGraphType::FiniteHorizon,
            annual_discount_rate: 0.0,
            transitions: vec![Transition {
                source_id: 10,
                target_id: 11,
                probability: 1.0,
                annual_discount_rate_override: None,
            }],
            nodes: vec![
                Node {
                    id: 10,
                    stage_id: 0,
                    scenario_id: Some(3),
                    label: Some("root".to_string()),
                },
                Node {
                    id: 11,
                    stage_id: 1,
                    scenario_id: None,
                    label: None,
                },
            ],
            season_map: None,
        };

        assert_eq!(graph.nodes.len(), 2);
        assert_eq!(graph.nodes[0].stage_id, 0);
        assert_eq!(graph.nodes[0].scenario_id, Some(3));
        assert_eq!(graph.nodes[1].scenario_id, None);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_horizon_graph_serde_roundtrip() {
        let graph = HorizonGraph {
            stage_discount_rate_overrides: std::collections::BTreeMap::new(),
            graph_type: PolicyGraphType::FiniteHorizon,
            annual_discount_rate: 0.06,
            transitions: vec![
                Transition {
                    source_id: 1,
                    target_id: 2,
                    probability: 1.0,
                    annual_discount_rate_override: None,
                },
                Transition {
                    source_id: 2,
                    target_id: 3,
                    probability: 1.0,
                    annual_discount_rate_override: None,
                },
            ],
            nodes: Vec::new(),
            season_map: None,
        };

        let json = serde_json::to_string(&graph).unwrap();

        assert!(
            json.contains("\"graph_type\":\"FiniteHorizon\""),
            "JSON did not contain expected graph_type: {json}"
        );
        assert!(
            json.contains("\"annual_discount_rate\":0.06"),
            "JSON did not contain expected annual_discount_rate: {json}"
        );

        let deserialized: HorizonGraph = serde_json::from_str(&json).unwrap();
        assert_eq!(graph, deserialized);
    }
}
