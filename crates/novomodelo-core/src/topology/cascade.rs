//! Resolved hydro cascade topology.
//!
//! `CascadeTopology` holds the validated, cycle-free directed graph of hydro plant
//! relationships. It is built during case loading after all `Hydro` entities have
//! been validated and their `downstream_id` cross-references verified.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};

use crate::{EntityId, Hydro};

/// Resolved hydro cascade graph for water balance traversal.
///
/// A directed forest where each hydro has at most one downstream plant; built
/// from hydro `downstream_id` fields during System construction and immutable
/// thereafter. Traversing in topological order guarantees every upstream
/// inflow is computed before the downstream plant that receives it.
#[derive(Debug, Clone, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CascadeTopology {
    /// Terminal nodes (no downstream) are absent from the map.
    downstream: HashMap<EntityId, EntityId>,

    /// Headwaters (no upstream) are absent from the map.
    upstream: HashMap<EntityId, Vec<EntityId>>,

    /// Upstream before downstream; ties broken by `EntityId`'s inner i32 for determinism.
    topological_order: Vec<EntityId>,
}

impl CascadeTopology {
    /// Build cascade topology from hydro entities, assumed to be in canonical ID order.
    ///
    /// Does not validate: cycles and `downstream_id`s referencing a non-existent
    /// hydro are stored as-is for the separate validation layer to catch.
    #[must_use]
    pub fn build(hydros: &[Hydro]) -> Self {
        let mut downstream: HashMap<EntityId, EntityId> = HashMap::new();
        for hydro in hydros {
            if let Some(ds_id) = hydro.downstream_id {
                downstream.insert(hydro.id, ds_id);
            }
        }

        let mut upstream: HashMap<EntityId, Vec<EntityId>> = HashMap::new();
        for (from, to) in &downstream {
            upstream.entry(*to).or_default().push(*from);
        }
        for upstream_list in upstream.values_mut() {
            upstream_list.sort_by_key(|id| id.0);
        }

        let mut in_degree: HashMap<EntityId, usize> = HashMap::new();
        for hydro in hydros {
            in_degree.insert(hydro.id, 0);
        }
        for to in downstream.values() {
            if let Some(deg) = in_degree.get_mut(to) {
                *deg += 1;
            }
        }

        let mut ready: BinaryHeap<Reverse<i32>> = in_degree
            .iter()
            .filter(|&(_, deg)| *deg == 0)
            .map(|(id, _)| Reverse(id.0))
            .collect();

        let mut topological_order: Vec<EntityId> = Vec::with_capacity(hydros.len());

        while let Some(Reverse(current_raw)) = ready.pop() {
            let current = EntityId(current_raw);
            topological_order.push(current);

            if let Some(&ds_id) = downstream.get(&current)
                && let Some(deg) = in_degree.get_mut(&ds_id)
            {
                *deg -= 1;
                if *deg == 0 {
                    ready.push(Reverse(ds_id.0));
                }
            }
        }

        Self {
            downstream,
            upstream,
            topological_order,
        }
    }

    /// Returns the downstream hydro, or `None` for a terminal node.
    #[must_use]
    pub fn downstream(&self, hydro_id: EntityId) -> Option<EntityId> {
        self.downstream.get(&hydro_id).copied()
    }

    /// Returns the upstream hydros, or an empty slice for a headwater.
    #[must_use]
    pub fn upstream(&self, hydro_id: EntityId) -> &[EntityId] {
        self.upstream.get(&hydro_id).map_or(&[], Vec::as_slice)
    }

    /// Returns the topological ordering of all hydro IDs (upstream before downstream).
    #[must_use]
    pub fn topological_order(&self) -> &[EntityId] {
        &self.topological_order
    }

    /// Returns true if the given hydro is a headwater (has no upstream plants).
    #[must_use]
    pub fn is_headwater(&self, hydro_id: EntityId) -> bool {
        !self.upstream.contains_key(&hydro_id)
    }

    /// Returns true if the given hydro is a terminal node (has no downstream plant).
    #[must_use]
    pub fn is_terminal(&self, hydro_id: EntityId) -> bool {
        !self.downstream.contains_key(&hydro_id)
    }

    /// Returns the number of hydros in the cascade.
    #[must_use]
    pub fn len(&self) -> usize {
        self.topological_order.len()
    }

    /// Returns true if the cascade has no hydros.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.topological_order.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::CascadeTopology;
    use crate::EntityId;
    use crate::test_support::{HydroSpec, make_hydro};

    #[test]
    fn test_empty_cascade() {
        let topo = CascadeTopology::build(&[]);
        assert_eq!(topo.len(), 0);
        assert!(topo.is_empty());
        assert_eq!(topo.topological_order(), &[]);
    }

    #[test]
    fn test_single_hydro_terminal() {
        let hydros = vec![make_hydro(HydroSpec {
            id: 1,
            ..Default::default()
        })];
        let topo = CascadeTopology::build(&hydros);
        assert_eq!(topo.len(), 1);
        assert!(!topo.is_empty());
        assert!(topo.is_headwater(EntityId(1)));
        assert!(topo.is_terminal(EntityId(1)));
        assert_eq!(topo.topological_order(), &[EntityId(1)]);
        assert_eq!(topo.downstream(EntityId(1)), None);
        assert_eq!(topo.upstream(EntityId(1)), &[]);
    }

    #[test]
    fn test_linear_chain() {
        let hydros = vec![
            make_hydro(HydroSpec {
                id: 0,
                downstream_id: Some(1),
                ..Default::default()
            }),
            make_hydro(HydroSpec {
                id: 1,
                downstream_id: Some(2),
                ..Default::default()
            }),
            make_hydro(HydroSpec {
                id: 2,
                ..Default::default()
            }),
        ];
        let topo = CascadeTopology::build(&hydros);
        assert_eq!(topo.len(), 3);

        assert_eq!(topo.downstream(EntityId(0)), Some(EntityId(1)));
        assert_eq!(topo.downstream(EntityId(1)), Some(EntityId(2)));
        assert_eq!(topo.downstream(EntityId(2)), None);

        assert_eq!(topo.upstream(EntityId(0)), &[]);
        assert_eq!(topo.upstream(EntityId(1)), &[EntityId(0)]);
        assert_eq!(topo.upstream(EntityId(2)), &[EntityId(1)]);

        let order = topo.topological_order();
        let pos_a = order.iter().position(|&id| id == EntityId(0)).unwrap();
        let pos_b = order.iter().position(|&id| id == EntityId(1)).unwrap();
        let pos_c = order.iter().position(|&id| id == EntityId(2)).unwrap();
        assert!(pos_a < pos_b);
        assert!(pos_b < pos_c);
    }

    #[test]
    fn test_fork_merge() {
        let hydros = vec![
            make_hydro(HydroSpec {
                id: 0,
                downstream_id: Some(2),
                ..Default::default()
            }),
            make_hydro(HydroSpec {
                id: 1,
                downstream_id: Some(2),
                ..Default::default()
            }),
            make_hydro(HydroSpec {
                id: 2,
                ..Default::default()
            }),
        ];
        let topo = CascadeTopology::build(&hydros);
        assert_eq!(topo.len(), 3);

        assert_eq!(topo.downstream(EntityId(0)), Some(EntityId(2)));
        assert_eq!(topo.downstream(EntityId(1)), Some(EntityId(2)));
        assert_eq!(topo.downstream(EntityId(2)), None);

        let upstream_c = topo.upstream(EntityId(2));
        assert_eq!(upstream_c.len(), 2);
        assert!(upstream_c.contains(&EntityId(0)));
        assert!(upstream_c.contains(&EntityId(1)));
        assert_eq!(upstream_c[0], EntityId(0));
        assert_eq!(upstream_c[1], EntityId(1));

        let order = topo.topological_order();
        let pos_a = order.iter().position(|&id| id == EntityId(0)).unwrap();
        let pos_b = order.iter().position(|&id| id == EntityId(1)).unwrap();
        let pos_c = order.iter().position(|&id| id == EntityId(2)).unwrap();
        assert!(pos_a < pos_c);
        assert!(pos_b < pos_c);
    }

    #[test]
    fn test_parallel_chains() {
        let hydros = vec![
            make_hydro(HydroSpec {
                id: 0,
                downstream_id: Some(1),
                ..Default::default()
            }),
            make_hydro(HydroSpec {
                id: 1,
                ..Default::default()
            }),
            make_hydro(HydroSpec {
                id: 2,
                downstream_id: Some(3),
                ..Default::default()
            }),
            make_hydro(HydroSpec {
                id: 3,
                ..Default::default()
            }),
        ];
        let topo = CascadeTopology::build(&hydros);
        assert_eq!(topo.len(), 4);

        let order = topo.topological_order();
        let pos_a = order.iter().position(|&id| id == EntityId(0)).unwrap();
        let pos_b = order.iter().position(|&id| id == EntityId(1)).unwrap();
        let pos_c = order.iter().position(|&id| id == EntityId(2)).unwrap();
        let pos_d = order.iter().position(|&id| id == EntityId(3)).unwrap();

        assert!(pos_a < pos_b);
        assert!(pos_c < pos_d);
        assert_eq!(order.len(), 4);
    }

    #[test]
    fn test_all_terminal() {
        let hydros = vec![
            make_hydro(HydroSpec {
                id: 1,
                ..Default::default()
            }),
            make_hydro(HydroSpec {
                id: 2,
                ..Default::default()
            }),
            make_hydro(HydroSpec {
                id: 3,
                ..Default::default()
            }),
        ];
        let topo = CascadeTopology::build(&hydros);
        assert_eq!(topo.len(), 3);

        for id in [1, 2, 3] {
            assert!(topo.is_headwater(EntityId(id)));
            assert!(topo.is_terminal(EntityId(id)));
        }

        assert_eq!(
            topo.topological_order(),
            &[EntityId(1), EntityId(2), EntityId(3)]
        );
    }

    #[test]
    fn test_deterministic_ordering() {
        let hydros = vec![
            make_hydro(HydroSpec {
                id: 5,
                downstream_id: Some(10),
                ..Default::default()
            }),
            make_hydro(HydroSpec {
                id: 3,
                downstream_id: Some(10),
                ..Default::default()
            }),
            make_hydro(HydroSpec {
                id: 10,
                ..Default::default()
            }),
        ];
        let topo_a = CascadeTopology::build(&hydros);
        let topo_b = CascadeTopology::build(&hydros);
        assert_eq!(topo_a, topo_b);
        assert_eq!(topo_a.upstream(EntityId(10)), topo_b.upstream(EntityId(10)));
        assert_eq!(topo_a.topological_order(), topo_b.topological_order());
    }

    #[test]
    fn test_is_headwater() {
        let hydros = vec![
            make_hydro(HydroSpec {
                id: 0,
                downstream_id: Some(2),
                ..Default::default()
            }),
            make_hydro(HydroSpec {
                id: 1,
                downstream_id: Some(2),
                ..Default::default()
            }),
            make_hydro(HydroSpec {
                id: 2,
                ..Default::default()
            }),
        ];
        let topo = CascadeTopology::build(&hydros);
        assert!(topo.is_headwater(EntityId(0)));
        assert!(topo.is_headwater(EntityId(1)));
        assert!(!topo.is_headwater(EntityId(2)));
    }

    #[test]
    fn test_is_terminal() {
        let hydros = vec![
            make_hydro(HydroSpec {
                id: 0,
                downstream_id: Some(2),
                ..Default::default()
            }),
            make_hydro(HydroSpec {
                id: 1,
                downstream_id: Some(2),
                ..Default::default()
            }),
            make_hydro(HydroSpec {
                id: 2,
                ..Default::default()
            }),
        ];
        let topo = CascadeTopology::build(&hydros);
        assert!(!topo.is_terminal(EntityId(0)));
        assert!(!topo.is_terminal(EntityId(1)));
        assert!(topo.is_terminal(EntityId(2)));
    }

    #[test]
    fn test_len() {
        let hydros = vec![
            make_hydro(HydroSpec {
                id: 0,
                downstream_id: Some(1),
                ..Default::default()
            }),
            make_hydro(HydroSpec {
                id: 1,
                downstream_id: Some(2),
                ..Default::default()
            }),
            make_hydro(HydroSpec {
                id: 2,
                ..Default::default()
            }),
        ];
        let topo = CascadeTopology::build(&hydros);
        assert_eq!(topo.len(), 3);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_topology_serde_roundtrip_cascade() {
        let hydros = vec![
            make_hydro(HydroSpec {
                id: 0,
                downstream_id: Some(1),
                ..Default::default()
            }),
            make_hydro(HydroSpec {
                id: 1,
                downstream_id: Some(2),
                ..Default::default()
            }),
            make_hydro(HydroSpec {
                id: 2,
                ..Default::default()
            }),
        ];
        let topo = CascadeTopology::build(&hydros);
        let json = serde_json::to_string(&topo).unwrap();
        let deserialized: CascadeTopology = serde_json::from_str(&json).unwrap();
        assert_eq!(topo, deserialized);
    }
}
