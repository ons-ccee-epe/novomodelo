//! Bus entity — an electrical network node with power balance constraint.

use crate::EntityId;
use chrono::NaiveDate;

/// A single segment of the piecewise-linear deficit cost curve.
///
/// Segments are cumulative: each covers its own `depth_mw` at its own
/// `cost_per_mwh`; the final segment has `depth_mw` = None (extends to infinity).
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DeficitSegment {
    /// Deficit covered by this segment \[MW\]. None for the final unbounded segment.
    pub depth_mw: Option<f64>,
    /// Cost per `MWh` of deficit in this segment \[$/`MWh`\].
    pub cost_per_mwh: f64,
}

/// Electrical network node where energy balance is maintained.
///
/// Each bus carries a piecewise-linear deficit cost curve, whose final unbounded
/// segment ensures LP feasibility when demand cannot be met.
///
/// See Input System Entities SS1.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Bus {
    /// Unique bus identifier.
    pub id: EntityId,
    /// Human-readable bus name.
    pub name: String,
    /// Date the entity enters service (ISO 8601).
    pub operational_start_date: NaiveDate,
    /// Deficit cost segments, ordered by ascending cost.
    pub deficit_segments: Vec<DeficitSegment>,
    /// Cost per `MWh` for surplus generation absorption \[$/`MWh`\].
    pub excess_cost: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bus_construction() {
        let bus = Bus {
            id: EntityId::from(1),
            name: "Bus A".to_string(),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            deficit_segments: vec![
                DeficitSegment {
                    depth_mw: Some(100.0),
                    cost_per_mwh: 500.0,
                },
                DeficitSegment {
                    depth_mw: Some(200.0),
                    cost_per_mwh: 1000.0,
                },
                DeficitSegment {
                    depth_mw: None,
                    cost_per_mwh: 5000.0,
                },
            ],
            excess_cost: 1.0,
        };

        assert_eq!(bus.id, EntityId::from(1));
        assert_eq!(bus.name, "Bus A");
        assert_eq!(bus.deficit_segments.len(), 3);
        assert_eq!(bus.deficit_segments[2].depth_mw, None);
        assert_eq!(bus.excess_cost, 1.0);
    }

    #[test]
    fn test_deficit_segment_unbounded() {
        let segment = DeficitSegment {
            depth_mw: None,
            cost_per_mwh: 9999.0,
        };

        assert_eq!(segment.depth_mw, None);
        assert_eq!(segment.cost_per_mwh, 9999.0);
    }

    #[test]
    fn test_bus_equality() {
        let bus_a = Bus {
            id: EntityId::from(1),
            name: "Bus A".to_string(),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 500.0,
            }],
            excess_cost: 1.0,
        };

        let bus_b = bus_a.clone();
        assert_eq!(bus_a, bus_b);

        let bus_c = Bus {
            id: EntityId::from(2),
            ..bus_a.clone()
        };

        assert_ne!(bus_a, bus_c);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_bus_serde_roundtrip() {
        let bus = Bus {
            id: EntityId::from(1),
            name: "Main Bus".to_string(),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            deficit_segments: vec![
                DeficitSegment {
                    depth_mw: Some(100.0),
                    cost_per_mwh: 500.0,
                },
                DeficitSegment {
                    depth_mw: Some(200.0),
                    cost_per_mwh: 1000.0,
                },
                DeficitSegment {
                    depth_mw: None,
                    cost_per_mwh: 5000.0,
                },
            ],
            excess_cost: 1.5,
        };
        let json = serde_json::to_string(&bus).unwrap();
        let deserialized: Bus = serde_json::from_str(&json).unwrap();
        assert_eq!(bus, deserialized);
    }
}
