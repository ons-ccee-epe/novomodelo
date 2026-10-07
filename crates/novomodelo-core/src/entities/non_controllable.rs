//! Non-controllable generation source entity — intermittent wind/solar.

use crate::EntityId;
use chrono::NaiveDate;

/// Intermittent generation source that cannot be dispatched.
///
/// Available generation is `max_generation_mw × α × factor`, where `α` is the
/// per-(stage, scenario) availability ratio and `factor` the per-(stage, block)
/// shape factor. `allow_curtailment` selects between the curtailable and must-run
/// dispatch regimes.
///
/// See Input System Entities SS1.9.8.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct NonControllableSource {
    /// Unique source identifier.
    pub id: EntityId,
    /// Human-readable source name.
    pub name: String,
    /// Date the entity enters service (ISO 8601).
    pub operational_start_date: NaiveDate,
    /// Bus to which this source's generation is injected.
    pub bus_id: EntityId,
    /// Stage index when the source enters service. None = always exists.
    pub entry_stage_id: Option<i32>,
    /// Stage index when the source is decommissioned. None = never decommissioned.
    pub exit_stage_id: Option<i32>,
    /// Maximum generation (installed capacity) \[MW\].
    pub max_generation_mw: f64,
    /// Whether the LP may curtail this source.
    ///
    /// `false` is the **must-run** regime: dispatch is pinned to realized
    /// availability (`col_lower = col_upper`), not merely bounded above by it.
    pub allow_curtailment: bool,
    /// Resolved cost per `MWh` of curtailed generation \[$/`MWh`\]. Unused when
    /// `allow_curtailment == false`.
    pub curtailment_cost: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_non_controllable_construction() {
        let source = NonControllableSource {
            id: EntityId::from(1),
            name: "Eólica Caetité".to_string(),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            bus_id: EntityId::from(7),
            entry_stage_id: None,
            exit_stage_id: None,
            max_generation_mw: 300.0,
            allow_curtailment: true,
            curtailment_cost: 0.01,
        };

        assert_eq!(source.id, EntityId::from(1));
        assert_eq!(source.name, "Eólica Caetité");
        assert_eq!(source.bus_id, EntityId::from(7));
        assert_eq!(source.entry_stage_id, None);
        assert_eq!(source.exit_stage_id, None);
        assert!(source.allow_curtailment);
        assert!((source.max_generation_mw - 300.0).abs() < f64::EPSILON);
        assert!((source.curtailment_cost - 0.01).abs() < f64::EPSILON);
    }

    #[test]
    fn test_non_controllable_curtailment_cost() {
        let source = NonControllableSource {
            id: EntityId::from(2),
            name: "Solar Pirapora".to_string(),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            bus_id: EntityId::from(3),
            entry_stage_id: Some(12),
            exit_stage_id: None,
            max_generation_mw: 400.0,
            allow_curtailment: true,
            curtailment_cost: 5.0,
        };

        assert!((source.curtailment_cost - 5.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_non_controllable_must_run() {
        let source = NonControllableSource {
            id: EntityId::from(3),
            name: "PCH Aggregate NE".to_string(),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            bus_id: EntityId::from(0),
            entry_stage_id: None,
            exit_stage_id: None,
            max_generation_mw: 120.0,
            allow_curtailment: false,
            curtailment_cost: 0.0344,
        };

        assert!(!source.allow_curtailment);
    }
}
