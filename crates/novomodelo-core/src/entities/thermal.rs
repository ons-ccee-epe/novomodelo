//! Thermal plant entity — generation with MW bounds and cost.

use crate::EntityId;
use chrono::NaiveDate;

/// Anticipated dispatch configuration for thermal plants requiring advance commitment.
///
/// Carries exactly one lead mode — a stage-count lead (`LeadStages`) or a
/// physical lead time in hours (`LeadTime`, delivery-anchored, same clock as a
/// water arc's `travel_time_hours`) — mutually exclusive by construction.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum AnticipatedConfig {
    /// Stage-count lead; the calendar is never consulted. Must be >= 1.
    LeadStages(u32),
    /// Physical lead time in hours, delivery-anchored. Must be finite and > 0.0.
    LeadTime(f64),
}

impl AnticipatedConfig {
    /// The stage count in `LeadStages` mode; `None` for `LeadTime`.
    #[must_use]
    pub fn lead_stages(&self) -> Option<u32> {
        match self {
            Self::LeadStages(lead_stages) => Some(*lead_stages),
            Self::LeadTime(_) => None,
        }
    }
}

/// Thermal power plant with a scalar marginal cost.
///
/// See Input System Entities SS1.9.5.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Thermal {
    /// Unique thermal plant identifier.
    pub id: EntityId,
    /// Human-readable plant name.
    pub name: String,
    /// Date the entity enters service (ISO 8601).
    pub operational_start_date: NaiveDate,
    /// Bus to which this plant's generation is injected.
    pub bus_id: EntityId,
    /// Stage index when the plant enters service. None = always exists.
    pub entry_stage_id: Option<i32>,
    /// Stage index when the plant is decommissioned. None = never decommissioned.
    pub exit_stage_id: Option<i32>,
    /// Marginal cost of generation \[$/`MWh`\].
    pub cost_per_mwh: f64,
    /// Minimum electrical generation (minimum stable load) \[MW\].
    pub min_generation_mw: f64,
    /// Maximum electrical generation (installed capacity) \[MW\].
    pub max_generation_mw: f64,
    /// Anticipated dispatch configuration. None = no anticipation lag.
    pub anticipated_config: Option<AnticipatedConfig>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_thermal_construction() {
        let thermal = Thermal {
            id: EntityId::from(1),
            name: "Angra 1".to_string(),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            bus_id: EntityId::from(10),
            entry_stage_id: None,
            exit_stage_id: None,
            cost_per_mwh: 50.0,
            min_generation_mw: 0.0,
            max_generation_mw: 657.0,
            anticipated_config: None,
        };

        assert_eq!(thermal.id, EntityId::from(1));
        assert_eq!(thermal.name, "Angra 1");
        assert_eq!(thermal.bus_id, EntityId::from(10));
        assert_eq!(thermal.entry_stage_id, None);
        assert_eq!(thermal.exit_stage_id, None);
        assert_eq!(thermal.cost_per_mwh, 50.0);
        assert_eq!(thermal.min_generation_mw, 0.0);
        assert_eq!(thermal.max_generation_mw, 657.0);
        assert_eq!(thermal.anticipated_config, None);
    }

    #[test]
    fn test_thermal_with_anticipated() {
        let thermal = Thermal {
            id: EntityId::from(2),
            name: "Pecém I".to_string(),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            bus_id: EntityId::from(20),
            entry_stage_id: Some(1),
            exit_stage_id: Some(120),
            cost_per_mwh: 120.0,
            min_generation_mw: 100.0,
            max_generation_mw: 360.0,
            anticipated_config: Some(AnticipatedConfig::LeadStages(2)),
        };

        assert_eq!(
            thermal.anticipated_config,
            Some(AnticipatedConfig::LeadStages(2))
        );
        assert_eq!(thermal.entry_stage_id, Some(1));
        assert_eq!(thermal.exit_stage_id, Some(120));
    }

    #[test]
    fn test_anticipated_config_lead_stages_accessor_returns_stage_count() {
        let config = AnticipatedConfig::LeadStages(5);
        assert_eq!(config.lead_stages(), Some(5));
    }

    #[test]
    fn test_anticipated_config_lead_stages_accessor_none_for_lead_time() {
        let config = AnticipatedConfig::LeadTime(720.0);
        assert_eq!(config.lead_stages(), None);
    }

    #[test]
    fn test_anticipated_config_is_copy() {
        let config = AnticipatedConfig::LeadStages(3);
        let copied = config;
        assert_eq!(config, copied);
    }

    #[test]
    fn test_anticipated_config_partial_eq() {
        assert_eq!(
            AnticipatedConfig::LeadStages(2),
            AnticipatedConfig::LeadStages(2)
        );
        assert_ne!(
            AnticipatedConfig::LeadStages(2),
            AnticipatedConfig::LeadStages(3)
        );
        assert_ne!(
            AnticipatedConfig::LeadStages(2),
            AnticipatedConfig::LeadTime(2.0)
        );
        assert_eq!(
            AnticipatedConfig::LeadTime(720.0),
            AnticipatedConfig::LeadTime(720.0)
        );
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_thermal_serde_roundtrip() {
        let thermal = Thermal {
            id: EntityId::from(2),
            name: "Pecém I".to_string(),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            bus_id: EntityId::from(20),
            entry_stage_id: Some(1),
            exit_stage_id: Some(120),
            cost_per_mwh: 80.0,
            min_generation_mw: 100.0,
            max_generation_mw: 360.0,
            anticipated_config: Some(AnticipatedConfig::LeadStages(2)),
        };
        let json = serde_json::to_string(&thermal).unwrap();
        let deserialized: Thermal = serde_json::from_str(&json).unwrap();
        assert_eq!(thermal, deserialized);
        // Externally-tagged (not `#[serde(untagged)]`) — postcard's Deserializer
        // does not implement `deserialize_any`, which `untagged` requires; this
        // shape keeps `Thermal` postcard-broadcast-safe (`cobre-io::broadcast`).
        assert!(json.contains("\"anticipated_config\":{\"LeadStages\":2}"));
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_thermal_serde_roundtrip_lead_time() {
        let thermal = Thermal {
            id: EntityId::from(3),
            name: "T_lead_time".to_string(),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            bus_id: EntityId::from(20),
            entry_stage_id: None,
            exit_stage_id: None,
            cost_per_mwh: 80.0,
            min_generation_mw: 100.0,
            max_generation_mw: 360.0,
            anticipated_config: Some(AnticipatedConfig::LeadTime(720.0)),
        };
        let json = serde_json::to_string(&thermal).unwrap();
        let deserialized: Thermal = serde_json::from_str(&json).unwrap();
        assert_eq!(thermal, deserialized);
    }
}
