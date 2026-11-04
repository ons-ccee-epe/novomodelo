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

/// Duration of the interval over which each ramp level is held \[h\].
pub const THERMAL_RAMP_INTERVAL_HOURS: f64 = 0.5;

/// One generating machine of a thermal plant.
///
/// The unit injects at the plant's bus and shares the plant's entry and exit stages.
///
/// Ramps are trajectories of held levels \[MW\], not rates: each element is the
/// output held over one [`THERMAL_RAMP_INTERVAL_HOURS`] interval. Every level lies
/// in `[0, min_generation_mw]`, and an empty trajectory is an instantaneous transition.
///
/// The startup trajectory begins at switch-on, is non-decreasing and ends at
/// `min_generation_mw`; the unit is dispatchable in `[min_generation_mw, max_generation_mw]`
/// from the next interval. The shutdown trajectory begins at `min_generation_mw` and is
/// non-increasing; the unit is off at 0 MW from the interval after its last level.
///
/// Durations are physical hours on one on/off clock. Minimum up time runs from
/// switch-on, with both trajectories counted as on-time. Minimum down time runs from
/// the end of the shutdown trajectory to the next switch-on.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ThermalUnit {
    /// Identifier, unique within the owning plant (not globally).
    pub id: EntityId,
    /// Human-readable unit name.
    pub name: String,
    /// Minimum generation once dispatchable \[MW\].
    pub min_generation_mw: f64,
    /// Maximum generation \[MW\].
    pub max_generation_mw: f64,
    /// Minimum time on before a switch-off \[h\]; `0` means no minimum.
    pub min_up_time_hours: f64,
    /// Minimum time off before a switch-on \[h\]; `0` means no minimum.
    pub min_down_time_hours: f64,
    /// Startup trajectory of held levels \[MW\].
    pub startup_ramp_mw: Vec<f64>,
    /// Shutdown trajectory of held levels \[MW\].
    pub shutdown_ramp_mw: Vec<f64>,
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
    /// Generating units owned by the plant; empty means the plant is modeled only in aggregate.
    /// The plant's `min_generation_mw`/`max_generation_mw` floor and cap the units' summed output,
    /// trajectory output included; a study that does not model units is bounded by them alone.
    #[cfg_attr(feature = "serde", serde(default))]
    pub units: Vec<ThermalUnit>,
}

impl Thermal {
    /// Puts `units` in canonical id order so results do not depend on declaration order.
    pub fn sort_units(&mut self) {
        self.units.sort_by_key(|u| u.id.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{ThermalSpec, make_thermal};
    #[cfg(feature = "serde")]
    use crate::test_support::{ThermalUnitSpec, f64_bits_eq, make_thermal_unit};

    #[cfg(feature = "serde")]
    const THERMAL_UNIT_KEYS: [&str; 8] = [
        "id",
        "name",
        "min_generation_mw",
        "max_generation_mw",
        "min_up_time_hours",
        "min_down_time_hours",
        "startup_ramp_mw",
        "shutdown_ramp_mw",
    ];

    #[cfg(feature = "serde")]
    fn thermal_unit_fixture() -> crate::ThermalUnit {
        make_thermal_unit(ThermalUnitSpec {
            id: 3,
            name: "U3".to_string(),
            min_generation_mw: 20.0,
            max_generation_mw: 60.0,
            min_up_time_hours: 4.0,
            min_down_time_hours: 2.5,
            startup_ramp_mw: vec![10.0, 20.0],
            shutdown_ramp_mw: vec![20.0, 15.0, 5.0],
        })
    }

    #[cfg(feature = "serde")]
    fn assert_f64_fields_bit_exact(a: &crate::ThermalUnit, b: &crate::ThermalUnit) {
        let f64_fields = |unit: &crate::ThermalUnit| -> Vec<f64> {
            [
                unit.min_generation_mw,
                unit.max_generation_mw,
                unit.min_up_time_hours,
                unit.min_down_time_hours,
            ]
            .into_iter()
            .chain(unit.startup_ramp_mw.iter().copied())
            .chain(unit.shutdown_ramp_mw.iter().copied())
            .collect()
        };
        let (a, b) = (f64_fields(a), f64_fields(b));
        assert_eq!(a.len(), b.len());
        assert!(a.iter().zip(&b).all(|(x, y)| f64_bits_eq(*x, *y)));
    }

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
            units: Vec::new(),
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
            entry_stage_id: Some(1),
            exit_stage_id: Some(120),
            anticipated_config: Some(AnticipatedConfig::LeadStages(2)),
            ..make_thermal(ThermalSpec {
                id: 2,
                name: "Pecém I".to_string(),
                bus_id: 20,
                cost_per_mwh: 120.0,
                min_generation_mw: 100.0,
                max_generation_mw: 360.0,
                ..Default::default()
            })
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
            entry_stage_id: Some(1),
            exit_stage_id: Some(120),
            anticipated_config: Some(AnticipatedConfig::LeadStages(2)),
            ..make_thermal(ThermalSpec {
                id: 2,
                name: "Pecém I".to_string(),
                bus_id: 20,
                cost_per_mwh: 80.0,
                min_generation_mw: 100.0,
                max_generation_mw: 360.0,
                ..Default::default()
            })
        };
        let json = serde_json::to_string(&thermal).unwrap();
        let deserialized: Thermal = serde_json::from_str(&json).unwrap();
        assert_eq!(thermal, deserialized);
        // Externally-tagged (not `#[serde(untagged)]`) — postcard's Deserializer
        // does not implement `deserialize_any`, which `untagged` requires; this
        // shape keeps `Thermal` postcard-broadcast-safe (`novomodelo-io::broadcast`).
        assert!(json.contains("\"anticipated_config\":{\"LeadStages\":2}"));
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_thermal_serde_roundtrip_lead_time() {
        let thermal = Thermal {
            anticipated_config: Some(AnticipatedConfig::LeadTime(720.0)),
            ..make_thermal(ThermalSpec {
                id: 3,
                name: "T_lead_time".to_string(),
                bus_id: 20,
                cost_per_mwh: 80.0,
                min_generation_mw: 100.0,
                max_generation_mw: 360.0,
                ..Default::default()
            })
        };
        let json = serde_json::to_string(&thermal).unwrap();
        let deserialized: Thermal = serde_json::from_str(&json).unwrap();
        assert_eq!(thermal, deserialized);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn thermal_json_without_units_key_loads_with_no_units_unlike_hydro_unit_groups() {
        let thermal = make_thermal(ThermalSpec {
            id: 7,
            bus_id: 1,
            ..Default::default()
        });
        let mut document = serde_json::to_value(&thermal).unwrap();
        document.as_object_mut().unwrap().remove("units");
        let deserialized: Thermal = serde_json::from_value(document).unwrap();
        assert!(deserialized.units.is_empty());
        assert_eq!(deserialized, thermal);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn thermal_units_postcard_field_order_is_last() {
        let base = make_thermal(ThermalSpec {
            id: 9,
            name: "T9".to_string(),
            bus_id: 4,
            cost_per_mwh: 87.5,
            min_generation_mw: 25.5,
            max_generation_mw: 340.25,
            ..Default::default()
        });
        let mut with_units = base.clone();
        with_units.units = vec![make_thermal_unit(ThermalUnitSpec {
            id: 1,
            name: "T9-U1".to_string(),
            max_generation_mw: 340.25,
            ..Default::default()
        })];

        let prefix = postcard::to_allocvec(&base).unwrap();
        let full = postcard::to_allocvec(&with_units).unwrap();

        assert_eq!(
            prefix.last(),
            Some(&0x00),
            "an empty units list encodes as one trailing length byte"
        );
        assert!(full.len() > prefix.len());
        assert_eq!(
            &full[..prefix.len() - 1],
            &prefix[..prefix.len() - 1],
            "units must serialize last (append-last wire contract)"
        );
    }

    #[cfg(feature = "serde")]
    #[test]
    fn thermal_unit_json_roundtrip_preserves_every_field_bit_exact() {
        let unit = thermal_unit_fixture();
        let json = serde_json::to_string(&unit).unwrap();
        let deserialized: crate::ThermalUnit = serde_json::from_str(&json).unwrap();
        assert_eq!(unit, deserialized);
        assert_f64_fields_bit_exact(&unit, &deserialized);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn thermal_unit_postcard_roundtrip_preserves_every_field_bit_exact() {
        let unit = thermal_unit_fixture();
        let bytes = postcard::to_allocvec(&unit).unwrap();
        let deserialized: crate::ThermalUnit = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(unit, deserialized);
        assert_f64_fields_bit_exact(&unit, &deserialized);
        assert_eq!(bytes, postcard::to_allocvec(&deserialized).unwrap());
    }

    #[cfg(feature = "serde")]
    #[test]
    fn thermal_unit_document_missing_any_key_fails_to_deserialize() {
        let value = serde_json::to_value(thermal_unit_fixture()).unwrap();
        assert_eq!(
            value.as_object().unwrap().len(),
            THERMAL_UNIT_KEYS.len(),
            "a field was added or removed without updating THERMAL_UNIT_KEYS"
        );
        for key in THERMAL_UNIT_KEYS {
            let mut document = value.clone();
            document.as_object_mut().unwrap().remove(key);
            assert!(
                serde_json::from_value::<crate::ThermalUnit>(document).is_err(),
                "document without `{key}` must fail to deserialize"
            );
        }
    }

    #[test]
    fn thermal_ramp_interval_is_half_an_hour() {
        assert_eq!(crate::THERMAL_RAMP_INTERVAL_HOURS, 0.5);
    }
}
