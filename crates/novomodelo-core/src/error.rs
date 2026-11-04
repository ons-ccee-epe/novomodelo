//! Error types produced during `System` construction and validation.

use core::fmt;

use crate::{EntityId, ThermalUnitTransition};

/// Errors produced during System construction and validation.
///
/// Each variant carries enough context to pinpoint the invalid input.
///
/// # Examples
///
/// ```
/// use novomodelo_core::{EntityId, ValidationError};
///
/// let err = ValidationError::DuplicateId {
///     entity_type: "Bus",
///     id: EntityId(1),
/// };
/// assert!(err.to_string().contains("Bus"));
/// ```
#[derive(Debug, Clone, PartialEq)]
pub enum ValidationError {
    /// A cross-reference field (e.g., `bus_id`, `downstream_id`) refers to
    /// an entity ID that does not exist in the system.
    InvalidReference {
        /// Entity type holding the dangling reference.
        source_entity_type: &'static str,
        /// ID of the entity holding the dangling reference.
        source_id: EntityId,
        /// Field holding the dangling reference.
        field_name: &'static str,
        /// Referenced ID that does not exist.
        referenced_id: EntityId,
        /// Entity type the reference was expected to resolve to.
        expected_type: &'static str,
    },
    /// Duplicate entity ID within a single entity collection.
    DuplicateId {
        /// Entity type carrying the duplicate.
        entity_type: &'static str,
        /// The duplicated ID.
        id: EntityId,
    },
    /// The hydro cascade contains a cycle.
    CascadeCycle {
        /// IDs of hydros forming the cycle.
        cycle_ids: Vec<EntityId>,
    },
    /// A hydro's filling configuration is invalid.
    InvalidFillingConfig {
        /// Hydro with the invalid configuration.
        hydro_id: EntityId,
        /// Why the configuration is invalid.
        reason: String,
    },
    /// A hydro declares no unit groups; at least one is required.
    MissingUnitGroups {
        /// Hydro declaring no unit groups.
        hydro_id: EntityId,
    },
    /// A scenario model table is not in its documented canonical order.
    UnsortedModelTable {
        /// Table that is out of order.
        table: &'static str,
        /// Position of the first element that is less than its predecessor;
        /// equal adjacent keys are allowed.
        position: usize,
    },
    /// A thermal unit breaks a [`ThermalUnitViolation`] rule.
    InvalidThermalUnit {
        /// Plant owning the unit.
        thermal_id: EntityId,
        /// Unit that breaks the rule.
        unit_id: EntityId,
        /// Rule the unit breaks.
        violation: ThermalUnitViolation,
    },
    /// A thermal plant's units sum to less than its maximum generation.
    InsufficientThermalUnitCapacity {
        /// Plant whose units fall short.
        thermal_id: EntityId,
        /// Compensated sum of the units' maximum generation \[MW\].
        units_max_mw: f64,
        /// Declared plant maximum generation \[MW\].
        plant_max_mw: f64,
    },
    /// A thermal unit initial state breaks a [`ThermalUnitStateViolation`] rule.
    InvalidThermalUnitInitialState {
        /// Plant of the record.
        thermal_id: EntityId,
        /// Unit of the record.
        unit_id: EntityId,
        /// Rule the record breaks.
        violation: ThermalUnitStateViolation,
    },
    /// A hydro unit group breaks a [`HydroUnitGroupViolation`] rule.
    InvalidHydroUnitGroup {
        /// Plant owning the group.
        hydro_id: EntityId,
        /// Group that breaks the rule.
        group_id: EntityId,
        /// Rule the group breaks.
        violation: HydroUnitGroupViolation,
    },
    /// A hydro unit breaks a [`HydroUnitViolation`] rule.
    InvalidHydroUnit {
        /// Plant owning the unit.
        hydro_id: EntityId,
        /// Group owning the unit.
        group_id: EntityId,
        /// Unit that breaks the rule.
        unit_id: EntityId,
        /// Rule the unit breaks.
        violation: HydroUnitViolation,
    },
    /// A hydro unit initial state breaks a [`HydroUnitStateViolation`] rule.
    InvalidHydroUnitInitialState {
        /// Plant of the record.
        hydro_id: EntityId,
        /// Group of the record.
        group_id: EntityId,
        /// Unit of the record.
        unit_id: EntityId,
        /// Rule the record breaks.
        violation: HydroUnitStateViolation,
    },
}

/// Rule a thermal unit breaks, carried by [`ValidationError::InvalidThermalUnit`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ThermalUnitViolation {
    /// Another unit of the same plant has this id.
    DuplicateId,
    /// Generation bounds are not finite with `0 <= min <= max`.
    InvalidGenerationBounds {
        /// Declared minimum generation \[MW\].
        min_generation_mw: f64,
        /// Declared maximum generation \[MW\].
        max_generation_mw: f64,
    },
    /// Minimum up time is negative or not finite.
    InvalidMinUpTime {
        /// Declared minimum up time \[h\].
        min_up_time_hours: f64,
    },
    /// Minimum down time is negative or not finite.
    InvalidMinDownTime {
        /// Declared minimum down time \[h\].
        min_down_time_hours: f64,
    },
    /// A ramp level lies outside `0..=min_generation_mw`.
    RampLevelOutOfRange {
        /// Trajectory holding the level.
        transition: ThermalUnitTransition,
        /// Position of the level within the trajectory.
        index: usize,
        /// Declared level \[MW\].
        level_mw: f64,
    },
    /// A ramp is decreasing at startup or increasing at shutdown.
    RampNotMonotone {
        /// Trajectory that is not monotone.
        transition: ThermalUnitTransition,
        /// Position of the first level that breaks the order.
        index: usize,
    },
    /// The last startup level or the first shutdown level is not `min_generation_mw`.
    RampEndpointNotAtMinimum {
        /// Trajectory whose endpoint is wrong.
        transition: ThermalUnitTransition,
        /// Declared endpoint level \[MW\].
        level_mw: f64,
    },
}

/// Rule a thermal unit initial state breaks, carried by
/// [`ValidationError::InvalidThermalUnitInitialState`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ThermalUnitStateViolation {
    /// Another record has the same `(thermal_id, unit_id)` key.
    DuplicateRecord,
    /// No thermal plant has the record's `thermal_id`.
    UnknownThermal,
    /// The plant declares no unit with the record's `unit_id`.
    UnknownUnit,
    /// Time in state is negative or not finite.
    InvalidTimeInState {
        /// Declared time in the current state \[h\].
        time_in_state_hours: f64,
    },
    /// Previous generation is negative or not finite.
    InvalidPreviousGeneration {
        /// Declared previous generation \[MW\].
        previous_generation_mw: f64,
    },
    /// Status `Off` with a non-zero previous generation.
    OffWithGeneration {
        /// Declared previous generation \[MW\].
        previous_generation_mw: f64,
    },
    /// Status `On` with a previous generation outside the unit's generation bounds.
    OnOutsideBounds {
        /// Declared previous generation \[MW\].
        previous_generation_mw: f64,
        /// Unit minimum generation \[MW\].
        min_generation_mw: f64,
        /// Unit maximum generation \[MW\].
        max_generation_mw: f64,
    },
    /// Transient status whose previous generation equals no sub-minimum level of its ramp.
    GenerationNotOnTrajectory {
        /// Trajectory the status follows.
        transition: ThermalUnitTransition,
        /// Declared previous generation \[MW\].
        previous_generation_mw: f64,
    },
}

/// Rule a hydro unit group breaks, carried by [`ValidationError::InvalidHydroUnitGroup`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HydroUnitGroupViolation {
    /// Another group of the same plant has this id.
    DuplicateId,
    /// The units' maximum generation sums to less than the group maximum.
    InsufficientGenerationCapacity {
        /// Compensated sum of the units' maximum generation \[MW\].
        units_max_mw: f64,
        /// Declared group maximum generation \[MW\].
        group_max_mw: f64,
    },
    /// The units' maximum turbined flow sums to less than the group maximum.
    InsufficientTurbinedCapacity {
        /// Compensated sum of the units' maximum turbined flow \[m³/s\].
        units_max_m3s: f64,
        /// Declared group maximum turbined flow \[m³/s\].
        group_max_m3s: f64,
    },
}

/// Rule a hydro unit breaks, carried by [`ValidationError::InvalidHydroUnit`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HydroUnitViolation {
    /// Another unit of the same group has this id.
    DuplicateId,
    /// Generation bounds are not finite with `0 <= min <= max`.
    InvalidGenerationBounds {
        /// Declared minimum generation \[MW\].
        min_generation_mw: f64,
        /// Declared maximum generation \[MW\].
        max_generation_mw: f64,
    },
    /// Turbined-flow bounds are not finite with `0 <= min <= max`.
    InvalidTurbinedBounds {
        /// Declared minimum turbined flow \[m³/s\].
        min_turbined_m3s: f64,
        /// Declared maximum turbined flow \[m³/s\].
        max_turbined_m3s: f64,
    },
    /// Minimum up time is negative or not finite.
    InvalidMinUpTime {
        /// Declared minimum up time \[h\].
        min_up_time_hours: f64,
    },
    /// Minimum down time is negative or not finite.
    InvalidMinDownTime {
        /// Declared minimum down time \[h\].
        min_down_time_hours: f64,
    },
}

/// Rule a hydro unit initial state breaks, carried by
/// [`ValidationError::InvalidHydroUnitInitialState`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HydroUnitStateViolation {
    /// Another record has the same `(hydro_id, group_id, unit_id)` key.
    DuplicateRecord,
    /// No hydro plant has the record's `hydro_id`.
    UnknownHydro,
    /// The plant declares no unit group with the record's `group_id`.
    UnknownGroup,
    /// The group declares no unit with the record's `unit_id`.
    UnknownUnit,
    /// Time in state is negative or not finite.
    InvalidTimeInState {
        /// Declared time in the current state \[h\].
        time_in_state_hours: f64,
    },
    /// Previous generation is negative or not finite.
    InvalidPreviousGeneration {
        /// Declared previous generation \[MW\].
        previous_generation_mw: f64,
    },
    /// Previous turbined flow is negative or not finite.
    InvalidPreviousTurbined {
        /// Declared previous turbined flow \[m³/s\].
        previous_turbined_m3s: f64,
    },
    /// Status `Off` with a non-zero previous generation.
    OffWithGeneration {
        /// Declared previous generation \[MW\].
        previous_generation_mw: f64,
    },
    /// Status `Off` with a non-zero previous turbined flow.
    OffWithTurbinedFlow {
        /// Declared previous turbined flow \[m³/s\].
        previous_turbined_m3s: f64,
    },
    /// Status `On` with a previous generation outside the unit's generation bounds.
    OnGenerationOutsideBounds {
        /// Declared previous generation \[MW\].
        previous_generation_mw: f64,
        /// Unit minimum generation \[MW\].
        min_generation_mw: f64,
        /// Unit maximum generation \[MW\].
        max_generation_mw: f64,
    },
    /// Status `On` with a previous turbined flow outside the unit's turbined-flow bounds.
    OnTurbinedOutsideBounds {
        /// Declared previous turbined flow \[m³/s\].
        previous_turbined_m3s: f64,
        /// Unit minimum turbined flow \[m³/s\].
        min_turbined_m3s: f64,
        /// Unit maximum turbined flow \[m³/s\].
        max_turbined_m3s: f64,
    },
}

fn ramp_name(transition: ThermalUnitTransition) -> &'static str {
    match transition {
        ThermalUnitTransition::StartingUp => "startup",
        ThermalUnitTransition::ShuttingDown => "shutdown",
    }
}

impl fmt::Display for ValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidReference {
                source_entity_type,
                source_id,
                field_name,
                referenced_id,
                expected_type,
            } => write!(
                f,
                "{source_entity_type} with id {source_id} has invalid cross-reference \
                 in field '{field_name}': referenced {expected_type} id {referenced_id} does not exist"
            ),
            Self::DuplicateId { entity_type, id } => write!(f, "duplicate {entity_type} id: {id}"),
            Self::CascadeCycle { cycle_ids } => {
                let ids = cycle_ids
                    .iter()
                    .map(EntityId::to_string)
                    .collect::<Vec<_>>()
                    .join(", ");
                write!(f, "hydro cascade contains a cycle: [{ids}]")
            }
            Self::InvalidFillingConfig { hydro_id, reason } => write!(
                f,
                "hydro {hydro_id} has invalid filling configuration: {reason}"
            ),
            Self::MissingUnitGroups { hydro_id } => write!(
                f,
                "hydro {hydro_id} declares no unit groups: at least one unit group is required"
            ),
            Self::UnsortedModelTable { table, position } => write!(
                f,
                "{table} is not sorted by its documented (id, stage_id) key: row {position} is out of order"
            ),
            Self::InvalidThermalUnit {
                thermal_id,
                unit_id,
                violation,
            } => write!(
                f,
                "thermal {thermal_id} unit {unit_id} is invalid: {violation}"
            ),
            Self::InsufficientThermalUnitCapacity {
                thermal_id,
                units_max_mw,
                plant_max_mw,
            } => write!(
                f,
                "thermal {thermal_id} units sum to {units_max_mw} MW, \
                 below the plant maximum of {plant_max_mw} MW"
            ),
            Self::InvalidThermalUnitInitialState {
                thermal_id,
                unit_id,
                violation,
            } => write!(
                f,
                "initial state of thermal {thermal_id} unit {unit_id} is invalid: {violation}"
            ),
            Self::InvalidHydroUnitGroup {
                hydro_id,
                group_id,
                violation,
            } => write!(
                f,
                "hydro {hydro_id} unit group {group_id} is invalid: {violation}"
            ),
            Self::InvalidHydroUnit {
                hydro_id,
                group_id,
                unit_id,
                violation,
            } => write!(
                f,
                "hydro {hydro_id} unit group {group_id} unit {unit_id} is invalid: {violation}"
            ),
            Self::InvalidHydroUnitInitialState {
                hydro_id,
                group_id,
                unit_id,
                violation,
            } => write!(
                f,
                "initial state of hydro {hydro_id} unit group {group_id} unit {unit_id} \
                 is invalid: {violation}"
            ),
        }
    }
}

impl fmt::Display for ThermalUnitViolation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateId => write!(f, "duplicate unit id within the plant"),
            Self::InvalidGenerationBounds {
                min_generation_mw,
                max_generation_mw,
            } => write!(
                f,
                "generation bounds must be finite with 0 <= min <= max, \
                 got min {min_generation_mw} MW and max {max_generation_mw} MW"
            ),
            Self::InvalidMinUpTime { min_up_time_hours } => write!(
                f,
                "min_up_time_hours must be finite and >= 0, got {min_up_time_hours}"
            ),
            Self::InvalidMinDownTime {
                min_down_time_hours,
            } => write!(
                f,
                "min_down_time_hours must be finite and >= 0, got {min_down_time_hours}"
            ),
            Self::RampLevelOutOfRange {
                transition,
                index,
                level_mw,
            } => write!(
                f,
                "{} ramp level {index} ({level_mw} MW) must lie in [0, min_generation_mw]",
                ramp_name(*transition)
            ),
            Self::RampNotMonotone { transition, index } => write!(
                f,
                "{} ramp is not monotone at level {index}",
                ramp_name(*transition)
            ),
            Self::RampEndpointNotAtMinimum {
                transition,
                level_mw,
            } => write!(
                f,
                "{} ramp endpoint ({level_mw} MW) must equal min_generation_mw",
                ramp_name(*transition)
            ),
        }
    }
}

impl fmt::Display for ThermalUnitStateViolation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateRecord => write!(f, "duplicate record for this unit"),
            Self::UnknownThermal => write!(f, "no thermal plant has this id"),
            Self::UnknownUnit => write!(f, "the plant declares no unit with this id"),
            Self::InvalidTimeInState {
                time_in_state_hours,
            } => write!(
                f,
                "time_in_state_hours must be finite and >= 0, got {time_in_state_hours}"
            ),
            Self::InvalidPreviousGeneration {
                previous_generation_mw,
            } => write!(
                f,
                "previous_generation_mw must be finite and >= 0, got {previous_generation_mw}"
            ),
            Self::OffWithGeneration {
                previous_generation_mw,
            } => write!(
                f,
                "status Off requires previous_generation_mw == 0, got {previous_generation_mw} MW"
            ),
            Self::OnOutsideBounds {
                previous_generation_mw,
                min_generation_mw,
                max_generation_mw,
            } => write!(
                f,
                "status On requires previous_generation_mw in \
                 [{min_generation_mw}, {max_generation_mw}] MW, got {previous_generation_mw} MW"
            ),
            Self::GenerationNotOnTrajectory {
                transition,
                previous_generation_mw,
            } => write!(
                f,
                "previous_generation_mw {previous_generation_mw} MW matches no {} ramp level \
                 below min_generation_mw",
                ramp_name(*transition)
            ),
        }
    }
}

impl fmt::Display for HydroUnitGroupViolation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateId => write!(f, "duplicate unit group id within the plant"),
            Self::InsufficientGenerationCapacity {
                units_max_mw,
                group_max_mw,
            } => write!(
                f,
                "units sum to {units_max_mw} MW, \
                 below the group maximum generation of {group_max_mw} MW"
            ),
            Self::InsufficientTurbinedCapacity {
                units_max_m3s,
                group_max_m3s,
            } => write!(
                f,
                "units sum to {units_max_m3s} m³/s, \
                 below the group maximum turbined flow of {group_max_m3s} m³/s"
            ),
        }
    }
}

impl fmt::Display for HydroUnitViolation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateId => write!(f, "duplicate unit id within the unit group"),
            Self::InvalidGenerationBounds {
                min_generation_mw,
                max_generation_mw,
            } => write!(
                f,
                "generation bounds must be finite with 0 <= min <= max, \
                 got min {min_generation_mw} MW and max {max_generation_mw} MW"
            ),
            Self::InvalidTurbinedBounds {
                min_turbined_m3s,
                max_turbined_m3s,
            } => write!(
                f,
                "turbined-flow bounds must be finite with 0 <= min <= max, \
                 got min {min_turbined_m3s} m³/s and max {max_turbined_m3s} m³/s"
            ),
            Self::InvalidMinUpTime { min_up_time_hours } => write!(
                f,
                "min_up_time_hours must be finite and >= 0, got {min_up_time_hours}"
            ),
            Self::InvalidMinDownTime {
                min_down_time_hours,
            } => write!(
                f,
                "min_down_time_hours must be finite and >= 0, got {min_down_time_hours}"
            ),
        }
    }
}

impl fmt::Display for HydroUnitStateViolation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateRecord => write!(f, "duplicate record for this unit"),
            Self::UnknownHydro => write!(f, "no hydro plant has this id"),
            Self::UnknownGroup => write!(f, "the plant declares no unit group with this id"),
            Self::UnknownUnit => write!(f, "the unit group declares no unit with this id"),
            Self::InvalidTimeInState {
                time_in_state_hours,
            } => write!(
                f,
                "time_in_state_hours must be finite and >= 0, got {time_in_state_hours}"
            ),
            Self::InvalidPreviousGeneration {
                previous_generation_mw,
            } => write!(
                f,
                "previous_generation_mw must be finite and >= 0, got {previous_generation_mw}"
            ),
            Self::InvalidPreviousTurbined {
                previous_turbined_m3s,
            } => write!(
                f,
                "previous_turbined_m3s must be finite and >= 0, got {previous_turbined_m3s}"
            ),
            Self::OffWithGeneration {
                previous_generation_mw,
            } => write!(
                f,
                "status Off requires previous_generation_mw == 0, got {previous_generation_mw} MW"
            ),
            Self::OffWithTurbinedFlow {
                previous_turbined_m3s,
            } => write!(
                f,
                "status Off requires previous_turbined_m3s == 0, got {previous_turbined_m3s} m³/s"
            ),
            Self::OnGenerationOutsideBounds {
                previous_generation_mw,
                min_generation_mw,
                max_generation_mw,
            } => write!(
                f,
                "status On requires previous_generation_mw in \
                 [{min_generation_mw}, {max_generation_mw}] MW, got {previous_generation_mw} MW"
            ),
            Self::OnTurbinedOutsideBounds {
                previous_turbined_m3s,
                min_turbined_m3s,
                max_turbined_m3s,
            } => write!(
                f,
                "status On requires previous_turbined_m3s in \
                 [{min_turbined_m3s}, {max_turbined_m3s}] m³/s, got {previous_turbined_m3s} m³/s"
            ),
        }
    }
}

impl std::error::Error for ValidationError {}

#[cfg(test)]
mod tests {
    use super::{
        HydroUnitGroupViolation, HydroUnitStateViolation, HydroUnitViolation,
        ThermalUnitStateViolation, ThermalUnitViolation, ValidationError,
    };
    use crate::{EntityId, ThermalUnitTransition};

    #[test]
    fn test_display_invalid_reference() {
        let err = ValidationError::InvalidReference {
            source_entity_type: "Hydro",
            source_id: EntityId(3),
            field_name: "bus_id",
            referenced_id: EntityId(99),
            expected_type: "Bus",
        };
        let msg = err.to_string();
        assert!(msg.contains("Hydro"), "missing source entity type: {msg}");
        assert!(msg.contains("bus_id"), "missing field name: {msg}");
        assert!(msg.contains("99"), "missing referenced id: {msg}");
    }

    #[test]
    fn test_display_duplicate_id() {
        let err = ValidationError::DuplicateId {
            entity_type: "Thermal",
            id: EntityId(5),
        };
        let msg = err.to_string();
        assert!(msg.contains("Thermal"), "missing entity type: {msg}");
        assert!(msg.contains('5'), "missing id: {msg}");
    }

    #[test]
    fn test_display_cascade_cycle() {
        let err = ValidationError::CascadeCycle {
            cycle_ids: vec![EntityId(1), EntityId(2), EntityId(3)],
        };
        let msg = err.to_string();
        assert!(msg.contains('1'), "missing id 1: {msg}");
        assert!(msg.contains('2'), "missing id 2: {msg}");
        assert!(msg.contains('3'), "missing id 3: {msg}");
    }

    #[test]
    fn test_display_missing_unit_groups() {
        let err = ValidationError::MissingUnitGroups {
            hydro_id: EntityId(4),
        };
        let msg = err.to_string();
        assert!(msg.contains('4'), "missing hydro id: {msg}");
        assert!(msg.contains("unit group"), "missing mention: {msg}");
    }

    #[test]
    fn test_display_invalid_thermal_unit() {
        let duplicate = ValidationError::InvalidThermalUnit {
            thermal_id: EntityId(3),
            unit_id: EntityId(7),
            violation: ThermalUnitViolation::DuplicateId,
        };
        let msg = duplicate.to_string();
        assert!(msg.contains('3'), "missing thermal id: {msg}");
        assert!(msg.contains('7'), "missing unit id: {msg}");
        assert!(msg.contains("duplicate unit id"), "missing mention: {msg}");

        let bounds = ThermalUnitViolation::InvalidGenerationBounds {
            min_generation_mw: 50.0,
            max_generation_mw: 40.0,
        };
        let msg = bounds.to_string();
        assert!(msg.contains("50"), "missing min: {msg}");
        assert!(msg.contains("40"), "missing max: {msg}");
    }

    #[test]
    fn test_display_thermal_unit_ramp_violations() {
        let range = ThermalUnitViolation::RampLevelOutOfRange {
            transition: ThermalUnitTransition::StartingUp,
            index: 2,
            level_mw: 45.5,
        };
        let msg = range.to_string();
        assert!(msg.contains("startup"), "missing ramp name: {msg}");
        assert!(msg.contains("level 2"), "missing index: {msg}");
        assert!(msg.contains("45.5"), "missing level: {msg}");

        let order = ThermalUnitViolation::RampNotMonotone {
            transition: ThermalUnitTransition::ShuttingDown,
            index: 1,
        };
        let msg = order.to_string();
        assert!(msg.contains("shutdown"), "missing ramp name: {msg}");

        let endpoint = ThermalUnitViolation::RampEndpointNotAtMinimum {
            transition: ThermalUnitTransition::StartingUp,
            level_mw: 15.5,
        };
        let msg = endpoint.to_string();
        assert!(msg.contains("15.5"), "missing level: {msg}");
    }

    #[test]
    fn test_display_insufficient_thermal_unit_capacity() {
        let err = ValidationError::InsufficientThermalUnitCapacity {
            thermal_id: EntityId(4),
            units_max_mw: 99.0,
            plant_max_mw: 100.0,
        };
        let msg = err.to_string();
        assert!(msg.contains('4'), "missing thermal id: {msg}");
        assert!(msg.contains("99"), "missing units sum: {msg}");
        assert!(msg.contains("100"), "missing plant maximum: {msg}");
    }

    #[test]
    fn test_display_invalid_thermal_unit_initial_state() {
        let err = ValidationError::InvalidThermalUnitInitialState {
            thermal_id: EntityId(6),
            unit_id: EntityId(8),
            violation: ThermalUnitStateViolation::UnknownUnit,
        };
        let msg = err.to_string();
        assert!(msg.contains('6'), "missing thermal id: {msg}");
        assert!(msg.contains('8'), "missing unit id: {msg}");
        assert!(msg.contains("no unit"), "missing mention: {msg}");
    }

    #[test]
    fn test_display_thermal_unit_state_consistency_violations() {
        let off = ThermalUnitStateViolation::OffWithGeneration {
            previous_generation_mw: 5.0,
        };
        let msg = off.to_string();
        assert!(msg.contains("Off"), "missing status: {msg}");
        assert!(msg.contains('5'), "missing generation: {msg}");

        let on = ThermalUnitStateViolation::OnOutsideBounds {
            previous_generation_mw: 10.0,
            min_generation_mw: 20.0,
            max_generation_mw: 60.0,
        };
        let msg = on.to_string();
        assert!(msg.contains("[20, 60]"), "missing bounds: {msg}");
        assert!(msg.contains("10"), "missing generation: {msg}");

        let trajectory = ThermalUnitStateViolation::GenerationNotOnTrajectory {
            transition: ThermalUnitTransition::ShuttingDown,
            previous_generation_mw: 7.0,
        };
        let msg = trajectory.to_string();
        assert!(msg.contains("shutdown"), "missing ramp name: {msg}");
        assert!(msg.contains('7'), "missing generation: {msg}");
    }

    #[test]
    fn test_display_invalid_hydro_unit_group() {
        let error = |violation| ValidationError::InvalidHydroUnitGroup {
            hydro_id: EntityId(3),
            group_id: EntityId(7),
            violation,
        };
        let cases: [(ValidationError, &[&str]); 3] = [
            (
                error(HydroUnitGroupViolation::DuplicateId),
                &["hydro 3 unit group 7 is invalid", "duplicate unit group id"],
            ),
            (
                error(HydroUnitGroupViolation::InsufficientGenerationCapacity {
                    units_max_mw: 99.5,
                    group_max_mw: 100.25,
                }),
                &["hydro 3 unit group 7", "99.5 MW", "100.25 MW"],
            ),
            (
                error(HydroUnitGroupViolation::InsufficientTurbinedCapacity {
                    units_max_m3s: 49.5,
                    group_max_m3s: 50.75,
                }),
                &["hydro 3 unit group 7", "49.5 m³/s", "50.75 m³/s"],
            ),
        ];
        for (err, needles) in cases {
            let msg = err.to_string();
            for needle in needles {
                assert!(msg.contains(needle), "missing {needle:?}: {msg}");
            }
        }
    }

    #[test]
    fn test_display_invalid_hydro_unit() {
        let error = |violation| ValidationError::InvalidHydroUnit {
            hydro_id: EntityId(3),
            group_id: EntityId(7),
            unit_id: EntityId(9),
            violation,
        };
        let cases: [(ValidationError, &[&str]); 5] = [
            (
                error(HydroUnitViolation::DuplicateId),
                &[
                    "hydro 3 unit group 7 unit 9 is invalid",
                    "duplicate unit id within the unit group",
                ],
            ),
            (
                error(HydroUnitViolation::InvalidGenerationBounds {
                    min_generation_mw: 50.5,
                    max_generation_mw: 40.5,
                }),
                &["min 50.5 MW", "max 40.5 MW"],
            ),
            (
                error(HydroUnitViolation::InvalidTurbinedBounds {
                    min_turbined_m3s: 1.5,
                    max_turbined_m3s: 0.5,
                }),
                &["min 1.5 m³/s", "max 0.5 m³/s"],
            ),
            (
                error(HydroUnitViolation::InvalidMinUpTime {
                    min_up_time_hours: -2.5,
                }),
                &["min_up_time_hours", "-2.5"],
            ),
            (
                error(HydroUnitViolation::InvalidMinDownTime {
                    min_down_time_hours: -3.5,
                }),
                &["min_down_time_hours", "-3.5"],
            ),
        ];
        for (err, needles) in cases {
            let msg = err.to_string();
            for needle in needles {
                assert!(msg.contains(needle), "missing {needle:?}: {msg}");
            }
        }
    }

    #[test]
    fn test_display_invalid_hydro_unit_initial_state() {
        let error = |violation| ValidationError::InvalidHydroUnitInitialState {
            hydro_id: EntityId(4),
            group_id: EntityId(6),
            unit_id: EntityId(8),
            violation,
        };
        let cases: [(ValidationError, &[&str]); 11] = [
            (
                error(HydroUnitStateViolation::DuplicateRecord),
                &[
                    "initial state of hydro 4 unit group 6 unit 8 is invalid",
                    "duplicate record",
                ],
            ),
            (
                error(HydroUnitStateViolation::UnknownHydro),
                &["no hydro plant"],
            ),
            (
                error(HydroUnitStateViolation::UnknownGroup),
                &["no unit group"],
            ),
            (
                error(HydroUnitStateViolation::UnknownUnit),
                &["the unit group declares no unit"],
            ),
            (
                error(HydroUnitStateViolation::InvalidTimeInState {
                    time_in_state_hours: -1.5,
                }),
                &["time_in_state_hours", "-1.5"],
            ),
            (
                error(HydroUnitStateViolation::InvalidPreviousGeneration {
                    previous_generation_mw: -2.5,
                }),
                &["previous_generation_mw", "-2.5"],
            ),
            (
                error(HydroUnitStateViolation::InvalidPreviousTurbined {
                    previous_turbined_m3s: -3.5,
                }),
                &["previous_turbined_m3s", "-3.5"],
            ),
            (
                error(HydroUnitStateViolation::OffWithGeneration {
                    previous_generation_mw: 5.5,
                }),
                &["status Off", "5.5 MW"],
            ),
            (
                error(HydroUnitStateViolation::OffWithTurbinedFlow {
                    previous_turbined_m3s: 6.5,
                }),
                &["status Off", "6.5 m³/s"],
            ),
            (
                error(HydroUnitStateViolation::OnGenerationOutsideBounds {
                    previous_generation_mw: 10.5,
                    min_generation_mw: 20.5,
                    max_generation_mw: 60.5,
                }),
                &["status On", "[20.5, 60.5] MW", "got 10.5 MW"],
            ),
            (
                error(HydroUnitStateViolation::OnTurbinedOutsideBounds {
                    previous_turbined_m3s: 70.5,
                    min_turbined_m3s: 10.5,
                    max_turbined_m3s: 30.5,
                }),
                &["status On", "[10.5, 30.5] m³/s", "got 70.5 m³/s"],
            ),
        ];
        for (err, needles) in cases {
            let msg = err.to_string();
            for needle in needles {
                assert!(msg.contains(needle), "missing {needle:?}: {msg}");
            }
        }
    }

    #[test]
    fn test_error_trait() {
        let err = ValidationError::MissingUnitGroups {
            hydro_id: EntityId(7),
        };
        let _: &dyn std::error::Error = &err;
    }
}
