//! Error types produced during `System` construction and validation.

use core::fmt;

use crate::EntityId;

/// Errors produced during System construction and validation.
///
/// Each variant carries enough context to pinpoint the invalid input.
///
/// # Examples
///
/// ```
/// use cobre_core::{EntityId, ValidationError};
///
/// let err = ValidationError::DuplicateId {
///     entity_type: "Bus",
///     id: EntityId(1),
/// };
/// assert!(err.to_string().contains("Bus"));
/// ```
#[derive(Debug, Clone)]
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
        }
    }
}

impl std::error::Error for ValidationError {}

#[cfg(test)]
mod tests {
    use super::ValidationError;
    use crate::EntityId;

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
    fn test_error_trait() {
        let err = ValidationError::MissingUnitGroups {
            hydro_id: EntityId(7),
        };
        let _: &dyn std::error::Error = &err;
    }
}
