//! Energy contract entity — import/export agreements with external systems.

use crate::EntityId;
use chrono::NaiveDate;

/// Direction of energy flow for a bilateral contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ContractType {
    /// External energy flows into the modeled system.
    Import,
    /// System energy flows out to an external entity.
    Export,
}

/// Bilateral energy contract with an external system.
///
/// See Input System Entities SS1.9.7.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct EnergyContract {
    /// Unique contract identifier.
    pub id: EntityId,
    /// Human-readable contract name.
    pub name: String,
    /// Date the entity enters service (ISO 8601).
    pub operational_start_date: NaiveDate,
    /// Bus at which the contracted power is injected or withdrawn.
    pub bus_id: EntityId,
    /// Direction of energy flow for this contract.
    pub contract_type: ContractType,
    /// Stage index when the contract enters service. None = always active.
    pub entry_stage_id: Option<i32>,
    /// Stage index when the contract expires. None = never expires.
    pub exit_stage_id: Option<i32>,
    /// Contract price per `MWh`. Negative values represent export revenue \[$/`MWh`\].
    pub price_per_mwh: f64,
    /// Minimum contracted power \[MW\].
    pub min_mw: f64,
    /// Maximum contracted power \[MW\].
    pub max_mw: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_import_contract() {
        let contract = EnergyContract {
            id: EntityId::from(1),
            name: "Importação Argentina".to_string(),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            bus_id: EntityId::from(5),
            contract_type: ContractType::Import,
            entry_stage_id: None,
            exit_stage_id: None,
            price_per_mwh: 200.0,
            min_mw: 0.0,
            max_mw: 1000.0,
        };

        assert_eq!(contract.contract_type, ContractType::Import);
        assert!(contract.price_per_mwh > 0.0);
    }

    #[test]
    fn test_export_contract() {
        let contract = EnergyContract {
            id: EntityId::from(2),
            name: "Exportação Uruguai".to_string(),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            bus_id: EntityId::from(6),
            contract_type: ContractType::Export,
            entry_stage_id: Some(1),
            exit_stage_id: Some(60),
            price_per_mwh: -150.0,
            min_mw: 0.0,
            max_mw: 500.0,
        };

        assert_eq!(contract.contract_type, ContractType::Export);
        assert!(contract.price_per_mwh < 0.0);
    }

    #[test]
    fn test_contract_type_equality() {
        assert_eq!(ContractType::Import, ContractType::Import);
        assert_eq!(ContractType::Export, ContractType::Export);
        assert_ne!(ContractType::Import, ContractType::Export);
    }
}
