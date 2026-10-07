//! User-supplied productivity override table: the
//! [`HydroEnergyProductivityOverride`] lookup struct and its
//! [`build_hydro_energy_productivity_override`] constructor.

use std::collections::{HashMap, HashSet};

use cobre_core::{EntityId, StageId};
use cobre_io::{HydroEnergyProductivityRow, LoadError};

/// Per-`(hydro, stage)` override table loaded from
/// `system/hydro_energy_productivity.parquet`.
///
/// Each of the three override columns (`ρ_eq`, `Q_ref`, `ρ_esp`) is stored in
/// two parallel lookup tables: one for stage-specific rows and one for
/// per-hydro defaults (rows whose `stage_id` is NULL in the source file).
///
/// Accessors apply a per-stage → per-hydro-default → `None` fallback chain.
/// `Default` yields an override where every accessor returns `None`.
#[derive(Debug, Default, Clone)]
pub struct HydroEnergyProductivityOverride {
    rho_eq_per_hydro_stage: HashMap<(EntityId, i32), f64>,
    rho_eq_per_hydro_default: HashMap<EntityId, f64>,
    q_ref_per_hydro_stage: HashMap<(EntityId, i32), f64>,
    q_ref_per_hydro_default: HashMap<EntityId, f64>,
    rho_esp_per_hydro_stage: HashMap<(EntityId, i32), f64>,
    rho_esp_per_hydro_default: HashMap<EntityId, f64>,
}

impl HydroEnergyProductivityOverride {
    /// Returns the user-supplied `ρ_eq` for `(hydro, stage)` if any.
    #[must_use]
    pub fn equivalent_productivity(&self, hydro: EntityId, stage: StageId) -> Option<f64> {
        if let Some(&v) = self.rho_eq_per_hydro_stage.get(&(hydro, stage.0)) {
            return Some(v);
        }
        self.rho_eq_per_hydro_default.get(&hydro).copied()
    }

    /// Returns the user-supplied `Q_ref` \[m³/s\] for `(hydro, stage)` if any.
    #[must_use]
    pub fn reference_outflow(&self, hydro: EntityId, stage: StageId) -> Option<f64> {
        if let Some(&v) = self.q_ref_per_hydro_stage.get(&(hydro, stage.0)) {
            return Some(v);
        }
        self.q_ref_per_hydro_default.get(&hydro).copied()
    }

    /// Returns the user-supplied `ρ_esp` \[MW/(m³/s)/m\] for `(hydro, stage)` if any.
    #[must_use]
    pub fn specific_productivity(&self, hydro: EntityId, stage: StageId) -> Option<f64> {
        if let Some(&v) = self.rho_esp_per_hydro_stage.get(&(hydro, stage.0)) {
            return Some(v);
        }
        self.rho_esp_per_hydro_default.get(&hydro).copied()
    }

    /// Resolves `ρ_esp` for `(hydro, stage)` as override → `entity_default`. The
    /// single owner of the `ρ_esp` precedence shared by the energy-conversion
    /// builder and the `SpecificProductivity` tag.
    #[must_use]
    pub fn resolve_specific_productivity(
        &self,
        hydro: EntityId,
        stage: StageId,
        entity_default: Option<f64>,
    ) -> Option<f64> {
        self.specific_productivity(hydro, stage).or(entity_default)
    }
}

/// Build a [`HydroEnergyProductivityOverride`] from parsed rows.
///
/// A NULL `stage_id` is a distinct key from any concrete stage (it is the
/// per-hydro default, not a wildcard).
///
/// # Errors
///
/// Returns [`LoadError::SchemaError`] with
/// `field = "hydro_energy_productivity.duplicate_entry"` when the same
/// `(hydro_id, stage_id)` key appears more than once in `rows`.
pub fn build_hydro_energy_productivity_override(
    rows: &[HydroEnergyProductivityRow],
) -> Result<HydroEnergyProductivityOverride, LoadError> {
    let mut seen: HashSet<(EntityId, Option<i32>)> = HashSet::with_capacity(rows.len());
    let mut out = HydroEnergyProductivityOverride::default();

    for row in rows {
        let key = (row.hydro_id, row.stage_id);
        if !seen.insert(key) {
            let stage_label = row
                .stage_id
                .map_or_else(|| "NULL".to_string(), |s| s.to_string());
            return Err(LoadError::SchemaError {
                path: std::path::PathBuf::from("<hydro_energy_productivity>"),
                field: "hydro_energy_productivity.duplicate_entry".to_string(),
                message: format!(
                    "duplicate (hydro_id={}, stage_id={}) key",
                    row.hydro_id.0, stage_label
                ),
            });
        }

        if let Some(s) = row.stage_id {
            if let Some(v) = row.equivalent_productivity_mw_per_m3s {
                out.rho_eq_per_hydro_stage.insert((row.hydro_id, s), v);
            }
            if let Some(v) = row.reference_outflow_m3s {
                out.q_ref_per_hydro_stage.insert((row.hydro_id, s), v);
            }
            if let Some(v) = row.specific_productivity_mw_per_m3s_per_m {
                out.rho_esp_per_hydro_stage.insert((row.hydro_id, s), v);
            }
        } else {
            if let Some(v) = row.equivalent_productivity_mw_per_m3s {
                out.rho_eq_per_hydro_default.insert(row.hydro_id, v);
            }
            if let Some(v) = row.reference_outflow_m3s {
                out.q_ref_per_hydro_default.insert(row.hydro_id, v);
            }
            if let Some(v) = row.specific_productivity_mw_per_m3s_per_m {
                out.rho_esp_per_hydro_default.insert(row.hydro_id, v);
            }
        }
    }

    Ok(out)
}

#[cfg(test)]
#[allow(
    clippy::doc_markdown,
    clippy::expect_used,
    clippy::float_cmp,
    clippy::panic,
    clippy::unwrap_used
)]
mod tests {
    use cobre_core::EntityId;
    use cobre_io::HydroEnergyProductivityRow;

    use super::*;

    #[test]
    fn test_override_three_column_lookup_precedence() {
        let rows = vec![
            HydroEnergyProductivityRow {
                hydro_id: EntityId(1),
                stage_id: Some(0),
                equivalent_productivity_mw_per_m3s: Some(3.6),
                reference_outflow_m3s: None,
                specific_productivity_mw_per_m3s_per_m: None,
            },
            HydroEnergyProductivityRow {
                hydro_id: EntityId(1),
                stage_id: None,
                equivalent_productivity_mw_per_m3s: Some(4.0),
                reference_outflow_m3s: None,
                specific_productivity_mw_per_m3s_per_m: Some(0.009),
            },
            HydroEnergyProductivityRow {
                hydro_id: EntityId(2),
                stage_id: None,
                equivalent_productivity_mw_per_m3s: Some(5.0),
                reference_outflow_m3s: Some(200.0),
                specific_productivity_mw_per_m3s_per_m: None,
            },
        ];
        let o = build_hydro_energy_productivity_override(&rows).expect("override builds");
        assert_eq!(
            o.equivalent_productivity(EntityId(1), StageId(0)),
            Some(3.6)
        );
        assert_eq!(
            o.equivalent_productivity(EntityId(1), StageId(1)),
            Some(4.0)
        );
        assert_eq!(
            o.equivalent_productivity(EntityId(2), StageId(0)),
            Some(5.0)
        );
        assert_eq!(o.equivalent_productivity(EntityId(3), StageId(0)), None);
        assert_eq!(o.reference_outflow(EntityId(2), StageId(0)), Some(200.0));
        assert_eq!(o.reference_outflow(EntityId(1), StageId(0)), None);
        assert_eq!(
            o.specific_productivity(EntityId(1), StageId(0)),
            Some(0.009)
        );
        assert_eq!(
            o.specific_productivity(EntityId(1), StageId(1)),
            Some(0.009)
        );
        assert_eq!(o.specific_productivity(EntityId(2), StageId(0)), None);
    }

    #[test]
    fn test_build_override_rejects_duplicate_hydro_stage() {
        let rows = vec![
            HydroEnergyProductivityRow {
                hydro_id: EntityId(1),
                stage_id: Some(0),
                equivalent_productivity_mw_per_m3s: Some(3.6),
                reference_outflow_m3s: None,
                specific_productivity_mw_per_m3s_per_m: None,
            },
            HydroEnergyProductivityRow {
                hydro_id: EntityId(1),
                stage_id: Some(0),
                equivalent_productivity_mw_per_m3s: None,
                reference_outflow_m3s: Some(200.0),
                specific_productivity_mw_per_m3s_per_m: None,
            },
        ];
        let err = build_hydro_energy_productivity_override(&rows).unwrap_err();
        match err {
            LoadError::SchemaError { field, .. } => {
                assert_eq!(field, "hydro_energy_productivity.duplicate_entry");
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    #[test]
    fn test_build_override_distinguishes_null_and_concrete_stages() {
        let rows = vec![
            HydroEnergyProductivityRow {
                hydro_id: EntityId(1),
                stage_id: None,
                equivalent_productivity_mw_per_m3s: Some(2.0),
                reference_outflow_m3s: None,
                specific_productivity_mw_per_m3s_per_m: None,
            },
            HydroEnergyProductivityRow {
                hydro_id: EntityId(1),
                stage_id: Some(0),
                equivalent_productivity_mw_per_m3s: Some(3.0),
                reference_outflow_m3s: None,
                specific_productivity_mw_per_m3s_per_m: None,
            },
        ];
        let o = build_hydro_energy_productivity_override(&rows).expect("override builds");
        assert_eq!(
            o.equivalent_productivity(EntityId(1), StageId(0)),
            Some(3.0)
        );
        assert_eq!(
            o.equivalent_productivity(EntityId(1), StageId(1)),
            Some(2.0)
        );
    }

    #[test]
    fn resolve_specific_productivity_applies_override_then_entity_default() {
        let rows = vec![HydroEnergyProductivityRow {
            hydro_id: EntityId(1),
            stage_id: None,
            equivalent_productivity_mw_per_m3s: None,
            reference_outflow_m3s: None,
            specific_productivity_mw_per_m3s_per_m: Some(0.009),
        }];
        let o = build_hydro_energy_productivity_override(&rows).expect("override builds");

        assert_eq!(
            o.resolve_specific_productivity(EntityId(1), StageId(0), Some(0.5)),
            Some(0.009),
            "override wins over entity_default"
        );
        assert_eq!(
            o.resolve_specific_productivity(EntityId(2), StageId(0), Some(0.5)),
            Some(0.5),
            "entity_default applies when no override exists"
        );
        assert_eq!(
            o.resolve_specific_productivity(EntityId(3), StageId(0), None),
            None,
            "None when neither source supplies a value"
        );
    }

    #[test]
    fn test_default_override_returns_none_for_every_accessor() {
        let o = HydroEnergyProductivityOverride::default();
        assert_eq!(o.equivalent_productivity(EntityId(1), StageId(0)), None);
        assert_eq!(o.reference_outflow(EntityId(1), StageId(0)), None);
        assert_eq!(o.specific_productivity(EntityId(1), StageId(0)), None);
    }

    /// A stage-specific row keyed at a non-0-based domain `stage_id` (e.g. a
    /// study whose stages start at id 60) resolves at `StageId(60)`, never at
    /// study position 0 — the accessor takes a domain id, so there is no
    /// position to accidentally key on.
    #[test]
    fn test_non_zero_based_stage_id_resolves_by_domain_id_not_position() {
        let rows = vec![HydroEnergyProductivityRow {
            hydro_id: EntityId(1),
            stage_id: Some(60),
            equivalent_productivity_mw_per_m3s: Some(7.2),
            reference_outflow_m3s: None,
            specific_productivity_mw_per_m3s_per_m: None,
        }];
        let o = build_hydro_energy_productivity_override(&rows).expect("override builds");
        assert_eq!(
            o.equivalent_productivity(EntityId(1), StageId(60)),
            Some(7.2)
        );
        assert_eq!(o.equivalent_productivity(EntityId(1), StageId(0)), None);
    }
}
