//! Postcard-safe mirror types for [`ScalarParameter`], used to broadcast
//! scalar parameters over MPI.
//!
//! [`ParameterKind`] and [`ComputedParameter`] use serde internally-tagged
//! enums (`#[serde(tag = "...")]`) to drive the user-facing JSON schema — a
//! representation postcard cannot encode. [`BroadcastScalarParameter`],
//! [`BroadcastParameterKind`] and [`BroadcastComputedParameter`] mirror them
//! with externally-tagged encoding instead, convertible via `From` in both
//! directions. This is the same pattern as `BroadcastConfig` in `cobre-cli`,
//! used for [`crate::Config`].
//!
//! A postcard round-trip of a [`cobre_core::System`] returns a value whose
//! lookup indices are rebuilt by its `Deserialize` impl, so `system.bus(id)`
//! works immediately.

use cobre_core::{ComputedParameter, EntityId, ParameterKind, ScalarParameter};
use serde::{Deserialize, Serialize};

// ── Broadcast mirror types (tag-free, postcard-compatible) ──────────────────

/// Postcard-safe mirror of [`ScalarParameter`], holding a [`BroadcastParameterKind`]
/// in place of [`ParameterKind`]. Convert with `From` in both directions.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BroadcastScalarParameter {
    /// Unique parameter identifier.
    pub id: EntityId,
    /// Short name used in reports and log output.
    pub name: String,
    /// Kind in the broadcast-safe representation.
    pub kind: BroadcastParameterKind,
}

/// Postcard-safe mirror of [`ParameterKind`]. Uses externally-tagged enum
/// encoding (the serde default), which postcard supports natively.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum BroadcastParameterKind {
    /// Single scalar value applied to every stage.
    Constant(f64),
    /// Dense `Vec<f64>` indexed by stage (0-based).
    PerStage(Vec<f64>),
    /// Sorted, deduplicated `(season_id, value)` pairs.
    Seasonal(Vec<(i32, f64)>),
    /// Computed-parameter specification.
    Computed(BroadcastComputedParameter),
    /// Sorted, unique-keyed `(stage_id, block_id, value)` triples.
    PerStageBlock(Vec<(i32, i32, f64)>),
}

/// Postcard-safe mirror of [`ComputedParameter`]. Externally-tagged.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BroadcastComputedParameter {
    /// Equivalent productivity coefficient (`ρ_eq`).
    EquivalentProductivity(EntityId),
    /// Accumulated productivity coefficient (`ρ_acum`).
    AccumulatedProductivity(EntityId),
    /// Reference reservoir volume (`V_ref`).
    ReferenceVolume(EntityId),
    /// Reference turbine flow (`Q_ref`).
    ReferenceTurbine(EntityId),
    /// Minimum operational storage (`V_min`).
    MinStorage(EntityId),
    /// Maximum operational storage (`V_max`).
    MaxStorage(EntityId),
    /// Specific productivity (`ρ_esp`).
    SpecificProductivity(EntityId),
    /// Storage-range mean equivalent productivity (own scope).
    IntegratedEquivalentProductivity(EntityId),
    /// Storage-range mean accumulated productivity (cascade scope).
    IntegratedAccumulatedProductivity(EntityId),
    /// Maximum stored energy over the physical per-stage storage range.
    MaxStoredEnergy(EntityId),
}

impl From<&ScalarParameter> for BroadcastScalarParameter {
    fn from(p: &ScalarParameter) -> Self {
        Self {
            id: p.id,
            name: p.name.clone(),
            kind: BroadcastParameterKind::from(&p.kind),
        }
    }
}

impl From<BroadcastScalarParameter> for ScalarParameter {
    fn from(b: BroadcastScalarParameter) -> Self {
        Self {
            id: b.id,
            name: b.name,
            kind: ParameterKind::from(b.kind),
        }
    }
}

impl From<&ParameterKind> for BroadcastParameterKind {
    fn from(k: &ParameterKind) -> Self {
        match k {
            ParameterKind::Constant { value } => Self::Constant(*value),
            ParameterKind::PerStage { values } => Self::PerStage(values.clone()),
            ParameterKind::Seasonal { values } => Self::Seasonal(values.clone()),
            ParameterKind::Computed { computed_spec } => {
                Self::Computed(BroadcastComputedParameter::from(*computed_spec))
            }
            ParameterKind::PerStageBlock { values } => Self::PerStageBlock(values.clone()),
        }
    }
}

impl From<BroadcastParameterKind> for ParameterKind {
    fn from(b: BroadcastParameterKind) -> Self {
        match b {
            BroadcastParameterKind::Constant(value) => Self::Constant { value },
            BroadcastParameterKind::PerStage(values) => Self::PerStage { values },
            BroadcastParameterKind::Seasonal(values) => Self::Seasonal { values },
            BroadcastParameterKind::Computed(c) => Self::Computed {
                computed_spec: ComputedParameter::from(c),
            },
            BroadcastParameterKind::PerStageBlock(values) => Self::PerStageBlock { values },
        }
    }
}

impl From<ComputedParameter> for BroadcastComputedParameter {
    fn from(c: ComputedParameter) -> Self {
        match c {
            ComputedParameter::EquivalentProductivity { hydro_id } => {
                Self::EquivalentProductivity(hydro_id)
            }
            ComputedParameter::AccumulatedProductivity { hydro_id } => {
                Self::AccumulatedProductivity(hydro_id)
            }
            ComputedParameter::ReferenceVolume { hydro_id } => Self::ReferenceVolume(hydro_id),
            ComputedParameter::ReferenceTurbine { hydro_id } => Self::ReferenceTurbine(hydro_id),
            ComputedParameter::MinStorage { hydro_id } => Self::MinStorage(hydro_id),
            ComputedParameter::MaxStorage { hydro_id } => Self::MaxStorage(hydro_id),
            ComputedParameter::SpecificProductivity { hydro_id } => {
                Self::SpecificProductivity(hydro_id)
            }
            ComputedParameter::IntegratedEquivalentProductivity { hydro_id } => {
                Self::IntegratedEquivalentProductivity(hydro_id)
            }
            ComputedParameter::IntegratedAccumulatedProductivity { hydro_id } => {
                Self::IntegratedAccumulatedProductivity(hydro_id)
            }
            ComputedParameter::MaxStoredEnergy { hydro_id } => Self::MaxStoredEnergy(hydro_id),
        }
    }
}

impl From<BroadcastComputedParameter> for ComputedParameter {
    fn from(b: BroadcastComputedParameter) -> Self {
        match b {
            BroadcastComputedParameter::EquivalentProductivity(hydro_id) => {
                Self::EquivalentProductivity { hydro_id }
            }
            BroadcastComputedParameter::AccumulatedProductivity(hydro_id) => {
                Self::AccumulatedProductivity { hydro_id }
            }
            BroadcastComputedParameter::ReferenceVolume(hydro_id) => {
                Self::ReferenceVolume { hydro_id }
            }
            BroadcastComputedParameter::ReferenceTurbine(hydro_id) => {
                Self::ReferenceTurbine { hydro_id }
            }
            BroadcastComputedParameter::MinStorage(hydro_id) => Self::MinStorage { hydro_id },
            BroadcastComputedParameter::MaxStorage(hydro_id) => Self::MaxStorage { hydro_id },
            BroadcastComputedParameter::SpecificProductivity(hydro_id) => {
                Self::SpecificProductivity { hydro_id }
            }
            BroadcastComputedParameter::IntegratedEquivalentProductivity(hydro_id) => {
                Self::IntegratedEquivalentProductivity { hydro_id }
            }
            BroadcastComputedParameter::IntegratedAccumulatedProductivity(hydro_id) => {
                Self::IntegratedAccumulatedProductivity { hydro_id }
            }
            BroadcastComputedParameter::MaxStoredEnergy(hydro_id) => {
                Self::MaxStoredEnergy { hydro_id }
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;
    use chrono::NaiveDate;
    use cobre_core::{
        AnticipatedCommitmentHistory, Bus, ComputedParameter, DeficitSegment, EntityId, Hydro,
        HydroGenerationModel, HydroPenalties, InitialConditions, ParameterKind, ScalarParameter,
        System, SystemBuilder, Thermal, entities::AnticipatedConfig,
    };

    fn minimal_bus(id: i32) -> Bus {
        Bus {
            id: EntityId(id),
            name: format!("Bus {id}"),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 500.0,
            }],
            excess_cost: 0.0,
        }
    }

    fn minimal_thermal(id: i32, bus_id: i32) -> Thermal {
        Thermal {
            id: EntityId(id),
            name: format!("Thermal {id}"),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            bus_id: EntityId(bus_id),
            entry_stage_id: None,
            exit_stage_id: None,
            cost_per_mwh: 50.0,
            min_generation_mw: 0.0,
            max_generation_mw: 100.0,
            anticipated_config: None,
        }
    }

    fn zero_hydro_penalties() -> HydroPenalties {
        HydroPenalties {
            spillage_cost: 0.0,
            diversion_cost: 0.0,
            turbined_cost: 0.0,
            storage_violation_below_cost: 0.0,
            filling_target_violation_cost: 0.0,
            turbined_violation_below_cost: 0.0,
            outflow_violation_below_cost: 0.0,
            outflow_violation_above_cost: 0.0,
            generation_violation_below_cost: 0.0,
            evaporation_violation_cost: 0.0,
            water_withdrawal_violation_cost: 0.0,
            water_withdrawal_violation_pos_cost: 0.0,
            water_withdrawal_violation_neg_cost: 0.0,
            evaporation_violation_pos_cost: 0.0,
            evaporation_violation_neg_cost: 0.0,
            inflow_nonnegativity_cost: 1000.0,
        }
    }

    fn minimal_hydro(id: i32, bus_id: i32) -> Hydro {
        let mut hydro = Hydro {
            unit_groups: Vec::new(),
            id: EntityId(id),
            name: format!("Hydro {id}"),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            downstream_id: None,
            travel_time_hours: None,
            entry_stage_id: None,
            exit_stage_id: None,
            min_storage_hm3: 0.0,
            max_storage_hm3: 1000.0,
            min_outflow_m3s: 0.0,
            max_outflow_m3s: None,
            generation_model: HydroGenerationModel::ConstantProductivity,
            min_turbined_m3s: 0.0,
            max_turbined_m3s: 200.0,
            specific_productivity_mw_per_m3s_per_m: None,
            min_generation_mw: 0.0,
            max_generation_mw: 200.0,
            tailrace: None,
            hydraulic_losses: None,
            efficiency: None,
            evaporation_coefficients_mm: None,
            evaporation_reference_volumes_hm3: None,
            diversion: None,
            filling: None,
            penalties: zero_hydro_penalties(),
        };
        hydro.declare_mirror_unit_group(EntityId(bus_id));
        hydro
    }

    #[test]
    fn test_round_trip_minimal_system() {
        let bus = minimal_bus(1);
        let system = SystemBuilder::new().buses(vec![bus]).build().unwrap();

        let bytes = postcard::to_allocvec(&system).unwrap();
        assert!(!bytes.is_empty());

        let restored = postcard::from_bytes::<System>(&bytes).unwrap();

        assert_eq!(restored.n_buses(), system.n_buses());
        assert!(restored.bus(EntityId(1)).is_some());
    }

    #[test]
    fn test_round_trip_populated_system() {
        let buses = vec![minimal_bus(1), minimal_bus(2)];
        let thermals = vec![minimal_thermal(1, 1), minimal_thermal(2, 2)];
        let hydros = vec![minimal_hydro(1, 1)];

        let system = SystemBuilder::new()
            .buses(buses)
            .thermals(thermals)
            .hydros(hydros)
            .build()
            .unwrap();

        let bytes = postcard::to_allocvec(&system).unwrap();
        let restored = postcard::from_bytes::<System>(&bytes).unwrap();

        assert_eq!(restored.n_buses(), system.n_buses());
        assert_eq!(restored.n_thermals(), system.n_thermals());
        assert_eq!(restored.n_hydros(), system.n_hydros());

        assert!(restored.bus(EntityId(1)).is_some());
        assert!(restored.bus(EntityId(2)).is_some());
        assert!(restored.thermal(EntityId(1)).is_some());
        assert!(restored.thermal(EntityId(2)).is_some());
        assert!(restored.hydro(EntityId(1)).is_some());

        assert_eq!(restored, system);
    }

    /// A generic constraint's affine bound remainder (`bound_upper_affine`) rides the
    /// constraint's derived serde through the `System` postcard wire — there is no
    /// hand-written mirror for `GenericConstraint`, so the restored System must equal
    /// the original. Uses a genuine multi-component remainder (a nonzero constant plus
    /// a scaled term), not just the `single` special case, to prove the whole shape
    /// round-trips.
    #[test]
    fn test_round_trip_generic_constraint_bound_ref() {
        use cobre_core::{
            AffineBound, ConstraintExpression, GenericConstraint, LinearTerm, SlackConfig,
            VariableRef,
        };

        let gc = GenericConstraint {
            id: EntityId(0),
            name: "demand_cap".to_string(),
            description: None,
            expression: ConstraintExpression {
                terms: vec![LinearTerm::literal(
                    1.0,
                    VariableRef::HydroGeneration {
                        hydro_id: EntityId(1),
                        block_id: None,
                        bus_id: None,
                    },
                )],
            },
            slack: SlackConfig {
                enabled: false,
                penalty: None,
            },
            bound_lower_affine: None,
            bound_upper_affine: Some(AffineBound {
                constant: 12.0,
                terms: vec![(0.5, EntityId(7))],
            }),
        };

        let system = SystemBuilder::new()
            .buses(vec![minimal_bus(1)])
            .hydros(vec![minimal_hydro(1, 1)])
            .generic_constraints(vec![gc])
            .build()
            .unwrap();

        let bytes = postcard::to_allocvec(&system).unwrap();
        let restored = postcard::from_bytes::<System>(&bytes).unwrap();

        assert_eq!(restored, system);
        assert_eq!(
            restored.generic_constraints()[0].bound_upper_affine,
            Some(AffineBound {
                constant: 12.0,
                terms: vec![(0.5, EntityId(7))],
            }),
            "affine bound remainder must survive broadcast round-trip"
        );
    }

    #[test]
    fn test_round_trip_anticipated_thermal_system() {
        // Guards against a future Thermal field reorder or AnticipatedConfig schema
        // change silently breaking broadcast deserialization of `anticipated_config`.
        let buses = vec![minimal_bus(1), minimal_bus(2)];
        let mut anticipated = minimal_thermal(10, 1);
        anticipated.anticipated_config = Some(AnticipatedConfig::LeadStages(2));
        let regular = minimal_thermal(20, 2);
        let thermals = vec![anticipated, regular];

        let system = SystemBuilder::new()
            .buses(buses)
            .thermals(thermals)
            .build()
            .unwrap();

        let bytes = postcard::to_allocvec(&system).unwrap();
        let restored = postcard::from_bytes::<System>(&bytes).unwrap();

        assert_eq!(restored.n_thermals(), system.n_thermals());
        let Some(restored_anticipated) = restored.thermal(EntityId(10)) else {
            panic!("thermal 10 must round-trip");
        };
        assert_eq!(
            restored_anticipated.anticipated_config,
            Some(AnticipatedConfig::LeadStages(2)),
            "anticipated_config must survive broadcast round-trip"
        );
        let Some(restored_regular) = restored.thermal(EntityId(20)) else {
            panic!("thermal 20 must round-trip");
        };
        assert_eq!(
            restored_regular.anticipated_config, None,
            "non-anticipated thermal must remain None after round-trip"
        );

        assert_eq!(restored, system);
    }

    #[test]
    fn test_serialized_size_reasonable() {
        let bus = minimal_bus(1);
        let system = SystemBuilder::new().buses(vec![bus]).build().unwrap();
        let bytes = postcard::to_allocvec(&system).unwrap();
        assert!(bytes.len() < 1024);
    }

    /// Build a `Vec<ScalarParameter>` with one instance of each of the five
    /// `ParameterKind` variants, covering all code paths through the
    /// postcard serialization layer.
    fn five_kinds_fixture() -> Vec<ScalarParameter> {
        vec![
            ScalarParameter {
                id: EntityId(1),
                name: "constant_param".to_string(),
                kind: ParameterKind::Constant { value: 1.5 },
            },
            ScalarParameter {
                id: EntityId(2),
                name: "per_stage_param".to_string(),
                kind: ParameterKind::PerStage {
                    values: vec![1.0, 2.0, 3.0],
                },
            },
            ScalarParameter {
                id: EntityId(3),
                name: "seasonal_param".to_string(),
                kind: ParameterKind::new_seasonal(vec![(2, 1.0), (1, 0.5)]),
            },
            ScalarParameter {
                id: EntityId(4),
                name: "computed_param".to_string(),
                kind: ParameterKind::Computed {
                    computed_spec: ComputedParameter::EquivalentProductivity {
                        hydro_id: EntityId(7),
                    },
                },
            },
            ScalarParameter {
                id: EntityId(5),
                name: "per_stage_block_param".to_string(),
                kind: ParameterKind::PerStageBlock {
                    values: vec![(0, 0, 1.0), (0, 1, 2.0), (1, 0, 3.0)],
                },
            },
        ]
    }

    #[test]
    fn round_trip_all_five_parameter_kinds() {
        let original = five_kinds_fixture();
        let mirror: Vec<BroadcastScalarParameter> = original
            .iter()
            .map(BroadcastScalarParameter::from)
            .collect();
        let bytes = postcard::to_allocvec(&mirror).unwrap();
        assert!(!bytes.is_empty());
        let restored_mirror: Vec<BroadcastScalarParameter> = postcard::from_bytes(&bytes).unwrap();
        let restored: Vec<ScalarParameter> = restored_mirror
            .into_iter()
            .map(ScalarParameter::from)
            .collect();
        assert_eq!(restored, original);
    }

    #[test]
    fn mirror_parameter_encoding_is_deterministic() {
        let params = five_kinds_fixture();
        let mirror: Vec<BroadcastScalarParameter> =
            params.iter().map(BroadcastScalarParameter::from).collect();
        let bytes_a = postcard::to_allocvec(&mirror).unwrap();
        let bytes_b = postcard::to_allocvec(&mirror).unwrap();
        assert_eq!(bytes_a, bytes_b);
    }

    /// Pins the postcard discriminant (first byte) of every
    /// `BroadcastComputedParameter` variant. Postcard encodes the variant index
    /// as a varint; for indices `< 0x80` the first byte equals the index. The
    /// integrated tags and `MaxStoredEnergy` are appended at the tail (0x07,
    /// 0x08, 0x09), so every existing variant keeps its byte and a previously
    /// serialized parameter still decodes.
    #[test]
    fn broadcast_computed_parameter_postcard_discriminant_pin() {
        let cases: &[(BroadcastComputedParameter, u8)] = &[
            (
                BroadcastComputedParameter::EquivalentProductivity(EntityId(0)),
                0x00,
            ),
            (
                BroadcastComputedParameter::AccumulatedProductivity(EntityId(0)),
                0x01,
            ),
            (
                BroadcastComputedParameter::ReferenceVolume(EntityId(0)),
                0x02,
            ),
            (
                BroadcastComputedParameter::ReferenceTurbine(EntityId(0)),
                0x03,
            ),
            (BroadcastComputedParameter::MinStorage(EntityId(0)), 0x04),
            (BroadcastComputedParameter::MaxStorage(EntityId(0)), 0x05),
            (
                BroadcastComputedParameter::SpecificProductivity(EntityId(0)),
                0x06,
            ),
            (
                BroadcastComputedParameter::IntegratedEquivalentProductivity(EntityId(0)),
                0x07,
            ),
            (
                BroadcastComputedParameter::IntegratedAccumulatedProductivity(EntityId(0)),
                0x08,
            ),
            (
                BroadcastComputedParameter::MaxStoredEnergy(EntityId(0)),
                0x09,
            ),
        ];
        for (variant, discriminant) in cases {
            let bytes = postcard::to_allocvec(variant).unwrap();
            assert_eq!(
                bytes[0], *discriminant,
                "{variant:?} must serialize to postcard discriminant {discriminant:#04x}"
            );
        }
    }

    #[test]
    fn broadcast_computed_parameter_integrated_tags_round_trip() {
        for variant in [
            BroadcastComputedParameter::IntegratedEquivalentProductivity(EntityId(11)),
            BroadcastComputedParameter::IntegratedAccumulatedProductivity(EntityId(13)),
        ] {
            let bytes = postcard::to_allocvec(&variant).unwrap();
            let restored: BroadcastComputedParameter = postcard::from_bytes(&bytes).unwrap();
            assert_eq!(restored, variant);
        }
    }

    #[test]
    fn broadcast_computed_parameter_max_stored_energy_round_trip() {
        let variant = BroadcastComputedParameter::MaxStoredEnergy(EntityId(17));
        let bytes = postcard::to_allocvec(&variant).unwrap();
        let restored: BroadcastComputedParameter = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(restored, variant);
    }

    #[test]
    fn test_broadcast_initial_conditions_round_trips_past_anticipated_commitments() {
        let original = InitialConditions {
            storage: vec![],
            filling_storage: vec![],
            past_anticipated_commitments: vec![
                AnticipatedCommitmentHistory {
                    thermal_id: EntityId(1),
                    start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                    end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
                    value_mw: 120.0,
                },
                AnticipatedCommitmentHistory {
                    thermal_id: EntityId(7),
                    start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                    end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
                    value_mw: 50.0,
                },
            ],
            recent_observations: vec![],
            past_defluences: vec![],
        };

        let bytes = postcard::to_allocvec(&original).unwrap();
        assert!(!bytes.is_empty());

        let restored: InitialConditions = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(restored, original);
        assert_eq!(restored.past_anticipated_commitments.len(), 2);
        assert_eq!(
            restored.past_anticipated_commitments[0].thermal_id,
            EntityId(1)
        );
        assert!((restored.past_anticipated_commitments[0].value_mw - 120.0).abs() < f64::EPSILON);
        assert_eq!(
            restored.past_anticipated_commitments[1].thermal_id,
            EntityId(7)
        );
        assert!((restored.past_anticipated_commitments[1].value_mw - 50.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_broadcast_initial_conditions_empty_past_anticipated_commitments() {
        let original = InitialConditions {
            storage: vec![],
            filling_storage: vec![],
            past_anticipated_commitments: vec![],
            recent_observations: vec![],
            past_defluences: vec![],
        };

        let bytes = postcard::to_allocvec(&original).unwrap();
        assert!(!bytes.is_empty());

        let restored: InitialConditions = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(restored, original);
        assert!(restored.past_anticipated_commitments.is_empty());
    }
}
