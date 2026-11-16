//! Provenance metadata for stochastic pipeline components.

use novomodelo_core::scenario::SamplingScheme;

use crate::tree::MomentMatching;

/// Origin of a single stochastic pipeline component.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComponentProvenance {
    /// Computed by the stochastic pipeline from system data.
    Generated,
    /// Provided by the caller as an external override.
    UserSupplied,
    /// Component not present — system has no relevant entities.
    NotApplicable,
}

/// Provenance records for all components of a [`StochasticContext`](crate::StochasticContext).
///
/// Each `*_scheme` field is `None` when its class's per-class config is not yet applied.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StochasticProvenance {
    /// Origin of the opening scenario tree.
    pub opening_tree: ComponentProvenance,
    /// Moment matching applied to the opening tree; `None` unless the tree was generated.
    pub opening_tree_moment_matching: Option<MomentMatching>,
    /// Origin of inflow PAR models (`Generated` when hydros present; `NotApplicable` otherwise).
    pub inflow_model: ComponentProvenance,
    /// Inflow entity class.
    pub inflow_scheme: Option<SamplingScheme>,
    /// Load entity class.
    pub load_scheme: Option<SamplingScheme>,
    /// NCS entity class.
    pub ncs_scheme: Option<SamplingScheme>,
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use novomodelo_core::{
        EntityId, SystemBuilder,
        scenario::{
            CorrelationEntity, CorrelationGroup, CorrelationModel, CorrelationProfile,
            SamplingScheme,
        },
        test_support::{InflowModelSpec, make_inflow_model},
    };

    use crate::{
        ComponentProvenance,
        context::{ClassSchemes, OpeningTree, OpeningTreeInputs, build_stochastic_context},
        test_support::{deficit_bus, saa_stage, sized_hydro},
        tree::MomentMatching,
    };

    fn identity_correlation(entity_ids: &[i32]) -> CorrelationModel {
        let n = entity_ids.len();
        let matrix: Vec<Vec<f64>> = (0..n)
            .map(|i| (0..n).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
            .collect();
        let mut profiles = BTreeMap::new();
        profiles.insert(
            "default".to_string(),
            CorrelationProfile {
                groups: vec![CorrelationGroup {
                    name: "g1".to_string(),
                    entities: entity_ids
                        .iter()
                        .map(|&id| CorrelationEntity {
                            entity_type: "inflow".to_string(),
                            id: EntityId(id),
                        })
                        .collect(),
                    matrix,
                }],
            },
        );
        CorrelationModel {
            method: "spectral".to_string(),
            profiles,
            schedule: vec![],
        }
    }

    #[test]
    fn provenance_with_user_tree() {
        let hydros = vec![sized_hydro(1)];
        let stages = vec![saa_stage(0, 0, 3), saa_stage(1, 1, 3)];
        let inflow_models = vec![
            make_inflow_model(InflowModelSpec {
                hydro_id: 1,
                stage_id: 0,
                ..Default::default()
            }),
            make_inflow_model(InflowModelSpec {
                hydro_id: 1,
                stage_id: 1,
                ..Default::default()
            }),
        ];

        let system = SystemBuilder::new()
            .buses(vec![deficit_bus(0)])
            .hydros(hydros)
            .stages(stages)
            .inflow_models(inflow_models)
            .correlation(identity_correlation(&[1]))
            .build()
            .unwrap();

        let user_tree = OpeningTree::from_parts(vec![1.0_f64; 2 * 2], vec![2, 2], 1);
        let ctx = build_stochastic_context(
            &system,
            42,
            None,
            &[],
            &[],
            OpeningTreeInputs {
                user_tree: Some(user_tree),
                historical_library: None,
                external_scenario_counts: None,
                noise_group_ids: None,
                moment_matching: MomentMatching::None,
            },
            ClassSchemes {
                inflow: Some(SamplingScheme::InSample),
                load: Some(SamplingScheme::InSample),
                ncs: Some(SamplingScheme::InSample),
            },
        )
        .unwrap();

        assert_eq!(
            ctx.provenance().opening_tree,
            ComponentProvenance::UserSupplied,
            "user-supplied tree must be recorded as UserSupplied"
        );
    }

    #[test]
    fn provenance_without_user_tree() {
        let hydros = vec![sized_hydro(1)];
        let stages = vec![saa_stage(0, 0, 3), saa_stage(1, 1, 3)];
        let inflow_models = vec![
            make_inflow_model(InflowModelSpec {
                hydro_id: 1,
                stage_id: 0,
                ..Default::default()
            }),
            make_inflow_model(InflowModelSpec {
                hydro_id: 1,
                stage_id: 1,
                ..Default::default()
            }),
        ];

        let system = SystemBuilder::new()
            .buses(vec![deficit_bus(0)])
            .hydros(hydros)
            .stages(stages)
            .inflow_models(inflow_models)
            .correlation(identity_correlation(&[1]))
            .build()
            .unwrap();

        let ctx = build_stochastic_context(
            &system,
            42,
            None,
            &[],
            &[],
            OpeningTreeInputs::default(),
            ClassSchemes {
                inflow: Some(SamplingScheme::InSample),
                load: Some(SamplingScheme::InSample),
                ncs: Some(SamplingScheme::InSample),
            },
        )
        .unwrap();

        assert_eq!(
            ctx.provenance().opening_tree,
            ComponentProvenance::Generated,
            "generated opening tree must be recorded as Generated"
        );
    }

    #[test]
    fn generated_tree_records_its_moment_matching() {
        let system = SystemBuilder::new()
            .buses(vec![deficit_bus(0)])
            .hydros(vec![sized_hydro(1)])
            .stages(vec![saa_stage(0, 0, 3), saa_stage(1, 1, 3)])
            .inflow_models(vec![
                make_inflow_model(InflowModelSpec {
                    hydro_id: 1,
                    stage_id: 0,
                    ..Default::default()
                }),
                make_inflow_model(InflowModelSpec {
                    hydro_id: 1,
                    stage_id: 1,
                    ..Default::default()
                }),
            ])
            .correlation(identity_correlation(&[1]))
            .build()
            .unwrap();
        let schemes = || ClassSchemes {
            inflow: Some(SamplingScheme::InSample),
            load: Some(SamplingScheme::InSample),
            ncs: Some(SamplingScheme::InSample),
        };

        let matched = build_stochastic_context(
            &system,
            42,
            None,
            &[],
            &[],
            OpeningTreeInputs {
                moment_matching: MomentMatching::PerCoordinate,
                ..Default::default()
            },
            schemes(),
        )
        .unwrap();
        let default = build_stochastic_context(
            &system,
            42,
            None,
            &[],
            &[],
            OpeningTreeInputs::default(),
            schemes(),
        )
        .unwrap();
        let supplied = build_stochastic_context(
            &system,
            42,
            None,
            &[],
            &[],
            OpeningTreeInputs {
                user_tree: Some(OpeningTree::from_parts(vec![1.0_f64; 2 * 2], vec![2, 2], 1)),
                moment_matching: MomentMatching::PerCoordinate,
                ..Default::default()
            },
            schemes(),
        )
        .unwrap();

        assert_eq!(
            [
                matched.provenance().opening_tree_moment_matching,
                default.provenance().opening_tree_moment_matching,
                supplied.provenance().opening_tree_moment_matching,
            ],
            [
                Some(MomentMatching::PerCoordinate),
                Some(MomentMatching::None),
                None
            ]
        );
    }

    #[test]
    fn provenance_no_entities() {
        // A system with no hydros and no stochastic load buses has dim == 0.
        let system = SystemBuilder::new()
            .buses(vec![deficit_bus(0)])
            .stages(vec![saa_stage(0, 0, 3)])
            .build()
            .unwrap();

        let ctx = build_stochastic_context(
            &system,
            42,
            None,
            &[],
            &[],
            OpeningTreeInputs::default(),
            ClassSchemes {
                inflow: Some(SamplingScheme::InSample),
                load: Some(SamplingScheme::InSample),
                ncs: Some(SamplingScheme::InSample),
            },
        )
        .unwrap();

        assert_eq!(
            ctx.provenance().opening_tree,
            ComponentProvenance::NotApplicable,
            "dim == 0 means opening tree is NotApplicable"
        );
    }

    #[test]
    fn provenance_inflow_with_hydros() {
        let hydros = vec![sized_hydro(1)];
        let stages = vec![saa_stage(0, 0, 3), saa_stage(1, 1, 3)];
        let inflow_models = vec![
            make_inflow_model(InflowModelSpec {
                hydro_id: 1,
                stage_id: 0,
                ..Default::default()
            }),
            make_inflow_model(InflowModelSpec {
                hydro_id: 1,
                stage_id: 1,
                ..Default::default()
            }),
        ];

        let system = SystemBuilder::new()
            .buses(vec![deficit_bus(0)])
            .hydros(hydros)
            .stages(stages)
            .inflow_models(inflow_models)
            .correlation(identity_correlation(&[1]))
            .build()
            .unwrap();

        let ctx = build_stochastic_context(
            &system,
            42,
            None,
            &[],
            &[],
            OpeningTreeInputs::default(),
            ClassSchemes {
                inflow: Some(SamplingScheme::InSample),
                load: Some(SamplingScheme::InSample),
                ncs: Some(SamplingScheme::InSample),
            },
        )
        .unwrap();

        assert_eq!(
            ctx.provenance().inflow_model,
            ComponentProvenance::Generated,
            "system with hydros must produce inflow_model == Generated"
        );
    }

    #[test]
    fn provenance_inflow_without_hydros() {
        let system = SystemBuilder::new()
            .buses(vec![deficit_bus(0)])
            .stages(vec![saa_stage(0, 0, 3)])
            .build()
            .unwrap();

        let ctx = build_stochastic_context(
            &system,
            42,
            None,
            &[],
            &[],
            OpeningTreeInputs::default(),
            ClassSchemes {
                inflow: Some(SamplingScheme::InSample),
                load: Some(SamplingScheme::InSample),
                ncs: Some(SamplingScheme::InSample),
            },
        )
        .unwrap();

        assert_eq!(
            ctx.provenance().inflow_model,
            ComponentProvenance::NotApplicable,
            "system with no hydros must produce inflow_model == NotApplicable"
        );
    }

    #[test]
    fn test_provenance_per_class_schemes_populated() {
        let system = SystemBuilder::new()
            .buses(vec![deficit_bus(0)])
            .stages(vec![saa_stage(0, 0, 3)])
            .build()
            .unwrap();

        let ctx = build_stochastic_context(
            &system,
            42,
            None,
            &[],
            &[],
            OpeningTreeInputs::default(),
            ClassSchemes {
                inflow: Some(SamplingScheme::InSample),
                load: Some(SamplingScheme::InSample),
                ncs: Some(SamplingScheme::InSample),
            },
        )
        .unwrap();

        assert!(
            ctx.provenance().inflow_scheme == Some(SamplingScheme::InSample),
            "inflow_scheme must be Some(InSample) when passed as argument"
        );
        assert!(
            ctx.provenance().load_scheme == Some(SamplingScheme::InSample),
            "load_scheme must be Some(InSample) when passed as argument"
        );
        assert!(
            ctx.provenance().ncs_scheme == Some(SamplingScheme::InSample),
            "ncs_scheme must be Some(InSample) when passed as argument"
        );
    }
}
