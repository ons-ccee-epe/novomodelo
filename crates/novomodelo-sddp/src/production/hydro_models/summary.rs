//! Display-summary builder for the hydro model preprocessing pipeline.
//!
//! Aggregates the resolved production and evaporation models plus their
//! provenance into a `HydroModelSummary` consumed by `novomodelo-cli` for display.
//! All counts are derived from the already-validated pipeline result; summary
//! construction is infallible.

use novomodelo_core::{EntityId, System};

use super::types::{
    EvaporationReferenceSource, EvaporationSource, FphaHydroDetail, HydroModelSummary,
    NoTurbineCapacityHydro, PrepareHydroModelsResult, ProductionModelSource,
    ResolvedProductionModel,
};
use crate::indexer::HydroCellIndex;
use crate::lp::builder::frozen_turbining_hydros;
// ── Summary builder ───────────────────────────────────────────────────────────

/// Build a [`HydroModelSummary`] from the pipeline result and the system;
/// infallible (all fields derive from the already-validated inputs).
///
/// `total_planes` sums plane counts from the first study stage only — taken as
/// representative for display.
#[must_use]
pub fn build_hydro_model_summary(
    result: &PrepareHydroModelsResult,
    system: &System,
) -> HydroModelSummary {
    let mut n_constant = 0usize;
    let mut n_fpha = 0usize;
    let mut total_planes = 0usize;
    let mut fpha_details: Vec<FphaHydroDetail> = Vec::new();
    let mut no_turbine_capacity: Vec<NoTurbineCapacityHydro> = Vec::new();

    let has_study_stage = system.stages().iter().any(|s| s.id >= 0);
    let representative_stage = 0usize;

    for (hydro_pos, (entity_id, source)) in result.provenance.production_sources.iter().enumerate()
    {
        match source {
            ProductionModelSource::DefaultConstant => n_constant += 1,
            ProductionModelSource::NoTurbineCapacity => {
                n_constant += 1;
                no_turbine_capacity.push(NoTurbineCapacityHydro {
                    hydro_id: *entity_id,
                    name: hydro_name(system, *entity_id),
                });
            }
            ProductionModelSource::PrecomputedHyperplanes
            | ProductionModelSource::ComputedFromGeometry => {
                n_fpha += 1;

                let n_planes = if has_study_stage {
                    match result.production.model(hydro_pos, representative_stage) {
                        ResolvedProductionModel::Fpha { planes, .. } => planes.len(),
                        ResolvedProductionModel::ConstantProductivity { .. } => 0,
                    }
                } else {
                    0
                };

                total_planes += n_planes;

                fpha_details.push(FphaHydroDetail {
                    hydro_id: *entity_id,
                    name: hydro_name(system, *entity_id),
                    source: *source,
                    n_planes,
                });
            }
        }
    }
    no_turbine_capacity.sort_by_key(|entry| entry.hydro_id);

    let mut n_evaporation = 0usize;
    let mut n_no_evaporation = 0usize;

    for (_, source) in &result.provenance.evaporation_sources {
        match source {
            EvaporationSource::LinearizedFromGeometry => n_evaporation += 1,
            EvaporationSource::NotModeled | EvaporationSource::DisabledNoArea => {
                n_no_evaporation += 1;
            }
        }
    }

    let mut n_user_supplied_ref = 0usize;
    let mut n_default_midpoint_ref = 0usize;

    for ((_, evap_src), (_, ref_src)) in result
        .provenance
        .evaporation_sources
        .iter()
        .zip(result.provenance.evaporation_reference_sources.iter())
    {
        if *evap_src == EvaporationSource::LinearizedFromGeometry {
            match ref_src {
                EvaporationReferenceSource::UserSupplied => n_user_supplied_ref += 1,
                EvaporationReferenceSource::DefaultMidpoint => n_default_midpoint_ref += 1,
            }
        }
    }

    HydroModelSummary {
        n_constant,
        n_fpha,
        total_planes,
        fpha_details,
        no_turbine_capacity,
        frozen_turbining: frozen_turbining_hydros(
            system.hydros(),
            system.stages(),
            system.bounds(),
            &HydroCellIndex::build(system.hydros()),
        ),
        n_evaporation,
        n_no_evaporation,
        n_user_supplied_ref,
        n_default_midpoint_ref,
    }
}

fn hydro_name(system: &System, entity_id: EntityId) -> String {
    system
        .hydros()
        .iter()
        .find(|h| h.id == entity_id)
        .map_or_else(|| entity_id.0.to_string(), |h| h.name.clone())
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(
    clippy::doc_markdown,
    clippy::match_wildcard_for_single_variants,
    clippy::cast_precision_loss,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic
)]
mod tests {
    use chrono::{Months, NaiveDate};
    use novomodelo_core::{
        BoundsCountsSpec, BoundsDefaults, Bus, ContractBlockBounds, DeficitSegment, EntityId,
        Hydro, HydroBlockBounds, HydroStageBounds, HydroUnitGroup, LineBlockBounds,
        PumpingBlockBounds, ResolvedBounds, SystemBuilder, TailraceFlow, ThermalBlockBounds,
        entities::hydro::{HydroGenerationModel, HydroPenalties},
        scenario::CorrelationModel,
        temporal::{
            Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
            StageStateConfig,
        },
    };

    // The summary tests construct production and evaporation result types owned by
    // sibling submodules; import the full re-exported surface from the directory
    // module rather than only `summary`'s own narrow `use` block.
    use crate::HydroEnergyProductivityOverride;
    use crate::production::hydro_models::*;
    use crate::test_support::{make_unit_group, minimal_hydros};

    // ── Test helpers ──────────────────────────────────────────────────────────

    fn zero_penalties() -> HydroPenalties {
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
            fpha_violation_cost: 10000.0,
            evaporation_violation_cost: 0.0,
            water_withdrawal_violation_cost: 0.0,
            water_withdrawal_violation_pos_cost: 0.0,
            water_withdrawal_violation_neg_cost: 0.0,
            evaporation_violation_pos_cost: 0.0,
            evaporation_violation_neg_cost: 0.0,
            inflow_nonnegativity_cost: 1000.0,
        }
    }

    fn make_hydro(id: i32, model: HydroGenerationModel) -> Hydro {
        let mut hydro = Hydro {
            unit_groups: Vec::new(),
            id: EntityId::from(id),
            name: format!("Hydro{id}"),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            downstream_id: None,
            travel_time_hours: None,
            entry_stage_id: None,
            exit_stage_id: None,
            min_storage_hm3: 100.0,
            max_storage_hm3: 2000.0,
            min_outflow_m3s: 0.0,
            max_outflow_m3s: None,
            generation_model: model,
            min_turbined_m3s: 0.0,
            max_turbined_m3s: 500.0,
            specific_productivity_mw_per_m3s_per_m: None,
            min_generation_mw: 0.0,
            max_generation_mw: 1000.0,
            tailrace: None,
            hydraulic_losses: None,
            efficiency: None,
            evaporation_coefficients_mm: None,
            evaporation_reference_volumes_hm3: None,
            diversion: None,
            filling: None,
            penalties: zero_penalties(),
            tailrace_flow: TailraceFlow::default(),
        };
        hydro.declare_mirror_unit_group(EntityId::from(10));
        hydro
    }

    /// Build a minimal single-bus `System` with the given hydros and one study stage.
    ///
    /// Uses bus `EntityId(10)` to match the `make_hydro` helper's `bus_id`.
    fn make_system_for_summary(hydros: Vec<Hydro>) -> novomodelo_core::System {
        make_system_with_bounds(hydros, 1, |_| {})
    }

    /// Build the single-bus `System` of [`make_system_for_summary`] with `n_stages`
    /// one-block study stages and a resolved bounds table whose every (hydro, stage)
    /// holds capacity, then let `edit` change it. `edit` addresses hydros by
    /// canonical position.
    fn make_system_with_bounds(
        hydros: Vec<Hydro>,
        n_stages: usize,
        edit: impl FnOnce(&mut ResolvedBounds),
    ) -> novomodelo_core::System {
        let bus = Bus {
            id: EntityId(10),
            name: "B10".to_string(),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 1000.0,
            }],
            excess_cost: 0.0,
        };
        let first_start = NaiveDate::from_ymd_opt(2024, 1, 1).unwrap();
        let stages = (0..n_stages)
            .map(|index| {
                let months = u32::try_from(index).unwrap();
                Stage {
                    index,
                    id: i32::try_from(index).unwrap(),
                    start_date: first_start + Months::new(months),
                    end_date: first_start + Months::new(months + 1),
                    season_id: Some(0),
                    blocks: vec![Block {
                        index: 0,
                        name: "S".to_string(),
                        duration_hours: 744.0,
                    }],
                    block_mode: BlockMode::Parallel,
                    state_config: StageStateConfig {
                        storage: true,
                        inflow_lags: false,
                    },
                    risk_config: StageRiskConfig::Expectation,
                    scenario_config: ScenarioSourceConfig {
                        branching_factor: 3,
                        noise_method: NoiseMethod::Saa,
                    },
                }
            })
            .collect();
        let mut bounds = ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: hydros.len(),
                n_thermals: 0,
                n_lines: 0,
                n_pumping: 0,
                n_contracts: 0,
                n_stages,
                k_max: 0,
            },
            &BoundsDefaults {
                hydro: HydroStageBounds {
                    min_storage_hm3: 0.0,
                    max_storage_hm3: 0.0,
                    filling_min_rate_m3s: 0.0,
                    water_withdrawal_m3s: 0.0,
                },
                hydro_block: HydroBlockBounds {
                    max_turbined_m3s: 500.0,
                    max_generation_mw: 1000.0,
                    ..HydroBlockBounds::default()
                },
                thermal_block: ThermalBlockBounds {
                    min_generation_mw: 0.0,
                    max_generation_mw: 0.0,
                    cost_per_mwh: 0.0,
                },
                line_block: LineBlockBounds {
                    direct_mw: 0.0,
                    reverse_mw: 0.0,
                },
                pumping_block: PumpingBlockBounds {
                    min_flow_m3s: 0.0,
                    max_flow_m3s: 0.0,
                },
                contract_block: ContractBlockBounds {
                    min_mw: 0.0,
                    max_mw: 0.0,
                    price_per_mwh: 0.0,
                },
            },
        );
        edit(&mut bounds);
        SystemBuilder::new()
            .buses(vec![bus])
            .hydros(hydros)
            .stages(stages)
            .bounds(bounds)
            .correlation(CorrelationModel::default())
            .build()
            .unwrap()
    }

    /// Build a `PrepareHydroModelsResult` for a set of hydro IDs where every
    /// hydro uses constant productivity and has no evaporation.
    fn make_result_all_constant(hydro_ids: &[i32]) -> PrepareHydroModelsResult {
        let n_hydros = hydro_ids.len();
        let n_stages = 1;
        let models: Vec<Vec<ResolvedProductionModel>> = hydro_ids
            .iter()
            .map(|_| {
                (0..n_stages)
                    .map(|_| ResolvedProductionModel::ConstantProductivity { productivity: 0.95 })
                    .collect()
            })
            .collect();
        let production = ProductionModelSet::new(models, &minimal_hydros(n_hydros), n_stages);
        let production_sources = hydro_ids
            .iter()
            .map(|&id| (EntityId(id), ProductionModelSource::DefaultConstant))
            .collect();
        let evaporation_sources = hydro_ids
            .iter()
            .map(|&id| (EntityId(id), EvaporationSource::NotModeled))
            .collect();
        let evaporation_reference_sources: Vec<(EntityId, EvaporationReferenceSource)> = hydro_ids
            .iter()
            .map(|&id| (EntityId(id), EvaporationReferenceSource::DefaultMidpoint))
            .collect();
        let evap_models: Vec<EvaporationModel> =
            hydro_ids.iter().map(|_| EvaporationModel::None).collect();
        PrepareHydroModelsResult {
            production,
            productivity_override: HydroEnergyProductivityOverride::default(),
            evaporation: EvaporationModelSet::new(evap_models),
            provenance: HydroModelProvenance {
                production_sources,
                evaporation_sources,
                evaporation_reference_sources,
            },
            fpha_export_rows: Vec::new(),
            reference_volumes_hm3: Vec::new(),
            vha_geometry_by_hydro: std::collections::HashMap::new(),
            fpha_fit_deviations: Vec::new(),
            fpha_deviation_point_rows: Vec::new(),
        }
    }

    /// Build a `PrepareHydroModelsResult` with mixed constant and FPHA hydros.
    ///
    /// The hydro list is merged and sorted by id ascending (canonical order).
    /// Each FPHA hydro gets `n_planes` hyperplanes at the single study stage.
    fn make_result_mixed(
        constant_ids: &[i32],
        fpha_ids: &[i32],
        n_planes: usize,
    ) -> PrepareHydroModelsResult {
        let n_stages = 1;
        let fpha_plane = FphaPlane {
            intercept: 1000.0,
            gamma_v: 0.002,
            gamma_q: 0.85,
            gamma_s: -0.01,
        };
        let mut all_ids: Vec<(i32, bool)> = constant_ids
            .iter()
            .map(|&id| (id, false))
            .chain(fpha_ids.iter().map(|&id| (id, true)))
            .collect();
        all_ids.sort_by_key(|(id, _)| *id);

        let n_hydros = all_ids.len();
        let models: Vec<Vec<ResolvedProductionModel>> = all_ids
            .iter()
            .map(|(_, is_fpha)| {
                (0..n_stages)
                    .map(|_| {
                        if *is_fpha {
                            ResolvedProductionModel::Fpha {
                                planes: vec![fpha_plane; n_planes],
                            }
                        } else {
                            ResolvedProductionModel::ConstantProductivity { productivity: 0.95 }
                        }
                    })
                    .collect()
            })
            .collect();
        let production = ProductionModelSet::new(models, &minimal_hydros(n_hydros), n_stages);
        let production_sources: Vec<(EntityId, ProductionModelSource)> = all_ids
            .iter()
            .map(|(id, is_fpha)| {
                let src = if *is_fpha {
                    ProductionModelSource::PrecomputedHyperplanes
                } else {
                    ProductionModelSource::DefaultConstant
                };
                (EntityId(*id), src)
            })
            .collect();
        let evaporation_sources: Vec<(EntityId, EvaporationSource)> = all_ids
            .iter()
            .map(|(id, _)| (EntityId(*id), EvaporationSource::NotModeled))
            .collect();
        let evaporation_reference_sources: Vec<(EntityId, EvaporationReferenceSource)> = all_ids
            .iter()
            .map(|(id, _)| (EntityId(*id), EvaporationReferenceSource::DefaultMidpoint))
            .collect();
        let evap_models: Vec<EvaporationModel> =
            all_ids.iter().map(|_| EvaporationModel::None).collect();
        PrepareHydroModelsResult {
            production,
            productivity_override: HydroEnergyProductivityOverride::default(),
            evaporation: EvaporationModelSet::new(evap_models),
            provenance: HydroModelProvenance {
                production_sources,
                evaporation_sources,
                evaporation_reference_sources,
            },
            fpha_export_rows: Vec::new(),
            reference_volumes_hm3: Vec::new(),
            vha_geometry_by_hydro: std::collections::HashMap::new(),
            fpha_fit_deviations: Vec::new(),
            fpha_deviation_point_rows: Vec::new(),
        }
    }

    /// Build a `PrepareHydroModelsResult` with evaporation for a subset of hydros.
    fn make_result_with_evaporation(
        hydro_ids: &[i32],
        evap_ids: &[i32],
    ) -> PrepareHydroModelsResult {
        let n_hydros = hydro_ids.len();
        let n_stages = 1;
        let models: Vec<Vec<ResolvedProductionModel>> = hydro_ids
            .iter()
            .map(|_| {
                (0..n_stages)
                    .map(|_| ResolvedProductionModel::ConstantProductivity { productivity: 0.95 })
                    .collect()
            })
            .collect();
        let production = ProductionModelSet::new(models, &minimal_hydros(n_hydros), n_stages);
        let production_sources = hydro_ids
            .iter()
            .map(|&id| (EntityId(id), ProductionModelSource::DefaultConstant))
            .collect();
        let evap_set: std::collections::HashSet<i32> = evap_ids.iter().copied().collect();
        let evaporation_sources: Vec<(EntityId, EvaporationSource)> = hydro_ids
            .iter()
            .map(|&id| {
                let src = if evap_set.contains(&id) {
                    EvaporationSource::LinearizedFromGeometry
                } else {
                    EvaporationSource::NotModeled
                };
                (EntityId(id), src)
            })
            .collect();
        let evap_models: Vec<EvaporationModel> = hydro_ids
            .iter()
            .map(|&id| {
                if evap_set.contains(&id) {
                    EvaporationModel::Linearized {
                        coefficients: vec![LinearizedEvaporation {
                            intercept_m3s: 1.0,
                            volume_slope_m3s_per_hm3: 0.01,
                            coefficient_sign: EvaporationCoefficientSign::Positive,
                        }],
                        reference_volumes_hm3: vec![500.0],
                    }
                } else {
                    EvaporationModel::None
                }
            })
            .collect();
        // All test hydros use the default midpoint (no per-season volumes in this fixture).
        let evaporation_reference_sources: Vec<(EntityId, EvaporationReferenceSource)> = hydro_ids
            .iter()
            .map(|&id| (EntityId(id), EvaporationReferenceSource::DefaultMidpoint))
            .collect();
        PrepareHydroModelsResult {
            production,
            productivity_override: HydroEnergyProductivityOverride::default(),
            evaporation: EvaporationModelSet::new(evap_models),
            provenance: HydroModelProvenance {
                production_sources,
                evaporation_sources,
                evaporation_reference_sources,
            },
            fpha_export_rows: Vec::new(),
            reference_volumes_hm3: Vec::new(),
            vha_geometry_by_hydro: std::collections::HashMap::new(),
            fpha_fit_deviations: Vec::new(),
            fpha_deviation_point_rows: Vec::new(),
        }
    }

    /// build_hydro_model_summary counts n_user_supplied_ref and n_default_midpoint_ref correctly.
    #[test]
    fn build_hydro_model_summary_ref_source_counts() {
        // 3 hydros: IDs 1, 2, 3.
        // IDs 1 and 2 have evaporation (1=UserSupplied, 2=DefaultMidpoint).
        // ID 3 has no evaporation (DefaultMidpoint, irrelevant for count).
        let hydro_ids = [1i32, 2, 3];
        let hydros = hydro_ids
            .iter()
            .map(|&id| make_hydro(id, HydroGenerationModel::ConstantProductivity))
            .collect();
        let system = make_system_for_summary(hydros);

        let n_hydros = hydro_ids.len();
        let n_stages = 1;
        let models: Vec<Vec<ResolvedProductionModel>> = (0..n_hydros)
            .map(|_| vec![ResolvedProductionModel::ConstantProductivity { productivity: 0.95 }])
            .collect();
        let production = ProductionModelSet::new(models, system.hydros(), n_stages);
        let production_sources = hydro_ids
            .iter()
            .map(|&id| (EntityId(id), ProductionModelSource::DefaultConstant))
            .collect();

        let evaporation_sources = vec![
            (EntityId(1), EvaporationSource::LinearizedFromGeometry),
            (EntityId(2), EvaporationSource::LinearizedFromGeometry),
            (EntityId(3), EvaporationSource::NotModeled),
        ];
        let evaporation_reference_sources = vec![
            (EntityId(1), EvaporationReferenceSource::UserSupplied),
            (EntityId(2), EvaporationReferenceSource::DefaultMidpoint),
            (EntityId(3), EvaporationReferenceSource::DefaultMidpoint),
        ];
        let evap_models = vec![
            EvaporationModel::Linearized {
                coefficients: vec![LinearizedEvaporation {
                    intercept_m3s: 1.0,
                    volume_slope_m3s_per_hm3: 0.01,
                    coefficient_sign: EvaporationCoefficientSign::Positive,
                }],
                reference_volumes_hm3: vec![200.0],
            },
            EvaporationModel::Linearized {
                coefficients: vec![LinearizedEvaporation {
                    intercept_m3s: 1.0,
                    volume_slope_m3s_per_hm3: 0.01,
                    coefficient_sign: EvaporationCoefficientSign::Positive,
                }],
                reference_volumes_hm3: vec![300.0],
            },
            EvaporationModel::None,
        ];

        let result = PrepareHydroModelsResult {
            production,
            productivity_override: HydroEnergyProductivityOverride::default(),
            evaporation: EvaporationModelSet::new(evap_models),
            provenance: HydroModelProvenance {
                production_sources,
                evaporation_sources,
                evaporation_reference_sources,
            },
            fpha_export_rows: Vec::new(),
            reference_volumes_hm3: Vec::new(),
            vha_geometry_by_hydro: std::collections::HashMap::new(),
            fpha_fit_deviations: Vec::new(),
            fpha_deviation_point_rows: Vec::new(),
        };

        let summary = build_hydro_model_summary(&result, &system);

        assert_eq!(summary.n_evaporation, 2, "n_evaporation must be 2");
        assert_eq!(summary.n_no_evaporation, 1, "n_no_evaporation must be 1");
        assert_eq!(
            summary.n_user_supplied_ref, 1,
            "n_user_supplied_ref must be 1 (ID 1)"
        );
        assert_eq!(
            summary.n_default_midpoint_ref, 1,
            "n_default_midpoint_ref must be 1 (ID 2; ID 3 has no evaporation)"
        );
    }

    /// All-constant system: n_constant = 4, n_fpha = 0, total_planes = 0.
    #[test]
    fn build_hydro_model_summary_all_constant() {
        let hydro_ids = [1i32, 2, 3, 4];
        let hydros = hydro_ids
            .iter()
            .map(|&id| make_hydro(id, HydroGenerationModel::ConstantProductivity))
            .collect();
        let system = make_system_for_summary(hydros);
        let result = make_result_all_constant(&hydro_ids);

        let summary = build_hydro_model_summary(&result, &system);

        assert_eq!(
            summary.n_constant, 4,
            "all-constant system must have n_constant = 4"
        );
        assert_eq!(
            summary.n_fpha, 0,
            "all-constant system must have n_fpha = 0"
        );
        assert_eq!(
            summary.total_planes, 0,
            "all-constant system must have total_planes = 0"
        );
        assert!(
            summary.fpha_details.is_empty(),
            "all-constant system must have empty fpha_details"
        );
    }

    /// Mixed system (2 constant + 2 FPHA, 5 planes each): correct counts and plane total.

    #[test]
    fn build_hydro_model_summary_mixed_counts_and_plane_total() {
        let constant_ids = [1i32, 2];
        let fpha_ids = [3i32, 4];
        let all_ids = [1i32, 2, 3, 4];
        let hydros = all_ids
            .iter()
            .map(|&id| make_hydro(id, HydroGenerationModel::ConstantProductivity))
            .collect();
        let system = make_system_for_summary(hydros);
        let result = make_result_mixed(&constant_ids, &fpha_ids, 5);

        let summary = build_hydro_model_summary(&result, &system);

        assert_eq!(summary.n_constant, 2, "n_constant must be 2");
        assert_eq!(summary.n_fpha, 2, "n_fpha must be 2");
        assert_eq!(
            summary.total_planes, 10,
            "total_planes must be 10 (2 × 5 planes)"
        );
        assert_eq!(
            summary.fpha_details.len(),
            2,
            "must have 2 fpha_details entries"
        );
        for detail in &summary.fpha_details {
            assert_eq!(
                detail.n_planes, 5,
                "each FPHA detail must have n_planes = 5"
            );
        }
    }

    /// 2 constant + 2 FPHA (10 planes) + 3 evaporation / 1 without.

    #[test]
    // Rationale: the test exercises constant + FPHA + evaporation counts
    // in a single fixture; splitting into helpers would obscure the coverage
    // contract.
    #[allow(clippy::too_many_lines)]
    fn build_hydro_model_summary_mixed_system() {
        // IDs 1,2 constant; IDs 3,4 FPHA with 5 planes each.
        // IDs 1,2,3 have evaporation; ID 4 does not.
        let constant_ids = [1i32, 2];
        let fpha_ids = [3i32, 4];
        let all_ids = [1i32, 2, 3, 4];
        let hydros = all_ids
            .iter()
            .map(|&id| make_hydro(id, HydroGenerationModel::ConstantProductivity))
            .collect();
        let system = make_system_for_summary(hydros);

        let n_stages = 1;
        let fpha_plane = FphaPlane {
            intercept: 1000.0,
            gamma_v: 0.002,
            gamma_q: 0.85,
            gamma_s: -0.01,
        };
        let mut sorted: Vec<(i32, bool)> = constant_ids
            .iter()
            .map(|&id| (id, false))
            .chain(fpha_ids.iter().map(|&id| (id, true)))
            .collect();
        sorted.sort_by_key(|(id, _)| *id);

        let models: Vec<Vec<ResolvedProductionModel>> = sorted
            .iter()
            .map(|(_, is_fpha)| {
                (0..n_stages)
                    .map(|_| {
                        if *is_fpha {
                            ResolvedProductionModel::Fpha {
                                planes: vec![fpha_plane; 5],
                            }
                        } else {
                            ResolvedProductionModel::ConstantProductivity { productivity: 0.95 }
                        }
                    })
                    .collect()
            })
            .collect();
        let production = ProductionModelSet::new(models, system.hydros(), n_stages);
        let production_sources: Vec<(EntityId, ProductionModelSource)> = sorted
            .iter()
            .map(|(id, is_fpha)| {
                (
                    EntityId(*id),
                    if *is_fpha {
                        ProductionModelSource::PrecomputedHyperplanes
                    } else {
                        ProductionModelSource::DefaultConstant
                    },
                )
            })
            .collect();
        let evap_set: std::collections::HashSet<i32> = [1, 2, 3].into_iter().collect();
        let evaporation_sources: Vec<(EntityId, EvaporationSource)> = sorted
            .iter()
            .map(|(id, _)| {
                (
                    EntityId(*id),
                    if evap_set.contains(id) {
                        EvaporationSource::LinearizedFromGeometry
                    } else {
                        EvaporationSource::NotModeled
                    },
                )
            })
            .collect();
        let evap_models: Vec<EvaporationModel> = sorted
            .iter()
            .map(|(id, _)| {
                if evap_set.contains(id) {
                    EvaporationModel::Linearized {
                        coefficients: vec![LinearizedEvaporation {
                            intercept_m3s: 1.0,
                            volume_slope_m3s_per_hm3: 0.01,
                            coefficient_sign: EvaporationCoefficientSign::Positive,
                        }],
                        reference_volumes_hm3: vec![500.0],
                    }
                } else {
                    EvaporationModel::None
                }
            })
            .collect();
        // All test hydros use the default midpoint (no per-season volumes in this fixture).
        let evaporation_reference_sources: Vec<(EntityId, EvaporationReferenceSource)> = sorted
            .iter()
            .map(|(id, _)| (EntityId(*id), EvaporationReferenceSource::DefaultMidpoint))
            .collect();
        let result = PrepareHydroModelsResult {
            production,
            productivity_override: HydroEnergyProductivityOverride::default(),
            evaporation: EvaporationModelSet::new(evap_models),
            provenance: HydroModelProvenance {
                production_sources,
                evaporation_sources,
                evaporation_reference_sources,
            },
            fpha_export_rows: Vec::new(),
            reference_volumes_hm3: Vec::new(),
            vha_geometry_by_hydro: std::collections::HashMap::new(),
            fpha_fit_deviations: Vec::new(),
            fpha_deviation_point_rows: Vec::new(),
        };

        let summary = build_hydro_model_summary(&result, &system);

        assert_eq!(summary.n_constant, 2, "n_constant must be 2");
        assert_eq!(summary.n_fpha, 2, "n_fpha must be 2");
        assert_eq!(summary.total_planes, 10, "total_planes must be 10");
        assert_eq!(summary.n_evaporation, 3, "n_evaporation must be 3");
        assert_eq!(summary.n_no_evaporation, 1, "n_no_evaporation must be 1");
    }

    /// Evaporation counts are derived from provenance, not model variant.
    #[test]
    fn build_hydro_model_summary_evaporation_counts_from_provenance() {
        let hydro_ids = [1i32, 2, 3, 4];
        let evap_ids = [1i32, 3];
        let hydros = hydro_ids
            .iter()
            .map(|&id| make_hydro(id, HydroGenerationModel::ConstantProductivity))
            .collect();
        let system = make_system_for_summary(hydros);
        let result = make_result_with_evaporation(&hydro_ids, &evap_ids);

        let summary = build_hydro_model_summary(&result, &system);

        assert_eq!(
            summary.n_evaporation, 2,
            "n_evaporation must be 2 (IDs 1 and 3)"
        );
        assert_eq!(
            summary.n_no_evaporation, 2,
            "n_no_evaporation must be 2 (IDs 2 and 4)"
        );
    }

    /// Summary with one computed-source hydro: `n_fpha == 1` and
    /// `fpha_details[0].source == ComputedFromGeometry`.
    #[test]
    fn computed_source_in_summary_counts_correctly() {
        // Single hydro with Fpha generation model (needed so build_hydro_model_summary
        // can look up the name from the system entity list).
        let hydro = make_hydro(5, HydroGenerationModel::Fpha);
        let system = make_system_for_summary(vec![hydro]);

        let fpha_plane = FphaPlane {
            intercept: 800.0,
            gamma_v: 0.003,
            gamma_q: 0.90,
            gamma_s: -0.005,
        };
        let n_planes = 4;
        let production = ProductionModelSet::new(
            vec![vec![ResolvedProductionModel::Fpha {
                planes: vec![fpha_plane; n_planes],
            }]],
            system.hydros(),
            1,
        );

        let result = PrepareHydroModelsResult {
            production,
            productivity_override: HydroEnergyProductivityOverride::default(),
            evaporation: EvaporationModelSet::new(vec![EvaporationModel::None]),
            provenance: HydroModelProvenance {
                production_sources: vec![(
                    EntityId::from(5),
                    ProductionModelSource::ComputedFromGeometry,
                )],
                evaporation_sources: vec![(EntityId::from(5), EvaporationSource::NotModeled)],
                evaporation_reference_sources: vec![(
                    EntityId::from(5),
                    EvaporationReferenceSource::DefaultMidpoint,
                )],
            },
            fpha_export_rows: Vec::new(),
            reference_volumes_hm3: Vec::new(),
            vha_geometry_by_hydro: std::collections::HashMap::new(),
            fpha_fit_deviations: Vec::new(),
            fpha_deviation_point_rows: Vec::new(),
        };

        let summary = build_hydro_model_summary(&result, &system);

        assert_eq!(
            summary.n_fpha, 1,
            "n_fpha must be 1 for one computed-source hydro"
        );
        assert_eq!(summary.n_constant, 0, "n_constant must be 0");
        assert_eq!(
            summary.total_planes, n_planes,
            "total_planes must equal the plane count from the representative stage"
        );
        assert_eq!(
            summary.fpha_details.len(),
            1,
            "must have one fpha_details entry"
        );
        assert_eq!(
            summary.fpha_details[0].source,
            ProductionModelSource::ComputedFromGeometry,
            "fpha_details[0].source must be ComputedFromGeometry"
        );
        assert_eq!(
            summary.fpha_details[0].n_planes, n_planes,
            "fpha_details[0].n_planes must match the fitted plane count"
        );
    }

    fn summary_with_no_turbine_capacity_out_of_id_order() -> HydroModelSummary {
        let hydro_ids = [1i32, 2, 3];
        let hydros = hydro_ids
            .iter()
            .map(|&id| make_hydro(id, HydroGenerationModel::Fpha))
            .collect();
        let system = make_system_for_summary(hydros);
        let mut result = make_result_all_constant(&hydro_ids);
        result.provenance.production_sources = vec![
            (EntityId(3), ProductionModelSource::NoTurbineCapacity),
            (EntityId(1), ProductionModelSource::NoTurbineCapacity),
            (EntityId(2), ProductionModelSource::DefaultConstant),
        ];
        build_hydro_model_summary(&result, &system)
    }

    #[test]
    fn no_turbine_capacity_lists_plants_ascending_by_hydro_id() {
        let summary = summary_with_no_turbine_capacity_out_of_id_order();

        assert_eq!(
            summary.no_turbine_capacity,
            vec![
                NoTurbineCapacityHydro {
                    hydro_id: EntityId(1),
                    name: "Hydro1".to_string(),
                },
                NoTurbineCapacityHydro {
                    hydro_id: EntityId(3),
                    name: "Hydro3".to_string(),
                },
            ]
        );
        assert_eq!(summary.n_constant, 3);
    }

    #[test]
    fn no_turbine_capacity_serializes_as_an_always_present_list() {
        let listed = serde_json::to_value(summary_with_no_turbine_capacity_out_of_id_order())
            .expect("summary serializes");
        assert_eq!(
            listed["no_turbine_capacity"],
            serde_json::json!([
                {"hydro_id": 1, "name": "Hydro1"},
                {"hydro_id": 3, "name": "Hydro3"},
            ])
        );

        let hydro_ids = [1i32, 2];
        let hydros = hydro_ids
            .iter()
            .map(|&id| make_hydro(id, HydroGenerationModel::ConstantProductivity))
            .collect();
        let system = make_system_for_summary(hydros);
        let empty = serde_json::to_value(build_hydro_model_summary(
            &make_result_all_constant(&hydro_ids),
            &system,
        ))
        .expect("summary serializes");
        assert_eq!(empty["no_turbine_capacity"], serde_json::json!([]));
    }

    fn hydro_with_groups(id: i32, groups: Vec<HydroUnitGroup>) -> Hydro {
        let mut hydro = make_hydro(id, HydroGenerationModel::ConstantProductivity);
        hydro.unit_groups = groups;
        hydro.sort_unit_groups();
        hydro
    }

    fn dropped(
        floor: DroppedFloorKind,
        group_id: Option<i32>,
        value: f64,
        reason: DroppedFloorReason,
    ) -> DroppedFloor {
        DroppedFloor {
            stage_id: 0,
            block_id: 0,
            floor,
            level: if group_id.is_some() {
                DroppedFloorLevel::UnitGroup
            } else {
                DroppedFloorLevel::Plant
            },
            hydro_unit_group_id: group_id.map(EntityId),
            bus_id: group_id.map(|_| EntityId(10)),
            value,
            reason,
        }
    }

    fn frozen_row(
        hydro_id: i32,
        no_capacity_stage_ids: Vec<i32>,
        dropped_floors: Vec<DroppedFloor>,
    ) -> FrozenTurbiningHydro {
        FrozenTurbiningHydro {
            hydro_id: EntityId(hydro_id),
            name: format!("Hydro{hydro_id}"),
            no_capacity_stage_ids,
            partial_capacity_stage_ids: Vec::new(),
            dropped_floors,
        }
    }

    /// derived: hydro 1's resolved `max_generation_mw` is 0 at both stages, so no
    /// (cell, block) of it has capacity; the twin keeps the fixture's 1000 MW.
    #[test]
    fn frozen_turbining_lists_a_plant_without_capacity_at_every_stage() {
        let hydros = [1, 2]
            .map(|id| make_hydro(id, HydroGenerationModel::ConstantProductivity))
            .to_vec();
        let system = make_system_with_bounds(hydros, 2, |bounds| {
            for stage in 0..2 {
                bounds.hydro_block_base_mut(0, stage).max_generation_mw = 0.0;
            }
        });

        let summary = build_hydro_model_summary(&make_result_all_constant(&[1, 2]), &system);

        assert_eq!(
            summary.frozen_turbining,
            vec![frozen_row(1, vec![0, 1], Vec::new())]
        );
    }

    /// derived: group 2's declared `max_generation_mw` is 0, so it has no capacity
    /// while group 1 keeps the one cell alive; its 3 m3/s minimum is the only one
    /// dropped (group 1's 4 m3/s is charged). Raising it to 400 MW gives the twin.
    #[test]
    fn frozen_turbining_records_a_dead_group_of_a_live_cell() {
        let groups = |group_2_max_generation_mw| {
            vec![
                make_unit_group(EntityId(1), EntityId(10), 0.0, 500.0, 4.0, 250.0),
                make_unit_group(
                    EntityId(2),
                    EntityId(10),
                    0.0,
                    group_2_max_generation_mw,
                    3.0,
                    250.0,
                ),
            ]
        };

        let system = make_system_for_summary(vec![hydro_with_groups(1, groups(0.0))]);
        let summary = build_hydro_model_summary(&make_result_all_constant(&[1]), &system);
        assert_eq!(
            summary.frozen_turbining,
            vec![frozen_row(
                1,
                Vec::new(),
                vec![dropped(
                    DroppedFloorKind::MinTurbinedM3s,
                    Some(2),
                    3.0,
                    DroppedFloorReason::NoCapacity,
                )],
            )]
        );

        let twin = make_system_for_summary(vec![hydro_with_groups(1, groups(400.0))]);
        let twin_summary = build_hydro_model_summary(&make_result_all_constant(&[1]), &twin);
        assert!(twin_summary.frozen_turbining.is_empty());
    }

    /// derived: group 2 sits alone on bus 11 with `max_generation_mw` 0, so that
    /// cell is dead while bus 10's cell keeps the plant producing; stage 0 is
    /// partial, not without capacity, and no minimum is declared to drop.
    #[test]
    fn frozen_turbining_lists_a_stage_with_a_dead_cell_as_partial_capacity() {
        let groups = vec![
            make_unit_group(EntityId(1), EntityId(10), 0.0, 500.0, 0.0, 250.0),
            make_unit_group(EntityId(2), EntityId(11), 0.0, 0.0, 0.0, 250.0),
        ];
        let system = make_system_for_summary(vec![hydro_with_groups(1, groups)]);

        let summary = build_hydro_model_summary(&make_result_all_constant(&[1]), &system);

        assert_eq!(summary.frozen_turbining.len(), 1);
        let row = &summary.frozen_turbining[0];
        assert!(row.no_capacity_stage_ids.is_empty());
        assert_eq!(row.partial_capacity_stage_ids, vec![0]);
        assert!(row.dropped_floors.is_empty());
    }

    /// derived: hydro 1 enters at stage 1, so it is `PreFilling` at stage 0 and
    /// its resolved 6 MW minimum is dropped there; capacity is intact, so neither
    /// stage list moves. At stage 1 it produces and the minimum is charged.
    #[test]
    fn frozen_turbining_records_the_floors_of_a_commissioning_plant() {
        let plant = |entry_stage_id| {
            let mut hydro = make_hydro(1, HydroGenerationModel::ConstantProductivity);
            hydro.entry_stage_id = entry_stage_id;
            hydro
        };
        let edit = |bounds: &mut ResolvedBounds| {
            for stage in 0..2 {
                bounds.hydro_block_base_mut(0, stage).min_generation_mw = 6.0;
            }
        };

        let system = make_system_with_bounds(vec![plant(Some(1))], 2, edit);
        let summary = build_hydro_model_summary(&make_result_all_constant(&[1]), &system);
        assert_eq!(
            summary.frozen_turbining,
            vec![frozen_row(
                1,
                Vec::new(),
                vec![dropped(
                    DroppedFloorKind::MinGenerationMw,
                    None,
                    6.0,
                    DroppedFloorReason::Commissioning,
                )],
            )]
        );

        let twin = make_system_with_bounds(vec![plant(None)], 2, edit);
        let twin_summary = build_hydro_model_summary(&make_result_all_constant(&[1]), &twin);
        assert!(twin_summary.frozen_turbining.is_empty());
    }

    /// derived: hydro 7 starts in 2020 and is canonical position 0, hydro 3 position 1;
    /// the rows follow `hydro_id`, not position. Both lack capacity at stage 0
    /// (`max_generation_mw` 0), so hydro 3's minima are all dropped: the plant's
    /// 5 m3/s and 6 MW and its groups' 2 and 1 m3/s. The records follow
    /// (level, group id, floor), not the order the minima are walked in
    /// (turbined, then generation, plant before groups within each).
    #[test]
    fn frozen_turbining_sorts_rows_by_hydro_id_and_records_by_block_level_and_kind() {
        let mut early = make_hydro(7, HydroGenerationModel::ConstantProductivity);
        early.operational_start_date = NaiveDate::from_ymd_opt(2020, 1, 1).unwrap();
        let grouped = hydro_with_groups(
            3,
            vec![
                make_unit_group(EntityId(5), EntityId(10), 0.0, 1000.0, 1.0, 250.0),
                make_unit_group(EntityId(4), EntityId(10), 0.0, 1000.0, 2.0, 250.0),
            ],
        );
        let system = make_system_with_bounds(vec![grouped, early], 1, |bounds| {
            for hydro in 0..2 {
                bounds.hydro_block_base_mut(hydro, 0).max_generation_mw = 0.0;
            }
            let plant = bounds.hydro_block_base_mut(1, 0);
            plant.min_turbined_m3s = 5.0;
            plant.min_generation_mw = 6.0;
        });

        let summary = build_hydro_model_summary(&make_result_all_constant(&[7, 3]), &system);

        let no_capacity = DroppedFloorReason::NoCapacity;
        assert_eq!(
            summary.frozen_turbining,
            vec![
                frozen_row(
                    3,
                    vec![0],
                    vec![
                        dropped(DroppedFloorKind::MinTurbinedM3s, None, 5.0, no_capacity),
                        dropped(DroppedFloorKind::MinGenerationMw, None, 6.0, no_capacity),
                        dropped(DroppedFloorKind::MinTurbinedM3s, Some(4), 2.0, no_capacity),
                        dropped(DroppedFloorKind::MinTurbinedM3s, Some(5), 1.0, no_capacity),
                    ],
                ),
                frozen_row(7, vec![0], Vec::new()),
            ]
        );
    }

    /// derived: the field sits between `no_turbine_capacity` and `n_evaporation` in
    /// the struct and serializes in that order; with capacity everywhere it is `[]`.
    #[test]
    fn frozen_turbining_serializes_after_no_turbine_capacity_as_an_always_present_list() {
        let summary = summary_with_no_turbine_capacity_out_of_id_order();

        let value = serde_json::to_value(&summary).expect("summary serializes");
        assert_eq!(value["frozen_turbining"], serde_json::json!([]));

        let text = serde_json::to_string(&summary).expect("summary serializes");
        let offset = |key: &str| text.find(key).expect("key is serialized");
        assert!(offset("\"no_turbine_capacity\":") < offset("\"frozen_turbining\":"));
        assert!(offset("\"frozen_turbining\":") < offset("\"n_evaporation\":"));
    }
}
