//! `SystemBuilder` — canonical-order assembly and validation of a [`System`].

use std::collections::HashSet;

use chrono::NaiveDate;

use super::System;
use super::validate::{
    CrossRefEntities, build_index, build_stage_index, check_duplicate_stages, check_duplicates,
    validate_cross_references, validate_filling_configs,
};
use crate::{
    Bus, CascadeTopology, CorrelationModel, EnergyContract, EntityId, ExternalLoadRow,
    ExternalNcsRow, ExternalScenarioRow, GenericConstraint, HorizonGraph, Hydro, InflowHistoryRow,
    InflowModel, InitialConditions, Line, LoadModel, NcsModel, NonControllableSource,
    PostStudyStages, PumpingStation, ResolvedBounds, ResolvedGenericConstraintBounds,
    ResolvedLoadFactors, ResolvedNcsBounds, ResolvedNcsFactors, ResolvedPenalties, Stage, Thermal,
    ValidationError,
};

/// Builder for constructing a validated, immutable [`System`].
///
/// All entity collections default to empty; supply only the ones you need.
///
/// # Examples
///
/// ```
/// use chrono::NaiveDate;
/// use cobre_core::{Bus, DeficitSegment, EntityId, SystemBuilder};
///
/// let early = NaiveDate::from_ymd_opt(2024, 1, 1).unwrap();
/// let late = NaiveDate::from_ymd_opt(2024, 2, 1).unwrap();
/// let system = SystemBuilder::new()
///     .buses(vec![
///         Bus { id: EntityId(1), name: "B".to_string(), operational_start_date: late, deficit_segments: vec![], excess_cost: 0.0 },
///         Bus { id: EntityId(2), name: "Z".to_string(), operational_start_date: early, deficit_segments: vec![], excess_cost: 0.0 },
///         Bus { id: EntityId(3), name: "A".to_string(), operational_start_date: early, deficit_segments: vec![], excess_cost: 0.0 },
///     ])
///     .build()
///     .expect("valid system");
///
/// // Canonical ordering: by operational_start_date, then by id; never by name.
/// // The two early-date buses order by id (2 then 3), not by name (which would be A then Z).
/// assert_eq!(system.buses()[0].id, EntityId(2));
/// assert_eq!(system.buses()[1].id, EntityId(3));
/// assert_eq!(system.buses()[2].id, EntityId(1));
/// ```
pub struct SystemBuilder {
    buses: Vec<Bus>,
    lines: Vec<Line>,
    hydros: Vec<Hydro>,
    thermals: Vec<Thermal>,
    pumping_stations: Vec<PumpingStation>,
    contracts: Vec<EnergyContract>,
    non_controllable_sources: Vec<NonControllableSource>,
    stages: Vec<Stage>,
    policy_graph: HorizonGraph,
    penalties: ResolvedPenalties,
    bounds: ResolvedBounds,
    resolved_generic_bounds: ResolvedGenericConstraintBounds,
    resolved_load_factors: ResolvedLoadFactors,
    resolved_ncs_bounds: ResolvedNcsBounds,
    resolved_ncs_factors: ResolvedNcsFactors,
    inflow_models: Vec<InflowModel>,
    load_models: Vec<LoadModel>,
    ncs_models: Vec<NcsModel>,
    correlation: CorrelationModel,
    initial_conditions: InitialConditions,
    generic_constraints: Vec<GenericConstraint>,
    inflow_history: Vec<InflowHistoryRow>,
    external_scenarios: Vec<ExternalScenarioRow>,
    external_load_scenarios: Vec<ExternalLoadRow>,
    external_ncs_scenarios: Vec<ExternalNcsRow>,
    post_study_stages: Option<PostStudyStages>,
}

impl Default for SystemBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl SystemBuilder {
    /// Create a new builder with every collection empty and every field at its default.
    #[must_use]
    pub fn new() -> Self {
        Self {
            buses: Vec::new(),
            lines: Vec::new(),
            hydros: Vec::new(),
            thermals: Vec::new(),
            pumping_stations: Vec::new(),
            contracts: Vec::new(),
            non_controllable_sources: Vec::new(),
            stages: Vec::new(),
            policy_graph: HorizonGraph::default(),
            penalties: ResolvedPenalties::empty(),
            bounds: ResolvedBounds::empty(),
            resolved_generic_bounds: ResolvedGenericConstraintBounds::empty(),
            resolved_load_factors: ResolvedLoadFactors::empty(),
            resolved_ncs_bounds: ResolvedNcsBounds::empty(),
            resolved_ncs_factors: ResolvedNcsFactors::empty(),
            inflow_models: Vec::new(),
            load_models: Vec::new(),
            ncs_models: Vec::new(),
            correlation: CorrelationModel::default(),
            initial_conditions: InitialConditions::default(),
            generic_constraints: Vec::new(),
            inflow_history: Vec::new(),
            external_scenarios: Vec::new(),
            external_load_scenarios: Vec::new(),
            external_ncs_scenarios: Vec::new(),
            post_study_stages: None,
        }
    }

    /// Set the bus collection.
    #[must_use]
    pub fn buses(mut self, buses: Vec<Bus>) -> Self {
        self.buses = buses;
        self
    }

    /// Set the line collection.
    #[must_use]
    pub fn lines(mut self, lines: Vec<Line>) -> Self {
        self.lines = lines;
        self
    }

    /// Set the hydro plant collection.
    #[must_use]
    pub fn hydros(mut self, hydros: Vec<Hydro>) -> Self {
        self.hydros = hydros;
        self
    }

    /// Set the thermal plant collection.
    #[must_use]
    pub fn thermals(mut self, thermals: Vec<Thermal>) -> Self {
        self.thermals = thermals;
        self
    }

    /// Set the pumping station collection.
    #[must_use]
    pub fn pumping_stations(mut self, stations: Vec<PumpingStation>) -> Self {
        self.pumping_stations = stations;
        self
    }

    /// Set the energy contract collection.
    #[must_use]
    pub fn contracts(mut self, contracts: Vec<EnergyContract>) -> Self {
        self.contracts = contracts;
        self
    }

    /// Set the non-controllable source collection.
    #[must_use]
    pub fn non_controllable_sources(mut self, sources: Vec<NonControllableSource>) -> Self {
        self.non_controllable_sources = sources;
        self
    }

    /// Set the stage collection (study and pre-study stages).
    ///
    /// Stages are sorted by `id` in [`build`](Self::build) to canonical order.
    #[must_use]
    pub fn stages(mut self, stages: Vec<Stage>) -> Self {
        self.stages = stages;
        self
    }

    /// Set the policy graph.
    #[must_use]
    pub fn policy_graph(mut self, policy_graph: HorizonGraph) -> Self {
        self.policy_graph = policy_graph;
        self
    }

    /// Set the pre-resolved penalty table.
    #[must_use]
    pub fn penalties(mut self, penalties: ResolvedPenalties) -> Self {
        self.penalties = penalties;
        self
    }

    /// Set the pre-resolved bounds table.
    #[must_use]
    pub fn bounds(mut self, bounds: ResolvedBounds) -> Self {
        self.bounds = bounds;
        self
    }

    /// Set the pre-resolved generic constraint RHS bound table.
    #[must_use]
    pub fn resolved_generic_bounds(
        mut self,
        resolved_generic_bounds: ResolvedGenericConstraintBounds,
    ) -> Self {
        self.resolved_generic_bounds = resolved_generic_bounds;
        self
    }

    /// Set the pre-resolved per-block load scaling factors.
    #[must_use]
    pub fn resolved_load_factors(mut self, resolved_load_factors: ResolvedLoadFactors) -> Self {
        self.resolved_load_factors = resolved_load_factors;
        self
    }

    /// Set the pre-resolved per-stage NCS available generation bounds.
    #[must_use]
    pub fn resolved_ncs_bounds(mut self, resolved_ncs_bounds: ResolvedNcsBounds) -> Self {
        self.resolved_ncs_bounds = resolved_ncs_bounds;
        self
    }

    /// Set the pre-resolved per-block NCS generation scaling factors.
    #[must_use]
    pub fn resolved_ncs_factors(mut self, resolved_ncs_factors: ResolvedNcsFactors) -> Self {
        self.resolved_ncs_factors = resolved_ncs_factors;
        self
    }

    /// Set the PAR(p) inflow model collection.
    #[must_use]
    pub fn inflow_models(mut self, inflow_models: Vec<InflowModel>) -> Self {
        self.inflow_models = inflow_models;
        self
    }

    /// Set the load model collection.
    #[must_use]
    pub fn load_models(mut self, load_models: Vec<LoadModel>) -> Self {
        self.load_models = load_models;
        self
    }

    /// Set the NCS availability noise model collection.
    #[must_use]
    pub fn ncs_models(mut self, ncs_models: Vec<NcsModel>) -> Self {
        self.ncs_models = ncs_models;
        self
    }

    /// Set the correlation model.
    #[must_use]
    pub fn correlation(mut self, correlation: CorrelationModel) -> Self {
        self.correlation = correlation;
        self
    }

    /// Set the initial conditions.
    #[must_use]
    pub fn initial_conditions(mut self, initial_conditions: InitialConditions) -> Self {
        self.initial_conditions = initial_conditions;
        self
    }

    /// Set the generic constraint collection.
    ///
    /// Constraints are sorted by `id` in [`build`](Self::build) to canonical order.
    #[must_use]
    pub fn generic_constraints(mut self, generic_constraints: Vec<GenericConstraint>) -> Self {
        self.generic_constraints = generic_constraints;
        self
    }

    /// Set the raw historical inflow observations; rows must be sorted by
    /// `(hydro_id, start_date)` ascending.
    #[must_use]
    pub fn inflow_history(mut self, rows: Vec<InflowHistoryRow>) -> Self {
        self.inflow_history = rows;
        self
    }

    /// Set the raw external inflow scenario rows; rows must be sorted by
    /// `(stage_id, scenario_id, hydro_id)` ascending.
    #[must_use]
    pub fn external_scenarios(mut self, rows: Vec<ExternalScenarioRow>) -> Self {
        self.external_scenarios = rows;
        self
    }

    /// Set the raw external load scenario rows; rows must be sorted by
    /// `(stage_id, scenario_id, bus_id)` ascending.
    #[must_use]
    pub fn external_load_scenarios(mut self, rows: Vec<ExternalLoadRow>) -> Self {
        self.external_load_scenarios = rows;
        self
    }

    /// Set the raw external NCS scenario rows; rows must be sorted by
    /// `(stage_id, scenario_id, ncs_id)` ascending.
    #[must_use]
    pub fn external_ncs_scenarios(mut self, rows: Vec<ExternalNcsRow>) -> Self {
        self.external_ncs_scenarios = rows;
        self
    }

    /// Set the post-study boundary calendar and cost/bounds; `None` when
    /// `post_study_stages.json` is absent.
    #[must_use]
    pub fn post_study_stages(mut self, post_study_stages: Option<PostStudyStages>) -> Self {
        self.post_study_stages = post_study_stages;
        self
    }

    /// Sort every collection into canonical order, validate, and assemble the
    /// immutable [`System`]. Operational entities sort by
    /// `(operational_start_date, id)`; stages and generic constraints sort by
    /// `id`; the resulting slice position is each entity's canonical index.
    /// All validation errors are collected before returning — no
    /// short-circuiting on the first error.
    ///
    /// # Errors
    ///
    /// Returns `Err(Vec<ValidationError>)` if:
    /// - Any hydro declares no unit groups.
    /// - Duplicate IDs are detected in any entity collection or in the stage collection.
    /// - Any cross-reference field refers to an entity ID that does not exist.
    /// - The hydro cascade graph contains a cycle.
    /// - Any hydro filling configuration is invalid (non-positive inflow or missing
    ///   `entry_stage_id`).
    /// - Any of `inflow_models`, `load_models`, `ncs_models` is not in its documented
    ///   canonical order.
    // Rationale: sort, duplicate/cross-ref/cycle checks, and `System` assembly share
    // one `errors` accumulator and the intermediate index maps; splitting them would
    // thread those through every call and lose the fail-fast-on-duplicates short-circuit.
    #[allow(clippy::too_many_lines)]
    pub fn build(mut self) -> Result<System, Vec<ValidationError>> {
        sort_canonical(&mut self.buses, |b| b.operational_start_date, |b| b.id.0);
        sort_canonical(&mut self.lines, |l| l.operational_start_date, |l| l.id.0);
        sort_canonical(&mut self.hydros, |h| h.operational_start_date, |h| h.id.0);

        let missing_unit_groups: Vec<ValidationError> = self
            .hydros
            .iter()
            .filter(|h| h.unit_groups.is_empty())
            .map(|h| ValidationError::MissingUnitGroups { hydro_id: h.id })
            .collect();
        if !missing_unit_groups.is_empty() {
            return Err(missing_unit_groups);
        }

        for hydro in &mut self.hydros {
            hydro.sort_unit_groups();
        }
        sort_canonical(&mut self.thermals, |t| t.operational_start_date, |t| t.id.0);
        sort_canonical(
            &mut self.pumping_stations,
            |p| p.operational_start_date,
            |p| p.id.0,
        );
        sort_canonical(
            &mut self.contracts,
            |c| c.operational_start_date,
            |c| c.id.0,
        );
        sort_canonical(
            &mut self.non_controllable_sources,
            |n| n.operational_start_date,
            |n| n.id.0,
        );
        self.stages.sort_by_key(|s| s.id);
        for (idx, stage) in self.stages.iter_mut().enumerate() {
            stage.index = idx;
        }
        self.generic_constraints.sort_by_key(|c| c.id.0);

        let mut errors: Vec<ValidationError> = Vec::new();
        check_duplicates(&self.buses, "Bus", &mut errors);
        check_duplicates(&self.lines, "Line", &mut errors);
        check_duplicates(&self.hydros, "Hydro", &mut errors);
        check_duplicates(&self.thermals, "Thermal", &mut errors);
        check_duplicates(&self.pumping_stations, "PumpingStation", &mut errors);
        check_duplicates(&self.contracts, "EnergyContract", &mut errors);
        check_duplicates(
            &self.non_controllable_sources,
            "NonControllableSource",
            &mut errors,
        );
        check_duplicate_stages(&self.stages, &mut errors);
        check_canonical_order(
            &self.inflow_models,
            |m| (m.hydro_id.0, m.stage_id),
            "inflow_models",
            &mut errors,
        );
        check_canonical_order(
            &self.load_models,
            |m| (m.bus_id.0, m.stage_id),
            "load_models",
            &mut errors,
        );
        check_canonical_order(
            &self.ncs_models,
            |m| (m.ncs_id.0, m.stage_id),
            "ncs_models",
            &mut errors,
        );

        if !errors.is_empty() {
            return Err(errors);
        }

        let bus_index = build_index(&self.buses);
        let line_index = build_index(&self.lines);
        let hydro_index = build_index(&self.hydros);
        let thermal_index = build_index(&self.thermals);
        let pumping_station_index = build_index(&self.pumping_stations);
        let contract_index = build_index(&self.contracts);
        let non_controllable_source_index = build_index(&self.non_controllable_sources);

        validate_cross_references(
            &CrossRefEntities {
                lines: &self.lines,
                hydros: &self.hydros,
                thermals: &self.thermals,
                pumping_stations: &self.pumping_stations,
                contracts: &self.contracts,
                non_controllable_sources: &self.non_controllable_sources,
            },
            &bus_index,
            &hydro_index,
            &mut errors,
        );

        if !errors.is_empty() {
            return Err(errors);
        }

        let cascade = CascadeTopology::build(&self.hydros);

        if cascade.topological_order().len() < self.hydros.len() {
            let in_topo: HashSet<EntityId> = cascade.topological_order().iter().copied().collect();
            let mut cycle_ids: Vec<EntityId> = self
                .hydros
                .iter()
                .map(|h| h.id)
                .filter(|id| !in_topo.contains(id))
                .collect();
            cycle_ids.sort_by_key(|id| id.0);
            errors.push(ValidationError::CascadeCycle { cycle_ids });
        }

        validate_filling_configs(&self.hydros, &mut errors);

        if !errors.is_empty() {
            return Err(errors);
        }

        let stage_index = build_stage_index(&self.stages);

        Ok(System {
            buses: self.buses,
            lines: self.lines,
            hydros: self.hydros,
            thermals: self.thermals,
            pumping_stations: self.pumping_stations,
            contracts: self.contracts,
            non_controllable_sources: self.non_controllable_sources,
            bus_index,
            line_index,
            hydro_index,
            thermal_index,
            pumping_station_index,
            contract_index,
            non_controllable_source_index,
            cascade,
            stages: self.stages,
            policy_graph: self.policy_graph,
            stage_index,
            penalties: self.penalties,
            bounds: self.bounds,
            resolved_generic_bounds: self.resolved_generic_bounds,
            resolved_load_factors: self.resolved_load_factors,
            resolved_ncs_bounds: self.resolved_ncs_bounds,
            resolved_ncs_factors: self.resolved_ncs_factors,
            inflow_models: self.inflow_models,
            load_models: self.load_models,
            ncs_models: self.ncs_models,
            correlation: self.correlation,
            initial_conditions: self.initial_conditions,
            generic_constraints: self.generic_constraints,
            inflow_history: self.inflow_history,
            external_scenarios: self.external_scenarios,
            external_load_scenarios: self.external_load_scenarios,
            external_ncs_scenarios: self.external_ncs_scenarios,
            post_study_stages: self.post_study_stages,
        })
    }
}

/// Sort entities by `(operational_start_date, id)`. The `id` tiebreak is unique
/// within an entity type (duplicates are rejected), so this is a total order and
/// upholds the declaration-order hard rule without relying on input order. The
/// secondary key is the id, not the name, because names are user-chosen and vary
/// between authors of the same system.
fn sort_canonical<T>(entities: &mut [T], date: impl Fn(&T) -> NaiveDate, id: impl Fn(&T) -> i32) {
    entities.sort_by_key(|e| (date(e), id(e)));
}

/// Validate that `rows` is non-decreasing under `key`, pushing at most one
/// [`ValidationError::UnsortedModelTable`] naming `table` and the position of
/// the first offending row. A duplicate key tuple is accepted: the scenario
/// model tables are documented as sorted, not unique.
pub(crate) fn check_canonical_order<T, K: Ord>(
    rows: &[T],
    key: impl Fn(&T) -> K,
    table: &'static str,
    errors: &mut Vec<ValidationError>,
) {
    if let Some(idx) = rows.windows(2).position(|w| key(&w[1]) < key(&w[0])) {
        errors.push(ValidationError::UnsortedModelTable {
            table,
            position: idx + 1,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{BusSpec, HydroSpec, MirrorUnitGroup, make_bus, make_hydro};
    use crate::{
        Block, BlockMode, DeficitSegment, HydroPenalties, NoiseMethod, ScenarioSourceConfig,
        StageRiskConfig, StageStateConfig,
    };

    pub(super) fn zero_penalties() -> HydroPenalties {
        HydroPenalties::uniform(0.0)
    }

    /// Given two hydros with no declared `unit_groups`, `build()` returns `Err`
    /// with exactly one `MissingUnitGroups` per offending hydro, naming both ids —
    /// proving errors are collected rather than short-circuited on the first.
    #[test]
    fn test_builder_rejects_hydro_with_no_unit_groups() {
        let alpha = make_hydro(HydroSpec {
            id: 1,
            name: "AlphaPlant".to_string(),
            max_storage_hm3: 1000.0,
            min_generation_mw: 10.0,
            max_generation_mw: 90.0,
            min_turbined_m3s: 5.0,
            max_turbined_m3s: 200.0,
            penalties: zero_penalties(),
            mirror_unit_group: MirrorUnitGroup::None,
            ..Default::default()
        });
        let beta = make_hydro(HydroSpec {
            id: 2,
            name: "BetaPlant".to_string(),
            max_storage_hm3: 1000.0,
            min_generation_mw: 25.0,
            max_generation_mw: 150.0,
            min_turbined_m3s: 15.0,
            max_turbined_m3s: 300.0,
            penalties: zero_penalties(),
            mirror_unit_group: MirrorUnitGroup::None,
            ..Default::default()
        });

        let result = SystemBuilder::new()
            .buses(vec![
                make_bus(BusSpec {
                    id: 10,
                    name: format!("bus-{}", 10),
                    deficit_segments: vec![DeficitSegment {
                        depth_mw: None,
                        cost_per_mwh: 5000.0,
                    }],
                    ..Default::default()
                }),
                make_bus(BusSpec {
                    id: 20,
                    name: format!("bus-{}", 20),
                    deficit_segments: vec![DeficitSegment {
                        depth_mw: None,
                        cost_per_mwh: 5000.0,
                    }],
                    ..Default::default()
                }),
            ])
            .hydros(vec![alpha, beta])
            .build();

        let errors = result.expect_err("hydros with no unit groups must be rejected");
        let missing_ids: Vec<EntityId> = errors
            .iter()
            .map(|e| match e {
                ValidationError::MissingUnitGroups { hydro_id } => *hydro_id,
                other => panic!("expected MissingUnitGroups, got {other:?}"),
            })
            .collect();
        assert_eq!(missing_ids, vec![EntityId(1), EntityId(2)]);
    }

    /// Given the same builder with only the first hydro's groups declared,
    /// `build()` reports exactly the hydro that omitted its group — proving the
    /// filter discriminates rather than rejecting every hydro unconditionally.
    #[test]
    fn test_builder_reports_only_the_hydro_missing_unit_groups() {
        let mut alpha = make_hydro(HydroSpec {
            id: 1,
            name: "AlphaPlant".to_string(),
            max_storage_hm3: 1000.0,
            min_generation_mw: 10.0,
            max_generation_mw: 90.0,
            min_turbined_m3s: 5.0,
            max_turbined_m3s: 200.0,
            penalties: zero_penalties(),
            mirror_unit_group: MirrorUnitGroup::None,
            ..Default::default()
        });
        alpha.declare_mirror_unit_group(EntityId(10));
        let beta = make_hydro(HydroSpec {
            id: 2,
            name: "BetaPlant".to_string(),
            max_storage_hm3: 1000.0,
            min_generation_mw: 25.0,
            max_generation_mw: 150.0,
            min_turbined_m3s: 15.0,
            max_turbined_m3s: 300.0,
            penalties: zero_penalties(),
            mirror_unit_group: MirrorUnitGroup::None,
            ..Default::default()
        });

        let result = SystemBuilder::new()
            .buses(vec![
                make_bus(BusSpec {
                    id: 10,
                    name: format!("bus-{}", 10),
                    deficit_segments: vec![DeficitSegment {
                        depth_mw: None,
                        cost_per_mwh: 5000.0,
                    }],
                    ..Default::default()
                }),
                make_bus(BusSpec {
                    id: 20,
                    name: format!("bus-{}", 20),
                    deficit_segments: vec![DeficitSegment {
                        depth_mw: None,
                        cost_per_mwh: 5000.0,
                    }],
                    ..Default::default()
                }),
            ])
            .hydros(vec![alpha, beta])
            .build();

        let errors = result.expect_err("hydro missing groups must be rejected");
        assert_eq!(errors.len(), 1);
        assert!(matches!(
            errors[0],
            ValidationError::MissingUnitGroups {
                hydro_id: EntityId(2)
            }
        ));
    }

    fn stage_with_index(id: i32, index: usize) -> Stage {
        Stage {
            index,
            id,
            start_date: NaiveDate::from_ymd_opt(2024, 1, 1).expect("valid date"),
            end_date: NaiveDate::from_ymd_opt(2024, 2, 1).expect("valid date"),
            season_id: None,
            blocks: vec![Block {
                index: 0,
                name: "B0".to_string(),
                duration_hours: 744.0,
            }],
            block_mode: BlockMode::Parallel,
            state_config: StageStateConfig {
                storage: true,
                inflow_lags: false,
            },
            risk_config: StageRiskConfig::Expectation,
            scenario_config: ScenarioSourceConfig {
                branching_factor: 1,
                noise_method: NoiseMethod::Saa,
            },
        }
    }

    /// Given stages out of id order with a deliberately wrong incoming `index`,
    /// `build()` overwrites `index` with the post-sort position instead of
    /// preserving what it was handed.
    #[test]
    fn build_assigns_stage_index_by_post_sort_position() {
        let stages = vec![
            stage_with_index(2, 0),
            stage_with_index(0, 0),
            stage_with_index(1, 0),
        ];

        let system = SystemBuilder::new()
            .stages(stages)
            .build()
            .expect("stage-only system is valid");

        let ids: Vec<i32> = system.stages().iter().map(|s| s.id).collect();
        assert_eq!(ids, vec![0, 1, 2]);
        for (i, stage) in system.stages().iter().enumerate() {
            assert_eq!(stage.index, i);
        }
    }

    fn inflow_model(hydro_id: i32, stage_id: i32) -> InflowModel {
        InflowModel {
            hydro_id: EntityId(hydro_id),
            stage_id,
            mean_m3s: 100.0,
            std_m3s: 10.0,
            ar_coefficients: vec![],
            residual_std_ratio: 1.0,
            annual: None,
        }
    }

    fn load_model(bus_id: i32, stage_id: i32) -> LoadModel {
        LoadModel {
            bus_id: EntityId(bus_id),
            stage_id,
            mean_mw: 50.0,
            std_mw: 5.0,
        }
    }

    fn ncs_model(ncs_id: i32, stage_id: i32) -> NcsModel {
        NcsModel {
            ncs_id: EntityId(ncs_id),
            stage_id,
            mean: 0.5,
            std: 0.1,
        }
    }

    /// Given `inflow_models` out of order by `(hydro_id, stage_id)`, `build()`
    /// rejects it naming the offending position.
    #[test]
    fn build_rejects_out_of_order_inflow_models() {
        let result = SystemBuilder::new()
            .inflow_models(vec![inflow_model(2, 0), inflow_model(1, 0)])
            .build();

        let errors = result.expect_err("out-of-order inflow_models must be rejected");
        assert!(
            matches!(
                errors.as_slice(),
                [ValidationError::UnsortedModelTable {
                    table: "inflow_models",
                    position: 1
                }]
            ),
            "expected a single UnsortedModelTable(inflow_models, 1), got {errors:?}"
        );
    }

    /// Given both `load_models` and `ncs_models` out of order, `build()` reports
    /// one `UnsortedModelTable` for each table — the checks do not short-circuit.
    #[test]
    fn build_reports_unsorted_load_and_ncs_tables_together() {
        let result = SystemBuilder::new()
            .load_models(vec![load_model(2, 0), load_model(1, 0)])
            .ncs_models(vec![ncs_model(2, 0), ncs_model(1, 0)])
            .build();

        let errors = result.expect_err("out-of-order load_models and ncs_models must be rejected");
        assert_eq!(errors.len(), 2, "expected one error per table: {errors:?}");
        assert!(matches!(
            errors[0],
            ValidationError::UnsortedModelTable {
                table: "load_models",
                position: 1
            }
        ));
        assert!(matches!(
            errors[1],
            ValidationError::UnsortedModelTable {
                table: "ncs_models",
                position: 1
            }
        ));
    }

    /// Two `inflow_models` rows sharing the same `(hydro_id, stage_id)` are
    /// accepted: the table is documented as sorted, not unique.
    #[test]
    fn build_accepts_duplicate_key_tuple_in_inflow_models() {
        let system = SystemBuilder::new()
            .inflow_models(vec![inflow_model(1, 0), inflow_model(1, 0)])
            .build()
            .expect("a duplicated key tuple must not be rejected as unsorted");

        assert_eq!(system.inflow_models().len(), 2);
    }

    /// Correctly ordered tables for all three model families build
    /// successfully — the happy path, so a too-strict comparison fails loudly.
    #[test]
    fn build_accepts_correctly_ordered_model_tables_for_all_three_families() {
        let system = SystemBuilder::new()
            .inflow_models(vec![
                inflow_model(1, 0),
                inflow_model(1, 1),
                inflow_model(2, 0),
            ])
            .load_models(vec![load_model(1, 0), load_model(2, 0), load_model(2, 1)])
            .ncs_models(vec![ncs_model(1, 0), ncs_model(1, 1), ncs_model(2, 0)])
            .build()
            .expect("correctly ordered model tables must build successfully");

        assert_eq!(system.inflow_models().len(), 3);
        assert_eq!(system.load_models().len(), 3);
        assert_eq!(system.ncs_models().len(), 3);
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use crate::test_support::{
        BusSpec, ContractSpec, HydroSpec, LineSpec, NcsSpec, PumpingSpec, StageSpec, ThermalSpec,
        make_bus, make_contract, make_hydro, make_line, make_ncs, make_pumping_station, make_stage,
        make_thermal,
    };
    use crate::{ConstraintExpression, ContractType, DeficitSegment, SlackConfig};
    use proptest::prelude::*;

    fn date_early() -> NaiveDate {
        NaiveDate::from_ymd_opt(2024, 1, 1).expect("valid date")
    }

    fn date_late() -> NaiveDate {
        NaiveDate::from_ymd_opt(2024, 2, 1).expect("valid date")
    }

    fn generic_constraint(id: i32) -> GenericConstraint {
        GenericConstraint {
            id: EntityId(id),
            name: format!("gc{id}"),
            description: None,
            expression: ConstraintExpression { terms: vec![] },
            slack: SlackConfig {
                enabled: false,
                penalty: None,
            },
            bound_lower_affine: None,
            bound_upper_affine: None,
        }
    }

    // Each operational collection mixes two distinct dates so the primary date key
    // is exercised; the bus set additionally carries a same-date pair ("Z"/"A") so
    // the id-not-name tiebreak is exercised. Declaration order is non-canonical so a
    // permutation that happens to be canonical is not the only case the property
    // sees. Cross-references resolve: bus ids {1,2,3}, hydro ids {1,2}.
    fn reference_buses() -> Vec<Bus> {
        vec![
            make_bus(BusSpec {
                id: 1,
                name: "B".to_string(),
                operational_start_date: date_late(),
                deficit_segments: vec![DeficitSegment {
                    depth_mw: None,
                    cost_per_mwh: 5000.0,
                }],
                ..Default::default()
            }),
            make_bus(BusSpec {
                id: 2,
                name: "Z".to_string(),
                operational_start_date: date_early(),
                deficit_segments: vec![DeficitSegment {
                    depth_mw: None,
                    cost_per_mwh: 5000.0,
                }],
                ..Default::default()
            }),
            make_bus(BusSpec {
                id: 3,
                name: "A".to_string(),
                operational_start_date: date_early(),
                deficit_segments: vec![DeficitSegment {
                    depth_mw: None,
                    cost_per_mwh: 5000.0,
                }],
                ..Default::default()
            }),
        ]
    }

    fn reference_lines() -> Vec<Line> {
        vec![
            make_line(LineSpec {
                id: 1,
                name: "LB".to_string(),
                operational_start_date: date_late(),
                source_bus_id: 1,
                target_bus_id: 2,
                ..Default::default()
            }),
            make_line(LineSpec {
                id: 2,
                name: "LA".to_string(),
                operational_start_date: date_early(),
                source_bus_id: 2,
                target_bus_id: 3,
                ..Default::default()
            }),
        ]
    }

    fn reference_hydros() -> Vec<Hydro> {
        vec![
            make_hydro(HydroSpec {
                id: 1,
                name: "HB".to_string(),
                operational_start_date: date_late(),
                bus_id: 1,
                max_storage_hm3: 1000.0,
                max_turbined_m3s: 100.0,
                max_generation_mw: 100.0,
                penalties: super::tests::zero_penalties(),
                ..Default::default()
            }),
            make_hydro(HydroSpec {
                id: 2,
                name: "HA".to_string(),
                operational_start_date: date_early(),
                bus_id: 2,
                max_storage_hm3: 1000.0,
                max_turbined_m3s: 100.0,
                max_generation_mw: 100.0,
                penalties: super::tests::zero_penalties(),
                ..Default::default()
            }),
        ]
    }

    fn reference_thermals() -> Vec<Thermal> {
        vec![
            make_thermal(ThermalSpec {
                id: 1,
                name: "TB".to_string(),
                operational_start_date: date_late(),
                bus_id: 1,
                cost_per_mwh: 10.0,
                ..Default::default()
            }),
            make_thermal(ThermalSpec {
                id: 2,
                name: "TA".to_string(),
                operational_start_date: date_early(),
                bus_id: 3,
                cost_per_mwh: 10.0,
                ..Default::default()
            }),
        ]
    }

    fn reference_pumping() -> Vec<PumpingStation> {
        vec![
            make_pumping_station(PumpingSpec {
                id: 1,
                name: "PB".to_string(),
                operational_start_date: date_late(),
                bus_id: 1,
                source_hydro_id: 1,
                destination_hydro_id: 2,
                max_flow_m3s: 100.0,
            }),
            make_pumping_station(PumpingSpec {
                id: 2,
                name: "PA".to_string(),
                operational_start_date: date_early(),
                bus_id: 2,
                source_hydro_id: 2,
                destination_hydro_id: 1,
                max_flow_m3s: 100.0,
            }),
        ]
    }

    fn reference_contracts() -> Vec<EnergyContract> {
        vec![
            make_contract(ContractSpec {
                id: 1,
                name: "CB".to_string(),
                operational_start_date: date_late(),
                bus_id: 1,
                contract_type: ContractType::Import,
                price_per_mwh: 100.0,
                ..Default::default()
            }),
            make_contract(ContractSpec {
                id: 2,
                name: "CA".to_string(),
                operational_start_date: date_early(),
                bus_id: 2,
                contract_type: ContractType::Export,
                price_per_mwh: 100.0,
                ..Default::default()
            }),
        ]
    }

    fn reference_ncs() -> Vec<NonControllableSource> {
        vec![
            make_ncs(NcsSpec {
                id: 1,
                name: "NB".to_string(),
                operational_start_date: date_late(),
                bus_id: 1,
                max_generation_mw: 100.0,
                ..Default::default()
            }),
            make_ncs(NcsSpec {
                id: 2,
                name: "NA".to_string(),
                operational_start_date: date_early(),
                bus_id: 3,
                max_generation_mw: 100.0,
                ..Default::default()
            }),
        ]
    }

    fn reference_stages() -> Vec<Stage> {
        vec![
            make_stage(StageSpec {
                id: 3,
                ..Default::default()
            }),
            make_stage(StageSpec {
                id: 1,
                ..Default::default()
            }),
            make_stage(StageSpec {
                id: 2,
                ..Default::default()
            }),
        ]
    }

    fn reference_generic_constraints() -> Vec<GenericConstraint> {
        vec![
            generic_constraint(30),
            generic_constraint(10),
            generic_constraint(20),
        ]
    }

    /// Canonical key the builder applies to operational entities:
    /// `(operational_start_date, id)`. Returned as owned tuples so the projected
    /// post-`build()` order can be compared against the expected order.
    trait OpKey {
        fn op_date(&self) -> NaiveDate;
        fn op_id(&self) -> i32;
    }

    macro_rules! impl_op_key {
        ($t:ty) => {
            impl OpKey for $t {
                fn op_date(&self) -> NaiveDate {
                    self.operational_start_date
                }
                fn op_id(&self) -> i32 {
                    self.id.0
                }
            }
        };
    }

    impl_op_key!(Bus);
    impl_op_key!(Line);
    impl_op_key!(Hydro);
    impl_op_key!(Thermal);
    impl_op_key!(PumpingStation);
    impl_op_key!(EnergyContract);
    impl_op_key!(NonControllableSource);

    fn project_op<T: OpKey>(entities: &[T]) -> Vec<(NaiveDate, i32)> {
        entities.iter().map(|e| (e.op_date(), e.op_id())).collect()
    }

    /// Expected canonical projection: clone the reference set and apply the SAME
    /// `(date, id)` key `build()` uses, then project. Computed, never hand-typed,
    /// so the expectation tracks the contract rather than a guessed order.
    fn expected_op<T: OpKey>(mut reference: Vec<T>) -> Vec<(NaiveDate, i32)> {
        reference.sort_by(|a, b| {
            a.op_date()
                .cmp(&b.op_date())
                .then_with(|| a.op_id().cmp(&b.op_id()))
        });
        project_op(&reference)
    }

    fn expected_stage_ids() -> Vec<i32> {
        let mut s = reference_stages();
        s.sort_by_key(|s| s.id);
        s.iter().map(|s| s.id).collect()
    }

    fn expected_gc_ids() -> Vec<i32> {
        let mut g = reference_generic_constraints();
        g.sort_by_key(|c| c.id.0);
        g.iter().map(|c| c.id.0).collect()
    }

    proptest! {
        /// Declaration-order invariance guard: `SystemBuilder::build()` canonicalizes
        /// every collection identically regardless of input order. Each parameter is
        /// an independent shuffle of a fixed valid reference set — only the ORDER is
        /// random — so each generated `System` stays valid while exercising the sort.
        #[test]
        fn build_canonical_order_invariant_under_input_permutation(
            buses in Just(reference_buses()).prop_shuffle(),
            lines in Just(reference_lines()).prop_shuffle(),
            hydros in Just(reference_hydros()).prop_shuffle(),
            thermals in Just(reference_thermals()).prop_shuffle(),
            pumping in Just(reference_pumping()).prop_shuffle(),
            contracts in Just(reference_contracts()).prop_shuffle(),
            ncs in Just(reference_ncs()).prop_shuffle(),
            stages in Just(reference_stages()).prop_shuffle(),
            gcs in Just(reference_generic_constraints()).prop_shuffle(),
        ) {
            let system = SystemBuilder::new()
                .buses(buses)
                .lines(lines)
                .hydros(hydros)
                .thermals(thermals)
                .pumping_stations(pumping)
                .contracts(contracts)
                .non_controllable_sources(ncs)
                .stages(stages)
                .generic_constraints(gcs)
                .build()
                .expect("reference system is valid");

            let expected_buses = expected_op(reference_buses());
            let expected_lines = expected_op(reference_lines());
            let expected_hydros = expected_op(reference_hydros());
            let expected_thermals = expected_op(reference_thermals());
            let expected_pumping = expected_op(reference_pumping());
            let expected_contracts = expected_op(reference_contracts());
            let expected_ncs = expected_op(reference_ncs());
            let expected_stages = expected_stage_ids();
            let expected_gcs = expected_gc_ids();

            // Sortedness: the precomputed expectation is itself non-decreasing under
            // the canonical key, so a mistake in the expectation cannot mask a sort bug.
            prop_assert!(expected_buses.is_sorted());
            prop_assert!(expected_lines.is_sorted());
            prop_assert!(expected_hydros.is_sorted());
            prop_assert!(expected_thermals.is_sorted());
            prop_assert!(expected_pumping.is_sorted());
            prop_assert!(expected_contracts.is_sorted());
            prop_assert!(expected_ncs.is_sorted());
            prop_assert!(expected_stages.is_sorted());
            prop_assert!(expected_gcs.is_sorted());

            prop_assert_eq!(project_op(system.buses()), expected_buses);
            prop_assert_eq!(project_op(system.lines()), expected_lines);
            prop_assert_eq!(project_op(system.hydros()), expected_hydros);
            prop_assert_eq!(project_op(system.thermals()), expected_thermals);
            prop_assert_eq!(project_op(system.pumping_stations()), expected_pumping);
            prop_assert_eq!(project_op(system.contracts()), expected_contracts);
            prop_assert_eq!(project_op(system.non_controllable_sources()), expected_ncs);
            prop_assert_eq!(
                system.stages().iter().map(|s| s.id).collect::<Vec<_>>(),
                expected_stages
            );
            prop_assert_eq!(
                system.generic_constraints().iter().map(|c| c.id.0).collect::<Vec<_>>(),
                expected_gcs
            );
        }
    }
}
