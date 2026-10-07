//! Test-only entity fixture builders shared across this crate's unit and
//! integration tests, and reachable from downstream crates via the
//! `test-support` feature. Compiles only under `cfg(test)` or that feature;
//! must never be enabled in a production build.
//!
//! Every entity gets a `<Entity>Spec` + `Default` + `make_<entity>` triple: a
//! spec value overrides only the fields a fixture cares about, so a call site
//! that needs nothing unusual writes `make_bus(BusSpec { id: 1, ..Default::default() })`.
//! Each `make_<entity>` destructures its spec with no `..`, so a spec that gains
//! a field is a compile error inside `make_<entity>` until that field is routed
//! to the entity — no field is silently dropped between the spec and the entity.

use chrono::NaiveDate;

use crate::{
    Block, BlockMode, Bus, ContractType, DeficitSegment, EnergyContract, EntityId, Hydro,
    HydroGenerationModel, HydroPenalties, HydroUnitGroup, Line, NoiseMethod, NonControllableSource,
    PumpingStation, ScenarioSourceConfig, Stage, StageRiskConfig, StageStateConfig, Thermal,
};

/// The `NaiveDate` for a calendar-valid `(year, month, day)` triple.
///
/// # Panics
///
/// Never for the literal triples the builders below pass.
#[must_use]
#[allow(clippy::expect_used)]
// Rationale: every call site below passes a calendar-valid literal triple, so
// `from_ymd_opt` cannot return `None`.
pub fn date(year: i32, month: u32, day: u32) -> NaiveDate {
    NaiveDate::from_ymd_opt(year, month, day).expect("caller passes a calendar-valid triple")
}

/// Compares two `f64` values by their IEEE-754 bit pattern, so `NaN == NaN`
/// and `+0.0 != -0.0`.
#[must_use]
pub fn f64_bits_eq(a: f64, b: f64) -> bool {
    a.to_bits() == b.to_bits()
}

/// Compares two `Option<f64>` values by IEEE-754 bit pattern when both are
/// `Some`, so `NaN == NaN` and `+0.0 != -0.0`; `None` equals only `None`.
#[must_use]
pub fn opt_f64_bits_eq(a: Option<f64>, b: Option<f64>) -> bool {
    match (a, b) {
        (None, None) => true,
        (Some(x), Some(y)) => x.to_bits() == y.to_bits(),
        _ => false,
    }
}

/// Fixture fields for [`make_bus`].
#[derive(Debug, Clone)]
pub struct BusSpec {
    /// Bus identifier.
    pub id: i32,
    /// Bus name.
    pub name: String,
    /// Date the bus enters service.
    pub operational_start_date: NaiveDate,
    /// Deficit cost segments, ordered by ascending cost.
    pub deficit_segments: Vec<DeficitSegment>,
    /// Cost per `MWh` for surplus generation absorption.
    pub excess_cost: f64,
}

impl Default for BusSpec {
    fn default() -> Self {
        Self {
            id: 0,
            name: String::new(),
            operational_start_date: date(2024, 1, 1),
            deficit_segments: Vec::new(),
            excess_cost: 0.0,
        }
    }
}

/// Build a [`Bus`] from `spec`.
#[must_use]
pub fn make_bus(
    BusSpec {
        id,
        name,
        operational_start_date,
        deficit_segments,
        excess_cost,
    }: BusSpec,
) -> Bus {
    Bus {
        id: EntityId(id),
        name,
        operational_start_date,
        deficit_segments,
        excess_cost,
    }
}

/// Fixture fields for [`make_line`].
#[derive(Debug, Clone)]
pub struct LineSpec {
    /// Line identifier.
    pub id: i32,
    /// Line name.
    pub name: String,
    /// Date the line enters service.
    pub operational_start_date: NaiveDate,
    /// Source bus for direct flow direction.
    pub source_bus_id: i32,
    /// Target bus for direct flow direction.
    pub target_bus_id: i32,
    /// Maximum flow from source to target.
    pub direct_capacity_mw: f64,
    /// Maximum flow from target to source.
    pub reverse_capacity_mw: f64,
}

impl Default for LineSpec {
    fn default() -> Self {
        Self {
            id: 0,
            name: String::new(),
            operational_start_date: date(2024, 1, 1),
            source_bus_id: 0,
            target_bus_id: 1,
            direct_capacity_mw: 100.0,
            reverse_capacity_mw: 100.0,
        }
    }
}

/// Build a [`Line`] from `spec`.
#[must_use]
pub fn make_line(
    LineSpec {
        id,
        name,
        operational_start_date,
        source_bus_id,
        target_bus_id,
        direct_capacity_mw,
        reverse_capacity_mw,
    }: LineSpec,
) -> Line {
    Line {
        id: EntityId(id),
        name,
        operational_start_date,
        source_bus_id: EntityId(source_bus_id),
        target_bus_id: EntityId(target_bus_id),
        entry_stage_id: None,
        exit_stage_id: None,
        direct_capacity_mw,
        reverse_capacity_mw,
        losses_percent: 0.0,
        exchange_cost: 0.0,
    }
}

/// Where a hydro fixture's implicit unit group is declared — the three states
/// [`Hydro::declare_mirror_unit_group`] callers across this crate use. Not
/// `Option<EntityId>`: that type cannot distinguish "mirror my own bus" from
/// "mirror entity 0".
#[derive(Debug, Clone, Copy, Default)]
pub enum MirrorUnitGroup {
    /// Declared on [`HydroSpec::bus_id`].
    #[default]
    OwnBus,
    /// Declared on a bus independent of [`HydroSpec::bus_id`].
    FixedBus(i32),
    /// No group is declared; `unit_groups` stays empty.
    None,
}

/// Fixture fields for [`make_hydro`].
#[derive(Debug, Clone)]
pub struct HydroSpec {
    /// Hydro plant identifier.
    pub id: i32,
    /// Plant name.
    pub name: String,
    /// Date the plant enters service.
    pub operational_start_date: NaiveDate,
    /// Bus [`MirrorUnitGroup::OwnBus`] mirrors the plant onto.
    pub bus_id: i32,
    /// Downstream cascade plant; `None` = run-of-river or final plant.
    pub downstream_id: Option<i32>,
    /// Minimum turbined flow.
    pub min_turbined_m3s: f64,
    /// Maximum turbined flow (installed turbine capacity).
    pub max_turbined_m3s: f64,
    /// Minimum electrical generation.
    pub min_generation_mw: f64,
    /// Maximum electrical generation (installed capacity).
    pub max_generation_mw: f64,
    /// Maximum operational storage (flood control level).
    pub max_storage_hm3: f64,
    /// Resolved entity-level penalty costs.
    pub penalties: HydroPenalties,
    /// Which unit group, if any, is declared on construction.
    pub mirror_unit_group: MirrorUnitGroup,
}

impl Default for HydroSpec {
    fn default() -> Self {
        Self {
            id: 0,
            name: String::new(),
            operational_start_date: date(2024, 1, 1),
            bus_id: 0,
            downstream_id: None,
            min_turbined_m3s: 0.0,
            max_turbined_m3s: 1.0,
            min_generation_mw: 0.0,
            max_generation_mw: 1.0,
            max_storage_hm3: 1.0,
            penalties: HydroPenalties {
                inflow_nonnegativity_cost: 1000.0,
                ..HydroPenalties::uniform(0.0)
            },
            mirror_unit_group: MirrorUnitGroup::OwnBus,
        }
    }
}

/// Build a [`Hydro`] from `spec`.
#[must_use]
pub fn make_hydro(
    HydroSpec {
        id,
        name,
        operational_start_date,
        bus_id,
        downstream_id,
        min_turbined_m3s,
        max_turbined_m3s,
        min_generation_mw,
        max_generation_mw,
        max_storage_hm3,
        penalties,
        mirror_unit_group,
    }: HydroSpec,
) -> Hydro {
    let mut hydro = Hydro {
        id: EntityId(id),
        name,
        operational_start_date,
        downstream_id: downstream_id.map(EntityId),
        travel_time_hours: None,
        entry_stage_id: None,
        exit_stage_id: None,
        min_storage_hm3: 0.0,
        max_storage_hm3,
        min_outflow_m3s: 0.0,
        max_outflow_m3s: None,
        generation_model: HydroGenerationModel::ConstantProductivity,
        min_turbined_m3s,
        max_turbined_m3s,
        specific_productivity_mw_per_m3s_per_m: None,
        min_generation_mw,
        max_generation_mw,
        unit_groups: Vec::new(),
        tailrace: None,
        hydraulic_losses: None,
        efficiency: None,
        evaporation_coefficients_mm: None,
        evaporation_reference_volumes_hm3: None,
        diversion: None,
        filling: None,
        penalties,
    };
    match mirror_unit_group {
        MirrorUnitGroup::OwnBus => hydro.declare_mirror_unit_group(EntityId(bus_id)),
        MirrorUnitGroup::FixedBus(fixed_bus_id) => {
            hydro.declare_mirror_unit_group(EntityId(fixed_bus_id));
        }
        MirrorUnitGroup::None => {}
    }
    hydro
}

/// Fixture fields for [`make_thermal`].
#[derive(Debug, Clone)]
pub struct ThermalSpec {
    /// Thermal plant identifier.
    pub id: i32,
    /// Plant name.
    pub name: String,
    /// Date the plant enters service.
    pub operational_start_date: NaiveDate,
    /// Bus receiving this plant's generation.
    pub bus_id: i32,
    /// Marginal cost of generation.
    pub cost_per_mwh: f64,
    /// Minimum electrical generation (minimum stable load).
    pub min_generation_mw: f64,
    /// Maximum electrical generation (installed capacity).
    pub max_generation_mw: f64,
}

impl Default for ThermalSpec {
    fn default() -> Self {
        Self {
            id: 0,
            name: String::new(),
            operational_start_date: date(2024, 1, 1),
            bus_id: 0,
            cost_per_mwh: 50.0,
            min_generation_mw: 0.0,
            max_generation_mw: 100.0,
        }
    }
}

/// Build a [`Thermal`] from `spec`.
#[must_use]
pub fn make_thermal(
    ThermalSpec {
        id,
        name,
        operational_start_date,
        bus_id,
        cost_per_mwh,
        min_generation_mw,
        max_generation_mw,
    }: ThermalSpec,
) -> Thermal {
    Thermal {
        id: EntityId(id),
        name,
        operational_start_date,
        bus_id: EntityId(bus_id),
        entry_stage_id: None,
        exit_stage_id: None,
        cost_per_mwh,
        min_generation_mw,
        max_generation_mw,
        anticipated_config: None,
    }
}

/// Fixture fields for [`make_ncs`].
#[derive(Debug, Clone)]
pub struct NcsSpec {
    /// Source identifier.
    pub id: i32,
    /// Source name.
    pub name: String,
    /// Date the source enters service.
    pub operational_start_date: NaiveDate,
    /// Bus receiving this source's generation.
    pub bus_id: i32,
    /// Maximum generation (installed capacity).
    pub max_generation_mw: f64,
    /// Resolved cost per `MWh` of curtailed generation.
    pub curtailment_cost: f64,
}

impl Default for NcsSpec {
    fn default() -> Self {
        Self {
            id: 0,
            name: String::new(),
            operational_start_date: date(2024, 1, 1),
            bus_id: 0,
            max_generation_mw: 50.0,
            curtailment_cost: 0.0,
        }
    }
}

/// Build a [`NonControllableSource`] from `spec`.
#[must_use]
pub fn make_ncs(
    NcsSpec {
        id,
        name,
        operational_start_date,
        bus_id,
        max_generation_mw,
        curtailment_cost,
    }: NcsSpec,
) -> NonControllableSource {
    NonControllableSource {
        id: EntityId(id),
        name,
        operational_start_date,
        bus_id: EntityId(bus_id),
        entry_stage_id: None,
        exit_stage_id: None,
        max_generation_mw,
        allow_curtailment: true,
        curtailment_cost,
    }
}

/// Fixture fields for [`make_contract`].
#[derive(Debug, Clone)]
pub struct ContractSpec {
    /// Contract identifier.
    pub id: i32,
    /// Contract name.
    pub name: String,
    /// Date the contract enters service.
    pub operational_start_date: NaiveDate,
    /// Bus at which the contracted power is injected or withdrawn.
    pub bus_id: i32,
    /// Direction of energy flow for this contract.
    pub contract_type: ContractType,
    /// Contract price per `MWh`.
    pub price_per_mwh: f64,
    /// Maximum contracted power.
    pub max_mw: f64,
}

impl Default for ContractSpec {
    fn default() -> Self {
        Self {
            id: 0,
            name: String::new(),
            operational_start_date: date(2024, 1, 1),
            bus_id: 0,
            contract_type: ContractType::Import,
            price_per_mwh: 0.0,
            max_mw: 100.0,
        }
    }
}

/// Build an [`EnergyContract`] from `spec`.
#[must_use]
pub fn make_contract(
    ContractSpec {
        id,
        name,
        operational_start_date,
        bus_id,
        contract_type,
        price_per_mwh,
        max_mw,
    }: ContractSpec,
) -> EnergyContract {
    EnergyContract {
        id: EntityId(id),
        name,
        operational_start_date,
        bus_id: EntityId(bus_id),
        contract_type,
        entry_stage_id: None,
        exit_stage_id: None,
        price_per_mwh,
        min_mw: 0.0,
        max_mw,
    }
}

/// Fixture fields for [`make_pumping_station`].
#[derive(Debug, Clone)]
pub struct PumpingSpec {
    /// Pumping station identifier.
    pub id: i32,
    /// Pumping station name.
    pub name: String,
    /// Date the station enters service.
    pub operational_start_date: NaiveDate,
    /// Bus from which electrical power is consumed.
    pub bus_id: i32,
    /// Hydro plant from whose reservoir water is extracted.
    pub source_hydro_id: i32,
    /// Hydro plant into whose reservoir water is injected.
    pub destination_hydro_id: i32,
    /// Maximum pumped flow (installed pump capacity).
    pub max_flow_m3s: f64,
}

impl Default for PumpingSpec {
    fn default() -> Self {
        Self {
            id: 0,
            name: String::new(),
            operational_start_date: date(2024, 1, 1),
            bus_id: 0,
            source_hydro_id: 0,
            destination_hydro_id: 1,
            max_flow_m3s: 10.0,
        }
    }
}

/// Build a [`PumpingStation`] from `spec`.
#[must_use]
pub fn make_pumping_station(
    PumpingSpec {
        id,
        name,
        operational_start_date,
        bus_id,
        source_hydro_id,
        destination_hydro_id,
        max_flow_m3s,
    }: PumpingSpec,
) -> PumpingStation {
    PumpingStation {
        id: EntityId(id),
        name,
        operational_start_date,
        bus_id: EntityId(bus_id),
        source_hydro_id: EntityId(source_hydro_id),
        destination_hydro_id: EntityId(destination_hydro_id),
        entry_stage_id: None,
        exit_stage_id: None,
        consumption_mw_per_m3s: 0.5,
        min_flow_m3s: 0.0,
        max_flow_m3s,
    }
}

/// Fixture fields for [`make_unit_group`].
#[derive(Debug, Clone)]
pub struct UnitGroupSpec {
    /// Group identifier, unique within the owning plant.
    pub id: i32,
    /// Group name.
    pub name: String,
    /// Bus to which this group's generation is injected.
    pub bus_id: i32,
    /// Minimum electrical generation.
    pub min_generation_mw: f64,
    /// Maximum electrical generation.
    pub max_generation_mw: f64,
    /// Minimum turbined flow.
    pub min_turbined_m3s: f64,
    /// Maximum turbined flow.
    pub max_turbined_m3s: f64,
}

impl Default for UnitGroupSpec {
    fn default() -> Self {
        Self {
            id: 0,
            name: String::new(),
            bus_id: 0,
            min_generation_mw: 0.0,
            max_generation_mw: 1.0,
            min_turbined_m3s: 0.0,
            max_turbined_m3s: 1.0,
        }
    }
}

/// Build a [`HydroUnitGroup`] from `spec`.
#[must_use]
pub fn make_unit_group(
    UnitGroupSpec {
        id,
        name,
        bus_id,
        min_generation_mw,
        max_generation_mw,
        min_turbined_m3s,
        max_turbined_m3s,
    }: UnitGroupSpec,
) -> HydroUnitGroup {
    HydroUnitGroup {
        id: EntityId(id),
        name,
        bus_id: EntityId(bus_id),
        min_generation_mw,
        max_generation_mw,
        min_turbined_m3s,
        max_turbined_m3s,
    }
}

/// Fixture fields for [`make_stage`].
#[derive(Debug, Clone)]
pub struct StageSpec {
    /// Stage identifier.
    pub id: i32,
    /// Positional index. `None` derives it from `id` — the behaviour every
    /// fixture with contiguous non-negative stage ids wants.
    pub index: Option<usize>,
    /// Stage start date (inclusive).
    pub start_date: NaiveDate,
    /// Stage end date (exclusive).
    pub end_date: NaiveDate,
    /// Season index; `None` for stages without seasonal structure.
    pub season_id: Option<usize>,
    /// Load blocks, sorted by index.
    pub blocks: Vec<Block>,
    /// Scenario source configuration (branching factor and noise method).
    pub scenario_config: ScenarioSourceConfig,
}

impl Default for StageSpec {
    fn default() -> Self {
        Self {
            id: 0,
            index: None,
            start_date: date(2024, 1, 1),
            end_date: date(2024, 2, 1),
            season_id: None,
            blocks: vec![Block {
                index: 0,
                name: "B0".to_string(),
                duration_hours: 744.0,
            }],
            scenario_config: ScenarioSourceConfig {
                branching_factor: 1,
                noise_method: NoiseMethod::Saa,
            },
        }
    }
}

/// Build a [`Stage`] from `spec`.
#[must_use]
pub fn make_stage(
    StageSpec {
        id,
        index,
        start_date,
        end_date,
        season_id,
        blocks,
        scenario_config,
    }: StageSpec,
) -> Stage {
    Stage {
        index: index.unwrap_or_else(|| usize::try_from(id.max(0)).unwrap_or(0)),
        id,
        start_date,
        end_date,
        season_id,
        blocks,
        block_mode: BlockMode::Parallel,
        state_config: StageStateConfig {
            storage: true,
            inflow_lags: false,
        },
        risk_config: StageRiskConfig::Expectation,
        scenario_config,
    }
}

/// A one-block vector — the shape nearly every single-block stage fixture
/// needs — with `index` 0.
#[must_use]
pub fn single_block(name: &str, duration_hours: f64) -> Vec<Block> {
    vec![Block {
        index: 0,
        name: name.to_string(),
        duration_hours,
    }]
}

/// Approximate `erf(x)` using the Horner-form rational approximation
/// (Abramowitz & Stegun 7.1.26, max error 1.5e-7).
#[must_use]
pub fn approx_erf(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0_f64 } else { 1.0_f64 };
    let t = 1.0 / (1.0 + 0.327_591_1 * x.abs());
    let poly = t
        * (0.254_829_592
            + t * (-0.284_496_736
                + t * (1.421_413_741 + t * (-1.453_152_027 + t * 1.061_405_429))));
    sign * (1.0 - poly * (-x * x).exp())
}

/// The standard normal CDF, via [`approx_erf`].
#[must_use]
pub fn norm_cdf(z: f64) -> f64 {
    0.5 * (1.0 + approx_erf(z / std::f64::consts::SQRT_2))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn make_hydro_mirror_unit_group_has_three_states() {
        let own_bus = make_hydro(HydroSpec {
            id: 1,
            bus_id: 7,
            ..Default::default()
        });
        assert_eq!(own_bus.unit_groups.len(), 1);
        assert_eq!(own_bus.unit_groups[0].bus_id, EntityId(7));

        let none = make_hydro(HydroSpec {
            id: 2,
            mirror_unit_group: MirrorUnitGroup::None,
            ..Default::default()
        });
        assert!(none.unit_groups.is_empty());

        let fixed = make_hydro(HydroSpec {
            id: 3,
            bus_id: 7,
            mirror_unit_group: MirrorUnitGroup::FixedBus(42),
            ..Default::default()
        });
        assert_eq!(fixed.unit_groups.len(), 1);
        assert_eq!(fixed.unit_groups[0].bus_id, EntityId(42));
    }

    #[test]
    fn make_stage_index_and_dates_are_axes() {
        let derived = make_stage(StageSpec {
            id: 5,
            ..Default::default()
        });
        assert_eq!(derived.index, 5);
        assert_eq!(derived.start_date, date(2024, 1, 1));
        assert_eq!(derived.end_date, date(2024, 2, 1));

        let negative_id = make_stage(StageSpec {
            id: -1,
            ..Default::default()
        });
        assert_eq!(negative_id.index, 0);

        let overridden = make_stage(StageSpec {
            id: 5,
            index: Some(1),
            start_date: date(2020, 3, 1),
            end_date: date(2020, 4, 1),
            ..Default::default()
        });
        assert_eq!(overridden.index, 1);
        assert_eq!(overridden.start_date, date(2020, 3, 1));
        assert_eq!(overridden.end_date, date(2020, 4, 1));
    }

    #[test]
    fn approx_erf_and_norm_cdf_match_known_values() {
        assert!((norm_cdf(0.0) - 0.5).abs() < 1e-8);
        assert!(approx_erf(0.0).abs() < 1e-8);
        assert!((norm_cdf(1.96) - 0.975).abs() < 1e-4);
        assert!((approx_erf(-0.7) + approx_erf(0.7)).abs() < 1e-12);
    }
}
