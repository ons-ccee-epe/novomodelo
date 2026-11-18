//! Default-bearing entity builders for `novomodelo-sddp` tests.
//!
//! The one-place field-addition invariant: a new required field on a `novomodelo-core`
//! entity is absorbed by editing only this module — add the field to the
//! matching `<Entity>Spec` (with a neutral `Default`) and map it in
//! `make_<entity>`. Call sites that spread `..Default::default()` recompile
//! unchanged, overriding every field they depend on. Each `make_<entity>` entity
//! literal omits `..`, so the compiler rejects a `Spec` that fails to mirror
//! every non-identity field.

use chrono::NaiveDate;
use novomodelo_core::EntityId;
use novomodelo_core::entities::bus::{Bus, DeficitSegment};
use novomodelo_core::entities::energy_contract::{ContractType, EnergyContract};
use novomodelo_core::entities::hydro::{
    DiversionChannel, EfficiencyModel, FillingConfig, HydraulicLossesModel, Hydro,
    HydroGenerationModel, HydroPenalties, HydroUnitGroup, TailraceModel,
};
use novomodelo_core::entities::line::Line;
use novomodelo_core::entities::non_controllable::NonControllableSource;
use novomodelo_core::entities::pumping_station::PumpingStation;
use novomodelo_core::entities::thermal::{AnticipatedConfig, Thermal};
use novomodelo_core::temporal::{
    Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig, StageStateConfig,
};

/// Non-identity fields of [`Stage`]; `index`/`id` are supplied positionally to
/// [`make_stage`].
#[derive(Clone)]
pub struct StageSpec {
    /// Stage start date (inclusive).
    pub start_date: NaiveDate,
    /// Stage end date (exclusive).
    pub end_date: NaiveDate,
    /// Season index; None when the stage has no seasonal structure.
    pub season_id: Option<usize>,
    /// Load blocks, sorted by index.
    pub blocks: Vec<Block>,
    /// How the stage's blocks relate in time.
    pub block_mode: BlockMode,
    /// State variables carried across the stage boundary.
    pub state_config: StageStateConfig,
    /// Risk measure applied at the stage.
    pub risk_config: StageRiskConfig,
    /// Per-stage scenario source (branching factor and noise method).
    pub scenario_config: ScenarioSourceConfig,
}

impl Default for StageSpec {
    fn default() -> Self {
        Self {
            start_date: super::ymd(2024, 1, 1),
            end_date: super::ymd(2024, 2, 1),
            season_id: None,
            blocks: vec![Block {
                index: 0,
                name: String::from("B0"),
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
}

/// Build a [`Stage`] with `index` as the positional identity.
///
/// The stage `id` is `index` as an `i32`; the fixtures this mirrors always use
/// `id == index`, so the identity has no spec default that could alias stages.
#[must_use]
pub fn make_stage(
    index: usize,
    StageSpec {
        start_date,
        end_date,
        season_id,
        blocks,
        block_mode,
        state_config,
        risk_config,
        scenario_config,
    }: StageSpec,
) -> Stage {
    Stage {
        index,
        id: i32::try_from(index).unwrap_or(i32::MAX),
        start_date,
        end_date,
        season_id,
        blocks,
        block_mode,
        state_config,
        risk_config,
        scenario_config,
    }
}

/// Non-identity fields of [`Hydro`]; `id` is supplied positionally to
/// [`make_hydro`].
#[derive(Clone)]
pub struct HydroSpec {
    /// Plant name.
    pub name: String,
    /// Date from which the plant is operational.
    pub operational_start_date: NaiveDate,
    /// Bus receiving this plant's generation.
    pub bus_id: EntityId,
    /// Downstream cascade plant; None = run-of-river or final.
    pub downstream_id: Option<EntityId>,
    /// Travel time on the cascade arc to `downstream_id`; None = instantaneous.
    pub travel_time_hours: Option<f64>,
    /// Commissioning stage; None = always present.
    pub entry_stage_id: Option<i32>,
    /// Decommissioning stage; None = never retired.
    pub exit_stage_id: Option<i32>,
    /// Minimum operational storage — dead volume.
    pub min_storage_hm3: f64,
    /// Maximum operational storage.
    pub max_storage_hm3: f64,
    /// Minimum outflow (m³/s).
    pub min_outflow_m3s: f64,
    /// Maximum outflow; None = unbounded.
    pub max_outflow_m3s: Option<f64>,
    /// Minimum spillage (m³/s); None = no lower bound.
    pub min_spillage_m3s: Option<f64>,
    /// Maximum spillage (m³/s); None = no upper bound.
    pub max_spillage_m3s: Option<f64>,
    /// Generation model.
    pub generation_model: HydroGenerationModel,
    /// Minimum turbined flow (m³/s).
    pub min_turbined_m3s: f64,
    /// Maximum turbined flow (m³/s).
    pub max_turbined_m3s: f64,
    /// Specific productivity (MW per (m³/s)·m); None = not supplied.
    pub specific_productivity_mw_per_m3s_per_m: Option<f64>,
    /// Minimum generation (MW).
    pub min_generation_mw: f64,
    /// Maximum generation (MW).
    pub max_generation_mw: f64,
    /// Turbine groups partitioning the plant's generation envelope.
    pub unit_groups: Vec<HydroUnitGroup>,
    /// Tailrace elevation model; None = constant zero tailrace.
    pub tailrace: Option<TailraceModel>,
    /// Penstock hydraulic-loss model; None = lossless.
    pub hydraulic_losses: Option<HydraulicLossesModel>,
    /// Turbine efficiency model; None = 100% efficient.
    pub efficiency: Option<EfficiencyModel>,
    /// Monthly evaporation coefficients (mm/month); None = no evaporation.
    pub evaporation_coefficients_mm: Option<[f64; 12]>,
    /// Monthly reference storage volumes for evaporation linearization.
    pub evaporation_reference_volumes_hm3: Option<[f64; 12]>,
    /// Diversion channel configuration; None = no channel.
    pub diversion: Option<DiversionChannel>,
    /// Reservoir filling configuration; None = no filling operation.
    pub filling: Option<FillingConfig>,
    /// Resolved entity-level penalty costs.
    pub penalties: HydroPenalties,
}

impl Default for HydroSpec {
    fn default() -> Self {
        Self {
            name: String::new(),
            operational_start_date: super::ymd(2024, 1, 1),
            bus_id: EntityId(0),
            downstream_id: None,
            travel_time_hours: None,
            entry_stage_id: None,
            exit_stage_id: None,
            min_storage_hm3: 0.0,
            max_storage_hm3: 0.0,
            min_outflow_m3s: 0.0,
            max_outflow_m3s: None,
            min_spillage_m3s: None,
            max_spillage_m3s: None,
            generation_model: HydroGenerationModel::ConstantProductivity,
            min_turbined_m3s: 0.0,
            max_turbined_m3s: 0.0,
            specific_productivity_mw_per_m3s_per_m: None,
            min_generation_mw: 0.0,
            max_generation_mw: 0.0,
            unit_groups: Vec::new(),
            tailrace: None,
            hydraulic_losses: None,
            efficiency: None,
            evaporation_coefficients_mm: None,
            evaporation_reference_volumes_hm3: None,
            diversion: None,
            filling: None,
            penalties: neutral_hydro_penalties(),
        }
    }
}

fn neutral_hydro_penalties() -> HydroPenalties {
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
        fpha_violation_cost: 5000.0,
        evaporation_violation_cost: 0.0,
        water_withdrawal_violation_cost: 0.0,
        water_withdrawal_violation_pos_cost: 0.0,
        water_withdrawal_violation_neg_cost: 0.0,
        evaporation_violation_pos_cost: 0.0,
        evaporation_violation_neg_cost: 0.0,
        inflow_nonnegativity_cost: 0.0,
    }
}

/// Build a [`Hydro`] with `id` as the positional identity.
#[must_use]
pub fn make_hydro(
    id: EntityId,
    HydroSpec {
        name,
        operational_start_date,
        bus_id,
        downstream_id,
        travel_time_hours,
        entry_stage_id,
        exit_stage_id,
        min_storage_hm3,
        max_storage_hm3,
        min_outflow_m3s,
        max_outflow_m3s,
        min_spillage_m3s,
        max_spillage_m3s,
        generation_model,
        min_turbined_m3s,
        max_turbined_m3s,
        specific_productivity_mw_per_m3s_per_m,
        min_generation_mw,
        max_generation_mw,
        unit_groups,
        tailrace,
        hydraulic_losses,
        efficiency,
        evaporation_coefficients_mm,
        evaporation_reference_volumes_hm3,
        diversion,
        filling,
        penalties,
    }: HydroSpec,
) -> Hydro {
    let mut hydro = Hydro {
        id,
        name,
        operational_start_date,
        downstream_id,
        travel_time_hours,
        entry_stage_id,
        exit_stage_id,
        min_storage_hm3,
        max_storage_hm3,
        min_outflow_m3s,
        max_outflow_m3s,
        min_spillage_m3s,
        max_spillage_m3s,
        generation_model,
        min_turbined_m3s,
        max_turbined_m3s,
        specific_productivity_mw_per_m3s_per_m,
        min_generation_mw,
        max_generation_mw,
        unit_groups,
        tailrace,
        hydraulic_losses,
        efficiency,
        evaporation_coefficients_mm,
        evaporation_reference_volumes_hm3,
        diversion,
        filling,
        penalties,
    };
    hydro.declare_mirror_unit_group(bus_id);
    hydro
}

/// Non-identity fields of [`Bus`]; `id` is supplied positionally to
/// [`make_bus`].
#[derive(Clone)]
pub struct BusSpec {
    /// Bus name.
    pub name: String,
    /// Date from which the bus is operational.
    pub operational_start_date: NaiveDate,
    /// Deficit cost segments, ordered by ascending cost.
    pub deficit_segments: Vec<DeficitSegment>,
    /// Cost per `MWh` for surplus generation absorption.
    pub excess_cost: f64,
}

impl Default for BusSpec {
    fn default() -> Self {
        Self {
            name: String::new(),
            operational_start_date: super::ymd(2024, 1, 1),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 0.0,
            }],
            excess_cost: 0.0,
        }
    }
}

/// Build a [`Bus`] with `id` as the positional identity.
#[must_use]
pub fn make_bus(
    id: EntityId,
    BusSpec {
        name,
        operational_start_date,
        deficit_segments,
        excess_cost,
    }: BusSpec,
) -> Bus {
    Bus {
        id,
        name,
        operational_start_date,
        deficit_segments,
        excess_cost,
    }
}

/// Non-identity fields of [`Thermal`]; `id` is supplied positionally to
/// [`make_thermal`].
#[derive(Clone)]
pub struct ThermalSpec {
    /// Plant name.
    pub name: String,
    /// Date from which the plant is operational.
    pub operational_start_date: NaiveDate,
    /// Bus receiving this plant's generation.
    pub bus_id: EntityId,
    /// Commissioning stage; None = always present.
    pub entry_stage_id: Option<i32>,
    /// Decommissioning stage; None = never retired.
    pub exit_stage_id: Option<i32>,
    /// Marginal cost of generation ($/`MWh`).
    pub cost_per_mwh: f64,
    /// Minimum generation (MW).
    pub min_generation_mw: f64,
    /// Maximum generation (MW).
    pub max_generation_mw: f64,
    /// Anticipated dispatch configuration; None = no anticipation lag.
    pub anticipated_config: Option<AnticipatedConfig>,
}

impl Default for ThermalSpec {
    fn default() -> Self {
        Self {
            name: String::new(),
            operational_start_date: super::ymd(2024, 1, 1),
            bus_id: EntityId(0),
            entry_stage_id: None,
            exit_stage_id: None,
            cost_per_mwh: 0.0,
            min_generation_mw: 0.0,
            max_generation_mw: 0.0,
            anticipated_config: None,
        }
    }
}

/// Build a [`Thermal`] with `id` as the positional identity.
#[must_use]
pub fn make_thermal(
    id: EntityId,
    ThermalSpec {
        name,
        operational_start_date,
        bus_id,
        entry_stage_id,
        exit_stage_id,
        cost_per_mwh,
        min_generation_mw,
        max_generation_mw,
        anticipated_config,
    }: ThermalSpec,
) -> Thermal {
    Thermal {
        id,
        name,
        operational_start_date,
        bus_id,
        entry_stage_id,
        exit_stage_id,
        cost_per_mwh,
        min_generation_mw,
        max_generation_mw,
        anticipated_config,
    }
}

/// Non-identity fields of [`Line`]; `id` is supplied positionally to
/// [`make_line`].
#[derive(Clone)]
pub struct LineSpec {
    /// Line name.
    pub name: String,
    /// Date from which the line is operational.
    pub operational_start_date: NaiveDate,
    /// Source bus of the direct flow direction.
    pub source_bus_id: EntityId,
    /// Target bus of the direct flow direction.
    pub target_bus_id: EntityId,
    /// Commissioning stage; None = always present.
    pub entry_stage_id: Option<i32>,
    /// Decommissioning stage; None = never retired.
    pub exit_stage_id: Option<i32>,
    /// Maximum flow from source to target (MW).
    pub direct_capacity_mw: f64,
    /// Maximum flow from target to source (MW).
    pub reverse_capacity_mw: f64,
    /// Transmission losses (percent).
    pub losses_percent: f64,
    /// Regularization cost per `MWh` exchanged ($/`MWh`).
    pub exchange_cost: f64,
}

impl Default for LineSpec {
    fn default() -> Self {
        Self {
            name: String::new(),
            operational_start_date: super::ymd(2024, 1, 1),
            source_bus_id: EntityId(0),
            target_bus_id: EntityId(0),
            entry_stage_id: None,
            exit_stage_id: None,
            direct_capacity_mw: 0.0,
            reverse_capacity_mw: 0.0,
            losses_percent: 0.0,
            exchange_cost: 0.0,
        }
    }
}

/// Build a [`Line`] with `id` as the positional identity.
#[must_use]
pub fn make_line(
    id: EntityId,
    LineSpec {
        name,
        operational_start_date,
        source_bus_id,
        target_bus_id,
        entry_stage_id,
        exit_stage_id,
        direct_capacity_mw,
        reverse_capacity_mw,
        losses_percent,
        exchange_cost,
    }: LineSpec,
) -> Line {
    Line {
        id,
        name,
        operational_start_date,
        source_bus_id,
        target_bus_id,
        entry_stage_id,
        exit_stage_id,
        direct_capacity_mw,
        reverse_capacity_mw,
        losses_percent,
        exchange_cost,
    }
}

/// Non-identity fields of [`NonControllableSource`]; `id` is supplied
/// positionally to [`make_ncs`].
#[derive(Clone)]
pub struct NcsSpec {
    /// Source name.
    pub name: String,
    /// Date from which the source is operational.
    pub operational_start_date: NaiveDate,
    /// Bus receiving this source's generation.
    pub bus_id: EntityId,
    /// Commissioning stage; None = always present.
    pub entry_stage_id: Option<i32>,
    /// Decommissioning stage; None = never retired.
    pub exit_stage_id: Option<i32>,
    /// Maximum generation (MW).
    pub max_generation_mw: f64,
    /// Whether the LP may curtail this source.
    pub allow_curtailment: bool,
    /// Cost per `MWh` of curtailed generation ($/`MWh`).
    pub curtailment_cost: f64,
}

impl Default for NcsSpec {
    fn default() -> Self {
        Self {
            name: String::new(),
            operational_start_date: super::ymd(2024, 1, 1),
            bus_id: EntityId(0),
            entry_stage_id: None,
            exit_stage_id: None,
            max_generation_mw: 100.0,
            allow_curtailment: true,
            curtailment_cost: 0.0,
        }
    }
}

/// Build a [`NonControllableSource`] with `id` as the positional identity.
#[must_use]
pub fn make_ncs(
    id: EntityId,
    NcsSpec {
        name,
        operational_start_date,
        bus_id,
        entry_stage_id,
        exit_stage_id,
        max_generation_mw,
        allow_curtailment,
        curtailment_cost,
    }: NcsSpec,
) -> NonControllableSource {
    NonControllableSource {
        id,
        name,
        operational_start_date,
        bus_id,
        entry_stage_id,
        exit_stage_id,
        max_generation_mw,
        allow_curtailment,
        curtailment_cost,
    }
}

/// Non-identity fields of [`EnergyContract`]; `id` is supplied positionally to
/// [`make_contract`].
#[derive(Clone)]
pub struct ContractSpec {
    /// Contract name.
    pub name: String,
    /// Date from which the contract is operational.
    pub operational_start_date: NaiveDate,
    /// Bus at which the contracted power is injected or withdrawn.
    pub bus_id: EntityId,
    /// Direction of energy flow.
    pub contract_type: ContractType,
    /// Commissioning stage; None = always present.
    pub entry_stage_id: Option<i32>,
    /// Decommissioning stage; None = never retired.
    pub exit_stage_id: Option<i32>,
    /// Price per `MWh` ($/`MWh`); negative for export revenue.
    pub price_per_mwh: f64,
    /// Minimum contracted power (MW).
    pub min_mw: f64,
    /// Maximum contracted power (MW).
    pub max_mw: f64,
}

impl Default for ContractSpec {
    fn default() -> Self {
        Self {
            name: String::new(),
            operational_start_date: super::ymd(2024, 1, 1),
            bus_id: EntityId(0),
            contract_type: ContractType::Import,
            entry_stage_id: None,
            exit_stage_id: None,
            price_per_mwh: 0.0,
            min_mw: 0.0,
            max_mw: 0.0,
        }
    }
}

/// Build an [`EnergyContract`] with `id` as the positional identity.
#[must_use]
pub fn make_contract(
    id: EntityId,
    ContractSpec {
        name,
        operational_start_date,
        bus_id,
        contract_type,
        entry_stage_id,
        exit_stage_id,
        price_per_mwh,
        min_mw,
        max_mw,
    }: ContractSpec,
) -> EnergyContract {
    EnergyContract {
        id,
        name,
        operational_start_date,
        bus_id,
        contract_type,
        entry_stage_id,
        exit_stage_id,
        price_per_mwh,
        min_mw,
        max_mw,
    }
}

/// Non-identity fields of [`PumpingStation`]; `id` is supplied positionally to
/// [`make_pumping_station`].
#[derive(Clone)]
pub struct PumpingSpec {
    /// Station name.
    pub name: String,
    /// Date from which the station is operational.
    pub operational_start_date: NaiveDate,
    /// Bus from which electrical power is consumed.
    pub bus_id: EntityId,
    /// Hydro plant whose reservoir water is extracted.
    pub source_hydro_id: EntityId,
    /// Hydro plant whose reservoir receives the pumped water.
    pub destination_hydro_id: EntityId,
    /// Commissioning stage; None = always present.
    pub entry_stage_id: Option<i32>,
    /// Decommissioning stage; None = never retired.
    pub exit_stage_id: Option<i32>,
    /// Power consumed per unit of pumped flow (MW per m³/s).
    pub consumption_mw_per_m3s: f64,
    /// Minimum pumped flow (m³/s).
    pub min_flow_m3s: f64,
    /// Maximum pumped flow (m³/s).
    pub max_flow_m3s: f64,
}

impl Default for PumpingSpec {
    fn default() -> Self {
        Self {
            name: String::new(),
            operational_start_date: super::ymd(2024, 1, 1),
            bus_id: EntityId(0),
            source_hydro_id: EntityId(0),
            destination_hydro_id: EntityId(0),
            entry_stage_id: None,
            exit_stage_id: None,
            consumption_mw_per_m3s: 0.0,
            min_flow_m3s: 0.0,
            max_flow_m3s: 0.0,
        }
    }
}

/// Build a [`PumpingStation`] with `id` as the positional identity.
#[must_use]
pub fn make_pumping_station(
    id: EntityId,
    PumpingSpec {
        name,
        operational_start_date,
        bus_id,
        source_hydro_id,
        destination_hydro_id,
        entry_stage_id,
        exit_stage_id,
        consumption_mw_per_m3s,
        min_flow_m3s,
        max_flow_m3s,
    }: PumpingSpec,
) -> PumpingStation {
    PumpingStation {
        id,
        name,
        operational_start_date,
        bus_id,
        source_hydro_id,
        destination_hydro_id,
        entry_stage_id,
        exit_stage_id,
        consumption_mw_per_m3s,
        min_flow_m3s,
        max_flow_m3s,
    }
}
