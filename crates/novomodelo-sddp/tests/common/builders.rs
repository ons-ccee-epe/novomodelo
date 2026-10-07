//! Default-bearing entity builders for `cobre-sddp` integration tests.
//!
//! The one-place field-addition invariant: a new required field on a `cobre-core`
//! entity is absorbed by editing only this module — add the field to the
//! matching `<Entity>Spec` (with a neutral `Default`) and map it in
//! `make_<entity>`. Call sites that spread `..Default::default()` recompile
//! unchanged, overriding every field they depend on. Each `make_<entity>` entity
//! literal omits `..`, so the compiler rejects a `Spec` that fails to mirror
//! every non-identity field.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::doc_markdown
)]

use chrono::NaiveDate;
use cobre_core::EntityId;
use cobre_core::entities::bus::{Bus, DeficitSegment};
use cobre_core::entities::hydro::{
    DiversionChannel, EfficiencyModel, FillingConfig, HydraulicLossesModel, Hydro,
    HydroGenerationModel, HydroPenalties, HydroUnitGroup, TailraceModel,
};
use cobre_core::entities::thermal::{AnticipatedConfig, Thermal};
use cobre_core::temporal::{
    Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig, StageStateConfig,
};

fn neutral_date() -> NaiveDate {
    NaiveDate::from_ymd_opt(2024, 1, 1).expect("2024-01-01 is a valid date")
}

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
    pub block_mode: BlockMode,
    pub state_config: StageStateConfig,
    pub risk_config: StageRiskConfig,
    /// Per-stage scenario source (branching factor and noise method).
    pub scenario_config: ScenarioSourceConfig,
}

impl Default for StageSpec {
    fn default() -> Self {
        Self {
            start_date: neutral_date(),
            end_date: NaiveDate::from_ymd_opt(2024, 2, 1).expect("2024-02-01 is a valid date"),
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
/// The stage `id` is set to `index as i32`; the fixtures this mirrors always use
/// `id == index`, so the identity has no spec default that could alias stages.
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
        id: index as i32,
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
    pub name: String,
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
    pub max_storage_hm3: f64,
    pub min_outflow_m3s: f64,
    /// Maximum outflow; None = unbounded.
    pub max_outflow_m3s: Option<f64>,
    pub generation_model: HydroGenerationModel,
    pub min_turbined_m3s: f64,
    pub max_turbined_m3s: f64,
    /// Specific productivity (MW per (m³/s)·m); None = not supplied.
    pub specific_productivity_mw_per_m3s_per_m: Option<f64>,
    pub min_generation_mw: f64,
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
            operational_start_date: neutral_date(),
            bus_id: EntityId(0),
            downstream_id: None,
            travel_time_hours: None,
            entry_stage_id: None,
            exit_stage_id: None,
            min_storage_hm3: 0.0,
            max_storage_hm3: 0.0,
            min_outflow_m3s: 0.0,
            max_outflow_m3s: None,
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
    pub name: String,
    pub operational_start_date: NaiveDate,
    /// Deficit cost segments, ordered by ascending cost.
    pub deficit_segments: Vec<DeficitSegment>,
    /// Cost per MWh for surplus generation absorption.
    pub excess_cost: f64,
}

impl Default for BusSpec {
    fn default() -> Self {
        Self {
            name: String::new(),
            operational_start_date: neutral_date(),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 0.0,
            }],
            excess_cost: 0.0,
        }
    }
}

/// Build a [`Bus`] with `id` as the positional identity.
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
    pub name: String,
    pub operational_start_date: NaiveDate,
    /// Bus receiving this plant's generation.
    pub bus_id: EntityId,
    /// Commissioning stage; None = always present.
    pub entry_stage_id: Option<i32>,
    /// Decommissioning stage; None = never retired.
    pub exit_stage_id: Option<i32>,
    /// Marginal cost of generation ($/MWh).
    pub cost_per_mwh: f64,
    pub min_generation_mw: f64,
    pub max_generation_mw: f64,
    /// Anticipated dispatch configuration; None = no anticipation lag.
    pub anticipated_config: Option<AnticipatedConfig>,
}

impl Default for ThermalSpec {
    fn default() -> Self {
        Self {
            name: String::new(),
            operational_start_date: neutral_date(),
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
