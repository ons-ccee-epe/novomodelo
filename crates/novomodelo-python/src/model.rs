//! `PyO3` wrapper classes exposing `cobre-core` entity types in `cobre.model`.
//!
//! Each entity wrapper holds the shared `Arc<System>` plus its index into the
//! matching collection, so a `System` getter (e.g. `System.hydros`) reads
//! rather than deep-clones. As a result, element identity across reads is
//! unspecified: two separate reads of the same getter (or of two different
//! getters over the same underlying entity) may or may not yield objects
//! that compare `is` equal in Python.

use std::sync::Arc;

use pyo3::prelude::*;
use pyo3::types::PyDict;

use cobre_core::Bus;
use cobre_core::ContractType;
use cobre_core::EnergyContract;
use cobre_core::Hydro;
use cobre_core::HydroGenerationModel::ConstantProductivity;
use cobre_core::HydroGenerationModel::Fpha;
use cobre_core::HydroGenerationModel::LinearizedHead;
use cobre_core::Line;
use cobre_core::NonControllableSource;
use cobre_core::PumpingStation;
use cobre_core::System;
use cobre_core::Thermal;

// ─── Bus ─────────────────────────────────────────────────────────────────────

/// Electrical network node where energy balance is maintained.
///
/// Each bus carries a power balance constraint satisfied at every stage and block.
#[pyclass(name = "Bus", frozen, skip_from_py_object)]
#[derive(Clone)]
pub struct PyBus {
    system: Arc<System>,
    index: usize,
}

#[pymethods]
impl PyBus {
    #[getter]
    fn id(&self) -> i32 {
        self.bus().id.0
    }

    #[getter]
    fn name(&self) -> &str {
        &self.bus().name
    }

    /// Pre-resolved piecewise-linear deficit cost segments.
    ///
    /// Each segment is returned as a dict with keys `"depth_mw"` (float or
    /// `None` for the final unbounded segment) and `"cost_per_mwh"` (float).
    #[getter]
    fn deficit_segments<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyDict>>> {
        self.bus()
            .deficit_segments
            .iter()
            .map(|seg| {
                let d = PyDict::new(py);
                match seg.depth_mw {
                    Some(v) => d.set_item("depth_mw", v)?,
                    None => d.set_item("depth_mw", py.None())?,
                }
                d.set_item("cost_per_mwh", seg.cost_per_mwh)?;
                Ok(d)
            })
            .collect()
    }

    /// Cost per `MWh` for surplus generation absorption [$/`MWh`].
    #[getter]
    fn excess_cost(&self) -> f64 {
        self.bus().excess_cost
    }

    fn __repr__(&self) -> String {
        format!("Bus(id={}, name='{}')", self.bus().id.0, self.bus().name)
    }
}

impl PyBus {
    pub(crate) fn from_index(system: Arc<System>, index: usize) -> Self {
        Self { system, index }
    }

    fn bus(&self) -> &Bus {
        &self.system.buses()[self.index]
    }
}

// ─── Line ────────────────────────────────────────────────────────────────────

/// Transmission interconnection carrying bidirectional power between two buses.
#[pyclass(name = "Line", frozen, skip_from_py_object)]
#[derive(Clone)]
pub struct PyLine {
    system: Arc<System>,
    index: usize,
}

#[pymethods]
impl PyLine {
    #[getter]
    fn id(&self) -> i32 {
        self.line().id.0
    }

    #[getter]
    fn name(&self) -> &str {
        &self.line().name
    }

    #[getter]
    fn source_bus_id(&self) -> i32 {
        self.line().source_bus_id.0
    }

    #[getter]
    fn target_bus_id(&self) -> i32 {
        self.line().target_bus_id.0
    }

    /// Maximum flow from source to target [MW].
    #[getter]
    fn direct_capacity_mw(&self) -> f64 {
        self.line().direct_capacity_mw
    }

    /// Maximum flow from target to source [MW].
    #[getter]
    fn reverse_capacity_mw(&self) -> f64 {
        self.line().reverse_capacity_mw
    }

    /// Transmission losses as percentage (e.g., 2.5 means 2.5%).
    #[getter]
    fn losses_percent(&self) -> f64 {
        self.line().losses_percent
    }

    /// Regularization cost per `MWh` exchanged [$/`MWh`].
    #[getter]
    fn exchange_cost(&self) -> f64 {
        self.line().exchange_cost
    }

    fn __repr__(&self) -> String {
        format!(
            "Line(id={}, name='{}', source_bus_id={}, target_bus_id={})",
            self.line().id.0,
            self.line().name,
            self.line().source_bus_id.0,
            self.line().target_bus_id.0
        )
    }
}

impl PyLine {
    pub(crate) fn from_index(system: Arc<System>, index: usize) -> Self {
        Self { system, index }
    }

    fn line(&self) -> &Line {
        &self.system.lines()[self.index]
    }
}

// ─── Thermal ─────────────────────────────────────────────────────────────────

/// Thermal power plant with a scalar marginal cost.
#[pyclass(name = "Thermal", frozen, skip_from_py_object)]
#[derive(Clone)]
pub struct PyThermal {
    system: Arc<System>,
    index: usize,
}

#[pymethods]
impl PyThermal {
    #[getter]
    fn id(&self) -> i32 {
        self.thermal().id.0
    }

    #[getter]
    fn name(&self) -> &str {
        &self.thermal().name
    }

    #[getter]
    fn bus_id(&self) -> i32 {
        self.thermal().bus_id.0
    }

    /// Minimum electrical generation (minimum stable load) [MW].
    #[getter]
    fn min_generation_mw(&self) -> f64 {
        self.thermal().min_generation_mw
    }

    /// Maximum electrical generation (installed capacity) [MW].
    #[getter]
    fn max_generation_mw(&self) -> f64 {
        self.thermal().max_generation_mw
    }

    /// Marginal cost of generation [$/`MWh`].
    #[getter]
    fn cost_per_mwh(&self) -> f64 {
        self.thermal().cost_per_mwh
    }

    fn __repr__(&self) -> String {
        format!(
            "Thermal(id={}, name='{}', bus_id={})",
            self.thermal().id.0,
            self.thermal().name,
            self.thermal().bus_id.0
        )
    }
}

impl PyThermal {
    pub(crate) fn from_index(system: Arc<System>, index: usize) -> Self {
        Self { system, index }
    }

    fn thermal(&self) -> &Thermal {
        &self.system.thermals()[self.index]
    }
}

// ─── Hydro ───────────────────────────────────────────────────────────────────

/// Hydroelectric power plant with reservoir storage and cascade topology.
///
/// Plants form a cascade via `downstream_id` references.
#[pyclass(name = "Hydro", frozen, skip_from_py_object)]
#[derive(Clone)]
pub struct PyHydro {
    system: Arc<System>,
    index: usize,
}

#[pymethods]
impl PyHydro {
    #[getter]
    fn id(&self) -> i32 {
        self.hydro().id.0
    }

    #[getter]
    fn name(&self) -> &str {
        &self.hydro().name
    }

    #[getter]
    fn downstream_id(&self) -> Option<i32> {
        self.hydro().downstream_id.map(|id| id.0)
    }

    /// Minimum operational storage (dead volume) [hm³].
    #[getter]
    fn min_storage_hm3(&self) -> f64 {
        self.hydro().min_storage_hm3
    }

    /// Maximum operational storage (flood control level) [hm³].
    #[getter]
    fn max_storage_hm3(&self) -> f64 {
        self.hydro().max_storage_hm3
    }

    /// Minimum turbined flow [m³/s].
    #[getter]
    fn min_turbined_m3s(&self) -> f64 {
        self.hydro().min_turbined_m3s
    }

    /// Maximum turbined flow (installed turbine capacity) [m³/s].
    #[getter]
    fn max_turbined_m3s(&self) -> f64 {
        self.hydro().max_turbined_m3s
    }

    /// Power output per unit of turbined flow [MW/(m³/s)].
    ///
    /// Always `None`: the productivity scalar is resolved per-stage from
    /// `system/hydro_production_models.json`; this getter exists for ABI continuity.
    #[getter]
    fn productivity_mw_per_m3s(&self) -> Option<f64> {
        match self.hydro().generation_model {
            ConstantProductivity | LinearizedHead | Fpha => None,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "Hydro(id={}, name='{}')",
            self.hydro().id.0,
            self.hydro().name
        )
    }
}

impl PyHydro {
    pub(crate) fn from_index(system: Arc<System>, index: usize) -> Self {
        Self { system, index }
    }

    fn hydro(&self) -> &Hydro {
        &self.system.hydros()[self.index]
    }
}

// ─── EnergyContract ──────────────────────────────────────────────────────────

/// Bilateral energy contract with an external system.
#[pyclass(name = "EnergyContract", frozen, skip_from_py_object)]
#[derive(Clone)]
pub struct PyEnergyContract {
    system: Arc<System>,
    index: usize,
}

#[pymethods]
impl PyEnergyContract {
    #[getter]
    fn id(&self) -> i32 {
        self.contract().id.0
    }

    #[getter]
    fn name(&self) -> &str {
        &self.contract().name
    }

    /// Date the entity enters service, as an ISO 8601 `YYYY-MM-DD` string.
    #[getter]
    fn operational_start_date(&self) -> String {
        self.contract().operational_start_date.to_string()
    }

    #[getter]
    fn bus_id(&self) -> i32 {
        self.contract().bus_id.0
    }

    /// Direction of energy flow: `"import"` or `"export"`.
    #[getter]
    fn contract_type(&self) -> &'static str {
        match self.contract().contract_type {
            ContractType::Import => "import",
            ContractType::Export => "export",
        }
    }

    /// Stage index when the contract enters service, or `None` if always active.
    #[getter]
    fn entry_stage_id(&self) -> Option<i32> {
        self.contract().entry_stage_id
    }

    /// Stage index when the contract expires, or `None` if it never expires.
    #[getter]
    fn exit_stage_id(&self) -> Option<i32> {
        self.contract().exit_stage_id
    }

    /// Contract price per `MWh`; negative values represent export revenue [$/`MWh`].
    #[getter]
    fn price_per_mwh(&self) -> f64 {
        self.contract().price_per_mwh
    }

    /// Minimum contracted power [MW].
    #[getter]
    fn min_mw(&self) -> f64 {
        self.contract().min_mw
    }

    /// Maximum contracted power [MW].
    #[getter]
    fn max_mw(&self) -> f64 {
        self.contract().max_mw
    }

    fn __repr__(&self) -> String {
        format!(
            "EnergyContract(id={}, name='{}')",
            self.contract().id.0,
            self.contract().name
        )
    }
}

impl PyEnergyContract {
    pub(crate) fn from_index(system: Arc<System>, index: usize) -> Self {
        Self { system, index }
    }

    fn contract(&self) -> &EnergyContract {
        &self.system.contracts()[self.index]
    }
}

// ─── PumpingStation ──────────────────────────────────────────────────────────

/// Pumping station that transfers water between hydro reservoirs.
#[pyclass(name = "PumpingStation", frozen, skip_from_py_object)]
#[derive(Clone)]
pub struct PyPumpingStation {
    system: Arc<System>,
    index: usize,
}

#[pymethods]
impl PyPumpingStation {
    #[getter]
    fn id(&self) -> i32 {
        self.pumping_station().id.0
    }

    #[getter]
    fn name(&self) -> &str {
        &self.pumping_station().name
    }

    /// Date the entity enters service, as an ISO 8601 `YYYY-MM-DD` string.
    #[getter]
    fn operational_start_date(&self) -> String {
        self.pumping_station().operational_start_date.to_string()
    }

    #[getter]
    fn bus_id(&self) -> i32 {
        self.pumping_station().bus_id.0
    }

    #[getter]
    fn source_hydro_id(&self) -> i32 {
        self.pumping_station().source_hydro_id.0
    }

    #[getter]
    fn destination_hydro_id(&self) -> i32 {
        self.pumping_station().destination_hydro_id.0
    }

    /// Stage index when the station enters service, or `None` if always active.
    #[getter]
    fn entry_stage_id(&self) -> Option<i32> {
        self.pumping_station().entry_stage_id
    }

    /// Stage index when the station is decommissioned, or `None` if never decommissioned.
    #[getter]
    fn exit_stage_id(&self) -> Option<i32> {
        self.pumping_station().exit_stage_id
    }

    /// Power consumption rate per unit of pumped flow [MW/(m³/s)].
    #[getter]
    fn consumption_mw_per_m3s(&self) -> f64 {
        self.pumping_station().consumption_mw_per_m3s
    }

    /// Minimum pumped flow [m³/s].
    #[getter]
    fn min_flow_m3s(&self) -> f64 {
        self.pumping_station().min_flow_m3s
    }

    /// Maximum pumped flow (installed pump capacity) [m³/s].
    #[getter]
    fn max_flow_m3s(&self) -> f64 {
        self.pumping_station().max_flow_m3s
    }

    fn __repr__(&self) -> String {
        format!(
            "PumpingStation(id={}, name='{}')",
            self.pumping_station().id.0,
            self.pumping_station().name
        )
    }
}

impl PyPumpingStation {
    pub(crate) fn from_index(system: Arc<System>, index: usize) -> Self {
        Self { system, index }
    }

    fn pumping_station(&self) -> &PumpingStation {
        &self.system.pumping_stations()[self.index]
    }
}

// ─── NonControllableSource ───────────────────────────────────────────────────

/// Intermittent generation source that cannot be dispatched.
#[pyclass(name = "NonControllableSource", frozen, skip_from_py_object)]
#[derive(Clone)]
pub struct PyNonControllableSource {
    system: Arc<System>,
    index: usize,
}

#[pymethods]
impl PyNonControllableSource {
    #[getter]
    fn id(&self) -> i32 {
        self.source().id.0
    }

    #[getter]
    fn name(&self) -> &str {
        &self.source().name
    }

    /// Date the entity enters service, as an ISO 8601 `YYYY-MM-DD` string.
    #[getter]
    fn operational_start_date(&self) -> String {
        self.source().operational_start_date.to_string()
    }

    #[getter]
    fn bus_id(&self) -> i32 {
        self.source().bus_id.0
    }

    /// Stage index when the source enters service, or `None` if always active.
    #[getter]
    fn entry_stage_id(&self) -> Option<i32> {
        self.source().entry_stage_id
    }

    /// Stage index when the source is decommissioned, or `None` if never decommissioned.
    #[getter]
    fn exit_stage_id(&self) -> Option<i32> {
        self.source().exit_stage_id
    }

    /// Maximum generation (installed capacity) [MW].
    #[getter]
    fn max_generation_mw(&self) -> f64 {
        self.source().max_generation_mw
    }

    /// Whether the LP may curtail this source; `False` is the must-run regime.
    #[getter]
    fn allow_curtailment(&self) -> bool {
        self.source().allow_curtailment
    }

    /// Resolved cost per `MWh` of curtailed generation [$/`MWh`]; unused when
    /// `allow_curtailment` is `False`.
    #[getter]
    fn curtailment_cost(&self) -> f64 {
        self.source().curtailment_cost
    }

    fn __repr__(&self) -> String {
        format!(
            "NonControllableSource(id={}, name='{}')",
            self.source().id.0,
            self.source().name
        )
    }
}

impl PyNonControllableSource {
    pub(crate) fn from_index(system: Arc<System>, index: usize) -> Self {
        Self { system, index }
    }

    fn source(&self) -> &NonControllableSource {
        &self.system.non_controllable_sources()[self.index]
    }
}

// ─── System ──────────────────────────────────────────────────────────────────

/// Top-level system representation wrapping a loaded Cobre case.
///
/// Cannot be constructed from Python — use `cobre.io.load_case()` to obtain one.
/// Every entity-list getter below returns its items in canonical ID order.
#[pyclass(name = "System", frozen)]
pub struct PySystem {
    inner: Arc<System>,
}

#[pymethods]
impl PySystem {
    #[getter]
    fn buses(&self) -> Vec<PyBus> {
        (0..self.inner.buses().len())
            .map(|index| PyBus::from_index(Arc::clone(&self.inner), index))
            .collect()
    }

    #[getter]
    fn lines(&self) -> Vec<PyLine> {
        (0..self.inner.lines().len())
            .map(|index| PyLine::from_index(Arc::clone(&self.inner), index))
            .collect()
    }

    #[getter]
    fn thermals(&self) -> Vec<PyThermal> {
        (0..self.inner.thermals().len())
            .map(|index| PyThermal::from_index(Arc::clone(&self.inner), index))
            .collect()
    }

    #[getter]
    fn hydros(&self) -> Vec<PyHydro> {
        (0..self.inner.hydros().len())
            .map(|index| PyHydro::from_index(Arc::clone(&self.inner), index))
            .collect()
    }

    #[getter]
    fn contracts(&self) -> Vec<PyEnergyContract> {
        (0..self.inner.contracts().len())
            .map(|index| PyEnergyContract::from_index(Arc::clone(&self.inner), index))
            .collect()
    }

    #[getter]
    fn pumping_stations(&self) -> Vec<PyPumpingStation> {
        (0..self.inner.pumping_stations().len())
            .map(|index| PyPumpingStation::from_index(Arc::clone(&self.inner), index))
            .collect()
    }

    #[getter]
    fn non_controllable_sources(&self) -> Vec<PyNonControllableSource> {
        (0..self.inner.non_controllable_sources().len())
            .map(|index| PyNonControllableSource::from_index(Arc::clone(&self.inner), index))
            .collect()
    }

    #[getter]
    fn n_buses(&self) -> usize {
        self.inner.n_buses()
    }

    #[getter]
    fn n_lines(&self) -> usize {
        self.inner.n_lines()
    }

    #[getter]
    fn n_hydros(&self) -> usize {
        self.inner.n_hydros()
    }

    #[getter]
    fn n_thermals(&self) -> usize {
        self.inner.n_thermals()
    }

    /// Number of stages (study and pre-study) in the system.
    #[getter]
    fn n_stages(&self) -> usize {
        self.inner.n_stages()
    }

    fn __repr__(&self) -> String {
        format!(
            "System(n_buses={}, n_lines={}, n_hydros={}, n_thermals={}, n_stages={})",
            self.inner.n_buses(),
            self.inner.n_lines(),
            self.inner.n_hydros(),
            self.inner.n_thermals(),
            self.inner.n_stages(),
        )
    }
}

impl PySystem {
    pub(crate) fn from_rust(system: System) -> Self {
        Self {
            inner: Arc::new(system),
        }
    }

    /// Wrap an already-shared [`cobre_core::System`] via a refcount bump rather
    /// than a full clone (used by [`crate::study::Study`]'s `system` getter).
    pub(crate) fn from_arc(inner: Arc<System>) -> Self {
        Self { inner }
    }
}
