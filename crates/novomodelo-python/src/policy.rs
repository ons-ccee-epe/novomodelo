//! `novomodelo.write_policy_checkpoint` — writes a policy checkpoint from plain
//! Python dicts/sequences, single-sourcing the `FlatBuffers` byte layout in
//! `novomodelo_io`.
//!
//! Input dict shapes mirror what [`crate::results::load_policy`] emits, so a
//! loaded checkpoint round-trips: load -> edit -> write. `season_manifest` and
//! `graph_manifest` round-trip.

use std::borrow::Cow;
use std::collections::HashMap;
use std::num::NonZeroU32;
use std::path::PathBuf;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use novomodelo_io::{
    BasisEntry, CheckpointManifest, ENTITY_SLOT_DATE_SENTINEL, EntitySlot, FORMAT_VERSION,
    GraphManifest, HydroSeasonOrders, ManifestEdge, ManifestNode, OutputError, PolicyBasisRecord,
    PolicyCutRecord, ProducerBlock, SOFTWARE_NAME, SOFTWARE_VERSION,
    STAGE_CUTS_GRAPH_STAGE_ID_SENTINEL, STAGE_CUTS_NODE_ID_SENTINEL,
    STAGE_CUTS_PRICED_STATE_DATE_SENTINEL, STAGE_STATES_NODE_ID_SENTINEL, SeasonManifest,
    StageCutsPayload, StageStatesPayload, StateFamily, StateHistory, check_checkpoint_payloads,
};
use novomodelo_sddp::{SddpError, reserve_boundary_inflow_lag_slots};

use crate::errors::{ErrorSource, convert_error};

#[derive(Debug, FromPyObject)]
#[pyo3(from_item_all)]
pub(crate) struct PyEntitySlot {
    entity_type: u8,
    entity_id: i32,
    subindex: u32,
    was_active: bool,
    #[pyo3(default = ENTITY_SLOT_DATE_SENTINEL)]
    reference_date: i32,
    #[pyo3(default = ENTITY_SLOT_DATE_SENTINEL)]
    interval_start: i32,
    #[pyo3(default = ENTITY_SLOT_DATE_SENTINEL)]
    interval_end: i32,
}

impl From<&PyEntitySlot> for EntitySlot {
    fn from(slot: &PyEntitySlot) -> Self {
        match StateFamily::from_code(slot.entity_type) {
            Some(StateFamily::HydroStorage) => Self::storage(slot.entity_id, slot.was_active),
            Some(StateFamily::HydroInflowLag) => {
                Self::inflow_lag(slot.entity_id, slot.subindex, slot.was_active)
            }
            Some(StateFamily::HydroTransitBucket) => {
                Self::transit_bucket(slot.entity_id, slot.subindex, slot.was_active)
            }
            Some(StateFamily::AnticipatedThermalState) => {
                Self::anticipated(slot.entity_id, slot.subindex, slot.was_active)
            }
            None => Self {
                entity_type: slot.entity_type,
                entity_id: slot.entity_id,
                subindex: slot.subindex,
                was_active: slot.was_active,
                reference_date: ENTITY_SLOT_DATE_SENTINEL,
                interval_start: ENTITY_SLOT_DATE_SENTINEL,
                interval_end: ENTITY_SLOT_DATE_SENTINEL,
            },
        }
        .with_reference_date(slot.reference_date)
        .with_interval(slot.interval_start, slot.interval_end)
    }
}

#[derive(Debug, FromPyObject)]
#[pyo3(from_item_all)]
pub(crate) struct PyCutRecord {
    slot_index: u32,
    iteration: u32,
    forward_pass_index: u32,
    intercept: f64,
    coefficients: Vec<f64>,
    is_active: bool,
    #[pyo3(default)]
    scored: bool,
    #[pyo3(default)]
    active_count: Option<u64>,
    #[pyo3(default)]
    last_active_iteration: Option<u64>,
    /// Inflow-lag gradient terms keyed by hydro id (`hydro_id -> [coef_by_depth]`,
    /// index `0` = lag depth 1). Requires the top-level `inflow_lag_depth`; empty
    /// (the default) leaves the checkpoint byte-identical.
    #[pyo3(default)]
    inflow_lag_coefficients: HashMap<i32, Vec<f64>>,
}

#[derive(Debug, FromPyObject)]
#[pyo3(from_item_all)]
pub(crate) struct PyStageCutsPayload {
    stage_id: u32,
    state_dimension: u32,
    capacity: u32,
    #[pyo3(default)]
    warm_start_count: u32,
    cuts: Vec<PyCutRecord>,
    #[pyo3(default)]
    populated_count: Option<u32>,
    #[pyo3(default)]
    entity_manifest: Vec<PyEntitySlot>,
    /// Required; read by [`pool_cost_scale_factors`], which rejects an absent value.
    #[pyo3(default)]
    cost_scale_factor: Option<f64>,
    #[pyo3(default = STAGE_CUTS_NODE_ID_SENTINEL)]
    node_id: i32,
    #[pyo3(default = STAGE_CUTS_GRAPH_STAGE_ID_SENTINEL)]
    graph_stage_id: i32,
    #[pyo3(default = STAGE_CUTS_PRICED_STATE_DATE_SENTINEL)]
    priced_state_date: i32,
}

#[derive(Debug, FromPyObject)]
#[pyo3(from_item_all)]
pub(crate) struct PyBasisEntry {
    scenario: u32,
    column_status: Vec<u8>,
    row_status: Vec<u8>,
    piece_row_slots: Vec<u32>,
}

#[derive(Debug, FromPyObject)]
#[pyo3(from_item_all)]
pub(crate) struct PyBasisRecord {
    stage_id: u32,
    iteration: u32,
    entries: Vec<PyBasisEntry>,
}

#[derive(Debug, FromPyObject)]
#[pyo3(from_item_all)]
pub(crate) struct PyStageStatesPayload {
    stage_id: u32,
    #[pyo3(default = STAGE_STATES_NODE_ID_SENTINEL)]
    node_id: i32,
    state_dimension: u32,
    count: u32,
    data: Vec<f64>,
    #[pyo3(default)]
    entity_manifest: Vec<PyEntitySlot>,
    slots: Vec<u32>,
}

#[derive(Debug, FromPyObject)]
#[pyo3(from_item_all)]
pub(crate) struct PyManifestNode {
    id: i32,
    stage_id: i32,
    pool_id: u32,
}

#[derive(Debug, FromPyObject)]
#[pyo3(from_item_all)]
pub(crate) struct PyManifestEdge {
    source_id: i32,
    target_id: i32,
    probability: f64,
}

#[derive(Debug, FromPyObject)]
#[pyo3(from_item_all)]
pub(crate) struct PyGraphManifest {
    n_pools: u32,
    nodes: Vec<PyManifestNode>,
    edges: Vec<PyManifestEdge>,
}

impl From<PyGraphManifest> for GraphManifest {
    fn from(g: PyGraphManifest) -> Self {
        Self {
            n_pools: g.n_pools,
            nodes: g
                .nodes
                .into_iter()
                .map(|n| ManifestNode {
                    id: n.id,
                    stage_id: n.stage_id,
                    pool_id: n.pool_id,
                })
                .collect(),
            edges: g
                .edges
                .into_iter()
                .map(|e| ManifestEdge {
                    source_id: e.source_id,
                    target_id: e.target_id,
                    probability: e.probability,
                })
                .collect(),
        }
    }
}

#[derive(Debug, FromPyObject)]
#[pyo3(from_item_all)]
pub(crate) struct PyHydroSeasonOrders {
    hydro_id: i32,
    orders: Vec<u32>,
}

#[derive(Debug, FromPyObject)]
#[pyo3(from_item_all)]
pub(crate) struct PySeasonManifest {
    cycle_code: u8,
    n_seasons: u32,
    hydro_orders: Vec<PyHydroSeasonOrders>,
}

impl From<PySeasonManifest> for SeasonManifest {
    fn from(s: PySeasonManifest) -> Self {
        Self {
            cycle_code: s.cycle_code,
            n_seasons: s.n_seasons,
            hydro_orders: s
                .hydro_orders
                .into_iter()
                .map(|h| HydroSeasonOrders {
                    hydro_id: h.hydro_id,
                    orders: h.orders,
                })
                .collect(),
        }
    }
}

/// The producer-namespaced metadata block, mirroring what
/// [`crate::results::load_policy`] emits under `metadata["producer"]`.
#[derive(Debug, FromPyObject)]
#[pyo3(from_item_all)]
pub(crate) struct PyProducerBlock {
    completed_iterations: u32,
    final_lower_bound: f64,
    #[pyo3(default)]
    best_upper_bound: Option<f64>,
    max_iterations: u32,
    forward_passes: u32,
    warm_start_cuts: u32,
    #[pyo3(default)]
    warm_start_counts: Vec<u32>,
    rng_seed: u64,
    #[pyo3(default)]
    total_visited_states: u64,
    #[pyo3(default)]
    training_block_mode: String,
    #[pyo3(default)]
    training_block_mode_per_stage: Vec<String>,
    #[pyo3(default)]
    cost_scale_factor: Option<f64>,
    #[pyo3(default)]
    lower_bound_history: Vec<f64>,
    #[pyo3(default)]
    state_history: Option<PyStateHistory>,
    #[pyo3(default)]
    activity_counters: bool,
    #[pyo3(default)]
    basis_store_complete: bool,
}

/// A producer's `state_history` key: `None` (no trial points), `"whole"`, or a
/// positive window of iterations — the shape [`crate::results::load_policy`]
/// emits. `Flag` catches a Python `bool`, which a `u32` extraction would take
/// for a window of 1.
#[derive(Debug, FromPyObject)]
pub(crate) enum PyStateHistory {
    Named(String),
    Flag(bool),
    Window(u32),
}

impl PyStateHistory {
    fn resolve(history: Option<&Self>) -> PyResult<StateHistory> {
        match history {
            None => Ok(StateHistory::None),
            Some(Self::Named(scope)) if scope == "whole" => Ok(StateHistory::Whole),
            Some(Self::Window(window)) => NonZeroU32::new(*window)
                .map(StateHistory::Window)
                .ok_or_else(|| PyValueError::new_err("state_history window must be positive")),
            Some(Self::Named(other)) => Err(PyValueError::new_err(format!(
                "state_history {other:?} is neither None, \"whole\" nor a positive window"
            ))),
            Some(Self::Flag(flag)) => Err(PyValueError::new_err(format!(
                "state_history {} is a bool, not None, \"whole\" or a positive window",
                if *flag { "True" } else { "False" }
            ))),
        }
    }
}

impl PyProducerBlock {
    fn into_producer_block(self) -> PyResult<ProducerBlock> {
        Ok(ProducerBlock {
            completed_iterations: self.completed_iterations,
            final_lower_bound: self.final_lower_bound,
            best_upper_bound: self.best_upper_bound,
            max_iterations: self.max_iterations,
            forward_passes: self.forward_passes,
            warm_start_cuts: self.warm_start_cuts,
            warm_start_counts: self.warm_start_counts,
            rng_seed: self.rng_seed,
            total_visited_states: self.total_visited_states,
            training_block_mode: self.training_block_mode,
            training_block_mode_per_stage: self.training_block_mode_per_stage,
            cost_scale_factor: self.cost_scale_factor,
            lower_bound_history: self.lower_bound_history,
            state_history: PyStateHistory::resolve(self.state_history.as_ref())?,
            activity_counters: self.activity_counters,
            basis_store_complete: self.basis_store_complete,
        })
    }
}

#[derive(Debug, FromPyObject)]
#[pyo3(from_item_all)]
pub(crate) struct PyPolicyCheckpointMetadata {
    #[pyo3(default = FORMAT_VERSION)]
    format_version: u32,
    created_at: String,
    num_stages: u32,
    #[pyo3(default)]
    graph_manifest: Option<PyGraphManifest>,
    #[pyo3(default)]
    season_manifest: Option<PySeasonManifest>,
    producer: PyProducerBlock,
}

impl PyPolicyCheckpointMetadata {
    fn into_manifest(self) -> PyResult<CheckpointManifest> {
        Ok(CheckpointManifest {
            format_version: self.format_version,
            software: Some(SOFTWARE_NAME.to_string()),
            software_version: SOFTWARE_VERSION.to_string(),
            created_at: self.created_at,
            num_stages: self.num_stages,
            graph_manifest: self
                .graph_manifest
                .map(GraphManifest::from)
                .unwrap_or_default(),
            season_manifest: self
                .season_manifest
                .map(SeasonManifest::from)
                .unwrap_or_default(),
            producer: self.producer.into_producer_block()?,
        })
    }
}

fn checked_u32_len(len: usize, what: &str) -> PyResult<u32> {
    u32::try_from(len)
        .map_err(|_| PyValueError::new_err(format!("{what} has {len} entries, exceeding u32::MAX")))
}

/// Validates `coefficients.len() == state_dimension` per cut; resolves
/// `populated_count` default to `cuts.len()`.
fn resolve_populated_counts(stage_cuts: &[PyStageCutsPayload]) -> PyResult<Vec<u32>> {
    let mut resolved = Vec::with_capacity(stage_cuts.len());
    for sc in stage_cuts {
        for cut in &sc.cuts {
            if cut.coefficients.len() != sc.state_dimension as usize {
                return Err(PyValueError::new_err(format!(
                    "stage {} cut at slot {}: coefficients has {} entries, expected \
                     state_dimension={}",
                    sc.stage_id,
                    cut.slot_index,
                    cut.coefficients.len(),
                    sc.state_dimension
                )));
            }
        }
        resolved.push(match sc.populated_count {
            Some(p) => p,
            None => checked_u32_len(sc.cuts.len(), "cuts")?,
        });
    }
    Ok(resolved)
}

/// The scale of a pool's values, read from its required `cost_scale_factor`.
fn pool_cost_scale_factors(stage_cuts: &[PyStageCutsPayload]) -> PyResult<Vec<f64>> {
    stage_cuts
        .iter()
        .enumerate()
        .map(|(index, sc)| {
            let problem = match sc.cost_scale_factor {
                None => "is required".to_string(),
                Some(factor) if !(factor.is_finite() && factor > 0.0) => {
                    format!("must be finite and positive, got {factor}")
                }
                Some(factor) => return Ok(factor),
            };
            Err(PyValueError::new_err(format!(
                "stage_cuts[{index}] (stage_id {}): cost_scale_factor {problem}; the intercepts \
                 and coefficients are in the units of this factor (pass 1.0 for values in \
                 currency units)",
                sc.stage_id
            )))
        })
        .collect()
}

/// A whole basis store claims every scenario's basis, which cannot be true of a
/// write that received none.
fn reject_unreceived_basis_store(
    producer: &PyProducerBlock,
    stage_bases: &[PyBasisRecord],
) -> PyResult<()> {
    if producer.basis_store_complete && stage_bases.iter().all(|b| b.entries.is_empty()) {
        return Err(PyValueError::new_err(
            "metadata['producer']['basis_store_complete'] is True but stage_bases holds no \
             basis; pass the bases or set it to False",
        ));
    }
    Ok(())
}

// `from_item_all` ignores keys it does not name, so a caller still passing a
// retired key would otherwise write without error.
const RETIRED_POOL_KEYS: [&str; 1] = ["active_cut_indices"];
const RETIRED_CUT_KEYS: [&str; 1] = ["cut_id"];
const RETIRED_BASIS_KEYS: [&str; 3] = ["num_cut_rows", "num_columns", "num_rows"];

fn reject_retired_keys(item: &Bound<'_, PyAny>, keys: &[&str], position: &str) -> PyResult<()> {
    for key in keys {
        if item.contains(key)? {
            return Err(PyValueError::new_err(format!(
                "{position}: key '{key}' is not part of a policy checkpoint (format \
                 {FORMAT_VERSION}); remove it"
            )));
        }
    }
    Ok(())
}

/// Runs on the raw arguments, before extraction.
fn reject_retired_checkpoint_keys(
    stage_cuts: &Bound<'_, PyAny>,
    stage_bases: Option<&Bound<'_, PyAny>>,
) -> PyResult<()> {
    for (pool_index, pool) in stage_cuts.try_iter()?.enumerate() {
        let pool = pool?;
        let position = format!("stage_cuts[{pool_index}]");
        reject_retired_keys(&pool, &RETIRED_POOL_KEYS, &position)?;
        if pool.contains("cuts")? {
            for (cut_index, cut) in pool.get_item("cuts")?.try_iter()?.enumerate() {
                let position = format!("{position}.cuts[{cut_index}]");
                reject_retired_keys(&cut?, &RETIRED_CUT_KEYS, &position)?;
            }
        }
    }
    if let Some(stage_bases) = stage_bases {
        for (node_index, basis) in stage_bases.try_iter()?.enumerate() {
            let position = format!("stage_bases[{node_index}]");
            reject_retired_keys(&basis?, &RETIRED_BASIS_KEYS, &position)?;
        }
    }
    Ok(())
}

/// Reject `inflow_lag_coefficients` supplied without a positive
/// `inflow_lag_depth` — they would be silently dropped.
fn reject_unreserved_lag_coefficients(
    stage_cuts: &[PyStageCutsPayload],
    reserve_depth: Option<u32>,
) -> PyResult<()> {
    if reserve_depth.is_some() {
        return Ok(());
    }
    for sc in stage_cuts {
        for cut in &sc.cuts {
            if !cut.inflow_lag_coefficients.is_empty() {
                return Err(PyValueError::new_err(format!(
                    "stage {} cut at slot {}: inflow_lag_coefficients supplied without \
                     inflow_lag_depth; pass inflow_lag_depth=N to reserve the lag slots",
                    sc.stage_id, cut.slot_index
                )));
            }
        }
    }
    Ok(())
}

/// Owns the manifest and coefficients so borrowed [`StageCutsPayload`]/[`PolicyCutRecord`]
/// views can reference them (reservation widens coefficients, requiring owned storage;
/// no-reservation borrows `sc.cuts[*].coefficients` directly).
struct StageCutsData<'a> {
    manifest: Vec<EntitySlot>,
    coefficients: Vec<Cow<'a, [f64]>>,
    state_dimension: u32,
}

/// Convert one stage's manifest and coefficients to owned form, reserving the
/// canonical `HydroInflowLag` slots when `reserve_depth` is set.
fn build_stage_cuts_data(
    sc: &PyStageCutsPayload,
    reserve_depth: Option<u32>,
) -> PyResult<StageCutsData<'_>> {
    let manifest: Vec<EntitySlot> = sc.entity_manifest.iter().map(EntitySlot::from).collect();
    match reserve_depth {
        Some(depth) => {
            let coefficients: Vec<Vec<f64>> =
                sc.cuts.iter().map(|c| c.coefficients.clone()).collect();
            let cut_lag: Vec<HashMap<i32, Vec<f64>>> = sc
                .cuts
                .iter()
                .map(|c| c.inflow_lag_coefficients.clone())
                .collect();
            let reserved =
                reserve_boundary_inflow_lag_slots(&manifest, &coefficients, &cut_lag, depth)
                    .map_err(|e| match e {
                        SddpError::Validation(msg) => PyValueError::new_err(msg),
                        other => PyValueError::new_err(other.to_string()),
                    })?;
            Ok(StageCutsData {
                manifest: reserved.manifest,
                coefficients: reserved.coefficients.into_iter().map(Cow::Owned).collect(),
                state_dimension: reserved.state_dimension,
            })
        }
        None => Ok(StageCutsData {
            manifest,
            coefficients: sc
                .cuts
                .iter()
                .map(|c| Cow::Borrowed(c.coefficients.as_slice()))
                .collect(),
            state_dimension: sc.state_dimension,
        }),
    }
}

/// Write a policy checkpoint from plain Python dicts/sequences to `path`.
///
/// `stage_cuts` and `metadata` mirror the `"stage_cuts"`/`"metadata"` keys
/// [`crate::results::load_policy`] returns; `stage_bases`/`stage_states`
/// default to empty when omitted (a checkpoint authored from raw cut data
/// carries neither).
///
/// Every pool's `cost_scale_factor` is required: the pool's values are in the
/// units of that factor, and the producer block's `cost_scale_factor` is
/// provenance only, never a fallback.
///
/// `inflow_lag_depth`, when set to `N > 0`, has novomodelo reserve `N` canonical
/// `HydroInflowLag` state slots per storage hydro in every stage's manifest
/// (via [`reserve_boundary_inflow_lag_slots`]), placing each cut's
/// `inflow_lag_coefficients` at their `(hydro, depth)` positions. This is the
/// authoring path for a boundary policy of a case with no PAR model to infer
/// the depth from (a DECOMP-bridge bootstrap). Absent or `0`, the checkpoint is
/// byte-identical to one written without the argument.
///
/// The checkpoint always records this build's identity
/// ([`novomodelo_io::SoftwareIdentity::THIS_BUILD`]); `software`, `software_version`
/// or `novomodelo_version` keys in `metadata` are ignored.
///
/// # Errors
///
/// `ValueError` when a pool dict holds the retired `active_cut_indices`, a cut
/// dict `cut_id` or a basis dict `num_cut_rows`, `num_columns` or `num_rows`; a
/// pool's `cost_scale_factor` is absent, not finite or not positive; a cut's
/// `coefficients` length does not match its stage's `state_dimension`; a cut
/// carries `inflow_lag_coefficients` without a positive `inflow_lag_depth`; or
/// (under `inflow_lag_depth`) a manifest lacks a leading storage block or an
/// inflow-lag coefficient is unplaceable. `ValueError` also when the producer
/// claims `basis_store_complete` with no basis in `stage_bases`, when a producer
/// `state_history` is other than `None`, `"whole"` or a positive window, and
/// when [`check_checkpoint_payloads`] refuses the payloads, with its message.
/// A `season_manifest` whose `hydro_orders` are not ascending by `hydro_id` or
/// whose `orders` lengths disagree with `n_seasons` is written as given and
/// rejected by [`crate::results::load_policy`] with `OutputError`. A failed
/// write raises the `novomodelo.errors` leaf mapped from the underlying
/// [`novomodelo_io::OutputError`].
#[pyfunction]
#[pyo3(signature = (path, stage_cuts, metadata, stage_bases=None, stage_states=None, inflow_lag_depth=None))]
#[allow(clippy::needless_pass_by_value)]
pub fn write_policy_checkpoint(
    py: Python<'_>,
    path: PathBuf,
    stage_cuts: &Bound<'_, PyAny>,
    metadata: PyPolicyCheckpointMetadata,
    stage_bases: Option<&Bound<'_, PyAny>>,
    stage_states: Option<Vec<PyStageStatesPayload>>,
    inflow_lag_depth: Option<u32>,
) -> PyResult<()> {
    reject_retired_checkpoint_keys(stage_cuts, stage_bases)?;
    let stage_cuts: Vec<PyStageCutsPayload> = stage_cuts.extract()?;
    let stage_bases: Vec<PyBasisRecord> = stage_bases
        .map(PyAnyMethods::extract)
        .transpose()?
        .unwrap_or_default();
    let stage_states = stage_states.unwrap_or_default();

    let populated_counts = resolve_populated_counts(&stage_cuts)?;
    let cost_scale_factors = pool_cost_scale_factors(&stage_cuts)?;
    reject_unreceived_basis_store(&metadata.producer, &stage_bases)?;

    let reserve_depth = inflow_lag_depth.filter(|&n| n > 0);
    reject_unreserved_lag_coefficients(&stage_cuts, reserve_depth)?;
    let stage_data: Vec<StageCutsData<'_>> = stage_cuts
        .iter()
        .map(|sc| build_stage_cuts_data(sc, reserve_depth))
        .collect::<PyResult<_>>()?;

    let metadata = metadata.into_manifest()?;

    let written = py.detach(|| -> PyResult<Result<(), OutputError>> {
        let cut_records: Vec<Vec<PolicyCutRecord<'_>>> = stage_cuts
            .iter()
            .zip(&stage_data)
            .map(|(sc, data)| {
                sc.cuts
                    .iter()
                    .zip(&data.coefficients)
                    .map(|(c, coefficients)| PolicyCutRecord {
                        slot_index: c.slot_index,
                        iteration: c.iteration,
                        forward_pass_index: c.forward_pass_index,
                        intercept: c.intercept,
                        coefficients,
                        is_active: c.is_active,
                        scored: c.scored,
                        active_count: c.active_count,
                        last_active_iteration: c.last_active_iteration,
                    })
                    .collect()
            })
            .collect();

        let stage_cuts_payloads: Vec<StageCutsPayload<'_>> = stage_cuts
            .iter()
            .zip(&stage_data)
            .enumerate()
            .map(|(i, (sc, data))| StageCutsPayload {
                stage_id: sc.stage_id,
                state_dimension: data.state_dimension,
                capacity: sc.capacity,
                warm_start_count: sc.warm_start_count,
                cuts: &cut_records[i],
                populated_count: populated_counts[i],
                entity_manifest: &data.manifest,
                cost_scale_factor: cost_scale_factors[i],
                node_id: sc.node_id,
                graph_stage_id: sc.graph_stage_id,
                priced_state_date: sc.priced_state_date,
            })
            .collect();

        let basis_entries: Vec<Vec<BasisEntry<'_>>> = stage_bases
            .iter()
            .map(|b| {
                b.entries
                    .iter()
                    .map(|e| BasisEntry {
                        scenario: e.scenario,
                        column_status: &e.column_status,
                        row_status: &e.row_status,
                        piece_row_slots: &e.piece_row_slots,
                    })
                    .collect()
            })
            .collect();
        let basis_records: Vec<PolicyBasisRecord<'_>> = stage_bases
            .iter()
            .zip(&basis_entries)
            .map(|(b, entries)| PolicyBasisRecord {
                stage_id: b.stage_id,
                iteration: b.iteration,
                entries,
            })
            .collect();

        let state_manifests: Vec<Vec<EntitySlot>> = stage_states
            .iter()
            .map(|ss| ss.entity_manifest.iter().map(EntitySlot::from).collect())
            .collect();

        let state_payloads: Vec<StageStatesPayload<'_>> = stage_states
            .iter()
            .enumerate()
            .map(|(i, ss)| StageStatesPayload {
                stage_id: ss.stage_id,
                node_id: ss.node_id,
                state_dimension: ss.state_dimension,
                count: ss.count,
                data: &ss.data,
                entity_manifest: &state_manifests[i],
                slots: &ss.slots,
            })
            .collect();

        check_checkpoint_payloads(
            &stage_cuts_payloads,
            &basis_records,
            &state_payloads,
            &metadata,
        )
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

        Ok(novomodelo_io::write_policy_checkpoint(
            &path,
            &stage_cuts_payloads,
            &basis_records,
            &metadata,
            &state_payloads,
        ))
    })?;
    written.map_err(|e| convert_error(ErrorSource::Output(&e)))
}

#[cfg(test)]
mod tests {
    use super::{EntitySlot, PyEntitySlot};

    #[test]
    fn py_entity_slot_unclassified_entity_type_round_trips() {
        let slot = PyEntitySlot {
            entity_type: 200,
            entity_id: 7,
            subindex: 3,
            was_active: true,
            reference_date: 20_260_102,
            interval_start: 20_260_103,
            interval_end: 20_260_104,
        };

        let converted = EntitySlot::from(&slot);

        assert_eq!(converted.entity_type, 200);
        assert_eq!(converted.entity_id, 7);
        assert_eq!(converted.subindex, 3);
        assert!(converted.was_active);
        assert_eq!(converted.reference_date, 20_260_102);
        assert_eq!(converted.interval_start, 20_260_103);
        assert_eq!(converted.interval_end, 20_260_104);
    }
}
