//! `External` scenario sampling scheme — library type and eta standardization.
//!
//! [`ExternalScenarioLibrary`] stores pre-standardized eta values loaded from
//! externally provided scenario files; the `ClassSampler::External` variant
//! indexes into it per (stage, scenario) during the forward pass. One library
//! instance per entity class.
//!
//! The `eta` buffer uses **stage-major** layout
//! (`eta[stage * n_scenarios * n_entities + scenario * n_entities + entity]`),
//! matching the forward-pass access pattern of all entities for one (stage,
//! scenario) pair.
//!
//! [`solve_par_noise`]: crate::par::evaluate::solve_par_noise

use std::collections::{HashMap, HashSet};

use cobre_core::{
    EntityId,
    scenario::{ExternalLoadRow, ExternalNcsRow, ExternalScenarioRow, LoadModel, NcsModel},
    temporal::{Stage, StageLagTransition},
};

use crate::StochasticError;
use crate::par::precompute::PrecomputedPar;
use crate::seeds::DerivedSeed;

use super::eta_inversion::run_eta_inversion;

// ---------------------------------------------------------------------------
// ExternalScenarioLibrary
// ---------------------------------------------------------------------------

/// Pre-standardized eta store for external scenario files.
///
/// A pure data container: the external-file parsing pass populates it, the
/// `ClassSampler::External` variant reads it during the forward pass. An eta
/// of `(μ = 0, σ = 1, ψ = 0)` is not a placeholder or an absence marker — it
/// IS the natural value, the identity that makes a stored entry interpretable
/// on its own.
///
/// # Examples
///
/// ```
/// use cobre_stochastic::ExternalScenarioLibrary;
///
/// let raw = vec![50usize; 12];
/// let mut lib = ExternalScenarioLibrary::new(12, 50, 5, "inflow", raw);
/// assert_eq!(lib.n_stages(), 12);
/// assert_eq!(lib.n_scenarios(), 50);
/// assert_eq!(lib.n_entities(), 5);
/// assert_eq!(lib.entity_class(), "inflow");
///
/// // Write and read eta values.
/// lib.eta_slice_mut(0, 1).copy_from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
/// assert_eq!(lib.eta_slice(0, 1), &[1.0, 2.0, 3.0, 4.0, 5.0]);
/// ```
#[derive(Debug, Clone)]
pub struct ExternalScenarioLibrary {
    eta: Box<[f64]>,
    n_stages: usize,
    /// Padded, uniform per-stage count used to size the eta buffer.
    n_scenarios: usize,
    n_entities: usize,
    entity_class: &'static str,
    /// Pre-padding scenario count per stage; a stage with fewer raw scenarios
    /// than `n_scenarios` keeps its original count so downstream code can tell
    /// padded from unpadded stages.
    raw_scenarios_per_stage: Vec<usize>,
}

impl ExternalScenarioLibrary {
    /// Construct a new library with zero-filled buffers.
    ///
    /// # Parameters
    ///
    /// - `n_scenarios` — eta-buffer per-stage count (max across all stages)
    /// - `raw_scenarios_per_stage` — pre-padding count per stage (length must equal `n_stages`)
    #[must_use]
    pub fn new(
        n_stages: usize,
        n_scenarios: usize,
        n_entities: usize,
        entity_class: &'static str,
        raw_scenarios_per_stage: Vec<usize>,
    ) -> Self {
        debug_assert_eq!(
            raw_scenarios_per_stage.len(),
            n_stages,
            "raw_scenarios_per_stage.len() ({}) must equal n_stages ({})",
            raw_scenarios_per_stage.len(),
            n_stages,
        );
        Self {
            eta: vec![0.0_f64; n_stages * n_scenarios * n_entities].into_boxed_slice(),
            n_stages,
            n_scenarios,
            n_entities,
            entity_class,
            raw_scenarios_per_stage,
        }
    }

    // -----------------------------------------------------------------------
    // Dimension accessors
    // -----------------------------------------------------------------------

    /// Returns the number of study stages.
    #[must_use]
    #[inline]
    pub fn n_stages(&self) -> usize {
        self.n_stages
    }

    /// Returns the number of scenarios per stage.
    #[must_use]
    #[inline]
    pub fn n_scenarios(&self) -> usize {
        self.n_scenarios
    }

    /// Returns the number of entities in the eta vector.
    #[must_use]
    #[inline]
    pub fn n_entities(&self) -> usize {
        self.n_entities
    }

    /// Returns the entity class label for diagnostic messages.
    #[must_use]
    #[inline]
    pub fn entity_class(&self) -> &str {
        self.entity_class
    }

    /// Returns the pre-padding scenario count per stage.
    #[must_use]
    #[inline]
    pub fn raw_scenarios_per_stage(&self) -> &[usize] {
        &self.raw_scenarios_per_stage
    }

    // -----------------------------------------------------------------------
    // Eta accessors
    // -----------------------------------------------------------------------

    /// Returns the `n_entities`-length slice of eta values for `(stage, scenario)`.
    ///
    /// # Panics
    ///
    /// Panics if `stage >= n_stages` or `scenario >= n_scenarios`.
    #[must_use]
    #[inline]
    pub fn eta_slice(&self, stage: usize, scenario: usize) -> &[f64] {
        assert!(
            stage < self.n_stages,
            "stage ({stage}) must be < n_stages ({})",
            self.n_stages
        );
        assert!(
            scenario < self.n_scenarios,
            "scenario ({scenario}) must be < n_scenarios ({})",
            self.n_scenarios
        );
        let offset = (stage * self.n_scenarios + scenario) * self.n_entities;
        &self.eta[offset..offset + self.n_entities]
    }

    /// Returns a mutable `n_entities`-length slice of eta values for `(stage, scenario)`.
    ///
    /// # Panics
    ///
    /// Panics if `stage >= n_stages` or `scenario >= n_scenarios`.
    #[must_use]
    #[inline]
    pub fn eta_slice_mut(&mut self, stage: usize, scenario: usize) -> &mut [f64] {
        assert!(
            stage < self.n_stages,
            "stage ({stage}) must be < n_stages ({})",
            self.n_stages
        );
        assert!(
            scenario < self.n_scenarios,
            "scenario ({scenario}) must be < n_scenarios ({})",
            self.n_scenarios
        );
        let offset = (stage * self.n_scenarios + scenario) * self.n_entities;
        &mut self.eta[offset..offset + self.n_entities]
    }
}

/// Map each study stage's declared `Stage.id` to its 0-based position in
/// `stages` — the declared-domain-id -> canonical-study-index resolution
/// cobre-io's `StageIdResolver` performs for the σ=0 validator (rule 47),
/// replicated here because `cobre-stochastic` cannot depend on `cobre-io`
/// (`cobre-io` depends on `cobre-stochastic`). `ExternalScenarioRow`/
/// `ExternalLoadRow`/`ExternalNcsRow`'s own `stage_id` doc is explicit that
/// the value is a declared domain id, "not a 0-based index" — every
/// External-row consumer below resolves through this map, never casts
/// `stage_id as usize` directly.
fn stage_id_to_index(stages: &[Stage]) -> HashMap<i32, usize> {
    stages.iter().enumerate().map(|(i, s)| (s.id, i)).collect()
}

// ---------------------------------------------------------------------------
// standardize_external_inflow
// ---------------------------------------------------------------------------

/// Populate `library` with standardized eta values from external inflow rows.
///
/// For each (stage, scenario, hydro), inverts the PAR(p) model via
/// [`solve_par_noise`](crate::par::evaluate::solve_par_noise) to produce the
/// noise `η` that the forward PAR pass would turn back into the raw external
/// value, using a lag chain seeded from `seed` ([`DerivedSeed`]) and advanced
/// by the `stage_lag_transitions` accumulate/finalize pattern (lags frozen
/// within a period; shifted with the period's weighted-average raw value at
/// each `finalize_period` boundary).
///
/// A `f64::NEG_INFINITY` from `solve_par_noise` (sigma=0, non-matching target)
/// is stored as-is; V3.7 in [`validate_external_library`] rejects it for a
/// stage's real (pre-padding) scenarios. A phantom padding slot may also hold
/// the sentinel, harmlessly — [`pad_library_to_uniform`] overwrites it with a
/// real replica before the library is ever read.
///
/// A stage with fewer real scenarios than another stage has its raw values
/// replicated to the library's uniform width BEFORE inversion (the same
/// wrap-around convention [`pad_library_to_uniform`] applies to the eta
/// output), so a later real stage's lag chain — `run_eta_inversion`'s
/// `advance_lag_chain` step — advances through that replica, never a
/// fabricated `0.0`: every branch descending from one earlier real scenario
/// shares that scenario's own realized value as its lag, matching what the
/// forward pass's own state carry-over feeds a shared parent's children.
///
/// # Parameters
///
/// - `library` — destination, must have `n_entities() == hydro_ids.len()`
/// - `external_rows` — raw rows sorted by `(stage_id, scenario_id, hydro_id)`
/// - `hydro_ids` — canonical-order hydro entity IDs
/// - `seed` — stage-0 lag/accumulator seed; see [`DerivedSeed`]
/// - `stage_lag_transitions` — one per stage, same length as `stages`
/// - `downstream_par_order` — PAR order of the downstream (coarser) resolution;
///   `0` for uniform-resolution studies. Reuse the same value the forward pass
///   was set up with — recomputing it independently here can size the sampler's
///   ring differently and desync the replay from the forward lag chain.
///
/// # Panics
///
/// Panics in debug builds if dimension mismatches are detected.
// Rationale: `seed` folds the former lag/accum/weight arguments into one
// aggregate; the remaining library/rows/entity-list/stage-list/par-model/
// lag-transition/downstream-order inputs are independent, with no further
// shared aggregate.
#[allow(clippy::too_many_arguments)]
pub fn standardize_external_inflow(
    library: &mut ExternalScenarioLibrary,
    external_rows: &[ExternalScenarioRow],
    hydro_ids: &[EntityId],
    stages: &[Stage],
    par: &PrecomputedPar,
    seed: DerivedSeed<'_>,
    stage_lag_transitions: &[StageLagTransition],
    downstream_par_order: usize,
) {
    let n_stages = library.n_stages();
    let n_scenarios = library.n_scenarios();
    let n_hydros = hydro_ids.len();
    let max_order = par.max_order();

    debug_assert_eq!(
        library.n_entities(),
        n_hydros,
        "library.n_entities() ({}) must equal hydro_ids.len() ({})",
        library.n_entities(),
        n_hydros,
    );
    debug_assert_eq!(
        n_stages,
        stages.len(),
        "library.n_stages() ({}) must equal stages.len() ({})",
        n_stages,
        stages.len(),
    );
    debug_assert_eq!(
        stage_lag_transitions.len(),
        n_stages,
        "stage_lag_transitions.len() ({}) must equal n_stages ({})",
        stage_lag_transitions.len(),
        n_stages,
    );

    if n_hydros == 0 || n_stages == 0 || n_scenarios == 0 {
        return;
    }

    let hydro_index: HashMap<EntityId, usize> = hydro_ids
        .iter()
        .enumerate()
        .map(|(i, &id)| (id, i))
        .collect();
    let stage_index = stage_id_to_index(stages);

    // raw[stage * n_scenarios * n_hydros + scenario * n_hydros + h_idx]
    let mut raw_values = vec![0.0_f64; n_stages * n_scenarios * n_hydros].into_boxed_slice();

    for row in external_rows {
        debug_assert!(
            row.stage_id >= 0,
            "negative stage_id in external scenario row"
        );
        debug_assert!(
            row.scenario_id >= 0,
            "negative scenario_id in external scenario row"
        );
        let Some(&stage_idx) = stage_index.get(&row.stage_id) else {
            continue;
        };
        #[allow(clippy::cast_sign_loss)]
        let scenario_idx = row.scenario_id as usize;
        let Some(&h_idx) = hydro_index.get(&row.hydro_id) else {
            continue;
        };
        // Defensive bound: cobre-io's A1 (exact scenario_id set) is the
        // load-bearing guard that rejects an out-of-range scenario_id at
        // load; this keeps a stray index from writing into a neighbouring
        // stage's realization 0 instead of being caught.
        if scenario_idx >= n_scenarios {
            continue;
        }
        raw_values[stage_idx * n_scenarios * n_hydros + scenario_idx * n_hydros + h_idx] =
            row.value_m3s;
    }

    for (stage_idx, &raw_count) in library
        .raw_scenarios_per_stage()
        .iter()
        .enumerate()
        .take(n_stages)
    {
        if raw_count == 0 || raw_count >= n_scenarios {
            continue;
        }
        replicate_to_uniform_width(&mut raw_values, n_scenarios, n_hydros, stage_idx, raw_count);
    }

    run_eta_inversion(
        n_stages,
        n_scenarios,
        n_hydros,
        max_order,
        par,
        seed,
        stage_lag_transitions,
        downstream_par_order,
        |t, scenario, h| raw_values[t * n_scenarios * n_hydros + scenario * n_hydros + h],
        |t, scenario, h, eta| library.eta_slice_mut(t, scenario)[h] = eta,
    );
}

// ---------------------------------------------------------------------------
// standardize_external_simple
// ---------------------------------------------------------------------------

/// Sole owner of the simple `η = (value - mean) / std` standardization shared by
/// every non-PAR(p) external entity class.
///
/// The `std == 0.0` case yields `η = 0.0` (deterministic entity, per the sigma=0
/// convention); dividing unconditionally would emit NaN/inf instead. The
/// `mean_std` lookup is `mean_std[stage * n_entities + idx]` — callers and this
/// body must agree on that index expression.
///
/// Determinism: `models`, `external_rows`, and `entity_ids` are traversed in the
/// order given with no collect/sort/reorder, so output depends only on input
/// declaration order.
///
/// Carries no entity vocabulary — the two accessor closures supply per-class
/// field access; [`standardize_external_load`] / [`standardize_external_ncs`] own it.
///
/// # Parameters
///
/// - `library` — destination, must have `n_entities() == entity_ids.len()`
/// - `model_fields` — yields `(entity_id, stage_id, mean, std)` for one model
/// - `row_fields` — yields `(entity_id, stage_id, scenario_id, value)` for one row
///
/// # Panics
///
/// Panics in debug builds if dimension mismatches are detected.
fn standardize_external_simple<R, M, FM, FR>(
    library: &mut ExternalScenarioLibrary,
    external_rows: &[R],
    entity_ids: &[EntityId],
    models: &[M],
    stages: &[Stage],
    model_fields: FM,
    row_fields: FR,
) where
    FM: Fn(&M) -> (EntityId, i32, f64, f64),
    FR: Fn(&R) -> (EntityId, i32, i32, f64),
{
    let n_entities = entity_ids.len();
    let n_stages = stages.len();
    let n_scenarios = library.n_scenarios();

    debug_assert_eq!(
        library.n_entities(),
        n_entities,
        "library.n_entities() ({}) must equal entity_ids.len() ({})",
        library.n_entities(),
        n_entities,
    );
    debug_assert_eq!(
        library.n_stages(),
        n_stages,
        "library.n_stages() ({}) must equal stages.len() ({})",
        library.n_stages(),
        n_stages,
    );

    if n_entities == 0 || n_stages == 0 || n_scenarios == 0 {
        return;
    }

    let entity_index: HashMap<EntityId, usize> = entity_ids
        .iter()
        .enumerate()
        .map(|(i, &id)| (id, i))
        .collect();
    let stage_index = stage_id_to_index(stages);

    // mean_std[stage * n_entities + entity_idx]
    let mut mean_std = vec![(0.0_f64, 0.0_f64); n_stages * n_entities];
    for model in models {
        let (entity_id, stage_id, mean, std) = model_fields(model);
        if let (Some(&e_idx), Some(&stage_idx)) =
            (entity_index.get(&entity_id), stage_index.get(&stage_id))
        {
            mean_std[stage_idx * n_entities + e_idx] = (mean, std);
        }
    }

    #[allow(clippy::cast_sign_loss)]
    for row in external_rows {
        let (entity_id, stage_id, scenario_id, value) = row_fields(row);
        let Some(&stage_idx) = stage_index.get(&stage_id) else {
            continue;
        };
        let scenario_idx = scenario_id as usize;
        let Some(&e_idx) = entity_index.get(&entity_id) else {
            continue;
        };
        // Defensive bound: cobre-io's A1 is the load-bearing guard that
        // rejects an out-of-range scenario_id at load; this keeps a stray
        // index from writing into a neighbouring stage's realization 0.
        if scenario_idx >= n_scenarios {
            continue;
        }
        let (mean, std) = mean_std[stage_idx * n_entities + e_idx];
        let eta = if std == 0.0 {
            0.0
        } else {
            (value - mean) / std
        };
        library.eta_slice_mut(stage_idx, scenario_idx)[e_idx] = eta;
    }
}

// ---------------------------------------------------------------------------
// derive_external_sample_moments
// ---------------------------------------------------------------------------

/// Derive per-`(entity, stage)` sample `(μ, σ)` over the scenario axis of
/// external scenario rows. Population (`1/N`) divisor, matching
/// [`SeasonalStats`](crate::par::SeasonalStats)'s convention — the
/// population-vs-sample choice only rescales the internal η, not the
/// standardize/reconstruct round trip.
///
/// Output layout matches the `mean_std` table `standardize_external_simple`
/// consumes: `moments[stage * entity_ids.len() + entity_idx]`. An
/// `(entity, stage)` cell with no rows defaults to `(0.0, 0.0)`.
///
/// `row_fields`' `stage_id` MUST already be the canonical 0-based study-stage
/// index, never the row's raw declared domain id — this reduction has no
/// stage list to resolve one against. Every caller resolves first (this
/// crate's `stage_id_to_index`, or the equivalent `cobre-io`
/// `StageIdResolver`); passing a raw `ExternalScenarioRow`/`ExternalLoadRow`/
/// `ExternalNcsRow::stage_id` directly silently mis-indexes or drops rows
/// whenever a study's declared stage ids are not exactly `0..n_stages`.
///
/// Determinism: each cell reduces its rows in ascending `(scenario_id, value)`
/// order, never `external_rows`' declaration order, so any permutation of the
/// same rows yields a bit-identical table (D5). A bit-identical constant cell
/// short-circuits to `σ = 0.0` exactly and `μ` equal to the constant — the
/// two-pass sum/variance below would otherwise round a constant column to a
/// tiny nonzero `σ` (catastrophic cancellation).
///
/// # Parameters
///
/// - `row_fields` — yields `(entity_id, stage_id, scenario_id, value)` for one row
///
/// # Panics
///
/// Panics in debug builds if `row_fields` yields a negative `stage_id` or `scenario_id`.
#[must_use]
pub fn derive_external_sample_moments<R, FR>(
    external_rows: &[R],
    entity_ids: &[EntityId],
    n_stages: usize,
    row_fields: FR,
) -> Vec<(f64, f64)>
where
    FR: Fn(&R) -> (EntityId, i32, i32, f64),
{
    let n_entities = entity_ids.len();
    let mut moments = vec![(0.0_f64, 0.0_f64); n_stages * n_entities];
    if n_entities == 0 || n_stages == 0 {
        return moments;
    }

    let entity_index: HashMap<EntityId, usize> = entity_ids
        .iter()
        .enumerate()
        .map(|(i, &id)| (id, i))
        .collect();

    let mut cells: Vec<Vec<(i32, f64)>> = vec![Vec::new(); n_stages * n_entities];

    #[allow(clippy::cast_sign_loss)]
    for row in external_rows {
        let (entity_id, stage_id, scenario_id, value) = row_fields(row);
        debug_assert!(stage_id >= 0, "negative stage_id in external scenario row");
        debug_assert!(
            scenario_id >= 0,
            "negative scenario_id in external scenario row"
        );
        let Some(&e_idx) = entity_index.get(&entity_id) else {
            continue;
        };
        let stage_idx = stage_id as usize;
        if stage_idx >= n_stages {
            continue;
        }
        cells[stage_idx * n_entities + e_idx].push((scenario_id, value));
    }

    for (cell_idx, cell) in cells.iter_mut().enumerate() {
        if cell.is_empty() {
            continue;
        }
        cell.sort_unstable_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.total_cmp(&b.1)));
        moments[cell_idx] = cell_population_moments(cell);
    }

    moments
}

/// Population `(μ, σ)` of one canonically-ordered `(scenario_id, value)` cell.
#[allow(clippy::float_cmp)] // intentional bit-identical equality, not a tolerance check
fn cell_population_moments(cell: &[(i32, f64)]) -> (f64, f64) {
    let first = cell[0].1;
    if cell.iter().all(|&(_, v)| v == first) {
        return (first, 0.0);
    }

    // Rationale: cell.len() is a per-(entity, stage) scenario count, far
    // within f64's 2^53 exact range.
    #[allow(clippy::cast_precision_loss)]
    let n = cell.len() as f64;
    let mean = cell.iter().map(|&(_, v)| v).sum::<f64>() / n;
    let variance = cell
        .iter()
        .map(|&(_, v)| (v - mean) * (v - mean))
        .sum::<f64>()
        / n;
    (mean, variance.sqrt())
}

// ---------------------------------------------------------------------------
// standardize_external_load
// ---------------------------------------------------------------------------

/// Populate `library` with standardized eta from external load rows, via
/// `standardize_external_simple` with the [`LoadModel`] field accessors.
///
/// `library` must have `n_entities() == bus_ids.len()`.
///
/// # Panics
///
/// Panics in debug builds if dimension mismatches are detected.
pub fn standardize_external_load(
    library: &mut ExternalScenarioLibrary,
    external_rows: &[ExternalLoadRow],
    bus_ids: &[EntityId],
    load_models: &[LoadModel],
    stages: &[Stage],
) {
    standardize_external_simple(
        library,
        external_rows,
        bus_ids,
        load_models,
        stages,
        |model| (model.bus_id, model.stage_id, model.mean_mw, model.std_mw),
        |row| (row.bus_id, row.stage_id, row.scenario_id, row.value_mw),
    );
}

// ---------------------------------------------------------------------------
// standardize_external_ncs
// ---------------------------------------------------------------------------

/// Populate `library` with standardized eta from external NCS rows, via
/// `standardize_external_simple` with the [`NcsModel`] field accessors.
///
/// `library` must have `n_entities() == ncs_ids.len()`.
///
/// # Panics
///
/// Panics in debug builds if dimension mismatches are detected.
pub fn standardize_external_ncs(
    library: &mut ExternalScenarioLibrary,
    external_rows: &[ExternalNcsRow],
    ncs_ids: &[EntityId],
    ncs_models: &[NcsModel],
    stages: &[Stage],
) {
    standardize_external_simple(
        library,
        external_rows,
        ncs_ids,
        ncs_models,
        stages,
        |model| (model.ncs_id, model.stage_id, model.mean, model.std),
        |row| (row.ncs_id, row.stage_id, row.scenario_id, row.value),
    );
}

// ---------------------------------------------------------------------------
// validate_external_library
// ---------------------------------------------------------------------------

/// Validate a populated [`ExternalScenarioLibrary`] against construction inputs.
///
/// Runs after per-class file parsing and eta standardization, before the
/// library is stored on `StudySetup`; the first failed error check returns
/// `Err`.
///
/// ## Checks performed
///
/// | ID   | Kind    | Description                                                              |
/// |------|---------|--------------------------------------------------------------------------|
/// | V3.2 | Error   | Every entity in `entity_ids` must have data in `row_entity_ids`.         |
/// | V3.3 | Error   | Every study stage must have at least one row in `rows_per_stage`.        |
/// | V3.4 | Error   | Each stage's row count must be divisible by `n_entities` (non-uniform counts allowed). |
/// | V3.5 | Error   | Every entity ID in `row_entity_ids` must exist in `entity_ids`.          |
/// | V3.6 | Assert  | All values in raw rows are finite (parser invariant — `debug_assert`).   |
/// | V3.7 | Error   | No eta value at a stage's real (pre-padding) scenario may be `f64::NEG_INFINITY` or `NaN`. |
/// | V3.8 | Warning | `library.n_scenarios() < forward_passes` — log a warning.               |
///
/// The caller pre-extracts `row_entity_ids` (entity IDs present in the raw rows)
/// and `rows_per_stage` (raw-row count per stage across all entities and
/// scenarios; length must equal `n_stages`).
///
/// # Errors
///
/// Returns [`StochasticError::InsufficientData`] with a message prefixed by the
/// check ID (e.g., `"V3.2: ..."`) for the first failed error check.
///
/// # Examples
///
/// ```
/// use std::collections::HashSet;
/// use cobre_core::EntityId;
/// use cobre_stochastic::{ExternalScenarioLibrary, sampling::external::validate_external_library};
///
/// let lib = ExternalScenarioLibrary::new(3, 50, 2, "inflow", vec![50usize; 3]);
/// let entity_ids = [EntityId(1), EntityId(2)];
/// let row_entity_ids: HashSet<EntityId> = entity_ids.iter().copied().collect();
/// // 50 scenarios × 2 entities = 100 rows per stage.
/// let rows_per_stage = vec![100usize; 3];
/// let result = validate_external_library(
///     &lib, &entity_ids, &row_entity_ids, &rows_per_stage, 3, 50,
/// );
/// assert!(result.is_ok());
/// ```
// Rationale: `implicit_hasher` wants a `<S: BuildHasher>` generic, but every
// caller passes a default-`RandomState` `HashSet` and the membership test is
// hasher-agnostic — declaration-order invariance rides the ordered `entity_ids`
// slice, not this set — so the concrete type is deliberate.
#[allow(clippy::implicit_hasher)]
pub fn validate_external_library(
    library: &ExternalScenarioLibrary,
    entity_ids: &[EntityId],
    row_entity_ids: &HashSet<EntityId>,
    rows_per_stage: &[usize],
    n_stages: usize,
    forward_passes: u32,
) -> Result<(), StochasticError> {
    let n_entities = entity_ids.len();
    let class = library.entity_class();

    for &id in entity_ids {
        if !row_entity_ids.contains(&id) {
            return Err(StochasticError::InsufficientData {
                context: format!(
                    "V3.2: external {class} library missing data for {class} {id}; \
                     entity has zero rows in the external file",
                    id = id.0,
                ),
            });
        }
    }

    for (stage_idx, &count) in rows_per_stage.iter().enumerate().take(n_stages) {
        if count == 0 {
            return Err(StochasticError::InsufficientData {
                context: format!(
                    "V3.3: external {class} library has no rows for stage {stage_idx}; \
                     every study stage must have at least one row",
                ),
            });
        }
    }

    // V3.4 accepts non-uniform counts; `pad_library_to_uniform` fills them
    // afterward. The zero-entity guard skips an empty (benign) library to avoid
    // a div-by-zero.
    if n_entities > 0 && n_stages > 0 {
        for (stage_idx, &count) in rows_per_stage.iter().enumerate().take(n_stages) {
            if count % n_entities != 0 {
                return Err(StochasticError::InsufficientData {
                    context: format!(
                        "V3.4: external {class} library has {count} rows for stage \
                         {stage_idx} which is not exactly divisible by {n_entities} \
                         entities; each stage must have a whole number of scenarios",
                    ),
                });
            }
        }
    }

    let entity_id_set: HashSet<EntityId> = entity_ids.iter().copied().collect();
    for &id in row_entity_ids {
        if !entity_id_set.contains(&id) {
            return Err(StochasticError::InsufficientData {
                context: format!(
                    "V3.5: external {class} library contains unknown entity ID {id}; \
                     the ID does not exist in the canonical {class} entity list",
                    id = id.0,
                ),
            });
        }
    }

    // V3.6 — finite values are a parser invariant; debug_assert only, no full rescan.
    debug_assert!(
        row_entity_ids.iter().all(|id| entity_id_set.contains(id)),
        "V3.6: row_entity_ids contains IDs not in entity_id_set (parser invariant violated)",
    );

    // V3.7 — no eta may be NEG_INFINITY (the sigma=0 non-matching-target sentinel
    // from standardize_external_inflow) or NaN (numerical failure). Bounded to
    // `raw_scenarios_per_stage` (each stage's REAL scenario count), never
    // `library.n_scenarios()` (the padded, uniform width): a phantom slot beyond
    // the real count is unconditionally overwritten by the later
    // `pad_library_to_uniform` call with a real replica, so checking the padded
    // width here would reject a stage that legitimately has fewer real scenarios
    // than another stage (e.g. a branching root's single external observation).
    for stage in 0..library.n_stages() {
        let real_scenarios = library.raw_scenarios_per_stage()[stage];
        for scenario in 0..real_scenarios {
            let eta = library.eta_slice(stage, scenario);
            for (entity_idx, &value) in eta.iter().enumerate() {
                if value == f64::NEG_INFINITY || value.is_nan() {
                    return Err(StochasticError::InsufficientData {
                        context: format!(
                            "V3.7: external {class} library contains non-finite eta at \
                             stage {stage}, scenario {scenario}, entity {entity_idx} \
                             (value = {value}) — sigma=0 with non-matching external value \
                             or numerical failure",
                        ),
                    });
                }
            }
        }
    }

    if library.n_scenarios() < forward_passes as usize {
        tracing::warn!(
            n_scenarios = library.n_scenarios(),
            forward_passes = forward_passes,
            entity_class = class,
            "external {class} library has fewer scenarios ({n_scenarios}) than forward \
             passes ({forward_passes}); scenarios will be reused across forward passes",
            n_scenarios = library.n_scenarios(),
            forward_passes = forward_passes,
        );
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// pad_library_to_uniform
// ---------------------------------------------------------------------------

/// Replicate one stage's real (pre-padding) scenario slots into that stage's
/// remaining `raw_count..n_scenarios` range via wrap-around indexing
/// (`k % raw_count`), on a flat `(stage, scenario)`-major buffer with an
/// `n_entities`-wide innermost stride — the shared wrap-around convention
/// [`pad_library_to_uniform`] (the eta output) and
/// [`standardize_external_inflow`] (the pre-inversion raw-value buffer) both
/// apply, so the two padding operations can never disagree on how a phantom
/// slot's value is chosen.
fn replicate_to_uniform_width(
    buffer: &mut [f64],
    n_scenarios: usize,
    n_entities: usize,
    stage_idx: usize,
    raw_count: usize,
) {
    for k in raw_count..n_scenarios {
        let src_k = k % raw_count;
        let src_offset = (stage_idx * n_scenarios + src_k) * n_entities;
        let dst_offset = (stage_idx * n_scenarios + k) * n_entities;
        buffer.copy_within(src_offset..src_offset + n_entities, dst_offset);
    }
}

/// Fill each under-populated stage to `library.n_scenarios()` by replicating its
/// raw scenario slots with wrap-around indexing (`k % raw_count`), so every
/// scenario index holds a valid eta vector. No-op for already-uniform input.
///
/// # Panics
///
/// Does not panic — all indices are derived from library dimensions.
pub fn pad_library_to_uniform(library: &mut ExternalScenarioLibrary) {
    let n_scenarios = library.n_scenarios();
    let n_stages = library.n_stages();
    let n_entities = library.n_entities();
    // Owned so the tracing macro can use it after the mutable-borrow loop.
    let class = library.entity_class().to_owned();

    let mut padded_stages: Vec<(usize, usize)> = Vec::new();

    for s in 0..n_stages {
        let raw_count = library.raw_scenarios_per_stage[s];
        if raw_count == 0 || raw_count >= n_scenarios {
            continue;
        }

        padded_stages.push((s, raw_count));
        replicate_to_uniform_width(&mut library.eta, n_scenarios, n_entities, s, raw_count);
    }

    if !padded_stages.is_empty() {
        let stage_list: Vec<String> = padded_stages
            .iter()
            .map(|(s, raw)| format!("stage {s} ({raw}→{n_scenarios})"))
            .collect();
        tracing::info!(
            entity_class = class,
            padded_to = n_scenarios,
            "external {class} library padded to {n_scenarios} scenarios: {}",
            stage_list.join(", "),
        );
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp
)]
mod tests {
    use chrono::NaiveDate;
    use cobre_core::{
        EntityId, Hydro, InflowHistoryRow, RecentObservation,
        scenario::{
            AnnualComponent, ExternalLoadRow, ExternalNcsRow, ExternalScenarioRow, InflowModel,
            LoadModel, NcsModel,
        },
        temporal::{Stage, StageLagTransition},
        test_support::{HydroSpec, MirrorUnitGroup, StageSpec, date, single_block},
    };

    use super::{
        DerivedSeed, ExternalScenarioLibrary, derive_external_sample_moments,
        standardize_external_inflow, standardize_external_load, standardize_external_ncs,
    };
    use crate::derive_inflow_seeds;
    use crate::par::{
        DownstreamLagAccum, EntityMajor, PrimaryLagAccum, advance_lag_chain,
        evaluate::{evaluate_par, solve_par_noise},
        precompute::PrecomputedPar,
        precompute_stage_lag_transitions,
    };
    use crate::test_support::{MonthlyLabels, monthly_season_map};

    /// Build `n_stages` uniform-monthly transitions: each stage finalizes its own
    /// period with full weight and no spillover (the simple per-stage path).
    fn uniform_monthly_transitions(n_stages: usize) -> Vec<StageLagTransition> {
        vec![
            StageLagTransition {
                accumulate_weight: 1.0,
                spillover_weight: 0.0,
                finalize_period: true,
                accumulate_downstream: false,
                downstream_accumulate_weight: 0.0,
                downstream_spillover_weight: 0.0,
                downstream_finalize: false,
                rebuild_from_downstream: false,
            };
            n_stages
        ]
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    fn make_stage(index: usize, id: i32, season_id: usize) -> Stage {
        cobre_core::test_support::make_stage(StageSpec {
            id,
            index: Some(index),
            season_id: Some(season_id),
            blocks: single_block("SINGLE", 744.0),
            ..Default::default()
        })
    }

    fn make_inflow_model(
        hydro_id: i32,
        stage_id: i32,
        mean: f64,
        std: f64,
        ar: Vec<f64>,
    ) -> InflowModel {
        InflowModel {
            hydro_id: EntityId(hydro_id),
            stage_id,
            mean_m3s: mean,
            std_m3s: std,
            ar_coefficients: ar,
            residual_std_ratio: 1.0,
            annual: None,
        }
    }

    // -----------------------------------------------------------------------
    // Inflow standardization tests
    // -----------------------------------------------------------------------

    /// AR(0): 1 hydro, 2 stages, 1 scenario. Values [120.0, 90.0].
    /// Expected: eta[stage=0] = (120-100)/30 = 0.6667, eta[stage=1] = (90-100)/30 = -0.3333.
    #[test]
    fn test_inflow_ar0_standardization() {
        let hydro_id = EntityId(1);
        let hydro_ids = vec![hydro_id];

        // Two stages, both season 0 (single-season system).
        let stages = vec![make_stage(0, 0, 0), make_stage(1, 1, 0)];

        let models = vec![
            make_inflow_model(1, 0, 100.0, 30.0, vec![]),
            make_inflow_model(1, 1, 100.0, 30.0, vec![]),
        ];
        let par = PrecomputedPar::build(&models, &stages, &hydro_ids, None).unwrap();

        let mut lib = ExternalScenarioLibrary::new(2, 1, 1, "inflow", vec![1, 1]);

        let rows = vec![
            ExternalScenarioRow {
                stage_id: 0,
                scenario_id: 0,
                hydro_id,
                value_m3s: 120.0,
            },
            ExternalScenarioRow {
                stage_id: 1,
                scenario_id: 0,
                hydro_id,
                value_m3s: 90.0,
            },
        ];
        // AR(0) has no lags; the derived seed is irrelevant but must be provided.
        let transitions = uniform_monthly_transitions(stages.len());
        standardize_external_inflow(
            &mut lib,
            &rows,
            &hydro_ids,
            &stages,
            &par,
            DerivedSeed {
                lag_values: &[],
                l_state: 0,
                accum: &[],
                weight: &[],
            },
            &transitions,
            0,
        );

        let eta_0 = lib.eta_slice(0, 0)[0];
        let eta_1 = lib.eta_slice(1, 0)[0];

        assert!(
            (eta_0 - (120.0_f64 - 100.0) / 30.0).abs() < 1e-10,
            "eta[stage=0] = {eta_0}"
        );
        assert!(
            (eta_1 - (90.0_f64 - 100.0) / 30.0).abs() < 1e-10,
            "eta[stage=1] = {eta_1}"
        );
    }

    /// AR(1): stage 0 must use the derived lag seed (110.0) as lag-1,
    /// stage 1 must use the raw external value from stage 0 (130.0) as lag-1.
    ///
    /// Parameters: base=80, psi=[0.5], sigma=25. Derived lag seed: `[110.0]`.
    #[test]
    fn test_inflow_ar1_uses_external_lags() {
        let hydro_id = EntityId(1);
        let hydro_ids = vec![hydro_id];

        let stages = vec![make_stage(0, 0, 0), make_stage(1, 1, 0)];

        // From mean=160, std=25, psi*=0.5, PrecomputedPar computes psi_val=0.5,
        // base=80.0, sigma=25.0.
        let models = vec![
            make_inflow_model(1, 0, 160.0, 25.0, vec![0.5]),
            make_inflow_model(1, 1, 160.0, 25.0, vec![0.5]),
        ];
        let par = PrecomputedPar::build(&models, &stages, &hydro_ids, None).unwrap();

        assert!((par.deterministic_base(0, 0) - 80.0).abs() < 1e-10);
        assert!((par.sigma(0, 0) - 25.0).abs() < 1e-10);
        assert!((par.psi_slice(0, 0)[0] - 0.5).abs() < 1e-10);

        let mut lib = ExternalScenarioLibrary::new(2, 1, 1, "inflow", vec![1, 1]);
        let rows = vec![
            ExternalScenarioRow {
                stage_id: 0,
                scenario_id: 0,
                hydro_id,
                value_m3s: 130.0,
            },
            ExternalScenarioRow {
                stage_id: 1,
                scenario_id: 0,
                hydro_id,
                value_m3s: 95.0,
            },
        ];

        let derived_lag_values = [110.0];
        let transitions = uniform_monthly_transitions(stages.len());
        standardize_external_inflow(
            &mut lib,
            &rows,
            &hydro_ids,
            &stages,
            &par,
            DerivedSeed {
                lag_values: &derived_lag_values,
                l_state: 1,
                accum: &[],
                weight: &[],
            },
            &transitions,
            0,
        );

        // Stage 0: lag-1 = 110.0 (from the derived seed).
        // eta_0 = (130.0 - 80.0 - 0.5 * 110.0) / 25.0 = (130 - 80 - 55) / 25 = -5/25 = -0.2
        let det_base_0 = par.deterministic_base(0, 0);
        let psi_0 = par.psi_slice(0, 0)[0];
        let sigma_0 = par.sigma(0, 0);
        let expected_eta_0 = (130.0 - det_base_0 - psi_0 * 110.0) / sigma_0;
        let eta_0 = lib.eta_slice(0, 0)[0];
        assert!(
            (eta_0 - expected_eta_0).abs() < 1e-10,
            "eta[stage=0] = {eta_0}, expected {expected_eta_0}"
        );

        // Stage 1: lag-1 = raw external value at stage 0 = 130.0.
        // eta_1 = (95.0 - 80.0 - 0.5 * 130.0) / 25.0 = (95 - 80 - 65) / 25 = -50/25 = -2.0
        let det_base_1 = par.deterministic_base(1, 0);
        let psi_1 = par.psi_slice(1, 0)[0];
        let sigma_1 = par.sigma(1, 0);
        let expected_eta_1 = (95.0 - det_base_1 - psi_1 * 130.0) / sigma_1;
        let eta_1 = lib.eta_slice(1, 0)[0];
        assert!(
            (eta_1 - expected_eta_1).abs() < 1e-10,
            "eta[stage=1] = {eta_1}, expected {expected_eta_1}"
        );
    }

    /// AR(1) with 3 weekly stages all within the same lag period:
    ///   stage 0: `accumulate_weight`=0.4, `finalize_period`=false
    ///   stage 1: `accumulate_weight`=0.4, `finalize_period`=false
    ///   stage 2: `accumulate_weight`=0.2, `finalize_period`=true
    ///
    /// Parameters: base=80, psi=\[0.5\], sigma=25. Derived lag seed provides lag-1 = 110.0.
    ///
    /// Stages 0 and 1 must use the frozen derived-seed lag (110.0), NOT the
    /// previous stage's raw value. Stage 2 also uses 110.0 (still frozen during
    /// that stage's `solve_par_noise` call, since the shift happens after). The
    /// weighted average computed at finalize is: (200*0.4 + 160*0.4 + 120*0.2) = 168.0.
    #[test]
    #[allow(clippy::too_many_lines)]
    fn test_inflow_ar1_weekly_frozen_lags() {
        let hydro_id = EntityId(1);
        let hydro_ids = vec![hydro_id];

        // 3 stages, all season 0 — same PAR parameters apply.
        let stages = vec![
            make_stage(0, 0, 0),
            make_stage(1, 1, 0),
            make_stage(2, 2, 0),
        ];

        // AR(1): mean=160, std=25, psi*=0.5 → base=80, sigma=25.
        let models = vec![
            make_inflow_model(1, 0, 160.0, 25.0, vec![0.5]),
            make_inflow_model(1, 1, 160.0, 25.0, vec![0.5]),
            make_inflow_model(1, 2, 160.0, 25.0, vec![0.5]),
        ];
        let par = PrecomputedPar::build(&models, &stages, &hydro_ids, None).unwrap();

        let transitions = vec![
            StageLagTransition {
                accumulate_weight: 0.4,
                spillover_weight: 0.0,
                finalize_period: false,
                accumulate_downstream: false,
                downstream_accumulate_weight: 0.0,
                downstream_spillover_weight: 0.0,
                downstream_finalize: false,
                rebuild_from_downstream: false,
            },
            StageLagTransition {
                accumulate_weight: 0.4,
                spillover_weight: 0.0,
                finalize_period: false,
                accumulate_downstream: false,
                downstream_accumulate_weight: 0.0,
                downstream_spillover_weight: 0.0,
                downstream_finalize: false,
                rebuild_from_downstream: false,
            },
            StageLagTransition {
                accumulate_weight: 0.2,
                spillover_weight: 0.0,
                finalize_period: true,
                accumulate_downstream: false,
                downstream_accumulate_weight: 0.0,
                downstream_spillover_weight: 0.0,
                downstream_finalize: false,
                rebuild_from_downstream: false,
            },
        ];

        let mut lib = ExternalScenarioLibrary::new(3, 1, 1, "inflow", vec![1, 1, 1]);
        let rows = vec![
            ExternalScenarioRow {
                stage_id: 0,
                scenario_id: 0,
                hydro_id,
                value_m3s: 200.0,
            },
            ExternalScenarioRow {
                stage_id: 1,
                scenario_id: 0,
                hydro_id,
                value_m3s: 160.0,
            },
            ExternalScenarioRow {
                stage_id: 2,
                scenario_id: 0,
                hydro_id,
                value_m3s: 120.0,
            },
        ];

        let derived_lag_values = [110.0];

        standardize_external_inflow(
            &mut lib,
            &rows,
            &hydro_ids,
            &stages,
            &par,
            DerivedSeed {
                lag_values: &derived_lag_values,
                l_state: 1,
                accum: &[],
                weight: &[],
            },
            &transitions,
            0,
        );

        let det_base = par.deterministic_base(0, 0);
        let psi = par.psi_slice(0, 0)[0];
        let sigma = par.sigma(0, 0);

        // All three stages use the frozen lag-1 = 110.0 (from the derived seed).
        // The lag state is NOT shifted until stage 2 finalizes the period —
        // and even at stage 2 the shift happens AFTER solve_par_noise.
        let frozen_lag = 110.0_f64;
        let expected_eta_0 = (200.0 - det_base - psi * frozen_lag) / sigma;
        let expected_eta_1 = (160.0 - det_base - psi * frozen_lag) / sigma;
        let expected_eta_2 = (120.0 - det_base - psi * frozen_lag) / sigma;

        let eta_0 = lib.eta_slice(0, 0)[0];
        let eta_1 = lib.eta_slice(1, 0)[0];
        let eta_2 = lib.eta_slice(2, 0)[0];

        assert!(
            (eta_0 - expected_eta_0).abs() < 1e-10,
            "eta[stage=0] = {eta_0}, expected {expected_eta_0} (frozen lag)"
        );
        assert!(
            (eta_1 - expected_eta_1).abs() < 1e-10,
            "eta[stage=1] = {eta_1}, expected {expected_eta_1} (frozen lag, not stage-0 raw)"
        );
        assert!(
            (eta_2 - expected_eta_2).abs() < 1e-10,
            "eta[stage=2] = {eta_2}, expected {expected_eta_2} (frozen lag before finalize)"
        );

        // Also verify these differ from what naive per-stage advancement would produce.
        // With naive advancement, stage 1 would use raw value at stage 0 = 200.0.
        let naive_eta_1 = (160.0 - det_base - psi * 200.0) / sigma;
        assert!(
            (eta_1 - naive_eta_1).abs() > 1e-6,
            "eta[stage=1] must differ from naive per-stage value; got {eta_1} == naive {naive_eta_1}"
        );
    }

    /// AR(1): 2 stages where stage 0 has `spillover_weight > 0` and finalizes.
    ///
    /// stage 0: `accumulate_weight`=0.7, `spillover_weight`=0.3, `finalize_period`=true
    /// stage 1: `accumulate_weight`=1.0, `spillover_weight`=0.0, `finalize_period`=true
    ///
    /// Parameters: base=80, psi=\[0.5\], sigma=25. Derived lag seed: lag-1 = 110.0.
    /// raw values: stage 0 = 150.0, stage 1 = 130.0.
    ///
    /// Stage 0 computation:
    ///   lag for `solve_par_noise` = 110.0 (frozen from the derived seed)
    ///   accumulate: `lag_accum[0]` = 150.0 * 0.7 = 105.0, `lag_weight` = 0.7
    ///   finalize: avg = 105.0 / 0.7 = 150.0; `lag_state[0]` shifts to 150.0
    ///   spillover seed: `lag_accum[0]` = 150.0 * 0.3 = 45.0, `lag_weight` = 0.3
    ///
    /// Stage 1 computation:
    ///   lag for `solve_par_noise` = 150.0 (shifted in at stage 0 finalize)
    ///   accumulate: `lag_accum[0]` += 130.0 * 1.0 → 45.0 + 130.0 = 175.0, `lag_weight` = 1.3
    ///   finalize: avg = 175.0 / 1.3 ≈ 134.615...
    #[test]
    fn test_inflow_ar1_spillover_accumulation() {
        let hydro_id = EntityId(1);
        let hydro_ids = vec![hydro_id];

        let stages = vec![make_stage(0, 0, 0), make_stage(1, 1, 0)];

        let models = vec![
            make_inflow_model(1, 0, 160.0, 25.0, vec![0.5]),
            make_inflow_model(1, 1, 160.0, 25.0, vec![0.5]),
        ];
        let par = PrecomputedPar::build(&models, &stages, &hydro_ids, None).unwrap();

        let transitions = vec![
            StageLagTransition {
                accumulate_weight: 0.7,
                spillover_weight: 0.3,
                finalize_period: true,
                accumulate_downstream: false,
                downstream_accumulate_weight: 0.0,
                downstream_spillover_weight: 0.0,
                downstream_finalize: false,
                rebuild_from_downstream: false,
            },
            StageLagTransition {
                accumulate_weight: 1.0,
                spillover_weight: 0.0,
                finalize_period: true,
                accumulate_downstream: false,
                downstream_accumulate_weight: 0.0,
                downstream_spillover_weight: 0.0,
                downstream_finalize: false,
                rebuild_from_downstream: false,
            },
        ];

        let mut lib = ExternalScenarioLibrary::new(2, 1, 1, "inflow", vec![1, 1]);
        let rows = vec![
            ExternalScenarioRow {
                stage_id: 0,
                scenario_id: 0,
                hydro_id,
                value_m3s: 150.0,
            },
            ExternalScenarioRow {
                stage_id: 1,
                scenario_id: 0,
                hydro_id,
                value_m3s: 130.0,
            },
        ];

        let derived_lag_values = [110.0];

        standardize_external_inflow(
            &mut lib,
            &rows,
            &hydro_ids,
            &stages,
            &par,
            DerivedSeed {
                lag_values: &derived_lag_values,
                l_state: 1,
                accum: &[],
                weight: &[],
            },
            &transitions,
            0,
        );

        let det_base = par.deterministic_base(0, 0);
        let psi = par.psi_slice(0, 0)[0];
        let sigma = par.sigma(0, 0);

        // Stage 0: lag-1 = 110.0 (frozen from the derived seed before any finalize).
        let expected_eta_0 = (150.0 - det_base - psi * 110.0) / sigma;
        let eta_0 = lib.eta_slice(0, 0)[0];
        assert!(
            (eta_0 - expected_eta_0).abs() < 1e-10,
            "eta[stage=0] = {eta_0}, expected {expected_eta_0}"
        );

        // Stage 1: lag-1 = 150.0, shifted in at stage 0's finalize — not the
        // spillover-seeded accumulator, which only finalizes at stage 1's end.
        let expected_eta_1 = (130.0 - det_base - psi * 150.0) / sigma;
        let eta_1 = lib.eta_slice(1, 0)[0];
        assert!(
            (eta_1 - expected_eta_1).abs() < 1e-10,
            "eta[stage=1] = {eta_1}, expected {expected_eta_1} (lag = 150.0 from spillover period)"
        );
    }

    // -----------------------------------------------------------------------
    // Monthly→quarterly downstream ring
    // -----------------------------------------------------------------------

    /// Monthly→quarterly transition: 3 monthly stages (0,1,2) feed the downstream
    /// ring (weight 1/3 each, finalized at stage 2), stage 3 rebuilds the primary
    /// lag from the ring (`rebuild_from_downstream`), and stage 4's AR(1) eta
    /// reads that rebuilt lag — the first point downstream of the transition
    /// whose value depends on whether the ring fired, since stage 3's own eta is
    /// computed from the PRE-rebuild lag (unaffected by the ring either way).
    ///
    /// Oracle: ring average = (130+140+150)/3 = 140.0 (mirrors the forward
    /// `noise.rs` ring's weighted-accumulate-then-average). The negative control
    /// hand-computes what stage 4's eta would be under the old primary-only
    /// advance (ignoring `rebuild_from_downstream`, so stage 3's own raw value
    /// 500.0 — not the ring average — would have shifted into the lag) and
    /// asserts it differs from the kernel-routed result.
    #[test]
    fn quarterly_ring_sampler_external_matches_oracle() {
        let hydro_id = EntityId(1);
        let hydro_ids = vec![hydro_id];
        let stages: Vec<Stage> = (0_i32..5)
            .map(|i| make_stage(usize::try_from(i).unwrap(), i, 0))
            .collect();

        let mut models: Vec<InflowModel> = (0..4)
            .map(|sid| make_inflow_model(1, sid, 100.0, 10.0, vec![]))
            .collect();
        // Stage 4: AR(1), base=80, psi=0.5, sigma=25 (mirrors the AR(1) fixture
        // in `test_inflow_ar1_uses_external_lags`).
        models.push(make_inflow_model(1, 4, 160.0, 25.0, vec![0.5]));
        let par = PrecomputedPar::build(&models, &stages, &hydro_ids, None).unwrap();
        assert_eq!(
            par.max_order(),
            1,
            "stage 4's AR(1) sets the global max_order"
        );

        let raw_values = [130.0, 140.0, 150.0, 500.0, 200.0];
        let rows: Vec<ExternalScenarioRow> = raw_values
            .iter()
            .enumerate()
            .map(|(stage_id, &value_m3s)| ExternalScenarioRow {
                stage_id: i32::try_from(stage_id).unwrap(),
                scenario_id: 0,
                hydro_id,
                value_m3s,
            })
            .collect();

        let downstream_transition = |downstream_finalize: bool| StageLagTransition {
            accumulate_weight: 1.0,
            spillover_weight: 0.0,
            finalize_period: true,
            accumulate_downstream: true,
            downstream_accumulate_weight: 1.0 / 3.0,
            downstream_spillover_weight: 0.0,
            downstream_finalize,
            rebuild_from_downstream: false,
        };
        let transitions = vec![
            downstream_transition(false),
            downstream_transition(false),
            downstream_transition(true),
            StageLagTransition {
                accumulate_weight: 1.0,
                spillover_weight: 0.0,
                finalize_period: true,
                accumulate_downstream: false,
                downstream_accumulate_weight: 0.0,
                downstream_spillover_weight: 0.0,
                downstream_finalize: false,
                rebuild_from_downstream: true,
            },
            uniform_monthly_transitions(1)[0],
        ];

        let mut lib = ExternalScenarioLibrary::new(5, 1, 1, "inflow", vec![1; 5]);
        standardize_external_inflow(
            &mut lib,
            &rows,
            &hydro_ids,
            &stages,
            &par,
            DerivedSeed {
                lag_values: &[],
                l_state: 0,
                accum: &[],
                weight: &[],
            },
            &transitions,
            1, // downstream_par_order: one completed quarter needed to rebuild
        );

        let det_base = par.deterministic_base(4, 0);
        let psi = par.psi_slice(4, 0)[0];
        let sigma = par.sigma(4, 0);

        let ring_average = (130.0 + 140.0 + 150.0) / 3.0;
        let expected_eta_4 = (raw_values[4] - det_base - psi * ring_average) / sigma;
        let eta_4 = lib.eta_slice(4, 0)[0];
        assert!(
            (eta_4 - expected_eta_4).abs() < 1e-10,
            "eta[stage=4] = {eta_4}, expected {expected_eta_4} (ring average lag = {ring_average})"
        );

        // Negative control: the old primary-only advance would have shifted
        // stage 3's own raw value (500.0), not the ring average, into the lag.
        let naive_lag = raw_values[3];
        let naive_eta_4 = (raw_values[4] - det_base - psi * naive_lag) / sigma;
        assert!(
            (eta_4 - naive_eta_4).abs() > 1e-6,
            "eta[stage=4] must differ from the primary-only naive value; \
             got {eta_4} == naive {naive_eta_4}"
        );
    }

    // -----------------------------------------------------------------------
    // Load standardization tests
    // -----------------------------------------------------------------------

    /// 1 bus, 1 stage, 1 scenario. `value_mw`=240, mean=200, std=40 → eta=1.0.
    #[test]
    fn test_load_standardization() {
        let bus_id = EntityId(3);
        let bus_ids = vec![bus_id];

        let load_models = vec![LoadModel {
            bus_id,
            stage_id: 0,
            mean_mw: 200.0,
            std_mw: 40.0,
        }];

        let mut lib = ExternalScenarioLibrary::new(1, 1, 1, "load", vec![1]);
        let rows = vec![ExternalLoadRow {
            stage_id: 0,
            scenario_id: 0,
            bus_id,
            value_mw: 240.0,
        }];
        let stages = vec![make_stage(0, 0, 0)];
        standardize_external_load(&mut lib, &rows, &bus_ids, &load_models, &stages);

        let eta = lib.eta_slice(0, 0)[0];
        assert!((eta - 1.0).abs() < 1e-10, "eta = {eta}");
    }

    /// The standardization bound is defensive, not load-bearing: cobre-io's A1
    /// (exact `scenario_id` set) rejects an out-of-range `scenario_id` at load. An
    /// index equal to `n_scenarios` must be dropped here, never written into the
    /// next stage's realization 0 (the release-build corruption A1 now prevents).
    #[test]
    fn external_standardization_bound_is_defensive() {
        let bus_id = EntityId(1);
        let bus_ids = vec![bus_id];
        let load_models = vec![
            LoadModel {
                bus_id,
                stage_id: 0,
                mean_mw: 0.0,
                std_mw: 1.0,
            },
            LoadModel {
                bus_id,
                stage_id: 1,
                mean_mw: 0.0,
                std_mw: 1.0,
            },
        ];

        let mut lib = ExternalScenarioLibrary::new(2, 2, 1, "load", vec![2, 2]);
        let rows = vec![
            ExternalLoadRow {
                stage_id: 0,
                scenario_id: 0,
                bus_id,
                value_mw: 5.0,
            },
            ExternalLoadRow {
                stage_id: 0,
                scenario_id: 2,
                bus_id,
                value_mw: 99.0,
            },
        ];
        let stages = vec![make_stage(0, 0, 0), make_stage(1, 1, 0)];
        standardize_external_load(&mut lib, &rows, &bus_ids, &load_models, &stages);

        assert!(
            (lib.eta_slice(0, 0)[0] - 5.0).abs() < 1e-12,
            "the in-range cell is written"
        );
        assert!(
            lib.eta_slice(1, 0)[0].abs() < 1e-12,
            "the out-of-range write is dropped, not aliased onto stage 1 realization 0"
        );
    }

    // -----------------------------------------------------------------------
    // NCS standardization tests
    // -----------------------------------------------------------------------

    /// 1 NCS, 1 stage, 1 scenario. value=0.7, mean=0.5, std=0.2 → eta=1.0.
    #[test]
    fn test_ncs_standardization() {
        let ncs_id = EntityId(7);
        let ncs_ids = vec![ncs_id];

        let ncs_models = vec![NcsModel {
            ncs_id,
            stage_id: 0,
            mean: 0.5,
            std: 0.2,
        }];

        let mut lib = ExternalScenarioLibrary::new(1, 1, 1, "ncs", vec![1]);
        let rows = vec![ExternalNcsRow {
            stage_id: 0,
            scenario_id: 0,
            ncs_id,
            value: 0.7,
        }];
        let stages = vec![make_stage(0, 0, 0)];
        standardize_external_ncs(&mut lib, &rows, &ncs_ids, &ncs_models, &stages);

        let eta = lib.eta_slice(0, 0)[0];
        assert!((eta - 1.0).abs() < 1e-10, "eta = {eta}");
    }

    // -----------------------------------------------------------------------
    // std=0 guard test
    // -----------------------------------------------------------------------

    /// When `std_mw`=0.0, eta must be 0.0 (not NaN or infinity).
    #[test]
    fn test_std_zero_returns_zero() {
        let bus_id = EntityId(5);
        let bus_ids = vec![bus_id];

        let load_models = vec![LoadModel {
            bus_id,
            stage_id: 0,
            mean_mw: 0.5,
            std_mw: 0.0,
        }];

        let mut lib = ExternalScenarioLibrary::new(1, 1, 1, "load", vec![1]);
        let rows = vec![ExternalLoadRow {
            stage_id: 0,
            scenario_id: 0,
            bus_id,
            value_mw: 0.5,
        }];
        let stages = vec![make_stage(0, 0, 0)];
        standardize_external_load(&mut lib, &rows, &bus_ids, &load_models, &stages);

        let eta = lib.eta_slice(0, 0)[0];
        assert_eq!(eta, 0.0, "eta must be 0.0 when std=0.0, got {eta}");
    }

    /// NCS counterpart of the std=0 guard: when `std`=0.0, eta must be 0.0.
    #[test]
    fn test_ncs_std_zero_returns_zero() {
        let ncs_id = EntityId(9);
        let ncs_ids = vec![ncs_id];

        let ncs_models = vec![NcsModel {
            ncs_id,
            stage_id: 0,
            mean: 0.4,
            std: 0.0,
        }];

        let mut lib = ExternalScenarioLibrary::new(1, 1, 1, "ncs", vec![1]);
        let rows = vec![ExternalNcsRow {
            stage_id: 0,
            scenario_id: 0,
            ncs_id,
            value: 0.4,
        }];
        let stages = vec![make_stage(0, 0, 0)];
        standardize_external_ncs(&mut lib, &rows, &ncs_ids, &ncs_models, &stages);

        let eta = lib.eta_slice(0, 0)[0];
        assert_eq!(eta, 0.0, "eta must be 0.0 when std=0.0, got {eta}");
    }

    /// Multi-(stage, entity) load fixture. Pins the stage-major `mean_std`
    /// storage layout and the per-(bus, stage) model lookup: a transposed index
    /// or a swapped mean/std accessor in the shared helper would change at least
    /// one eta here, so the four asserts together guard the load wrapper's four
    /// field accessors and the storage-layout contract.
    #[test]
    fn test_load_standardization_multi_stage_entity() {
        let bus_a = EntityId(1);
        let bus_b = EntityId(2);
        let bus_ids = vec![bus_a, bus_b];

        let load_models = vec![
            LoadModel {
                bus_id: bus_a,
                stage_id: 0,
                mean_mw: 100.0,
                std_mw: 10.0,
            },
            LoadModel {
                bus_id: bus_a,
                stage_id: 1,
                mean_mw: 200.0,
                std_mw: 20.0,
            },
            LoadModel {
                bus_id: bus_b,
                stage_id: 0,
                mean_mw: 300.0,
                std_mw: 30.0,
            },
            LoadModel {
                bus_id: bus_b,
                stage_id: 1,
                mean_mw: 400.0,
                std_mw: 40.0,
            },
        ];

        let mut lib = ExternalScenarioLibrary::new(2, 1, 2, "load", vec![2, 2]);
        let rows = vec![
            ExternalLoadRow {
                stage_id: 0,
                scenario_id: 0,
                bus_id: bus_a,
                value_mw: 115.0, // (115-100)/10 = 1.5
            },
            ExternalLoadRow {
                stage_id: 0,
                scenario_id: 0,
                bus_id: bus_b,
                value_mw: 360.0, // (360-300)/30 = 2.0
            },
            ExternalLoadRow {
                stage_id: 1,
                scenario_id: 0,
                bus_id: bus_a,
                value_mw: 250.0, // (250-200)/20 = 2.5
            },
            ExternalLoadRow {
                stage_id: 1,
                scenario_id: 0,
                bus_id: bus_b,
                value_mw: 520.0, // (520-400)/40 = 3.0
            },
        ];
        let stages = vec![make_stage(0, 0, 0), make_stage(1, 1, 0)];
        standardize_external_load(&mut lib, &rows, &bus_ids, &load_models, &stages);

        assert!((lib.eta_slice(0, 0)[0] - 1.5).abs() < 1e-10);
        assert!((lib.eta_slice(0, 0)[1] - 2.0).abs() < 1e-10);
        assert!((lib.eta_slice(1, 0)[0] - 2.5).abs() < 1e-10);
        assert!((lib.eta_slice(1, 0)[1] - 3.0).abs() < 1e-10);
    }

    /// Multi-(stage, entity) NCS counterpart of
    /// `test_load_standardization_multi_stage_entity`: pins the same layout and
    /// lookup for the NCS wrapper's distinct field accessors (`value`, `mean`,
    /// `std`, `ncs_id`).
    #[test]
    fn test_ncs_standardization_multi_stage_entity() {
        let ncs_a = EntityId(4);
        let ncs_b = EntityId(8);
        let ncs_ids = vec![ncs_a, ncs_b];

        let ncs_models = vec![
            NcsModel {
                ncs_id: ncs_a,
                stage_id: 0,
                mean: 0.10,
                std: 0.10,
            },
            NcsModel {
                ncs_id: ncs_a,
                stage_id: 1,
                mean: 0.20,
                std: 0.20,
            },
            NcsModel {
                ncs_id: ncs_b,
                stage_id: 0,
                mean: 0.30,
                std: 0.30,
            },
            NcsModel {
                ncs_id: ncs_b,
                stage_id: 1,
                mean: 0.40,
                std: 0.40,
            },
        ];

        let mut lib = ExternalScenarioLibrary::new(2, 1, 2, "ncs", vec![2, 2]);
        let rows = vec![
            ExternalNcsRow {
                stage_id: 0,
                scenario_id: 0,
                ncs_id: ncs_a,
                value: 0.25, // (0.25-0.10)/0.10 = 1.5
            },
            ExternalNcsRow {
                stage_id: 0,
                scenario_id: 0,
                ncs_id: ncs_b,
                value: 0.90, // (0.90-0.30)/0.30 = 2.0
            },
            ExternalNcsRow {
                stage_id: 1,
                scenario_id: 0,
                ncs_id: ncs_a,
                value: 0.70, // (0.70-0.20)/0.20 = 2.5
            },
            ExternalNcsRow {
                stage_id: 1,
                scenario_id: 0,
                ncs_id: ncs_b,
                value: 1.60, // (1.60-0.40)/0.40 = 3.0
            },
        ];
        let stages = vec![make_stage(0, 0, 0), make_stage(1, 1, 0)];
        standardize_external_ncs(&mut lib, &rows, &ncs_ids, &ncs_models, &stages);

        assert!((lib.eta_slice(0, 0)[0] - 1.5).abs() < 1e-10);
        assert!((lib.eta_slice(0, 0)[1] - 2.0).abs() < 1e-10);
        assert!((lib.eta_slice(1, 0)[0] - 2.5).abs() < 1e-10);
        assert!((lib.eta_slice(1, 0)[1] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_new_allocates_correct_sizes() {
        let lib = ExternalScenarioLibrary::new(12, 50, 5, "inflow", vec![50usize; 12]);
        assert_eq!(lib.n_stages(), 12);
        assert_eq!(lib.n_scenarios(), 50);
        assert_eq!(lib.n_entities(), 5);
        assert_eq!(lib.eta_slice(0, 0).len(), 5);
        assert_eq!(lib.eta_slice(11, 49).len(), 5);
    }

    #[test]
    fn test_eta_roundtrip() {
        let mut lib = ExternalScenarioLibrary::new(3, 2, 4, "load", vec![2, 2, 2]);
        let written = [1.0_f64, 2.0, 3.0, 4.0];
        lib.eta_slice_mut(1, 0).copy_from_slice(&written);
        assert_eq!(lib.eta_slice(1, 0), &written);
    }

    #[test]
    fn test_entity_class_metadata() {
        let lib = ExternalScenarioLibrary::new(1, 1, 1, "ncs", vec![1]);
        assert_eq!(lib.entity_class(), "ncs");

        let lib2 = ExternalScenarioLibrary::new(1, 1, 1, "inflow", vec![1]);
        assert_eq!(lib2.entity_class(), "inflow");
    }

    #[test]
    fn test_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<ExternalScenarioLibrary>();
    }

    #[test]
    fn test_zero_initialized() {
        let lib = ExternalScenarioLibrary::new(2, 3, 4, "inflow", vec![3, 3]);
        for stage in 0..2 {
            for scenario in 0..3 {
                for &v in lib.eta_slice(stage, scenario) {
                    assert_eq!(v, 0.0_f64);
                }
            }
        }
    }

    #[test]
    fn test_eta_roundtrip_multiple_cells() {
        let mut lib = ExternalScenarioLibrary::new(3, 2, 4, "inflow", vec![2, 2, 2]);
        lib.eta_slice_mut(0, 0)
            .copy_from_slice(&[0.1, 0.2, 0.3, 0.4]);
        lib.eta_slice_mut(2, 1)
            .copy_from_slice(&[9.0, 8.0, 7.0, 6.0]);

        assert_eq!(lib.eta_slice(0, 0), &[0.1, 0.2, 0.3, 0.4]);
        assert_eq!(lib.eta_slice(2, 1), &[9.0, 8.0, 7.0, 6.0]);
        assert_eq!(lib.eta_slice(1, 0), &[0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_clone_is_independent() {
        let mut lib = ExternalScenarioLibrary::new(2, 2, 2, "ncs", vec![2, 2]);
        lib.eta_slice_mut(0, 0).copy_from_slice(&[1.0, 2.0]);

        let mut cloned = lib.clone();
        cloned.eta_slice_mut(0, 0).copy_from_slice(&[99.0, 99.0]);

        assert_eq!(lib.eta_slice(0, 0), &[1.0, 2.0]);
        assert_eq!(cloned.eta_slice(0, 0), &[99.0, 99.0]);
    }

    // -----------------------------------------------------------------------
    // validate_external_library tests
    // -----------------------------------------------------------------------

    use std::collections::HashSet;

    use super::validate_external_library;
    use crate::StochasticError;

    fn make_valid_library(
        n_stages: usize,
        n_scenarios: usize,
        n_entities: usize,
        class: &'static str,
    ) -> ExternalScenarioLibrary {
        let raw = vec![n_scenarios; n_stages];
        let mut lib = ExternalScenarioLibrary::new(n_stages, n_scenarios, n_entities, class, raw);
        // Fill with a known finite value so V3.7 passes.
        for stage in 0..n_stages {
            for scenario in 0..n_scenarios {
                for entity in 0..n_entities {
                    lib.eta_slice_mut(stage, scenario)[entity] = 0.5;
                }
            }
        }
        lib
    }

    fn entity_id_set(ids: impl IntoIterator<Item = i32>) -> HashSet<EntityId> {
        ids.into_iter().map(EntityId).collect()
    }

    fn uniform_rows_per_stage(
        n_stages: usize,
        n_scenarios: usize,
        n_entities: usize,
    ) -> Vec<usize> {
        vec![n_scenarios * n_entities; n_stages]
    }

    #[test]
    fn test_valid_library_passes() {
        let n_stages = 12;
        let n_scenarios = 50;
        let n_entities = 5;
        let lib = make_valid_library(n_stages, n_scenarios, n_entities, "inflow");
        let entity_ids: Vec<EntityId> = (1..=5).map(EntityId).collect();
        let row_entity_ids = entity_id_set(1..=5);
        let rows_per_stage = uniform_rows_per_stage(n_stages, n_scenarios, n_entities);

        let result = validate_external_library(
            &lib,
            &entity_ids,
            &row_entity_ids,
            &rows_per_stage,
            n_stages,
            50,
        );
        assert!(result.is_ok(), "expected Ok(()), got: {result:?}");
    }

    #[test]
    fn test_missing_entity_fails_v3_2() {
        let n_stages = 3;
        let n_scenarios = 10;
        let n_entities = 3;
        let lib = make_valid_library(n_stages, n_scenarios, n_entities, "inflow");
        // Entity IDs include 7, but row_entity_ids omits it.
        let entity_ids = vec![EntityId(5), EntityId(7), EntityId(9)];
        let row_entity_ids = entity_id_set([5, 9]); // 7 is missing
        let rows_per_stage = uniform_rows_per_stage(n_stages, n_scenarios, n_entities);

        let result = validate_external_library(
            &lib,
            &entity_ids,
            &row_entity_ids,
            &rows_per_stage,
            n_stages,
            10,
        );
        match result {
            Err(StochasticError::InsufficientData { context }) => {
                assert!(
                    context.contains("V3.2"),
                    "expected message to contain 'V3.2', got: {context}",
                );
                assert!(
                    context.contains('7'),
                    "expected message to contain entity ID '7', got: {context}",
                );
            }
            other => panic!("expected Err(InsufficientData), got: {other:?}"),
        }
    }

    #[test]
    fn test_nonuniform_divisible_counts_accepted_v3_4() {
        let n_stages = 3;
        let n_scenarios = 50;
        let n_entities = 2;
        let lib = make_valid_library(n_stages, n_scenarios, n_entities, "load");
        let entity_ids = vec![EntityId(1), EntityId(2)];
        let row_entity_ids = entity_id_set([1, 2]);
        // Stage 0: 50*2=100, Stage 1: 49*2=98 (non-uniform but divisible), Stage 2: 50*2=100.
        let rows_per_stage = vec![100usize, 98, 100];

        let result = validate_external_library(
            &lib,
            &entity_ids,
            &row_entity_ids,
            &rows_per_stage,
            n_stages,
            50,
        );
        assert!(
            result.is_ok(),
            "expected Ok(()) for non-uniform but divisible counts, got: {result:?}",
        );
    }

    #[test]
    fn test_nan_eta_fails_v3_7() {
        let n_stages = 5;
        let n_scenarios = 20;
        let n_entities = 4;
        let mut lib = make_valid_library(n_stages, n_scenarios, n_entities, "ncs");
        lib.eta_slice_mut(3, 10)[2] = f64::NAN;

        let entity_ids: Vec<EntityId> = (1..=4).map(EntityId).collect();
        let row_entity_ids = entity_id_set(1..=4);
        let rows_per_stage = uniform_rows_per_stage(n_stages, n_scenarios, n_entities);

        let result = validate_external_library(
            &lib,
            &entity_ids,
            &row_entity_ids,
            &rows_per_stage,
            n_stages,
            20,
        );
        match result {
            Err(StochasticError::InsufficientData { context }) => {
                assert!(
                    context.contains("V3.7"),
                    "expected message to contain 'V3.7', got: {context}",
                );
            }
            other => panic!("expected Err(InsufficientData), got: {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Round-trip standardization tests
    // -----------------------------------------------------------------------

    /// Build the fixture for the weekly+monthly AR(1) round-trip test.
    ///
    /// Returns `(stages, par, stage_lag_transitions, targets, past_lag, hydro_ids)` for
    /// a 4-weekly + 1-monthly layout matching the PMO\_APR\_2026 excerpt in the design doc.
    ///
    /// Stage layout (`season_id=3` for April, `season_id=4` for May):
    /// - W1 `[2026-03-28, 2026-04-04)` — 3 April days
    /// - W2 `[2026-04-04, 2026-04-11)` — 7 April days
    /// - W3 `[2026-04-11, 2026-04-18)` — 7 April days
    /// - W4 `[2026-04-18, 2026-04-25)` — 7 April days, finalizes April
    /// - M2 `[2026-05-02, 2026-06-01)` — 30 May days, finalizes May
    ///
    /// `StageLagTransition` weights: April = 720 h, May = 744 h.
    /// `psi=[0.3]`, `mean=500`, `std=50`, past lag-1 = 450.
    #[allow(clippy::type_complexity)]
    fn make_round_trip_fixture() -> (
        Vec<Stage>,
        PrecomputedPar,
        Vec<StageLagTransition>,
        [[f64; 5]; 2],
        f64,
        Vec<EntityId>,
    ) {
        const N_STAGES: usize = 5;
        let hydro_ids = vec![EntityId(1)];

        let stages = vec![
            make_stage(0, 0, 3), // W1 — April
            make_stage(1, 1, 3), // W2 — April
            make_stage(2, 2, 3), // W3 — April
            make_stage(3, 3, 3), // W4 — April (finalizes April period)
            make_stage(4, 4, 4), // M2 — May
        ];

        // AR(1): mean=500, std=50, psi=[0.3], residual_std_ratio=1.0 → sigma=50.
        let models: Vec<_> = (0..i32::try_from(N_STAGES).unwrap())
            .map(|stage_id| make_inflow_model(1, stage_id, 500.0, 50.0, vec![0.3]))
            .collect();
        let par = PrecomputedPar::build(&models, &stages, &hydro_ids, None).unwrap();

        // StageLagTransition weights (hand-computed from date boundaries).
        // April = 30 days = 720 h.  W1 covers only 3 April days; W2/W3/W4 cover 7 each.
        // May = 31 days = 744 h. M2 covers 30 May days.
        let weight_w1 = 3.0 * 24.0 / 720.0;
        let weight_weekly = 7.0 * 24.0 / 720.0;
        let weight_may = 30.0 * 24.0 / 744.0;
        let stage_lag_transitions = vec![
            StageLagTransition {
                accumulate_weight: weight_w1,
                spillover_weight: 0.0,
                finalize_period: false,
                accumulate_downstream: false,
                downstream_accumulate_weight: 0.0,
                downstream_spillover_weight: 0.0,
                downstream_finalize: false,
                rebuild_from_downstream: false,
            },
            StageLagTransition {
                accumulate_weight: weight_weekly,
                spillover_weight: 0.0,
                finalize_period: false,
                accumulate_downstream: false,
                downstream_accumulate_weight: 0.0,
                downstream_spillover_weight: 0.0,
                downstream_finalize: false,
                rebuild_from_downstream: false,
            },
            StageLagTransition {
                accumulate_weight: weight_weekly,
                spillover_weight: 0.0,
                finalize_period: false,
                accumulate_downstream: false,
                downstream_accumulate_weight: 0.0,
                downstream_spillover_weight: 0.0,
                downstream_finalize: false,
                rebuild_from_downstream: false,
            },
            StageLagTransition {
                accumulate_weight: weight_weekly,
                spillover_weight: 0.0,
                finalize_period: true, // last April stage → finalize the April period
                accumulate_downstream: false,
                downstream_accumulate_weight: 0.0,
                downstream_spillover_weight: 0.0,
                downstream_finalize: false,
                rebuild_from_downstream: false,
            },
            StageLagTransition {
                accumulate_weight: weight_may,
                spillover_weight: 0.0,
                finalize_period: true, // only May stage → finalizes itself
                accumulate_downstream: false,
                downstream_accumulate_weight: 0.0,
                downstream_spillover_weight: 0.0,
                downstream_finalize: false,
                rebuild_from_downstream: false,
            },
        ];

        let targets = [
            [480.0_f64, 520.0, 490.0, 510.0, 530.0], // scenario 0
            [550.0_f64, 470.0, 500.0, 540.0, 460.0], // scenario 1
        ];

        // Past inflow lag-1 = 450.0 (December monthly average before the study).
        let past_lag = 450.0_f64;

        (
            stages,
            par,
            stage_lag_transitions,
            targets,
            past_lag,
            hydro_ids,
        )
    }

    /// Round-trip consistency: `standardize_external_inflow` followed by
    /// [`evaluate_par`] must reconstruct the original external targets for a
    /// mixed 4-weekly + 1-monthly layout with AR(1) lags.
    ///
    /// The reconstruction loop replicates the lag state standardization used:
    /// frozen within each lag period, advanced by weighted average at period
    /// boundaries. See `make_round_trip_fixture` for the stage layout.
    #[test]
    fn test_round_trip_weekly_monthly_ar1() {
        let (stages, par, stage_lag_transitions, targets, past_lag, hydro_ids) =
            make_round_trip_fixture();
        let hydro_id = hydro_ids[0];
        let n_stages = stages.len();
        let n_scenarios = targets.len();

        let mut rows = Vec::with_capacity(n_stages * n_scenarios);
        for (scenario, scenario_targets) in targets.iter().enumerate() {
            for (stage, &value) in scenario_targets.iter().enumerate() {
                rows.push(ExternalScenarioRow {
                    stage_id: i32::try_from(stage).unwrap(),
                    scenario_id: i32::try_from(scenario).unwrap(),
                    hydro_id,
                    value_m3s: value,
                });
            }
        }

        let derived_lag_values = [past_lag];

        let raw = vec![n_scenarios; n_stages];
        let mut lib = ExternalScenarioLibrary::new(n_stages, n_scenarios, 1, "inflow", raw);
        standardize_external_inflow(
            &mut lib,
            &rows,
            &hydro_ids,
            &stages,
            &par,
            DerivedSeed {
                lag_values: &derived_lag_values,
                l_state: 1,
                accum: &[],
                weight: &[],
            },
            &stage_lag_transitions,
            0,
        );

        for (scenario, scenario_targets) in targets.iter().enumerate() {
            let mut lag_buf = vec![past_lag]; // lag-1 initialized from the derived seed
            let mut accum = 0.0_f64;
            let mut weight_accum = 0.0_f64;

            for (t, (&target, slt)) in scenario_targets
                .iter()
                .zip(&stage_lag_transitions)
                .enumerate()
            {
                let eta = lib.eta_slice(t, scenario)[0];
                let det_base = par.deterministic_base(t, 0);
                let psi = par.psi_slice(t, 0);
                let sigma = par.sigma(t, 0);

                let reconstructed = evaluate_par(det_base, psi, &lag_buf, sigma, eta);
                assert!(
                    (reconstructed - target).abs() < 1e-10,
                    "stage={t}, scenario={scenario}: reconstructed={reconstructed:.15}, \
                     target={target:.15}, diff={:.2e}",
                    (reconstructed - target).abs()
                );

                accum += target * slt.accumulate_weight;
                weight_accum += slt.accumulate_weight;

                if slt.finalize_period && weight_accum > 0.0 {
                    lag_buf[0] = accum / weight_accum;
                    accum = 0.0;
                    weight_accum = 0.0;
                }
                // Non-finalizing stages: lag_buf stays frozen (unchanged).
            }
        }
    }

    #[test]
    fn test_scenario_count_warning_returns_ok() {
        let n_stages = 2;
        let n_scenarios = 10;
        let n_entities = 2;
        let lib = make_valid_library(n_stages, n_scenarios, n_entities, "inflow");
        let entity_ids = vec![EntityId(1), EntityId(2)];
        let row_entity_ids = entity_id_set([1, 2]);
        let rows_per_stage = uniform_rows_per_stage(n_stages, n_scenarios, n_entities);

        // 10 scenarios < 50 forward passes — must warn but not error.
        let result = validate_external_library(
            &lib,
            &entity_ids,
            &row_entity_ids,
            &rows_per_stage,
            n_stages,
            50,
        );
        assert!(
            result.is_ok(),
            "V3.8 warning path must return Ok(()), got: {result:?}",
        );
    }

    // -----------------------------------------------------------------------
    // relaxed V3.4, raw_scenarios_per_stage, pad_library_to_uniform
    // -----------------------------------------------------------------------

    use super::pad_library_to_uniform;

    #[test]
    fn test_v34_accepts_nonuniform_scenario_counts() {
        // 5 stages: 4 with 1 scenario (2 rows each) and 1 with 50 scenarios (100 rows).
        let n_entities = 2;
        let n_stages = 5;
        // Library must be big enough to hold V3.7 pass (50 scenarios max).
        let lib = make_valid_library(n_stages, 50, n_entities, "inflow");
        let entity_ids = vec![EntityId(1), EntityId(2)];
        let row_entity_ids = entity_id_set([1, 2]);
        let rows_per_stage = vec![2usize, 2, 2, 2, 100];

        let result = validate_external_library(
            &lib,
            &entity_ids,
            &row_entity_ids,
            &rows_per_stage,
            n_stages,
            50,
        );
        assert!(
            result.is_ok(),
            "V3.4 must accept non-uniform but divisible counts, got: {result:?}",
        );
    }

    #[test]
    fn test_v34_still_rejects_indivisible_rows() {
        let n_entities = 2;
        let n_stages = 2;
        let lib = make_valid_library(n_stages, 2, n_entities, "inflow");
        let entity_ids = vec![EntityId(1), EntityId(2)];
        let row_entity_ids = entity_id_set([1, 2]);
        // Stage 0: 3 rows (not divisible by 2), Stage 1: 2 rows (ok).
        let rows_per_stage = vec![3usize, 2];

        let result = validate_external_library(
            &lib,
            &entity_ids,
            &row_entity_ids,
            &rows_per_stage,
            n_stages,
            1,
        );
        match result {
            Err(StochasticError::InsufficientData { context }) => {
                assert!(
                    context.contains("V3.4"),
                    "expected error to contain 'V3.4', got: {context}",
                );
            }
            other => panic!("expected Err(InsufficientData) with V3.4, got: {other:?}"),
        }
    }

    #[test]
    fn test_raw_scenarios_per_stage_uniform() {
        let n_stages = 4;
        let n_scenarios = 10;
        let raw = vec![n_scenarios; n_stages];
        let lib = ExternalScenarioLibrary::new(n_stages, n_scenarios, 2, "inflow", raw);
        assert_eq!(lib.raw_scenarios_per_stage(), &[10, 10, 10, 10]);
    }

    #[test]
    fn test_raw_scenarios_per_stage_nonuniform() {
        let n_stages = 3;
        let n_scenarios = 50; // max (padded-to) count
        let raw = vec![1usize, 1, 50];
        let lib = ExternalScenarioLibrary::new(n_stages, n_scenarios, 1, "inflow", raw);
        assert_eq!(lib.raw_scenarios_per_stage(), &[1, 1, 50]);
        assert_eq!(lib.n_scenarios(), 50);
    }

    /// `pad_library_to_uniform` replicates stage 0's single eta value into
    /// all `n_scenarios` slots so that `eta_slice(0, k)` is identical for all k.
    #[test]
    fn test_pad_library_replicates_eta() {
        // 2 stages, 1 entity, raw counts [1, 3], padded to n_scenarios=3.
        let raw = vec![1usize, 3];
        let mut lib = ExternalScenarioLibrary::new(2, 3, 1, "inflow", raw);

        // Write known values only to the raw slots.
        // Stage 0 raw slot: scenario 0
        lib.eta_slice_mut(0, 0).copy_from_slice(&[7.0]);
        // Stage 1 raw slots: scenarios 0..3
        lib.eta_slice_mut(1, 0).copy_from_slice(&[1.0]);
        lib.eta_slice_mut(1, 1).copy_from_slice(&[2.0]);
        lib.eta_slice_mut(1, 2).copy_from_slice(&[3.0]);

        pad_library_to_uniform(&mut lib);

        // Stage 0: all three scenario slots must equal the single raw value.
        assert_eq!(lib.eta_slice(0, 0), &[7.0], "stage 0 scenario 0");
        assert_eq!(lib.eta_slice(0, 1), &[7.0], "stage 0 scenario 1 (padded)");
        assert_eq!(lib.eta_slice(0, 2), &[7.0], "stage 0 scenario 2 (padded)");

        // Stage 1: unchanged (already had 3 raw scenarios == n_scenarios).
        assert_eq!(lib.eta_slice(1, 0), &[1.0], "stage 1 scenario 0");
        assert_eq!(lib.eta_slice(1, 1), &[2.0], "stage 1 scenario 1");
        assert_eq!(lib.eta_slice(1, 2), &[3.0], "stage 1 scenario 2");
    }

    /// `pad_library_to_uniform` is a no-op when all stages already have the
    /// maximum scenario count — eta values must not change.
    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn test_pad_library_noop_when_uniform() {
        let n_stages = 3;
        let n_scenarios = 5;
        let raw = vec![n_scenarios; n_stages];
        let mut lib = ExternalScenarioLibrary::new(n_stages, n_scenarios, 2, "load", raw);

        // Fill with recognizable values.
        for s in 0..n_stages {
            for k in 0..n_scenarios {
                lib.eta_slice_mut(s, k)
                    .copy_from_slice(&[s as f64, k as f64]);
            }
        }

        pad_library_to_uniform(&mut lib);

        // Values must be identical to what was written.
        for s in 0..n_stages {
            for k in 0..n_scenarios {
                assert_eq!(
                    lib.eta_slice(s, k),
                    &[s as f64, k as f64],
                    "stage {s} scenario {k} must be unchanged",
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // Derived-seed replay: the replay seeds from the same derived lag state as
    // the forward pass, so z == v.
    // -----------------------------------------------------------------------

    fn dated_stage(
        index: usize,
        id: i32,
        start: NaiveDate,
        end: NaiveDate,
        season_id: usize,
    ) -> Stage {
        cobre_core::test_support::make_stage(StageSpec {
            id,
            index: Some(index),
            start_date: start,
            end_date: end,
            season_id: Some(season_id),
            blocks: single_block("SINGLE", 744.0),
            ..Default::default()
        })
    }

    fn make_hydro(id: i32) -> Hydro {
        cobre_core::test_support::make_hydro(HydroSpec {
            id,
            name: format!("H{id}"),
            max_storage_hm3: 100.0,
            max_turbined_m3s: 100.0,
            max_generation_mw: 100.0,
            operational_start_date: date(2020, 1, 1),
            mirror_unit_group: MirrorUnitGroup::None,
            ..Default::default()
        })
    }

    /// With no conditioning, the derived lag seed carries exactly the same
    /// values a pre-windowing literal seed would have held, so threading it
    /// through the per-(stage, hydro) `solve_par_noise` calls reproduces the
    /// eta a hard-coded seed of those values would have produced. Covers 2
    /// hydros and a 2-lag stride so a transposed `(hydro, lag)` index in the
    /// fill loop would be caught.
    #[test]
    fn standardize_external_eta_matches_positional_seed() {
        let h1 = EntityId(1);
        let h2 = EntityId(2);
        let hydro_ids = vec![h1, h2];
        let hydros = vec![make_hydro(1), make_hydro(2)];
        let season_map = monthly_season_map(MonthlyLabels::OneBased);

        let stages = vec![dated_stage(0, 0, date(2024, 1, 1), date(2024, 2, 1), 0)];
        let first_stage = stages[0].clone();

        let models = vec![
            make_inflow_model(1, 0, 300.0, 40.0, vec![0.3, 0.1]),
            make_inflow_model(2, 0, 500.0, 60.0, vec![0.2, 0.05]),
        ];
        let par = PrecomputedPar::build(&models, &stages, &hydro_ids, None).unwrap();
        let l_state = par.max_order();
        assert_eq!(l_state, 2);

        let record = vec![
            InflowHistoryRow {
                hydro_id: h1,
                start_date: date(2023, 12, 1),
                end_date: date(2024, 1, 1),
                value_m3s: 110.0,
            },
            InflowHistoryRow {
                hydro_id: h1,
                start_date: date(2023, 11, 1),
                end_date: date(2023, 12, 1),
                value_m3s: 120.0,
            },
            InflowHistoryRow {
                hydro_id: h2,
                start_date: date(2023, 12, 1),
                end_date: date(2024, 1, 1),
                value_m3s: 210.0,
            },
            InflowHistoryRow {
                hydro_id: h2,
                start_date: date(2023, 11, 1),
                end_date: date(2023, 12, 1),
                value_m3s: 220.0,
            },
        ];
        let derived =
            derive_inflow_seeds(&record, &[], &hydros, &first_stage, &season_map, l_state);
        assert_eq!(derived.lag_values, vec![110.0, 120.0, 210.0, 220.0]);

        let rows = vec![
            ExternalScenarioRow {
                stage_id: 0,
                scenario_id: 0,
                hydro_id: h1,
                value_m3s: 250.0,
            },
            ExternalScenarioRow {
                stage_id: 0,
                scenario_id: 0,
                hydro_id: h2,
                value_m3s: 300.0,
            },
        ];
        let transitions = uniform_monthly_transitions(1);

        let mut lib = ExternalScenarioLibrary::new(1, 1, 2, "inflow", vec![1]);
        standardize_external_inflow(
            &mut lib,
            &rows,
            &hydro_ids,
            &stages,
            &par,
            DerivedSeed {
                lag_values: &derived.lag_values,
                l_state,
                accum: &[],
                weight: &[],
            },
            &transitions,
            0,
        );

        let expected_h1 = solve_par_noise(
            par.deterministic_base(0, 0),
            par.psi_slice(0, 0),
            &[110.0, 120.0],
            par.sigma(0, 0),
            250.0,
        );
        let expected_h2 = solve_par_noise(
            par.deterministic_base(0, 1),
            par.psi_slice(0, 1),
            &[210.0, 220.0],
            par.sigma(0, 1),
            300.0,
        );

        let eta = lib.eta_slice(0, 0);
        assert_eq!(
            eta[0], expected_h1,
            "hydro 1 eta must match the positional-seed formula"
        );
        assert_eq!(
            eta[1], expected_h2,
            "hydro 2 eta must match the positional-seed formula"
        );
    }

    /// External replay seeds from the same derived lag state as the forward
    /// pass, so the inverted noise reconstructs the raw value exactly even
    /// with PAR(p)-A annual coupling, a mid-year study start (April, not
    /// January), and a conditioning window shadowing the most recent pre-study
    /// history.
    #[test]
    #[allow(clippy::too_many_lines)]
    fn external_eta_round_trip_exact_under_conditioning() {
        let hydro_id = EntityId(1);
        let hydro_ids = vec![hydro_id];
        let hydros = vec![make_hydro(1)];
        let season_map = monthly_season_map(MonthlyLabels::OneBased);

        let stages = vec![
            dated_stage(0, 0, date(2026, 4, 1), date(2026, 5, 1), 3),
            dated_stage(1, 1, date(2026, 5, 1), date(2026, 6, 1), 4),
            dated_stage(2, 2, date(2026, 6, 1), date(2026, 7, 1), 5),
            dated_stage(3, 3, date(2026, 7, 1), date(2026, 8, 1), 6),
        ];
        let first_stage = stages[0].clone();

        let annual = AnnualComponent {
            coefficient: -0.18,
            mean_m3s: 95.0,
            std_m3s: 22.0,
        };
        let mut all_models: Vec<InflowModel> = (-12_i32..0)
            .map(|sid| InflowModel {
                hydro_id,
                stage_id: sid,
                mean_m3s: 90.0,
                std_m3s: 20.0,
                ar_coefficients: vec![0.4],
                residual_std_ratio: 0.85,
                annual: Some(annual.clone()),
            })
            .collect();
        all_models.extend((0_i32..4).map(|sid| InflowModel {
            hydro_id,
            stage_id: sid,
            mean_m3s: 100.0 + f64::from(sid) * 5.0,
            std_m3s: 25.0 + f64::from(sid),
            ar_coefficients: vec![0.4],
            residual_std_ratio: 0.85,
            annual: Some(annual.clone()),
        }));
        let par = PrecomputedPar::build(&all_models, &stages, &hydro_ids, None).unwrap();
        let l_state = par.max_order();
        assert_eq!(
            l_state, 12,
            "PAR-A annual coupling must widen max_order to 12"
        );

        // 12 months of inflow_history immediately preceding the study (April
        // 2025 through March 2026).
        let record_months = [
            (2025, 4),
            (2025, 5),
            (2025, 6),
            (2025, 7),
            (2025, 8),
            (2025, 9),
            (2025, 10),
            (2025, 11),
            (2025, 12),
            (2026, 1),
            (2026, 2),
            (2026, 3),
        ];
        let record: Vec<InflowHistoryRow> = record_months
            .iter()
            .enumerate()
            .map(|(i, &(year, month))| {
                let start = date(year, month, 1);
                let end = if month == 12 {
                    date(year + 1, 1, 1)
                } else {
                    date(year, month + 1, 1)
                };
                InflowHistoryRow {
                    hydro_id,
                    start_date: start,
                    end_date: end,
                    value_m3s: 100.0 + f64::from(i32::try_from(i).unwrap()) * 3.0,
                }
            })
            .collect();

        // The conditioning window shadows the two most recent pre-study
        // months (Feb and March 2026) with distinct, more-recent observations
        // — the scenario a plain historical read would miss.
        let conditioning = vec![
            RecentObservation {
                hydro_id,
                start_date: date(2026, 2, 1),
                end_date: date(2026, 3, 1),
                value_m3s: 555.0,
            },
            RecentObservation {
                hydro_id,
                start_date: date(2026, 3, 1),
                end_date: date(2026, 4, 1),
                value_m3s: 777.0,
            },
        ];

        let derived = derive_inflow_seeds(
            &record,
            &conditioning,
            &hydros,
            &first_stage,
            &season_map,
            l_state,
        );
        assert_eq!(derived.lag_values.len(), l_state);
        // The conditioning window must actually have shadowed the two most
        // recent record months, or this scenario would not exercise
        // conditioning at all.
        assert_eq!(derived.lag_values[0], 777.0);
        assert_eq!(derived.lag_values[1], 555.0);

        // Forward-generate raw external targets from the derived seed with an
        // arbitrary noise sequence, advancing the lag chain the same way the
        // uniform-monthly transitions below do.
        let stage_lag_transitions = uniform_monthly_transitions(stages.len());
        let eta_sequence = [0.35_f64, -0.6, 0.9, -0.2];
        let mut lag_state = derived.lag_values.clone();
        let mut targets = Vec::with_capacity(stages.len());
        for (t, &eta) in eta_sequence.iter().enumerate() {
            let det_base = par.deterministic_base(t, 0);
            let psi = par.psi_slice(t, 0);
            let sigma = par.sigma(t, 0);
            let value = evaluate_par(det_base, psi, &lag_state, sigma, eta);
            targets.push(value);
            for l in (1..lag_state.len()).rev() {
                lag_state[l] = lag_state[l - 1];
            }
            lag_state[0] = value;
        }

        let rows: Vec<ExternalScenarioRow> = targets
            .iter()
            .enumerate()
            .map(|(t, &value_m3s)| ExternalScenarioRow {
                stage_id: i32::try_from(t).unwrap(),
                scenario_id: 0,
                hydro_id,
                value_m3s,
            })
            .collect();

        let mut lib =
            ExternalScenarioLibrary::new(stages.len(), 1, 1, "inflow", vec![1; stages.len()]);
        standardize_external_inflow(
            &mut lib,
            &rows,
            &hydro_ids,
            &stages,
            &par,
            DerivedSeed {
                lag_values: &derived.lag_values,
                l_state,
                accum: &derived.accum,
                weight: &derived.weight,
            },
            &stage_lag_transitions,
            0,
        );

        // Replay from the same derived seed: inverting the stored eta must
        // reconstruct each forward-generated target exactly (z == v).
        let mut replay_lag_state = derived.lag_values.clone();
        for (t, &target) in targets.iter().enumerate() {
            let eta = lib.eta_slice(t, 0)[0];
            let det_base = par.deterministic_base(t, 0);
            let psi = par.psi_slice(t, 0);
            let sigma = par.sigma(t, 0);
            let reconstructed = evaluate_par(det_base, psi, &replay_lag_state, sigma, eta);
            assert!(
                (reconstructed - target).abs() < 1e-9,
                "stage {t}: replay reconstructed {reconstructed:.12} != forward target {target:.12}",
            );
            for l in (1..replay_lag_state.len()).rev() {
                replay_lag_state[l] = replay_lag_state[l - 1];
            }
            replay_lag_state[0] = target;
        }
    }

    /// A study whose stage 0 starts mid-coarse-period, with pre-study record
    /// coverage seeding a genuine partial `accum`/`weight`, exercises a
    /// divergence no monthly-boundary fixture above can reach (every one of
    /// those starts exactly on the 1st, making the in-progress seed inert).
    /// Forward generation and `standardize_external_inflow`'s replay reset
    /// both advance the lag chain from the same `derived.accum`/`derived.weight`
    /// seed, so `z == v`.
    #[test]
    #[allow(clippy::too_many_lines)]
    fn external_eta_round_trip_exact_mid_coarse_period() {
        let hydro_id = EntityId(1);
        let hydro_ids = vec![hydro_id];
        let hydros = vec![make_hydro(1)];
        let season_map = monthly_season_map(MonthlyLabels::OneBased);

        // Stage 0 starts April 11: the in-progress occurrence [April 1,
        // April 11) is non-empty, and the remaining 20 of April's 30 days
        // still finalize within stage 0.
        let stages = vec![
            dated_stage(0, 0, date(2026, 4, 11), date(2026, 5, 1), 3),
            dated_stage(1, 1, date(2026, 5, 1), date(2026, 6, 1), 4),
        ];
        let first_stage = stages[0].clone();

        let models = vec![
            make_inflow_model(1, 0, 160.0, 25.0, vec![0.5]),
            make_inflow_model(1, 1, 160.0, 25.0, vec![0.5]),
        ];
        let par = PrecomputedPar::build(&models, &stages, &hydro_ids, None).unwrap();
        let l_state = par.max_order();
        assert_eq!(l_state, 1, "AR(1) with no annual coupling stays order 1");

        let record = vec![
            InflowHistoryRow {
                hydro_id,
                start_date: date(2026, 3, 1),
                end_date: date(2026, 4, 1),
                value_m3s: 300.0,
            },
            InflowHistoryRow {
                hydro_id,
                start_date: date(2026, 4, 1),
                end_date: date(2026, 4, 11),
                value_m3s: 200.0,
            },
        ];

        let derived =
            derive_inflow_seeds(&record, &[], &hydros, &first_stage, &season_map, l_state);
        assert_eq!(derived.lag_values.len(), l_state);
        assert_eq!(derived.lag_values[0], 300.0);
        assert!(
            derived.weight[0] > 0.0 && derived.weight[0] < 1.0,
            "the accumulator seed must be a genuine partial-coverage fraction \
             in (0, 1), or this test is a tautology (every monthly-boundary \
             fixture misses the bug this way); got weight={}",
            derived.weight[0]
        );

        let stage_lag_transitions = precompute_stage_lag_transitions(&stages, &season_map, 0);
        assert!(
            stage_lag_transitions[0].finalize_period && stage_lag_transitions[1].finalize_period,
            "both stages must finalize their own period for this fixture to \
             exercise a mid-period accumulate/finalize transition"
        );

        // Forward-generate targets by advancing the SEEDED lag chain exactly
        // as the training/simulation forward pass does: the accumulator
        // starts from `derived.accum`/`derived.weight`, not zero.
        let eta_sequence = [0.35_f64, -0.6];
        let mut lag_state = derived.lag_values.clone();
        let mut accum = derived.accum.clone();
        let mut weight = derived.weight.clone();
        let mut incoming_scratch = vec![0.0_f64; l_state];
        let mut downstream_accumulator: Vec<f64> = Vec::new();
        let mut downstream_weight_accum = 0.0_f64;
        let mut downstream_completed_lags: Vec<f64> = Vec::new();
        let mut downstream_n_completed = 0_usize;
        let mut targets = Vec::with_capacity(stages.len());
        for (t, &eta) in eta_sequence.iter().enumerate() {
            let det_base = par.deterministic_base(t, 0);
            let psi = par.psi_slice(t, 0);
            let sigma = par.sigma(t, 0);
            let value = evaluate_par(det_base, psi, &lag_state, sigma, eta);
            targets.push(value);

            incoming_scratch.copy_from_slice(&lag_state);
            let mut primary = PrimaryLagAccum {
                accumulator: &mut accum,
                weight_accum: &mut weight,
            };
            let mut downstream = DownstreamLagAccum {
                accumulator: &mut downstream_accumulator,
                weight_accum: &mut downstream_weight_accum,
                completed_lags: &mut downstream_completed_lags,
                n_completed: &mut downstream_n_completed,
                par_order: 0,
            };
            advance_lag_chain(
                EntityMajor {
                    entity_count: 1,
                    max_order: l_state,
                },
                &mut lag_state,
                &incoming_scratch,
                &[value],
                &stage_lag_transitions[t],
                &mut primary,
                &mut downstream,
            );
        }

        let rows: Vec<ExternalScenarioRow> = targets
            .iter()
            .enumerate()
            .map(|(t, &value_m3s)| ExternalScenarioRow {
                stage_id: i32::try_from(t).unwrap(),
                scenario_id: 0,
                hydro_id,
                value_m3s,
            })
            .collect();

        let mut lib =
            ExternalScenarioLibrary::new(stages.len(), 1, 1, "inflow", vec![1; stages.len()]);
        standardize_external_inflow(
            &mut lib,
            &rows,
            &hydro_ids,
            &stages,
            &par,
            DerivedSeed {
                lag_values: &derived.lag_values,
                l_state,
                accum: &derived.accum,
                weight: &derived.weight,
            },
            &stage_lag_transitions,
            0,
        );

        // Replay from the SAME seeded lag chain the production forward pass
        // would carry, inverting the stored eta; z must equal v.
        let mut replay_lag_state = derived.lag_values.clone();
        let mut replay_accum = derived.accum.clone();
        let mut replay_weight = derived.weight.clone();
        let mut replay_downstream_accumulator: Vec<f64> = Vec::new();
        let mut replay_downstream_weight_accum = 0.0_f64;
        let mut replay_downstream_completed_lags: Vec<f64> = Vec::new();
        let mut replay_downstream_n_completed = 0_usize;
        for (t, &target) in targets.iter().enumerate() {
            let eta = lib.eta_slice(t, 0)[0];
            let det_base = par.deterministic_base(t, 0);
            let psi = par.psi_slice(t, 0);
            let sigma = par.sigma(t, 0);
            let reconstructed = evaluate_par(det_base, psi, &replay_lag_state, sigma, eta);
            assert!(
                (reconstructed - target).abs() < 1e-9,
                "stage {t}: mid-period replay reconstructed {reconstructed:.12} \
                 (z) != forward target {target:.12} (v)",
            );

            incoming_scratch.copy_from_slice(&replay_lag_state);
            let mut primary = PrimaryLagAccum {
                accumulator: &mut replay_accum,
                weight_accum: &mut replay_weight,
            };
            let mut downstream = DownstreamLagAccum {
                accumulator: &mut replay_downstream_accumulator,
                weight_accum: &mut replay_downstream_weight_accum,
                completed_lags: &mut replay_downstream_completed_lags,
                n_completed: &mut replay_downstream_n_completed,
                par_order: 0,
            };
            advance_lag_chain(
                EntityMajor {
                    entity_count: 1,
                    max_order: l_state,
                },
                &mut replay_lag_state,
                &incoming_scratch,
                &[target],
                &stage_lag_transitions[t],
                &mut primary,
                &mut downstream,
            );
        }
    }

    // -----------------------------------------------------------------------
    // Sample-moment reduction tests
    // -----------------------------------------------------------------------

    /// Bit-identical constant column: `σ = 0.0` exactly, `μ` equal to the
    /// constant exactly (guards the catastrophic-cancellation trap a naive
    /// two-pass sum would fall into).
    #[test]
    fn test_derive_external_sample_moments_constant_column_is_exact() {
        let bus_id = EntityId(1);
        let bus_ids = vec![bus_id];
        let rows = vec![
            ExternalLoadRow {
                stage_id: 0,
                scenario_id: 0,
                bus_id,
                value_mw: 42.0,
            },
            ExternalLoadRow {
                stage_id: 0,
                scenario_id: 1,
                bus_id,
                value_mw: 42.0,
            },
            ExternalLoadRow {
                stage_id: 0,
                scenario_id: 2,
                bus_id,
                value_mw: 42.0,
            },
        ];
        let moments =
            derive_external_sample_moments(&rows, &bus_ids, 1, |row: &ExternalLoadRow| {
                (row.bus_id, row.stage_id, row.scenario_id, row.value_mw)
            });
        let (mean, std) = moments[0];
        assert_eq!(
            std, 0.0,
            "constant column must yield sigma = 0.0, got {std}"
        );
        assert_eq!(
            mean, 42.0,
            "constant column mean must equal the constant, got {mean}"
        );
    }

    /// Two distinct values: `σ > 0` and `(μ, σ)` match the hand-computed
    /// population moments (mean=15.0, population std=5.0).
    #[test]
    fn test_derive_external_sample_moments_two_distinct_values() {
        let bus_id = EntityId(1);
        let bus_ids = vec![bus_id];
        let rows = vec![
            ExternalLoadRow {
                stage_id: 0,
                scenario_id: 0,
                bus_id,
                value_mw: 10.0,
            },
            ExternalLoadRow {
                stage_id: 0,
                scenario_id: 1,
                bus_id,
                value_mw: 20.0,
            },
        ];
        let moments =
            derive_external_sample_moments(&rows, &bus_ids, 1, |row: &ExternalLoadRow| {
                (row.bus_id, row.stage_id, row.scenario_id, row.value_mw)
            });
        let (mean, std) = moments[0];
        assert!(std > 0.0, "distinct values must yield sigma > 0, got {std}");
        assert!((mean - 15.0).abs() < 1e-12, "mean = {mean}, expected 15.0");
        assert!((std - 5.0).abs() < 1e-12, "std = {std}, expected 5.0");
    }

    /// An `(entity, stage)` cell with no rows defaults to `(0.0, 0.0)`.
    #[test]
    fn test_derive_external_sample_moments_empty_cell_defaults_to_zero() {
        let bus_a = EntityId(1);
        let bus_b = EntityId(2);
        let bus_ids = vec![bus_a, bus_b];
        let rows = vec![ExternalLoadRow {
            stage_id: 0,
            scenario_id: 0,
            bus_id: bus_a,
            value_mw: 7.0,
        }];
        let moments =
            derive_external_sample_moments(&rows, &bus_ids, 1, |row: &ExternalLoadRow| {
                (row.bus_id, row.stage_id, row.scenario_id, row.value_mw)
            });
        assert_eq!(moments[0], (7.0, 0.0), "entity with rows keeps its moments");
        assert_eq!(
            moments[1],
            (0.0, 0.0),
            "entity with no rows must default to (0.0, 0.0)"
        );
    }

    mod sample_moment_reduction_proptests {
        use proptest::prelude::*;

        use super::{EntityId, ExternalLoadRow, derive_external_sample_moments};

        proptest! {
            /// A permutation of the same external rows must reduce to a
            /// bit-for-bit identical (mu, sigma) table (declaration-order
            /// invariance).
            #[test]
            fn derive_external_sample_moments_is_declaration_order_invariant(
                cell_values in prop::collection::vec(
                    prop::collection::vec(-1000.0_f64..1000.0_f64, 0..=5),
                    4,
                ),
                rotate_by in 0_usize..37,
            ) {
                let bus_ids = vec![EntityId(1), EntityId(2)];
                let n_stages = 2;

                let mut rows = Vec::new();
                for (cell_idx, values) in cell_values.iter().enumerate() {
                    let entity_idx = cell_idx % 2;
                    let stage_idx = cell_idx / 2;
                    for (scenario_id, &value) in values.iter().enumerate() {
                        rows.push(ExternalLoadRow {
                            stage_id: i32::try_from(stage_idx).unwrap(),
                            scenario_id: i32::try_from(scenario_id).unwrap(),
                            bus_id: bus_ids[entity_idx],
                            value_mw: value,
                        });
                    }
                }

                let row_fields = |row: &ExternalLoadRow| {
                    (row.bus_id, row.stage_id, row.scenario_id, row.value_mw)
                };

                let baseline = derive_external_sample_moments(&rows, &bus_ids, n_stages, row_fields);

                let mut permuted = rows.clone();
                permuted.reverse();
                if !permuted.is_empty() {
                    let len = permuted.len();
                    permuted.rotate_left(rotate_by % len);
                }
                let reordered =
                    derive_external_sample_moments(&permuted, &bus_ids, n_stages, row_fields);

                prop_assert_eq!(baseline, reordered);
            }
        }
    }
}
