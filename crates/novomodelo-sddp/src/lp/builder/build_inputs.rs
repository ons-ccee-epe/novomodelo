//! [`LpBuildInputs`]: the builder's resolved study inputs, each resolved once
//! by setup's `resolve_lp_build_inputs`.

use std::collections::{BTreeMap, HashMap};

use novomodelo_core::EntityId;
use novomodelo_core::scenario::LoadModel;

use crate::indexer::{EntityPositions, HydroCellIndex, HydroSys, StudyDimensions};
use crate::policy::resolved_generic_coefficients::ResolvedGenericCoefficients;
use crate::resolved_parameters::ResolvedParameters;
use crate::time_value::TimeValue;

/// One hydro's tailrace-flow terms with every influencing plant resolved to its
/// [`HydroSys`] index, each list in ascending index order; all factors are
/// dimensionless weights.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ResolvedTailraceLateral {
    /// Weight on the hydro's own turbined plus spilled outflow.
    pub(crate) own_outflow_factor: f64,
    /// Plants whose outflow adds to the tailrace flow.
    pub(crate) outflow: Vec<(HydroSys, f64)>,
    /// Plants whose incremental inflow adds to the tailrace flow.
    pub(crate) incremental_inflow: Vec<(HydroSys, f64)>,
}

#[cfg(test)]
impl ResolvedTailraceLateral {
    /// The hydro's own outflow at weight `1.0`, no influences. Not `Default`:
    /// a derived default weighs the own outflow `0.0`.
    pub(crate) fn undeclared() -> Self {
        Self {
            own_outflow_factor: 1.0,
            outflow: Vec::new(),
            incremental_inflow: Vec::new(),
        }
    }
}

/// Resolved study inputs [`build_stage_templates`](super::build_stage_templates)
/// consumes by value; every field is the single derivation of its fact, either
/// owned here or borrowed from the same setup `resolve_stage_data` step that
/// resolves the other borrowed fields.
pub(crate) struct LpBuildInputs<'a> {
    /// Canonical entity-id → slot maps for every position-addressed family.
    pub(crate) positions: EntityPositions,
    /// Per-stage minimum target-storage trajectory, keyed `(hydro_idx,
    /// stage_id) → V_target` \[hm³\].
    pub(crate) filling_v_target: BTreeMap<(usize, i32), f64>,
    /// Declared load-balance models for buses outside the stochastic
    /// load-noise membership.
    pub(crate) deterministic_load_models: Vec<LoadModel>,
    /// Bus-slice positions of the stochastic load-noise-member buses.
    pub(crate) load_bus_indices: Vec<usize>,
    /// Target hydro ID → system indices of hydros diverting to it.
    pub(crate) diversion_upstream: HashMap<EntityId, Vec<usize>>,
    /// Per-hydro tailrace-flow terms, one record per hydro in system order.
    pub(crate) tailrace_lateral: Vec<ResolvedTailraceLateral>,
    /// Per-stage hydro productivities (MW per m³/s); FPHA hydros carry `0.0`.
    pub(crate) hydro_productivities_per_stage: Vec<Vec<f64>>,
    /// Study-invariant, non-state LP shape, from setup's
    /// `build_study_dimensions`.
    pub(crate) study_dims: &'a StudyDimensions,
    /// Present-value discounting and delivery hours/ids, from
    /// [`crate::time_value::TimeValue::from_system`].
    pub(crate) time_value: &'a TimeValue,
    /// Study-scope hydro-cell partition, from [`HydroCellIndex::build`].
    pub(crate) hydro_cell_index: &'a HydroCellIndex,
    /// Resolved `(parameter_id, stage_idx, block_idx)` table, from
    /// [`crate::resolved_parameters::build_resolved_parameters`].
    pub(crate) resolved_parameters: &'a ResolvedParameters,
    /// Per-`(stage, block)` generic-constraint term values and bound
    /// remainders, from [`ResolvedGenericCoefficients::build`].
    pub(crate) resolved_generic_coefficients: &'a ResolvedGenericCoefficients,
}
