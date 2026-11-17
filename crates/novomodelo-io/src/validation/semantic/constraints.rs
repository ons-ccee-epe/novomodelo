//! Layer 5a — generic-constraint semantic validation: per-block references against stage mode,
//! and the terms, parameters and bounds rows a stage-volume constraint may take.
//!
//! Rejects per-block variable references that cannot resolve to a real column on a
//! stage: interior storage boundaries (`HydroStorageInitial` / `HydroStorageFinal`)
//! on parallel stages, and out-of-range block selectors on any per-block reference.

use std::collections::{BTreeMap, HashMap};

use novomodelo_core::{
    ComputedParameter, EntityId, GenericConstraint, ParameterKind, ScalarParameter, VariableRef,
    expand_expression_parameters, temporal::BlockMode,
};

use super::super::{ValidationContext, rules, rules::ValidationRule, schema::ParsedData};
use crate::expression::catalog::variable_call;

/// Rule 20. Layer 5a — rejects per-block variable references that address a column a stage
/// cannot expose, honoring per-`(constraint, stage)` activation.
///
/// `HydroStorageInitial{Some(b)}` references boundary `b` (start of block `b`);
/// `HydroStorageFinal{Some(b)}` references boundary `b + 1` (end of block `b`); a
/// boundary is interior iff `0 < k < K`. On a `Parallel` stage with `K > 1` only
/// the two endpoints (`0`, `K`) exist — the `storage_internal` family is empty — so
/// an interior `Some(..)` reference resolves outside the storage family and is
/// rejected. A `None` selector is the stage endpoint (`S⁰` / `Sᴷ`), which always
/// exists, so it always passes. On a `Parallel` stage with `K > 1`,
/// `HydroEvaporation` is one stage-level quantity: `None` and `Some(0)` resolve to
/// it and pass, while `Some(1..K)` names a block the collapse discards and is
/// rejected. On a `Chronological` stage each block keeps its own evaporation
/// column, so any in-range `Some(b)` passes, but a bare `HydroEvaporation{None}`
/// with `K > 1` is ambiguous (the blocks differ) and is rejected — a block must
/// be named.
///
/// An out-of-range `block_id` (`b >= K`) is rejected on every stage: it would
/// otherwise resolve to no column and silently drop the term.
///
/// A `(constraint, stage)` pair the constraint does not activate (no bound row for
/// that stage) is skipped: no LP row is emitted there, so validating it would
/// false-reject a legitimate mixed-mode study.
pub(super) fn check_per_block_storage_interior_reference(
    data: &ParsedData,
    ctx: &mut ValidationContext,
) {
    for stage in data.stages.stages.iter().filter(|s| s.id >= 0) {
        let k = stage.blocks.len();
        let stage_id = stage.id;
        for constraint in &data.generic_constraints {
            if !constraint_active_on_stage(data, constraint.id.0, stage_id) {
                continue;
            }
            for term in &constraint.expression.terms {
                validate_block_ref(
                    constraint,
                    &term.variable,
                    k,
                    stage_id,
                    stage.block_mode,
                    ctx,
                );
            }
        }
    }
}

/// Whether `constraint` has a bound row at `stage_id` — mirrors
/// `ResolvedGenericConstraintBounds::is_active`; only an active `(constraint, stage)`
/// pair contributes an LP row.
fn constraint_active_on_stage(data: &ParsedData, constraint_id: i32, stage_id: i32) -> bool {
    data.generic_constraint_bounds
        .iter()
        .any(|r| r.constraint_id == constraint_id && r.stage_id == stage_id)
}

/// Unit family of a stored-energy computed tag.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EnergyUnit {
    /// `MW/(m3/s)`, and `MW/(m3/s) * hm3` for `max_stored_energy`.
    Raw,
    /// `MWh/hm3`, and `MWh` for `max_stored_energy_mwh`.
    Mwh,
}

/// Role of a computed tag in a stored-energy constraint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EnergyRole {
    /// The right-hand-side reference (`max_stored_energy*`).
    Ceiling,
    /// A productivity coefficient on a volume or flow term.
    Productivity,
}

/// A computed tag Rule 51 inspects: its role, unit, hydro and JSON tag.
#[derive(Debug, Clone, Copy)]
struct EnergyTag {
    role: EnergyRole,
    unit: EnergyUnit,
    hydro_id: EntityId,
    name: &'static str,
}

fn energy_tag(cp: ComputedParameter) -> Option<EnergyTag> {
    use ComputedParameter as C;
    use EnergyRole::{Ceiling, Productivity};
    use EnergyUnit::{Mwh, Raw};
    let (role, unit, hydro_id, name) = match cp {
        C::MaxStoredEnergy { hydro_id } => (Ceiling, Raw, hydro_id, "max_stored_energy"),
        C::MaxStoredEnergyMwh { hydro_id } => (Ceiling, Mwh, hydro_id, "max_stored_energy_mwh"),
        C::EquivalentProductivity { hydro_id } => {
            (Productivity, Raw, hydro_id, "equivalent_productivity")
        }
        C::AccumulatedProductivity { hydro_id } => {
            (Productivity, Raw, hydro_id, "accumulated_productivity")
        }
        C::IntegratedEquivalentProductivity { hydro_id } => (
            Productivity,
            Raw,
            hydro_id,
            "integrated_equivalent_productivity",
        ),
        C::IntegratedAccumulatedProductivity { hydro_id } => (
            Productivity,
            Raw,
            hydro_id,
            "integrated_accumulated_productivity",
        ),
        C::IntegratedEquivalentProductivityMwhPerHm3 { hydro_id } => (
            Productivity,
            Mwh,
            hydro_id,
            "integrated_equivalent_productivity_mwh_per_hm3",
        ),
        C::IntegratedAccumulatedProductivityMwhPerHm3 { hydro_id } => (
            Productivity,
            Mwh,
            hydro_id,
            "integrated_accumulated_productivity_mwh_per_hm3",
        ),
        C::ReferenceVolume { .. }
        | C::ReferenceTurbine { .. }
        | C::MinStorage { .. }
        | C::MaxStorage { .. }
        | C::SpecificProductivity { .. } => return None,
    };
    Some(EnergyTag {
        role,
        unit,
        hydro_id,
        name,
    })
}

/// The only productivity tag that cancels a ceiling of `unit`: same unit,
/// cascade scope, useful-range mean evaluator.
fn matching_productivity(unit: EnergyUnit) -> &'static str {
    match unit {
        EnergyUnit::Raw => "integrated_accumulated_productivity",
        EnergyUnit::Mwh => "integrated_accumulated_productivity_mwh_per_hm3",
    }
}

/// Rule 51. Layer 5a — warns when a constraint's stored-energy tags would not
/// cancel: a `max_stored_energy*(h)` ceiling beside a same-unit productivity tag
/// for the same hydro `h` other than the matching one (another scope or
/// evaluator), or raw and `MWh` tags mixed in one constraint (a factor of
/// `10^6/3600` apart). It reads tags reached through expression parameters too.
/// Warning only — never rejected.
pub(super) fn check_productivity_tag_pairing(data: &ParsedData, ctx: &mut ValidationContext) {
    let computed: HashMap<EntityId, EnergyTag> = data
        .scalar_parameters
        .iter()
        .filter_map(|p| match p.kind {
            ParameterKind::Computed { computed_spec } => {
                energy_tag(computed_spec).map(|tag| (p.id, tag))
            }
            _ => None,
        })
        .collect();

    for constraint in &data.generic_constraints {
        let mut tags: Vec<EnergyTag> =
            expand_expression_parameters(constraint.parameter_ids(), &data.scalar_parameters)
                .iter()
                .filter_map(|id| computed.get(id).copied())
                .collect();
        tags.sort_by_key(|t| (t.hydro_id, t.name));
        tags.dedup_by_key(|t| (t.hydro_id, t.name));

        for ceiling in tags.iter().filter(|t| t.role == EnergyRole::Ceiling) {
            let matching = matching_productivity(ceiling.unit);
            for other in tags.iter().filter(|t| {
                t.role == EnergyRole::Productivity
                    && t.hydro_id == ceiling.hydro_id
                    && t.unit == ceiling.unit
                    && t.name != matching
            }) {
                ctx.emit(
                    &rules::SEMANTIC_GENERIC_PRODUCTIVITY_TAG_MISMATCH,
                    "constraints/generic_constraints.json",
                    Some(&constraint.name),
                    format!(
                        "Constraint \"{}\": pairs {}({hid}) with {}({hid}) for hydro \
                         {hid}; the two differ in scope or evaluator and would not \
                         cancel. The coefficient matching {} is {matching}.",
                        constraint.name,
                        ceiling.name,
                        other.name,
                        ceiling.name,
                        hid = ceiling.hydro_id.0,
                    ),
                );
            }
        }

        let names_in = |unit: EnergyUnit| -> Vec<&'static str> {
            let mut names: Vec<&'static str> = tags
                .iter()
                .filter(|t| t.unit == unit)
                .map(|t| t.name)
                .collect();
            names.sort_unstable();
            names.dedup();
            names
        };
        let (raw, mwh) = (names_in(EnergyUnit::Raw), names_in(EnergyUnit::Mwh));
        if !raw.is_empty() && !mwh.is_empty() {
            ctx.emit(
                &rules::SEMANTIC_GENERIC_PRODUCTIVITY_TAG_MISMATCH,
                "constraints/generic_constraints.json",
                Some(&constraint.name),
                format!(
                    "Constraint \"{}\": mixes MWh tags ({}) with raw productivity tags \
                     ({}); the two units differ by a factor of 10^6/3600. Use one family \
                     throughout.",
                    constraint.name,
                    mwh.join(", "),
                    raw.join(", "),
                ),
            );
        }
    }
}

/// Which per-block storage boundary a term references, and the block selector.
///
/// `Initial` references boundary `block_id`; `Final` references `block_id + 1`.
#[derive(Clone, Copy)]
enum StorageRef {
    Initial(Option<usize>),
    Final(Option<usize>),
}

fn add_constraint_error(
    ctx: &mut ValidationContext,
    rule: &ValidationRule,
    constraint: &GenericConstraint,
    message: String,
) {
    ctx.emit(
        rule,
        "constraints/generic_constraints.json",
        Some(format!("constraint[id={}]", constraint.id.0)),
        message,
    );
}

/// Dispatch a term to its per-block validity check. Storage boundaries get the
/// full interior + out-of-range check; evaporation gets the out-of-range check
/// plus, on a `Parallel` stage with `K > 1`, a reject for any block past the
/// stage-level slot (`Some(1..K)`) and, on a `Chronological` stage with `K > 1`,
/// a reject for the ambiguous bare reference; every other variant is
/// unrestricted.
fn validate_block_ref(
    constraint: &GenericConstraint,
    variable: &VariableRef,
    k: usize,
    stage_id: i32,
    block_mode: BlockMode,
    ctx: &mut ValidationContext,
) {
    if let Some(boundary) = storage_boundary_ref(variable) {
        validate_storage_ref(constraint, variable, boundary, k, stage_id, block_mode, ctx);
        return;
    }
    match variable {
        VariableRef::HydroEvaporation {
            block_id: Some(b), ..
        } if *b >= k => {
            let term = variable_call(variable);
            add_constraint_error(
                ctx,
                &rules::SEMANTIC_GENERIC_PER_BLOCK_REFERENCE_UNRESOLVABLE,
                constraint,
                format!(
                    "Constraint \"{}\": per-block evaporation reference \
                     `{term}` at stage {stage_id} references block {b} \
                     which does not exist at stage {stage_id} (K = {k})",
                    constraint.name
                ),
            );
        }
        VariableRef::HydroEvaporation {
            block_id: Some(b), ..
        } if block_mode == BlockMode::Parallel && k > 1 && *b >= 1 => {
            let term = variable_call(variable);
            add_constraint_error(
                ctx,
                &rules::SEMANTIC_GENERIC_PER_BLOCK_REFERENCE_UNRESOLVABLE,
                constraint,
                format!(
                    "Constraint \"{}\": per-block evaporation reference \
                     `{term}` at stage {stage_id} names block {b}, past \
                     the stage-level evaporation, which requires chronological block \
                     mode (stage {stage_id} is parallel with {k} blocks); use block 0 \
                     or no block",
                    constraint.name
                ),
            );
        }
        VariableRef::HydroEvaporation {
            hydro_id,
            block_id: None,
        } if block_mode == BlockMode::Chronological && k > 1 => {
            let term = variable_call(variable);
            let example = variable_call(&VariableRef::HydroEvaporation {
                hydro_id: *hydro_id,
                block_id: Some(0),
            });
            add_constraint_error(
                ctx,
                &rules::SEMANTIC_GENERIC_PER_BLOCK_REFERENCE_UNRESOLVABLE,
                constraint,
                format!(
                    "Constraint \"{}\": stage-level `{term}` at \
                     chronological stage {stage_id} is ambiguous — evaporation is \
                     per-block there (K = {k}); name a block, e.g. `{example}`",
                    constraint.name
                ),
            );
        }
        _ => {}
    }
}

/// The boundary a storage-boundary term references, `None` for every other variant.
fn storage_boundary_ref(variable: &VariableRef) -> Option<StorageRef> {
    match variable {
        VariableRef::HydroStorageInitial { block_id, .. }
        | VariableRef::HydroUsefulVolumeInitial { block_id, .. } => {
            Some(StorageRef::Initial(*block_id))
        }
        VariableRef::HydroStorageFinal { block_id, .. }
        | VariableRef::HydroUsefulVolumeFinal { block_id, .. } => {
            Some(StorageRef::Final(*block_id))
        }
        _ => None,
    }
}

fn validate_storage_ref(
    constraint: &GenericConstraint,
    variable: &VariableRef,
    boundary: StorageRef,
    k: usize,
    stage_id: i32,
    block_mode: BlockMode,
    ctx: &mut ValidationContext,
) {
    let (StorageRef::Initial(block_id) | StorageRef::Final(block_id)) = boundary;

    if let Some(b) = block_id
        && b >= k
    {
        let rendered = variable_call(variable);
        add_constraint_error(
            ctx,
            &rules::SEMANTIC_GENERIC_PER_BLOCK_REFERENCE_UNRESOLVABLE,
            constraint,
            format!(
                "Constraint \"{}\": per-block storage reference `{rendered}` at \
                 stage {stage_id} references block {b} which does not exist at \
                 stage {stage_id} (K = {k})",
                constraint.name
            ),
        );
        return;
    }

    match block_mode {
        BlockMode::Chronological => {}
        BlockMode::Parallel => {
            if k <= 1 {
                return;
            }
            let Some(b) = block_id else {
                return;
            };
            let interior = match boundary {
                StorageRef::Initial(_) => boundary_is_interior(b, k),
                StorageRef::Final(_) => boundary_is_interior(b + 1, k),
            };
            if interior {
                let rendered = variable_call(variable);
                add_constraint_error(
                    ctx,
                    &rules::SEMANTIC_GENERIC_PER_BLOCK_REFERENCE_UNRESOLVABLE,
                    constraint,
                    format!(
                        "Constraint \"{}\": per-block storage reference `{rendered}` at \
                         stage {stage_id} resolves to an interior boundary, which requires \
                         chronological block mode (stage {stage_id} is parallel with {k} blocks)",
                        constraint.name
                    ),
                );
            }
        }
    }
}

/// A boundary `k` is interior iff it is neither the stage-initial anchor
/// (`k == 0`) nor the stage-final boundary (`k == K`).
fn boundary_is_interior(k: usize, num_blocks: usize) -> bool {
    k > 0 && k < num_blocks
}

fn is_stage_volume(variable: &VariableRef) -> bool {
    matches!(
        variable,
        VariableRef::HydroOutflowVolume { .. }
            | VariableRef::HydroDiversionVolume { .. }
            | VariableRef::PumpingVolume { .. }
    )
}

/// Block-independent is not enough: evaporation and the anticipated decision are not amounts.
fn is_stage_amount(variable: &VariableRef) -> bool {
    match variable {
        VariableRef::HydroOutflowVolume { .. }
        | VariableRef::HydroDiversionVolume { .. }
        | VariableRef::PumpingVolume { .. }
        | VariableRef::HydroStorage { .. } => true,
        VariableRef::HydroStorageInitial { block_id, .. }
        | VariableRef::HydroStorageFinal { block_id, .. }
        | VariableRef::HydroUsefulVolumeInitial { block_id, .. }
        | VariableRef::HydroUsefulVolumeFinal { block_id, .. } => block_id.is_none(),
        VariableRef::HydroTurbined { .. }
        | VariableRef::HydroSpillage { .. }
        | VariableRef::HydroDiversion { .. }
        | VariableRef::HydroOutflow { .. }
        | VariableRef::HydroGeneration { .. }
        | VariableRef::HydroEvaporation { .. }
        | VariableRef::HydroWithdrawal { .. }
        | VariableRef::HydroInflow { .. }
        | VariableRef::ThermalGeneration { .. }
        | VariableRef::LineDirect { .. }
        | VariableRef::LineReverse { .. }
        | VariableRef::LineExchange { .. }
        | VariableRef::BusDeficit { .. }
        | VariableRef::BusExcess { .. }
        | VariableRef::PumpingFlow { .. }
        | VariableRef::PumpingPower { .. }
        | VariableRef::ContractImport { .. }
        | VariableRef::ContractExport { .. }
        | VariableRef::NonControllableGeneration { .. }
        | VariableRef::NonControllableCurtailment { .. }
        | VariableRef::AnticipatedDecision { .. } => false,
    }
}

/// Rules 56, 57 and 58. Layer 5a — refuses, beside a stage-volume term, every term that is not a
/// block-free stage amount, every `per_stage_block` parameter a coefficient or bound reads,
/// directly or through expression parameters, and every bounds row that names a `block_id`.
/// The first two refusals do not depend on the constraint's bounds rows.
pub(super) fn check_stage_volume_constraints(data: &ParsedData, ctx: &mut ValidationContext) {
    let parameters: HashMap<EntityId, &ScalarParameter> =
        data.scalar_parameters.iter().map(|p| (p.id, p)).collect();
    let mut volume_constraints: BTreeMap<i32, &str> = BTreeMap::new();
    for constraint in &data.generic_constraints {
        let terms = &constraint.expression.terms;
        if !terms.iter().any(|t| is_stage_volume(&t.variable)) {
            continue;
        }
        volume_constraints.insert(constraint.id.0, &constraint.name);
        let sources = data
            .generic_constraint_named_sources
            .get(&constraint.id)
            .map_or(&[][..], Vec::as_slice);
        for term in terms.iter().filter(|t| !is_stage_amount(&t.variable)) {
            let introduced_by: Vec<&str> = sources
                .iter()
                .filter(|source| source.variables.contains(&term.variable))
                .map(|source| source.name.as_str())
                .collect();
            let origin = if introduced_by.is_empty() {
                String::new()
            } else {
                format!(" (from @{})", introduced_by.join(", @"))
            };
            add_constraint_error(
                ctx,
                &rules::SEMANTIC_GENERIC_STAGE_VOLUME_NON_AMOUNT_TERM,
                constraint,
                format!(
                    "Constraint \"{}\": `{}`{origin} cannot appear beside a stage-volume term; \
                     such a constraint takes only stage volumes, hydro_storage, and \
                     hydro_storage_* or hydro_useful_volume_* without a block argument",
                    constraint.name,
                    variable_call(&term.variable),
                ),
            );
        }
        let reached =
            expand_expression_parameters(constraint.parameter_ids(), &data.scalar_parameters);
        for parameter in reached
            .iter()
            .filter_map(|id| parameters.get(id))
            .filter(|p| matches!(p.kind, ParameterKind::PerStageBlock { .. }))
        {
            add_constraint_error(
                ctx,
                &rules::SEMANTIC_GENERIC_STAGE_VOLUME_PER_BLOCK_PARAMETER,
                constraint,
                format!(
                    "Constraint \"{}\": parameter `@{}` is per_stage_block; a constraint \
                     with a stage-volume term takes no coefficient or bound that varies \
                     by block",
                    constraint.name, parameter.name,
                ),
            );
        }
    }
    for row in &data.generic_constraint_bounds {
        let (Some(name), Some(block)) = (volume_constraints.get(&row.constraint_id), row.block_id)
        else {
            continue;
        };
        ctx.emit(
            &rules::SEMANTIC_GENERIC_STAGE_VOLUME_BLOCK_BOUND_ROW,
            "constraints/generic_constraint_bounds.parquet",
            Some(format!(
                "constraint_id={}, stage_id={}",
                row.constraint_id, row.stage_id
            )),
            format!(
                "Constraint \"{name}\": the bounds row at stage_id={} has block_id={block}; a \
                 constraint with a stage-volume term takes one row per stage with a null block_id",
                row.stage_id,
            ),
        );
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use novomodelo_core::{
        ComputedParameter, ConstraintExpression, EntityId, GenericConstraint, LinearTerm,
        ParameterKind, ScalarExpr, ScalarParameter, VariableRef,
        temporal::{Block, BlockMode},
    };

    use super::super::validate_semantic_hydro_thermal;
    use super::{check_productivity_tag_pairing, check_stage_volume_constraints};
    use crate::ValidationEntry;
    use crate::constraints::GenericConstraintBoundsRow;
    use crate::expression::catalog::{build_variable_ref, variable_call};
    use crate::expression::lower::NamedSource;
    use crate::test_support::*;
    use crate::validation::schema::ParsedData;
    use crate::validation::{ErrorKind, ValidationContext};

    fn make_blocks(k: usize) -> Vec<Block> {
        (0..k)
            .map(|i| Block {
                index: i,
                name: format!("B{i}"),
                duration_hours: 168.0,
            })
            .collect()
    }

    /// Build `ParsedData` with a single stage of `k` blocks in `block_mode` and a
    /// generic constraint whose sole term references the given variant, active on
    /// stage 0 (a bound row exists, so the per-block check runs).
    fn make_data_storage_ref(block_mode: BlockMode, k: usize, variable: VariableRef) -> ParsedData {
        let mut data = make_data(
            vec![make_hydro(1, None)],
            vec![],
            vec![],
            make_stages(vec![0]),
            vec![],
            vec![],
        );
        data.stages.stages[0].block_mode = block_mode;
        data.stages.stages[0].blocks = make_blocks(k);
        data.generic_constraints = vec![GenericConstraint {
            id: EntityId::from(1),
            name: "storage_constraint".to_string(),
            description: None,
            expression: ConstraintExpression {
                terms: vec![LinearTerm::literal(1.0, variable)],
            },
            slack: None,
            bound_lower: None,
            bound_upper: None,
        }];
        data.generic_constraint_bounds = vec![GenericConstraintBoundsRow {
            constraint_id: 1,
            stage_id: 0,
            block_id: None,
            bound_lower: Some(0.0),
            bound_upper: None,
        }];
        data
    }

    fn evaporation(block_id: Option<usize>) -> VariableRef {
        VariableRef::HydroEvaporation {
            hydro_id: EntityId::from(1),
            block_id,
        }
    }

    fn interior_errors(data: &ParsedData) -> Vec<ValidationEntry> {
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(data, &mut ctx);
        ctx.errors()
            .iter()
            .filter(|e| {
                e.kind == ErrorKind::BusinessRuleViolation
                    && e.file
                        .to_string_lossy()
                        .contains("constraints/generic_constraints.json")
            })
            .map(|e| (*e).clone())
            .collect()
    }

    fn initial(block_id: Option<usize>) -> VariableRef {
        VariableRef::HydroStorageInitial {
            hydro_id: EntityId::from(1),
            block_id,
        }
    }

    fn final_(block_id: Option<usize>) -> VariableRef {
        VariableRef::HydroStorageFinal {
            hydro_id: EntityId::from(1),
            block_id,
        }
    }

    fn useful_initial(block_id: Option<usize>) -> VariableRef {
        VariableRef::HydroUsefulVolumeInitial {
            hydro_id: EntityId::from(1),
            block_id,
        }
    }

    fn useful_final(block_id: Option<usize>) -> VariableRef {
        VariableRef::HydroUsefulVolumeFinal {
            hydro_id: EntityId::from(1),
            block_id,
        }
    }

    /// The single error a reference raises, with `accessor` rewritten to the
    /// storage sibling's name so the two messages can be compared verbatim.
    fn sole_error_as_storage(
        block_mode: BlockMode,
        variable: VariableRef,
        accessor: &str,
        storage_accessor: &str,
    ) -> String {
        let errors = interior_errors(&make_data_storage_ref(block_mode, 3, variable));
        assert_eq!(errors.len(), 1, "expected one error, got: {errors:?}");
        let msg = &errors[0].message;
        assert!(
            msg.contains(accessor),
            "message should name `{accessor}`, got: {msg}"
        );
        msg.replace(accessor, storage_accessor)
    }

    #[test]
    fn useful_volume_out_of_range_block_matches_storage_sibling() {
        for block_mode in [BlockMode::Parallel, BlockMode::Chronological] {
            for (useful, storage, accessor, storage_accessor) in [
                (
                    useful_initial(Some(5)),
                    initial(Some(5)),
                    "hydro_useful_volume_initial",
                    "hydro_storage_initial",
                ),
                (
                    useful_final(Some(5)),
                    final_(Some(5)),
                    "hydro_useful_volume_final",
                    "hydro_storage_final",
                ),
            ] {
                assert_eq!(
                    sole_error_as_storage(block_mode, useful, accessor, storage_accessor),
                    sole_error_as_storage(block_mode, storage, storage_accessor, storage_accessor),
                );
            }
        }
    }

    #[test]
    fn useful_volume_parallel_interior_boundary_matches_storage_sibling() {
        // Initial{1} and Final{0} both reference boundary k=1, interior for K=3.
        for (useful, storage, accessor, storage_accessor) in [
            (
                useful_initial(Some(1)),
                initial(Some(1)),
                "hydro_useful_volume_initial",
                "hydro_storage_initial",
            ),
            (
                useful_final(Some(0)),
                final_(Some(0)),
                "hydro_useful_volume_final",
                "hydro_storage_final",
            ),
        ] {
            assert_eq!(
                sole_error_as_storage(BlockMode::Parallel, useful, accessor, storage_accessor),
                sole_error_as_storage(
                    BlockMode::Parallel,
                    storage,
                    storage_accessor,
                    storage_accessor
                ),
            );
        }
    }

    #[test]
    fn non_storage_per_block_reference_is_unrestricted() {
        // An interior block on a parallel stage would be rejected for a storage boundary.
        for block_mode in [BlockMode::Parallel, BlockMode::Chronological] {
            let turbined = VariableRef::HydroTurbined {
                hydro_id: EntityId::from(1),
                block_id: Some(1),
                bus_id: None,
            };
            let data = make_data_storage_ref(block_mode, 3, turbined);
            assert!(interior_errors(&data).is_empty());
        }
    }

    #[test]
    fn useful_volume_valid_references_accepted() {
        for (block_mode, variable) in [
            (BlockMode::Parallel, useful_initial(Some(0))),
            (BlockMode::Parallel, useful_final(Some(2))),
            (BlockMode::Parallel, useful_initial(None)),
            (BlockMode::Parallel, useful_final(None)),
            (BlockMode::Chronological, useful_initial(Some(1))),
            (BlockMode::Chronological, useful_final(Some(1))),
        ] {
            let data = make_data_storage_ref(block_mode, 3, variable);
            assert!(interior_errors(&data).is_empty());
        }
    }

    #[test]
    fn parallel_k3_interior_initial_rejected() {
        let data = make_data_storage_ref(BlockMode::Parallel, 3, initial(Some(1)));
        let errors = interior_errors(&data);
        assert_eq!(errors.len(), 1, "expected one error, got: {errors:?}");
        let msg = &errors[0].message;
        assert!(
            msg.contains("storage_constraint") && msg.contains("interior boundary"),
            "message should name the constraint and the interior requirement, got: {msg}"
        );
        assert!(
            msg.contains("parallel with 3 blocks"),
            "message should state the parallel mode and block count, got: {msg}"
        );
    }

    #[test]
    fn parallel_k3_interior_final_rejected() {
        // Final{0} references boundary k=1, interior for K=3.
        let data = make_data_storage_ref(BlockMode::Parallel, 3, final_(Some(0)));
        let errors = interior_errors(&data);
        assert_eq!(errors.len(), 1, "expected one error, got: {errors:?}");
    }

    #[test]
    fn parallel_k3_endpoint_initial_accepted() {
        // Initial{0} references boundary k=0, the S⁰ endpoint.
        let data = make_data_storage_ref(BlockMode::Parallel, 3, initial(Some(0)));
        assert!(interior_errors(&data).is_empty());
    }

    #[test]
    fn parallel_k3_endpoint_final_accepted() {
        // Final{2} references boundary k=3=K, the Sᴷ endpoint.
        let data = make_data_storage_ref(BlockMode::Parallel, 3, final_(Some(2)));
        assert!(interior_errors(&data).is_empty());
    }

    #[test]
    fn parallel_k3_none_initial_accepted() {
        // Initial{None} is the S⁰ endpoint, which exists on a parallel stage.
        let data = make_data_storage_ref(BlockMode::Parallel, 3, initial(None));
        assert!(interior_errors(&data).is_empty());
    }

    #[test]
    fn parallel_k3_none_final_accepted() {
        // Final{None} is the Sᴷ endpoint, which exists on a parallel stage.
        let data = make_data_storage_ref(BlockMode::Parallel, 3, final_(None));
        assert!(interior_errors(&data).is_empty());
    }

    #[test]
    fn parallel_k1_all_references_accepted() {
        for variable in [
            initial(Some(0)),
            final_(Some(0)),
            initial(None),
            final_(None),
        ] {
            let data = make_data_storage_ref(BlockMode::Parallel, 1, variable);
            assert!(
                interior_errors(&data).is_empty(),
                "K=1 parallel reference must be accepted"
            );
        }
    }

    #[test]
    fn chronological_k3_all_references_accepted() {
        for variable in [
            initial(Some(0)),
            initial(Some(1)),
            initial(Some(2)),
            final_(Some(0)),
            final_(Some(1)),
            final_(Some(2)),
            initial(None),
            final_(None),
        ] {
            let data = make_data_storage_ref(BlockMode::Chronological, 3, variable);
            assert!(
                interior_errors(&data).is_empty(),
                "chronological reference must be accepted"
            );
        }
    }

    #[test]
    fn parallel_k3_out_of_range_initial_rejected() {
        let data = make_data_storage_ref(BlockMode::Parallel, 3, initial(Some(5)));
        let errors = interior_errors(&data);
        assert_eq!(errors.len(), 1, "expected one error, got: {errors:?}");
        let msg = &errors[0].message;
        assert!(
            msg.contains("block 5 which does not exist") && msg.contains("K = 3"),
            "message should be the out-of-range message, got: {msg}"
        );
    }

    #[test]
    fn parallel_k3_out_of_range_final_rejected() {
        let data = make_data_storage_ref(BlockMode::Parallel, 3, final_(Some(5)));
        let errors = interior_errors(&data);
        assert_eq!(errors.len(), 1, "expected one error, got: {errors:?}");
        let msg = &errors[0].message;
        assert!(
            msg.contains("block 5 which does not exist") && msg.contains("K = 3"),
            "message should be the out-of-range message, got: {msg}"
        );
    }

    #[test]
    fn chronological_k3_out_of_range_initial_rejected() {
        let data = make_data_storage_ref(BlockMode::Chronological, 3, initial(Some(5)));
        let errors = interior_errors(&data);
        assert_eq!(errors.len(), 1, "expected one error, got: {errors:?}");
        assert!(errors[0].message.contains("block 5 which does not exist"));
    }

    #[test]
    fn chronological_k3_out_of_range_final_rejected() {
        let data = make_data_storage_ref(BlockMode::Chronological, 3, final_(Some(5)));
        let errors = interior_errors(&data);
        assert_eq!(errors.len(), 1, "expected one error, got: {errors:?}");
        assert!(errors[0].message.contains("block 5 which does not exist"));
    }

    #[test]
    fn parallel_k3_interior_ref_on_inactive_constraint_accepted() {
        // An interior reference the constraint never activates on this stage
        // (no bound row) emits no LP row, so it must not be rejected.
        let mut data = make_data_storage_ref(BlockMode::Parallel, 3, initial(Some(1)));
        data.generic_constraint_bounds.clear();
        assert!(
            interior_errors(&data).is_empty(),
            "inactive (constraint, stage) pair must be skipped"
        );
    }

    #[test]
    fn mixed_mode_interior_ref_active_only_on_chronological_stage_accepted() {
        // Stage 0 chronological (K=3), stage 1 parallel (K=3); the constraint
        // references an interior boundary but is active only on the chronological
        // stage, so the parallel stage must not be flagged.
        let mut data = make_data_storage_ref(BlockMode::Chronological, 3, initial(Some(1)));
        data.stages.stages.push({
            let mut s = data.stages.stages[0].clone();
            s.id = 1;
            s.block_mode = BlockMode::Parallel;
            s
        });
        assert!(
            interior_errors(&data).is_empty(),
            "constraint inactive on the parallel stage must not be flagged"
        );
    }

    #[test]
    fn parallel_k3_evaporation_block_past_the_stage_slot_rejected() {
        let data = make_data_storage_ref(BlockMode::Parallel, 3, evaporation(Some(2)));
        let errors = interior_errors(&data);
        assert_eq!(errors.len(), 1, "expected one error, got: {errors:?}");
        let msg = &errors[0].message;
        assert!(
            msg.contains("hydro_evaporation(1, 2)")
                && msg.contains("parallel with 3 blocks")
                && msg.contains("block 0"),
            "message should name the block, the parallel mode and block count, \
             and the fix, got: {msg}"
        );
    }

    #[test]
    fn parallel_k3_evaporation_block_zero_accepted() {
        let data = make_data_storage_ref(BlockMode::Parallel, 3, evaporation(Some(0)));
        assert!(interior_errors(&data).is_empty());
    }

    #[test]
    fn parallel_evaporation_block_reference_skipped_where_inactive() {
        // A block-past-the-slot reference the constraint never activates on this
        // stage (no bound row) emits no LP row, so it must not be rejected.
        let mut data = make_data_storage_ref(BlockMode::Parallel, 3, evaporation(Some(2)));
        data.generic_constraint_bounds.clear();
        assert!(
            interior_errors(&data).is_empty(),
            "inactive (constraint, stage) pair must be skipped"
        );
    }

    #[test]
    fn evaporation_bare_accepted_in_parallel() {
        // In parallel mode every block shares the stage endpoints, so bare
        // evaporation is the well-defined stage rate.
        let data = make_data_storage_ref(BlockMode::Parallel, 3, evaporation(None));
        assert!(interior_errors(&data).is_empty());
    }

    #[test]
    fn evaporation_bare_accepted_in_chronological_k1() {
        // K == 1: the single block is the whole stage, so bare is unambiguous.
        let data = make_data_storage_ref(BlockMode::Chronological, 1, evaporation(None));
        assert!(interior_errors(&data).is_empty());
    }

    #[test]
    fn evaporation_block_accepted_in_chronological() {
        let data = make_data_storage_ref(BlockMode::Chronological, 3, evaporation(Some(0)));
        assert!(interior_errors(&data).is_empty());
    }

    #[test]
    fn evaporation_bare_rejected_in_chronological_multiblock() {
        // K > 1 chronological: blocks differ, so a bare (stage-level) reference is
        // ambiguous and must name a block.
        let data = make_data_storage_ref(BlockMode::Chronological, 3, evaporation(None));
        let errors = interior_errors(&data);
        assert_eq!(errors.len(), 1, "expected one error, got: {errors:?}");
        assert!(
            errors[0].message.contains("ambiguous") && errors[0].message.contains("name a block"),
            "message should ask for an explicit block, got: {}",
            errors[0].message
        );
    }

    #[test]
    fn evaporation_out_of_range_block_rejected() {
        let data = make_data_storage_ref(BlockMode::Parallel, 3, evaporation(Some(5)));
        let errors = interior_errors(&data);
        assert_eq!(errors.len(), 1, "expected one error, got: {errors:?}");
        let msg = &errors[0].message;
        assert!(
            msg.contains("hydro_evaporation(1, 5)") && msg.contains("block 5 which does not exist"),
            "message should be the evaporation out-of-range message, got: {msg}"
        );
    }

    fn sole_error_for_hydro_7(block_mode: BlockMode, variable: VariableRef) -> String {
        let mut data = make_data_storage_ref(block_mode, 3, variable);
        data.hydros = vec![make_hydro(7, None)];
        let errors = interior_errors(&data);
        assert_eq!(errors.len(), 1, "expected one error, got: {errors:?}");
        errors[0].message.clone()
    }

    fn evaporation_of_hydro_7(block_id: Option<usize>) -> VariableRef {
        VariableRef::HydroEvaporation {
            hydro_id: EntityId::from(7),
            block_id,
        }
    }

    #[test]
    fn evaporation_block_messages_name_the_hydro_and_the_block() {
        let out_of_range =
            sole_error_for_hydro_7(BlockMode::Parallel, evaporation_of_hydro_7(Some(5)));
        assert!(
            out_of_range.contains("`hydro_evaporation(7, 5)`") && out_of_range.contains("block 5"),
            "got: {out_of_range}"
        );

        let past_the_slot =
            sole_error_for_hydro_7(BlockMode::Parallel, evaporation_of_hydro_7(Some(2)));
        assert!(
            past_the_slot.contains("`hydro_evaporation(7, 2)`")
                && past_the_slot.contains("names block 2"),
            "got: {past_the_slot}"
        );

        let bare = sole_error_for_hydro_7(BlockMode::Chronological, evaporation_of_hydro_7(None));
        assert!(
            bare.contains("`hydro_evaporation(7)`") && bare.contains("`hydro_evaporation(7, 0)`"),
            "got: {bare}"
        );

        for msg in [&out_of_range, &past_the_slot] {
            assert!(
                !msg.contains("hydro_evaporation(5)") && !msg.contains("hydro_evaporation(2)"),
                "block rendered as a hydro id: {msg}"
            );
        }
    }

    #[test]
    fn storage_block_messages_name_the_hydro_and_the_block() {
        for (variable, term) in [
            (
                VariableRef::HydroStorageInitial {
                    hydro_id: EntityId::from(7),
                    block_id: Some(5),
                },
                "`hydro_storage_initial(7, 5)`",
            ),
            (
                VariableRef::HydroStorageInitial {
                    hydro_id: EntityId::from(7),
                    block_id: Some(1),
                },
                "`hydro_storage_initial(7, 1)`",
            ),
            (
                VariableRef::HydroStorageFinal {
                    hydro_id: EntityId::from(7),
                    block_id: Some(0),
                },
                "`hydro_storage_final(7, 0)`",
            ),
        ] {
            let msg = sole_error_for_hydro_7(BlockMode::Parallel, variable);
            assert!(msg.contains(term), "expected {term}, got: {msg}");
        }
    }

    // ── check_productivity_tag_pairing ──────────────────────────────────────

    fn computed_scalar_param(id: i32, name: &str, spec: ComputedParameter) -> ScalarParameter {
        ScalarParameter {
            id: EntityId::from(id),
            name: name.to_string(),
            kind: ParameterKind::Computed {
                computed_spec: spec,
            },
        }
    }

    fn expression_scalar_param(id: i32, name: &str, expression: ScalarExpr) -> ScalarParameter {
        ScalarParameter {
            id: EntityId::from(id),
            name: name.to_string(),
            kind: ParameterKind::Expression { expression },
        }
    }

    fn param_term(coefficient_id: EntityId) -> LinearTerm {
        LinearTerm {
            coefficient: ScalarExpr::param(coefficient_id),
            variable: VariableRef::HydroTurbined {
                hydro_id: EntityId::from(1),
                block_id: None,
                bus_id: None,
            },
        }
    }

    fn constraint_with_lhs_and_upper_bound(
        lhs_param: EntityId,
        upper_bound_param: EntityId,
    ) -> GenericConstraint {
        GenericConstraint {
            id: EntityId::from(1),
            name: "energy_constraint".to_string(),
            description: None,
            expression: ConstraintExpression {
                terms: vec![param_term(lhs_param)],
            },
            slack: None,
            bound_lower: None,
            bound_upper: Some(ScalarExpr::param(upper_bound_param)),
        }
    }

    fn constraint_with_two_lhs_terms(param_a: EntityId, param_b: EntityId) -> GenericConstraint {
        GenericConstraint {
            id: EntityId::from(1),
            name: "energy_constraint".to_string(),
            description: None,
            expression: ConstraintExpression {
                terms: vec![param_term(param_a), param_term(param_b)],
            },
            slack: None,
            bound_lower: None,
            bound_upper: None,
        }
    }

    fn constraint_with_literal_term() -> GenericConstraint {
        GenericConstraint {
            id: EntityId::from(1),
            name: "no_computed_params".to_string(),
            description: None,
            expression: ConstraintExpression {
                terms: vec![LinearTerm::literal(
                    1.0,
                    VariableRef::HydroTurbined {
                        hydro_id: EntityId::from(1),
                        block_id: None,
                        bus_id: None,
                    },
                )],
            },
            slack: None,
            bound_lower: None,
            bound_upper: None,
        }
    }

    /// Build `ParsedData` with a single hydro/stage, the given scalar parameters,
    /// and a single generic constraint.
    fn make_data_with_constraint(
        scalar_parameters: Vec<ScalarParameter>,
        constraint: GenericConstraint,
    ) -> ParsedData {
        let mut data = make_data(
            vec![make_hydro(1, None)],
            vec![],
            vec![],
            make_stages(vec![0]),
            vec![],
            vec![],
        );
        data.scalar_parameters = scalar_parameters;
        data.generic_constraints = vec![constraint];
        data
    }

    #[test]
    fn mismatched_pairing_same_hydro_warns_once() {
        let data = make_data_with_constraint(
            vec![
                computed_scalar_param(
                    1,
                    "rho_acum",
                    ComputedParameter::AccumulatedProductivity {
                        hydro_id: EntityId::from(7),
                    },
                ),
                computed_scalar_param(
                    2,
                    "e_max",
                    ComputedParameter::MaxStoredEnergy {
                        hydro_id: EntityId::from(7),
                    },
                ),
            ],
            constraint_with_lhs_and_upper_bound(EntityId::from(1), EntityId::from(2)),
        );
        let mut ctx = ValidationContext::new();
        check_productivity_tag_pairing(&data, &mut ctx);

        let messages: Vec<String> = ctx
            .warnings()
            .iter()
            .map(|e| {
                assert_eq!(e.kind, ErrorKind::SemanticAmbiguity);
                e.message.clone()
            })
            .collect();
        assert_eq!(messages.len(), 1, "expected one warning, got: {messages:?}");
        assert!(
            messages[0].contains('7'),
            "message should name hydro 7, got: {}",
            messages[0]
        );
        assert!(
            messages[0].contains("integrated_accumulated_productivity"),
            "message should name the matching coefficient, got: {}",
            messages[0]
        );

        assert!(
            ctx.into_result().is_ok(),
            "a warning must not fail into_result"
        );
    }

    #[test]
    fn mismatched_pairing_both_as_lhs_coefficients_warns_once() {
        let data = make_data_with_constraint(
            vec![
                computed_scalar_param(
                    1,
                    "rho_acum",
                    ComputedParameter::AccumulatedProductivity {
                        hydro_id: EntityId::from(7),
                    },
                ),
                computed_scalar_param(
                    2,
                    "e_max",
                    ComputedParameter::MaxStoredEnergy {
                        hydro_id: EntityId::from(7),
                    },
                ),
            ],
            constraint_with_two_lhs_terms(EntityId::from(1), EntityId::from(2)),
        );
        let mut ctx = ValidationContext::new();
        check_productivity_tag_pairing(&data, &mut ctx);

        let warnings = ctx.warnings();
        assert_eq!(warnings.len(), 1, "expected one warning, got: {warnings:?}");
        assert_eq!(warnings[0].kind, ErrorKind::SemanticAmbiguity);
    }

    #[test]
    fn matching_integrated_coefficient_no_warning() {
        let data = make_data_with_constraint(
            vec![
                computed_scalar_param(
                    1,
                    "rho_acum_int",
                    ComputedParameter::IntegratedAccumulatedProductivity {
                        hydro_id: EntityId::from(7),
                    },
                ),
                computed_scalar_param(
                    2,
                    "e_max",
                    ComputedParameter::MaxStoredEnergy {
                        hydro_id: EntityId::from(7),
                    },
                ),
            ],
            constraint_with_lhs_and_upper_bound(EntityId::from(1), EntityId::from(2)),
        );
        let mut ctx = ValidationContext::new();
        check_productivity_tag_pairing(&data, &mut ctx);

        assert!(
            ctx.warnings().is_empty(),
            "the matching integrated_accumulated_productivity coefficient must not warn"
        );
    }

    #[test]
    fn mismatched_pairing_behind_an_expression_parameter_warns_once() {
        let data = make_data_with_constraint(
            vec![
                computed_scalar_param(
                    1,
                    "rho_acum",
                    ComputedParameter::AccumulatedProductivity {
                        hydro_id: EntityId::from(7),
                    },
                ),
                computed_scalar_param(
                    2,
                    "e_max",
                    ComputedParameter::MaxStoredEnergy {
                        hydro_id: EntityId::from(7),
                    },
                ),
                expression_scalar_param(
                    3,
                    "e",
                    ScalarExpr::param(EntityId(1)) * ScalarExpr::literal(2.0),
                ),
            ],
            constraint_with_lhs_and_upper_bound(EntityId(3), EntityId(2)),
        );
        let mut ctx = ValidationContext::new();
        check_productivity_tag_pairing(&data, &mut ctx);

        let messages: Vec<String> = ctx
            .warnings()
            .iter()
            .map(|e| {
                assert_eq!(e.kind, ErrorKind::SemanticAmbiguity);
                e.message.clone()
            })
            .collect();
        assert_eq!(messages.len(), 1, "expected one warning, got: {messages:?}");
        assert!(
            messages[0].contains('7'),
            "message should name hydro 7, got: {}",
            messages[0]
        );
        assert!(
            messages[0].contains("integrated_accumulated_productivity"),
            "message should name the matching coefficient, got: {}",
            messages[0]
        );

        assert!(
            ctx.into_result().is_ok(),
            "a warning must not fail into_result"
        );
    }

    #[test]
    fn matching_integrated_coefficient_behind_an_expression_parameter_no_warning() {
        let data = make_data_with_constraint(
            vec![
                computed_scalar_param(
                    1,
                    "rho_acum_int",
                    ComputedParameter::IntegratedAccumulatedProductivity {
                        hydro_id: EntityId::from(7),
                    },
                ),
                computed_scalar_param(
                    2,
                    "e_max",
                    ComputedParameter::MaxStoredEnergy {
                        hydro_id: EntityId::from(7),
                    },
                ),
                expression_scalar_param(
                    3,
                    "e",
                    ScalarExpr::param(EntityId(1)) * ScalarExpr::literal(2.0),
                ),
            ],
            constraint_with_lhs_and_upper_bound(EntityId(3), EntityId(2)),
        );
        let mut ctx = ValidationContext::new();
        check_productivity_tag_pairing(&data, &mut ctx);

        assert!(
            ctx.warnings().is_empty(),
            "the matching coefficient reached through an expression must not warn"
        );
    }

    #[test]
    fn different_hydro_pairing_no_warning() {
        let data = make_data_with_constraint(
            vec![
                computed_scalar_param(
                    1,
                    "rho_acum_h9",
                    ComputedParameter::AccumulatedProductivity {
                        hydro_id: EntityId::from(9),
                    },
                ),
                computed_scalar_param(
                    2,
                    "e_max_h7",
                    ComputedParameter::MaxStoredEnergy {
                        hydro_id: EntityId::from(7),
                    },
                ),
            ],
            constraint_with_lhs_and_upper_bound(EntityId::from(1), EntityId::from(2)),
        );
        let mut ctx = ValidationContext::new();
        check_productivity_tag_pairing(&data, &mut ctx);

        assert!(
            ctx.warnings().is_empty(),
            "a different-hydro pairing must not warn"
        );
    }

    fn warnings_for_pair(a: ComputedParameter, b: ComputedParameter) -> Vec<String> {
        let data = make_data_with_constraint(
            vec![
                computed_scalar_param(1, "coefficient", a),
                computed_scalar_param(2, "ceiling", b),
            ],
            constraint_with_lhs_and_upper_bound(EntityId::from(1), EntityId::from(2)),
        );
        let mut ctx = ValidationContext::new();
        check_productivity_tag_pairing(&data, &mut ctx);
        ctx.warnings().iter().map(|e| e.message.clone()).collect()
    }

    #[test]
    fn mwh_ceiling_with_matching_mwh_coefficient_no_warning() {
        let hydro_id = EntityId::from(7);
        let warnings = warnings_for_pair(
            ComputedParameter::IntegratedAccumulatedProductivityMwhPerHm3 { hydro_id },
            ComputedParameter::MaxStoredEnergyMwh { hydro_id },
        );
        assert!(warnings.is_empty(), "{warnings:?}");
    }

    #[test]
    fn mwh_ceiling_with_raw_coefficient_warns_unit_mix_once() {
        let hydro_id = EntityId::from(7);
        let warnings = warnings_for_pair(
            ComputedParameter::IntegratedAccumulatedProductivity { hydro_id },
            ComputedParameter::MaxStoredEnergyMwh { hydro_id },
        );
        assert_eq!(warnings.len(), 1, "{warnings:?}");
        assert!(warnings[0].contains("10^6/3600"), "{}", warnings[0]);
    }

    #[test]
    fn raw_and_mwh_tags_on_different_hydros_warn_unit_mix() {
        let warnings = warnings_for_pair(
            ComputedParameter::IntegratedAccumulatedProductivityMwhPerHm3 {
                hydro_id: EntityId::from(7),
            },
            ComputedParameter::MaxStoredEnergy {
                hydro_id: EntityId::from(9),
            },
        );
        assert_eq!(warnings.len(), 1, "{warnings:?}");
        assert!(warnings[0].contains("max_stored_energy"), "{}", warnings[0]);
    }

    #[test]
    fn mwh_ceiling_with_own_scope_mwh_coefficient_warns_and_names_the_match() {
        let hydro_id = EntityId::from(7);
        let warnings = warnings_for_pair(
            ComputedParameter::IntegratedEquivalentProductivityMwhPerHm3 { hydro_id },
            ComputedParameter::MaxStoredEnergyMwh { hydro_id },
        );
        assert_eq!(warnings.len(), 1, "{warnings:?}");
        assert!(
            warnings[0].contains("is integrated_accumulated_productivity_mwh_per_hm3"),
            "{}",
            warnings[0]
        );
    }

    #[test]
    fn no_computed_scalar_parameters_no_warning_no_panic() {
        let data = make_data_with_constraint(vec![], constraint_with_literal_term());
        let mut ctx = ValidationContext::new();
        check_productivity_tag_pairing(&data, &mut ctx);

        assert!(ctx.warnings().is_empty());
    }

    const VOLUME_REFUSAL_TAIL: &str = "cannot appear beside a stage-volume term; such a constraint \
         takes only stage volumes, hydro_storage, and hydro_storage_* or \
         hydro_useful_volume_* without a block argument";

    fn volume_refusal(call: &str) -> String {
        format!("Constraint \"volume_constraint\": `{call}` {VOLUME_REFUSAL_TAIL}")
    }

    fn volume_constraint_data(terms: Vec<(f64, VariableRef)>) -> ParsedData {
        make_data_with_constraint(
            vec![],
            GenericConstraint {
                id: EntityId::from(1),
                name: "volume_constraint".to_string(),
                description: None,
                expression: ConstraintExpression {
                    terms: terms
                        .into_iter()
                        .map(|(coefficient, variable)| LinearTerm::literal(coefficient, variable))
                        .collect(),
                },
                slack: None,
                bound_lower: None,
                bound_upper: None,
            },
        )
    }

    fn outflow_volume() -> VariableRef {
        VariableRef::HydroOutflowVolume {
            hydro_id: EntityId::from(1),
        }
    }

    fn volume_refusal_entries(data: &ParsedData) -> Vec<ValidationEntry> {
        let mut ctx = ValidationContext::new();
        check_stage_volume_constraints(data, &mut ctx);
        ctx.errors().into_iter().cloned().collect()
    }

    #[test]
    fn stage_volume_constraint_refuses_each_non_amount_term() {
        let data = volume_constraint_data(vec![
            (1.0, outflow_volume()),
            (
                1.0,
                VariableRef::HydroTurbined {
                    hydro_id: EntityId::from(1),
                    block_id: Some(0),
                    bus_id: None,
                },
            ),
            (-1.0, final_(Some(0))),
        ]);
        let entries = volume_refusal_entries(&data);
        let messages: Vec<&str> = entries.iter().map(|e| e.message.as_str()).collect();
        assert_eq!(
            messages,
            [
                volume_refusal("hydro_turbined(1, 0)"),
                volume_refusal("hydro_storage_final(1, 0)"),
            ]
        );
        for entry in &entries {
            assert_eq!(entry.kind, ErrorKind::BusinessRuleViolation);
            assert_eq!(
                entry.file.to_string_lossy(),
                "constraints/generic_constraints.json"
            );
            assert_eq!(entry.entity.as_deref(), Some("constraint[id=1]"));
        }
    }

    #[test]
    fn stage_volume_refusal_names_the_named_expression_that_introduced_the_term() {
        let turbined = VariableRef::HydroTurbined {
            hydro_id: EntityId::from(1),
            block_id: None,
            bus_id: None,
        };
        let elsewhere = VariableRef::ThermalGeneration {
            thermal_id: EntityId::from(1),
            block_id: None,
        };
        let source = |name: &str, variables: Vec<VariableRef>| NamedSource {
            name: name.to_owned(),
            variables,
        };
        let messages = |data: &ParsedData| -> Vec<String> {
            volume_refusal_entries(data)
                .into_iter()
                .map(|e| e.message)
                .collect()
        };
        let named = |from: &str| {
            format!(
                "Constraint \"volume_constraint\": `hydro_turbined(1)` ({from}) \
                 {VOLUME_REFUSAL_TAIL}"
            )
        };
        let mut data = volume_constraint_data(vec![(1.0, outflow_volume()), (1.0, turbined)]);
        assert_eq!(messages(&data), [volume_refusal("hydro_turbined(1)")]);

        data.generic_constraint_named_sources
            .insert(EntityId::from(1), vec![source("f", vec![turbined])]);
        assert_eq!(messages(&data), [named("from @f")]);

        data.generic_constraint_named_sources.insert(
            EntityId::from(1),
            vec![
                source("f", vec![turbined]),
                source("g", vec![turbined]),
                source("h", vec![elsewhere]),
            ],
        );
        assert_eq!(messages(&data), [named("from @f, @g")]);
    }

    #[test]
    fn stage_volume_constraint_refuses_every_other_variable_form() {
        const NON_AMOUNT_NAMES: [&str; 21] = [
            "hydro_withdrawal",
            "hydro_evaporation",
            "hydro_inflow",
            "hydro_turbined",
            "hydro_spillage",
            "hydro_diversion",
            "hydro_outflow",
            "hydro_generation",
            "thermal_generation",
            "line_direct",
            "line_reverse",
            "line_exchange",
            "bus_deficit",
            "bus_excess",
            "pumping_flow",
            "pumping_power",
            "contract_import",
            "contract_export",
            "non_controllable_generation",
            "non_controllable_curtailment",
            "anticipated_decision",
        ];
        const BOUNDARY_NAMES: [&str; 4] = [
            "hydro_storage_initial",
            "hydro_storage_final",
            "hydro_useful_volume_initial",
            "hydro_useful_volume_final",
        ];
        let build = |name: &str, block: Option<usize>| {
            build_variable_ref(name, EntityId::from(1), block, None).unwrap()
        };
        let refused: Vec<VariableRef> = NON_AMOUNT_NAMES
            .iter()
            .map(|name| build(name, None))
            .chain(BOUNDARY_NAMES.iter().map(|name| build(name, Some(0))))
            .collect();

        for variable in refused {
            let call = variable_call(&variable);
            let data = volume_constraint_data(vec![(1.0, outflow_volume()), (1.0, variable)]);
            let messages: Vec<String> = volume_refusal_entries(&data)
                .into_iter()
                .map(|e| e.message)
                .collect();
            assert_eq!(messages, [volume_refusal(&call)], "{call}");
        }
    }

    #[test]
    fn stage_volume_constraint_accepts_volume_and_block_free_storage_terms() {
        let id = EntityId::from(1);
        for terms in [
            vec![
                (1.0, outflow_volume()),
                (1.0, VariableRef::HydroDiversionVolume { hydro_id: id }),
                (-1.0, VariableRef::HydroStorage { hydro_id: id }),
                (1.0, useful_initial(None)),
            ],
            vec![
                (1.0, VariableRef::PumpingVolume { station_id: id }),
                (1.0, initial(None)),
                (1.0, final_(None)),
                (1.0, useful_final(None)),
            ],
        ] {
            let data = volume_constraint_data(terms);
            assert!(data.generic_constraint_bounds.is_empty());
            let entries = volume_refusal_entries(&data);
            assert!(entries.is_empty(), "{entries:?}");
        }
    }

    #[test]
    fn constraint_without_stage_volume_term_is_not_checked() {
        let data = volume_constraint_data(vec![
            (
                1.0,
                VariableRef::HydroTurbined {
                    hydro_id: EntityId::from(1),
                    block_id: None,
                    bus_id: None,
                },
            ),
            (1.0, final_(Some(0))),
        ]);
        let entries = volume_refusal_entries(&data);
        assert!(entries.is_empty(), "{entries:?}");
    }

    fn per_block_refusal(parameter: &str) -> String {
        format!(
            "Constraint \"volume_constraint\": parameter `@{parameter}` is per_stage_block; \
             a constraint with a stage-volume term takes no coefficient or bound that \
             varies by block"
        )
    }

    fn scalar_param(id: i32, name: &str, kind: ParameterKind) -> ScalarParameter {
        ScalarParameter {
            id: EntityId::from(id),
            name: name.to_string(),
            kind,
        }
    }

    fn per_stage_block_param(id: i32, name: &str) -> ScalarParameter {
        scalar_param(
            id,
            name,
            ParameterKind::PerStageBlock {
                values: vec![(0, 0, 1.0)],
            },
        )
    }

    fn one_term_constraint_data(
        parameters: Vec<ScalarParameter>,
        coefficient: ScalarExpr,
        variable: VariableRef,
        bound_upper: Option<ScalarExpr>,
    ) -> ParsedData {
        make_data_with_constraint(
            parameters,
            GenericConstraint {
                id: EntityId::from(1),
                name: "volume_constraint".to_string(),
                description: None,
                expression: ConstraintExpression {
                    terms: vec![LinearTerm {
                        coefficient,
                        variable,
                    }],
                },
                slack: None,
                bound_lower: None,
                bound_upper,
            },
        )
    }

    #[test]
    fn stage_volume_constraint_refuses_a_per_stage_block_parameter() {
        let data = one_term_constraint_data(
            vec![per_stage_block_param(1, "k")],
            ScalarExpr::param(EntityId::from(1)),
            outflow_volume(),
            None,
        );
        let entries = volume_refusal_entries(&data);
        let messages: Vec<&str> = entries.iter().map(|e| e.message.as_str()).collect();
        assert_eq!(messages, [per_block_refusal("k")]);
        assert_eq!(entries[0].kind, ErrorKind::BusinessRuleViolation);
        assert_eq!(
            entries[0].file.to_string_lossy(),
            "constraints/generic_constraints.json"
        );
        assert_eq!(entries[0].entity.as_deref(), Some("constraint[id=1]"));
    }

    #[test]
    fn stage_volume_constraint_refuses_a_per_stage_block_parameter_once_however_reached() {
        let k = ScalarExpr::param(EntityId::from(1));
        let e = ScalarExpr::param(EntityId::from(2));
        let one = ScalarExpr::literal(1.0);
        for (case, coefficient, bound_upper) in [
            ("coefficient is an expression reading @k", e, None),
            ("bound_upper is @k", one, Some(k.clone())),
            (
                "coefficient and bound_upper both read @k",
                k.clone(),
                Some(k.clone()),
            ),
        ] {
            let data = one_term_constraint_data(
                vec![
                    per_stage_block_param(1, "k"),
                    expression_scalar_param(2, "e", k.clone() * ScalarExpr::literal(2.0)),
                ],
                coefficient,
                outflow_volume(),
                bound_upper,
            );
            let messages: Vec<String> = volume_refusal_entries(&data)
                .into_iter()
                .map(|entry| entry.message)
                .collect();
            assert_eq!(messages, [per_block_refusal("k")], "{case}");
        }
    }

    #[test]
    fn per_stage_parameter_in_a_stage_volume_constraint_is_accepted() {
        let k = ScalarExpr::param(EntityId::from(1));
        let per_stage = one_term_constraint_data(
            vec![scalar_param(
                1,
                "p",
                ParameterKind::PerStage { values: vec![1.0] },
            )],
            k.clone(),
            outflow_volume(),
            None,
        );
        let no_volume_term = one_term_constraint_data(
            vec![per_stage_block_param(1, "k")],
            k,
            VariableRef::HydroTurbined {
                hydro_id: EntityId::from(1),
                block_id: None,
                bus_id: None,
            },
            None,
        );
        for data in [per_stage, no_volume_term] {
            let entries = volume_refusal_entries(&data);
            assert!(entries.is_empty(), "{entries:?}");
        }
    }

    #[test]
    fn stage_volume_refusal_runs_in_the_semantic_layer() {
        let data = volume_constraint_data(vec![
            (1.0, outflow_volume()),
            (
                1.0,
                VariableRef::HydroTurbined {
                    hydro_id: EntityId::from(1),
                    block_id: Some(0),
                    bus_id: None,
                },
            ),
            (-1.0, final_(Some(0))),
        ]);
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        let messages: Vec<&str> = ctx
            .errors()
            .into_iter()
            .map(|e| e.message.as_str())
            .filter(|m| m.contains(VOLUME_REFUSAL_TAIL))
            .collect();
        assert_eq!(
            messages,
            [
                volume_refusal("hydro_turbined(1, 0)"),
                volume_refusal("hydro_storage_final(1, 0)"),
            ]
        );
    }

    fn bounds_row(
        stage_id: i32,
        block_id: Option<i32>,
        bound_lower: Option<f64>,
        bound_upper: Option<f64>,
    ) -> GenericConstraintBoundsRow {
        GenericConstraintBoundsRow {
            constraint_id: 1,
            stage_id,
            block_id,
            bound_lower,
            bound_upper,
        }
    }

    fn block_bound_refusal(stage_id: i32, block_id: i32) -> String {
        format!(
            "Constraint \"volume_constraint\": the bounds row at stage_id={stage_id} has \
             block_id={block_id}; a constraint with a stage-volume term takes one row per \
             stage with a null block_id"
        )
    }

    #[test]
    fn stage_volume_constraint_refuses_each_bounds_row_with_a_block_id() {
        let mut data = volume_constraint_data(vec![(1.0, outflow_volume())]);
        data.generic_constraint_bounds = vec![
            bounds_row(0, None, Some(9.0), Some(9.0)),
            bounds_row(0, Some(1), None, Some(9.0)),
            bounds_row(0, Some(2), Some(1.0), None),
        ];
        let entries = volume_refusal_entries(&data);
        let messages: Vec<&str> = entries.iter().map(|e| e.message.as_str()).collect();
        assert_eq!(
            messages,
            [block_bound_refusal(0, 1), block_bound_refusal(0, 2)]
        );
        for entry in &entries {
            assert_eq!(entry.kind, ErrorKind::BusinessRuleViolation);
            assert_eq!(
                entry.file.to_string_lossy(),
                "constraints/generic_constraint_bounds.parquet"
            );
            assert_eq!(entry.entity.as_deref(), Some("constraint_id=1, stage_id=0"));
        }
    }

    #[test]
    fn stage_volume_constraint_accepts_null_block_bounds_rows_of_every_shape() {
        let mut data = volume_constraint_data(vec![(1.0, outflow_volume())]);
        data.generic_constraint_bounds = vec![
            bounds_row(0, None, Some(1.0), None),
            bounds_row(1, None, None, Some(2.0)),
            bounds_row(2, None, Some(3.0), Some(3.0)),
        ];
        let entries = volume_refusal_entries(&data);
        assert!(entries.is_empty(), "{entries:?}");
    }

    #[test]
    fn block_bounds_row_of_a_constraint_without_a_stage_volume_term_is_accepted() {
        let mut data = volume_constraint_data(vec![(
            1.0,
            VariableRef::HydroTurbined {
                hydro_id: EntityId::from(1),
                block_id: None,
                bus_id: None,
            },
        )]);
        data.generic_constraint_bounds = vec![bounds_row(0, Some(0), None, Some(5.0))];
        let entries = volume_refusal_entries(&data);
        assert!(entries.is_empty(), "{entries:?}");
    }
}
