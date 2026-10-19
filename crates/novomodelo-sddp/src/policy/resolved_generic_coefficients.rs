//! The one owner of every generic-constraint term value and bound remainder per
//! study stage and block, built once at setup from [`ResolvedParameters`]. The
//! LP builder and the generic-constraint echo read numbers from it and resolve
//! nothing themselves.

use novomodelo_core::{GenericConstraint, ScalarExpr, StageId};
use thiserror::Error;

use crate::generic_constraint_echo::render_variable;
use crate::policy::resolved_parameters::ResolvedParameters;

/// One resolved number per study stage, or per block of every study stage when
/// `block_varying` (then indexed through the owner's `block_offsets`).
#[derive(Debug, Default)]
pub(crate) struct ResolvedSeries {
    block_varying: bool,
    values: Vec<f64>,
}

impl ResolvedSeries {
    fn value(&self, block_offsets: &[usize], stage_idx: usize, block_idx: usize) -> f64 {
        if self.block_varying {
            self.values[block_offsets[stage_idx]..block_offsets[stage_idx + 1]][block_idx]
        } else {
            self.values[stage_idx]
        }
    }
}

/// An absent bound side is `None`, never a zero series.
#[derive(Debug)]
struct ResolvedConstraint {
    terms: Vec<ResolvedSeries>,
    lower: Option<ResolvedSeries>,
    upper: Option<ResolvedSeries>,
}

/// Resolved term values and bound remainders of every generic constraint, in
/// the system's constraint order.
#[derive(Debug, Default)]
pub(crate) struct ResolvedGenericCoefficients {
    constraints: Vec<ResolvedConstraint>,
    /// `block_offsets[s]..block_offsets[s + 1]` is study stage `s`'s range in a
    /// block-varying series.
    block_offsets: Vec<usize>,
}

/// Which bound remainder of a generic constraint to read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BoundSide {
    Lower,
    Upper,
}

/// A generic-constraint coefficient or bound that cannot be resolved.
#[derive(Debug, Error)]
pub(crate) enum ResolvedGenericCoefficientsError {
    #[error(
        "generic constraint {constraint_id} '{constraint_name}' references scalar parameter id={parameter_id} not present in the resolved parameter table"
    )]
    UnresolvedParameter {
        constraint_id: i32,
        constraint_name: String,
        parameter_id: i32,
    },

    #[error(
        "generic constraint {constraint_id} '{constraint_name}': {location} resolves to the non-finite value {value} at stage {stage_id}, block {block}"
    )]
    NonFinite {
        constraint_id: i32,
        constraint_name: String,
        location: String,
        value: f64,
        stage_id: i32,
        block: usize,
    },
}

struct NonFiniteCell {
    value: f64,
    stage_idx: usize,
    block: usize,
}

struct StudyGrid<'a> {
    stage_ids: &'a [StageId],
    block_counts: &'a [usize],
    n_cells: usize,
}

impl ResolvedGenericCoefficients {
    /// Resolve every term and present bound side of `constraints` over every
    /// study stage and block.
    ///
    /// # Errors
    ///
    /// [`ResolvedGenericCoefficientsError::UnresolvedParameter`] when a
    /// constraint reads a parameter id `resolved` does not hold, and
    /// [`ResolvedGenericCoefficientsError::NonFinite`] on the first non-finite
    /// cell, whether or not the constraint is active at that stage.
    pub(crate) fn build(
        constraints: &[GenericConstraint],
        resolved: &ResolvedParameters,
        study_stage_ids: &[StageId],
        stage_block_counts: &[usize],
    ) -> Result<Self, ResolvedGenericCoefficientsError> {
        debug_assert_eq!(study_stage_ids.len(), stage_block_counts.len());
        let mut block_offsets = Vec::with_capacity(stage_block_counts.len() + 1);
        let mut n_cells = 0;
        block_offsets.push(n_cells);
        for &count in stage_block_counts {
            n_cells += count;
            block_offsets.push(n_cells);
        }
        let grid = StudyGrid {
            stage_ids: study_stage_ids,
            block_counts: stage_block_counts,
            n_cells,
        };
        let constraints = constraints
            .iter()
            .map(|constraint| resolve_constraint(constraint, resolved, &grid))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            constraints,
            block_offsets,
        })
    }

    /// The LP coefficient of term `term_idx` of constraint `constraint_idx` at
    /// `(stage_idx, block_idx)`.
    #[must_use]
    pub(crate) fn term_value(
        &self,
        constraint_idx: usize,
        term_idx: usize,
        stage_idx: usize,
        block_idx: usize,
    ) -> f64 {
        self.constraints[constraint_idx].terms[term_idx].value(
            &self.block_offsets,
            stage_idx,
            block_idx,
        )
    }

    /// The resolved remainder of bound `side` of constraint `constraint_idx` at
    /// `(stage_idx, block_idx)`; `None` when that side is absent.
    #[must_use]
    pub(crate) fn bound_value(
        &self,
        constraint_idx: usize,
        side: BoundSide,
        stage_idx: usize,
        block_idx: usize,
    ) -> Option<f64> {
        let constraint = &self.constraints[constraint_idx];
        let series = match side {
            BoundSide::Lower => constraint.lower.as_ref(),
            BoundSide::Upper => constraint.upper.as_ref(),
        };
        series.map(|series| series.value(&self.block_offsets, stage_idx, block_idx))
    }

    /// Whether any term or bound series of constraint `constraint_idx` varies
    /// by block.
    #[must_use]
    pub(crate) fn is_block_varying(&self, constraint_idx: usize) -> bool {
        let constraint = &self.constraints[constraint_idx];
        constraint
            .terms
            .iter()
            .chain(constraint.lower.as_ref())
            .chain(constraint.upper.as_ref())
            .any(|series| series.block_varying)
    }
}

fn resolve_constraint(
    constraint: &GenericConstraint,
    resolved: &ResolvedParameters,
    grid: &StudyGrid<'_>,
) -> Result<ResolvedConstraint, ResolvedGenericCoefficientsError> {
    // Before any `get`: it asserts on an unknown id in debug builds.
    if let Some(id) = constraint
        .parameter_ids()
        .find(|&id| !resolved.contains(id))
    {
        return Err(ResolvedGenericCoefficientsError::UnresolvedParameter {
            constraint_id: constraint.id.0,
            constraint_name: constraint.name.clone(),
            parameter_id: id.0,
        });
    }
    let terms = constraint
        .expression
        .terms
        .iter()
        .map(|term| {
            let block_varying = term
                .coefficient
                .params()
                .any(|id| resolved.is_block_varying(id));
            resolve_series(grid, block_varying, |s, b| {
                term.coefficient.evaluate(|id| resolved.get(id, s, b))
            })
            .map_err(|cell| {
                let (_, label) = render_variable(&term.variable);
                non_finite(
                    constraint,
                    format!("the coefficient of {label}"),
                    grid,
                    &cell,
                )
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let lower = resolve_bound(
        constraint,
        constraint.bound_lower.as_ref(),
        "lower",
        resolved,
        grid,
    )?;
    let upper = resolve_bound(
        constraint,
        constraint.bound_upper.as_ref(),
        "upper",
        resolved,
        grid,
    )?;
    Ok(ResolvedConstraint {
        terms,
        lower,
        upper,
    })
}

fn resolve_bound(
    constraint: &GenericConstraint,
    bound: Option<&ScalarExpr>,
    side: &str,
    resolved: &ResolvedParameters,
    grid: &StudyGrid<'_>,
) -> Result<Option<ResolvedSeries>, ResolvedGenericCoefficientsError> {
    let Some(bound) = bound else {
        return Ok(None);
    };
    let block_varying = bound.params().any(|id| resolved.is_block_varying(id));
    resolve_series(grid, block_varying, |s, b| {
        bound.evaluate(|id| resolved.get(id, s, b))
    })
    .map(Some)
    .map_err(|cell| non_finite(constraint, format!("the {side} bound"), grid, &cell))
}

fn resolve_series(
    grid: &StudyGrid<'_>,
    block_varying: bool,
    cell: impl Fn(usize, usize) -> f64,
) -> Result<ResolvedSeries, NonFiniteCell> {
    let mut values = Vec::with_capacity(if block_varying {
        grid.n_cells
    } else {
        grid.block_counts.len()
    });
    for (stage_idx, &n_blocks) in grid.block_counts.iter().enumerate() {
        let n_evaluated = if block_varying { n_blocks } else { 1 };
        for block in 0..n_evaluated {
            let value = cell(stage_idx, block);
            if !value.is_finite() {
                return Err(NonFiniteCell {
                    value,
                    stage_idx,
                    block,
                });
            }
            values.push(value);
        }
    }
    Ok(ResolvedSeries {
        block_varying,
        values,
    })
}

fn non_finite(
    constraint: &GenericConstraint,
    location: String,
    grid: &StudyGrid<'_>,
    cell: &NonFiniteCell,
) -> ResolvedGenericCoefficientsError {
    ResolvedGenericCoefficientsError::NonFinite {
        constraint_id: constraint.id.0,
        constraint_name: constraint.name.clone(),
        location,
        value: cell.value,
        stage_id: grid.stage_ids[cell.stage_idx].0,
        block: cell.block,
    }
}

#[cfg(test)]
#[allow(clippy::panic, clippy::unwrap_used)]
mod tests {
    use novomodelo_core::{
        ConstraintExpression, EntityId, GenericConstraint, LinearTerm, ScalarExpr, StageId,
        VariableRef,
    };

    use super::{BoundSide, ResolvedGenericCoefficients, ResolvedGenericCoefficientsError};
    use crate::policy::resolved_parameters::ResolvedParameters;

    fn hydro_storage(id: i32) -> VariableRef {
        VariableRef::HydroStorage {
            hydro_id: EntityId(id),
        }
    }

    fn lit(value: f64) -> ScalarExpr {
        ScalarExpr::literal(value)
    }

    fn param(id: i32) -> ScalarExpr {
        ScalarExpr::param(EntityId(id))
    }

    fn term(coefficient: ScalarExpr, variable: VariableRef) -> LinearTerm {
        LinearTerm {
            coefficient,
            variable,
        }
    }

    fn constraint(terms: Vec<LinearTerm>) -> GenericConstraint {
        GenericConstraint {
            id: EntityId(500),
            name: "probe".to_string(),
            description: None,
            expression: ConstraintExpression { terms },
            slack: None,
            bound_lower: None,
            bound_upper: None,
        }
    }

    fn table(per_param: Vec<Vec<Vec<f64>>>, ids: &[i32]) -> ResolvedParameters {
        ResolvedParameters {
            id_to_slot: ids
                .iter()
                .enumerate()
                .map(|(slot, &id)| (id, slot))
                .collect(),
            per_param,
            ..Default::default()
        }
    }

    const TWO_STAGES: [StageId; 2] = [StageId(3), StageId(4)];

    /// Slot 42 stores two block values at the one study stage (block-varying);
    /// slot 43 stores a length-1 inner (block-invariant broadcast).
    fn bound_fold_table() -> ResolvedParameters {
        table(vec![vec![vec![1.0, 2.0]], vec![vec![5.0]]], &[42, 43])
    }

    fn lower_bound_at_block_1(bound: ScalarExpr) -> f64 {
        let mut c = constraint(vec![LinearTerm::literal(1.0, hydro_storage(1))]);
        c.bound_lower = Some(bound);
        let rgc =
            ResolvedGenericCoefficients::build(&[c], &bound_fold_table(), &[StageId(0)], &[2])
                .unwrap();
        rgc.bound_value(0, BoundSide::Lower, 0, 1).unwrap()
    }

    /// derived: the literal term is `0.87 * 2.0` at every stage; the parameter
    /// term is each stage's value times `-1.0`, which is its negation.
    #[test]
    fn literal_and_parameter_terms_store_one_value_per_stage() {
        let r = table(vec![vec![vec![3.0], vec![4.0]]], &[5]);
        let c = constraint(vec![
            term(lit(0.87) * lit(2.0), hydro_storage(1)),
            term(param(5) * lit(-1.0), hydro_storage(2)),
        ]);
        let rgc = ResolvedGenericCoefficients::build(&[c], &r, &TWO_STAGES, &[2, 2]).unwrap();
        assert!(!rgc.is_block_varying(0));
        for (stage, param) in [(0, 3.0_f64), (1, 4.0)] {
            assert_eq!(rgc.constraints[0].terms[0].values.len(), 2);
            for block in 0..2 {
                assert_eq!(
                    rgc.term_value(0, 0, stage, block).to_bits(),
                    (0.87_f64 * 2.0).to_bits()
                );
                assert_eq!(
                    rgc.term_value(0, 1, stage, block).to_bits(),
                    (-param).to_bits()
                );
            }
        }
    }

    /// derived: every `(stage, block)` cell is that cell's parameter value times
    /// the term's `1.5` factor.
    #[test]
    fn a_block_varying_parameter_term_stores_every_block() {
        let r = table(vec![vec![vec![1.0, 2.0], vec![3.0, 4.0]]], &[6]);
        let c = constraint(vec![term(param(6) * lit(1.5), hydro_storage(1))]);
        let rgc = ResolvedGenericCoefficients::build(&[c], &r, &TWO_STAGES, &[2, 2]).unwrap();
        assert!(rgc.is_block_varying(0));
        for stage in 0..2 {
            for block in 0..2 {
                assert_eq!(
                    rgc.term_value(0, 0, stage, block).to_bits(),
                    (r.get(EntityId(6), stage, block) * 1.5).to_bits()
                );
            }
        }
    }

    /// derived: a bare parameter evaluates to its resolved value.
    #[test]
    fn bound_of_a_single_parameter_equals_its_value() {
        let r = bound_fold_table();
        let value = lower_bound_at_block_1(param(42));
        assert_eq!(value.to_bits(), r.get(EntityId(42), 0, 1).to_bits());
    }

    /// derived: `100 + 2 * p42 - 1 * p43` at block 1.
    #[test]
    fn bound_sums_constant_then_terms() {
        let r = bound_fold_table();
        let value =
            lower_bound_at_block_1(lit(100.0) + lit(2.0) * param(42) - lit(1.0) * param(43));
        let expected = 100.0 + 2.0 * r.get(EntityId(42), 0, 1) - 1.0 * r.get(EntityId(43), 0, 1);
        assert_eq!(value.to_bits(), expected.to_bits());
    }

    /// derived: `p42 * p43` at block 1.
    #[test]
    fn bound_of_a_product_only() {
        let r = bound_fold_table();
        let value = lower_bound_at_block_1(param(42) * param(43));
        let expected = r.get(EntityId(42), 0, 1) * r.get(EntityId(43), 0, 1);
        assert_eq!(value.to_bits(), expected.to_bits());
    }

    /// derived: `2.5 * p42 * p43` at block 1.
    #[test]
    fn bound_of_a_scaled_product() {
        let r = bound_fold_table();
        let value = lower_bound_at_block_1(lit(2.5) * param(42) * param(43));
        let expected = 2.5 * r.get(EntityId(42), 0, 1) * r.get(EntityId(43), 0, 1);
        assert_eq!(value.to_bits(), expected.to_bits());
    }

    /// derived: `(5 + 2 * p42) + 0.5 * p42 * p43` at block 1.
    #[test]
    fn bound_mixes_terms_then_products() {
        let r = bound_fold_table();
        let value = lower_bound_at_block_1(
            lit(5.0) + lit(2.0) * param(42) + lit(0.5) * param(42) * param(43),
        );
        let linear = 5.0 + 2.0 * r.get(EntityId(42), 0, 1);
        let product = 0.5 * r.get(EntityId(42), 0, 1) * r.get(EntityId(43), 0, 1);
        assert_eq!(value.to_bits(), (linear + product).to_bits());
    }

    /// derived: the lower bound is parameter 43's only value, `5.0`.
    #[test]
    fn an_absent_bound_side_has_no_value() {
        let mut c = constraint(vec![LinearTerm::literal(1.0, hydro_storage(1))]);
        c.bound_lower = Some(param(43));
        let rgc =
            ResolvedGenericCoefficients::build(&[c], &bound_fold_table(), &[StageId(0)], &[2])
                .unwrap();
        assert_eq!(rgc.bound_value(0, BoundSide::Upper, 0, 0), None);
        assert_eq!(
            rgc.bound_value(0, BoundSide::Lower, 0, 0).map(f64::to_bits),
            Some(5.0_f64.to_bits())
        );
    }

    #[test]
    fn block_variation_follows_term_and_bound_leaves() {
        let literal = || vec![LinearTerm::literal(1.0, hydro_storage(1))];
        let varying_term = constraint(vec![term(param(42), hydro_storage(1))]);
        let mut varying_lower = constraint(literal());
        varying_lower.bound_lower = Some(param(42));
        let mut varying_product = constraint(literal());
        varying_product.bound_upper = Some(param(43) + param(43) * param(42));
        let mut invariant = constraint(vec![term(param(43), hydro_storage(1))]);
        invariant.bound_lower = Some(param(43));
        invariant.bound_upper = Some(lit(1.0) + lit(2.0) * param(43) + param(43) * param(43));
        let rgc = ResolvedGenericCoefficients::build(
            &[varying_term, varying_lower, varying_product, invariant],
            &bound_fold_table(),
            &[StageId(0)],
            &[2],
        )
        .unwrap();
        assert!(rgc.is_block_varying(0));
        assert!(rgc.is_block_varying(1));
        assert!(rgc.is_block_varying(2));
        assert!(!rgc.is_block_varying(3));
    }

    #[test]
    fn an_unresolved_parameter_names_the_constraint_and_id() {
        let mut c = constraint(vec![LinearTerm::literal(1.0, hydro_storage(1))]);
        c.bound_lower = Some(param(999));
        let err =
            ResolvedGenericCoefficients::build(&[c], &bound_fold_table(), &[StageId(0)], &[2])
                .unwrap_err();
        assert!(matches!(
            err,
            ResolvedGenericCoefficientsError::UnresolvedParameter {
                constraint_id: 500,
                parameter_id: 999,
                ..
            }
        ));
        let message = err.to_string();
        for needle in ["500", "'probe'", "id=999"] {
            assert!(message.contains(needle), "{needle} missing from: {message}");
        }
    }

    #[test]
    fn a_non_finite_literal_names_the_variable_stage_and_block() {
        let c = constraint(vec![LinearTerm::literal(f64::INFINITY, hydro_storage(1))]);
        let err =
            ResolvedGenericCoefficients::build(&[c], &table(vec![], &[]), &TWO_STAGES, &[2, 2])
                .unwrap_err();
        assert!(matches!(
            err,
            ResolvedGenericCoefficientsError::NonFinite {
                stage_id: 3,
                block: 0,
                ..
            }
        ));
        let message = err.to_string();
        for needle in ["the coefficient of hydro_storage(", "stage 3", "block 0"] {
            assert!(message.contains(needle), "{needle} missing from: {message}");
        }
    }

    #[test]
    fn a_non_finite_parameter_at_any_study_stage_is_an_error() {
        let r = table(vec![vec![vec![1.0], vec![f64::NAN]]], &[7]);
        let mut c = constraint(vec![LinearTerm::literal(1.0, hydro_storage(1))]);
        c.bound_lower = Some(param(7));
        let err = ResolvedGenericCoefficients::build(&[c], &r, &TWO_STAGES, &[1, 1]).unwrap_err();
        assert!(matches!(
            err,
            ResolvedGenericCoefficientsError::NonFinite { stage_id: 4, .. }
        ));
        let message = err.to_string();
        for needle in ["the lower bound", "stage 4"] {
            assert!(message.contains(needle), "{needle} missing from: {message}");
        }
    }
}
