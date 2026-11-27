//! The types with which a constraint reads data that depends on the scenario: the data atoms and
//! plant capacities a scalar parameter stands for, the data-only conditions, the selectors an
//! `if()` lowers to, and the layer that carries them beside the constraints; the scan that finds
//! the atoms and capacities an expression reads and the ids allocated to them; and the fold that
//! moves the capacities a relation reads into its bounds.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use novomodelo_core::{EntityId, LinearTerm, ScalarExpr, VariableRef};
use serde::{Deserialize, Serialize};

use crate::expression::catalog::{LineBusPairIndex, data_atom, variable_from_call};
use crate::expression::lower::{CallClass, classify_call};
use crate::expression::parser::{Condition, Expr, ExprKind, VariableCall};

pub use crate::expression::catalog::DataAtom;
pub use crate::expression::parser::CompareOp;

/// The capacity of a plant that a scalar parameter stands for: `hydro_availability(h)` or, with a
/// block, `hydro_availability(h, k)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct AvailabilityAtom {
    /// The plant.
    pub hydro_id: EntityId,
    /// Block of the stage whose capacity is read; `None` for the block of the row.
    pub block_id: Option<usize>,
}

/// The value of an `if()`: `then` where the condition holds, `else_` elsewhere.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConditionalScalar {
    /// The test, over data atoms and parameters only.
    pub cond: ResolvedCondition,
    /// The value where the condition holds.
    pub then: ScalarExpr,
    /// The value where it does not.
    pub else_: ScalarExpr,
}

/// A data-only test: comparisons joined by `&` and `|`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ResolvedCondition {
    /// `lhs op rhs`.
    Compare {
        /// Left side.
        lhs: ScalarExpr,
        /// The relation tested.
        op: CompareOp,
        /// Right side.
        rhs: ScalarExpr,
    },
    /// Both hold.
    And(Box<ResolvedCondition>, Box<ResolvedCondition>),
    /// At least one holds.
    Or(Box<ResolvedCondition>, Box<ResolvedCondition>),
}

impl CompareOp {
    /// Whether `lhs` stands in this relation to `rhs`; `Eq` is exact equality.
    #[must_use]
    #[expect(clippy::float_cmp, reason = "the DSL's == compares exactly")]
    pub fn holds(self, lhs: f64, rhs: f64) -> bool {
        match self {
            Self::Lt => lhs < rhs,
            Self::Le => lhs <= rhs,
            Self::Gt => lhs > rhs,
            Self::Ge => lhs >= rhs,
            Self::Eq => lhs == rhs,
        }
    }
}

impl ResolvedCondition {
    /// Whether the condition holds, `value` giving the value of each side of a comparison.
    pub fn holds(&self, value: &mut impl FnMut(&ScalarExpr) -> f64) -> bool {
        match self {
            Self::Compare { lhs, op, rhs } => {
                let lhs = value(lhs);
                op.holds(lhs, value(rhs))
            }
            Self::And(left, right) => left.holds(value) && right.holds(value),
            Self::Or(left, right) => left.holds(value) || right.holds(value),
        }
    }

    /// Push both sides of every comparison of the condition onto `out`, left to right.
    pub fn operands<'a>(&'a self, out: &mut Vec<&'a ScalarExpr>) {
        match self {
            Self::Compare { lhs, rhs, .. } => out.extend([lhs, rhs]),
            Self::And(left, right) | Self::Or(left, right) => {
                left.operands(out);
                right.operands(out);
            }
        }
    }
}

/// What the generic constraints read per scenario, carried beside them: the scalar parameters
/// that stand for data atoms, capacities and `if()` selectors, and each constraint's condition and
/// endpoint shift.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ScenarioLayer {
    /// The parameter id of each data atom read, ascending.
    pub data_atoms: Vec<(EntityId, DataAtom)>,
    /// The parameter id of each capacity read, ascending; follows those of `data_atoms`.
    pub availability_atoms: Vec<(EntityId, AvailabilityAtom)>,
    /// The parameter of each `if()`, ascending by id.
    pub selectors: Vec<(EntityId, ConditionalScalar)>,
    /// One entry per constraint with a condition or a shift, ascending by id.
    pub constraints: Vec<ConstraintScenario>,
}

impl ScenarioLayer {
    pub(crate) fn new(
        data_ids: BTreeMap<DataAtom, EntityId>,
        capacity_ids: BTreeMap<AvailabilityAtom, EntityId>,
        mut selectors: Vec<(EntityId, ConditionalScalar)>,
        mut constraints: Vec<ConstraintScenario>,
    ) -> Self {
        let mut data_atoms: Vec<(EntityId, DataAtom)> =
            data_ids.into_iter().map(|(atom, id)| (id, atom)).collect();
        data_atoms.sort_unstable_by_key(|&(id, _)| id);
        let mut availability_atoms: Vec<(EntityId, AvailabilityAtom)> = capacity_ids
            .into_iter()
            .map(|(atom, id)| (id, atom))
            .collect();
        availability_atoms.sort_unstable_by_key(|&(id, _)| id);
        selectors.sort_unstable_by_key(|&(id, _)| id);
        constraints.sort_unstable_by_key(|constraint| constraint.constraint_id);
        Self {
            data_atoms,
            availability_atoms,
            selectors,
            constraints,
        }
    }

    /// Whether all four lists are empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data_atoms.is_empty()
            && self.availability_atoms.is_empty()
            && self.selectors.is_empty()
            && self.constraints.is_empty()
    }
}

/// The condition and endpoint shift of one constraint.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConstraintScenario {
    /// The constraint.
    pub constraint_id: EntityId,
    /// The `active_when` condition; `None` when the constraint always holds.
    pub active_when: Option<ResolvedCondition>,
    /// Added to each endpoint the bounds table gives to an expression without an operator.
    pub endpoint_shift: Option<ScalarExpr>,
}

/// The data atoms and plant capacities read by the expressions and conditions visited.
pub(crate) struct AtomScan<'a> {
    line_index: &'a LineBusPairIndex,
    pub(crate) atoms: BTreeSet<DataAtom>,
    pub(crate) capacities: BTreeSet<AvailabilityAtom>,
}

impl<'a> AtomScan<'a> {
    pub(crate) fn new(line_index: &'a LineBusPairIndex) -> Self {
        Self {
            line_index,
            atoms: BTreeSet::new(),
            capacities: BTreeSet::new(),
        }
    }

    pub(crate) fn visit_expr(&mut self, expr: &Expr) {
        for_each_call(expr, &mut |call| self.visit(call));
    }

    pub(crate) fn visit_condition(&mut self, condition: &Condition) {
        for_each_condition_call(condition, &mut |call| self.visit(call));
    }

    /// Records the data atom or plant capacity `call` reads.
    fn visit(&mut self, call: &VariableCall) {
        match classify_call(call) {
            CallClass::Data => {
                if let Ok(atom) = data_atom(call) {
                    self.atoms.insert(atom);
                }
            }
            CallClass::Variable => {
                if let Ok((VariableRef::HydroAvailability { hydro_id, block_id }, _)) =
                    variable_from_call(call, self.line_index)
                {
                    self.capacities
                        .insert(AvailabilityAtom { hydro_id, block_id });
                }
            }
        }
    }
}

/// Calls `f` on every variable call under `expr`.
pub(crate) fn for_each_call(expr: &Expr, f: &mut impl FnMut(&VariableCall)) {
    match &expr.kind {
        ExprKind::Number(_) | ExprKind::Name(_) => {}
        ExprKind::Variable(call) => f(call),
        ExprKind::Neg(operand) => for_each_call(operand, f),
        ExprKind::Sum { head, tail } => {
            for_each_call(head, f);
            for (_, operand) in tail {
                for_each_call(operand, f);
            }
        }
        ExprKind::Product { head, tail } => {
            for_each_call(head, f);
            for (_, operand) in tail {
                for_each_call(operand, f);
            }
        }
        ExprKind::Conditional {
            cond,
            then_br,
            else_br,
        } => {
            for_each_condition_call(cond, f);
            for_each_call(then_br, f);
            for_each_call(else_br, f);
        }
    }
}

pub(crate) fn for_each_condition_call(condition: &Condition, f: &mut impl FnMut(&VariableCall)) {
    match condition {
        Condition::Compare { lhs, rhs, .. } => {
            for_each_call(lhs, f);
            for_each_call(rhs, f);
        }
        Condition::And(left, right) | Condition::Or(left, right) => {
            for_each_condition_call(left, f);
            for_each_condition_call(right, f);
        }
    }
}

/// Gives each data atom and then each plant capacity, in key order, the next id above every
/// scalar-parameter id, and returns the first id after them, where the selectors start; `None`
/// when the ids run out.
pub(crate) fn allocate_ids(
    params: &HashMap<String, EntityId>,
    atoms: BTreeSet<DataAtom>,
    capacities: BTreeSet<AvailabilityAtom>,
) -> Option<Ids> {
    let first = params
        .values()
        .map(|id| id.0)
        .max()
        .map_or(Some(0), |max| max.checked_add(1))?;
    let (data, first) = assign_ids(atoms, first)?;
    let (capacities, first_selector) = assign_ids(capacities, first)?;
    Some(Ids {
        data,
        capacities,
        first_selector,
    })
}

pub(crate) struct Ids {
    pub(crate) data: BTreeMap<DataAtom, EntityId>,
    pub(crate) capacities: BTreeMap<AvailabilityAtom, EntityId>,
    pub(crate) first_selector: i32,
}

/// The terms of `terms` that are not a plant capacity; each capacity term is subtracted from each
/// bound present as its coefficient times the parameter `capacity_ids` gives it. `terms` must be
/// canonical, so that a plant read twice subtracts one summed coefficient.
///
/// # Errors
///
/// The message for the first capacity `capacity_ids` gives no parameter id.
pub(crate) fn fold_availability(
    terms: Vec<LinearTerm>,
    lower: &mut Option<ScalarExpr>,
    upper: &mut Option<ScalarExpr>,
    capacity_ids: &BTreeMap<AvailabilityAtom, EntityId>,
) -> Result<Vec<LinearTerm>, String> {
    let mut kept = Vec::with_capacity(terms.len());
    for term in terms {
        let VariableRef::HydroAvailability { hydro_id, block_id } = term.variable else {
            kept.push(term);
            continue;
        };
        let id = capacity_ids
            .get(&AvailabilityAtom { hydro_id, block_id })
            .ok_or_else(|| format!("the availability of hydro {hydro_id} has no parameter id"))?;
        let capacity = term.coefficient * ScalarExpr::param(*id);
        *lower = lower.take().map(|bound| bound - capacity.clone());
        *upper = upper.take().map(|bound| bound - capacity);
    }
    Ok(kept)
}

pub(crate) fn assign_ids<K: Ord>(
    keys: BTreeSet<K>,
    first: i32,
) -> Option<(BTreeMap<K, EntityId>, i32)> {
    let after = first.checked_add(i32::try_from(keys.len()).ok()?)?;
    let ids = keys
        .into_iter()
        .zip(first..)
        .map(|(key, id)| (key, EntityId(id)))
        .collect();
    Some((ids, after))
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::expression::parser::parse;

    fn lit(value: f64) -> ScalarExpr {
        ScalarExpr::literal(value)
    }

    fn param(id: i32) -> ScalarExpr {
        ScalarExpr::param(EntityId(id))
    }

    fn compare(lhs: ScalarExpr, op: CompareOp, rhs: ScalarExpr) -> ResolvedCondition {
        ResolvedCondition::Compare { lhs, op, rhs }
    }

    fn round_trip(layer: &ScenarioLayer) -> ScenarioLayer {
        let bytes = postcard::to_allocvec(layer).expect("the layer serializes");
        postcard::from_bytes(&bytes).expect("the layer deserializes")
    }

    #[test]
    fn an_empty_layer_is_empty_and_round_trips() {
        let layer = ScenarioLayer::default();
        assert!(layer.is_empty());
        assert_eq!(round_trip(&layer), layer);
    }

    #[test]
    fn a_layer_with_any_list_filled_is_not_empty_and_round_trips() {
        let layers = [
            ScenarioLayer {
                constraints: vec![ConstraintScenario {
                    constraint_id: EntityId(7),
                    active_when: None,
                    endpoint_shift: Some(lit(2.0)),
                }],
                ..ScenarioLayer::default()
            },
            ScenarioLayer {
                data_atoms: vec![(
                    EntityId(1),
                    DataAtom::BusLoadValue {
                        bus_id: EntityId(3),
                        block_id: None,
                    },
                )],
                ..ScenarioLayer::default()
            },
            ScenarioLayer {
                availability_atoms: vec![(
                    EntityId(2),
                    AvailabilityAtom {
                        hydro_id: EntityId(4),
                        block_id: Some(1),
                    },
                )],
                ..ScenarioLayer::default()
            },
            ScenarioLayer {
                selectors: vec![(
                    EntityId(3),
                    ConditionalScalar {
                        cond: compare(param(1), CompareOp::Gt, lit(10.0)),
                        then: lit(1.0),
                        else_: lit(2.0),
                    },
                )],
                ..ScenarioLayer::default()
            },
        ];
        for layer in layers {
            assert!(!layer.is_empty(), "{layer:?}");
            assert_eq!(round_trip(&layer), layer);
        }
    }

    /// derived: `82000 < x & x <= 95000` holds for `x` in `(82000, 95000]`, and
    /// `x <= 85000 | y <= 3000` wherever either side does.
    #[test]
    fn a_condition_holds_by_its_comparisons_joined_by_and_and_or() {
        let range = ResolvedCondition::And(
            Box::new(compare(lit(82000.0), CompareOp::Lt, param(8))),
            Box::new(compare(param(8), CompareOp::Le, lit(95000.0))),
        );
        let either = ResolvedCondition::Or(
            Box::new(compare(param(8), CompareOp::Le, lit(85000.0))),
            Box::new(compare(param(9), CompareOp::Le, lit(3000.0))),
        );
        for (x, y, in_range, either_holds) in [
            (82000.0, 3001.0, false, true),
            (82001.0, 3001.0, true, true),
            (95000.0, 3000.0, true, true),
            (95000.0, 3001.0, true, false),
            (95001.0, 3000.0, false, true),
            (95001.0, 3001.0, false, false),
        ] {
            let mut value =
                |side: &ScalarExpr| side.evaluate(|id| if id == EntityId(8) { x } else { y });
            assert_eq!(
                (range.holds(&mut value), either.holds(&mut value)),
                (in_range, either_holds),
                "x = {x}, y = {y}"
            );
        }
        for (op, holds) in [
            (CompareOp::Lt, false),
            (CompareOp::Le, true),
            (CompareOp::Gt, false),
            (CompareOp::Ge, true),
            (CompareOp::Eq, true),
        ] {
            assert_eq!(op.holds(1.0, 1.0), holds, "{op:?}");
        }
        assert!(!CompareOp::Eq.holds(1.0, 1.0 + f64::EPSILON));
    }

    /// derived: with the capacities at 100 and 40, each bound moves by `-(2 · 100) - (-3 · 40)`,
    /// so 10 becomes -70 and 20 becomes -60.
    #[test]
    fn availability_folds_out_of_the_terms_into_both_bounds() {
        let availability = |block_id| VariableRef::HydroAvailability {
            hydro_id: EntityId(7),
            block_id,
        };
        let generation = LinearTerm::literal(
            1.0,
            VariableRef::HydroGeneration {
                hydro_id: EntityId(7),
                block_id: None,
                bus_id: None,
            },
        );
        let terms = vec![
            generation.clone(),
            LinearTerm::literal(2.0, availability(None)),
            LinearTerm::literal(-3.0, availability(Some(1))),
        ];
        let capacity_ids = BTreeMap::from([
            (
                AvailabilityAtom {
                    hydro_id: EntityId(7),
                    block_id: None,
                },
                EntityId(5),
            ),
            (
                AvailabilityAtom {
                    hydro_id: EntityId(7),
                    block_id: Some(1),
                },
                EntityId(6),
            ),
        ]);
        let (mut lower, mut upper) = (Some(lit(10.0)), Some(lit(20.0)));

        let kept = fold_availability(terms.clone(), &mut lower, &mut upper, &capacity_ids)
            .expect("every capacity has an id");

        assert_eq!(kept, [generation]);
        let capacity = |id: EntityId| match id.0 {
            5 => 100.0,
            6 => 40.0,
            other => panic!("no value for parameter {other}"),
        };
        let evaluate =
            |bound: Option<ScalarExpr>| bound.expect("the bound stays").evaluate(capacity);
        assert_eq!(evaluate(lower).to_bits(), (-70.0_f64).to_bits());
        assert_eq!(evaluate(upper).to_bits(), (-60.0_f64).to_bits());

        let (mut lower, mut upper) = (None, Some(lit(20.0)));
        assert_eq!(
            fold_availability(terms, &mut lower, &mut upper, &BTreeMap::new()),
            Err("the availability of hydro 7 has no parameter id".to_owned())
        );
    }

    #[test]
    fn an_atom_scan_collects_data_atoms_and_capacities() {
        let relation = parse(
            "hydro_generation(1) - bus_load_value(3) \
             + if(bus_load_value(3, 1) > 10, 1, 2) * hydro_availability(7)",
        )
        .expect("the expression parses");
        let line_index = LineBusPairIndex::default();
        let mut scan = AtomScan::new(&line_index);
        scan.visit_expr(&relation.lhs);
        assert_eq!(
            scan.atoms,
            BTreeSet::from([
                DataAtom::BusLoadValue {
                    bus_id: EntityId(3),
                    block_id: None,
                },
                DataAtom::BusLoadValue {
                    bus_id: EntityId(3),
                    block_id: Some(1),
                },
            ])
        );
        assert_eq!(
            scan.capacities,
            BTreeSet::from([AvailabilityAtom {
                hydro_id: EntityId(7),
                block_id: None,
            }])
        );
    }
}
