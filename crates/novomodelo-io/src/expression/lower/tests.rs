#![allow(clippy::panic, clippy::unwrap_used)]

use std::collections::{BTreeMap, HashMap};

use novomodelo_core::{EntityId, Line, ScalarExpr, VariableRef};

use super::{
    Fault, LinearForm, LoweredRelation, Lowering, NamedTable, Scope, Slot,
    collect_named_references, lower_activation_condition, lower_relation, lower_relation_from,
    lower_scalar_body,
};
use crate::constraints::scenario::{CompareOp, ConditionalScalar, ResolvedCondition};
use crate::expression::Diagnostic;
use crate::expression::catalog::{DataAtom, LineBusPairIndex, build_line_bus_pair_index};
use crate::expression::parser::{Expr, parse, parse_condition};
use crate::test_support::make_line;

fn params() -> HashMap<String, EntityId> {
    HashMap::from([
        ("a".to_owned(), EntityId(7)),
        ("b".to_owned(), EntityId(8)),
        ("c".to_owned(), EntityId(9)),
        ("p".to_owned(), EntityId(10)),
        ("q".to_owned(), EntityId(11)),
    ])
}

fn lowered(src: &str) -> ScalarExpr {
    lower_scalar_body(src, &params()).unwrap()
}

fn rendered_rejection(src: &str) -> String {
    lower_scalar_body(src, &params()).unwrap_err().render(src)
}

fn value_at(expr: &ScalarExpr, a: f64, b: f64) -> f64 {
    expr.evaluate(|id| match id {
        EntityId(7) => a,
        EntityId(8) => b,
        _ => panic!("unexpected parameter id {id:?}"),
    })
}

/// derived: `2 * 3 + 1` is `(2.0 * 3.0) + 1.0 = 7.0`, one literal; `-(4 / 2)` is
/// `-(4.0 / 2.0) = -2.0`.
#[test]
fn scalar_body_folds_literal_only_subexpressions() {
    let expr = lowered("2 * 3 + 1");
    assert_eq!(expr.as_literal().map(f64::to_bits), Some(7.0_f64.to_bits()));
    assert_eq!(expr.op_count(), 1);

    let expr = lowered("-(4 / 2)");
    assert_eq!(
        expr.as_literal().map(f64::to_bits),
        Some((-2.0_f64).to_bits())
    );
}

/// derived: `2 * @a * 3` is the program `2, a, mul, 3, mul` (5 ops) and evaluates at
/// `a = 0.1` to `(2.0 * 0.1) * 3.0`; `2 * (@a * 3)` is `2, a, 3, mul, mul` and evaluates to
/// `2.0 * (0.1 * 3.0)`; `@a + 0.1 + 0.2` at `a = 3.0` is `(3.0 + 0.1) + 0.2 =
/// 3.3000000000000003`, which a fold of `0.1 + 0.2` would turn into `3.3`; `@a + @b * 2` is
/// the program `a, b, 2, mul, add` (5 ops) and at `a = 1.0`, `b = 3.0` is `1.0 + 3.0 * 2.0 =
/// 7.0`.
#[test]
fn scalar_body_keeps_the_authored_association() {
    let left = lowered("2 * @a * 3");
    assert_eq!(left.op_count(), 5);
    assert_eq!(left, (lit(2.0) * a()) * lit(3.0));
    assert_eq!(
        value_at(&left, 0.1, 0.0).to_bits(),
        ((2.0_f64 * 0.1) * 3.0).to_bits()
    );

    let right = lowered("2 * (@a * 3)");
    assert_eq!(right.op_count(), 5);
    assert_eq!(right, lit(2.0) * (a() * lit(3.0)));
    assert_ne!(left, right);
    assert_eq!(
        value_at(&right, 0.1, 0.0).to_bits(),
        (2.0_f64 * (0.1 * 3.0)).to_bits()
    );

    let chain = lowered("@a + 0.1 + 0.2");
    assert_eq!(chain.op_count(), 5);
    assert_eq!(
        value_at(&chain, 3.0, 0.0).to_bits(),
        ((3.0_f64 + 0.1) + 0.2).to_bits()
    );
    assert_ne!(
        value_at(&chain, 3.0, 0.0).to_bits(),
        (3.0_f64 + (0.1 + 0.2)).to_bits()
    );

    let mixed = lowered("@a + @b * 2");
    assert_eq!(mixed, a() + b() * lit(2.0));
    assert_eq!(mixed.op_count(), 5);
    assert_eq!(
        mixed.params().collect::<Vec<_>>(),
        [EntityId(7), EntityId(8)]
    );
    assert_eq!(value_at(&mixed, 1.0, 3.0).to_bits(), 7.0_f64.to_bits());
}

/// derived: `@a + 0` is `a, 0, add` (3 ops), so at `a = -0.0` it is `-0.0 + 0.0 = +0.0`,
/// not `a`'s own `-0.0`.
#[test]
fn scalar_body_applies_no_identity() {
    let expr = lowered("@a + 0");
    assert_eq!(expr.op_count(), 3);
    assert_eq!(expr.as_param(), None);
    assert_eq!(value_at(&expr, -0.0, 0.0).to_bits(), 0.0_f64.to_bits());
}

/// derived: `1 / @a` is `1, a, div`; at `a = 4.0` it is `0.25`, and at `a = 0.0` it is
/// `+inf`, a value the lowering cannot see.
#[test]
fn scalar_body_with_a_parameter_divisor_is_accepted() {
    let expr = lowered("1 / @a");
    assert_eq!(expr.op_count(), 3);
    assert_eq!(value_at(&expr, 4.0, 0.0).to_bits(), 0.25_f64.to_bits());
    assert_eq!(value_at(&expr, 0.0, 0.0).to_bits(), f64::INFINITY.to_bits());
}

/// derived: `(2 - 2)` folds to `2.0 - 2.0 = +0.0` and `-0` folds to `-0.0`; both are exact
/// zeros, reported over the division step from the chain head.
#[test]
fn scalar_body_division_by_a_folded_zero_is_rejected() {
    assert_eq!(
        rendered_rejection("@a / (2 - 2)"),
        "`@a / (2 - 2)`: division by zero"
    );
    assert_eq!(rendered_rejection("@a / -0"), "`@a / -0`: division by zero");
    assert_eq!(
        rendered_rejection("1 + @a * 3 / (1 - 1) * 2"),
        "`@a * 3 / (1 - 1)`: division by zero"
    );
}

/// derived: `1e308 * 1e308` overflows to `+inf`; in `1 / (1e308 * 10)` the inner step
/// `1e308 * 10` overflows to `+inf` before the division; `-1e308 - 1e308` overflows to
/// `-inf`.
#[test]
fn scalar_body_non_finite_fold_is_rejected() {
    assert_eq!(
        rendered_rejection("1e308 * 1e308"),
        "`1e308 * 1e308`: evaluates to a non-finite value (inf)"
    );
    assert_eq!(
        rendered_rejection("1 / (1e308 * 10)"),
        "`1e308 * 10`: evaluates to a non-finite value (inf)"
    );
    assert_eq!(
        rendered_rejection("-1e308 - 1e308"),
        "`-1e308 - 1e308`: evaluates to a non-finite value (-inf)"
    );
}

#[test]
fn scalar_body_rejects_an_lp_variable() {
    assert_eq!(
        rendered_rejection("@a * hydro_storage(1)"),
        "`hydro_storage(1)`: an LP variable cannot appear in a scalar-parameter expression"
    );
}

#[test]
fn a_scalar_parameter_cannot_read_scenario_data() {
    assert_eq!(
        lower_scalar_body("2 * bus_load_value(1)", &params()).unwrap_err(),
        Diagnostic {
            span: 4..21,
            message: "\"bus_load_value\" reads scenario data, which a scalar parameter cannot read"
                .to_owned(),
        }
    );
}

#[test]
fn scalar_body_rejects_an_unknown_name_with_the_hint() {
    assert_eq!(
        rendered_rejection("@a + 2 * @demand"),
        "`@demand`: unknown scalar parameter \"@demand\"; a named expression cannot appear \
         in a scalar-parameter expression"
    );
}

#[test]
fn scalar_body_rejects_a_relation() {
    for src in ["@a <= 1", "2 * @a == @b"] {
        assert_eq!(
            rendered_rejection(src),
            format!("`{src}`: a scalar-parameter expression cannot contain a relational operator"),
            "{src:?}"
        );
    }
}

/// Runs `f` under `params()`, the named expressions `decls` and `line_index()`, without data ids.
fn with_named<T>(decls: &[(&str, &str)], f: impl FnOnce(&Scope) -> T) -> T {
    let params = params();
    let line_index = line_index();
    let named = match NamedTable::declare(decls, &params, &line_index, &BTreeMap::new()) {
        Ok(named) => named,
        Err((j, d)) => panic!("expressions[{j}]: {}", d.render(decls[j].1)),
    };
    f(&Scope {
        params: &params,
        line_index: &line_index,
        named: &named,
        data_ids: None,
    })
}

/// A line `id` running from bus `source` to bus `target`.
fn line(id: i32, source: i32, target: i32) -> Line {
    let mut line = make_line(id, source, target);
    line.name = format!("L{id}");
    line
}

/// A single line 5 running from bus 3 to bus 7.
fn line_index() -> LineBusPairIndex {
    build_line_bus_pair_index(&[line(5, 3, 7)]).unwrap()
}

fn with_scope<T>(f: impl FnOnce(&Scope) -> T) -> T {
    with_named(&[], f)
}

fn lower_side(side: &Expr) -> Result<LinearForm, Diagnostic> {
    with_scope(|scope| {
        Lowering::new(scope, side, None)
            .form(side)
            .map_err(Fault::into_diagnostic)
    })
}

fn form(src: &str) -> LinearForm {
    let relation = parse(src).unwrap();
    assert!(relation.rhs.is_none(), "{src:?} is a relation");
    lower_side(&relation.lhs).unwrap_or_else(|d| panic!("{}", d.render(src)))
}

fn sides(src: &str) -> (LinearForm, LinearForm) {
    let relation = parse(src).unwrap();
    let (_, rhs) = relation.rhs.unwrap();
    (
        lower_side(&relation.lhs).unwrap(),
        lower_side(&rhs).unwrap(),
    )
}

fn terms(src: &str) -> Vec<(VariableRef, ScalarExpr)> {
    form(src).into_terms()
}

/// Every coefficient of `src` as the bits of its folded literal.
fn literal_terms(src: &str) -> Vec<(VariableRef, u64)> {
    literal_bits_of(src, terms(src))
}

fn literal_bits_of(src: &str, terms: Vec<(VariableRef, ScalarExpr)>) -> Vec<(VariableRef, u64)> {
    terms
        .into_iter()
        .map(|(variable, coefficient)| {
            let value = coefficient
                .as_literal()
                .unwrap_or_else(|| panic!("{src:?}: {variable:?} has a non-literal program"));
            (variable, value.to_bits())
        })
        .collect()
}

fn linear_rejection(src: &str) -> String {
    parse(src)
        .and_then(|relation| lower_side(&relation.lhs))
        .unwrap_err()
        .render(src)
}

fn a() -> ScalarExpr {
    ScalarExpr::param(EntityId(7))
}

fn b() -> ScalarExpr {
    ScalarExpr::param(EntityId(8))
}

fn lit(value: f64) -> ScalarExpr {
    ScalarExpr::literal(value)
}

fn hg(id: i32) -> VariableRef {
    VariableRef::HydroGeneration {
        hydro_id: EntityId(id),
        block_id: None,
        bus_id: None,
    }
}

fn tg(id: i32) -> VariableRef {
    VariableRef::ThermalGeneration {
        thermal_id: EntityId(id),
        block_id: None,
    }
}

fn exchange(line: i32) -> VariableRef {
    VariableRef::LineExchange {
        line_id: EntityId(line),
        block_id: None,
    }
}

/// derived: `2 * h0 + 3 * h0` is `2.0 + 3.0 = 5.0`, one folded literal;
/// `t5 + 2 * h0 + 3 * t5` keeps `t5` first at `1.0 + 3.0 = 4.0`, then `h0` at `2.0`;
/// `h10 + h11` is `[1.0, 1.0]`; `2.5 * t5 - h3` is `[2.5, -1.0]`; `t5 - h3` is
/// `[1.0, -1.0]`; the storage ramp is `[1.0, -1.0]`; `(h0 + h1)` lowers like `h0 + h1`;
/// `h5 - (h0 + h1)` is `[1.0, -1.0, -1.0]`.
#[test]
fn repeated_literal_coefficients_merge_in_authoring_order() {
    let merged = terms("2 * hydro_generation(0) + 3 * hydro_generation(0)");
    assert_eq!(merged.len(), 1);
    assert_eq!(merged[0].0, hg(0));
    assert_eq!(merged[0].1.op_count(), 1);
    assert_eq!(
        merged[0].1.as_literal().map(f64::to_bits),
        Some(5.0_f64.to_bits())
    );

    let storage = |final_side: bool| {
        if final_side {
            VariableRef::HydroStorageFinal {
                hydro_id: EntityId(5),
                block_id: Some(1),
            }
        } else {
            VariableRef::HydroStorageInitial {
                hydro_id: EntityId(5),
                block_id: Some(1),
            }
        }
    };
    for (src, expected) in [
        (
            "thermal_generation(5) + 2 * hydro_generation(0) + 3 * thermal_generation(5)",
            vec![(tg(5), 4.0), (hg(0), 2.0)],
        ),
        (
            "hydro_generation(10) + hydro_generation(11)",
            vec![(hg(10), 1.0), (hg(11), 1.0)],
        ),
        (
            "2.5 * thermal_generation(5) - hydro_generation(3)",
            vec![(tg(5), 2.5), (hg(3), -1.0)],
        ),
        (
            "thermal_generation(5) - hydro_generation(3)",
            vec![(tg(5), 1.0), (hg(3), -1.0)],
        ),
        (
            "hydro_storage_final(5, 1) - hydro_storage_initial(5, 1)",
            vec![(storage(true), 1.0), (storage(false), -1.0)],
        ),
        (
            "(hydro_generation(0) + hydro_generation(1))",
            vec![(hg(0), 1.0), (hg(1), 1.0)],
        ),
        (
            "hydro_generation(5) - (hydro_generation(0) + hydro_generation(1))",
            vec![(hg(5), 1.0), (hg(0), -1.0), (hg(1), -1.0)],
        ),
    ] {
        let expected: Vec<(VariableRef, u64)> = expected
            .into_iter()
            .map(|(variable, value)| (variable, f64::to_bits(value)))
            .collect();
        assert_eq!(literal_terms(src), expected, "{src:?}");
    }
    assert_eq!(
        form("(hydro_generation(0) + hydro_generation(1))"),
        form("hydro_generation(0) + hydro_generation(1)")
    );
}

/// derived: `2 * h0 + @a * h0` is the program `2, a, add`, at `a = 0.1` equal to
/// `2.0 + 0.1`; `@a * h0` is the bare parameter; `2.5 * @a * h0` is `2.5, a, mul`;
/// `2.0 * @a * h0` is `2, a, mul`; in `t5 - @a * h3`, `h3` is `a, neg`, at `a = 0.5` equal
/// to `-0.5`.
#[test]
fn parameter_and_literal_on_one_variable_merge_into_one_program() {
    let merged = terms("2 * hydro_generation(0) + @a * hydro_generation(0)");
    assert_eq!(merged.len(), 1);
    assert_eq!(merged[0].1, lit(2.0) + a());
    assert_ne!(merged[0].1, a() + lit(2.0));
    assert_eq!(merged[0].1.op_count(), 3);
    assert_eq!(
        merged[0].1.evaluate(|_| 0.1).to_bits(),
        (2.0_f64 + 0.1).to_bits()
    );

    let implicit = terms("@a * hydro_generation(0)");
    assert_eq!(implicit.len(), 1);
    assert_eq!(implicit[0].0, hg(0));
    assert_eq!(implicit[0].1.as_param(), Some(EntityId(7)));
    assert_eq!(implicit[0].1.op_count(), 1);

    assert_eq!(
        terms("2.5 * @a * hydro_generation(0)"),
        [(hg(0), lit(2.5) * a())]
    );
    assert_eq!(
        terms("2.0 * @a * hydro_generation(0)"),
        [(hg(0), lit(2.0) * a())]
    );

    let negated = terms("thermal_generation(5) - @a * hydro_generation(3)");
    assert_eq!(negated, [(tg(5), lit(1.0)), (hg(3), -a())]);
    assert_eq!(negated[1].1.op_count(), 2);
    assert_eq!(
        negated[1].1.evaluate(|_| 0.5).to_bits(),
        (-0.5_f64).to_bits()
    );
}

/// derived: `h0 - h0` is `1.0 - 1.0 = +0.0`, an exact zero, kept in the form and dropped by
/// `into_terms`; `h0 - h0 + h0` is `(1.0 - 1.0) + 1.0 = 1.0`; `h0 - h0 + @a * h0` is the
/// program `0, a, add`, which is not a literal and stays.
#[test]
fn a_variable_cancelled_to_exact_zero_is_dropped() {
    let cancelled = form("hydro_generation(0) - hydro_generation(0)");
    assert_eq!(cancelled.terms.len(), 1);
    assert_eq!(
        cancelled.terms[0]
            .1
            .as_ref()
            .and_then(ScalarExpr::as_literal)
            .map(f64::to_bits),
        Some(0.0_f64.to_bits())
    );
    assert!(cancelled.into_terms().is_empty());

    assert_eq!(
        literal_terms("2 * hydro_generation(1) + hydro_generation(0) - hydro_generation(0)"),
        [(hg(1), 2.0_f64.to_bits())]
    );
    assert_eq!(
        literal_terms("hydro_generation(0) - hydro_generation(0) + hydro_generation(0)"),
        [(hg(0), 1.0_f64.to_bits())]
    );

    let restored = terms("hydro_generation(0) - hydro_generation(0) + @a * hydro_generation(0)");
    assert_eq!(restored, [(hg(0), lit(0.0) + a())]);
    assert_eq!(restored[0].1.op_count(), 3);
}

/// derived: `0 * h0` types linear and holds `h0` at `+0.0`; `-0 * h0` holds it at `-0.0`;
/// both are exact zeros. `0 * h0 + 3` keeps the constant `3.0`. `@a * 0 * h0` is the
/// program `a, 0, mul`, which is not a literal and stays.
#[test]
fn zero_times_a_variable_types_linear_and_drops_its_term() {
    let zero = form("0 * hydro_generation(0)");
    assert_eq!(
        zero,
        LinearForm {
            terms: vec![(hg(0), Some(lit(0.0)))],
            constant: None,
        }
    );
    assert!(zero.into_terms().is_empty());

    let negative_zero = form("-0 * hydro_generation(0)");
    assert_eq!(
        negative_zero.terms[0]
            .1
            .as_ref()
            .and_then(ScalarExpr::as_literal)
            .map(f64::to_bits),
        Some((-0.0_f64).to_bits())
    );
    assert!(negative_zero.into_terms().is_empty());

    let with_constant = form("0 * hydro_generation(0) + 3");
    assert_eq!(
        with_constant
            .constant
            .as_ref()
            .and_then(ScalarExpr::as_literal)
            .map(f64::to_bits),
        Some(3.0_f64.to_bits())
    );
    assert!(with_constant.into_terms().is_empty());

    assert_eq!(
        terms("@a * 0 * hydro_generation(0)"),
        [(hg(0), a() * lit(0.0))]
    );
}

/// derived: `@a * (h0 + 2 * h1)` is `h0: a`, `h1: a, 2, mul`; `(h0 + 2 * h1) * @a` is
/// `h0: a`, `h1: 2, a, mul`; the two `h1` programs differ only in operand order.
#[test]
fn parameter_scaling_distributes_over_a_group() {
    let left = terms("@a * (hydro_generation(0) + 2 * hydro_generation(1))");
    assert_eq!(left, [(hg(0), a()), (hg(1), a() * lit(2.0))]);
    assert_eq!(left[0].1.as_param(), Some(EntityId(7)));

    let right = terms("(hydro_generation(0) + 2 * hydro_generation(1)) * @a");
    assert_eq!(right, [(hg(0), a()), (hg(1), lit(2.0) * a())]);
    assert_eq!(right[0].1.as_param(), Some(EntityId(7)));

    assert_ne!(left[1].1, right[1].1);
}

/// derived: `2 * (3 * (@a * h0))` is `2, 3, a, mul, mul` (5 ops), not the `6, a, mul` of
/// `2 * 3 * @a * h0`; `2.0 * (@a * h0)` is `2, a, mul`; `2 * (h0 - h1)` and
/// `2 * h0 - 2 * h1` are both `[2.0, -2.0]`; `3 * (h0 + h1)` is `[3.0, 3.0]`;
/// `2 * (h0 - 3 * (h1 + h2))` is `[2.0, 2.0 * -3.0, 2.0 * -3.0] = [2.0, -6.0, -6.0]`.
#[test]
fn nested_literal_scales_keep_the_authored_order() {
    let nested = terms("2 * (3 * (@a * hydro_generation(0)))");
    assert_eq!(nested, [(hg(0), lit(2.0) * (lit(3.0) * a()))]);
    assert_eq!(nested[0].1.op_count(), 5);
    let flat = terms("2 * 3 * @a * hydro_generation(0)");
    assert_eq!(flat, [(hg(0), lit(6.0) * a())]);
    assert_ne!(nested[0].1, flat[0].1);

    assert_eq!(
        terms("2.0 * (@a * hydro_generation(0))"),
        [(hg(0), lit(2.0) * a())]
    );

    let distributed = [(hg(0), 2.0_f64.to_bits()), (hg(1), (-2.0_f64).to_bits())];
    assert_eq!(
        literal_terms("2 * (hydro_generation(0) - hydro_generation(1))"),
        distributed
    );
    assert_eq!(
        literal_terms("2 * hydro_generation(0) - 2 * hydro_generation(1)"),
        distributed
    );
    assert_eq!(
        literal_terms("3 * (hydro_generation(0) + hydro_generation(1))"),
        [(hg(0), 3.0_f64.to_bits()), (hg(1), 3.0_f64.to_bits())]
    );
    assert_eq!(
        literal_terms(
            "2 * (hydro_generation(0) - 3 * (hydro_generation(1) + hydro_generation(2)))"
        ),
        [
            (hg(0), 2.0_f64.to_bits()),
            (hg(1), (-6.0_f64).to_bits()),
            (hg(2), (-6.0_f64).to_bits())
        ]
    );
}

/// derived: in `(h0 + @a * h1) / 2`, `h0` is `1.0 / 2.0 = 0.5` and `h1` is `a, 2, div`;
/// `(h0 + 3) / 2` has the constant `3.0 / 2.0 = 1.5`; `h0 / @a` is `1, a, div`, at
/// `a = 4.0` equal to `0.25`.
#[test]
fn division_by_a_scalar_divides_every_coefficient() {
    let divided = terms("(hydro_generation(0) + @a * hydro_generation(1)) / 2");
    assert_eq!(divided, [(hg(0), lit(0.5)), (hg(1), a() / lit(2.0))]);
    assert_eq!(
        divided[0].1.as_literal().map(f64::to_bits),
        Some(0.5_f64.to_bits())
    );
    assert_eq!(divided[1].1.op_count(), 3);

    let with_constant = form("(hydro_generation(0) + 3) / 2");
    assert_eq!(
        with_constant
            .constant
            .as_ref()
            .and_then(ScalarExpr::as_literal)
            .map(f64::to_bits),
        Some(1.5_f64.to_bits())
    );

    let by_parameter = terms("hydro_generation(0) / @a");
    assert_eq!(by_parameter, [(hg(0), lit(1.0) / a())]);
    assert_eq!(
        by_parameter[0].1.evaluate(|_| 4.0).to_bits(),
        0.25_f64.to_bits()
    );
}

/// derived: the pair (7, 3) runs against line 5's 3 -> 7 direction, so its coefficient is
/// `-1.0`; `2 * pair` is `2.0 * -1.0 = -2.0`; `line_exchange(5) + pair` is
/// `1.0 + -1.0 = 0.0`, dropped.
#[test]
fn reversed_line_pair_contributes_minus_one() {
    let reversed = form("line_exchange(source_bus=7, target_bus=3)");
    assert_eq!(
        reversed,
        LinearForm {
            terms: vec![(exchange(5), Some(lit(-1.0)))],
            constant: None,
        }
    );
    assert_eq!(
        literal_terms("line_exchange(source_bus=7, target_bus=3)"),
        [(exchange(5), (-1.0_f64).to_bits())]
    );
    assert_eq!(
        literal_terms("2 * line_exchange(source_bus=7, target_bus=3)"),
        [(exchange(5), (-2.0_f64).to_bits())]
    );

    let cancelled = form("line_exchange(5) + line_exchange(source_bus=7, target_bus=3)");
    assert_eq!(cancelled.terms.len(), 1);
    assert!(cancelled.into_terms().is_empty());
}

/// derived: the pair (3, 7) follows line 5's direction, so it is a unit coefficient, `1.0`
/// once materialized, exactly as `line_exchange(5)`.
#[test]
fn forward_line_pair_contributes_plus_one() {
    let forward = form("line_exchange(source_bus=3, target_bus=7)");
    assert_eq!(
        forward,
        LinearForm {
            terms: vec![(exchange(5), None)],
            constant: None,
        }
    );
    assert_eq!(forward, form("line_exchange(5)"));
    assert_eq!(
        literal_terms("line_exchange(source_bus=3, target_bus=7)"),
        [(exchange(5), 1.0_f64.to_bits())]
    );
}

/// derived: line 10 runs from bus 3 to bus 7, so the flow 3 -> 7 is its direct variable at
/// `1.0` and the flow 7 -> 3 its reverse variable, which the `-` makes `-1.0`, exactly as
/// `line_direct(10) - line_reverse(10)`.
#[test]
fn line_flow_lowers_to_the_directional_variable_of_its_line() {
    let (params, named) = (params(), NamedTable::default());
    let line_index = build_line_bus_pair_index(&[line(10, 3, 7)]).unwrap();
    let scope = Scope {
        params: &params,
        line_index: &line_index,
        named: &named,
        data_ids: None,
    };
    let src =
        "line_flow(source_bus=3, target_bus=7) - line_flow(source_bus=7, target_bus=3) <= 100";
    let flow = lower_relation(src, &scope).unwrap_or_else(|d| panic!("{}", d.render(src)));
    let (line_id, block_id) = (EntityId(10), None);
    assert_eq!(
        literal_bits_of(src, flow.terms.clone()),
        [
            (
                VariableRef::LineDirect { line_id, block_id },
                1.0_f64.to_bits()
            ),
            (
                VariableRef::LineReverse { line_id, block_id },
                (-1.0_f64).to_bits()
            ),
        ]
    );
    assert_eq!(
        Ok(flow),
        lower_relation("line_direct(10) - line_reverse(10) <= 100", &scope)
    );
}

#[test]
fn products_and_divisions_of_lp_variables_are_rejected() {
    let product = "product of LP variables is not linear";
    let division = "division by an expression containing LP variables";
    let unknown = |name: &str| {
        format!(
            "unknown name \"@{name}\": no scalar parameter or named expression with this \
             name was declared"
        )
    };
    for (src, fragment, message) in [
        (
            "hydro_generation(0) * hydro_generation(1)",
            "hydro_generation(0) * hydro_generation(1)",
            product.to_owned(),
        ),
        (
            "@a / hydro_generation(0)",
            "@a / hydro_generation(0)",
            division.to_owned(),
        ),
        (
            "hydro_generation(0) / hydro_generation(1)",
            "hydro_generation(0) / hydro_generation(1)",
            division.to_owned(),
        ),
        (
            "1 + 2 * hydro_generation(0) * (hydro_generation(1) + 1)",
            "2 * hydro_generation(0) * (hydro_generation(1) + 1)",
            product.to_owned(),
        ),
        (
            "@nobody * hydro_generation(0)",
            "@nobody",
            unknown("nobody"),
        ),
        (
            "@nobody + hydro_generation(0)",
            "@nobody",
            unknown("nobody"),
        ),
        ("3 * ()", ")", "unexpected ')'".to_owned()),
    ] {
        assert_eq!(
            linear_rejection(src),
            format!("`{fragment}`: {message}"),
            "{src:?}"
        );
    }
    assert_eq!(
        relation_rejection("hydro_generation(0) >= @p * @missing"),
        format!("`@missing`: {}", unknown("missing"))
    );
}

/// derived: the inner `1e308 * h0` holds `1e308`, and the outer `1e308 * 1e308` overflows
/// to `+inf`; `2 - 2` folds to `+0.0` and `-0` to `-0.0`, both exact zeros;
/// `1e308 + 1e308` overflows to `+inf` when two `h0` coefficients merge; the constant
/// `-1e308 - 1e308` overflows to `-inf`.
#[test]
fn folded_faults_in_linear_lowering_are_rejected() {
    let overflow = "evaluates to a non-finite value";
    for (src, message) in [
        (
            "1e308 * (1e308 * hydro_generation(0))",
            format!("{overflow} (inf)"),
        ),
        (
            "1e308 * hydro_generation(0) + 1e308 * hydro_generation(0)",
            format!("{overflow} (inf)"),
        ),
        (
            "hydro_generation(0) - 1e308 - 1e308",
            format!("{overflow} (-inf)"),
        ),
        (
            "hydro_generation(0) / (2 - 2)",
            "division by zero".to_owned(),
        ),
        ("hydro_generation(0) / -0", "division by zero".to_owned()),
        (
            "(hydro_generation(0) + 1) / (1 - 1)",
            "division by zero".to_owned(),
        ),
    ] {
        assert_eq!(
            linear_rejection(src),
            format!("`{src}`: {message}"),
            "{src:?}"
        );
    }
}

/// derived: each accepted lone reference is a unit coefficient, `1.0` once materialized;
/// `2.5 * anticipated_decision(5)` is `2.5`; two distinct references keep `1.0` each; two
/// equal `hydro_turbined(5, 0)` references merge into `1.0 + 1.0 = 2.0`.
#[test]
// Rationale: one table per rule set (every rejection message, every accepted form) reads
// as the argument catalogue; splitting it would scatter the cases across tests.
#[allow(clippy::too_many_lines)]
fn variable_argument_rules_keep_their_messages() {
    for (src, fragment, message) in [
        (
            "hydro_turbined(5.5)",
            "hydro_turbined(5.5)",
            "entity_id must be a non-negative integer, got 5.5 in variable \"hydro_turbined\"",
        ),
        (
            "hydro_turbined(bus=2)",
            "hydro_turbined(bus=2)",
            "expected integer entity_id in variable \"hydro_turbined\"",
        ),
        (
            "hydro_turbined(5, bus=2, 0)",
            "hydro_turbined(5, bus=2, 0)",
            "positional block argument must precede the named \"bus=\" argument in variable \
             \"hydro_turbined\"",
        ),
        (
            "2 * (hydro_turbined(5, 0, 1))",
            "(hydro_turbined(5, 0, 1))",
            "repeated block argument in variable \"hydro_turbined\"",
        ),
        (
            "hydro_turbined(5, 0.5)",
            "hydro_turbined(5, 0.5)",
            "block_id must be a non-negative integer, got 0.5 in variable \"hydro_turbined\"",
        ),
        (
            "hydro_turbined(5, bus=2, bus=3)",
            "hydro_turbined(5, bus=2, bus=3)",
            "repeated \"bus=\" argument in variable \"hydro_turbined\"",
        ),
        (
            "hydro_turbined(5, bus=2.5)",
            "hydro_turbined(5, bus=2.5)",
            "bus_id must be a non-negative integer, got 2.5 in variable \"hydro_turbined\"",
        ),
        (
            "hydro_generation(0, source_bus=3, target_bus=7)",
            "hydro_generation(0, source_bus=3, target_bus=7)",
            "the (source_bus=, target_bus=) pair form is only accepted on \"line_exchange\" \
             or \"line_flow\" as its sole argument, e.g. \"line_exchange(source_bus=3, \
             target_bus=7)\"; it cannot follow a positional argument in \"hydro_generation\"",
        ),
        (
            "hydro_turbined(5, foo=2)",
            "hydro_turbined(5, foo=2)",
            "unknown named argument \"foo\" in variable \"hydro_turbined\": only \"bus\" and \
             \"unit_group\" are supported",
        ),
        (
            "hydro_generation(source_bus=3, target_bus=7)",
            "hydro_generation(source_bus=3, target_bus=7)",
            "the (source_bus=, target_bus=) pair form addresses a line by its endpoint buses \
             and is only accepted on \"line_exchange\" or \"line_flow\", not \"hydro_generation\"",
        ),
        (
            "line_exchange(source_bus=3, 7)",
            "line_exchange(source_bus=3, 7)",
            "expected \"source_bus\" or \"target_bus\" in \"line_exchange\" pair form; write \
             \"line_exchange(source_bus=X, target_bus=Y)\"",
        ),
        (
            "line_exchange(source_bus=3, bus=7)",
            "line_exchange(source_bus=3, bus=7)",
            "expected \"source_bus\" or \"target_bus\" in \"line_exchange\" pair form; write \
             \"line_exchange(source_bus=X, target_bus=Y)\"",
        ),
        (
            "line_exchange(source_bus=3.5, target_bus=7)",
            "line_exchange(source_bus=3.5, target_bus=7)",
            "source_bus must be a non-negative integer, got 3.5 in \"line_exchange\"",
        ),
        (
            "line_exchange(source_bus=3, source_bus=7)",
            "line_exchange(source_bus=3, source_bus=7)",
            "repeated \"source_bus=\" argument in \"line_exchange\"",
        ),
        (
            "line_exchange(target_bus=3, target_bus=7)",
            "line_exchange(target_bus=3, target_bus=7)",
            "repeated \"target_bus=\" argument in \"line_exchange\"",
        ),
        (
            "line_exchange(source_bus=3)",
            "line_exchange(source_bus=3)",
            "the \"line_exchange\" pair form requires both \"source_bus=\" and \"target_bus=\"; \
             write \"line_exchange(source_bus=X, target_bus=Y)\"",
        ),
        (
            "line_exchange(target_bus=7)",
            "line_exchange(target_bus=7)",
            "the \"line_exchange\" pair form requires both \"source_bus=\" and \"target_bus=\"; \
             write \"line_exchange(source_bus=X, target_bus=Y)\"",
        ),
        (
            "line_exchange(source_bus=1, target_bus=2)",
            "line_exchange(source_bus=1, target_bus=2)",
            "no line connects buses source_bus=1 and target_bus=2",
        ),
        (
            "hydro_storage(7, 0)",
            "hydro_storage(7, 0)",
            "variable \"hydro_storage\" does not accept a block argument",
        ),
        (
            "invalid_var(0)",
            "invalid_var(0)",
            "unknown variable name \"invalid_var\": not a recognised LP variable name",
        ),
        (
            "not_a_variable(5, bus=2)",
            "not_a_variable(5, bus=2)",
            "unknown variable name \"not_a_variable\": not a recognised LP variable name",
        ),
        (
            "hydro_storage(5, bus=2)",
            "hydro_storage(5, bus=2)",
            "variable \"hydro_storage\" does not accept a bus selector; only \
             \"hydro_turbined\" and \"hydro_generation\" accept a \"bus=\" argument",
        ),
        (
            "anticipated_decision(5, 0)",
            "anticipated_decision(5, 0)",
            "variable \"anticipated_decision\" is a stage-level scalar and does not accept a \
             block_id — write \"anticipated_decision(5)\", not \"anticipated_decision(5, ...)\"",
        ),
    ] {
        assert_eq!(
            linear_rejection(src),
            format!("`{fragment}`: {message}"),
            "{src:?}"
        );
    }
    for (src, name, message) in [
        (
            "hydro_spillage(5, bus=2)",
            "hydro_spillage",
            "does not accept a bus selector; only \"hydro_turbined\" and \"hydro_generation\" \
             accept a \"bus=\" argument",
        ),
        (
            "hydro_outflow(5, bus=2)",
            "hydro_outflow",
            "does not accept a bus selector; only \"hydro_turbined\" and \"hydro_generation\" \
             accept a \"bus=\" argument",
        ),
        ("hydro_turbined(5, 0, 0)", "hydro_turbined", ""),
        ("hydro_generation(5, 1, 2, bus=3)", "hydro_generation", ""),
        ("thermal_generation(0, 0, 0)", "thermal_generation", ""),
    ] {
        let expected = if message.is_empty() {
            format!("`{src}`: repeated block argument in variable \"{name}\"")
        } else {
            format!("`{src}`: variable \"{name}\" {message}")
        };
        assert_eq!(linear_rejection(src), expected, "{src:?}");
    }

    let turbined = |block_id, bus_id: Option<i32>| VariableRef::HydroTurbined {
        hydro_id: EntityId(5),
        block_id,
        bus_id: bus_id.map(EntityId),
    };
    let accepted: Vec<(&str, VariableRef)> = vec![
        (
            "hydro_storage(0)",
            VariableRef::HydroStorage {
                hydro_id: EntityId(0),
            },
        ),
        (
            "hydro_turbined(0)",
            VariableRef::HydroTurbined {
                hydro_id: EntityId(0),
                block_id: None,
                bus_id: None,
            },
        ),
        (
            "hydro_spillage(0)",
            VariableRef::HydroSpillage {
                hydro_id: EntityId(0),
                block_id: None,
            },
        ),
        (
            "hydro_diversion(0)",
            VariableRef::HydroDiversion {
                hydro_id: EntityId(0),
                block_id: None,
            },
        ),
        (
            "hydro_outflow(0)",
            VariableRef::HydroOutflow {
                hydro_id: EntityId(0),
                block_id: None,
            },
        ),
        ("hydro_generation(0)", hg(0)),
        (
            "hydro_evaporation(0)",
            VariableRef::HydroEvaporation {
                hydro_id: EntityId(0),
                block_id: None,
            },
        ),
        (
            "hydro_withdrawal(0)",
            VariableRef::HydroWithdrawal {
                hydro_id: EntityId(0),
            },
        ),
        (
            "hydro_inflow(0)",
            VariableRef::HydroInflow {
                hydro_id: EntityId(0),
                block_id: None,
            },
        ),
        ("thermal_generation(0)", tg(0)),
        (
            "line_direct(0)",
            VariableRef::LineDirect {
                line_id: EntityId(0),
                block_id: None,
            },
        ),
        (
            "line_reverse(0)",
            VariableRef::LineReverse {
                line_id: EntityId(0),
                block_id: None,
            },
        ),
        ("line_exchange(0)", exchange(0)),
        (
            "bus_deficit(0)",
            VariableRef::BusDeficit {
                bus_id: EntityId(0),
                block_id: None,
            },
        ),
        (
            "bus_excess(0)",
            VariableRef::BusExcess {
                bus_id: EntityId(0),
                block_id: None,
            },
        ),
        (
            "pumping_flow(0)",
            VariableRef::PumpingFlow {
                station_id: EntityId(0),
                block_id: None,
            },
        ),
        (
            "pumping_power(0)",
            VariableRef::PumpingPower {
                station_id: EntityId(0),
                block_id: None,
            },
        ),
        (
            "contract_import(0)",
            VariableRef::ContractImport {
                contract_id: EntityId(0),
                block_id: None,
            },
        ),
        (
            "contract_export(0)",
            VariableRef::ContractExport {
                contract_id: EntityId(0),
                block_id: None,
            },
        ),
        (
            "non_controllable_generation(0)",
            VariableRef::NonControllableGeneration {
                source_id: EntityId(0),
                block_id: None,
            },
        ),
        (
            "non_controllable_curtailment(0)",
            VariableRef::NonControllableCurtailment {
                source_id: EntityId(0),
                block_id: None,
            },
        ),
        (
            "anticipated_decision(5)",
            VariableRef::AnticipatedDecision {
                thermal_id: EntityId(5),
            },
        ),
        (
            "anticipated_decision( 5 )",
            VariableRef::AnticipatedDecision {
                thermal_id: EntityId(5),
            },
        ),
        ("hydro_turbined(5)", turbined(None, None)),
        ("hydro_turbined(5, 0)", turbined(Some(0), None)),
        ("hydro_turbined(5, bus=2)", turbined(None, Some(2))),
        ("hydro_turbined(5, 0, bus=2)", turbined(Some(0), Some(2))),
        ("hydro_turbined(5, 1, bus=2)", turbined(Some(1), Some(2))),
        (
            "line_exchange(0, 1)",
            VariableRef::LineExchange {
                line_id: EntityId(0),
                block_id: Some(1),
            },
        ),
        (
            "hydro_useful_volume_final(3, 0)",
            VariableRef::HydroUsefulVolumeFinal {
                hydro_id: EntityId(3),
                block_id: Some(0),
            },
        ),
    ];
    for (src, variable) in accepted {
        assert_eq!(
            literal_terms(src),
            [(variable, 1.0_f64.to_bits())],
            "{src:?}"
        );
    }

    let anticipated = |thermal_id| VariableRef::AnticipatedDecision {
        thermal_id: EntityId(thermal_id),
    };
    for (src, expected) in [
        ("2.5 * anticipated_decision(5)", vec![(anticipated(5), 2.5)]),
        (
            "anticipated_decision(5) + anticipated_decision(6)",
            vec![(anticipated(5), 1.0), (anticipated(6), 1.0)],
        ),
        (
            "hydro_turbined(5, 0) + hydro_turbined(5, 1)",
            vec![
                (turbined(Some(0), None), 1.0),
                (turbined(Some(1), None), 1.0),
            ],
        ),
        (
            "hydro_turbined(5, 0) + hydro_turbined(5, 0)",
            vec![(turbined(Some(0), None), 2.0)],
        ),
    ] {
        let expected: Vec<(VariableRef, u64)> = expected
            .into_iter()
            .map(|(variable, value)| (variable, f64::to_bits(value)))
            .collect();
        assert_eq!(literal_terms(src), expected, "{src:?}");
    }
}

/// derived: `@a * (h0 + h1)` gives each unit coefficient the scale itself, `a`.
#[test]
fn parameter_scaled_group_distributes_the_parameter() {
    assert_eq!(
        terms("@a * (hydro_generation(0) + hydro_generation(1))"),
        [(hg(0), a()), (hg(1), a())]
    );
}

/// derived: `2.0 * @a` folds first to the program `2, a, mul`, which each unit coefficient
/// of the group takes as is.
#[test]
fn literal_times_parameter_scaled_group_distributes_the_product() {
    let scale = lit(2.0) * a();
    assert_eq!(
        terms("2.0 * @a * (hydro_generation(0) + hydro_generation(1))"),
        [(hg(0), scale.clone()), (hg(1), scale)]
    );
}

/// derived: `@a * @b * h0` is `a, b, mul`, at `a = 3.0`, `b = 0.5` equal to `1.5`.
#[test]
fn two_parameters_in_one_term_multiply() {
    let product = terms("@a * @b * hydro_generation(0)");
    assert_eq!(product, [(hg(0), a() * b())]);
    assert_eq!(
        product[0]
            .1
            .evaluate(|id| if id == EntityId(7) { 3.0 } else { 0.5 })
            .to_bits(),
        1.5_f64.to_bits()
    );
}

/// derived: the left side `@a * @b * h0` gives `h0` the program `a, b, mul`; the right side
/// is the constant `0.0` with no term.
#[test]
fn parameter_product_times_a_variable_is_its_coefficient() {
    let (lhs, rhs) = sides("@a * @b * hydro_generation(0) >= 0");
    assert_eq!(
        lhs,
        LinearForm {
            terms: vec![(hg(0), Some(a() * b()))],
            constant: None,
        }
    );
    assert_eq!(
        rhs.constant
            .as_ref()
            .and_then(ScalarExpr::as_literal)
            .map(f64::to_bits),
        Some(0.0_f64.to_bits())
    );
    assert!(rhs.terms.is_empty());
}

/// derived: the right side `@a * @b * @c` is the constant program `a, b, mul, c, mul`.
#[test]
fn chained_parameter_product_lowers_to_one_constant() {
    let (lhs, rhs) = sides("hydro_generation(0) >= @a * @b * @c");
    assert_eq!(
        lhs,
        LinearForm {
            terms: vec![(hg(0), None)],
            constant: None,
        }
    );
    assert_eq!(
        rhs,
        LinearForm {
            terms: Vec::new(),
            constant: Some(a() * b() * param(9)),
        }
    );
}

/// derived: the right side `2 * (@a * @b)` is the constant program `2, a, b, mul, mul`.
#[test]
fn parameter_product_inside_a_group_lowers_to_a_constant() {
    let (_, rhs) = sides("hydro_generation(0) >= 2 * (@a * @b)");
    assert_eq!(
        rhs,
        LinearForm {
            terms: Vec::new(),
            constant: Some(lit(2.0) * (a() * b())),
        }
    );
    assert_eq!(rhs.constant.as_ref().map(ScalarExpr::op_count), Some(5));
}

fn relation(src: &str) -> LoweredRelation {
    with_scope(|scope| lower_relation(src, scope)).unwrap_or_else(|d| panic!("{}", d.render(src)))
}

fn relation_rejection(src: &str) -> String {
    with_scope(|scope| lower_relation(src, scope))
        .unwrap_err()
        .render(src)
}

fn relation_terms(src: &str) -> Vec<(VariableRef, u64)> {
    literal_bits_of(src, relation(src).terms)
}

fn literal_bits(endpoint: Option<&ScalarExpr>) -> Option<u64> {
    endpoint.and_then(ScalarExpr::as_literal).map(f64::to_bits)
}

fn p() -> ScalarExpr {
    ScalarExpr::param(EntityId(10))
}

fn q() -> ScalarExpr {
    ScalarExpr::param(EntityId(11))
}

/// derived: `h0 >= @p` targets the lower endpoint with `0, p, add` (3 ops), at `p = -0.0`
/// equal to `0.0 + -0.0 = +0.0` and at `p = 1.25` to `1.25`;
/// `h0 >= 2 * @p * @q - @a * @b + 5` targets it with `0` added to the right side's program
/// `2, p, mul, q, mul, a, b, mul, sub, 5, add` (13 ops in all), at `p = 3`, `q = 0.5`,
/// `a = 2`, `b = 0.25` equal to `0.0 + (((2.0 * 3.0) * 0.5 - 2.0 * 0.25) + 5.0) = 7.5`.
#[test]
fn a_bare_parameter_bound_starts_from_zero() {
    let bare = relation("hydro_generation(0) >= @p");
    assert_eq!(bare.terms, [(hg(0), lit(1.0))]);
    let lower = bare.lower.unwrap();
    assert_eq!(lower, lit(0.0) + p());
    assert_eq!(lower.as_param(), None);
    assert_eq!(lower.op_count(), 3);
    assert_eq!(lower.evaluate(|_| -0.0).to_bits(), 0.0_f64.to_bits());
    assert_eq!(lower.evaluate(|_| 1.25).to_bits(), 1.25_f64.to_bits());
    assert_eq!(bare.upper, None);

    let products = relation("hydro_generation(0) >= 2 * @p * @q - @a * @b + 5");
    let lower = products.lower.unwrap();
    assert_eq!(
        lower,
        lit(0.0) + (lit(2.0) * p() * q() - a() * b() + lit(5.0))
    );
    assert_eq!(lower.op_count(), 13);
    let value = lower.evaluate(|id| match id {
        EntityId(10) => 3.0,
        EntityId(11) => 0.5,
        EntityId(7) => 2.0,
        EntityId(8) => 0.25,
        _ => panic!("unexpected parameter id {id:?}"),
    });
    assert_eq!(value.to_bits(), 7.5_f64.to_bits());
    assert_eq!(products.upper, None);
}

/// derived: each expected value replays the remainder fold, `0.0` plus each parameter term
/// `coef * leaf` and then each product `scale * left * right`, right side first and left
/// side negated: `h0 <= -2 * @p * @q` is `0.0 + (-2.0 * p) * q`, `h0 >= h1 - @p * @q` is
/// `0.0 + (-1.0 * p) * q`, `h0 <= @a + @b` is `(0.0 + 1.0 * a) + 1.0 * b`, `h0 <= -@a` is
/// `0.0 + -1.0 * a`, `h0 <= 2 * @a` is `0.0 + 2.0 * a`, `h0 >= @p` is `0.0 + 1.0 * p` and
/// `h0 + @a >= @b` is `(0.0 + 1.0 * b) + -1.0 * a`; the `+0.0` start absorbs a `-0.0` sum,
/// so none is `-0.0`. Away from `-0.0` values cannot tell `0, c, add` from `c`, so the
/// programs are also compared with constructor-built ones.
#[test]
fn right_only_and_two_sided_remainders_are_never_negative_zero() {
    struct Shape {
        src: &'static str,
        program: ScalarExpr,
        terms: &'static [(f64, i32)],
        products: &'static [(f64, i32, i32)],
    }
    let shapes = [
        Shape {
            src: "hydro_generation(0) <= -2 * @p * @q",
            program: lit(0.0) + lit(-2.0) * p() * q(),
            terms: &[],
            products: &[(-2.0, 10, 11)],
        },
        Shape {
            src: "hydro_generation(0) >= hydro_generation(1) - @p * @q",
            program: lit(0.0) + -(p() * q()),
            terms: &[],
            products: &[(-1.0, 10, 11)],
        },
        Shape {
            src: "hydro_generation(0) <= @a + @b",
            program: lit(0.0) + (a() + b()),
            terms: &[(1.0, 7), (1.0, 8)],
            products: &[],
        },
        Shape {
            src: "hydro_generation(0) <= -@a",
            program: lit(0.0) + -a(),
            terms: &[(-1.0, 7)],
            products: &[],
        },
        Shape {
            src: "hydro_generation(0) <= 2 * @a",
            program: lit(0.0) + lit(2.0) * a(),
            terms: &[(2.0, 7)],
            products: &[],
        },
        Shape {
            src: "hydro_generation(0) >= @p",
            program: lit(0.0) + p(),
            terms: &[(1.0, 10)],
            products: &[],
        },
        Shape {
            src: "hydro_generation(0) + @a >= @b",
            program: (lit(0.0) + b()) - a(),
            terms: &[(1.0, 8), (-1.0, 7)],
            products: &[],
        },
    ];
    let valuations: [[(i32, f64); 4]; 3] = [
        [(7, 0.0), (8, 0.0), (10, 0.0), (11, 0.0)],
        [(7, -0.0), (8, -0.0), (10, -0.0), (11, -0.0)],
        [(7, 0.0), (8, -0.0), (10, 0.0), (11, -0.0)],
    ];
    for shape in shapes {
        let lowered = relation(shape.src);
        let remainder = match (lowered.lower, lowered.upper) {
            (Some(remainder), None) | (None, Some(remainder)) => remainder,
            endpoints => panic!("{:?} targets {endpoints:?}", shape.src),
        };
        assert_eq!(remainder, shape.program, "{:?}", shape.src);
        for valuation in valuations {
            let leaf = |id: i32| match valuation.iter().find(|&&(known, _)| known == id) {
                Some(&(_, value)) => value,
                None => panic!("unexpected parameter id {id}"),
            };
            let value = remainder.evaluate(|EntityId(id)| leaf(id));
            let linear = shape
                .terms
                .iter()
                .fold(0.0, |acc, &(coef, id)| acc + coef * leaf(id));
            let old = shape
                .products
                .iter()
                .fold(linear, |acc, &(scale, left, right)| {
                    acc + scale * leaf(left) * leaf(right)
                });
            assert_ne!(
                value.to_bits(),
                (-0.0_f64).to_bits(),
                "{:?} at {valuation:?}",
                shape.src
            );
            assert_eq!(
                value.to_bits(),
                old.to_bits(),
                "{:?} at {valuation:?}",
                shape.src
            );
        }
    }
}

/// derived: `h0 + 2 >= 5` is `5.0 - 2.0 = 3.0`; `h0 + 2 >= 10 - 73` is
/// `(10.0 - 73.0) - 2.0 = -65.0`; `@p * @q + h0 >= 0` is the program `0, p, q, mul, sub`
/// (5 ops), at `p = q = 0` equal to `0.0 - (0.0 * 0.0) = +0.0`; `h0 + 1e308 >= -1e308` is
/// `-1e308 - 1e308 = -inf`.
#[test]
fn constants_on_both_sides_subtract_right_minus_left() {
    let folded = relation("hydro_generation(0) + 2 >= 5");
    assert_eq!(
        relation_terms("hydro_generation(0) + 2 >= 5"),
        [(hg(0), 1.0_f64.to_bits())]
    );
    assert_eq!(literal_bits(folded.lower.as_ref()), Some(3.0_f64.to_bits()));
    assert_eq!(folded.upper, None);

    assert_eq!(
        literal_bits(
            relation("hydro_generation(0) + 2 >= 10 - 73")
                .lower
                .as_ref()
        ),
        Some((-65.0_f64).to_bits())
    );

    let product = relation("@p * @q + hydro_generation(0) >= 0")
        .lower
        .unwrap();
    assert_eq!(product, lit(0.0) - p() * q());
    assert_ne!(product, -(p() * q()));
    assert_eq!(product.op_count(), 5);
    assert_eq!(product.evaluate(|_| 0.0).to_bits(), 0.0_f64.to_bits());

    let src = "hydro_generation(0) + 1e308 >= -1e308";
    assert_eq!(
        relation_rejection(src),
        format!("`{src}`: evaluates to a non-finite value (-inf)")
    );
}

/// derived: `@p * @q + h0 >= h1` is the program `0, p, q, mul, sub`, at `p = q = 0` equal to
/// `0.0 - 0.0 = +0.0`; `h0 + 0 <= h1` folds `0.0 - 0.0` to the literal `+0.0`;
/// `h0 - @a >= h1` is `0, a, neg, sub`, at `a = 0` equal to `0.0 - (-0.0) = +0.0` and at
/// `a = 1.5` to `1.5`; `10 + 2 * @a <= h1` is `0, 10, 2, a, mul, add, sub`.
#[test]
fn a_left_only_constant_is_subtracted_from_zero() {
    let src = "@p * @q + hydro_generation(0) >= hydro_generation(1)";
    let product = relation(src);
    assert_eq!(
        literal_bits_of(src, product.terms),
        [(hg(0), 1.0_f64.to_bits()), (hg(1), (-1.0_f64).to_bits())]
    );
    let lower = product.lower.unwrap();
    assert_eq!(lower, lit(0.0) - p() * q());
    assert_ne!(lower, -(p() * q()));
    assert_eq!(lower.evaluate(|_| 0.0).to_bits(), 0.0_f64.to_bits());
    assert_eq!(product.upper, None);

    let zero = relation("hydro_generation(0) + 0 <= hydro_generation(1)");
    assert_eq!(literal_bits(zero.upper.as_ref()), Some(0.0_f64.to_bits()));
    assert_eq!(zero.lower, None);

    let negated = relation("hydro_generation(0) - @a >= hydro_generation(1)")
        .lower
        .unwrap();
    assert_eq!(negated, lit(0.0) - -a());
    assert_ne!(negated, a());
    assert_eq!(negated.op_count(), 4);
    assert_eq!(negated.evaluate(|_| 0.0).to_bits(), 0.0_f64.to_bits());
    assert_eq!(negated.evaluate(|_| 1.5).to_bits(), 1.5_f64.to_bits());

    let no_variable = relation("10 + 2 * @a <= hydro_generation(1)");
    assert_eq!(no_variable.terms, [(hg(1), lit(-1.0))]);
    assert_eq!(
        no_variable.upper,
        Some(lit(0.0) - (lit(10.0) + lit(2.0) * a()))
    );
}

/// derived: with no constant on either side the targeted endpoint is the literal `+0.0`;
/// `h1` moves left at `-1.0`.
#[test]
fn a_relation_without_constants_targets_a_zero_remainder() {
    let src = "hydro_generation(0) >= hydro_generation(1)";
    let ge = relation(src);
    assert_eq!(
        relation_terms(src),
        [(hg(0), 1.0_f64.to_bits()), (hg(1), (-1.0_f64).to_bits())]
    );
    assert_eq!(literal_bits(ge.lower.as_ref()), Some(0.0_f64.to_bits()));
    assert_eq!(ge.upper, None);

    let le = relation("hydro_generation(0) <= hydro_generation(1)");
    assert_eq!(literal_bits(le.upper.as_ref()), Some(0.0_f64.to_bits()));
    assert_eq!(le.lower, None);
}

/// derived: `h0 == 3` puts `3.0` on both endpoints; `h0 + 2 == h1` puts `0.0 - 2.0 = -2.0`
/// on both; `h0 == @p` puts `0, p, add` on both.
#[test]
fn equality_targets_both_endpoints() {
    let three = relation("hydro_generation(0) == 3");
    assert_eq!(literal_bits(three.lower.as_ref()), Some(3.0_f64.to_bits()));
    assert_eq!(literal_bits(three.upper.as_ref()), Some(3.0_f64.to_bits()));

    let moved = relation("hydro_generation(0) + 2 == hydro_generation(1)");
    assert_eq!(
        literal_bits(moved.lower.as_ref()),
        Some((-2.0_f64).to_bits())
    );
    assert_eq!(
        literal_bits(moved.upper.as_ref()),
        Some((-2.0_f64).to_bits())
    );

    let parameter = relation("hydro_generation(0) == @p");
    assert_eq!(parameter.lower, Some(lit(0.0) + p()));
    assert_eq!(parameter.upper, Some(lit(0.0) + p()));
}

/// derived: `2 * h0 <= 3 * h0 + @a` gives `h0` the literal `2.0 - 3.0 = -1.0` and the upper
/// endpoint `0, a, add`; `t5 <= 0.87 * h140` gives `[1.0, -0.87]` and the upper endpoint `+0.0`;
/// `2 * @a * h0 <= 3 * @a * h0` gives `h0` the program `2, a, mul, 3, a, mul, sub` (7 ops),
/// at `a = 0.1` equal to `(2.0 * 0.1) - (3.0 * 0.1)`.
#[test]
fn right_side_variables_move_left_with_flipped_signs() {
    let src = "2 * hydro_generation(0) <= 3 * hydro_generation(0) + @a";
    assert_eq!(relation_terms(src), [(hg(0), (-1.0_f64).to_bits())]);
    let merged = relation(src);
    assert_eq!(merged.upper, Some(lit(0.0) + a()));
    assert_eq!(merged.lower, None);

    let src = "thermal_generation(5) <= 0.87 * hydro_generation(140)";
    assert_eq!(
        relation_terms(src),
        [(tg(5), 1.0_f64.to_bits()), (hg(140), (-0.87_f64).to_bits())]
    );
    assert_eq!(
        literal_bits(relation(src).upper.as_ref()),
        Some(0.0_f64.to_bits())
    );

    let repeated = relation("2 * @a * hydro_generation(0) <= 3 * @a * hydro_generation(0)");
    assert_eq!(repeated.terms, [(hg(0), lit(2.0) * a() - lit(3.0) * a())]);
    assert_eq!(repeated.terms[0].1.op_count(), 7);
    assert_eq!(
        repeated.terms[0].1.evaluate(|_| 0.1).to_bits(),
        ((2.0_f64 * 0.1) - (3.0 * 0.1)).to_bits()
    );
}

/// derived: `h0 - h0` and `h0` against `h0 + 5` both cancel to `1.0 - 1.0 = +0.0` and drop,
/// leaving `5.0`; `-0 * h1 - 0 * h1` is `-0.0 - 0.0 = -0.0`, an exact zero; `3 <= 5` has no
/// variable and the remainder `5.0 - 3.0 = 2.0`.
#[test]
fn cancelled_variables_leave_the_bound() {
    for (src, upper) in [
        ("hydro_generation(0) - hydro_generation(0) <= 5", 5.0),
        ("hydro_generation(0) <= hydro_generation(0) + 5", 5.0),
        (
            "-0 * hydro_generation(1) - 0 * hydro_generation(1) <= 1",
            1.0,
        ),
        ("3 <= 5", 2.0),
    ] {
        let lowered = relation(src);
        assert!(lowered.terms.is_empty(), "{src:?}");
        assert_eq!(
            literal_bits(lowered.upper.as_ref()),
            Some(f64::to_bits(upper)),
            "{src:?}"
        );
        assert_eq!(lowered.lower, None, "{src:?}");
    }
}

/// derived: `h0 + h1` without an operator keeps `[1.0, 1.0]` and targets no endpoint.
#[test]
fn an_operator_free_constant_is_rejected() {
    for src in [
        "hydro_generation(0) + 5",
        "(hydro_generation(0) + 2)",
        "hydro_generation(0) + 0",
        "hydro_generation(0) + (hydro_generation(1) + 5)",
        "2 * (hydro_generation(1) - 3)",
        "3",
        "@p * @q",
    ] {
        assert_eq!(
            relation_rejection(src),
            format!(
                "`{src}`: an expression without a relational operator cannot have a constant \
                 part; put the bound in generic_constraint_bounds.parquet or use <=, >= or =="
            ),
            "{src:?}"
        );
    }

    let src = "hydro_generation(0) + hydro_generation(1)";
    assert_eq!(
        relation_terms(src),
        [(hg(0), 1.0_f64.to_bits()), (hg(1), 1.0_f64.to_bits())]
    );
    let free = relation(src);
    assert_eq!((free.lower, free.upper), (None, None));
}

/// derived: `@p * @q * h0 <= 1` gives `h0` the program `p, q, mul`, at `p = 3`, `q = 0.5`
/// equal to `1.5`, and the upper endpoint `1.0`; `@p * @q + h0 >= 0` keeps `h0` at `1.0`
/// with the lower endpoint `0, p, q, mul, sub`.
#[test]
fn left_hand_parameter_products_are_accepted() {
    let scaled = relation("@p * @q * hydro_generation(0) <= 1");
    assert_eq!(scaled.terms, [(hg(0), p() * q())]);
    assert_eq!(
        scaled.terms[0]
            .1
            .evaluate(|id| if id == EntityId(10) { 3.0 } else { 0.5 })
            .to_bits(),
        1.5_f64.to_bits()
    );
    assert_eq!(literal_bits(scaled.upper.as_ref()), Some(1.0_f64.to_bits()));

    let standalone = relation("@p * @q + hydro_generation(0) >= 0");
    assert_eq!(standalone.terms, [(hg(0), lit(1.0))]);
    assert_eq!(standalone.lower, Some(lit(0.0) - p() * q()));
}

/// derived: `h0 + (h1 + 5) <= 10` keeps `[1.0, 1.0]` and the upper endpoint
/// `10.0 - 5.0 = 5.0`; `2 * (h1 - 3) >= 0` gives `h1` the coefficient `2.0` and the lower
/// endpoint `0.0 - (2.0 * -3.0) = 6.0`.
#[test]
fn a_constant_inside_a_group_reaches_the_bound() {
    let src = "hydro_generation(0) + (hydro_generation(1) + 5) <= 10";
    assert_eq!(
        relation_terms(src),
        [(hg(0), 1.0_f64.to_bits()), (hg(1), 1.0_f64.to_bits())]
    );
    assert_eq!(
        literal_bits(relation(src).upper.as_ref()),
        Some(5.0_f64.to_bits())
    );

    let src = "2 * (hydro_generation(1) - 3) >= 0";
    assert_eq!(relation_terms(src), [(hg(1), 2.0_f64.to_bits())]);
    assert_eq!(
        literal_bits(relation(src).lower.as_ref()),
        Some(6.0_f64.to_bits())
    );
}

fn named_relation(decls: &[(&str, &str)], src: &str) -> LoweredRelation {
    with_named(decls, |scope| lower_relation(src, scope))
        .unwrap_or_else(|d| panic!("{}", d.render(src)))
}

fn named_rejection(decls: &[(&str, &str)], src: &str) -> String {
    with_named(decls, |scope| lower_relation(src, scope))
        .unwrap_err()
        .render(src)
}

fn declaration_rejection(decls: &[(&str, &str)]) -> (usize, String) {
    match NamedTable::declare(decls, &params(), &line_index(), &BTreeMap::new()) {
        Ok(_) => panic!("{decls:?} declared"),
        Err((j, d)) => (j, d.render(decls[j].1)),
    }
}

/// One constraint's lowering, with how often each body was lowered, the most bodies cached
/// at once and whether the cache was empty at the end.
struct Trace {
    result: Result<LoweredRelation, Diagnostic>,
    lowerings: Vec<usize>,
    peak_cached: usize,
    cache_empty: bool,
}

fn trace_in(scope: &Scope, src: &str) -> Trace {
    let relation = parse(src).unwrap();
    let rhs = relation.rhs.as_ref().map(|(_, rhs)| rhs);
    let mut lowering = Lowering::new(scope, &relation.lhs, rhs);
    let result = lowering
        .relation(&relation.lhs, relation.rhs.as_ref(), 0..src.len())
        .map_err(Fault::into_diagnostic);
    Trace {
        result,
        lowerings: lowering.lowerings.clone(),
        peak_cached: lowering.peak_cached,
        cache_empty: lowering
            .cache
            .iter()
            .all(|slot| matches!(slot, Slot::Pending)),
    }
}

fn traced(decls: &[(&str, &str)], src: &str) -> Trace {
    with_named(decls, |scope| trace_in(scope, src))
}

/// `e0 = base` and `ek = @e(k-1) + @e(k-1)` for `k` up to `levels`.
fn doubling_chain(base: &str, levels: usize) -> Vec<(String, String)> {
    let mut chain = vec![("e0".to_owned(), base.to_owned())];
    for k in 1..=levels {
        chain.push((
            format!("e{k}"),
            format!("@e{prev} + @e{prev}", prev = k - 1),
        ));
    }
    chain
}

fn borrowed(decls: &[(String, String)]) -> Vec<(&str, &str)> {
    decls
        .iter()
        .map(|(name, source)| (name.as_str(), source.as_str()))
        .collect()
}

/// derived: `e = h0 + 2 * h1`, so `3 * @e` gives `h0` the scale itself, `3.0`, and `h1`
/// `3.0 * 2.0 = 6.0`; the upper endpoint is `1.0`; `-@e` is `[-1.0, -2.0]`; `s = @a * h0`,
/// so `2 * @s` gives `h0` the program `2, a, mul`.
#[test]
fn a_reference_inlines_its_body_scaled() {
    let decls = [
        ("e", "hydro_generation(0) + 2 * hydro_generation(1)"),
        ("s", "@a * hydro_generation(0)"),
    ];
    let src = "3 * @e <= 1";
    let lowered = named_relation(&decls, src);
    assert_eq!(
        literal_bits_of(src, lowered.terms),
        [(hg(0), 3.0_f64.to_bits()), (hg(1), 6.0_f64.to_bits())]
    );
    assert_eq!(
        literal_bits(lowered.upper.as_ref()),
        Some(1.0_f64.to_bits())
    );
    assert_eq!(lowered.lower, None);

    assert_eq!(
        literal_bits_of("-@e", named_relation(&decls, "-@e").terms),
        [(hg(0), (-1.0_f64).to_bits()), (hg(1), (-2.0_f64).to_bits())]
    );
    assert_eq!(
        named_relation(&decls, "2 * @s").terms,
        [(hg(0), lit(2.0) * a())]
    );
}

/// derived: `e = h0 + 5`, so `@e <= 7` keeps `h0` at `1.0` and targets the upper endpoint
/// `7.0 - 5.0 = 2.0`; `@e >= @p` targets the lower endpoint with the program
/// `0, p, add, 5, sub`.
#[test]
fn an_affine_body_moves_its_constant_to_the_bound() {
    let decls = [("e", "hydro_generation(0) + 5")];
    let src = "@e <= 7";
    let lowered = named_relation(&decls, src);
    assert_eq!(
        literal_bits_of(src, lowered.terms),
        [(hg(0), 1.0_f64.to_bits())]
    );
    assert_eq!(
        literal_bits(lowered.upper.as_ref()),
        Some(2.0_f64.to_bits())
    );
    assert_eq!(lowered.lower, None);

    let parameter = named_relation(&decls, "@e >= @p");
    assert_eq!(parameter.lower, Some(lit(0.0) + p() - lit(5.0)));
    assert_eq!(parameter.upper, None);
}

#[test]
fn an_affine_body_in_an_operator_free_expression_is_rejected() {
    let decls = [
        ("e", "hydro_generation(0) + 5"),
        ("f", "hydro_generation(0)"),
    ];
    for src in ["@e", "2 * @e + hydro_generation(1)"] {
        assert_eq!(
            named_rejection(&decls, src),
            format!(
                "`{src}`: an expression without a relational operator cannot have a constant \
                 part; put the bound in generic_constraint_bounds.parquet or use <=, >= or =="
            ),
            "{src:?}"
        );
    }
    assert_eq!(named_relation(&decls, "@f").terms, [(hg(0), lit(1.0))]);
}

/// derived: `outer = 2 * @middle`, `middle = @inner + h1`, `inner = h0`, so `@outer` is
/// `2 * (h0 + h1)`, `[2.0, 2.0]`; `@outer - @inner >= 0` gives `h0` `2.0 - 1.0 = 1.0` and
/// keeps `h1` at `2.0`.
#[test]
fn composition_resolves_transitively() {
    let decls = [
        ("outer", "2 * @middle"),
        ("middle", "@inner + hydro_generation(1)"),
        ("inner", "hydro_generation(0)"),
    ];
    let lowered = named_relation(&decls, "@outer");
    assert_eq!(
        literal_bits_of("@outer", lowered.terms),
        [(hg(0), 2.0_f64.to_bits()), (hg(1), 2.0_f64.to_bits())]
    );

    let src = "@outer - @inner >= 0";
    let shared = traced(&decls, src);
    assert_eq!(
        literal_bits_of(src, shared.result.unwrap().terms),
        [(hg(0), 1.0_f64.to_bits()), (hg(1), 2.0_f64.to_bits())]
    );
    assert_eq!(shared.lowerings, [1, 1, 1]);
    assert!(shared.cache_empty);
}

#[test]
fn a_body_error_is_reported_at_its_declaration() {
    assert_eq!(
        declaration_rejection(&[("e", "hydro_generation(0) * hydro_generation(1)")]),
        (
            0,
            "`hydro_generation(0) * hydro_generation(1)`: product of LP variables is not \
             linear"
                .to_owned()
        )
    );
    assert_eq!(
        declaration_rejection(&[("top", "@mid"), ("mid", "@missing + hydro_generation(0)")]),
        (
            1,
            "`@missing`: unknown name \"@missing\": no scalar parameter or named expression \
             with this name was declared"
                .to_owned()
        )
    );
    assert_eq!(
        declaration_rejection(&[("r", "hydro_generation(0) <= 1")]),
        (
            0,
            "`hydro_generation(0) <= 1`: a named expression cannot contain a relational \
             operator"
                .to_owned()
        )
    );
    assert_eq!(
        declaration_rejection(&[
            ("e", "hydro_generation(1)"),
            ("bad", "hydro_turbined(5.5) + @e")
        ]),
        (
            1,
            "`hydro_turbined(5.5)`: entity_id must be a non-negative integer, got 5.5 in \
             variable \"hydro_turbined\""
                .to_owned()
        )
    );
    assert_eq!(
        declaration_rejection(&[("pair", "line_exchange(source_bus=1, target_bus=2)")]),
        (
            0,
            "`line_exchange(source_bus=1, target_bus=2)`: no line connects buses source_bus=1 \
             and target_bus=2"
                .to_owned()
        )
    );
    assert_eq!(
        declaration_rejection(&[
            ("f", "hydro_generation(0)"),
            ("g", "@f * hydro_generation(1)")
        ]),
        (
            1,
            "`@f * hydro_generation(1)`: product of LP variables is not linear (\"@f\" is the \
             named expression at expressions[0], which contains LP variables)"
                .to_owned()
        )
    );
}

#[test]
fn an_unreferenced_body_with_a_misspelled_variable_is_rejected_at_its_declaration() {
    assert_eq!(
        declaration_rejection(&[
            ("e", "hydro_generation(0)"),
            ("typo", "2 * hydro_generaton(1) + @e")
        ]),
        (
            1,
            "`hydro_generaton(1)`: unknown variable name \"hydro_generaton\": not a recognised \
             LP variable name"
                .to_owned()
        )
    );
}

#[test]
fn declare_reports_the_first_failing_declaration_in_file_order() {
    assert_eq!(
        declaration_rejection(&[
            ("lowering", "invalid_var(0)"),
            ("syntax", "hydro_generation(")
        ]),
        (
            0,
            "`invalid_var(0)`: unknown variable name \"invalid_var\": not a recognised LP \
             variable name"
                .to_owned()
        )
    );
}

#[test]
fn an_unreferenced_body_dividing_by_zero_is_rejected_at_its_declaration() {
    assert_eq!(
        declaration_rejection(&[
            ("e", "hydro_generation(0)"),
            ("zero", "hydro_generation(1) / (1 - 1)")
        ]),
        (
            1,
            "`hydro_generation(1) / (1 - 1)`: division by zero".to_owned()
        )
    );
    assert_eq!(
        declaration_rejection(&[("e", "hydro_generation(0)"), ("zero", "2 * @e / 0")]),
        (1, "`2 * @e / 0`: division by zero".to_owned())
    );
}

/// derived: `1e308 * @e` with `e = 1e308 * h0` folds `1e308 * 1e308` to `+inf`, in the
/// constraint and, through `f = 1e308 * @e`, inside the body of `f`.
#[test]
fn a_composition_error_names_the_declaration() {
    let note = "(\"@e\" is the named expression at expressions[0], which contains LP variables)";
    let decls = [("e", "hydro_generation(1)")];
    assert_eq!(
        named_rejection(&decls, "@e * hydro_generation(0) <= 1"),
        format!("`@e * hydro_generation(0)`: product of LP variables is not linear {note}")
    );
    assert_eq!(
        named_rejection(&decls, "hydro_generation(0) / @e >= 0"),
        format!(
            "`hydro_generation(0) / @e`: division by an expression containing LP variables \
             {note}"
        )
    );
    assert_eq!(
        named_rejection(&[("e", "1e308 * hydro_generation(0)")], "1e308 * @e <= 1"),
        "`1e308 * @e`: evaluates to a non-finite value (inf)"
    );

    let overflowing = [
        ("e", "1e308 * hydro_generation(0)"),
        ("f", "1e308 * @e"),
        ("g", "@f + hydro_generation(1)"),
    ];
    let composing = "(composing \"@f\", the named expression at expressions[1])";
    assert_eq!(
        named_rejection(&overflowing, "@f <= 1"),
        format!("`@f`: evaluates to a non-finite value (inf) {composing}")
    );
    assert_eq!(
        named_rejection(&overflowing, "hydro_generation(2) + @g >= 0"),
        format!("`@g`: evaluates to a non-finite value (inf) {composing}")
    );
    assert_eq!(
        named_rejection(&[("e", "@e + hydro_generation(0)")], "@e <= 1"),
        "`@e`: named-expression reference cycle detected"
    );
    assert_eq!(
        named_rejection(
            &[("ping", "@pong + hydro_generation(0)"), ("pong", "@ping")],
            "hydro_generation(1) + @ping <= 1"
        ),
        "`@ping`: named-expression reference cycle detected"
    );
}

/// derived: `e = h1 + 2 * h2`; `@p * @e` gives `h1` the scale itself, `p`, and `h2`
/// `p, 2, mul`, at `p = 3` equal to `3.0` and `6.0`; `@e * @p` gives `h2` `2, p, mul`; in
/// `h0 >= @p * @e` the right side moves left as `p, neg` and `p, 2, mul, neg` and the lower
/// endpoint is `+0.0`. Values cannot tell `p, 2, mul` from `2, p, mul`, so the programs are
/// also compared with constructor-built ones.
#[test]
fn a_parameter_may_scale_a_named_expression() {
    let decls = [("e", "hydro_generation(1) + 2 * hydro_generation(2)")];
    let scaled = named_relation(&decls, "@p * @e");
    assert_eq!(scaled.terms, [(hg(1), p()), (hg(2), p() * lit(2.0))]);
    assert_ne!(scaled.terms[1].1, lit(2.0) * p());
    let values: Vec<u64> = scaled
        .terms
        .iter()
        .map(|(_, c)| c.evaluate(|_| 3.0).to_bits())
        .collect();
    assert_eq!(values, [3.0_f64.to_bits(), 6.0_f64.to_bits()]);

    assert_eq!(
        named_relation(&decls, "@e * @p").terms,
        [(hg(1), p()), (hg(2), lit(2.0) * p())]
    );

    let moved = named_relation(&decls, "hydro_generation(0) >= @p * @e");
    assert_eq!(
        moved.terms,
        [(hg(0), lit(1.0)), (hg(1), -p()), (hg(2), -(p() * lit(2.0)))]
    );
    assert_eq!(literal_bits(moved.lower.as_ref()), Some(0.0_f64.to_bits()));
}

/// derived: `base = @p * h0` reaches `top` through `left` and `right`, so `@top` gives `h0`
/// the program `p, p, add`; `@base + @top` gives it `p, p, p, add, add`.
#[test]
fn each_body_is_lowered_once() {
    let decls = [
        ("top", "@left + @right"),
        ("left", "@base"),
        ("right", "@base"),
        ("base", "@p * hydro_generation(0)"),
    ];
    let graph = with_named(&decls, |scope| {
        scope
            .named
            .reference_graph()
            .into_iter()
            .map(|(name, refs)| (name.to_owned(), refs.join(" ")))
            .collect::<Vec<_>>()
    });
    assert_eq!(
        graph,
        [
            ("top".to_owned(), "left right".to_owned()),
            ("left".to_owned(), "base".to_owned()),
            ("right".to_owned(), "base".to_owned()),
            ("base".to_owned(), String::new())
        ]
    );

    let diamond = traced(&decls, "@top <= 1");
    assert_eq!(diamond.result.unwrap().terms, [(hg(0), p() + p())]);
    assert_eq!(diamond.lowerings, [1, 1, 1, 1]);
    assert_eq!(diamond.peak_cached, 2);
    assert!(diamond.cache_empty);

    let shared = traced(&decls, "@base + @top <= 1");
    let terms = shared.result.unwrap().terms;
    assert_eq!(terms, [(hg(0), p() + (p() + p()))]);
    assert_eq!(terms[0].1.op_count(), 5);
    assert_eq!(shared.lowerings, [1, 1, 1, 1]);
    assert!(shared.cache_empty);
}

/// derived: every level doubles a literal, `1.0 + 1.0 = 2.0` up to `2^59 + 2^59 = 2^60`,
/// each exact.
#[test]
fn a_literal_doubling_chain_folds_to_one_term() {
    let chain = doubling_chain("hydro_generation(0)", 60);
    let decls = borrowed(&chain);
    let trace = traced(&decls, "@e60");
    let terms = trace.result.unwrap().terms;
    assert_eq!(terms.len(), 1);
    assert_eq!(terms[0].0, hg(0));
    assert_eq!(terms[0].1.op_count(), 1);
    assert_eq!(
        terms[0].1.as_literal().map(f64::to_bits),
        Some(2.0_f64.powi(60).to_bits())
    );
    assert!(trace.lowerings.iter().all(|&n| n == 1));
    assert!(trace.peak_cached <= 2);
    assert!(trace.cache_empty);

    let graph = with_named(&decls, |scope| {
        let graph = scope.named.reference_graph();
        (graph[1].0.to_owned(), graph[1].1.join(" "))
    });
    assert_eq!(graph, ("e1".to_owned(), "e0 e0".to_owned()));
}

/// derived: level `k` of the chain over `@a * h0` is `c, c, add` on level `k - 1`, so it
/// holds `2^(k+1) - 1` ops: `e15` holds 65535 and `e16` would hold 131071, the first above
/// the budget; `@e15 + @e15` would also hold 131071.
#[test]
fn a_parameter_doubling_chain_hits_the_budget_fast() {
    let chain = doubling_chain("@a * hydro_generation(0)", 60);
    let decls = borrowed(&chain);
    let trace = traced(&decls, "@e60 <= 1");
    assert_eq!(
        trace.result.unwrap_err().render("@e60 <= 1"),
        "`@e60`: named-expression reference \"@e60\" expands to more than 100000 coefficient \
         operations; this indicates an exponential reference pattern (e.g. a doubling chain \
         \"@e = @prev + @prev\")"
    );
    assert!(trace.lowerings[..=16].iter().all(|&n| n == 1));
    assert!(trace.lowerings[17..].iter().all(|&n| n == 0));

    let src = "@e15 + @e15 <= 1";
    assert_eq!(
        named_rejection(&decls, src),
        "`@e15 + @e15`: expression expands to more than 100000 coefficient operations"
    );
}

#[test]
fn an_unreferenced_doubling_chain_declares_fast() {
    let chain = doubling_chain("@a * hydro_generation(0)", 60);
    let decls = borrowed(&chain);
    let trace = traced(&decls, "hydro_generation(0) <= 1");
    assert_eq!(trace.result.unwrap().terms, [(hg(0), lit(1.0))]);
    assert_eq!(trace.lowerings.len(), 61);
    assert!(trace.lowerings.iter().all(|&n| n == 0));
    assert!(trace.cache_empty);
}

/// derived: 1,000 distinct unit coefficients materialize to `1.0` each; scaled by `@p` each
/// is `p` itself, 1,000 ops in all.
#[test]
fn a_large_flat_expression_is_under_the_budget() {
    let body = (0..1000)
        .map(|id| format!("hydro_generation({id})"))
        .collect::<Vec<_>>()
        .join(" + ");
    let decls = [("e", body.as_str())];
    let expected: Vec<(VariableRef, u64)> =
        (0..1000).map(|id| (hg(id), 1.0_f64.to_bits())).collect();

    let src = "@e <= 1";
    assert_eq!(
        literal_bits_of(src, named_relation(&decls, src).terms),
        expected
    );
    let direct = format!("{body} <= 1");
    assert_eq!(literal_bits_of(&direct, relation(&direct).terms), expected);

    let scaled = named_relation(&decls, "@p * @e <= 1").terms;
    assert_eq!(scaled.len(), 1000);
    assert!(
        scaled
            .iter()
            .all(|(_, c)| c.as_param() == Some(EntityId(10)))
    );
}

/// derived: `e200` sums 200 distinct unit coefficients, each `1.0` once materialized.
#[test]
fn a_linear_chain_keeps_at_most_two_bodies_cached() {
    let mut chain = vec![("e1".to_owned(), "hydro_generation(1)".to_owned())];
    for k in 2..=200 {
        chain.push((
            format!("e{k}"),
            format!("@e{} + hydro_generation({k})", k - 1),
        ));
    }
    let decls = borrowed(&chain);
    let src = "@e200 <= 1";
    let trace = traced(&decls, src);
    let expected: Vec<(VariableRef, u64)> =
        (1..=200).map(|id| (hg(id), 1.0_f64.to_bits())).collect();
    assert_eq!(literal_bits_of(src, trace.result.unwrap().terms), expected);
    assert!(trace.peak_cached <= 2, "peak {}", trace.peak_cached);
    assert!(trace.lowerings.iter().all(|&n| n == 1));
    assert!(trace.cache_empty);
}

/// derived: `e = h0 + 2 * h1` gives `h0` `1.0` and `h1` `2.0` under both constraints.
#[test]
fn an_evicted_body_is_lowered_again_for_the_next_constraint() {
    let decls = [("e", "hydro_generation(0) + 2 * hydro_generation(1)")];
    let src = "@e <= 1";
    let (first, second) = with_named(&decls, |scope| (trace_in(scope, src), trace_in(scope, src)));
    for trace in [&first, &second] {
        assert_eq!(trace.lowerings, [1]);
        assert!(trace.cache_empty);
    }
    let first = first.result.unwrap();
    assert_eq!(first, second.result.unwrap());
    assert_eq!(
        literal_bits_of(src, first.terms),
        [(hg(0), 1.0_f64.to_bits()), (hg(1), 2.0_f64.to_bits())]
    );
}

const CONDITIONAL_UNSUPPORTED: &str = "()` is not supported in this expression context";
const BRANCHES_DIFFER: &str =
    "() must read the same LP variables; they may differ only in coefficients and constants";

/// Asserts that `message` is `before`, then `if`, then `after`.
fn assert_around_conditional(message: &str, before: &str, after: &str) {
    assert_eq!(message, format!("{before}if{after}"));
}

#[test]
fn a_conditional_is_rejected_in_a_scalar_body() {
    for (src, call) in [
        ("if(@a<=1, 10, 20)", "if(@a<=1, 10, 20)"),
        ("2 * @a + if(@a<=1, 10, 20)", "if(@a<=1, 10, 20)"),
        (
            "-if(@a<=1, if(@b<=2, 1, 2), 20)",
            "if(@a<=1, if(@b<=2, 1, 2), 20)",
        ),
    ] {
        assert_around_conditional(
            &rendered_rejection(src),
            &format!("`{call}`: `"),
            CONDITIONAL_UNSUPPORTED,
        );
    }
}

#[test]
fn a_conditional_is_rejected_in_a_linear_form_and_a_relation() {
    let call = "if(@a<=1, 10, 20)";
    let before = format!("`{call}`: `");
    for src in [
        format!("hydro_storage(1) + {call}"),
        format!("2 * {call}"),
        format!("{call} * hydro_storage(1)"),
    ] {
        assert_around_conditional(&linear_rejection(&src), &before, CONDITIONAL_UNSUPPORTED);
    }
    for src in [
        format!("hydro_storage(1) <= {call}"),
        format!("{call} >= hydro_storage(1)"),
        format!("hydro_storage(1) + {call} == 5"),
    ] {
        assert_around_conditional(&relation_rejection(&src), &before, CONDITIONAL_UNSUPPORTED);
    }
}

/// derived: the references of `if(@n <= 1 & @m > 2, @m, -@n)` in source order are the
/// condition's `n`, `m`, then the branches' `m`, `n`.
#[test]
fn named_references_inside_a_conditional_are_collected_in_source_order() {
    let named = HashMap::from([("n".to_owned(), 0), ("m".to_owned(), 1)]);
    let expr = parse("if(@n <= 1 & @m > 2 | @x == 3, @m, -@n)")
        .unwrap()
        .lhs;
    let mut references = Vec::new();
    collect_named_references(&expr, &named, &mut references);
    assert_eq!(references, [0, 1, 1, 0]);
}

/// Runs `f` in a constraint scope where the loads of bus 3 and of bus 3 in block 1 are parameters
/// 20 and 21.
fn with_atoms<T>(f: impl FnOnce(&Scope) -> T) -> T {
    let params = params();
    let line_index = line_index();
    let named = NamedTable::default();
    let data_ids = BTreeMap::from([
        (
            DataAtom::BusLoadValue {
                bus_id: EntityId(3),
                block_id: None,
            },
            EntityId(20),
        ),
        (
            DataAtom::BusLoadValue {
                bus_id: EntityId(3),
                block_id: Some(1),
            },
            EntityId(21),
        ),
    ]);
    f(&Scope {
        params: &params,
        line_index: &line_index,
        named: &named,
        data_ids: Some(&data_ids),
    })
}

fn param(id: i32) -> ScalarExpr {
    ScalarExpr::param(EntityId(id))
}

/// derived: a data term times a decision variable is a coefficient program over the parameter,
/// `p20 * h14`, and a data term alone on a side is a constant.
#[test]
fn a_data_term_scales_a_variable_and_stands_alone_as_a_constant() {
    let relation = with_atoms(|scope| {
        lower_relation(
            "bus_load_value(3) * hydro_generation(14) <= bus_load_value(3)",
            scope,
        )
    })
    .unwrap();
    assert_eq!(relation.terms, [(hg(14), param(20))]);
    assert_eq!(relation.upper, Some(lit(0.0) + param(20)));
}

#[test]
fn a_data_term_needs_data_ids_and_a_listed_atom() {
    let without =
        with_scope(|scope| lower_relation("hydro_generation(14) >= bus_load_value(3)", scope));
    assert_eq!(
        without.unwrap_err().message,
        "\"bus_load_value\" reads scenario data, which a scalar parameter cannot read"
    );
    assert!(lower_scalar_body("bus_load_value(3)", &params()).is_err());

    let unlisted =
        with_atoms(|scope| lower_relation("hydro_generation(14) >= bus_load_value(7)", scope));
    assert_eq!(
        unlisted.unwrap_err().message,
        "data term \"bus_load_value\" has no parameter id"
    );
}

const FIRST_SELECTOR: i32 = 30;

/// Runs `f` with the data atoms of a constraint scope: parameters 20 (load of bus 3) and 21 (load
/// of bus 7).
fn with_data_ids<T>(
    f: impl FnOnce(&BTreeMap<DataAtom, EntityId>, &HashMap<String, EntityId>, &LineBusPairIndex) -> T,
) -> T {
    let params = params();
    let line_index = line_index();
    let data_ids = BTreeMap::from([
        (
            DataAtom::BusLoadValue {
                bus_id: EntityId(3),
                block_id: None,
            },
            EntityId(20),
        ),
        (
            DataAtom::BusLoadValue {
                bus_id: EntityId(7),
                block_id: None,
            },
            EntityId(21),
        ),
    ]);
    f(&data_ids, &params, &line_index)
}

fn with_constraint_scope<T>(decls: &[(&str, &str)], f: impl FnOnce(&Scope) -> T) -> T {
    with_data_ids(|data_ids, params, line_index| {
        let named = match NamedTable::declare(decls, params, line_index, data_ids) {
            Ok(named) => named,
            Err((j, d)) => panic!("declaration {j}: {}", d.render(decls[j].1)),
        };
        f(&Scope {
            params,
            line_index,
            named: &named,
            data_ids: Some(data_ids),
        })
    })
}

fn constraint_result(decls: &[(&str, &str)], src: &str) -> Result<LoweredRelation, Diagnostic> {
    with_constraint_scope(decls, |scope| {
        lower_relation_from(src, scope, EntityId(FIRST_SELECTOR))
    })
}

fn constraint_relation(decls: &[(&str, &str)], src: &str) -> LoweredRelation {
    constraint_result(decls, src).unwrap_or_else(|d| panic!("{}", d.render(src)))
}

fn constraint_rejection(decls: &[(&str, &str)], src: &str) -> String {
    constraint_result(decls, src).unwrap_err().render(src)
}

fn hg_block(id: i32, block: usize) -> VariableRef {
    VariableRef::HydroGeneration {
        hydro_id: EntityId(id),
        block_id: Some(block),
        bus_id: None,
    }
}

fn compare(lhs: ScalarExpr, op: CompareOp, rhs: ScalarExpr) -> ResolvedCondition {
    ResolvedCondition::Compare { lhs, op, rhs }
}

fn selector(cond: ResolvedCondition, then: ScalarExpr, else_: ScalarExpr) -> ConditionalScalar {
    ConditionalScalar { cond, then, else_ }
}

#[test]
fn a_bare_name_is_rejected() {
    let rejection = with_scope(|scope| lower_relation("FNS <= 1", scope))
        .unwrap_err()
        .render("FNS <= 1");
    assert_eq!(rejection, "`FNS`: expected '(' after variable name \"FNS\"");
}

/// derived: only `if` opens a conditional. In `when(1 <= 2, 1, 2)` the comparison stands among
/// the arguments of a call, which the parser rejects, and a `when(...)` whose arguments parse is
/// a call the catalog does not know.
#[test]
fn only_if_opens_a_conditional() {
    assert_eq!(
        constraint_rejection(&[], "when(1 <= 2, 1, 2)"),
        "`<=`: a relational operator is only allowed at the top level"
    );
    assert_eq!(
        constraint_rejection(&[], "hydro_generation(14) <= when(1)"),
        "`when(1)`: unknown variable name \"when\": not a recognised LP variable name"
    );
}

#[test]
fn an_entity_term_without_arguments_is_rejected_at_lowering() {
    let src = "hydro_generation() <= 1";
    assert!(parse(src).is_ok());
    let rejection = with_scope(|scope| lower_relation(src, scope))
        .unwrap_err()
        .render(src);
    assert!(
        rejection.contains("variable \"hydro_generation\" has an empty argument list"),
        "{rejection}"
    );
}

/// derived: the bound is `0.0 + p30` and the selector compares the load of bus 7, parameter 21,
/// with 85000 and takes 100 or 200.
#[test]
fn a_conditional_bound_lowers_to_a_selector_parameter() {
    let relation = constraint_relation(
        &[],
        "hydro_generation(14) <= if(bus_load_value(7) <= 85000, 100, 200)",
    );
    assert_eq!(relation.terms, [(hg(14), lit(1.0))]);
    assert_eq!(relation.upper, Some(lit(0.0) + param(30)));
    assert_eq!(
        relation.selectors,
        [(
            EntityId(30),
            selector(
                compare(param(21), CompareOp::Le, lit(85000.0)),
                lit(100.0),
                lit(200.0)
            )
        )]
    );
}

#[test]
fn a_conditional_coefficient_lowers_to_a_selector_parameter() {
    let relation = constraint_relation(
        &[],
        "if(bus_load_value(7) <= 85000, 2, 3) * hydro_generation(14) <= 1",
    );
    assert_eq!(relation.terms, [(hg(14), param(30))]);
    assert_eq!(
        relation.selectors,
        [(
            EntityId(30),
            selector(
                compare(param(21), CompareOp::Le, lit(85000.0)),
                lit(2.0),
                lit(3.0)
            )
        )]
    );
}

/// derived: the branches `h14 + 1` and `h14 + 2` share the variable and the unit coefficient, so
/// only the constant is a selector, and `h14 + p30 <= 10` has the bound `10 - p30`.
#[test]
fn conditional_branches_over_the_same_variables_select_only_the_difference() {
    let relation = constraint_relation(
        &[],
        "if(bus_load_value(7) <= 1, hydro_generation(14) + 1, hydro_generation(14) + 2) <= 10",
    );
    assert_eq!(relation.terms, [(hg(14), lit(1.0))]);
    assert_eq!(relation.upper, Some(lit(10.0) - param(30)));
    assert_eq!(
        relation.selectors,
        [(
            EntityId(30),
            selector(
                compare(param(21), CompareOp::Le, lit(1.0)),
                lit(1.0),
                lit(2.0)
            )
        )]
    );

    let coefficient = constraint_relation(
        &[],
        "if(bus_load_value(7) <= 1, 2 * hydro_generation(14), 3 * hydro_generation(14)) >= 0",
    );
    assert_eq!(coefficient.terms, [(hg(14), param(30))]);
    assert_eq!(coefficient.selectors.len(), 1);

    let same = constraint_relation(
        &[],
        "if(bus_load_value(7) <= 1, 2 * hydro_generation(14), 2 * hydro_generation(14)) >= 0",
    );
    assert_eq!(same.terms, [(hg(14), lit(2.0))]);
    assert!(same.selectors.is_empty());
}

#[test]
fn conditional_branches_over_different_variables_are_rejected() {
    for src in [
        "if(bus_load_value(7) <= 1, hydro_generation(14), 0) <= 1",
        "if(bus_load_value(7) <= 1, 0, hydro_generation(14)) <= 1",
        "if(bus_load_value(7) <= 1, hydro_generation(14), hydro_generation(14, 1)) <= 1",
        "if(bus_load_value(7) <= 1, hydro_generation(14) + hydro_generation(14, 1), \
         hydro_generation(14)) <= 1",
    ] {
        let rejection = constraint_rejection(&[], src);
        let message = rejection
            .split_once("`: ")
            .map_or("", |(_, message)| message);
        assert_around_conditional(message, "the branches of ", BRANCHES_DIFFER);
    }
}

#[test]
fn a_condition_that_reads_an_lp_variable_is_rejected() {
    let decls = [("RSUL", "hydro_generation(14) + 1")];
    for (src, fragment) in [
        (
            "if(hydro_generation(14) <= 1, 1, 2) <= 5",
            "hydro_generation(14)",
        ),
        (
            "if(1 <= @RSUL & bus_load_value(7) <= 2, 1, 2) <= 5",
            "@RSUL",
        ),
    ] {
        assert_around_conditional(
            &constraint_rejection(&decls, src),
            &format!("`{fragment}`: an "),
            "() condition may read only data, not an LP variable",
        );
    }
}

const CAPACITY_IN_CONDITION: &str = "a condition cannot read hydro_availability: it is the \
                                      capacity of a plant, resolved per stage and block, not a \
                                      scenario value";

fn activation_rejection(decls: &[(&str, &str)], src: &str) -> String {
    with_constraint_scope(decls, |scope| {
        lower_activation_condition(&parse_condition(src).unwrap(), scope)
    })
    .unwrap_err()
    .render(src)
}

/// The fault sits on the call, not on the side that holds it; outside a condition the term keeps
/// its meaning.
#[test]
fn a_condition_naming_a_plant_capacity_says_so() {
    let capacity = CAPACITY_IN_CONDITION;
    let activation = |src: &str| {
        with_constraint_scope(&[], |scope| {
            lower_activation_condition(&parse_condition(src).unwrap(), scope)
        })
        .unwrap_err()
        .render(src)
    };
    for (rejection, call) in [
        (
            activation("hydro_availability(14) > 10"),
            "hydro_availability(14)",
        ),
        (
            activation("bus_load_value(7) > 1 & 2 * hydro_availability(14, 2) > 10"),
            "hydro_availability(14, 2)",
        ),
        (
            constraint_rejection(
                &[],
                "hydro_generation(14) <= if(2 * hydro_availability(14) > 1, 1, 2)",
            ),
            "hydro_availability(14)",
        ),
    ] {
        assert_eq!(rejection, format!("`{call}`: {capacity}"));
        assert!(!rejection.contains("LP variable"), "{rejection}");
    }

    assert_eq!(
        rendered_rejection("2 * hydro_availability(14)"),
        "`hydro_availability(14)`: an LP variable cannot appear in a scalar-parameter expression"
    );
    let relation = constraint_relation(
        &[],
        "hydro_generation(14) <= if(bus_load_value(7) <= 1, 2 * hydro_availability(14), \
         hydro_availability(14))",
    );
    let variables: Vec<_> = relation.terms.into_iter().map(|(v, _)| v).collect();
    assert_eq!(
        variables,
        [
            hg(14),
            VariableRef::HydroAvailability {
                hydro_id: EntityId(14),
                block_id: None,
            },
        ]
    );
}

/// An unknown name is refused in a condition as it is in an expression, not as an LP variable.
#[test]
fn an_unknown_name_in_a_condition_is_refused_as_it_is_in_an_expression() {
    for call in ["foo(3)", "foo()", "foo(3, 1)"] {
        let body = constraint_rejection(&[], &format!("hydro_generation(14) + {call} <= 1"));
        let message = body
            .strip_prefix(&format!("`{call}`: "))
            .unwrap_or_else(|| panic!("{body}"));
        for rejection in [
            activation_rejection(&[], &format!("{call} > 1")),
            activation_rejection(&[], &format!("bus_load_value(7) > 1 & 2 * {call} > 1")),
            constraint_rejection(
                &[],
                &format!("hydro_generation(14) <= if({call} > 1, 1, 2)"),
            ),
        ] {
            assert_eq!(rejection, format!("`{call}`: {message}"));
        }
    }
    let unknown = "`foo(3)`: unknown variable name \"foo\": not a recognised LP variable name";
    assert_eq!(activation_rejection(&[], "foo(3) > 1"), unknown);
    assert_eq!(rendered_rejection("@a * foo(3)"), unknown);
}

/// A condition side is faulted at its first term in source order that a condition cannot read,
/// as an `if()` condition and as an activation condition alike: a capacity before a variable or
/// an unknown name, whatever follows it, and a variable, a named expression reading one, or an
/// unknown name before a capacity.
#[test]
fn a_condition_side_is_faulted_at_its_first_refused_term() {
    let decls = [("RSUL", "hydro_generation(14) + 1")];
    let capacity = format!("`hydro_availability(14)`: {CAPACITY_IN_CONDITION}");
    let in_if = |lhs: &str| {
        let src = format!("hydro_generation(14) <= if({lhs} > 1, 1, 2)");
        constraint_rejection(&decls, &src)
    };
    let in_activation = |lhs: &str| activation_rejection(&decls, &format!("{lhs} > 1"));
    let side_reads_a_variable =
        |lhs: &str| format!("`{lhs}`: an if() condition may read only data, not an LP variable");
    let unknown = "`foo(1)`: unknown variable name \"foo\": not a recognised LP variable name";

    for lhs in [
        "hydro_availability(14) + hydro_generation(14)",
        "hydro_availability(14) * hydro_generation(14)",
        "2 / hydro_availability(14)",
        "hydro_availability(14) + foo(1)",
        "hydro_availability(14) + @RSUL",
    ] {
        assert_eq!(in_if(lhs), capacity, "{lhs}");
        assert_eq!(in_activation(lhs), capacity, "{lhs}");
    }
    for (lhs, activation) in [
        (
            "hydro_generation(14) + hydro_availability(14)",
            "`hydro_generation(14)`: an LP variable cannot appear in a scalar-parameter \
             expression",
        ),
        (
            "2 * hydro_generation(14) - 3 * hydro_availability(14, 2)",
            "`hydro_generation(14)`: an LP variable cannot appear in a scalar-parameter \
             expression",
        ),
        (
            "@RSUL + hydro_availability(14)",
            "`@RSUL`: unknown scalar parameter \"@RSUL\"; a named expression cannot appear in a \
             scalar-parameter expression",
        ),
    ] {
        assert_eq!(in_if(lhs), side_reads_a_variable(lhs), "{lhs}");
        assert_eq!(in_activation(lhs), activation, "{lhs}");
    }
    let lhs = "foo(1) + hydro_availability(14)";
    assert_eq!(in_if(lhs), unknown);
    assert_eq!(in_activation(lhs), unknown);

    let nested = "if(hydro_generation(14) > 0, 1, 2) + hydro_availability(14)";
    assert_eq!(in_if(nested), side_reads_a_variable("hydro_generation(14)"));
    assert_eq!(
        in_activation(nested),
        "`hydro_generation(14)`: an LP variable cannot appear in a scalar-parameter expression"
    );
}

/// derived: the inner `if()` is lowered before the outer one, so it takes id 30 and the outer
/// selector, id 31, takes it as its `then`.
#[test]
fn nested_conditionals_number_the_inner_selector_first() {
    let relation = constraint_relation(
        &[],
        "hydro_generation(14) <= if(bus_load_value(7) <= 1, if(bus_load_value(7) <= 2, 3, 4), 5)",
    );
    assert_eq!(relation.upper, Some(lit(0.0) + param(31)));
    assert_eq!(
        relation.selectors,
        [
            (
                EntityId(30),
                selector(
                    compare(param(21), CompareOp::Le, lit(2.0)),
                    lit(3.0),
                    lit(4.0)
                )
            ),
            (
                EntityId(31),
                selector(
                    compare(param(21), CompareOp::Le, lit(1.0)),
                    param(30),
                    lit(5.0)
                )
            ),
        ]
    );
}

/// derived: `&` binds tighter than `|`; the loads of buses 3 and 7 are parameters 20 and 21, and
/// `@b` and `@a` are parameters 8 and 7.
#[test]
fn a_condition_reads_loads_data_terms_and_parameters() {
    let relation = constraint_relation(
        &[],
        "hydro_generation(14) <= if(@b > 1 & bus_load_value(7) <= @a | bus_load_value(3) == 3, 1, 2)",
    );
    let expected = ResolvedCondition::Or(
        Box::new(ResolvedCondition::And(
            Box::new(compare(param(8), CompareOp::Gt, lit(1.0))),
            Box::new(compare(param(21), CompareOp::Le, param(7))),
        )),
        Box::new(compare(param(20), CompareOp::Eq, lit(3.0))),
    );
    assert_eq!(relation.selectors.len(), 1);
    assert_eq!(relation.selectors[0].1.cond, expected);
}

#[test]
fn selector_ids_run_out_at_the_largest_id() {
    let one = "hydro_generation(14) <= if(bus_load_value(7) <= 1, 1, 2)";
    let two = "hydro_generation(14) <= if(bus_load_value(7) <= 1, 1, 2) + if(bus_load_value(7) <= 2, 3, 4)";
    let lower_from_max = |src: &str| {
        with_constraint_scope(&[], |scope| {
            lower_relation_from(src, scope, EntityId(i32::MAX))
        })
    };
    assert_eq!(
        lower_from_max(one).unwrap().selectors[0].0,
        EntityId(i32::MAX)
    );
    assert_around_conditional(
        &lower_from_max(two).unwrap_err().message,
        "the scalar-parameter ids leave no room for the ",
        "() selectors of this constraint",
    );
}

/// derived: the body `E` is lowered once, so its selector is created once and both uses read
/// parameter 30: `@E + @E` is `2 * h14 + (p30 + p30)` and the bound is `10 - (p30 + p30)`.
#[test]
fn a_named_expression_with_a_conditional_creates_its_selector_once() {
    let relation = constraint_relation(
        &[(
            "E",
            "if(bus_load_value(7) <= 1, hydro_generation(14) + 1, hydro_generation(14) + 2)",
        )],
        "@E + @E <= 10",
    );
    assert_eq!(relation.terms, [(hg(14), lit(2.0))]);
    assert_eq!(relation.upper, Some(lit(10.0) - (param(30) + param(30))));
    assert_eq!(relation.selectors.len(), 1);
}

/// derived: `restriction_405_formula` is `h14 - 5` and `FNS` is `h14`, against `2 * h14`, so the
/// coefficient selects 1 or 2 (parameter 30) and, for `restriction_405_formula`, the constant
/// selects -5 or 0 (parameter 31), which `@E <= 10` moves to the bound `10 - p31`.
#[test]
fn a_named_body_whose_conditional_branch_reads_a_reference_is_accepted_at_declaration() {
    let cond = || compare(param(21), CompareOp::Le, lit(1.0));
    let restriction = constraint_relation(
        &[
            ("restriction_405_formula", "hydro_generation(14) - 5"),
            (
                "E",
                "if(bus_load_value(7) <= 1, @restriction_405_formula, 2 * hydro_generation(14))",
            ),
        ],
        "@E <= 10",
    );
    assert_eq!(restriction.terms, [(hg(14), param(30))]);
    assert_eq!(restriction.upper, Some(lit(10.0) - param(31)));
    assert_eq!(
        restriction.selectors,
        [
            (EntityId(30), selector(cond(), lit(1.0), lit(2.0))),
            (EntityId(31), selector(cond(), lit(-5.0), lit(0.0))),
        ]
    );

    let expression = constraint_relation(
        &[
            ("FNS", "hydro_generation(14)"),
            (
                "E",
                "if(bus_load_value(7) <= 1, @FNS, 2 * hydro_generation(14))",
            ),
        ],
        "@E <= 10",
    );
    assert_eq!(expression.terms, [(hg(14), param(30))]);
    assert_eq!(expression.upper, Some(lit(10.0)));
    assert_eq!(
        expression.selectors,
        [(EntityId(30), selector(cond(), lit(1.0), lit(2.0)))]
    );
}

#[test]
fn a_reference_in_a_conditional_branch_over_different_variables_is_rejected_at_the_constraint() {
    let composed =
        format!("{BRANCHES_DIFFER} (composing \"@E\", the named expression at expressions[1])");
    let restriction = [
        ("restriction_405_formula", "hydro_generation(14, 1) - 5"),
        (
            "E",
            "if(bus_load_value(7) <= 1, @restriction_405_formula, 2 * hydro_generation(14))",
        ),
    ];
    assert_around_conditional(
        &constraint_rejection(&restriction, "@E <= 10"),
        "`@E`: the branches of ",
        &composed,
    );
    let expression = [
        ("FNS", "hydro_generation(14, 1)"),
        (
            "E",
            "if(bus_load_value(7) <= 1, @FNS, 2 * hydro_generation(14))",
        ),
    ];
    assert_around_conditional(
        &constraint_rejection(&expression, "@E <= 10"),
        "`@E`: the branches of ",
        &composed,
    );
    let inline = "if(bus_load_value(7) <= 1, @FNS, 2 * hydro_generation(14)) <= 10";
    assert_around_conditional(
        &constraint_rejection(&expression, inline),
        "`if(bus_load_value(7) <= 1, @FNS, 2 * hydro_generation(14))`: the branches of ",
        BRANCHES_DIFFER,
    );
}

#[test]
fn a_named_body_whose_conditional_branches_read_different_variables_is_rejected_at_declaration() {
    with_data_ids(|data_ids, params, line_index| {
        let (j, diagnostic) = NamedTable::declare(
            &[
                ("FNS", "hydro_generation(14)"),
                (
                    "E",
                    "if(bus_load_value(7) <= 1, hydro_generation(14, 1), 2 * hydro_generation(14))",
                ),
            ],
            params,
            line_index,
            data_ids,
        )
        .unwrap_err();
        assert_eq!(j, 1);
        assert_around_conditional(&diagnostic.message, "the branches of ", BRANCHES_DIFFER);
    });
}

#[test]
fn a_reference_cycle_between_linear_expressions_is_rejected_at_lowering() {
    assert_eq!(
        constraint_rejection(
            &[
                ("A", "@B + hydro_generation(14)"),
                ("B", "@A + hydro_generation(14)")
            ],
            "@A <= 1"
        ),
        "`@A`: named-expression reference cycle detected"
    );
}

/// derived: `K` is `p20 - p8` (the load of bus 3 minus `@b`), so `h14 <= @K` has the bound
/// `0.0 + (p20 - p8)`, as with the body written in parentheses.
#[test]
fn a_body_without_a_variable_is_a_data_term_in_the_bound() {
    let decls = [("K", "bus_load_value(3) - @b")];
    let relation = constraint_relation(&decls, "hydro_generation(14) <= @K");
    assert_eq!(relation.terms, [(hg(14), lit(1.0))]);
    assert_eq!(relation.upper, Some(lit(0.0) + (param(20) - param(8))));
    assert_eq!(
        relation,
        constraint_relation(&decls, "hydro_generation(14) <= (bus_load_value(3) - @b)")
    );
}

/// derived: `C_NRed_cMMGD` is `K = (p20 - p9) - p8` and `@q` is parameter 11, so the constant of
/// `FNXG` is `(p11 - K) + p10` and `@FNXG <= 1000` has the bound `1000 - ((p11 - K) + p10)`; at
/// the load 100 or 250 with `p8..=p11` at 1, 2, 3 and 4 it is
/// `1000 - ((4 - ((100 - 2) - 1)) + 3)` or the same with 250, whatever the declaration order.
#[test]
fn a_data_only_expression_composes_into_another_as_written_in_place() {
    let expression = (
        "FNXG",
        "@Ger_NRed + @q - @C_NRed_cMMGD + @p - line_flow(source_bus=3, target_bus=7) \
         + line_flow(source_bus=7, target_bus=3)",
    );
    let generation = ("Ger_NRed", "hydro_generation(14) + hydro_generation(14, 1)");
    let data = ("C_NRed_cMMGD", "bus_load_value(3) - @c - @b");
    let relation = constraint_relation(&[expression, generation, data], "@FNXG <= 1000");

    let line = |direct| {
        let line_id = EntityId(5);
        if direct {
            VariableRef::LineDirect {
                line_id,
                block_id: None,
            }
        } else {
            VariableRef::LineReverse {
                line_id,
                block_id: None,
            }
        }
    };
    assert_eq!(
        relation.terms,
        [
            (hg(14), lit(1.0)),
            (hg_block(14, 1), lit(1.0)),
            (line(true), lit(-1.0)),
            (line(false), lit(1.0)),
        ]
    );
    let k = (param(20) - param(9)) - param(8);
    assert_eq!(
        relation.upper,
        Some(lit(1000.0) - ((param(11) - k) + param(10)))
    );
    let bound = relation.upper.as_ref().unwrap();
    for load in [100.0, 250.0] {
        let value = bound.evaluate(|id| match id.0 {
            20 => load,
            8..=11 => f64::from(id.0 - 7),
            other => panic!("unexpected parameter {other}"),
        });
        let expected = 1000.0_f64 - ((4.0 - ((load - 2.0) - 1.0)) + 3.0);
        assert_eq!(value.to_bits(), expected.to_bits(), "load {load}");
    }

    assert_eq!(
        constraint_relation(&[data, generation, expression], "@FNXG <= 1000"),
        relation
    );
    let inline = (
        "FNXG",
        "@Ger_NRed + @q - (bus_load_value(3) - @c - @b) + @p \
         - line_flow(source_bus=3, target_bus=7) + line_flow(source_bus=7, target_bus=3)",
    );
    assert_eq!(
        constraint_relation(&[inline, generation], "@FNXG <= 1000"),
        relation
    );
}

/// derived: `K` is `p20 - p8`; it scales and divides a form as its body would, so `@K * h14`
/// has the coefficient `p20 - p8` and `h14 / @K` the coefficient `1 / (p20 - p8)`. `K2`,
/// declared before the `K` it reads, is data-only too.
#[test]
fn a_data_only_expression_is_a_coefficient_and_a_linear_one_still_is_not() {
    let decls = [
        ("K2", "2 * @K"),
        ("K", "bus_load_value(3) - @b"),
        ("E", "hydro_generation(14)"),
    ];
    let scaled = constraint_relation(&decls, "@K * hydro_generation(14) <= 1");
    assert_eq!(scaled.terms, [(hg(14), param(20) - param(8))]);
    for (named, inline) in [
        (
            "@K2 * hydro_generation(14) <= 1",
            "(2 * (bus_load_value(3) - @b)) * hydro_generation(14) <= 1",
        ),
        (
            "@K * hydro_generation(14) <= 1",
            "(bus_load_value(3) - @b) * hydro_generation(14) <= 1",
        ),
        (
            "hydro_generation(14) / @K <= 1",
            "hydro_generation(14) / (bus_load_value(3) - @b) <= 1",
        ),
        (
            "@E * @K <= @K",
            "@E * (bus_load_value(3) - @b) <= (bus_load_value(3) - @b)",
        ),
    ] {
        assert_eq!(
            constraint_relation(&decls, named),
            constraint_relation(&decls, inline),
            "{named}"
        );
    }
    assert_eq!(
        constraint_rejection(&decls, "@E * @E <= 1"),
        "`@E * @E`: product of LP variables is not linear (\"@E\" is the named expression at \
         expressions[2], which contains LP variables)"
    );
    assert_eq!(
        constraint_rejection(&decls, "@K / @E <= 1"),
        "`@K / @E`: division by an expression containing LP variables (\"@E\" is the named \
         expression at expressions[2], which contains LP variables)"
    );
}

/// derived: a relation over data-only expressions alone has no term, as its inline form has.
#[test]
fn a_relation_over_data_only_expressions_keeps_its_inline_form() {
    let decls = [("K", "bus_load_value(3) - @b")];
    let relation = constraint_relation(&decls, "@K <= 5");
    assert!(relation.terms.is_empty());
    assert_eq!(
        relation,
        constraint_relation(&decls, "(bus_load_value(3) - @b) <= 5")
    );
}

/// derived: in an `if()` condition and in an activation condition `K` reads as `p20 - p8`; the
/// load of bus 7 is parameter 21.
#[test]
fn a_data_only_expression_reads_as_its_body_in_a_condition() {
    let decls = [("K", "bus_load_value(3) - @b")];
    let k = || param(20) - param(8);
    let relation = constraint_relation(&decls, "hydro_generation(14) <= if(@K > 0, 1, 2)");
    assert_eq!(
        relation.selectors,
        [(
            EntityId(30),
            selector(compare(k(), CompareOp::Gt, lit(0.0)), lit(1.0), lit(2.0))
        )]
    );
    assert_eq!(
        relation,
        constraint_relation(
            &decls,
            "hydro_generation(14) <= if(bus_load_value(3) - @b > 0, 1, 2)"
        )
    );

    let condition = parse_condition("@K > 0 | bus_load_value(7) <= 5").unwrap();
    let rule = with_constraint_scope(&decls, |scope| {
        lower_activation_condition(&condition, scope)
    });
    assert_eq!(
        rule.unwrap(),
        ResolvedCondition::Or(
            Box::new(compare(k(), CompareOp::Gt, lit(0.0))),
            Box::new(compare(param(21), CompareOp::Le, lit(5.0))),
        )
    );

    let src = "@S > 0";
    let rejection = with_constraint_scope(&[("S", "if(@b > 1, 1, 2)")], |scope| {
        lower_activation_condition(&parse_condition(src).unwrap(), scope)
    });
    assert_around_conditional(
        &rejection.unwrap_err().render(src),
        "`@S`: an activation condition cannot use ",
        "() (composing \"@S\", the named expression at expressions[0])",
    );
}

/// derived: `Z` folds to `0` only once `K` is composed into a constraint, so the declarations
/// are accepted and `1 / @Z` is a division by zero at the reference.
#[test]
fn a_data_only_reference_folds_only_once_a_constraint_composes_it() {
    let decls = [("K", "1 / @Z"), ("Z", "0")];
    assert_eq!(
        constraint_rejection(&decls, "hydro_generation(14) <= @K"),
        "`@K`: division by zero (composing \"@K\", the named expression at expressions[0])"
    );
    with_data_ids(|data_ids, params, line_index| {
        let (j, diagnostic) = NamedTable::declare(
            &[("K", "bus_load_value(3) / 0")],
            params,
            line_index,
            data_ids,
        )
        .unwrap_err();
        assert_eq!((j, diagnostic.message.as_str()), (0, "division by zero"));
    });
}

#[test]
fn a_reference_cycle_through_data_only_expressions_is_rejected_at_lowering() {
    assert_eq!(
        constraint_rejection(
            &[("A", "@B + bus_load_value(3)"), ("B", "@A - 1")],
            "hydro_generation(14) <= @A"
        ),
        "`@A`: named-expression reference cycle detected"
    );
}

/// derived: `D{k}` holds `2^(k+1) - 1` operations, so `D16`, at 131071, is the first past the
/// budget of 100000.
#[test]
fn a_doubling_chain_of_data_only_expressions_is_refused_by_the_budget() {
    let mut decls = vec![("D0".to_owned(), "bus_load_value(3)".to_owned())];
    for k in 1..=20 {
        decls.push((format!("D{k}"), format!("@D{0} + @D{0}", k - 1)));
    }
    let decls = borrowed(&decls);
    assert_eq!(
        constraint_rejection(&decls, "hydro_generation(14) <= @D20"),
        "`@D20`: named-expression reference \"@D20\" expands to more than 100000 coefficient \
         operations; this indicates an exponential reference pattern (e.g. a doubling chain \
         \"@e = @prev + @prev\")"
    );
}
