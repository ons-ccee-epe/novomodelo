//! Lowers a parsed expression to [`ScalarExpr`] programs. The program's constructors run in
//! source order, so a sub-expression without a parameter folds to one literal and nothing is
//! reassociated. A fold that divides by an exact zero or yields a non-finite value is rejected
//! here, where the source span is still known; a divisor or value that reads a parameter is
//! left to whoever supplies the parameter values.
//!
//! An expression that reads an LP variable lowers to a [`LinearForm`]: one coefficient program
//! per structurally distinct [`VariableRef`], merged in authoring order. Every node is typed
//! before anything folds, so `0 * x` still holds a term for `x`.
//!
//! A `@name` reference to a named expression lowers to the form of its body, each body lowered
//! at most once per constraint. Every form operation is refused before it builds a form whose
//! programs hold more than [`MAX_INLINED_OPS`] operations in total. Each body is also lowered
//! once at declaration with its own references read as empty forms, so a fault in its source is
//! reported there and only faults of the composition reach a constraint.

use std::collections::HashMap;
use std::mem;
use std::ops::Range;

use novomodelo_core::{EntityId, ScalarExpr, VariableRef, is_exact_zero};

use super::Diagnostic;
use super::catalog::{LineBusPairIndex, LinePairOrientation, variable_from_call};
use super::parser::{AddOp, Expr, ExprKind, MulOp, RelOp, parse};

/// The most operations the coefficient programs and constant of one form may hold together.
pub(crate) const MAX_INLINED_OPS: usize = 100_000;

/// The names an expression can read.
pub(crate) struct Scope<'a> {
    pub(crate) params: &'a HashMap<String, EntityId>,
    pub(crate) line_index: &'a LineBusPairIndex,
    pub(crate) named: &'a NamedTable,
}

/// The named expressions of one file, parsed, typed and checked, but expanded only where a
/// constraint references them.
#[derive(Debug)]
pub(crate) struct NamedTable {
    declarations: Vec<Declaration>,
    index: HashMap<String, usize>,
}

#[derive(Debug)]
struct Declaration {
    name: String,
    body: Expr,
    /// Declaration indices of the body's named references, in source order, repeats kept.
    references: Vec<usize>,
}

impl NamedTable {
    /// Parses and types each body, reading every named reference as linear, then lowers it with
    /// every named reference read as an empty form, so no body is expanded. Names must be unique
    /// and must not collide with `params`.
    ///
    /// # Errors
    ///
    /// Returns the index of the first declaration, in file order, whose body does not parse,
    /// contains a relational operator, reads an unknown name, is not linear, reads no LP
    /// variable, or fails to lower on its own (a variable call that does not resolve, a fold
    /// that divides by zero or is not finite), with a diagnostic on that body's source.
    pub(crate) fn declare(
        decls: &[(&str, &str)],
        params: &HashMap<String, EntityId>,
        line_index: &LineBusPairIndex,
    ) -> Result<NamedTable, (usize, Diagnostic)> {
        let index: HashMap<String, usize> = decls
            .iter()
            .enumerate()
            .map(|(j, (name, _))| ((*name).to_owned(), j))
            .collect();
        let mut declarations = Vec::with_capacity(decls.len());
        let mut unparsed = None;
        for (j, &(name, source)) in decls.iter().enumerate() {
            match declared_body(name, source, params, &index) {
                Ok(body) => {
                    let mut references = Vec::new();
                    collect_named_references(&body, &index, &mut references);
                    declarations.push(Declaration {
                        name: name.to_owned(),
                        body,
                        references,
                    });
                }
                Err(fault) => {
                    unparsed = Some((j, fault));
                    break;
                }
            }
        }
        let table = NamedTable {
            declarations,
            index,
        };
        {
            let scope = Scope {
                params,
                line_index,
                named: &table,
            };
            let mut lowering = Lowering::opaque(&scope);
            for (j, declaration) in table.declarations.iter().enumerate() {
                lowering
                    .form(&declaration.body)
                    .map_err(|fault| (j, fault.into_diagnostic()))?;
            }
        }
        unparsed.map_or(Ok(table), Err)
    }

    /// Each declaration in order with the named references of its body in source order,
    /// repeats kept.
    pub(crate) fn reference_graph(&self) -> Vec<(&str, Vec<&str>)> {
        self.declarations
            .iter()
            .map(|declaration| {
                let references = declaration
                    .references
                    .iter()
                    .map(|&k| self.declarations[k].name.as_str())
                    .collect();
                (declaration.name.as_str(), references)
            })
            .collect()
    }
}

fn declared_body(
    name: &str,
    source: &str,
    params: &HashMap<String, EntityId>,
    named: &HashMap<String, usize>,
) -> Result<Expr, Diagnostic> {
    let relation = parse(source)?;
    let whole = 0..source.len();
    if relation.rhs.is_some() {
        return Err(Diagnostic {
            span: whole,
            message: "a named expression cannot contain a relational operator".to_owned(),
        });
    }
    match type_of(&relation.lhs, params, named)? {
        Type::Linear => Ok(relation.lhs),
        Type::Scalar => Err(Diagnostic {
            span: whole,
            message: format!(
                "\"{name}\" contains no LP variable; declare it in generic_parameters.json as a \
                 kind \"expression\" parameter"
            ),
        }),
    }
}

fn collect_named_references(expr: &Expr, named: &HashMap<String, usize>, out: &mut Vec<usize>) {
    match &expr.kind {
        ExprKind::Number(_) | ExprKind::Variable(_) => {}
        ExprKind::Name(name) => out.extend(named.get(name)),
        ExprKind::Neg(operand) => collect_named_references(operand, named, out),
        ExprKind::Sum { head, tail } => {
            collect_named_references(head, named, out);
            for (_, operand) in tail {
                collect_named_references(operand, named, out);
            }
        }
        ExprKind::Product { head, tail } => {
            collect_named_references(head, named, out);
            for (_, operand) in tail {
                collect_named_references(operand, named, out);
            }
        }
    }
}

/// A coefficient of `None` is a unit coefficient, built as a program only when an operation or
/// [`LinearForm::into_terms`] needs one, so `s * x` stays `s` and not `s * 1`. A `None` constant
/// is no constant at all, which a relation tells apart from a literal `0`.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct LinearForm {
    pub(crate) terms: Vec<(VariableRef, Option<ScalarExpr>)>,
    pub(crate) constant: Option<ScalarExpr>,
}

impl LinearForm {
    /// Materializes unit coefficients and drops every term whose coefficient folded to an exact
    /// zero, keeping order; the constant is left to the caller.
    pub(crate) fn into_terms(self) -> Vec<(VariableRef, ScalarExpr)> {
        self.terms
            .into_iter()
            .map(|(variable, coefficient)| (variable, coefficient.unwrap_or_else(one)))
            .filter(|(_, coefficient)| !coefficient.as_literal().is_some_and(is_exact_zero))
            .collect()
    }
}

/// A constraint's variables, moved to the left, and the remainder on each endpoint its
/// relational operator targets.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct LoweredRelation {
    pub(crate) terms: Vec<(VariableRef, ScalarExpr)>,
    pub(crate) lower: Option<ScalarExpr>,
    pub(crate) upper: Option<ScalarExpr>,
}

/// Parses and lowers a constraint expression. A relation `L op R` moves every right-side
/// variable left with a flipped sign and targets its endpoint(s) with a remainder that starts
/// from `literal(0.0)`, adds `R.c` and subtracts `L.c`, each only when present, so it is never
/// `-0.0`; `R.c` alone or `-L.c` would keep or create one. An expression without an operator
/// targets no endpoint and cannot have a constant part; a named body's constant counts as its
/// side's.
pub(crate) fn lower_relation(src: &str, scope: &Scope) -> Result<LoweredRelation, Diagnostic> {
    let relation = parse(src)?;
    let rhs = relation.rhs.as_ref().map(|(_, rhs)| rhs);
    Lowering::new(scope, &relation.lhs, rhs)
        .relation(&relation.lhs, relation.rhs.as_ref(), 0..src.len())
        .map_err(Fault::into_diagnostic)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Type {
    Scalar,
    Linear,
}

/// Parses `src` and lowers it as the body of a scalar parameter, which has no relational
/// operator.
#[cfg_attr(not(test), expect(dead_code, reason = "no loader reads it yet"))]
pub(crate) fn lower_scalar_body(
    src: &str,
    params: &HashMap<String, EntityId>,
) -> Result<ScalarExpr, Diagnostic> {
    let relation = parse(src)?;
    if relation.rhs.is_some() {
        return Err(Diagnostic {
            span: 0..src.len(),
            message: "a scalar-parameter expression cannot contain a relational operator"
                .to_owned(),
        });
    }
    lower_scalar(&relation.lhs, params)
}

/// Lowers a tree that reads no LP variable; each chain step is checked over the source of its
/// partial result, from the chain's head to the step's operand.
fn lower_scalar(expr: &Expr, params: &HashMap<String, EntityId>) -> Result<ScalarExpr, Diagnostic> {
    match &expr.kind {
        ExprKind::Number(value) => Ok(ScalarExpr::literal(*value)),
        ExprKind::Name(name) => params
            .get(name)
            .map(|id| ScalarExpr::param(*id))
            .ok_or_else(|| Diagnostic {
                span: expr.span.clone(),
                message: format!(
                    "unknown scalar parameter \"@{name}\"; a named expression cannot appear in a \
                     scalar-parameter expression"
                ),
            }),
        ExprKind::Variable(_) => Err(Diagnostic {
            span: expr.span.clone(),
            message: "an LP variable cannot appear in a scalar-parameter expression".to_owned(),
        }),
        ExprKind::Neg(operand) => checked(-lower_scalar(operand, params)?, &expr.span),
        ExprKind::Sum { head, tail } => {
            let mut acc = lower_scalar(head, params)?;
            for (op, operand) in tail {
                let rhs = lower_scalar(operand, params)?;
                let step = head.span.start..operand.span.end;
                acc = match op {
                    AddOp::Add => checked(acc + rhs, &step)?,
                    AddOp::Sub => checked(acc - rhs, &step)?,
                };
            }
            Ok(acc)
        }
        ExprKind::Product { head, tail } => {
            let mut acc = lower_scalar(head, params)?;
            for (op, operand) in tail {
                let rhs = lower_scalar(operand, params)?;
                let step = head.span.start..operand.span.end;
                acc = match op {
                    MulOp::Mul => checked(acc * rhs, &step)?,
                    MulOp::Div => checked_div(acc, rhs, &step)?,
                };
            }
            Ok(acc)
        }
    }
}

/// Types `expr`, reading a named reference as linear without looking at its body.
fn type_of(
    expr: &Expr,
    params: &HashMap<String, EntityId>,
    named: &HashMap<String, usize>,
) -> Result<Type, Diagnostic> {
    match &expr.kind {
        ExprKind::Number(_) => Ok(Type::Scalar),
        ExprKind::Name(name) if params.contains_key(name) => Ok(Type::Scalar),
        ExprKind::Name(name) if named.contains_key(name) => Ok(Type::Linear),
        ExprKind::Name(name) => Err(Diagnostic {
            span: expr.span.clone(),
            message: format!(
                "unknown name \"@{name}\": no scalar parameter or named expression with this \
                 name was declared"
            ),
        }),
        ExprKind::Variable(_) => Ok(Type::Linear),
        ExprKind::Neg(operand) => type_of(operand, params, named),
        ExprKind::Sum { head, tail } => {
            let mut ty = type_of(head, params, named)?;
            for (_, operand) in tail {
                if type_of(operand, params, named)? == Type::Linear {
                    ty = Type::Linear;
                }
            }
            Ok(ty)
        }
        ExprKind::Product { head, tail } => {
            let mut ty = type_of(head, params, named)?;
            let mut first_linear = (ty == Type::Linear).then_some(&**head);
            for (op, operand) in tail {
                let rhs = type_of(operand, params, named)?;
                ty = match (op, ty, rhs) {
                    (MulOp::Mul, Type::Linear, Type::Linear) | (MulOp::Div, _, Type::Linear) => {
                        let culprit = match op {
                            MulOp::Mul => named_reference(operand, named)
                                .or_else(|| first_linear.and_then(|e| named_reference(e, named))),
                            MulOp::Div => named_reference(operand, named),
                        };
                        let mut error = not_linear(op, head.span.start..operand.span.end);
                        if let Some((name, j)) = culprit {
                            error.message = format!(
                                "{} (\"@{name}\" is the named expression at expressions[{j}], \
                                 which contains LP variables)",
                                error.message
                            );
                        }
                        return Err(error);
                    }
                    (_, Type::Scalar, Type::Scalar) => Type::Scalar,
                    _ => Type::Linear,
                };
                if rhs == Type::Linear && first_linear.is_none() {
                    first_linear = Some(operand);
                }
            }
            Ok(ty)
        }
    }
}

fn named_reference<'e>(expr: &'e Expr, named: &HashMap<String, usize>) -> Option<(&'e str, usize)> {
    match &expr.kind {
        ExprKind::Name(name) => named.get(name).map(|&j| (name.as_str(), j)),
        _ => None,
    }
}

fn not_linear(op: &MulOp, span: Range<usize>) -> Diagnostic {
    let message = match op {
        MulOp::Mul => "product of LP variables is not linear",
        MulOp::Div => "division by an expression containing LP variables",
    };
    Diagnostic {
        span,
        message: message.to_owned(),
    }
}

/// A rejection on its way out of a lowering.
enum Fault {
    /// Located in the source being lowered, which may be a named body.
    Local(Diagnostic),
    /// Already located on the constraint's outermost reference.
    Anchored(Diagnostic),
}

impl From<Diagnostic> for Fault {
    fn from(diagnostic: Diagnostic) -> Self {
        Fault::Local(diagnostic)
    }
}

impl Fault {
    fn into_diagnostic(self) -> Diagnostic {
        match self {
            Fault::Local(diagnostic) | Fault::Anchored(diagnostic) => diagnostic,
        }
    }
}

enum Slot {
    Pending,
    InProgress,
    Lowered(LinearForm),
}

/// The state of one constraint's lowering: the forms of the named bodies it has lowered and
/// still needs, indexed by declaration.
struct Lowering<'a> {
    scope: &'a Scope<'a>,
    cache: Vec<Slot>,
    /// Uses left per body, counted over the constraint and every reachable body; a form leaves
    /// the cache at its last use, which keeps a long chain from holding every link at once.
    remaining: Vec<usize>,
    /// The constraint's reference whose bodies are being lowered.
    origin: Option<(usize, Range<usize>)>,
    /// Every named reference lowers to an empty form, which checks a body without expanding it.
    opaque_references: bool,
    #[cfg(test)]
    lowerings: Vec<usize>,
    #[cfg(test)]
    peak_cached: usize,
}

impl<'a> Lowering<'a> {
    fn opaque(scope: &'a Scope<'a>) -> Self {
        let mut lowering = Self::with_remaining(scope, Vec::new());
        lowering.opaque_references = true;
        lowering
    }
    fn new(scope: &'a Scope<'a>, lhs: &Expr, rhs: Option<&Expr>) -> Self {
        let declarations = &scope.named.declarations;
        let mut remaining = vec![0; declarations.len()];
        let mut reached = vec![false; declarations.len()];
        let mut uses = Vec::new();
        collect_named_references(lhs, &scope.named.index, &mut uses);
        if let Some(rhs) = rhs {
            collect_named_references(rhs, &scope.named.index, &mut uses);
        }
        let mut frontier = Vec::new();
        while !uses.is_empty() {
            for j in uses.drain(..) {
                remaining[j] += 1;
                if !reached[j] {
                    reached[j] = true;
                    frontier.push(j);
                }
            }
            for j in frontier.drain(..) {
                uses.extend_from_slice(&declarations[j].references);
            }
        }
        Self::with_remaining(scope, remaining)
    }

    fn with_remaining(scope: &'a Scope<'a>, remaining: Vec<usize>) -> Self {
        let declarations = &scope.named.declarations;
        Lowering {
            scope,
            cache: declarations.iter().map(|_| Slot::Pending).collect(),
            remaining,
            origin: None,
            opaque_references: false,
            #[cfg(test)]
            lowerings: vec![0; declarations.len()],
            #[cfg(test)]
            peak_cached: 0,
        }
    }

    fn relation(
        &mut self,
        lhs: &Expr,
        rhs: Option<&(RelOp, Expr)>,
        whole: Range<usize>,
    ) -> Result<LoweredRelation, Fault> {
        let Some((op, rhs)) = rhs else {
            let form = self.form(lhs)?;
            if form.constant.is_some() {
                return Err(Fault::Local(Diagnostic {
                    span: whole,
                    message: "an expression without a relational operator cannot have a constant \
                              part; put the bound in generic_constraint_bounds.parquet or use <=, \
                              >= or =="
                        .to_owned(),
                }));
            }
            return Ok(LoweredRelation {
                terms: form.into_terms(),
                lower: None,
                upper: None,
            });
        };

        let left = self.form(lhs)?;
        let right = self.form(rhs)?;
        let variables = self.combine(
            LinearForm {
                terms: left.terms,
                constant: None,
            },
            LinearForm {
                terms: right.terms,
                constant: None,
            },
            &AddOp::Sub,
            &whole,
        )?;
        let mut remainder = ScalarExpr::literal(0.0);
        if let Some(right_constant) = right.constant {
            self.check_budget(binary_ops(Some(&remainder), Some(&right_constant)), &whole)?;
            remainder = checked(remainder + right_constant, &whole)?;
        }
        if let Some(left_constant) = left.constant {
            self.check_budget(binary_ops(Some(&remainder), Some(&left_constant)), &whole)?;
            remainder = checked(remainder - left_constant, &whole)?;
        }
        let (lower, upper) = match op {
            RelOp::Le => (None, Some(remainder)),
            RelOp::Ge => (Some(remainder), None),
            RelOp::Eq => (Some(remainder.clone()), Some(remainder)),
        };
        Ok(LoweredRelation {
            terms: variables.into_terms(),
            lower,
            upper,
        })
    }

    fn type_of(&self, expr: &Expr) -> Result<Type, Diagnostic> {
        type_of(expr, self.scope.params, &self.scope.named.index)
    }

    /// Lowers a tree that reads an LP variable or not; a tree that reads none becomes a form
    /// with only a constant.
    fn form(&mut self, expr: &Expr) -> Result<LinearForm, Fault> {
        enum Acc {
            Scalar(ScalarExpr),
            Form(LinearForm),
        }
        fn operand(lowering: &mut Lowering, expr: &Expr) -> Result<Acc, Fault> {
            Ok(match lowering.type_of(expr)? {
                Type::Scalar => Acc::Scalar(lower_scalar(expr, lowering.scope.params)?),
                Type::Linear => Acc::Form(lowering.form(expr)?),
            })
        }

        let linear = self.type_of(expr)? == Type::Linear;
        if let Some((_, j)) = named_reference(expr, &self.scope.named.index) {
            return self.reference(j, &expr.span);
        }
        match &expr.kind {
            ExprKind::Variable(call) => {
                let (variable, orientation) = variable_from_call(call, self.scope.line_index)
                    .map_err(|message| Diagnostic {
                        span: expr.span.clone(),
                        message,
                    })?;
                let coefficient = match orientation {
                    LinePairOrientation::Forward => None,
                    LinePairOrientation::Reversed => Some(ScalarExpr::literal(-1.0)),
                };
                Ok(LinearForm {
                    terms: vec![(variable, coefficient)],
                    constant: None,
                })
            }
            ExprKind::Neg(inner) if linear => {
                let form = self.form(inner)?;
                self.map_form(form, Scaling::Negate, &expr.span)
            }
            ExprKind::Sum { head, tail } if linear => {
                let mut acc = self.form(head)?;
                for (op, rhs) in tail {
                    let other = self.form(rhs)?;
                    acc = self.combine(acc, other, op, &(head.span.start..rhs.span.end))?;
                }
                Ok(acc)
            }
            ExprKind::Product { head, tail } if linear => {
                let mut acc = operand(self, head)?;
                for (op, rhs) in tail {
                    let step = head.span.start..rhs.span.end;
                    acc = match (acc, op, operand(self, rhs)?) {
                        (Acc::Scalar(lhs), MulOp::Mul, Acc::Scalar(rhs)) => {
                            Acc::Scalar(checked(lhs * rhs, &step)?)
                        }
                        (Acc::Scalar(lhs), MulOp::Div, Acc::Scalar(rhs)) => {
                            Acc::Scalar(checked_div(lhs, rhs, &step)?)
                        }
                        (Acc::Scalar(s), MulOp::Mul, Acc::Form(form)) => {
                            Acc::Form(self.map_form(form, Scaling::ScaleLeft(&s), &step)?)
                        }
                        (Acc::Form(form), MulOp::Mul, Acc::Scalar(s)) => {
                            Acc::Form(self.map_form(form, Scaling::ScaleRight(&s), &step)?)
                        }
                        (Acc::Form(form), MulOp::Div, Acc::Scalar(s)) => {
                            Acc::Form(self.map_form(form, Scaling::DivideBy(&s), &step)?)
                        }
                        (Acc::Form(_), MulOp::Mul, Acc::Form(_))
                        | (_, MulOp::Div, Acc::Form(_)) => {
                            return Err(not_linear(op, step).into());
                        }
                    };
                }
                Ok(match acc {
                    Acc::Scalar(value) => LinearForm {
                        terms: Vec::new(),
                        constant: Some(value),
                    },
                    Acc::Form(form) => form,
                })
            }
            _ => Ok(LinearForm {
                terms: Vec::new(),
                constant: Some(lower_scalar(expr, self.scope.params)?),
            }),
        }
    }

    /// The form of body `j` for one of its uses at `span`; the last use takes the cached form,
    /// every earlier one clones it.
    fn reference(&mut self, j: usize, span: &Range<usize>) -> Result<LinearForm, Fault> {
        if self.opaque_references {
            return Ok(LinearForm {
                terms: Vec::new(),
                constant: None,
            });
        }
        let form = match mem::replace(&mut self.cache[j], Slot::Pending) {
            Slot::Lowered(form) => form,
            Slot::InProgress => return Err(Fault::Anchored(cycle(span))),
            Slot::Pending => {
                let outermost = self.origin.is_none();
                if outermost {
                    self.origin = Some((j, span.clone()));
                }
                let anchor = self.origin.as_ref().map_or(span, |(_, at)| at).clone();
                let walked = self.walk(j, &anchor);
                if outermost {
                    self.origin = None;
                }
                walked?
            }
        };
        self.remaining[j] = self.remaining[j].saturating_sub(1);
        if self.remaining[j] > 0 {
            self.store(j, form.clone());
        }
        Ok(form)
    }

    /// Lowers body `root` after every body it reaches, in an iterative post-order so a long
    /// chain never recurses; `root`'s form is returned, not cached.
    fn walk(&mut self, root: usize, anchor: &Range<usize>) -> Result<LinearForm, Fault> {
        let named = self.scope.named;
        self.cache[root] = Slot::InProgress;
        let mut stack = vec![(root, 0_usize)];
        while let Some((node, next)) = stack.last_mut() {
            let node = *node;
            if let Some(&child) = named.declarations[node].references.get(*next) {
                *next += 1;
                match self.cache[child] {
                    Slot::Lowered(_) => {}
                    Slot::InProgress => return Err(Fault::Anchored(cycle(anchor))),
                    Slot::Pending => {
                        self.cache[child] = Slot::InProgress;
                        stack.push((child, 0));
                    }
                }
                continue;
            }
            stack.pop();
            if stack.is_empty() {
                break;
            }
            let form = self.lower_body(node, anchor)?;
            self.store(node, form);
        }
        let form = self.lower_body(root, anchor);
        self.cache[root] = Slot::Pending;
        form
    }

    /// Lowers body `j`, whose references are all cached. [`NamedTable::declare`] already lowered
    /// its source alone, so a fault here comes from the forms it composes; it moves to `anchor`
    /// on the constraint's source and names the declaration.
    fn lower_body(&mut self, j: usize, anchor: &Range<usize>) -> Result<LinearForm, Fault> {
        #[cfg(test)]
        {
            self.lowerings[j] += 1;
        }
        let named = self.scope.named;
        let declaration = &named.declarations[j];
        self.form(&declaration.body).map_err(|fault| match fault {
            Fault::Local(diagnostic) => Fault::Anchored(Diagnostic {
                span: anchor.clone(),
                message: format!(
                    "{} (composing \"@{}\", the named expression at expressions[{j}])",
                    diagnostic.message, declaration.name
                ),
            }),
            anchored @ Fault::Anchored(_) => anchored,
        })
    }

    fn store(&mut self, j: usize, form: LinearForm) {
        self.cache[j] = Slot::Lowered(form);
        #[cfg(test)]
        {
            let cached = self
                .cache
                .iter()
                .filter(|slot| matches!(slot, Slot::Lowered(_)))
                .count();
            self.peak_cached = self.peak_cached.max(cached);
        }
    }

    /// Refuses a form of `ops` total operations above [`MAX_INLINED_OPS`], naming the
    /// constraint's reference under expansion if there is one.
    fn check_budget(&self, ops: usize, span: &Range<usize>) -> Result<(), Fault> {
        if ops <= MAX_INLINED_OPS {
            return Ok(());
        }
        Err(match &self.origin {
            Some((j, at)) => Fault::Anchored(Diagnostic {
                span: at.clone(),
                message: format!(
                    "named-expression reference \"@{}\" expands to more than {MAX_INLINED_OPS} \
                     coefficient operations; this indicates an exponential reference pattern \
                     (e.g. a doubling chain \"@e = @prev + @prev\")",
                    self.scope.named.declarations[*j].name
                ),
            }),
            None => Fault::Local(Diagnostic {
                span: span.clone(),
                message: format!(
                    "expression expands to more than {MAX_INLINED_OPS} coefficient operations"
                ),
            }),
        })
    }

    /// Applies `scaling` to every coefficient and to the constant.
    fn map_form(
        &self,
        form: LinearForm,
        scaling: Scaling,
        span: &Range<usize>,
    ) -> Result<LinearForm, Fault> {
        let ops = form
            .terms
            .iter()
            .map(|(_, coefficient)| scaling.ops(coefficient.as_ref()))
            .chain(
                form.constant
                    .iter()
                    .map(|constant| scaling.ops(Some(constant))),
            )
            .fold(0, usize::saturating_add);
        self.check_budget(ops, span)?;
        // Checked before mapping so a form without terms, an opaque reference, still rejects it.
        if let Scaling::DivideBy(divisor) = scaling
            && divisor.as_literal().is_some_and(is_exact_zero)
        {
            return Err(division_by_zero(span).into());
        }
        let terms = form
            .terms
            .into_iter()
            .map(|(variable, coefficient)| Ok((variable, Some(scaling.apply(coefficient, span)?))))
            .collect::<Result<_, Diagnostic>>()?;
        let constant = form
            .constant
            .map(|constant| scaling.apply(Some(constant), span))
            .transpose()?;
        Ok(LinearForm { terms, constant })
    }

    /// Adds or subtracts `other` into `acc`: a variable already in `acc` (structural equality)
    /// has its coefficient combined in place, a new one is appended. No exact-zero coefficient
    /// is dropped here, so a later term can still change it.
    fn combine(
        &self,
        mut acc: LinearForm,
        other: LinearForm,
        op: &AddOp,
        span: &Range<usize>,
    ) -> Result<LinearForm, Fault> {
        let positions: Vec<Option<usize>> = other
            .terms
            .iter()
            .map(|(variable, _)| acc.terms.iter().position(|(known, _)| known == variable))
            .collect();
        let mut partners = vec![None; acc.terms.len()];
        let mut ops = 0_usize;
        for ((_, c2), position) in other.terms.iter().zip(&positions) {
            match position {
                Some(i) => partners[*i] = Some(c2.as_ref()),
                None => ops = ops.saturating_add(signed_ops(op, c2.as_ref())),
            }
        }
        for ((_, c1), partner) in acc.terms.iter().zip(&partners) {
            let merged = match partner {
                Some(c2) => binary_ops(c1.as_ref(), *c2),
                None => ops_of(c1.as_ref()),
            };
            ops = ops.saturating_add(merged);
        }
        ops = ops.saturating_add(match (&acc.constant, &other.constant) {
            (Some(k1), Some(k2)) => binary_ops(Some(k1), Some(k2)),
            (Some(k1), None) => k1.op_count(),
            (None, Some(k2)) => signed_ops(op, Some(k2)),
            (None, None) => 0,
        });
        self.check_budget(ops, span)?;

        let apply = |lhs: ScalarExpr, rhs: ScalarExpr| match op {
            AddOp::Add => checked(lhs + rhs, span),
            AddOp::Sub => checked(lhs - rhs, span),
        };
        for ((variable, c2), position) in other.terms.into_iter().zip(positions) {
            if let Some(i) = position {
                let c1 = &mut acc.terms[i].1;
                let lhs = c1.take().unwrap_or_else(one);
                *c1 = Some(apply(lhs, c2.unwrap_or_else(one))?);
            } else {
                let coefficient = match op {
                    AddOp::Add => c2,
                    AddOp::Sub => Some(checked(-c2.unwrap_or_else(one), span)?),
                };
                acc.terms.push((variable, coefficient));
            }
        }
        acc.constant = match (acc.constant, other.constant) {
            (Some(k1), Some(k2)) => Some(apply(k1, k2)?),
            (k1, None) => k1,
            (None, Some(k2)) => Some(match op {
                AddOp::Add => k2,
                AddOp::Sub => checked(-k2, span)?,
            }),
        };
        Ok(acc)
    }
}

fn cycle(span: &Range<usize>) -> Diagnostic {
    Diagnostic {
        span: span.clone(),
        message: "named-expression reference cycle detected".to_owned(),
    }
}

/// An operation `map_form` applies to every coefficient and to the constant; a unit coefficient
/// becomes the scale itself under a product and the operation applied to `literal(1.0)`
/// otherwise.
#[derive(Clone, Copy)]
enum Scaling<'s> {
    Negate,
    ScaleLeft(&'s ScalarExpr),
    ScaleRight(&'s ScalarExpr),
    DivideBy(&'s ScalarExpr),
}

impl Scaling<'_> {
    fn apply(
        self,
        coefficient: Option<ScalarExpr>,
        span: &Range<usize>,
    ) -> Result<ScalarExpr, Diagnostic> {
        match (self, coefficient) {
            (Scaling::ScaleLeft(s) | Scaling::ScaleRight(s), None) => Ok(s.clone()),
            (Scaling::ScaleLeft(s), Some(c)) => checked(s.clone() * c, span),
            (Scaling::ScaleRight(s), Some(c)) => checked(c * s.clone(), span),
            (Scaling::Negate, c) => checked(-c.unwrap_or_else(one), span),
            (Scaling::DivideBy(s), c) => checked_div(c.unwrap_or_else(one), s.clone(), span),
        }
    }

    /// The operation count of what [`Scaling::apply`] builds from `coefficient`.
    fn ops(self, coefficient: Option<&ScalarExpr>) -> usize {
        match (self, coefficient) {
            (Scaling::ScaleLeft(s) | Scaling::ScaleRight(s), None) => s.op_count(),
            (Scaling::Negate, c) => unary_ops(c),
            (Scaling::ScaleLeft(s) | Scaling::ScaleRight(s) | Scaling::DivideBy(s), c) => {
                binary_ops(c, Some(s))
            }
        }
    }
}

/// The operation count of a coefficient once built; a unit coefficient is the one literal
/// `1.0`.
fn ops_of(coefficient: Option<&ScalarExpr>) -> usize {
    coefficient.map_or(1, ScalarExpr::op_count)
}

fn is_literal(coefficient: Option<&ScalarExpr>) -> bool {
    coefficient.is_none_or(|c| c.as_literal().is_some())
}

/// Literal operands fold to one literal; anything else appends one operation to both operands.
fn binary_ops(lhs: Option<&ScalarExpr>, rhs: Option<&ScalarExpr>) -> usize {
    if is_literal(lhs) && is_literal(rhs) {
        1
    } else {
        ops_of(lhs).saturating_add(ops_of(rhs)).saturating_add(1)
    }
}

fn unary_ops(operand: Option<&ScalarExpr>) -> usize {
    if is_literal(operand) {
        1
    } else {
        ops_of(operand).saturating_add(1)
    }
}

fn signed_ops(op: &AddOp, operand: Option<&ScalarExpr>) -> usize {
    match op {
        AddOp::Add => ops_of(operand),
        AddOp::Sub => unary_ops(operand),
    }
}

fn one() -> ScalarExpr {
    ScalarExpr::literal(1.0)
}

fn checked(value: ScalarExpr, span: &Range<usize>) -> Result<ScalarExpr, Diagnostic> {
    match value.as_literal() {
        Some(v) if !v.is_finite() => Err(Diagnostic {
            span: span.clone(),
            message: format!("evaluates to a non-finite value ({v})"),
        }),
        _ => Ok(value),
    }
}

fn checked_div(
    lhs: ScalarExpr,
    rhs: ScalarExpr,
    span: &Range<usize>,
) -> Result<ScalarExpr, Diagnostic> {
    if rhs.as_literal().is_some_and(is_exact_zero) {
        return Err(division_by_zero(span));
    }
    checked(lhs / rhs, span)
}

fn division_by_zero(span: &Range<usize>) -> Diagnostic {
    Diagnostic {
        span: span.clone(),
        message: "division by zero".to_owned(),
    }
}

#[cfg(test)]
#[allow(clippy::panic, clippy::unwrap_used)]
mod tests {
    use std::collections::HashMap;

    use novomodelo_core::{EntityId, Line, ScalarExpr, VariableRef};

    use super::{
        Fault, LinearForm, LoweredRelation, Lowering, NamedTable, Scope, Slot, lower_relation,
        lower_scalar_body,
    };
    use crate::expression::Diagnostic;
    use crate::expression::catalog::{LineBusPairIndex, build_line_bus_pair_index};
    use crate::expression::parser::{Expr, parse};

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
        let a = || ScalarExpr::param(EntityId(7));
        let lit = ScalarExpr::literal;

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
        assert_eq!(mixed, a() + ScalarExpr::param(EntityId(8)) * lit(2.0));
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
                format!(
                    "`{src}`: a scalar-parameter expression cannot contain a relational operator"
                ),
                "{src:?}"
            );
        }
    }

    /// Runs `f` under `params()`, the named expressions `decls` and `line_index()`.
    fn with_named<T>(decls: &[(&str, &str)], f: impl FnOnce(&Scope) -> T) -> T {
        let params = params();
        let line_index = line_index();
        let named = match NamedTable::declare(decls, &params, &line_index) {
            Ok(named) => named,
            Err((j, d)) => panic!("expressions[{j}]: {}", d.render(decls[j].1)),
        };
        f(&Scope {
            params: &params,
            line_index: &line_index,
            named: &named,
        })
    }

    /// A single line 5 running from bus 3 to bus 7.
    fn line_index() -> LineBusPairIndex {
        let line = Line {
            id: EntityId(5),
            name: "L5".to_owned(),
            operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            source_bus_id: EntityId(3),
            target_bus_id: EntityId(7),
            entry_stage_id: None,
            exit_stage_id: None,
            direct_capacity_mw: 100.0,
            reverse_capacity_mw: 100.0,
            losses_percent: 0.0,
            exchange_cost: 0.0,
        };
        build_line_bus_pair_index(&[line]).unwrap()
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

    fn literal_bits_of(
        src: &str,
        terms: Vec<(VariableRef, ScalarExpr)>,
    ) -> Vec<(VariableRef, u64)> {
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

        let restored =
            terms("hydro_generation(0) - hydro_generation(0) + @a * hydro_generation(0)");
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
                 as its sole argument, e.g. \"line_exchange(source_bus=3, target_bus=7)\"; it \
                 cannot follow a positional argument in \"hydro_generation\"",
            ),
            (
                "hydro_turbined(5, foo=2)",
                "hydro_turbined(5, foo=2)",
                "unknown named argument \"foo\" in variable \"hydro_turbined\": only \"bus\" is \
                 supported",
            ),
            (
                "hydro_generation(source_bus=3, target_bus=7)",
                "hydro_generation(source_bus=3, target_bus=7)",
                "the (source_bus=, target_bus=) pair form addresses a line by its endpoint buses \
                 and is only accepted on \"line_exchange\", not \"hydro_generation\"",
            ),
            (
                "line_exchange(source_bus=3, 7)",
                "line_exchange(source_bus=3, 7)",
                "expected \"source_bus\" or \"target_bus\" in \"line_exchange\" pair form",
            ),
            (
                "line_exchange(source_bus=3, bus=7)",
                "line_exchange(source_bus=3, bus=7)",
                "expected \"source_bus\" or \"target_bus\" in \"line_exchange\" pair form",
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
                constant: Some(a() * b() * ScalarExpr::param(EntityId(9))),
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
        with_scope(|scope| lower_relation(src, scope))
            .unwrap_or_else(|d| panic!("{}", d.render(src)))
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
        match NamedTable::declare(decls, &params(), &line_index()) {
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
    fn a_scalar_body_is_rejected_with_the_parameter_hint() {
        let hint = |name: &str| {
            format!(
                "\"{name}\" contains no LP variable; declare it in generic_parameters.json as a \
                 kind \"expression\" parameter"
            )
        };
        assert_eq!(
            declaration_rejection(&[("x0", "hydro_generation(0)"), ("k", "2 * @p")]),
            (1, format!("`2 * @p`: {}", hint("k")))
        );
        assert_eq!(
            declaration_rejection(&[("three", "3")]),
            (0, format!("`3`: {}", hint("three")))
        );
        assert_eq!(
            declaration_rejection(&[("k", "@p")]),
            (0, format!("`@p`: {}", hint("k")))
        );
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
        let note =
            "(\"@e\" is the named expression at expressions[0], which contains LP variables)";
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
        let (first, second) = with_named(&decls, |scope| {
            let first = trace_in(scope, src);
            let second = trace_in(scope, src);
            (first, second)
        });
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
}
