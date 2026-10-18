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

/// The most operations the coefficient programs and constant of one form may hold together; a
/// guard against expansion blowup, set far above the size of an authored constraint so that
/// authored files never meet it.
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
mod tests;
