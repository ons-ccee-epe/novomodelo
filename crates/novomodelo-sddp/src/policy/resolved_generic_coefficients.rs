//! The one owner of every generic-constraint term value, bound remainder and endpoint shift per
//! study stage and block, built once at setup from [`ResolvedParameters`], the plant capacities
//! and the scenario layer. The LP builder and the generic-constraint echo read numbers from it
//! and resolve nothing themselves.
//!
//! A bound remainder or shift that reads scenario data, a data atom or a selector, is a program
//! each solve evaluates, and so is the `active_when` condition of a constraint: a row of such a
//! constraint is bounded per solve, and is left open where its condition does not hold.

mod per_solve;

use std::collections::{BTreeMap, BTreeSet};

use novomodelo_core::{
    EntityId, GenericConstraint, Hydro, ResolvedHydroUnitGroupBounds, ScalarExpr, StageId,
    VariableRef,
};
use novomodelo_io::constraints::scenario::{AvailabilityAtom, ConditionalScalar, DataAtom};
use novomodelo_io::constraints::{ConstraintScenario, ScenarioLayer};
use novomodelo_io::{AvailabilityError, resolve_hydro_availability};
use thiserror::Error;

use crate::generic_constraint_echo::render_variable;
use crate::lp::builder::PerSolveRow;
use crate::policy::resolved_parameters::ResolvedParameters;

use per_solve::{BusLoadRead, LoadSource};
pub(crate) use per_solve::{PerSolveTables, RowBoundScratch, StageLoads};

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

/// A bound remainder or endpoint shift: its setup value per cell, or the program each solve
/// evaluates because it reads scenario data.
#[derive(Debug)]
enum ResolvedProgram {
    Fixed(ResolvedSeries),
    PerSolve(ScalarExpr),
}

/// An absent bound side or shift is `None`, never a zero series.
#[derive(Debug)]
struct ResolvedConstraint {
    terms: Vec<ResolvedSeries>,
    lower: Option<ResolvedProgram>,
    upper: Option<ResolvedProgram>,
    shift: Option<ResolvedProgram>,
    /// Position in [`PerSolveTables::conditions`] of the constraint's `active_when`; `None` when
    /// the constraint always holds.
    gate: Option<usize>,
    /// Whether a program a solve evaluates for the constraint depends on the block.
    per_solve_block_varying: bool,
    /// The plant capacities the bounds and the shift read, each with its parameter id,
    /// ascending by plant and then block.
    capacity_reads: Vec<(AvailabilityAtom, EntityId)>,
}

impl ResolvedConstraint {
    /// The per-solve bound remainders and shift, in this order.
    fn per_solve_programs(&self) -> impl Iterator<Item = &ScalarExpr> {
        [&self.lower, &self.upper, &self.shift]
            .into_iter()
            .filter_map(|program| match program {
                Some(ResolvedProgram::PerSolve(program)) => Some(program),
                Some(ResolvedProgram::Fixed(_)) | None => None,
            })
    }

    fn is_per_solve(&self) -> bool {
        self.gate.is_some() || self.per_solve_programs().next().is_some()
    }
}

/// Resolved term values, bound remainders and endpoint shifts of every generic
/// constraint, in the system's constraint order.
#[derive(Debug)]
pub(crate) struct ResolvedGenericCoefficients {
    constraints: Vec<ResolvedConstraint>,
    /// What a solve reads besides the rows; its `block_offsets[s]..block_offsets[s + 1]` is
    /// study stage `s`'s range in a block-varying series.
    tables: PerSolveTables,
}

#[cfg(any(test, feature = "test-support"))]
pub(crate) static NO_GENERIC_ROWS: ResolvedGenericCoefficients =
    ResolvedGenericCoefficients::empty();

impl Default for ResolvedGenericCoefficients {
    fn default() -> Self {
        Self::empty()
    }
}

/// What the generic constraints read besides the scalar parameters: the scenario layer, the
/// plants and unit-group bounds its capacities resolve from, the plants whose unit groups the
/// unit-group generation terms name, and the buses its loads come from.
pub(crate) struct ScenarioSources<'a> {
    pub(crate) layer: &'a ScenarioLayer,
    pub(crate) hydros: &'a [Hydro],
    pub(crate) group_bounds: &'a ResolvedHydroUnitGroupBounds,
    pub(crate) loads: LoadBuses<'a>,
}

/// The buses with a load, the buses whose load a solve draws, and the buses of the system.
#[derive(Debug, Default)]
pub(crate) struct LoadBuses<'a> {
    /// Ascending: a load read of any other bus is
    /// [`ResolvedGenericCoefficientsError::UnknownLoadBus`].
    pub(crate) buses_with_load: &'a [EntityId],
    /// Ascending: the load noise members, in the order a solve's realized loads follow.
    pub(crate) noise_members: &'a [EntityId],
    /// In system order, the bus positions of the stage LP.
    pub(crate) bus_ids: &'a [EntityId],
}

/// The endpoints of a generic-constraint row in the stage template.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct RowEndpoints {
    pub(crate) lower: Option<f64>,
    pub(crate) upper: Option<f64>,
    /// Whether each solve writes the row's bounds; a side whose value a solve evaluates is then
    /// present and infinite here.
    pub(crate) per_solve: bool,
}

/// Which bound remainder of a generic constraint to read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BoundSide {
    Lower,
    Upper,
}

/// A generic-constraint term, coefficient, bound, shift or capacity that cannot be resolved.
#[derive(Debug, Error)]
pub(crate) enum ResolvedGenericCoefficientsError {
    #[error(transparent)]
    Availability(#[from] AvailabilityError),

    #[error(
        "the capacity of hydro {hydro_id} reads block {block}, which stage {stage_id} does not have"
    )]
    AvailabilityBlockOutOfRange {
        hydro_id: EntityId,
        block: usize,
        stage_id: i32,
    },

    #[error(
        "generic constraint {constraint_id} '{constraint_name}' references {variable}, a term the stage LP cannot place in a generic-constraint row"
    )]
    UnsupportedTerm {
        constraint_id: i32,
        constraint_name: String,
        variable: String,
    },

    #[error(
        "generic constraint {constraint_id} '{constraint_name}' reads the generation of unit group {group_id} of hydro {hydro_id}, which the stage LP cannot resolve: {reason}"
    )]
    GroupGenerationUnresolvable {
        constraint_id: i32,
        constraint_name: String,
        hydro_id: EntityId,
        group_id: EntityId,
        reason: &'static str,
    },

    #[error(
        "generic constraint {constraint_id} '{constraint_name}' references scalar parameter id={parameter_id} not present in the resolved parameter table"
    )]
    UnresolvedParameter {
        constraint_id: i32,
        constraint_name: String,
        parameter_id: i32,
    },

    #[error(
        "generic constraint {constraint_id} '{constraint_name}': the coefficient of {variable} reads scenario data, which a row of fixed structure cannot hold"
    )]
    ScenarioCoefficient {
        constraint_id: i32,
        constraint_name: String,
        variable: String,
    },

    #[error(
        "generic constraint {constraint_id} '{constraint_name}' reads the load of bus {bus_id}, which has no load"
    )]
    UnknownLoadBus {
        constraint_id: i32,
        constraint_name: String,
        bus_id: EntityId,
    },

    #[error(
        "generic constraint {constraint_id} '{constraint_name}' reads the load at block {block}, which stage {stage_id} does not have"
    )]
    LoadBlockOutOfRange {
        constraint_id: i32,
        constraint_name: String,
        block: usize,
        stage_id: i32,
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

/// Where a program's ids resolve, in this order: a scalar parameter, a plant capacity (both
/// setup values), then a data atom or selector of the layer, read per scenario.
struct Leaves<'a> {
    parameters: &'a ResolvedParameters,
    /// The value of every cell, per capacity id, ascending by id.
    capacities: Vec<(EntityId, Vec<f64>)>,
    block_offsets: &'a [usize],
    layer: &'a ScenarioLayer,
}

impl Leaves<'_> {
    fn capacity(&self, id: EntityId) -> Option<&[f64]> {
        let idx = self
            .capacities
            .binary_search_by_key(&id, |&(capacity_id, _)| capacity_id)
            .ok()?;
        Some(&self.capacities[idx].1)
    }

    fn is_setup(&self, id: EntityId) -> bool {
        self.parameters.contains(id) || self.capacity(id).is_some()
    }

    fn resolves(&self, id: EntityId) -> bool {
        self.is_setup(id)
            || self
                .layer
                .data_atoms
                .binary_search_by_key(&id, |&(atom_id, _)| atom_id)
                .is_ok()
            || self
                .layer
                .selectors
                .binary_search_by_key(&id, |&(selector_id, _)| selector_id)
                .is_ok()
    }

    /// Whether `program`, whose every id [`Self::resolves`], reads scenario data.
    fn reads_scenario(&self, program: &ScalarExpr) -> bool {
        program.params().any(|id| !self.is_setup(id))
    }

    fn is_block_varying(&self, program: &ScalarExpr) -> bool {
        program
            .params()
            .any(|id| self.parameters.is_block_varying(id) || self.capacity(id).is_some())
    }

    fn value(&self, id: EntityId, stage_idx: usize, block_idx: usize) -> f64 {
        if self.parameters.contains(id) {
            self.parameters.get(id, stage_idx, block_idx)
        } else {
            self.capacity(id).map_or(f64::NAN, |cells| {
                cells[self.block_offsets[stage_idx] + block_idx]
            })
        }
    }

    /// The value of setup id `id` at every cell of `grid`.
    fn cells(&self, id: EntityId, grid: &StudyGrid<'_>) -> Vec<f64> {
        self.capacity(id).map_or_else(
            || {
                (0..)
                    .zip(grid.block_counts)
                    .flat_map(|(stage_idx, &n_blocks)| {
                        (0..n_blocks).map(move |block| self.parameters.get(id, stage_idx, block))
                    })
                    .collect()
            },
            <[f64]>::to_vec,
        )
    }

    /// Whether the value a solve reads for `id` depends on the block: a parameter varying by
    /// block, a capacity or data atom without a block of its own, or a selector `selectors`
    /// marks, which holds every selector of a lower id.
    fn varies_by_block(&self, id: EntityId, selectors: &BTreeMap<EntityId, bool>) -> bool {
        if self.parameters.contains(id) {
            return self.parameters.is_block_varying(id);
        }
        if let Some(atom) = find(&self.layer.availability_atoms, id) {
            return atom.block_id.is_none();
        }
        if let Some(DataAtom::BusLoadValue { block_id, .. }) = find(&self.layer.data_atoms, id) {
            return block_id.is_none();
        }
        debug_assert!(
            selectors.contains_key(&id),
            "selector {id:?} reads a selector of a higher id"
        );
        selectors.get(&id).copied().unwrap_or(false)
    }
}

fn find<T>(items: &[(EntityId, T)], id: EntityId) -> Option<&T> {
    items
        .binary_search_by_key(&id, |(item_id, _)| *item_id)
        .ok()
        .map(|idx| &items[idx].1)
}

/// Push the programs a solve evaluates for `selector` onto `out`: the sides of its comparisons,
/// then its two values.
fn selector_programs<'a>(selector: &'a ConditionalScalar, out: &mut Vec<&'a ScalarExpr>) {
    selector.cond.operands(out);
    out.extend([&selector.then, &selector.else_]);
}

impl ResolvedGenericCoefficients {
    /// [`Self::build_with_scenario`] over an empty scenario layer, no plant and no load.
    ///
    /// # Errors
    ///
    /// As [`Self::build_with_scenario`].
    #[cfg(any(test, feature = "test-support"))]
    pub(crate) fn build(
        constraints: &[GenericConstraint],
        resolved: &ResolvedParameters,
        study_stage_ids: &[StageId],
        stage_block_counts: &[usize],
    ) -> Result<Self, ResolvedGenericCoefficientsError> {
        Self::build_with_scenario(
            constraints,
            &ScenarioSources {
                layer: &ScenarioLayer::default(),
                hydros: &[],
                group_bounds: &ResolvedHydroUnitGroupBounds::default(),
                loads: LoadBuses::default(),
            },
            resolved,
            study_stage_ids,
            stage_block_counts,
        )
    }

    /// Resolve every term, present bound side and endpoint shift of `constraints` over every
    /// study stage and block. A program reads the scalar parameters of `resolved` and the plant
    /// capacities of `sources` at their setup value; each constraint takes the shift and the
    /// condition of its entry in the scenario layer. A bound or shift reading a data atom or a
    /// selector is kept as a program each solve evaluates, and the tables a solve reads hold
    /// only what those programs, the conditions and the selectors they read reach, and the
    /// capacities the bounds and shifts read.
    ///
    /// # Errors
    ///
    /// [`ResolvedGenericCoefficientsError::Availability`] and
    /// [`ResolvedGenericCoefficientsError::AvailabilityBlockOutOfRange`] when a capacity cannot
    /// be resolved. Then, per constraint and in this order:
    /// [`ResolvedGenericCoefficientsError::UnsupportedTerm`] for a term the stage LP cannot
    /// place in a generic-constraint row,
    /// [`ResolvedGenericCoefficientsError::GroupGenerationUnresolvable`] for the generation of a
    /// unit group of `sources.hydros` without a bus of its own in its plant,
    /// [`ResolvedGenericCoefficientsError::UnresolvedParameter`] for an id that is neither a
    /// parameter `resolved` holds, a capacity, a data atom nor a selector,
    /// [`ResolvedGenericCoefficientsError::ScenarioCoefficient`] for a coefficient reading
    /// scenario data, and [`ResolvedGenericCoefficientsError::NonFinite`] on the first
    /// non-finite cell of a setup value, whether or not the constraint is active at that stage.
    /// Then, for the first constraint in system order whose per-solve programs reach it:
    /// [`ResolvedGenericCoefficientsError::UnresolvedParameter`] for an id a selector reads
    /// that nothing resolves, [`ResolvedGenericCoefficientsError::UnknownLoadBus`] for the load
    /// of a bus outside `sources.loads.buses_with_load` or without a load source, and
    /// [`ResolvedGenericCoefficientsError::LoadBlockOutOfRange`] for a load at a block some
    /// study stage does not have.
    pub(crate) fn build_with_scenario(
        constraints: &[GenericConstraint],
        sources: &ScenarioSources<'_>,
        resolved: &ResolvedParameters,
        study_stage_ids: &[StageId],
        stage_block_counts: &[usize],
    ) -> Result<Self, ResolvedGenericCoefficientsError> {
        debug_assert_eq!(study_stage_ids.len(), stage_block_counts.len());
        let layer = sources.layer;
        debug_assert!(
            layer.data_atoms.is_sorted_by(|a, b| a.0 < b.0)
                && layer.availability_atoms.is_sorted_by(|a, b| a.0 < b.0)
                && layer.selectors.is_sorted_by(|a, b| a.0 < b.0)
                && layer
                    .constraints
                    .is_sorted_by(|a, b| a.constraint_id < b.constraint_id),
            "the scenario layer is binary-searched"
        );
        debug_assert!(
            sources.loads.buses_with_load.is_sorted_by(|a, b| a < b)
                && sources.loads.noise_members.is_sorted_by(|a, b| a < b),
            "the load buses and the noise members are binary-searched"
        );
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
        let stage_ids: Vec<i32> = study_stage_ids.iter().map(|id| id.0).collect();
        let capacities = layer.availability_atoms.iter().map(|&(id, atom)| {
            let capacity = resolve_hydro_availability(
                sources.hydros,
                sources.group_bounds,
                atom.hydro_id,
                &stage_ids,
                stage_block_counts,
            )?;
            let mut values = Vec::with_capacity(n_cells);
            for (stage, &stage_id) in capacity.iter().zip(&stage_ids) {
                let Some(block) = atom.block_id else {
                    values.extend_from_slice(stage);
                    continue;
                };
                let &value = stage.get(block).ok_or(
                    ResolvedGenericCoefficientsError::AvailabilityBlockOutOfRange {
                        hydro_id: atom.hydro_id,
                        block,
                        stage_id,
                    },
                )?;
                values.resize(values.len() + stage.len(), value);
            }
            Ok((id, values))
        });
        let leaves = Leaves {
            parameters: resolved,
            capacities: capacities.collect::<Result<_, ResolvedGenericCoefficientsError>>()?,
            block_offsets: &block_offsets,
            layer,
        };
        let mut entries = Vec::with_capacity(constraints.len());
        let mut resolved_constraints = Vec::with_capacity(constraints.len());
        let mut n_conditions = 0;
        for constraint in constraints {
            let entry = layer
                .constraints
                .binary_search_by_key(&constraint.id, |entry| entry.constraint_id)
                .ok()
                .map(|idx| &layer.constraints[idx]);
            let gate = entry.and_then(|entry| entry.active_when.as_ref()).map(|_| {
                n_conditions += 1;
                n_conditions - 1
            });
            resolved_constraints.push(resolve_constraint(
                constraint,
                entry,
                gate,
                sources.hydros,
                &leaves,
                &grid,
            )?);
            entries.push(entry);
        }
        let mut tables = per_solve_tables(
            constraints,
            &entries,
            &mut resolved_constraints,
            &leaves,
            &sources.loads,
            &grid,
        )?;
        tables.block_offsets = block_offsets;
        Ok(Self {
            constraints: resolved_constraints,
            tables,
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
            &self.tables.block_offsets,
            stage_idx,
            block_idx,
        )
    }

    /// The setup value of `program` at `(stage_idx, block_idx)`; `None` for a per-solve one.
    fn fixed_value(
        &self,
        program: &ResolvedProgram,
        stage_idx: usize,
        block_idx: usize,
    ) -> Option<f64> {
        match program {
            ResolvedProgram::Fixed(series) => {
                Some(series.value(&self.tables.block_offsets, stage_idx, block_idx))
            }
            ResolvedProgram::PerSolve(_) => None,
        }
    }

    /// The resolved remainder of bound `side` of constraint `constraint_idx` at
    /// `(stage_idx, block_idx)`; `None` when that side is absent or read per solve.
    #[cfg(test)]
    fn bound_value(
        &self,
        constraint_idx: usize,
        side: BoundSide,
        stage_idx: usize,
        block_idx: usize,
    ) -> Option<f64> {
        let constraint = &self.constraints[constraint_idx];
        let program = match side {
            BoundSide::Lower => constraint.lower.as_ref(),
            BoundSide::Upper => constraint.upper.as_ref(),
        };
        program.and_then(|program| self.fixed_value(program, stage_idx, block_idx))
    }

    /// The endpoints of a row of constraint `constraint_idx` at `(stage_idx, block_idx)` whose
    /// bounds-table entry gives `base_lower` and `base_upper`: each side is [`fold_endpoint`] of
    /// its base and bound remainder, then a present side moves by the constraint's endpoint
    /// shift, which never creates an endpoint. A side whose remainder or shift a solve evaluates
    /// is present and open: `-inf` below, `+inf` above.
    #[must_use]
    pub(crate) fn endpoints(
        &self,
        constraint_idx: usize,
        stage_idx: usize,
        block_idx: usize,
        base_lower: Option<f64>,
        base_upper: Option<f64>,
    ) -> RowEndpoints {
        let constraint = &self.constraints[constraint_idx];
        let side = |remainder: Option<&ResolvedProgram>, base: Option<f64>, open: f64| {
            let side = match remainder {
                None => fold_endpoint(base, None),
                Some(remainder) => match self.fixed_value(remainder, stage_idx, block_idx) {
                    Some(remainder) => fold_endpoint(base, Some(remainder)),
                    None => return Some(open),
                },
            }?;
            Some(match &constraint.shift {
                None => side,
                Some(shift) => self
                    .fixed_value(shift, stage_idx, block_idx)
                    .map_or(open, |shift| side + shift),
            })
        };
        RowEndpoints {
            lower: side(constraint.lower.as_ref(), base_lower, f64::NEG_INFINITY),
            upper: side(constraint.upper.as_ref(), base_upper, f64::INFINITY),
            per_solve: constraint.is_per_solve(),
        }
    }

    /// Whether any term, bound or shift series of constraint `constraint_idx` varies by block,
    /// or any program a solve evaluates for it depends on the block.
    #[must_use]
    pub(crate) fn is_block_varying(&self, constraint_idx: usize) -> bool {
        let constraint = &self.constraints[constraint_idx];
        constraint.per_solve_block_varying
            || constraint
                .terms
                .iter()
                .chain(
                    [&constraint.lower, &constraint.upper, &constraint.shift]
                        .into_iter()
                        .filter_map(|program| match program {
                            Some(ResolvedProgram::Fixed(series)) => Some(series),
                            Some(ResolvedProgram::PerSolve(_)) | None => None,
                        }),
                )
                .any(|series| series.block_varying)
    }

    /// Whether a present side `side` of constraint `constraint_idx` is evaluated per solve: its
    /// remainder or the endpoint shift reads scenario data.
    #[must_use]
    pub(crate) fn side_is_per_solve(&self, constraint_idx: usize, side: BoundSide) -> bool {
        let constraint = &self.constraints[constraint_idx];
        let remainder = match side {
            BoundSide::Lower => &constraint.lower,
            BoundSide::Upper => &constraint.upper,
        };
        [remainder, &constraint.shift]
            .into_iter()
            .any(|program| matches!(program, Some(ResolvedProgram::PerSolve(_))))
    }

    /// Whether constraint `constraint_idx` has an `active_when` condition.
    #[must_use]
    pub(crate) fn is_gated(&self, constraint_idx: usize) -> bool {
        self.constraints[constraint_idx].gate.is_some()
    }

    /// The plant capacities the bounds and the endpoint shift of constraint `constraint_idx`
    /// read, each with its parameter id, ascending by plant and then block.
    #[must_use]
    pub(crate) fn capacity_reads(&self, constraint_idx: usize) -> &[(AvailabilityAtom, EntityId)] {
        &self.constraints[constraint_idx].capacity_reads
    }

    /// The value setup id `id` holds at `(stage_idx, block_idx)`: a plant capacity in MW some
    /// constraint reads, or a scalar parameter a solve reads; `None` when `id` is neither.
    #[must_use]
    pub(crate) fn setup_value(
        &self,
        id: EntityId,
        stage_idx: usize,
        block_idx: usize,
    ) -> Option<f64> {
        self.tables.setup_value(id, stage_idx, block_idx)
    }

    #[must_use]
    pub(crate) fn per_solve_tables(&self) -> &PerSolveTables {
        &self.tables
    }

    /// Write into `scratch` the bounds and activation of each row of `rows`, rows of study stage
    /// `stage_idx`, in their order, under the scenario load `loads`. A row whose condition does
    /// not hold at its block is open; otherwise each side is evaluated like
    /// [`endpoints`](Self::endpoints) folds it, a per-solve remainder or shift taking its value
    /// this solve, and a present side then moves by the row's `volume_shift`; an absent side is
    /// open.
    pub(crate) fn evaluate_bounds(
        &self,
        stage_idx: usize,
        rows: &[PerSolveRow],
        loads: &StageLoads<'_>,
        scratch: &mut RowBoundScratch,
    ) {
        self.tables.begin_stage(stage_idx, loads, scratch);
        for row in rows {
            let constraint = &self.constraints[row.constraint_idx];
            let active = scratch.holds(constraint.gate, row.block_idx);
            let (lower, upper) = if active {
                let lower = self.solved_side(
                    constraint,
                    constraint.lower.as_ref(),
                    row.base_lower,
                    stage_idx,
                    row.block_idx,
                    scratch,
                );
                let upper = self.solved_side(
                    constraint,
                    constraint.upper.as_ref(),
                    row.base_upper,
                    stage_idx,
                    row.block_idx,
                    scratch,
                );
                let moved = |side: Option<f64>| {
                    side.map(|side| {
                        row.volume_shift
                            .map_or(side, |volume_shift| side + volume_shift)
                    })
                };
                (
                    moved(lower).unwrap_or(f64::NEG_INFINITY),
                    moved(upper).unwrap_or(f64::INFINITY),
                )
            } else {
                (f64::NEG_INFINITY, f64::INFINITY)
            };
            scratch.push_row_bounds(lower, upper, active);
        }
    }

    /// One side of a row at block `block_idx` this solve: `base` folded with `remainder`, then
    /// the shift; `None` when the side is absent.
    fn solved_side(
        &self,
        constraint: &ResolvedConstraint,
        remainder: Option<&ResolvedProgram>,
        base: Option<f64>,
        stage_idx: usize,
        block_idx: usize,
        scratch: &mut RowBoundScratch,
    ) -> Option<f64> {
        let remainder =
            remainder.map(|remainder| self.solved_value(remainder, stage_idx, block_idx, scratch));
        let side = fold_endpoint(base, remainder)?;
        Some(match &constraint.shift {
            None => side,
            Some(shift) => side + self.solved_value(shift, stage_idx, block_idx, scratch),
        })
    }

    fn solved_value(
        &self,
        program: &ResolvedProgram,
        stage_idx: usize,
        block_idx: usize,
        scratch: &mut RowBoundScratch,
    ) -> f64 {
        match program {
            ResolvedProgram::Fixed(series) => {
                series.value(&self.tables.block_offsets, stage_idx, block_idx)
            }
            ResolvedProgram::PerSolve(program) => {
                self.tables.evaluate(program, stage_idx, block_idx, scratch)
            }
        }
    }

    pub(crate) const fn empty() -> Self {
        Self {
            constraints: Vec::new(),
            tables: PerSolveTables::empty(),
        }
    }
}

/// Fold a generic-constraint endpoint's parquet base with its resolved bound
/// remainder: a present remainder SHIFTS the base rather than replacing it, so
/// `(Some(base), Some(remainder))` folds to `base + remainder`, never `remainder`
/// alone. A `(None, None)` endpoint is untargeted and stays `None` (the open LP
/// direction), never shifted.
fn fold_endpoint(base: Option<f64>, remainder: Option<f64>) -> Option<f64> {
    match (base, remainder) {
        (None, None) => None,
        (Some(base), None) => Some(base),
        (None, Some(remainder)) => Some(remainder),
        (Some(base), Some(remainder)) => Some(base + remainder),
    }
}

fn resolve_constraint(
    constraint: &GenericConstraint,
    scenario: Option<&ConstraintScenario>,
    gate: Option<usize>,
    hydros: &[Hydro],
    leaves: &Leaves<'_>,
    grid: &StudyGrid<'_>,
) -> Result<ResolvedConstraint, ResolvedGenericCoefficientsError> {
    // `resolve_variable_ref` alone would drop or misplace these terms.
    if let Some(term) = constraint
        .expression
        .terms
        .iter()
        .find(|term| matches!(term.variable, VariableRef::HydroAvailability { .. }))
    {
        return Err(ResolvedGenericCoefficientsError::UnsupportedTerm {
            constraint_id: constraint.id.0,
            constraint_name: constraint.name.clone(),
            variable: render_variable(&term.variable).1,
        });
    }
    let unresolvable = constraint.expression.terms.iter().find_map(|term| {
        let VariableRef::HydroGroupGeneration {
            hydro_id, group_id, ..
        } = term.variable
        else {
            return None;
        };
        unresolvable_group(hydros, hydro_id, group_id).map(|reason| (hydro_id, group_id, reason))
    });
    if let Some((hydro_id, group_id, reason)) = unresolvable {
        return Err(
            ResolvedGenericCoefficientsError::GroupGenerationUnresolvable {
                constraint_id: constraint.id.0,
                constraint_name: constraint.name.clone(),
                hydro_id,
                group_id,
                reason,
            },
        );
    }
    let shift = scenario.and_then(|entry| entry.endpoint_shift.as_ref());
    let mut condition_operands = Vec::new();
    if let Some(condition) = scenario.and_then(|entry| entry.active_when.as_ref()) {
        condition.operands(&mut condition_operands);
    }
    // Before any `get`: it asserts on an unknown id in debug builds.
    if let Some(id) = constraint
        .parameter_ids()
        .chain(shift.into_iter().flat_map(ScalarExpr::params))
        .chain(
            condition_operands
                .iter()
                .flat_map(|operand| operand.params()),
        )
        .find(|&id| !leaves.resolves(id))
    {
        return Err(ResolvedGenericCoefficientsError::UnresolvedParameter {
            constraint_id: constraint.id.0,
            constraint_name: constraint.name.clone(),
            parameter_id: id.0,
        });
    }
    if let Some(term) = constraint
        .expression
        .terms
        .iter()
        .find(|term| leaves.reads_scenario(&term.coefficient))
    {
        return Err(ResolvedGenericCoefficientsError::ScenarioCoefficient {
            constraint_id: constraint.id.0,
            constraint_name: constraint.name.clone(),
            variable: render_variable(&term.variable).1,
        });
    }
    let terms = constraint
        .expression
        .terms
        .iter()
        .map(|term| {
            resolve_program(constraint, &term.coefficient, leaves, grid, || {
                format!("the coefficient of {}", render_variable(&term.variable).1)
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut capacity_reads: Vec<(AvailabilityAtom, EntityId)> = constraint
        .bound_lower
        .iter()
        .chain(&constraint.bound_upper)
        .chain(shift)
        .flat_map(ScalarExpr::params)
        .filter_map(|id| find(&leaves.layer.availability_atoms, id).map(|&atom| (atom, id)))
        .collect();
    capacity_reads.sort_unstable();
    capacity_reads.dedup();
    let resolve_optional = |program: Option<&ScalarExpr>, location: &str| {
        program
            .map(|program| {
                if leaves.reads_scenario(program) {
                    Ok(ResolvedProgram::PerSolve(program.clone()))
                } else {
                    resolve_program(constraint, program, leaves, grid, || location.to_string())
                        .map(ResolvedProgram::Fixed)
                }
            })
            .transpose()
    };
    Ok(ResolvedConstraint {
        terms,
        lower: resolve_optional(constraint.bound_lower.as_ref(), "the lower bound")?,
        upper: resolve_optional(constraint.bound_upper.as_ref(), "the upper bound")?,
        shift: resolve_optional(shift, "the endpoint shift")?,
        gate,
        per_solve_block_varying: false,
        capacity_reads,
    })
}

/// Why the stage LP has no generation column of its own for unit group `group_id` of plant
/// `hydro_id`: it holds one generation per bus of a plant, so a group sharing its bus shares
/// that column.
fn unresolvable_group(
    hydros: &[Hydro],
    hydro_id: EntityId,
    group_id: EntityId,
) -> Option<&'static str> {
    let Some(hydro) = hydros.iter().find(|hydro| hydro.id == hydro_id) else {
        return Some("no such plant");
    };
    let Some(group) = hydro.unit_groups.iter().find(|group| group.id == group_id) else {
        return Some("no such unit group");
    };
    hydro
        .unit_groups
        .iter()
        .any(|other| other.id != group_id && other.bus_id == group.bus_id)
        .then_some("unit group shares its bus with another group of the plant")
}

/// What a solve reads for `resolved`, the resolved `constraints` whose layer entries are
/// `entries`: the conditions of the constraints in system order, a constraint's gate its
/// position there, and, for only what the per-solve remainders and shifts, the conditions and
/// the selectors they read reach, the selectors, the load values and the buses they read with
/// where a solve finds each load, and the value per cell of each parameter and capacity, every
/// capacity a constraint's bounds or shift read included. Sets each constraint's
/// `per_solve_block_varying`.
fn per_solve_tables(
    constraints: &[GenericConstraint],
    entries: &[Option<&ConstraintScenario>],
    resolved: &mut [ResolvedConstraint],
    leaves: &Leaves<'_>,
    loads: &LoadBuses<'_>,
    grid: &StudyGrid<'_>,
) -> Result<PerSolveTables, ResolvedGenericCoefficientsError> {
    let layer = leaves.layer;
    let mut selectors: BTreeMap<EntityId, &ConditionalScalar> = BTreeMap::new();
    let mut atoms: BTreeMap<EntityId, (EntityId, Option<usize>)> = BTreeMap::new();
    let mut bus_readers = BTreeMap::new();
    let mut setup_ids = BTreeSet::new();
    let mut conditions = Vec::new();
    let mut max_op_count = 0;
    let mut programs = Vec::new();
    for (constraint_idx, (entry, constraint)) in entries.iter().zip(&*resolved).enumerate() {
        let reader = &constraints[constraint_idx];
        setup_ids.extend(constraint.capacity_reads.iter().map(|&(_, id)| id));
        programs.clear();
        programs.extend(constraint.per_solve_programs());
        if let Some(condition) = entry.and_then(|entry| entry.active_when.as_ref()) {
            debug_assert_eq!(constraint.gate, Some(conditions.len()));
            condition.operands(&mut programs);
            conditions.push(condition.clone());
        }
        let mut next = 0;
        while let Some(&program) = programs.get(next) {
            next += 1;
            max_op_count = max_op_count.max(program.op_count());
            for id in program.params() {
                if leaves.is_setup(id) {
                    setup_ids.insert(id);
                } else if let Some(selector) = find(&layer.selectors, id) {
                    if selectors.insert(id, selector).is_none() {
                        selector_programs(selector, &mut programs);
                    }
                } else if let Some(&DataAtom::BusLoadValue { bus_id, block_id }) =
                    find(&layer.data_atoms, id)
                {
                    if loads.buses_with_load.binary_search(&bus_id).is_err() {
                        return Err(unknown_load_bus(reader, bus_id));
                    }
                    bus_readers.entry(bus_id).or_insert(constraint_idx);
                    check_load_block(reader, block_id, grid)?;
                    atoms.insert(id, (bus_id, block_id));
                } else {
                    return Err(ResolvedGenericCoefficientsError::UnresolvedParameter {
                        constraint_id: reader.id.0,
                        constraint_name: reader.name.clone(),
                        parameter_id: id.0,
                    });
                }
            }
        }
    }
    let load_buses = bus_readers
        .into_iter()
        .map(|(bus, reader)| {
            let source = match loads.noise_members.binary_search(&bus) {
                Ok(member) => Some(LoadSource::Noise { member }),
                Err(_) => loads
                    .bus_ids
                    .iter()
                    .position(|&id| id == bus)
                    .map(|bus_pos| LoadSource::Balance { bus_pos }),
            };
            source
                .map(|source| (bus, source))
                .ok_or_else(|| unknown_load_bus(&constraints[reader], bus))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let load_atoms = atoms
        .into_iter()
        .map(|(id, (bus, block))| {
            let slot = load_buses.partition_point(|&(load_bus, _)| load_bus < bus);
            (id, BusLoadRead { slot, block })
        })
        .collect();
    let mut selector_varies: BTreeMap<EntityId, bool> = BTreeMap::new();
    for (&id, selector) in &selectors {
        programs.clear();
        selector_programs(selector, &mut programs);
        let varies = programs.iter().any(|program| {
            program
                .params()
                .any(|id| leaves.varies_by_block(id, &selector_varies))
        });
        selector_varies.insert(id, varies);
    }
    for constraint in resolved.iter_mut() {
        let mut reads: Vec<&ScalarExpr> = constraint.per_solve_programs().collect();
        if let Some(gate) = constraint.gate {
            conditions[gate].operands(&mut reads);
        }
        let varies = reads.iter().any(|program| {
            program
                .params()
                .any(|id| leaves.varies_by_block(id, &selector_varies))
        });
        constraint.per_solve_block_varying = varies;
    }
    Ok(PerSolveTables {
        setup_values: setup_ids
            .into_iter()
            .map(|id| (id, leaves.cells(id, grid)))
            .collect(),
        block_offsets: Vec::new(),
        load_buses,
        load_atoms,
        conditions,
        selectors: selectors
            .into_iter()
            .map(|(id, selector)| (id, selector.clone()))
            .collect(),
        max_op_count,
    })
}

/// [`ResolvedGenericCoefficientsError::LoadBlockOutOfRange`] when `reader` reads the load at a
/// `block` some study stage does not have.
fn check_load_block(
    reader: &GenericConstraint,
    block: Option<usize>,
    grid: &StudyGrid<'_>,
) -> Result<(), ResolvedGenericCoefficientsError> {
    if let Some(block) = block
        && let Some((stage_id, _)) = grid
            .stage_ids
            .iter()
            .zip(grid.block_counts)
            .find(|&(_, &n_blocks)| block >= n_blocks)
    {
        return Err(ResolvedGenericCoefficientsError::LoadBlockOutOfRange {
            constraint_id: reader.id.0,
            constraint_name: reader.name.clone(),
            block,
            stage_id: stage_id.0,
        });
    }
    Ok(())
}

fn unknown_load_bus(
    constraint: &GenericConstraint,
    bus_id: EntityId,
) -> ResolvedGenericCoefficientsError {
    ResolvedGenericCoefficientsError::UnknownLoadBus {
        constraint_id: constraint.id.0,
        constraint_name: constraint.name.clone(),
        bus_id,
    }
}

fn resolve_program(
    constraint: &GenericConstraint,
    program: &ScalarExpr,
    leaves: &Leaves<'_>,
    grid: &StudyGrid<'_>,
    location: impl FnOnce() -> String,
) -> Result<ResolvedSeries, ResolvedGenericCoefficientsError> {
    resolve_series(grid, leaves.is_block_varying(program), |s, b| {
        program.evaluate(|id| leaves.value(id, s, b))
    })
    .map_err(|cell| non_finite(constraint, location(), grid, &cell))
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
    use novomodelo_core::test_support::{HydroSpec, make_hydro};
    use novomodelo_core::{
        ConstraintExpression, EntityId, GenericConstraint, Hydro, HydroUnitGroupBoundsCountsSpec,
        LinearTerm, ResolvedHydroUnitGroupBounds, ScalarExpr, StageId, VariableRef,
    };
    use novomodelo_io::constraints::scenario::{
        AvailabilityAtom, CompareOp, ConditionalScalar, DataAtom, ResolvedCondition,
    };
    use novomodelo_io::constraints::{ConstraintScenario, ScenarioLayer};
    use novomodelo_io::{AvailabilityError, resolve_hydro_availability};

    use super::{
        BoundSide, LoadBuses, ResolvedGenericCoefficients, ResolvedGenericCoefficientsError,
        RowBoundScratch, RowEndpoints, ScenarioSources, StageLoads, fold_endpoint,
    };
    use crate::lp::builder::PerSolveRow;
    use crate::policy::resolved_parameters::ResolvedParameters;
    use crate::test_support::{
        equipment_free_geometry, geometry_with_load_balance, make_unit_group,
    };

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

    /// Built without plants, so the unit-group term would fail if it were checked first.
    #[test]
    fn an_unsupported_term_names_the_constraint_and_the_variable() {
        let c = constraint(vec![
            LinearTerm::literal(1.0, group_generation(1)),
            LinearTerm::literal(
                1.0,
                VariableRef::HydroAvailability {
                    hydro_id: EntityId(14),
                    block_id: None,
                },
            ),
        ]);
        let err =
            ResolvedGenericCoefficients::build(&[c], &table(vec![], &[]), &TWO_STAGES, &[1, 1])
                .unwrap_err();
        assert!(matches!(
            err,
            ResolvedGenericCoefficientsError::UnsupportedTerm {
                constraint_id: 500,
                ..
            }
        ));
        let message = err.to_string();
        for expected in [
            "500",
            "'probe'",
            "hydro_availability(id=14",
            "cannot place in a generic-constraint row",
        ] {
            assert!(
                message.contains(expected),
                "{expected} missing from: {message}"
            );
        }
        assert!(!message.contains("unit_group"), "{message}");
    }

    fn group_generation(group_id: i32) -> VariableRef {
        VariableRef::HydroGroupGeneration {
            hydro_id: EntityId(14),
            group_id: EntityId(group_id),
            block_id: None,
        }
    }

    /// Plant 14 of [`plant_14`] holding the unit groups `(id, bus)` of `groups`.
    fn plant_14_with_groups(groups: &[(i32, i32)]) -> Vec<Hydro> {
        let (mut hydros, _) = plant_14();
        hydros[0].unit_groups = groups
            .iter()
            .map(|&(id, bus)| {
                make_unit_group(EntityId(id), EntityId(bus), 0.0, DECLARED_MW, 0.0, 1.0)
            })
            .collect();
        hydros
    }

    /// The owner of `probe` reading the generation of unit group 1 of plant 14, over `hydros`
    /// and two stages of two blocks.
    fn build_with_hydros(
        hydros: &[Hydro],
    ) -> Result<ResolvedGenericCoefficients, ResolvedGenericCoefficientsError> {
        ResolvedGenericCoefficients::build_with_scenario(
            &[constraint(vec![LinearTerm::literal(
                1.0,
                group_generation(1),
            )])],
            &ScenarioSources {
                layer: &ScenarioLayer::default(),
                hydros,
                group_bounds: &ResolvedHydroUnitGroupBounds::default(),
                loads: LoadBuses::default(),
            },
            &table(vec![], &[]),
            &TWO_STAGES,
            &[2, 2],
        )
    }

    #[test]
    fn a_unit_group_alone_on_its_bus_keeps_its_generation_term() {
        let rgc = build_with_hydros(&plant_14_with_groups(&[(1, 10), (2, 20)])).unwrap();
        for stage in 0..2 {
            for block in 0..2 {
                assert_eq!(
                    rgc.term_value(0, 0, stage, block).to_bits(),
                    1.0_f64.to_bits()
                );
            }
        }
    }

    #[test]
    fn the_generation_of_a_unit_group_the_lp_cannot_resolve_names_its_reason() {
        let mut plant_15 = plant_14_with_groups(&[(1, 10), (2, 20)]);
        plant_15[0].id = EntityId(15);
        for (hydros, expected) in [
            (plant_15, "no such plant"),
            (
                plant_14_with_groups(&[(0, 10), (2, 20)]),
                "no such unit group",
            ),
            (
                plant_14_with_groups(&[(1, 10), (2, 10)]),
                "unit group shares its bus with another group of the plant",
            ),
        ] {
            let error = build_with_hydros(&hydros).unwrap_err();
            let message = error.to_string();
            let ResolvedGenericCoefficientsError::GroupGenerationUnresolvable {
                constraint_id,
                hydro_id,
                group_id,
                reason,
                ..
            } = error
            else {
                panic!("{error:?}");
            };
            assert_eq!(
                (constraint_id, hydro_id, group_id, reason),
                (500, EntityId(14), EntityId(1), expected)
            );
            for needle in ["500", "'probe'", "unit group 1 of hydro 14", expected] {
                assert!(message.contains(needle), "{needle} missing from: {message}");
            }
        }
    }

    const DECLARED_MW: f64 = 300.0;
    const BLOCK_1_MW: f64 = 120.0;

    /// Plant 14 with one unit group of [`DECLARED_MW`], cut to [`BLOCK_1_MW`] in block 1 of the
    /// first of two stages of two blocks.
    fn plant_14() -> (Vec<Hydro>, ResolvedHydroUnitGroupBounds) {
        let hydros = vec![make_hydro(HydroSpec {
            id: 14,
            max_generation_mw: DECLARED_MW,
            ..HydroSpec::default()
        })];
        let mut group_bounds = ResolvedHydroUnitGroupBounds::new(&HydroUnitGroupBoundsCountsSpec {
            groups_per_plant: &[1],
            n_stages: 2,
            max_blocks: 2,
        });
        group_bounds
            .block_override_mut(0, 0, 0, 1)
            .unwrap()
            .max_generation_mw = Some(BLOCK_1_MW);
        (hydros, group_bounds)
    }

    fn probe() -> GenericConstraint {
        constraint(vec![LinearTerm::literal(1.0, hydro_storage(1))])
    }

    fn layer_entry(
        active_when: Option<ResolvedCondition>,
        endpoint_shift: Option<ScalarExpr>,
    ) -> ScenarioLayer {
        ScenarioLayer {
            constraints: vec![ConstraintScenario {
                constraint_id: EntityId(500),
                active_when,
                endpoint_shift,
            }],
            ..ScenarioLayer::default()
        }
    }

    /// The owner of `constraint` alone over the first `stage_block_counts.len()` stages of
    /// [`TWO_STAGES`], plant 14 behind the capacities of `layer`.
    fn build_with_layer(
        constraint: GenericConstraint,
        layer: &ScenarioLayer,
        resolved: &ResolvedParameters,
        stage_block_counts: &[usize],
    ) -> Result<ResolvedGenericCoefficients, ResolvedGenericCoefficientsError> {
        build_with_loads(
            &[constraint],
            layer,
            resolved,
            stage_block_counts,
            LoadBuses::default(),
        )
    }

    /// [`build_with_layer`] over `constraints`, their loads from `loads`.
    fn build_with_loads(
        constraints: &[GenericConstraint],
        layer: &ScenarioLayer,
        resolved: &ResolvedParameters,
        stage_block_counts: &[usize],
        loads: LoadBuses<'_>,
    ) -> Result<ResolvedGenericCoefficients, ResolvedGenericCoefficientsError> {
        let (hydros, group_bounds) = plant_14();
        ResolvedGenericCoefficients::build_with_scenario(
            constraints,
            &ScenarioSources {
                layer,
                hydros: &hydros,
                group_bounds: &group_bounds,
                loads,
            },
            resolved,
            &TWO_STAGES[..stage_block_counts.len()],
            stage_block_counts,
        )
    }

    /// Fixed endpoints, as a row bounded at setup takes them.
    fn fixed(lower: Option<f64>, upper: Option<f64>) -> RowEndpoints {
        RowEndpoints {
            lower,
            upper,
            per_solve: false,
        }
    }

    /// derived: the table gives only the lower endpoint `100`, which the shift `-5` moves to
    /// `95`; parameter 42 is `2.0` at block 1, so the shift `-1 · p42` moves `(100, 200)` to
    /// `(98, 198)` there.
    #[test]
    fn an_endpoint_shift_moves_only_the_endpoints_the_table_gives() {
        let literal = layer_entry(None, Some(lit(-5.0)));
        let rgc = build_with_layer(probe(), &literal, &table(vec![], &[]), &[2]).unwrap();
        assert_eq!(
            rgc.endpoints(0, 0, 0, Some(100.0), None),
            fixed(Some(95.0), None)
        );
        assert_eq!(
            rgc.endpoints(0, 0, 0, None, Some(100.0)),
            fixed(None, Some(95.0))
        );
        assert!(!rgc.is_block_varying(0));

        let parameter = layer_entry(None, Some(lit(-1.0) * param(42)));
        let rgc = build_with_layer(probe(), &parameter, &bound_fold_table(), &[2]).unwrap();
        assert_eq!(
            rgc.endpoints(0, 0, 1, Some(100.0), Some(200.0)),
            fixed(Some(98.0), Some(198.0))
        );
        assert!(rgc.is_block_varying(0));
    }

    /// derived: 900 is the capacity of plant 14 at the row's block and 901 its capacity at block
    /// 1 of the row's stage.
    #[test]
    fn a_bound_reading_a_capacity_is_fixed_at_its_setup_value() {
        let atom = |block_id| AvailabilityAtom {
            hydro_id: EntityId(14),
            block_id,
        };
        let layer = ScenarioLayer {
            availability_atoms: vec![(EntityId(900), atom(None)), (EntityId(901), atom(Some(1)))],
            ..ScenarioLayer::default()
        };
        let mut c = probe();
        c.bound_lower = Some(param(901));
        c.bound_upper = Some(param(900));
        let rgc = build_with_layer(c, &layer, &table(vec![], &[]), &[2, 2]).unwrap();
        let (hydros, group_bounds) = plant_14();
        let capacity = resolve_hydro_availability(
            &hydros,
            &group_bounds,
            EntityId(14),
            &TWO_STAGES.map(|id| id.0),
            &[2, 2],
        )
        .unwrap();
        assert_eq!(
            capacity,
            [[DECLARED_MW, BLOCK_1_MW], [DECLARED_MW, DECLARED_MW]]
        );
        for (stage, cells) in capacity.iter().enumerate() {
            for (block, value) in cells.iter().enumerate() {
                let endpoints = rgc.endpoints(0, stage, block, None, None);
                assert!(!endpoints.per_solve);
                assert_eq!(endpoints.upper.map(f64::to_bits), Some(value.to_bits()));
                assert_eq!(endpoints.lower.map(f64::to_bits), Some(cells[1].to_bits()));
            }
        }
        assert!(rgc.is_block_varying(0));
    }

    /// derived: the upper bound reads 900 twice and the shift reads 901, both setup values, and
    /// 902 is read by nothing; plant 14's capacity at the row's block (901) orders before its
    /// capacity at block 1 (900).
    #[test]
    fn a_constraint_reads_the_capacities_of_its_bounds_and_shift_at_their_setup_values() {
        let atom = |block_id| AvailabilityAtom {
            hydro_id: EntityId(14),
            block_id,
        };
        let mut layer = layer_entry(None, Some(param(901)));
        layer.availability_atoms = vec![
            (EntityId(900), atom(Some(1))),
            (EntityId(901), atom(None)),
            (EntityId(902), atom(Some(0))),
        ];
        let mut c = probe();
        c.bound_upper = Some(param(900) + param(900));
        let rgc = build_with_layer(c, &layer, &table(vec![], &[]), &[2, 2]).unwrap();
        let (hydros, group_bounds) = plant_14();
        let capacity = resolve_hydro_availability(
            &hydros,
            &group_bounds,
            EntityId(14),
            &TWO_STAGES.map(|id| id.0),
            &[2, 2],
        )
        .unwrap();

        assert_eq!(
            rgc.capacity_reads(0),
            [(atom(None), EntityId(901)), (atom(Some(1)), EntityId(900))]
        );
        for (stage, cells) in capacity.iter().enumerate() {
            for (block, &cell) in cells.iter().enumerate() {
                assert_eq!(rgc.setup_value(EntityId(901), stage, block), Some(cell));
                assert_eq!(rgc.setup_value(EntityId(900), stage, block), Some(cells[1]));
            }
        }
    }

    #[test]
    fn a_capacity_of_a_block_some_stage_lacks_names_that_stage() {
        let layer = ScenarioLayer {
            availability_atoms: vec![(
                EntityId(900),
                AvailabilityAtom {
                    hydro_id: EntityId(14),
                    block_id: Some(1),
                },
            )],
            ..ScenarioLayer::default()
        };
        let err = build_with_layer(probe(), &layer, &table(vec![], &[]), &[2, 1]).unwrap_err();
        assert_eq!(
            err.to_string(),
            "the capacity of hydro 14 reads block 1, which stage 4 does not have"
        );
    }

    #[test]
    fn a_capacity_of_a_block_no_stage_has_names_the_first_stage() {
        let layer = ScenarioLayer {
            availability_atoms: vec![(
                EntityId(900),
                AvailabilityAtom {
                    hydro_id: EntityId(14),
                    block_id: Some(3),
                },
            )],
            ..ScenarioLayer::default()
        };
        let err = build_with_layer(probe(), &layer, &table(vec![], &[]), &[2, 2]).unwrap_err();
        assert_eq!(
            err.to_string(),
            "the capacity of hydro 14 reads block 3, which stage 3 does not have"
        );
    }

    #[test]
    fn an_id_that_is_not_a_capacity_has_no_setup_value() {
        let layer = ScenarioLayer {
            availability_atoms: vec![(
                EntityId(900),
                AvailabilityAtom {
                    hydro_id: EntityId(14),
                    block_id: None,
                },
            )],
            ..ScenarioLayer::default()
        };
        let mut c = probe();
        c.bound_upper = Some(param(900));
        let rgc = build_with_layer(c, &layer, &table(vec![], &[]), &[2, 2]).unwrap();
        assert_eq!(rgc.setup_value(EntityId(900), 0, 0), Some(DECLARED_MW));
        assert_eq!(rgc.setup_value(EntityId(5), 0, 0), None);
    }

    #[test]
    fn an_endpoint_shift_resolves_its_ids_and_values_like_a_bound() {
        let unresolved = layer_entry(None, Some(param(999)));
        let err = build_with_layer(probe(), &unresolved, &table(vec![], &[]), &[2]).unwrap_err();
        assert!(
            matches!(
                err,
                ResolvedGenericCoefficientsError::UnresolvedParameter {
                    parameter_id: 999,
                    ..
                }
            ),
            "{err:?}"
        );
        let infinite = layer_entry(None, Some(lit(f64::INFINITY)));
        let err = build_with_layer(probe(), &infinite, &table(vec![], &[]), &[2]).unwrap_err();
        assert!(
            err.to_string().contains("the endpoint shift resolves to"),
            "{err}"
        );
    }

    /// The coefficient is refused first, although the lower bound also reads scenario data; a
    /// lower bound reading scenario data is present and open in the template, and a condition
    /// bounds per solve a row whose sides are fixed.
    #[test]
    fn scenario_reads_are_a_coefficient_error_or_a_per_solve_program() {
        let positive_load = || ResolvedCondition::Compare {
            lhs: param(940),
            op: CompareOp::Gt,
            rhs: lit(0.0),
        };
        let mut layer = ScenarioLayer {
            data_atoms: vec![(
                EntityId(940),
                DataAtom::BusLoadValue {
                    bus_id: EntityId(1),
                    block_id: None,
                },
            )],
            selectors: vec![(
                EntityId(950),
                ConditionalScalar {
                    cond: positive_load(),
                    then: lit(1.0),
                    else_: lit(0.0),
                },
            )],
            ..ScenarioLayer::default()
        };
        let resolved = table(vec![], &[]);
        let build = |c: GenericConstraint, layer: &ScenarioLayer| {
            build_with_loads(&[c], layer, &resolved, &[2], bus_1_noise())
        };
        let mut coefficient = constraint(vec![term(param(950), hydro_storage(1))]);
        coefficient.bound_lower = Some(param(940));
        let coefficient_err = build(coefficient, &layer).unwrap_err();
        let mut lower = probe();
        lower.bound_lower = Some(param(940));
        lower.bound_upper = Some(lit(7.0));
        let lower_rgc = build(lower, &layer).unwrap();
        layer.constraints = layer_entry(Some(positive_load()), None).constraints;
        let mut gated = probe();
        gated.bound_upper = Some(lit(7.0));
        let gated_rgc = build(gated, &layer).unwrap();

        assert!(
            matches!(
                coefficient_err,
                ResolvedGenericCoefficientsError::ScenarioCoefficient {
                    constraint_id: 500,
                    ..
                }
            ),
            "{coefficient_err:?}"
        );
        assert_eq!(
            coefficient_err.to_string(),
            "generic constraint 500 'probe': the coefficient of hydro_storage(id=1) reads scenario \
             data, which a row of fixed structure cannot hold"
        );
        assert_eq!(
            lower_rgc.endpoints(0, 0, 0, None, Some(3.0)),
            RowEndpoints {
                lower: Some(f64::NEG_INFINITY),
                upper: Some(10.0),
                per_solve: true,
            }
        );
        assert_eq!(
            gated_rgc.endpoints(0, 0, 0, None, None),
            RowEndpoints {
                lower: None,
                upper: Some(7.0),
                per_solve: true,
            }
        );
    }

    const LOAD: i32 = 940;

    /// Bus 1, the one noise member, whose load parameter [`LOAD`] stands for.
    fn bus_1_load() -> ScenarioLayer {
        ScenarioLayer {
            data_atoms: vec![(
                EntityId(LOAD),
                DataAtom::BusLoadValue {
                    bus_id: EntityId(1),
                    block_id: None,
                },
            )],
            ..ScenarioLayer::default()
        }
    }

    const BUS_1: [EntityId; 1] = [EntityId(1)];

    /// Bus 1 with a load drawn as noise member 0.
    fn bus_1_noise() -> LoadBuses<'static> {
        LoadBuses {
            buses_with_load: &BUS_1,
            noise_members: &BUS_1,
            bus_ids: &BUS_1,
        }
    }

    /// The row of constraint 0 (id 500) at block `block_idx`, with the table bases `base`.
    fn per_solve_row(block_idx: usize, base: (Option<f64>, Option<f64>)) -> PerSolveRow {
        PerSolveRow {
            row: 0,
            constraint_idx: 0,
            constraint_id: 500,
            block_idx,
            is_stage_level: false,
            base_lower: base.0,
            base_upper: base.1,
            volume_shift: None,
        }
    }

    /// The bounds and activation of `rows` of stage 0 of `rgc`, one block per row, under the
    /// realized loads `noise` of bus 1.
    fn evaluate(
        rgc: &ResolvedGenericCoefficients,
        rows: &[PerSolveRow],
        noise: &[f64],
        scratch: &mut RowBoundScratch,
    ) -> Vec<(f64, f64, bool)> {
        let geometry = equipment_free_geometry(&[noise.len()]).remove(0);
        let loads = StageLoads {
            noise,
            row_lower: &[],
            row_scale: &[],
            geometry: &geometry,
        };
        rgc.evaluate_bounds(0, rows, &loads, scratch);
        (0..rows.len())
            .map(|i| {
                (
                    scratch.lower()[i],
                    scratch.upper()[i],
                    scratch.row_active()[i],
                )
            })
            .collect()
    }

    /// derived: the upper bound is the load of bus 1, realized at 120, with no table base.
    #[test]
    fn a_solve_bounds_a_generic_row_by_the_realized_load_without_allocating() {
        let mut c = probe();
        c.bound_upper = Some(param(LOAD));
        let rgc = build_with_loads(
            &[c],
            &bus_1_load(),
            &table(vec![], &[]),
            &[1],
            bus_1_noise(),
        )
        .unwrap();
        assert_eq!(
            rgc.endpoints(0, 0, 0, None, None),
            RowEndpoints {
                lower: None,
                upper: Some(f64::INFINITY),
                per_solve: true,
            }
        );
        let rows = [per_solve_row(0, (None, None))];
        let geometry = equipment_free_geometry(&[1]).remove(0);
        let loads = StageLoads {
            noise: &[120.0],
            row_lower: &[],
            row_scale: &[],
            geometry: &geometry,
        };
        let mut scratch = RowBoundScratch::new(rgc.per_solve_tables(), rows.len());
        let reserved = scratch.footprint();

        rgc.evaluate_bounds(0, &rows, &loads, &mut scratch);
        let before_second = scratch.footprint();
        rgc.evaluate_bounds(0, &rows, &loads, &mut scratch);

        assert_eq!(before_second, reserved);

        assert_eq!(scratch.upper()[0].to_bits(), 120.0_f64.to_bits());
        assert_eq!(scratch.lower()[0], f64::NEG_INFINITY);
        assert!(scratch.row_active()[0]);
        assert_eq!(scratch.footprint(), reserved);
    }

    /// derived: the condition `load > 100` fails at 50, and at 150 the shift `-1 · load` moves
    /// the table's `(100, 200)` by -150.
    #[test]
    fn a_gated_formula_row_is_open_or_shifted_by_the_realized_load() {
        let layer = ScenarioLayer {
            constraints: layer_entry(
                Some(ResolvedCondition::Compare {
                    lhs: param(LOAD),
                    op: CompareOp::Gt,
                    rhs: lit(100.0),
                }),
                Some(lit(-1.0) * param(LOAD)),
            )
            .constraints,
            ..bus_1_load()
        };
        let rgc =
            build_with_loads(&[probe()], &layer, &table(vec![], &[]), &[1], bus_1_noise()).unwrap();
        assert_eq!(
            rgc.endpoints(0, 0, 0, Some(100.0), Some(200.0)),
            RowEndpoints {
                lower: Some(f64::NEG_INFINITY),
                upper: Some(f64::INFINITY),
                per_solve: true,
            }
        );
        let rows = [per_solve_row(0, (Some(100.0), Some(200.0)))];
        let mut scratch = RowBoundScratch::new(rgc.per_solve_tables(), rows.len());

        let low = evaluate(&rgc, &rows, &[50.0], &mut scratch);
        assert_eq!(low, [(f64::NEG_INFINITY, f64::INFINITY, false)]);
        let [(lower, upper, active)] = evaluate(&rgc, &rows, &[150.0], &mut scratch)[..] else {
            panic!("one row");
        };
        assert_eq!(lower.to_bits(), (100.0_f64 + -150.0).to_bits());
        assert_eq!(upper.to_bits(), (200.0_f64 + -150.0).to_bits());
        assert_eq!((lower, upper, active), (-50.0, 50.0, true));
    }

    /// derived: selector 950 is 5 where the load of bus 1 exceeds 100 and 2 elsewhere; the table
    /// gives the lower base 10, so the lower bound is `10 + 2` at 50 and `10 + 5` at 150.
    #[test]
    fn a_bound_reading_a_selector_takes_the_branch_its_condition_picks() {
        let layer = ScenarioLayer {
            selectors: vec![(
                EntityId(950),
                ConditionalScalar {
                    cond: ResolvedCondition::Compare {
                        lhs: param(LOAD),
                        op: CompareOp::Gt,
                        rhs: lit(100.0),
                    },
                    then: lit(5.0),
                    else_: lit(2.0),
                },
            )],
            ..bus_1_load()
        };
        let mut c = probe();
        c.bound_lower = Some(param(950));
        let rgc = build_with_loads(&[c], &layer, &table(vec![], &[]), &[1], bus_1_noise()).unwrap();
        let rows = [per_solve_row(0, (Some(10.0), None))];
        let mut scratch = RowBoundScratch::new(rgc.per_solve_tables(), rows.len());

        assert_eq!(
            evaluate(&rgc, &rows, &[50.0], &mut scratch),
            [(12.0, f64::INFINITY, true)]
        );
        assert_eq!(
            evaluate(&rgc, &rows, &[150.0], &mut scratch),
            [(15.0, f64::INFINITY, true)]
        );
    }

    /// derived: the bound reads selector 952, which reads selector 951 and the load of bus 1;
    /// selector 950, the load 941 of bus 1 and the condition of constraint 777, which the system
    /// does not hold, are reached by nothing. At 150, 951 is 5 and 952 takes it.
    #[test]
    fn only_the_programs_some_constraint_reaches_are_evaluated() {
        let compare = |id, rhs| ResolvedCondition::Compare {
            lhs: param(id),
            op: CompareOp::Gt,
            rhs: lit(rhs),
        };
        let selector = |cond, then, else_| ConditionalScalar { cond, then, else_ };
        let mut layer = bus_1_load();
        layer.data_atoms.push((
            EntityId(941),
            DataAtom::BusLoadValue {
                bus_id: EntityId(1),
                block_id: None,
            },
        ));
        layer.selectors = vec![
            (
                EntityId(950),
                selector(compare(941, 0.0), lit(1.0), lit(0.0)),
            ),
            (
                EntityId(951),
                selector(compare(LOAD, 100.0), lit(5.0), lit(2.0)),
            ),
            (
                EntityId(952),
                selector(compare(LOAD, 0.0), param(951), lit(0.0)),
            ),
        ];
        layer.constraints = vec![ConstraintScenario {
            constraint_id: EntityId(777),
            active_when: Some(compare(941, 0.0)),
            endpoint_shift: None,
        }];
        let mut c = probe();
        c.bound_upper = Some(param(952));
        let rgc = build_with_loads(&[c], &layer, &table(vec![], &[]), &[1], bus_1_noise()).unwrap();

        let tables = rgc.per_solve_tables();
        let ids = |ids: Vec<EntityId>| ids.into_iter().map(|id| id.0).collect::<Vec<_>>();
        assert_eq!(
            ids(tables.selectors.iter().map(|&(id, _)| id).collect()),
            [951, 952]
        );
        assert_eq!(
            ids(tables.load_atoms.iter().map(|&(id, _)| id).collect()),
            [LOAD]
        );
        assert_eq!(
            ids(tables.load_buses.iter().map(|&(id, _)| id).collect()),
            [1]
        );
        assert!(tables.conditions.is_empty());
        let mut scratch = RowBoundScratch::new(tables, 1);
        assert_eq!(
            evaluate(
                &rgc,
                &[per_solve_row(0, (None, None))],
                &[150.0],
                &mut scratch
            ),
            [(f64::NEG_INFINITY, 5.0, true)]
        );
    }

    /// derived: parameter 42 varies by block and 43 does not; capacity 901 and the load atom 939
    /// name block 1 and block 0, so only the reads without a block of their own vary.
    #[test]
    fn a_per_solve_constraint_collapses_only_when_nothing_it_evaluates_depends_on_the_block() {
        let mut layer = bus_1_load();
        layer.data_atoms.insert(
            0,
            (
                EntityId(939),
                DataAtom::BusLoadValue {
                    bus_id: EntityId(1),
                    block_id: Some(0),
                },
            ),
        );
        layer.availability_atoms = vec![(
            EntityId(901),
            AvailabilityAtom {
                hydro_id: EntityId(14),
                block_id: Some(1),
            },
        )];
        layer.selectors = vec![
            (
                EntityId(950),
                ConditionalScalar {
                    cond: ResolvedCondition::Compare {
                        lhs: param(939),
                        op: CompareOp::Gt,
                        rhs: lit(0.0),
                    },
                    then: lit(1.0),
                    else_: param(LOAD),
                },
            ),
            (
                EntityId(951),
                ConditionalScalar {
                    cond: ResolvedCondition::Compare {
                        lhs: param(939),
                        op: CompareOp::Gt,
                        rhs: param(43),
                    },
                    then: lit(1.0),
                    else_: param(901),
                },
            ),
        ];
        let gated = |condition_lhs: i32| {
            let mut layer = layer.clone();
            layer.constraints = layer_entry(
                Some(ResolvedCondition::Compare {
                    lhs: param(condition_lhs),
                    op: CompareOp::Gt,
                    rhs: lit(0.0),
                }),
                None,
            )
            .constraints;
            build_with_loads(&[probe()], &layer, &bound_fold_table(), &[2], bus_1_noise())
                .unwrap()
                .is_block_varying(0)
        };
        let bounded = |bound: ScalarExpr| {
            let mut c = probe();
            c.bound_upper = Some(bound);
            build_with_loads(&[c], &layer, &bound_fold_table(), &[2], bus_1_noise())
                .unwrap()
                .is_block_varying(0)
        };

        assert!(bounded(param(LOAD)));
        assert!(bounded(param(939) + param(42)));
        assert!(bounded(param(950)));
        assert!(gated(LOAD));
        assert!(!bounded(param(939) + param(43) * param(901)));
        assert!(!bounded(param(951)));
        assert!(!gated(939));
    }

    /// derived: bus 2 has no load, bus 3 has one that neither the noise nor the system holds, and
    /// the second study stage has one block, so block 1 is out of range there; constraint 600,
    /// first in system order, reaches each atom through selector 950.
    #[test]
    fn a_load_value_of_a_bus_without_load_or_of_a_missing_block_is_rejected() {
        let buses_with_load = [EntityId(1), EntityId(3)];
        let loads = || LoadBuses {
            buses_with_load: &buses_with_load,
            noise_members: &BUS_1,
            bus_ids: &BUS_1,
        };
        let reject = |atom: DataAtom| {
            let layer = ScenarioLayer {
                data_atoms: vec![(EntityId(LOAD), atom)],
                selectors: vec![(
                    EntityId(950),
                    ConditionalScalar {
                        cond: ResolvedCondition::Compare {
                            lhs: param(LOAD),
                            op: CompareOp::Gt,
                            rhs: lit(0.0),
                        },
                        then: lit(1.0),
                        else_: lit(0.0),
                    },
                )],
                ..ScenarioLayer::default()
            };
            let mut first = probe();
            first.id = EntityId(600);
            first.name = "first".to_string();
            first.bound_lower = Some(param(950));
            let mut second = probe();
            second.bound_upper = Some(param(LOAD));
            build_with_loads(
                &[first, second],
                &layer,
                &table(vec![], &[]),
                &[2, 1],
                loads(),
            )
            .unwrap_err()
            .to_string()
        };
        let bus = |bus_id, block_id| DataAtom::BusLoadValue {
            bus_id: EntityId(bus_id),
            block_id,
        };

        for bus_id in [2, 3] {
            assert_eq!(
                reject(bus(bus_id, None)),
                format!(
                    "generic constraint 600 'first' reads the load of bus {bus_id}, which has no \
                     load"
                )
            );
        }
        assert_eq!(
            reject(bus(1, Some(1))),
            "generic constraint 600 'first' reads the load at block 1, which stage 4 does not \
             have"
        );
    }

    /// derived: a present side is evaluated per solve when its remainder or the shift reads the
    /// load value [`LOAD`]; a condition alone leaves the sides fixed.
    #[test]
    fn a_side_is_per_solve_when_a_solve_evaluates_it() {
        let flags = |constraint: GenericConstraint, layer: &ScenarioLayer| {
            let rgc = build_with_loads(
                &[constraint],
                layer,
                &table(vec![], &[]),
                &[1],
                bus_1_noise(),
            )
            .unwrap();
            (
                rgc.side_is_per_solve(0, BoundSide::Lower),
                rgc.side_is_per_solve(0, BoundSide::Upper),
                rgc.is_gated(0),
            )
        };
        let mut upper = probe();
        upper.bound_lower = Some(lit(1.0));
        upper.bound_upper = Some(param(LOAD));
        let mut shifted = bus_1_load();
        shifted.constraints = layer_entry(None, Some(param(LOAD))).constraints;
        let mut gated = bus_1_load();
        gated.constraints = layer_entry(
            Some(ResolvedCondition::Compare {
                lhs: param(LOAD),
                op: CompareOp::Gt,
                rhs: lit(0.0),
            }),
            None,
        )
        .constraints;

        assert_eq!(flags(upper, &bus_1_load()), (false, true, false));
        assert_eq!(flags(probe(), &shifted), (true, true, false));
        assert_eq!(flags(probe(), &gated), (false, false, true));
    }

    #[test]
    fn a_capacity_of_a_plant_without_unit_groups_is_an_availability_error() {
        let layer = ScenarioLayer {
            availability_atoms: vec![(
                EntityId(900),
                AvailabilityAtom {
                    hydro_id: EntityId(99),
                    block_id: None,
                },
            )],
            ..ScenarioLayer::default()
        };
        let err = build_with_layer(probe(), &layer, &table(vec![], &[]), &[2, 2]).unwrap_err();
        assert!(
            matches!(
                err,
                ResolvedGenericCoefficientsError::Availability(AvailabilityError::NoUnitGroups {
                    hydro_id: EntityId(99),
                    stage_id: 3,
                })
            ),
            "{err:?}"
        );
    }

    const PISO: i32 = 5;
    const MARGIN: i32 = 4;

    /// `@PISO` at study stage `s`, block `b` is `10 + 2s + b`.
    fn piso() -> ResolvedParameters {
        table(vec![vec![vec![10.0, 11.0], vec![12.0, 13.0]]], &[PISO])
    }

    #[test]
    fn a_non_finite_bound_names_the_stage_and_the_block_of_its_cell() {
        let mut r = piso();
        r.per_param[0][1][1] = f64::NAN;
        let mut c = probe();
        c.bound_lower = Some(param(PISO));
        let err = ResolvedGenericCoefficients::build(&[c], &r, &TWO_STAGES, &[2, 2]).unwrap_err();
        assert!(
            matches!(
                err,
                ResolvedGenericCoefficientsError::NonFinite {
                    constraint_id: 500,
                    stage_id: 4,
                    block: 1,
                    ..
                }
            ),
            "{err:?}"
        );
        let message = err.to_string();
        for needle in ["500", "the lower bound", "stage 4", "block 1"] {
            assert!(message.contains(needle), "{needle} missing from: {message}");
        }
    }

    const BUS_2_LOAD: i32 = 942;
    const BUS_3_LOAD: i32 = 943;
    const BUS_2: [EntityId; 1] = [EntityId(2)];
    const BUSES_2_AND_3: [EntityId; 2] = [EntityId(2), EntityId(3)];

    /// Bus 2, noise member 0, and bus 3, at position 1 of the system, whose load its
    /// load-balance rows hold.
    fn buses_2_and_3() -> LoadBuses<'static> {
        LoadBuses {
            buses_with_load: &BUSES_2_AND_3,
            noise_members: &BUS_2,
            bus_ids: &BUSES_2_AND_3,
        }
    }

    /// The load of bus 2, [`BUS_2_LOAD`], and of bus 3, [`BUS_3_LOAD`], at the row's block.
    fn buses_2_and_3_loads() -> ScenarioLayer {
        ScenarioLayer {
            data_atoms: vec![
                (
                    EntityId(BUS_2_LOAD),
                    DataAtom::BusLoadValue {
                        bus_id: EntityId(2),
                        block_id: None,
                    },
                ),
                (
                    EntityId(BUS_3_LOAD),
                    DataAtom::BusLoadValue {
                        bus_id: EntityId(3),
                        block_id: None,
                    },
                ),
            ],
            ..ScenarioLayer::default()
        }
    }

    /// `bus_load_value(2) + bus_load_value(3)`: the load of every bus of [`buses_2_and_3`].
    fn sum_of_bus_loads() -> ScalarExpr {
        param(BUS_2_LOAD) + param(BUS_3_LOAD)
    }

    fn compare(lhs: ScalarExpr, op: CompareOp, rhs: ScalarExpr) -> ResolvedCondition {
        ResolvedCondition::Compare { lhs, op, rhs }
    }

    fn probe_as(id: i32) -> GenericConstraint {
        GenericConstraint {
            id: EntityId(id),
            ..probe()
        }
    }

    /// The rows of `constraints` at blocks `0..n_blocks`, constraint-major, without table bases.
    fn rows_of(constraints: &[GenericConstraint], n_blocks: usize) -> Vec<PerSolveRow> {
        (0..)
            .zip(constraints)
            .flat_map(|(constraint_idx, constraint)| {
                (0..n_blocks).map(move |block_idx| PerSolveRow {
                    row: 0,
                    constraint_idx,
                    constraint_id: constraint.id.0,
                    block_idx,
                    is_stage_level: false,
                    base_lower: None,
                    base_upper: None,
                    volume_shift: None,
                })
            })
            .collect()
    }

    /// derived: bus 2 is noise member 0, realized at 80 and 90, and bus 3 holds 40 and 50 on its
    /// load-balance rows, unscaled or scaled by 2 and 4; so `100 - 0.5·load(2)` is 60 and 55,
    /// `0.05·(load(2) + load(3))` is `0.05·(80 + 40)` and `0.05·(90 + 50)`, and the fixed lower
    /// bound 10 holds at both blocks.
    #[test]
    fn a_solve_bounds_generic_rows_by_the_noise_and_balance_loads_without_allocating() {
        let mut bus = probe_as(701);
        bus.bound_upper = Some(lit(100.0) - lit(0.5) * param(BUS_2_LOAD));
        let mut system = probe_as(702);
        system.bound_lower = Some(lit(0.05) * sum_of_bus_loads());
        let mut fixed = probe_as(703);
        fixed.bound_lower = Some(lit(10.0));
        let constraints = [bus, system, fixed];
        let rgc = build_with_loads(
            &constraints,
            &buses_2_and_3_loads(),
            &table(vec![], &[]),
            &[2],
            buses_2_and_3(),
        )
        .unwrap();
        let rows = rows_of(&constraints, 2);
        let geometry = geometry_with_load_balance(0, 2, 2);
        let unscaled = StageLoads {
            noise: &[80.0, 90.0],
            row_lower: &[0.0, 0.0, 40.0, 50.0],
            row_scale: &[],
            geometry: &geometry,
        };
        let scaled = StageLoads {
            row_lower: &[0.0, 0.0, 80.0, 200.0],
            row_scale: &[1.0, 1.0, 2.0, 4.0],
            ..unscaled
        };
        let mut scratch = RowBoundScratch::new(rgc.per_solve_tables(), rows.len());
        let reserved = scratch.footprint();

        for loads in [&unscaled, &scaled, &unscaled] {
            rgc.evaluate_bounds(0, &rows, loads, &mut scratch);
            assert_eq!(scratch.upper()[..2], [60.0, 55.0]);
            assert_eq!(scratch.lower()[..2], [f64::NEG_INFINITY; 2]);
            assert_eq!(scratch.lower()[2..4], [6.0, 7.0]);
            assert_eq!(scratch.upper()[2..4], [f64::INFINITY; 2]);
            assert_eq!(scratch.lower()[4..], [10.0, 10.0]);
            assert_eq!(scratch.footprint(), reserved);
        }
    }

    /// derived: bus 2 realizes 80 and 90, then 20 and 30, and `@MARGIN` is 5 and 7 at the blocks
    /// of the stage, so the upper bound `100 - 0.5·load(2) - @MARGIN` is `100 - 0.5·80 - 5` and
    /// `100 - 0.5·90 - 7`, then 85 and 78.
    #[test]
    fn a_solve_bounds_a_generic_row_by_the_realized_load_and_a_parameter() {
        let mut c = probe_as(701);
        c.bound_upper = Some(lit(100.0) - lit(0.5) * param(BUS_2_LOAD) - param(MARGIN));
        let constraints = [c];
        let rgc = build_with_loads(
            &constraints,
            &buses_2_and_3_loads(),
            &table(vec![vec![vec![5.0, 7.0]]], &[MARGIN]),
            &[2],
            buses_2_and_3(),
        )
        .unwrap();
        let rows = rows_of(&constraints, 2);
        let geometry = geometry_with_load_balance(0, 2, 2);
        let mut scratch = RowBoundScratch::new(rgc.per_solve_tables(), rows.len());

        for (noise, upper) in [([80.0, 90.0], [55.0, 48.0]), ([20.0, 30.0], [85.0, 78.0])] {
            let loads = StageLoads {
                noise: &noise,
                row_lower: &[],
                row_scale: &[],
                geometry: &geometry,
            };
            rgc.evaluate_bounds(0, &rows, &loads, &mut scratch);
            assert_eq!(scratch.upper(), upper);
            assert_eq!(scratch.lower(), [f64::NEG_INFINITY; 2]);
        }
    }

    /// derived: buses 2 and 3 load 50 + 30 = 80 at block 0 and 70 + 50 = 120 at block 1, bus 3
    /// on its load-balance rows, unscaled or scaled by 2 and 4; so the condition
    /// `load(2) + load(3) <= 100` of constraint 701 holds at block 0 only, and the selector
    /// bounding 702 is 2, then 5.
    #[test]
    fn a_solve_opens_a_generic_row_where_its_condition_fails_and_reads_a_selector_per_block() {
        let mut layer = buses_2_and_3_loads();
        layer.selectors = vec![(
            EntityId(950),
            ConditionalScalar {
                cond: compare(sum_of_bus_loads(), CompareOp::Gt, lit(100.0)),
                then: lit(5.0),
                else_: lit(2.0),
            },
        )];
        layer.constraints = vec![ConstraintScenario {
            constraint_id: EntityId(701),
            active_when: Some(compare(sum_of_bus_loads(), CompareOp::Le, lit(100.0))),
            endpoint_shift: None,
        }];
        let mut gated = probe_as(701);
        gated.bound_upper = Some(lit(10.0));
        let mut selected = probe_as(702);
        selected.bound_lower = Some(param(950));
        let constraints = [gated, selected];
        let rgc = build_with_loads(
            &constraints,
            &layer,
            &table(vec![], &[]),
            &[2],
            buses_2_and_3(),
        )
        .unwrap();
        let rows = rows_of(&constraints, 2);
        let geometry = geometry_with_load_balance(0, 2, 2);
        let unscaled = StageLoads {
            noise: &[50.0, 70.0],
            row_lower: &[0.0, 0.0, 30.0, 50.0],
            row_scale: &[],
            geometry: &geometry,
        };
        let scaled = StageLoads {
            row_lower: &[0.0, 0.0, 60.0, 200.0],
            row_scale: &[1.0, 1.0, 2.0, 4.0],
            ..unscaled
        };
        let mut scratch = RowBoundScratch::new(rgc.per_solve_tables(), rows.len());
        let reserved = scratch.footprint();

        for loads in [&unscaled, &scaled, &unscaled] {
            rgc.evaluate_bounds(0, &rows, loads, &mut scratch);
            assert_eq!(scratch.upper()[..2], [10.0, f64::INFINITY]);
            assert_eq!(scratch.lower()[..2], [f64::NEG_INFINITY; 2]);
            assert_eq!(scratch.lower()[2..], [2.0, 5.0]);
            assert_eq!(scratch.upper()[2..], [f64::INFINITY; 2]);
            assert_eq!(scratch.row_active(), [true, false, true, true]);
            assert_eq!(scratch.footprint(), reserved);
        }
    }

    /// The bounds and activation of `rows` of stage 0 of `rgc`, one block, under the load
    /// `bus_2` of bus 2, realized as noise member 0, and `bus_3` of bus 3, held on its
    /// load-balance row.
    fn evaluate_buses_2_and_3(
        rgc: &ResolvedGenericCoefficients,
        rows: &[PerSolveRow],
        [bus_2, bus_3]: [f64; 2],
        scratch: &mut RowBoundScratch,
    ) -> Vec<(f64, f64, bool)> {
        let geometry = geometry_with_load_balance(0, 2, 1);
        let loads = StageLoads {
            noise: &[bus_2],
            row_lower: &[0.0, bus_3],
            row_scale: &[],
            geometry: &geometry,
        };
        rgc.evaluate_bounds(0, rows, &loads, scratch);
        (0..rows.len())
            .map(|i| {
                (
                    scratch.lower()[i],
                    scratch.upper()[i],
                    scratch.row_active()[i],
                )
            })
            .collect()
    }

    /// derived: buses 2 and 3 load 50 + 30 = 80 in the first solve and 60 + 40 = 100 in the
    /// second, so `bus_load_value(2) + bus_load_value(3) <= 85` holds in the first, where the row
    /// takes its upper bound 10, and fails in the second, where the row is open.
    #[test]
    fn a_condition_over_the_sum_of_bus_loads_gates_the_row() {
        let mut layer = buses_2_and_3_loads();
        layer.constraints = layer_entry(
            Some(compare(sum_of_bus_loads(), CompareOp::Le, lit(85.0))),
            None,
        )
        .constraints;
        let mut c = probe();
        c.bound_upper = Some(lit(10.0));
        let rgc =
            build_with_loads(&[c], &layer, &table(vec![], &[]), &[1], buses_2_and_3()).unwrap();
        let rows = [per_solve_row(0, (None, None))];
        let mut scratch = RowBoundScratch::new(rgc.per_solve_tables(), rows.len());

        assert_eq!(
            evaluate_buses_2_and_3(&rgc, &rows, [50.0, 30.0], &mut scratch),
            [(f64::NEG_INFINITY, 10.0, true)]
        );
        assert_eq!(
            evaluate_buses_2_and_3(&rgc, &rows, [60.0, 40.0], &mut scratch),
            [(f64::NEG_INFINITY, f64::INFINITY, false)]
        );
    }

    /// derived: buses 2 and 3 load 50 + 30 = 80 in the first solve and 60 + 40 = 100 in the
    /// second, so the upper bound `0.1·(bus_load_value(2) + bus_load_value(3))` is `0.1·80` and
    /// `0.1·100`, which round to exactly 8 and 10.
    #[test]
    fn a_bound_reads_the_sum_of_bus_loads() {
        let upper_bounds = |bound: ScalarExpr, layer: &ScenarioLayer| {
            let mut c = probe();
            c.bound_upper = Some(bound);
            let rgc =
                build_with_loads(&[c], layer, &table(vec![], &[]), &[1], buses_2_and_3()).unwrap();
            let rows = [per_solve_row(0, (None, None))];
            let mut scratch = RowBoundScratch::new(rgc.per_solve_tables(), rows.len());
            [[50.0, 30.0], [60.0, 40.0]]
                .map(|loads| evaluate_buses_2_and_3(&rgc, &rows, loads, &mut scratch)[0].1)
        };

        let sum = upper_bounds(lit(0.1) * sum_of_bus_loads(), &buses_2_and_3_loads());
        assert_eq!(sum, [8.0, 10.0]);
    }

    /// derived: a per-solve bound, a selector and a condition read `@PISO` from the setup values,
    /// `10 + 2s + b` at study stage `s` and block `b`.
    #[test]
    fn a_parameter_a_generic_solve_reads_is_a_setup_value() {
        let mut bound = probe();
        bound.bound_upper = Some(param(PISO) + param(BUS_2_LOAD));
        let mut selector_layer = buses_2_and_3_loads();
        selector_layer.selectors = vec![(
            EntityId(950),
            ConditionalScalar {
                cond: compare(sum_of_bus_loads(), CompareOp::Gt, param(PISO)),
                then: lit(5.0),
                else_: lit(6.0),
            },
        )];
        let mut selected = probe();
        selected.bound_upper = Some(param(950));
        let mut gated_layer = buses_2_and_3_loads();
        gated_layer.constraints = layer_entry(
            Some(compare(sum_of_bus_loads(), CompareOp::Le, param(PISO))),
            None,
        )
        .constraints;
        let mut gated = probe();
        gated.bound_upper = Some(lit(10.0));

        for (c, layer) in [
            (bound, buses_2_and_3_loads()),
            (selected, selector_layer),
            (gated, gated_layer),
        ] {
            let rgc = build_with_loads(&[c], &layer, &piso(), &[2, 2], buses_2_and_3()).unwrap();
            for (stage, block, value) in [(0, 0, 10.0), (0, 1, 11.0), (1, 0, 12.0), (1, 1, 13.0)] {
                assert_eq!(rgc.setup_value(EntityId(PISO), stage, block), Some(value));
            }
        }
    }

    /// derived: each constraint resolves on its own, so its terms, its endpoints and the bounds a
    /// solve writes on its rows are those of the same constraint in either order, and the buses,
    /// loads and selectors a solve reads are ascending by id. Buses 2 and 3 load 50 + 30 = 80 at
    /// block 0 and 70 + 50 = 120 at block 1: the condition of 401 holds at block 0, that of 705
    /// at block 1, and selector 950, the upper bound of 701, is 2, then 5.
    #[test]
    fn the_resolution_of_a_constraint_does_not_depend_on_the_order_of_the_constraints() {
        let mut layer = buses_2_and_3_loads();
        layer.selectors = vec![(
            EntityId(950),
            ConditionalScalar {
                cond: compare(sum_of_bus_loads(), CompareOp::Gt, lit(100.0)),
                then: lit(5.0),
                else_: lit(2.0),
            },
        )];
        layer.constraints = vec![
            ConstraintScenario {
                constraint_id: EntityId(401),
                active_when: Some(compare(sum_of_bus_loads(), CompareOp::Le, lit(100.0))),
                endpoint_shift: None,
            },
            ConstraintScenario {
                constraint_id: EntityId(705),
                active_when: Some(compare(param(BUS_2_LOAD), CompareOp::Gt, lit(60.0))),
                endpoint_shift: None,
            },
        ];
        let mut parameter = probe_as(705);
        parameter.bound_lower = Some(param(PISO));
        parameter.bound_upper = Some(lit(0.5) * param(BUS_2_LOAD));
        let mut coefficient = GenericConstraint {
            id: EntityId(401),
            ..constraint(vec![term(lit(2.0) * param(PISO), hydro_storage(2))])
        };
        coefficient.bound_upper = Some(lit(10.0));
        let mut selected = probe_as(701);
        selected.bound_upper = Some(param(950));
        let declared = [parameter, coefficient, selected];
        let mut permuted = declared.clone();
        permuted.rotate_left(1);
        let ids = |constraints: &[GenericConstraint]| -> Vec<i32> {
            constraints.iter().map(|c| c.id.0).collect()
        };
        assert_ne!(ids(&declared), ids(&permuted));

        let resolve = |constraints: &[GenericConstraint]| {
            let rgc =
                build_with_loads(constraints, &layer, &piso(), &[2], buses_2_and_3()).unwrap();
            let rows = rows_of(constraints, 2);
            let geometry = geometry_with_load_balance(0, 2, 2);
            let loads = StageLoads {
                noise: &[50.0, 70.0],
                row_lower: &[0.0, 0.0, 30.0, 50.0],
                row_scale: &[],
                geometry: &geometry,
            };
            let mut scratch = RowBoundScratch::new(rgc.per_solve_tables(), rows.len());
            rgc.evaluate_bounds(0, &rows, &loads, &mut scratch);
            let mut solved = Vec::new();
            let mut resolved = Vec::new();
            for (constraint_idx, c) in constraints.iter().enumerate() {
                let blocks = constraint_idx * 2..constraint_idx * 2 + 2;
                solved.push((
                    c.id.0,
                    blocks
                        .map(|i| {
                            (
                                scratch.lower()[i],
                                scratch.upper()[i],
                                scratch.row_active()[i],
                            )
                        })
                        .collect::<Vec<_>>(),
                ));
                let terms: Vec<f64> = (0..c.expression.terms.len())
                    .flat_map(|t| (0..2).map(move |b| (t, b)))
                    .map(|(t, b)| rgc.term_value(constraint_idx, t, 0, b))
                    .collect();
                let endpoints: Vec<RowEndpoints> = (0..2)
                    .map(|b| rgc.endpoints(constraint_idx, 0, b, None, None))
                    .collect();
                resolved.push((
                    c.id.0,
                    format!(
                        "{terms:?} {endpoints:?} {} {}",
                        rgc.is_gated(constraint_idx),
                        rgc.is_block_varying(constraint_idx)
                    ),
                ));
            }
            solved.sort_by_key(|&(id, _)| id);
            resolved.sort_by_key(|&(id, _)| id);
            let tables = rgc.per_solve_tables();
            let read = format!(
                "{:?} {:?} {:?}",
                tables.load_buses, tables.load_atoms, tables.selectors
            );
            (solved, resolved, read)
        };
        let (declared_solved, declared_resolved, declared_read) = resolve(&declared);
        let (permuted_solved, permuted_resolved, permuted_read) = resolve(&permuted);

        let open = (f64::NEG_INFINITY, f64::INFINITY, false);
        assert_eq!(
            declared_solved,
            [
                (401, vec![(f64::NEG_INFINITY, 10.0, true), open]),
                (
                    701,
                    vec![
                        (f64::NEG_INFINITY, 2.0, true),
                        (f64::NEG_INFINITY, 5.0, true)
                    ]
                ),
                (705, vec![open, (11.0, 0.5 * 70.0, true)]),
            ]
        );
        assert_eq!(
            format!("{declared_solved:?}"),
            format!("{permuted_solved:?}")
        );
        assert_eq!(declared_resolved, permuted_resolved);
        assert_eq!(declared_read, permuted_read);
    }

    /// `(None, None)`: an endpoint neither side targets stays untargeted.
    #[test]
    fn fold_endpoint_both_absent_stays_none() {
        assert_eq!(fold_endpoint(None, None), None);
    }

    /// `(Some(base), None)`: a pure literal passes through unchanged.
    #[test]
    fn fold_endpoint_pure_literal_is_unchanged() {
        assert_eq!(fold_endpoint(Some(95.0), None), Some(95.0));
    }

    /// `(None, Some(remainder))`: with no parquet base, the remainder alone
    /// establishes the endpoint; `7.0` stands for a single-parameter remainder
    /// resolved to `7.0`.
    #[test]
    fn fold_endpoint_pure_affine_establishes_endpoint() {
        assert_eq!(fold_endpoint(None, Some(7.0)), Some(7.0));
    }

    /// `(Some(base), Some(remainder))` with a constant-only remainder (`-5.0`)
    /// shifts the base by the constant — the fold is `base + R`, never `R`
    /// replacing `base`.
    #[test]
    fn fold_endpoint_literal_and_constant_affine_composes() {
        assert_eq!(fold_endpoint(Some(100.0), Some(-5.0)), Some(95.0));
    }

    /// `(Some(base), Some(remainder))` with a `@param`-bearing remainder (`4.0`,
    /// the parameter's value at block 1) folds exactly as a constant-only one does —
    /// the fold does not distinguish the shape.
    #[test]
    fn fold_endpoint_literal_and_param_affine_composes() {
        assert_eq!(fold_endpoint(Some(100.0), Some(4.0)), Some(104.0));
    }

    /// An `==`-normalized constant remainder (`2.0`) assigned to both endpoints
    /// folds each independently against its own parquet base.
    #[test]
    fn fold_endpoint_equals_normalized_folds_both_endpoints() {
        assert_eq!(
            fold_endpoint(Some(10.0), Some(2.0)),
            Some(12.0),
            "lower endpoint folds against its own base"
        );
        assert_eq!(
            fold_endpoint(Some(50.0), Some(2.0)),
            Some(52.0),
            "upper endpoint folds against its own base"
        );
    }
}
