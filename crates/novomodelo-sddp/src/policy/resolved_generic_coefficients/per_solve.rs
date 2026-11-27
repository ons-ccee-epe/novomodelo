//! The bounds every solve writes on the rows of its stage whose bounds read scenario data.

use novomodelo_core::{EntityId, ScalarExpr};
use novomodelo_io::constraints::scenario::{ConditionalScalar, ResolvedCondition};

use crate::lp::builder::StageGeometry;
use crate::lp::indexer::{BlockIdx, BusSys};

/// Where a solve finds the load of a bus.
#[derive(Debug, Clone, Copy)]
pub(crate) enum LoadSource {
    /// The realized load of noise member `member`.
    Noise { member: usize },
    /// The right-hand side of the load-balance row of the bus at `bus_pos`: the stage template
    /// owns a deterministic load.
    Balance { bus_pos: usize },
}

/// A load value a per-solve bound reads: the load of the bus at `slot` of the load buses, at
/// `block` or the block of the row.
#[derive(Debug, Clone, Copy)]
pub(crate) struct BusLoadRead {
    pub(crate) slot: usize,
    pub(crate) block: Option<usize>,
}

/// What a solve reads besides its rows.
#[derive(Debug)]
pub(crate) struct PerSolveTables {
    /// The value of every cell, per plant-capacity or per-solve-read parameter, ascending by id.
    pub(crate) setup_values: Vec<(EntityId, Vec<f64>)>,
    /// `block_offsets[s]..block_offsets[s + 1]` is study stage `s`'s range in a series.
    pub(crate) block_offsets: Vec<usize>,
    /// The buses whose load a program a solve evaluates reads, ascending, with where a solve
    /// finds it.
    pub(crate) load_buses: Vec<(EntityId, LoadSource)>,
    /// The load values the programs a solve evaluates read, ascending by parameter id.
    pub(crate) load_atoms: Vec<(EntityId, BusLoadRead)>,
    /// The activation conditions gating some row; a row's gate is a position here.
    pub(crate) conditions: Vec<ResolvedCondition>,
    /// The selectors of the rows' relations, ascending by id.
    pub(crate) selectors: Vec<(EntityId, ConditionalScalar)>,
    /// The largest [`ScalarExpr::op_count`] of the programs a solve evaluates.
    pub(crate) max_op_count: usize,
}

/// The bounds and activation of each row a solve bounds, in the order of its
/// [`RowBoundScratch::push_row_bounds`] calls, with the loads, values and condition states they
/// read.
#[derive(Debug, Default)]
pub(crate) struct RowBoundScratch {
    lower: Vec<f64>,
    upper: Vec<f64>,
    /// Whether each row's gate holds at its block; a row without a gate is always active. An
    /// active row may still be open on both sides, so this is not derivable from the bounds.
    row_active: Vec<bool>,
    /// The load of each load bus, bus-major and block-minor.
    loads: Vec<f64>,
    /// The value of each load atom and then of each selector, value-major and block-minor.
    values: Vec<f64>,
    /// Whether each condition holds, condition-major and block-minor.
    active: Vec<bool>,
    stack: Vec<f64>,
    /// The block count of the stage [`PerSolveTables::begin_stage`] last began: the stride of
    /// `loads`, `values` and `active`.
    n_blks: usize,
}

/// The scenario load of one solve's stage.
pub(crate) struct StageLoads<'a> {
    /// The realized load of each noise member, member-major and block-minor; empty without load
    /// noise.
    pub(crate) noise: &'a [f64],
    /// The stage template's row lower bounds, scaled by `row_scale`.
    pub(crate) row_lower: &'a [f64],
    /// The stage template's row scale; empty when the rows are unscaled.
    pub(crate) row_scale: &'a [f64],
    pub(crate) geometry: &'a StageGeometry,
}

impl RowBoundScratch {
    /// Room for `max_rows` rows, the loads, values and condition states of `tables` at the most
    /// blocks of a stage, and the longest program a solve evaluates, so evaluating any stage of
    /// at most `max_rows` rows allocates nothing.
    #[must_use]
    pub(crate) fn new(tables: &PerSolveTables, max_rows: usize) -> Self {
        let max_blocks = tables
            .block_offsets
            .windows(2)
            .map(|stage| stage[1] - stage[0])
            .max()
            .unwrap_or(0);
        Self {
            lower: Vec::with_capacity(max_rows),
            upper: Vec::with_capacity(max_rows),
            row_active: Vec::with_capacity(max_rows),
            loads: Vec::with_capacity(tables.load_buses.len() * max_blocks),
            values: Vec::with_capacity(
                (tables.load_atoms.len() + tables.selectors.len()) * max_blocks,
            ),
            active: Vec::with_capacity(tables.conditions.len() * max_blocks),
            stack: Vec::with_capacity(tables.max_op_count),
            n_blks: 0,
        }
    }

    #[must_use]
    pub(crate) fn lower(&self) -> &[f64] {
        &self.lower
    }

    #[must_use]
    pub(crate) fn upper(&self) -> &[f64] {
        &self.upper
    }

    #[must_use]
    pub(crate) fn row_active(&self) -> &[bool] {
        &self.row_active
    }

    /// Whether the condition at position `gate` of [`PerSolveTables::conditions`] holds at block
    /// `block_idx` of the stage [`PerSolveTables::begin_stage`] last began; always without a
    /// gate.
    #[must_use]
    pub(crate) fn holds(&self, gate: Option<usize>, block_idx: usize) -> bool {
        gate.is_none_or(|gate| self.active[gate * self.n_blks + block_idx])
    }

    /// Push the bounds and activation of the next row.
    pub(crate) fn push_row_bounds(&mut self, lower: f64, upper: f64, active: bool) {
        debug_assert!(
            self.lower.len() < self.lower.capacity(),
            "the stage has more rows than the scratch holds without allocating ({})",
            self.lower.capacity(),
        );
        self.lower.push(lower);
        self.upper.push(upper);
        self.row_active.push(active);
    }
}

#[cfg(test)]
impl RowBoundScratch {
    /// A scratch holding only the row bounds and activation of one stage's rows.
    pub(crate) fn with_row_bounds(lower: Vec<f64>, upper: Vec<f64>, row_active: Vec<bool>) -> Self {
        Self {
            lower,
            upper,
            row_active,
            ..Self::default()
        }
    }

    /// The address and capacity of every buffer.
    pub(crate) fn footprint(&self) -> [(*const (), usize); 7] {
        let of = |buffer: &Vec<f64>| (buffer.as_ptr().cast(), buffer.capacity());
        let of_flags = |buffer: &Vec<bool>| (buffer.as_ptr().cast(), buffer.capacity());
        [
            of(&self.lower),
            of(&self.upper),
            of_flags(&self.row_active),
            of(&self.loads),
            of(&self.values),
            of_flags(&self.active),
            of(&self.stack),
        ]
    }
}

impl StageLoads<'_> {
    fn load(&self, source: LoadSource, blk: usize) -> f64 {
        match source {
            LoadSource::Noise { member } => self.noise[member * self.geometry.n_blks + blk],
            LoadSource::Balance { bus_pos } => {
                let row = self
                    .geometry
                    .load_balance_row(BusSys::new(bus_pos), BlockIdx::new(blk));
                if self.row_scale.is_empty() {
                    self.row_lower[row]
                } else {
                    self.row_lower[row] / self.row_scale[row]
                }
            }
        }
    }
}

impl PerSolveTables {
    pub(crate) const fn empty() -> Self {
        Self {
            setup_values: Vec::new(),
            block_offsets: Vec::new(),
            load_buses: Vec::new(),
            load_atoms: Vec::new(),
            conditions: Vec::new(),
            selectors: Vec::new(),
            max_op_count: 0,
        }
    }

    /// The value setup parameter `id` holds at `(stage_idx, block_idx)`; `None` when `id` is not
    /// one.
    #[must_use]
    pub(crate) fn setup_value(
        &self,
        id: EntityId,
        stage_idx: usize,
        block_idx: usize,
    ) -> Option<f64> {
        let idx = self.setup_idx(id)?;
        let stage = self.block_offsets[stage_idx]..self.block_offsets[stage_idx + 1];
        Some(self.setup_values[idx].1[stage][block_idx])
    }

    pub(crate) fn setup_idx(&self, id: EntityId) -> Option<usize> {
        self.setup_values
            .binary_search_by_key(&id, |&(setup_id, _)| setup_id)
            .ok()
    }

    /// Clear the rows of `scratch` and set, per block of study stage `stage_idx` under the
    /// scenario load `loads`, the load of each load bus, the value of each load atom, the value
    /// its condition picks for each selector, in ascending id order so a selector another reads
    /// is already set, and whether each condition holds.
    pub(crate) fn begin_stage(
        &self,
        stage_idx: usize,
        loads: &StageLoads<'_>,
        scratch: &mut RowBoundScratch,
    ) {
        let n_blks = loads.geometry.n_blks;
        scratch.n_blks = n_blks;
        scratch.loads.clear();
        for &(_, source) in &self.load_buses {
            for blk in 0..n_blks {
                scratch.loads.push(loads.load(source, blk));
            }
        }
        scratch.values.clear();
        for &(_, BusLoadRead { slot, block }) in &self.load_atoms {
            for blk in 0..n_blks {
                let value = scratch.loads[slot * n_blks + block.unwrap_or(blk)];
                scratch.values.push(value);
            }
        }
        for (_, selector) in &self.selectors {
            for blk in 0..n_blks {
                let leaf = |id| self.per_solve_value(id, stage_idx, blk, n_blks, &scratch.values);
                let stack = &mut scratch.stack;
                let branch = if selector
                    .cond
                    .holds(&mut |side| side.evaluate_in(stack, leaf))
                {
                    &selector.then
                } else {
                    &selector.else_
                };
                let value = branch.evaluate_in(stack, leaf);
                scratch.values.push(value);
            }
        }
        scratch.active.clear();
        for condition in &self.conditions {
            for blk in 0..n_blks {
                let leaf = |id| self.per_solve_value(id, stage_idx, blk, n_blks, &scratch.values);
                let stack = &mut scratch.stack;
                let holds = condition.holds(&mut |side| side.evaluate_in(stack, leaf));
                scratch.active.push(holds);
            }
        }
        scratch.lower.clear();
        scratch.upper.clear();
        scratch.row_active.clear();
    }

    /// The value of `program` over the setup values, the load values and the selector values at
    /// block `block_idx` of study stage `stage_idx`, the stage
    /// [`begin_stage`](Self::begin_stage) last began.
    pub(crate) fn evaluate(
        &self,
        program: &ScalarExpr,
        stage_idx: usize,
        block_idx: usize,
        scratch: &mut RowBoundScratch,
    ) -> f64 {
        let RowBoundScratch {
            stack,
            values,
            n_blks,
            ..
        } = scratch;
        program.evaluate_in(stack, |id| {
            self.per_solve_value(id, stage_idx, block_idx, *n_blks, values)
        })
    }

    fn per_solve_value(
        &self,
        id: EntityId,
        stage_idx: usize,
        block_idx: usize,
        n_blks: usize,
        values: &[f64],
    ) -> f64 {
        if let Some(value) = self.setup_value(id, stage_idx, block_idx) {
            return value;
        }
        let slot = self
            .load_atoms
            .binary_search_by_key(&id, |&(atom_id, _)| atom_id)
            .or_else(|_| {
                self.selectors
                    .binary_search_by_key(&id, |&(selector_id, _)| selector_id)
                    .map(|selector| self.load_atoms.len() + selector)
            });
        debug_assert!(
            slot.is_ok(),
            "{id:?} is read per solve but is neither a setup value, a load value nor a selector"
        );
        slot.map_or(f64::NAN, |slot| values[slot * n_blks + block_idx])
    }
}
