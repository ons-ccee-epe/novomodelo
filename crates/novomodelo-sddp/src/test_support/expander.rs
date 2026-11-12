//! Rebuilds the LP that carries the incoming-lag columns, z columns and z rows
//! from the compact template, the unscaled inflow coupling and
//! `PrecomputedPar`, addressing the lag columns through its own alias, and
//! renders a pool's cuts as rows of it. It is the oracle for the compact LP.

use std::ops::Range;

use novomodelo_core::System;
use novomodelo_solver::{RowBatch, StageTemplate};
use novomodelo_stochastic::par::precompute::PrecomputedPar;

use crate::cut::CutPool;
use crate::indexer::{CutSlot, CutStateProjection, HydroSys, StateDim, StateSpace};
use crate::lp::builder::{InflowCoupling, StageTemplates, assemble_csc};
use crate::setup::StudySetup;
use crate::setup::template_postprocess::postprocess_templates;
use crate::test_support::template_structure::column_entries;

/// Index maps from a compact stage LP to its expanded shape.
#[derive(Debug)]
pub struct ExpandedLayout {
    state: StateSpace,
    compact_storage_in_start: usize,
}

fn shifted(range: &Range<usize>, by: usize) -> Range<usize> {
    if range.is_empty() {
        range.clone()
    } else {
        range.start + by..range.end + by
    }
}

impl ExpandedLayout {
    /// Moves every region of a compact [`StateSpace`] after the lag block to its expanded column.
    #[must_use]
    pub fn of(compact: &StateSpace) -> Self {
        let n = compact.hydro_count;
        let lags = n * compact.max_par_order;
        let storage_in_start = compact.storage_in.start;
        let past_z = lags + n;

        let mut state = compact.clone();
        state.transit_buckets_out = shifted(&compact.transit_buckets_out, lags);
        state.commit_out = shifted(&compact.commit_out, lags);
        state.storage_in = shifted(&compact.storage_in, past_z);
        state.transit_buckets_in = shifted(&compact.transit_buckets_in, past_z);
        state.commit_in = shifted(&compact.commit_in, past_z);
        state.theta = compact.theta + past_z;
        state.finalize_state_column_map();

        Self {
            state,
            compact_storage_in_start: storage_in_start,
        }
    }

    /// The compact state space with every region after the lag block moved to its expanded
    /// column; read the lag and z columns from this layout, never from the state.
    #[must_use]
    pub fn state(&self) -> &StateSpace {
        &self.state
    }

    fn lag_columns(&self) -> usize {
        self.state.hydro_count * self.state.max_par_order
    }

    /// The expanded column of compact column `c`.
    #[must_use]
    pub fn col(&self, c: usize) -> usize {
        if c < self.state.hydro_count {
            c
        } else if c < self.compact_storage_in_start {
            c + self.lag_columns()
        } else {
            c + self.lag_columns() + self.state.hydro_count
        }
    }

    /// The expanded row of compact row `r`.
    #[must_use]
    pub fn row(&self, r: usize) -> usize {
        r + self.state.hydro_count
    }

    /// Hydro `h`'s realized-inflow column.
    #[must_use]
    pub fn z_col(&self, h: HydroSys) -> usize {
        self.compact_storage_in_start + self.lag_columns() + h.get()
    }

    /// Hydro `h`'s realized-inflow definition row.
    #[must_use]
    pub fn z_row(&self, h: HydroSys) -> usize {
        h.get()
    }

    /// Hydro `h`'s incoming lag column at lag `l`.
    #[must_use]
    pub fn incoming_lag_col(&self, h: HydroSys, l: usize) -> usize {
        let n = self.state.hydro_count;
        n + l * n + h.get()
    }

    /// Hydro `h`'s outgoing lag `l`: the z column at lag 0, else the incoming lag `l - 1`.
    #[must_use]
    pub fn outgoing_lag_col(&self, h: HydroSys, l: usize) -> usize {
        if l == 0 {
            self.z_col(h)
        } else {
            self.incoming_lag_col(h, l - 1)
        }
    }

    fn lag_of(&self, j: StateDim) -> Option<(HydroSys, usize)> {
        let lags = self.state.state_dim_lag_range();
        let n = self.state.hydro_count;
        lags.contains(&j.get()).then(|| {
            let offset = j.get() - lags.start;
            (HydroSys::new(offset % n), offset / n)
        })
    }

    /// The column carrying state dimension `j` out of the stage.
    #[must_use]
    pub fn outgoing_state_col(&self, j: StateDim) -> usize {
        match self.lag_of(j) {
            Some((h, l)) => self.outgoing_lag_col(h, l),
            None => self.state.lp_column_for_state(j).get(),
        }
    }

    /// The column pinned to state dimension `j` on entry to the stage.
    #[must_use]
    pub fn incoming_state_col(&self, j: StateDim) -> usize {
        match self.lag_of(j) {
            Some((h, l)) => self.incoming_lag_col(h, l),
            None => self.state.state_to_lp_incoming_column(j).get(),
        }
    }
}

/// The expanded stage templates rebuilt from a compact setup, unscaled and scaled.
#[derive(Debug)]
pub struct ExpandedStudy {
    /// The index maps the templates follow.
    pub layout: ExpandedLayout,
    /// Per-stage unscaled expanded templates.
    pub unscaled: Vec<StageTemplate>,
    /// Per-stage expanded templates after the column and row scaling.
    pub scaled: Vec<StageTemplate>,
}

fn expand_template(
    layout: &ExpandedLayout,
    par: &PrecomputedPar,
    coupling: &InflowCoupling,
    stage: usize,
    compact: &StageTemplate,
) -> StageTemplate {
    let n = layout.state.hydro_count;
    let lag_order = layout.state.max_par_order;
    let num_cols = compact.num_cols + n * (lag_order + 1);
    let num_rows = compact.num_rows + n;
    let has_par = par.n_stages() > 0;

    let mut entries = vec![Vec::new(); num_cols];
    let mut col_lower = vec![f64::NEG_INFINITY; num_cols];
    let mut col_upper = vec![f64::INFINITY; num_cols];
    let mut objective = vec![0.0; num_cols];
    for c in 0..compact.num_cols {
        let e = layout.col(c);
        entries[e] = column_entries(compact, c)
            .map(|(r, v)| (layout.row(r), v))
            .collect();
        col_lower[e] = compact.col_lower[c];
        col_upper[e] = compact.col_upper[c];
        objective[e] = compact.objective[c];
    }

    let mut row_lower = vec![0.0; num_rows];
    let mut row_upper = vec![0.0; num_rows];
    for r in 0..compact.num_rows {
        row_lower[layout.row(r)] = compact.row_lower[r];
        row_upper[layout.row(r)] = compact.row_upper[r];
    }

    for h in (0..n).map(HydroSys::new) {
        let z_row = layout.z_row(h);
        entries[layout.z_col(h)] = std::iter::once((z_row, 1.0))
            .chain(coupling.hydro_entries(h).map(|(r, v)| (layout.row(r), v)))
            .collect();
        let base = if has_par {
            par.deterministic_base(stage, h.get())
        } else {
            0.0
        };
        row_lower[z_row] = base;
        row_upper[z_row] = base;
        if has_par {
            for l in 0..lag_order.min(par.max_order()) {
                let psi = par.psi_slice(stage, h.get())[l];
                if psi != 0.0 {
                    entries[layout.incoming_lag_col(h, l)] = vec![(z_row, -psi)];
                }
            }
        }
    }

    let (col_starts, row_indices, values) = assemble_csc(&entries);
    StageTemplate {
        num_cols,
        num_rows,
        num_nz: values.len(),
        col_starts,
        row_indices,
        values,
        col_lower,
        col_upper,
        objective,
        row_lower,
        row_upper,
        col_scale: Vec::new(),
        row_scale: Vec::new(),
    }
}

/// Rebuilds the unscaled and scaled expanded stage templates from a compact setup.
#[must_use]
pub fn expand_stage_templates(system: &System, compact: &StudySetup) -> ExpandedStudy {
    let layout = ExpandedLayout::of(compact.stage_state());
    let stage_data = &compact.inputs.stage_data;
    let par = compact.inputs.stochastic.par();
    let coupling = compact.stage_ctx().inflow_coupling;

    let mut local: StageTemplates = stage_data.stage_templates.clone();
    local.templates = stage_data
        .stage_templates
        .pre_scaling_templates()
        .iter()
        .enumerate()
        .map(|(stage, template)| expand_template(&layout, par, &coupling[stage], stage, template))
        .collect();
    let cost_scale_factor = local.cost_scale_factor;
    postprocess_templates(
        &mut local,
        system,
        layout.state(),
        &stage_data.study_dims.anticipated_plants,
        cost_scale_factor,
        &stage_data.time_value,
    );

    ExpandedStudy {
        layout,
        unscaled: local.pre_scaling_templates().to_vec(),
        scaled: local.templates,
    }
}

#[expect(
    clippy::expect_used,
    reason = "the solver C API takes i32 offsets and column indices, and a stage LP is far below i32::MAX"
)]
fn csr_index(n: usize) -> i32 {
    i32::try_from(n).expect("CSR index exceeds i32::MAX")
}

/// Writes the cuts of `slots` of `pool` into `batch` as rows of the expanded LP, in list order,
/// scaled by the pool's stage's expanded `col_scale`.
///
/// # Panics
///
/// Panics if a cut's coefficient count differs from `cut_state.n_slots()`, or
/// a column index or the entry count exceeds `i32::MAX`.
pub fn expanded_cut_rows_into(
    batch: &mut RowBatch,
    pool: &CutPool,
    slots: &[usize],
    cut_state: &CutStateProjection,
    layout: &ExpandedLayout,
    col_scale: &[f64],
) {
    batch.clear();
    let state = layout.state();
    for &slot in slots {
        let (intercept, coefficients) = (pool.intercept(slot), pool.coefficient_row(slot));
        assert_eq!(
            coefficients.len(),
            cut_state.n_slots(),
            "cut slot {slot}: coefficient count"
        );
        batch.row_starts.push(csr_index(batch.values.len()));
        for (j, &coefficient) in coefficients.iter().enumerate() {
            let d = cut_state.global_state_index(CutSlot::new(j));
            if state.nonzero_state_indices.binary_search(&d).is_ok() {
                let c = layout.outgoing_state_col(d);
                batch.col_indices.push(csr_index(c));
                batch.values.push(-coefficient * col_scale[c]);
            }
        }
        batch.col_indices.push(csr_index(state.theta));
        batch.values.push(col_scale[state.theta]);
        batch.row_lower.push(intercept);
        batch.row_upper.push(f64::INFINITY);
        batch.num_rows += 1;
    }
    batch.row_starts.push(csr_index(batch.values.len()));
}

/// Per-solve bounds that exist only in the expanded LP, each written to both bounds of its row or column.
#[derive(Debug)]
pub struct ExpandedPatch {
    /// `(expanded z row, scaled right-hand side)`, one per hydro.
    pub z_rows: Vec<(usize, f64)>,
    /// `(expanded incoming column, scaled pinned value)`, one per state dimension.
    pub incoming_cols: Vec<(usize, f64)>,
}

/// The expanded LP's z-row bounds and incoming-state pins for one solve of `stage` at hydro noise
/// `inflow_eta` and incoming state `incoming_state`.
///
/// A pin is divided by its column's scale, never multiplied.
///
/// # Panics
///
/// Panics if `inflow_eta.len()` differs from the hydro count, or `incoming_state.len()` from `n_state`.
#[must_use]
pub fn expanded_solve_patch(
    compact: &StudySetup,
    expanded: &ExpandedStudy,
    stage: usize,
    inflow_eta: &[f64],
    incoming_state: &[f64],
) -> ExpandedPatch {
    let layout = &expanded.layout;
    let scaled = &expanded.scaled[stage];
    let par = compact.inputs.stochastic.par();
    assert_eq!(
        inflow_eta.len(),
        layout.state().hydro_count,
        "inflow_eta length"
    );
    assert_eq!(
        incoming_state.len(),
        layout.state().n_state,
        "incoming_state length"
    );

    let z_rows = inflow_eta
        .iter()
        .enumerate()
        .map(|(h, &eta)| {
            let row = layout.z_row(HydroSys::new(h));
            let rhs = if par.n_stages() > 0 {
                par.deterministic_base(stage, h) + par.sigma(stage, h) * eta
            } else {
                0.0
            };
            (row, rhs * scaled.row_scale[row])
        })
        .collect();
    let incoming_cols = incoming_state
        .iter()
        .enumerate()
        .map(|(j, &pinned)| {
            let col = layout.incoming_state_col(StateDim::new(j));
            (col, pinned / scaled.col_scale[col])
        })
        .collect();
    ExpandedPatch {
        z_rows,
        incoming_cols,
    }
}
