//! The generic-constraint row family of each stage LP as facts keyed by identity: a row by
//! `(constraint_id, block_id)`, a column by its position among the columns outside the family.
//! The same formulation built in another row or column order gives the same facts. They are read
//! from the unscaled templates, and every `f64` is its `{:?}` text, which parses back to the same
//! bits.

use std::fmt::Write;

use novomodelo_solver::StageTemplate;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::StudySetup;
use crate::lp::builder::RowOrigin;
use crate::policy::resolved_generic_coefficients::{BoundSide, RowBoundScratch, StageLoads};

/// The row family of every study stage.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConstraintRowFacts {
    /// One per study stage, in stage order.
    pub stages: Vec<StageRowFacts>,
}

/// The row family of one study stage.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StageRowFacts {
    /// Study stage id.
    pub stage_id: i32,
    /// Blocks of the stage.
    pub n_blocks: usize,
    /// Lowercase hex SHA-256 of the stage LP without the family's rows and columns.
    pub outside_family_sha256: String,
    /// One per family row, ascending by `(constraint_id, block_id)`.
    pub rows: Vec<RowFacts>,
    /// One per probe, in the order given.
    pub probes: Vec<ProbeFacts>,
}

/// One row of the family.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RowFacts {
    /// Generic constraint id.
    pub constraint_id: i32,
    /// Block index within the stage; `0` for a stage-level row.
    pub block_id: usize,
    /// Whether the row stands for every block of the stage.
    pub is_stage_level: bool,
    /// The row's entries outside its own slack columns, ascending by column.
    pub entries: Vec<EntryFacts>,
    /// Lower side.
    pub lower: BoundFacts,
    /// Upper side.
    pub upper: BoundFacts,
    /// `None` for a row without slack columns.
    pub slack: Option<SlackFacts>,
    /// Whether an `active_when` condition gates the row.
    pub gated: bool,
}

/// One entry of a family row.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EntryFacts {
    /// The entry's column.
    pub column: ColumnKey,
    /// Coefficient.
    pub value: String,
}

/// A column a family row reads, by identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ColumnKey {
    /// A column outside the family.
    Outside {
        /// Position once the family's columns are removed.
        index: usize,
    },
}

/// One side of a family row.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum BoundFacts {
    /// The side is open.
    Absent,
    /// The template's bound.
    Fixed(String),
    /// Each solve evaluates the side; the template leaves it open.
    PerSolve,
}

/// The slack of a family row.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SlackFacts {
    /// Penalty per unit of violation.
    pub penalty: String,
    /// `{:?}` of the penalty basis.
    pub basis: String,
    /// The slack that relaxes the lower side, or the upper side of a row without one.
    pub plus: SlackColumnFacts,
    /// The slack that relaxes the upper side of a two-sided row.
    pub minus: Option<SlackColumnFacts>,
}

/// One slack column of a family row.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SlackColumnFacts {
    /// The column's coefficient in its row.
    pub coefficient: String,
    /// Objective coefficient.
    pub objective: String,
    /// Column lower bound.
    pub lower: String,
    /// Column upper bound.
    pub upper: String,
}

/// The rows of one stage that a solve gates or bounds, evaluated under one probe load.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProbeFacts {
    /// The probe's load vector at the stage.
    pub loads: Vec<String>,
    /// One per gated or per-solve row, in the order of [`StageRowFacts::rows`].
    pub rows: Vec<ProbeRowFacts>,
}

/// One row under one probe load.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProbeRowFacts {
    /// Generic constraint id.
    pub constraint_id: i32,
    /// Block index within the stage.
    pub block_id: usize,
    /// Whether the row's condition holds; a row without one is always active.
    pub active: bool,
    /// The lower bound the solve writes.
    pub lower: String,
    /// The upper bound the solve writes.
    pub upper: String,
}

/// Where a template column sits relative to the family.
#[derive(Clone, Copy)]
enum Column {
    Key(ColumnKey),
    /// A slack column of the family row at `row` (relative to the first family row).
    Slack {
        row: usize,
    },
}

/// The facts of `setup`'s row family. `probes[p][stage_idx]` is the realized load of probe `p`
/// at study stage `stage_idx`: member-major and block-minor over the load-noise buses.
///
/// # Panics
///
/// Naming the stage, when a probe has no load vector for the stage or its vector does not hold one
/// value per load-noise bus and block, when the template read is scaled, when a family row has an
/// entry on a column that is neither outside the family nor the row's own slack, or when a slack
/// column has no entry in its row.
#[must_use]
pub fn constraint_row_facts(setup: &StudySetup, probes: &[Vec<Vec<f64>>]) -> ConstraintRowFacts {
    let stage_templates = &setup.inputs.stage_data.stage_templates;
    let templates = match stage_templates.pre_scaling_templates() {
        [] => stage_templates.templates.as_slice(),
        pre_scaling => pre_scaling,
    };
    let max_rows = stage_templates
        .geometry_per_stage
        .iter()
        .map(|geometry| geometry.per_solve_rows.len())
        .max()
        .unwrap_or(0);
    let mut scratch = RowBoundScratch::new(
        setup.inputs.stage_data.generic_rows.per_solve_tables(),
        max_rows,
    );
    let stages = templates
        .iter()
        .enumerate()
        .map(|(stage_idx, template)| stage_facts(setup, template, stage_idx, probes, &mut scratch))
        .collect();
    ConstraintRowFacts { stages }
}

#[expect(
    clippy::cast_sign_loss,
    reason = "CSC col_starts/row_indices are non-negative by construction"
)]
#[expect(
    clippy::panic,
    reason = "a family entry with no identity, or a slack column missing from its row, would drop a fact from the snapshot"
)]
#[expect(
    clippy::too_many_lines,
    reason = "one pass over the stage's CSC fills the row and outside facts together"
)]
fn stage_facts(
    setup: &StudySetup,
    template: &StageTemplate,
    stage_idx: usize,
    probes: &[Vec<Vec<f64>>],
    scratch: &mut RowBoundScratch,
) -> StageRowFacts {
    let StageTemplate {
        num_cols,
        num_rows,
        num_nz: _,
        col_starts,
        row_indices,
        values,
        col_lower,
        col_upper,
        objective,
        row_lower,
        row_upper,
        col_scale,
        row_scale,
    } = template;
    let stage_data = &setup.inputs.stage_data;
    let stage_id = stage_data.stages[stage_idx].id;
    assert!(
        row_scale.is_empty() && col_scale.is_empty(),
        "stage {stage_id}: the template the facts read is scaled"
    );
    let entries = &stage_data.stage_templates.generic_constraint_row_entries[stage_idx];
    let geometry = &stage_data.stage_templates.geometry_per_stage[stage_idx];
    let generic = &stage_data.generic_rows;
    let n_blks = geometry.n_blks;
    let family_start = *num_rows - entries.len();

    let mut family: Vec<Option<Column>> = vec![None; *num_cols];
    for (row, entry) in entries.iter().enumerate() {
        for col in entry
            .slack_plus_col
            .into_iter()
            .chain(entry.slack_minus_col)
        {
            family[col] = Some(Column::Slack { row });
        }
    }
    let mut outside_cols = Vec::new();
    let columns: Vec<Column> = family
        .into_iter()
        .enumerate()
        .map(|(col, column)| {
            column.unwrap_or_else(|| {
                outside_cols.push(col);
                Column::Key(ColumnKey::Outside {
                    index: outside_cols.len() - 1,
                })
            })
        })
        .collect();

    let mut row_entries: Vec<Vec<(usize, f64)>> = vec![Vec::new(); entries.len()];
    let mut outside_col_starts = vec![0];
    let mut outside_row_indices = Vec::new();
    let mut outside_values = Vec::new();
    for (col, column) in columns.iter().enumerate() {
        let csc = col_starts[col] as usize..col_starts[col + 1] as usize;
        for (&row, &value) in row_indices[csc.clone()].iter().zip(&values[csc]) {
            let row = row as usize;
            if row >= family_start {
                row_entries[row - family_start].push((col, value));
                continue;
            }
            if let Column::Key(ColumnKey::Outside { .. }) = column {
                outside_row_indices.push(row);
                outside_values.push(value);
            }
        }
        if matches!(column, Column::Key(ColumnKey::Outside { .. })) {
            outside_col_starts.push(outside_row_indices.len());
        }
    }

    let outside_family_sha256 = {
        let words = [outside_cols.len(), family_start, outside_values.len()]
            .into_iter()
            .chain(outside_col_starts)
            .chain(outside_row_indices)
            .map(|word| word as u64)
            .chain(outside_values.iter().map(|v| v.to_bits()))
            .chain(outside_cols.iter().map(|&c| col_lower[c].to_bits()))
            .chain(outside_cols.iter().map(|&c| col_upper[c].to_bits()))
            .chain(row_lower[..family_start].iter().map(|v| v.to_bits()))
            .chain(row_upper[..family_start].iter().map(|v| v.to_bits()))
            .chain(outside_cols.iter().map(|&c| objective[c].to_bits()));
        let mut hasher = Sha256::new();
        for word in words {
            hasher.update(word.to_le_bytes());
        }
        hasher
            .finalize()
            .iter()
            .fold(String::with_capacity(64), |mut hex, byte| {
                let _ = write!(hex, "{byte:02x}");
                hex
            })
    };

    let mut rows: Vec<RowFacts> = entries
        .iter()
        .enumerate()
        .map(|(i, entry)| {
            let RowOrigin::Generic { constraint_idx } = entry.origin;
            let row = family_start + i;
            let mut keyed = Vec::new();
            let mut own_slack = Vec::new();
            for &(col, value) in &row_entries[i] {
                match columns[col] {
                    Column::Key(column) => keyed.push(EntryFacts {
                        column,
                        value: fact(value),
                    }),
                    Column::Slack { row: owner } if owner == i => own_slack.push((col, value)),
                    Column::Slack { .. } => panic!(
                        "stage {stage_id}: row ({}, {}) has an entry on column {col}, which is \
                         neither outside the family nor the row's own slack",
                        entry.entity_id, entry.block_idx
                    ),
                }
            }
            keyed.sort_by_key(|entry| entry.column);
            let slack_column = |col: usize| SlackColumnFacts {
                coefficient: own_slack.iter().find(|&&(own, _)| own == col).map_or_else(
                    || {
                        panic!(
                            "stage {stage_id}: row ({}, {}) has no entry on its slack column {col}",
                            entry.entity_id, entry.block_idx
                        )
                    },
                    |&(_, value)| fact(value),
                ),
                objective: fact(objective[col]),
                lower: fact(col_lower[col]),
                upper: fact(col_upper[col]),
            };
            let slack = entry
                .slack
                .zip(entry.slack_plus_col)
                .map(|(config, plus)| SlackFacts {
                    penalty: fact(config.penalty),
                    basis: format!("{:?}", config.penalty_basis),
                    plus: slack_column(plus),
                    minus: entry.slack_minus_col.map(slack_column),
                });
            let per_solve = |endpoint: Option<f64>, side| {
                endpoint.is_some() && generic.side_is_per_solve(constraint_idx, side)
            };
            RowFacts {
                constraint_id: entry.entity_id,
                block_id: entry.block_idx,
                is_stage_level: entry.is_stage_level,
                entries: keyed,
                lower: bound(
                    per_solve(entry.bound_lower, BoundSide::Lower),
                    entry.bound_lower,
                    row_lower[row],
                ),
                upper: bound(
                    per_solve(entry.bound_upper, BoundSide::Upper),
                    entry.bound_upper,
                    row_upper[row],
                ),
                slack,
                gated: generic.is_gated(constraint_idx),
            }
        })
        .collect();
    rows.sort_by_key(|facts| (facts.constraint_id, facts.block_id));

    let n_loads = stage_data.stage_templates.n_load_buses() * n_blks;
    let probes = probes
        .iter()
        .enumerate()
        .map(|(p, probe)| {
            let noise = probe.get(stage_idx).unwrap_or_else(|| {
                panic!(
                    "stage {stage_id}: probe {p} has no load vector for it, only {} stage vectors",
                    probe.len()
                )
            });
            assert_eq!(
                noise.len(),
                n_loads,
                "stage {stage_id}: probe {p} needs one load per load-noise bus and block"
            );
            generic.evaluate_bounds(
                stage_idx,
                &geometry.per_solve_rows,
                &StageLoads {
                    noise,
                    row_lower,
                    row_scale,
                    geometry,
                },
                scratch,
            );
            let mut rows: Vec<ProbeRowFacts> = geometry
                .per_solve_rows
                .iter()
                .enumerate()
                .map(|(i, row)| ProbeRowFacts {
                    constraint_id: row.constraint_id,
                    block_id: row.block_idx,
                    active: scratch.row_active()[i],
                    lower: fact(scratch.lower()[i]),
                    upper: fact(scratch.upper()[i]),
                })
                .collect();
            rows.sort_by_key(|row| (row.constraint_id, row.block_id));
            ProbeFacts {
                loads: noise.iter().copied().map(fact).collect(),
                rows,
            }
        })
        .collect();

    StageRowFacts {
        stage_id,
        n_blocks: n_blks,
        outside_family_sha256,
        rows,
        probes,
    }
}

/// A side is per solve before it is open: the template holds a per-solve side as an endpoint at
/// infinity.
fn bound(per_solve: bool, endpoint: Option<f64>, template_value: f64) -> BoundFacts {
    match endpoint {
        _ if per_solve => BoundFacts::PerSolve,
        None => BoundFacts::Absent,
        Some(_) => BoundFacts::Fixed(fact(template_value)),
    }
}

fn fact(value: f64) -> String {
    format!("{value:?}")
}
