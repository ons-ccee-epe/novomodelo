//! Recording wrapper over a real solver, for tests that assert on the LP a solve saw.

use novomodelo_solver::{
    Basis, RowBatch, SolutionView, SolverError, SolverInterface, SolverStatistics, StageTemplate,
};

/// One bound-affecting call a solve site made, in the form it received it.
#[derive(Debug, Clone, PartialEq)]
pub enum BoundWrite {
    /// `add_rows`: `count` rows appended at row `first_row`.
    AddRows {
        /// Index of the first appended row in the loaded LP.
        first_row: usize,
        /// Number of rows appended.
        count: usize,
    },
    /// `set_row_bounds`.
    RowBounds {
        /// Rows written.
        indices: Vec<usize>,
        /// Lower bounds, parallel to `indices`.
        lower: Vec<f64>,
        /// Upper bounds, parallel to `indices`.
        upper: Vec<f64>,
    },
    /// `set_row_lower_bounds_range`: rows `[first_row, first_row + lower.len())` get `[lower[i], +inf)`.
    RowLowerRange {
        /// First row of the range.
        first_row: usize,
        /// Lower bounds of the range.
        lower: Vec<f64>,
    },
    /// `set_col_bounds`.
    ColBounds {
        /// Columns written.
        indices: Vec<usize>,
        /// Lower bounds, parallel to `indices`.
        lower: Vec<f64>,
        /// Upper bounds, parallel to `indices`.
        upper: Vec<f64>,
    },
}

/// The LP one `solve` call saw, and the writes that led to it.
#[derive(Debug, Clone)]
pub struct RecordedSolve {
    /// The loaded template with every bound write applied.
    pub template: StageTemplate,
    /// Row count of the loaded template; appended row `r` is LP row `base_rows + r`.
    pub base_rows: usize,
    /// Rows appended since the load, with their current bounds; rows at or above `base_rows` live here, not in `template`.
    pub appended: RowBatch,
    /// Calls since the previous solve or load, in call order.
    pub writes: Vec<BoundWrite>,
}

/// [`SolverInterface`] wrapper that forwards every call to `inner` and records, per `solve`, the LP it saw.
pub struct BoundRecordingSolver<S: SolverInterface> {
    inner: S,
    current: Option<StageTemplate>,
    base_rows: usize,
    appended: RowBatch,
    writes: Vec<BoundWrite>,
    recorded: Vec<RecordedSolve>,
}

impl<S: SolverInterface> BoundRecordingSolver<S> {
    /// Wraps `inner` with nothing loaded and nothing recorded.
    #[must_use]
    pub fn new(inner: S) -> Self {
        Self {
            inner,
            current: None,
            base_rows: 0,
            appended: RowBatch {
                num_rows: 0,
                row_starts: vec![0],
                col_indices: Vec::new(),
                values: Vec::new(),
                row_lower: Vec::new(),
                row_upper: Vec::new(),
            },
            writes: Vec::new(),
            recorded: Vec::new(),
        }
    }

    /// One entry per `solve` call so far, in call order.
    #[must_use]
    pub fn recorded(&self) -> &[RecordedSolve] {
        &self.recorded
    }

    /// Consumes the wrapper, returning the recorded solves.
    #[must_use]
    pub fn into_recorded(self) -> Vec<RecordedSolve> {
        self.recorded
    }
}

impl<S: SolverInterface> SolverInterface for BoundRecordingSolver<S> {
    type Profile = S::Profile;

    fn apply_profile(&mut self, profile: &Self::Profile) {
        self.inner.apply_profile(profile);
    }

    fn load_model(&mut self, template: &StageTemplate) {
        self.current = Some(template.clone());
        self.base_rows = template.num_rows;
        self.appended.clear();
        self.appended.row_starts.push(0);
        self.writes.clear();
        self.inner.load_model(template);
    }

    fn add_rows(&mut self, rows: &RowBatch) {
        let appended = &mut self.appended;
        let offset = appended.row_starts[appended.num_rows];
        self.writes.push(BoundWrite::AddRows {
            first_row: self.base_rows + appended.num_rows,
            count: rows.num_rows,
        });
        appended.row_starts.extend(
            rows.row_starts
                .iter()
                .skip(1)
                .take(rows.num_rows)
                .map(|&end| end + offset),
        );
        appended.col_indices.extend_from_slice(&rows.col_indices);
        appended.values.extend_from_slice(&rows.values);
        appended.row_lower.extend_from_slice(&rows.row_lower);
        appended.row_upper.extend_from_slice(&rows.row_upper);
        appended.num_rows += rows.num_rows;
        self.inner.add_rows(rows);
    }

    #[expect(
        clippy::expect_used,
        reason = "every caller invokes load_model before set_row_bounds, so current is always Some"
    )]
    fn set_row_bounds(&mut self, indices: &[usize], lower: &[f64], upper: &[f64]) {
        let t = self
            .current
            .as_mut()
            .expect("BoundRecordingSolver: load_model precedes set_row_bounds");
        for (k, &i) in indices.iter().enumerate() {
            if i < self.base_rows {
                t.row_lower[i] = lower[k];
                t.row_upper[i] = upper[k];
            } else {
                self.appended.row_lower[i - self.base_rows] = lower[k];
                self.appended.row_upper[i - self.base_rows] = upper[k];
            }
        }
        self.writes.push(BoundWrite::RowBounds {
            indices: indices.to_vec(),
            lower: lower.to_vec(),
            upper: upper.to_vec(),
        });
        self.inner.set_row_bounds(indices, lower, upper);
    }

    #[expect(
        clippy::expect_used,
        reason = "every caller invokes load_model before set_row_lower_bounds_range, so current is always Some"
    )]
    fn set_row_lower_bounds_range(&mut self, first_row: usize, lower: &[f64]) {
        let t = self
            .current
            .as_mut()
            .expect("BoundRecordingSolver: load_model precedes set_row_lower_bounds_range");
        for (k, &v) in lower.iter().enumerate() {
            let row = first_row + k;
            if row < self.base_rows {
                t.row_lower[row] = v;
                t.row_upper[row] = f64::INFINITY;
            } else {
                self.appended.row_lower[row - self.base_rows] = v;
                self.appended.row_upper[row - self.base_rows] = f64::INFINITY;
            }
        }
        self.writes.push(BoundWrite::RowLowerRange {
            first_row,
            lower: lower.to_vec(),
        });
        self.inner.set_row_lower_bounds_range(first_row, lower);
    }

    #[expect(
        clippy::expect_used,
        reason = "every caller invokes load_model before set_col_bounds, so current is always Some"
    )]
    fn set_col_bounds(&mut self, indices: &[usize], lower: &[f64], upper: &[f64]) {
        let t = self
            .current
            .as_mut()
            .expect("BoundRecordingSolver: load_model precedes set_col_bounds");
        for (k, &i) in indices.iter().enumerate() {
            t.col_lower[i] = lower[k];
            t.col_upper[i] = upper[k];
        }
        self.writes.push(BoundWrite::ColBounds {
            indices: indices.to_vec(),
            lower: lower.to_vec(),
            upper: upper.to_vec(),
        });
        self.inner.set_col_bounds(indices, lower, upper);
    }

    #[expect(
        clippy::expect_used,
        reason = "every caller invokes load_model before solve, so current is always Some"
    )]
    fn solve(&mut self, basis: Option<&Basis>) -> Result<SolutionView<'_>, SolverError> {
        let template = self
            .current
            .clone()
            .expect("BoundRecordingSolver: load_model precedes solve");
        self.recorded.push(RecordedSolve {
            template,
            base_rows: self.base_rows,
            appended: self.appended.clone(),
            writes: std::mem::take(&mut self.writes),
        });
        self.inner.solve(basis)
    }

    fn get_basis(&mut self, out: &mut Basis) {
        self.inner.get_basis(out);
    }

    fn statistics(&self) -> SolverStatistics {
        self.inner.statistics()
    }

    fn statistics_into(&self, out: &mut SolverStatistics) {
        self.inner.statistics_into(out);
    }

    fn name(&self) -> &'static str {
        self.inner.name()
    }

    fn solver_name_version(&self) -> String {
        self.inner.solver_name_version()
    }

    fn record_reconstruction_stats(&mut self) {
        self.inner.record_reconstruction_stats();
    }

    fn reset_solver_state(&mut self) {
        self.inner.reset_solver_state();
    }
}

#[cfg(test)]
mod tests {
    use novomodelo_solver::{ActiveSolver, RowBatch, SolverInterface, StageTemplate};

    use super::{BoundRecordingSolver, BoundWrite};

    fn template() -> StageTemplate {
        StageTemplate {
            num_cols: 2,
            num_rows: 1,
            num_nz: 2,
            col_starts: vec![0, 1, 2],
            row_indices: vec![0, 0],
            values: vec![1.0, 1.0],
            col_lower: vec![0.0, 0.0],
            col_upper: vec![10.0, 10.0],
            objective: vec![1.0, 1.0],
            row_lower: vec![0.0],
            row_upper: vec![f64::INFINITY],
            n_state: 0,
            col_scale: Vec::new(),
            row_scale: Vec::new(),
        }
    }

    fn row_on(col: i32) -> RowBatch {
        RowBatch {
            num_rows: 1,
            row_starts: vec![0, 1],
            col_indices: vec![col],
            values: vec![10.0],
            row_lower: vec![0.0],
            row_upper: vec![f64::INFINITY],
        }
    }

    fn recorder() -> BoundRecordingSolver<ActiveSolver> {
        BoundRecordingSolver::new(ActiveSolver::new().expect("ActiveSolver::new must succeed"))
    }

    #[test]
    fn appended_rows_take_their_bound_writes() {
        let mut rec = recorder();
        rec.load_model(&template());
        rec.add_rows(&row_on(0));
        rec.add_rows(&row_on(1));
        rec.set_row_lower_bounds_range(1, &[3.0, 4.0]);
        rec.set_row_bounds(&[0, 1], &[1.0, 5.0], &[2.0, 6.0]);
        rec.solve(None).expect("the recorded LP is feasible");

        let solve = rec.recorded().last().expect("one solve recorded");
        assert_eq!(solve.base_rows, 1);
        assert_eq!(solve.template.row_lower, [1.0]);
        assert_eq!(solve.template.row_upper, [2.0]);
        assert_eq!(solve.appended.num_rows, 2);
        assert_eq!(solve.appended.row_lower, [5.0, 4.0]);
        assert_eq!(solve.appended.row_upper, [6.0, f64::INFINITY]);
        assert_eq!(solve.appended.row_starts, [0, 1, 2]);
        assert_eq!(solve.appended.col_indices, [0, 1]);
    }

    #[test]
    fn each_solve_logs_the_writes_since_the_previous_one() {
        let mut rec = recorder();
        rec.load_model(&template());
        rec.add_rows(&row_on(0));
        rec.set_row_bounds(&[0], &[1.0], &[2.0]);
        rec.solve(None).expect("first LP is feasible");
        rec.set_col_bounds(&[0], &[0.0], &[5.0]);
        rec.set_row_lower_bounds_range(1, &[0.2]);
        rec.solve(None).expect("second LP is feasible");

        let [first, second] = rec.recorded() else {
            panic!("two solves recorded");
        };
        assert_eq!(
            first.writes,
            [
                BoundWrite::AddRows {
                    first_row: 1,
                    count: 1
                },
                BoundWrite::RowBounds {
                    indices: vec![0],
                    lower: vec![1.0],
                    upper: vec![2.0]
                },
            ]
        );
        assert_eq!(
            second.writes,
            [
                BoundWrite::ColBounds {
                    indices: vec![0],
                    lower: vec![0.0],
                    upper: vec![5.0]
                },
                BoundWrite::RowLowerRange {
                    first_row: 1,
                    lower: vec![0.2]
                },
            ]
        );
        assert_eq!(second.appended.num_rows, 1);
    }

    #[test]
    fn load_model_clears_the_appended_rows() {
        let mut rec = recorder();
        rec.load_model(&template());
        rec.add_rows(&row_on(0));
        rec.solve(None).expect("first LP is feasible");
        rec.load_model(&template());
        rec.solve(None).expect("second LP is feasible");

        let [first, second] = rec.recorded() else {
            panic!("two solves recorded");
        };
        assert_eq!(first.appended.num_rows, 1);
        assert_eq!(second.appended.num_rows, 0);
        assert_eq!(second.appended.row_starts, [0]);
        assert!(second.writes.is_empty());
    }
}
