//! Recording wrapper over a real solver, for tests that assert on the LP a solve saw.

use std::sync::{Arc, Mutex, MutexGuard};

use novomodelo_solver::{
    Basis, RowBatch, SolutionView, SolverError, SolverInterface, SolverStatistics, StageTemplate,
};

use crate::test_support::template_structure::column_entries;

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

/// What one `solve` saw of the cut rows. Rows are LP row indices; a cut row is fresh once its RHS
/// was written since the last change of pinned state or noise.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CutRowPushRecord {
    /// First cut row of the loaded template.
    pub first_cut_row: usize,
    /// Cut rows of the loaded template.
    pub frozen_rows: usize,
    /// Rows appended since the load.
    pub appended_rows: usize,
    /// Rows appended since the previous solve or load.
    pub added_in_window: usize,
    /// A structural row or a column bound was written since the previous solve or load.
    pub structural_window: bool,
    /// Cut rows, frozen and appended, that are not fresh.
    pub stale_rows: usize,
    /// Lowest stale cut row.
    pub first_stale_row: Option<usize>,
}

/// How a recorder counts a cut row's own bound when the row is appended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AppendedCutRowBound {
    /// The appended bound is not a write: the row needs a later write.
    Patched,
    /// The appended bound is final: it counts as a write unless a structural write follows in the same window.
    Final,
}

/// The records of every recorder built on it, in solve order; clones share one list.
#[derive(Clone, Debug, Default)]
pub struct CutRowPushLog(Arc<Mutex<Vec<CutRowPushRecord>>>);

impl CutRowPushLog {
    /// An empty log.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// A copy of the records so far.
    #[must_use]
    pub fn records(&self) -> Vec<CutRowPushRecord> {
        self.lock().clone()
    }

    fn push(&self, record: CutRowPushRecord) {
        self.lock().push(record);
    }

    #[expect(
        clippy::expect_used,
        reason = "a poisoned log means a recorder panicked while holding it"
    )]
    fn lock(&self) -> MutexGuard<'_, Vec<CutRowPushRecord>> {
        self.0.lock().expect("CutRowPushLog: lock poisoned")
    }
}

/// [`SolverInterface`] wrapper that forwards every call to `inner` and records, per `solve`, the LP it saw.
pub struct BoundRecordingSolver<S: SolverInterface> {
    inner: S,
    current: Option<StageTemplate>,
    base_rows: usize,
    appended: RowBatch,
    writes: Vec<BoundWrite>,
    recorded: Vec<RecordedSolve>,
    log: Option<CutRowPushLog>,
    theta_col: usize,
    appended_bound: AppendedCutRowBound,
    first_cut_row: usize,
    fresh: Vec<bool>,
    written: Vec<bool>,
    credit_from: usize,
    structural_window: bool,
    added_in_window: usize,
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
            log: None,
            theta_col: 0,
            appended_bound: AppendedCutRowBound::Patched,
            first_cut_row: 0,
            fresh: Vec::new(),
            written: Vec::new(),
            credit_from: 0,
            structural_window: false,
            added_in_window: 0,
        }
    }

    /// Wraps `inner` so that each `solve` pushes one [`CutRowPushRecord`] to `log` and keeps no
    /// [`RecordedSolve`].
    ///
    /// A cut row is a row with an entry on `theta_col`; the loaded template's cut rows are its
    /// last rows and every appended row is one. A cut row's RHS must be written after the last
    /// change of pinned state or noise, which is a write to a structural row or to a column bound.
    /// After such a write every cut row must be written again before the next solve; between two
    /// solves without one, the rows already fresh stay fresh and only the rows appended since need
    /// a write. Under [`AppendedCutRowBound::Patched`] an appended row's own bound is not a write;
    /// under [`AppendedCutRowBound::Final`] a row appended after the window's last structural write
    /// counts as written in that window.
    ///
    /// `Patched` is the production append-then-patch rule; `Final` models an append whose bound is
    /// its RHS and needs no later write. `load_model` and `add_rows` panic on a template or an
    /// appended row that breaks the cut-row shape.
    #[must_use]
    pub fn logging_cut_row_pushes(
        inner: S,
        theta_col: usize,
        log: CutRowPushLog,
        appended_bound: AppendedCutRowBound,
    ) -> Self {
        Self {
            log: Some(log),
            theta_col,
            appended_bound,
            ..Self::new(inner)
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

    fn open_window(&mut self, template: &StageTemplate) {
        if self.log.is_none() {
            return;
        }
        let mut theta_rows: Vec<usize> = column_entries(template, self.theta_col)
            .map(|(row, _)| row)
            .collect();
        theta_rows.sort_unstable();
        self.first_cut_row = theta_rows.first().copied().unwrap_or(template.num_rows);
        for (k, row) in (self.first_cut_row..template.num_rows).enumerate() {
            assert_eq!(
                theta_rows.get(k),
                Some(&row),
                "BoundRecordingSolver: template row {row} follows a cut row and has no theta entry"
            );
        }
        let cut_rows = template.num_rows - self.first_cut_row;
        self.fresh.clear();
        self.fresh.resize(cut_rows, false);
        self.written.clear();
        self.written.resize(cut_rows, false);
        self.credit_from = cut_rows;
        self.structural_window = false;
        self.added_in_window = 0;
    }

    fn note_structural_write(&mut self) {
        self.structural_window = true;
        self.credit_from = self.fresh.len();
    }

    fn note_added_rows(&mut self, rows: &RowBatch) {
        if self.log.is_none() {
            return;
        }
        let first_row = self.base_rows + self.appended.num_rows;
        for k in 0..rows.num_rows {
            assert!(
                row_has_column(rows, k, self.theta_col),
                "BoundRecordingSolver: appended row {} has no theta entry",
                first_row + k
            );
        }
        self.fresh.resize(self.fresh.len() + rows.num_rows, false);
        self.written
            .resize(self.written.len() + rows.num_rows, false);
        self.added_in_window += rows.num_rows;
    }

    fn note_row_write(&mut self, row: usize) {
        if self.log.is_none() {
            return;
        }
        match row.checked_sub(self.first_cut_row) {
            Some(cut) => self.written[cut] = true,
            None => self.note_structural_write(),
        }
    }

    fn close_window(&mut self) -> CutRowPushRecord {
        let structural_window = self.structural_window;
        let credited_from = match self.appended_bound {
            AppendedCutRowBound::Patched => self.fresh.len(),
            AppendedCutRowBound::Final => self.credit_from,
        };
        for (cut, (fresh, &written)) in self.fresh.iter_mut().zip(&self.written).enumerate() {
            *fresh = written || cut >= credited_from || (*fresh && !structural_window);
        }
        let first_stale = self.fresh.iter().position(|&fresh| !fresh);
        let record = CutRowPushRecord {
            first_cut_row: self.first_cut_row,
            frozen_rows: self.base_rows - self.first_cut_row,
            appended_rows: self.appended.num_rows,
            added_in_window: self.added_in_window,
            structural_window,
            stale_rows: self.fresh.iter().filter(|&&fresh| !fresh).count(),
            first_stale_row: first_stale.map(|cut| self.first_cut_row + cut),
        };
        self.written.fill(false);
        self.credit_from = self.fresh.len();
        self.structural_window = false;
        self.added_in_window = 0;
        record
    }
}

#[expect(
    clippy::cast_sign_loss,
    reason = "CSR row_starts and col_indices are non-negative by construction"
)]
fn row_has_column(rows: &RowBatch, row: usize, col: usize) -> bool {
    let span = rows.row_starts[row] as usize..rows.row_starts[row + 1] as usize;
    rows.col_indices[span].iter().any(|&c| c as usize == col)
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
        self.open_window(template);
        self.inner.load_model(template);
    }

    fn add_rows(&mut self, rows: &RowBatch) {
        self.note_added_rows(rows);
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
        for &i in indices {
            self.note_row_write(i);
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
        for row in first_row..first_row + lower.len() {
            self.note_row_write(row);
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
        if self.log.is_some() {
            self.note_structural_write();
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
        if self.log.is_none() {
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
        } else {
            self.writes.clear();
        }
        let record = self.log.is_some().then(|| self.close_window());
        if let Some((log, record)) = self.log.as_ref().zip(record) {
            log.push(record);
        }
        self.inner.solve(basis)
    }

    fn warm_starts(&self) -> bool {
        self.inner.warm_starts()
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
    use novomodelo_solver::{
        ActiveProfile, ActiveSolver, RowBatch, SolverInterface, StageTemplate,
    };

    use super::{
        AppendedCutRowBound, BoundRecordingSolver, BoundWrite, CutRowPushLog, CutRowPushRecord,
    };

    const THETA_COL: usize = 1;

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
        BoundRecordingSolver::new(solver())
    }

    fn solver() -> ActiveSolver {
        ActiveSolver::new().expect("ActiveSolver::new must succeed")
    }

    fn theta_template() -> StageTemplate {
        StageTemplate {
            num_cols: 2,
            num_rows: 3,
            num_nz: 3,
            col_starts: vec![0, 1, 3],
            row_indices: vec![0, 1, 2],
            values: vec![1.0, 1.0, 1.0],
            col_lower: vec![0.0, 0.0],
            col_upper: vec![10.0, 10.0],
            objective: vec![1.0, 1.0],
            row_lower: vec![1.0, 2.0, 3.0],
            row_upper: vec![f64::INFINITY; 3],
            n_state: 0,
            col_scale: Vec::new(),
            row_scale: Vec::new(),
        }
    }

    fn theta_row() -> RowBatch {
        row_on(1)
    }

    fn logging_recorder(log: &CutRowPushLog) -> BoundRecordingSolver<ActiveSolver> {
        recorder_with(log, AppendedCutRowBound::Patched)
    }

    fn recorder_with(
        log: &CutRowPushLog,
        appended_bound: AppendedCutRowBound,
    ) -> BoundRecordingSolver<ActiveSolver> {
        BoundRecordingSolver::logging_cut_row_pushes(
            solver(),
            THETA_COL,
            log.clone(),
            appended_bound,
        )
    }

    fn write_structural_row(rec: &mut BoundRecordingSolver<ActiveSolver>) {
        rec.set_row_bounds(&[0], &[1.0], &[f64::INFINITY]);
    }

    #[cfg(all(feature = "highs", not(feature = "clp")))]
    fn cold_profiles() -> Vec<ActiveProfile> {
        use novomodelo_solver::HighsProfile;

        vec![HighsProfile {
            use_warm_start: false,
            ..HighsProfile::default()
        }]
    }

    #[cfg(feature = "clp")]
    fn cold_profiles() -> Vec<ActiveProfile> {
        Vec::new()
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

    #[test]
    fn a_structural_window_requires_every_cut_row_written() {
        let log = CutRowPushLog::new();
        let mut rec = logging_recorder(&log);
        rec.load_model(&theta_template());

        write_structural_row(&mut rec);
        rec.set_row_lower_bounds_range(1, &[2.0]);
        rec.solve(None).expect("first LP is feasible");

        write_structural_row(&mut rec);
        rec.set_row_lower_bounds_range(1, &[2.0, 3.0]);
        rec.solve(None).expect("second LP is feasible");

        write_structural_row(&mut rec);
        rec.set_row_lower_bounds_range(2, &[3.0]);
        rec.solve(None).expect("third LP is feasible");

        let records = log.records();
        assert!(records.iter().all(|record| record.structural_window));
        let stale: Vec<_> = records
            .iter()
            .map(|record| (record.stale_rows, record.first_stale_row))
            .collect();
        assert_eq!(stale, [(1, Some(2)), (0, None), (1, Some(1))]);
    }

    #[test]
    fn a_round_without_structural_writes_keeps_earlier_rows_fresh() {
        let log = CutRowPushLog::new();
        let mut rec = logging_recorder(&log);
        rec.load_model(&theta_template());
        write_structural_row(&mut rec);
        rec.set_row_lower_bounds_range(1, &[2.0, 3.0]);
        rec.solve(None).expect("first LP is feasible");

        rec.add_rows(&theta_row());
        rec.set_row_lower_bounds_range(3, &[4.0]);
        rec.solve(None).expect("second LP is feasible");

        rec.add_rows(&theta_row());
        rec.solve(None).expect("third LP is feasible");

        let [_, second, third] = log.records().try_into().expect("three solves logged");
        assert_eq!(
            second,
            CutRowPushRecord {
                first_cut_row: 1,
                frozen_rows: 2,
                appended_rows: 1,
                added_in_window: 1,
                structural_window: false,
                stale_rows: 0,
                first_stale_row: None,
            }
        );
        assert_eq!(
            third,
            CutRowPushRecord {
                first_cut_row: 1,
                frozen_rows: 2,
                appended_rows: 2,
                added_in_window: 1,
                structural_window: false,
                stale_rows: 1,
                first_stale_row: Some(4),
            }
        );
    }

    #[test]
    fn a_final_append_after_the_pin_is_fresh_without_a_push() {
        let log = CutRowPushLog::new();
        let mut rec = recorder_with(&log, AppendedCutRowBound::Final);
        rec.load_model(&theta_template());
        rec.set_col_bounds(&[0], &[0.0], &[10.0]);
        rec.set_row_lower_bounds_range(1, &[2.0, 3.0]);
        rec.add_rows(&theta_row());
        rec.solve(None).expect("the recorded LP is feasible");

        let [record] = log.records().try_into().expect("one solve logged");
        assert!(record.structural_window);
        assert_eq!(record.stale_rows, 0);
        assert_eq!(record.added_in_window, 1);
    }

    #[test]
    fn a_final_append_before_a_later_pin_still_needs_a_push() {
        let log = CutRowPushLog::new();
        let mut rec = recorder_with(&log, AppendedCutRowBound::Final);
        rec.load_model(&theta_template());
        rec.set_row_lower_bounds_range(1, &[2.0, 3.0]);
        rec.add_rows(&theta_row());
        rec.set_col_bounds(&[0], &[0.0], &[10.0]);
        rec.solve(None).expect("first LP is feasible");

        rec.set_col_bounds(&[0], &[0.0], &[10.0]);
        rec.set_row_lower_bounds_range(1, &[2.0, 3.0, 4.0]);
        rec.solve(None).expect("second LP is feasible");

        let [stale, written] = log.records().try_into().expect("two solves logged");
        assert_eq!((stale.stale_rows, stale.first_stale_row), (1, Some(3)));
        assert_eq!(written.stale_rows, 0);
    }

    #[test]
    fn a_patched_append_needs_a_push_even_after_the_pin() {
        let log = CutRowPushLog::new();
        let mut rec = recorder_with(&log, AppendedCutRowBound::Patched);
        rec.load_model(&theta_template());
        write_structural_row(&mut rec);
        rec.set_row_lower_bounds_range(1, &[2.0, 3.0]);
        rec.add_rows(&theta_row());
        rec.solve(None).expect("the recorded LP is feasible");

        let [record] = log.records().try_into().expect("one solve logged");
        assert_eq!((record.stale_rows, record.first_stale_row), (1, Some(3)));
    }

    #[test]
    fn load_model_resets_the_push_flags() {
        let log = CutRowPushLog::new();
        let mut rec = logging_recorder(&log);
        rec.load_model(&theta_template());
        write_structural_row(&mut rec);
        rec.set_row_lower_bounds_range(1, &[2.0, 3.0]);
        rec.solve(None).expect("first LP is feasible");

        rec.load_model(&theta_template());
        rec.solve(None).expect("second LP is feasible");

        let [first, second] = log.records().try_into().expect("two solves logged");
        assert_eq!(first.stale_rows, 0);
        assert_eq!(
            second,
            CutRowPushRecord {
                first_cut_row: 1,
                frozen_rows: 2,
                appended_rows: 0,
                added_in_window: 0,
                structural_window: false,
                stale_rows: 2,
                first_stale_row: Some(1),
            }
        );
    }

    #[test]
    fn the_log_mode_keeps_no_solves() {
        let log = CutRowPushLog::new();
        let mut rec = logging_recorder(&log);
        rec.load_model(&theta_template());
        rec.solve(None).expect("the LP is feasible");

        assert!(rec.recorded().is_empty());
        assert_eq!(log.records().len(), 1);
    }

    #[test]
    fn warm_starts_follows_the_inner_solver() {
        let mut inner = solver();
        let mut rec = BoundRecordingSolver::new(solver());
        assert_eq!(rec.warm_starts(), inner.warm_starts());

        for cold in cold_profiles() {
            inner.apply_profile(&cold);
            rec.apply_profile(&cold);
            assert!(!rec.warm_starts(), "a cold profile reports cold");
            assert_eq!(rec.warm_starts(), inner.warm_starts());
        }
    }
}
