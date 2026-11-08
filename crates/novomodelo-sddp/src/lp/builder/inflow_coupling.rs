use crate::indexer::HydroSys;

/// The only route by which the builder references the realized inflow: every entry
/// enters through [`Self::record`].
pub(crate) struct InflowCouplingSink {
    per_hydro: Vec<Vec<(usize, f64)>>,
}

impl InflowCouplingSink {
    pub(crate) fn new(n_hydros: usize) -> Self {
        Self {
            per_hydro: vec![Vec::new(); n_hydros],
        }
    }

    pub(crate) fn record(&mut self, hydro: HydroSys, row: usize, coefficient: f64) {
        debug_assert!(
            hydro.get() < self.per_hydro.len(),
            "inflow-coupling sink: hydro index out of range"
        );
        self.per_hydro[hydro.get()].push((row, coefficient));
    }

    pub(crate) fn finish(mut self, row_lower: &[f64], row_upper: &[f64]) -> InflowCoupling {
        debug_assert_eq!(
            row_lower.len(),
            row_upper.len(),
            "inflow-coupling sink: row bound slices differ in length"
        );
        let n_base_rows = row_lower.len();
        let n_entries = self.per_hydro.iter().map(Vec::len).sum();
        let mut hydro_starts = Vec::with_capacity(self.per_hydro.len() + 1);
        let mut rows = Vec::with_capacity(n_entries);
        let mut coefficients = Vec::with_capacity(n_entries);
        hydro_starts.push(0);
        for entries in &mut self.per_hydro {
            entries.sort_by_key(|&(row, _)| row);
            for &(row, coefficient) in entries.iter() {
                debug_assert!(
                    row < n_base_rows,
                    "inflow-coupling sink: entry is not a base row of this stage"
                );
                debug_assert!(
                    coefficient.is_finite(),
                    "inflow-coupling sink: non-finite coefficient"
                );
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "base rows are below the template row count, which `load_model` requires to fit in i32"
                )]
                rows.push(row as u32);
                coefficients.push(coefficient);
            }
            hydro_starts.push(rows.len());
        }
        let mut touched_rows = rows.clone();
        touched_rows.sort_unstable();
        touched_rows.dedup();
        #[expect(
            clippy::cast_possible_truncation,
            reason = "a position is below the touched-row count, at most the base-row count"
        )]
        let positions = rows
            .iter()
            .map(|&row| touched_rows.partition_point(|&touched| touched < row) as u32)
            .collect();
        let touched_row_lower = touched_rows
            .iter()
            .map(|&row| row_lower[row as usize])
            .collect();
        let touched_row_upper = touched_rows
            .iter()
            .map(|&row| row_upper[row as usize])
            .collect();
        InflowCoupling {
            hydro_starts,
            positions,
            coefficients,
            touched_rows,
            touched_row_lower,
            touched_row_upper,
        }
    }
}

/// Hydro-major `(touched-row position, unscaled coefficient)` entries, positions
/// ascending within a hydro and duplicates kept, where a position indexes
/// [`Self::touched_rows`], plus the unscaled template bounds of each of those rows.
#[derive(Debug, Clone)]
pub struct InflowCoupling {
    hydro_starts: Vec<usize>,
    positions: Vec<u32>,
    coefficients: Vec<f64>,
    touched_rows: Vec<u32>,
    touched_row_lower: Vec<f64>,
    touched_row_upper: Vec<f64>,
}

impl InflowCoupling {
    /// Number of hydros the coupling is indexed over.
    #[must_use]
    pub fn n_hydros(&self) -> usize {
        self.hydro_starts.len() - 1
    }

    /// Hydro `h`'s `(base row, coefficient)` entries, rows ascending.
    pub fn hydro_entries(&self, h: HydroSys) -> impl Iterator<Item = (usize, f64)> + '_ {
        self.hydro_shift_entries(h)
            .map(|(position, coefficient)| (self.touched_rows[position] as usize, coefficient))
    }

    /// Hydro `h`'s `(touched-row position, coefficient)` entries, positions ascending;
    /// a position indexes [`Self::touched_rows`].
    pub(crate) fn hydro_shift_entries(
        &self,
        h: HydroSys,
    ) -> impl Iterator<Item = (usize, f64)> + '_ {
        let span = self.hydro_starts[h.get()]..self.hydro_starts[h.get() + 1];
        self.positions[span.clone()]
            .iter()
            .zip(&self.coefficients[span])
            .map(|(&position, &coefficient)| (position as usize, coefficient))
    }

    /// Distinct base rows the entries touch, ascending.
    #[must_use]
    pub fn touched_rows(&self) -> &[u32] {
        &self.touched_rows
    }

    /// Unscaled lower bound of each row in [`Self::touched_rows`].
    #[must_use]
    pub fn touched_row_lower(&self) -> &[f64] {
        &self.touched_row_lower
    }

    /// Unscaled upper bound of each row in [`Self::touched_rows`].
    #[must_use]
    pub fn touched_row_upper(&self) -> &[f64] {
        &self.touched_row_upper
    }
}

#[cfg(test)]
mod tests;
