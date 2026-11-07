use crate::indexer::HydroSys;

/// The only route by which the builder references the realized inflow.
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

    /// Per-hydro `(row, coefficient)` lists, indexed by hydro position. An entry pushed
    /// here is checked by [`Self::finish`] exactly like one passed to [`Self::record`].
    pub(crate) fn per_hydro_mut(&mut self) -> &mut [Vec<(usize, f64)>] {
        &mut self.per_hydro
    }

    pub(crate) fn finish(mut self, n_base_rows: usize) -> InflowCoupling {
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
        InflowCoupling {
            hydro_starts,
            rows,
            coefficients,
        }
    }
}

pub(crate) struct InflowCoupling {
    hydro_starts: Vec<usize>,
    rows: Vec<u32>,
    coefficients: Vec<f64>,
}

impl InflowCoupling {
    pub(crate) fn n_hydros(&self) -> usize {
        self.hydro_starts.len() - 1
    }

    pub(crate) fn hydro_entries(&self, h: HydroSys) -> impl Iterator<Item = (usize, f64)> + '_ {
        let span = self.hydro_starts[h.get()]..self.hydro_starts[h.get() + 1];
        self.rows[span.clone()]
            .iter()
            .zip(&self.coefficients[span])
            .map(|(&row, &coefficient)| (row as usize, coefficient))
    }
}

#[cfg(test)]
mod tests;
