//! The σ·η and cut-row W tables of one cut-generating node on the compact LP, filled once per
//! node and read by every opening of every trial point. Row ω is outcome ω of the node's
//! successor outcome set, in its canonical order.

use novomodelo_stochastic::evaluate_par_noise_term_batch;
use rayon::{
    iter::{IndexedParallelIterator, ParallelIterator},
    slice::ParallelSliceMut,
};

use super::{OpeningNoise, SuccessorOutcomes, resolve_opening_noise};
use crate::{
    SddpError, context::TrainingContext, cut::rhs::cut_noise_term, setup::node_graph::StageIdx,
};

#[derive(Debug, Default)]
pub(crate) struct StageNoiseTable {
    noise_terms: Vec<f64>,
    cut_noise: Vec<f64>,
    n_hydros: usize,
    stride: usize,
    external_noise: Vec<f64>,
}

fn reserve_to(buffer: &mut Vec<f64>, capacity: usize) {
    buffer.reserve(capacity.saturating_sub(buffer.len()));
}

impl StageNoiseTable {
    /// Grows every buffer to hold `max_outcomes` rows of `n_hydros` terms and `max_rows` cut
    /// terms, and one external noise column of `noise_dim`; never shrinks.
    pub(crate) fn reserve(
        &mut self,
        max_outcomes: usize,
        n_hydros: usize,
        max_rows: usize,
        noise_dim: usize,
    ) {
        reserve_to(&mut self.noise_terms, max_outcomes * n_hydros);
        reserve_to(&mut self.cut_noise, max_outcomes * max_rows);
        reserve_to(&mut self.external_noise, noise_dim);
    }

    /// Writes σ·η of every outcome, then the W of every resident cut row of each outcome's child.
    ///
    /// # Errors
    ///
    /// [`SddpError::Validation`] from [`resolve_opening_noise`] for an external child.
    pub(crate) fn fill(
        &mut self,
        training_ctx: &TrainingContext<'_>,
        successor: StageIdx,
        outcomes: &SuccessorOutcomes<'_>,
    ) -> Result<(), SddpError> {
        let dims = training_ctx.stochastic.class_dimensions();
        let par = training_ctx.stochastic.par();
        let tree_view = training_ctx.stochastic.tree_view();
        let n_hydros = dims.n_hydros;
        let total = outcomes.total_outcomes();
        let Self {
            noise_terms,
            cut_noise,
            n_hydros: row_width,
            stride,
            external_noise,
        } = self;
        *row_width = n_hydros;

        noise_terms.clear();
        noise_terms.resize(total * n_hydros, 0.0);
        let mut write_noise_term = |omega: usize, raw_noise: &[f64]| {
            if par.n_stages() > 0 {
                evaluate_par_noise_term_batch(
                    par,
                    successor.0,
                    &raw_noise[dims.hydro_range()],
                    &mut noise_terms[omega * n_hydros..][..n_hydros],
                );
            }
        };
        for ci in 0..outcomes.n_children() {
            let child = outcomes.child(ci);
            let first = child.outcome_range.start;
            match resolve_opening_noise(
                training_ctx,
                successor,
                &child.openings,
                child.successor_node_id,
                external_noise,
            )? {
                OpeningNoise::External(column) => write_noise_term(first, column),
                OpeningNoise::Generated => {
                    for local_omega in 0..child.openings.len {
                        write_noise_term(
                            first + local_omega,
                            tree_view.opening(successor.0, local_omega),
                        );
                    }
                }
            }
        }

        *stride = (0..outcomes.n_children())
            .map(|ci| outcomes.child(ci).successor_active_slots.len())
            .max()
            .unwrap_or(0);
        let stride = *stride;
        cut_noise.clear();
        cut_noise.resize(total * stride, 0.0);
        if stride == 0 {
            return Ok(());
        }

        let noise_terms = noise_terms.as_slice();
        let layouts = training_ctx.cut_state_layouts;
        cut_noise
            .par_chunks_mut(stride)
            .enumerate()
            .for_each(|(omega, row)| {
                let child = outcomes.child_of_outcome(omega);
                let lag_point = layouts[child.pool_id].lag_point();
                let noise_term = &noise_terms[omega * n_hydros..][..n_hydros];
                for (w, &slot) in row.iter_mut().zip(child.successor_active_slots) {
                    *w = cut_noise_term(
                        child.successor_pool.coefficient_row(slot),
                        lag_point,
                        noise_term,
                    );
                }
            });
        Ok(())
    }

    /// σ·η of outcome `omega`, one term per hydro.
    pub(crate) fn noise_term(&self, omega: usize) -> &[f64] {
        &self.noise_terms[omega * self.n_hydros..][..self.n_hydros]
    }

    /// W of the first `rows` cut rows of outcome `omega`'s child, in its LP cut-row order.
    pub(crate) fn cut_noise(&self, omega: usize, rows: usize) -> &[f64] {
        debug_assert!(rows <= self.stride, "more cut rows than the table's stride");
        &self.cut_noise[omega * self.stride..][..rows]
    }

    #[cfg(test)]
    pub(crate) fn buffer_identity(&self) -> [(usize, usize); 3] {
        [
            (
                self.noise_terms.as_ptr().addr(),
                self.noise_terms.capacity(),
            ),
            (self.cut_noise.as_ptr().addr(), self.cut_noise.capacity()),
            (
                self.external_noise.as_ptr().addr(),
                self.external_noise.capacity(),
            ),
        ]
    }
}
