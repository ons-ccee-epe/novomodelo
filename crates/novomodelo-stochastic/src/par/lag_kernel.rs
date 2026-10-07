//! Shared lag-chain advancement kernel.
//!
//! Three call sites accumulate a stage's realized value into a lag period,
//! finalize and shift the lag chain, and — for multi-resolution studies —
//! roll completed periods into a coarser downstream ring.
//! [`advance_lag_chain`] is the single kernel they all route through, generic
//! over the lag-storage layout via [`LagIndex`].
//!
//! The shift is never in-place: `advance_lag_chain` reads the lags it moves
//! forward from a separate `incoming_lags` slice, distinct from the `lag_state`
//! it writes into. A caller whose realized value has already landed in its lag
//! storage before the call would lose the old lags to an in-place shift, so it
//! passes a pre-saved snapshot as `incoming_lags`.

use cobre_core::temporal::StageLagTransition;

/// Maps a `(lag, entity)` pair to a flat-buffer index for one lag-storage
/// order, plus the two dimensions the advancer loops over. Static-dispatched
/// (`impl LagIndex`), never `Box<dyn>`.
pub trait LagIndex: Copy {
    /// Flat-buffer index for `(lag, entity)` under this layout's storage order.
    fn index(self, lag: usize, entity: usize) -> usize;

    /// Number of entities this layout indexes.
    fn entity_count(self) -> usize;

    /// Maximum lag order this layout indexes.
    fn max_order(self) -> usize;
}

/// Lag-major order: `index = lag * entity_count + entity`
/// (a contiguous `[f64]` window laid out lag-major).
#[derive(Clone, Copy)]
pub struct LagMajor {
    /// Number of entities this layout indexes.
    pub entity_count: usize,
    /// Maximum lag order this layout indexes.
    pub max_order: usize,
}

impl LagIndex for LagMajor {
    #[inline]
    fn index(self, lag: usize, entity: usize) -> usize {
        lag * self.entity_count + entity
    }

    fn entity_count(self) -> usize {
        self.entity_count
    }

    fn max_order(self) -> usize {
        self.max_order
    }
}

/// Entity-major order: `index = entity * max_order + lag`.
#[derive(Clone, Copy)]
pub struct EntityMajor {
    /// Number of entities this layout indexes.
    pub entity_count: usize,
    /// Maximum lag order this layout indexes.
    pub max_order: usize,
}

impl LagIndex for EntityMajor {
    #[inline]
    fn index(self, lag: usize, entity: usize) -> usize {
        entity * self.max_order + lag
    }

    fn entity_count(self) -> usize {
        self.entity_count
    }

    fn max_order(self) -> usize {
        self.max_order
    }
}

/// Primary lag-accumulation buffers for [`advance_lag_chain`].
pub struct PrimaryLagAccum<'a> {
    /// Weighted-sum accumulator for the current lag period; length `>= entity_count`.
    pub accumulator: &'a mut [f64],
    /// Total weight accumulated so far in the current lag period, per entity;
    /// finalize divides each entity by its own entry, length `>= entity_count`.
    pub weight_accum: &'a mut [f64],
}

/// Downstream (coarser-resolution) lag-accumulation buffers for
/// [`advance_lag_chain`].
///
/// Uniform-resolution studies pass `accumulator: &mut []` (empty slice) and
/// `par_order: 0`; every downstream branch short-circuits on
/// `accumulator.is_empty()`.
pub struct DownstreamLagAccum<'a> {
    /// Weighted-sum accumulator for the current downstream period; empty for
    /// uniform-resolution studies.
    pub accumulator: &'a mut [f64],
    /// Total weight accumulated so far in the current downstream period. Stays
    /// scalar: no seed path reaches the downstream ring, so its per-stage
    /// weight is uniform across entities by construction.
    pub weight_accum: &'a mut f64,
    /// Slot-major ring buffer (`completed_lags[slot * entity_count +
    /// entity]`), length `entity_count * par_order` or empty.
    pub completed_lags: &'a mut [f64],
    /// Number of completed downstream periods currently held in the ring
    /// (capped at `par_order`).
    pub n_completed: &'a mut usize,
    /// Lag order for the downstream resolution; `0` for uniform-resolution
    /// studies.
    pub par_order: usize,
}

/// Accumulate this stage's realized value into the current lag period and,
/// when the period finalizes, shift the lag chain — reproducing, for the
/// monthly identity transition (`accumulate_weight=1.0, spillover_weight=0.0,
/// finalize_period=true`), a direct highest-lag-first shift seeded with the
/// realized value.
///
/// Also advances the downstream (coarser-resolution) ring in the same call:
/// when `stage_lag.rebuild_from_downstream` fires and the ring holds at
/// least one completed period, the newest-first ring contents overwrite
/// `lag_state` and the primary finalize below is skipped for this call.
///
/// # Panics (debug only)
///
/// Panics in debug builds if `primary.accumulator.len() < layout.entity_count()`,
/// if `primary.weight_accum.len() < layout.entity_count()`, or if
/// `downstream.par_order != 0` and `downstream.completed_lags.len() <
/// layout.entity_count() * downstream.par_order`.
pub fn advance_lag_chain<L: LagIndex>(
    layout: L,
    lag_state: &mut [f64],
    incoming_lags: &[f64],
    realized: &[f64],
    stage_lag: &StageLagTransition,
    primary: &mut PrimaryLagAccum<'_>,
    downstream: &mut DownstreamLagAccum<'_>,
) {
    let entity_count = layout.entity_count();
    let max_order = layout.max_order();
    if max_order == 0 || entity_count == 0 {
        return;
    }

    debug_assert!(
        primary.accumulator.len() >= entity_count,
        "primary accumulator too short: {} < {entity_count}",
        primary.accumulator.len()
    );
    debug_assert!(
        primary.weight_accum.len() >= entity_count,
        "primary weight_accum too short: {} < {entity_count}",
        primary.weight_accum.len()
    );

    let w = stage_lag.accumulate_weight;
    for (entity, &r) in realized.iter().enumerate().take(entity_count) {
        primary.accumulator[entity] += r * w;
        primary.weight_accum[entity] += w;
    }

    if !downstream.accumulator.is_empty() && stage_lag.accumulate_downstream {
        debug_assert!(
            downstream.par_order == 0
                || downstream.completed_lags.len() >= entity_count * downstream.par_order,
            "downstream completed_lags too short: {} < {}",
            downstream.completed_lags.len(),
            entity_count * downstream.par_order
        );

        let dw = stage_lag.downstream_accumulate_weight;
        for (acc, r) in downstream.accumulator[..entity_count]
            .iter_mut()
            .zip(realized)
        {
            *acc += r * dw;
        }
        *downstream.weight_accum += dw;

        if stage_lag.downstream_finalize && *downstream.weight_accum > 0.0 {
            let inv = 1.0 / *downstream.weight_accum;
            for v in &mut downstream.accumulator[..entity_count] {
                *v *= inv;
            }

            let slot = (*downstream.n_completed).min(downstream.par_order.saturating_sub(1));
            let offset = slot * entity_count;
            downstream.completed_lags[offset..offset + entity_count]
                .copy_from_slice(&downstream.accumulator[..entity_count]);
            *downstream.n_completed = (*downstream.n_completed + 1).min(downstream.par_order);

            if stage_lag.downstream_spillover_weight > 0.0 {
                let dsw = stage_lag.downstream_spillover_weight;
                for (acc, r) in downstream.accumulator[..entity_count]
                    .iter_mut()
                    .zip(realized)
                {
                    *acc = r * dsw;
                }
                *downstream.weight_accum = dsw;
            } else {
                downstream.accumulator[..entity_count].fill(0.0);
                *downstream.weight_accum = 0.0;
            }
        }
    }

    // Rebuild takes priority: return here so the primary finalize below does
    // not overwrite the ring-rebuilt lags for this call.
    if stage_lag.rebuild_from_downstream && *downstream.n_completed > 0 {
        let n_fill = (*downstream.n_completed).min(max_order);
        for lag_idx in 0..n_fill {
            // lag_idx 0 = newest = slot n_completed-1, descending into older slots.
            let src_slot = *downstream.n_completed - 1 - lag_idx;
            let src_offset = src_slot * entity_count;
            for entity in 0..entity_count {
                lag_state[layout.index(lag_idx, entity)] =
                    downstream.completed_lags[src_offset + entity];
            }
        }
        *downstream.n_completed = 0;
        downstream.completed_lags.fill(0.0);
        if !downstream.accumulator.is_empty() {
            downstream.accumulator[..entity_count].fill(0.0);
        }
        *downstream.weight_accum = 0.0;
        return;
    }

    if stage_lag.finalize_period {
        for entity in 0..entity_count {
            if primary.weight_accum[entity] > 0.0 {
                let inv = 1.0 / primary.weight_accum[entity];
                primary.accumulator[entity] *= inv;
            }
        }

        for entity in 0..entity_count {
            for lag in (1..max_order).rev() {
                lag_state[layout.index(lag, entity)] = incoming_lags[layout.index(lag - 1, entity)];
            }
            lag_state[layout.index(0, entity)] = primary.accumulator[entity];
        }

        // Spillover seeds the NEXT period with RAW realized value, not the averaged one.
        if stage_lag.spillover_weight > 0.0 {
            let sw = stage_lag.spillover_weight;
            for (entity, &r) in realized.iter().enumerate().take(entity_count) {
                primary.accumulator[entity] = r * sw;
                primary.weight_accum[entity] = sw;
            }
        } else {
            primary.accumulator[..entity_count].fill(0.0);
            primary.weight_accum[..entity_count].fill(0.0);
        }
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::{DownstreamLagAccum, EntityMajor, LagMajor, PrimaryLagAccum, advance_lag_chain};
    use cobre_core::temporal::StageLagTransition;

    /// Builds a `StageLagTransition` exercising only the primary path — the
    /// `downstream_*` fields stay inert (`0.0` / `false`).
    fn primary_transition(
        accumulate_weight: f64,
        spillover_weight: f64,
        finalize_period: bool,
    ) -> StageLagTransition {
        StageLagTransition {
            accumulate_weight,
            spillover_weight,
            finalize_period,
            accumulate_downstream: false,
            downstream_accumulate_weight: 0.0,
            downstream_spillover_weight: 0.0,
            downstream_finalize: false,
            rebuild_from_downstream: false,
        }
    }

    /// Independent reimplementation of the reference monthly-identity shift
    /// (highest-lag-first, newest lag set to the realized value) over a
    /// lag-major window, so a regression in [`advance_lag_chain`] diverges
    /// from this oracle rather than agreeing with itself.
    fn oracle_shift_lag_major(
        lag_state: &mut [f64],
        incoming_lags: &[f64],
        realized: &[f64],
        entity_count: usize,
        max_order: usize,
    ) {
        for entity in 0..entity_count {
            for lag in (1..max_order).rev() {
                lag_state[lag * entity_count + entity] =
                    incoming_lags[(lag - 1) * entity_count + entity];
            }
            lag_state[entity] = realized[entity];
        }
    }

    /// Independent reimplementation of the reference monthly-identity shift
    /// over an entity-major buffer.
    fn oracle_shift_entity_major(
        lag_state: &mut [f64],
        incoming_lags: &[f64],
        realized: &[f64],
        entity_count: usize,
        max_order: usize,
    ) {
        for (entity, &r) in realized.iter().enumerate().take(entity_count) {
            let base = entity * max_order;
            for lag in (1..max_order).rev() {
                lag_state[base + lag] = incoming_lags[base + lag - 1];
            }
            lag_state[base] = r;
        }
    }

    /// Independent reimplementation of the downstream ring (accumulate,
    /// weighted-average finalize into a slot-major ring, newest-first
    /// rebuild), used to discriminate a regression in the kernel's ring path
    /// from a bug in the test fixture itself.
    struct OracleRing {
        accumulator: Vec<f64>,
        weight_accum: f64,
        completed_lags: Vec<f64>,
        n_completed: usize,
        par_order: usize,
        entity_count: usize,
    }

    impl OracleRing {
        fn new(entity_count: usize, par_order: usize) -> Self {
            Self {
                accumulator: vec![0.0; entity_count],
                weight_accum: 0.0,
                completed_lags: vec![0.0; entity_count * par_order],
                n_completed: 0,
                par_order,
                entity_count,
            }
        }

        fn accumulate(&mut self, realized: &[f64], weight: f64) {
            for (acc, r) in self.accumulator.iter_mut().zip(realized) {
                *acc += r * weight;
            }
            self.weight_accum += weight;
        }

        fn finalize(&mut self) {
            if self.weight_accum <= 0.0 {
                return;
            }
            let inv = 1.0 / self.weight_accum;
            for v in &mut self.accumulator {
                *v *= inv;
            }
            let slot = self.n_completed.min(self.par_order.saturating_sub(1));
            let offset = slot * self.entity_count;
            self.completed_lags[offset..offset + self.entity_count]
                .copy_from_slice(&self.accumulator);
            self.n_completed = (self.n_completed + 1).min(self.par_order);
            self.accumulator.fill(0.0);
            self.weight_accum = 0.0;
        }

        fn rebuild(&mut self, max_order: usize) -> Vec<f64> {
            let n_fill = self.n_completed.min(max_order);
            let mut out = vec![0.0; max_order * self.entity_count];
            for lag_idx in 0..n_fill {
                let src_slot = self.n_completed - 1 - lag_idx;
                let src_offset = src_slot * self.entity_count;
                let dst_offset = lag_idx * self.entity_count;
                out[dst_offset..dst_offset + self.entity_count].copy_from_slice(
                    &self.completed_lags[src_offset..src_offset + self.entity_count],
                );
            }
            self.n_completed = 0;
            self.completed_lags.fill(0.0);
            self.weight_accum = 0.0;
            out
        }
    }

    #[test]
    fn advance_lag_chain_noop_when_max_order_zero() {
        let layout = LagMajor {
            entity_count: 3,
            max_order: 0,
        };
        let mut lag_state = vec![7.0; 3];
        let incoming_lags = vec![0.0; 3];
        let realized = vec![1.0, 2.0, 3.0];
        let stage_lag = primary_transition(1.0, 0.0, true);

        let mut accumulator = vec![0.0; 3];
        let mut weight_accum = vec![0.0; 3];
        let mut primary = PrimaryLagAccum {
            accumulator: &mut accumulator,
            weight_accum: &mut weight_accum,
        };
        let mut downstream_accumulator: Vec<f64> = Vec::new();
        let mut downstream_weight_accum = 0.0;
        let mut completed_lags: Vec<f64> = Vec::new();
        let mut n_completed = 0;
        let mut downstream = DownstreamLagAccum {
            accumulator: &mut downstream_accumulator,
            weight_accum: &mut downstream_weight_accum,
            completed_lags: &mut completed_lags,
            n_completed: &mut n_completed,
            par_order: 0,
        };

        advance_lag_chain(
            layout,
            &mut lag_state,
            &incoming_lags,
            &realized,
            &stage_lag,
            &mut primary,
            &mut downstream,
        );

        assert_eq!(lag_state, vec![7.0; 3]);
        assert_eq!(accumulator, vec![0.0; 3]);
        assert_eq!(weight_accum, vec![0.0; 3]);
    }

    #[test]
    fn advance_lag_chain_monthly_identity_lag_major_matches_shift_oracle() {
        let entity_count = 3;
        let max_order = 3;
        let layout = LagMajor {
            entity_count,
            max_order,
        };

        let incoming_lags = vec![
            10.0, 11.0, 12.0, // lag 0
            20.0, 21.0, 22.0, // lag 1
            30.0, 31.0, 32.0, // lag 2
        ];
        let realized = vec![1.0, 2.0, 3.0];

        let mut lag_state = vec![0.0; entity_count * max_order];
        let mut oracle_state = lag_state.clone();
        oracle_shift_lag_major(
            &mut oracle_state,
            &incoming_lags,
            &realized,
            entity_count,
            max_order,
        );

        let stage_lag = primary_transition(1.0, 0.0, true);
        let mut accumulator = vec![0.0; entity_count];
        let mut weight_accum = vec![0.0; entity_count];
        let mut primary = PrimaryLagAccum {
            accumulator: &mut accumulator,
            weight_accum: &mut weight_accum,
        };
        let mut downstream_accumulator: Vec<f64> = Vec::new();
        let mut downstream_weight_accum = 0.0;
        let mut completed_lags: Vec<f64> = Vec::new();
        let mut n_completed = 0;
        let mut downstream = DownstreamLagAccum {
            accumulator: &mut downstream_accumulator,
            weight_accum: &mut downstream_weight_accum,
            completed_lags: &mut completed_lags,
            n_completed: &mut n_completed,
            par_order: 0,
        };

        advance_lag_chain(
            layout,
            &mut lag_state,
            &incoming_lags,
            &realized,
            &stage_lag,
            &mut primary,
            &mut downstream,
        );

        assert_eq!(lag_state, oracle_state);
    }

    #[test]
    fn advance_lag_chain_monthly_identity_entity_major_matches_shift_oracle() {
        let entity_count = 3;
        let max_order = 3;
        let layout = EntityMajor {
            entity_count,
            max_order,
        };

        let incoming_lags = vec![
            10.0, 20.0, 30.0, // entity 0: lag 0, 1, 2
            11.0, 21.0, 31.0, // entity 1: lag 0, 1, 2
            12.0, 22.0, 32.0, // entity 2: lag 0, 1, 2
        ];
        let realized = vec![1.0, 2.0, 3.0];

        let mut lag_state = vec![0.0; entity_count * max_order];
        let mut oracle_state = lag_state.clone();
        oracle_shift_entity_major(
            &mut oracle_state,
            &incoming_lags,
            &realized,
            entity_count,
            max_order,
        );

        let stage_lag = primary_transition(1.0, 0.0, true);
        let mut accumulator = vec![0.0; entity_count];
        let mut weight_accum = vec![0.0; entity_count];
        let mut primary = PrimaryLagAccum {
            accumulator: &mut accumulator,
            weight_accum: &mut weight_accum,
        };
        let mut downstream_accumulator: Vec<f64> = Vec::new();
        let mut downstream_weight_accum = 0.0;
        let mut completed_lags: Vec<f64> = Vec::new();
        let mut n_completed = 0;
        let mut downstream = DownstreamLagAccum {
            accumulator: &mut downstream_accumulator,
            weight_accum: &mut downstream_weight_accum,
            completed_lags: &mut completed_lags,
            n_completed: &mut n_completed,
            par_order: 0,
        };

        advance_lag_chain(
            layout,
            &mut lag_state,
            &incoming_lags,
            &realized,
            &stage_lag,
            &mut primary,
            &mut downstream,
        );

        assert_eq!(lag_state, oracle_state);
    }

    #[test]
    fn advance_lag_chain_weighted_average_across_finalizing_stages() {
        let entity_count = 2;
        let max_order = 1;
        let layout = LagMajor {
            entity_count,
            max_order,
        };

        let incoming_lags = vec![0.0; entity_count];
        let weights = [1.0, 2.0, 3.0];
        let realized_per_stage = [[10.0, 100.0], [20.0, 200.0], [30.0, 300.0]];

        let mut lag_state = vec![0.0; entity_count];
        let mut accumulator = vec![0.0; entity_count];
        let mut weight_accum = vec![0.0; entity_count];
        let mut downstream_accumulator: Vec<f64> = Vec::new();
        let mut downstream_weight_accum = 0.0;
        let mut completed_lags: Vec<f64> = Vec::new();
        let mut n_completed = 0;

        for (stage, &w) in weights.iter().enumerate() {
            let finalize = stage == weights.len() - 1;
            let stage_lag = primary_transition(w, 0.0, finalize);
            let realized = realized_per_stage[stage];
            let mut primary = PrimaryLagAccum {
                accumulator: &mut accumulator,
                weight_accum: &mut weight_accum,
            };
            let mut downstream = DownstreamLagAccum {
                accumulator: &mut downstream_accumulator,
                weight_accum: &mut downstream_weight_accum,
                completed_lags: &mut completed_lags,
                n_completed: &mut n_completed,
                par_order: 0,
            };
            advance_lag_chain(
                layout,
                &mut lag_state,
                &incoming_lags,
                &realized,
                &stage_lag,
                &mut primary,
                &mut downstream,
            );
        }

        let total_weight: f64 = weights.iter().sum();
        for entity in 0..entity_count {
            let sum: f64 = weights
                .iter()
                .zip(realized_per_stage.iter())
                .map(|(&w, r)| w * r[entity])
                .sum();
            let expected = sum * (1.0 / total_weight);
            assert!(
                (lag_state[entity] - expected).abs() < 1e-10,
                "entity {entity}: {} vs {expected}",
                lag_state[entity]
            );
        }
    }

    /// A seeded accumulator (mirroring `derive_inflow_seeds`'s `accum = rate *
    /// coverage`, `crates/cobre-stochastic/src/seeds.rs`) for a 3-day
    /// conditioning window at 500 m3/s out of a 30-day month, carried through
    /// the finalize where the remaining 27 days complete at 480 m3/s — the
    /// seed-unit-bug killer: an accumulator seeded in raw hours (or days) but
    /// divided by the wrong denominator would miss this exact average.
    #[test]
    fn test_finalized_lag1_mixes_conditioning_and_completing_month_exactly() {
        let entity_count = 1;
        let max_order = 1;
        let layout = LagMajor {
            entity_count,
            max_order,
        };

        let conditioning_rate = 500.0_f64;
        let conditioning_hours = 3.0 * 24.0;
        let completing_rate = 480.0_f64;
        let completing_hours = 27.0 * 24.0;
        let month_hours = 30.0 * 24.0;

        let conditioning_weight = conditioning_hours / month_hours;
        let completing_weight = completing_hours / month_hours;

        let incoming_lags = vec![0.0; entity_count];
        let realized = [completing_rate];

        let mut lag_state = vec![0.0; entity_count];
        let mut accumulator = vec![conditioning_rate * conditioning_weight];
        let mut weight_accum = vec![conditioning_weight];
        let mut downstream_accumulator: Vec<f64> = Vec::new();
        let mut downstream_weight_accum = 0.0;
        let mut completed_lags: Vec<f64> = Vec::new();
        let mut n_completed = 0;

        let stage_lag = primary_transition(completing_weight, 0.0, true);
        let mut primary = PrimaryLagAccum {
            accumulator: &mut accumulator,
            weight_accum: &mut weight_accum,
        };
        let mut downstream = DownstreamLagAccum {
            accumulator: &mut downstream_accumulator,
            weight_accum: &mut downstream_weight_accum,
            completed_lags: &mut completed_lags,
            n_completed: &mut n_completed,
            par_order: 0,
        };
        advance_lag_chain(
            layout,
            &mut lag_state,
            &incoming_lags,
            &realized,
            &stage_lag,
            &mut primary,
            &mut downstream,
        );

        let expected = (conditioning_rate * conditioning_hours
            + completing_rate * completing_hours)
            / month_hours;
        assert_eq!(
            lag_state[0], expected,
            "finalized lag-1 must mix the conditioning seed and the completing \
             month exactly: got {}, expected {expected}",
            lag_state[0]
        );
    }

    #[test]
    fn advance_lag_chain_spillover_seeds_next_period_with_raw_value() {
        let entity_count = 2;
        let max_order = 1;
        let layout = LagMajor {
            entity_count,
            max_order,
        };

        let incoming_lags = vec![0.0; entity_count];
        let realized = [5.0, 7.0];
        let spillover_weight = 0.25;

        let mut lag_state = vec![0.0; entity_count];
        let mut accumulator = vec![0.0; entity_count];
        let mut weight_accum = vec![0.0; entity_count];
        let mut downstream_accumulator: Vec<f64> = Vec::new();
        let mut downstream_weight_accum = 0.0;
        let mut completed_lags: Vec<f64> = Vec::new();
        let mut n_completed = 0;
        let mut primary = PrimaryLagAccum {
            accumulator: &mut accumulator,
            weight_accum: &mut weight_accum,
        };
        let mut downstream = DownstreamLagAccum {
            accumulator: &mut downstream_accumulator,
            weight_accum: &mut downstream_weight_accum,
            completed_lags: &mut completed_lags,
            n_completed: &mut n_completed,
            par_order: 0,
        };

        let stage_lag = primary_transition(1.0, spillover_weight, true);
        advance_lag_chain(
            layout,
            &mut lag_state,
            &incoming_lags,
            &realized,
            &stage_lag,
            &mut primary,
            &mut downstream,
        );

        assert_eq!(lag_state, vec![realized[0], realized[1]]);
        assert_eq!(
            accumulator,
            vec![
                realized[0] * spillover_weight,
                realized[1] * spillover_weight
            ]
        );
        assert_eq!(weight_accum, vec![spillover_weight; entity_count]);
    }

    #[test]
    fn lag_kernel_downstream_ring_rebuild_matches_oracle() {
        let entity_count = 2;
        let par_order = 1;
        let max_order = 1;
        let layout = LagMajor {
            entity_count,
            max_order,
        };

        let incoming_lags = vec![0.0; entity_count * max_order];
        let monthly_realized = [[10.0, 20.0], [11.0, 21.0], [12.0, 22.0]];
        let downstream_weight_per_month = 1.0;

        let mut lag_state = vec![0.0; entity_count * max_order];
        let mut accumulator = vec![0.0; entity_count];
        let mut weight_accum = vec![0.0; entity_count];
        let mut downstream_accumulator = vec![0.0; entity_count];
        let mut downstream_weight_accum = 0.0;
        let mut completed_lags = vec![0.0; entity_count * par_order];
        let mut n_completed = 0;

        let mut oracle = OracleRing::new(entity_count, par_order);

        for (stage, realized) in monthly_realized.iter().enumerate() {
            let finalize = stage == monthly_realized.len() - 1;
            let stage_lag = StageLagTransition {
                accumulate_weight: 0.0,
                spillover_weight: 0.0,
                finalize_period: false,
                accumulate_downstream: true,
                downstream_accumulate_weight: downstream_weight_per_month,
                downstream_spillover_weight: 0.0,
                downstream_finalize: finalize,
                rebuild_from_downstream: false,
            };
            let mut primary = PrimaryLagAccum {
                accumulator: &mut accumulator,
                weight_accum: &mut weight_accum,
            };
            let mut downstream = DownstreamLagAccum {
                accumulator: &mut downstream_accumulator,
                weight_accum: &mut downstream_weight_accum,
                completed_lags: &mut completed_lags,
                n_completed: &mut n_completed,
                par_order,
            };
            advance_lag_chain(
                layout,
                &mut lag_state,
                &incoming_lags,
                realized,
                &stage_lag,
                &mut primary,
                &mut downstream,
            );

            oracle.accumulate(realized, downstream_weight_per_month);
            if finalize {
                oracle.finalize();
            }
        }

        let rebuild_realized = [0.0, 0.0];
        let stage_lag = StageLagTransition {
            accumulate_weight: 0.0,
            spillover_weight: 0.0,
            finalize_period: false,
            accumulate_downstream: false,
            downstream_accumulate_weight: 0.0,
            downstream_spillover_weight: 0.0,
            downstream_finalize: false,
            rebuild_from_downstream: true,
        };
        let mut primary = PrimaryLagAccum {
            accumulator: &mut accumulator,
            weight_accum: &mut weight_accum,
        };
        let mut downstream = DownstreamLagAccum {
            accumulator: &mut downstream_accumulator,
            weight_accum: &mut downstream_weight_accum,
            completed_lags: &mut completed_lags,
            n_completed: &mut n_completed,
            par_order,
        };
        advance_lag_chain(
            layout,
            &mut lag_state,
            &incoming_lags,
            &rebuild_realized,
            &stage_lag,
            &mut primary,
            &mut downstream,
        );

        let expected = oracle.rebuild(max_order);
        assert_eq!(lag_state, expected);
        assert_eq!(n_completed, 0);
        assert_eq!(completed_lags, vec![0.0; entity_count * par_order]);
    }

    #[test]
    fn advance_lag_chain_per_entity_weight_divides_independently() {
        let entity_count = 2;
        let max_order = 1;
        let layout = LagMajor {
            entity_count,
            max_order,
        };

        let incoming_lags = vec![0.0; entity_count];
        let realized = [0.0, 0.0];

        let mut lag_state = vec![0.0; entity_count];
        let mut accumulator = vec![10.0, 10.0];
        // Entity 0 has accumulated total weight 1.0; entity 1 only 0.5.
        let mut weight_accum = vec![1.0, 0.5];
        let mut downstream_accumulator: Vec<f64> = Vec::new();
        let mut downstream_weight_accum = 0.0;
        let mut completed_lags: Vec<f64> = Vec::new();
        let mut n_completed = 0;
        let mut primary = PrimaryLagAccum {
            accumulator: &mut accumulator,
            weight_accum: &mut weight_accum,
        };
        let mut downstream = DownstreamLagAccum {
            accumulator: &mut downstream_accumulator,
            weight_accum: &mut downstream_weight_accum,
            completed_lags: &mut completed_lags,
            n_completed: &mut n_completed,
            par_order: 0,
        };

        let stage_lag = primary_transition(0.0, 0.0, true);
        advance_lag_chain(
            layout,
            &mut lag_state,
            &incoming_lags,
            &realized,
            &stage_lag,
            &mut primary,
            &mut downstream,
        );

        // Hand-computed: entity 0 divides its 10.0 sum by its own weight 1.0
        // (-> 10.0); entity 1 divides the same 10.0 sum by its own weight 0.5
        // (-> 20.0). A shared scalar weight (e.g. the combined total 1.5)
        // would instead divide both entities by the same denominator,
        // producing neither entity's per-entity result.
        assert_eq!(lag_state, vec![10.0, 20.0]);
        let shared_scalar_result = 10.0 / (1.0 + 0.5);
        assert_ne!(lag_state[0], shared_scalar_result);
        assert_ne!(lag_state[1], shared_scalar_result);
    }

    /// Independent reimplementation of the pre-widening scalar-weight
    /// accumulate-then-finalize arithmetic (one shared weight divides every
    /// entity), used to pin the per-entity kernel's output bit-for-bit under
    /// uniform per-stage coverage.
    fn oracle_scalar_weight_finalize(
        accumulator: &mut [f64],
        weight: &mut f64,
        realized_per_stage: &[[f64; 2]],
        weights: &[f64],
    ) {
        for (&w, r) in weights.iter().zip(realized_per_stage) {
            for (acc, &rv) in accumulator.iter_mut().zip(r) {
                *acc += rv * w;
            }
            *weight += w;
        }
        if *weight > 0.0 {
            let inv = 1.0 / *weight;
            for acc in accumulator.iter_mut() {
                *acc *= inv;
            }
        }
    }

    #[test]
    fn advance_lag_chain_uniform_weight_matches_scalar_result() {
        let entity_count = 2;
        let max_order = 1;
        let layout = LagMajor {
            entity_count,
            max_order,
        };

        let incoming_lags = vec![0.0; entity_count];
        let weights = [1.0, 2.0, 3.0];
        let realized_per_stage = [[10.0, 100.0], [20.0, 200.0], [30.0, 300.0]];

        let mut oracle_accumulator = vec![0.0; entity_count];
        let mut oracle_weight = 0.0_f64;
        oracle_scalar_weight_finalize(
            &mut oracle_accumulator,
            &mut oracle_weight,
            &realized_per_stage,
            &weights,
        );

        let mut lag_state = vec![0.0; entity_count];
        let mut accumulator = vec![0.0; entity_count];
        let mut weight_accum = vec![0.0; entity_count];
        let mut downstream_accumulator: Vec<f64> = Vec::new();
        let mut downstream_weight_accum = 0.0;
        let mut completed_lags: Vec<f64> = Vec::new();
        let mut n_completed = 0;

        for (stage, &w) in weights.iter().enumerate() {
            let finalize = stage == weights.len() - 1;
            let stage_lag = primary_transition(w, 0.0, finalize);
            let realized = realized_per_stage[stage];
            let mut primary = PrimaryLagAccum {
                accumulator: &mut accumulator,
                weight_accum: &mut weight_accum,
            };
            let mut downstream = DownstreamLagAccum {
                accumulator: &mut downstream_accumulator,
                weight_accum: &mut downstream_weight_accum,
                completed_lags: &mut completed_lags,
                n_completed: &mut n_completed,
                par_order: 0,
            };
            advance_lag_chain(
                layout,
                &mut lag_state,
                &incoming_lags,
                &realized,
                &stage_lag,
                &mut primary,
                &mut downstream,
            );
        }

        assert_eq!(
            lag_state, oracle_accumulator,
            "uniform per-entity weight broadcast must match the shared-scalar-weight result bit-for-bit"
        );
    }

    /// Per-hydro uneven coverage from a hand-built `DerivedInflowSeeds`:
    /// entity 0 (hydro A) carries a pre-study seed (`accum`/`weight` from a
    /// partial in-progress window); entity 1 (hydro B) has no windows at all
    /// (`accum = weight = 0.0`). Both finalize over the same two in-study
    /// stages. Hydro B's zero seed weight must not make its finalize divide
    /// by zero or fall back to a zero-inflow imputation — the
    /// `weight_accum[entity] > 0.0` finalize gate must still fire once the
    /// in-study stages contribute weight, averaging hydro B's own realized
    /// values.
    #[test]
    fn test_no_window_hydro_finalizes_to_in_study_average() {
        let entity_count = 2;
        let max_order = 1;
        let layout = LagMajor {
            entity_count,
            max_order,
        };

        let derived = crate::seeds::DerivedInflowSeeds {
            lag_values: vec![0.0; entity_count],
            accum: vec![100.0, 0.0],
            weight: vec![0.4, 0.0],
        };

        let incoming_lags = vec![0.0; entity_count];
        let mut lag_state = vec![0.0; entity_count];
        let mut accumulator = derived.accum.clone();
        let mut weight_accum = derived.weight.clone();
        let mut downstream_accumulator: Vec<f64> = Vec::new();
        let mut downstream_weight_accum = 0.0;
        let mut completed_lags: Vec<f64> = Vec::new();
        let mut n_completed = 0;

        // Two in-study stages of the same finalizing period: hydro A reaches
        // full coverage (seed 0.4 + 0.3 + 0.3 = 1.0); hydro B, with no seed,
        // reaches only the in-study weight (0.3 + 0.3 = 0.6).
        let weights = [0.3, 0.3];
        let realized_per_stage = [[300.0, 150.0], [340.0, 170.0]];
        for (stage, &w) in weights.iter().enumerate() {
            let finalize = stage == weights.len() - 1;
            let stage_lag = primary_transition(w, 0.0, finalize);
            let realized = realized_per_stage[stage];
            let mut primary = PrimaryLagAccum {
                accumulator: &mut accumulator,
                weight_accum: &mut weight_accum,
            };
            let mut downstream = DownstreamLagAccum {
                accumulator: &mut downstream_accumulator,
                weight_accum: &mut downstream_weight_accum,
                completed_lags: &mut completed_lags,
                n_completed: &mut n_completed,
                par_order: 0,
            };
            advance_lag_chain(
                layout,
                &mut lag_state,
                &incoming_lags,
                &realized,
                &stage_lag,
                &mut primary,
                &mut downstream,
            );
        }

        assert_eq!(
            lag_state[0], 292.0,
            "hydro A finalizes to its own coverage-weighted value, mixing the \
             pre-study seed with the in-study accumulation"
        );
        assert_eq!(
            lag_state[1],
            f64::midpoint(150.0, 170.0),
            "the no-window hydro must finalize to the in-study average of its \
             own realized values, not a zero-inflow imputation"
        );
    }
}
