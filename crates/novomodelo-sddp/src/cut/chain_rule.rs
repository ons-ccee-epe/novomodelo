//! Per-opening chain rule: the cut lag coefficients and intercept rebuilt from the compact LP's
//! duals through ẑ = p(â) + σ·η. In unscaled cost units, with ψ the PAR coefficient per lag,
//! `G` the stored coupling, `π` the base-row duals and `y` the resident cut duals:
//!
//! ```text
//! g(h)      = -(Gᵀπ)(h) + Σ_k y(k)·β(k, (h, 0))
//! w(h, l)   = Σ_k y(k)·β(k, (h, l+1))
//! β′(h, l)  = ψ(h, l)·g(h) + w(h, l)       w absent at the last lag
//! α′        = Q − Σ_j β′(j)·x̂(j)
//! ```

use novomodelo_stochastic::par::precompute::PrecomputedPar;

use crate::cut::CutPool;
use crate::indexer::{ChainRuleSlots, CutStateProjection};

/// Per-(pool pair, stage) inputs of the kernel; none owns a buffer.
pub struct ChainRuleInputs<'a> {
    /// Slot ranges of the (resident, extracted) pool pair.
    pub slots: &'a ChainRuleSlots,
    /// The pool whose cut duals the opening reports.
    pub resident_pool: &'a CutPool,
    /// Projection of the pool the new cut joins.
    pub extracted: &'a CutStateProjection,
    /// Source of ψ.
    pub par: &'a PrecomputedPar,
    /// The solved child stage's index into `par`.
    pub stage: usize,
}

/// One opening's unscaled values.
pub struct OpeningDuals<'a> {
    /// Q, in cost units.
    pub objective: f64,
    /// The state duals of the extracted slots outside `extracted_lag`, in slot order, with no
    /// entry for a lag slot; [`ChainRuleSlots::non_lag`] maps each position to its slot.
    pub state_duals: &'a [f64],
    /// (Gᵀπ)_h, length N.
    pub coupled_duals: &'a [f64],
    /// The resident cut duals `(slot, y)` by strictly ascending slot.
    pub cut_duals: &'a [(usize, f64)],
}

/// Reusable accumulators of the kernel; both buffers are overwritten on every call.
#[derive(Debug, Clone, Default)]
pub struct ChainRuleScratch {
    g: Vec<f64>,
    w: Vec<f64>,
}

impl ChainRuleScratch {
    /// Scratch built from the layout's N (`n_hydros`) and L (`lag_depth`): g holds N values and w
    /// holds N·(L−1). Every pool pair of one layout shares that shape.
    #[must_use]
    pub fn new(n_hydros: usize, lag_depth: usize) -> Self {
        Self {
            g: vec![0.0; n_hydros],
            w: vec![0.0; n_hydros * lag_depth.saturating_sub(1)],
        }
    }
}

/// On entry `scratch.g` holds the starting value of each hydro's g.
fn lag_coefficients_into(
    inputs: &ChainRuleInputs<'_>,
    cut_duals: &[(usize, f64)],
    scratch: &mut ChainRuleScratch,
    lag_out: &mut [f64],
) {
    let slots = inputs.slots;
    let (n_hydros, lag_depth) = (slots.n_hydros(), slots.lag_depth());
    debug_assert!(
        cut_duals.windows(2).all(|pair| pair[0].0 < pair[1].0),
        "cut duals slots must strictly ascend"
    );
    debug_assert_eq!(lag_out.len(), n_hydros * lag_depth);

    let ChainRuleScratch { g, w } = scratch;
    debug_assert!(
        g.len() == n_hydros && w.len() == n_hydros * lag_depth.saturating_sub(1),
        "scratch was built for another pair"
    );
    w.fill(-0.0);
    let (lag0, shifted) = (slots.resident_lag0(), slots.resident_shifted());
    if !lag0.is_empty() {
        for &(slot, y) in cut_duals {
            if y == 0.0 {
                continue;
            }
            let row = inputs.resident_pool.coefficient_row(slot);
            for (g_h, &beta) in g.iter_mut().zip(&row[lag0.clone()]) {
                *g_h += y * beta;
            }
            for (w_i, &beta) in w.iter_mut().zip(&row[shifted.clone()]) {
                *w_i += y * beta;
            }
        }
    }

    for (h, &g_h) in g.iter().enumerate() {
        let psi = if inputs.par.max_order() > 0 {
            inputs.par.psi_slice(inputs.stage, h)
        } else {
            &[]
        };
        debug_assert!(
            psi.len() <= lag_depth,
            "chain rule: the PAR order exceeds the lag depth"
        );
        for l in 0..lag_depth {
            // A lag past the PAR order carries ψ = 0: the lag state is widened past it.
            let psi_l = psi.get(l).copied().unwrap_or(0.0);
            let at = l * n_hydros + h;
            lag_out[at] = if l + 1 < lag_depth {
                psi_l * g_h + w[at]
            } else {
                psi_l * g_h
            };
        }
    }
}

/// Writes the new cut for one opening into `coefficients` and returns its intercept α.
///
/// The non-lag slots copy `state_duals` and the `extracted_lag` slots hold ψ·g + w, as in the
/// module formula. α is Q minus the extracted projection's trial-state dot of the result.
///
/// # Panics
///
/// In debug builds, panics if `coefficients.len() != extracted.n_slots()`, if `state_duals` is
/// not lag-free (its length is not `extracted.n_slots()` minus the lag width), if
/// `coupled_duals.len() != N`, if the slots of `cut_duals` do not strictly ascend, if a PAR order
/// is deeper than the lag depth, or if `scratch` was built for another pair.
pub fn per_opening_cut_into(
    inputs: &ChainRuleInputs<'_>,
    opening: &OpeningDuals<'_>,
    x_hat: &[f64],
    scratch: &mut ChainRuleScratch,
    coefficients: &mut [f64],
) -> f64 {
    debug_assert_eq!(coefficients.len(), inputs.extracted.n_slots());
    debug_assert_eq!(opening.state_duals.len(), inputs.slots.non_lag()[1].1.end);
    debug_assert_eq!(opening.coupled_duals.len(), inputs.slots.n_hydros());

    for (slots, positions) in inputs.slots.non_lag() {
        coefficients[slots].copy_from_slice(&opening.state_duals[positions]);
    }
    let lag = inputs.slots.extracted_lag();
    if !lag.is_empty() {
        for (g_h, &coupled) in scratch.g.iter_mut().zip(opening.coupled_duals) {
            *g_h = -coupled;
        }
        lag_coefficients_into(inputs, opening.cut_duals, scratch, &mut coefficients[lag]);
    }
    opening.objective - inputs.extracted.dot_trial_state(coefficients, x_hat)
}

/// One successor child's inputs and its openings in canonical order.
pub struct ChildOpenings<'a> {
    /// The child's own resident pool and slot ranges, with the shared extracted projection.
    pub inputs: ChainRuleInputs<'a>,
    /// The child's openings, by canonical ω.
    pub openings: &'a [OpeningDuals<'a>],
}

/// Reusable buffers of [`aggregate_first_cut_into`], sized by [`AggregateScratch::reserve`]; every ȳ
/// slot is 0.0, every bitmap word 0 and `pairs` empty between calls.
#[derive(Debug, Clone)]
pub struct AggregateScratch {
    y_bar: Vec<f64>,
    touched: Vec<u64>,
    pairs: Vec<(usize, f64)>,
    chain: ChainRuleScratch,
    child_lag: Vec<f64>,
}

impl AggregateScratch {
    /// Scratch built from the layout's N and L; the ȳ buffers start empty and
    /// [`AggregateScratch::reserve`] sizes them.
    #[must_use]
    pub fn new(n_hydros: usize, lag_depth: usize) -> Self {
        Self {
            y_bar: Vec::new(),
            touched: Vec::new(),
            pairs: Vec::new(),
            chain: ChainRuleScratch::new(n_hydros, lag_depth),
            child_lag: vec![0.0; n_hydros * lag_depth],
        }
    }

    /// Grows ȳ and the bitmap to cover `pool_capacity` pool slots and the pair buffer to `pair_hint`
    /// entries, never shrinking them. Call it wherever the resident pool grows:
    /// [`aggregate_first_cut_into`] does not grow them.
    pub fn reserve(&mut self, pool_capacity: usize, pair_hint: usize) {
        if self.y_bar.len() < pool_capacity {
            self.y_bar.resize(pool_capacity, 0.0);
            self.touched.resize(pool_capacity.div_ceil(64), 0);
        }
        self.pairs.reserve(pair_hint);
    }
}

/// Writes the weighted cut over a fan of children into `coefficients` and returns its intercept ᾱ.
///
/// `weights` is μ over the flattened children × openings, from one weight-kernel call over the
/// whole set: each child's partial sums use those joint weights and are never renormalised per
/// child. The duals are aggregated first (ȳ per child in slot space, then Q̄, the non-lag
/// coefficients and ḡ) and the chain rule of the module formula runs once per child over ȳ. Every
/// accumulator starts at −0.0, so one opening at weight 1.0 returns `per_opening_cut_into`'s bits.
///
/// # Panics
///
/// In debug builds, panics if `children` is empty, if `weights.len()` is not the total opening
/// count, if the children differ in `extracted_lag` or in `extracted.n_slots()`, if
/// `coefficients.len() != extracted.n_slots()`, if an opening's `state_duals` is not lag-free or
/// its `coupled_duals.len()` is not N, if an opening's objective is not finite, if a PAR order is
/// deeper than the lag depth, or if `scratch` was not reserved for the resident pool.
pub fn aggregate_first_cut_into(
    children: &[ChildOpenings<'_>],
    weights: &[f64],
    x_hat: &[f64],
    scratch: &mut AggregateScratch,
    coefficients: &mut [f64],
) -> f64 {
    debug_assert!(
        !children.is_empty(),
        "aggregate_first_cut_into: no children"
    );
    let (slots, extracted) = (children[0].inputs.slots, children[0].inputs.extracted);
    let lag = slots.extracted_lag();
    debug_assert_eq!(
        weights.len(),
        children.iter().map(|c| c.openings.len()).sum::<usize>()
    );
    debug_assert!(
        children
            .iter()
            .all(|c| c.inputs.slots.extracted_lag() == lag
                && c.inputs.extracted.n_slots() == extracted.n_slots()),
        "children must share the extracted projection"
    );
    debug_assert_eq!(coefficients.len(), extracted.n_slots());

    let AggregateScratch {
        y_bar,
        touched,
        pairs,
        chain,
        child_lag,
    } = scratch;
    let non_lag = slots.non_lag();
    coefficients.fill(-0.0);
    let mut q_bar = -0.0;
    let mut rest = weights;
    for child in children {
        let (child_weights, tail) = rest.split_at(child.openings.len());
        rest = tail;
        for (opening, &mu) in child.openings.iter().zip(child_weights) {
            debug_assert_eq!(opening.state_duals.len(), non_lag[1].1.end);
            debug_assert_eq!(opening.coupled_duals.len(), slots.n_hydros());
            debug_assert!(
                opening.objective.is_finite(),
                "aggregate-first chain rule: an opening objective is not finite"
            );
            if mu == 0.0 {
                continue;
            }
            q_bar += mu * opening.objective;
            for (range, positions) in &non_lag {
                let dst = &mut coefficients[range.clone()];
                for (c, &d) in dst.iter_mut().zip(&opening.state_duals[positions.clone()]) {
                    *c += mu * d;
                }
            }
        }
        if lag.is_empty() {
            continue;
        }

        let populated = child.inputs.resident_pool.populated();
        debug_assert!(
            y_bar.len() >= populated,
            "aggregate scratch is not reserved for the resident pool"
        );
        chain.g.fill(-0.0);
        let (mut first_word, mut last_word) = (usize::MAX, 0);
        for (opening, &mu) in child.openings.iter().zip(child_weights) {
            if mu == 0.0 {
                continue;
            }
            for (g_h, &coupled) in chain.g.iter_mut().zip(opening.coupled_duals) {
                *g_h -= mu * coupled;
            }
            for &(slot, y) in opening.cut_duals {
                if y == 0.0 {
                    continue;
                }
                y_bar[slot] += mu * y;
                let word = slot / 64;
                touched[word] |= 1u64 << (slot % 64);
                first_word = first_word.min(word);
                last_word = last_word.max(word);
            }
        }
        for (word_at, word) in touched
            .iter_mut()
            .enumerate()
            .take(last_word + 1)
            .skip(first_word)
        {
            let mut bits = std::mem::take(word);
            while bits != 0 {
                let slot = word_at * 64 + bits.trailing_zeros() as usize;
                bits &= bits - 1;
                pairs.push((slot, std::mem::take(&mut y_bar[slot])));
            }
        }

        lag_coefficients_into(&child.inputs, pairs, chain, child_lag);
        for (acc, &v) in coefficients[lag.clone()].iter_mut().zip(child_lag.iter()) {
            *acc += v;
        }
        pairs.clear();
    }
    q_bar - extracted.dot_trial_state(coefficients, x_hat)
}

#[cfg(test)]
mod tests {
    use novomodelo_core::EntityId;
    use novomodelo_core::temporal::StageStateConfig;

    use super::{
        AggregateScratch, ChainRuleInputs, ChainRuleScratch, ChainRuleSlots, ChildOpenings,
        OpeningDuals, PrecomputedPar, aggregate_first_cut_into, per_opening_cut_into,
    };
    use crate::cut::CutPool;
    use crate::indexer::CutStateProjection;
    use crate::risk_measure::{BackwardOutcome, RiskMeasure, RiskMeasureScratch};
    use crate::setup::NodeId;
    use crate::test_support::{par_with_ar_coefficients, state_layout};

    const ALL: StageStateConfig = StageStateConfig {
        storage: true,
        inflow_lags: true,
    };
    const STORAGE_ONLY: StageStateConfig = StageStateConfig {
        storage: true,
        inflow_lags: false,
    };

    struct Pair {
        slots: ChainRuleSlots,
        pool: CutPool,
        extracted: CutStateProjection,
        par: PrecomputedPar,
    }

    impl Pair {
        fn new(
            (n_hydros, lag_depth): (usize, usize),
            (resident, extracted): (StageStateConfig, StageStateConfig),
            rows: &[&[f64]],
            par: PrecomputedPar,
        ) -> Self {
            let global = state_layout(n_hydros, lag_depth);
            let resident = CutStateProjection::new(&global, resident);
            let extracted = CutStateProjection::new(&global, extracted);
            let mut pool = CutPool::new(rows.len(), resident.n_slots(), 1, 0);
            for (k, row) in rows.iter().enumerate() {
                pool.add_cut(NodeId(0), k as u64, 0, 0.0, row);
            }
            Self {
                slots: ChainRuleSlots::compose(&global, &resident, &extracted),
                pool,
                extracted,
                par,
            }
        }

        fn inputs(&self) -> ChainRuleInputs<'_> {
            ChainRuleInputs {
                slots: &self.slots,
                resident_pool: &self.pool,
                extracted: &self.extracted,
                par: &self.par,
                stage: 0,
            }
        }

        fn run(&self, opening: &OpeningDuals<'_>, x_hat: &[f64]) -> (Vec<f64>, f64) {
            let mut scratch = ChainRuleScratch::new(self.slots.n_hydros(), self.slots.lag_depth());
            let mut coefficients = vec![0.0; self.extracted.n_slots()];
            let alpha = per_opening_cut_into(
                &self.inputs(),
                opening,
                x_hat,
                &mut scratch,
                &mut coefficients,
            );
            (coefficients, alpha)
        }
    }

    fn aggregate(
        children: &[ChildOpenings<'_>],
        weights: &[f64],
        x_hat: &[f64],
        scratch: &mut AggregateScratch,
    ) -> (Vec<f64>, f64) {
        let mut coefficients = vec![0.0; children[0].inputs.extracted.n_slots()];
        let alpha = aggregate_first_cut_into(children, weights, x_hat, scratch, &mut coefficients);
        (coefficients, alpha)
    }

    fn reserved_scratch(pairs: &[&Pair]) -> AggregateScratch {
        let first = pairs[0];
        let mut scratch = AggregateScratch::new(first.slots.n_hydros(), first.slots.lag_depth());
        for pair in pairs {
            scratch.reserve(pair.pool.capacity, pair.pool.populated());
        }
        scratch
    }

    fn par_with_psi(n_hydros: usize, psi: &[f64]) -> PrecomputedPar {
        let ids: Vec<EntityId> = (0..n_hydros)
            .map(|h| EntityId::from(i32::try_from(h).unwrap()))
            .collect();
        let par = par_with_ar_coefficients(&ids, psi);
        for h in 0..n_hydros {
            let held: Vec<u64> = par.psi_slice(0, h).iter().map(|v| v.to_bits()).collect();
            let wanted: Vec<u64> = psi.iter().map(|v| v.to_bits()).collect();
            assert_eq!(
                held, wanted,
                "psi_slice(0, {h}) must hold the requested bits"
            );
        }
        par
    }

    fn bits(values: &[f64]) -> Vec<u64> {
        values.iter().map(|v| v.to_bits()).collect()
    }

    const CLOSED_FORM_X_HAT: [f64; 6] = [10.0, 20.0, 1.0, 2.0, 3.0, 4.0];
    const CLOSED_FORM_ROWS: [[f64; 6]; 2] = [
        [9.0, 9.0, 1.0, 2.0, 3.0, 4.0],
        [9.0, 9.0, 5.0, 6.0, 7.0, 8.0],
    ];

    #[test]
    fn per_opening_cut_matches_the_closed_form_on_a_par2_pair() {
        let pair = Pair::new(
            (2, 2),
            (ALL, ALL),
            &[
                &CLOSED_FORM_ROWS[0],
                &CLOSED_FORM_ROWS[1],
                &[9.0, 9.0, 100.0, 100.0, 100.0, 100.0],
            ],
            par_with_psi(2, &[0.5, 0.25]),
        );
        let opening = OpeningDuals {
            objective: 100.0,
            state_duals: &[-1.5, -2.5],
            coupled_duals: &[4.0, 8.0],
            cut_duals: &[(0, 2.0), (1, 1.0), (2, 0.0)],
        };

        let (coefficients, alpha) = pair.run(&opening, &CLOSED_FORM_X_HAT);

        assert_eq!(coefficients, [-1.5, -2.5, 14.5, 17.0, 0.75, 0.5]);
        assert_eq!(alpha, 112.25);
    }

    #[test]
    fn one_scratch_serves_every_pair_of_the_layout() {
        let a = Pair::new(
            (2, 2),
            (ALL, ALL),
            &[&CLOSED_FORM_ROWS[0], &CLOSED_FORM_ROWS[1]],
            par_with_psi(2, &[0.5, 0.25]),
        );
        let b = Pair::new(
            (2, 2),
            (STORAGE_ONLY, ALL),
            &[&[100.0, 50.0], &[25.0, 75.0]],
            par_with_psi(2, &[0.5, 0.25]),
        );
        let opening_a = OpeningDuals {
            objective: 100.0,
            state_duals: &[-1.5, -2.5],
            coupled_duals: &[4.0, 8.0],
            cut_duals: &[(0, 2.0), (1, 1.0)],
        };
        let opening_b = OpeningDuals {
            objective: 40.0,
            state_duals: &[0.25, -0.5],
            coupled_duals: &[3.0, -6.0],
            cut_duals: &[(0, 3.0), (1, 1.5)],
        };

        let mut scratch = ChainRuleScratch::new(2, 2);
        for (pair, opening) in [(&a, &opening_a), (&b, &opening_b)] {
            let mut coefficients = vec![0.0; pair.extracted.n_slots()];
            let alpha = per_opening_cut_into(
                &pair.inputs(),
                opening,
                &CLOSED_FORM_X_HAT,
                &mut scratch,
                &mut coefficients,
            );

            let (fresh, fresh_alpha) = pair.run(opening, &CLOSED_FORM_X_HAT);

            assert_eq!(bits(&coefficients), bits(&fresh));
            assert_eq!(alpha.to_bits(), fresh_alpha.to_bits());
        }
    }

    #[test]
    fn zero_cut_duals_leave_the_bits_unchanged() {
        assert!((0.0 * f64::INFINITY).is_nan());
        let poison = [f64::INFINITY; 6];
        let pair = Pair::new(
            (2, 2),
            (ALL, ALL),
            &[&CLOSED_FORM_ROWS[0], &poison, &CLOSED_FORM_ROWS[1], &poison],
            par_with_psi(2, &[0.5, 0.25]),
        );
        let run = |cut_duals: &[(usize, f64)]| {
            pair.run(
                &OpeningDuals {
                    objective: 100.0,
                    state_duals: &[-1.5, -2.5],
                    coupled_duals: &[4.0, 8.0],
                    cut_duals,
                },
                &CLOSED_FORM_X_HAT,
            )
        };

        let (sparse, sparse_alpha) = run(&[(0, 2.0), (2, 1.0)]);
        let (padded, padded_alpha) = run(&[(0, 2.0), (1, 0.0), (2, 1.0), (3, -0.0)]);

        assert!(sparse.iter().all(|v| !v.is_nan()) && !sparse_alpha.is_nan());
        assert_eq!(bits(&sparse), bits(&padded));
        assert_eq!(sparse_alpha.to_bits(), padded_alpha.to_bits());
    }

    #[test]
    fn single_accumulator_keeps_the_sign_of_a_negated_zero_coupling() {
        let forbidden_form = 0.5 * (-0.0_f64 + 0.0);
        assert_eq!(forbidden_form.to_bits(), 0.0_f64.to_bits());
        let pair = Pair::new((1, 1), (ALL, ALL), &[], par_with_psi(1, &[0.5]));
        let opening = OpeningDuals {
            objective: 0.0,
            state_duals: &[0.0],
            coupled_duals: &[0.0],
            cut_duals: &[],
        };

        let (coefficients, _) = pair.run(&opening, &[0.0, 0.0]);

        assert_eq!(coefficients[1].to_bits(), (-0.0_f64).to_bits());
    }

    #[test]
    fn an_empty_cut_sum_keeps_the_sign_of_a_negated_zero_coefficient() {
        let plus_zero_start = 0.5 * -0.0_f64 + 0.0;
        assert_eq!(plus_zero_start.to_bits(), 0.0_f64.to_bits());
        let pair = Pair::new((1, 2), (ALL, ALL), &[], par_with_psi(1, &[0.5, 0.25]));
        let opening = OpeningDuals {
            objective: 0.0,
            state_duals: &[0.0],
            coupled_duals: &[0.0],
            cut_duals: &[],
        };

        let (coefficients, _) = pair.run(&opening, &[0.0; 3]);

        assert_eq!(bits(&coefficients[1..]), bits(&[-0.0, -0.0]));
    }

    #[test]
    fn lagless_resident_gives_psi_times_the_negated_coupling() {
        let pair = Pair::new(
            (2, 2),
            (STORAGE_ONLY, ALL),
            &[&[100.0, 100.0]],
            par_with_psi(2, &[0.5, 0.25]),
        );
        let opening = OpeningDuals {
            objective: 0.0,
            state_duals: &[0.0; 2],
            coupled_duals: &[4.0, 8.0],
            cut_duals: &[(0, 3.0)],
        };

        let (coefficients, _) = pair.run(&opening, &CLOSED_FORM_X_HAT);

        assert_eq!(coefficients[2..], [-2.0, -4.0, -1.0, -2.0]);
    }

    #[test]
    fn lagless_extracted_cut_is_the_state_duals() {
        let pair = Pair::new(
            (2, 2),
            (ALL, STORAGE_ONLY),
            &[&CLOSED_FORM_ROWS[0]],
            par_with_psi(2, &[0.5, 0.25]),
        );
        let opening = OpeningDuals {
            objective: 100.0,
            state_duals: &[-1.5, -2.5],
            coupled_duals: &[4.0, 8.0],
            cut_duals: &[(0, 2.0)],
        };

        let (coefficients, alpha) = pair.run(&opening, &CLOSED_FORM_X_HAT);

        assert_eq!(bits(&coefficients), bits(&[-1.5, -2.5]));
        assert_eq!(alpha, 165.0);
    }

    #[test]
    fn lags_past_the_par_order_take_a_zero_psi() {
        let pair = Pair::new((1, 3), (ALL, ALL), &[], par_with_psi(1, &[0.5, 0.25]));
        let opening = OpeningDuals {
            objective: 0.0,
            state_duals: &[0.0],
            coupled_duals: &[-2.0],
            cut_duals: &[],
        };

        let (coefficients, _) = pair.run(&opening, &[0.0; 4]);

        assert_eq!(coefficients[1..], [1.0, 0.5, 0.0]);
    }

    #[test]
    fn orderless_par_never_reads_psi() {
        let pair = Pair::new((1, 1), (ALL, ALL), &[], PrecomputedPar::default());
        let opening = OpeningDuals {
            objective: 0.0,
            state_duals: &[0.0],
            coupled_duals: &[-2.0],
            cut_duals: &[],
        };

        let (coefficients, _) = pair.run(&opening, &[0.0; 2]);

        assert_eq!(coefficients[1].to_bits(), 0.0_f64.to_bits());
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "strictly ascend")]
    fn non_ascending_cut_duals_panic_in_debug() {
        let pair = Pair::new(
            (2, 2),
            (ALL, ALL),
            &[&CLOSED_FORM_ROWS[0], &CLOSED_FORM_ROWS[1]],
            par_with_psi(2, &[0.5, 0.25]),
        );
        let opening = OpeningDuals {
            objective: 0.0,
            state_duals: &[0.0; 2],
            coupled_duals: &[4.0, 8.0],
            cut_duals: &[(1, 1.0), (0, 2.0)],
        };

        let _ = pair.run(&opening, &CLOSED_FORM_X_HAT);
    }

    #[test]
    fn aggregate_of_one_opening_at_unit_weight_equals_the_per_opening_cut() {
        let flipped_by_a_plus_zero_start = 0.0_f64 + -0.0_f64;
        assert_eq!(flipped_by_a_plus_zero_start.to_bits(), 0.0_f64.to_bits());
        let pair = Pair::new(
            (2, 2),
            (ALL, ALL),
            &[
                &CLOSED_FORM_ROWS[0],
                &CLOSED_FORM_ROWS[1],
                &[9.0, 9.0, 100.0, 100.0, 100.0, 100.0],
            ],
            par_with_psi(2, &[0.5, 0.25]),
        );
        let openings = [OpeningDuals {
            objective: 100.0,
            state_duals: &[-0.0, -2.5],
            coupled_duals: &[4.0, 8.0],
            cut_duals: &[(0, 2.0), (1, 1.0), (2, 0.0)],
        }];
        let (expected, expected_alpha) = pair.run(&openings[0], &CLOSED_FORM_X_HAT);
        assert_eq!(expected[0].to_bits(), (-0.0_f64).to_bits());

        let children = [ChildOpenings {
            inputs: pair.inputs(),
            openings: &openings,
        }];
        let mut scratch = reserved_scratch(&[&pair]);
        let (coefficients, alpha) = aggregate(&children, &[1.0], &CLOSED_FORM_X_HAT, &mut scratch);

        assert_eq!(bits(&coefficients), bits(&expected));
        assert_eq!(alpha.to_bits(), expected_alpha.to_bits());
    }

    #[test]
    fn aggregate_on_a_cvar_fan_uses_joint_weights_never_renormalised() {
        let x_hat = [10.0, 1.0, 2.0];
        let pair0 = Pair::new(
            (1, 2),
            (ALL, ALL),
            &[&[0.0, 1.0, 2.0], &[0.0, 3.0, 1.0]],
            par_with_psi(1, &[0.5, 0.25]),
        );
        let pair1 = Pair::new(
            (1, 2),
            (ALL, ALL),
            &[&[0.0, 2.0, 2.0]],
            par_with_psi(1, &[0.5, 0.25]),
        );
        let child0 = [
            OpeningDuals {
                objective: 40.0,
                state_duals: &[-1.0],
                coupled_duals: &[2.0],
                cut_duals: &[(0, 1.0)],
            },
            OpeningDuals {
                objective: 30.0,
                state_duals: &[-2.0],
                coupled_duals: &[4.0],
                cut_duals: &[(0, 1.0), (1, 2.0)],
            },
        ];
        let child1 = [
            OpeningDuals {
                objective: 20.0,
                state_duals: &[-1.0],
                coupled_duals: &[1.0],
                cut_duals: &[(0, 3.0)],
            },
            OpeningDuals {
                objective: 10.0,
                state_duals: &[-3.0],
                coupled_duals: &[0.0],
                cut_duals: &[],
            },
        ];
        let measure = RiskMeasure::CVaR {
            alpha: 0.5,
            lambda: 0.5,
        };
        let mut risk_scratch = RiskMeasureScratch::new();
        let weights =
            measure.opening_weights_into(&[40.0, 30.0, 20.0, 10.0], &[0.25; 4], &mut risk_scratch);
        assert_eq!(weights, [0.375, 0.375, 0.125, 0.125]);
        assert_eq!(weights[0] + weights[1], 0.75);

        let per_opening = [
            (&pair0, &child0[0]),
            (&pair0, &child0[1]),
            (&pair1, &child1[0]),
            (&pair1, &child1[1]),
        ];
        let outcomes: Vec<BackwardOutcome> = per_opening
            .iter()
            .map(|(pair, opening)| {
                let (coefficients, intercept) = pair.run(opening, &x_hat);
                BackwardOutcome {
                    intercept,
                    coefficients,
                    objective_value: opening.objective,
                }
            })
            .collect();
        let (reference_alpha, reference) = measure.aggregate_cut(&outcomes, &[0.25; 4]);

        let children = [
            ChildOpenings {
                inputs: pair0.inputs(),
                openings: &child0,
            },
            ChildOpenings {
                inputs: pair1.inputs(),
                openings: &child1,
            },
        ];
        let mut scratch = reserved_scratch(&[&pair0, &pair1]);
        let (coefficients, alpha) = aggregate(&children, weights, &x_hat, &mut scratch);

        assert_eq!(coefficients, [-1.625, 3.6875, 0.34375]);
        assert_eq!(alpha, 41.875);
        assert_eq!(coefficients, reference);
        assert_eq!(alpha, reference_alpha);
    }

    const SCAN_X_HAT: [f64; 3] = [1.0; 3];

    fn scan_fixture() -> (Pair, [OpeningDuals<'static>; 2]) {
        let mut rows = vec![[0.0; 3]; 71];
        rows[0] = [0.0, 1.0, 1.0];
        rows[3] = [0.0, 2.0, 1.0];
        rows[65] = [0.0, 4.0, 1.0];
        rows[70] = [0.0, 8.0, 1.0];
        let row_refs: Vec<&[f64]> = rows.iter().map(<[f64; 3]>::as_slice).collect();
        let pair = Pair::new((1, 2), (ALL, ALL), &row_refs, par_with_psi(1, &[0.5, 0.25]));
        assert_eq!(pair.pool.populated(), 71);
        let openings = [
            OpeningDuals {
                objective: 10.0,
                state_duals: &[-1.0],
                coupled_duals: &[0.0],
                cut_duals: &[(3, 1.0), (70, 1.0)],
            },
            OpeningDuals {
                objective: 10.0,
                state_duals: &[-1.0],
                coupled_duals: &[0.0],
                cut_duals: &[(0, 1.0), (65, 1.0)],
            },
        ];
        (pair, openings)
    }

    #[test]
    fn aggregate_scan_yields_ascending_slots_and_resets_only_touched_words() {
        let (pair, openings) = scan_fixture();
        let x_hat = SCAN_X_HAT;
        let children = [ChildOpenings {
            inputs: pair.inputs(),
            openings: &openings,
        }];
        let mut scratch = reserved_scratch(&[&pair]);

        let (coefficients, alpha) = aggregate(&children, &[0.5, 0.5], &x_hat, &mut scratch);

        assert_eq!(coefficients, [-1.0, 5.75, 1.875]);
        assert_eq!(alpha, 3.375);
        assert_eq!(scratch.touched.len(), 2);
        assert!(scratch.y_bar.iter().all(|y| y.to_bits() == 0));
        assert!(scratch.touched.iter().all(|&word| word == 0));

        let untouched = 99.0;
        scratch.y_bar[10] = untouched;
        let single = [OpeningDuals {
            objective: 10.0,
            state_duals: &[-1.0],
            coupled_duals: &[0.0],
            cut_duals: &[(65, 1.0)],
        }];
        let again = [ChildOpenings {
            inputs: pair.inputs(),
            openings: &single,
        }];
        let (reused, reused_alpha) = aggregate(&again, &[1.0], &x_hat, &mut scratch);
        let (fresh, fresh_alpha) =
            aggregate(&again, &[1.0], &x_hat, &mut reserved_scratch(&[&pair]));

        assert_eq!(reused, [-1.0, 3.0, 1.0]);
        assert_eq!(reused_alpha, 7.0);
        assert_eq!(bits(&reused), bits(&fresh));
        assert_eq!(reused_alpha.to_bits(), fresh_alpha.to_bits());
        assert_eq!(scratch.y_bar[10], untouched);
        assert!(
            scratch
                .y_bar
                .iter()
                .enumerate()
                .all(|(slot, y)| slot == 10 || y.to_bits() == 0)
        );
        assert!(scratch.touched.iter().all(|&word| word == 0));
    }

    #[test]
    fn aggregate_ignores_the_cut_duals_of_a_zero_weight_opening() {
        assert!((0.0 * f64::INFINITY).is_nan());
        let x_hat = [10.0, 1.0, 2.0];
        let pair = Pair::new(
            (1, 2),
            (ALL, ALL),
            &[&[0.0, 1.0, 2.0], &[f64::INFINITY; 3]],
            par_with_psi(1, &[0.5, 0.25]),
        );
        let openings = [
            OpeningDuals {
                objective: 40.0,
                state_duals: &[-1.0],
                coupled_duals: &[2.0],
                cut_duals: &[(0, 1.0)],
            },
            OpeningDuals {
                objective: 30.0,
                state_duals: &[-2.0],
                coupled_duals: &[4.0],
                cut_duals: &[(1, 5.0)],
            },
        ];
        let (expected, expected_alpha) = pair.run(&openings[0], &x_hat);
        assert_eq!(expected, [-1.0, 1.5, -0.25]);
        assert_eq!(expected_alpha, 49.0);

        let children = [ChildOpenings {
            inputs: pair.inputs(),
            openings: &openings,
        }];
        let (coefficients, alpha) = aggregate(
            &children,
            &[1.0, 0.0],
            &x_hat,
            &mut reserved_scratch(&[&pair]),
        );

        assert!(coefficients.iter().all(|v| !v.is_nan()) && !alpha.is_nan());
        assert_eq!(bits(&coefficients), bits(&expected));
        assert_eq!(alpha.to_bits(), expected_alpha.to_bits());
    }

    #[test]
    fn a_reserved_scratch_never_grows() {
        let (pair, openings) = scan_fixture();
        let mut scratch = AggregateScratch::new(pair.slots.n_hydros(), pair.slots.lag_depth());
        scratch.reserve(71, 4);
        assert!(scratch.pairs.capacity() < pair.pool.populated());
        let held = |scratch: &AggregateScratch| {
            (
                scratch.y_bar.as_ptr(),
                scratch.y_bar.capacity(),
                scratch.touched.as_ptr(),
                scratch.touched.capacity(),
                scratch.pairs.as_ptr(),
                scratch.pairs.capacity(),
            )
        };
        let before = held(&scratch);
        let children = [ChildOpenings {
            inputs: pair.inputs(),
            openings: &openings,
        }];

        let _ = aggregate(&children, &[0.5, 0.5], &SCAN_X_HAT, &mut scratch);

        assert_eq!(held(&scratch), before);
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "aggregate scratch is not reserved for the resident pool")]
    fn an_unreserved_scratch_panics_in_debug() {
        let pair = Pair::new(
            (1, 2),
            (ALL, ALL),
            &[&[0.0, 1.0, 1.0]],
            par_with_psi(1, &[0.5, 0.25]),
        );
        let mut scratch = AggregateScratch::new(1, 2);
        assert!(scratch.y_bar.len() < pair.pool.populated());
        let openings = [OpeningDuals {
            objective: 10.0,
            state_duals: &[-1.0],
            coupled_duals: &[2.0],
            cut_duals: &[(0, 1.0)],
        }];
        let children = [ChildOpenings {
            inputs: pair.inputs(),
            openings: &openings,
        }];

        let _ = aggregate(&children, &[1.0], &[1.0; 3], &mut scratch);
    }

    #[test]
    fn a_zero_weight_opening_changes_no_bit() {
        let pair = Pair::new((1, 1), (ALL, ALL), &[&[0.0, 1.0]], par_with_psi(1, &[0.5]));
        let x_hat = [-1.0, -1.0];
        let openings = [
            OpeningDuals {
                objective: -0.0,
                state_duals: &[-0.0],
                coupled_duals: &[0.0],
                cut_duals: &[],
            },
            OpeningDuals {
                objective: 30.0,
                state_duals: &[2.0],
                coupled_duals: &[-4.0],
                cut_duals: &[(0, 5.0)],
            },
        ];
        let (expected, expected_alpha) = pair.run(&openings[0], &x_hat);
        assert_eq!(bits(&expected), bits(&[-0.0, -0.0]));
        assert_eq!(expected_alpha.to_bits(), (-0.0_f64).to_bits());

        let children = [ChildOpenings {
            inputs: pair.inputs(),
            openings: &openings,
        }];
        let (coefficients, alpha) = aggregate(
            &children,
            &[1.0, 0.0],
            &x_hat,
            &mut reserved_scratch(&[&pair]),
        );

        assert_eq!(bits(&coefficients), bits(&expected));
        assert_eq!(alpha.to_bits(), expected_alpha.to_bits());
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "chain rule: the PAR order exceeds the lag depth")]
    fn a_par_order_deeper_than_the_lag_depth_panics_in_debug() {
        let pair = Pair::new((1, 1), (ALL, ALL), &[], par_with_psi(1, &[0.5, 0.25]));
        assert!(pair.par.psi_slice(0, 0).len() > pair.slots.lag_depth());
        let opening = OpeningDuals {
            objective: 0.0,
            state_duals: &[0.0],
            coupled_duals: &[-2.0],
            cut_duals: &[],
        };

        let _ = pair.run(&opening, &[0.0; 2]);
    }
}
