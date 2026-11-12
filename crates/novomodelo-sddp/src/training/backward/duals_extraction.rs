//! Benders subgradient extraction from a solved backward LP.
//!
//! The subgradient is `rc_scaled / col_scale[col]` — **divided, not multiplied** —
//! because the incoming-state column pin sets `v_scaled = v_orig / col_scale`
//! (sddp.md "Benders cut sign & subgradient extraction").

use novomodelo_solver::SolutionView;
use novomodelo_stochastic::par::precompute::PrecomputedPar;

use crate::{
    cut::{
        CutPool, CutRowMap,
        chain_rule::{ChainRuleInputs, ChainRuleScratch, OpeningDuals, per_opening_cut_into},
    },
    indexer::{ChainRuleSlots, CutStateProjection, HydroSys, InCol},
    lp::builder::InflowCoupling,
};

fn unscaled_reduced_cost(view: &SolutionView<'_>, col_scale: &[f64], col: InCol) -> f64 {
    // Divided, not multiplied (module-level contract). Empty col_scale ⇒ raw rc.
    let col = col.get();
    let rc = view.reduced_costs[col];
    if col_scale.is_empty() {
        rc
    } else {
        rc / col_scale[col]
    }
}

/// Appends `(slot, y)` for each cut row `k` whose dual `y = dual[first_cut_row + k]`
/// is not zero of either sign, in row order. The filter is `!= 0.0` exactly: a bit
/// test keeps the `-0.0` of a basic row, a tolerance drops tiny real duals.
fn push_nonzero_cut_row_duals(
    dual: &[f64],
    first_cut_row: usize,
    cut_row_slots: &[usize],
    out: &mut Vec<(usize, f64)>,
) {
    for (k, &slot) in cut_row_slots.iter().enumerate() {
        let y = dual[first_cut_row + k];
        if y != 0.0 {
            out.push((slot, y));
        }
    }
}

/// What the extraction reads of one solved opening.
pub(crate) struct DualExtractionSpec<'a> {
    /// Projection the opening's state duals are read in.
    pub(crate) extracted: &'a CutStateProjection,
    /// The loaded compact template's column scales; empty when unscaled.
    pub(crate) col_scale: &'a [f64],
    /// The loaded compact template's row scales; empty when unscaled.
    pub(crate) row_scale: &'a [f64],
    /// The stage's coupling of the realized inflow into the base rows.
    pub(crate) coupling: &'a InflowCoupling,
    /// LP row of the first cut row.
    pub(crate) first_cut_row: usize,
    /// Resident pool slot of each cut row, in LP-row order.
    pub(crate) cut_row_slots: &'a [usize],
    pub(crate) read_cut_duals: bool,
}

/// The per-worker buffers the extraction fills.
pub(crate) struct OpeningDualBuffers<'a> {
    pub(crate) state_duals: &'a mut Vec<f64>,
    pub(crate) coupled_duals: &'a mut Vec<f64>,
    pub(crate) cut_dual_slots: &'a mut Vec<(usize, f64)>,
}

/// The chain-rule inputs of one opening other than the extracted projection, which
/// [`DualExtractionSpec`] carries.
pub(crate) struct ChainRuleStageInputs<'a> {
    pub(crate) slots: &'a ChainRuleSlots,
    pub(crate) resident_pool: &'a CutPool,
    pub(crate) par: &'a PrecomputedPar,
    pub(crate) stage: usize,
}

/// Fills `out` from a solved compact opening and returns its objective Q.
/// `state_duals` holds the unscaled reduced cost of each slot of
/// `spec.extracted.incoming_pairs()`, in that order, with no entry for a lag slot,
/// whose coefficient the chain rule writes. `coupled_duals` holds Gᵀπ per hydro in
/// unscaled units, from the coupled base rows' duals `row_scale[r] * dual[r]`.
/// `cut_dual_slots` holds the zero-free (slot, y) cut-row duals in strictly
/// ascending slot order, and is left empty unless `read_cut_duals`. Set
/// `read_cut_duals` when the resident pool's projection carries lag slots or
/// binding activity is tracked.
pub(crate) fn extract_opening_duals(
    view: &SolutionView<'_>,
    spec: &DualExtractionSpec<'_>,
    out: &mut OpeningDualBuffers<'_>,
) -> f64 {
    let extracted = spec.extracted;
    debug_assert_eq!(
        extracted.columned_len(),
        extracted.n_slots() - extracted.lag_point().full().len(),
        "extraction reads a projection whose lag slots have no LP column"
    );
    out.state_duals.clear();
    out.state_duals.extend(
        extracted
            .incoming_pairs()
            .map(|(_, col)| unscaled_reduced_cost(view, spec.col_scale, col)),
    );

    out.coupled_duals.clear();
    for h in 0..spec.coupling.n_hydros() {
        let mut acc = -0.0_f64;
        for (row, coefficient) in spec.coupling.hydro_entries(HydroSys::new(h)) {
            let pi = if spec.row_scale.is_empty() {
                view.dual[row]
            } else {
                view.dual[row] * spec.row_scale[row]
            };
            acc += coefficient * pi;
        }
        out.coupled_duals.push(acc);
    }

    out.cut_dual_slots.clear();
    if spec.read_cut_duals {
        push_nonzero_cut_row_duals(
            view.dual,
            spec.first_cut_row,
            spec.cut_row_slots,
            out.cut_dual_slots,
        );
        debug_assert!(
            out.cut_dual_slots.windows(2).all(|w| w[0].0 < w[1].0),
            "cut-row slots must strictly ascend in LP-row order"
        );
    }

    view.objective
}

/// [`extract_opening_duals`], then [`per_opening_cut_into`] on the same
/// `spec.extracted`: writes the opening's cut coefficients into `coefficients`, whose
/// length is `spec.extracted.n_slots()`, and returns the objective and the intercept.
pub(crate) fn extract_opening_cut(
    view: &SolutionView<'_>,
    spec: &DualExtractionSpec<'_>,
    chain: &ChainRuleStageInputs<'_>,
    x_hat: &[f64],
    duals: &mut OpeningDualBuffers<'_>,
    scratch: &mut ChainRuleScratch,
    coefficients: &mut [f64],
) -> (f64, f64) {
    let objective = extract_opening_duals(view, spec, duals);
    let intercept = per_opening_cut_into(
        &ChainRuleInputs {
            slots: chain.slots,
            resident_pool: chain.resident_pool,
            extracted: spec.extracted,
            par: chain.par,
            stage: chain.stage,
        },
        &OpeningDuals {
            objective,
            state_duals: duals.state_duals,
            coupled_duals: duals.coupled_duals,
            cut_duals: duals.cut_dual_slots,
        },
        x_hat,
        scratch,
        coefficients,
    );
    (objective, intercept)
}

/// Fills `out` with (pool slot, raw cut-row dual) pairs for the lazy solve's
/// resident cuts, in ascending slot order.
///
/// Under a lazy solve the resident rows are a row-map-ordered subset, not the
/// frozen cut-row order, so a resident slot's dual is
/// `dual[row_map.lp_row_for_slot(slot)]`. A non-resident slot did not bind (by
/// exactness; else the lazy loop would have added it), so leaving it out is
/// correct.
///
/// A zero dual (either sign) yields no pair.
///
/// `dual` must be the FINAL all-satisfied solve's dual vector and `row_map` the
/// residency that produced it.
pub(crate) fn fill_resident_cut_dual_slots(
    dual: &[f64],
    row_map: &CutRowMap,
    pool: &CutPool,
    out: &mut Vec<(usize, f64)>,
) {
    out.clear();
    out.extend((0..pool.populated()).filter_map(|slot| {
        let lp_row = row_map.lp_row_for_slot(slot)?;
        let y = dual[lp_row];
        (y != 0.0).then_some((slot, y))
    }));
}

#[cfg(test)]
#[allow(clippy::cast_precision_loss)]
mod tests {
    use novomodelo_solver::{ActiveSolver, SolutionView, SolverInterface, StageTemplate};

    use super::{
        DualExtractionSpec, OpeningDualBuffers, extract_opening_duals,
        fill_resident_cut_dual_slots, push_nonzero_cut_row_duals,
    };
    use crate::cut::{CutPool, CutRowMap};
    use crate::indexer::{CutSlot, CutStateProjection, HydroSys, StateDim, StateSpace};
    use crate::lead_time::AnticipatedResolution;
    use crate::lp::builder::{InflowCoupling, inflow_coupling::InflowCouplingSink};
    use crate::setup::NodeId;
    use crate::training::backward::outcome_aggregation::count_binding_slots;
    use novomodelo_core::temporal::StageStateConfig;

    const ALL_ENABLED: StageStateConfig = StageStateConfig {
        storage: true,
        inflow_lags: true,
    };
    const STORAGE_ONLY: StageStateConfig = StageStateConfig {
        storage: true,
        inflow_lags: false,
    };
    const OBJECTIVE: f64 = 12.5;

    fn state_layout(hydro_count: usize, max_par_order: usize) -> StateSpace {
        let lag_counts = vec![max_par_order; hydro_count];
        StateSpace::new(
            hydro_count,
            max_par_order,
            Vec::new(),
            vec![],
            AnticipatedResolution::default(),
            &lag_counts,
        )
    }

    /// Build a `SolutionView` whose `reduced_costs[col]` equals `col as f64`, so
    /// an extracted subgradient at LP column `col` is exactly `col as f64`
    /// (`col_scale` empty ⇒ raw rc).
    fn indexed_view(reduced_costs: &[f64]) -> SolutionView<'_> {
        SolutionView {
            objective: 0.0,
            primal: &[],
            dual: &[],
            reduced_costs,
            iterations: 0,
            solve_time_seconds: 0.0,
        }
    }

    fn compact_layout(
        hydro_count: usize,
        max_par_order: usize,
        buckets: Vec<(HydroSys, usize)>,
    ) -> StateSpace {
        let lag_counts = vec![max_par_order; hydro_count];
        StateSpace::new(
            hydro_count,
            max_par_order,
            buckets,
            vec![],
            AnticipatedResolution::default(),
            &lag_counts,
        )
    }

    fn view_with_duals<'a>(reduced_costs: &'a [f64], dual: &'a [f64]) -> SolutionView<'a> {
        SolutionView {
            objective: OBJECTIVE,
            primal: &[],
            dual,
            reduced_costs,
            iterations: 0,
            solve_time_seconds: 0.0,
        }
    }

    fn empty_coupling(n_hydros: usize) -> InflowCoupling {
        InflowCouplingSink::new(n_hydros).finish(&[], &[])
    }

    #[derive(Default)]
    struct Buffers {
        state_duals: Vec<f64>,
        coupled_duals: Vec<f64>,
        cut_dual_slots: Vec<(usize, f64)>,
    }

    impl Buffers {
        fn extract(&mut self, view: &SolutionView<'_>, spec: &DualExtractionSpec<'_>) -> f64 {
            extract_opening_duals(
                view,
                spec,
                &mut OpeningDualBuffers {
                    state_duals: &mut self.state_duals,
                    coupled_duals: &mut self.coupled_duals,
                    cut_dual_slots: &mut self.cut_dual_slots,
                },
            )
        }
    }

    /// Storage-only `CutStateProjection` (`N=3, L=2`): extraction yields an `N`-length
    /// subgradient reading exactly the storage columns `[0, N)`, no lag entries.
    #[test]
    fn storage_only_yields_n_length_subgradient_with_storage_columns() {
        let global = state_layout(3, 2);
        let cut_state = CutStateProjection::new(&global, STORAGE_ONLY);
        assert_eq!(cut_state.n_slots(), 3);

        let rc: Vec<f64> = (0..=global.theta).map(|c| c as f64).collect();
        let coupling = empty_coupling(3);
        let mut buffers = Buffers::default();

        buffers.extract(
            &indexed_view(&rc),
            &DualExtractionSpec {
                extracted: &cut_state,
                col_scale: &[],
                row_scale: &[],
                coupling: &coupling,
                first_cut_row: 0,
                cut_row_slots: &[],
                read_cut_duals: false,
            },
        );

        assert_eq!(buffers.state_duals.len(), 3);
        for (j, &dual) in buffers.state_duals.iter().enumerate() {
            assert_eq!(
                dual,
                global.state_to_lp_incoming_column(StateDim::new(j)).get() as f64,
                "storage slot {j} reads its storage_in column reduced cost"
            );
        }
    }

    /// Slots 3 then 1 are resident, so the row map's order (3, 1) is not slot
    /// order; the pairs follow ascending slot and slot 1's zero dual yields no pair.
    #[test]
    fn fill_resident_cut_dual_slots_skips_non_resident_slots() {
        let mut pool = CutPool::new(8, 1, 8, 0);
        for forward_pass_index in 0..4 {
            pool.add_cut(NodeId(0), 0, forward_pass_index, 0.0, &[0.0]);
        }
        let mut row_map = CutRowMap::new(8, 2);
        row_map.insert(3);
        row_map.insert(1);
        let dual = [9.0, 9.0, 0.5, 0.0];
        let mut out = vec![(99, 99.0)];

        fill_resident_cut_dual_slots(&dual, &row_map, &pool, &mut out);

        assert_eq!(out, [(3, 0.5)]);
    }

    #[test]
    #[should_panic(expected = "index out of bounds")]
    fn resident_cut_dual_pairs_index_the_dual_slice_directly() {
        let mut pool = CutPool::new(8, 1, 8, 0);
        for forward_pass_index in 0..4 {
            pool.add_cut(NodeId(0), 0, forward_pass_index, 0.0, &[0.0]);
        }
        let mut row_map = CutRowMap::new(8, 2);
        row_map.insert(3);
        row_map.insert(1);
        let dual = [9.0, 9.0, 0.5];
        let mut out = Vec::new();

        fill_resident_cut_dual_slots(&dual, &row_map, &pool, &mut out);
    }

    #[test]
    fn frozen_cut_dual_pairs_skip_zero_duals_of_either_sign() {
        let dual = [7.0, 7.0, 0.0, -0.0, 2.5, -1e-12];
        let mut out = vec![(99, 99.0)];

        push_nonzero_cut_row_duals(&dual, 2, &[4, 6, 9, 11], &mut out);

        assert_eq!(out, [(99, 99.0), (9, 2.5), (11, -1e-12)]);
    }

    #[test]
    fn zero_free_pairs_leave_the_binding_counts_unchanged_at_a_zero_threshold() {
        let dense = [(0, 0.0), (1, -0.0), (2, 1e-12), (3, -3.0), (4, 2.0)];
        let zero_free = [(2, 1e-12), (3, -3.0), (4, 2.0)];
        let mut from_dense = vec![0_u64; 6];
        let mut from_zero_free = vec![0_u64; 6];

        count_binding_slots(&dense, 0.0, &mut from_dense, 1);
        count_binding_slots(&zero_free, 0.0, &mut from_zero_free, 1);

        assert_eq!(from_zero_free, [0, 0, 0, 1, 0, 1]);
        assert_eq!(from_dense, from_zero_free);
    }

    #[test]
    fn folded_state_duals_hold_only_the_columned_slots() {
        let global = compact_layout(2, 2, vec![(HydroSys::new(0), 1), (HydroSys::new(1), 1)]);
        let projection = CutStateProjection::new(&global, ALL_ENABLED);
        let lag = projection.lag_point().full();
        assert_eq!(lag.len(), 4);
        assert!(lag.start > 0 && lag.end < projection.n_slots());
        let rc: Vec<f64> = (0..=global.theta).map(|c| c as f64).collect();
        let col_scale = vec![2.0; rc.len()];
        let coupling = empty_coupling(2);
        let mut buffers = Buffers::default();

        buffers.extract(
            &indexed_view(&rc),
            &DualExtractionSpec {
                extracted: &projection,
                col_scale: &col_scale,
                row_scale: &[],
                coupling: &coupling,
                first_cut_row: 0,
                cut_row_slots: &[],
                read_cut_duals: false,
            },
        );

        let non_lag: Vec<usize> = (0..projection.n_slots())
            .filter(|j| !lag.contains(j))
            .collect();
        assert_eq!(buffers.state_duals.len(), non_lag.len());
        for (p, (&dual, &j)) in buffers.state_duals.iter().zip(&non_lag).enumerate() {
            let col = global
                .state_to_lp_incoming_column(projection.global_state_index(CutSlot::new(j)))
                .get();
            assert_eq!(
                dual,
                rc[col] / 2.0,
                "position {p} divides slot {j} by col_scale"
            );
        }
    }

    #[test]
    fn folded_coupled_duals_sum_unscaled_base_row_duals_in_entry_order() {
        let global = compact_layout(2, 1, Vec::new());
        let projection = CutStateProjection::new(&global, STORAGE_ONLY);
        let mut sink = InflowCouplingSink::new(2);
        sink.record(HydroSys::new(0), 3, 0.5);
        sink.record(HydroSys::new(0), 1, -2.0);
        let coupling = sink.finish(&[0.0; 4], &[0.0; 4]);
        let rc = vec![0.0; global.theta + 1];
        let dual = [0.0, 4.0, 0.0, 8.0];
        let row_scale = [1.0, 0.25, 1.0, 3.0];
        let mut buffers = Buffers::default();
        let extract = |buffers: &mut Buffers, row_scale: &[f64]| {
            buffers.extract(
                &view_with_duals(&rc, &dual),
                &DualExtractionSpec {
                    extracted: &projection,
                    col_scale: &[],
                    row_scale,
                    coupling: &coupling,
                    first_cut_row: 4,
                    cut_row_slots: &[],
                    read_cut_duals: false,
                },
            )
        };

        let q = extract(&mut buffers, &row_scale);

        assert_eq!(q.to_bits(), OBJECTIVE.to_bits());
        assert_eq!(buffers.coupled_duals.len(), 2);
        let scaled = (-0.0_f64 + -2.0 * (4.0 * 0.25)) + 0.5 * (8.0 * 3.0);
        assert_eq!(buffers.coupled_duals[0].to_bits(), scaled.to_bits());
        assert_eq!(buffers.coupled_duals[1].to_bits(), (-0.0_f64).to_bits());

        extract(&mut buffers, &[]);

        let raw = (-0.0_f64 + -2.0 * 4.0) + 0.5 * 8.0;
        assert_eq!(buffers.coupled_duals[0].to_bits(), raw.to_bits());
    }

    #[test]
    fn folded_cut_pairs_skip_zero_duals_and_follow_lp_row_order() {
        let global = compact_layout(2, 1, Vec::new());
        let projection = CutStateProjection::new(&global, STORAGE_ONLY);
        let coupling = empty_coupling(2);
        let rc = vec![0.0; global.theta + 1];
        let dual = [7.0, 7.0, 7.0, 7.0, 0.0, -0.0, 1.5, -1e-12];
        let mut buffers = Buffers {
            cut_dual_slots: vec![(99, 99.0)],
            ..Buffers::default()
        };

        let q = buffers.extract(
            &view_with_duals(&rc, &dual),
            &DualExtractionSpec {
                extracted: &projection,
                col_scale: &[],
                row_scale: &[],
                coupling: &coupling,
                first_cut_row: 4,
                cut_row_slots: &[2, 5, 7, 9],
                read_cut_duals: true,
            },
        );

        assert_eq!(q.to_bits(), OBJECTIVE.to_bits());
        assert_eq!(buffers.cut_dual_slots, [(7, 1.5), (9, -1e-12)]);
    }

    #[test]
    fn folded_extraction_reads_no_cut_duals_when_not_asked() {
        let global = compact_layout(2, 1, Vec::new());
        let projection = CutStateProjection::new(&global, STORAGE_ONLY);
        let coupling = empty_coupling(2);
        let rc = vec![0.0; global.theta + 1];
        let dual = [1.5, 2.5];
        let mut buffers = Buffers {
            cut_dual_slots: vec![(99, 99.0)],
            ..Buffers::default()
        };

        buffers.extract(
            &view_with_duals(&rc, &dual),
            &DualExtractionSpec {
                extracted: &projection,
                col_scale: &[],
                row_scale: &[],
                coupling: &coupling,
                first_cut_row: 0,
                cut_row_slots: &[0, 1],
                read_cut_duals: false,
            },
        );

        assert!(buffers.cut_dual_slots.is_empty());
    }

    #[test]
    fn folded_extraction_reuses_its_buffers() {
        let global = compact_layout(2, 2, Vec::new());
        let projection = CutStateProjection::new(&global, ALL_ENABLED);
        let mut sink = InflowCouplingSink::new(2);
        sink.record(HydroSys::new(0), 0, 1.0);
        sink.record(HydroSys::new(1), 1, 2.0);
        let coupling = sink.finish(&[0.0; 2], &[0.0; 2]);
        let rc = vec![1.0; global.theta + 1];
        let dual = [3.0, 4.0, 0.5, 0.0, -2.0];
        let mut buffers = Buffers::default();
        let extract = |buffers: &mut Buffers| {
            buffers.extract(
                &view_with_duals(&rc, &dual),
                &DualExtractionSpec {
                    extracted: &projection,
                    col_scale: &[],
                    row_scale: &[],
                    coupling: &coupling,
                    first_cut_row: 2,
                    cut_row_slots: &[1, 4, 6],
                    read_cut_duals: true,
                },
            );
        };

        extract(&mut buffers);
        assert_eq!(buffers.cut_dual_slots, [(1, 0.5), (6, -2.0)]);
        let before = (
            buffers.state_duals.capacity(),
            buffers.state_duals.as_ptr(),
            buffers.coupled_duals.capacity(),
            buffers.coupled_duals.as_ptr(),
            buffers.cut_dual_slots.capacity(),
            buffers.cut_dual_slots.as_ptr(),
        );

        extract(&mut buffers);

        let after = (
            buffers.state_duals.capacity(),
            buffers.state_duals.as_ptr(),
            buffers.coupled_duals.capacity(),
            buffers.coupled_duals.as_ptr(),
            buffers.cut_dual_slots.capacity(),
            buffers.cut_dual_slots.as_ptr(),
        );
        assert_eq!(before, after);
    }

    /// `x_out` at `x_col`, θ at `theta`: minimise θ over the rows
    /// `θ − β_k·x_out ≥ α_k`, every column in `[0, 10]` except θ.
    fn cut_row_template(
        num_cols: usize,
        x_col: usize,
        theta: usize,
        cuts: &[(f64, f64)],
    ) -> StageTemplate {
        let mut col_starts = vec![0_i32];
        let mut row_indices = Vec::new();
        let mut values = Vec::new();
        for col in 0..num_cols {
            for (row, &(_, beta)) in cuts.iter().enumerate() {
                let value = if col == theta {
                    1.0
                } else if col == x_col && beta != 0.0 {
                    -beta
                } else {
                    continue;
                };
                row_indices.push(i32::try_from(row).expect("row fits i32"));
                values.push(value);
            }
            col_starts.push(i32::try_from(values.len()).expect("nnz fits i32"));
        }
        let mut col_lower = vec![0.0; num_cols];
        let mut col_upper = vec![10.0; num_cols];
        col_lower[theta] = -1.0e6;
        col_upper[theta] = 1.0e6;
        let mut objective = vec![0.0; num_cols];
        objective[theta] = 1.0;
        StageTemplate {
            num_cols,
            num_rows: cuts.len(),
            num_nz: values.len(),
            col_starts,
            row_indices,
            values,
            col_lower,
            col_upper,
            objective,
            row_lower: cuts.iter().map(|&(alpha, _)| alpha).collect(),
            row_upper: vec![f64::INFINITY; cuts.len()],
            n_state: 0,
            col_scale: Vec::new(),
            row_scale: Vec::new(),
        }
    }

    #[test]
    fn folded_pair_count_stays_within_the_structural_bound() {
        let global = compact_layout(1, 1, Vec::new());
        let projection = CutStateProjection::new(&global, ALL_ENABLED);
        assert_eq!(projection.structural_dual_bound(), 2);
        let x_col = global
            .lp_column_for_state(projection.global_state_index(CutSlot::new(0)))
            .get();
        let coupling = empty_coupling(1);
        let non_degenerate = [
            (0.0, -1.0),
            (-4.0, 1.0),
            (-20.0, 2.0),
            (-20.0, -3.0),
            (-50.0, 0.0),
        ];
        let mut degenerate = non_degenerate.to_vec();
        degenerate.push((-2.0, 0.0));

        for (cuts, degenerate) in [(&non_degenerate[..], false), (&degenerate[..], true)] {
            let template = cut_row_template(global.theta + 1, x_col, global.theta, cuts);
            let slots: Vec<usize> = (0..cuts.len()).collect();
            let mut solver = ActiveSolver::new().expect("ActiveSolver::new()");
            solver.load_model(&template);
            let view = solver.solve(None).expect("the cut-row LP is feasible");
            let mut buffers = Buffers::default();

            let q = buffers.extract(
                &view,
                &DualExtractionSpec {
                    extracted: &projection,
                    col_scale: &[],
                    row_scale: &[],
                    coupling: &coupling,
                    first_cut_row: 0,
                    cut_row_slots: &slots,
                    read_cut_duals: true,
                },
            );

            assert!((q - -2.0).abs() < 1e-9, "objective {q}");
            let pairs = buffers.cut_dual_slots.len();
            if degenerate {
                assert!(
                    (1..=projection.structural_dual_bound()).contains(&pairs),
                    "degenerate pair count {pairs}"
                );
            } else {
                assert_eq!(pairs, projection.structural_dual_bound());
                assert_eq!(
                    buffers
                        .cut_dual_slots
                        .iter()
                        .map(|&(slot, _)| slot)
                        .collect::<Vec<_>>(),
                    [0, 1]
                );
            }
        }
    }
}
