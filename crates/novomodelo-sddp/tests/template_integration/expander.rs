//! `expander` section tests.

use std::fmt::Debug;

use novomodelo_sddp::indexer::{HydroSys, StateDim};
use novomodelo_sddp::setup::NodePos;
use novomodelo_sddp::test_support::expander::{
    ExpandedLayout, expand_stage_templates, expanded_solve_patch,
};
use novomodelo_sddp::test_support::template_structure::z_inflow_column;
use novomodelo_sddp::test_support::{
    capture_patched_node_template_at, oracle_initial_state, stage_state_box_bounds,
};
use novomodelo_solver::StageTemplate;

use super::common::{expected_study_count, for_each_study_in_both_forms};

const FPHA_LAG_STUDY: &str = "structural/fpha-lag";

fn equal(a: i32, b: i32) -> bool {
    a == b
}

fn bit_equal(a: f64, b: f64) -> bool {
    a.to_bits() == b.to_bits()
}

fn assert_same<T: Copy + Debug>(
    context: &str,
    what: &str,
    got: &[T],
    want: &[T],
    same: fn(T, T) -> bool,
    locate: impl Fn(usize) -> String,
) {
    assert_eq!(got.len(), want.len(), "{context}: {what} length");
    if let Some(i) = (0..got.len()).find(|&i| !same(got[i], want[i])) {
        panic!(
            "{context}: {what} at {}: expanded {:?}, today's {:?}",
            locate(i),
            got[i],
            want[i]
        );
    }
}

fn assert_template_matches(
    context: &str,
    got: &StageTemplate,
    want: &StageTemplate,
    compare_scales: bool,
) {
    assert_eq!(got.num_cols, want.num_cols, "{context}: num_cols");
    assert_eq!(got.num_rows, want.num_rows, "{context}: num_rows");
    assert_eq!(got.num_nz, want.num_nz, "{context}: num_nz");
    assert_eq!(got.n_state, want.n_state, "{context}: n_state");

    let column = |c| format!("column {c}");
    let row = |r| format!("row {r}");
    let entry = |k: usize| {
        let c = want.col_starts.partition_point(|&s| (s as usize) <= k) - 1;
        format!("entry {k} of column {c}")
    };
    assert_same(
        context,
        "col_starts",
        &got.col_starts,
        &want.col_starts,
        equal,
        column,
    );
    assert_same(
        context,
        "row_indices",
        &got.row_indices,
        &want.row_indices,
        equal,
        entry,
    );
    assert_same(
        context,
        "values",
        &got.values,
        &want.values,
        bit_equal,
        entry,
    );
    assert_same(
        context,
        "col_lower",
        &got.col_lower,
        &want.col_lower,
        bit_equal,
        column,
    );
    assert_same(
        context,
        "col_upper",
        &got.col_upper,
        &want.col_upper,
        bit_equal,
        column,
    );
    assert_same(
        context,
        "objective",
        &got.objective,
        &want.objective,
        bit_equal,
        column,
    );
    assert_same(
        context,
        "row_lower",
        &got.row_lower,
        &want.row_lower,
        bit_equal,
        row,
    );
    assert_same(
        context,
        "row_upper",
        &got.row_upper,
        &want.row_upper,
        bit_equal,
        row,
    );
    if compare_scales {
        assert_same(
            context,
            "col_scale",
            &got.col_scale,
            &want.col_scale,
            bit_equal,
            column,
        );
        assert_same(
            context,
            "row_scale",
            &got.row_scale,
            &want.row_scale,
            bit_equal,
            row,
        );
    }
}

#[test]
fn expanded_templates_rebuild_todays_emission() {
    let mut saw_second_lag_entry = false;
    let mut saw_commit_out_with_lags = false;
    let visited = for_each_study_in_both_forms(|key, system, logical, compact| {
        let expanded = expand_stage_templates(system, compact);
        let logical_templates = &logical.inputs.stage_data.stage_templates;
        let logical_state = logical.stage_state();
        let n = logical_state.hydro_count;
        assert_eq!(
            expanded.unscaled.len(),
            logical_templates.templates.len(),
            "{key}: unscaled template count"
        );
        assert_eq!(
            expanded.scaled.len(),
            logical_templates.templates.len(),
            "{key}: scaled template count"
        );

        for (stage, today) in logical_templates.pre_scaling_templates().iter().enumerate() {
            assert_template_matches(
                &format!("{key} stage {stage} unscaled"),
                &expanded.unscaled[stage],
                today,
                false,
            );
            assert_template_matches(
                &format!("{key} stage {stage} scaled"),
                &expanded.scaled[stage],
                &logical_templates.templates[stage],
                true,
            );
            saw_second_lag_entry |= logical_state.max_par_order >= 2
                && (0..n).any(|h| {
                    let c = logical_state.inflow_lags.start + n + h;
                    today.col_starts[c + 1] > today.col_starts[c]
                });
            saw_commit_out_with_lags |=
                !logical_state.commit_out.is_empty() && logical_state.max_par_order >= 1;
        }
    });

    assert_eq!(visited, expected_study_count(), "every study is visited");
    assert!(
        saw_second_lag_entry,
        "no visited stage has a lag-1 column carrying an entry"
    );
    assert!(
        saw_commit_out_with_lags,
        "no visited stage has a non-empty commit_out beside inflow lags"
    );
}

#[test]
fn expanded_state_columns_match_todays_state_map() {
    let mut saw_deep_lags = false;
    let visited = for_each_study_in_both_forms(|key, _, logical, compact| {
        let layout = ExpandedLayout::of(compact.stage_state());
        let state = layout.state();
        let logical_state = logical.stage_state();

        for j in 0..logical_state.n_state {
            let dim = StateDim::new(j);
            assert_eq!(
                layout.outgoing_state_col(dim),
                logical_state.lp_column_for_state(dim).get(),
                "{key}: outgoing column of state dimension {j}"
            );
            assert_eq!(
                layout.incoming_state_col(dim),
                logical_state.state_to_lp_incoming_column(dim).get(),
                "{key}: incoming column of state dimension {j}"
            );
        }
        for h in (0..logical_state.hydro_count).map(HydroSys::new) {
            assert_eq!(
                layout.z_col(h),
                z_inflow_column(logical_state, h),
                "{key}: z column of hydro {}",
                h.get()
            );
            assert_eq!(
                layout.z_row(h),
                logical_state.z_inflow_row(h),
                "{key}: z row of hydro {}",
                h.get()
            );
        }

        let ranges = [
            ("storage", &state.storage, &logical_state.storage),
            (
                "inflow_lags",
                &state.inflow_lags,
                &logical_state.inflow_lags,
            ),
            (
                "transit_buckets_out",
                &state.transit_buckets_out,
                &logical_state.transit_buckets_out,
            ),
            ("commit_out", &state.commit_out, &logical_state.commit_out),
            ("z_inflow", &state.z_inflow, &logical_state.z_inflow),
            ("storage_in", &state.storage_in, &logical_state.storage_in),
            (
                "transit_buckets_in",
                &state.transit_buckets_in,
                &logical_state.transit_buckets_in,
            ),
            ("commit_in", &state.commit_in, &logical_state.commit_in),
        ];
        for (name, got, want) in ranges {
            assert_eq!(got, want, "{key}: expanded {name} range");
        }
        assert_eq!(state.theta, logical_state.theta, "{key}: expanded theta");
        assert_eq!(
            state.n_state, logical_state.n_state,
            "{key}: expanded n_state"
        );
        saw_deep_lags |= logical_state.max_par_order >= 2;
    });

    assert_eq!(visited, expected_study_count(), "every study is visited");
    assert!(saw_deep_lags, "no visited study has PAR order 2 or more");
}

#[test]
fn expanded_scaling_follows_the_compact_template() {
    let mut saw_fpha_lag = false;
    let mut saw_scaled_z = false;
    let mut saw_scaled_lag = false;
    let mut saw_rescaled_touched_row = false;
    let visited = for_each_study_in_both_forms(|key, system, _, compact| {
        let expanded = expand_stage_templates(system, compact);
        let layout = &expanded.layout;
        let state = compact.stage_state();
        let (n, l) = (state.hydro_count, state.max_par_order);
        let compact_templates = &compact.inputs.stage_data.stage_templates;
        let coupling = compact.stage_ctx().inflow_coupling;
        assert_eq!(
            expanded.scaled.len(),
            compact_templates.templates.len(),
            "{key}: scaled template count"
        );
        let is_fpha_lag = key == FPHA_LAG_STUDY;
        saw_fpha_lag |= is_fpha_lag;

        for (stage, scaled) in expanded.scaled.iter().enumerate() {
            let compact_template = &compact_templates.templates[stage];
            assert_eq!(
                scaled.num_cols,
                compact_template.num_cols + n * (l + 1),
                "{key} stage {stage}: num_cols"
            );
            assert_eq!(
                scaled.num_rows,
                compact_template.num_rows + n,
                "{key} stage {stage}: num_rows"
            );

            for c in 0..compact_template.num_cols {
                assert!(
                    bit_equal(
                        scaled.col_scale[layout.col(c)],
                        compact_template.col_scale[c]
                    ),
                    "{key} stage {stage}: col_scale of compact column {c}: expanded {:?}, compact {:?}",
                    scaled.col_scale[layout.col(c)],
                    compact_template.col_scale[c]
                );
            }
            let touched = coupling[stage].touched_rows();
            for r in (0..compact_template.num_rows).filter(|&r| !touched.contains(&(r as u32))) {
                assert!(
                    bit_equal(
                        scaled.row_scale[layout.row(r)],
                        compact_template.row_scale[r]
                    ),
                    "{key} stage {stage}: row_scale of compact row {r}: expanded {:?}, compact {:?}",
                    scaled.row_scale[layout.row(r)],
                    compact_template.row_scale[r]
                );
            }

            if is_fpha_lag {
                let hydros = (0..n).map(HydroSys::new);
                saw_scaled_z |= hydros
                    .clone()
                    .any(|h| scaled.col_scale[layout.z_col(h)] != 1.0);
                saw_scaled_lag |= hydros.clone().any(|h| {
                    (0..l).any(|lag| scaled.col_scale[layout.incoming_lag_col(h, lag)] != 1.0)
                });
                saw_rescaled_touched_row |= touched.iter().any(|&r| {
                    let r = r as usize;
                    !bit_equal(
                        scaled.row_scale[layout.row(r)],
                        compact_template.row_scale[r],
                    )
                });
            }
        }
    });

    assert_eq!(visited, expected_study_count(), "every study is visited");
    assert!(saw_fpha_lag, "the {FPHA_LAG_STUDY} study is not visited");
    assert!(
        saw_scaled_z,
        "no {FPHA_LAG_STUDY} stage has a z column with a non-unit scale"
    );
    assert!(
        saw_scaled_lag,
        "no {FPHA_LAG_STUDY} stage has a lag column with a non-unit scale"
    );
    assert!(
        saw_rescaled_touched_row,
        "no {FPHA_LAG_STUDY} stage has a coupled row scaled differently from the compact row"
    );
}

const SAMPLES: usize = 3;

fn assert_both_bounds(context: &str, what: &str, (lower, upper): (f64, f64), expanded: f64) {
    for (side, today) in [("lower", lower), ("upper", upper)] {
        assert!(
            bit_equal(expanded, today),
            "{context}: {what} {side} bound: expanded {:#018x} ({expanded:?}), today's {:#018x} ({today:?})",
            expanded.to_bits(),
            today.to_bits()
        );
    }
}

fn all_distinct(values: [f64; SAMPLES]) -> bool {
    values
        .iter()
        .enumerate()
        .all(|(i, a)| values[..i].iter().all(|b| !bit_equal(*a, *b)))
}

#[test]
fn expanded_solve_patch_matches_todays_patched_template() {
    let mut saw_distinct_z = false;
    let mut saw_fpha_lag = false;
    let mut saw_lag_scale_split = false;
    let visited = for_each_study_in_both_forms(|key, system, logical, compact| {
        let logical_state = logical.stage_state();
        let n = logical_state.hydro_count;
        let n_state = logical_state.n_state;
        if n == 0 {
            return;
        }
        let expanded = expand_stage_templates(system, compact);
        let hydro_range = logical.inputs.stochastic.class_dimensions().hydro_range();
        let is_lag: Vec<bool> = (0..n_state)
            .map(|j| {
                let col = logical_state
                    .state_to_lp_incoming_column(StateDim::new(j))
                    .get();
                logical_state.inflow_lags.contains(&col)
            })
            .collect();
        let is_fpha_lag = key == FPHA_LAG_STUDY;
        saw_fpha_lag |= is_fpha_lag;

        for stage in 0..expanded.scaled.len() {
            let scaled = &expanded.scaled[stage];
            let producer_box = (stage > 0).then(|| stage_state_box_bounds(logical, stage - 1));
            let mut z_by_sample = Vec::with_capacity(SAMPLES);

            for k in 0..SAMPLES {
                let context = format!("{key} stage {stage} sample {k}");
                let eta: Vec<f64> = (0..n)
                    .map(|h| [-1.5, 0.0, 0.8 + 0.1 * h as f64][k])
                    .collect();
                let mut x = oracle_initial_state(logical);
                for (j, value) in x.iter_mut().enumerate() {
                    if is_lag[j] {
                        *value *= [0.5, 1.0, 1.7][k];
                    } else if let Some((lower, upper)) = &producer_box {
                        *value = value.clamp(lower[j], upper[j]);
                    }
                }
                let mut raw = vec![0.0; logical.inputs.stochastic.dim()];
                raw[hydro_range.clone()].copy_from_slice(&eta);

                let want = capture_patched_node_template_at(logical, NodePos(stage), &raw, &x);
                let got = expanded_solve_patch(compact, &expanded, stage, &eta, &x);

                assert_eq!(got.z_rows.len(), n, "{context}: z row count");
                assert_eq!(got.incoming_cols.len(), n_state, "{context}: pin count");
                for &(row, value) in &got.z_rows {
                    assert_both_bounds(
                        &context,
                        &format!("z row {row}"),
                        (want.row_lower[row], want.row_upper[row]),
                        value,
                    );
                }
                for (j, &(col, value)) in got.incoming_cols.iter().enumerate() {
                    assert_eq!(
                        col,
                        logical_state
                            .state_to_lp_incoming_column(StateDim::new(j))
                            .get(),
                        "{context}: incoming column of state dimension {j}"
                    );
                    assert_both_bounds(
                        &context,
                        &format!("incoming column {col}"),
                        (want.col_lower[col], want.col_upper[col]),
                        value,
                    );
                }
                z_by_sample.push(got.z_rows);
            }

            saw_distinct_z |= (0..n).any(|h| {
                let z = |k: usize| z_by_sample[k][h].1;
                all_distinct([z(0), z(1), z(2)])
            });
            if is_fpha_lag {
                let layout = &expanded.layout;
                saw_lag_scale_split |= (0..n_state).filter(|&j| is_lag[j]).any(|j| {
                    let dim = StateDim::new(j);
                    !bit_equal(
                        scaled.col_scale[layout.incoming_state_col(dim)],
                        scaled.col_scale[layout.outgoing_state_col(dim)],
                    )
                });
            }
        }
    });

    assert_eq!(visited, expected_study_count(), "every study is visited");
    assert!(
        saw_distinct_z,
        "no study, stage and hydro has three distinct z-row values across the noise samples (needs a PAR model with a non-zero sigma)"
    );
    assert!(saw_fpha_lag, "the {FPHA_LAG_STUDY} study is not visited");
    assert!(
        saw_lag_scale_split,
        "no {FPHA_LAG_STUDY} stage has a lag dimension whose incoming and outgoing columns scale differently"
    );
}
