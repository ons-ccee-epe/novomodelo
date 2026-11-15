use novomodelo_sddp::indexer::HydroSys;
use novomodelo_sddp::test_support::expander::expand_stage_templates;
use novomodelo_sddp::test_support::template_structure::column_entries;

use super::common::in_code_studies::{FPHA_LAG_STUDY_KEY, structural_studies};
use super::common::{expected_study_count, for_each_study};

fn bit_equal(a: f64, b: f64) -> bool {
    a.to_bits() == b.to_bits()
}

#[test]
fn expanded_scaling_follows_the_compact_template() {
    let mut saw_fpha_lag = false;
    let mut saw_scaled_z = false;
    let mut saw_scaled_lag = false;
    let mut saw_rescaled_touched_row = false;
    let visited = for_each_study(|key, system, compact| {
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
        let is_fpha_lag = key == FPHA_LAG_STUDY_KEY;
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
    assert!(
        saw_fpha_lag,
        "the {FPHA_LAG_STUDY_KEY} study is not visited"
    );
    assert!(
        saw_scaled_z,
        "no {FPHA_LAG_STUDY_KEY} stage has a z column with a non-unit scale"
    );
    assert!(
        saw_scaled_lag,
        "no {FPHA_LAG_STUDY_KEY} stage has a lag column with a non-unit scale"
    );
    assert!(
        saw_rescaled_touched_row,
        "no {FPHA_LAG_STUDY_KEY} stage has a coupled row scaled differently from the compact row"
    );
}

/// derived: σ's contract is `[0, ∞)` with `−1` in each plane row of its cell and block, so each
/// expanded σ column holds those bounds and the compact column's price and entries, rows mapped.
#[test]
fn expanded_template_carries_the_fpha_slack_column() {
    let (key, system, compact) = structural_studies()
        .into_iter()
        .find(|(key, _, _)| key == FPHA_LAG_STUDY_KEY)
        .unwrap_or_else(|| panic!("the {FPHA_LAG_STUDY_KEY} study is registered"));
    let expanded = expand_stage_templates(&system, &compact);
    let layout = &expanded.layout;
    let pre_scaling = compact
        .inputs
        .stage_data
        .stage_templates
        .pre_scaling_templates();
    let ctx = compact.stage_ctx();
    assert_eq!(
        expanded.unscaled.len(),
        pre_scaling.len(),
        "{key}: unscaled template count"
    );

    for (stage, unscaled) in expanded.unscaled.iter().enumerate() {
        let compact_template = &pre_scaling[stage];
        let slack = ctx.geometry_per_stage[stage].fpha_slack.clone();
        assert!(
            !slack.is_empty(),
            "{key} stage {stage}: the FPHA slack family is empty"
        );
        for c in slack {
            let e = layout.col(c);
            assert_eq!(
                unscaled.col_lower[e], 0.0,
                "{key} stage {stage}: col_lower of FPHA slack column {c}"
            );
            assert_eq!(
                unscaled.col_upper[e],
                f64::INFINITY,
                "{key} stage {stage}: col_upper of FPHA slack column {c}"
            );
            assert!(
                bit_equal(unscaled.objective[e], compact_template.objective[c]),
                "{key} stage {stage}: objective of FPHA slack column {c}: expanded {:?}, compact \
                 {:?}",
                unscaled.objective[e],
                compact_template.objective[c]
            );
            let want: Vec<(usize, f64)> = column_entries(compact_template, c)
                .map(|(r, v)| (layout.row(r), v))
                .collect();
            assert!(
                !want.is_empty() && want.iter().all(|&(_, v)| v == -1.0),
                "{key} stage {stage}: FPHA slack column {c} must carry −1 in each of its plane \
                 rows, got {want:?}"
            );
            let got: Vec<(usize, f64)> = column_entries(unscaled, e).collect();
            assert_eq!(
                got, want,
                "{key} stage {stage}: entries of FPHA slack column {c} (expanded rows)"
            );
        }
    }
}
