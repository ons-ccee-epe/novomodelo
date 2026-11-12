use novomodelo_sddp::indexer::HydroSys;
use novomodelo_sddp::test_support::expander::expand_stage_templates;

use super::common::in_code_studies::FPHA_LAG_STUDY_KEY;
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
