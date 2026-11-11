use std::collections::BTreeMap;

use novomodelo_sddp::test_support::expander::expand_stage_templates;

use crate::common::row_families::{RowClass, census, classify_expanded_rows, classify_stage_rows};
use crate::common::{expected_study_count, for_each_study};

const CUT_ROWS: usize = 2;

fn assert_trailing_cuts(context: &str, classes: &[RowClass], base: usize) {
    assert_eq!(classes.len(), base + CUT_ROWS, "{context}: row count");
    assert_eq!(
        classes[base..],
        [RowClass::Cut; CUT_ROWS],
        "{context}: the rows past the structural rows"
    );
}

#[test]
fn the_row_classifier_places_every_row_of_every_study() {
    let mut union: BTreeMap<&'static str, usize> = BTreeMap::new();
    let visited = for_each_study(|key, system, compact| {
        let expanded = expand_stage_templates(system, compact);
        let compact_templates = &compact.inputs.stage_data.stage_templates.templates;
        for (stage, scaled) in expanded.scaled.iter().enumerate() {
            let context = format!("{key} stage {stage}");

            let compact_base = compact_templates[stage].num_rows;
            let compact_classes = classify_stage_rows(compact, stage, compact_base + CUT_ROWS);
            assert_trailing_cuts(&context, &compact_classes, compact_base);
            for (name, rows) in census(&compact_classes) {
                *union.entry(name).or_insert(0) += rows;
            }

            let expanded_classes = classify_expanded_rows(
                compact,
                &expanded.layout,
                stage,
                scaled.num_rows + CUT_ROWS,
            );
            assert_trailing_cuts(
                &format!("{context} expanded"),
                &expanded_classes,
                scaled.num_rows,
            );
        }
    });
    assert_eq!(
        visited,
        expected_study_count(),
        "the sweep visited a different number of studies than there are"
    );

    for family in [
        "water_balance",
        "load_balance",
        "fpha",
        "evaporation",
        "filling_target",
        "generic_constraint",
    ] {
        assert!(
            union.contains_key(family),
            "no study has a {family} row; reached {union:?}"
        );
    }
    assert!(
        [
            "anticipated_fishing",
            "anticipated_state_out_def",
            "anticipated_slot_definition",
        ]
        .iter()
        .any(|family| union.contains_key(family)),
        "no study has an anticipated row; reached {union:?}"
    );
}
