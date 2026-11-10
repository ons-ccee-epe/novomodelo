//! `compact_form` section tests.

use super::*;
use common::compact_map::{
    CompactMap, assert_compact_generic_rows_match, assert_compact_geometry_matches,
    assert_compact_scales_match, assert_compact_template_matches,
};
use common::row_families::{RowClass, census, classify_stage_rows, name};
use common::{
    Emission, expected_study_count, for_each_study_in_both_forms, fresh_system_and_setup_as,
};
use novomodelo_sddp::lp::builder::RowFamily;
use novomodelo_sddp::test_support::{
    assert_all_templates_byte_identical, bits, stage_state_box_bounds, template_fact_groups,
};
use std::collections::BTreeSet;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::Path;

fn nested_bits(rows: &[Vec<f64>]) -> Vec<Vec<u64>> {
    rows.iter().map(|row| bits(row)).collect()
}

#[test]
fn compact_emission_of_a_hydro_free_deck_equals_the_logical_template() {
    let case_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../examples/deterministic/d01-thermal-dispatch");
    let (system, logical) = fresh_system_and_setup_as(&case_dir, Emission::Logical);
    let (_, compact) = fresh_system_and_setup_as(&case_dir, Emission::Compact);

    assert!(system.hydros().is_empty(), "fixture sanity: no hydro");
    assert_all_templates_byte_identical(
        &logical.inputs.stage_data.stage_templates.templates,
        &compact.inputs.stage_data.stage_templates.templates,
        "d01-thermal-dispatch",
    );
    assert_eq!(
        template_fact_groups(&logical),
        template_fact_groups(&compact),
        "template fact groups"
    );
}

#[test]
fn every_study_compacts_to_its_logical_template() {
    let mut saw_commit_out = false;
    let mut saw_par_lags = false;
    let visited = for_each_study_in_both_forms(|key, _, logical, compact| {
        let logical_templates = &logical.inputs.stage_data.stage_templates;
        let compact_templates = &compact.inputs.stage_data.stage_templates;
        assert_eq!(
            logical_templates.templates.len(),
            compact_templates.templates.len(),
            "{key}: template count"
        );
        assert_eq!(
            nested_bits(&logical_templates.block_hours_per_stage),
            nested_bits(&compact_templates.block_hours_per_stage),
            "{key}: block_hours_per_stage"
        );
        assert_eq!(
            logical_templates.cost_scale_factor.to_bits(),
            compact_templates.cost_scale_factor.to_bits(),
            "{key}: cost_scale_factor"
        );
        assert_eq!(
            logical_templates.load_bus_indices, compact_templates.load_bus_indices,
            "{key}: load_bus_indices"
        );
        assert_eq!(
            logical_templates.diversion_upstream, compact_templates.diversion_upstream,
            "{key}: diversion_upstream"
        );
        assert_eq!(
            nested_bits(&logical_templates.hydro_productivities_per_stage),
            nested_bits(&compact_templates.hydro_productivities_per_stage),
            "{key}: hydro_productivities_per_stage"
        );
        let logical_state = logical.stage_state();
        let compact_state = compact.stage_state();
        saw_commit_out |= !logical_state.commit_out.is_empty();
        saw_par_lags |= !logical_state.inflow_lags.is_empty();

        for (stage, (logical_template, compact_template)) in logical_templates
            .templates
            .iter()
            .zip(&compact_templates.templates)
            .enumerate()
        {
            let map = CompactMap::of_stage(
                logical_state,
                compact_state,
                logical_template,
                compact_template,
            );
            assert_compact_scales_match(
                key,
                stage,
                &map,
                logical_template,
                compact_template,
                compact_templates.inflow_coupling_per_stage[stage].touched_rows(),
            );
            assert_compact_template_matches(
                key,
                stage,
                &map,
                logical_state,
                &logical_templates.pre_scaling_templates()[stage],
                &compact_templates.pre_scaling_templates()[stage],
            );
            assert_compact_geometry_matches(
                key,
                stage,
                &map,
                &logical_templates.geometry_per_stage[stage],
                &compact_templates.geometry_per_stage[stage],
            );
            assert_compact_generic_rows_match(
                key,
                stage,
                &map,
                &logical_templates.generic_constraint_row_entries[stage],
                &compact_templates.generic_constraint_row_entries[stage],
            );
            let (logical_lower, logical_upper) = stage_state_box_bounds(logical, stage);
            let (compact_lower, compact_upper) = stage_state_box_bounds(compact, stage);
            assert_eq!(
                bits(&logical_lower),
                bits(&compact_lower),
                "{key} stage {stage}: state box lower"
            );
            assert_eq!(
                bits(&logical_upper),
                bits(&compact_upper),
                "{key} stage {stage}: state box upper"
            );
        }
    });

    assert_eq!(visited, expected_study_count(), "every study is visited");
    assert!(saw_commit_out, "no visited study has a commit_out range");
    assert!(saw_par_lags, "no visited study has PAR lags");
}

#[test]
fn row_families_tile_every_stage_row_of_both_forms() {
    let z_inflow = name(RowClass::Base(RowFamily::ZInflow));
    let mut compact_families = BTreeSet::new();
    let visited = for_each_study_in_both_forms(|key, _, logical, compact| {
        let hydro_count = logical.stage_state().hydro_count;
        let stages = logical.inputs.stage_data.stage_templates.templates.len();
        assert_eq!(
            stages,
            compact.inputs.stage_data.stage_templates.templates.len(),
            "{key}: template count"
        );
        for stage in 0..stages {
            let [mut logical_census, mut compact_census] =
                [("logical", logical), ("compact", compact)].map(|(form, setup)| {
                    let num_rows =
                        setup.inputs.stage_data.stage_templates.templates[stage].num_rows;
                    let classes = catch_unwind(AssertUnwindSafe(|| {
                        classify_stage_rows(setup, stage, num_rows)
                    }))
                    .unwrap_or_else(|payload| {
                        let message = payload
                            .downcast_ref::<String>()
                            .map(String::as_str)
                            .or_else(|| payload.downcast_ref::<&str>().copied())
                            .unwrap_or("non-string panic payload");
                        panic!("{key}, {form} form: {message}")
                    });
                    census(&classes)
                });
            assert_eq!(
                logical_census.remove(z_inflow).unwrap_or(0),
                hydro_count,
                "{key} stage {stage}: logical z_inflow rows"
            );
            assert_eq!(
                compact_census.remove(z_inflow).unwrap_or(0),
                0,
                "{key} stage {stage}: compact z_inflow rows"
            );
            assert_eq!(
                logical_census, compact_census,
                "{key} stage {stage}: row families outside z_inflow differ between forms"
            );
            compact_families.extend(compact_census.into_keys());
        }
    });

    assert_eq!(visited, expected_study_count(), "every study is visited");
    for family in [
        RowFamily::Fpha,
        RowFamily::Evaporation,
        RowFamily::GenericConstraint,
        RowFamily::WaterBalance,
    ] {
        assert!(
            compact_families.contains(name(RowClass::Base(family))),
            "no study reaches {family:?} on the compact form; reached {compact_families:?}"
        );
    }
    assert!(
        [
            RowFamily::AnticipatedFishing,
            RowFamily::AnticipatedStateOutDef,
            RowFamily::AnticipatedSlotDefinition,
        ]
        .into_iter()
        .any(|family| compact_families.contains(name(RowClass::Base(family)))),
        "no study reaches an anticipated family on the compact form; reached {compact_families:?}"
    );
}
