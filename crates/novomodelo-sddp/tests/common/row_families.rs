//! The row classes of a loaded stage LP, read from the builder's row-family record: every
//! structural row of the stage or expanded LP gets the family the builder allocated it
//! to, and every row loaded past the structural rows is a cut. Positions come from the record and
//! the expander's index maps only, so a row that no family places fails loudly.

use std::collections::BTreeMap;

use novomodelo_sddp::StudySetup;
use novomodelo_sddp::indexer::HydroSys;
use novomodelo_sddp::lp::builder::RowFamily;
use novomodelo_sddp::test_support::expander::ExpandedLayout;

/// What one loaded row is: a structural row of a family, or an appended cut.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RowClass {
    Base(RowFamily),
    ExpandedInflowDefinition,
    Cut,
}

/// The snake-case name of `class`.
pub fn name(class: RowClass) -> &'static str {
    match class {
        RowClass::ExpandedInflowDefinition => "expanded_inflow_definition",
        RowClass::Base(RowFamily::WaterBalance) => "water_balance",
        RowClass::Base(RowFamily::TransitBucketDefinition) => "transit_bucket_definition",
        RowClass::Base(RowFamily::LoadBalance) => "load_balance",
        RowClass::Base(RowFamily::Fpha) => "fpha",
        RowClass::Base(RowFamily::Evaporation) => "evaporation",
        RowClass::Base(RowFamily::MinOutflow) => "min_outflow_violation",
        RowClass::Base(RowFamily::MaxOutflow) => "max_outflow_violation",
        RowClass::Base(RowFamily::MinTurbine) => "min_turbine_violation",
        RowClass::Base(RowFamily::MinGeneration) => "min_generation_violation",
        RowClass::Base(RowFamily::PlantMinTurbine) => "plant_min_turbine_violation",
        RowClass::Base(RowFamily::PlantMinGeneration) => "plant_min_generation_violation",
        RowClass::Base(RowFamily::PlantMaxTurbine) => "plant_max_turbine",
        RowClass::Base(RowFamily::PlantMaxGeneration) => "plant_max_generation",
        RowClass::Base(RowFamily::FillingTarget) => "filling_target",
        RowClass::Base(RowFamily::FilledMinStorageFloor) => "filled_min_storage_floor",
        RowClass::Base(RowFamily::AnticipatedFishing) => "anticipated_fishing",
        RowClass::Base(RowFamily::AnticipatedStateOutDef) => "anticipated_state_out_def",
        RowClass::Base(RowFamily::AnticipatedSlotDefinition) => "anticipated_slot_definition",
        RowClass::Base(RowFamily::GenericConstraint) => "generic_constraint",
        RowClass::Cut => "cut_row",
    }
}

fn place(classes: &mut [Option<RowClass>], stage: usize, row: usize, class: RowClass) {
    assert!(
        row < classes.len(),
        "stage {stage}: {class:?} places row {row}, outside the {} structural rows",
        classes.len()
    );
    assert!(
        classes[row].is_none(),
        "stage {stage}: row {row} is placed as both {:?} and {class:?}",
        classes[row]
    );
    classes[row] = Some(class);
}

fn with_cut_rows(
    placed: &[Option<RowClass>],
    stage: usize,
    num_rows_loaded: usize,
) -> Vec<RowClass> {
    let mut classes: Vec<RowClass> = placed
        .iter()
        .enumerate()
        .map(|(row, class)| {
            class.unwrap_or_else(|| panic!("stage {stage}: row {row} is in no family"))
        })
        .collect();
    classes.resize(num_rows_loaded, RowClass::Cut);
    classes
}

/// The class of each of the `num_rows_loaded` rows of `stage`'s LP: rows below
/// the stage template's own row count by the builder's record, the rest cuts.
pub fn classify_stage_rows(
    setup: &StudySetup,
    stage: usize,
    num_rows_loaded: usize,
) -> Vec<RowClass> {
    let templates = &setup.inputs.stage_data.stage_templates;
    let base = templates.templates[stage].num_rows;
    assert!(
        num_rows_loaded >= base,
        "stage {stage}: {num_rows_loaded} loaded rows are fewer than the {base} structural rows"
    );
    let mut placed = vec![None; base];
    for (family, range) in &templates.row_families()[stage] {
        for row in range.clone() {
            place(&mut placed, stage, row, RowClass::Base(*family));
        }
    }
    with_cut_rows(&placed, stage, num_rows_loaded)
}

/// The class of each of the `num_rows_loaded` rows of `stage`'s expanded LP: the z rows and the
/// compact rows' classes at their expanded positions, the rest cuts.
pub fn classify_expanded_rows(
    compact: &StudySetup,
    layout: &ExpandedLayout,
    stage: usize,
    num_rows_loaded: usize,
) -> Vec<RowClass> {
    let templates = &compact.inputs.stage_data.stage_templates;
    let n = layout.state().hydro_count;
    let base = templates.templates[stage].num_rows + n;
    assert!(
        num_rows_loaded >= base,
        "stage {stage}: {num_rows_loaded} loaded rows are fewer than the {base} structural rows"
    );
    let mut placed = vec![None; base];
    for h in 0..n {
        place(
            &mut placed,
            stage,
            layout.z_row(HydroSys::new(h)),
            RowClass::ExpandedInflowDefinition,
        );
    }
    for (family, range) in &templates.row_families()[stage] {
        for row in range.clone() {
            place(&mut placed, stage, layout.row(row), RowClass::Base(*family));
        }
    }
    with_cut_rows(&placed, stage, num_rows_loaded)
}

/// How many rows `classes` holds of each class, keyed by [`name`].
pub fn census(classes: &[RowClass]) -> BTreeMap<&'static str, usize> {
    let mut counts = BTreeMap::new();
    for &class in classes {
        *counts.entry(name(class)).or_insert(0) += 1;
    }
    counts
}
