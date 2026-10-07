//! `split_plant_bounds` section tests.
//!
//! Pins spec section 5 item 4 (available generation capacity) against the
//! `d51-split-plant-two-bus` fixture: the resolved per-cell FPHA-generation
//! `col_upper` on the REAL loaded deck, read structurally (no solve), must
//! equal the group-wise fold-then-sum from the deck's own declared group
//! values and plant-axis `hydro_bounds.parquet` override, and must never
//! collapse to the fold-after-sum plant product.

use std::path::Path;

use cobre_core::EntityId;
use cobre_core::scenario::ScenarioSource;
use cobre_sddp::hydro_models::prepare_hydro_models;
use cobre_sddp::indexer::{BlockIdx, HydroCell, HydroCellIndex, HydroSys};
use cobre_sddp::{StageTemplates, StudyParams, prepare_stochastic};

use super::*;

/// Structural (Tier 3) read of d51's `col_upper` — no solve.
fn build_d51_templates() -> (cobre_core::System, StageTemplates) {
    let case_dir = Path::new("../../examples/deterministic/d51-split-plant-two-bus");
    let config_path = case_dir.join("config.json");
    let config = cobre_io::parse_config(&config_path).expect("d51 config must parse");
    let system = cobre_io::load_case(case_dir).expect("d51 load_case must succeed");

    let prepare_result = prepare_stochastic(
        system,
        case_dir,
        &config,
        42,
        &ScenarioSource::default(),
        None,
    )
    .expect("prepare_stochastic must succeed for d51");
    let system = prepare_result.system;
    let stochastic = prepare_result.stochastic;

    let hydro_models = prepare_hydro_models(&system, case_dir, false)
        .expect("prepare_hydro_models must succeed for d51");
    let params = StudyParams::from_config(&config, Vec::new())
        .expect("StudyParams::from_config must succeed");

    let templates = build_stage_templates_resolving_layout(
        &system,
        params.inflow_method,
        stochastic.par(),
        stochastic.normal(),
        &hydro_models.production,
        &hydro_models.evaporation,
        &ResolvedParameters::default(),
    )
    .expect("build_stage_templates_resolving_layout must succeed for d51");

    (system, templates)
}

/// Reads `StageTemplate.col_upper` directly, never hand-recomputing
/// `cell_max_generation`'s formula.
fn cell_generation_bounds_at(
    system: &cobre_core::System,
    templates: &StageTemplates,
    stage_idx: usize,
    blk: usize,
) -> (f64, f64) {
    let hydro_cell_index = HydroCellIndex::build(system.hydros());
    let cell_range = hydro_cell_index.cells_of(HydroSys::new(0));
    assert_eq!(
        cell_range.len(),
        2,
        "d51's H0 must partition into exactly 2 cells (one per declared bus)"
    );
    let cell0 = HydroCell::new(cell_range.start);
    let cell1 = HydroCell::new(cell_range.start + 1);
    assert_eq!(
        hydro_cell_index.bus_of(cell0),
        EntityId(0),
        "cell 0 must own bus 0 (HydroCellIndex partitions bus-ascending)"
    );
    assert_eq!(
        hydro_cell_index.bus_of(cell1),
        EntityId(1),
        "cell 1 must own bus 1"
    );

    let geometry = &templates.geometry_per_stage[stage_idx];
    let n_blks = templates.block_hours_per_stage[stage_idx].len();
    let grid = BlockGrid::new(n_blks, 1);
    // H0 is the only FPHA hydro, so its cells occupy FPHA-local generation
    // offsets [0, 2) in the same ascending order `HydroCellIndex::cells_of` yields.
    let col0 = grid.flat(geometry.generation.start, 0, BlockIdx::new(blk));
    let col1 = grid.flat(geometry.generation.start, 1, BlockIdx::new(blk));

    let template = &templates.templates[stage_idx];
    (template.col_upper[col0], template.col_upper[col1])
}

/// Available capacity (structural check): the resolved per-cell column upper bound.
///
/// H0's two groups keep their own declared envelope at every stage (H0-B0:
/// `30.0` MW; H0-B1: `20.0` MW) — no group-axis override exists in this deck.
/// `hydro_bounds.parquet`'s single, plant-axis, LOWERING row drops the
/// plant's resolved envelope to `28.0` MW at stage 0 only, strictly between
/// the two groups' declared values: H0-B0's cell resolves to the PLANT term
/// (`min(30.0, 28.0) == 28.0`), H0-B1's cell resolves to its own GROUP term
/// (`min(20.0, 28.0) == 20.0`). Stage 1 carries no override, so the plant's
/// resolved envelope falls back to its declared `60.0` MW — above both
/// groups — and every cell resolves to its own group term.
#[test]
fn split_plant_ac1_resolved_cell_bound_matches_group_wise_overlay() {
    let (system, templates) = build_d51_templates();

    let cases: [(usize, f64, f64); 2] = [(0, 28.0, 20.0), (1, 30.0, 20.0)];

    for (stage_idx, expected_cell0, expected_cell1) in cases {
        let n_blks = templates.block_hours_per_stage[stage_idx].len();
        for blk in 0..n_blks {
            let (bound0, bound1) = cell_generation_bounds_at(&system, &templates, stage_idx, blk);
            assert_eq!(
                bound0.to_bits(),
                expected_cell0.to_bits(),
                "stage {stage_idx} block {blk}: cell 0 (bus 0) col_upper = {bound0}, \
                 expected {expected_cell0} (H0-B0's resolved bound)"
            );
            assert_eq!(
                bound1.to_bits(),
                expected_cell1.to_bits(),
                "stage {stage_idx} block {blk}: cell 1 (bus 1) col_upper = {bound1}, \
                 expected {expected_cell1} (H0-B1's resolved bound)"
            );
        }
    }
}

/// The plant-level product is unreachable (negative check).
///
/// At stage 0, H0-B0's cell resolves to the plant's lowered envelope
/// (`28.0` MW, the PLANT term of its closing `min`) while H0-B1's cell
/// resolves to its own declared value (`20.0` MW, the GROUP term) — available
/// capacity is the group-wise fold-then-sum of these two PER-CELL resolutions
/// (`28.0 + 20.0 == 48.0` MW), never a single plant-wide fold. A resolver
/// that summed the two groups' raw DECLARED values first (`30.0 + 20.0 ==
/// 50.0`) and folded once against the plant's resolved envelope — the
/// fold-after-sum alternative — would report `50.0.min(28.0) == 28.0`
/// instead: a different, and strictly smaller, number (it collapses to
/// exactly the plant's own resolved envelope, discarding H0-B1's cell
/// entirely).
#[test]
fn split_plant_ac2_available_capacity_is_not_the_fold_after_sum_plant_product() {
    let (system, templates) = build_d51_templates();

    let (cell0_bound, cell1_bound) = cell_generation_bounds_at(&system, &templates, 0, 0);
    let available_capacity = cell0_bound + cell1_bound;

    // Hand-derived from the deck alone (generate_parquet.py's override row and
    // hydros.json's declared group/plant envelopes) -- never routed through
    // `cell_max_generation` or any other cobre-sddp/-io function, so this is
    // an independent check on what a fold-after-sum implementation would give.
    let raw_group_sum_stage0 = 30.0_f64 + 20.0;
    let plant_resolved_envelope_stage0 = 28.0_f64;
    let fold_after_sum_plant_product = raw_group_sum_stage0.min(plant_resolved_envelope_stage0);

    assert_eq!(
        available_capacity.to_bits(),
        48.0_f64.to_bits(),
        "available capacity must be the group-wise fold-then-sum, 48.0 MW"
    );
    assert_eq!(
        fold_after_sum_plant_product.to_bits(),
        28.0_f64.to_bits(),
        "the fold-after-sum alternative must collapse to the plant's resolved envelope, 28.0 MW"
    );
    assert_ne!(
        available_capacity.to_bits(),
        fold_after_sum_plant_product.to_bits(),
        "available capacity ({available_capacity}) must not equal the fold-after-sum \
         plant product ({fold_after_sum_plant_product}) -- a resolver that summed the \
         groups before folding against the plant envelope would silently manufacture a \
         SMALLER, wrong bound (min(Σa, C) <= Σ min(a, C) whenever more than one term binds \
         below C individually while the sum exceeds it)"
    );
}
