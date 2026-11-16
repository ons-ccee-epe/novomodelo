//! `split_plant_bounds` section tests.
//!
//! Pins spec section 5 item 4 (available generation capacity) against the
//! `d51-split-plant-two-bus` fixture: the resolved per-cell FPHA-generation
//! `col_upper` on the REAL loaded deck, read structurally (no solve), must
//! equal the group-wise fold-then-sum from the deck's own declared group
//! values and plant-axis `hydro_bounds.parquet` override, and must never
//! collapse to the fold-after-sum plant product.

use std::path::Path;

use novomodelo_core::EntityId;
use novomodelo_core::scenario::ScenarioSource;
use novomodelo_sddp::hydro_models::prepare_hydro_models;
use novomodelo_sddp::indexer::{BlockIdx, FphaCellLocal, HydroCell, HydroCellIndex, HydroSys};
use novomodelo_sddp::{StageTemplates, StudyParams, prepare_stochastic};
use tempfile::TempDir;

use super::*;

const D51_DIR: &str = "../../examples/deterministic/d51-split-plant-two-bus";

/// Structural (Tier 3) read of d51's `col_upper` — no solve.
fn build_d51_templates() -> (novomodelo_core::System, StageTemplates) {
    build_d51_templates_from(Path::new(D51_DIR))
}

fn build_d51_templates_from(case_dir: &Path) -> (novomodelo_core::System, StageTemplates) {
    let config_path = case_dir.join("config.json");
    let config = novomodelo_io::parse_config(&config_path).expect("d51 config must parse");
    let system = novomodelo_io::load_case(case_dir).expect("d51 load_case must succeed");

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
    system: &novomodelo_core::System,
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
    // `cell_max_generation` or any other novomodelo-sddp/-io function, so this is
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

/// d51 with group 1 (`H0-B1`, bus 1) declaring no generation capacity.
fn d51_with_dead_bus1_group() -> TempDir {
    let case = TempDir::new().expect("create the d51 copy directory");
    common::copy_toy_case(Path::new(D51_DIR), case.path());
    let hydros_file = case.path().join("system/hydros.json");
    let mut document: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(&hydros_file).expect("read the copy's system/hydros.json"),
    )
    .expect("parse the copy's system/hydros.json");
    document["hydros"][0]["unit_groups"][1]["max_generation_mw"] = serde_json::json!(0.0);
    std::fs::write(&hydros_file, document.to_string())
        .expect("write the copy's system/hydros.json");
    case
}

/// derived: cell 0's `(generation MW, turbine m3/s)` column upper bounds on the
/// dead-bus-1 copy, the fold `min(group, plant)` of group 0 (30 MW, 42 m3/s)
/// against the plant's resolved limits: stage 0 `(min(30, 28), min(42, 32))`
/// under the lowered 28 MW / 32 m3/s, stage 1 `(min(30, 60), min(42, 80))`
/// under the declared 60 MW / 80 m3/s.
const DEAD_COPY_CELL0_BOUNDS: [(f64, f64); 2] = [(28.0, 32.0), (30.0, 42.0)];

/// A cell of a multi-cell FPHA plant that is dead at every block keeps the live
/// cell's structure intact.
///
/// derived: cell 1's generation and turbine columns are `[0, 0]` because group 1
/// has no generation capacity. Cell 1 is dead at every block, so its plane rows
/// are free (`[-inf, +inf]`) and its plane slack is `[0, 0]`; cell 0 is live, so
/// its slack is `[0, +inf)` and its rows are the shipped build's, bit for bit.
/// The shipped build is the non-degeneracy twin: its cell 1 rows are `<= 0`.
#[test]
fn split_plant_dead_cell_freezes_its_columns_and_frees_its_plane_rows() {
    let case = d51_with_dead_bus1_group();
    let (system, dead) = build_d51_templates_from(case.path());
    let (_, shipped) = build_d51_templates();
    let first_cell = HydroCellIndex::build(system.hydros())
        .cells_of(HydroSys::new(0))
        .start;

    assert_eq!(dead.templates.len(), DEAD_COPY_CELL0_BOUNDS.len());
    assert_eq!(shipped.templates.len(), DEAD_COPY_CELL0_BOUNDS.len());

    for (stage_idx, (expected_generation, expected_turbine)) in
        DEAD_COPY_CELL0_BOUNDS.into_iter().enumerate()
    {
        let n_blks = dead.block_hours_per_stage[stage_idx].len();
        let geometry = &dead.geometry_per_stage[stage_idx];
        let shipped_geometry = &shipped.geometry_per_stage[stage_idx];
        let template = &dead.templates[stage_idx];
        let shipped_template = &shipped.templates[stage_idx];

        assert_eq!(
            geometry.fpha_hydro_indices,
            [HydroSys::new(0)],
            "stage {stage_idx}: the live cell keeps the plant FPHA"
        );
        assert_eq!(
            geometry.fpha.len(),
            shipped_geometry.fpha.len(),
            "stage {stage_idx}: the dead cell keeps its plane rows, free"
        );
        assert_eq!(
            geometry.fpha_slack.len(),
            2 * n_blks,
            "stage {stage_idx}: one plane slack per cell per block"
        );
        assert_eq!(
            geometry.fpha.len() % (2 * n_blks),
            0,
            "stage {stage_idx}: plane rows must split evenly over 2 cells x {n_blks} blocks"
        );
        let n_planes = geometry.fpha.len() / (2 * n_blks);
        assert!(n_planes > 0, "stage {stage_idx}: no plane rows to check");

        let grid = BlockGrid::new(n_blks, 1);
        let cell0_base = geometry.fpha.start;
        let cell1_base = grid.advance_fpha_base(cell0_base, n_planes);
        let shipped_cell0_base = shipped_geometry.fpha.start;
        let shipped_cell1_base = grid.advance_fpha_base(shipped_cell0_base, n_planes);

        for blk in (0..n_blks).map(BlockIdx::new) {
            let (generation0, generation1) =
                cell_generation_bounds_at(&system, &dead, stage_idx, blk.get());
            let turbine0 =
                template.col_upper[geometry.turbine_col(HydroCell::new(first_cell), blk)];
            let turbine1 =
                template.col_upper[geometry.turbine_col(HydroCell::new(first_cell + 1), blk)];
            for (name, got, expected) in [
                ("cell 0 generation", generation0, expected_generation),
                ("cell 0 turbine", turbine0, expected_turbine),
                ("cell 1 generation", generation1, 0.0),
                ("cell 1 turbine", turbine1, 0.0),
            ] {
                assert_eq!(
                    got.to_bits(),
                    expected.to_bits(),
                    "stage {stage_idx} block {}: {name} col_upper = {got}, expected {expected}",
                    blk.get()
                );
            }

            for (cell, expected_upper) in [(0, f64::INFINITY), (1, 0.0)] {
                let col = geometry.fpha_slack_col(FphaCellLocal::new(cell), blk);
                assert_eq!(
                    (template.col_lower[col], template.col_upper[col]),
                    (0.0, expected_upper),
                    "stage {stage_idx} block {}: cell {cell} plane-slack column",
                    blk.get()
                );
            }

            for p_idx in 0..n_planes {
                let row = |base| grid.fpha_plane(base, blk, p_idx, n_planes);
                let at = format!("stage {stage_idx} block {} plane {p_idx}", blk.get());

                let dead_cell1 = row(cell1_base);
                assert_eq!(
                    (
                        template.row_lower[dead_cell1],
                        template.row_upper[dead_cell1]
                    ),
                    (f64::NEG_INFINITY, f64::INFINITY),
                    "{at}: the dead cell's plane row must be free"
                );

                let live_cell0 = row(cell0_base);
                let shipped_cell0 = row(shipped_cell0_base);
                assert_eq!(
                    template.row_upper[live_cell0].to_bits(),
                    shipped_template.row_upper[shipped_cell0].to_bits(),
                    "{at}: the live cell's plane row must match the shipped build's"
                );

                let shipped_cell1 = row(shipped_cell1_base);
                assert_eq!(
                    shipped_template.row_upper[shipped_cell1], 0.0,
                    "{at}: the shipped build's cell 1 row is bounded (non-degeneracy twin)"
                );
            }
        }
    }
}

/// derived: the copy's cells sum to `28 + 0` MW at stage 0, which no longer
/// exceeds the plant's lowered 28 MW (and `32 + 0` m3/s no longer exceeds 32), so
/// `plant_cap_rhs` leaves the plant uncapped; the shipped cells sum to
/// `28 + 20` MW and `32 + 28` m3/s.
#[test]
fn split_plant_dead_cell_drops_the_binding_cap_row() {
    let case = d51_with_dead_bus1_group();
    let (_, dead) = build_d51_templates_from(case.path());
    let (_, shipped) = build_d51_templates();

    assert_eq!(
        shipped.geometry_per_stage[0].plant_cap_hydro_indices,
        [HydroSys::new(0)],
        "the shipped deck needs its plant cap row at stage 0"
    );
    assert!(
        dead.geometry_per_stage[0]
            .plant_cap_hydro_indices
            .is_empty(),
        "the dead cell's zero capacity leaves no binding plant cap row at stage 0"
    );
}
