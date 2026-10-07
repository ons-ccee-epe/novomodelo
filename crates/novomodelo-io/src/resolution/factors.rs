//! Resolution of parsed load- and NCS-factor entries into dense lookup tables.
//!
//! One generic routine builds both [`ResolvedLoadFactors`] and [`ResolvedNcsFactors`]
//! tables, indexed by `(entity_index, stage_index, block_index)` for O(1) lookup during
//! LP construction. Unknown entity/stage IDs are silently skipped (already caught
//! upstream); the default factor is `1.0` (no scaling).

use std::collections::HashMap;

use cobre_core::{Bus, NonControllableSource, ResolvedLoadFactors, ResolvedNcsFactors, Stage};

use crate::StageIdResolver;
use crate::scenarios::{BlockFactor, LoadFactorEntry, NcsFactorEntry};

/// Which id an entry/entity carries, and which dense table receives resolved values.
trait FactorKind {
    /// The sparse parsed entry this kind resolves.
    type Entry;
    /// The parsed entity slice indexed by position to build the id→index map.
    type Entity;
    /// The dense resolved table this kind fills.
    type Table;

    fn entry_entity_id(entry: &Self::Entry) -> i32;
    fn entry_stage_id(entry: &Self::Entry) -> i32;
    fn entry_block_factors(entry: &Self::Entry) -> &[BlockFactor];
    fn entity_id(entity: &Self::Entity) -> i32;
    fn empty_table() -> Self::Table;
    fn new_table(n_entities: usize, n_stages: usize, max_blocks: usize) -> Self::Table;
    fn set(
        table: &mut Self::Table,
        entity_idx: usize,
        stage_idx: usize,
        block_idx: usize,
        factor: f64,
    );
}

struct LoadFactors;

impl FactorKind for LoadFactors {
    type Entry = LoadFactorEntry;
    type Entity = Bus;
    type Table = ResolvedLoadFactors;

    fn entry_entity_id(entry: &Self::Entry) -> i32 {
        entry.bus_id.0
    }

    fn entry_stage_id(entry: &Self::Entry) -> i32 {
        entry.stage_id
    }

    fn entry_block_factors(entry: &Self::Entry) -> &[BlockFactor] {
        &entry.block_factors
    }

    fn entity_id(entity: &Self::Entity) -> i32 {
        entity.id.0
    }

    fn empty_table() -> Self::Table {
        ResolvedLoadFactors::empty()
    }

    fn new_table(n_entities: usize, n_stages: usize, max_blocks: usize) -> Self::Table {
        ResolvedLoadFactors::new(n_entities, n_stages, max_blocks)
    }

    fn set(
        table: &mut Self::Table,
        entity_idx: usize,
        stage_idx: usize,
        block_idx: usize,
        factor: f64,
    ) {
        table.set(entity_idx, stage_idx, block_idx, factor);
    }
}

struct NcsFactors;

impl FactorKind for NcsFactors {
    type Entry = NcsFactorEntry;
    type Entity = NonControllableSource;
    type Table = ResolvedNcsFactors;

    fn entry_entity_id(entry: &Self::Entry) -> i32 {
        entry.ncs_id.0
    }

    fn entry_stage_id(entry: &Self::Entry) -> i32 {
        entry.stage_id
    }

    fn entry_block_factors(entry: &Self::Entry) -> &[BlockFactor] {
        &entry.block_factors
    }

    fn entity_id(entity: &Self::Entity) -> i32 {
        entity.id.0
    }

    fn empty_table() -> Self::Table {
        ResolvedNcsFactors::empty()
    }

    fn new_table(n_entities: usize, n_stages: usize, max_blocks: usize) -> Self::Table {
        ResolvedNcsFactors::new(n_entities, n_stages, max_blocks)
    }

    fn set(
        table: &mut Self::Table,
        entity_idx: usize,
        stage_idx: usize,
        block_idx: usize,
        factor: f64,
    ) {
        table.set(entity_idx, stage_idx, block_idx, factor);
    }
}

/// Builds a dense factor table from sparse parsed entries, shared by both axes.
fn resolve_factors<K: FactorKind>(
    entries: &[K::Entry],
    entities: &[K::Entity],
    stages: &[Stage],
) -> K::Table {
    if entries.is_empty() || entities.is_empty() || stages.is_empty() {
        return K::empty_table();
    }

    let id_to_idx: HashMap<i32, usize> = entities
        .iter()
        .enumerate()
        .map(|(idx, entity)| (K::entity_id(entity), idx))
        .collect();

    let study_stage_ids: Vec<i32> = stages.iter().filter(|s| s.id >= 0).map(|s| s.id).collect();
    let stage_resolver = StageIdResolver::from_study_stage_ids(&study_stage_ids);
    let stage_id_to_idx = stage_resolver.index_map();

    let n_entities = entities.len();
    let n_stages = stage_id_to_idx.len();
    let max_blocks = stages
        .iter()
        .filter(|s| s.id >= 0)
        .map(|s| s.blocks.len())
        .max()
        .unwrap_or(0);

    let mut table = K::new_table(n_entities, n_stages, max_blocks);

    for entry in entries {
        let Some(&entity_idx) = id_to_idx.get(&K::entry_entity_id(entry)) else {
            continue;
        };
        let Some(&stage_idx) = stage_id_to_idx.get(&K::entry_stage_id(entry)) else {
            continue;
        };
        for bf in K::entry_block_factors(entry) {
            let Ok(block_idx) = usize::try_from(bf.block_id) else {
                continue;
            };
            if block_idx < max_blocks {
                K::set(&mut table, entity_idx, stage_idx, block_idx, bf.factor);
            }
        }
    }

    table
}

/// Build a resolved load factor table from parsed entries.
///
/// `buses` and `stages` must each be in the order
/// [`SystemBuilder::build`](cobre_core::SystemBuilder::build) establishes;
/// slice position becomes the entity index. The stage axis spans study
/// stages only (`id >= 0`); the block axis is sized to the largest
/// per-stage block count.
#[must_use]
pub fn resolve_load_factors(
    entries: &[LoadFactorEntry],
    buses: &[Bus],
    stages: &[Stage],
) -> ResolvedLoadFactors {
    resolve_factors::<LoadFactors>(entries, buses, stages)
}

/// Build a resolved NCS factor table from parsed entries.
///
/// `non_controllable_sources` and `stages` must each be in the order
/// [`SystemBuilder::build`](cobre_core::SystemBuilder::build) establishes;
/// slice position becomes the entity index. The stage axis spans study
/// stages only (`id >= 0`); the block axis is sized to the largest
/// per-stage block count.
#[must_use]
pub fn resolve_ncs_factors(
    entries: &[NcsFactorEntry],
    non_controllable_sources: &[NonControllableSource],
    stages: &[Stage],
) -> ResolvedNcsFactors {
    resolve_factors::<NcsFactors>(entries, non_controllable_sources, stages)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::float_cmp)]
mod tests {
    use super::*;
    use chrono::NaiveDate;
    use cobre_core::EntityId;
    use cobre_core::temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, StageRiskConfig, StageStateConfig,
    };

    trait FactorFixtures: FactorKind {
        fn make_entity(id: i32) -> Self::Entity;
        fn make_entry(id: i32, stage_id: i32, factors: &[(i32, f64)]) -> Self::Entry;
        fn factor(
            table: &Self::Table,
            entity_idx: usize,
            stage_idx: usize,
            block_idx: usize,
        ) -> f64;
    }

    impl FactorFixtures for LoadFactors {
        fn make_entity(id: i32) -> Bus {
            Bus {
                id: EntityId(id),
                name: format!("B{id}"),
                operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                deficit_segments: vec![],
                excess_cost: 0.0,
            }
        }

        fn make_entry(id: i32, stage_id: i32, factors: &[(i32, f64)]) -> LoadFactorEntry {
            LoadFactorEntry {
                bus_id: EntityId(id),
                stage_id,
                block_factors: factors
                    .iter()
                    .map(|&(block_id, factor)| BlockFactor { block_id, factor })
                    .collect(),
            }
        }

        fn factor(
            table: &ResolvedLoadFactors,
            entity_idx: usize,
            stage_idx: usize,
            block_idx: usize,
        ) -> f64 {
            table.factor(entity_idx, stage_idx, block_idx)
        }
    }

    impl FactorFixtures for NcsFactors {
        fn make_entity(id: i32) -> NonControllableSource {
            NonControllableSource {
                id: EntityId(id),
                name: format!("NCS{id}"),
                operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                bus_id: EntityId(0),
                entry_stage_id: None,
                exit_stage_id: None,
                max_generation_mw: 100.0,
                allow_curtailment: true,
                curtailment_cost: 5.0,
            }
        }

        fn make_entry(id: i32, stage_id: i32, factors: &[(i32, f64)]) -> NcsFactorEntry {
            NcsFactorEntry {
                ncs_id: EntityId(id),
                stage_id,
                block_factors: factors
                    .iter()
                    .map(|&(block_id, factor)| BlockFactor { block_id, factor })
                    .collect(),
            }
        }

        fn factor(
            table: &ResolvedNcsFactors,
            entity_idx: usize,
            stage_idx: usize,
            block_idx: usize,
        ) -> f64 {
            table.factor(entity_idx, stage_idx, block_idx)
        }
    }

    fn make_stage(id: i32, n_blocks: usize) -> Stage {
        Stage {
            index: 0,
            id,
            start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            end_date: chrono::NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
            season_id: None,
            blocks: (0..n_blocks)
                .map(|b| Block {
                    index: b,
                    name: format!("B{b}"),
                    duration_hours: 100.0,
                })
                .collect(),
            block_mode: BlockMode::Parallel,
            state_config: StageStateConfig {
                storage: true,
                inflow_lags: false,
            },
            risk_config: StageRiskConfig::Expectation,
            scenario_config: ScenarioSourceConfig {
                branching_factor: 1,
                noise_method: NoiseMethod::Saa,
            },
        }
    }

    fn scenario_empty_entries_returns_empty<K: FactorFixtures>() {
        let entities = vec![K::make_entity(0)];
        let stages = vec![make_stage(0, 2)];
        let table = resolve_factors::<K>(&[], &entities, &stages);
        assert!((K::factor(&table, 0, 0, 0) - 1.0).abs() < f64::EPSILON);
    }

    fn scenario_basic_resolution<K: FactorFixtures>(first: f64, second: f64) {
        let entities = vec![K::make_entity(0), K::make_entity(1)];
        let stages = vec![make_stage(0, 3)];
        let entries = vec![K::make_entry(0, 0, &[(0, first), (1, second)])];

        let table = resolve_factors::<K>(&entries, &entities, &stages);
        assert!((K::factor(&table, 0, 0, 0) - first).abs() < 1e-10);
        assert!((K::factor(&table, 0, 0, 1) - second).abs() < 1e-10);
        assert!((K::factor(&table, 0, 0, 2) - 1.0).abs() < f64::EPSILON);
        assert!((K::factor(&table, 1, 0, 0) - 1.0).abs() < f64::EPSILON);
    }

    fn scenario_unknown_entity_id_skipped<K: FactorFixtures>() {
        let entities = vec![K::make_entity(0)];
        let stages = vec![make_stage(0, 1)];
        let entries = vec![K::make_entry(99, 0, &[(0, 0.5)])];

        let table = resolve_factors::<K>(&entries, &entities, &stages);
        assert!((K::factor(&table, 0, 0, 0) - 1.0).abs() < f64::EPSILON);
    }

    fn scenario_unknown_stage_id_skipped<K: FactorFixtures>() {
        let entities = vec![K::make_entity(0)];
        let stages = vec![make_stage(0, 1)];
        let entries = vec![K::make_entry(0, 99, &[(0, 0.5)])];

        let table = resolve_factors::<K>(&entries, &entities, &stages);
        assert!((K::factor(&table, 0, 0, 0) - 1.0).abs() < f64::EPSILON);
    }

    fn scenario_pre_study_stages_excluded<K: FactorFixtures>(value: f64) {
        let entities = vec![K::make_entity(0)];
        let stages = vec![make_stage(-1, 1), make_stage(0, 2)];
        let entries = vec![K::make_entry(0, 0, &[(0, value)])];

        let table = resolve_factors::<K>(&entries, &entities, &stages);
        assert!((K::factor(&table, 0, 0, 0) - value).abs() < 1e-10);
    }

    #[test]
    fn test_load_empty_entries_returns_empty() {
        scenario_empty_entries_returns_empty::<LoadFactors>();
    }

    #[test]
    fn test_load_basic_resolution() {
        scenario_basic_resolution::<LoadFactors>(0.85, 1.15);
    }

    #[test]
    fn test_load_unknown_entity_id_skipped() {
        scenario_unknown_entity_id_skipped::<LoadFactors>();
    }

    #[test]
    fn test_load_unknown_stage_id_skipped() {
        scenario_unknown_stage_id_skipped::<LoadFactors>();
    }

    #[test]
    fn test_load_pre_study_stages_excluded() {
        scenario_pre_study_stages_excluded::<LoadFactors>(0.9);
    }

    #[test]
    fn test_ncs_empty_entries_returns_empty() {
        scenario_empty_entries_returns_empty::<NcsFactors>();
    }

    #[test]
    fn test_ncs_basic_resolution() {
        scenario_basic_resolution::<NcsFactors>(0.6, 0.8);
    }

    #[test]
    fn test_ncs_unknown_entity_id_skipped() {
        scenario_unknown_entity_id_skipped::<NcsFactors>();
    }

    #[test]
    fn test_ncs_unknown_stage_id_skipped() {
        scenario_unknown_stage_id_skipped::<NcsFactors>();
    }

    #[test]
    fn test_ncs_pre_study_stages_excluded() {
        scenario_pre_study_stages_excluded::<NcsFactors>(0.7);
    }
}
