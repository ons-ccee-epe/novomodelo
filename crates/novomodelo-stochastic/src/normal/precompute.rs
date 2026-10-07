//! Pre-computation of normal noise model parameter arrays for LP RHS patching.
//!
//! [`PrecomputedNormal`] is a cache built once from raw parameters, exposing flat
//! arrays in stage-major layout so callers patch LP right-hand sides without
//! per-scenario recomputation.
//!
//! ## Array layout
//!
//! Stage-major keeps all per-stage data contiguous, the access order of sequential
//! stage iteration within a scenario trajectory:
//!
//! - mean, std: `array[stage * n_entities + entity_idx]`.
//! - factors (entity-minor, block-innermost):
//!   `factors[stage * n_entities * max_blocks + entity_idx * max_blocks + block_idx]`.
//!
//! ## Normal noise model
//!
//! `x = μ_{e,m} + σ_{e,m} · f_{e,m,b} · ε`, with `ε ~ N(0, 1)` i.i.d. standard
//! normal, for entity `e`, stage `m`, block `b`.

use std::collections::HashMap;

use cobre_core::{EntityId, scenario::LoadModel, temporal::Stage};

use crate::StochasticError;

// ---------------------------------------------------------------------------
// Helper types
// ---------------------------------------------------------------------------

/// A `(block_id, factor)` pair input to [`PrecomputedNormal::build`]: 0-based block
/// index, multiplicative scaling on the stage-level noise realization.
pub type BlockFactorPair = (i32, f64);

/// An `(entity_id, stage_id, block_factors)` factor entry input to
/// [`PrecomputedNormal::build`].
pub type EntityFactorEntry<'a> = (EntityId, i32, &'a [BlockFactorPair]);

// ---------------------------------------------------------------------------
// PrecomputedNormal
// ---------------------------------------------------------------------------

/// Cache-friendly normal noise model data for LP RHS patching, built once and
/// consumed read-only. Entities are in canonical entity-ID order; see the
/// [module documentation](self) for the array layout and noise model.
///
/// # Examples
///
/// ```
/// use cobre_core::{EntityId, scenario::LoadModel, temporal::{Stage, Block, BlockMode, StageStateConfig, StageRiskConfig, ScenarioSourceConfig, NoiseMethod}};
/// use cobre_stochastic::normal::precompute::PrecomputedNormal;
/// use chrono::NaiveDate;
///
/// let date = NaiveDate::from_ymd_opt(2024, 1, 1).unwrap();
/// let stage = Stage {
///     index: 0,
///     id: 0,
///     start_date: date,
///     end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
///     season_id: Some(0),
///     blocks: vec![Block { index: 0, name: "SINGLE".to_string(), duration_hours: 744.0 }],
///     block_mode: BlockMode::Parallel,
///     state_config: StageStateConfig { storage: true, inflow_lags: false },
///     risk_config: StageRiskConfig::Expectation,
///     scenario_config: ScenarioSourceConfig { branching_factor: 10, noise_method: NoiseMethod::Saa },
/// };
///
/// let model = LoadModel {
///     bus_id: EntityId(1),
///     stage_id: 0,
///     mean_mw: 100.0,
///     std_mw: 20.0,
/// };
///
/// let lp = PrecomputedNormal::build(&[model], &[], &[stage], &[EntityId(1)], 1).unwrap();
/// assert_eq!(lp.n_stages(), 1);
/// assert_eq!(lp.n_entities(), 1);
/// assert!((lp.mean(0, 0) - 100.0).abs() < f64::EPSILON);
/// assert!((lp.std(0, 0) - 20.0).abs() < f64::EPSILON);
/// assert!((lp.block_factor(0, 0, 0) - 1.0).abs() < f64::EPSILON);
/// ```
#[must_use]
#[derive(Debug)]
pub struct PrecomputedNormal {
    /// Stage-level mean, length `n_stages * n_entities`.
    mean: Box<[f64]>,

    /// Stage-level standard deviation, length `n_stages * n_entities`.
    std: Box<[f64]>,

    /// Per-block scaling factor, length `n_stages * n_entities * max_blocks`;
    /// `1.0` where the input factor data has no entry.
    factors: Box<[f64]>,

    n_stages: usize,

    n_entities: usize,

    max_blocks: usize,
}

impl PrecomputedNormal {
    // -----------------------------------------------------------------------
    // Builder
    // -----------------------------------------------------------------------

    /// Build a [`PrecomputedNormal`] from raw normal noise model parameters.
    ///
    /// # Parameters
    ///
    /// - `models`: entries whose `bus_id` is absent from `entity_ids` are silently
    ///   ignored; missing `(entity, stage)` pairs default to zero mean/std.
    /// - `factors`: per-`(entity, stage, block)` scaling; missing entries default
    ///   to `1.0`.
    /// - `entity_ids`: canonical sorted order; determines entity array index order.
    ///
    /// # Errors
    ///
    /// Always `Ok` today; the `Result` is retained so future validation is not a
    /// breaking change.
    pub fn build(
        models: &[LoadModel],
        factors: &[EntityFactorEntry<'_>],
        study_stages: &[Stage],
        entity_ids: &[EntityId],
        max_blocks: usize,
    ) -> Result<Self, StochasticError> {
        let n_stages = study_stages.len();
        let n_entities = entity_ids.len();

        let entity_index: HashMap<EntityId, usize> = entity_ids
            .iter()
            .enumerate()
            .map(|(i, &id)| (id, i))
            .collect();

        let stage_index: HashMap<i32, usize> = study_stages
            .iter()
            .enumerate()
            .map(|(i, s)| (s.id, i))
            .collect();

        let n2 = n_stages * n_entities;
        let n3 = n_stages * n_entities * max_blocks.max(1);
        let mut mean_arr = vec![0.0f64; n2];
        let mut std_arr = vec![0.0f64; n2];
        let mut factors_arr = vec![1.0f64; n3];

        for model in models {
            let Some(&e_idx) = entity_index.get(&model.bus_id) else {
                continue;
            };
            let Some(&s_idx) = stage_index.get(&model.stage_id) else {
                continue;
            };
            let flat2 = s_idx * n_entities + e_idx;
            mean_arr[flat2] = model.mean_mw;
            std_arr[flat2] = model.std_mw;
        }

        if max_blocks > 0 {
            for (entity_id, stage_id, block_factors) in factors {
                let Some(&e_idx) = entity_index.get(entity_id) else {
                    continue;
                };
                let Some(&s_idx) = stage_index.get(stage_id) else {
                    continue;
                };
                for &(block_id, factor) in *block_factors {
                    let b_idx = usize::try_from(block_id).unwrap_or(usize::MAX);
                    if b_idx < max_blocks {
                        let flat3 = s_idx * n_entities * max_blocks + e_idx * max_blocks + b_idx;
                        factors_arr[flat3] = factor;
                    }
                }
            }
        }

        Ok(Self {
            mean: mean_arr.into_boxed_slice(),
            std: std_arr.into_boxed_slice(),
            factors: factors_arr.into_boxed_slice(),
            n_stages,
            n_entities,
            max_blocks,
        })
    }

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------

    fn debug_assert_in_bounds(&self, stage: usize, entity: usize) {
        debug_assert!(
            stage < self.n_stages,
            "stage index {stage} is out of bounds (n_stages = {})",
            self.n_stages
        );
        debug_assert!(
            entity < self.n_entities,
            "entity index {entity} is out of bounds (n_entities = {})",
            self.n_entities
        );
    }

    /// Stage-level mean for the given stage and entity indices.
    ///
    /// # Panics
    ///
    /// Panics if `stage >= n_stages` or `entity >= n_entities`.
    #[must_use]
    pub fn mean(&self, stage: usize, entity: usize) -> f64 {
        self.debug_assert_in_bounds(stage, entity);
        self.mean[stage * self.n_entities + entity]
    }

    /// Stage-level standard deviation for the given stage and entity indices.
    ///
    /// # Panics
    ///
    /// Panics if `stage >= n_stages` or `entity >= n_entities`.
    #[must_use]
    pub fn std(&self, stage: usize, entity: usize) -> f64 {
        self.debug_assert_in_bounds(stage, entity);
        self.std[stage * self.n_entities + entity]
    }

    /// Per-block scaling factor for the given stage, entity, and block indices.
    ///
    /// Returns `1.0` for any (stage, entity, block) combination that was not
    /// explicitly provided in the input factor data.
    ///
    /// # Panics
    ///
    /// Panics if `stage >= n_stages`, `entity >= n_entities`, or
    /// `block >= max_blocks`.
    #[must_use]
    pub fn block_factor(&self, stage: usize, entity: usize, block: usize) -> f64 {
        self.debug_assert_in_bounds(stage, entity);
        debug_assert!(
            block < self.max_blocks,
            "block index {block} is out of bounds (max_blocks = {})",
            self.max_blocks
        );
        self.factors[stage * self.n_entities * self.max_blocks + entity * self.max_blocks + block]
    }

    /// Number of study stages.
    #[must_use]
    pub fn n_stages(&self) -> usize {
        self.n_stages
    }

    /// Number of entities.
    #[must_use]
    pub fn n_entities(&self) -> usize {
        self.n_entities
    }

    /// Maximum block count across all stages.
    #[must_use]
    pub fn max_blocks(&self) -> usize {
        self.max_blocks
    }
}

impl Default for PrecomputedNormal {
    /// Empty sentinel for callers without any normal noise model entities.
    fn default() -> Self {
        Self {
            mean: Box::new([]),
            std: Box::new([]),
            factors: Box::new([]),
            n_stages: 0,
            n_entities: 0,
            max_blocks: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use cobre_core::{
        EntityId,
        scenario::LoadModel,
        temporal::{NoiseMethod, ScenarioSourceConfig, Stage},
        test_support::{StageSpec, single_block},
    };

    use super::{BlockFactorPair, EntityFactorEntry, PrecomputedNormal};

    fn make_stage(index: usize, id: i32) -> Stage {
        cobre_core::test_support::make_stage(StageSpec {
            id,
            index: Some(index),
            season_id: Some(0),
            blocks: single_block("SINGLE", 744.0),
            scenario_config: ScenarioSourceConfig {
                branching_factor: 10,
                noise_method: NoiseMethod::Saa,
            },
            ..Default::default()
        })
    }

    fn make_model(entity_id: i32, stage_id: i32, mean: f64, std: f64) -> LoadModel {
        LoadModel {
            bus_id: EntityId(entity_id),
            stage_id,
            mean_mw: mean,
            std_mw: std,
        }
    }

    #[test]
    fn build_empty_returns_zero_mean_std_one_factors() {
        let stages = [make_stage(0, 0)];
        let entity_ids = [EntityId(1)];

        let lp = PrecomputedNormal::build(&[], &[], &stages, &entity_ids, 2).unwrap();

        assert_eq!(lp.n_stages(), 1);
        assert_eq!(lp.n_entities(), 1);
        assert_eq!(lp.max_blocks(), 2);

        assert!(
            (lp.mean(0, 0)).abs() < f64::EPSILON,
            "empty models → mean = 0.0"
        );
        assert!(
            (lp.std(0, 0)).abs() < f64::EPSILON,
            "empty models → std = 0.0"
        );
        assert!(
            (lp.block_factor(0, 0, 0) - 1.0).abs() < f64::EPSILON,
            "empty factors → factor = 1.0"
        );
        assert!(
            (lp.block_factor(0, 0, 1) - 1.0).abs() < f64::EPSILON,
            "empty factors → factor = 1.0 for block 1"
        );
    }

    #[test]
    fn build_with_models_populates_mean_std() {
        let stages = [make_stage(0, 0), make_stage(1, 1), make_stage(2, 2)];
        let entity_ids = [EntityId(10), EntityId(20)];

        let models = vec![
            make_model(10, 0, 100.0, 15.0),
            make_model(10, 1, 110.0, 18.0),
            make_model(10, 2, 95.0, 12.0),
            make_model(20, 0, 200.0, 30.0),
            make_model(20, 1, 210.0, 35.0),
            make_model(20, 2, 195.0, 28.0),
        ];

        let lp = PrecomputedNormal::build(&models, &[], &stages, &entity_ids, 1).unwrap();

        assert_eq!(lp.n_stages(), 3);
        assert_eq!(lp.n_entities(), 2);

        // entity 0 = EntityId(10), entity 1 = EntityId(20)
        assert!((lp.mean(0, 0) - 100.0).abs() < f64::EPSILON);
        assert!((lp.std(0, 0) - 15.0).abs() < f64::EPSILON);
        assert!((lp.mean(1, 0) - 110.0).abs() < f64::EPSILON);
        assert!((lp.std(1, 0) - 18.0).abs() < f64::EPSILON);
        assert!((lp.mean(2, 0) - 95.0).abs() < f64::EPSILON);
        assert!((lp.std(2, 0) - 12.0).abs() < f64::EPSILON);

        assert!((lp.mean(0, 1) - 200.0).abs() < f64::EPSILON);
        assert!((lp.std(0, 1) - 30.0).abs() < f64::EPSILON);
        assert!((lp.mean(2, 1) - 195.0).abs() < f64::EPSILON);
        assert!((lp.std(2, 1) - 28.0).abs() < f64::EPSILON);
    }

    #[test]
    fn build_with_factors_populates_block_factors() {
        let stages = [make_stage(0, 0)];
        let entity_ids = [EntityId(1), EntityId(2)];

        let block_factors_entity2: &[BlockFactorPair] = &[(0, 0.85), (1, 1.15)];
        let factors: &[EntityFactorEntry<'_>] = &[(EntityId(2), 0, block_factors_entity2)];

        let lp = PrecomputedNormal::build(&[], factors, &stages, &entity_ids, 2).unwrap();

        // entity_ids[0] = EntityId(1), entity_ids[1] = EntityId(2)
        // entity 1 (EntityId(2)) has factors for stage 0
        assert!(
            (lp.block_factor(0, 1, 0) - 0.85).abs() < f64::EPSILON,
            "block 0 factor for entity 1 should be 0.85"
        );
        assert!(
            (lp.block_factor(0, 1, 1) - 1.15).abs() < f64::EPSILON,
            "block 1 factor for entity 1 should be 1.15"
        );

        // entity 0 (EntityId(1)) has no factors → defaults to 1.0
        assert!(
            (lp.block_factor(0, 0, 0) - 1.0).abs() < f64::EPSILON,
            "missing factor → 1.0"
        );
        assert!(
            (lp.block_factor(0, 0, 1) - 1.0).abs() < f64::EPSILON,
            "missing factor → 1.0"
        );
    }

    #[test]
    fn build_with_missing_entity_stage_defaults_to_zero() {
        let stages = [make_stage(0, 0), make_stage(1, 1)];
        let entity_ids = [EntityId(1), EntityId(2)];

        // Only entity 1 at stage 0 has a model
        let models = [make_model(1, 0, 150.0, 25.0)];

        let lp = PrecomputedNormal::build(&models, &[], &stages, &entity_ids, 1).unwrap();

        // Present entry
        assert!((lp.mean(0, 0) - 150.0).abs() < f64::EPSILON);
        assert!((lp.std(0, 0) - 25.0).abs() < f64::EPSILON);

        // Missing: entity 1 at stage 1
        assert!(
            (lp.mean(1, 0)).abs() < f64::EPSILON,
            "missing stage → mean = 0.0"
        );
        assert!(
            (lp.std(1, 0)).abs() < f64::EPSILON,
            "missing stage → std = 0.0"
        );

        // Missing: entity 2 at all stages
        assert!(
            (lp.mean(0, 1)).abs() < f64::EPSILON,
            "missing entity → mean = 0.0"
        );
        assert!(
            (lp.std(0, 1)).abs() < f64::EPSILON,
            "missing entity → std = 0.0"
        );
        assert!(
            (lp.mean(1, 1)).abs() < f64::EPSILON,
            "missing (entity, stage) → mean = 0.0"
        );
        assert!(
            (lp.std(1, 1)).abs() < f64::EPSILON,
            "missing (entity, stage) → std = 0.0"
        );
    }

    #[test]
    fn build_with_missing_factor_defaults_to_one() {
        let stages = [make_stage(0, 0), make_stage(1, 1)];
        let entity_ids = [EntityId(1), EntityId(2)];

        // Only entity 1 at stage 0, block 0 has a factor
        let present_factors: &[BlockFactorPair] = &[(0, 0.9)];
        let factors: &[EntityFactorEntry<'_>] = &[(EntityId(1), 0, present_factors)];

        let lp = PrecomputedNormal::build(&[], factors, &stages, &entity_ids, 3).unwrap();

        // Present entry
        assert!((lp.block_factor(0, 0, 0) - 0.9).abs() < f64::EPSILON);

        // Missing blocks for the same (stage, entity) → 1.0
        assert!((lp.block_factor(0, 0, 1) - 1.0).abs() < f64::EPSILON);
        assert!((lp.block_factor(0, 0, 2) - 1.0).abs() < f64::EPSILON);

        // Entirely missing (entity, stage) combinations → 1.0
        assert!((lp.block_factor(0, 1, 0) - 1.0).abs() < f64::EPSILON);
        assert!((lp.block_factor(1, 0, 0) - 1.0).abs() < f64::EPSILON);
        assert!((lp.block_factor(1, 1, 0) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn accessor_consistency_across_stages() {
        // 2 entities, 3 stages, 2 blocks — the acceptance criterion scenario.
        let stages = [make_stage(0, 0), make_stage(1, 1), make_stage(2, 2)];
        let entity_ids = [EntityId(5), EntityId(7)];

        let models = vec![
            // EntityId(5) = entity_idx 0
            make_model(5, 0, 50.0, 5.0),
            make_model(5, 1, 55.0, 6.0),
            make_model(5, 2, 48.0, 4.0),
            // EntityId(7) = entity_idx 1
            make_model(7, 0, 70.0, 8.0),
            make_model(7, 1, 75.0, 9.0),
            make_model(7, 2, 68.0, 7.0),
        ];

        let bf_e0_s1: &[BlockFactorPair] = &[(0, 0.8), (1, 1.2)];
        let bf_e1_s2: &[BlockFactorPair] = &[(0, 0.95), (1, 1.05)];
        let factors: &[EntityFactorEntry<'_>] =
            &[(EntityId(5), 1, bf_e0_s1), (EntityId(7), 2, bf_e1_s2)];

        let lp = PrecomputedNormal::build(&models, factors, &stages, &entity_ids, 2).unwrap();

        assert_eq!(lp.n_stages(), 3);
        assert_eq!(lp.n_entities(), 2);
        assert_eq!(lp.max_blocks(), 2);

        // mean(1, 0): stage_idx=1 maps to stage id=1, entity_idx=0 maps to EntityId(5)
        assert!(
            (lp.mean(1, 0) - 55.0).abs() < f64::EPSILON,
            "mean(1, 0) should be 55.0 (stage id=1, EntityId(5))"
        );
        assert!((lp.std(1, 0) - 6.0).abs() < f64::EPSILON);

        // Check factors: EntityId(5) at stage 1 → block 0=0.8, block 1=1.2
        assert!((lp.block_factor(1, 0, 0) - 0.8).abs() < f64::EPSILON);
        assert!((lp.block_factor(1, 0, 1) - 1.2).abs() < f64::EPSILON);

        // EntityId(7) at stage 2 → block 0=0.95, block 1=1.05
        assert!((lp.block_factor(2, 1, 0) - 0.95).abs() < f64::EPSILON);
        assert!((lp.block_factor(2, 1, 1) - 1.05).abs() < f64::EPSILON);

        // EntityId(5) at stage 0 → no factors → 1.0
        assert!((lp.block_factor(0, 0, 0) - 1.0).abs() < f64::EPSILON);
        assert!((lp.block_factor(0, 0, 1) - 1.0).abs() < f64::EPSILON);

        // All stage/entity combinations have correct mean/std
        assert!((lp.mean(0, 0) - 50.0).abs() < f64::EPSILON);
        assert!((lp.mean(2, 0) - 48.0).abs() < f64::EPSILON);
        assert!((lp.mean(0, 1) - 70.0).abs() < f64::EPSILON);
        assert!((lp.mean(1, 1) - 75.0).abs() < f64::EPSILON);
        assert!((lp.mean(2, 1) - 68.0).abs() < f64::EPSILON);
    }

    #[test]
    fn acceptance_criterion_mean_stage1_entity0() {
        let stages = [make_stage(0, 0), make_stage(1, 1), make_stage(2, 2)];
        let entity_ids = [EntityId(3), EntityId(9)];

        let models = vec![
            make_model(3, 0, 120.0, 10.0),
            make_model(3, 1, 130.0, 11.0), // stage_ids[1] = id 1, entity_ids[0] = EntityId(3)
            make_model(3, 2, 115.0, 9.0),
            make_model(9, 0, 220.0, 20.0),
            make_model(9, 1, 230.0, 22.0),
            make_model(9, 2, 210.0, 18.0),
        ];

        let lp = PrecomputedNormal::build(&models, &[], &stages, &entity_ids, 1).unwrap();

        // mean(1, 0): stage_idx=1 corresponds to study_stages[1] (id=1),
        //             entity_idx=0 corresponds to entity_ids[0] = EntityId(3)
        assert!(
            (lp.mean(1, 0) - 130.0).abs() < f64::EPSILON,
            "mean(1, 0) should equal the LoadModel with EntityId(3) at stage_id=1: \
             expected 130.0, got {}",
            lp.mean(1, 0)
        );
    }

    #[test]
    fn declaration_order_invariance() {
        let stage = make_stage(0, 0);
        let entity_ids = [EntityId(3), EntityId(7)];

        // Models provided in reverse entity order
        let models = vec![make_model(7, 0, 200.0, 30.0), make_model(3, 0, 100.0, 15.0)];

        let lp = PrecomputedNormal::build(&models, &[], &[stage], &entity_ids, 1).unwrap();

        // entity_ids[0] = EntityId(3) → mean 100.0
        // entity_ids[1] = EntityId(7) → mean 200.0
        assert!(
            (lp.mean(0, 0) - 100.0).abs() < f64::EPSILON,
            "entity index 0 should be EntityId(3) with mean=100, got {}",
            lp.mean(0, 0)
        );
        assert!(
            (lp.mean(0, 1) - 200.0).abs() < f64::EPSILON,
            "entity index 1 should be EntityId(7) with mean=200, got {}",
            lp.mean(0, 1)
        );
    }

    /// A deterministic load (`std_mw == 0.0`) with block factors `{1.2, 1.0,
    /// 0.8}` still applies the factors to the realized per-block value:
    /// `mean_mw * factor[b]` exactly, since `factors_arr` (block factors) is
    /// populated independent of `std_arr` (see the module docs' noise model).
    #[test]
    fn test_block_factors_applied_at_zero_sigma() {
        let stage = make_stage(0, 0);
        let entity_ids = [EntityId(1)];
        let mean_mw = 100.0;
        let model = make_model(1, 0, mean_mw, 0.0);

        let block_factors: &[BlockFactorPair] = &[(0, 1.2), (1, 1.0), (2, 0.8)];
        let factors: &[EntityFactorEntry<'_>] = &[(EntityId(1), 0, block_factors)];

        let lp = PrecomputedNormal::build(&[model], factors, &[stage], &entity_ids, 3).unwrap();

        assert!(
            (lp.std(0, 0)).abs() < f64::EPSILON,
            "std_mw == 0.0 (deterministic load)"
        );

        for (block, &(_, factor)) in block_factors.iter().enumerate() {
            let eta = 0.0;
            let realization = (lp.mean(0, 0) + lp.std(0, 0) * eta).max(0.0);
            let per_block_value = realization * lp.block_factor(0, 0, block);
            let expected = mean_mw * factor;
            assert!(
                (per_block_value - expected).abs() < 1e-10,
                "block {block}: expected mean_mw * factor = {expected}, got {per_block_value}"
            );
        }
    }

    // debug-only: release lets native Vec indexing panic with a different message
    // that would fail #[should_panic].
    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "stage index 1 is out of bounds")]
    fn mean_out_of_bounds_panics() {
        let stage = make_stage(0, 0);
        let model = make_model(1, 0, 100.0, 20.0);
        let lp = PrecomputedNormal::build(&[model], &[], &[stage], &[EntityId(1)], 1).unwrap();
        let _ = lp.mean(1, 0);
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "entity index 1 is out of bounds")]
    fn mean_entity_out_of_bounds_panics() {
        let stage = make_stage(0, 0);
        let model = make_model(1, 0, 100.0, 20.0);
        let lp = PrecomputedNormal::build(&[model], &[], &[stage], &[EntityId(1)], 1).unwrap();
        let _ = lp.mean(0, 1);
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "stage index 1 is out of bounds")]
    fn std_out_of_bounds_panics() {
        let stage = make_stage(0, 0);
        let model = make_model(1, 0, 100.0, 20.0);
        let lp = PrecomputedNormal::build(&[model], &[], &[stage], &[EntityId(1)], 1).unwrap();
        let _ = lp.std(1, 0);
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "block index 1 is out of bounds")]
    fn block_factor_out_of_bounds_panics() {
        let stage = make_stage(0, 0);
        let lp = PrecomputedNormal::build(&[], &[], &[stage], &[EntityId(1)], 1).unwrap();
        let _ = lp.block_factor(0, 0, 1);
    }

    #[test]
    fn default_is_empty() {
        let lp = PrecomputedNormal::default();
        assert_eq!(lp.n_stages(), 0);
        assert_eq!(lp.n_entities(), 0);
        assert_eq!(lp.max_blocks(), 0);
    }
}
