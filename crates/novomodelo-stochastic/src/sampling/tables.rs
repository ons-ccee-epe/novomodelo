//! Scenario-invariant per-class noise tables and the builder that fills them,
//! rebuilt once per training iteration and shared by reference across
//! forward-pass workers.

use cobre_core::temporal::NoiseMethod;

use crate::{
    StochasticError,
    tree::{
        lhs::LhsPrecomputed,
        qmc_halton::HaltonPrecomputed,
        qmc_sobol::{MAX_SOBOL_DIM, SobolPrecomputed},
    },
};

/// Precomputed scenario-invariant state for one `(noise_group_id,
/// noise_method)` pair.
#[derive(Debug, Clone)]
pub enum NoiseTable {
    /// SAA, Selective and `HistoricalResiduals` rebuild nothing per draw.
    Direct,
    /// Precomputed LHS per-dimension stratum permutations.
    Lhs(LhsPrecomputed),
    /// Precomputed Sobol direction matrix and scramble parameters.
    Sobol(SobolPrecomputed),
    /// Precomputed Halton prime table and scramble tables.
    Halton(HaltonPrecomputed),
}

/// One entity class's [`NoiseTable`]s for one training iteration, indexed per
/// stage by [`Self::table_at`]. Keys on the `(noise_group_id, noise_method)`
/// pair, not the group alone: `precompute_noise_groups` derives groups from
/// `(season_id, year)` independent of `sampling_method`, so two stages sharing
/// a group can carry different methods.
#[derive(Debug, Default)]
pub struct ClassNoiseTables {
    slots: Vec<u32>,
    tables: Vec<NoiseTable>,
    keys: Vec<(u32, NoiseMethod)>,
    iteration: u32,
    total_scenarios: u32,
}

impl ClassNoiseTables {
    /// Table for `stage_idx`, or `None` past the end of the populated range —
    /// always `None` for a class that is not sampled out of sample.
    #[must_use]
    pub fn table_at(&self, stage_idx: usize) -> Option<&NoiseTable> {
        self.slots
            .get(stage_idx)
            .and_then(|&slot| self.tables.get(slot as usize))
    }

    /// Group `stage_idx`'s table was built for, or `None` past the end.
    #[must_use]
    pub fn group_at(&self, stage_idx: usize) -> Option<u32> {
        self.slots
            .get(stage_idx)
            .and_then(|&slot| self.keys.get(slot as usize))
            .map(|&(group, _)| group)
    }

    /// The `(iteration, total_scenarios)` the last `refill` recorded.
    #[must_use]
    pub fn built_for(&self) -> (u32, u32) {
        (self.iteration, self.total_scenarios)
    }

    /// Clear all three vectors, keeping their allocated capacity, and reset
    /// the stamped `(iteration, total_scenarios)`.
    pub(crate) fn clear(&mut self) {
        self.slots.clear();
        self.tables.clear();
        self.keys.clear();
        self.iteration = 0;
        self.total_scenarios = 0;
    }

    /// Refill from `noise_methods` (one entry per stage) and `noise_group_ids`,
    /// deduplicating tables by `(noise_group_id, noise_method)` in
    /// first-appearance stage order. Reuses existing `Vec` capacity via
    /// [`Self::clear`], so a repeat call allocates only when a table itself
    /// grows.
    ///
    /// # Errors
    ///
    /// Returns [`StochasticError::DimensionExceedsCapacity`] when `dim >
    /// MAX_SOBOL_DIM` and `noise_methods` contains `QmcSobol`; `slots`,
    /// `tables`, and `keys` stay empty.
    pub(crate) fn refill(
        &mut self,
        seed: u64,
        dim: usize,
        iteration: u32,
        total_scenarios: u32,
        noise_group_ids: &[u32],
        noise_methods: &[NoiseMethod],
    ) -> Result<(), StochasticError> {
        self.clear();
        self.iteration = iteration;
        self.total_scenarios = total_scenarios;
        if dim > MAX_SOBOL_DIM && noise_methods.contains(&NoiseMethod::QmcSobol) {
            return Err(StochasticError::DimensionExceedsCapacity {
                dim,
                max_dim: MAX_SOBOL_DIM,
                method: "sobol".to_string(),
            });
        }
        for (stage_idx, &method) in noise_methods.iter().enumerate() {
            // An empty `noise_group_ids` means every stage is its own group —
            // mirrors `StageContext::noise_group_id_at`'s empty-slice behavior;
            // falling back to group `0` would collapse every stage onto one
            // table.
            #[allow(clippy::cast_possible_truncation)]
            let group = noise_group_ids
                .get(stage_idx)
                .copied()
                .unwrap_or(stage_idx as u32);
            let key = (group, method);
            let slot = self.keys.iter().position(|&k| k == key).unwrap_or_else(|| {
                self.keys.push(key);
                self.tables.push(build_table(
                    seed,
                    dim,
                    iteration,
                    total_scenarios,
                    group,
                    method,
                ));
                self.tables.len() - 1
            });
            #[allow(clippy::cast_possible_truncation)]
            self.slots.push(slot as u32);
        }
        Ok(())
    }
}

fn build_table(
    seed: u64,
    dim: usize,
    iteration: u32,
    total_scenarios: u32,
    group: u32,
    method: NoiseMethod,
) -> NoiseTable {
    match method {
        NoiseMethod::QmcSobol => {
            NoiseTable::Sobol(SobolPrecomputed::new(seed, iteration, group, dim))
        }
        NoiseMethod::QmcHalton => NoiseTable::Halton(HaltonPrecomputed::new(
            seed,
            iteration,
            group,
            dim,
            total_scenarios,
        )),
        NoiseMethod::Lhs => NoiseTable::Lhs(LhsPrecomputed::new(
            seed,
            iteration,
            group,
            dim,
            total_scenarios,
        )),
        NoiseMethod::Saa | NoiseMethod::Selective | NoiseMethod::HistoricalResiduals => {
            NoiseTable::Direct
        }
    }
}

/// Per-class forward noise tables for one training iteration. Mirrors
/// `ForwardSampler`'s own inflow/load/NCS split so a reader pairs them
/// without a class enum or an index.
#[derive(Debug, Default)]
pub struct ForwardNoiseTables {
    pub(crate) inflow: ClassNoiseTables,
    pub(crate) load: ClassNoiseTables,
    pub(crate) ncs: ClassNoiseTables,
}

impl ForwardNoiseTables {
    /// Inflow class tables.
    #[must_use]
    pub fn inflow(&self) -> &ClassNoiseTables {
        &self.inflow
    }

    /// Load class tables.
    #[must_use]
    pub fn load(&self) -> &ClassNoiseTables {
        &self.load
    }

    /// NCS class tables.
    #[must_use]
    pub fn ncs(&self) -> &ClassNoiseTables {
        &self.ncs
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use cobre_core::temporal::NoiseMethod;

    use super::{ClassNoiseTables, NoiseTable};
    use crate::{StochasticError, tree::qmc_sobol::MAX_SOBOL_DIM};

    #[test]
    fn table_at_returns_none_for_default_constructed() {
        let tables = ClassNoiseTables::default();
        assert!(tables.table_at(0).is_none());
    }

    #[test]
    fn table_at_returns_none_past_populated_range() {
        let mut tables = ClassNoiseTables::default();
        tables
            .refill(1, 2, 0, 4, &[0, 1], &[NoiseMethod::Saa, NoiseMethod::Saa])
            .expect("Saa never exceeds the Sobol dimension cap");
        assert!(tables.table_at(2).is_none());
    }

    #[test]
    fn refill_dedups_by_group_and_method() {
        let mut tables = ClassNoiseTables::default();
        let methods = [NoiseMethod::Lhs, NoiseMethod::Lhs, NoiseMethod::Lhs];
        let groups = [0u32, 0, 1];
        tables
            .refill(7, 2, 0, 8, &groups, &methods)
            .expect("Lhs never exceeds the Sobol dimension cap");
        assert_eq!(tables.slots, vec![0, 0, 1]);
        assert_eq!(tables.tables.len(), 2);
    }

    #[test]
    fn built_for_and_group_at_reflect_the_last_refill() {
        let mut tables = ClassNoiseTables::default();
        let methods = [NoiseMethod::Lhs, NoiseMethod::Lhs, NoiseMethod::Lhs];
        let groups = [0u32, 0, 1];
        tables
            .refill(7, 2, 3, 8, &groups, &methods)
            .expect("Lhs never exceeds the Sobol dimension cap");

        assert_eq!(tables.built_for(), (3, 8));
        assert_eq!(tables.group_at(0), Some(0));
        assert_eq!(tables.group_at(1), Some(0));
        assert_eq!(tables.group_at(2), Some(1));
        assert_eq!(tables.group_at(3), None);
    }

    #[test]
    fn refill_is_repeatable_without_growing_tables() {
        let mut tables = ClassNoiseTables::default();
        let methods = [NoiseMethod::Lhs, NoiseMethod::Lhs, NoiseMethod::Lhs];
        let groups = [0u32, 0, 1];
        tables
            .refill(7, 2, 0, 8, &groups, &methods)
            .expect("Lhs never exceeds the Sobol dimension cap");
        let first_slots = tables.slots.clone();

        tables
            .refill(7, 2, 0, 8, &groups, &methods)
            .expect("Lhs never exceeds the Sobol dimension cap");

        assert_eq!(tables.slots, first_slots);
        assert_eq!(tables.tables.len(), 2);
    }

    #[test]
    fn refill_rejects_a_sobol_class_wider_than_the_table() {
        let mut tables = ClassNoiseTables::default();
        let dim = MAX_SOBOL_DIM + 1;

        let result = tables.refill(1, dim, 0, 1, &[0], &[NoiseMethod::QmcSobol]);

        match result {
            Err(StochasticError::DimensionExceedsCapacity {
                dim: got_dim,
                max_dim,
                method,
            }) => {
                assert_eq!(got_dim, dim, "dim field");
                assert_eq!(max_dim, MAX_SOBOL_DIM, "max_dim field");
                assert!(
                    method.contains("sobol"),
                    "method must contain 'sobol', got: {method}"
                );
            }
            other => panic!("expected Err(DimensionExceedsCapacity), got {other:?}"),
        }
        assert!(tables.table_at(0).is_none());
    }

    #[test]
    fn refill_builds_a_wide_class_without_sobol() {
        let mut tables = ClassNoiseTables::default();
        let dim = MAX_SOBOL_DIM + 1;

        let result = tables.refill(1, dim, 0, 1, &[0], &[NoiseMethod::Lhs]);

        assert!(result.is_ok(), "expected Ok, got: {result:?}");
        assert!(matches!(tables.table_at(0), Some(NoiseTable::Lhs(_))));
    }
}
