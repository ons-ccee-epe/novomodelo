//! The effective capacity of a hydro plant, which is what `hydro_availability` reads.
//!
//! The capacity is data fixed for each stage and block: the sum of the maximum generation of
//! the plant's unit groups. A group contributes the maximum the study resolves for the block or
//! the stage, where a maintenance reduction lives, and its declared maximum where the study
//! resolves none. The groups are added in ascending group-id order, so the sum is bit-identical
//! whatever order the plant declares them in. The capacity is 0 at a stage where the plant does
//! not operate ([`hydro_operating_active`]): it generates nothing there.

use novomodelo_core::{
    EntityId,
    commissioning::hydro_operating_active,
    entities::{Hydro, HydroUnitGroup},
    resolved::ResolvedHydroUnitGroupBounds,
};

/// A plant whose capacity cannot be resolved.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum AvailabilityError {
    /// The plant is not in the system or declares no unit group.
    #[error(
        "hydro {hydro_id} has no unit group, so its availability at stage {stage_id} has no capacity to sum"
    )]
    NoUnitGroups {
        /// The plant.
        hydro_id: EntityId,
        /// The first study stage, where the resolution starts.
        stage_id: i32,
    },
    /// The summed capacity is `NaN` or infinite.
    #[error(
        "hydro {hydro_id} has the non-finite availability {value} at stage {stage_id}, block {block}"
    )]
    NonFinite {
        /// The plant.
        hydro_id: EntityId,
        /// The stage of the cell.
        stage_id: i32,
        /// The block of the cell.
        block: usize,
        /// The sum.
        value: f64,
    },
}

/// The capacity in MW of plant `hydro_id` at every block of every study stage, indexed by stage
/// position and then by block; the capacity is 0 at a stage where the plant does not operate.
///
/// `hydros` is the system's plants in the order `group_bounds` was resolved against, and
/// `study_stage_ids` and `stage_block_counts` hold the study stages' ids and block counts, one
/// entry per stage.
///
/// # Errors
///
/// [`AvailabilityError::NoUnitGroups`] when the plant has no unit group, and
/// [`AvailabilityError::NonFinite`] on the first cell whose capacity is not finite.
///
/// # Panics
///
/// When `study_stage_ids` and `stage_block_counts` differ in length.
///
/// # Examples
///
/// ```
/// use novomodelo_core::{EntityId, resolved::ResolvedHydroUnitGroupBounds};
/// use novomodelo_io::resolve_hydro_availability;
///
/// let capacity = resolve_hydro_availability(
///     &[],
///     &ResolvedHydroUnitGroupBounds::empty(),
///     EntityId(14),
///     &[],
///     &[],
/// );
/// assert_eq!(capacity, Ok(Vec::new()));
/// ```
pub fn resolve_hydro_availability(
    hydros: &[Hydro],
    group_bounds: &ResolvedHydroUnitGroupBounds,
    hydro_id: EntityId,
    study_stage_ids: &[i32],
    stage_block_counts: &[usize],
) -> Result<Vec<Vec<f64>>, AvailabilityError> {
    assert_eq!(
        study_stage_ids.len(),
        stage_block_counts.len(),
        "one block count per study stage"
    );
    let located = hydros
        .iter()
        .enumerate()
        .find(|(_, hydro)| hydro.id == hydro_id)
        .map(|(hydro_idx, hydro)| {
            let mut groups: Vec<(usize, &HydroUnitGroup)> =
                hydro.unit_groups.iter().enumerate().collect();
            groups.sort_by_key(|&(_, group)| group.id.0);
            (hydro_idx, hydro, groups)
        })
        .filter(|(_, _, groups)| !groups.is_empty());
    let Some((hydro_idx, hydro, groups)) = located else {
        return study_stage_ids.first().map_or_else(
            || Ok(Vec::new()),
            |&stage_id| Err(AvailabilityError::NoUnitGroups { hydro_id, stage_id }),
        );
    };

    study_stage_ids
        .iter()
        .zip(stage_block_counts)
        .enumerate()
        .map(|(stage_idx, (&stage_id, &n_blocks))| {
            if !hydro_operating_active(
                hydro.filling.as_ref(),
                hydro.entry_stage_id,
                hydro.exit_stage_id,
                stage_id,
            ) {
                return Ok(vec![0.0; n_blocks]);
            }
            (0..n_blocks)
                .map(|block| {
                    let value = groups.iter().fold(0.0, |sum, &(group_pos, group)| {
                        sum + group_bounds
                            .override_at_block(hydro_idx, group_pos, stage_idx, block)
                            .max_generation_mw
                            .unwrap_or(group.max_generation_mw)
                    });
                    if value.is_finite() {
                        Ok(value)
                    } else {
                        Err(AvailabilityError::NonFinite {
                            hydro_id,
                            stage_id,
                            block,
                            value,
                        })
                    }
                })
                .collect()
        })
        .collect()
}

#[cfg(test)]
#[allow(clippy::panic, clippy::unwrap_used)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::constraints::HydroUnitGroupBoundsRow;
    use crate::resolution::resolve_hydro_unit_group_bounds;
    use crate::test_support::{make_hydro, make_unit_group};

    fn plant(id: i32, maxima: &[(i32, f64)]) -> Hydro {
        let mut hydro = make_hydro(id, None);
        hydro.unit_groups = maxima
            .iter()
            .map(|&(group, mw)| make_unit_group(group, 1, 0.0, mw, 0.0, 100.0))
            .collect();
        hydro
    }

    fn row(
        hydro: i32,
        group: i32,
        stage: i32,
        block: Option<i32>,
        max_generation_mw: Option<f64>,
    ) -> HydroUnitGroupBoundsRow {
        HydroUnitGroupBoundsRow {
            hydro_id: EntityId(hydro),
            hydro_unit_group_id: EntityId(group),
            stage_id: stage,
            min_turbined_m3s: None,
            max_turbined_m3s: Some(7.0),
            min_generation_mw: None,
            max_generation_mw,
            block_id: block,
        }
    }

    fn resolve(
        hydros: &[Hydro],
        rows: &[HydroUnitGroupBoundsRow],
        hydro_id: i32,
        stages: &[i32],
        blocks: &[usize],
    ) -> Result<Vec<Vec<f64>>, AvailabilityError> {
        let stage_index: HashMap<i32, usize> = stages
            .iter()
            .enumerate()
            .map(|(idx, &id)| (id, idx))
            .collect();
        let table =
            resolve_hydro_unit_group_bounds(hydros, stages.len(), &stage_index, rows, blocks);
        resolve_hydro_availability(hydros, &table, EntityId(hydro_id), stages, blocks)
    }

    /// derived: one group of 500 MW and no override gives 500 in every block of every stage,
    /// whatever the stage's block count.
    #[test]
    fn a_group_without_overrides_gives_its_declared_maximum_in_every_cell() {
        let hydros = [plant(14, &[(1, 500.0)])];
        let capacity = resolve(&hydros, &[], 14, &[0, 1, 2], &[2, 2, 3]).unwrap();
        assert_eq!(capacity, [vec![500.0; 2], vec![500.0; 2], vec![500.0; 3]]);
    }

    /// derived: groups of 300 and 200 MW sum to 500; group 1 cut to 250 in stage 2 gives
    /// 250 + 200 = 450 there and 500 elsewhere.
    #[test]
    fn a_stage_override_of_a_group_is_a_maintenance_reduction() {
        let hydros = [plant(14, &[(1, 300.0), (2, 200.0)])];
        let rows = [row(14, 1, 2, None, Some(250.0))];
        let capacity = resolve(&hydros, &rows, 14, &[0, 1, 2, 3], &[1, 1, 1, 1]).unwrap();
        assert_eq!(capacity, [[500.0], [500.0], [450.0], [500.0]]);
    }

    /// derived: a block row cuts group 2 to 50 in block 1 of stage 1 only, so that cell is
    /// 300 + 50; its stage row, set on the same group, is 120 elsewhere in stage 1, which is
    /// 300 + 120 in block 0; a row that sets no generation maximum changes nothing.
    #[test]
    fn a_block_override_wins_over_the_stage_override_and_only_a_generation_maximum_counts() {
        let hydros = [plant(14, &[(1, 300.0), (2, 200.0)])];
        let rows = [
            row(14, 2, 1, None, Some(120.0)),
            row(14, 2, 1, Some(1), Some(50.0)),
            row(14, 1, 0, None, None),
        ];
        let capacity = resolve(&hydros, &rows, 14, &[0, 1], &[2, 2]).unwrap();
        assert_eq!(capacity, [[500.0, 500.0], [420.0, 350.0]]);
    }

    /// derived: the sum is `(0.1 + 0.2) + 0.3` in group-id order, and stage 1 cuts group 3 to
    /// 0.7, giving `(0.1 + 0.2) + 0.7`. The two plants declare the same groups in opposite
    /// order, so their override tables address different positions. Adding in declaration order
    /// gives `(0.3 + 0.2) + 0.1`, which differs in the last bit, so the comparison has power.
    #[test]
    fn the_sum_is_bit_identical_whatever_order_the_groups_are_declared_in() {
        let by_id = [plant(14, &[(1, 0.1), (2, 0.2), (3, 0.3)])];
        let reversed = [plant(14, &[(3, 0.3), (2, 0.2), (1, 0.1)])];
        assert_ne!(
            ((0.3_f64 + 0.2) + 0.1).to_bits(),
            ((0.1_f64 + 0.2) + 0.3).to_bits()
        );
        let rows = [row(14, 3, 1, None, Some(0.7))];
        let expected = [
            ((0.1_f64 + 0.2) + 0.3).to_bits(),
            ((0.1_f64 + 0.2) + 0.7).to_bits(),
        ];

        for hydros in [&by_id, &reversed] {
            let capacity = resolve(hydros, &rows, 14, &[0, 1], &[1, 1]).unwrap();
            let bits: Vec<u64> = capacity.iter().map(|stage| stage[0].to_bits()).collect();
            assert_eq!(bits, expected);
        }
    }

    #[test]
    fn a_plant_has_no_capacity_before_its_entry_stage() {
        let mut hydro = plant(14, &[(1, 100.0)]);
        hydro.entry_stage_id = Some(2);
        let capacity = resolve(&[hydro], &[], 14, &[0, 1, 2, 3], &[2, 2, 2, 2]).unwrap();
        assert_eq!(
            capacity,
            [[0.0, 0.0], [0.0, 0.0], [100.0, 100.0], [100.0, 100.0]]
        );
    }

    #[test]
    fn a_plant_has_no_capacity_from_its_exit_stage_on() {
        let mut hydro = plant(14, &[(1, 100.0)]);
        hydro.exit_stage_id = Some(1);
        let capacity = resolve(&[hydro], &[], 14, &[0, 1, 2], &[1, 1, 1]).unwrap();
        assert_eq!(capacity, [[100.0], [0.0], [0.0]]);
    }

    #[test]
    fn a_plant_without_unit_groups_is_an_error_naming_it_and_the_first_stage() {
        let hydros = [plant(14, &[]), plant(15, &[(1, 10.0)])];
        for unresolvable in [14, 99] {
            let error = resolve(&hydros, &[], unresolvable, &[5, 6], &[1, 1]).unwrap_err();
            assert_eq!(
                error,
                AvailabilityError::NoUnitGroups {
                    hydro_id: EntityId(unresolvable),
                    stage_id: 5,
                }
            );
            let message = error.to_string();
            assert!(message.contains(&unresolvable.to_string()), "{message}");
            assert!(message.contains("stage 5"), "{message}");
        }
        assert_eq!(resolve(&hydros, &[], 14, &[], &[]), Ok(Vec::new()));
    }

    #[test]
    fn a_non_finite_capacity_is_an_error_naming_the_cell() {
        for maxima in [
            vec![(1, 10.0), (2, f64::NAN)],
            vec![(1, f64::MAX), (2, f64::MAX)],
        ] {
            let hydros = [plant(14, &maxima)];
            let error = resolve(&hydros, &[], 14, &[5, 6], &[2, 2]).unwrap_err();
            let AvailabilityError::NonFinite {
                hydro_id,
                stage_id,
                block,
                value,
            } = error
            else {
                panic!("{error:?}");
            };
            assert_eq!((hydro_id, stage_id, block), (EntityId(14), 5, 0));
            assert!(!value.is_finite());
            assert!(error.to_string().contains("stage 5, block 0"));
        }
    }

    #[test]
    #[should_panic(expected = "one block count per study stage")]
    fn a_grid_of_unequal_lengths_is_a_caller_bug() {
        let hydros = [plant(14, &[(1, 10.0)])];
        let _ = resolve_hydro_availability(
            &hydros,
            &ResolvedHydroUnitGroupBounds::empty(),
            EntityId(14),
            &[0, 1],
            &[1],
        );
    }
}
