//! Resolution of `penalty_overrides_bus_deficit` rows into the per-`(bus,
//! stage, block, segment)` deficit table: a stage-wide row replaces the stage
//! layer's segment, a block row the block layer's, and rows carry depth only, so
//! cost always comes from the base curve. Validation upstream guarantees
//! complete keys over known buses, study stages and in-range blocks.

use std::collections::HashMap;

use novomodelo_core::{Bus, DeficitSegment, EntityId, ResolvedDeficitSegments};

use crate::constraints::BusDeficitOverrideRow;

/// Build a resolved bus deficit table from parsed override rows; each row is
/// one independent cell write, so row order does not matter.
///
/// `buses` must be in canonical order. A row naming no bus, stage or base
/// segment, or a block outside the stage, is skipped silently.
// implicit_hasher: callers pass a concrete `HashMap`; a `BuildHasher` generic buys nothing.
#[must_use]
#[allow(clippy::implicit_hasher)]
pub fn resolve_bus_deficit(
    buses: &[Bus],
    n_stages: usize,
    stage_index: &HashMap<i32, usize>,
    rows: &[BusDeficitOverrideRow],
    blocks_per_stage: &[usize],
) -> ResolvedDeficitSegments {
    let max_blocks = blocks_per_stage.iter().copied().max().unwrap_or(0);
    let mut table = ResolvedDeficitSegments::seeded(buses, n_stages, max_blocks);
    let bus_position: HashMap<EntityId, usize> = buses
        .iter()
        .enumerate()
        .map(|(position, bus)| (bus.id, position))
        .collect();

    for row in rows {
        let Some((bus, stage, seg, segment)) = row_target(row, &bus_position, buses, stage_index)
        else {
            continue;
        };
        match row.block_id {
            None => {
                if let Some(cell) = table.stage_segment_mut(bus, stage, seg) {
                    *cell = segment;
                }
            }
            Some(block_id) => {
                let Ok(block) = usize::try_from(block_id) else {
                    continue;
                };
                if blocks_per_stage
                    .get(stage)
                    .is_none_or(|&n_blocks| block >= n_blocks)
                {
                    continue;
                }
                if let Some(cell) = table.block_segment_mut(bus, stage, block, seg) {
                    *cell = Some(segment);
                }
            }
        }
    }

    table
}

/// Resolve a row's `(bus_idx, stage_idx, seg, segment)` target, where `segment`
/// carries the row's depth and the base curve's cost; `None` skips the row.
fn row_target(
    row: &BusDeficitOverrideRow,
    bus_position: &HashMap<EntityId, usize>,
    buses: &[Bus],
    stage_index: &HashMap<i32, usize>,
) -> Option<(usize, usize, usize, DeficitSegment)> {
    let &bus = bus_position.get(&row.bus_id)?;
    let &stage = stage_index.get(&row.stage_id)?;
    let seg = usize::try_from(row.segment_index).ok()?;
    let base = buses.get(bus)?.deficit_segments.get(seg)?;
    Some((
        bus,
        stage,
        seg,
        DeficitSegment {
            depth_mw: row.depth_mw,
            cost_per_mwh: base.cost_per_mwh,
        },
    ))
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use std::collections::HashMap;

    use novomodelo_core::{Bus, DeficitSegment, EntityId, ResolvedDeficitSegments};

    use super::resolve_bus_deficit;
    use crate::constraints::BusDeficitOverrideRow;
    use crate::test_support::make_bus_with_segments;

    const BLOCKS_PER_STAGE: [usize; 2] = [2, 2];

    fn si(n: usize) -> HashMap<i32, usize> {
        (0..n).map(|i| (i32::try_from(i).unwrap(), i)).collect()
    }

    fn buses() -> Vec<Bus> {
        vec![make_bus_with_segments(1, 2), make_bus_with_segments(2, 3)]
    }

    fn row(
        bus: i32,
        stage: i32,
        block: Option<i32>,
        segment_index: i32,
        depth_mw: Option<f64>,
    ) -> BusDeficitOverrideRow {
        BusDeficitOverrideRow {
            bus_id: EntityId::from(bus),
            stage_id: stage,
            block_id: block,
            segment_index,
            depth_mw,
        }
    }

    fn segment(depth_mw: Option<f64>, cost_per_mwh: f64) -> DeficitSegment {
        DeficitSegment {
            depth_mw,
            cost_per_mwh,
        }
    }

    /// Bus 1 at stage id 0 stage-wide, and at stage id 1 block 1, in parser order.
    fn valid_rows() -> Vec<BusDeficitOverrideRow> {
        vec![
            row(1, 0, None, 0, Some(250.0)),
            row(1, 0, None, 1, None),
            row(1, 1, Some(1), 0, Some(400.0)),
            row(1, 1, Some(1), 1, None),
        ]
    }

    fn resolve(rows: &[BusDeficitOverrideRow]) -> ResolvedDeficitSegments {
        resolve_bus_deficit(&buses(), 2, &si(2), rows, &BLOCKS_PER_STAGE)
    }

    #[test]
    fn stage_row_sets_every_block_of_its_stage() {
        let table = resolve(&valid_rows());
        for block in 0..2 {
            assert_eq!(
                table.segment_at_block(0, 0, block, 0),
                segment(Some(250.0), 500.0)
            );
            assert_eq!(
                table.segment_at_block(0, 0, block, 1),
                segment(None, 1000.0)
            );
        }
    }

    #[test]
    fn block_row_sets_only_its_block() {
        let buses = buses();
        let table = resolve(&valid_rows());
        assert_eq!(
            table.segment_at_block(0, 1, 1, 0),
            segment(Some(400.0), 500.0)
        );
        assert_eq!(
            table.segment_at_block(0, 1, 0, 0),
            buses[0].deficit_segments[0]
        );
    }

    #[test]
    fn bus_without_rows_reads_its_base_curve() {
        let buses = buses();
        let table = resolve(&valid_rows());
        for (seg, base) in buses[1].deficit_segments.iter().enumerate() {
            for stage in 0..2 {
                for block in 0..2 {
                    assert_eq!(table.segment_at_block(1, stage, block, seg), *base);
                }
            }
        }
    }

    #[test]
    fn unresolvable_rows_are_skipped() {
        let buses = buses();
        let mut rows = valid_rows();
        rows.extend([
            row(9, 0, None, 0, Some(1.0)),
            row(1, 7, None, 0, Some(1.0)),
            row(1, 0, Some(5), 0, Some(1.0)),
            row(1, 0, Some(-1), 0, Some(1.0)),
            row(1, 0, None, 3, Some(1.0)),
            row(1, 0, None, -1, Some(1.0)),
        ]);
        let table = resolve(&rows);
        assert_eq!(table, resolve(&valid_rows()));
        for (bus, spec) in buses.iter().enumerate() {
            assert_eq!(
                table.bus_segment_count(bus),
                Some(spec.deficit_segments.len())
            );
        }
    }

    #[test]
    fn resolution_is_invariant_to_row_order() {
        let sorted = valid_rows();
        let reversed: Vec<_> = sorted.iter().rev().cloned().collect();
        let mut rotated = sorted.clone();
        rotated.rotate_left(1);

        let expected = resolve(&sorted);
        let expected_bytes = postcard::to_allocvec(&expected).unwrap();
        for rows in [reversed, rotated] {
            let table = resolve(&rows);
            assert_eq!(table, expected);
            assert_eq!(postcard::to_allocvec(&table).unwrap(), expected_bytes);
        }
    }

    #[test]
    fn no_rows_equals_seed() {
        assert_eq!(
            resolve(&[]),
            ResolvedDeficitSegments::seeded(&buses(), 2, 2)
        );
    }
}
