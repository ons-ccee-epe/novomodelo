//! Pre-resolved per-`(bus, stage, block, segment)` deficit curve table.
//!
//! The segment axis is ragged, so it is addressed as `(bus, seg)` and stored
//! through CSR offsets: bus `b`'s segments occupy slots
//! `bus_segment_start[b]..bus_segment_start[b + 1]`, and each bus's count
//! mirrors `Bus::deficit_segments.len()`. The stage layer is dense, indexed
//! `slot * n_stages + stage` and filled from each bus's base curve by
//! [`ResolvedDeficitSegments::seeded`]. The block layer, indexed
//! `(slot * n_stages + stage) * max_blocks + block`, stays empty until the
//! first block write. A table built by [`ResolvedDeficitSegments::empty`] is
//! unseeded and carries no bus.

use crate::{Bus, DeficitSegment};

/// Per-`(bus, stage, block, segment)` deficit segment table.
///
/// # Examples
///
/// ```
/// use novomodelo_core::ResolvedDeficitSegments;
///
/// let empty = ResolvedDeficitSegments::empty();
/// assert!(!empty.is_seeded());
/// assert_eq!(empty.n_buses(), 0);
/// ```
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ResolvedDeficitSegments {
    n_stages: usize,
    max_blocks: usize,
    bus_segment_start: Vec<usize>,
    stage: Vec<DeficitSegment>,
    block: Vec<Option<DeficitSegment>>,
}

impl ResolvedDeficitSegments {
    /// `max_blocks` for a table that takes no block write; every block reads its stage cell.
    pub const NO_BLOCK_LAYER: usize = 0;

    /// Create the unseeded table that carries no bus.
    #[must_use]
    pub fn empty() -> Self {
        Self {
            n_stages: 0,
            max_blocks: 0,
            bus_segment_start: Vec::new(),
            stage: Vec::new(),
            block: Vec::new(),
        }
    }

    /// Create a table holding each bus's base curve at every stage and block.
    #[must_use]
    pub fn seeded(buses: &[Bus], n_stages: usize, max_blocks: usize) -> Self {
        let mut bus_segment_start = Vec::with_capacity(buses.len() + 1);
        bus_segment_start.push(0);
        let mut total = 0usize;
        let mut stage = Vec::new();
        for bus in buses {
            for &segment in &bus.deficit_segments {
                stage.extend(std::iter::repeat_n(segment, n_stages));
            }
            total += bus.deficit_segments.len();
            bus_segment_start.push(total);
        }
        Self {
            n_stages,
            max_blocks,
            bus_segment_start,
            stage,
            block: Vec::new(),
        }
    }

    /// Returns `true` once the table carries a bus axis, even an empty one.
    #[must_use]
    pub fn is_seeded(&self) -> bool {
        !self.bus_segment_start.is_empty()
    }

    /// Return the number of stages in this table.
    #[must_use]
    pub fn n_stages(&self) -> usize {
        self.n_stages
    }

    /// Return the number of buses in this table.
    #[must_use]
    pub fn n_buses(&self) -> usize {
        self.bus_segment_start.len().saturating_sub(1)
    }

    /// Return the segment count of `bus`, or `None` when `bus` is out of range.
    #[must_use]
    pub fn bus_segment_count(&self, bus: usize) -> Option<usize> {
        let &start = self.bus_segment_start.get(bus)?;
        let &end = self.bus_segment_start.get(bus + 1)?;
        Some(end - start)
    }

    fn total_segments(&self) -> usize {
        self.bus_segment_start.last().copied().unwrap_or(0)
    }

    /// Resolve `(bus, seg)` to its CSR slot, or `None` when `bus` is out of
    /// range or `seg` is not one of that bus's segments — never aliasing a
    /// sibling bus's slot.
    fn slot(&self, bus: usize, seg: usize) -> Option<usize> {
        let &start = self.bus_segment_start.get(bus)?;
        let &end = self.bus_segment_start.get(bus + 1)?;
        let slot = start + seg;
        (slot < end).then_some(slot)
    }

    fn stage_flat_index(&self, bus: usize, stage: usize, seg: usize) -> Option<usize> {
        let slot = self.slot(bus, seg)?;
        (stage < self.n_stages).then_some(slot * self.n_stages + stage)
    }

    fn block_flat_index(
        &self,
        bus: usize,
        stage: usize,
        block: usize,
        seg: usize,
    ) -> Option<usize> {
        let stage_flat = self.stage_flat_index(bus, stage, seg)?;
        (block < self.max_blocks).then_some(stage_flat * self.max_blocks + block)
    }

    /// Return a mutable handle to the stage-wide cell, or `None` when any
    /// index is out of range.
    #[inline]
    pub fn stage_segment_mut(
        &mut self,
        bus: usize,
        stage: usize,
        seg: usize,
    ) -> Option<&mut DeficitSegment> {
        let idx = self.stage_flat_index(bus, stage, seg)?;
        self.stage.get_mut(idx)
    }

    /// Return a mutable handle to the per-block cell, growing the block layer
    /// to full size on its first call, or `None` when any index is out of
    /// range.
    #[inline]
    pub fn block_segment_mut(
        &mut self,
        bus: usize,
        stage: usize,
        block: usize,
        seg: usize,
    ) -> Option<&mut Option<DeficitSegment>> {
        let idx = self.block_flat_index(bus, stage, block, seg)?;
        if self.block.is_empty() {
            self.block = vec![None; self.total_segments() * self.n_stages * self.max_blocks];
        }
        self.block.get_mut(idx)
    }

    /// Return the block cell when one is set, else the stage cell.
    #[inline]
    #[must_use]
    pub fn segment_at_block(
        &self,
        bus: usize,
        stage: usize,
        block: usize,
        seg: usize,
    ) -> DeficitSegment {
        let start = self.bus_segment_start[bus];
        debug_assert!(
            seg < self.bus_segment_start[bus + 1] - start,
            "deficit segment {seg} is past the segment count of bus {bus}"
        );
        debug_assert!(
            stage < self.n_stages,
            "stage {stage} is past the {} stages of the deficit table",
            self.n_stages
        );
        let stage_flat = (start + seg) * self.n_stages + stage;
        let from_block = if block < self.max_blocks {
            self.block
                .get(stage_flat * self.max_blocks + block)
                .copied()
                .flatten()
        } else {
            None
        };
        from_block.unwrap_or(self.stage[stage_flat])
    }
}

#[cfg(test)]
mod tests {
    use super::ResolvedDeficitSegments;
    #[cfg(feature = "serde")]
    use crate::ResolvedPenalties;
    use crate::test_support::{BusSpec, make_bus};
    use crate::{Bus, DeficitSegment};

    const N_STAGES: usize = 2;
    const MAX_BLOCKS: usize = 2;

    fn segment(depth_mw: Option<f64>, cost_per_mwh: f64) -> DeficitSegment {
        DeficitSegment {
            depth_mw,
            cost_per_mwh,
        }
    }

    fn buses() -> Vec<Bus> {
        vec![
            make_bus(BusSpec {
                id: 1,
                deficit_segments: vec![segment(Some(100.0), 500.0), segment(None, 1000.0)],
                ..Default::default()
            }),
            make_bus(BusSpec {
                id: 2,
                deficit_segments: vec![
                    segment(Some(50.0), 300.0),
                    segment(Some(150.0), 600.0),
                    segment(None, 900.0),
                ],
                ..Default::default()
            }),
        ]
    }

    fn seeded() -> ResolvedDeficitSegments {
        ResolvedDeficitSegments::seeded(&buses(), N_STAGES, MAX_BLOCKS)
    }

    #[test]
    fn seeded_table_reads_base_curve_everywhere() {
        let buses = buses();
        let table = seeded();
        assert!(table.is_seeded());
        assert_eq!(table.n_buses(), 2);
        assert_eq!(table.n_stages(), N_STAGES);
        assert_eq!(table.bus_segment_count(0), Some(2));
        assert_eq!(table.bus_segment_count(1), Some(3));
        for (bus, spec) in buses.iter().enumerate() {
            for (seg, expected) in spec.deficit_segments.iter().enumerate() {
                for stage in 0..N_STAGES {
                    for block in 0..MAX_BLOCKS {
                        assert_eq!(table.segment_at_block(bus, stage, block, seg), *expected);
                    }
                }
            }
        }
    }

    #[test]
    fn block_write_changes_only_its_block() {
        let buses = buses();
        let mut table = seeded();
        let written = segment(Some(7.0), 42.0);
        *table.block_segment_mut(1, 1, 0, 2).unwrap() = Some(written);
        for (bus, spec) in buses.iter().enumerate() {
            for (seg, base) in spec.deficit_segments.iter().enumerate() {
                for stage in 0..N_STAGES {
                    for block in 0..MAX_BLOCKS {
                        let expected = if (bus, stage, block, seg) == (1, 1, 0, 2) {
                            written
                        } else {
                            *base
                        };
                        assert_eq!(table.segment_at_block(bus, stage, block, seg), expected);
                    }
                }
            }
        }
    }

    #[test]
    fn stage_write_reaches_blocks_without_block_cell() {
        let buses = buses();
        let mut table = seeded();
        let stage_value = segment(Some(11.0), 21.0);
        let block_value = segment(Some(13.0), 23.0);
        *table.block_segment_mut(0, 1, 1, 0).unwrap() = Some(block_value);
        *table.stage_segment_mut(0, 1, 0).unwrap() = stage_value;
        assert_eq!(table.segment_at_block(0, 1, 0, 0), stage_value);
        assert_eq!(table.segment_at_block(0, 1, 1, 0), block_value);
        assert_eq!(
            table.segment_at_block(0, 0, 0, 0),
            buses[0].deficit_segments[0]
        );
        assert_eq!(
            table.segment_at_block(0, 1, 0, 1),
            buses[0].deficit_segments[1]
        );
        assert_eq!(
            table.segment_at_block(1, 1, 0, 0),
            buses[1].deficit_segments[0]
        );
    }

    #[test]
    fn writers_refuse_out_of_range_indices() {
        let mut table = seeded();
        let fresh = seeded();
        let n_buses = table.n_buses();
        let count = table.bus_segment_count(0).unwrap();
        assert!(table.stage_segment_mut(0, 0, count).is_none());
        assert!(table.stage_segment_mut(0, N_STAGES, 0).is_none());
        assert!(table.stage_segment_mut(n_buses, 0, 0).is_none());
        assert!(table.block_segment_mut(0, 0, MAX_BLOCKS, 0).is_none());
        assert!(table.block_segment_mut(0, 0, 0, count).is_none());
        assert!(table.block_segment_mut(0, N_STAGES, 0, 0).is_none());
        assert!(table.block_segment_mut(n_buses, 0, 0, 0).is_none());
        assert_eq!(table, fresh);
    }

    #[test]
    fn empty_table_is_unseeded() {
        let mut table = ResolvedDeficitSegments::empty();
        assert!(!table.is_seeded());
        assert_eq!(table.n_buses(), 0);
        assert_eq!(table.n_stages(), 0);
        assert_eq!(table.bus_segment_count(0), None);
        assert!(table.stage_segment_mut(0, 0, 0).is_none());
        assert!(table.block_segment_mut(0, 0, 0, 0).is_none());
    }

    #[test]
    fn seed_over_zero_buses_is_seeded() {
        let table = ResolvedDeficitSegments::seeded(&[], N_STAGES, MAX_BLOCKS);
        assert!(table.is_seeded());
        assert_eq!(table.n_buses(), 0);
    }

    #[test]
    #[cfg(debug_assertions)]
    fn segment_at_block_panics_in_debug_for_seg_past_bus_count() {
        let table = seeded();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = table.segment_at_block(0, 0, 0, 2);
        }));
        assert!(
            result.is_err(),
            "segment_at_block(0, 0, 0, 2) must panic in debug builds: bus 0 has 2 segments"
        );
    }

    #[test]
    #[cfg(feature = "serde")]
    fn resolved_deficit_segments_postcard_roundtrip() {
        let mut table = seeded();
        *table.stage_segment_mut(0, 0, 1).unwrap() = segment(None, 77.0);
        let written = segment(Some(5.0), 66.0);
        *table.block_segment_mut(1, 1, 1, 0).unwrap() = Some(written);
        let mut original = ResolvedPenalties::empty();
        original.set_bus_deficit(table);

        let bytes = postcard::to_allocvec(&original).unwrap();
        let restored: ResolvedPenalties = postcard::from_bytes(&bytes).unwrap();

        assert_eq!(restored.bus_deficit(), original.bus_deficit());
        assert_eq!(restored.bus_deficit_segment_at_block(1, 1, 1, 0), written);
    }
}
