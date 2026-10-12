//! Per-slot trial-state rows of a [`CutPool`](super::CutPool).

use std::ops::Range;

const NO_ROW: u32 = u32::MAX;

/// A lookup from slot to row, `u32` per slot, over rows of `dimension` values
/// handed out from a free list. A store from [`new`](Self::new) reserves one row
/// per slot; one from [`bounded`](Self::bounded) reserves a fixed number of rows.
#[derive(Debug, Clone)]
pub(crate) struct TrialStates {
    rows: Vec<f64>,
    lookup: Vec<u32>,
    free: Vec<u32>,
    reserved: usize,
    fresh: usize,
    dimension: usize,
    stored_count: usize,
    bounded: bool,
}

impl TrialStates {
    pub(crate) fn new(capacity: usize, dimension: usize) -> Self {
        Self {
            rows: vec![0.0; capacity * dimension],
            lookup: vec![NO_ROW; capacity],
            free: Vec::new(),
            reserved: capacity,
            fresh: 0,
            dimension,
            stored_count: 0,
            bounded: false,
        }
    }

    pub(crate) fn bounded(capacity: usize, rows: usize, dimension: usize) -> Self {
        Self {
            rows: vec![0.0; rows * dimension],
            lookup: vec![NO_ROW; capacity],
            free: Vec::with_capacity(rows),
            reserved: rows,
            fresh: 0,
            dimension,
            stored_count: 0,
            bounded: true,
        }
    }

    /// The only place a slot becomes a row.
    fn row_of(&self, slot: usize) -> Option<u32> {
        let row = self.lookup[slot];
        (row != NO_ROW).then_some(row)
    }

    fn row_range(&self, row: u32) -> Range<usize> {
        let row = row as usize;
        row * self.dimension..(row + 1) * self.dimension
    }

    /// Hands out a row for `slot`, to be written: a released row first, then a
    /// never-used one, and past the reserve a new row is appended.
    pub(crate) fn claim(&mut self, slot: usize) -> &mut [f64] {
        debug_assert!(
            self.row_of(slot).is_none(),
            "slot {slot} already holds a trial state"
        );
        let row = self.free.pop().unwrap_or_else(|| self.fresh_row());
        self.lookup[slot] = row;
        self.stored_count += 1;
        let range = self.row_range(row);
        &mut self.rows[range]
    }

    /// Returns the row of `slot` to the free list; `false` when it held none.
    pub(crate) fn release(&mut self, slot: usize) -> bool {
        let Some(row) = self.row_of(slot) else {
            return false;
        };
        self.lookup[slot] = NO_ROW;
        self.free.push(row);
        self.stored_count -= 1;
        true
    }

    pub(crate) fn get(&self, slot: usize) -> Option<&[f64]> {
        self.row_of(slot).map(|row| &self.rows[self.row_range(row)])
    }

    pub(crate) fn stored_count(&self) -> usize {
        self.stored_count
    }

    /// Extends the lookup to `capacity` slots; every stored row keeps its
    /// address. A store that is not bounded also appends a row per new slot.
    pub(crate) fn grow(&mut self, capacity: usize) {
        self.lookup.resize(capacity, NO_ROW);
        if !self.bounded {
            self.reserve_rows(capacity);
        }
    }

    #[allow(clippy::cast_possible_truncation)]
    fn fresh_row(&mut self) -> u32 {
        if self.fresh == self.reserved {
            self.reserve_rows(self.reserved + 1);
            self.free.reserve(self.reserved - self.free.len());
        }
        debug_assert!(self.fresh < NO_ROW as usize);
        self.fresh += 1;
        (self.fresh - 1) as u32
    }

    fn reserve_rows(&mut self, reserved: usize) {
        self.reserved = reserved;
        self.rows.resize(reserved * self.dimension, 0.0);
    }

    #[cfg(test)]
    pub(crate) fn reserved_rows(&self) -> usize {
        self.reserved
    }
}

#[cfg(test)]
mod tests {
    use super::TrialStates;

    #[test]
    fn a_trial_state_row_is_addressed_by_its_slot() {
        let mut store = TrialStates::new(4, 3);
        assert_eq!(store.rows.len(), 4 * 3, "one row per slot of the capacity");
        assert!((0..4).all(|slot| store.get(slot).is_none()));

        store.claim(2).copy_from_slice(&[1.0, 2.0, 3.0]);
        store.claim(0).copy_from_slice(&[4.0, 5.0, 6.0]);
        assert_eq!(store.get(2), Some(&[1.0, 2.0, 3.0][..]));
        assert_eq!(store.get(0), Some(&[4.0, 5.0, 6.0][..]));
        assert_eq!(store.get(1), None);
        assert_eq!(store.get(3), None);
        assert_eq!(store.stored_count(), 2);

        store.grow(10);
        assert_eq!(store.rows.len(), 10 * 3);
        assert_eq!(store.get(2), Some(&[1.0, 2.0, 3.0][..]));
        assert_eq!(store.get(0), Some(&[4.0, 5.0, 6.0][..]));
        assert!((3..10).all(|slot| store.get(slot).is_none()));
        assert_eq!(store.stored_count(), 2);
    }

    #[allow(clippy::cast_precision_loss)]
    fn values_of(slot: usize) -> [f64; 2] {
        [slot as f64, 100.0 + slot as f64]
    }

    #[test]
    fn a_bounded_store_frees_the_row_of_a_dropped_state() {
        const SLOTS: usize = 10;
        const ROWS: usize = 3;
        let mut store = TrialStates::bounded(SLOTS, ROWS, 2);
        assert_eq!(store.rows.len(), ROWS * 2);
        let free_capacity = store.free.capacity();
        assert!(free_capacity >= ROWS);

        for slot in 0..SLOTS {
            if let Some(dropped) = slot.checked_sub(ROWS) {
                assert!(store.release(dropped), "slot {dropped} held a state");
                assert!(!store.release(dropped), "a released slot holds none");
            }
            store.claim(slot).copy_from_slice(&values_of(slot));

            assert_eq!(store.reserved_rows(), ROWS, "after storing slot {slot}");
            assert_eq!(store.rows.len(), ROWS * 2, "after storing slot {slot}");
            assert_eq!(store.free.capacity(), free_capacity, "slot {slot}");
            assert_eq!(store.stored_count(), (slot + 1).min(ROWS));
            for other in 0..SLOTS {
                let held = other <= slot && slot - other < ROWS;
                assert_eq!(
                    store.get(other),
                    held.then_some(&values_of(other)[..]),
                    "slot {other} after storing slot {slot}"
                );
            }
        }
    }

    #[test]
    fn a_bounded_store_extends_when_more_states_are_held_than_reserved() {
        const ROWS: usize = 2;
        let mut store = TrialStates::bounded(6, ROWS, 2);

        for slot in 0..=ROWS {
            store.claim(slot).copy_from_slice(&values_of(slot));
        }

        assert_eq!(store.reserved_rows(), ROWS + 1);
        assert!(
            store.free.capacity() > ROWS,
            "the free list holds every row"
        );
        assert_eq!(store.stored_count(), ROWS + 1);
        for slot in 0..=ROWS {
            assert_eq!(store.get(slot), Some(&values_of(slot)[..]), "slot {slot}");
        }
        assert_eq!(store.get(ROWS + 1), None);
    }
}
