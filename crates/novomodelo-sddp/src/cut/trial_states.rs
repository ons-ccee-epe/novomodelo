//! Per-slot trial-state rows of a [`CutPool`](super::CutPool).

use std::ops::Range;

/// One row of `dimension` values and one "state stored" flag per pool slot.
#[derive(Debug, Clone)]
pub(crate) struct TrialStates {
    rows: Vec<f64>,
    stored: Vec<bool>,
    dimension: usize,
    stored_count: usize,
}

impl TrialStates {
    pub(crate) fn new(capacity: usize, dimension: usize) -> Self {
        Self {
            rows: vec![0.0; capacity * dimension],
            stored: vec![false; capacity],
            dimension,
            stored_count: 0,
        }
    }

    /// The only place a slot becomes a row.
    fn row_of(slot: usize) -> usize {
        slot
    }

    fn row_range(&self, row: usize) -> Range<usize> {
        row * self.dimension..(row + 1) * self.dimension
    }

    /// Marks `slot` stored and returns its row to be written.
    pub(crate) fn claim(&mut self, slot: usize) -> &mut [f64] {
        let row = Self::row_of(slot);
        debug_assert!(!self.stored[row], "slot {slot} already holds a trial state");
        self.stored[row] = true;
        self.stored_count += 1;
        let range = self.row_range(row);
        &mut self.rows[range]
    }

    pub(crate) fn get(&self, slot: usize) -> Option<&[f64]> {
        let row = Self::row_of(slot);
        self.stored[row].then(|| &self.rows[self.row_range(row)])
    }

    pub(crate) fn stored_count(&self) -> usize {
        self.stored_count
    }

    /// Appends rows for the new slots; every stored row keeps its address.
    pub(crate) fn grow(&mut self, capacity: usize) {
        self.rows.resize(capacity * self.dimension, 0.0);
        self.stored.resize(capacity, false);
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
}
