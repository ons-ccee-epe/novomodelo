//! Per-state ownership record of a [`CutPool`]: a cache of the Level 1
//! ownership function of de Matos, Philpott and Finardi (2015). The owner of a
//! stored trial state is the lowest live slot whose value there equals the live
//! maximum, compared exactly; `takes_over` owns the direction of that
//! comparison. The record is never persisted and never crosses ranks.

use std::ops::Range;

use rayon::prelude::*;

use super::pool::{CutPool, SlotOrigin};
use super::trial_states::TrialStates;
use crate::gemm::gemm_block;

/// Slots per `gemm_block` call, at most the rows its block-shape property pins.
const ROW_BLOCK: usize = 64;

/// States per `gemm_block` call and per parallel task, at most the columns its
/// block-shape property pins.
pub(crate) const M_BLOCK: usize = 8;

const NO_OWNER: u32 = u32::MAX;

/// A state's best value so far and the slot it belongs to.
type Incumbent = Option<(f64, usize)>;

#[derive(Debug, Clone)]
pub(crate) struct Ownership {
    best: Vec<f64>,
    owner: Vec<u32>,
    owned: Vec<u32>,
    scanned: usize,
    expired: usize,
}

struct Scratch {
    states: Vec<f64>,
    values: Vec<f64>,
}

impl Ownership {
    pub(crate) fn new(capacity: usize) -> Self {
        debug_assert!(capacity < NO_OWNER as usize);
        Self {
            best: vec![0.0; capacity],
            owner: vec![NO_OWNER; capacity],
            owned: vec![0; capacity],
            scanned: 0,
            expired: 0,
        }
    }

    pub(crate) fn grow(&mut self, capacity: usize) {
        debug_assert!(capacity < NO_OWNER as usize);
        self.best.resize(capacity, 0.0);
        self.owner.resize(capacity, NO_OWNER);
        self.owned.resize(capacity, 0);
    }

    pub(crate) fn reset(&mut self) {
        self.best.fill(0.0);
        self.owner.fill(NO_OWNER);
        self.owned.fill(0);
        self.scanned = 0;
    }

    pub(crate) fn owner_of(&self, state_slot: usize) -> Option<usize> {
        let owner = self.owner[state_slot];
        (owner != NO_OWNER).then_some(owner as usize)
    }

    pub(crate) fn best_of(&self, state_slot: usize) -> Option<f64> {
        self.owner_of(state_slot).map(|_| self.best[state_slot])
    }

    pub(crate) fn owned_by(&self, slot: usize) -> usize {
        self.owned[slot] as usize
    }

    /// Drops the stored states of the slots from the previous boundary up to
    /// `until`: each takes its count from its owner and returns its row to
    /// `states`. The boundary only moves up.
    pub(crate) fn expire(&mut self, states: &mut TrialStates, until: usize) {
        debug_assert!(
            self.expired <= until,
            "the boundary moved down from {} to {until}",
            self.expired
        );
        for slot in self.expired..until {
            if states.release(slot)
                && let Some(owner) = self.owner_of(slot)
            {
                self.owned[owner] -= 1;
                self.owner[slot] = NO_OWNER;
                self.best[slot] = 0.0;
            }
        }
        self.expired = until;
    }

    /// Brings the record up to date with the slots written since the last call:
    /// each new stored state against every live slot, then each new live slot
    /// against the states stored before. A slot written or a state stored below
    /// the previous populated mark is not seen until `reset` and a new update.
    pub(crate) fn update(&mut self, pool: &CutPool) {
        let populated = pool.populated();
        debug_assert!(
            self.scanned <= populated,
            "scanned mark {} is past populated {populated}",
            self.scanned
        );
        debug_assert!(
            (0..self.scanned)
                .all(|slot| pool.trial_state(slot).is_none() || self.owner_of(slot).is_some()),
            "a state stored below the scanned mark {} has no owner; only `reset` brings it in",
            self.scanned
        );

        let new_states: Vec<(usize, &[f64])> = (self.scanned..populated)
            .filter_map(|slot| Some((slot, pool.trial_state(slot)?)))
            .collect();
        let mut found = vec![None; new_states.len()];
        scan(pool, &new_states, 0..populated, &mut found);
        for (&(slot, _), incumbent) in new_states.iter().zip(&found) {
            debug_assert!(
                is_live(pool, slot),
                "the state stored at slot {slot} is not live"
            );
            if let &Some((best, owner)) = incumbent {
                self.assign(slot, best, owner);
            }
        }

        let old_states: Vec<(usize, &[f64])> = (0..self.scanned)
            .filter(|&slot| self.owner_of(slot).is_some())
            .filter_map(|slot| Some((slot, pool.trial_state(slot)?)))
            .collect();
        let mut incumbents: Vec<Incumbent> = old_states
            .iter()
            .map(|&(slot, _)| self.owner_of(slot).map(|owner| (self.best[slot], owner)))
            .collect();
        scan(pool, &old_states, self.scanned..populated, &mut incumbents);
        for (&(slot, _), incumbent) in old_states.iter().zip(&incumbents) {
            if let &Some((best, owner)) = incumbent
                && self.owner_of(slot) != Some(owner)
            {
                self.assign(slot, best, owner);
            }
        }

        self.scanned = populated;
    }

    #[allow(clippy::cast_possible_truncation)]
    fn assign(&mut self, state_slot: usize, best: f64, owner: usize) {
        if let Some(previous) = self.owner_of(state_slot) {
            self.owned[previous] -= 1;
        }
        self.owned[owner] += 1;
        self.best[state_slot] = best;
        self.owner[state_slot] = owner as u32;
    }
}

/// A scored slot is live whatever its activity flag; an unscored loaded slot is
/// live while active.
fn is_live(pool: &CutPool, slot: usize) -> bool {
    let role = pool.role(slot);
    role.scored || (role.origin == SlotOrigin::Prefix && pool.is_active(slot))
}

/// The only comparison of two cut values: the lowest live slot keeps a tie.
fn takes_over(candidate: f64, incumbent: f64) -> bool {
    candidate > incumbent
}

/// Runs every live slot of `rows`, in ascending order, against each of `states`
/// and leaves the winners in `incumbents`.
fn scan(
    pool: &CutPool,
    states: &[(usize, &[f64])],
    rows: Range<usize>,
    incumbents: &mut [Incumbent],
) {
    let dimension = pool.state_dimension;
    states
        .par_chunks(M_BLOCK)
        .zip(incumbents.par_chunks_mut(M_BLOCK))
        .for_each_init(
            || Scratch {
                states: vec![0.0; M_BLOCK * dimension],
                values: vec![0.0; ROW_BLOCK * M_BLOCK],
            },
            |scratch, (block, incumbents)| scan_block(pool, block, &rows, incumbents, scratch),
        );
}

fn scan_block(
    pool: &CutPool,
    block: &[(usize, &[f64])],
    rows: &Range<usize>,
    incumbents: &mut [Incumbent],
    scratch: &mut Scratch,
) {
    let dimension = pool.state_dimension;
    let m_len = block.len();
    for (j, &(_, state)) in block.iter().enumerate() {
        scratch.states[j * dimension..(j + 1) * dimension].copy_from_slice(state);
    }
    let coefficients = pool.coefficients_prefix();
    let intercepts = pool.intercepts_prefix();
    for start in rows.clone().step_by(ROW_BLOCK) {
        let end = (start + ROW_BLOCK).min(rows.end);
        let k_rows = end - start;
        let values = &mut scratch.values[..k_rows * m_len];
        gemm_block(
            &coefficients[start * dimension..end * dimension],
            &scratch.states[..m_len * dimension],
            k_rows,
            dimension,
            m_len,
            values,
        );
        for slot in (start..end).filter(|&slot| is_live(pool, slot)) {
            for (j, incumbent) in incumbents.iter_mut().enumerate() {
                let candidate = values[(slot - start) * m_len + j] + intercepts[slot];
                if incumbent.is_none_or(|(best, _)| takes_over(candidate, best)) {
                    *incumbent = Some((candidate, slot));
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;
    use std::num::NonZeroU64;

    use novomodelo_io::OwnedPolicyCutRecord;
    use novomodelo_io::test_support::owned_piece_defaults;
    use proptest::prelude::*;
    use proptest::test_runner::RngSeed;

    use super::{M_BLOCK, ROW_BLOCK};
    use crate::cut::pool::{CutPool, SlotOrigin, SlotRole};
    use crate::cut::selection_reference::tests::{Fixture, fixtures};
    use crate::cut::selection_reference::{RefCut, RefRole, active_after, owners, selected, value};
    use crate::gemm::{PINNED_MAX_COLUMNS, PINNED_MAX_ROWS};
    use crate::setup::NodeId;

    /// The slots of a pool, one entry per slot in slot order.
    #[derive(Debug)]
    struct Spec {
        cuts: Vec<RefCut>,
        active: Vec<bool>,
        states: Vec<Option<Vec<f64>>>,
    }

    impl Spec {
        /// The reference states go to the scored slots in slot order.
        fn from_fixture((cuts, states): &Fixture) -> Self {
            let mut unplaced = states.iter();
            let placed = cuts
                .iter()
                .map(|cut| {
                    let scored = matches!(cut.role, RefRole::Generated | RefRole::ScoredPrefix);
                    scored.then(|| unplaced.next().cloned()).flatten()
                })
                .collect();
            assert!(unplaced.next().is_none(), "more states than scored slots");
            Self {
                active: cuts
                    .iter()
                    .map(|cut| !matches!(cut.role, RefRole::StatelessPrefix { active: false }))
                    .collect(),
                cuts: cuts.clone(),
                states: placed,
            }
        }

        fn stored_states(&self) -> Vec<Vec<f64>> {
            self.states.iter().flatten().cloned().collect()
        }

        fn state_slots(&self) -> Vec<usize> {
            (0..self.states.len())
                .filter(|&slot| self.states[slot].is_some())
                .collect()
        }
    }

    fn fixture(name: &str) -> Fixture {
        fixtures()
            .into_iter()
            .find(|(candidate, _)| *candidate == name)
            .map(|(_, fixture)| fixture)
            .expect("a named fixture")
    }

    fn write_slot(pool: &mut CutPool, slot: usize, spec: &Spec) {
        let cut = &spec.cuts[slot];
        let (origin, scored) = match cut.role {
            RefRole::Vacant => return,
            RefRole::Generated => (SlotOrigin::Generated, true),
            RefRole::StatelessPrefix { .. } => (SlotOrigin::Prefix, false),
            RefRole::ScoredPrefix => (SlotOrigin::Prefix, true),
        };
        pool.add_cut(NodeId(0), slot as u64, 0, cut.intercept, &cut.coefficients);
        pool.set_role_for_test(slot, SlotRole { origin, scored });
        pool.set_active(slot as u32, spec.active[slot]);
        if let Some(state) = &spec.states[slot] {
            pool.set_trial_state_for_test(slot, state);
        }
    }

    fn storing_pool(capacity: usize, dimension: usize) -> CutPool {
        let mut pool = CutPool::new(capacity, dimension, 1, 0);
        pool.enable_trial_states();
        pool
    }

    /// Writes the slots in ascending batches, updating the record after each and
    /// growing the pool between some.
    fn reveal(spec: &Spec, batches: &[usize]) -> CutPool {
        let total = spec.cuts.len();
        let mut pool = storing_pool(1, spec.cuts[0].coefficients.len());
        let mut revealed = 0;
        for (batch, &len) in batches.iter().cycle().enumerate() {
            if revealed == total {
                break;
            }
            let end = (revealed + len).min(total);
            if end > pool.capacity {
                pool.grow(end + batch % 3);
            }
            for slot in revealed..end {
                write_slot(&mut pool, slot, spec);
            }
            pool.update_ownership();
            revealed = end;
        }
        pool
    }

    fn snapshot(pool: &CutPool) -> Vec<(Option<usize>, Option<u64>, usize)> {
        (0..pool.capacity)
            .map(|slot| {
                (
                    pool.state_owner(slot),
                    pool.state_best_value(slot).map(f64::to_bits),
                    pool.owned_states(slot),
                )
            })
            .collect()
    }

    fn stored_state_bits(pool: &CutPool) -> Vec<Option<Vec<u64>>> {
        (0..pool.populated())
            .map(|slot| {
                pool.trial_state(slot)
                    .map(|state| state.iter().map(|value| value.to_bits()).collect())
            })
            .collect()
    }

    fn rebuilt(pool: &CutPool) -> CutPool {
        let mut rebuilt = pool.clone();
        rebuilt.rebuild_ownership();
        rebuilt
    }

    fn assert_matches_reference(pool: &CutPool, spec: &Spec, label: &str) {
        let states = spec.stored_states();
        let state_slots = spec.state_slots();
        let expected = owners(&spec.cuts, &states);
        for ((&slot, state), owner) in state_slots.iter().zip(&states).zip(&expected) {
            assert_eq!(
                pool.state_owner(slot),
                *owner,
                "{label}: owner of the state at slot {slot}"
            );
            assert_eq!(
                pool.state_best_value(slot),
                owner.map(|owner| value(&spec.cuts[owner], state)),
                "{label}: best value of the state at slot {slot}"
            );
        }
        for slot in 0..pool.capacity {
            let owned = expected
                .iter()
                .filter(|&&owner| owner == Some(slot))
                .count();
            assert_eq!(
                pool.owned_states(slot),
                owned,
                "{label}: states owned by slot {slot}"
            );
            if !state_slots.contains(&slot) {
                assert_eq!(pool.state_owner(slot), None, "{label}: slot {slot}");
            }
        }
    }

    #[test]
    fn the_record_reproduces_every_reference_fixture() {
        for (name, fixture) in fixtures() {
            let spec = Spec::from_fixture(&fixture);
            let mut pool = storing_pool(spec.cuts.len(), spec.cuts[0].coefficients.len());
            for slot in 0..spec.cuts.len() {
                write_slot(&mut pool, slot, &spec);
            }
            pool.update_ownership();
            assert_matches_reference(&pool, &spec, name);
        }
    }

    /// derived: at the state 2 the four `figure_four` cuts are worth `10 - 8 = 2`,
    /// `6.5 - 3 = 3.5`, `5 - 3 = 2` and `4 - 1 = 3`, so the second cut owns the
    /// state although it was deactivated before the state arrived.
    #[test]
    fn an_inactive_scored_cut_wins_a_new_state() {
        let spec = Spec::from_fixture(&fixture("figure_four_with_a_new_state"));
        let mut pool = storing_pool(4, 1);
        for slot in 0..3 {
            write_slot(&mut pool, slot, &spec);
        }
        pool.update_ownership();
        pool.set_active(1, false);

        write_slot(&mut pool, 3, &spec);
        pool.update_ownership();

        assert_eq!(pool.state_owner(3), Some(1));
        assert_eq!(pool.state_best_value(3), Some(3.5));
        assert_matches_reference(&pool, &spec, "figure_four_with_a_new_state");
    }

    /// derived: the lines `x`, `-x`, the flat `4` and `5 - x` are worth `4` and
    /// `-4` at the state 4, `-4` and `4` at the state -4. The flat line ties both
    /// owners and takes neither state; `5 - x` is worth `9` at -4, `1` at 4, and
    /// takes only the state -4.
    #[test]
    fn a_new_cut_takes_a_state_only_when_strictly_higher() {
        let line = |intercept, slope| RefCut {
            intercept,
            coefficients: vec![slope],
            role: RefRole::Generated,
        };
        let spec = Spec {
            cuts: vec![
                line(0.0, 1.0),
                line(0.0, -1.0),
                line(4.0, 0.0),
                line(5.0, -1.0),
            ],
            active: vec![true; 4],
            states: [4.0, -4.0, 100.0, -100.0].map(|x| Some(vec![x])).to_vec(),
        };
        let mut pool = storing_pool(4, 1);
        for slot in 0..2 {
            write_slot(&mut pool, slot, &spec);
        }
        pool.update_ownership();
        assert_eq!(
            (pool.state_owner(0), pool.state_owner(1)),
            (Some(0), Some(1))
        );

        for slot in 2..4 {
            write_slot(&mut pool, slot, &spec);
        }
        pool.update_ownership();

        assert_eq!(pool.state_owner(0), Some(0), "the flat line ties the owner");
        assert_eq!(pool.state_owner(1), Some(3), "5 - x exceeds the owner");
        assert_eq!(pool.state_best_value(1), Some(9.0));
        assert_matches_reference(&pool, &spec, "two updates");
    }

    #[derive(Debug, Clone, Copy)]
    enum Kind {
        StatelessActive,
        StatelessInactive,
        ScoredWithState,
        ScoredWithoutState,
        Generated { active: bool },
        Vacant,
    }

    #[derive(Debug)]
    struct Draw {
        kinds: Vec<Kind>,
        cuts: Vec<(f64, Vec<f64>)>,
        states: Vec<Vec<f64>>,
        batches: Vec<usize>,
    }

    impl Draw {
        fn spec(&self) -> Spec {
            let role = |kind| match kind {
                Kind::StatelessActive => RefRole::StatelessPrefix { active: true },
                Kind::StatelessInactive => RefRole::StatelessPrefix { active: false },
                Kind::ScoredWithState | Kind::ScoredWithoutState => RefRole::ScoredPrefix,
                Kind::Generated { .. } => RefRole::Generated,
                Kind::Vacant => RefRole::Vacant,
            };
            Spec {
                cuts: self
                    .kinds
                    .iter()
                    .zip(&self.cuts)
                    .map(|(&kind, (intercept, coefficients))| RefCut {
                        intercept: *intercept,
                        coefficients: coefficients.clone(),
                        role: role(kind),
                    })
                    .collect(),
                active: self
                    .kinds
                    .iter()
                    .map(|kind| {
                        !matches!(
                            kind,
                            Kind::StatelessInactive | Kind::Generated { active: false }
                        )
                    })
                    .collect(),
                states: self
                    .kinds
                    .iter()
                    .zip(&self.states)
                    .map(|(kind, state)| {
                        matches!(kind, Kind::ScoredWithState | Kind::Generated { .. })
                            .then(|| state.clone())
                    })
                    .collect(),
            }
        }
    }

    fn small_cut(dimension: usize) -> impl Strategy<Value = (f64, Vec<f64>)> {
        (
            (-6..=6i32).prop_map(f64::from),
            prop::collection::vec((-3..=3i32).prop_map(f64::from), dimension),
        )
    }

    fn small_state(dimension: usize) -> impl Strategy<Value = Vec<f64>> {
        prop::collection::vec((-3..=3i32).prop_map(f64::from), dimension)
    }

    prop_compose! {
        /// Integer-valued cuts and states from small ranges; each cut is a
        /// repeat of one of three palette cuts or a fresh draw, so ties are
        /// frequent.
        fn exact_draw()(
            dimension in 1..=3usize,
            kinds in (
                prop::collection::vec(
                    prop::sample::select(vec![
                        Kind::StatelessActive,
                        Kind::StatelessInactive,
                        Kind::ScoredWithState,
                        Kind::ScoredWithoutState,
                    ]),
                    0..=6,
                ),
                prop::collection::vec(
                    prop_oneof![
                        3 => Just(Kind::Generated { active: true }),
                        1 => Just(Kind::Generated { active: false }),
                        1 => Just(Kind::Vacant),
                    ],
                    2..=40,
                ),
            )
                .prop_map(|(prefix, tail)| [prefix, tail].concat()),
            batches in prop::collection::vec(1..=9usize, 1..=6),
        )(
            palette in prop::collection::vec(small_cut(dimension), 3),
            fresh in prop::collection::vec(small_cut(dimension), kinds.len()),
            picks in prop::collection::vec(prop::option::of(0..3usize), kinds.len()),
            states in prop::collection::vec(small_state(dimension), kinds.len()),
            kinds in Just(kinds),
            batches in Just(batches),
        ) -> Draw {
            let cuts = picks
                .iter()
                .zip(fresh)
                .map(|(pick, fresh)| pick.map_or(fresh, |pick| palette[pick].clone()))
                .collect();
            Draw { kinds, cuts, states, batches }
        }
    }

    fn inexact_value() -> impl Strategy<Value = f64> {
        (any::<bool>(), 1000..10_000_i32, -6..=2_i32).prop_map(|(negative, mantissa, exponent)| {
            let magnitude = f64::from(mantissa) * 10_f64.powi(exponent);
            if negative { -magnitude } else { magnitude }
        })
    }

    prop_compose! {
        fn inexact_draw()(
            dimension in prop::sample::select(vec![3usize, 257, 300, 600]),
            slots in 2..=24usize,
            batches in prop::collection::vec(1..=6usize, 1..=4),
        )(
            cuts in prop::collection::vec(
                (inexact_value(), prop::collection::vec(inexact_value(), dimension)),
                slots,
            ),
            states in prop::collection::vec(
                prop::collection::vec(inexact_value(), dimension),
                slots,
            ),
            batches in Just(batches),
        ) -> (Spec, Vec<usize>) {
            let spec = Spec {
                active: vec![true; cuts.len()],
                states: states.into_iter().map(Some).collect(),
                cuts: cuts
                    .into_iter()
                    .map(|(intercept, coefficients)| RefCut {
                        intercept,
                        coefficients,
                        role: RefRole::Generated,
                    })
                    .collect(),
            };
            (spec, batches)
        }
    }

    #[derive(Debug)]
    struct WindowedDraw {
        draw: Draw,
        stride: usize,
        check_frequency: u64,
        lookback: u64,
    }

    prop_compose! {
        fn windowed_draw()(
            draw in exact_draw(),
            stride in 1..=3usize,
            check_frequency in 1..=3u64,
            lookback in 1..=4u64,
        ) -> WindowedDraw {
            WindowedDraw { draw, stride, check_frequency, lookback }
        }
    }

    /// Reveals the draw one iteration at a time, `stride` slots each, after its
    /// prefix kinds loaded as records of another run; every `check_frequency`
    /// iterations the pool expires, updates and applies the derived flags, and
    /// is compared with the reference rule on the states of the window. A clone
    /// that updates before it expires must end in the same record, states and
    /// derived flags.
    ///
    /// The states of the window are computed from the rule alone: a generated
    /// state is inside at a run of iteration `k` when its generating iteration
    /// exceeds `k - lookback`, and a loaded one while `k < lookback`.
    #[allow(clippy::too_many_lines)]
    fn assert_window_follows_the_reference(windowed: &WindowedDraw) {
        let WindowedDraw {
            draw,
            stride,
            check_frequency,
            lookback,
        } = windowed;
        let (stride, check_frequency, lookback) = (*stride, *check_frequency, *lookback);
        let spec = draw.spec();
        let loaded = draw
            .kinds
            .iter()
            .take_while(|kind| !matches!(kind, Kind::Generated { .. } | Kind::Vacant))
            .count();
        let iterations = (spec.cuts.len() - loaded).div_ceil(stride) as u64;
        let records: Vec<OwnedPolicyCutRecord> = (0..loaded)
            .map(|slot| OwnedPolicyCutRecord {
                slot_index: slot as u32,
                iteration: 1000,
                intercept: spec.cuts[slot].intercept,
                coefficients: spec.cuts[slot].coefficients.clone(),
                is_active: spec.active[slot],
                ..owned_piece_defaults()
            })
            .collect();

        let dimension = spec.cuts[0].coefficients.len();
        let mut pool = CutPool::new_with_warm_start(dimension, stride as u32, iterations, &records);
        pool.set_iteration_base(1);
        pool.enable_trial_states_within(
            NonZeroU64::new(lookback).expect("a lookback"),
            check_frequency,
        );
        let reserve = pool.reserved_trial_rows();
        for slot in 0..loaded {
            if matches!(spec.cuts[slot].role, RefRole::ScoredPrefix) {
                pool.set_role_for_test(
                    slot,
                    SlotRole {
                        origin: SlotOrigin::Prefix,
                        scored: true,
                    },
                );
            }
            if let Some(state) = &spec.states[slot] {
                pool.set_trial_state_for_test(slot, state);
            }
        }
        let loaded_states = pool.stored_trial_states();
        let mut peak = loaded_states;

        for iteration in 1..=iterations {
            for pass in 0..stride {
                let slot = loaded + (iteration as usize - 1) * stride + pass;
                let Some(&Kind::Generated { active }) = draw.kinds.get(slot) else {
                    continue;
                };
                pool.add_cut(
                    NodeId(0),
                    iteration,
                    pass as u32,
                    spec.cuts[slot].intercept,
                    &spec.cuts[slot].coefficients,
                );
                pool.set_active(slot as u32, active);
                let state = spec.states[slot].as_ref().expect("a generated state");
                pool.set_trial_state_for_test(slot, state);
                peak = peak.max(pool.stored_trial_states());
            }
            if iteration % check_frequency != 0 {
                continue;
            }
            let inside = |slot: usize| {
                if slot < loaded {
                    iteration < lookback
                } else {
                    ((slot - loaded) / stride) as u64 + 1 + lookback > iteration
                }
            };
            let revealed = (loaded + iteration as usize * stride).min(spec.cuts.len());
            let populated = (0..revealed)
                .rev()
                .find(|&slot| !matches!(draw.kinds[slot], Kind::Vacant))
                .map_or(0, |slot| slot + 1)
                .max(loaded);
            let window = Spec {
                cuts: spec.cuts[..populated].to_vec(),
                active: spec.active[..populated].to_vec(),
                states: (0..populated)
                    .map(|slot| spec.states[slot].clone().filter(|_| inside(slot)))
                    .collect(),
            };
            let was_active: Vec<bool> = (0..populated).map(|slot| pool.is_active(slot)).collect();
            let label = format!("run at iteration {iteration}");

            let mut swapped = pool.clone();
            pool.expire_trial_states(iteration);
            pool.update_ownership();
            swapped.update_ownership();
            swapped.expire_trial_states(iteration);
            assert_eq!(
                snapshot(&swapped),
                snapshot(&pool),
                "{label}: swapped order"
            );
            assert_eq!(
                stored_state_bits(&swapped),
                stored_state_bits(&pool),
                "{label}: swapped order, stored states"
            );
            let updates = pool.ownership_activity_updates();
            assert_eq!(
                swapped.ownership_activity_updates(),
                updates,
                "{label}: swapped order, activity derivation"
            );
            let owning = selected(&window.cuts, &window.stored_states());
            let is_scored = |slot: usize| {
                matches!(
                    window.cuts[slot].role,
                    RefRole::Generated | RefRole::ScoredPrefix
                )
            };
            let flagged = |wanted: bool| -> Vec<u32> {
                (0..populated)
                    .filter(|&slot| {
                        is_scored(slot)
                            && was_active[slot] != wanted
                            && owning.contains(&slot) == wanted
                    })
                    .map(|slot| slot as u32)
                    .collect()
            };
            assert_eq!(
                updates.updates,
                flagged(false),
                "run at {iteration}: updates"
            );
            assert_eq!(
                updates.reactivations,
                flagged(true),
                "run at {iteration}: reactivations"
            );
            pool.apply_updates(&updates);

            assert_eq!(snapshot(&pool), snapshot(&rebuilt(&pool)), "{label}");
            assert_matches_reference(&pool, &window, &label);
            let active: BTreeSet<usize> = (0..pool.populated())
                .filter(|&slot| pool.is_active(slot))
                .collect();
            assert_eq!(
                active,
                active_after(&window.cuts, &window.stored_states()),
                "{label}: active set"
            );
            for slot in 0..populated {
                assert_eq!(
                    pool.role(slot).scored,
                    is_scored(slot),
                    "{label}: slot {slot}"
                );
            }
            assert_eq!(pool.reserved_trial_rows(), reserve.max(peak), "{label}");
            if loaded_states == 0 {
                assert!(peak <= reserve, "{label}: the store extended");
            }
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 256,
            rng_seed: RngSeed::Fixed(42),
            ..ProptestConfig::default()
        })]

        #[test]
        fn windowed_record_equals_rebuild_and_reference(windowed in windowed_draw()) {
            assert_window_follows_the_reference(&windowed);
        }

        #[test]
        fn incremental_record_equals_rebuild_and_reference(draw in exact_draw()) {
            let spec = draw.spec();
            let pool = reveal(&spec, &draw.batches);
            prop_assert_eq!(snapshot(&pool), snapshot(&rebuilt(&pool)));
            assert_matches_reference(&pool, &spec, "incremental");
        }

        #[test]
        fn incremental_record_equals_rebuild_on_inexact_values(
            (spec, batches) in inexact_draw(),
        ) {
            let pool = reveal(&spec, &batches);
            prop_assert_eq!(snapshot(&pool), snapshot(&rebuilt(&pool)));
        }
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "below the scanned mark")]
    fn a_state_stored_below_the_scanned_mark_panics_in_debug() {
        let mut pool = storing_pool(3, 1);
        for iteration in 0..3 {
            pool.add_cut(NodeId(0), iteration, 0, 1.0, &[0.0]);
        }
        pool.set_trial_state_for_test(0, &[0.0]);
        pool.update_ownership();
        pool.set_trial_state_for_test(1, &[1.0]);
        pool.update_ownership();
    }

    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn ownership_record_is_identical_across_thread_counts() {
        let slots = 150;
        assert!(slots > ROW_BLOCK.max(M_BLOCK));
        let fraction = |i: usize, modulus: usize, step: f64, shift: f64| {
            (i * 7 % modulus) as f64 * step + shift
        };
        let spec = Spec {
            cuts: (0..slots)
                .map(|i| RefCut {
                    intercept: fraction(i, 13, 0.37, -2.0),
                    coefficients: vec![
                        fraction(i, 5, 0.23, -0.4),
                        fraction(i + 1, 7, 0.11, -0.3),
                        fraction(i + 2, 4, 0.19, 0.0),
                    ],
                    role: RefRole::Generated,
                })
                .collect(),
            active: vec![true; slots],
            states: (0..slots)
                .map(|i| {
                    Some(vec![
                        fraction(i, 11, 0.21, -1.0),
                        fraction(i + 3, 9, 0.13, 0.0),
                        fraction(i + 5, 7, 0.31, -0.7),
                    ])
                })
                .collect(),
        };
        let run = |threads: usize| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("a rayon pool")
                .install(|| {
                    let pool = reveal(&spec, &[ROW_BLOCK + 26, 60]);
                    assert_eq!(snapshot(&pool), snapshot(&rebuilt(&pool)));
                    snapshot(&pool)
                })
        };

        let one = run(1);
        let distinct_owners: BTreeSet<_> = one.iter().filter_map(|(owner, _, _)| *owner).collect();
        assert!(
            distinct_owners.len() > 1,
            "the record must have several owners"
        );
        assert_eq!(one, run(4), "1 against 4 threads");
        assert_eq!(one, run(8), "1 against 8 threads");
    }

    #[test]
    #[allow(clippy::assertions_on_constants)]
    fn ownership_blocks_stay_inside_the_pinned_gemm_envelope() {
        assert!(ROW_BLOCK <= PINNED_MAX_ROWS);
        assert!(M_BLOCK <= PINNED_MAX_COLUMNS);
    }
}
