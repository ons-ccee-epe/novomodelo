//! Bit-level fingerprint of a trained cut pool, the power counters of a
//! selection-on gate, and the envelope invariant read off a trained run.
//!
//! Every float is stored as its bit pattern so `-0.0` and NaN payloads compare
//! by bits. Wall-clock fields and solver statistics are deliberately absent.
//! The ownership record is read through the public readers of `CutPool` only.

use std::collections::BTreeMap;
use std::fmt::Debug;
use std::ops::RangeInclusive;

use novomodelo_core::TrainingEvent;
use novomodelo_sddp::cut::{CutPool, SlotOrigin, SlotRole};
use novomodelo_sddp::setup::StageIdx;
use novomodelo_sddp::{StudySetup, TrainingResult};

/// One pool's cuts, one entry per populated slot in slot order.
#[derive(Debug, PartialEq)]
pub struct PoolCuts {
    /// `f64::to_bits` of each slot's intercept.
    pub intercept_bits: Vec<u64>,
    /// `f64::to_bits` of each slot's coefficient row, concatenated in slot order.
    pub coefficient_bits: Vec<u64>,
    /// Each slot's activity flag.
    pub active: Vec<bool>,
}

/// One pool's per-slot generation provenance, in slot order.
#[derive(Debug, PartialEq)]
pub struct PoolProvenance {
    /// Iteration that generated the cut.
    pub iteration_generated: Vec<u64>,
    /// Forward pass that generated the cut.
    pub forward_pass_index: Vec<u32>,
}

/// One pool's per-slot binding counters, in slot order.
#[derive(Debug, PartialEq)]
pub struct PoolBinding {
    /// Cumulative number of iterations the cut was binding.
    pub active_count: Vec<u64>,
    /// Most recent iteration the cut was binding.
    pub last_active_iter: Vec<u64>,
}

/// One pool's slot roles, stored trial states and ownership record, in slot
/// order. A pool that stores no trial state reads `None` and 0 throughout.
#[derive(Debug, PartialEq)]
pub struct PoolOwnership {
    /// Each slot's origin and scored flag.
    pub roles: Vec<SlotRole>,
    /// `f64::to_bits` of each coordinate of the slot's stored trial state.
    pub trial_state_bits: Vec<Option<Vec<u64>>>,
    /// `f64::to_bits` of the owner's value at the slot's stored state.
    pub best_value_bits: Vec<Option<u64>>,
    /// Slot that owns the slot's stored state.
    pub owner: Vec<Option<usize>>,
    /// Number of stored states the slot's cut owns.
    pub owned_states: Vec<usize>,
}

/// The bounds a training run reports.
#[derive(Debug, PartialEq)]
pub struct Bounds {
    /// `f64::to_bits` of each iteration's lower bound.
    pub lower_bound_bits: Vec<u64>,
    /// `f64::to_bits` of the final upper bound.
    pub final_ub_bits: u64,
}

/// Everything a determinism gate compares between two training runs; one
/// `Vec` entry per pool of `StudySetup::fcf`.
#[derive(Debug, PartialEq)]
pub struct SelectionFingerprint {
    /// Cut values and activity flags.
    pub cuts: Vec<PoolCuts>,
    /// Generation provenance.
    pub provenance: Vec<PoolProvenance>,
    /// Binding counters.
    pub binding: Vec<PoolBinding>,
    /// Slot roles, stored trial states and the ownership record.
    pub ownership: Vec<PoolOwnership>,
    /// Reported bounds.
    pub bounds: Bounds,
}

impl SelectionFingerprint {
    /// Reads the fingerprint off a trained `setup` and its `result`.
    pub fn capture(setup: &StudySetup, result: &TrainingResult) -> Self {
        let mut cuts = Vec::new();
        let mut provenance = Vec::new();
        let mut binding = Vec::new();
        let mut ownership = Vec::new();
        for pool in &setup.fcf.pools {
            let slots = || 0..pool.populated();
            cuts.push(PoolCuts {
                intercept_bits: slots().map(|s| pool.intercept(s).to_bits()).collect(),
                coefficient_bits: slots()
                    .flat_map(|s| pool.coefficient_row(s).iter().map(|c| c.to_bits()))
                    .collect(),
                active: slots().map(|s| pool.is_active(s)).collect(),
            });
            provenance.push(PoolProvenance {
                iteration_generated: slots()
                    .map(|s| pool.metadata(s).iteration_generated)
                    .collect(),
                forward_pass_index: slots()
                    .map(|s| pool.metadata(s).forward_pass_index)
                    .collect(),
            });
            binding.push(PoolBinding {
                active_count: slots().map(|s| pool.metadata(s).active_count).collect(),
                last_active_iter: slots().map(|s| pool.metadata(s).last_active_iter).collect(),
            });
            ownership.push(PoolOwnership {
                roles: slots().map(|s| pool.role(s)).collect(),
                trial_state_bits: slots().map(|s| pool.trial_state(s).map(bits_of)).collect(),
                best_value_bits: slots()
                    .map(|s| pool.state_best_value(s).map(f64::to_bits))
                    .collect(),
                owner: slots().map(|s| pool.state_owner(s)).collect(),
                owned_states: slots().map(|s| pool.owned_states(s)).collect(),
            });
        }
        Self {
            cuts,
            provenance,
            binding,
            ownership,
            bounds: Bounds {
                lower_bound_bits: result
                    .lower_bound_history
                    .iter()
                    .map(|lb| lb.to_bits())
                    .collect(),
                final_ub_bits: result.final_ub.to_bits(),
            },
        }
    }

    /// Asserts `self == other`, naming the first differing field.
    ///
    /// # Panics
    ///
    /// Panics with `context` and the field, its pool, the position in it and
    /// both values.
    pub fn assert_matches(&self, other: &Self, context: &str) {
        assert!(
            self == other,
            "{context}: {}",
            self.first_difference(other)
                .unwrap_or_else(|| "the fingerprints differ in an unnamed field".to_string())
        );
    }

    /// Asserts that every pool of `self` and `other` agrees in every field,
    /// naming the first difference. The bounds are not compared.
    ///
    /// # Panics
    ///
    /// Panics with `context` and the field, its pool, the position in it and
    /// both values.
    pub fn assert_pools_match(&self, other: &Self, context: &str) {
        let difference = self.first_pool_difference(other);
        assert!(
            difference.is_none(),
            "{context}: {}",
            difference.unwrap_or_default()
        );
    }

    /// The first field in which `self` and `other` differ, with its pool, its
    /// position in the field and both values.
    fn first_difference(&self, other: &Self) -> Option<String> {
        self.first_pool_difference(other).or_else(|| {
            [
                mismatch(
                    "lower_bound_bits",
                    &self.bounds.lower_bound_bits,
                    &other.bounds.lower_bound_bits,
                ),
                mismatch(
                    "final_ub_bits",
                    &[self.bounds.final_ub_bits],
                    &[other.bounds.final_ub_bits],
                ),
            ]
            .into_iter()
            .flatten()
            .next()
        })
    }

    /// [`first_difference`](Self::first_difference) over the pools only.
    fn first_pool_difference(&self, other: &Self) -> Option<String> {
        if self.cuts.len() != other.cuts.len() {
            return Some(format!(
                "pool count: {} vs {}",
                self.cuts.len(),
                other.cuts.len()
            ));
        }
        let in_pool = |pool: usize| {
            let (cuts, their_cuts) = (&self.cuts[pool], &other.cuts[pool]);
            let (origin, their_origin) = (&self.provenance[pool], &other.provenance[pool]);
            let (binding, their_binding) = (&self.binding[pool], &other.binding[pool]);
            let (record, their_record) = (&self.ownership[pool], &other.ownership[pool]);
            [
                mismatch(
                    "intercept_bits",
                    &cuts.intercept_bits,
                    &their_cuts.intercept_bits,
                ),
                mismatch(
                    "coefficient_bits",
                    &cuts.coefficient_bits,
                    &their_cuts.coefficient_bits,
                ),
                mismatch("active", &cuts.active, &their_cuts.active),
                mismatch(
                    "iteration_generated",
                    &origin.iteration_generated,
                    &their_origin.iteration_generated,
                ),
                mismatch(
                    "forward_pass_index",
                    &origin.forward_pass_index,
                    &their_origin.forward_pass_index,
                ),
                mismatch(
                    "active_count",
                    &binding.active_count,
                    &their_binding.active_count,
                ),
                mismatch(
                    "last_active_iter",
                    &binding.last_active_iter,
                    &their_binding.last_active_iter,
                ),
                mismatch("roles", &record.roles, &their_record.roles),
                mismatch(
                    "trial_state_bits",
                    &record.trial_state_bits,
                    &their_record.trial_state_bits,
                ),
                mismatch(
                    "best_value_bits",
                    &record.best_value_bits,
                    &their_record.best_value_bits,
                ),
                mismatch("owner", &record.owner, &their_record.owner),
                mismatch(
                    "owned_states",
                    &record.owned_states,
                    &their_record.owned_states,
                ),
            ]
            .into_iter()
            .flatten()
            .next()
            .map(|difference| format!("pool {pool}: {difference}"))
        };
        (0..self.cuts.len()).find_map(in_pool)
    }
}

fn bits_of(values: &[f64]) -> Vec<u64> {
    values.iter().map(|value| value.to_bits()).collect()
}

fn mismatch<T: PartialEq + Debug>(field: &str, ours: &[T], theirs: &[T]) -> Option<String> {
    if ours.len() != theirs.len() {
        return Some(format!(
            "{field}: length {} vs {}",
            ours.len(),
            theirs.len()
        ));
    }
    ours.iter()
        .zip(theirs)
        .position(|(a, b)| a != b)
        .map(|at| format!("{field}[{at}]: {:?} vs {:?}", ours[at], theirs[at]))
}

/// How much selection work a training run did; a gate with no deactivation
/// compares nothing the selection method could have changed.
#[derive(Debug, PartialEq)]
pub struct SelectionPower {
    /// Rows deactivated at interior nodes.
    pub deactivations: u64,
    /// Rows reactivated at interior nodes.
    pub reactivations: u64,
    /// Rows evicted by budget enforcement.
    pub budget_evictions: u64,
}

impl SelectionPower {
    /// Sums the counters over a training run's events. The root and terminal
    /// pools are never selected, so their records carry no selection counts and
    /// the sum over every record of a `PolicyPoolSummary` is the interior sum.
    pub fn from_events(events: &[TrainingEvent]) -> Self {
        let mut power = Self {
            deactivations: 0,
            reactivations: 0,
            budget_evictions: 0,
        };
        for event in events {
            match event {
                TrainingEvent::PolicyPoolSummary { pools, .. } => {
                    for record in pools {
                        power.deactivations += u64::from(record.rows_deactivated.unwrap_or(0));
                        power.reactivations += u64::from(record.rows_reactivated.unwrap_or(0));
                    }
                }
                TrainingEvent::PolicyBudgetEnforcementComplete { rows_evicted, .. } => {
                    power.budget_evictions += u64::from(*rows_evicted);
                }
                _ => {}
            }
        }
        power
    }
}

/// The row-management events of `iterations`, in the order they were emitted.
pub fn row_management_events(
    events: &[TrainingEvent],
    iterations: RangeInclusive<u64>,
) -> Vec<TrainingEvent> {
    events
        .iter()
        .filter(|event| match event {
            TrainingEvent::PolicySelectionComplete { iteration, .. }
            | TrainingEvent::PolicyBudgetEnforcementComplete { iteration, .. }
            | TrainingEvent::PolicyPoolSummary { iteration, .. } => iterations.contains(iteration),
            _ => false,
        })
        .cloned()
        .collect()
}

/// Whether a selection pass ran at the run's last iteration. The ownership
/// record is brought up to date by a selection pass only, so otherwise the
/// states of the later iterations have no owner yet.
pub fn selection_ran_at_the_last_iteration(
    events: &[TrainingEvent],
    result: &TrainingResult,
) -> bool {
    events.iter().any(|event| {
        matches!(
            event,
            TrainingEvent::PolicySelectionComplete { iteration, .. }
                if *iteration == result.iterations
        )
    })
}

/// Pools of the nodes whose cuts selection scores: stage 1 or later, with a
/// successor.
pub fn interior_pool_ids(setup: &StudySetup) -> Vec<usize> {
    let graph = &setup.inputs.node_graph;
    graph
        .nodes
        .iter()
        .zip(graph.successors.iter())
        .filter(|(node, successors)| node.stage >= StageIdx(1) && !successors.is_empty())
        .map(|(node, _)| node.pool_id)
        .collect()
}

/// What the ownership part of a fingerprint holds, summed over the interior
/// pools. A gate with no state owned by another slot compares a record in
/// which every cut won its own state, so no cut lost its last one.
#[derive(Debug, PartialEq)]
pub struct OwnershipPower {
    /// Slots that hold a trial state.
    pub stored_states: usize,
    /// Stored states whose owner is not the slot they are stored at.
    pub states_owned_by_another_slot: usize,
    /// Scored slots that hold no trial state: a window dropped it.
    pub scored_slots_without_a_state: usize,
    /// The scored slots without a state that a load wrote.
    pub loaded_scored_slots_without_a_state: usize,
}

impl OwnershipPower {
    /// Reads the counters off a trained `setup`.
    pub fn of(setup: &StudySetup) -> Self {
        let mut power = Self {
            stored_states: 0,
            states_owned_by_another_slot: 0,
            scored_slots_without_a_state: 0,
            loaded_scored_slots_without_a_state: 0,
        };
        for pool_id in interior_pool_ids(setup) {
            let pool = &setup.fcf.pools[pool_id];
            for slot in 0..pool.populated() {
                if pool.trial_state(slot).is_some() {
                    power.stored_states += 1;
                    if pool.state_owner(slot).is_some_and(|owner| owner != slot) {
                        power.states_owned_by_another_slot += 1;
                    }
                } else if pool.role(slot).scored {
                    power.scored_slots_without_a_state += 1;
                    if pool.is_prefix_slot(slot) {
                        power.loaded_scored_slots_without_a_state += 1;
                    }
                }
            }
        }
        power
    }
}

/// A scored slot whatever its flag, or a loaded slot while active: the cuts
/// that compete for a stored state.
fn competes(pool: &CutPool, slot: usize) -> bool {
    let role = pool.role(slot);
    role.scored || (role.origin == SlotOrigin::Prefix && pool.is_active(slot))
}

/// The value of the cut at `slot` at `state`, summed left to right, and a bound
/// on how far another summation order of the same terms can round from it.
///
/// derived: the value has `n = dimension + 1` terms, and a sum of `n` rounded
/// products evaluated in any order lies within `gamma_n * magnitude` of the
/// exact value, `gamma_n = n*u / (1 - n*u)` with `u = f64::EPSILON / 2` and
/// `magnitude = |intercept| + sum |coefficient * coordinate|` (Higham (2002),
/// Accuracy and Stability of Numerical Algorithms, section 3.1). Two orders are
/// therefore within `2 * gamma_n * magnitude` of each other, and the bound
/// returned, `2 * n * f64::EPSILON * magnitude`, is twice that to first order,
/// which covers the rounding of `magnitude` itself.
fn value_and_rounding_bound(pool: &CutPool, slot: usize, state: &[f64]) -> (f64, f64) {
    let mut value = pool.intercept(slot);
    let mut magnitude = value.abs();
    for (coefficient, coordinate) in pool.coefficient_row(slot).iter().zip(state) {
        let term = coefficient * coordinate;
        value += term;
        magnitude += term.abs();
    }
    #[allow(clippy::cast_precision_loss)]
    let terms = (state.len() + 1) as f64;
    (value, 2.0 * terms * f64::EPSILON * magnitude)
}

fn same_cut_bits(pool: &CutPool, a: usize, b: usize) -> bool {
    pool.intercept(a).to_bits() == pool.intercept(b).to_bits()
        && bits_of(pool.coefficient_row(a)) == bits_of(pool.coefficient_row(b))
}

/// Asserts, on every interior pool of a run whose last iteration ran selection,
/// that the recorded owner of each stored state attains the maximum of the
/// competing cuts there, from cut values this function computes itself.
///
/// The record's values come from a blocked kernel, so a value is compared with
/// the plain sum within the rounding bound of
/// [`value_and_rounding_bound`], never by bits. The tie direction is exact
/// where it can be: a cut with the owner's bits has the owner's value at every
/// state, so none may compete from a lower slot.
///
/// # Panics
///
/// Panics naming the run, the pool, the state's slot, the owner and the
/// offending cut with both values.
pub fn assert_owners_attain_the_maximum(setup: &StudySetup, run: &str) {
    for pool_id in interior_pool_ids(setup) {
        let pool = &setup.fcf.pools[pool_id];
        let mut states_won = vec![0_usize; pool.populated()];
        for slot in 0..pool.populated() {
            let Some(state) = pool.trial_state(slot) else {
                assert_eq!(
                    (pool.state_owner(slot), pool.state_best_value(slot)),
                    (None, None),
                    "{run}: pool {pool_id} slot {slot} holds no state and has an owner"
                );
                continue;
            };
            let at = format!("{run}: pool {pool_id}, state at slot {slot}");
            let owner = pool
                .state_owner(slot)
                .unwrap_or_else(|| panic!("{at}: the state has no owner"));
            assert!(
                owner < pool.populated() && competes(pool, owner),
                "{at}: owner {owner} is not a competing cut ({:?}, active {})",
                pool.role(owner),
                pool.is_active(owner)
            );
            states_won[owner] += 1;

            let (owner_value, owner_bound) = value_and_rounding_bound(pool, owner, state);
            let recorded = pool
                .state_best_value(slot)
                .unwrap_or_else(|| panic!("{at}: an owned state has no best value"));
            assert!(
                (recorded - owner_value).abs() <= owner_bound,
                "{at}: the recorded best value {recorded:e} is not owner {owner}'s value \
                 {owner_value:e} (rounding bound {owner_bound:e})"
            );
            for rival in (0..pool.populated()).filter(|&rival| competes(pool, rival)) {
                let (rival_value, rival_bound) = value_and_rounding_bound(pool, rival, state);
                assert!(
                    rival_value <= owner_value + owner_bound + rival_bound,
                    "{at}: owner {owner} has value {owner_value:e} and competing slot {rival} \
                     has {rival_value:e}, above it by more than the rounding bound {:e}",
                    owner_bound + rival_bound
                );
                assert!(
                    rival >= owner || !same_cut_bits(pool, rival, owner),
                    "{at}: owner {owner} is bit-identical to the competing lower slot {rival}, \
                     which keeps a tie"
                );
            }
        }
        for (slot, &count) in states_won.iter().enumerate() {
            assert_eq!(
                pool.owned_states(slot),
                count,
                "{run}: pool {pool_id} slot {slot}: the owned count differs from the number of \
                 states that name it their owner"
            );
        }
    }
}

/// Asserts, on every interior pool, that a scored cut is active exactly when
/// some stored state names it its owner. It holds right after a selection pass
/// in which no budget evicted a cut.
///
/// # Panics
///
/// Panics naming the run, the pool and the slot.
pub fn assert_active_scored_cuts_are_the_owners(setup: &StudySetup, run: &str) {
    for pool_id in interior_pool_ids(setup) {
        let pool = &setup.fcf.pools[pool_id];
        let mut owns = vec![false; pool.populated()];
        for owner in (0..pool.populated()).filter_map(|slot| pool.state_owner(slot)) {
            owns[owner] = true;
        }
        for (slot, &owns_a_state) in owns.iter().enumerate() {
            if pool.role(slot).scored {
                assert_eq!(
                    pool.is_active(slot),
                    owns_a_state,
                    "{run}: pool {pool_id} scored slot {slot}: active {} and owner of a stored \
                     state {owns_a_state}",
                    pool.is_active(slot)
                );
            } else {
                assert!(
                    !owns_a_state || pool.is_active(slot),
                    "{run}: pool {pool_id} slot {slot} owns a stored state and is inactive"
                );
            }
        }
    }
}

/// Asserts that every interior pool holds two bit-identical cuts and that, of
/// the cuts bit-identical to each other, none above the lowest slot is active.
///
/// derived: bit-identical cuts have equal values at every state, so only the
/// lowest slot among them can own one; it holds right after a selection pass
/// in which no budget evicted a cut, when the active scored cuts are the owners.
///
/// # Panics
///
/// Panics naming the run, the pool and the two slots.
pub fn assert_identical_cuts_keep_their_lowest_slot(setup: &StudySetup, run: &str) {
    for pool_id in interior_pool_ids(setup) {
        let pool = &setup.fcf.pools[pool_id];
        let mut lowest_slot_of: BTreeMap<Vec<u64>, usize> = BTreeMap::new();
        let mut duplicates = 0_usize;
        for slot in (0..pool.populated()).filter(|&slot| pool.role(slot).scored) {
            let mut bits = bits_of(pool.coefficient_row(slot));
            bits.push(pool.intercept(slot).to_bits());
            let lowest = *lowest_slot_of.entry(bits).or_insert(slot);
            if lowest != slot {
                duplicates += 1;
                assert!(
                    !pool.is_active(slot),
                    "{run}: pool {pool_id} slot {slot} is active and bit-identical to the lower \
                     slot {lowest}"
                );
            }
        }
        assert!(
            duplicates > 0,
            "{run}: pool {pool_id} holds no two bit-identical cuts, so the tie direction is not \
             exercised"
        );
    }
}
