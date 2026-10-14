//! Cut selection strategies for controlling cut pool growth during SDDP
//! training.
//!
//! # Level 1
//!
//! `Level1` is the Level 1 strategy of de Matos, Philpott and Finardi (2015),
//! section 4.2: each trial state has one owner, the cut with the highest value
//! there, and a selection run keeps the cuts that own a state.
//!
//! **Stored states.** For the pool of each node of stage 1 or later that has a
//! successor: the trial state of every cut this run generated for that pool,
//! one per cut, on every rank; such a cut is scored. A loaded cut is unscored
//! and has no state: no load writes a scored role or a state. Every stored
//! state of the run is kept, so the memory of a run grows with its cut count,
//! unless the strategy carries a window of `w` iterations
//! ([`lookback_iterations`](CutSelectionStrategy::Level1::lookback_iterations)).
//! A selection run of iteration `k` then drops the states of the cuts generated
//! before iteration `k - w + 1`, a loaded cut counting as iteration 0, so the
//! scored set is the trial states of the cuts generated in the last `w`
//! iterations, and the memory is bounded by the window
//! ([`CutPool::enable_trial_states_within`]). Only states leave: a cut whose
//! state left keeps its role, stays live and competes at the states that
//! remain.
//!
//! **Rule.** The live cuts of such a pool are its scored cuts, whatever their
//! activity flag, and its unscored loaded cuts while active. The owner of a
//! stored state is the lowest slot among the live cuts whose value at the state
//! equals the live maximum. Values are compared exactly: there is no tolerance,
//! and a cut of the current iteration is judged like any other. A selection run
//! makes a scored cut active when it owns a stored state and inactive when it
//! owns none ([`CutPool::ownership_activity_updates`]); an unscored cut keeps
//! its flag. A later cut takes a state only when strictly higher, as in the
//! paper's update; the paper names no rule for a tie at a new state, and the
//! lowest slot is this implementation's choice.
//!
//! **Invariant.** Right after a selection run in which the active-cut budget
//! evicts nothing, the active cuts of such a pool attain the maximum of its
//! live cuts at every stored state, which with a window are the states inside
//! it. The budget can deactivate a cut that owns states; the next selection run
//! reactivates it, and the budget is applied again in that iteration.
//!
//! **Departures from the paper.** The paper selects in every stage problem and
//! updates the owners each time a cut is added. Here the root pool is never
//! selected, and selection runs every `check_frequency` iterations.
//!
//! **Convergence.** The paper presents Level 1 as a heuristic for its dominance
//! test and states no convergence result for it: a cut Level 1 does not select
//! "might be binding at some other feasible point" (its Fig. 4). None is
//! claimed here.
//!
//! [`CutPool::enable_trial_states_within`]: crate::cut::CutPool::enable_trial_states_within
//! [`CutPool::ownership_activity_updates`]: crate::cut::CutPool::ownership_activity_updates
//!
//! # Usage
//!
//! ```rust
//! use novomodelo_sddp::cut_selection::CutSelectionStrategy;
//!
//! let strategy = CutSelectionStrategy::Level1 {
//!     check_frequency: 5,
//!     lookback_iterations: None,
//! };
//!
//! // Should run at multiples of check_frequency (excluding 0).
//! assert!(!strategy.should_run(0));
//! assert!(!strategy.should_run(3));
//! assert!(strategy.should_run(5));
//! assert!(strategy.should_run(10));
//! ```

use std::num::{NonZeroU32, NonZeroU64};

use crate::setup::NodeId;

use novomodelo_io::config::RowSelectionConfig;

// ---------------------------------------------------------------------------
// CutMetadata
// ---------------------------------------------------------------------------

/// Per-cut tracking metadata for cut selection strategies.
#[derive(Debug, Clone)]
pub struct CutMetadata {
    /// Iteration at which this cut was generated (1-based). The active-cut
    /// budget never evicts a cut of the current iteration; `Level1` judges it
    /// like any other.
    pub iteration_generated: u64,

    /// Forward pass index that generated this cut.
    pub forward_pass_index: u32,

    /// Declared id (`NodeGraph::node_ids`) of the node that generated this cut;
    /// carried onto the MPI cut wire ([`crate::cut::wire::serialize_cut`]) so a
    /// shared-leaf pool's records stay distinguishable by generating node. Not
    /// read by the value-based selection logic.
    pub node: NodeId,

    /// Binding solves counted since the slot was written; a resumed cut keeps
    /// the checkpoint's count, and a warm start or boundary load restarts it at
    /// 0. Stays 0 while no reader of binding activity is configured.
    pub active_count: u64,

    /// Iteration of the last recorded binding, on this run's clock. Starts at
    /// the generating iteration for a cut of this run, at the checkpoint's
    /// value for a resumed cut (the generating iteration when the checkpoint
    /// carries no counters), and at 0 for a cut loaded by a warm start or as a
    /// boundary record. Read by the budget key and the `dynamic` seed window.
    pub last_active_iter: u64,
}

// ---------------------------------------------------------------------------
// CutActivityUpdates
// ---------------------------------------------------------------------------

/// The activity changes one selection pass makes to a pool.
///
/// The caller applies them to the activity bitmap via [`CutPool::apply_updates`].
///
/// [`CutPool::apply_updates`]: crate::cut::CutPool::apply_updates
#[derive(Debug, Clone, PartialEq)]
pub struct CutActivityUpdates {
    /// Slot indices to deactivate.
    pub updates: Vec<u32>,
    /// Slot indices to reactivate.
    pub reactivations: Vec<u32>,
}

// ---------------------------------------------------------------------------
// CutSelectionStrategy
// ---------------------------------------------------------------------------

/// Cut selection strategy for controlling cut pool growth during SDDP training.
///
/// One strategy is active for the entire training run; all stages use it.
///
/// The strategy is sent by postcard inside the broadcast config, within one run
/// of one binary, by variant index and field position; names are not encoded.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum CutSelectionStrategy {
    /// Level 1 of de Matos, Philpott and Finardi (2015): each stored trial state
    /// has one owner, the lowest slot among the cuts with the highest value
    /// there, compared exactly, and a selection run keeps the scored cuts
    /// that own a state. The [module documentation](crate::cut::cut_selection)
    /// states the rule, the stored states, the invariant, the departures from
    /// the paper and what is said of convergence. With a window, the scored set
    /// of each interior pool is the trial states of the cuts generated in the
    /// last `w` iterations, and the invariant holds at those states.
    Level1 {
        /// Number of iterations between selection runs. Must be > 0.
        check_frequency: u64,

        /// `Some(w)` keeps the trial states of the last `w` iterations of the
        /// run, `None` every one; set from
        /// `training.cut_selection.selection.lookback_iterations`.
        lookback_iterations: Option<NonZeroU64>,
    },

    /// Lazy row generation inside each LP solve: this software's own method, not
    /// the Dynamic Cut Selection of de Matos, Philpott and Finardi (2015). A
    /// solve starts from a seed of active cuts and appends the candidates it
    /// violates until none is; it never deactivates a cut, so [`should_run`]
    /// always returns `false`.
    ///
    /// [`should_run`]: CutSelectionStrategy::should_run
    Dynamic {
        /// Candidate-recency window. `None` ⇒ ∞ (every active cut a candidate;
        /// exactness-preserving). `Some(n)` ⇒ only cuts generated within the
        /// last `n` iterations, and loaded cuts, are candidates — deliberately
        /// NOT exact: older cuts are never added even when violated.
        candidate_recency: Option<u32>,

        /// Seed window: a lazy solve starts with the active cuts whose
        /// [`CutMetadata::last_active_iter`] is at most `seed_window`
        /// iterations old, and the cuts of the current iteration.
        seed_window: u32,

        /// Number of most-violated candidate cuts considered per round. Must be
        /// `>= 1`.
        max_added_per_round: u32,

        /// Relative violation factor: a candidate cut is added when its value
        /// exceeds the future-cost value by more than the factor times the
        /// larger of the two magnitudes. `None` takes the primal feasibility
        /// tolerance of each phase's solver profile; a value is finite, `> 0`
        /// and `< 1`.
        violation_tolerance: Option<f64>,

        /// First training iteration at which the lazy loop becomes active.
        start_iteration: u64,
    },
}

impl CutSelectionStrategy {
    /// Determine whether cut selection should run at the given iteration.
    ///
    /// `true` if `iteration > 0` and `iteration` is a multiple of the variant's
    /// `check_frequency`. Always `false` for `Dynamic`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use novomodelo_sddp::cut_selection::CutSelectionStrategy;
    ///
    /// let s = CutSelectionStrategy::Level1 {
    ///     check_frequency: 5,
    ///     lookback_iterations: None,
    /// };
    /// assert!(!s.should_run(0));
    /// assert!(!s.should_run(3));
    /// assert!(s.should_run(5));
    /// assert!(s.should_run(10));
    /// ```
    #[must_use]
    pub fn should_run(&self, iteration: u64) -> bool {
        let freq = match self {
            Self::Level1 {
                check_frequency, ..
            } => *check_frequency,
            Self::Dynamic { .. } => return false,
        };
        iteration > 0 && iteration.is_multiple_of(freq)
    }
}

// ---------------------------------------------------------------------------
// Config parsing
// ---------------------------------------------------------------------------

/// Validate a selection tolerance: finite and `>= 0`, or finite and `> 0` when
/// `strict`.
///
/// # Errors
///
/// Returns `Err(String)` naming `key` and the offending value.
fn validate_tolerance(key: &str, value: f64, strict: bool) -> Result<f64, String> {
    let in_range = value.is_finite() && if strict { value > 0.0 } else { value >= 0.0 };
    if !in_range {
        let bound = if strict { ">" } else { ">=" };
        return Err(format!(
            "cut_selection.{key} must be finite and {bound} 0 (got {value})"
        ));
    }
    Ok(value)
}

/// Parse a [`RowSelectionConfig`] into an optional
/// [`CutSelectionStrategy`].
///
/// Returns `None` when `selection` is absent (row selection disabled).
/// Otherwise maps the chosen [`SelectionMethod`] onto the matching
/// [`CutSelectionStrategy`], validating the variant's numeric constraints.
///
/// # Errors
///
/// Returns `Err(String)` when `row_activity_tolerance` (checked even with no
/// method) is not finite and `>= 0`, or when a method's numeric constraint is
/// violated: for `level1`, a `lookback_iterations` below `check_frequency`; for
/// `dynamic`, a `violation_tolerance` that is not finite and `> 0`, or is
/// `>= 1`.
///
/// [`SelectionMethod`]: novomodelo_io::config::SelectionMethod
pub fn parse_cut_selection_config(
    config: &RowSelectionConfig,
) -> Result<Option<CutSelectionStrategy>, String> {
    use novomodelo_io::config::SelectionMethod;

    validate_tolerance(
        "row_activity_tolerance",
        config.row_activity_tolerance,
        false,
    )?;

    let Some(selection) = config.selection.as_ref() else {
        return Ok(None);
    };

    match *selection {
        SelectionMethod::Level1 {
            check_frequency,
            lookback_iterations,
        } => {
            // A shorter window drops the states of the iterations between two
            // selection runs before any run scores them.
            if let Some(window) = lookback_iterations
                && window.get() < check_frequency.get()
            {
                return Err(format!(
                    "cut_selection.lookback_iterations ({window}) must be >= \
                     cut_selection.check_frequency ({check_frequency})"
                ));
            }
            Ok(Some(CutSelectionStrategy::Level1 {
                check_frequency: u64::from(check_frequency.get()),
                lookback_iterations: lookback_iterations.map(NonZeroU64::from),
            }))
        }
        SelectionMethod::Dynamic {
            start_iteration,
            seed_window,
            candidate_recency,
            max_added_per_round,
            violation_tolerance,
        } => {
            let violation_tolerance = violation_tolerance
                .map(|value| validate_tolerance("violation_tolerance", value, true))
                .transpose()?;
            if let Some(value) = violation_tolerance
                && value >= 1.0
            {
                return Err(format!(
                    "cut_selection.violation_tolerance must be < 1 (got {value})"
                ));
            }

            Ok(Some(CutSelectionStrategy::Dynamic {
                candidate_recency: candidate_recency.map(NonZeroU32::get),
                seed_window,
                max_added_per_round: max_added_per_round.get(),
                violation_tolerance,
                start_iteration: u64::from(start_iteration.get()),
            }))
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;
    use std::num::{NonZeroU32, NonZeroU64};
    use std::ops::RangeInclusive;

    use super::parse_cut_selection_config;
    use super::{CutActivityUpdates, CutMetadata, CutSelectionStrategy, NodeId};
    use crate::cut::ownership::M_BLOCK;
    use crate::cut::selection_reference::tests::fixtures;
    use crate::cut::selection_reference::{RefCut, RefRole, active_after};
    use crate::cut::{CutPool, SlotOrigin, SlotRole};
    use novomodelo_io::config::{RowSelectionConfig, SelectionMethod};
    use novomodelo_io::test_support::owned_piece_defaults;
    use proptest::prelude::*;
    use proptest::test_runner::RngSeed;

    fn positive(n: u32) -> NonZeroU32 {
        NonZeroU32::new(n).expect("a positive count")
    }

    fn make_meta(active_count: u64, last_active_iter: u64) -> CutMetadata {
        CutMetadata {
            iteration_generated: 1,
            forward_pass_index: 0,
            node: NodeId(0),
            active_count,
            last_active_iter,
        }
    }

    #[test]
    fn should_run_false_at_zero() {
        let s = CutSelectionStrategy::Level1 {
            check_frequency: 5,
            lookback_iterations: None,
        };
        assert!(!s.should_run(0));
    }

    #[test]
    fn should_run_false_between_multiples() {
        let s = CutSelectionStrategy::Level1 {
            check_frequency: 5,
            lookback_iterations: None,
        };
        assert!(!s.should_run(3));
        assert!(!s.should_run(7));
    }

    #[test]
    fn should_run_true_at_multiples() {
        let s = CutSelectionStrategy::Level1 {
            check_frequency: 5,
            lookback_iterations: None,
        };
        assert!(s.should_run(5));
        assert!(s.should_run(10));
        assert!(s.should_run(15));
    }

    #[test]
    fn cut_metadata_derives_debug_and_clone() {
        let meta = make_meta(5, 10);
        let cloned = meta.clone();
        assert_eq!(cloned.active_count, 5);
        assert!(!format!("{meta:?}").is_empty());
    }

    // -----------------------------------------------------------------------
    // parse_cut_selection_config tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_disabled_default() {
        let cfg = RowSelectionConfig::default();
        let result = parse_cut_selection_config(&cfg);
        assert!(result.is_ok());
        assert!(
            result.unwrap().is_none(),
            "default config (no selection) must produce None (disabled)"
        );
    }

    #[test]
    fn test_parse_level1() {
        let cfg = RowSelectionConfig {
            selection: Some(SelectionMethod::Level1 {
                check_frequency: positive(5),
                lookback_iterations: None,
            }),
            ..RowSelectionConfig::default()
        };
        let strategy = parse_cut_selection_config(&cfg)
            .expect("level1 must parse")
            .expect("must produce Some for level1");
        assert!(
            matches!(
                strategy,
                CutSelectionStrategy::Level1 {
                    check_frequency: 5,
                    lookback_iterations: None,
                }
            ),
            "unexpected variant: {strategy:?}"
        );
    }

    fn level1_with_lookback(check_frequency: u32, lookback: Option<u32>) -> RowSelectionConfig {
        RowSelectionConfig {
            selection: Some(SelectionMethod::Level1 {
                check_frequency: positive(check_frequency),
                lookback_iterations: lookback.and_then(NonZeroU32::new),
            }),
            ..RowSelectionConfig::default()
        }
    }

    fn parsed_lookback(cfg: &RowSelectionConfig) -> Option<NonZeroU64> {
        match parse_cut_selection_config(cfg) {
            Ok(Some(CutSelectionStrategy::Level1 {
                lookback_iterations,
                ..
            })) => lookback_iterations,
            other => panic!("expected a level1 strategy, got {other:?}"),
        }
    }

    #[test]
    fn parse_level1_lookback_absent_gives_none() {
        assert_eq!(parsed_lookback(&level1_with_lookback(2, None)), None);
    }

    #[test]
    fn parse_level1_lookback_value_gives_some() {
        assert_eq!(
            parsed_lookback(&level1_with_lookback(2, Some(7))),
            NonZeroU64::new(7)
        );
    }

    #[test]
    fn parse_level1_lookback_equal_to_check_frequency_is_admitted() {
        assert_eq!(
            parsed_lookback(&level1_with_lookback(3, Some(3))),
            NonZeroU64::new(3)
        );
    }

    #[test]
    fn parse_level1_lookback_below_check_frequency_is_rejected() {
        let msg = parse_cut_selection_config(&level1_with_lookback(3, Some(2)))
            .expect_err("a window below check_frequency must be rejected");
        assert_eq!(
            msg,
            "cut_selection.lookback_iterations (2) must be >= cut_selection.check_frequency (3)"
        );
    }

    /// One config per tolerance key, each carrying `value`.
    fn configs_with_tolerance(value: f64) -> Vec<(&'static str, RowSelectionConfig)> {
        let with_method = |selection| RowSelectionConfig {
            selection: Some(selection),
            ..RowSelectionConfig::default()
        };
        vec![
            (
                "row_activity_tolerance",
                RowSelectionConfig {
                    row_activity_tolerance: value,
                    ..RowSelectionConfig::default()
                },
            ),
            (
                "violation_tolerance",
                with_method(SelectionMethod::Dynamic {
                    start_iteration: positive(2),
                    seed_window: 5,
                    candidate_recency: None,
                    max_added_per_round: positive(10),
                    violation_tolerance: Some(value),
                }),
            ),
        ]
    }

    /// The rejection message of every config of `configs_with_tolerance(value)`
    /// that carries `key`.
    fn rejections(key: &str, value: f64) -> Vec<String> {
        configs_with_tolerance(value)
            .into_iter()
            .filter(|(config_key, _)| *config_key == key)
            .map(|(_, cfg)| {
                parse_cut_selection_config(&cfg)
                    .expect_err(&format!("{key} = {value} must be rejected"))
            })
            .collect()
    }

    #[test]
    fn parse_accepts_the_default_row_selection() {
        for row_activity_tolerance in [RowSelectionConfig::default().row_activity_tolerance, 0.0] {
            let cfg = RowSelectionConfig {
                row_activity_tolerance,
                ..RowSelectionConfig::default()
            };
            let strategy =
                parse_cut_selection_config(&cfg).expect("the default row selection must parse");
            assert!(
                strategy.is_none(),
                "no selection method means selection is off: {strategy:?}"
            );
        }
    }

    #[test]
    fn parse_accepts_a_zero_tolerance_where_zero_is_legal() {
        for (key, cfg) in configs_with_tolerance(0.0) {
            if key == "violation_tolerance" {
                continue;
            }
            parse_cut_selection_config(&cfg)
                .unwrap_or_else(|msg| panic!("{key} = 0 must parse, got: {msg}"));
        }
    }

    #[test]
    fn parse_rejects_a_nan_violation_tolerance() {
        let messages = rejections("violation_tolerance", f64::NAN);
        assert_eq!(messages.len(), 1);
        assert!(
            messages[0].contains("violation_tolerance must be finite and > 0 (got NaN)"),
            "error must name the key, the rule and the value, got: {}",
            messages[0]
        );
    }

    #[test]
    fn parse_rejects_a_negative_row_activity_tolerance_without_a_method() {
        let messages = rejections("row_activity_tolerance", -1.0);
        assert_eq!(messages.len(), 1);
        assert!(
            messages[0].contains("row_activity_tolerance must be finite and >= 0 (got -1)"),
            "error must name the key, the rule and the value, got: {}",
            messages[0]
        );
    }

    #[test]
    fn parse_rejects_an_infinite_tolerance_of_every_kind() {
        for value in [f64::INFINITY, f64::NEG_INFINITY] {
            for (key, cfg) in configs_with_tolerance(value) {
                let msg = parse_cut_selection_config(&cfg)
                    .expect_err(&format!("{key} = {value} must be rejected"));
                assert!(
                    msg.contains(key) && msg.contains(&format!("(got {value})")),
                    "error must name the key and the value, got: {msg}"
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // Level1 closed-form fixtures
    // -----------------------------------------------------------------------

    /// The cuts of Fig. 4 of de Matos, Philpott and Finardi (2015): slot 1 is
    /// undominated yet never the highest at the states 1, 3 and 4, where the
    /// values are 6, 5, 3.5, 3.5, then -2, 2, 0.5, 2.5, then -6, 0.5, -1, 2.
    const FIGURE_FOUR: [(f64, f64); 4] = [(10.0, -4.0), (6.5, -1.5), (5.0, -1.5), (4.0, -0.5)];

    /// Writes the `(intercept, slope)` cuts in slot order, one per iteration, and
    /// stores `states` at the first slots of a pool that already stores states.
    fn fill_scored_pool(mut pool: CutPool, lines: &[(f64, f64)], states: &[f64]) -> CutPool {
        for (slot, &(intercept, slope)) in lines.iter().enumerate() {
            pool.add_cut(NodeId(0), slot as u64, 0, intercept, &[slope]);
        }
        for (slot, &state) in states.iter().enumerate() {
            pool.set_trial_state_for_test(slot, &[state]);
        }
        pool
    }

    /// One-dimensional pool of scored `(intercept, slope)` cuts in slot order,
    /// with `states` stored at the first slots.
    fn scored_pool_from_lines(lines: &[(f64, f64)], states: &[f64]) -> CutPool {
        let mut pool = CutPool::new(lines.len(), 1, 1, 0);
        pool.enable_trial_states();
        fill_scored_pool(pool, lines, states)
    }

    /// One `level1` selection run: the record is brought up to date and the
    /// updates it implies are applied and returned.
    fn select_level1(pool: &mut CutPool) -> CutActivityUpdates {
        pool.update_ownership();
        let updates = pool.ownership_activity_updates();
        pool.apply_updates(&updates);
        updates
    }

    fn active_slots(pool: &CutPool) -> BTreeSet<usize> {
        (0..pool.populated())
            .filter(|&slot| pool.is_active(slot))
            .collect()
    }

    #[test]
    fn level1_selects_the_two_steep_cuts_of_paper_figure_four() {
        let mut pool = scored_pool_from_lines(&FIGURE_FOUR, &[1.0, 3.0, 4.0]);
        let result = select_level1(&mut pool);
        assert_eq!(result.updates, vec![1, 2]);
        assert_eq!(result.reactivations, Vec::<u32>::new());
    }

    #[test]
    fn level1_loses_the_middle_cut_of_paper_figure_four_at_an_unvisited_state() {
        let mut pool = scored_pool_from_lines(&FIGURE_FOUR, &[1.0, 3.0, 4.0]);
        select_level1(&mut pool);
        let max_of_all_four = FIGURE_FOUR
            .iter()
            .map(|&(intercept, slope)| intercept + slope * 2.0)
            .fold(f64::NEG_INFINITY, f64::max);
        assert_eq!(max_of_all_four, 3.5);
        assert_eq!(pool.evaluate_at_state(&[2.0]), 3.0);
    }

    #[test]
    fn level1_reactivates_the_owner_of_a_new_state() {
        let mut pool = scored_pool_from_lines(&FIGURE_FOUR, &[1.0, 3.0, 4.0]);
        select_level1(&mut pool);
        // At 2 the values are 2, 3.5, 2, 3: the inactive slot 1 owns the new state.
        pool.set_trial_state_for_test(3, &[2.0]);
        // A state stored below the scanned mark is seen only by a rebuild.
        pool.rebuild_ownership();
        let second = select_level1(&mut pool);
        assert_eq!(second.updates, Vec::<u32>::new());
        assert_eq!(second.reactivations, vec![1]);
    }

    /// derived: lookback 3, one cut per iteration, the four cuts written before the
    /// first run. At iteration 2 (`r = 3`) the window starts at slot 0 and holds
    /// the states 1, 3 and 4. At iteration 3 (`r = 4`) it starts at slot 1, so the
    /// state 1 leaves: the values at 3 and 4 are won by slot 3, the values at 2 are
    /// 2, 3.5, 2, 3 and are won by slot 1, and slot 0 owns none.
    #[test]
    fn level1_drops_the_owner_of_a_state_that_left_the_window() {
        let mut pool = CutPool::new(4, 1, 1, 0);
        pool.enable_trial_states_within(NonZeroU64::new(3).unwrap(), 1);
        let mut pool = fill_scored_pool(pool, &FIGURE_FOUR, &[1.0, 3.0, 4.0]);
        pool.expire_trial_states(2);
        let first = select_level1(&mut pool);
        assert_eq!(first.updates, vec![1, 2]);
        assert_eq!(first.reactivations, Vec::<u32>::new());

        pool.set_trial_state_for_test(3, &[2.0]);
        // The state stored below the scanned mark needs a rebuild, run before the
        // expiry so that the record holds the state 1 that the expiry removes.
        pool.rebuild_ownership();
        assert_eq!(pool.owned_states(0), 1);
        pool.expire_trial_states(3);
        assert_eq!((pool.owned_states(0), pool.state_owner(0)), (0, None));
        assert_eq!(pool.trial_state(0), None);
        assert_eq!(pool.stored_trial_states(), 3);
        let second = select_level1(&mut pool);
        assert_eq!(second.updates, vec![0]);
        assert_eq!(second.reactivations, vec![1]);
    }

    /// derived: the two loaded cuts are worth 5 - x and 2x, the generated ones
    /// 1 + x/2, 2 + x/2 and 0. The loaded cut of iteration 40 is read against this
    /// run's clock: it is at iteration 0 of it, so with lookback 2 its state is
    /// inside at iteration 1 and out from iteration 2.
    /// - iteration 1: values at 0 are 5, 0, 1 (owner 0), at 4 are 1, 8, 3
    ///   (owner 1);
    /// - iteration 2 (`r = 2`, start slot 2): the loaded state leaves; values at 4
    ///   are 1, 8, 3, 4 and at 5 are 0, 10, 3.5, 4.5 (owner 1 at both);
    /// - iteration 3 (start slot 3): values at 5 are 0, 10, 3.5, 4.5, 0 (owner 1),
    ///   at -10 are 15, -20, -4, -3, 0 (owner 0).
    #[test]
    fn a_loaded_scored_cut_competes_only_at_this_runs_states_once_its_state_left_the_window() {
        let loaded = |intercept, slope, is_active| novomodelo_io::OwnedPolicyCutRecord {
            slot_index: 0,
            iteration: 40,
            intercept,
            coefficients: vec![slope],
            is_active,
            ..owned_piece_defaults()
        };
        let mut pool = CutPool::new_with_warm_start(
            1,
            1,
            3,
            &[loaded(5.0, -1.0, true), loaded(0.0, 2.0, false)],
        );
        pool.set_iteration_base(1);
        pool.enable_trial_states_within(NonZeroU64::new(2).unwrap(), 1);
        for slot in 0..2 {
            pool.set_role_for_test(
                slot,
                SlotRole {
                    origin: SlotOrigin::Prefix,
                    scored: true,
                },
            );
        }
        pool.set_trial_state_for_test(0, &[0.0]);
        let run = |pool: &mut CutPool, iteration, intercept, slope, state| {
            pool.add_cut(NodeId(0), iteration, 0, intercept, &[slope]);
            let slot = pool.populated() - 1;
            pool.set_trial_state_for_test(slot, &[state]);
            pool.expire_trial_states(iteration);
            select_level1(pool)
        };

        let first = run(&mut pool, 1, 1.0, 0.5, 4.0);
        assert_eq!(first.updates, vec![2]);
        assert_eq!(first.reactivations, vec![1]);

        let second = run(&mut pool, 2, 2.0, 0.5, 5.0);
        assert_eq!(pool.trial_state(0), None);
        assert_eq!(second.updates, vec![0, 3]);
        assert_eq!(second.reactivations, Vec::<u32>::new());

        let third = run(&mut pool, 3, 0.0, 0.0, -10.0);
        assert_eq!(third.updates, vec![4]);
        assert_eq!(third.reactivations, vec![0]);
        assert_eq!(active_slots(&pool), BTreeSet::from([0, 1]));
    }

    /// derived: the values at 1 are 3, 3, 3, 2 and at 3 are 1, 7, 3, 2. Slot 0 is
    /// the lowest of the three cuts tied at 1 and slot 1 is alone at 3, so slots 2
    /// and 3 own no state.
    #[test]
    fn level1_gives_a_tied_state_to_the_lowest_slot() {
        let mut pool = scored_pool_from_lines(
            &[(4.0, -1.0), (1.0, 2.0), (3.0, 0.0), (2.0, 0.0)],
            &[1.0, 3.0],
        );
        let result = select_level1(&mut pool);
        assert_eq!(result.updates, vec![2, 3]);
        assert_eq!(result.reactivations, Vec::<u32>::new());
    }

    /// derived: the values at 0 are 5, 5, 1 and at 7 are 5, 5, 8. The active
    /// loaded cut at slot 0 owns the state 0 and slot 2 owns the state 7, so the
    /// generated slot 1 owns no state; the loaded cut has no state and keeps its
    /// flag.
    #[test]
    fn level1_deactivates_a_generated_cut_tied_with_a_lower_loaded_cut() {
        let loaded = vec![novomodelo_io::OwnedPolicyCutRecord {
            slot_index: 0,
            iteration: 1,
            intercept: 5.0,
            coefficients: vec![0.0],
            is_active: true,
            ..owned_piece_defaults()
        }];
        let mut pool = CutPool::new_with_warm_start(1, 1, 2, &loaded);
        pool.enable_trial_states();
        pool.add_cut(NodeId(0), 0, 0, 5.0, &[0.0]); // slot 1
        pool.add_cut(NodeId(0), 1, 0, 1.0, &[1.0]); // slot 2
        pool.set_trial_state_for_test(1, &[0.0]);
        pool.set_trial_state_for_test(2, &[7.0]);
        let result = select_level1(&mut pool);
        assert_eq!(result.updates, vec![1]);
        assert_eq!(result.reactivations, Vec::<u32>::new());
        assert_eq!(active_slots(&pool), BTreeSet::from([0, 2]));
    }

    /// derived: the values at 1 are 0, 3, 3 and at 3 are 0, 1, 7, so slot 1 owns
    /// the state 1 and slot 2 the state 3. Slot 0, the only cut of the latest
    /// iteration, owns no state.
    #[test]
    fn level1_deactivates_a_current_iteration_cut_that_owns_no_state() {
        let mut pool = scored_pool_from_lines(&[(0.0, 0.0), (4.0, -1.0), (1.0, 2.0)], &[1.0, 3.0]);
        pool.set_iteration_generated_for_test(0, 5);
        let result = select_level1(&mut pool);
        assert_eq!(result.updates, vec![0]);
        assert_eq!(result.reactivations, Vec::<u32>::new());
    }

    /// derived: with zero slopes every state sees the intercepts 0, 7, 7, 0.
    /// Slot 1 is the lowest maximiser of all four states, so slot 2, identical
    /// to it, owns none.
    #[test]
    fn level1_keeps_one_of_two_identical_cuts() {
        let mut pool =
            scored_pool_from_lines(&[(0.0, 0.0), (7.0, 0.0), (7.0, 0.0), (0.0, 0.0)], &[0.0; 4]);
        select_level1(&mut pool);
        assert_eq!(active_slots(&pool), BTreeSet::from([1]));
    }

    /// A pool that stores trial states with one slot per reference cut;
    /// `scored_active` gives each scored slot's starting flag.
    fn pool_from_reference(cuts: &[RefCut], scored_active: impl Fn(usize) -> bool) -> CutPool {
        let mut pool = CutPool::new(cuts.len(), cuts[0].coefficients.len(), 1, 0);
        pool.enable_trial_states();
        for (slot, cut) in cuts.iter().enumerate() {
            let (origin, scored, active) = match cut.role {
                RefRole::Vacant => continue,
                RefRole::Generated => (SlotOrigin::Generated, true, scored_active(slot)),
                RefRole::ScoredPrefix => (SlotOrigin::Prefix, true, scored_active(slot)),
                RefRole::StatelessPrefix { active } => (SlotOrigin::Prefix, false, active),
            };
            pool.add_cut(NodeId(0), slot as u64, 0, cut.intercept, &cut.coefficients);
            pool.set_role_for_test(slot, SlotRole { origin, scored });
            pool.set_active(slot as u32, active);
        }
        pool
    }

    fn scored_slots(pool: &CutPool) -> Vec<usize> {
        (0..pool.populated())
            .filter(|&slot| pool.role(slot).scored)
            .collect()
    }

    /// Stores `states[i]` at `slots[i]`.
    fn store_states(pool: &mut CutPool, slots: &[usize], states: &[Vec<f64>]) {
        assert!(states.len() <= slots.len(), "more states than slots");
        for (&slot, state) in slots.iter().zip(states) {
            pool.set_trial_state_for_test(slot, state);
        }
    }

    /// derived: `active_after` of the reference rule, whose fixtures hold scored
    /// generated cuts, scored and stateless loaded cuts and a vacant slot. A
    /// scored cut's starting flag is an output of the rule, so both starts give
    /// the same set.
    #[test]
    fn level1_follows_the_reference_rule_on_every_fixture() {
        for (name, (cuts, states)) in fixtures() {
            for scored_start_active in [true, false] {
                let mut pool = pool_from_reference(&cuts, |_| scored_start_active);
                let slots = scored_slots(&pool);
                store_states(&mut pool, &slots, &states);
                select_level1(&mut pool);
                assert_eq!(
                    active_slots(&pool),
                    active_after(&cuts, &states),
                    "{name}, scored cuts starting active: {scored_start_active}"
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // Level1 against the reference rule
    // -----------------------------------------------------------------------

    const CUT_COUNTS: RangeInclusive<usize> = 2..=40;

    #[derive(Debug)]
    struct PoolDraw {
        intercepts: Vec<f64>,
        coefficients: Vec<Vec<f64>>,
        states: Vec<Vec<f64>>,
        active: Vec<bool>,
    }

    prop_compose! {
        /// Integer-valued cuts and states: every cut value and every
        /// power-of-two rescaling of them is exact in `f64`. A pool stores a
        /// state with a cut, so a draw holds at most one state per cut.
        fn pool_draw()(
            (cut_count, state_count) in CUT_COUNTS.prop_flat_map(|cuts| (Just(cuts), 1..=cuts)),
            dimension in 1..=4usize,
        )(
            intercepts in prop::collection::vec((-50..=50i32).prop_map(f64::from), cut_count),
            coefficients in prop::collection::vec(
                prop::collection::vec((-50..=50i32).prop_map(f64::from), dimension),
                cut_count,
            ),
            states in prop::collection::vec(
                prop::collection::vec((-8..=8i32).prop_map(f64::from), dimension),
                state_count,
            ),
            active in prop::collection::vec(any::<bool>(), cut_count),
        ) -> PoolDraw {
            PoolDraw { intercepts, coefficients, states, active }
        }
    }

    fn reference_cuts(draw: &PoolDraw, scale: f64) -> Vec<RefCut> {
        draw.intercepts
            .iter()
            .zip(&draw.coefficients)
            .map(|(intercept, coefficients)| RefCut {
                intercept: intercept * scale,
                coefficients: coefficients.iter().map(|c| c * scale).collect(),
                role: RefRole::Generated,
            })
            .collect()
    }

    /// The active set after one selection run on the draw's cuts scaled by
    /// `scale`, with `states[i]` stored at `slots[i]`.
    fn active_after_level1(
        draw: &PoolDraw,
        scale: f64,
        slots: &[usize],
        states: &[Vec<f64>],
    ) -> BTreeSet<usize> {
        let mut pool = pool_from_reference(&reference_cuts(draw, scale), |slot| draw.active[slot]);
        store_states(&mut pool, slots, states);
        select_level1(&mut pool);
        active_slots(&pool)
    }

    fn first_slots(draw: &PoolDraw) -> Vec<usize> {
        (0..draw.intercepts.len()).collect()
    }

    #[test]
    fn property_generator_crosses_the_block_size() {
        assert!(*CUT_COUNTS.end() > M_BLOCK);
    }

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 256,
            rng_seed: RngSeed::Fixed(42),
            ..ProptestConfig::default()
        })]

        #[test]
        fn level1_matches_the_reference_rule(draw in pool_draw()) {
            prop_assert_eq!(
                active_after_level1(&draw, 1.0, &first_slots(&draw), &draw.states),
                active_after(&reference_cuts(&draw, 1.0), &draw.states)
            );
        }

        #[test]
        fn level1_selection_ignores_state_order_and_duplicates(
            draw in pool_draw(),
            duplicates in prop::collection::vec(
                any::<prop::sample::Index>(),
                0..=*CUT_COUNTS.end(),
            ),
            slot_keys in prop::collection::vec(any::<u32>(), *CUT_COUNTS.end()),
        ) {
            let slots = first_slots(&draw);
            let mut states = draw.states.clone();
            states.extend(
                duplicates
                    .iter()
                    .take(slots.len() - draw.states.len())
                    .map(|index| draw.states[index.index(draw.states.len())].clone()),
            );
            let mut permuted_slots = slots.clone();
            permuted_slots.sort_by_key(|&slot| slot_keys[slot]);

            prop_assert_eq!(
                active_after_level1(&draw, 1.0, &slots, &draw.states),
                active_after_level1(&draw, 1.0, &permuted_slots, &states)
            );
        }

        #[test]
        fn level1_selection_is_idempotent(draw in pool_draw()) {
            let mut pool =
                pool_from_reference(&reference_cuts(&draw, 1.0), |slot| draw.active[slot]);
            store_states(&mut pool, &first_slots(&draw), &draw.states);
            select_level1(&mut pool);
            prop_assert_eq!(
                select_level1(&mut pool),
                CutActivityUpdates {
                    updates: vec![],
                    reactivations: vec![],
                }
            );
        }

        #[test]
        fn adding_a_state_never_deactivates_a_selected_cut(draw in pool_draw()) {
            prop_assume!(draw.states.len() >= 2);
            let slots = first_slots(&draw);
            let (without_last, last) = draw.states.split_at(draw.states.len() - 1);
            let fewer = active_after_level1(&draw, 1.0, &slots, without_last);
            let more = active_after_level1(&draw, 1.0, &slots, &draw.states);
            prop_assert!(
                fewer.is_subset(&more),
                "adding the state {last:?} deactivated {:?}",
                fewer.difference(&more).collect::<Vec<_>>()
            );
        }

        #[test]
        fn level1_selection_is_invariant_to_power_of_two_scaling(
            draw in pool_draw(),
            exponent in -20..=20i32,
        ) {
            let slots = first_slots(&draw);
            prop_assert_eq!(
                active_after_level1(&draw, 1.0, &slots, &draw.states),
                active_after_level1(&draw, 2.0_f64.powi(exponent), &slots, &draw.states)
            );
        }
    }

    // -----------------------------------------------------------------------
    // Dynamic (DCS) variant tests
    // -----------------------------------------------------------------------

    /// Build a `dynamic` selection block from JSON so serde fills the
    /// per-field defaults exactly as a user config would.
    fn dynamic_from_json(body: &str) -> RowSelectionConfig {
        serde_json::from_str(body).expect("dynamic selection block must parse")
    }

    /// A `dynamic` block with no parameters maps to the documented defaults; the
    /// violation tolerance stays absent.
    #[test]
    fn parse_dynamic_defaults() {
        let cfg = dynamic_from_json(r#"{"selection": {"method": "dynamic"}}"#);
        let strategy = parse_cut_selection_config(&cfg)
            .expect("dynamic with defaults must parse")
            .expect("must produce Some for dynamic");
        assert!(
            matches!(
                strategy,
                CutSelectionStrategy::Dynamic {
                    candidate_recency: None,
                    seed_window: 5,
                    max_added_per_round: 10,
                    violation_tolerance: None,
                    start_iteration: 2,
                }
            ),
            "unexpected variant or defaults: {strategy:?}"
        );
    }

    /// `candidate_recency`/`seed_window`/`max_added_per_round`/
    /// `violation_tolerance` flow into the `Dynamic` variant's fields of the
    /// same names; `start_iteration` keeps its default.
    #[test]
    fn parse_dynamic_overrides() {
        let cfg = RowSelectionConfig {
            selection: Some(SelectionMethod::Dynamic {
                start_iteration: positive(2),
                seed_window: 7,
                candidate_recency: Some(positive(20)),
                max_added_per_round: positive(3),
                violation_tolerance: Some(1e-9),
            }),
            ..RowSelectionConfig::default()
        };
        let strategy = parse_cut_selection_config(&cfg)
            .expect("dynamic with overrides must parse")
            .expect("must produce Some for dynamic");
        assert!(
            matches!(
                strategy,
                CutSelectionStrategy::Dynamic {
                    candidate_recency: Some(20),
                    seed_window: 7,
                    max_added_per_round: 3,
                    violation_tolerance: Some(violation_tolerance),
                    start_iteration: 2,
                } if (violation_tolerance - 1e-9).abs() < f64::EPSILON
            ),
            "unexpected variant or overrides: {strategy:?}"
        );
    }

    /// A non-positive `violation_tolerance` is rejected for the dynamic method.
    #[test]
    fn parse_dynamic_rejects_nonpositive_violation_tolerance() {
        for violation_tolerance in [-1.0, 0.0] {
            let cfg = RowSelectionConfig {
                selection: Some(SelectionMethod::Dynamic {
                    start_iteration: positive(2),
                    seed_window: 5,
                    candidate_recency: None,
                    max_added_per_round: positive(10),
                    violation_tolerance: Some(violation_tolerance),
                }),
                ..RowSelectionConfig::default()
            };
            let msg = parse_cut_selection_config(&cfg)
                .expect_err("non-positive violation_tolerance must be rejected for dynamic");
            assert!(
                msg.contains("violation_tolerance must be finite and > 0"),
                "error must mention the violation_tolerance constraint, got: {msg}"
            );
        }
    }

    #[test]
    fn parse_dynamic_rejects_a_violation_tolerance_of_one() {
        for value in [1.0, 1.5] {
            let messages = rejections("violation_tolerance", value);
            assert_eq!(messages.len(), 1);
            assert_eq!(
                messages[0],
                format!("cut_selection.violation_tolerance must be < 1 (got {value})")
            );
        }
        for (key, cfg) in configs_with_tolerance(0.5) {
            parse_cut_selection_config(&cfg)
                .unwrap_or_else(|msg| panic!("{key} = 0.5 must parse, got: {msg}"));
        }
    }

    /// Every explicit dynamic field flows into the `Dynamic` variant.
    #[test]
    fn parse_dynamic_explicit_fields() {
        let cfg = RowSelectionConfig {
            selection: Some(SelectionMethod::Dynamic {
                start_iteration: positive(5),
                seed_window: 7,
                candidate_recency: Some(positive(20)),
                max_added_per_round: positive(3),
                violation_tolerance: Some(1e-9),
            }),
            ..RowSelectionConfig::default()
        };
        let strategy = parse_cut_selection_config(&cfg)
            .expect("dynamic with explicit fields must parse")
            .expect("must produce Some for dynamic");
        assert!(
            matches!(
                strategy,
                CutSelectionStrategy::Dynamic {
                    candidate_recency: Some(20),
                    seed_window: 7,
                    max_added_per_round: 3,
                    violation_tolerance: Some(violation_tolerance),
                    start_iteration: 5,
                } if (violation_tolerance - 1e-9).abs() < f64::EPSILON
            ),
            "unexpected variant or explicit fields: {strategy:?}"
        );
    }

    /// `0` is a valid `seed_window` (seeds only the current iteration) and the
    /// absent default is `5`.
    #[test]
    fn parse_dynamic_seed_window_sets_the_seed_window() {
        let cfg_zero =
            dynamic_from_json(r#"{"selection": {"method": "dynamic", "seed_window": 0}}"#);
        let strategy = parse_cut_selection_config(&cfg_zero)
            .expect("seed_window = 0 must parse for dynamic (0 is valid)")
            .expect("must produce Some for dynamic");
        assert!(
            matches!(
                strategy,
                CutSelectionStrategy::Dynamic { seed_window: 0, .. }
            ),
            "seed_window = 0 must be carried: {strategy:?}"
        );

        let cfg_absent = dynamic_from_json(r#"{"selection": {"method": "dynamic"}}"#);
        let strategy = parse_cut_selection_config(&cfg_absent)
            .expect("absent seed_window must parse for dynamic")
            .expect("must produce Some for dynamic");
        assert!(
            matches!(
                strategy,
                CutSelectionStrategy::Dynamic { seed_window: 5, .. }
            ),
            "absent seed_window must default to 5: {strategy:?}"
        );
    }

    /// `should_run` returns `false` for `Dynamic` at every iteration.
    #[test]
    fn dynamic_should_run_always_false() {
        let strategy = CutSelectionStrategy::Dynamic {
            candidate_recency: None,
            seed_window: 5,
            max_added_per_round: 10,
            violation_tolerance: None,
            start_iteration: 2,
        };
        for iteration in [0_u64, 1, 2, 5, 100] {
            assert!(
                !strategy.should_run(iteration),
                "Dynamic must never run as a pool-level pass (iteration {iteration})"
            );
        }
    }
}
