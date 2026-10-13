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
//! # The windowed kernel
//!
//! `Lml1` and `Dominated` share one value-evaluation kernel,
//! [`CutSelectionStrategy::select_for_stage`]: it evaluates ALL written cuts
//! (active AND inactive) at each state of the retained window of trial states
//! (`VisitedStatesArchive::trim_to_window`) and applies the method's rule:
//!
//! - **Lml1**: at each visited state, retain only the oldest eligible cut
//!   within `tie_tolerance` of the maximum; the overall selected set is the
//!   union across all visited states (Guigues & Bandarra 2019).
//! - **Dominated**: retain every eligible cut within `threshold` of the maximum
//!   at some visited state.
//!
//! `matrixmultiply`'s micro-kernel is bit-deterministic for any given input
//! shape, and the per-task OR-merge is commutative + associative, so
//! `RAYON_NUM_THREADS=1` and `=96` produce identical output on the same binary.
//!
//! # Usage
//!
//! ```rust
//! use novomodelo_sddp::cut::CutPool;
//! use novomodelo_sddp::cut_selection::{
//!     CutActivityUpdates, CutMetadata, CutSelectionStrategy,
//! };
//!
//! let strategy = CutSelectionStrategy::Level1 {
//!     check_frequency: 5,
//!     tie_tolerance: 1e-10,
//!     lookback_iterations: None,
//! };
//!
//! // Should run at multiples of check_frequency (excluding 0).
//! assert!(!strategy.should_run(0));
//! assert!(!strategy.should_run(3));
//! assert!(strategy.should_run(5));
//! assert!(strategy.should_run(10));
//! ```

use std::num::NonZeroU64;

use rayon::prelude::*;

use crate::cut::{CutPool, SlotOrigin};
use crate::gemm::gemm_block;
use crate::setup::NodeId;

use novomodelo_io::config::RowSelectionConfig;

/// Number of trial points evaluated per `crate::gemm::gemm_block` call.
///
/// 8 yields 24-48 tasks per stage for M = 192-384 — above the rayon
/// scheduling-overhead floor, below the load-imbalance threshold.
pub(crate) const M_BLOCK: usize = 8;

// ---------------------------------------------------------------------------
// CutMetadata
// ---------------------------------------------------------------------------

/// Per-cut tracking metadata for cut selection strategies.
#[derive(Debug, Clone)]
pub struct CutMetadata {
    /// Iteration at which this cut was generated (1-based). `Lml1`, `Dominated`
    /// and the active-cut budget leave a cut of the current iteration active;
    /// `Level1` judges it like any other.
    pub iteration_generated: u64,

    /// Forward pass index that generated this cut.
    pub forward_pass_index: u32,

    /// Declared id (`NodeGraph::node_ids`) of the node that generated this cut;
    /// carried onto the MPI cut wire ([`crate::cut::wire::serialize_cut`]) so a
    /// shared-leaf pool's records stay distinguishable by generating node. Not
    /// read by the value-based selection logic.
    pub node: NodeId,

    /// Binding solves counted since the slot was written or loaded; stays 0
    /// while no reader of binding activity is configured.
    pub active_count: u64,

    /// Iteration of the last recorded binding, on this run's clock. Starts at
    /// the generating iteration for a cut of this run or of the run being
    /// resumed, and at 0 for a cut loaded by a warm start or as a boundary
    /// record. Read by the budget key and the `dynamic` seed window.
    pub last_active_iter: u64,
}

// ---------------------------------------------------------------------------
// CutActivityUpdates
// ---------------------------------------------------------------------------

/// Set of cut activity updates at a single stage.
///
/// The caller applies the changes to the activity bitmap via
/// [`CutPool::apply_updates`].
#[derive(Debug, Clone, PartialEq)]
pub struct CutActivityUpdates {
    /// Stage index (0-based) that this update set belongs to.
    pub stage_index: u32,
    /// Slot indices to deactivate.
    pub updates: Vec<u32>,
    /// Slot indices to reactivate.
    pub reactivations: Vec<u32>,
}

impl CutActivityUpdates {
    /// Construct a deactivation-only update set, leaving `reactivations` empty.
    #[must_use]
    pub fn deactivations_only(stage_index: u32, indices: Vec<u32>) -> Self {
        Self {
            stage_index,
            updates: indices,
            reactivations: vec![],
        }
    }
}

// ---------------------------------------------------------------------------
// CutSelectionStrategy
// ---------------------------------------------------------------------------

/// Cut selection strategy for controlling cut pool growth during SDDP training.
///
/// One strategy is active for the entire training run; all stages use it.
///
/// Variant names and field names are stable wire-format identifiers (postcard
/// MPI broadcast via the derived serde impls) — do not rename them without a
/// migration.
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

        /// Parsed and validated; no step of a run reads it, since ownership
        /// compares values exactly.
        tie_tolerance: f64,

        /// `Some(w)` keeps the trial states of the last `w` iterations of the
        /// run, `None` every one.
        lookback_iterations: Option<NonZeroU64>,
    },

    /// Limited Memory Level-1 (LML1): retain only the oldest eligible cut within
    /// `tie_tolerance` of the maximum per visited state, unioned across states
    /// (Guigues & Bandarra 2019).
    Lml1 {
        /// Number of iterations between selection runs. Must be > 0.
        check_frequency: u64,

        /// Tie-break tolerance: a cut is active at a state when within
        /// `tie_tolerance` of the best value there. Default: `1e-10`.
        tie_tolerance: f64,
    },

    /// Dominated cut detection: deactivate cuts dominated (by more than
    /// `threshold`) at every visited state; reactivate inactive cuts that
    /// achieve the maximum.
    Dominated {
        /// A cut survives if its value is within `threshold` of the maximum at
        /// any visited state.
        threshold: f64,

        /// Number of iterations between selection runs. Must be > 0.
        check_frequency: u64,
    },

    /// DCS: lazy row generation inside each LP solve. It is Novomodelo's own method, not
    /// the Dynamic Cut Selection of de Matos, Philpott and Finardi (2015), and NOT
    /// a pool-level deactivation pass. [`should_run`] always returns `false` and
    /// [`select_for_stage`] always returns empty [`CutActivityUpdates`]. Because
    /// exactness rests on the `k1 = None` (∞) default, DCS is mutually exclusive
    /// with the value-based pool passes.
    ///
    /// [`should_run`]: CutSelectionStrategy::should_run
    /// [`select_for_stage`]: CutSelectionStrategy::select_for_stage
    Dynamic {
        /// Candidate-recency window. `None` ⇒ ∞ (every pool cut a candidate;
        /// exactness-preserving). `Some(n)` ⇒ only cuts generated within the
        /// last `n` iterations are candidates — deliberately NOT exact: older
        /// cuts are never added even when violated.
        k1: Option<u32>,

        /// Seed window: a lazy solve starts with the active cuts whose
        /// [`CutMetadata::last_active_iter`] is at most `k2` iterations old.
        k2: u32,

        /// Number of most-violated candidate cuts considered per round. Must be
        /// `>= 1`.
        nadic: u32,

        /// Absolute violation tolerance for accepting a candidate cut. Must be
        /// `> 0`.
        epsilon_viol: f64,

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
    ///     tie_tolerance: 1e-10,
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
            }
            | Self::Lml1 {
                check_frequency, ..
            }
            | Self::Dominated {
                check_frequency, ..
            } => *check_frequency,
            Self::Dynamic { .. } => return false,
        };
        iteration > 0 && iteration.is_multiple_of(freq)
    }

    /// Whether the strategy reads the per-node archive of visited states: the
    /// windowed kernel of `Lml1` and `Dominated`. `Level1` reads the states
    /// stored with its cuts and `Dynamic` selects inside each solve.
    #[must_use]
    pub fn scores_visited_states(&self) -> bool {
        match self {
            Self::Lml1 { .. } | Self::Dominated { .. } => true,
            Self::Level1 { .. } | Self::Dynamic { .. } => false,
        }
    }

    /// Whether each interior pool stores the trial state of its generated cuts.
    #[must_use]
    pub fn stores_trial_states(&self) -> bool {
        match self {
            Self::Level1 { .. } => true,
            Self::Lml1 { .. } | Self::Dominated { .. } | Self::Dynamic { .. } => false,
        }
    }

    /// Run the windowed kernel of `Lml1` and `Dominated` for stage 0 (a pure
    /// query: the pool is not modified).
    ///
    /// Delegates to [`select_for_stage`](Self::select_for_stage) with
    /// `stage_index = 0`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use novomodelo_sddp::cut::{CutPool};
    /// use novomodelo_sddp::cut_selection::{CutMetadata, CutSelectionStrategy};
    /// use novomodelo_sddp::setup::NodeId;
    ///
    /// let strategy = CutSelectionStrategy::Dominated { threshold: 0.0, check_frequency: 5 };
    /// let mut pool = CutPool::new(2, 1, 1, 0);
    /// pool.add_cut(NodeId(0), 0, 0, 1.0, &[1.0]);
    /// pool.add_cut(NodeId(0), 1, 0, 2.0, &[2.0]);
    /// // Empty visited_states returns empty updates.
    /// let deact = strategy.select(&pool, &[], 10);
    /// assert!(deact.updates.is_empty());
    /// ```
    #[must_use]
    pub fn select(
        &self,
        pool: &CutPool,
        visited_states: &[f64],
        current_iteration: u64,
    ) -> CutActivityUpdates {
        let expected_trial_count = if pool.state_dimension == 0 {
            0
        } else {
            visited_states.len() / pool.state_dimension
        };
        self.select_for_stage(
            pool,
            visited_states,
            expected_trial_count,
            current_iteration,
            0,
        )
    }

    /// Run the windowed kernel of `Lml1` and `Dominated` for a specific stage (a
    /// pure query: the pool is not modified). `Level1` and `Dynamic` get no
    /// update from it.
    ///
    /// `visited_states` is a flat row-major `&[f64]`, one state per
    /// `pool.state_dimension` elements, already projected into the pool's slot
    /// space; `expected_trial_count` is the archive's own trial-point count for
    /// that node, cross-checked against the buffer stride. When empty, returns
    /// empty updates.
    ///
    /// # Panics
    ///
    /// Panics if `visited_states.len()` is not `expected_trial_count *
    /// pool.state_dimension` — a packing-stride/scoring-stride mismatch that
    /// would score cuts against misaligned memory.
    ///
    /// # Determinism
    ///
    /// Trial points are partitioned into `M_BLOCK` blocks scored independently;
    /// the per-task selection bitmaps are OR-merged. The union merge is
    /// commutative and associative, so the result is bit-for-bit identical
    /// regardless of thread count or block boundaries — including for Lml1,
    /// where the union of per-block "oldest at max" over disjoint trial-point
    /// subsets equals the global oldest-at-max union.
    #[must_use]
    pub fn select_for_stage(
        &self,
        pool: &CutPool,
        visited_states: &[f64],
        expected_trial_count: usize,
        current_iteration: u64,
        stage_index: u32,
    ) -> CutActivityUpdates {
        match self {
            Self::Level1 { .. } | Self::Dynamic { .. } => {
                return CutActivityUpdates {
                    stage_index,
                    updates: vec![],
                    reactivations: vec![],
                };
            }
            Self::Lml1 { .. } | Self::Dominated { .. } => {}
        }

        let populated = pool.populated();
        let n_state = pool.state_dimension;

        // Release-active stride/count guard (never debug_assert; the misread
        // lands in release). The count check — not the exact-multiple test alone —
        // catches a global-packed buffer strided by the pool's projected
        // dimension: `n_state_global % n_state == 0` still holds, but the derived
        // count over-counts the archive's own trial points.
        if n_state != 0 {
            let derived = visited_states.len() / n_state;
            assert!(
                visited_states.len().is_multiple_of(n_state) && derived == expected_trial_count,
                "cut selection stride mismatch (stage {stage_index}, stride {n_state}): \
                 buffer length {} is not the archive trial count {expected_trial_count} at \
                 that stride (derived {derived})",
                visited_states.len(),
            );
        }

        if populated == 0 || visited_states.is_empty() || n_state == 0 {
            return CutActivityUpdates {
                stage_index,
                updates: vec![],
                reactivations: vec![],
            };
        }

        let eligible: Vec<bool> = (0..populated)
            .map(|k| {
                pool.role(k).origin == SlotOrigin::Generated
                    && pool.metadata(k).iteration_generated < current_iteration
            })
            .collect();
        let n_eligible = eligible.iter().filter(|&&e| e).count();
        if n_eligible < 2 {
            return CutActivityUpdates {
                stage_index,
                updates: vec![],
                reactivations: vec![],
            };
        }

        let n_states = visited_states.len() / n_state;

        let n_blocks = n_states.div_ceil(M_BLOCK);
        let m_block_starts: Vec<usize> = (0..n_blocks).map(|i| i * M_BLOCK).collect();

        let coef_slice = pool.coefficients_prefix();
        let intercepts = pool.intercepts_prefix();

        let is_selected: Vec<bool> = m_block_starts
            .par_iter()
            .fold(
                || (vec![0.0_f64; populated * M_BLOCK], vec![false; populated]),
                |(mut v_block_local, mut bitmap_local), &m_start| {
                    let m_end = (m_start + M_BLOCK).min(n_states);
                    let m_len = m_end - m_start;
                    let state_block = &visited_states[m_start * n_state..m_end * n_state];

                    let v_block_active = &mut v_block_local[..populated * m_len];
                    gemm_block(
                        coef_slice,
                        state_block,
                        populated,
                        n_state,
                        m_len,
                        v_block_active,
                    );

                    // Broadcast-add each cut's intercept in linear order (D5:
                    // deterministic accumulation). Row-major v_block[k*m_len+col].
                    for (k, &intercept) in intercepts.iter().enumerate().take(populated) {
                        let row = k * m_len;
                        for col in 0..m_len {
                            v_block_active[row + col] += intercept;
                        }
                    }

                    // A vacant slot holds no cut: its zero must not enter the maximum.
                    for k in (0..populated).filter(|&k| pool.role(k).origin == SlotOrigin::Vacant) {
                        v_block_active[k * m_len..(k + 1) * m_len].fill(f64::NEG_INFINITY);
                    }

                    for col in 0..m_len {
                        apply_column_rule(
                            self,
                            v_block_active,
                            populated,
                            m_len,
                            col,
                            &eligible,
                            &mut bitmap_local,
                        );
                    }

                    (v_block_local, bitmap_local)
                },
            )
            .map(|(_, bitmap)| bitmap)
            .reduce(
                || vec![false; populated],
                |mut a, b| {
                    for (ai, bi) in a.iter_mut().zip(b.iter()) {
                        *ai |= *bi;
                    }
                    a
                },
            );

        let mut deactivations: Vec<u32> = Vec::new();
        let mut reactivations: Vec<u32> = Vec::new();
        #[allow(clippy::cast_possible_truncation)]
        for k in 0..populated {
            if eligible[k] {
                let currently_active = pool.is_active(k);
                if is_selected[k] && !currently_active {
                    reactivations.push(k as u32);
                } else if !is_selected[k] && currently_active {
                    deactivations.push(k as u32);
                }
            }
        }

        CutActivityUpdates {
            stage_index,
            updates: deactivations,
            reactivations,
        }
    }
}

/// Mark selected slots for column `col` of the row-major
/// `populated × m_len` `v_block` panel into `bitmap`.
///
/// The per-column max is computed over ALL written cuts (active and inactive).
/// Dominated marks every eligible cut within `threshold` of the max; Lml1 marks
/// only the oldest cut within `tie_tolerance` of it. The ascending slot walk
/// makes Lml1's "oldest at max" deterministic (D5).
#[inline]
fn apply_column_rule(
    method: &CutSelectionStrategy,
    v_block: &[f64],
    populated: usize,
    m_len: usize,
    col: usize,
    eligible: &[bool],
    bitmap: &mut [bool],
) {
    let mut max_val = f64::NEG_INFINITY;
    for k in 0..populated {
        let v = v_block[k * m_len + col];
        if v > max_val {
            max_val = v;
        }
    }

    match method {
        CutSelectionStrategy::Dominated { threshold, .. } => {
            let cutoff = max_val - threshold;
            for k in 0..populated {
                if eligible[k] && v_block[k * m_len + col] >= cutoff {
                    bitmap[k] = true;
                }
            }
        }
        CutSelectionStrategy::Lml1 { tie_tolerance, .. } => {
            let cutoff = max_val - tie_tolerance;
            for k in 0..populated {
                if eligible[k] && v_block[k * m_len + col] >= cutoff {
                    bitmap[k] = true;
                    break;
                }
            }
        }
        CutSelectionStrategy::Level1 { .. } | CutSelectionStrategy::Dynamic { .. } => {
            unreachable!("Level1 and Dynamic do not run the windowed kernel")
        }
    }
}

// ---------------------------------------------------------------------------
// Config parsing
// ---------------------------------------------------------------------------

/// Validate the periodic-pruning cadence (`check_frequency`).
///
/// # Errors
///
/// Returns `Err(String)` when `check_frequency = 0`.
fn validate_check_frequency(check_frequency: u32) -> Result<u32, String> {
    if check_frequency == 0 {
        return Err("cut_selection.check_frequency must be > 0".to_string());
    }
    Ok(check_frequency)
}

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
/// violated: `check_frequency = 0` or a `tie_tolerance`/`domination_tolerance`
/// that is not finite and `>= 0` for a periodic method; or, for the dynamic
/// method, `start_iteration = 0`, `candidate_recency = Some(0)`,
/// `max_added_per_round = 0`, or a `violation_tolerance` that is not finite
/// and `> 0`.
///
/// [`SelectionMethod`]: novomodelo_io::config::SelectionMethod
pub fn parse_cut_selection_config(
    config: &RowSelectionConfig,
) -> Result<Option<CutSelectionStrategy>, String> {
    use novomodelo_io::config::SelectionMethod;

    if let Some(tolerance) = config.row_activity_tolerance {
        validate_tolerance("row_activity_tolerance", tolerance, false)?;
    }

    let Some(selection) = config.selection.as_ref() else {
        return Ok(None);
    };

    match *selection {
        SelectionMethod::Level1 {
            tie_tolerance,
            check_frequency,
        } => Ok(Some(CutSelectionStrategy::Level1 {
            check_frequency: u64::from(validate_check_frequency(check_frequency)?),
            tie_tolerance: validate_tolerance("tie_tolerance", tie_tolerance, false)?,
            lookback_iterations: None,
        })),
        SelectionMethod::Lml1 {
            tie_tolerance,
            check_frequency,
        } => Ok(Some(CutSelectionStrategy::Lml1 {
            check_frequency: u64::from(validate_check_frequency(check_frequency)?),
            tie_tolerance: validate_tolerance("tie_tolerance", tie_tolerance, false)?,
        })),
        SelectionMethod::Domination {
            domination_tolerance,
            check_frequency,
        } => Ok(Some(CutSelectionStrategy::Dominated {
            check_frequency: u64::from(validate_check_frequency(check_frequency)?),
            threshold: validate_tolerance("domination_tolerance", domination_tolerance, false)?,
        })),
        SelectionMethod::Dynamic {
            start_iteration,
            seed_window,
            candidate_recency,
            max_added_per_round,
            violation_tolerance,
        } => {
            if start_iteration == 0 {
                return Err(
                    "cut_selection.start_iteration must be >= 1 for method='dynamic'".to_string(),
                );
            }

            if candidate_recency == Some(0) {
                return Err(
                    "cut_selection.candidate_recency must be >= 1 for method='dynamic' \
                     (omit it for the unbounded default)"
                        .to_string(),
                );
            }

            if max_added_per_round == 0 {
                return Err(
                    "cut_selection.max_added_per_round must be >= 1 for method='dynamic'"
                        .to_string(),
                );
            }

            Ok(Some(CutSelectionStrategy::Dynamic {
                k1: candidate_recency,
                k2: seed_window,
                nadic: max_added_per_round,
                epsilon_viol: validate_tolerance("violation_tolerance", violation_tolerance, true)?,
                start_iteration: u64::from(start_iteration),
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
    use std::num::NonZeroU64;
    use std::ops::RangeInclusive;

    use super::parse_cut_selection_config;
    use super::{CutActivityUpdates, CutMetadata, CutSelectionStrategy, M_BLOCK, NodeId};
    use crate::cut::selection_reference::tests::fixtures;
    use crate::cut::selection_reference::{RefCut, RefRole, active_after};
    use crate::cut::{CutPool, SlotOrigin, SlotRole};
    use novomodelo_io::config::{RowSelectionConfig, SelectionMethod};
    use proptest::prelude::*;
    use proptest::test_runner::RngSeed;

    fn make_meta(active_count: u64, last_active_iter: u64) -> CutMetadata {
        CutMetadata {
            iteration_generated: 1,
            forward_pass_index: 0,
            node: NodeId(0),
            active_count,
            last_active_iter,
        }
    }

    /// Build a `CutPool` pre-populated with the given metadata and active flags.
    #[allow(clippy::cast_possible_truncation)]
    fn make_pool(metadata: &[CutMetadata], active: &[bool]) -> CutPool {
        let n = metadata.len();
        let mut pool = CutPool::new(n, 1, 1, 0);
        // Populate dummy cuts so populated_count advances.
        for i in 0..n {
            pool.add_cut(NodeId(0), 0, i as u32, 0.0, &[0.0]);
        }
        pool.replace_selection(metadata, active);
        pool
    }

    #[test]
    fn should_run_false_at_zero() {
        let s = CutSelectionStrategy::Level1 {
            check_frequency: 5,
            tie_tolerance: 1e-10,
            lookback_iterations: None,
        };
        assert!(!s.should_run(0));
    }

    #[test]
    fn should_run_false_between_multiples() {
        let s = CutSelectionStrategy::Level1 {
            check_frequency: 5,
            tie_tolerance: 1e-10,
            lookback_iterations: None,
        };
        assert!(!s.should_run(3));
        assert!(!s.should_run(7));
    }

    #[test]
    fn should_run_true_at_multiples() {
        let s = CutSelectionStrategy::Level1 {
            check_frequency: 5,
            tie_tolerance: 1e-10,
            lookback_iterations: None,
        };
        assert!(s.should_run(5));
        assert!(s.should_run(10));
        assert!(s.should_run(15));
    }

    #[test]
    fn should_run_lml1_respects_check_frequency() {
        let s = CutSelectionStrategy::Lml1 {
            check_frequency: 5,
            tie_tolerance: 1e-10,
        };
        assert!(!s.should_run(0));
        assert!(!s.should_run(3));
        assert!(s.should_run(5));
        assert!(s.should_run(10));
    }

    #[test]
    fn should_run_dominated_respects_check_frequency() {
        let s = CutSelectionStrategy::Dominated {
            threshold: 0.0,
            check_frequency: 10,
        };
        assert!(!s.should_run(5));
        assert!(s.should_run(10));
    }

    // -----------------------------------------------------------------------
    // Dominated value-based kernel tests
    // -----------------------------------------------------------------------

    /// Pool with 3 cuts (intercepts [1,5,3], coeff all 0), state [0.0].
    /// Dominated `threshold=0.0` → cut 1 (value 5) survives; cuts 0,2 deactivated.
    #[test]
    fn dominated_deactivates_cuts_below_the_maximum_at_a_state() {
        let strategy = CutSelectionStrategy::Dominated {
            threshold: 0.0,
            check_frequency: 5,
        };
        let mut pool = CutPool::new(3, 1, 1, 0);
        pool.add_cut(NodeId(0), 0, 0, 1.0, &[0.0]);
        pool.add_cut(NodeId(0), 1, 0, 5.0, &[0.0]);
        pool.add_cut(NodeId(0), 2, 0, 3.0, &[0.0]);
        // All from iteration 1, current_iteration=10 → all eligible.
        let deact = strategy.select(&pool, &[0.0], 10);
        let mut deact_idx = deact.updates;
        deact_idx.sort_unstable();
        assert_eq!(deact_idx, vec![0, 2], "cuts 0 and 2 must be deactivated");
        assert!(deact.reactivations.is_empty(), "no reactivations expected");
    }

    /// Pool with 3 cuts (intercepts [5,5,3]), state [0.0].
    /// Dominated threshold=1e-10 → cuts 0 and 1 both survive (tie kept).
    #[test]
    fn dominated_retains_tied_cuts() {
        let strategy = CutSelectionStrategy::Dominated {
            threshold: 1e-10,
            check_frequency: 5,
        };
        let mut pool = CutPool::new(3, 1, 1, 0);
        pool.add_cut(NodeId(0), 0, 0, 5.0, &[0.0]);
        pool.add_cut(NodeId(0), 1, 0, 5.0, &[0.0]);
        pool.add_cut(NodeId(0), 2, 0, 3.0, &[0.0]);
        let deact = strategy.select(&pool, &[0.0], 10);
        assert_eq!(
            deact.updates,
            vec![2],
            "only cut 2 (value 3) is deactivated; ties 0 and 1 kept"
        );
        assert!(deact.reactivations.is_empty());
    }

    #[test]
    fn dominated_keeps_two_cuts_tied_at_the_maximum() {
        let strategy = CutSelectionStrategy::Dominated {
            threshold: 0.0,
            check_frequency: 5,
        };
        // Two cuts with equal values at state [1.0]:
        // cut0: 1.0 + 2.0*1.0 = 3.0, cut1: 3.0 + 0.0*1.0 = 3.0 → tied, both survive.
        let mut pool = CutPool::new(2, 1, 1, 0);
        pool.add_cut(NodeId(0), 0, 0, 1.0, &[2.0]);
        pool.add_cut(NodeId(0), 1, 0, 3.0, &[0.0]);
        let deact = strategy.select(&pool, &[1.0], 10);
        assert!(
            deact.updates.is_empty(),
            "no cuts deactivated when all tied at max"
        );
    }

    #[test]
    fn dominated_deactivates_cuts_below_the_tolerance_band() {
        let strategy = CutSelectionStrategy::Dominated {
            threshold: 0.5,
            check_frequency: 5,
        };
        // Values at [0.0]: 1, 3, 2. max=3, cutoff=2.5: only cut1 is selected;
        // cut0 and cut2 fall below the band and are deactivated.
        let mut pool = CutPool::new(3, 1, 1, 0);
        pool.add_cut(NodeId(0), 0, 0, 1.0, &[0.0]);
        pool.add_cut(NodeId(0), 1, 0, 3.0, &[0.0]);
        pool.add_cut(NodeId(0), 2, 0, 2.0, &[0.0]);
        let deact = strategy.select(&pool, &[0.0], 10);
        let mut deact_idx = deact.updates;
        deact_idx.sort_unstable();
        assert_eq!(deact_idx, vec![0, 2]);
    }

    #[test]
    fn dominated_empty_metadata_returns_empty_set() {
        let strategy = CutSelectionStrategy::Dominated {
            threshold: 1e-10,
            check_frequency: 5,
        };
        let pool = CutPool::new(0, 1, 1, 0);
        let deact = strategy.select(&pool, &[], 10);
        assert!(deact.updates.is_empty());
    }

    #[test]
    fn dominated_empty_states_returns_empty() {
        let strategy = CutSelectionStrategy::Dominated {
            threshold: 1e-10,
            check_frequency: 5,
        };
        let mut pool = CutPool::new(3, 1, 1, 0);
        pool.add_cut(NodeId(0), 0, 0, 1.0, &[0.0]);
        pool.add_cut(NodeId(0), 1, 0, 5.0, &[0.0]);
        pool.add_cut(NodeId(0), 2, 0, 3.0, &[0.0]);
        let deact = strategy.select(&pool, &[], 10);
        assert!(deact.updates.is_empty());
        assert!(deact.reactivations.is_empty());
    }

    // -----------------------------------------------------------------------
    // Lml1 value-based kernel tests
    // -----------------------------------------------------------------------

    /// Same pool as `dominated_retains_tied_cuts` (intercepts [5,5,3]), state [0.0].
    /// Lml1 tie_tolerance=1e-10 → only cut 0 (the oldest by slot) survives.
    #[test]
    fn lml1_only_oldest_survives_at_each_state() {
        let strategy = CutSelectionStrategy::Lml1 {
            check_frequency: 5,
            tie_tolerance: 1e-10,
        };
        let mut pool = CutPool::new(3, 1, 1, 0);
        pool.add_cut(NodeId(0), 0, 0, 5.0, &[0.0]);
        pool.add_cut(NodeId(0), 1, 0, 5.0, &[0.0]);
        pool.add_cut(NodeId(0), 2, 0, 3.0, &[0.0]);
        let deact = strategy.select(&pool, &[0.0], 10);
        let mut deact_idx = deact.updates;
        deact_idx.sort_unstable();
        assert_eq!(
            deact_idx,
            vec![1, 2],
            "cuts 1 and 2 deactivated; only oldest (cut 0) at max survives"
        );
        assert!(deact.reactivations.is_empty());
    }

    /// Lml1 with two trial points selecting different oldest cuts.
    /// state [0.0]: cut0(val=2)>cut1(val=1) → oldest at max = cut0
    /// state [1.0]: cut0=2+0=2, cut1=0+3=3 → cut1 is max, oldest at max = cut1
    /// Union: both cut0 and cut1 selected → cut2 deactivated.
    #[test]
    fn lml1_union_across_trial_states() {
        let strategy = CutSelectionStrategy::Lml1 {
            check_frequency: 5,
            tie_tolerance: 1e-10,
        };
        let mut pool = CutPool::new(3, 1, 1, 0);
        pool.add_cut(NodeId(0), 0, 0, 2.0, &[0.0]); // constant 2
        pool.add_cut(NodeId(0), 1, 0, 0.0, &[3.0]); // 3x
        pool.add_cut(NodeId(0), 2, 0, 0.5, &[0.0]); // constant 0.5 (never at max)
        let deact = strategy.select(&pool, &[0.0, 1.0], 10);
        assert_eq!(
            deact.updates,
            vec![2],
            "only cut 2 (never at max) deactivated"
        );
        assert!(deact.reactivations.is_empty());
    }

    #[test]
    fn lml1_keeps_the_older_of_two_tied_cuts() {
        let strategy = CutSelectionStrategy::Lml1 {
            check_frequency: 5,
            tie_tolerance: 1e-10,
        };
        let mut pool = CutPool::new(2, 1, 1, 0);
        pool.add_cut(NodeId(0), 0, 0, 3.0, &[0.0]);
        pool.add_cut(NodeId(0), 1, 0, 3.0, &[0.0]);
        let deact = strategy.select(&pool, &[0.0], 10);
        // Both tied → oldest (cut 0) selected → cut 1 deactivated.
        assert_eq!(
            deact.updates,
            vec![1],
            "cut 1 deactivated; cut 0 (oldest) retained"
        );
    }

    #[test]
    fn lml1_keeps_the_oldest_of_the_cuts_tied_at_the_maximum() {
        let strategy = CutSelectionStrategy::Lml1 {
            check_frequency: 5,
            tie_tolerance: 1e-10,
        };
        // cut0=5, cut1=4, cut2=5 (tied). Oldest at max = cut0.
        let mut pool = CutPool::new(3, 1, 1, 0);
        pool.add_cut(NodeId(0), 0, 0, 5.0, &[0.0]);
        pool.add_cut(NodeId(0), 1, 0, 4.0, &[0.0]);
        pool.add_cut(NodeId(0), 2, 0, 5.0, &[0.0]);
        let deact = strategy.select(&pool, &[0.0], 10);
        let mut deact_idx = deact.updates;
        deact_idx.sort_unstable();
        assert_eq!(
            deact_idx,
            vec![1, 2],
            "cuts 1 and 2 deactivated; cut 0 (oldest at max) retained"
        );
    }

    /// Lml1 with 3 cuts, 2 trial points, each selecting a different oldest cut.
    #[test]
    fn lml1_mixed_cuts_deactivates_correct_indices() {
        let strategy = CutSelectionStrategy::Lml1 {
            check_frequency: 5,
            tie_tolerance: 1e-10,
        };
        // cut0: constant 1 (at max for state [0]: max=3→no)
        // cut1: 2x (at state [1]: 2, state [2]: 4 → max at [2])
        // cut2: constant 3 (at max for state [0]: max=3)
        // state [0.0]: values=[1,0,3] → max=3, oldest at max = cut2 (cut0<3, cut1<3)
        // state [2.0]: values=[1,4,3] → max=4, oldest at max = cut1
        // Union: cut1 and cut2 selected → cut0 deactivated.
        let mut pool = CutPool::new(3, 1, 1, 0);
        pool.add_cut(NodeId(0), 0, 0, 1.0, &[0.0]);
        pool.add_cut(NodeId(0), 1, 0, 0.0, &[2.0]);
        pool.add_cut(NodeId(0), 2, 0, 3.0, &[0.0]);
        let deact = strategy.select(&pool, &[0.0, 2.0], 10);
        assert_eq!(
            deact.updates,
            vec![0],
            "only cut 0 (never at max) deactivated"
        );
    }

    #[test]
    fn lml1_empty_states_returns_empty() {
        let strategy = CutSelectionStrategy::Lml1 {
            check_frequency: 5,
            tie_tolerance: 1e-10,
        };
        let mut pool = CutPool::new(2, 1, 1, 0);
        pool.add_cut(NodeId(0), 0, 0, 5.0, &[0.0]);
        pool.add_cut(NodeId(0), 1, 0, 3.0, &[0.0]);
        let deact = strategy.select(&pool, &[], 10);
        assert!(deact.updates.is_empty());
        assert!(deact.reactivations.is_empty());
    }

    #[test]
    fn dominated_deactivates_the_lower_of_two_cuts() {
        let strategy = CutSelectionStrategy::Dominated {
            threshold: 0.0,
            check_frequency: 5,
        };
        // cut0: value=1 at state [0]. cut1: value=3 at state [0].
        // max=3, cutoff=3. cut0(1<3) not selected → deactivated.
        let mut pool = CutPool::new(2, 1, 1, 0);
        pool.add_cut(NodeId(0), 0, 0, 1.0, &[0.0]);
        pool.add_cut(NodeId(0), 1, 0, 3.0, &[0.0]);
        let deact = strategy.select(&pool, &[0.0], 10);
        assert_eq!(deact.updates, vec![0]);
        assert_eq!(deact.reactivations, Vec::<u32>::new());
    }

    #[test]
    fn lml1_deactivates_the_lower_of_two_cuts() {
        let strategy = CutSelectionStrategy::Lml1 {
            check_frequency: 5,
            tie_tolerance: 1e-10,
        };
        // cut0: value=5 at state [0]. cut1: value=3 at state [0].
        // Oldest at max = cut0. cut1 deactivated.
        let mut pool = CutPool::new(2, 1, 1, 0);
        pool.add_cut(NodeId(0), 0, 0, 5.0, &[0.0]);
        pool.add_cut(NodeId(0), 1, 0, 3.0, &[0.0]);
        let deact = strategy.select(&pool, &[0.0], 20);
        assert_eq!(deact.updates, vec![1]);
        assert_eq!(deact.reactivations, Vec::<u32>::new());
    }

    #[test]
    fn select_for_stage_sets_stage_index() {
        let strategy = CutSelectionStrategy::Dominated {
            threshold: 1e-10,
            check_frequency: 5,
        };
        let pool = make_pool(&[make_meta(0, 1)], &[true]);
        let deact = strategy.select_for_stage(&pool, &[], 0, 10, 7);
        assert_eq!(deact.stage_index, 7);
    }

    /// The stride/count guard: a buffer whose derived trial count exceeds the
    /// archive's own count — the exact shape of a global-packed buffer strided by
    /// a reduced pool's projected dimension — panics rather than scoring
    /// misaligned memory.
    #[test]
    #[should_panic(expected = "stride mismatch")]
    fn select_for_stage_guard_fires_on_stride_mismatch() {
        let strategy = CutSelectionStrategy::Dominated {
            threshold: 0.0,
            check_frequency: 1,
        };
        let mut pool = CutPool::new(2, 2, 1, 0);
        pool.add_cut(NodeId(0), 0, 0, 1.0, &[0.0, 0.0]);
        pool.add_cut(NodeId(0), 1, 0, 3.0, &[0.0, 0.0]);
        // stride 2, buffer holds 3 trial points (6 values), but the archive
        // claims 2 — derived 3 != expected 2 → guard fires.
        let _ = strategy.select_for_stage(&pool, &[0.0; 6], 2, 10, 0);
    }

    #[test]
    fn select_sets_stage_index_to_zero() {
        let strategy = CutSelectionStrategy::Dominated {
            threshold: 1e-10,
            check_frequency: 5,
        };
        let pool = CutPool::new(0, 1, 1, 0);
        let deact = strategy.select(&pool, &[], 10);
        assert_eq!(deact.stage_index, 0);
    }

    #[test]
    fn deactivation_set_derives_debug_and_clone() {
        let deact = CutActivityUpdates::deactivations_only(2, vec![0, 3, 7]);
        let cloned = deact.clone();
        assert_eq!(cloned.stage_index, 2);
        assert_eq!(cloned.updates, vec![0, 3, 7]);
        assert!(!format!("{deact:?}").is_empty());
    }

    #[test]
    fn cut_metadata_derives_debug_and_clone() {
        let meta = make_meta(5, 10);
        let cloned = meta.clone();
        assert_eq!(cloned.active_count, 5);
        assert!(!format!("{meta:?}").is_empty());
    }

    // -----------------------------------------------------------------------
    // Current-iteration guard tests
    // -----------------------------------------------------------------------

    /// Cuts generated in the current iteration must never be deactivated by
    /// Dominated, even if they have lower values. They haven't been tested yet
    /// and deactivating them would cause the lower bound to stagnate.
    #[test]
    fn dominated_spares_cuts_from_current_iteration() {
        let strategy = CutSelectionStrategy::Dominated {
            threshold: 1e-10,
            check_frequency: 1,
        };
        // cut0: iteration_generated=10 (current), value=1 at [0.0] → protected
        // cut1: iteration_generated=5 (older), value=1 at [0.0] → eligible
        // cut2: iteration_generated=5 (older), value=5 at [0.0] → eligible
        // max=5 (from cut2, which is eligible), cutoff=5.
        // cut1(1<5) not selected → deactivated. cut0 is not eligible → unchanged.
        let mut pool = CutPool::new(3, 1, 1, 0);
        pool.add_cut(NodeId(0), 0, 0, 1.0, &[0.0]);
        pool.add_cut(NodeId(0), 1, 0, 1.0, &[0.0]);
        pool.add_cut(NodeId(0), 2, 0, 5.0, &[0.0]);
        pool.set_iteration_generated_for_test(0, 10); // current iteration
        pool.set_iteration_generated_for_test(1, 5);
        pool.set_iteration_generated_for_test(2, 5);
        let deact = strategy.select(&pool, &[0.0], 10);
        assert_eq!(
            deact.updates,
            vec![1],
            "only the older cut (slot 1) should be deactivated; \
             the current-iteration cut (slot 0) must be spared"
        );
    }

    #[test]
    fn lml1_keeps_one_owner_across_two_states() {
        let strategy = CutSelectionStrategy::Lml1 {
            check_frequency: 1,
            tie_tolerance: 1e-10,
        };
        // 5 cuts with different values at 2 states:
        // cut0: constant 1 (coeff=0, intercept=1)
        // cut1: constant 2 (coeff=0, intercept=2)
        // cut2: 2x (coeff=2, intercept=0)
        // cut3: constant 3 (coeff=0, intercept=3)
        // cut4: x+1 (coeff=1, intercept=1)
        //
        // state [0.0]: values=[1,2,0,3,1] → max=3 (cut3). Oldest at max: cut3.
        // state [1.0]: values=[1,2,2,3,2] → max=3 (cut3). Oldest at max: cut3 again.
        // Union: only cut3 selected → cuts 0,1,2,4 deactivated.
        let mut pool = CutPool::new(5, 1, 1, 0);
        pool.add_cut(NodeId(0), 0, 0, 1.0, &[0.0]);
        pool.add_cut(NodeId(0), 1, 0, 2.0, &[0.0]);
        pool.add_cut(NodeId(0), 2, 0, 0.0, &[2.0]);
        pool.add_cut(NodeId(0), 3, 0, 3.0, &[0.0]);
        pool.add_cut(NodeId(0), 4, 0, 1.0, &[1.0]);
        let deact = strategy.select_for_stage(&pool, &[0.0, 1.0], 2, 10, 0);
        let mut deact_idx = deact.updates;
        deact_idx.sort_unstable();
        assert_eq!(
            deact_idx,
            vec![0, 1, 2, 4],
            "only cut 3 (oldest at max at both states) survives"
        );
    }

    // -----------------------------------------------------------------------
    // Reactivation tests
    // -----------------------------------------------------------------------

    #[test]
    fn dominated_reactivates_inactive_cut_at_max() {
        let strategy = CutSelectionStrategy::Dominated {
            threshold: 1e-10,
            check_frequency: 1,
        };
        // cut0: intercept=5, active=false (inactive but achieves max)
        // cut1: intercept=3, active=true
        // cut2: intercept=1, active=true
        // max=5 (cut0). cut0 selected → reactivate. cut1,cut2 not selected → deactivate.
        let mut pool = CutPool::new(3, 1, 1, 0);
        pool.add_cut(NodeId(0), 0, 0, 5.0, &[0.0]);
        pool.add_cut(NodeId(0), 1, 0, 3.0, &[0.0]);
        pool.add_cut(NodeId(0), 2, 0, 1.0, &[0.0]);
        // Manually deactivate cut 0.
        pool.set_active(0, false);
        assert_eq!(pool.active_count(), 2);

        let result = strategy.select(&pool, &[0.0], 10);
        assert_eq!(
            result.reactivations,
            vec![0],
            "inactive cut 0 (at max) must be reactivated"
        );
        let mut deact_idx = result.updates;
        deact_idx.sort_unstable();
        assert_eq!(
            deact_idx,
            vec![1, 2],
            "active cuts 1 and 2 (below max) must be deactivated"
        );
    }

    /// Lml1 also emits reactivation for an inactive cut that is oldest-at-max.
    #[test]
    fn lml1_reactivates_inactive_oldest_at_max() {
        let strategy = CutSelectionStrategy::Lml1 {
            check_frequency: 1,
            tie_tolerance: 1e-10,
        };
        // cut0: intercept=5, inactive (oldest, at max)
        // cut1: intercept=5, active (also at max, but younger than cut0)
        // Oldest at max = cut0 (slot 0 < slot 1). cut1 not selected → deactivate.
        let mut pool = CutPool::new(2, 1, 1, 0);
        pool.add_cut(NodeId(0), 0, 0, 5.0, &[0.0]);
        pool.add_cut(NodeId(0), 1, 0, 5.0, &[0.0]);
        pool.set_active(0, false);

        let result = strategy.select(&pool, &[0.0], 10);
        assert_eq!(
            result.reactivations,
            vec![0],
            "inactive cut 0 (oldest at max) must be reactivated"
        );
        assert_eq!(
            result.updates,
            vec![1],
            "active cut 1 (younger at max) must be deactivated"
        );
    }

    #[test]
    fn select_returns_empty_when_all_cuts_from_current_iteration() {
        let strategy = CutSelectionStrategy::Dominated {
            threshold: 1e-10,
            check_frequency: 1,
        };
        let mut pool = CutPool::new(3, 1, 1, 0);
        pool.add_cut(NodeId(0), 0, 0, 1.0, &[0.0]);
        pool.add_cut(NodeId(0), 1, 0, 5.0, &[0.0]);
        pool.add_cut(NodeId(0), 2, 0, 3.0, &[0.0]);
        // All from current iteration.
        pool.set_iteration_generated_for_test(0, 10);
        pool.set_iteration_generated_for_test(1, 10);
        pool.set_iteration_generated_for_test(2, 10);
        let result = strategy.select(&pool, &[0.0], 10);
        assert!(
            result.updates.is_empty(),
            "no activity changes when all cuts from current iteration"
        );
        assert!(result.reactivations.is_empty());
    }

    // -----------------------------------------------------------------------
    // Warm-start slot protection tests
    // -----------------------------------------------------------------------

    /// Loaded cuts participate in max computation but are not candidates.
    /// With `prefix_len() == 1`: slot 0 is loaded; only slots 1+ are eligible.
    #[allow(clippy::cast_possible_truncation)]
    #[test]
    fn dominated_warm_start_cuts_not_deactivated() {
        let strategy = CutSelectionStrategy::Dominated {
            threshold: 0.0,
            check_frequency: 1,
        };
        // Slot 0 is the loaded record; add_cut writes from `slot_base` (1), so
        // slots 1,2 are generated cuts.
        // 3 slots total: slot 0 (loaded, intercept=10), slot 1 (eligible, =1), slot 2 (eligible, =3).
        // max=10 (slot 0). Cutoff=10. Eligible cuts 1,2 both below cutoff → deactivated.
        // Slot 0 is not eligible (loaded) → not deactivated.
        let warm_start_records = vec![novomodelo_io::OwnedPolicyCutRecord {
            cut_id: 0,
            slot_index: 0,
            iteration: 1,
            forward_pass_index: 0,
            intercept: 10.0,
            coefficients: vec![0.0],
            is_active: true,
        }];
        let mut pool = CutPool::new_with_warm_start(1, 1, 2, &warm_start_records);
        pool.add_cut(NodeId(0), 0, 0, 1.0, &[0.0]); // slot 1
        pool.add_cut(NodeId(0), 1, 0, 3.0, &[0.0]); // slot 2

        let result = strategy.select(&pool, &[0.0], 10);
        let mut deact_idx = result.updates;
        deact_idx.sort_unstable();
        assert_eq!(deact_idx, vec![1, 2]);
    }

    // -----------------------------------------------------------------------
    // Skip already-inactive slot tests
    // -----------------------------------------------------------------------

    /// An already-inactive cut that is not at max produces no change; one that
    /// IS at max is reactivated.
    #[test]
    fn select_skips_already_inactive_slots() {
        let mut pool = CutPool::new(10, 1, 1, 0);
        pool.add_cut(NodeId(0), 0, 0, 1.0, &[0.0]); // slot 0: value=1
        pool.add_cut(NodeId(0), 1, 0, 5.0, &[0.0]); // slot 1: value=5 (max)
        pool.add_cut(NodeId(0), 2, 0, 3.0, &[0.0]); // slot 2: value=3
        assert_eq!(pool.active_count(), 3);

        // Manually deactivate slot 0 before selection.
        pool.set_active(0, false);
        assert_eq!(pool.active_count(), 2);

        // Dominated threshold=0: only slot 1 (value=5) is at max.
        // slot 0: eligible, not selected, inactive → no change (not reactivated).
        // slot 1: eligible, selected, active → no change.
        // slot 2: eligible, not selected, active → deactivated.
        let strategy = CutSelectionStrategy::Dominated {
            threshold: 0.0,
            check_frequency: 1,
        };
        let deact = strategy.select_for_stage(&pool, &[0.0], 1, 5, 0);
        assert_eq!(
            deact.updates,
            vec![2],
            "only slot 2 (active, below max) deactivated"
        );
        assert!(
            deact.reactivations.is_empty(),
            "slot 0 (inactive, below max) must not be reactivated"
        );
    }

    // -----------------------------------------------------------------------
    // select_for_stage returns CutActivityUpdates with deactivations
    // -----------------------------------------------------------------------

    #[test]
    fn select_for_stage_returns_cut_activity_updates_with_deactivations() {
        let mut pool = CutPool::new(3, 1, 1, 0);
        pool.add_cut(NodeId(0), 0, 0, 1.0, &[0.0]); // value=1
        pool.add_cut(NodeId(0), 1, 0, 2.0, &[0.0]); // value=2 (max)

        let strategy = CutSelectionStrategy::Dominated {
            threshold: 0.0,
            check_frequency: 1,
        };
        let result = strategy.select_for_stage(&pool, &[0.0], 1, 10, 0);

        // cut0 (value=1) not at max → deactivated. cut1 (value=2) at max → no change.
        assert_eq!(result.updates, vec![0]);
        assert_eq!(result.reactivations, Vec::<u32>::new());
    }

    #[test]
    fn cut_activity_updates_deactivations_only_constructor() {
        let updates = CutActivityUpdates::deactivations_only(7, vec![0, 1, 2]);
        assert_eq!(updates.stage_index, 7);
        assert_eq!(updates.updates.len(), 3);
        assert_eq!(updates.updates, vec![0, 1, 2]);
        assert!(updates.reactivations.is_empty());
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
                tie_tolerance: 1e-8,
                check_frequency: 5,
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
                    tie_tolerance,
                    lookback_iterations: None,
                } if (tie_tolerance - 1e-8).abs() < f64::EPSILON
            ),
            "unexpected variant: {strategy:?}"
        );
    }

    /// `level1` with serde-default fields resolves to the documented defaults
    /// (`tie_tolerance = 1e-10`, `check_frequency = 5`).
    #[test]
    fn test_parse_level1_default_tie_tolerance() {
        let json = r#"{"selection": {"method": "level1"}}"#;
        let cfg: RowSelectionConfig = serde_json::from_str(json).expect("level1 defaults parse");
        let strategy = parse_cut_selection_config(&cfg)
            .expect("level1 must parse")
            .expect("must produce Some for level1");
        assert!(
            matches!(
                strategy,
                CutSelectionStrategy::Level1 {
                    check_frequency: 5,
                    tie_tolerance,
                    lookback_iterations: None,
                } if (tie_tolerance - 1e-10).abs() < 1e-20
            ),
            "unexpected variant or wrong default tie_tolerance: {strategy:?}"
        );
    }

    #[test]
    fn test_parse_lml1() {
        let cfg = RowSelectionConfig {
            selection: Some(SelectionMethod::Lml1 {
                tie_tolerance: 1e-8,
                check_frequency: 5,
            }),
            ..RowSelectionConfig::default()
        };
        let strategy = parse_cut_selection_config(&cfg)
            .expect("lml1 must parse")
            .expect("must produce Some for lml1");
        assert!(
            matches!(
                strategy,
                CutSelectionStrategy::Lml1 {
                    check_frequency: 5,
                    tie_tolerance,
                } if (tie_tolerance - 1e-8).abs() < f64::EPSILON
            ),
            "unexpected variant: {strategy:?}"
        );
    }

    /// `lml1` with serde-default fields resolves to the default
    /// `tie_tolerance = 1e-10`.
    #[test]
    fn test_parse_lml1_defaults() {
        let json = r#"{"selection": {"method": "lml1"}}"#;
        let cfg: RowSelectionConfig = serde_json::from_str(json).expect("lml1 defaults parse");
        let strategy = parse_cut_selection_config(&cfg)
            .expect("lml1 must parse")
            .expect("must produce Some for lml1");
        assert!(
            matches!(
                strategy,
                CutSelectionStrategy::Lml1 {
                    check_frequency: 5,
                    tie_tolerance,
                } if (tie_tolerance - 1e-10).abs() < 1e-20
            ),
            "unexpected variant or wrong default tie_tolerance: {strategy:?}"
        );
    }

    #[test]
    fn test_parse_domination() {
        let cfg = RowSelectionConfig {
            selection: Some(SelectionMethod::Domination {
                domination_tolerance: 1e-6,
                check_frequency: 10,
            }),
            ..RowSelectionConfig::default()
        };
        let strategy = parse_cut_selection_config(&cfg)
            .expect("domination must parse")
            .expect("must produce Some for domination");
        assert!(
            matches!(
                strategy,
                CutSelectionStrategy::Dominated {
                    threshold,
                    check_frequency: 10,
                } if (threshold - 1e-6).abs() < f64::EPSILON
            ),
            "unexpected variant: {strategy:?}"
        );
    }

    #[test]
    fn test_parse_zero_check_frequency() {
        let cfg = RowSelectionConfig {
            selection: Some(SelectionMethod::Level1 {
                tie_tolerance: 1e-10,
                check_frequency: 0,
            }),
            ..RowSelectionConfig::default()
        };
        let result = parse_cut_selection_config(&cfg);
        assert!(result.is_err());
        let msg = result.unwrap_err();
        assert!(
            msg.contains("check_frequency"),
            "error message must mention check_frequency, got: {msg}"
        );
    }

    /// One config per tolerance key, each carrying `value`; `tie_tolerance`
    /// appears once per method that has it.
    fn configs_with_tolerance(value: f64) -> Vec<(&'static str, RowSelectionConfig)> {
        let with_method = |selection| RowSelectionConfig {
            selection: Some(selection),
            ..RowSelectionConfig::default()
        };
        vec![
            (
                "row_activity_tolerance",
                RowSelectionConfig {
                    row_activity_tolerance: Some(value),
                    ..RowSelectionConfig::default()
                },
            ),
            (
                "tie_tolerance",
                with_method(SelectionMethod::Level1 {
                    tie_tolerance: value,
                    check_frequency: 5,
                }),
            ),
            (
                "tie_tolerance",
                with_method(SelectionMethod::Lml1 {
                    tie_tolerance: value,
                    check_frequency: 5,
                }),
            ),
            (
                "domination_tolerance",
                with_method(SelectionMethod::Domination {
                    domination_tolerance: value,
                    check_frequency: 5,
                }),
            ),
            (
                "violation_tolerance",
                with_method(SelectionMethod::Dynamic {
                    start_iteration: 2,
                    seed_window: 5,
                    candidate_recency: None,
                    max_added_per_round: 10,
                    violation_tolerance: value,
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
        for row_activity_tolerance in [None, Some(0.0)] {
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
    fn parse_rejects_a_negative_tie_tolerance() {
        let messages = rejections("tie_tolerance", -1.0);
        assert_eq!(
            messages.len(),
            2,
            "level1 and lml1 both carry tie_tolerance"
        );
        for msg in messages {
            assert!(
                msg.contains("tie_tolerance must be finite and >= 0 (got -1)"),
                "error must name the key, the rule and the value, got: {msg}"
            );
        }
    }

    #[test]
    fn parse_rejects_a_nan_tie_tolerance() {
        let messages = rejections("tie_tolerance", f64::NAN);
        assert_eq!(
            messages.len(),
            2,
            "level1 and lml1 both carry tie_tolerance"
        );
        for msg in messages {
            assert!(
                msg.contains("tie_tolerance") && msg.contains("(got NaN)"),
                "error must name the key and the value, got: {msg}"
            );
        }
    }

    #[test]
    fn parse_rejects_a_negative_domination_tolerance() {
        let messages = rejections("domination_tolerance", -1.0);
        assert_eq!(messages.len(), 1);
        assert!(
            messages[0].contains("domination_tolerance must be finite and >= 0 (got -1)"),
            "error must name the key, the rule and the value, got: {}",
            messages[0]
        );
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
    // Dominated algorithm tests
    // -----------------------------------------------------------------------

    /// Build a `CutPool` with known coefficients, intercepts, and metadata
    /// for testing the dominated selection algorithm.
    #[allow(clippy::cast_possible_truncation)]
    fn make_dominated_pool(
        intercepts: &[f64],
        coefficients: &[Vec<f64>],
        active: &[bool],
        metadata: &[CutMetadata],
    ) -> CutPool {
        let n = intercepts.len();
        let state_dim = coefficients[0].len();
        let mut pool = CutPool::new(n, state_dim, 1, 0);
        for i in 0..n {
            pool.add_cut(NodeId(0), 0, i as u32, intercepts[i], &coefficients[i]);
        }
        pool.replace_selection(metadata, active);
        pool
    }

    fn default_meta_at(iter: u64) -> CutMetadata {
        CutMetadata {
            iteration_generated: iter,
            forward_pass_index: 0,
            node: NodeId(0),
            active_count: 0,
            last_active_iter: iter,
        }
    }

    fn default_meta_vec(n: usize, iter: u64) -> Vec<CutMetadata> {
        (0..n).map(|_| default_meta_at(iter)).collect()
    }

    /// 5 cuts, 3 states (1D). Cuts 0,3,4 dominated at all states.
    #[test]
    fn dominated_select_deactivate_dominated() {
        let strategy = CutSelectionStrategy::Dominated {
            threshold: 0.0,
            check_frequency: 1,
        };
        let pool = make_dominated_pool(
            &[1.0, 0.0, 3.0, 0.5, 0.0],
            &[
                vec![0.0],  // cut 0: constant 1
                vec![2.0],  // cut 1: 2x
                vec![-1.0], // cut 2: 3 - x
                vec![0.0],  // cut 3: constant 0.5
                vec![0.5],  // cut 4: 0.5x
            ],
            &[true; 5],
            &default_meta_vec(5, 1),
        );
        let states: Vec<f64> = vec![0.0, 1.0, 3.0];
        let deact = strategy.select(&pool, &states, 10);
        assert_eq!(deact.updates, vec![0, 3, 4]);
    }

    /// Cut dominated at 2/3 states but tied at 1 -> retained.
    #[test]
    fn dominated_select_partial_domination_retained() {
        let strategy = CutSelectionStrategy::Dominated {
            threshold: 0.0,
            check_frequency: 1,
        };
        // Cut 0: intercept=2, slope=0 (constant 2)
        // Cut 1: intercept=0, slope=2 (2x)
        // At x=0: values=[2, 0] -> max=2, cut 0 achieves max -> not dominated
        // At x=1: values=[2, 2] -> max=2, cut 0 achieves max -> not dominated
        // At x=3: values=[2, 6] -> max=6, cut 0 below -> dominated at this state
        // Net: cut 0 is NOT dominated (achieves max at x=0 and x=1)
        let pool = make_dominated_pool(
            &[2.0, 0.0],
            &[vec![0.0], vec![2.0]],
            &[true, true],
            &default_meta_vec(2, 1),
        );
        let states: Vec<f64> = vec![0.0, 1.0, 3.0];
        let deact = strategy.select(&pool, &states, 10);
        assert!(
            deact.updates.is_empty(),
            "cut 0 achieves max at x=0 and x=1, must not be deactivated"
        );
    }

    /// Cut 2 (constant 2) never achieves max → deactivated.
    #[test]
    fn dominated_select_none_dominated_when_all_achieve_max() {
        let strategy = CutSelectionStrategy::Dominated {
            threshold: 0.0,
            check_frequency: 1,
        };
        // Cut 0: 5 - 2x (max at x=0: 5)
        // Cut 1: 0 + 3x (max at x=3: 9)
        // Cut 2: 2 + 0x (constant 2, never achieves max)
        let pool = make_dominated_pool(
            &[5.0, 0.0, 2.0],
            &[vec![-2.0], vec![3.0], vec![0.0]],
            &[true; 3],
            &default_meta_vec(3, 1),
        );
        let states: Vec<f64> = vec![0.0, 1.0, 3.0];
        let deact = strategy.select(&pool, &states, 10);
        assert_eq!(
            deact.updates,
            vec![2],
            "only cut 2 (constant 2) should be dominated"
        );
    }

    #[test]
    fn dominated_select_empty_states() {
        let strategy = CutSelectionStrategy::Dominated {
            threshold: 0.0,
            check_frequency: 1,
        };
        let pool = make_dominated_pool(
            &[1.0, 2.0],
            &[vec![0.0], vec![0.0]],
            &[true, true],
            &default_meta_vec(2, 1),
        );
        let deact = strategy.select(&pool, &[], 10);
        assert!(
            deact.updates.is_empty(),
            "empty visited_states must produce empty deactivation set"
        );
    }

    /// With 1 active and 2 inactive cuts, the max is taken over ALL cuts: the
    /// highest inactive cut (intercept=3) is reactivated and the only active cut
    /// (intercept=1) is deactivated.
    #[test]
    fn dominated_select_single_active_cut() {
        let strategy = CutSelectionStrategy::Dominated {
            threshold: 0.0,
            check_frequency: 1,
        };
        let pool = make_dominated_pool(
            &[1.0, 2.0, 3.0],
            &[vec![0.0], vec![0.0], vec![0.0]],
            &[true, false, false],
            &default_meta_vec(3, 1),
        );
        let states: Vec<f64> = vec![0.0, 1.0];
        let deact = strategy.select(&pool, &states, 10);
        // max=3 (cut2), cutoff=3. Only cut2 selected.
        // cut0: active, not selected → deactivated.
        // cut1: inactive, not selected → no change.
        // cut2: inactive, selected → reactivated.
        assert_eq!(
            deact.updates,
            vec![0],
            "active cut 0 (below max) must be deactivated"
        );
        assert_eq!(
            deact.reactivations,
            vec![2],
            "inactive cut 2 (at max) must be reactivated"
        );
    }

    // -----------------------------------------------------------------------
    // Dominated epsilon-tolerance test
    // -----------------------------------------------------------------------

    /// A cut 1e-7 below max everywhere survives with epsilon=1e-6 but is
    /// deactivated with epsilon=0.
    ///
    /// Fixture: 2 cuts (1D constant), states [0.0].
    /// cut0: constant 5.0  (max)
    /// cut1: constant 4.9999999  (max - 1e-7)
    ///
    /// With epsilon=1e-6: cutoff = 5.0 - 1e-6 = 4.999999. cut1(4.9999999) > 4.999999 → survives.
    /// With epsilon=0:    cutoff = 5.0.       cut1(4.9999999) < 5.0 → deactivated.
    #[test]
    fn dominated_epsilon_tolerance_cut_barely_below_max() {
        let meta = default_meta_vec(2, 1);
        let pool = make_dominated_pool(
            &[5.0, 4.999_999_9],
            &[vec![0.0], vec![0.0]],
            &[true; 2],
            &meta,
        );
        let states: Vec<f64> = vec![0.0];

        // With epsilon=1e-6: cut1 is within tolerance → survives.
        let dom_loose = CutSelectionStrategy::Dominated {
            threshold: 1e-6,
            check_frequency: 1,
        };
        let deact_loose = dom_loose.select(&pool, &states, 10);
        assert!(
            deact_loose.updates.is_empty(),
            "cut1 (1e-7 below max) must survive when epsilon=1e-6"
        );

        // With epsilon=0: cut1 is strictly below max → deactivated.
        let dom_strict = CutSelectionStrategy::Dominated {
            threshold: 0.0,
            check_frequency: 1,
        };
        let deact_strict = dom_strict.select(&pool, &states, 10);
        assert_eq!(
            deact_strict.updates,
            vec![1],
            "cut1 (1e-7 below max) must be deactivated when epsilon=0"
        );
    }

    // -----------------------------------------------------------------------
    // Edge case: single eligible cut
    // -----------------------------------------------------------------------

    /// Dominated with exactly one eligible cut returns empty updates (`n_eligible` < 2
    /// guard fires before any evaluation).
    ///
    /// Two slots total: slot 0 is from `current_iteration` (ineligible); slot 1 is
    /// eligible. `n_eligible=1` → empty.
    #[test]
    fn dominated_single_eligible_cut_returns_empty() {
        let strategy = CutSelectionStrategy::Dominated {
            threshold: 1e-10,
            check_frequency: 1,
        };
        let mut pool = CutPool::new(2, 1, 1, 0);
        pool.add_cut(NodeId(0), 0, 0, 10.0, &[0.0]); // higher value
        pool.add_cut(NodeId(0), 1, 0, 1.0, &[0.0]); // lower value
        // Slot 0 from current iteration → ineligible. Slot 1 eligible.
        pool.set_iteration_generated_for_test(0, 10); // current_iteration
        pool.set_iteration_generated_for_test(1, 5);
        // n_eligible = 1 (only slot 1). Guard returns empty.
        let result = strategy.select(&pool, &[0.0], 10);
        assert!(
            result.updates.is_empty(),
            "single eligible cut must not trigger any deactivations"
        );
        assert!(result.reactivations.is_empty());
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
        let updates = pool.ownership_activity_updates(0);
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
            cut_id: 0,
            slot_index: 0,
            iteration: 40,
            forward_pass_index: 0,
            intercept,
            coefficients: vec![slope],
            is_active,
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
            cut_id: 0,
            slot_index: 0,
            iteration: 1,
            forward_pass_index: 0,
            intercept: 5.0,
            coefficients: vec![0.0],
            is_active: true,
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

    /// derived: the cuts at `(iteration, pass)` `(0, 0)`, `(1, 0)`, `(1, 1)` of a
    /// stride-2 pool sit at slots 0, 2 and 3, so slot 1 is vacant. At the state 0
    /// the cut values are -5, -1 and -3: the maximum is -1 at slot 2, so slots 0
    /// and 3 are deactivated and the vacant slot's zero is neither the maximum nor
    /// reactivated.
    #[test]
    fn a_vacant_slot_takes_no_part_in_selection() {
        let strategy = CutSelectionStrategy::Dominated {
            threshold: 0.0,
            check_frequency: 1,
        };
        let mut pool = CutPool::new(4, 1, 2, 0);
        pool.add_cut(NodeId(0), 0, 0, -5.0, &[0.0]);
        pool.add_cut(NodeId(0), 1, 0, -1.0, &[0.0]);
        pool.add_cut(NodeId(0), 1, 1, -3.0, &[0.0]);
        let result = strategy.select_for_stage(&pool, &[0.0], 1, 5, 0);
        assert_eq!(result.updates, vec![0, 3]);
        assert_eq!(result.reactivations, Vec::<u32>::new());
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
                    stage_index: 0,
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
    // Intra-stage parallelism determinism tests
    // -----------------------------------------------------------------------

    /// Build a 100-cut, 1-D pool whose intercepts and coefficients are
    /// deterministic functions of the slot index — designed so that different
    /// cuts achieve the max at different trial points, exercising both
    /// Dominated's "any-max" and Lml1's "oldest-at-max" branches.
    #[allow(clippy::cast_precision_loss)]
    fn make_determinism_pool() -> CutPool {
        const N: usize = 100;
        let mut pool = CutPool::new(N, 1, 1, 0);
        for i in 0..N {
            // Use varied (intercept, slope) pairs so values are non-trivial.
            // intercept = i mod 7, slope = ((i + 3) mod 5) - 2  (range [-2, 2]).
            let intercept = (i % 7) as f64;
            let slope = ((i + 3) % 5) as f64 - 2.0;
            #[allow(clippy::cast_possible_truncation)]
            pool.add_cut(NodeId(0), 0, i as u32, intercept, &[slope]);
            // Make every cut eligible (iteration_generated < current_iteration in
            // the tests below).
            pool.set_iteration_generated_for_test(i, 1);
        }
        pool
    }

    /// Build >= 1000 trial points spanning a representative range.
    #[allow(clippy::cast_precision_loss)]
    fn make_determinism_states(count: usize) -> Vec<f64> {
        (0..count).map(|i| (i as f64) * 0.01 - 5.0).collect()
    }

    /// Run `select_for_stage` for `strategy` inside a rayon thread pool with
    /// `num_threads` workers, returning the resulting `CutActivityUpdates`.
    fn run_in_pool(
        strategy: &CutSelectionStrategy,
        pool: &CutPool,
        states: &[f64],
        current_iteration: u64,
        num_threads: usize,
    ) -> CutActivityUpdates {
        let rayon_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .expect("rayon pool must build for determinism test");
        rayon_pool.install(|| {
            strategy.select_for_stage(
                pool,
                states,
                states.len() / pool.state_dimension,
                current_iteration,
                0,
            )
        })
    }

    /// Determinism for Lml1: per-chunk "oldest at max" picks unioned with
    /// bitwise-OR across disjoint trial-point chunks must reproduce the
    /// sequential global "oldest at max" union.
    #[test]
    fn select_for_stage_deterministic_across_thread_counts_lml1() {
        let pool = make_determinism_pool();
        let states = make_determinism_states(1024);
        let strategy = CutSelectionStrategy::Lml1 {
            check_frequency: 1,
            tie_tolerance: 1e-10,
        };
        let r1 = run_in_pool(&strategy, &pool, &states, 10, 1);
        let r4 = run_in_pool(&strategy, &pool, &states, 10, 4);
        let r8 = run_in_pool(&strategy, &pool, &states, 10, 8);
        assert_eq!(
            r1, r4,
            "Lml1: 1-thread vs 4-thread results must be bit-identical"
        );
        assert_eq!(
            r4, r8,
            "Lml1: 4-thread vs 8-thread results must be bit-identical"
        );
    }

    /// Determinism for Dominated at a scale that exercises many m-blocks per
    /// stage: the per-task OR-merge in `select_for_stage` is commutative and
    /// associative, so any worker assignment yields identical
    /// `CutActivityUpdates`.
    #[test]
    fn select_for_stage_deterministic_across_thread_counts_dominated() {
        let pool = make_determinism_pool();
        let states = make_determinism_states(1024);
        let strategy = CutSelectionStrategy::Dominated {
            threshold: 0.0,
            check_frequency: 1,
        };
        let r1 = run_in_pool(&strategy, &pool, &states, 10, 1);
        let r4 = run_in_pool(&strategy, &pool, &states, 10, 4);
        let r8 = run_in_pool(&strategy, &pool, &states, 10, 8);
        assert_eq!(
            r1, r4,
            "Dominated: 1-thread vs 4-thread results must be bit-identical"
        );
        assert_eq!(
            r4, r8,
            "Dominated: 4-thread vs 8-thread results must be bit-identical"
        );
    }

    /// Sequential (1-thread) vs parallel (4-thread) kernel results must
    /// agree on the same input. Fixture size chosen to trigger multiple
    /// m-blocks under `M_BLOCK` = 8 (`n_states` = 263 → 33 m-blocks).
    #[test]
    fn select_for_stage_parallel_matches_sequential_multiple_m_blocks() {
        let pool = make_determinism_pool();
        // 263 = 32 * 8 + 7 — exercises 33 m-blocks with the last block
        // partial (m_len = 7 < M_BLOCK).
        let n_states = 263;
        let states = make_determinism_states(n_states);

        for strategy in [
            CutSelectionStrategy::Lml1 {
                check_frequency: 1,
                tie_tolerance: 1e-10,
            },
            CutSelectionStrategy::Dominated {
                threshold: 0.0,
                check_frequency: 1,
            },
        ] {
            let seq = run_in_pool(&strategy, &pool, &states, 10, 1);
            let par = run_in_pool(&strategy, &pool, &states, 10, 4);
            assert_eq!(
                seq, par,
                "strategy {strategy:?}: parallel must equal sequential \
                 across multiple m-blocks with partial last block"
            );
        }
    }

    /// New kernel: bit-identical output across thread counts at a
    /// moderate scale that exercises multiple m-blocks per stage.
    #[test]
    fn select_for_stage_deterministic_across_thread_counts() {
        let pool = make_determinism_pool();
        let states = make_determinism_states(64); // 64 trial points >> M_BLOCK
        let strategy = CutSelectionStrategy::Lml1 {
            check_frequency: 1,
            tie_tolerance: 1e-10,
        };

        let r1 = {
            let rp = rayon::ThreadPoolBuilder::new()
                .num_threads(1)
                .build()
                .expect("rayon pool");
            rp.install(|| {
                strategy.select_for_stage(
                    &pool,
                    &states,
                    states.len() / pool.state_dimension,
                    10,
                    0,
                )
            })
        };

        let r8 = {
            let rp = rayon::ThreadPoolBuilder::new()
                .num_threads(8)
                .build()
                .expect("rayon pool");
            rp.install(|| {
                strategy.select_for_stage(
                    &pool,
                    &states,
                    states.len() / pool.state_dimension,
                    10,
                    0,
                )
            })
        };

        assert_eq!(
            r1, r8,
            "select_for_stage must be byte-identical across thread counts"
        );
    }

    // -----------------------------------------------------------------------
    // Dynamic (DCS) variant tests
    // -----------------------------------------------------------------------

    /// Build a `dynamic` selection block from JSON so serde fills the
    /// per-field defaults exactly as a user config would.
    fn dynamic_from_json(body: &str) -> RowSelectionConfig {
        serde_json::from_str(body).expect("dynamic selection block must parse")
    }

    /// A `dynamic` block with no parameters maps to the documented defaults.
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
                    k1: None,
                    k2: 5,
                    nadic: 10,
                    epsilon_viol,
                    start_iteration: 2,
                } if (epsilon_viol - 1e-10).abs() < 1e-20
            ),
            "unexpected variant or defaults: {strategy:?}"
        );
    }

    /// `candidate_recency`/`seed_window`/`max_added_per_round`/
    /// `violation_tolerance` flow into the wire-format `k1`/`k2`/`nadic`/
    /// `epsilon_viol`; `start_iteration` keeps its default.
    #[test]
    fn parse_dynamic_overrides() {
        let cfg = RowSelectionConfig {
            selection: Some(SelectionMethod::Dynamic {
                start_iteration: 2,
                seed_window: 7,
                candidate_recency: Some(20),
                max_added_per_round: 3,
                violation_tolerance: 1e-9,
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
                    k1: Some(20),
                    k2: 7,
                    nadic: 3,
                    epsilon_viol,
                    start_iteration: 2,
                } if (epsilon_viol - 1e-9).abs() < f64::EPSILON
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
                    start_iteration: 2,
                    seed_window: 5,
                    candidate_recency: None,
                    max_added_per_round: 10,
                    violation_tolerance,
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

    /// `candidate_recency = Some(0)` is rejected for the dynamic method.
    /// (Absent/`None` is valid and means the unbounded default.)
    #[test]
    fn parse_dynamic_rejects_zero_candidate_recency() {
        let cfg = RowSelectionConfig {
            selection: Some(SelectionMethod::Dynamic {
                start_iteration: 2,
                seed_window: 5,
                candidate_recency: Some(0),
                max_added_per_round: 10,
                violation_tolerance: 1e-10,
            }),
            ..RowSelectionConfig::default()
        };
        let msg = parse_cut_selection_config(&cfg)
            .expect_err("zero candidate_recency must be rejected for dynamic");
        assert!(
            msg.contains("candidate_recency must be >= 1"),
            "error must mention the candidate_recency constraint, got: {msg}"
        );
    }

    /// Every explicit dynamic field flows into the `Dynamic` variant.
    #[test]
    fn parse_dynamic_explicit_fields() {
        let cfg = RowSelectionConfig {
            selection: Some(SelectionMethod::Dynamic {
                start_iteration: 5,
                seed_window: 7,
                candidate_recency: Some(20),
                max_added_per_round: 3,
                violation_tolerance: 1e-9,
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
                    k1: Some(20),
                    k2: 7,
                    nadic: 3,
                    epsilon_viol,
                    start_iteration: 5,
                } if (epsilon_viol - 1e-9).abs() < f64::EPSILON
            ),
            "unexpected variant or explicit fields: {strategy:?}"
        );
    }

    /// `start_iteration = 0` is rejected with a message naming the field and the
    /// `>= 1` constraint.
    #[test]
    fn parse_dynamic_rejects_zero_start_iteration() {
        let cfg = RowSelectionConfig {
            selection: Some(SelectionMethod::Dynamic {
                start_iteration: 0,
                seed_window: 5,
                candidate_recency: None,
                max_added_per_round: 10,
                violation_tolerance: 1e-10,
            }),
            ..RowSelectionConfig::default()
        };
        let msg = parse_cut_selection_config(&cfg)
            .expect_err("zero start_iteration must be rejected for dynamic");
        assert!(
            msg.contains("start_iteration") && msg.contains(">= 1"),
            "error must mention start_iteration and '>= 1', got: {msg}"
        );
    }

    /// A zero `max_added_per_round` is rejected with a naming message.
    #[test]
    fn parse_dynamic_rejects_zero_max_added_per_round() {
        let cfg = RowSelectionConfig {
            selection: Some(SelectionMethod::Dynamic {
                start_iteration: 2,
                seed_window: 5,
                candidate_recency: None,
                max_added_per_round: 0,
                violation_tolerance: 1e-10,
            }),
            ..RowSelectionConfig::default()
        };
        let msg = parse_cut_selection_config(&cfg)
            .expect_err("zero max_added_per_round must be rejected for dynamic");
        assert!(
            msg.contains("max_added_per_round") && msg.contains(">= 1"),
            "error must name max_added_per_round and '>= 1', got: {msg}"
        );
    }

    /// `seed_window` feeds `k2`: `0` is valid (seeds only the current
    /// iteration) and the absent default is `5`.
    #[test]
    fn parse_dynamic_seed_window_sets_k2() {
        let cfg_zero =
            dynamic_from_json(r#"{"selection": {"method": "dynamic", "seed_window": 0}}"#);
        let strategy = parse_cut_selection_config(&cfg_zero)
            .expect("seed_window = 0 must parse for dynamic (0 is valid)")
            .expect("must produce Some for dynamic");
        assert!(
            matches!(strategy, CutSelectionStrategy::Dynamic { k2: 0, .. }),
            "seed_window = 0 must set k2 = 0: {strategy:?}"
        );

        let cfg_absent = dynamic_from_json(r#"{"selection": {"method": "dynamic"}}"#);
        let strategy = parse_cut_selection_config(&cfg_absent)
            .expect("absent seed_window must parse for dynamic")
            .expect("must produce Some for dynamic");
        assert!(
            matches!(strategy, CutSelectionStrategy::Dynamic { k2: 5, .. }),
            "absent seed_window must default k2 to 5: {strategy:?}"
        );
    }

    /// `should_run` returns `false` for `Dynamic` at every iteration.
    #[test]
    fn dynamic_should_run_always_false() {
        let strategy = CutSelectionStrategy::Dynamic {
            k1: None,
            k2: 5,
            nadic: 10,
            epsilon_viol: 1e-10,
            start_iteration: 2,
        };
        for iteration in [0_u64, 1, 2, 5, 100] {
            assert!(
                !strategy.should_run(iteration),
                "Dynamic must never run as a pool-level pass (iteration {iteration})"
            );
        }
    }

    /// `select_for_stage` returns empty updates for `Dynamic`, regardless of a
    /// non-empty pool and visited states.
    #[test]
    fn dynamic_select_returns_empty() {
        let strategy = CutSelectionStrategy::Dynamic {
            k1: None,
            k2: 5,
            nadic: 10,
            epsilon_viol: 1e-10,
            start_iteration: 2,
        };
        let mut pool = CutPool::new(3, 1, 1, 0);
        pool.add_cut(NodeId(0), 0, 0, 1.0, &[0.0]);
        pool.add_cut(NodeId(0), 1, 0, 5.0, &[0.0]);
        pool.add_cut(NodeId(0), 2, 0, 3.0, &[0.0]);

        let result = strategy.select_for_stage(&pool, &[0.0], 1, 10, 4);
        assert!(
            result.updates.is_empty(),
            "Dynamic must produce no deactivations"
        );
        assert!(
            result.reactivations.is_empty(),
            "Dynamic must produce no reactivations"
        );

        // `select` delegates to `select_for_stage` with stage 0.
        let via_select = strategy.select(&pool, &[0.0], 10);
        assert!(via_select.updates.is_empty());
        assert!(via_select.reactivations.is_empty());
    }

    /// derived: at the state 0 the cut values are 1, 5 and 3, so `Dominated` at a
    /// zero threshold deactivates slots 0 and 2 of the same pool; `Level1` and
    /// `Dynamic` are not served by the kernel.
    #[test]
    fn the_windowed_kernel_selects_nothing_for_level1_or_dynamic() {
        let mut pool = CutPool::new(3, 1, 1, 0);
        pool.add_cut(NodeId(0), 0, 0, 1.0, &[0.0]);
        pool.add_cut(NodeId(0), 1, 0, 5.0, &[0.0]);
        pool.add_cut(NodeId(0), 2, 0, 3.0, &[0.0]);
        let no_update = |stage_index| CutActivityUpdates {
            stage_index,
            updates: vec![],
            reactivations: vec![],
        };

        let dominated = CutSelectionStrategy::Dominated {
            threshold: 0.0,
            check_frequency: 1,
        };
        assert_eq!(
            dominated.select_for_stage(&pool, &[0.0], 1, 10, 4).updates,
            vec![0, 2]
        );

        let level1 = CutSelectionStrategy::Level1 {
            check_frequency: 1,
            tie_tolerance: 1e-10,
            lookback_iterations: None,
        };
        let dynamic = CutSelectionStrategy::Dynamic {
            k1: None,
            k2: 5,
            nadic: 10,
            epsilon_viol: 1e-10,
            start_iteration: 2,
        };
        for strategy in [level1, dynamic] {
            assert_eq!(
                strategy.select_for_stage(&pool, &[0.0], 1, 10, 4),
                no_update(4),
                "{strategy:?}"
            );
            assert_eq!(
                strategy.select(&pool, &[0.0], 10),
                no_update(0),
                "{strategy:?}"
            );
        }
    }
}
