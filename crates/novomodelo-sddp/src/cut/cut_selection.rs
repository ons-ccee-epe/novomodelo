//! Cut selection strategies for controlling cut pool growth during SDDP
//! training.
//!
//! # Determinism
//!
//! `matrixmultiply`'s micro-kernel is bit-deterministic for any given input
//! shape, and the per-task OR-merge is commutative + associative, so
//! `RAYON_NUM_THREADS=1` and `=96` produce identical output on the same binary.
//!
//! # Algorithm semantics
//!
//! The value-based variants share a single value-evaluation kernel in
//! [`CutSelectionStrategy::select_for_stage`], which evaluates ALL populated
//! cuts (active AND inactive) at each visited forward-pass state and applies the
//! method-specific survival rule:
//!
//! - **Level1**: retain any cut within `tie_tolerance` of the per-state maximum
//!   at any visited state (de Matos 2015).
//! - **Lml1**: at each visited state, retain only the oldest eligible cut
//!   within `tie_tolerance` of the maximum; the overall selected set is the
//!   union across all visited states (Guigues & Bandarra 2019).
//! - **Dominated**: same max-survival logic as Level1 using `threshold` as the
//!   tolerance, applied across ALL populated cuts.
//!
//! # Usage
//!
//! ```rust
//! use cobre_sddp::cut::CutPool;
//! use cobre_sddp::cut_selection::{
//!     CutActivityUpdates, CutMetadata, CutSelectionStrategy,
//! };
//!
//! let strategy = CutSelectionStrategy::Level1 {
//!     check_frequency: 5,
//!     tie_tolerance: 1e-10,
//! };
//!
//! // Should run at multiples of check_frequency (excluding 0).
//! assert!(!strategy.should_run(0));
//! assert!(!strategy.should_run(3));
//! assert!(strategy.should_run(5));
//! assert!(strategy.should_run(10));
//! ```

use rayon::prelude::*;

use crate::cut::CutPool;
use crate::gemm::gemm_block;
use crate::setup::NodeId;

use cobre_io::config::RowSelectionConfig;

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
    /// Iteration at which this cut was generated (1-based). Cuts from the
    /// current iteration are never deactivated.
    pub iteration_generated: u64,

    /// Forward pass index that generated this cut.
    pub forward_pass_index: u32,

    /// Declared id (`NodeGraph::node_ids`) of the node that generated this cut;
    /// carried onto the MPI cut wire ([`crate::cut::wire::serialize_cut`]) so a
    /// shared-leaf pool's records stay distinguishable by generating node. Not
    /// read by the value-based selection logic.
    pub node: NodeId,

    /// Cumulative number of times this cut was binding at an LP solution. Used
    /// by budget enforcement and diagnostics, NOT by the value-based selection
    /// logic.
    pub active_count: u64,

    /// Most recent iteration at which this cut was binding. Used by budget
    /// enforcement (staleness eviction) and diagnostics, NOT by the value-based
    /// selection logic.
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
    /// Level-1 selection: retain any cut within `tie_tolerance` of the per-state
    /// maximum at some visited state (de Matos 2015). The least aggressive
    /// value-based strategy; preserves the convergence guarantee.
    Level1 {
        /// Number of iterations between selection runs. Must be > 0.
        check_frequency: u64,

        /// Tie-break tolerance: a cut is active at a state when within
        /// `tie_tolerance` of the best value there. Default: `1e-10`.
        tie_tolerance: f64,
    },

    /// Limited Memory Level-1 (LML1): retain only the oldest eligible cut within
    /// `tie_tolerance` of the maximum per visited state, unioned across states
    /// (Guigues & Bandarra 2019). More aggressive than Level1.
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

    /// Dynamic Cut Selection (DCS): a per-solve lazy selection loop, NOT a
    /// pool-level deactivation pass. [`should_run`] always returns `false` and
    /// [`select_for_stage`] always returns empty [`CutActivityUpdates`];
    /// selection happens lazily inside each LP solve. Because exactness rests on
    /// the `k1 = None` (∞) default, DCS is mutually exclusive with the
    /// value-based pool passes.
    ///
    /// [`should_run`]: CutSelectionStrategy::should_run
    /// [`select_for_stage`]: CutSelectionStrategy::select_for_stage
    Dynamic {
        /// Candidate-recency window. `None` ⇒ ∞ (every pool cut a candidate;
        /// exactness-preserving). `Some(n)` ⇒ only cuts generated within the
        /// last `n` iterations are candidates — deliberately NOT exact: older
        /// cuts are never added even when violated.
        k1: Option<u32>,

        /// Maximum number of cuts added per lazy-solve round.
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
    /// use cobre_sddp::cut_selection::CutSelectionStrategy;
    ///
    /// let s = CutSelectionStrategy::Level1 { check_frequency: 5, tie_tolerance: 1e-10 };
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

    /// Scan the cut pool for stage 0 (a pure query: the pool is not modified).
    ///
    /// Delegates to [`select_for_stage`](Self::select_for_stage) with
    /// `stage_index = 0`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cobre_sddp::cut::{CutPool};
    /// use cobre_sddp::cut_selection::{CutMetadata, CutSelectionStrategy};
    /// use cobre_sddp::setup::NodeId;
    ///
    /// let strategy = CutSelectionStrategy::Level1 { check_frequency: 5, tie_tolerance: 1e-10 };
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

    /// Scan the cut pool for a specific stage (a pure query: the pool is not
    /// modified).
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
        if let CutSelectionStrategy::Dynamic { .. } = self {
            return CutActivityUpdates {
                stage_index,
                updates: vec![],
                reactivations: vec![],
            };
        }

        let populated = pool.populated();
        let n_state = pool.state_dimension;
        let warm_start = pool.warm_start_count as usize;

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
            .map(|k| k >= warm_start && pool.metadata(k).iteration_generated < current_iteration)
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

                    for col in 0..m_len {
                        apply_column_rule(
                            self,
                            v_block_active,
                            populated,
                            m_len,
                            col,
                            warm_start,
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
        for k in warm_start..populated {
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
/// The per-column max is computed over ALL populated cuts (active and inactive).
/// Level1/Dominated mark every eligible cut within `tie_tolerance` of the max;
/// Lml1 marks only the oldest such cut. The ascending slot walk makes Lml1's
/// "oldest at max" deterministic (D5).
// Rationale: each parameter is a distinct kernel dimension with no natural
// grouping; collapsing any pair would obscure the O(populated × m_len) sweep.
#[allow(clippy::too_many_arguments)]
#[inline]
fn apply_column_rule(
    method: &CutSelectionStrategy,
    v_block: &[f64],
    populated: usize,
    m_len: usize,
    col: usize,
    warm_start: usize,
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
        CutSelectionStrategy::Level1 { tie_tolerance, .. }
        | CutSelectionStrategy::Dominated {
            threshold: tie_tolerance,
            ..
        } => {
            let cutoff = max_val - tie_tolerance;
            for k in warm_start..populated {
                if eligible[k] && v_block[k * m_len + col] >= cutoff {
                    bitmap[k] = true;
                }
            }
        }
        CutSelectionStrategy::Lml1 { tie_tolerance, .. } => {
            let cutoff = max_val - tie_tolerance;
            for k in warm_start..populated {
                if eligible[k] && v_block[k * m_len + col] >= cutoff {
                    bitmap[k] = true;
                    break;
                }
            }
        }
        CutSelectionStrategy::Dynamic { .. } => {
            unreachable!("Dynamic cut selection does not run the value-evaluation kernel")
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

/// Parse a [`RowSelectionConfig`] into an optional
/// [`CutSelectionStrategy`].
///
/// Returns `None` when `selection` is absent (row selection disabled).
/// Otherwise maps the chosen [`SelectionMethod`] onto the matching
/// [`CutSelectionStrategy`], validating the variant's numeric constraints.
///
/// # Errors
///
/// Returns `Err(String)` when a method's numeric constraint is violated:
/// `check_frequency = 0` for a periodic method; or, for the dynamic method,
/// `start_iteration = 0`, `candidate_recency = Some(0)`,
/// `max_added_per_round = 0`, or `violation_tolerance <= 0`.
///
/// [`SelectionMethod`]: cobre_io::config::SelectionMethod
pub fn parse_cut_selection_config(
    config: &RowSelectionConfig,
) -> Result<Option<CutSelectionStrategy>, String> {
    use cobre_io::config::SelectionMethod;

    let Some(selection) = config.selection.as_ref() else {
        return Ok(None);
    };

    match *selection {
        SelectionMethod::Level1 {
            tie_tolerance,
            check_frequency,
        } => Ok(Some(CutSelectionStrategy::Level1 {
            check_frequency: u64::from(validate_check_frequency(check_frequency)?),
            tie_tolerance,
        })),
        SelectionMethod::Lml1 {
            tie_tolerance,
            check_frequency,
        } => Ok(Some(CutSelectionStrategy::Lml1 {
            check_frequency: u64::from(validate_check_frequency(check_frequency)?),
            tie_tolerance,
        })),
        SelectionMethod::Domination {
            domination_tolerance,
            check_frequency,
        } => Ok(Some(CutSelectionStrategy::Dominated {
            threshold: domination_tolerance,
            check_frequency: u64::from(validate_check_frequency(check_frequency)?),
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

            if violation_tolerance <= 0.0 {
                return Err(
                    "cut_selection.violation_tolerance must be > 0 for method='dynamic'"
                        .to_string(),
                );
            }

            Ok(Some(CutSelectionStrategy::Dynamic {
                k1: candidate_recency,
                k2: seed_window,
                nadic: max_added_per_round,
                epsilon_viol: violation_tolerance,
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
    use super::parse_cut_selection_config;
    use super::{CutActivityUpdates, CutMetadata, CutSelectionStrategy, NodeId};
    use crate::cut::CutPool;
    use cobre_io::config::{RowSelectionConfig, SelectionMethod};

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
        };
        assert!(!s.should_run(0));
    }

    #[test]
    fn should_run_false_between_multiples() {
        let s = CutSelectionStrategy::Level1 {
            check_frequency: 5,
            tie_tolerance: 1e-10,
        };
        assert!(!s.should_run(3));
        assert!(!s.should_run(7));
    }

    #[test]
    fn should_run_true_at_multiples() {
        let s = CutSelectionStrategy::Level1 {
            check_frequency: 5,
            tie_tolerance: 1e-10,
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
    // Level1 value-based kernel tests
    // -----------------------------------------------------------------------

    /// Pool with 3 cuts (intercepts [1,5,3], coeff all 0), state [0.0].
    /// Level1 `tie_tolerance=0.0` → cut 1 (value 5) survives; cuts 0,2 deactivated.
    #[test]
    fn level1_deactivates_dominated_cuts_at_state() {
        let strategy = CutSelectionStrategy::Level1 {
            check_frequency: 5,
            tie_tolerance: 0.0,
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
    /// Level1 tie_tolerance=1e-10 → cuts 0 and 1 both survive (tie kept).
    #[test]
    fn level1_retains_tied_cuts() {
        let strategy = CutSelectionStrategy::Level1 {
            check_frequency: 5,
            tie_tolerance: 1e-10,
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

    /// Level1 retains all cuts when two have equal max values.
    #[test]
    fn level1_retains_positive_activity_cuts() {
        let strategy = CutSelectionStrategy::Level1 {
            check_frequency: 5,
            tie_tolerance: 0.0,
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

    /// Level1 with three cuts where two are clearly below max.
    #[test]
    fn level1_threshold_1_deactivates_cuts_with_count_at_most_1() {
        let strategy = CutSelectionStrategy::Level1 {
            check_frequency: 5,
            tie_tolerance: 0.5,
        };
        // cut0: value=1, cut1: value=3, cut2: value=2 at state [0.0].
        // max=3, cutoff=3-0.5=2.5; only cut1(3>=2.5) and cut2(2<2.5→no) wait:
        // cut2=2 < 2.5 → not selected. cut0=1 < 2.5 → not selected. cut1=3 >= 2.5 → selected.
        // Deactivate: cut0, cut2.
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
    fn level1_empty_metadata_returns_empty_set() {
        let strategy = CutSelectionStrategy::Level1 {
            check_frequency: 5,
            tie_tolerance: 1e-10,
        };
        let pool = CutPool::new(0, 1, 1, 0);
        let deact = strategy.select(&pool, &[], 10);
        assert!(deact.updates.is_empty());
    }

    #[test]
    fn level1_empty_states_returns_empty() {
        let strategy = CutSelectionStrategy::Level1 {
            check_frequency: 5,
            tie_tolerance: 1e-10,
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

    /// Same pool as `level1_retains_tied_cuts` (intercepts [5,5,3]), state [0.0].
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

    /// Lml1: single eligible cut → `n_eligible` < 2 → empty.
    #[test]
    fn lml1_deactivates_cuts_outside_memory_window() {
        let strategy = CutSelectionStrategy::Lml1 {
            check_frequency: 5,
            tie_tolerance: 1e-10,
        };
        // Only 1 eligible cut → n_eligible < 2 → returns empty.
        let pool = make_pool(&[make_meta(0, 5)], &[true]);
        let deact = strategy.select(&pool, &[0.0], 20);
        assert!(deact.updates.is_empty());
    }

    /// Lml1 with two eligible cuts at the same value: oldest (slot 0) retained.
    #[test]
    fn lml1_retains_cuts_within_memory_window() {
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

    /// Lml1 with 3 cuts at state [0.0]: only oldest at max survives.
    #[test]
    fn lml1_retains_cuts_exactly_at_boundary() {
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

    /// A cut with lower value is deactivated.
    #[test]
    fn ac_level1_threshold_0_deactivates_zero_activity_cut() {
        let strategy = CutSelectionStrategy::Level1 {
            check_frequency: 5,
            tie_tolerance: 0.0,
        };
        // cut0: value=1 at state [0]. cut1: value=3 at state [0].
        // max=3, cutoff=3. cut0(1<3) not selected → deactivated.
        let mut pool = CutPool::new(2, 1, 1, 0);
        pool.add_cut(NodeId(0), 0, 0, 1.0, &[0.0]);
        pool.add_cut(NodeId(0), 1, 0, 3.0, &[0.0]);
        let deact = strategy.select(&pool, &[0.0], 10);
        assert!(deact.updates.contains(&0));
    }

    /// Lml1 deactivates a cut that is never oldest-at-max.
    #[test]
    fn ac_lml1_deactivates_cut_outside_memory_window() {
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
        assert!(deact.updates.contains(&1));
    }

    #[test]
    fn select_for_stage_sets_stage_index() {
        let strategy = CutSelectionStrategy::Level1 {
            check_frequency: 5,
            tie_tolerance: 1e-10,
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
        let strategy = CutSelectionStrategy::Level1 {
            check_frequency: 1,
            tie_tolerance: 0.0,
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
        let strategy = CutSelectionStrategy::Level1 {
            check_frequency: 5,
            tie_tolerance: 1e-10,
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
    /// Level1, even if they have lower values. They haven't been tested yet
    /// and deactivating them would cause the lower bound to stagnate.
    #[test]
    fn level1_spares_cuts_from_current_iteration() {
        let strategy = CutSelectionStrategy::Level1 {
            check_frequency: 1,
            tie_tolerance: 1e-10,
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

    /// Lml1 also spares cuts from the current iteration via the
    /// `iteration_generated` guard.
    #[test]
    fn lml1_spares_cuts_from_current_iteration() {
        let strategy = CutSelectionStrategy::Lml1 {
            check_frequency: 1,
            tie_tolerance: 1e-10,
        };
        let pool = make_pool(
            &[CutMetadata {
                iteration_generated: 10,
                forward_pass_index: 0,
                node: NodeId(0),
                active_count: 0,
                last_active_iter: 10,
            }],
            &[true],
        );
        // Only 1 cut, from current iteration (not eligible) → n_eligible = 0 < 2 → empty.
        let deact = strategy.select(&pool, &[0.0], 10);
        assert!(
            deact.updates.is_empty(),
            "current-iteration cut must not be deactivated"
        );
    }

    /// Lml1 memory window boundary behavior: 5 cuts, 2 trial points.
    /// Each trial point selects a different oldest cut; union determines survivors.
    #[test]
    fn lml1_memory_window_boundary_behavior() {
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
    fn level1_reactivates_inactive_cut_at_max() {
        let strategy = CutSelectionStrategy::Level1 {
            check_frequency: 1,
            tie_tolerance: 1e-10,
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
        let strategy = CutSelectionStrategy::Level1 {
            check_frequency: 1,
            tie_tolerance: 1e-10,
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

    /// Warm-start cuts participate in max computation but are not candidates.
    /// With `warm_start_count=1`: slot 0 is protected; only slots 1+ are eligible.
    #[allow(clippy::cast_possible_truncation)]
    #[test]
    fn level1_warm_start_cuts_not_deactivated() {
        let strategy = CutSelectionStrategy::Level1 {
            check_frequency: 1,
            tie_tolerance: 0.0,
        };
        // warm_start_count=1 → slot 0 is warm-start (protected).
        // Slot 0 comes from a warm-start record (add_cut cannot address it once
        // warm_start_count > 0); slots 1,2 come from add_cut.
        // 3 slots total: slot 0 (warm-start, intercept=10), slot 1 (eligible, =1), slot 2 (eligible, =3).
        // max=10 (slot 0). Cutoff=10. Eligible cuts 1,2 both below cutoff → deactivated.
        // Slot 0 is not eligible (warm-start) → not deactivated.
        let warm_start_records = vec![cobre_io::OwnedPolicyCutRecord {
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
        // Warm-start slot 0 must not appear in deactivations.
        assert!(
            !result.updates.contains(&0),
            "warm-start slot 0 must not be deactivated"
        );
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

        // Level1 tie_tolerance=0: only slot 1 (value=5) is at max.
        // slot 0: eligible, not selected, inactive → no change (not reactivated).
        // slot 1: eligible, selected, active → no change.
        // slot 2: eligible, not selected, active → deactivated.
        let strategy = CutSelectionStrategy::Level1 {
            check_frequency: 1,
            tie_tolerance: 0.0,
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

        let strategy = CutSelectionStrategy::Level1 {
            check_frequency: 1,
            tie_tolerance: 0.0,
        };
        let result = strategy.select_for_stage(&pool, &[0.0], 1, 10, 0);

        // cut0 (value=1) not at max → deactivated. cut1 (value=2) at max → no change.
        assert!(
            !result.updates.is_empty(),
            "deactivations must be non-empty"
        );
        assert!(
            result.updates.contains(&0),
            "slot 0 (below max) must be deactivated"
        );
        assert!(
            !result.updates.contains(&1),
            "slot 1 (at max) must not be deactivated"
        );
    }

    // -----------------------------------------------------------------------
    // Aggressiveness ordering
    // -----------------------------------------------------------------------

    /// Level1 and Lml1 with empty states return empty; Dominated with states
    /// returns some. Ordering: |Level1| <= |Lml1| <= |Dominated|.
    #[test]
    fn aggressiveness_ordering_level1_leq_lml1_leq_dominated() {
        // 5 cuts (1D):
        // Cut 0: intercept=0, slope=0 (constant 0)
        // Cut 1: intercept=0, slope=0.1
        // Cut 2: intercept=1, slope=0
        // Cut 3: intercept=0, slope=2
        // Cut 4: intercept=5, slope=-1
        let meta = [
            CutMetadata {
                iteration_generated: 1,
                forward_pass_index: 0,
                node: NodeId(0),
                active_count: 0,
                last_active_iter: 1,
            },
            CutMetadata {
                iteration_generated: 1,
                forward_pass_index: 1,
                node: NodeId(0),
                active_count: 0,
                last_active_iter: 2,
            },
            CutMetadata {
                iteration_generated: 1,
                forward_pass_index: 2,
                node: NodeId(0),
                active_count: 3,
                last_active_iter: 3,
            },
            CutMetadata {
                iteration_generated: 1,
                forward_pass_index: 3,
                node: NodeId(0),
                active_count: 5,
                last_active_iter: 10,
            },
            CutMetadata {
                iteration_generated: 1,
                forward_pass_index: 4,
                node: NodeId(0),
                active_count: 5,
                last_active_iter: 10,
            },
        ];
        let pool = make_dominated_pool(
            &[0.0, 0.0, 1.0, 0.0, 5.0],
            &[vec![0.0], vec![0.1], vec![0.0], vec![2.0], vec![-1.0]],
            &[true; 5],
            &meta,
        );
        let states: Vec<f64> = vec![0.0, 1.0, 3.0, 5.0];

        // Level1 and Lml1 use value-based evaluation with actual states.
        let l1 = CutSelectionStrategy::Level1 {
            check_frequency: 1,
            tie_tolerance: 1e-10,
        };
        let deact_l1 = l1.select(&pool, &states, 11);

        let lml1 = CutSelectionStrategy::Lml1 {
            check_frequency: 1,
            tie_tolerance: 1e-10,
        };
        let deact_lml1 = lml1.select(&pool, &states, 11);

        // Dominated threshold=0
        let dom = CutSelectionStrategy::Dominated {
            threshold: 0.0,
            check_frequency: 1,
        };
        let deact_dom = dom.select(&pool, &states, 11);

        assert!(
            deact_l1.updates.len() <= deact_lml1.updates.len(),
            "Level1 ({}) should deactivate <= LML1 ({})",
            deact_l1.updates.len(),
            deact_lml1.updates.len()
        );
        assert!(
            deact_lml1.updates.len() <= deact_dom.updates.len(),
            "LML1 ({}) should deactivate <= Dominated ({})",
            deact_lml1.updates.len(),
            deact_dom.updates.len()
        );
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

    // -----------------------------------------------------------------------
    // Dominated algorithm tests (conformance + aggressiveness ordering)
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

    #[test]
    fn dominated_select_current_iteration_excluded() {
        let strategy = CutSelectionStrategy::Dominated {
            threshold: 0.0,
            check_frequency: 1,
        };
        // Cut 0: constant 1 (from current iteration 10 -- protected)
        // Cut 1: constant 5 (from iteration 1 -- dominates cut 0)
        let pool = make_dominated_pool(
            &[1.0, 5.0],
            &[vec![0.0], vec![0.0]],
            &[true, true],
            &[default_meta_at(10), default_meta_at(1)],
        );
        let states: Vec<f64> = vec![0.0, 1.0];
        let deact = strategy.select(&pool, &states, 10);
        assert!(
            deact.updates.is_empty(),
            "cut from current iteration must not be deactivated even if dominated"
        );
    }

    // -----------------------------------------------------------------------
    // Set-inclusion property: Level1_selected ⊇ Lml1_selected
    // -----------------------------------------------------------------------

    /// Every cut that Lml1 keeps must also be kept by
    /// Level1 — i.e., every slot in Lml1's deactivation list also appears in
    /// Level1's deactivation list.  The existing aggressiveness ordering test
    /// only checks that `|deact_L1|` <= `|deact_Lml1|`; this test directly verifies
    /// the subset relationship on slot indices.
    ///
    /// Fixture: 4 cuts (1D), 2 trial points.
    /// cut0: constant 1  (coeff=0, intercept=1)
    /// cut1: constant 3  (coeff=0, intercept=3)
    /// cut2: 2x          (coeff=2, intercept=0)
    /// cut3: constant 0  (coeff=0, intercept=0) -- always below max
    ///
    /// At state [0.0]: values=[1,3,0,0] → max=3 (cut1). Level1 keeps cut1.
    ///   Lml1 keeps oldest at max = cut1.
    /// At state [2.0]: values=[1,3,4,0] → max=4 (cut2). Level1 keeps cut1,cut2.
    ///   Lml1 keeps oldest at max = cut2.
    ///
    /// Level1 deactivates: {cut0, cut3}.
    /// Lml1 deactivates: {cut0, cut3} (same here, but the property holds).
    ///
    /// Set-inclusion check: every slot in `deact_lml1` must also be in `deact_l1`.
    #[test]
    fn level1_selected_is_superset_of_lml1_selected() {
        let meta = default_meta_vec(4, 1);
        let pool = make_dominated_pool(
            &[1.0, 3.0, 0.0, 0.0],
            &[vec![0.0], vec![0.0], vec![2.0], vec![0.0]],
            &[true; 4],
            &meta,
        );
        let states: Vec<f64> = vec![0.0, 2.0];

        let l1 = CutSelectionStrategy::Level1 {
            check_frequency: 1,
            tie_tolerance: 1e-10,
        };
        let lml1 = CutSelectionStrategy::Lml1 {
            check_frequency: 1,
            tie_tolerance: 1e-10,
        };

        let deact_l1 = l1.select(&pool, &states, 10);
        let deact_lml1 = lml1.select(&pool, &states, 10);

        // Set-inclusion: every slot Lml1 deactivates must also be deactivated by Level1.
        // Equivalently: the Level1 survivor set is a superset of the Lml1 survivor set.
        for &slot in &deact_lml1.updates {
            assert!(
                deact_l1.updates.contains(&slot),
                "slot {slot} deactivated by Lml1 but not by Level1; \
                 Level1_selected must be a superset of Lml1_selected"
            );
        }
        // Sanity: Level1 must not deactivate more than it could (count check too).
        assert!(
            deact_l1.updates.len() <= deact_lml1.updates.len(),
            "Level1 must deactivate <= Lml1"
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

    /// Level1 with exactly one eligible cut returns empty updates (`n_eligible` < 2
    /// guard fires before any evaluation).
    ///
    /// Two slots total: slot 0 is from `current_iteration` (ineligible); slot 1 is
    /// eligible. `n_eligible=1` → empty.
    #[test]
    fn level1_single_eligible_cut_returns_empty() {
        let strategy = CutSelectionStrategy::Level1 {
            check_frequency: 1,
            tie_tolerance: 1e-10,
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
    // Intra-stage parallelism determinism tests
    // -----------------------------------------------------------------------

    /// Build a 100-cut, 1-D pool whose intercepts and coefficients are
    /// deterministic functions of the slot index — designed so that different
    /// cuts achieve the max at different trial points, exercising both
    /// Level1's "any-max" and Lml1's "oldest-at-max" branches.
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

    /// Determinism: identical bit-for-bit output for 1 vs 4 vs 8 threads at
    /// a scale that exercises many m-blocks per stage.
    ///
    /// 1024 trial points yields 128 m-blocks under `M_BLOCK = 8`. The
    /// per-task OR-merge is commutative and associative, so any worker
    /// assignment must yield the same `is_selected` bitmap and therefore
    /// identical `CutActivityUpdates`.
    #[test]
    fn select_for_stage_deterministic_across_thread_counts_level1() {
        let pool = make_determinism_pool();
        let states = make_determinism_states(1024);
        let strategy = CutSelectionStrategy::Level1 {
            check_frequency: 1,
            tie_tolerance: 1e-10,
        };
        let r1 = run_in_pool(&strategy, &pool, &states, 10, 1);
        let r4 = run_in_pool(&strategy, &pool, &states, 10, 4);
        let r8 = run_in_pool(&strategy, &pool, &states, 10, 8);
        assert_eq!(
            r1, r4,
            "Level1: 1-thread vs 4-thread results must be bit-identical"
        );
        assert_eq!(
            r4, r8,
            "Level1: 4-thread vs 8-thread results must be bit-identical"
        );
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

    /// Determinism for Dominated: same proof structure as Level1.
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
            CutSelectionStrategy::Level1 {
                check_frequency: 1,
                tie_tolerance: 1e-10,
            },
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
        let cfg = RowSelectionConfig {
            selection: Some(SelectionMethod::Dynamic {
                start_iteration: 2,
                seed_window: 5,
                candidate_recency: None,
                max_added_per_round: 10,
                violation_tolerance: -1.0,
            }),
            ..RowSelectionConfig::default()
        };
        let msg = parse_cut_selection_config(&cfg)
            .expect_err("non-positive violation_tolerance must be rejected for dynamic");
        assert!(
            msg.contains("violation_tolerance must be > 0"),
            "error must mention the violation_tolerance constraint, got: {msg}"
        );
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
}
