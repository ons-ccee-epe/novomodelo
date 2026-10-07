//! Per-class noise source ([`ClassSampler`]) combined three-up (inflow, load,
//! NCS) by the composite `ForwardSampler`.
//!
//! ## Correlation contract
//!
//! `InSample`/`Historical`/`External` write pre-correlated noise (tree and
//! libraries store pre-standardized eta). Only `OutOfSample` writes independent
//! N(0,1); the composite `ForwardSampler` correlates it, not this module.
//!
//! ## Window / scenario selection
//!
//! `Historical` and `External` selection is deterministic in `(iteration,
//! scenario)` and excludes `stage`, so one window/scenario serves all stages of
//! a forward trajectory.

use cobre_core::temporal::NoiseMethod;

use crate::{
    StochasticError,
    noise::seed::derive_forward_seed,
    sampling::{
        ClassNoiseTables, ExternalScenarioLibrary, HistoricalScenarioLibrary,
        out_of_sample::{FreshNoiseSpec, fill_uncorrelated},
    },
    tree::opening_tree::OpeningTreeView,
};

use super::insample;

use std::fmt;

/// Per-call arguments for [`ClassSampler::fill`].
#[derive(Debug, Clone, Copy)]
pub struct ClassSampleRequest {
    /// Training iteration counter (0-based).
    pub iteration: u32,
    /// Global scenario index (includes MPI offset).
    pub scenario: u32,
    /// Stage domain ID used for seed derivation.
    pub stage: u32,
    /// Stage array index used for tree/method lookup.
    pub stage_idx: usize,
    /// Total scenario count across all ranks (for LHS stratification).
    pub total_scenarios: u32,
    /// Seed-derivation identifier: stages sharing a `(season_id, year)` bucket
    /// share a `noise_group_id` so their `OutOfSample` draws are identical.
    pub noise_group_id: u32,
    /// Sampled node's Ω sub-range within this stage's generated opening set —
    /// `ClassSampler::InSample` draws its opening index inside
    /// `node_opening_offset..node_opening_offset + node_opening_len` instead of
    /// the full stage set. A chain-degenerate node's view spans the whole stage
    /// (`NodeGraph`'s synthesized `(0, tree.n_openings(stage))`), so a chain's
    /// draw is unchanged. Every other scheme ignores both fields.
    pub node_opening_offset: usize,
    /// See [`Self::node_opening_offset`].
    pub node_opening_len: usize,
    /// Optional scenario-column pin. `External` reads column `k` when `Some(k)`,
    /// else hash-selects; every other arm ignores it.
    pub pinned_scenario: Option<usize>,
}

/// Per-class noise source for one entity class (inflow, load, or NCS).
pub enum ClassSampler<'a> {
    /// In-sample scheme: copies a segment from the pre-generated opening tree.
    InSample {
        /// View into the pre-generated opening tree.
        tree: OpeningTreeView<'a>,
        /// Base seed for deterministic opening selection.
        base_seed: u64,
        /// Start offset within the full noise vector for this class.
        offset: usize,
        /// Number of entities in this class (segment length).
        len: usize,
    },
    /// Out-of-sample scheme: fresh independent N(0,1) noise. Correlation is NOT
    /// applied here — the composite `ForwardSampler` applies it after all classes
    /// have filled their segments.
    OutOfSample {
        /// Seed for the forward-pass noise generator.
        forward_seed: u64,
        /// Per-class dimension (number of entities in this class).
        dim: usize,
        /// Per-stage noise generation method for this class.
        noise_methods: Box<[NoiseMethod]>,
    },
    /// Historical replay scheme: replays a pre-standardized window from the library.
    Historical {
        /// Borrowed pre-standardized historical library.
        library: &'a HistoricalScenarioLibrary,
    },
    /// External scenario file scheme: reads from a pre-standardized external library.
    External {
        /// Borrowed pre-standardized external library.
        library: &'a ExternalScenarioLibrary,
    },
}

impl std::fmt::Debug for ClassSampler<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ClassSampler::InSample {
                base_seed,
                offset,
                len,
                ..
            } => f
                .debug_struct("ClassSampler::InSample")
                .field("base_seed", base_seed)
                .field("offset", offset)
                .field("len", len)
                .finish_non_exhaustive(),
            ClassSampler::OutOfSample {
                forward_seed,
                dim,
                noise_methods,
            } => f
                .debug_struct("ClassSampler::OutOfSample")
                .field("forward_seed", forward_seed)
                .field("dim", dim)
                .field("noise_methods", noise_methods)
                .finish_non_exhaustive(),
            ClassSampler::Historical { .. } => write!(f, "ClassSampler::Historical(..)"),
            ClassSampler::External { .. } => write!(f, "ClassSampler::External(..)"),
        }
    }
}

/// Window-selection seed base, distinct from the forward-pass noise domain so the
/// two hash domains never collide.
const HISTORICAL_SELECTION_BASE_SEED: u64 = 0x6869_7374_6f72_6963; // b"historic" as u64 LE

/// Scenario-selection seed base, distinct from both the forward-pass noise and
/// historical-selection domains.
const EXTERNAL_SELECTION_BASE_SEED: u64 = 0x6578_7465_726e_616c; // b"external" as u64 LE

/// Node-graph transition-draw seed base, distinct from the forward-pass noise,
/// historical-selection, and external-selection domains.
const TRANSITION_SELECTION_BASE_SEED: u64 = 0x6e74_6973_6e61_7274; // b"transitn" as u64 LE

/// Select an out-edge by inverse-CDF over `weights` (a node's successors,
/// ascending child-position order — the caller supplies the canonical order;
/// this function does not know about `NodeGraph`), using
/// `u = (h >> 11) * 2^-53` from
/// `derive_forward_seed(TRANSITION_SELECTION_BASE_SEED, iteration, path,
/// stage)`. `weights` need not be pre-normalized to sum to exactly 1.0 (the
/// walk here compensates its own running sum); a compensated sum that still
/// falls short of `u` due to floating-point rounding resolves to the last
/// index, never a panic or an out-of-range index.
///
/// A single out-edge must never reach this function — the caller short-
/// circuits to that edge without deriving a seed at all, so the within-node
/// noise stream is never perturbed by a branch that had no choice to make
/// (chain byte-parity depends on this).
///
/// # Panics (debug builds only)
///
/// Panics if `weights` is empty.
#[must_use]
pub fn select_transition_child<I>(iteration: u32, path: u32, stage: u32, weights: I) -> usize
where
    I: IntoIterator<Item = f64>,
{
    let h = derive_forward_seed(TRANSITION_SELECTION_BASE_SEED, iteration, path, stage);
    #[allow(clippy::cast_precision_loss)]
    let u = ((h >> 11) as f64) * (1.0_f64 / (1_u64 << 53) as f64);

    let mut cumsum = 0.0_f64;
    let mut compensation = 0.0_f64;
    let mut last_idx = 0_usize;
    let mut any = false;
    for (i, w) in weights.into_iter().enumerate() {
        any = true;
        last_idx = i;
        // Neumaier-compensated running sum (mirrors cobre-io's
        // `normalize_weights`), so a long successor list does not drift the
        // cumulative boundary away from the true sum.
        let t = cumsum + w;
        if cumsum.abs() >= w.abs() {
            compensation += (cumsum - t) + w;
        } else {
            compensation += (w - t) + cumsum;
        }
        cumsum = t;
        if u < cumsum + compensation {
            return i;
        }
    }
    debug_assert!(any, "select_transition_child: weights must be non-empty");
    last_idx
}

impl ClassSampler<'_> {
    /// Deterministic historical window index from `(iteration, scenario)`.
    #[allow(clippy::cast_possible_truncation)]
    fn select_historical_window(req: &ClassSampleRequest, n_windows: usize) -> usize {
        let hash = derive_forward_seed(
            HISTORICAL_SELECTION_BASE_SEED,
            req.iteration,
            req.scenario,
            0,
        );
        (hash as usize) % n_windows
    }

    /// Deterministic external-scenario index from `(iteration, scenario)`.
    #[allow(clippy::cast_possible_truncation)]
    fn select_external_scenario(req: &ClassSampleRequest, n_scenarios: usize) -> usize {
        let hash =
            derive_forward_seed(EXTERNAL_SELECTION_BASE_SEED, req.iteration, req.scenario, 0);
        (hash as usize) % n_scenarios
    }

    /// No-op for every variant — reserved for future per-class initial-state
    /// injection.
    ///
    /// `Historical` deliberately does NOT inject window-preceding lags: initial
    /// inflow lags come from the derived inflow seed (`DerivedInflowSeeds`) for
    /// every scenario regardless of the replayed window, so forward, backward, and
    /// lower-bound evaluators all consume the same `x_0` (a window-dependent lag
    /// would make the reported convergence gap meaningless). The window
    /// contributes only standardized noise residuals via [`ClassSampler::fill`].
    pub fn apply_initial_state(
        &self,
        _req: &ClassSampleRequest,
        _state: &mut [f64],
        _lag_offset: usize,
    ) {
    }

    /// Fill `output` with noise for the given `(iteration, scenario, stage)` triple.
    ///
    /// `InSample`/`Historical`/`External` produce pre-correlated noise; only
    /// `OutOfSample` leaves it uncorrelated for the composite `ForwardSampler` to
    /// correlate downstream.
    ///
    /// # Errors
    ///
    /// - [`StochasticError::InsufficientData`] — when `OutOfSample` and
    ///   `req.stage_idx` is out of bounds for the per-stage noise methods or
    ///   for `tables`, or `tables`'s resolved variant does not match the
    ///   stage's noise method.
    ///
    /// # Panics
    ///
    /// `InSample`: panics in debug builds if `output.len() != self.len`
    /// (programming error — caller is responsible for buffer sizing).
    ///
    /// `OutOfSample`: panics in debug builds if `tables` was not rebuilt for
    /// `req`'s `(iteration, total_scenarios)`, or if `req.stage_idx`'s table
    /// was not built for `req.noise_group_id`.
    pub fn fill(
        &self,
        req: &ClassSampleRequest,
        tables: &ClassNoiseTables,
        output: &mut [f64],
    ) -> Result<(), StochasticError> {
        match self {
            ClassSampler::InSample {
                tree,
                base_seed,
                offset,
                len,
            } => {
                // An empty class copies nothing but must still skip sample_forward,
                // whose opening-range assert would otherwise trip on an offset it
                // never reads.
                if *len == 0 {
                    return Ok(());
                }
                debug_assert_eq!(
                    output.len(),
                    *len,
                    "ClassSampler::InSample::fill: output.len() ({}) != self.len ({})",
                    output.len(),
                    len,
                );
                let (_idx, slice) = insample::sample_forward(
                    tree,
                    *base_seed,
                    req.iteration,
                    req.scenario,
                    req.stage,
                    req.stage_idx,
                    req.node_opening_offset,
                    req.node_opening_len,
                );
                output.copy_from_slice(&slice[*offset..*offset + *len]);
                Ok(())
            }

            ClassSampler::OutOfSample {
                forward_seed,
                dim,
                noise_methods,
            } => {
                let noise_method = noise_methods.get(req.stage_idx).copied().ok_or_else(|| {
                    StochasticError::InsufficientData {
                        context: format!(
                            "stage_idx {} out of bounds for {} noise methods",
                            req.stage_idx,
                            noise_methods.len(),
                        ),
                    }
                })?;
                debug_assert!(
                    tables.built_for() == (req.iteration, req.total_scenarios),
                    "ClassSampler::OutOfSample::fill: tables built for {:?} but request \
                     carries (iteration={}, total_scenarios={})",
                    tables.built_for(),
                    req.iteration,
                    req.total_scenarios,
                );
                let table = tables.table_at(req.stage_idx).ok_or_else(|| {
                    StochasticError::InsufficientData {
                        context: format!(
                            "stage_idx {} out of bounds for the noise tables",
                            req.stage_idx,
                        ),
                    }
                })?;
                debug_assert!(
                    tables.group_at(req.stage_idx) == Some(req.noise_group_id),
                    "ClassSampler::OutOfSample::fill: stage_idx {} table built for group {:?} \
                     but request carries noise_group_id {}",
                    req.stage_idx,
                    tables.group_at(req.stage_idx),
                    req.noise_group_id,
                );
                let spec = FreshNoiseSpec {
                    forward_seed: *forward_seed,
                    noise_method,
                    iteration: req.iteration,
                    scenario: req.scenario,
                    noise_group_id: req.noise_group_id,
                    dim: *dim,
                    total_scenarios: req.total_scenarios,
                };
                fill_uncorrelated(spec, table, output)
            }

            ClassSampler::Historical { library } => {
                let window_idx = Self::select_historical_window(req, library.n_windows());
                output.copy_from_slice(library.eta_slice(window_idx, req.stage_idx));
                Ok(())
            }

            ClassSampler::External { library } => {
                let scenario_idx = req
                    .pinned_scenario
                    .unwrap_or_else(|| Self::select_external_scenario(req, library.n_scenarios()));
                output.copy_from_slice(library.eta_slice(req.stage_idx, scenario_idx));
                Ok(())
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp
)]
mod tests {
    use cobre_core::temporal::NoiseMethod;

    use crate::{
        StochasticError, sample_forward,
        sampling::{ClassNoiseTables, ExternalScenarioLibrary, HistoricalScenarioLibrary},
        test_support::uniform_tree,
    };

    use super::{ClassSampleRequest, ClassSampler, select_transition_child};

    /// `ClassNoiseTables` with one `Saa` slot per `groups` entry, built for
    /// `(iteration, total_scenarios)` — enough for every `OutOfSample` case in
    /// this module: all use `NoiseMethod::Saa`, whose `fill_uncorrelated` arm
    /// ignores the table's contents, so only `table_at`'s `Some` resolution
    /// and the stamped `(iteration, total_scenarios, group)` matter here.
    fn saa_tables(iteration: u32, total_scenarios: u32, groups: &[u32]) -> ClassNoiseTables {
        let mut tables = ClassNoiseTables::default();
        let methods = vec![NoiseMethod::Saa; groups.len()];
        tables
            .refill(1, 1, iteration, total_scenarios, groups, &methods)
            .expect("Saa never exceeds the Sobol dimension cap");
        tables
    }

    // -----------------------------------------------------------------------
    // select_transition_child
    // -----------------------------------------------------------------------

    #[test]
    fn transition_child_deterministic() {
        let weights = [0.5_f64, 0.5];
        assert_eq!(
            select_transition_child(3, 7, 1, weights),
            select_transition_child(3, 7, 1, weights),
        );
    }

    #[test]
    fn transition_child_single_weight_always_index_zero() {
        // Never reached on the hot path (callers short-circuit a single
        // out-edge without deriving a seed), but the function itself must
        // still resolve correctly if called this way directly.
        for path in 0_u32..50 {
            assert_eq!(select_transition_child(0, path, 0, [1.0_f64]), 0);
        }
    }

    #[test]
    fn transition_child_index_always_in_bounds() {
        let weights = [0.2_f64, 0.3, 0.1, 0.4];
        for path in 0_u32..500 {
            let idx = select_transition_child(2, path, 5, weights);
            assert!(idx < weights.len(), "index {idx} out of bounds");
        }
    }

    #[test]
    fn transition_child_empirical_distribution_matches_weights() {
        // A 3-child split at (0.7, 0.2, 0.1): over many draws (varying `path`),
        // the realized frequency should land close to the declared weight —
        // a coarse statistical sanity check on the inverse-CDF, not a proof.
        let weights = [0.7_f64, 0.2, 0.1];
        let n = 20_000_u32;
        let mut counts = [0_u32; 3];
        for path in 0..n {
            counts[select_transition_child(11, path, 3, weights)] += 1;
        }
        #[allow(clippy::cast_precision_loss)]
        let freq: Vec<f64> = counts
            .iter()
            .map(|&c| f64::from(c) / f64::from(n))
            .collect();
        for (f, w) in freq.iter().zip(weights) {
            assert!(
                (f - w).abs() < 0.02,
                "empirical frequency {f} too far from declared weight {w}"
            );
        }
    }

    #[test]
    fn transition_child_ascending_order_low_weight_first_child_rarely_wins() {
        // (0.01, 0.99): the first child should win far less often than the
        // second — pins that cumulative weight walks in the GIVEN (ascending
        // child-position) order, not some other order.
        let weights = [0.01_f64, 0.99];
        let n = 5_000_u32;
        let mut child0_wins = 0_u32;
        for path in 0..n {
            if select_transition_child(4, path, 0, weights) == 0 {
                child0_wins += 1;
            }
        }
        assert!(
            child0_wins < n / 10,
            "expected child 0 (weight 0.01) to win rarely, won {child0_wins}/{n} times"
        );
    }

    #[test]
    fn transition_child_varies_with_iteration_path_and_stage() {
        let weights = [0.5_f64, 0.5];
        let base = select_transition_child(0, 0, 0, weights);
        let differs_by_any = (1..30_u32).any(|i| select_transition_child(i, 0, 0, weights) != base)
            || (1..30_u32).any(|p| select_transition_child(0, p, 0, weights) != base)
            || (1..30_u32).any(|s| select_transition_child(0, 0, s, weights) != base);
        assert!(
            differs_by_any,
            "expected the selected child to vary across at least one of iteration/path/stage"
        );
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    fn base_req() -> ClassSampleRequest {
        ClassSampleRequest {
            iteration: 0,
            scenario: 0,
            stage: 0,
            stage_idx: 0,
            total_scenarios: 10,
            noise_group_id: 0,
            node_opening_offset: 0,
            node_opening_len: 0,
            pinned_scenario: None,
        }
    }

    // -----------------------------------------------------------------------
    // InSample tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_in_sample_fill_copies_correct_segment() {
        let tree = uniform_tree(1, 3, 5);
        let view = tree.view();

        let sampler = ClassSampler::InSample {
            tree: view,
            base_seed: 42,
            offset: 2,
            len: 3,
        };

        let mut output = vec![0.0f64; 3];
        let req = ClassSampleRequest {
            node_opening_len: tree.view().n_openings(0),
            ..base_req()
        };

        sampler
            .fill(&req, &ClassNoiseTables::default(), &mut output)
            .unwrap();

        assert_eq!(output.len(), 3);
        for v in &output {
            assert!(v.is_finite(), "output value {v} is not finite");
        }

        let (_opening_idx, full_slice) = sample_forward(
            &tree.view(),
            42,
            req.iteration,
            req.scenario,
            req.stage,
            req.stage_idx,
            0,
            tree.view().n_openings(req.stage_idx),
        );
        assert_eq!(&output, &full_slice[2..5]);
    }

    #[test]
    fn in_sample_zero_entity_skips_out_of_range_opening() {
        // A node whose opening offset exceeds the stage's opening count would trip
        // sample_forward's range assert; a zero-entity class must return Ok before
        // reaching it, leaving output untouched.
        let tree = uniform_tree(1, 1, 5); // stage 0 carries exactly one opening
        let view = tree.view();
        let sampler = ClassSampler::InSample {
            tree: view,
            base_seed: 42,
            offset: 0,
            len: 0,
        };
        // offset 2 + len 1 exceeds the single opening: sample_forward would assert.
        let req = ClassSampleRequest {
            node_opening_offset: 2,
            node_opening_len: 1,
            ..base_req()
        };
        let mut output: Vec<f64> = Vec::new();
        sampler
            .fill(&req, &ClassNoiseTables::default(), &mut output)
            .unwrap();
        assert!(
            output.is_empty(),
            "zero-entity InSample fill must leave output untouched"
        );
    }

    #[test]
    fn test_in_sample_fill_deterministic() {
        let tree = uniform_tree(2, 5, 5);
        let view = tree.view();

        let sampler = ClassSampler::InSample {
            tree: view,
            base_seed: 99,
            offset: 1,
            len: 2,
        };

        let req = ClassSampleRequest {
            iteration: 3,
            scenario: 7,
            stage: 1,
            stage_idx: 1,
            total_scenarios: 5,
            noise_group_id: 0,
            node_opening_offset: 0,
            node_opening_len: tree.view().n_openings(1),
            pinned_scenario: None,
        };

        let mut out_a = vec![0.0f64; 2];
        let mut out_b = vec![0.0f64; 2];

        sampler
            .fill(&req, &ClassNoiseTables::default(), &mut out_a)
            .unwrap();
        sampler
            .fill(&req, &ClassNoiseTables::default(), &mut out_b)
            .unwrap();

        assert_eq!(out_a, out_b, "InSample::fill must be deterministic");
    }

    /// An incidental `select_transition_child` call for the same `(iteration,
    /// scenario, stage)` — its own domain-separated seed sharing no state with
    /// the noise draw's — must not change what `fill` produces. This is the
    /// invariant a single-out-edge short-circuit relies on to leave the
    /// within-node stream untouched (chain bit-equality): the short circuit is
    /// a call-site choice, not a defense against a stateful draw.
    #[test]
    fn transition_draw_call_does_not_perturb_subsequent_within_node_noise() {
        let tree = uniform_tree(2, 5, 5);
        let view = tree.view();

        let sampler = ClassSampler::InSample {
            tree: view,
            base_seed: 99,
            offset: 1,
            len: 2,
        };

        let req = ClassSampleRequest {
            iteration: 3,
            scenario: 7,
            stage: 1,
            stage_idx: 1,
            total_scenarios: 5,
            noise_group_id: 0,
            node_opening_offset: 0,
            node_opening_len: tree.view().n_openings(1),
            pinned_scenario: None,
        };

        let mut out_no_draw = vec![0.0f64; 2];
        let mut out_with_draw = vec![0.0f64; 2];

        sampler
            .fill(&req, &ClassNoiseTables::default(), &mut out_no_draw)
            .unwrap();

        let _ = select_transition_child(req.iteration, req.scenario, req.stage, [0.5_f64, 0.5]);

        sampler
            .fill(&req, &ClassNoiseTables::default(), &mut out_with_draw)
            .unwrap();

        assert_eq!(
            out_no_draw, out_with_draw,
            "an incidental transition draw for the same (iteration, scenario, stage) must not \
             perturb the within-node noise fill"
        );
    }

    // -----------------------------------------------------------------------
    // OutOfSample tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_out_of_sample_fill_deterministic() {
        let noise_methods: Box<[NoiseMethod]> =
            vec![NoiseMethod::Saa, NoiseMethod::Saa].into_boxed_slice();

        let sampler = ClassSampler::OutOfSample {
            forward_seed: 42,
            dim: 3,
            noise_methods,
        };

        let req = ClassSampleRequest {
            iteration: 1,
            scenario: 2,
            stage: 0,
            stage_idx: 0,
            total_scenarios: 10,
            noise_group_id: 0,
            node_opening_offset: 0,
            node_opening_len: 0,
            pinned_scenario: None,
        };

        let mut out_a = vec![0.0f64; 3];
        let mut out_b = vec![0.0f64; 3];

        sampler
            .fill(&req, &saa_tables(1, 10, &[0]), &mut out_a)
            .unwrap();
        sampler
            .fill(&req, &saa_tables(1, 10, &[0]), &mut out_b)
            .unwrap();

        assert_eq!(
            out_a, out_b,
            "OutOfSample::fill must produce bit-identical output for same inputs"
        );
    }

    #[test]
    fn test_out_of_sample_fill_stage_idx_out_of_bounds() {
        let noise_methods: Box<[NoiseMethod]> = vec![NoiseMethod::Saa].into_boxed_slice();
        let sampler = ClassSampler::OutOfSample {
            forward_seed: 1,
            dim: 2,
            noise_methods,
        };

        let req = ClassSampleRequest {
            stage_idx: 5, // out of bounds: only 1 method
            ..base_req()
        };

        let mut output = vec![0.0f64; 2];

        let result = sampler.fill(&req, &ClassNoiseTables::default(), &mut output);
        assert!(
            matches!(result, Err(StochasticError::InsufficientData { .. })),
            "expected InsufficientData error, got: {result:?}"
        );
    }

    // -----------------------------------------------------------------------
    // Historical tests
    // -----------------------------------------------------------------------

    #[allow(clippy::cast_precision_loss)]
    fn make_historical_library() -> HistoricalScenarioLibrary {
        // new(n_windows, n_stages, n_hydros, max_order, years)
        let mut lib = HistoricalScenarioLibrary::new(3, 4, 2, 1, vec![1990, 1995, 2000]);
        for w in 0..3 {
            for s in 0..4 {
                let base = (w * 100 + s * 10) as f64;
                lib.eta_slice_mut(w, s).copy_from_slice(&[base, base + 1.0]);
            }
        }
        lib
    }

    #[test]
    fn test_historical_fill_deterministic() {
        let lib = make_historical_library();
        let sampler = ClassSampler::Historical { library: &lib };

        let req = ClassSampleRequest {
            iteration: 5,
            scenario: 3,
            stage: 0,
            stage_idx: 1,
            total_scenarios: 10,
            noise_group_id: 0,
            node_opening_offset: 0,
            node_opening_len: 0,
            pinned_scenario: None,
        };

        let mut out_a = vec![0.0f64; 2];
        let mut out_b = vec![0.0f64; 2];

        sampler
            .fill(&req, &ClassNoiseTables::default(), &mut out_a)
            .unwrap();
        sampler
            .fill(&req, &ClassNoiseTables::default(), &mut out_b)
            .unwrap();

        assert_eq!(
            out_a, out_b,
            "Historical::fill must be deterministic for same (iteration, scenario)"
        );
    }

    #[test]
    fn test_historical_fill_different_scenarios_may_differ() {
        let lib = make_historical_library();
        let sampler = ClassSampler::Historical { library: &lib };

        let mut outputs: Vec<Vec<f64>> = (0_u32..20)
            .map(|scenario| {
                let req = ClassSampleRequest {
                    iteration: 0,
                    scenario,
                    stage: 0,
                    stage_idx: 0,
                    total_scenarios: 20,
                    noise_group_id: 0,
                    node_opening_offset: 0,
                    node_opening_len: 0,
                    pinned_scenario: None,
                };
                let mut out = vec![0.0f64; 2];
                sampler
                    .fill(&req, &ClassNoiseTables::default(), &mut out)
                    .unwrap();
                out
            })
            .collect();

        outputs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        outputs.dedup();
        assert!(
            outputs.len() > 1,
            "expected at least 2 distinct outputs across 20 scenarios, got only 1"
        );
    }

    #[test]
    fn test_historical_window_stable_across_stages() {
        let lib = make_historical_library();
        let sampler = ClassSampler::Historical { library: &lib };

        let req_stage0 = ClassSampleRequest {
            iteration: 2,
            scenario: 7,
            stage: 0,
            stage_idx: 0,
            total_scenarios: 10,
            noise_group_id: 0,
            node_opening_offset: 0,
            node_opening_len: 0,
            pinned_scenario: None,
        };
        let req_stage1 = ClassSampleRequest {
            stage_idx: 1,
            ..req_stage0
        };

        let mut out0 = vec![0.0f64; 2];
        let mut out1 = vec![0.0f64; 2];

        sampler
            .fill(&req_stage0, &ClassNoiseTables::default(), &mut out0)
            .unwrap();
        sampler
            .fill(&req_stage1, &ClassNoiseTables::default(), &mut out1)
            .unwrap();

        // Layout eta[w, s] = [w*100 + s*10, …]; same window, adjacent stage ⇒ +10.
        assert_eq!(
            out1[0] - out0[0],
            10.0,
            "Expected stage_idx difference of 10.0, got {}",
            out1[0] - out0[0]
        );
    }

    // -----------------------------------------------------------------------
    // External tests
    // -----------------------------------------------------------------------

    #[allow(clippy::cast_precision_loss)]
    fn make_external_library() -> ExternalScenarioLibrary {
        // new(n_stages, n_scenarios, n_entities, kind, per_stage_scenarios)
        let mut lib = ExternalScenarioLibrary::new(4, 50, 3, "inflow", vec![50usize; 4]);
        for s in 0..4_usize {
            for sc in 0..50_usize {
                let base = (s * 1000 + sc * 10) as f64;
                lib.eta_slice_mut(s, sc)
                    .copy_from_slice(&[base, base + 1.0, base + 2.0]);
            }
        }
        lib
    }

    #[test]
    fn test_external_fill_copies_eta_slice() {
        let lib = make_external_library();
        let sampler = ClassSampler::External { library: &lib };

        let req = ClassSampleRequest {
            iteration: 0,
            scenario: 0,
            stage: 0,
            stage_idx: 2,
            total_scenarios: 10,
            noise_group_id: 0,
            node_opening_offset: 0,
            node_opening_len: 0,
            pinned_scenario: None,
        };

        let mut output = vec![0.0f64; 3];

        sampler
            .fill(&req, &ClassNoiseTables::default(), &mut output)
            .unwrap();

        let scenario_idx = ClassSampler::select_external_scenario(&req, lib.n_scenarios());
        let expected = lib.eta_slice(req.stage_idx, scenario_idx);

        assert_eq!(
            &output, expected,
            "External::fill must match library.eta_slice(stage_idx, scenario_idx)"
        );
    }

    #[test]
    fn test_external_fill_scenario_selection_still_deterministic() {
        // fill()'s scenario selection is independent of apply_initial_state,
        // mirroring test_historical_fill_window_selection_still_deterministic.
        let lib = make_external_library();
        let sampler = ClassSampler::External { library: &lib };

        for scenario in 0..20_u32 {
            let req = ClassSampleRequest {
                iteration: 3,
                scenario,
                stage: 0,
                stage_idx: 0,
                total_scenarios: 20,
                noise_group_id: 0,
                node_opening_offset: 0,
                node_opening_len: 0,
                pinned_scenario: None,
            };
            let scenario_via_helper =
                ClassSampler::select_external_scenario(&req, lib.n_scenarios());

            let mut output = vec![0.0f64; 3];
            sampler
                .fill(&req, &ClassNoiseTables::default(), &mut output)
                .unwrap();
            let expected_eta = lib.eta_slice(req.stage_idx, scenario_via_helper);
            assert_eq!(
                &output, expected_eta,
                "fill() must use select_external_scenario for scenario={scenario}"
            );
        }
    }

    #[test]
    fn test_external_fill_deterministic() {
        let lib = make_external_library();
        let sampler = ClassSampler::External { library: &lib };

        let req = ClassSampleRequest {
            iteration: 3,
            scenario: 17,
            stage: 1,
            stage_idx: 1,
            total_scenarios: 10,
            noise_group_id: 0,
            node_opening_offset: 0,
            node_opening_len: 0,
            pinned_scenario: None,
        };

        let mut out_a = vec![0.0f64; 3];
        let mut out_b = vec![0.0f64; 3];

        sampler
            .fill(&req, &ClassNoiseTables::default(), &mut out_a)
            .unwrap();
        sampler
            .fill(&req, &ClassNoiseTables::default(), &mut out_b)
            .unwrap();

        assert_eq!(
            out_a, out_b,
            "External::fill must be deterministic for same (iteration, scenario)"
        );
    }

    #[test]
    fn test_external_fill_pinned_scenario_reads_that_column() {
        let lib = make_external_library();
        let sampler = ClassSampler::External { library: &lib };

        // A pinned column is read verbatim, independent of (iteration, scenario)
        // — and distinct from what the hash would have chosen for the same request.
        for pinned in [0_usize, 7, 23, 49] {
            let req = ClassSampleRequest {
                iteration: 3,
                scenario: 17,
                stage: 1,
                stage_idx: 1,
                total_scenarios: 10,
                noise_group_id: 0,
                node_opening_offset: 0,
                node_opening_len: 0,
                pinned_scenario: Some(pinned),
            };
            let mut output = vec![0.0f64; 3];
            sampler
                .fill(&req, &ClassNoiseTables::default(), &mut output)
                .unwrap();
            assert_eq!(
                &output,
                lib.eta_slice(req.stage_idx, pinned),
                "pinned_scenario=Some({pinned}) must read column {pinned}"
            );
        }
    }

    #[test]
    fn test_external_fill_pin_none_matches_hash_selection() {
        // With no pin the External arm reproduces the hash selection bit-for-bit
        // (the sampled path is byte-identical to the pre-pin behavior).
        let lib = make_external_library();
        let sampler = ClassSampler::External { library: &lib };

        for scenario in 0..20_u32 {
            let req = ClassSampleRequest {
                iteration: 5,
                scenario,
                stage: 0,
                stage_idx: 2,
                total_scenarios: 20,
                noise_group_id: 0,
                node_opening_offset: 0,
                node_opening_len: 0,
                pinned_scenario: None,
            };
            let hash_idx = ClassSampler::select_external_scenario(&req, lib.n_scenarios());
            let mut output = vec![0.0f64; 3];
            sampler
                .fill(&req, &ClassNoiseTables::default(), &mut output)
                .unwrap();
            assert_eq!(
                &output,
                lib.eta_slice(req.stage_idx, hash_idx),
                "pinned_scenario=None must hash-select exactly as before for scenario={scenario}"
            );
        }
    }

    #[test]
    fn test_external_scenario_stable_across_stages() {
        let lib = make_external_library();
        let sampler = ClassSampler::External { library: &lib };

        let req_stage0 = ClassSampleRequest {
            iteration: 1,
            scenario: 5,
            stage: 0,
            stage_idx: 0,
            total_scenarios: 10,
            noise_group_id: 0,
            node_opening_offset: 0,
            node_opening_len: 0,
            pinned_scenario: None,
        };
        let req_stage1 = ClassSampleRequest {
            stage_idx: 1,
            ..req_stage0
        };

        let mut out0 = vec![0.0f64; 3];
        let mut out1 = vec![0.0f64; 3];

        sampler
            .fill(&req_stage0, &ClassNoiseTables::default(), &mut out0)
            .unwrap();
        sampler
            .fill(&req_stage1, &ClassNoiseTables::default(), &mut out1)
            .unwrap();

        // Layout eta[s, sc] = [s*1000 + sc*10, …]; same scenario, adjacent stage ⇒ +1000.
        assert_eq!(
            out1[0] - out0[0],
            1000.0,
            "Expected stage difference of 1000.0 (same scenario, adjacent stage), got {}",
            out1[0] - out0[0]
        );
    }

    // -----------------------------------------------------------------------
    // apply_initial_state tests
    // -----------------------------------------------------------------------

    /// Construct a library with known eta values for `apply_initial_state` tests.
    ///
    /// 3 windows, 4 stages, 2 hydros, `max_order`=2.
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap
    )]
    fn make_historical_library_with_lags() -> HistoricalScenarioLibrary {
        let mut lib = HistoricalScenarioLibrary::new(3, 4, 2, 2, vec![1990, 1995, 2000]);
        for w in 0..3 {
            for s in 0..4 {
                let base = (w * 100 + s * 10) as f64;
                lib.eta_slice_mut(w, s).copy_from_slice(&[base, base + 1.0]);
            }
        }
        lib
    }

    #[test]
    fn test_historical_apply_initial_state_noop() {
        let lib = make_historical_library_with_lags();
        let sampler = ClassSampler::Historical { library: &lib };

        for scenario in 0..20_u32 {
            let req = ClassSampleRequest {
                iteration: 0,
                scenario,
                stage: 0,
                stage_idx: 0,
                total_scenarios: 20,
                noise_group_id: 0,
                node_opening_offset: 0,
                node_opening_len: 0,
                pinned_scenario: None,
            };

            let lag_offset = 5;
            let lag_len = lib.max_order() * lib.n_hydros();
            let total_len = u32::try_from(lag_offset + lag_len + 3).unwrap();
            let original: Vec<f64> = (0..total_len).map(|i| 1.0 + f64::from(i)).collect();
            let mut state = original.clone();

            sampler.apply_initial_state(&req, &mut state, lag_offset);

            assert_eq!(
                state, original,
                "Historical::apply_initial_state must be a no-op for scenario={scenario}"
            );
        }
    }

    #[test]
    fn test_historical_fill_window_selection_still_deterministic() {
        // fill()'s window selection is independent of apply_initial_state.
        let lib = make_historical_library_with_lags();
        let sampler = ClassSampler::Historical { library: &lib };

        for scenario in 0..20_u32 {
            let req = ClassSampleRequest {
                iteration: 3,
                scenario,
                stage: 0,
                stage_idx: 0,
                total_scenarios: 20,
                noise_group_id: 0,
                node_opening_offset: 0,
                node_opening_len: 0,
                pinned_scenario: None,
            };
            let window_via_helper = ClassSampler::select_historical_window(&req, lib.n_windows());

            let mut output = vec![0.0f64; lib.n_hydros()];
            sampler
                .fill(&req, &ClassNoiseTables::default(), &mut output)
                .unwrap();
            let expected_eta = lib.eta_slice(window_via_helper, req.stage_idx);
            assert_eq!(
                &output, expected_eta,
                "fill() must use select_historical_window for scenario={scenario}"
            );
        }
    }

    #[test]
    fn test_in_sample_apply_initial_state_noop() {
        let tree = uniform_tree(1, 3, 5);
        let sampler = ClassSampler::InSample {
            tree: tree.view(),
            base_seed: 42,
            offset: 0,
            len: 3,
        };

        let original = vec![1.0f64, 2.0, 3.0, 4.0, 5.0];
        let mut state = original.clone();
        let req = base_req();

        sampler.apply_initial_state(&req, &mut state, 0);

        assert_eq!(
            state, original,
            "InSample::apply_initial_state must be a no-op"
        );
    }

    #[test]
    fn test_out_of_sample_apply_initial_state_noop() {
        let noise_methods: Box<[NoiseMethod]> = vec![NoiseMethod::Saa].into_boxed_slice();
        let sampler = ClassSampler::OutOfSample {
            forward_seed: 7,
            dim: 3,
            noise_methods,
        };

        let original = vec![5.0f64, 6.0, 7.0];
        let mut state = original.clone();
        let req = base_req();

        sampler.apply_initial_state(&req, &mut state, 0);

        assert_eq!(
            state, original,
            "OutOfSample::apply_initial_state must be a no-op"
        );
    }

    #[test]
    fn test_external_apply_initial_state_noop() {
        let lib = make_external_library();
        let sampler = ClassSampler::External { library: &lib };

        let original = vec![8.0f64, 9.0, 10.0];
        let mut state = original.clone();
        let req = base_req();

        sampler.apply_initial_state(&req, &mut state, 0);

        assert_eq!(
            state, original,
            "External::apply_initial_state must be a no-op"
        );
    }

    // -----------------------------------------------------------------------
    // Debug impl test
    // -----------------------------------------------------------------------

    #[test]
    fn test_debug_all_variants() {
        let tree = uniform_tree(1, 2, 3);

        let in_sample_debug = format!(
            "{:?}",
            ClassSampler::InSample {
                tree: tree.view(),
                base_seed: 1,
                offset: 0,
                len: 2,
            }
        );
        assert!(
            !in_sample_debug.is_empty(),
            "Debug output must not be empty"
        );

        let out_of_sample_debug = format!(
            "{:?}",
            ClassSampler::OutOfSample {
                forward_seed: 2,
                dim: 3,
                noise_methods: vec![NoiseMethod::Saa].into_boxed_slice(),
            }
        );
        assert!(
            !out_of_sample_debug.is_empty(),
            "Debug output must not be empty"
        );

        let hist_lib = make_historical_library();
        let ext_lib = make_external_library();
        let hist_debug = format!("{:?}", ClassSampler::Historical { library: &hist_lib });
        let ext_debug = format!("{:?}", ClassSampler::External { library: &ext_lib });
        assert!(hist_debug.contains("Historical"));
        assert!(ext_debug.contains("External"));
    }
    // -----------------------------------------------------------------------
    // noise_group_id propagation tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_out_of_sample_same_group_produces_identical_noise() {
        let noise_methods: Box<[NoiseMethod]> =
            vec![NoiseMethod::Saa, NoiseMethod::Saa].into_boxed_slice();
        let sampler = ClassSampler::OutOfSample {
            forward_seed: 42,
            dim: 3,
            noise_methods,
        };

        let req_stage0 = ClassSampleRequest {
            iteration: 1,
            scenario: 2,
            stage: 0,
            stage_idx: 0,
            total_scenarios: 10,
            noise_group_id: 5,
            node_opening_offset: 0,
            node_opening_len: 0,
            pinned_scenario: None,
        };
        let req_stage1 = ClassSampleRequest {
            stage: 1,
            stage_idx: 1,
            ..req_stage0
        };

        let mut out0 = vec![0.0f64; 3];
        let mut out1 = vec![0.0f64; 3];
        let tables = saa_tables(1, 10, &[5, 5]);

        sampler.fill(&req_stage0, &tables, &mut out0).unwrap();
        sampler.fill(&req_stage1, &tables, &mut out1).unwrap();

        assert_eq!(
            out0, out1,
            "OutOfSample::fill with same noise_group_id must produce identical noise              regardless of stage"
        );
    }

    #[test]
    fn test_out_of_sample_different_group_produces_different_noise() {
        let noise_methods: Box<[NoiseMethod]> =
            vec![NoiseMethod::Saa, NoiseMethod::Saa].into_boxed_slice();
        let sampler = ClassSampler::OutOfSample {
            forward_seed: 42,
            dim: 3,
            noise_methods,
        };

        let req_group0 = ClassSampleRequest {
            iteration: 1,
            scenario: 2,
            stage: 0,
            stage_idx: 0,
            total_scenarios: 10,
            noise_group_id: 0,
            node_opening_offset: 0,
            node_opening_len: 0,
            pinned_scenario: None,
        };
        let req_group1 = ClassSampleRequest {
            noise_group_id: 1,
            ..req_group0
        };

        let mut out0 = vec![0.0f64; 3];
        let mut out1 = vec![0.0f64; 3];

        sampler
            .fill(&req_group0, &saa_tables(1, 10, &[0]), &mut out0)
            .unwrap();
        sampler
            .fill(&req_group1, &saa_tables(1, 10, &[1]), &mut out1)
            .unwrap();

        let any_differ = out0.iter().zip(&out1).any(|(a, b)| a != b);
        assert!(
            any_differ,
            "OutOfSample::fill with different noise_group_id must produce different noise"
        );
    }
}
