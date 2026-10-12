//! Bit-level fingerprint of a trained cut pool and the power counters of a
//! selection-on gate.
//!
//! Every float is stored as its bit pattern so `-0.0` and NaN payloads compare
//! by bits. Wall-clock fields and solver statistics are deliberately absent.

use novomodelo_core::TrainingEvent;
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
    /// Reported bounds.
    pub bounds: Bounds,
}

impl SelectionFingerprint {
    /// Reads the fingerprint off a trained `setup` and its `result`.
    pub fn capture(setup: &StudySetup, result: &TrainingResult) -> Self {
        let mut cuts = Vec::new();
        let mut provenance = Vec::new();
        let mut binding = Vec::new();
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
        }
        Self {
            cuts,
            provenance,
            binding,
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
    /// Sums the counters over a training run's events. The first record of
    /// each `PolicySelectionComplete` is the root and is skipped.
    pub fn from_events(events: &[TrainingEvent]) -> Self {
        let mut power = Self {
            deactivations: 0,
            reactivations: 0,
            budget_evictions: 0,
        };
        for event in events {
            match event {
                TrainingEvent::PolicySelectionComplete { per_stage, .. } => {
                    for record in per_stage.iter().skip(1) {
                        power.deactivations += u64::from(record.rows_deactivated);
                        power.reactivations += u64::from(record.rows_reactivated);
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
