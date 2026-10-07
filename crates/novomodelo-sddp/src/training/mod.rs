//! SDDP training-phase passes and the iterative training loop.
//!
//! This directory clusters everything that runs during training — the
//! orchestrating loop, the per-pass execution kernels, and the per-pass scratch
//! state — as a sibling to [`crate::simulation`]:
//!
//! - `training` — `train`: wires the forward pass, forward sync, state exchange,
//!   backward pass, cut sync, lower-bound evaluation, and convergence check into
//!   a single iterative loop.
//! - `session` — `TrainingSession`: owns all scratch buffers for one `train`
//!   call and drives `run_iteration`.
//! - `forward` / `backward` — the forward and backward pass kernels.
//! - `forward_pass_state` / `backward_pass_state` — pre-allocated per-pass
//!   scratch state and the borrowed-inputs bundles their `run` boundaries take.
//! - `lower_bound` — `evaluate_lower_bound`.
//! - `state_exchange` — `ExchangeBuffers`: `allgatherv` of trial points between
//!   the forward and backward passes.
//! - `trajectory` — `TrajectoryRecord`: the forward→backward data unit.
//! - `visited_states` — archived trial-point states for dominated cut selection.
//! - `training_output` — converts the training summary plus event log into the
//!   structured output the `cobre-io` writers consume.

// Rationale: renaming this submodule out of its parent's `training` name would
// break the `crate::training::{train, TrainingResult, TrainingOutcome}`
// re-export paths call sites and intra-doc links rely on, for no behavioural gain.
#[allow(clippy::module_inception)]
pub mod training;

pub mod forward;
pub mod lower_bound;

pub(crate) mod backward;
pub(crate) mod backward_pass_state;
pub(crate) mod forward_pass_state;
pub(crate) mod rank_reconcile;
pub(crate) mod session;
pub(crate) mod stage_solve_prep;
pub(crate) mod state_exchange;
pub(crate) mod training_output;
pub(crate) mod trajectory;
pub(crate) mod visited_states;

pub use forward::SyncResult;
pub use state_exchange::ExchangeBuffers;
pub use training::{TrainingOutcome, TrainingResult, train};
pub use training_output::build_training_output;
pub use trajectory::TrajectoryRecord;

// `pub(crate)` keeps this MPI basis-cache helper (sole consumer `session/mod.rs`)
// off the curated public surface.
pub(crate) use training::broadcast_basis_cache;
