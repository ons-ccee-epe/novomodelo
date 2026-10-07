//! Deterministic static work partition shared by the training passes and
//! simulation.
//!
//! A single owner keeps forward, backward, and simulation splits bit-identical.

/// Compute the scenario range `[start, end)` for worker `worker_id` when
/// distributing `n_scenarios` scenarios across `n_workers` workers.
///
/// Static partition: scenario-to-worker assignment is identical regardless of
/// thread scheduling order. Extra scenarios go to lower-index workers.
///
/// Returns `(start, end)` where `start == end` when `worker_id` receives zero
/// scenarios (only occurs when `n_workers > n_scenarios`).
#[inline]
pub(crate) fn partition(n_scenarios: usize, n_workers: usize, worker_id: usize) -> (usize, usize) {
    if n_workers == 0 {
        return (0, 0);
    }
    let base = n_scenarios / n_workers;
    let remainder = n_scenarios % n_workers;
    let start = base * worker_id + worker_id.min(remainder);
    let end = start + base + usize::from(worker_id < remainder);
    (start, end)
}
