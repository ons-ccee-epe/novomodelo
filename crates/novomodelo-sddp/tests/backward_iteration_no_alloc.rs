//! Pins one warm backward phase to zero Rust heap allocation on every thread, under each
//! backward scheduler, with the bracket slab and the noise-term tables on. Allocations inside the
//! linked C++ solver are not counted.
//!
//! The counter is global, so this binary holds exactly one test.

#![cfg(feature = "test-support")]
#![allow(clippy::expect_used, clippy::panic)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use novomodelo_io::config::{BackwardScheduler, StoppingRuleConfig, TrainingSelection};
use novomodelo_sddp::test_support::with_backward_phase_probe;

mod common;
use common::in_code_studies::{FphaLagOptions, fpha_lag_study_with};
use common::stored_cut_oracles::apply_oracle_config;
use common::{StubComm, build_setup_in_code_with_models};

const THREADS: usize = 2;
const FORWARD_PASSES: u32 = 4;
const OPENINGS: usize = 3;
const MAX_WARM_UP_ITERATIONS: u32 = 5;

struct CountingAllocator;

static ARMED: AtomicBool = AtomicBool::new(false);
static ALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);

fn count_alloc() {
    if ARMED.load(Ordering::Relaxed) {
        ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
    }
}

// SAFETY: every method is a verbatim forward to `System`, which upholds
// `GlobalAlloc`'s contract on its own; this wrapper only adds a counter
// increment around the call.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        count_alloc();
        // SAFETY: `layout` is forwarded unchanged to `System::alloc`.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr`/`layout` are forwarded unchanged to `System::dealloc`.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        count_alloc();
        // SAFETY: `ptr`/`layout`/`new_size` are forwarded unchanged to
        // `System::realloc`.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

#[test]
fn a_warm_backward_iteration_allocates_nothing_on_any_thread() {
    for backward_scheduler in [
        BackwardScheduler::ByScenario {},
        BackwardScheduler::ByNode { block_size: None },
    ] {
        let label = format!("{backward_scheduler:?}");
        let (system, mut config, hydro_models) = fpha_lag_study_with(FphaLagOptions {
            openings: OPENINGS,
            ..FphaLagOptions::default()
        });
        apply_oracle_config(&mut config);
        config.training.selection = Some(TrainingSelection::Sampled {
            forward_passes: FORWARD_PASSES,
        });
        config.training.stopping_rules = Some(vec![StoppingRuleConfig::IterationLimit {
            limit: MAX_WARM_UP_ITERATIONS + 1,
        }]);
        let mut setup = build_setup_in_code_with_models(system, &config, hydro_models);

        let (warm_up_iterations, slab_misses, frozen_rows, observed) = with_backward_phase_probe(
            &mut setup,
            &StubComm,
            THREADS,
            backward_scheduler,
            |mut probe| {
                let mut warm_up_iterations = 0;
                while warm_up_iterations < u64::from(MAX_WARM_UP_ITERATIONS)
                    && probe.frozen_rows_past_root() == 0
                {
                    warm_up_iterations += 1;
                    probe
                        .run_iteration(warm_up_iterations)
                        .expect("a warm-up iteration must succeed");
                }
                let measured = warm_up_iterations + 1;
                probe
                    .run_forward_phase(measured)
                    .expect("the measured forward phase must succeed");
                ALLOC_COUNT.store(0, Ordering::Relaxed);
                ARMED.store(true, Ordering::Relaxed);
                let backward = probe.run_backward_phase(measured);
                ARMED.store(false, Ordering::Relaxed);
                backward.expect("the measured backward phase must succeed");
                (
                    warm_up_iterations,
                    probe.slab_misses(),
                    probe.frozen_rows_past_root(),
                    ALLOC_COUNT.load(Ordering::Relaxed),
                )
            },
        )
        .expect("the probe session must build");

        assert!(
            frozen_rows > 0,
            "{label}: no cut row is frozen at a pinned successor stage after {warm_up_iterations} warm-up iterations"
        );
        assert_eq!(slab_misses, 0, "{label}: slab misses");
        assert_eq!(
            observed, 0,
            "{label}: the warm backward phase allocated {observed} times"
        );
    }
}
