//! Training, fingerprinting and world-split helpers shared by the sampled
//! determinism gates of `mpi_wire.rs`.

use std::collections::BTreeSet;

use novomodelo_comm::Communicator;
use novomodelo_io::Config;
use novomodelo_io::config::{StoppingRuleConfig, TrainingSelection};
use novomodelo_sddp::{StudySetup, TrainingResult};
use novomodelo_solver::ActiveSolver;

use super::StubComm;
use super::selection_fingerprint::SelectionFingerprint;

/// Binding is recorded only for a reader, and a budget no pool reaches is a
/// reader that evicts nothing.
pub const BUDGET_THAT_EVICTS_NOTHING: u32 = 100_000;

pub fn sample(config: &mut Config, forward_passes: u32, iterations: u32) {
    config.training.selection = Some(TrainingSelection::sampled(forward_passes));
    config.training.stopping_rules = Some(vec![StoppingRuleConfig::IterationLimit {
        limit: iterations,
    }]);
    config.training.cut_selection.max_active_per_stage = Some(BUDGET_THAT_EVICTS_NOTHING);
}

/// Trains `setup` on a fresh solver and returns its fingerprint, after
/// asserting the run's own power checks. `budget` is the
/// `max_active_per_stage` the fixture was built with.
pub fn train(
    mut setup: StudySetup,
    comm: &impl Communicator,
    n_threads: usize,
    fixture: &str,
    budget: Option<u32>,
) -> SelectionFingerprint {
    let mut solver = ActiveSolver::new().expect("ActiveSolver::new must succeed");
    let outcome = setup
        .train(&mut solver, comm, n_threads, ActiveSolver::new, None, None)
        .expect("train must return Ok");
    assert!(
        outcome.error.is_none(),
        "{fixture}: training must not error: {:?}",
        outcome.error
    );
    assert_frozen_rows_exist(fixture, &setup, &outcome.result);
    if let Some(budget) = budget {
        assert_budget_evicted_nothing(fixture, &setup, budget);
        assert_lag_bearing_binding_cut(fixture, &setup);
    }
    SelectionFingerprint::capture(&setup, &outcome.result)
}

fn assert_budget_evicted_nothing(fixture: &str, setup: &StudySetup, budget: u32) {
    let budget = usize::try_from(budget).expect("a u32 budget fits usize");
    for (pool_id, pool) in setup.fcf.pools.iter().enumerate() {
        assert!(
            pool.populated() <= budget,
            "{fixture}: pool {pool_id} populates {} slots, above the budget of {budget}, so \
             the budget may have evicted a cut",
            pool.populated()
        );
    }
}

fn cut_generating_pools(setup: &StudySetup) -> BTreeSet<usize> {
    let graph = &setup.inputs.node_graph;
    graph
        .nodes
        .iter()
        .zip(graph.successors.iter())
        .filter(|(_, successors)| !successors.is_empty())
        .map(|(node, _)| node.pool_id)
        .collect()
}

fn assert_frozen_rows_exist(fixture: &str, setup: &StudySetup, result: &TrainingResult) {
    assert!(
        result.iterations >= 2,
        "{fixture}: trained {} iteration(s), so no cut reached a later iteration's templates",
        result.iterations
    );
    let frozen = cut_generating_pools(setup).into_iter().any(|pool_id| {
        let pool = &setup.fcf.pools[pool_id];
        (0..pool.populated()).any(|slot| {
            pool.is_active(slot) && pool.metadata(slot).iteration_generated < result.iterations
        })
    });
    assert!(
        frozen,
        "{fixture}: no cut-generating pool holds an active cut generated before the last of \
         {} iterations",
        result.iterations
    );
}

fn assert_lag_bearing_binding_cut(fixture: &str, setup: &StudySetup) {
    let context = setup.training_ctx();
    let mut lag_pools = Vec::new();
    let mut binding_cuts = 0;
    let mut lag_coefficient_cuts = 0;
    let mut lag_bearing_binding_cuts = 0;
    for (pool_id, layout) in context.cut_state_layouts.iter().enumerate() {
        if !layout.has_lag_slots() {
            continue;
        }
        lag_pools.push(pool_id);
        let pool = &setup.fcf.pools[pool_id];
        let lags = layout.lag_point().full();
        for slot in 0..pool.populated() {
            let binding = pool.metadata(slot).active_count > 0;
            let lag_bearing = pool.coefficient_row(slot)[lags.clone()]
                .iter()
                .any(|&coefficient| coefficient != 0.0);
            binding_cuts += usize::from(binding);
            lag_coefficient_cuts += usize::from(lag_bearing);
            lag_bearing_binding_cuts += usize::from(binding && lag_bearing);
        }
    }
    assert!(
        lag_bearing_binding_cuts > 0,
        "{fixture}: no binding cut carries a nonzero lag coefficient: over the lag-bearing \
         pools {lag_pools:?}, {binding_cuts} cut(s) bound and {lag_coefficient_cuts} carry a \
         nonzero lag coefficient"
    );
}

/// Trains one fresh `build()` per shape and returns the first shape's fingerprint.
pub fn assert_thread_shapes_agree(
    fixture: &str,
    budget: Option<u32>,
    build: impl Fn() -> StudySetup,
    shapes: &[usize],
) -> SelectionFingerprint {
    let reference = train(build(), &StubComm, shapes[0], fixture, budget);
    for (index, &n_threads) in shapes.iter().enumerate().skip(1) {
        let repeat = if shapes[..index].contains(&n_threads) {
            " repeat"
        } else {
            ""
        };
        train(build(), &StubComm, n_threads, fixture, budget).assert_matches(
            &reference,
            &format!("{fixture} threads={n_threads}{repeat}"),
        );
    }
    reference
}

/// Every rank of the world must hold a trial and, on two or more ranks, a
/// strict share of them, so a rank's forward window differs from the single
/// rank's.
pub fn assert_world_splits_the_trials(fixture: &str, world: &impl Communicator, trials: usize) {
    assert!(
        world.size() <= trials,
        "a world of {} ranks leaves a rank without a trial",
        world.size()
    );
    if world.size() >= 2 {
        let share = trials.div_ceil(world.size());
        assert!(
            share < trials,
            "{fixture}: a rank trains {share} of the {trials} trials on {} ranks, so its \
             forward window is the single rank's",
            world.size()
        );
    }
}

pub fn assert_matches_single_rank(
    world: &impl Communicator,
    on_world: &SelectionFingerprint,
    reference: &SelectionFingerprint,
) {
    on_world.assert_matches(
        reference,
        &format!(
            "rank {} of {} must be bit-identical to the single-rank reference",
            world.rank(),
            world.size()
        ),
    );
}
