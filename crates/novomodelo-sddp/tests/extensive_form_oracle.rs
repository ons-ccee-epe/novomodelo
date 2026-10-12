//! Extensive-form LP oracle: closes node-native `enumerated` training's
//! `final_lb`/`final_ub` to the graph's true first-stage value on three
//! validation fixtures — (a) 3-stage K-fan (a root, K nodes, one leaf each: the
//! DECOMP shape), (b) 3-stage binary tree branching at two stages, (d) chain.
//!
//! A fourth fixture, a fan-then-recombine DAG (a node reached from two or more
//! parent nodes), is out of scope here: `setup::reject_recombining_node_enumeration`
//! hard-rejects any node graph with in-degree `>= 2` under `enumerated` selection
//! (`crates/novomodelo-sddp/src/setup/mod.rs`) — per-prefix state reconstruction for a
//! multi-parent node is a reserved seam, not built, and this binary's own
//! `extensive_form_optimum` adoption (below) would need the same prefix-aware
//! rewrite before it could expand one either. Tracked separately; not attempted.
//!
//! # The oracle
//!
//! [`extensive_form_optimum`] (`novomodelo_sddp::test_support`, adopted unchanged from
//! `branching_value_oracle.rs`'s harness — not re-derived here) expands one LP
//! column block per graph node. On these `|Ω| = 1`-per-node trees every node has
//! exactly one predecessor, so one column block per node IS one column block per
//! reachable `(node, realization)` prefix; [`assert_reachable_prefixes_match`]
//! checks that precondition before any bound comparison runs. Non-anticipativity is
//! structural: a parent's outgoing-state column is tied to its child's
//! incoming-state column by one equality row per state dimension, so no
//! non-anticipativity constraint row exists and none can be mis-signed.
//!
//! # Fixture (b): the headline
//!
//! [`branching_tree_setup_enumerated`] trains node-native `enumerated` over a
//! 3-stage binary tree branching at BOTH interior stages (root -> 2 -> 4 leaves)
//! under non-uniform weights — the shape a shape-based enumerated-execution
//! admission clause would have rejected outright. Its `final_lb`/`final_ub`
//! closing to the extensive-form optimum is the empirical discharge of that
//! claim, and it runs in the same (un-`#[ignore]`d) mode as fixtures (a)/(d).
//!
//! # Selection on
//!
//! The three fixtures run with productivity `0.0`, so each of their pools
//! holds one distinct cut and cut selection has nothing to choose between. The
//! `with_level1` legs train the same two graphs on the kinked system of
//! [`branching_tree_setup_enumerated_with_selection`] and
//! [`oracle_chain_setup_with_selection`], whose pools hold cuts of several
//! gradients, and assert that with selection on `final_lb` stays at or below
//! the optimum, a cut is deactivated, both bounds still close, and of the
//! bit-identical cuts a converged run keeps generating only the lowest slot
//! stays active.
//!
//! # LP tolerance
//!
//! [`REL_TOL`]/[`ABS_TOL`] mirror `branching_value_oracle.rs`: both LP backends'
//! `primal_feasibility_tolerance`/`dual_feasibility_tolerance` default to `1e-9`
//! (`novomodelo-solver`'s `clp`/`highs` backend configs); scaled by the objective
//! magnitude, this bounds the floating-point gap between the extensive-form LP and
//! the node-native engine's own accumulated solves — two independent
//! formulations, never bit equality.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::float_cmp,
    clippy::panic,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]

mod common;

use std::collections::BTreeSet;
use std::sync::mpsc::{self, Sender};

use novomodelo_core::TrainingEvent;
use novomodelo_io::config::SelectionMethod;
use novomodelo_sddp::cut::CutPool;
use novomodelo_sddp::setup::{
    NodeGraph, NodeId, NodeOpenings, NodePos, NodeRuntime, NodeSuccessor, OpeningSource, StageIdx,
};
use novomodelo_sddp::test_support::{
    branching_tree_setup_enumerated, branching_tree_setup_enumerated_with_selection,
    extensive_form_optimum, k_fan_setup_enumerated, node_prefix_counts, oracle_chain_setup,
    oracle_chain_setup_with_selection, single_path_enumerated_setup,
};
use novomodelo_sddp::{StudySetup, TrainingResult};
use novomodelo_solver::ActiveSolver;

use common::StubComm;
use common::selection_fingerprint::SelectionPower;

/// Relative + absolute LP tolerance for oracle-vs-engine value comparison. The
/// backend primal/dual feasibility tolerance is `≈ 1e-9`; scaled by the objective
/// magnitude this bounds the gap between the two formulations. Never bit-equality —
/// this is a value oracle across two different LP encodings.
const REL_TOL: f64 = 1e-6;
const ABS_TOL: f64 = 1e-4;

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= ABS_TOL + REL_TOL * a.abs().max(b.abs())
}

/// Train `setup` single-rank/single-thread to its iteration limit.
fn train(setup: &mut StudySetup, event_sender: Option<Sender<TrainingEvent>>) -> TrainingResult {
    let comm = StubComm;
    let mut solver = ActiveSolver::new().expect("ActiveSolver::new must succeed");
    let outcome = setup
        .train(&mut solver, &comm, 1, ActiveSolver::new, event_sender, None)
        .expect("training must return Ok");
    assert!(
        outcome.error.is_none(),
        "training must not error: {:?}",
        outcome.error
    );
    outcome.result
}

/// [`train`], returning `(final_lb, final_ub)`.
fn train_bounds(setup: &mut StudySetup) -> (f64, f64) {
    let result = train(setup, None);
    (result.final_lb, result.final_ub)
}

/// Assert `graph`'s reachable `(node, realization)` prefix set matches
/// [`extensive_form_optimum`]'s one-column-block-per-node expansion: every node
/// must be reached by EXACTLY one root-to-node prefix
/// ([`node_prefix_counts`] `== 1` everywhere), which is what makes "one LP column
/// block per node" the correct "one LP column block per reachable prefix"
/// expansion. Panics with a descriptive message on any mismatch, so a fixture
/// whose graph carries a recombination join (or any other multi-prefix node)
/// fails loudly rather than silently comparing the oracle's expansion against a
/// different problem.
///
/// # Panics
///
/// Panics if the prefix-count vector's length disagrees with the graph's node
/// count, if any node is reached by other than exactly one prefix, or if the
/// prefix-count total disagrees with the node count.
fn assert_reachable_prefixes_match(graph: &NodeGraph) {
    let n = graph.nodes.len();
    let prefix_counts = node_prefix_counts(graph)
        .expect("assert_reachable_prefixes_match: node prefix counts must not overflow");
    assert_eq!(
        prefix_counts.len(),
        n,
        "reachable-prefix count entries ({}) must equal the graph's node count ({n})",
        prefix_counts.len()
    );
    for (pos, &count) in prefix_counts.iter().enumerate() {
        assert!(
            count == 1,
            "node at canonical position {pos} is reached by {count} distinct \
             root-to-node prefixes, not exactly one: extensive_form_optimum \
             allocates a single LP column block per node, which expands the \
             correct problem only when every node has exactly one predecessor \
             (this graph carries a recombination join)"
        );
    }
    let prefix_total: u64 = prefix_counts.iter().sum();
    assert_eq!(
        prefix_total as usize, n,
        "reachable-prefix total ({prefix_total}) must equal the graph's node count \
         ({n}) — extensive_form_optimum's node-indexed column allocation is its \
         prefix allocation only under this equality"
    );
}

/// Cross-check the reachable prefixes, solve the extensive-form LP, train
/// `enumerated` to convergence, and assert both `final_lb` and `final_ub` close to
/// the LP optimum within [`REL_TOL`]/[`ABS_TOL`].
fn assert_closes_to_extensive_form(mut setup: StudySetup, fixture: &str) {
    assert_reachable_prefixes_match(&setup.inputs.node_graph);
    let optimum = extensive_form_optimum(&setup);
    let (lb, ub) = train_bounds(&mut setup);
    assert!(
        close(lb, optimum),
        "{fixture} final_lb {lb} must equal extensive-form optimum {optimum} (gap {})",
        lb - optimum
    );
    assert!(
        close(ub, optimum),
        "{fixture} final_ub {ub} must equal extensive-form optimum {optimum} (gap {})",
        ub - optimum
    );
}

// ── Three-fixture convergence gate ──────────────────────────────────────

#[test]
fn k_fan_matches_extensive_form() {
    let fixture = k_fan_setup_enumerated(3, 30);
    assert_closes_to_extensive_form(fixture.setup, "K-fan");
}

/// The headline: a 3-stage binary tree branching at TWO stages — the shape a
/// shape-based enumerated-execution admission clause would have rejected. Not
/// `#[ignore]`d; runs in the same mode as the other fixtures.
#[test]
fn branching_tree_matches_extensive_form() {
    let setup = branching_tree_setup_enumerated(30);
    assert_closes_to_extensive_form(setup, "3-stage binary tree");
}

#[test]
fn chain_matches_extensive_form() {
    let setup = oracle_chain_setup(30);
    assert_closes_to_extensive_form(setup, "chain");
}

// ── Selection on: the bound stays below the optimum and still closes ────

const LEVEL1_MAX_ITERATIONS: u32 = 30;

fn level1_every_iteration() -> SelectionMethod {
    SelectionMethod::Level1 {
        tie_tolerance: 1e-10,
        check_frequency: 1,
    }
}

/// Pools of the nodes whose cuts selection scores: stage 1 or later, with a
/// successor.
fn interior_cut_pools(graph: &NodeGraph) -> Vec<usize> {
    graph
        .nodes
        .iter()
        .zip(graph.successors.iter())
        .filter(|(node, successors)| node.stage >= StageIdx(1) && !successors.is_empty())
        .map(|(node, _)| node.pool_id)
        .collect()
}

/// Populated cuts of pool `pool_id` that differ in some coefficient bit.
fn distinct_cut_gradients(setup: &StudySetup, pool_id: usize) -> usize {
    let pool = &setup.fcf.pools[pool_id];
    (0..pool.populated())
        .map(|slot| {
            pool.coefficient_row(slot)
                .iter()
                .map(|coefficient| coefficient.to_bits())
                .collect::<Vec<u64>>()
        })
        .collect::<BTreeSet<_>>()
        .len()
}

/// The intercept and coefficient bits of each populated cut, in slot order.
fn cut_bits(pool: &CutPool) -> Vec<Vec<u64>> {
    (0..pool.populated())
        .map(|slot| {
            std::iter::once(pool.intercept(slot))
                .chain(pool.coefficient_row(slot).iter().copied())
                .map(f64::to_bits)
                .collect()
        })
        .collect()
}

/// derived: bit-identical cuts have equal values at every state, so among them
/// only the lowest slot can own one; and selection ran last at the final
/// iteration, so the owner of every stored state is active.
fn assert_identical_cuts_keep_their_lowest_slot(
    setup: &StudySetup,
    interior_pools: &[usize],
    fixture: &str,
) {
    for &pool_id in interior_pools {
        let pool = &setup.fcf.pools[pool_id];
        let cuts = cut_bits(pool);
        let lowest_identical = |slot: usize| {
            cuts.iter()
                .position(|cut| *cut == cuts[slot])
                .expect("a cut is identical to itself")
        };
        assert!(
            (0..cuts.len()).any(|slot| lowest_identical(slot) < slot),
            "{fixture} pool {pool_id} must hold a pair of bit-identical cuts"
        );
        assert!(
            pool.stored_trial_states() > 0,
            "{fixture} pool {pool_id} must store a trial state"
        );
        for slot in 0..cuts.len() {
            assert!(
                !pool.is_active(slot) || lowest_identical(slot) == slot,
                "{fixture} pool {pool_id}: slot {slot} is active and identical to the lower \
                 slot {}",
                lowest_identical(slot)
            );
            if pool.trial_state(slot).is_some() {
                let owner = pool.state_owner(slot).unwrap_or_else(|| {
                    panic!("{fixture} pool {pool_id}: the state at slot {slot} is unowned")
                });
                assert!(
                    pool.is_active(owner),
                    "{fixture} pool {pool_id}: slot {owner} owns the state at slot {slot} and \
                     is inactive"
                );
            }
        }
    }
}

/// Train `setup` with selection on and assert, in order: an interior node
/// exists for selection to act on, an interior pool holds cuts of different
/// gradients (the fixture's value function has a visited breakpoint),
/// `final_lb` does not exceed `optimum`, selection deactivated an interior cut,
/// each interior pool keeps the lowest slot of its bit-identical cuts and the
/// owner of every stored state, and both bounds close to `optimum` within
/// [`REL_TOL`]/[`ABS_TOL`].
fn assert_selection_stays_below_and_closes(mut setup: StudySetup, optimum: f64, fixture: &str) {
    let interior_pools = interior_cut_pools(&setup.inputs.node_graph);
    assert!(
        !interior_pools.is_empty(),
        "{fixture} must have an interior cut-generating node for selection to act on"
    );

    let (tx, rx) = mpsc::channel();
    let result = train(&mut setup, Some(tx));
    let events: Vec<TrainingEvent> = rx.try_iter().collect();
    let (lb, ub) = (result.final_lb, result.final_ub);

    let distinct_gradients: Vec<usize> = interior_pools
        .iter()
        .map(|&pool_id| distinct_cut_gradients(&setup, pool_id))
        .collect();
    assert!(
        distinct_gradients.iter().any(|&count| count >= 2),
        "{fixture} must leave an interior pool holding cuts of at least two gradients, \
         found {distinct_gradients:?} per interior pool"
    );

    assert!(
        lb <= optimum + ABS_TOL,
        "{fixture} final_lb {lb} must not exceed extensive-form optimum {optimum} (excess {})",
        lb - optimum
    );

    // `SelectionPower` reads every record after the first as an interior node.
    for event in &events {
        if let TrainingEvent::PolicySelectionComplete { per_stage, .. } = event {
            assert_eq!(
                per_stage.len(),
                1 + interior_pools.len(),
                "{fixture} selection must record the root and each interior node"
            );
        }
    }
    let power = SelectionPower::from_events(&events);
    assert!(
        power.deactivations > 0,
        "{fixture} selection must deactivate an interior cut within \
         {LEVEL1_MAX_ITERATIONS} iterations: {power:?}"
    );
    assert_identical_cuts_keep_their_lowest_slot(&setup, &interior_pools, fixture);

    assert!(
        close(lb, optimum),
        "{fixture} final_lb {lb} must equal extensive-form optimum {optimum} (gap {})",
        lb - optimum
    );
    assert!(
        close(ub, optimum),
        "{fixture} final_ub {ub} must equal extensive-form optimum {optimum} (gap {})",
        ub - optimum
    );
}

#[test]
fn branching_tree_with_level1_stays_below_and_closes_to_the_extensive_form() {
    let setup = branching_tree_setup_enumerated_with_selection(
        LEVEL1_MAX_ITERATIONS,
        level1_every_iteration(),
    );
    assert_reachable_prefixes_match(&setup.inputs.node_graph);
    let optimum = extensive_form_optimum(&setup);
    assert_selection_stays_below_and_closes(setup, optimum, "3-stage binary tree");
}

/// derived: the kinked chain's optimum. Each stage turbines against an 80 MW
/// load with a 100 m³/s limit and a 60 m³/s inflow, at 0.25, 0.5 and 0.75 MW
/// per m³/s, so water is worth more the later it is used: stage 0 leaves the
/// 200 hm³ storage limit (from 100 hm³), stage 1 leaves the volume stage 2 can
/// still turbine (100 − 60 m³/s over the stage), and stage 2 turbines at its
/// limit. Deficit costs 400 $/`MWh` up to 20 MW and 500 $/`MWh` beyond.
fn kinked_chain_optimum() -> f64 {
    const STAGE_HOURS: f64 = 744.0;
    let hm3_per_m3s = STAGE_HOURS * 3600.0 / 1e6;
    let turbined_m3s = [
        60.0 + (100.0 - 200.0) / hm3_per_m3s,
        60.0 + 200.0 / hm3_per_m3s - (100.0 - 60.0),
        100.0,
    ];
    let productivity = [0.25, 0.5, 0.75];
    turbined_m3s
        .iter()
        .zip(productivity)
        .map(|(turbined, productivity)| {
            let deficit_mw: f64 = 80.0 - productivity * turbined;
            STAGE_HOURS * (400.0 * deficit_mw.min(20.0) + 500.0 * (deficit_mw - 20.0).max(0.0))
        })
        .sum()
}

#[test]
fn chain_with_level1_stays_below_and_closes_to_the_extensive_form() {
    let setup = oracle_chain_setup_with_selection(LEVEL1_MAX_ITERATIONS, level1_every_iteration());
    assert_reachable_prefixes_match(&setup.inputs.node_graph);
    let optimum = extensive_form_optimum(&setup);
    let closed_form = kinked_chain_optimum();
    assert!(
        close(optimum, closed_form),
        "chain extensive-form optimum {optimum} must equal the closed form {closed_form}"
    );
    assert_selection_stays_below_and_closes(setup, optimum, "chain");
}

// ── Reachable-prefix cross-check failure path ───────────────────────────

/// A 4-node fan-then-recombine `NodeGraph`, hand-built (bypassing `StudySetup`,
/// which `enumerated`'s own admission guard would reject before construction
/// completes — see this file's module doc): root (position 0) fans into two
/// nodes (1, 2), both of which point at a shared node (3) — a recombination
/// join, `node_prefix_counts()[3] == 2`.
fn recombination_join_graph() -> NodeGraph {
    let openings = NodeOpenings {
        source: OpeningSource::Generated,
        offset: 0,
        len: 1,
        q: 1.0,
    };
    let node = |stage: usize, pool_id: usize| NodeRuntime {
        stage: StageIdx(stage),
        pool_id,
        openings,
    };
    let succ = |child: usize, probability: f64| NodeSuccessor {
        child: NodePos(child),
        probability,
    };
    NodeGraph {
        node_ids: (0..4_i32).map(NodeId).collect(),
        nodes: vec![node(0, 0), node(1, 1), node(1, 2), node(2, 3)].into(),
        successors: vec![
            vec![succ(1, 0.5), succ(2, 0.5)],
            vec![succ(3, 1.0)],
            vec![succ(3, 1.0)],
            vec![],
        ]
        .into(),
        n_pools: 4,
        pool_stage: vec![StageIdx(0), StageIdx(1), StageIdx(1), StageIdx(2)],
    }
}

#[test]
#[should_panic(expected = "distinct root-to-node prefixes")]
fn reachable_prefix_cross_check_panics_on_recombination_join() {
    assert_reachable_prefixes_match(&recombination_join_graph());
}

// ── Unit test: hand-checkable tiny graph ─────────────────────────────────────

/// The oracle on a hand-checkable 2-stage chain
/// ([`single_path_enumerated_setup`]'s system, built via
/// `PrepareHydroModelsResult::default_from_system`, which resolves every hydro's
/// `ConstantProductivity` at the documented `0.0` placeholder — no
/// `hydro_production_models.json` backs this fixture, so the plant can never
/// generate): with generation always `0`, the `80` MW load is met entirely by
/// deficit at `500` `$/MWh` on both `744`-hour stages, and turbining any water
/// that would otherwise spill is free (`turbined_cost 0.0`) and always within
/// the `100` m³/s turbine cap, so no spillage cost accrues either — the
/// closed-form optimum is exactly `500.0 * 80.0 * 744.0 * 2.0 = 59_520_000.0`.
#[test]
fn two_stage_chain_extensive_form_matches_deficit_closed_form() {
    let setup = single_path_enumerated_setup(30);
    assert_reachable_prefixes_match(&setup.inputs.node_graph);
    let optimum = extensive_form_optimum(&setup);
    let closed_form = 500.0 * 80.0 * 744.0 * 2.0;
    assert!(
        close(optimum, closed_form),
        "hand-checkable 2-stage chain extensive-form optimum {optimum} must equal \
         the closed-form deficit cost {closed_form}"
    );
}
