//! Observes the end-of-stage states of a forward pass over a [`StudySetup`]: a
//! replay of the first iteration, or the last iteration's from the cut pools.

use novomodelo_sddp::{
    StudySetup, TrajectoryRecord,
    forward::{ForwardPassBatch, run_forward_pass},
    workspace::BasisStore,
};
use novomodelo_solver::{ActiveSolver, StageTemplate};

use super::StubComm;

/// The end-of-stage state at every stage of the first trial of the first iteration's forward
/// pass over `setup`, which holds no cuts, so its stage templates are the frozen ones.
pub fn first_iteration_forward_states(setup: &StudySetup) -> Vec<Vec<f64>> {
    assert_eq!(
        setup.fcf.total_active_cuts(),
        0,
        "the first iteration's forward pass runs against cut-free templates"
    );
    let stage_ctx = setup.stage_ctx();
    let training_ctx = setup.training_ctx();
    let frozen: Vec<StageTemplate> = training_ctx
        .node_graph
        .pool_stage
        .iter()
        .map(|stage| stage_ctx.templates[stage.0].clone())
        .collect();
    let num_stages = setup.num_stages();
    let mut pool = setup
        .create_workspace_pool(&StubComm, 1, ActiveSolver::new)
        .expect("workspace pool");
    let mut basis_store = BasisStore::new(1, training_ctx.node_graph.nodes.len());
    let mut records = vec![TrajectoryRecord::default(); num_stages];
    // The total stays the run's forward passes: the first trial's noise depends on it.
    let batch = ForwardPassBatch {
        local_forward_passes: 1,
        total_forward_passes: setup.fcf.forward_passes as usize,
        iteration: 1,
        fwd_offset: 0,
        event_sender: None,
    };
    let _ = run_forward_pass(
        &mut pool.workspaces,
        &mut basis_store,
        &stage_ctx,
        &frozen,
        &setup.fcf,
        &training_ctx,
        &batch,
        &mut records,
    )
    .expect("forward pass");
    records.into_iter().map(|record| record.state).collect()
}

/// The end-of-stage state of the last forward pass of the chain run `setup`
/// trained, at every stage but the last: the state the last iteration's cut of
/// pool `t` was computed at.
///
/// # Panics
///
/// Panics if `setup` did not train with `set_export_states(true)` and one
/// forward pass, if a pool holds no state, or if a pool drops a state dimension.
pub fn final_forward_states(setup: &StudySetup) -> Vec<Vec<f64>> {
    assert_eq!(
        setup.fcf.forward_passes, 1,
        "the last stored state of a pool is one pass's"
    );
    let n_state = setup.stage_state().n_state;
    let layouts = &setup.training_ctx().cut_state_layouts;
    (0..setup.num_stages() - 1)
        .map(|t| {
            assert_eq!(
                layouts[t].n_slots(),
                n_state,
                "pool {t} drops a state dimension, so its stored state is not the whole state"
            );
            let pool = &setup.fcf.pools[t];
            (0..pool.populated())
                .rev()
                .find_map(|slot| pool.trial_state(slot))
                .unwrap_or_else(|| panic!("pool {t} stores no trial state"))
                .to_vec()
        })
        .collect()
}
