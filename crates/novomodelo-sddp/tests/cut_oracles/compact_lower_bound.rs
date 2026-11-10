use novomodelo_sddp::StudySetup;
use novomodelo_sddp::indexer::CutSlot;
use novomodelo_sddp::setup::{NodePos, StageIdx};
use novomodelo_sddp::test_support::no_cut_root_lower_bound;
use novomodelo_sddp::test_support::template_structure::state_lag_dims;
use novomodelo_solver::ActiveSolver;

use super::expander_cut_rows::{LAG_STUDY, assert_forms_share_cut_geometry};
use crate::common::in_code_studies::fpha_lag_study;
use crate::common::oracle_tolerance::{ORACLE_ABS_TOL, ORACLE_REL_TOL};
use crate::common::stored_cut_oracles::apply_oracle_config;
use crate::common::{Emission, build_setup_in_code_with_models_as, train_storing_trial_states};

fn trained_forms() -> (StudySetup, StudySetup) {
    let build = |emission| {
        let (system, mut config, hydro_models) = fpha_lag_study();
        apply_oracle_config(&mut config);
        build_setup_in_code_with_models_as(system, &config, hydro_models, emission)
    };
    let mut logical = build(Emission::Logical);
    let mut compact = build(Emission::Compact);
    train_storing_trial_states(LAG_STUDY, &mut logical);
    assert_forms_share_cut_geometry(LAG_STUDY, &logical, &compact);
    compact.fcf = logical.fcf.clone();
    (logical, compact)
}

fn root_pool(setup: &StudySetup) -> usize {
    let graph = &setup.inputs.node_graph;
    (0..graph.nodes.len())
        .map(|pos| &graph.nodes[NodePos(pos)])
        .find(|node| node.stage == StageIdx(0))
        .expect("study must have a stage-0 node")
        .pool_id
}

fn root_lag_cuts(setup: &StudySetup) -> usize {
    let state = setup.stage_state();
    let pool = root_pool(setup);
    let layout = &setup.training_ctx().cut_state_layouts[pool];
    setup
        .fcf
        .active_cuts(pool)
        .filter(|(_, _, coefficients)| {
            coefficients.iter().enumerate().any(|(j, &c)| {
                let d = layout.global_state_index(CutSlot::new(j));
                c != 0.0
                    && state.nonzero_state_indices.binary_search(&d).is_ok()
                    && state_lag_dims(state).contains(&d.get())
            })
        })
        .count()
}

fn root_lower_bound(setup: &StudySetup) -> f64 {
    let mut solver = ActiveSolver::new().expect("ActiveSolver::new must succeed");
    no_cut_root_lower_bound(setup, &mut solver).expect("the root lower bound must solve")
}

fn assert_forms_agree(label: &str, logical: &StudySetup, compact: &StudySetup) -> f64 {
    let (want, got) = (root_lower_bound(logical), root_lower_bound(compact));
    let tolerance = ORACLE_ABS_TOL + ORACLE_REL_TOL * want.abs();
    assert!(
        (got - want).abs() <= tolerance,
        "{label}: compact root lower bound {got:?} vs logical {want:?}, tolerance {tolerance:e}"
    );
    got
}

#[test]
fn compact_root_lower_bound_matches_the_logical_form_with_trained_cuts() {
    let (logical, compact) = trained_forms();
    assert!(
        root_lag_cuts(&logical) > 0,
        "{LAG_STUDY}: the root pool holds no active cut with a nonzero lag coefficient"
    );

    let got = assert_forms_agree(LAG_STUDY, &logical, &compact);

    assert_eq!(
        root_lower_bound(&compact).to_bits(),
        got.to_bits(),
        "{LAG_STUDY}: a second compact evaluation on a fresh solver moved the bound"
    );
}

#[test]
fn compact_root_lower_bound_tracks_a_deactivated_cut_like_the_logical_form() {
    let (mut logical, mut compact) = trained_forms();
    let pool = root_pool(&logical);
    let (slot, _, _) = logical
        .fcf
        .active_cuts(pool)
        .next()
        .expect("the root pool holds an active cut");
    let slot = u32::try_from(slot).expect("a cut slot fits u32");
    logical.fcf.pools[pool].set_active(slot, false);
    compact.fcf.pools[pool].set_active(slot, false);

    assert_forms_agree(LAG_STUDY, &logical, &compact);
}
