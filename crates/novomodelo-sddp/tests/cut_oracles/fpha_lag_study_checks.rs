use novomodelo_core::temporal::BlockMode;
use novomodelo_sddp::StudySetup;
use novomodelo_sddp::lp::builder::RowFamily;
use novomodelo_sddp::setup::{NodePos, StageIdx};
use novomodelo_sddp::test_support::{capture_patched_node_template_at, oracle_initial_state};
use novomodelo_solver::{ActiveSolver, SolverInterface};

use crate::common::build_setup_in_code_with_models;
use crate::common::in_code_studies::fpha_lag_study;
use crate::common::power_guards::{DualFamily, Guard, PowerEvidence};
use crate::common::row_families::{RowClass, classify_stage_rows};
use crate::shared_dual_identity::dual_family;

const CHRONOLOGICAL_STAGE: usize = 1;
const UNIT_SCALE_BITS: u64 = 1.0_f64.to_bits();

fn build() -> StudySetup {
    let (system, config, hydro_models) = fpha_lag_study();
    build_setup_in_code_with_models(system, &config, hydro_models)
}

#[test]
fn fpha_lag_study_has_every_required_feature() {
    let (system, _, _) = fpha_lag_study();
    let setup = build();
    let state = setup.stage_state();

    assert_eq!(state.max_par_order, 2, "the lag state must be PAR(2)");
    let ctx = setup.training_ctx();
    let par = ctx.stochastic.par();
    for stage in 0..system.stages().len() {
        for hydro in 0..state.hydro_count {
            for (lag, &psi) in par.psi_slice(stage, hydro).iter().enumerate() {
                assert!(
                    psi != 0.0 && psi.abs() != 1.0,
                    "stage {stage} hydro {hydro} lag {lag}: psi {psi} must be nonzero with magnitude not 1"
                );
            }
        }
    }

    let stages = system.stages();
    let modes: Vec<BlockMode> = stages.iter().map(|stage| stage.block_mode).collect();
    assert_eq!(
        modes,
        [
            BlockMode::Parallel,
            BlockMode::Chronological,
            BlockMode::Parallel
        ],
        "only the middle stage is chronological"
    );
    let hours: Vec<f64> = stages[CHRONOLOGICAL_STAGE]
        .blocks
        .iter()
        .map(|block| block.duration_hours)
        .collect();
    assert!(
        hours.len() == 2 && hours[0] != hours[1],
        "the chronological stage needs two blocks of different hours, got {hours:?}"
    );

    let templates = &setup.inputs.stage_data.stage_templates;
    let classes = classify_stage_rows(
        &setup,
        CHRONOLOGICAL_STAGE,
        templates.templates[CHRONOLOGICAL_STAGE].num_rows,
    );
    let coupling = &setup.stage_ctx().inflow_coupling[CHRONOLOGICAL_STAGE];
    assert!(
        coupling
            .touched_rows()
            .iter()
            .any(|&row| classes[row as usize] == RowClass::Base(RowFamily::GenericConstraint)),
        "stage {CHRONOLOGICAL_STAGE}: no touched inflow-coupling row is a generic-constraint \
         row; touched rows {:?}",
        coupling.touched_rows()
    );

    assert!(
        state.n_anticipated > 0 && !state.commit_out.is_empty(),
        "anticipated state absent (n_anticipated {}, commit_out {:?})",
        state.n_anticipated,
        state.commit_out
    );

    let coupled_row_scales: Vec<(usize, u32, f64)> = (0..system.stages().len())
        .flat_map(|stage| {
            let template = &templates.templates[stage];
            templates.inflow_coupling_per_stage[stage]
                .touched_rows()
                .iter()
                .map(|&row| (stage, row, template.row_scale[row as usize]))
                .collect::<Vec<_>>()
        })
        .collect();
    assert!(
        coupled_row_scales
            .iter()
            .any(|&(_, _, scale)| scale.to_bits() != UNIT_SCALE_BITS),
        "no coupled row's scale differs from 1.0; (stage, row, scale): {coupled_row_scales:?}"
    );
}

#[test]
fn fpha_lag_study_duals_bind_on_a_no_cut_stage_solve() {
    let setup = build();
    assert_eq!(
        setup.inputs.node_graph.nodes[NodePos(CHRONOLOGICAL_STAGE)].stage,
        StageIdx(CHRONOLOGICAL_STAGE),
        "node position and stage coincide on the chain"
    );
    let zero_noise = vec![0.0; setup.inputs.stochastic.dim()];
    let template = capture_patched_node_template_at(
        &setup,
        NodePos(CHRONOLOGICAL_STAGE),
        &zero_noise,
        &oracle_initial_state(&setup),
    );
    let mut solver = ActiveSolver::new().expect("ActiveSolver::new");
    solver.load_model(&template);
    let view = solver.solve(None).expect("the no-cut stage LP must solve");

    let state = setup.stage_state();
    let mut evidence = PowerEvidence::new(state.hydro_count, state.max_par_order);
    let classes = classify_stage_rows(&setup, CHRONOLOGICAL_STAGE, template.num_rows);
    for (row, &class) in classes.iter().enumerate() {
        if let Some(family) = dual_family(class) {
            evidence.record_family_dual(family, view.dual[row]);
        }
    }
    evidence.assert_holds(
        "no-cut chronological stage",
        &[
            Guard::FamilyDual(DualFamily::Fpha),
            Guard::FamilyDual(DualFamily::Evaporation),
            Guard::FamilyDual(DualFamily::Generic),
        ],
    );
}
