use novomodelo_core::temporal::BlockMode;
use novomodelo_sddp::StudySetup;
use novomodelo_sddp::lp::builder::RowFamily;
use novomodelo_sddp::setup::{NodePos, StageIdx};
use novomodelo_sddp::test_support::{capture_patched_node_template_at, oracle_initial_state};
use novomodelo_solver::{ActiveSolver, SolverInterface};

use super::shared_dual_identity::dual_family;
use crate::common::compact_map::CompactMap;
use crate::common::in_code_studies::fpha_lag_study;
use crate::common::power_guards::{DualFamily, Guard, PowerEvidence};
use crate::common::row_families::{RowClass, classify_stage_rows};
use crate::common::{Emission, build_setup_in_code_with_models_as};

const CHRONOLOGICAL_STAGE: usize = 1;
const UNIT_SCALE_BITS: u64 = 1.0_f64.to_bits();

fn build(emission: Emission) -> StudySetup {
    let (system, config, hydro_models) = fpha_lag_study();
    build_setup_in_code_with_models_as(system, &config, hydro_models, emission)
}

#[test]
fn fpha_lag_study_has_every_required_feature() {
    let (system, _, _) = fpha_lag_study();
    let logical = build(Emission::Logical);
    let compact = build(Emission::Compact);
    let logical_state = logical.stage_state();
    let compact_state = compact.stage_state();

    assert_eq!(
        logical_state.max_par_order, 2,
        "the logical form's lag state must be PAR(2)"
    );
    let logical_ctx = logical.training_ctx();
    let par = logical_ctx.stochastic.par();
    for stage in 0..system.stages().len() {
        for hydro in 0..logical_state.hydro_count {
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

    let compact_templates = &compact.inputs.stage_data.stage_templates;
    let compact_classes = classify_stage_rows(
        &compact,
        CHRONOLOGICAL_STAGE,
        compact_templates.templates[CHRONOLOGICAL_STAGE].num_rows,
    );
    let coupling = &compact.stage_ctx().inflow_coupling[CHRONOLOGICAL_STAGE];
    assert!(
        coupling.touched_rows().iter().any(|&row| {
            compact_classes[row as usize] == RowClass::Base(RowFamily::GenericConstraint)
        }),
        "stage {CHRONOLOGICAL_STAGE}: no touched inflow-coupling row is a generic-constraint \
         row; touched rows {:?}",
        coupling.touched_rows()
    );

    for (form, state) in [("logical", logical_state), ("compact", compact_state)] {
        assert!(
            state.n_anticipated > 0 && !state.commit_out.is_empty(),
            "{form}: anticipated state absent (n_anticipated {}, commit_out {:?})",
            state.n_anticipated,
            state.commit_out
        );
    }

    let logical_templates = &logical.inputs.stage_data.stage_templates;
    let scales_by_stage = |columns: &std::ops::Range<usize>| -> Vec<Vec<f64>> {
        (0..system.stages().len())
            .map(|stage| logical_templates.templates[stage].col_scale[columns.clone()].to_vec())
            .collect()
    };
    let any_off_unit = |scales: &[Vec<f64>]| {
        scales
            .iter()
            .flatten()
            .any(|scale| scale.to_bits() != UNIT_SCALE_BITS)
    };
    let z_scales = scales_by_stage(&logical_state.z_inflow);
    let lag_scales = scales_by_stage(&logical_state.inflow_lags);
    assert!(
        any_off_unit(&z_scales),
        "no z column scale differs from 1.0 on the logical templates; z scales by stage {z_scales:?}"
    );
    assert!(
        any_off_unit(&lag_scales),
        "no inflow-lag column scale differs from 1.0 on the logical templates; \
         lag scales by stage {lag_scales:?}"
    );

    let coupled_row_scales: Vec<(usize, u32, f64, f64)> = (0..system.stages().len())
        .flat_map(|stage| {
            let logical_template = &logical_templates.templates[stage];
            let compact_template = &compact_templates.templates[stage];
            let map = CompactMap::of_stage(
                logical_state,
                compact_state,
                logical_template,
                compact_template,
            );
            compact_templates.inflow_coupling_per_stage[stage]
                .touched_rows()
                .iter()
                .map(|&row| {
                    let logical_row = (0..logical_template.num_rows)
                        .find(|&r| map.row(r) == Some(row as usize))
                        .expect("a coupled compact row is a kept logical row");
                    (
                        stage,
                        row,
                        logical_template.row_scale[logical_row],
                        compact_template.row_scale[row as usize],
                    )
                })
                .collect::<Vec<_>>()
        })
        .collect();
    assert!(
        coupled_row_scales
            .iter()
            .any(|&(_, _, logical, compact)| logical.to_bits() != compact.to_bits()),
        "no compact coupled row's scale differs from its logical row's; \
         (stage, compact row, logical scale, compact scale): {coupled_row_scales:?}"
    );
}

#[test]
fn fpha_lag_study_duals_bind_on_a_no_cut_stage_solve() {
    let setup = build(Emission::Logical);
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
