//! The boundary cuts a target loads from `policy.boundary` are not part of any
//! stage-LP template fact group: two targets whose boundary checkpoints differ,
//! and one whose checkpoint is absent, produce byte-identical
//! [`template_fact_groups`].

#![allow(clippy::unwrap_used, clippy::expect_used)]

mod common;

use std::fs;
use std::path::Path;

use novomodelo_io::Config;
use novomodelo_sddp::hydro_models::prepare_hydro_models;
use novomodelo_sddp::setup::{StudyParams, prepare_stochastic};
use novomodelo_sddp::test_support::template_fact_groups;
use novomodelo_sddp::{BoundaryStateRequirements, StudySetup, resolve_boundary_state_requirements};
use serde_json::Value;
use tempfile::TempDir;

use common::{copy_dir_recursive, prepare_dual_calendar_boundary_case_with};

const SEED: u64 = 42;
const BOUNDARY_CUTS: &str = "source_policy/cuts/000.bin";

fn boundary_requirements(case_dir: &Path, config: &Config) -> BoundaryStateRequirements {
    let Some(bp) = config.policy.boundary.as_ref() else {
        return BoundaryStateRequirements::none();
    };
    if !bp.checkpoint_path(case_dir).join("manifest.bin").exists() {
        return BoundaryStateRequirements::present(0);
    }
    resolve_boundary_state_requirements(case_dir, config)
        .unwrap_or_else(|_| BoundaryStateRequirements::present(0))
}

fn setup_for(case_dir: &Path) -> StudySetup {
    let config_path = case_dir.join("config.json");
    let config = novomodelo_io::parse_config(&config_path).unwrap();
    let system = novomodelo_io::load_case(case_dir).unwrap();
    let training_source = config.training_scenario_source(&config_path).unwrap();
    let simulation_source = config.simulation_scenario_source(&config_path).unwrap();
    let inflow_lag_depth = boundary_requirements(case_dir, &config).inflow_lag_depth();
    let prepared = prepare_stochastic(
        system,
        case_dir,
        &config,
        SEED,
        &training_source,
        inflow_lag_depth,
    )
    .unwrap();
    let hydro_models = prepare_hydro_models(&prepared.system, case_dir, false).unwrap();

    let mut construction = StudyParams::from_config(&config, Vec::new()).unwrap();
    construction.boundary = boundary_requirements(case_dir, &config);
    construction.scalar_parameters = novomodelo_io::load_case_with_artifacts(case_dir)
        .unwrap()
        .artifacts
        .scalar_parameters;
    StudySetup::from_broadcast_params(
        &prepared.system,
        prepared.stochastic,
        construction,
        hydro_models,
        &training_source,
        &simulation_source,
    )
    .unwrap()
}

fn differing_groups(a: &StudySetup, b: &StudySetup) -> Vec<&'static str> {
    let (a, b) = (template_fact_groups(a), template_fact_groups(b));
    assert!(
        a.keys().eq(b.keys()),
        "fact-group key sets differ (a={:?}, b={:?})",
        a.keys().collect::<Vec<_>>(),
        b.keys().collect::<Vec<_>>(),
    );
    a.iter()
        .filter(|(group, bytes)| b.get(*group) != Some(*bytes))
        .map(|(group, _)| *group)
        .collect()
}

fn set_every_iteration_limit(config: &mut Value, limit: u64) {
    let rules = config
        .pointer_mut("/training/stopping_rules")
        .and_then(Value::as_array_mut)
        .expect("the source config must hold training stopping rules");
    for rule in rules.iter_mut().filter(|r| r["type"] == "iteration_limit") {
        rule["limit"] = limit.into();
    }
}

#[test]
fn boundary_policy_does_not_enter_lp_fact_groups() {
    let work = TempDir::new().unwrap();
    let case_a = work.path().join("a");
    let case_b = work.path().join("b");
    let case_without = work.path().join("without");
    prepare_dual_calendar_boundary_case_with(&case_a, |_| {});
    prepare_dual_calendar_boundary_case_with(&case_b, |config| {
        set_every_iteration_limit(config, 1);
    });
    assert_ne!(
        fs::read(case_a.join(BOUNDARY_CUTS)).unwrap(),
        fs::read(case_b.join(BOUNDARY_CUTS)).unwrap(),
        "the two boundary checkpoints must differ for the comparison to mean anything",
    );
    copy_dir_recursive(&case_a, &case_without);
    fs::remove_dir_all(case_without.join("source_policy")).unwrap();

    let setup_a = setup_for(&case_a);
    let setup_b = setup_for(&case_b);
    let setup_without = setup_for(&case_without);

    assert_eq!(
        differing_groups(&setup_a, &setup_b),
        Vec::<&str>::new(),
        "fact groups that differ between boundary checkpoints",
    );
    assert_eq!(
        differing_groups(&setup_a, &setup_without),
        Vec::<&str>::new(),
        "fact groups that differ between a case with and without its boundary checkpoint",
    );
}
