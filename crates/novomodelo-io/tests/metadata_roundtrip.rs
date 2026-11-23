//! Training and simulation `metadata.json` survive a write-read round trip, and
//! a product-named version key is refused.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::float_cmp)]

use novomodelo_io::{
    DeviationSummary, DeviationWorstEntry, DistributionInfo, HostLayout, MetadataBounds,
    MetadataConfiguration, MetadataConvergence, MetadataCost, MetadataIterations,
    MetadataProblemDimensions, MetadataRowPool, MetadataScenarios, MetadataSimulationSolveStats,
    MetadataTrainingSolveStats, RunStatus, SimulationMetadata, TrainingMetadata,
    read_simulation_metadata, read_training_metadata, write_simulation_metadata,
    write_training_metadata,
};

/// Frozen multi-node `DistributionInfo` pinning the on-disk per-host rank shape.
const MULTI_NODE_DISTRIBUTION_JSON: &str = r#"{
  "backend": "mpi",
  "world_size": 8,
  "ranks_participated": 8,
  "num_hosts": 2,
  "threads_per_rank": 4,
  "mpi_library": "Open MPI v4.1.6",
  "mpi_standard": "MPI 4.0",
  "thread_level": "Funneled",
  "hosts": [
    { "hostname": "node01", "ranks": [0, 1, 2, 3] },
    { "hostname": "node02", "ranks": [4, 5, 6, 7] }
  ]
}"#;

#[test]
fn multi_node_distribution_deserializes_with_correct_rank_lists() {
    let decoded: DistributionInfo = serde_json::from_str(MULTI_NODE_DISTRIBUTION_JSON)
        .expect("multi-node distribution fixture must deserialize");

    assert_eq!(decoded.backend, "mpi");
    assert_eq!(decoded.world_size, 8);
    assert_eq!(decoded.num_hosts, 2);
    assert_eq!(decoded.hosts.len(), 2);

    assert_eq!(decoded.hosts[0].hostname, "node01");
    assert_eq!(decoded.hosts[0].ranks, vec![0, 1, 2, 3]);
    assert_eq!(decoded.hosts[1].hostname, "node02");
    assert_eq!(decoded.hosts[1].ranks, vec![4, 5, 6, 7]);
}

fn fully_populated_distribution() -> DistributionInfo {
    DistributionInfo {
        backend: "mpi".to_string(),
        world_size: 8,
        ranks_participated: 8,
        num_hosts: 2,
        threads_per_rank: 4,
        mpi_library: Some("Open MPI v4.1.6".to_string()),
        mpi_standard: Some("MPI 4.0".to_string()),
        thread_level: Some("Funneled".to_string()),
        slurm_job_id: Some("123456".to_string()),
        hosts: vec![
            HostLayout {
                hostname: "node01".to_string(),
                ranks: vec![0, 1, 2, 3],
            },
            HostLayout {
                hostname: "node02".to_string(),
                ranks: vec![4, 5, 6, 7],
            },
        ],
    }
}

fn fully_populated_training_metadata() -> TrainingMetadata {
    TrainingMetadata {
        software: "novomodelo".to_string(),
        software_version: "0.1.6".to_string(),
        hostname: "node01".to_string(),
        solver: "highs".to_string(),
        solver_version: Some("1.8.0".to_string()),
        started_at: "2026-01-17T08:00:00Z".to_string(),
        completed_at: "2026-01-17T12:30:00Z".to_string(),
        duration_seconds: 16_200.0,
        status: RunStatus::Complete,
        configuration: MetadataConfiguration {
            seed: Some(42),
            max_iterations: Some(100),
            forward_passes: Some(192),
            stopping_mode: "any".to_string(),
            policy_mode: "fresh".to_string(),
        },
        problem_dimensions: MetadataProblemDimensions {
            num_stages: 12,
            num_hydros: 160,
            num_thermals: 200,
            num_buses: 5,
            num_lines: 8,
        },
        iterations: MetadataIterations {
            completed: 100,
            converged_at: Some(95),
        },
        convergence: MetadataConvergence {
            achieved: true,
            final_gap_percent: Some(0.45),
            termination_reason: "bound_stalling".to_string(),
        },
        row_pool: MetadataRowPool {
            total_generated: 1_250_000,
            total_active: 980_000,
            peak_active: 1_100_000,
            cuts_active: 980_000,
            rows_in_lp_total: 0,
            rows_in_lp_solve_count: 0,
            rows_in_lp_max: 0,
            total_loaded: 50_000,
        },
        bounds: MetadataBounds {
            final_lower_bound: 48_500.0,
            final_upper_bound: Some(49_000.0),
            final_upper_bound_std: Some(250.0),
            final_upper_bound_kind: "statistical".to_string(),
        },
        solve_stats: MetadataTrainingSolveStats {
            total_lp_solves: Some(84_000),
            first_try: Some(80_000),
            retried: Some(3_800),
            failed: Some(200),
            forward_solve_seconds: Some(123.5),
            backward_solve_seconds: Some(456.75),
            parallelism: Some(8),
        },
        setup: None,
        production_fit_deviation: Some(DeviationSummary {
            n_entries: 3,
            mean_abs: 4.1,
            max_abs: 31.7,
            worst_relative: 0.062,
            worst_entry: Some(DeviationWorstEntry {
                entity_id: 12,
                stage_id: 4,
                relative: 0.062,
                mean_abs: 8.0,
                max_abs: 31.7,
            }),
        }),
        distribution: fully_populated_distribution(),
    }
}

fn fully_populated_simulation_metadata() -> SimulationMetadata {
    SimulationMetadata {
        software: "novomodelo".to_string(),
        software_version: "0.1.6".to_string(),
        hostname: "node01".to_string(),
        solver: "highs".to_string(),
        solver_version: Some("1.8.0".to_string()),
        started_at: "2026-01-17T13:00:00Z".to_string(),
        completed_at: "2026-01-17T13:15:00Z".to_string(),
        duration_seconds: 900.0,
        status: RunStatus::Complete,
        scenarios: MetadataScenarios {
            total: 100,
            completed: 100,
            failed: 0,
        },
        cost: Some(MetadataCost {
            mean_cost: 12_345.6,
            std_cost: 200.0,
        }),
        solve_stats: MetadataSimulationSolveStats {
            total_lp_solves: Some(50_000),
            first_try: Some(48_000),
            retried: Some(1_900),
            failed: Some(100),
            solve_seconds: Some(321.0),
            parallelism: Some(8),
        },
        distribution: fully_populated_distribution(),
    }
}

#[test]
fn fully_populated_training_metadata_survives_write_read_roundtrip() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("metadata.json");
    let original = fully_populated_training_metadata();

    write_training_metadata(&path, &original).expect("write must succeed");
    let decoded = read_training_metadata(&path).expect("read must succeed");

    assert_eq!(
        decoded.row_pool.total_loaded,
        original.row_pool.total_loaded
    );
    assert_eq!(
        decoded.bounds.final_lower_bound,
        original.bounds.final_lower_bound
    );
    assert_eq!(
        decoded.bounds.final_upper_bound,
        original.bounds.final_upper_bound
    );
    assert_eq!(
        decoded.bounds.final_upper_bound_std,
        original.bounds.final_upper_bound_std
    );

    assert_eq!(
        decoded.solve_stats.total_lp_solves,
        original.solve_stats.total_lp_solves
    );
    assert_eq!(
        decoded.solve_stats.first_try,
        original.solve_stats.first_try
    );
    assert_eq!(decoded.solve_stats.retried, original.solve_stats.retried);
    assert_eq!(decoded.solve_stats.failed, original.solve_stats.failed);
    assert_eq!(
        decoded.solve_stats.forward_solve_seconds,
        original.solve_stats.forward_solve_seconds
    );
    assert_eq!(
        decoded.solve_stats.backward_solve_seconds,
        original.solve_stats.backward_solve_seconds
    );
    assert_eq!(
        decoded.solve_stats.parallelism,
        original.solve_stats.parallelism
    );

    assert_eq!(decoded.distribution.hosts.len(), 2);
    assert_eq!(decoded.distribution.hosts[0].hostname, "node01");
    assert_eq!(decoded.distribution.hosts[0].ranks, vec![0, 1, 2, 3]);
    assert_eq!(decoded.distribution.hosts[1].hostname, "node02");
    assert_eq!(decoded.distribution.hosts[1].ranks, vec![4, 5, 6, 7]);

    let summary = decoded
        .production_fit_deviation
        .expect("deviation summary must survive round-trip");
    assert_eq!(summary.n_entries, 3);
    assert_eq!(summary.mean_abs, 4.1);
    assert_eq!(summary.max_abs, 31.7);
    assert_eq!(summary.worst_relative, 0.062);
    let worst = summary.worst_entry.expect("worst entry must survive");
    assert_eq!(worst.entity_id, 12);
    assert_eq!(worst.stage_id, 4);
}

#[test]
fn fully_populated_simulation_metadata_survives_write_read_roundtrip() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("metadata.json");
    let original = fully_populated_simulation_metadata();

    write_simulation_metadata(&path, &original).expect("write must succeed");
    let decoded = read_simulation_metadata(&path).expect("read must succeed");

    let original_cost = original.cost.expect("fixture must populate cost");
    let decoded_cost = decoded.cost.expect("cost must survive roundtrip");
    assert_eq!(decoded_cost.mean_cost, original_cost.mean_cost);
    assert_eq!(decoded_cost.std_cost, original_cost.std_cost);

    assert_eq!(
        decoded.solve_stats.total_lp_solves,
        original.solve_stats.total_lp_solves
    );
    assert_eq!(
        decoded.solve_stats.first_try,
        original.solve_stats.first_try
    );
    assert_eq!(decoded.solve_stats.retried, original.solve_stats.retried);
    assert_eq!(decoded.solve_stats.failed, original.solve_stats.failed);
    assert_eq!(
        decoded.solve_stats.solve_seconds,
        original.solve_stats.solve_seconds
    );
    assert_eq!(
        decoded.solve_stats.parallelism,
        original.solve_stats.parallelism
    );

    assert_eq!(decoded.distribution.hosts.len(), 2);
    assert_eq!(decoded.distribution.hosts[0].hostname, "node01");
    assert_eq!(decoded.distribution.hosts[0].ranks, vec![0, 1, 2, 3]);
    assert_eq!(decoded.distribution.hosts[1].hostname, "node02");
    assert_eq!(decoded.distribution.hosts[1].ranks, vec![4, 5, 6, 7]);
}

#[test]
fn product_named_version_key_is_a_deserialize_error() {
    let mut training = serde_json::to_value(fully_populated_training_metadata()).unwrap();
    let mut simulation = serde_json::to_value(fully_populated_simulation_metadata()).unwrap();
    for value in [&mut training, &mut simulation] {
        let object = value.as_object_mut().unwrap();
        object.remove("software");
        let version = object.remove("software_version").unwrap();
        object.insert("novomodelo_version".to_string(), version);
    }

    assert!(serde_json::from_value::<TrainingMetadata>(training).is_err());
    assert!(serde_json::from_value::<SimulationMetadata>(simulation).is_err());
}
