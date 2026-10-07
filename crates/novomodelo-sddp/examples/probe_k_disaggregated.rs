//! Probe: print populated (`K`) and active (`A`) cut counts per stage after a
//! single SDDP iteration. A one-shot diagnostic, not a registered test/CI target.
//!
//! One iteration only: a multi-iteration disaggregated run is prohibitively
//! expensive under the current cut-selection kernel, and iter-1 already
//! populates the pool to the value the sizing estimate needs.
//!
//! ```text
//! cargo build --release -p cobre-sddp --example probe_k_disaggregated
//! ./target/release/examples/probe_k_disaggregated <study-config-path>
//! ```
//!
//! `<study-config-path>` is a `config.json`; the case directory is its parent
//! (mirroring the CLI's `cobre run <CASE_DIR>` convention).
//!
//! Output on stdout: one `stage=<t> populated_count=<K> active_count=<A>` line
//! per stage, then a `summary D=<D> M=<M> max_K=… mean_K=… min_K=…` line.
//!
//! Exit codes: `0` success; `2` bad arguments; `3` training ran but no pool was
//! populated; non-zero otherwise.

#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::panic,
    clippy::print_stdout
)]

use std::process::ExitCode;

#[cfg(feature = "highs")]
use std::path::{Path, PathBuf};

#[cfg(feature = "highs")]
use cobre_comm::LocalBackend;
#[cfg(feature = "highs")]
use cobre_core::scenario::ScenarioSource;
#[cfg(feature = "highs")]
use cobre_io::{config::StoppingRuleConfig, parse_config};
#[cfg(feature = "highs")]
use cobre_sddp::{StudySetup, hydro_models::prepare_hydro_models, setup::prepare_stochastic};
#[cfg(feature = "highs")]
use cobre_solver::highs::HighsSolver;

#[cfg(feature = "highs")]
const PROBE_MAX_ITERATIONS: u32 = 1;

#[cfg(not(feature = "highs"))]
fn main() -> ExitCode {
    eprintln!(
        "probe_k_disaggregated example requires the `highs` feature; rebuild with --features highs"
    );
    ExitCode::from(2)
}

#[cfg(feature = "highs")]
fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2 || args.iter().any(|a| a == "--help" || a == "-h") {
        eprintln!("usage: probe_k_disaggregated <study-config-path>");
        return ExitCode::from(2);
    }
    let config_path = PathBuf::from(&args[1]);

    if let Err(code) = run_probe(&config_path) {
        return code;
    }
    ExitCode::SUCCESS
}

#[cfg(feature = "highs")]
fn run_probe(config_path: &Path) -> Result<(), ExitCode> {
    let case_dir = match config_path.parent() {
        Some(dir) if !dir.as_os_str().is_empty() => dir.to_path_buf(),
        _ => {
            eprintln!(
                "error: cannot derive case directory from config path '{}'",
                config_path.display()
            );
            return Err(ExitCode::from(2));
        }
    };

    if !config_path.exists() {
        eprintln!(
            "error: config file does not exist: {}",
            config_path.display()
        );
        return Err(ExitCode::from(2));
    }

    let mut config = parse_config(config_path).map_err(|e| {
        eprintln!("error: failed to parse config: {e}");
        ExitCode::from(2)
    })?;

    // FCF pools size from stopping rules, so override here for one-iteration sizing.
    config.training.stopping_rules = Some(vec![StoppingRuleConfig::IterationLimit {
        limit: PROBE_MAX_ITERATIONS,
    }]);

    let system = cobre_io::load_case(&case_dir).map_err(|e| {
        eprintln!("error: load_case failed: {e}");
        ExitCode::from(1)
    })?;

    let prepared = prepare_stochastic(
        system,
        &case_dir,
        &config,
        42,
        &ScenarioSource::default(),
        None,
    )
    .map_err(|e| {
        eprintln!("error: prepare_stochastic failed: {e}");
        ExitCode::from(1)
    })?;
    let system = prepared.system;
    let stochastic = prepared.stochastic;

    let hydro_models = prepare_hydro_models(&system, &case_dir, false).map_err(|e| {
        eprintln!("error: prepare_hydro_models failed: {e}");
        ExitCode::from(1)
    })?;

    let mut setup = StudySetup::new(&system, &config, stochastic, hydro_models, Vec::new())
        .map_err(|e| {
            eprintln!("error: StudySetup::new failed: {e}");
            ExitCode::from(1)
        })?;

    // Guards direct reads of max_iterations.
    setup.loop_params.max_iterations = u64::from(PROBE_MAX_ITERATIONS);

    let comm = LocalBackend;
    let mut solver = HighsSolver::new().map_err(|e| {
        eprintln!("error: HiGHS init failed: {e}");
        ExitCode::from(1)
    })?;

    eprintln!(
        "probe_k_disaggregated: starting 1-iteration run (case_dir={}, D={}, M={})",
        case_dir.display(),
        setup.fcf.state_dimension,
        setup.loop_params.forward_passes,
    );

    let outcome = setup
        .train(&mut solver, &comm, 1, HighsSolver::new, None, None)
        .map_err(|e| {
            eprintln!("error: training failed: {e}");
            ExitCode::from(1)
        })?;
    if let Some(err) = outcome.error {
        eprintln!("error: training reported mid-iteration failure: {err}");
        return Err(ExitCode::from(1));
    }

    match print_pool_report(&setup) {
        PoolReport::Ok => Ok(()),
        PoolReport::EmptyPools => Err(ExitCode::from(3)),
    }
}

/// Outcome of [`print_pool_report`].
#[cfg(feature = "highs")]
enum PoolReport {
    Ok,
    EmptyPools,
}

/// Print the per-stage and summary lines. Returns [`PoolReport::EmptyPools`] if
/// every pool is empty.
#[cfg(feature = "highs")]
fn print_pool_report(setup: &StudySetup) -> PoolReport {
    let pools = &setup.fcf.pools;
    let d = setup.fcf.state_dimension;
    let m = setup.loop_params.forward_passes;

    let mut max_k: usize = 0;
    let mut min_k: usize = usize::MAX;
    let mut total_k: u64 = 0;

    for (t, pool) in pools.iter().enumerate() {
        let k = pool.populated();
        let a = pool.active_count();
        println!("stage={t} populated_count={k} active_count={a}");
        max_k = max_k.max(k);
        min_k = min_k.min(k);
        total_k = total_k.saturating_add(k as u64);
    }

    if max_k == 0 {
        eprintln!(
            "error: every stage pool is empty after iter 1 — expected disaggregated scale to populate at least one cut per stage"
        );
        return PoolReport::EmptyPools;
    }

    let n_stages = pools.len() as u64;
    #[allow(clippy::cast_precision_loss)]
    let mean_k = total_k as f64 / n_stages as f64;
    // Defensive: maps the init sentinel to 0.
    let min_k_out = if min_k == usize::MAX { 0 } else { min_k };

    println!("summary D={d} M={m} max_K={max_k} mean_K={mean_k:.2} min_K={min_k_out}");

    PoolReport::Ok
}
