//! DHAT heap-allocation baseline for the backward-pass hot path (D19 case).
//!
//! ```text
//! cargo run --example dhat_baseline --features dhat-heap -p cobre-sddp --profile profiling
//! ```
//!
//! The `profiling` profile (not plain `--release`) is mandatory: a `release`
//! optimisation level is required to be representative, but `profiling` also
//! keeps line-number debug info so DHAT resolves allocation sites to source
//! rather than bare addresses.
//!
//! On exit DHAT writes `dhat-heap.json` in the working directory; view it at
//! <https://nnethercote.github.io/dh_view/dh_view.html>.

#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::panic,
    clippy::print_stdout
)]

#[cfg(feature = "highs")]
use cobre_io::config::TrainingSelection;
#[cfg(feature = "dhat-heap")]
#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

#[cfg(feature = "highs")]
use std::path::Path;

#[cfg(feature = "highs")]
use cobre_comm::{CommData, CommError, Communicator, ReduceOp};
#[cfg(feature = "highs")]
use cobre_core::scenario::ScenarioSource;
#[cfg(feature = "highs")]
use cobre_io::{config::StoppingRuleConfig, parse_config};
#[cfg(feature = "highs")]
use cobre_sddp::{StudySetup, hydro_models::prepare_hydro_models, setup::prepare_stochastic};
#[cfg(feature = "highs")]
use cobre_solver::highs::HighsSolver;

/// Single-rank stub communicator (mirrors `tests/deterministic.rs`).
#[cfg(feature = "highs")]
struct StubComm;

#[cfg(feature = "highs")]
impl Communicator for StubComm {
    fn allgatherv<T: CommData>(
        &self,
        send: &[T],
        recv: &mut [T],
        _counts: &[usize],
        _displs: &[usize],
    ) -> Result<(), CommError> {
        recv[..send.len()].clone_from_slice(send);
        Ok(())
    }

    fn allreduce<T: CommData>(
        &self,
        send: &[T],
        recv: &mut [T],
        _op: ReduceOp,
    ) -> Result<(), CommError> {
        recv.clone_from_slice(send);
        Ok(())
    }

    fn broadcast<T: CommData>(&self, _buf: &mut [T], _root: usize) -> Result<(), CommError> {
        Ok(())
    }

    fn barrier(&self) -> Result<(), CommError> {
        Ok(())
    }

    fn rank(&self) -> usize {
        0
    }

    fn size(&self) -> usize {
        1
    }

    fn abort(&self, error_code: i32) -> ! {
        std::process::exit(error_code)
    }
}

#[cfg(not(feature = "highs"))]
fn main() {
    eprintln!("dhat_baseline example requires the `highs` feature; rebuild with --features highs");
}

#[cfg(feature = "highs")]
fn main() {
    #[cfg(feature = "dhat-heap")]
    let _profiler = dhat::Profiler::new_heap();

    let case_dir = Path::new("examples/deterministic/d19-multi-hydro-par");

    println!("Loading D19 case from {:?}", case_dir.canonicalize().ok());

    let config_path = case_dir.join("config.json");
    let mut config = parse_config(&config_path).expect("config must parse");

    config.training.selection = Some(TrainingSelection::Sampled { forward_passes: 3 });
    config.training.stopping_rules = Some(vec![StoppingRuleConfig::IterationLimit { limit: 10 }]);

    let system = cobre_io::load_case(case_dir).expect("load_case must succeed");

    let cobre_sddp::PrepareStochasticResult {
        system, stochastic, ..
    } = prepare_stochastic(
        system,
        case_dir,
        &config,
        42,
        &ScenarioSource::default(),
        None,
    )
    .expect("prepare_stochastic must succeed");

    let hydro_models =
        prepare_hydro_models(&system, case_dir, false).expect("prepare_hydro_models must succeed");

    let mut setup = StudySetup::new(&system, &config, stochastic, hydro_models, Vec::new())
        .expect("StudySetup must build");

    let comm = StubComm;
    let mut solver = HighsSolver::new().expect("HighsSolver::new must succeed");

    println!("Starting training (3 forward passes, 10 iterations)...");

    let outcome = setup
        .train(&mut solver, &comm, 1, HighsSolver::new, None, None)
        .expect("train must return Ok");

    let result = outcome.result;

    println!(
        "Training complete: {} iterations, final_lb = {:.4}, reason = {}",
        result.iterations, result.final_lb, result.reason
    );

    #[cfg(feature = "dhat-heap")]
    println!("DHAT profile written to dhat-heap.json");

    #[cfg(not(feature = "dhat-heap"))]
    println!(
        "Warning: dhat-heap feature not enabled. Re-run with --features dhat-heap to produce a profile."
    );
}
