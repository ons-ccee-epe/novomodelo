//! `cobre version` subcommand.
//!
//! Prints the binary version, active solver backend, and build profile.

use crate::error::CliError;

/// Print version, solver, and build information to stdout.
///
/// # Errors
///
/// Currently infallible; the `Result` is kept for uniform subcommand dispatch.
#[allow(clippy::unnecessary_wraps)]
pub fn execute() -> Result<(), CliError> {
    let version = env!("CARGO_PKG_VERSION");
    println!("cobre   v{version}");
    println!(
        "solver: {} {}",
        cobre_solver::active_solver_name(),
        cobre_solver::active_solver_version()
    );
    if cfg!(feature = "mpi") {
        println!("comm:   mpi");
    } else {
        println!("comm:   local");
    }
    println!("zstd:   enabled");
    println!(
        "arch:   {}-{}",
        std::env::consts::ARCH,
        std::env::consts::OS
    );
    if cfg!(debug_assertions) {
        println!("build:  debug");
    } else {
        println!("build:  release (lto=thin)");
    }

    Ok(())
}
