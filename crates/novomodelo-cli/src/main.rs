//! # novomodelo
//!
//! Command-line interface for the [Novomodelo](https://github.com/ons-ccee-epe/novomodelo) power systems ecosystem.
//!
//! Provides commands for running optimization studies, validating input data,
//! and inspecting results from the terminal.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

mod banner;
mod commands;
mod error;
mod progress;
mod summary;
mod templates;

use std::sync::atomic::{AtomicBool, Ordering};

use clap::{Parser, Subcommand, ValueEnum};
use tracing::Level;
use tracing_subscriber::{
    Layer, filter::dynamic_filter_fn, layer::SubscriberExt, util::SubscriberInitExt,
};

use commands::{
    init::{self, InitArgs},
    run::{self, RunArgs, RunOutcome},
    schema::{self, SchemaArgs},
    validate::{self, ValidateArgs},
    version,
};
use error::CliError;

/// Color output mode for stdout and stderr (no environment variable override).
#[derive(Clone, Copy, Debug, ValueEnum)]
pub(crate) enum ColorWhen {
    /// Enable color on each of stdout and stderr that is a TTY (default).
    Auto,
    /// Always emit ANSI color escapes, even on a stream that is not a TTY.
    Always,
    /// Never emit ANSI color escapes.
    Never,
}

pub(crate) fn resolve_color(cli_color: ColorWhen) {
    let enabled = match cli_color {
        ColorWhen::Always => true,
        ColorWhen::Never => false,
        ColorWhen::Auto => return,
    };
    console::set_colors_enabled(enabled);
    console::set_colors_enabled_stderr(enabled);
}

static NON_ROOT_RANK: AtomicBool = AtomicBool::new(false);

#[allow(clippy::trivially_copy_pass_by_ref)]
fn keeps_event(level: &Level, non_root: bool) -> bool {
    !non_root || *level == Level::ERROR
}

pub(crate) fn mark_non_root_rank() {
    NON_ROOT_RANK.store(true, Ordering::Relaxed);
}

#[derive(Debug, Parser)]
#[command(name = "novomodelo", about = "Power system optimization in Rust")]
struct Cli {
    /// Control ANSI color output on stdout and stderr.
    ///
    /// `always` forces color on (useful under `mpiexec` which pipes output through
    /// a non-TTY). `never` disables all color. `auto` lets the terminal detection
    /// decide.
    #[arg(long, global = true, default_value = "auto")]
    color: ColorWhen,

    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Scaffold a new case directory from an embedded template.
    Init(InitArgs),
    /// Load a case directory, train a policy, and run simulation.
    Run(RunArgs),
    /// Validate a case directory and print a structured diagnostic report.
    Validate(ValidateArgs),
    /// Manage JSON Schema files for case directory input types.
    Schema(SchemaArgs),
    /// Print version, solver backend, and build information.
    Version,
}

fn main() {
    let cli = Cli::parse();

    resolve_color(cli.color);

    let env_filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn"));
    // A subscriber can only install once, so a failed init must not stop the command.
    let _ = tracing_subscriber::registry()
        .with(
            tracing_subscriber::fmt::layer()
                .with_writer(std::io::stderr)
                .with_target(false)
                .without_time()
                .with_filter(env_filter)
                // The rank is set after this install, so the predicate must run per
                // event; a statically cached callsite interest would outlive the flag.
                .with_filter(dynamic_filter_fn(|metadata, _| {
                    keeps_event(metadata.level(), NON_ROOT_RANK.load(Ordering::Relaxed))
                })),
        )
        .try_init();

    let result: Result<i32, CliError> = match cli.command {
        Command::Init(args) => init::execute(args).map(|()| 0),
        Command::Run(ref args) => run::execute(args).map(RunOutcome::exit_code),
        Command::Validate(ref args) => validate::execute(args).map(|()| 0),
        Command::Schema(ref args) => schema::execute(args).map(|()| 0),
        Command::Version => version::execute().map(|()| 0),
    };

    match result {
        Ok(code) => std::process::exit(code),
        Err(e) => {
            e.format_error(&console::Term::stderr());
            std::process::exit(e.exit_code());
        }
    }
}

#[cfg(test)]
#[path = "../tests/common/varying_external_deck.rs"]
mod varying_external_deck;

#[cfg(test)]
mod tests {
    use tracing::Level;

    use super::{ColorWhen, keeps_event, resolve_color};

    #[test]
    fn non_root_ranks_keep_only_errors() {
        for level in [Level::WARN, Level::INFO, Level::DEBUG, Level::TRACE] {
            assert!(!keeps_event(&level, true), "{level} must be dropped");
        }
        assert!(keeps_event(&Level::ERROR, true));
        for level in [
            Level::ERROR,
            Level::WARN,
            Level::INFO,
            Level::DEBUG,
            Level::TRACE,
        ] {
            assert!(keeps_event(&level, false), "{level} must be kept");
        }
    }

    #[test]
    fn test_resolve_color_always_enables_color() {
        resolve_color(ColorWhen::Always);
        assert!(
            console::colors_enabled() && console::colors_enabled_stderr(),
            "Always must enable color on stdout and stderr"
        );
    }

    #[test]
    fn test_resolve_color_never_disables_color() {
        resolve_color(ColorWhen::Never);
        assert!(
            !console::colors_enabled() && !console::colors_enabled_stderr(),
            "Never must disable color on stdout and stderr"
        );
    }
}
