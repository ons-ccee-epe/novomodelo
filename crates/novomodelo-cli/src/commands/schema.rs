//! `cobre schema export [--output-dir DIR]` subcommand.
//!
//! `export` is a sub-subcommand to leave room for future `validate`/`list` siblings.

use std::path::PathBuf;

use clap::{Args, Subcommand};
use cobre_io::schema::SchemaExportError;
use console::Term;

use crate::error::CliError;

/// Arguments for the `cobre schema` subcommand.
#[derive(Debug, Args)]
#[command(about = "Manage JSON Schema files for case directory input types")]
pub struct SchemaArgs {
    /// Schema operation to perform.
    #[command(subcommand)]
    pub command: SchemaCommand,
}

/// Sub-subcommands for `cobre schema`.
#[derive(Debug, Subcommand)]
pub enum SchemaCommand {
    /// Export JSON Schema files for all input types to a directory.
    Export(ExportArgs),
}

/// Arguments for the `cobre schema export` sub-subcommand.
#[derive(Debug, Args)]
#[command(about = "Export JSON Schema files for all input types")]
pub struct ExportArgs {
    /// Directory to write schema files into.
    ///
    /// The directory is created if it does not exist. Existing schema files
    /// are overwritten without prompting (schemas are generated, not hand-edited).
    #[arg(long, default_value = ".")]
    pub output_dir: PathBuf,
}

/// Execute the `schema` subcommand.
///
/// # Errors
///
/// Returns [`CliError::Internal`] if schema generation fails.
/// Returns [`CliError::Io`] if the output directory cannot be created or a
/// file write fails.
pub fn execute(args: &SchemaArgs) -> Result<(), CliError> {
    match args.command {
        SchemaCommand::Export(ref export_args) => execute_export(export_args),
    }
}

fn execute_export(args: &ExportArgs) -> Result<(), CliError> {
    let count = cobre_io::schema::export_schemas(&args.output_dir).map_err(|err| match err {
        SchemaExportError::Generation(e) => CliError::Internal {
            message: format!("schema generation failed: {e}"),
        },
        SchemaExportError::Serialization { filename, source } => CliError::Internal {
            message: format!("serialization error for schema {filename}: {source}"),
        },
        SchemaExportError::Io { path, source } => CliError::Io {
            source,
            context: path.display().to_string(),
        },
    })?;

    let _ = Term::stderr().write_line(&format!(
        "Exported {count} schema files to {}",
        args.output_dir.display()
    ));

    Ok(())
}
