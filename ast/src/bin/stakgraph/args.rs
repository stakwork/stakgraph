use clap::{ArgAction, Args, Parser, Subcommand};
use shared::{Error, Result};

#[derive(Debug, Parser)]
#[command(name = "stakgraph")]
#[command(version)]
#[command(about = "Parse files and print a graph-oriented summary")]
pub struct CliArgs {
    #[command(subcommand)]
    pub command: Option<Commands>,

    /// Include unverified function calls in the graph
    #[arg(long, action = ArgAction::SetTrue)]
    pub allow: bool,

    /// Skip extracting function call relationships
    #[arg(long = "skip-calls", action = ArgAction::SetTrue)]
    pub skip_calls: bool,

    /// Exclude nodes nested inside other nodes
    #[arg(long = "no-nested", action = ArgAction::SetTrue)]
    pub no_nested: bool,

    /// Suppress all logs except errors (overrides RUST_LOG)
    #[arg(long, short = 'q', action = ArgAction::SetTrue)]
    pub quiet: bool,

    /// Show info and debug logs (overrides RUST_LOG)
    #[arg(long, short = 'v', action = ArgAction::SetTrue, conflicts_with = "quiet")]
    pub verbose: bool,

    /// Show [perf][memory] logs (implies verbose)
    #[arg(long, action = ArgAction::SetTrue, conflicts_with = "quiet")]
    pub perf: bool,

    /// Input files or directories (comma-separated or multiple args)
    #[arg(value_name = "FILE_OR_DIR", num_args = 0..)]
    pub files: Vec<String>,
}

#[derive(Debug, Subcommand)]
pub enum Commands {
    /// Print a token-budget-aware high-level summary of a directory
    Summarize(SummarizeArgs),
}

#[derive(Debug, Args)]
pub struct SummarizeArgs {
    /// Token budget for the output (default: 2000)
    #[arg(long, default_value = "2000")]
    pub max_tokens: usize,

    /// Maximum directory depth to display (default: adaptive, starts at 1)
    #[arg(long)]
    pub depth: Option<usize>,

    /// Path to summarize (default: current directory)
    #[arg(value_name = "PATH", default_value = ".")]
    pub path: String,
}

impl CliArgs {
    pub fn parse_and_expand() -> Result<Self> {
        let mut args = Self::parse();
        args.files = args
            .files
            .into_iter()
            .flat_map(|value| {
                value
                    .split(',')
                    .map(str::trim)
                    .filter(|part| !part.is_empty())
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
            })
            .collect();

        if args.command.is_none() && args.files.is_empty() {
            return Err(Error::validation("No file path provided"));
        }

        Ok(args)
    }
}
