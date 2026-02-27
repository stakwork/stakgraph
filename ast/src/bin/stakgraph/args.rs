use clap::{ArgAction, Parser};
use shared::{Error, Result};

#[derive(Debug, Parser)]
#[command(name = "stakgraph")]
#[command(version)]
#[command(about = "Parse files and print a graph-oriented summary")]
pub struct CliArgs {
    /// Include unverified function calls in the graph
    #[arg(long, action = ArgAction::SetTrue)]
    pub allow: bool,

    /// Skip extracting function call relationships
    #[arg(long = "skip-calls", action = ArgAction::SetTrue)]
    pub skip_calls: bool,

    /// Suppress all logs except errors (overrides RUST_LOG)
    #[arg(long, short = 'q', action = ArgAction::SetTrue)]
    pub quiet: bool,

    /// Show info and debug logs (overrides RUST_LOG)
    #[arg(long, short = 'v', action = ArgAction::SetTrue, conflicts_with = "quiet")]
    pub verbose: bool,

    /// Show [perf][memory] logs (implies verbose)
    #[arg(long, action = ArgAction::SetTrue, conflicts_with = "quiet")]
    pub perf: bool,

    #[arg(value_name = "FILE", required = true, num_args = 1..)]
    pub files: Vec<String>,
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

        if args.files.is_empty() {
            return Err(Error::Custom("No file path provided".into()));
        }

        Ok(args)
    }
}
