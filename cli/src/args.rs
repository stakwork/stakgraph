use clap::{ArgAction, Args, Parser, Subcommand};
use clap_complete::Shell;
use shared::Result;

#[derive(Debug, Parser)]
#[command(name = "stakgraph")]
#[command(version)]
#[command(about = "Parse files and print a graph-oriented summary")]
pub struct CliArgs {
    #[command(subcommand)]
    pub command: Option<Commands>,

    /// Emit machine-readable JSON instead of the default human output
    #[arg(long, action = ArgAction::SetTrue)]
    pub json: bool,

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

    /// Only emit certain node types, comma-separated (e.g. --type Endpoint,Request)
    #[arg(long, value_delimiter = ',')]
    pub r#type: Vec<String>,

    /// Print only the named node (use with a single file; optional --type to disambiguate)
    #[arg(long)]
    pub name: Option<String>,

    /// Print counts by node type as a summary table
    #[arg(long, action = ArgAction::SetTrue)]
    pub stats: bool,

    /// Token budget for output; activates budget-aware summary mode
    #[arg(long)]
    pub max_tokens: Option<usize>,

    /// Maximum directory tree depth (used with --max-tokens on a directory)
    #[arg(long)]
    pub depth: Option<usize>,

    /// Input files or directories (comma-separated or multiple args)
    #[arg(value_name = "FILE_OR_DIR", num_args = 0..)]
    pub files: Vec<String>,
}

#[derive(Debug, Subcommand)]
pub enum Commands {
    /// Generate shell completions
    Completions(CompletionsArgs),
    /// Explore git changes summaries scoped to specific files or directories
    Changes(ChangesArgs),
    /// Show a dependency tree for a named node
    Deps(DepsArgs),
}

#[derive(Debug, Args)]
pub struct CompletionsArgs {
    /// Shell to generate completions for
    #[arg(value_enum)]
    pub shell: Shell,
}

#[derive(Debug, Args)]
pub struct ChangesArgs {
    #[command(subcommand)]
    pub command: ChangesCommand,
}

#[derive(Debug, Subcommand)]
pub enum ChangesCommand {
    /// List commits that touched the scoped paths
    List(ListArgs),
    /// Compute graph delta between two points (default: working tree vs HEAD)
    Diff(DiffArgs),
}

#[derive(Debug, Args)]
pub struct ListArgs {
    /// Maximum number of commits to show (default: 20)
    #[arg(long, default_value = "20")]
    pub max: usize,

    /// Files or directories to scope the changes to (default: all files)
    #[arg(value_name = "PATH", num_args = 0..)]
    pub paths: Vec<String>,
}

#[derive(Debug, Args)]
pub struct DiffArgs {
    /// Compare staged changes only
    #[arg(long, conflicts_with_all = &["last", "since", "range"])]
    pub staged: bool,

    /// Compare HEAD~n..HEAD
    #[arg(long, conflicts_with_all = &["staged", "since", "range"])]
    pub last: Option<usize>,

    /// Compare <ref>..HEAD
    #[arg(long, conflicts_with_all = &["staged", "last", "range"])]
    pub since: Option<String>,

    /// Compare explicit range <a>..<b>
    #[arg(long, conflicts_with_all = &["staged", "last", "since"])]
    pub range: Option<String>,

    /// Only show nodes of these types, comma-separated (e.g. Function,Endpoint)
    #[arg(long, value_delimiter = ',')]
    pub types: Vec<String>,

    /// Files or directories to scope the changes to (default: all files)
    #[arg(value_name = "PATH", num_args = 0..)]
    pub paths: Vec<String>,
}

#[derive(Debug, Args)]
pub struct DepsArgs {
    /// Name of the function or node to inspect
    #[arg(value_name = "NAME")]
    pub name: String,

    /// Maximum traversal depth (0 = unlimited, default: 3)
    #[arg(long, default_value = "3")]
    pub depth: usize,

    /// Only show nodes of this type (e.g. Function, Class, Endpoint)
    #[arg(long)]
    pub r#type: Option<String>,

    /// Include unverified (cross-file unresolved) calls (default: true)
    #[arg(long, default_value_t = true, action = ArgAction::Set)]
    pub allow: bool,

    /// Files or directories to parse
    #[arg(value_name = "FILE_OR_DIR", num_args = 1..)]
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

        if args.command.is_none() && args.files.is_empty() {
            eprintln!("Error: no file path provided. Run with --help for usage.");
            std::process::exit(1);
        }

        if let Some(Commands::Completions(_)) = &args.command {
            return Ok(args);
        }

        if let Some(Commands::Changes(_)) = &args.command {
            return Ok(args);
        }

        if let Some(Commands::Deps(_)) = &args.command {
            return Ok(args);
        }

        Ok(args)
    }
}
