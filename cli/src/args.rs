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

    /// Use Neo4j graph for enhanced output (requires --features neo4j build)
    #[arg(long, action = ArgAction::SetTrue)]
    pub neo4j: bool,

    /// Input files or directories (comma-separated or multiple args)
    #[arg(value_name = "FILE_OR_DIR", num_args = 0..)]
    pub files: Vec<String>,
}

#[derive(Debug, Subcommand)]
pub enum Commands {
    /// Print a token-budget-aware high-level summary of a directory
    Summarize(SummarizeArgs),
    /// Generate shell completions
    Completions(CompletionsArgs),
    /// Explore git changes summaries scoped to specific files or directories
    Changes(ChangesArgs),
    /// Query and manage the Neo4j knowledge graph (requires --features neo4j build)
    Graph(GraphArgs),
}

#[derive(Debug, Args)]
pub struct SummarizeArgs {
    /// Token budget for the output (default: 5000)
    #[arg(long, default_value = "5000")]
    pub max_tokens: usize,

    /// Maximum directory depth to display (default: adaptive, starts at 1)
    #[arg(long)]
    pub depth: Option<usize>,

    /// Path to summarize (default: current directory)
    #[arg(value_name = "PATH", default_value = ".")]
    pub path: String,
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
pub struct GraphArgs {
    #[command(subcommand)]
    pub command: GraphCommand,
}

#[derive(Debug, Subcommand)]
pub enum GraphCommand {
    /// Ingest a local repository into Neo4j
    Ingest(GraphIngestArgs),
    /// Search for nodes by name substring
    Search(GraphSearchArgs),
    /// Show details for a named node
    Node(GraphNodeArgs),
    /// Traverse the graph from a node and render an ASCII tree
    Map(GraphMapArgs),
    /// Print all known node and edge types
    Schema,
    /// Clear all nodes and edges from the graph
    Clear,
    /// Show graph size statistics
    Stats,
}

#[derive(Debug, Args)]
pub struct GraphIngestArgs {
    /// Path to the repository to ingest
    #[arg(value_name = "PATH", default_value = ".")]
    pub path: String,
}

#[derive(Debug, Args)]
pub struct GraphSearchArgs {
    /// Name substring to search for
    pub query: String,

    /// Limit to these node types, comma-separated (e.g. Function,Endpoint)
    #[arg(long, value_delimiter = ',')]
    pub node_type: Vec<String>,

    /// Maximum number of results to return
    #[arg(long, default_value = "20")]
    pub limit: usize,
}

#[derive(Debug, Args)]
pub struct GraphNodeArgs {
    /// Exact node name to look up
    pub name: String,

    /// Limit lookup to this node type (e.g. Function, Endpoint)
    #[arg(long)]
    pub node_type: Option<String>,

    /// Filter by file path (exact match) to disambiguate duplicate names
    #[arg(long)]
    pub file: Option<String>,
}

#[derive(Debug, Args)]
pub struct GraphMapArgs {
    /// Name of the node to start traversal from
    pub name: String,

    /// Node type label (e.g. Function, Endpoint, Class)
    #[arg(long)]
    pub node_type: Option<String>,

    /// Traversal direction: down, up, or both
    #[arg(long, default_value = "down")]
    pub direction: String,

    /// Maximum traversal depth
    #[arg(long, default_value_t = 10)]
    pub depth: usize,

    /// Include test nodes in the traversal
    #[arg(long)]
    pub tests: bool,

    /// Node names to exclude from the tree, comma-separated
    #[arg(long, value_delimiter = ',')]
    pub trim: Vec<String>,
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

        if let Some(Commands::Graph(_)) = &args.command {
            return Ok(args);
        }

        Ok(args)
    }
}
