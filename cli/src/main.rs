use std::io::ErrorKind;

use shared::{Error, Result};
use tracing_subscriber::filter::{EnvFilter, LevelFilter};

mod args;
mod changes;
mod completions;
mod deps;
mod git;
mod output;
mod parse;
mod progress;
mod render;
mod summarize;
mod utils;

use args::{CliArgs, Commands};
use output::Output;

#[tokio::main]
async fn main() {
    if let Err(e) = run().await {
        if let Error::Io(io_err) = &e {
            if io_err.kind() == ErrorKind::BrokenPipe {
                std::process::exit(0);
            }
        }
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}

fn init_logging(cli: &CliArgs) {
    let level = if cli.quiet {
        LevelFilter::ERROR
    } else if cli.perf || cli.verbose {
        LevelFilter::INFO
    } else {
        LevelFilter::WARN
    };

    let filter = EnvFilter::builder()
        .with_default_directive(level.into())
        .from_env_lossy();
    tracing_subscriber::fmt()
        .with_target(false)
        .with_env_filter(filter)
        .init();
}

async fn run() -> Result<()> {
    let cli = CliArgs::parse_and_expand()?;
    init_logging(&cli);

    match &cli.command {
        Some(Commands::Completions(args)) => {
            completions::run(args);
            Ok(())
        }
        Some(Commands::Changes(args)) => {
            changes::run(args, &mut Output::new(), cli.verbose || cli.perf).await
        }
        Some(Commands::Deps(args)) => {
            deps::run(args, &mut Output::new(), cli.verbose || cli.perf).await
        }
        None => parse::run(&cli, &mut Output::new()).await,
    }
}
