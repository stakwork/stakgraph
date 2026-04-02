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
use output::{write_json_error, Output, OutputMode};

#[tokio::main]
async fn main() {
    let cli = match CliArgs::parse_and_expand() {
        Ok(cli) => cli,
        Err(e) => {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        }
    };
    let output_mode = OutputMode::from_json_flag(cli.json);
    let command_name = command_name(&cli);
    init_logging(&cli);

    if let Err(e) = run(cli).await {
        if let Error::Io(io_err) = &e {
            if io_err.kind() == ErrorKind::BrokenPipe {
                std::process::exit(0);
            }
        }
        if output_mode.is_json() {
            let mut out = Output::new();
            if write_json_error(&mut out, command_name, e.to_string()).is_err() {
                eprintln!("Error: {}", e);
            }
        } else {
            eprintln!("Error: {}", e);
        }
        std::process::exit(1);
    }
}

fn command_name(cli: &CliArgs) -> &'static str {
    match &cli.command {
        Some(Commands::Completions(_)) => "completions",
        Some(Commands::Changes(_)) => "changes",
        Some(Commands::Deps(_)) => "deps",
        None => "parse",
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

async fn run(cli: CliArgs) -> Result<()> {
    let output_mode = OutputMode::from_json_flag(cli.json);
    match &cli.command {
        Some(Commands::Completions(args)) => {
            if output_mode.is_json() {
                return Err(Error::validation(
                    "--json is not yet supported for completions",
                ));
            }
            completions::run(args);
            Ok(())
        }
        Some(Commands::Changes(args)) => {
            changes::run(args, &mut Output::new(), cli.verbose || cli.perf, output_mode).await
        }
        Some(Commands::Deps(args)) => {
            deps::run(args, &mut Output::new(), cli.verbose || cli.perf, output_mode).await
        }
        None => parse::run(&cli, &mut Output::new(), output_mode).await,
    }
}
