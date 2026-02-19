use clap::Parser;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct CliArgs {
    /// Optional name to operate on
    #[arg(short, long)]
    name: Option<String>,

    /// Sets a custom config file
    #[arg(short, long, value_name = "FILE")]
    config: Option<PathBuf>,

    /// Turn debugging information on
    #[arg(short, long, action = clap::ArgAction::Count)]
    debug: u8,

    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(clap::Subcommand, Debug)]
pub enum Commands {
    /// adds things
    Add {
        /// stuff to add
        #[arg(required = true)]
        path: Vec<String>,
    },
    /// lists things
    List {
        #[arg(short, long)]
        all: bool,
    },
}
