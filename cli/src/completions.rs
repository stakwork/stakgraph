use clap::CommandFactory;

use super::args::{CliArgs, CompletionsArgs};

pub fn run(args: &CompletionsArgs) {
    let mut cmd = CliArgs::command();
    clap_complete::generate(args.shell, &mut cmd, "stakgraph", &mut std::io::stdout());
}
