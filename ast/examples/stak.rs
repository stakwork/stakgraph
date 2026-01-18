use anyhow::Result;
use ast::utils::{logger, print_json};
use ast::{
    lang::Lang,
    repo::{Repo, Repos},
};
use std::str::FromStr;

/*
OUTPUT_FORMAT=jsonl cargo run --example stak
*/

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<()> {
    logger();
    let repo1 = Repo::new(
        "ast/examples/senza-lnd",
        Lang::from_str("ts")?,
        true,
        Vec::new(),
        Vec::new(),
    )?;
    let repo2 = Repo::new(
        "ast/examples/senza-lnd",
        Lang::from_str("ruby")?,
        false,
        Vec::new(),
        Vec::new(),
    )?;
    println!("building graph...");

    let repos = Repos(vec![repo1, repo2], Vec::new(), None);
    // let repos = Repos(vec![repo2], Vec::new(), None);
    let graph = repos.build_graphs().await?;;
    print_json(&graph, "stak")?;
    Ok(())
}
