use anyhow::Result;
use ast::utils::{logger, print_json};
use ast::{self, lang::Lang, repo::Repo, repo::Repos};
use std::str::FromStr;

// OUTPUT_FORMAT=jsonl cargo run --example tribes

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<()> {
    logger();

    let repo1 = Repo::new(
        "ast/examples/sphinx-tribes",
        Lang::from_str("go")?,
        true,
        Vec::new(),
        Vec::new(),
    )?;
    let repo2 = Repo::new(
        "ast/examples/sphinx-tribes-frontend",
        Lang::from_str("ts")?,
        true,
        Vec::new(),
        Vec::new(),
    )?;
    println!("building graph...");
    let repos = Repos(vec![repo1, repo2]);
    let graph = repos.build_graphs().await?;
    print_json(&graph, "tribes")?;
    Ok(())
}
