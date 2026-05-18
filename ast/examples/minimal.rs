use anyhow::Result;
use ast::utils::{logger, print_json};
use ast::{self, lang::Lang, repo::Repo};
use std::str::FromStr;

/*
PARSE_LANG=go     cargo run --example minimal
PARSE_LANG=python cargo run --example minimal
PARSE_LANG=react  cargo run --example minimal   (default)
PARSE_LANG=ruby   cargo run --example minimal
*/

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<()> {
    logger();

    let language = std::env::var("PARSE_LANG").unwrap_or("react".to_string());
    println!("parsing: {language}");

    let lang = Lang::from_str(&language)?;
    let repo = Repo::new("ast/examples/minimal", lang, true, Vec::new(), Vec::new())?;
    let graph = repo.build_graph().await?;

    println!("nodes: {}, edges: {}", graph.nodes.len(), graph.edges.len());
    print_json(&graph, "minimal")?;
    Ok(())
}
