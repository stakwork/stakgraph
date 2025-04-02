use anyhow::Result;
use ast::utils::logger;
use ast::{
    self,
    lang::{ArrayGraph, Lang},
    repo::Repo,
};
use std::str::FromStr;

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<()> {
    logger();

    let language = std::env::var("LANG").unwrap_or("react".to_string());
    let files_filter = match std::env::var("FILES_FILTER") {
        Ok(filter) => filter.split(',').map(|s| s.to_string()).collect(),
        Err(_) => Vec::new(),
    };

    println!("minimal example for {}:", language);
    let lang = Lang::from_str(&language)?;
    let repo = Repo::new("ast/examples/minimal", lang, true, files_filter, Vec::new())?;
    println!("building graph...");
    let graph = repo.build_graph::<ArrayGraph>().await?;
    let pretty = serde_json::to_string_pretty(&graph)?;
    let final_path = format!("ast/examples/minimal/{}.json", language);
    std::fs::write(final_path, pretty)?;
    Ok(())
}
