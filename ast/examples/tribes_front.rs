use anyhow::Result;
use ast::utils::{logger, print_json};
use ast::{
    self,
    lang::{ArrayGraph, Lang},
    repo::Repo,
};
use std::str::FromStr;

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<()> {
    logger();

    let language = "react";
    let lang = Lang::from_str(language)?;

    let repo = Repo::new(
        "ast/examples/sphinx-tribes-frontend",
        lang,
        true,
        Vec::new(),
        Vec::new(),
    )?;
    println!("building graph...");
    let graph = repo.build_graph::<ArrayGraph>().await?;
    print_json(&graph, "tribes-front")?;
    Ok(())
}
