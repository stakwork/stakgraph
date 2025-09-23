use anyhow::Result;
use ast::utils::print_json;
use ast::{self};
#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<()> {
    let language = "Python";
    let repo_url = "https://github.com/pypa/sampleproject";

    let repo = ast::repo::Repo::new_clone_to_tmp(
        repo_url,
        Some(language),
        true,
        None,
        None,
        Vec::new(),
        Vec::new(),
        None,
    )
    .await?;
    let graph = repo.build_graph().await?;

    print_json(&graph, "python")?;

    Ok(())
}
