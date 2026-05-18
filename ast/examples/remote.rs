use anyhow::Result;
use ast::utils::{logger, print_json};
use ast::{self, repo::Repo};

/*
cargo run --example remote
URL=https://github.com/some/repo cargo run --example remote
*/

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<()> {
    logger();

    let url = std::env::var("URL")
        .unwrap_or("https://github.com/pypa/sampleproject".to_string());

    let repos =
        Repo::new_clone_multi_detect(&url, None, None, Vec::new(), Vec::new(), None, None, None)
            .await?;
    let graph = repos.build_graphs().await?;

    println!("nodes: {}, edges: {}", graph.nodes.len(), graph.edges.len());
    print_json(&graph, "remote")?;
    Ok(())
}
