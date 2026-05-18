use anyhow::Result;
use ast::utils::{logger, print_json};
use ast::{self, repo::Repo};

/*
URL=https://github.com/stakwork/sphinx-tribes.git cargo run --example url
URL=https://github.com/pypa/sampleproject.git    cargo run --example url
URL=https://github.com/rails/rails.git           cargo run --example url

# with a specific revision:
URL=https://github.com/stakwork/sphinx-tribes.git REV=<sha> cargo run --example url

# with LSP-based cross-file linking:
USE_LSP=true URL=https://github.com/stakwork/sphinx-tribes.git cargo run --example url
*/

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<()> {
    logger();

    let url = std::env::var("URL").expect("URL env var is required");
    let use_lsp = std::env::var("USE_LSP").ok().map(|v| v == "true");
    let revs = std::env::var("REV")
        .ok()
        .filter(|v| !v.is_empty())
        .map(|r| r.split(',').map(|s| s.to_string()).collect())
        .unwrap_or_default();

    let repos =
        Repo::new_clone_multi_detect(&url, None, None, Vec::new(), revs, None, None, use_lsp)
            .await?;
    let graph = repos.build_graphs().await?;

    println!("nodes: {}, edges: {}", graph.nodes.len(), graph.edges.len());
    print_json(&graph, "url")?;
    Ok(())
}
