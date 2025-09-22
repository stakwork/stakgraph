use anyhow::Result;
use ast::utils::{logger, print_json};
use ast::{self, repo::Repo};

/*
REV=a858f95a6e89b0ff8e7f1797685b73e126e4c5ae URL=https://github.com/stakwork/sphinx-tribes.git cargo run --example url
*/

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<()> {
    logger();
    let url = std::env::var("URL").expect("URL is not set");
    let use_lsp = std::env::var("USE_LSP").ok().map(|v| v == "true");
    let rev = env_not_empty("REV");
    let revs = rev
        .map(|r| r.split(',').map(|s| s.to_string()).collect())
        .unwrap_or_default();

    let repos =
        Repo::new_clone_multi_detect(&url, None, None, Vec::new(), revs, None, None, use_lsp)
            .await?;
    let graph = repos.build_graphs().await?;
    println!(
        "Final Graph => {} nodes and {} edges",
        graph.nodes.len(),
        graph.edges.len()
    );
    print_json(&graph, "url")?;
    Ok(())
}

fn env_not_empty(name: &str) -> Option<String> {
    // return None if it doesn't exist or is empty string
    std::env::var(name).ok().filter(|v| !v.is_empty())
}
