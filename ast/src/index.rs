use ast::lang::Graph;
use ast::repo::Repo;
use ast::utils::{logger, print_json};
use shared::{Context, Error, Result};
use std::env;

/*

export REPO_URL="https://github.com/stakwork/sphinx-tribes.git,https://github.com/stakwork/sphinx-tribes-frontend.git"
export OUTPUT_NAME=tribes
cargo run --bin index

export REPO_URL="https://github.com/stakwork/sphinx-tribes.git"
export OUTPUT_NAME=tribes
cargo run --bin index

export REPO_URL="https://github.com/stakwork/demo-repo.git"

export REPO_URL="https://github.com/stakwork/sphinx-tribes-frontend.git"

export REPO_URL="https://github.com/stakwork/sphinx-tribes.git"

export REPO_PATH=/Users/evanfeenstra/code/sphinx2/tribes-workspace/sphinx-tribes

export REPO_PATH=/Users/evanfeenstra/code/sphinx2/stakgraph/ast/examples/senza-lnd

export REPO_URL=https://github.com/stakwork/sphinx-ios-v2

*/

#[tokio::main]
async fn main() -> Result<()> {
    logger();

    let repo_path = env::var("REPO_PATH").ok();
    let repo_urls = env::var("REPO_URL").ok();
    let use_lsp = env::var("USE_LSP").ok().map(|v| v == "true");
    if repo_path.is_none() && repo_urls.is_none() {
        return Err(Error::Custom("no REPO_PATH or REPO_URL".into()));
    }
    let rev = env_not_empty("REV");
    let revs: Vec<String> = rev
        .map(|r| r.split(',').map(|s| s.to_string()).collect())
        .unwrap_or_default();

    let repos = if let Some(repo_path) = &repo_path {
        Repo::new_multi_detect(repo_path, None, Vec::new(), revs.clone(), use_lsp).await?
    } else {
        let username = env_not_empty("USERNAME");
        let pat = env_not_empty("PAT");
        let repo_urls = &repo_urls.clone().context("no REPO_URL")?;
        Repo::new_clone_multi_detect(
            repo_urls,
            username.clone(),
            pat.clone(),
            Vec::new(),
            revs.clone(),
            None,
            None,
            use_lsp,
        )
        .await?
    };

    let name = env::var("OUTPUT_NAME").unwrap_or_else(|_| {
        repo_urls
            .unwrap_or_else(|| repo_path.context("no REPO_PATH").unwrap())
            .split('/')
            .next_back()
            .unwrap()
            .trim_end_matches(".git")
            .to_string()
    });
    println!("{}", name);

    //let graph = repos.build_graphs_btree().await?;
    let graph = repos.build_graphs_local().await?;

    if std::env::var("OUTPUT_FORMAT")
        .unwrap_or_else(|_| "jsonl".to_string())
        .as_str()
        == "jsonl"
    {
        println!("writing to ast/examples/{}-nodes.jsonl", &name);
    }

    print_json(&graph, &name)?;

    graph.analysis();

    Ok(())
}

fn env_not_empty(name: &str) -> Option<String> {
    // return None if it doesn't exist or is empty string
    std::env::var(name).ok().filter(|v| !v.is_empty())
}
