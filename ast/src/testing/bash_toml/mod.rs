use crate::lang::graphs::{EdgeType, NodeType};
use crate::lang::{Graph, Node};
use crate::{lang::Lang, repo::Repo};
use shared::error::Result;
use std::str::FromStr;

pub async fn test_bash_generic<G: Graph + Sync>() -> Result<()> {
    let repo = Repo::new(
        "src/testing/bash_toml/bash",
        Lang::from_str("bash").unwrap(),
        false,
        Vec::new(),
        Vec::new(),
    )
    .unwrap();

    let graph = repo.build_graph_inner::<G>().await?;

    let language_nodes = graph.find_nodes_by_type(NodeType::Language);
    assert_eq!(language_nodes.len(), 1, "Expected 1 language node");
    assert_eq!(language_nodes[0].name, "bash", "Language should be bash");

    let file_nodes = graph.find_nodes_by_type(NodeType::File);
    assert_eq!(file_nodes.len(), 1, "Expected 1 file node");

    let functions = graph.find_nodes_by_type(NodeType::Function);
    assert_eq!(functions.len(), 2, "Expected 2 function nodes");
    assert!(
        functions.iter().any(|f| f.name == "fetch_user"),
        "Expected fetch_user function"
    );
    assert!(
        functions.iter().any(|f| f.name == "run"),
        "Expected run function"
    );

    let run_fn = functions
        .iter()
        .find(|f| f.name == "run")
        .map(|f| Node::new(NodeType::Function, f.clone()))
        .expect("run function node not found");
    let fetch_user_fn = functions
        .iter()
        .find(|f| f.name == "fetch_user")
        .map(|f| Node::new(NodeType::Function, f.clone()))
        .expect("fetch_user function node not found");

    assert!(
        graph.has_edge(&run_fn, &fetch_user_fn, EdgeType::Calls),
        "Expected run to call fetch_user"
    );

    Ok(())
}

pub async fn test_toml_generic<G: Graph + Sync>() -> Result<()> {
    let repo = Repo::new(
        "src/testing/bash_toml/toml",
        Lang::from_str("toml").unwrap(),
        false,
        Vec::new(),
        Vec::new(),
    )
    .unwrap();

    let graph = repo.build_graph_inner::<G>().await?;

    let language_nodes = graph.find_nodes_by_type(NodeType::Language);
    assert_eq!(language_nodes.len(), 1, "Expected 1 language node");
    assert_eq!(language_nodes[0].name, "toml", "Language should be toml");

    let file_nodes = graph.find_nodes_by_type(NodeType::File);
    assert_eq!(file_nodes.len(), 1, "Expected 1 file node");

    let libraries = graph.find_nodes_by_type(NodeType::Library);
    assert!(
        libraries.iter().any(|l| l.name == "serde"),
        "Expected serde dependency"
    );
    assert!(
        libraries.iter().any(|l| l.name == "tokio"),
        "Expected tokio dependency"
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lang::graphs::{ArrayGraph, BTreeMapGraph};

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_bash() {
        test_bash_generic::<ArrayGraph>().await.unwrap();
        test_bash_generic::<BTreeMapGraph>().await.unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_toml() {
        test_toml_generic::<ArrayGraph>().await.unwrap();
        test_toml_generic::<BTreeMapGraph>().await.unwrap();
    }
}
