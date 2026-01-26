use crate::lang::{BTreeMapGraph, Graph, NodeType};
use crate::repo::Repo;
use shared::error::Result;

async fn build_and_detect(relative_path: &str) -> Result<BTreeMapGraph> {
    let repos = Repo::new_multi_detect(
        &format!("src/testing/{}", relative_path),
        None,
        vec![],
        vec![],
        Some(false),
    )
    .await?;
    repos.build_graphs().await
}

fn assert_language(graph: &BTreeMapGraph, expected: &str) -> Result<()> {
    let langs = graph.find_nodes_by_type(NodeType::Language);
    assert!(
        !langs.is_empty(),
        "No language detected in graph"
    );
    assert_eq!(
        langs[0].name, expected,
        "Expected language '{}' but got '{}'",
        expected, langs[0].name
    );
    Ok(())
}

#[tokio::test]
async fn test_vanilla_js() -> Result<()> {
    let graph = build_and_detect("vanila/vanilla_js").await?;
    assert_language(&graph, "typescript")?;
    Ok(())
}

#[tokio::test]
async fn test_python_minimal() -> Result<()> {
    let graph = build_and_detect("vanila/python_minimal").await?;
    assert_language(&graph, "python")?;
    Ok(())
}

#[tokio::test]
async fn test_python_setuppy() -> Result<()> {
    let graph = build_and_detect("vanila/python_setuppy").await?;
    assert_language(&graph, "python")?;
    Ok(())
}

#[tokio::test]
async fn test_typescript_bare() -> Result<()> {
    let graph = build_and_detect("vanila/typescript_bare").await?;
    assert_language(&graph, "typescript")?;
    Ok(())
}

#[tokio::test]
async fn test_go_scripts() -> Result<()> {
    let graph = build_and_detect("vanila/go_scripts").await?;
    assert_language(&graph, "go")?;
    Ok(())
}

#[tokio::test]
async fn test_ruby_minimal() -> Result<()> {
    let graph = build_and_detect("vanila/ruby_minimal").await?;
    assert_language(&graph, "ruby")?;
    Ok(())
}

#[tokio::test]
async fn test_php_minimal() -> Result<()> {
    let graph = build_and_detect("vanila/php_minimal").await?;
    assert_language(&graph, "php")?;
    Ok(())
}
