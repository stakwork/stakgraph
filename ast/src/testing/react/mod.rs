use crate::lang::graphs::Graph;
use crate::{lang::Lang, repo::Repo};
use crate::testing::annotations;
use lsp::Language;
use shared::error::Result;
use std::path::Path;
use std::str::FromStr;

pub async fn test_react_typescript_generic<G: Graph + Sync>() -> Result<()> {
    let repo = Repo::new(
        "src/testing/react",
        Lang::from_str("tsx").unwrap(),
        false,
        Vec::new(),
        Vec::new(),
    )
    .unwrap();

    let graph = repo.build_graph_inner::<G>().await?;

    let fixture_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/testing/react");
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let annotation_failures =
        annotations::walk_and_verify(&fixture_dir, root, &graph, &Language::Typescript);
    if !annotation_failures.is_empty() {
        for f in &annotation_failures {
            eprintln!("{}", f);
        }
        panic!("{} annotation verification failure(s)", annotation_failures.len());
    }

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_react_typescript() {
    #[cfg(not(feature = "neo4j"))]
    {
        use crate::lang::graphs::{ArrayGraph, BTreeMapGraph};
        test_react_typescript_generic::<ArrayGraph>().await.unwrap();
        test_react_typescript_generic::<BTreeMapGraph>()
            .await
            .unwrap();
    }

    #[cfg(feature = "neo4j")]
    {
        use crate::lang::graphs::Neo4jGraph;
        let graph = Neo4jGraph::default();
        graph.clear().await.unwrap();
        test_react_typescript_generic::<Neo4jGraph>().await.unwrap();
    }
}
