use crate::testing::annotations;
use crate::utils::get_use_lsp;
use crate::{
    lang::Lang,
    repo::{Repo, Repos},
};
use crate::lang::Graph;
use lsp::Language;
use shared::error::Result;
use std::path::Path;
use std::str::FromStr;

pub async fn test_nextjs_generic<G: Graph + Sync>() -> Result<()> {
    let use_lsp = get_use_lsp();
    let repo = Repo::new(
        "src/testing/nextjs",
        Lang::from_str("tsx").unwrap(),
        use_lsp,
        Vec::new(),
        Vec::new(),
    )
    .unwrap();

    let repos = Repos(vec![repo]);
    let graph = repos.build_graphs_inner::<G>().await?;

    graph.analysis();

    let fixture_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/testing/nextjs");
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
async fn test_nextjs() {
    #[cfg(not(feature = "neo4j"))]
    {
        use crate::lang::graphs::{ArrayGraph, BTreeMapGraph};
        test_nextjs_generic::<ArrayGraph>().await.unwrap();
        test_nextjs_generic::<BTreeMapGraph>().await.unwrap();
    }

    #[cfg(feature = "neo4j")]
    {
        use crate::lang::graphs::Neo4jGraph;
        let graph = Neo4jGraph::default();
        graph.clear().await.unwrap();
        test_nextjs_generic::<Neo4jGraph>().await.unwrap();
    }
}
