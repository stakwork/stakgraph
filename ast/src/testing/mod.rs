use crate::lang::{ArrayGraph, Lang, NodeData, BTreeMapGraph};
use lsp::Language;
use std::env;
use std::str::FromStr;
// use tracing_test::traced_test;

pub mod angular;
pub mod annotations;
pub mod bash_toml;

pub mod c;
#[cfg(test)]
pub mod coverage;
pub mod cpp;
pub mod csharp;
pub mod go;
pub mod graphs;
pub mod java;
pub mod kotlin;
#[cfg(test)]
pub mod monorepo;
pub mod php;
pub mod python;
pub mod ruby;
pub mod rust_test;
pub mod svelte;
pub mod swift;
pub mod test_backend;
pub mod test_frontend;
#[cfg(test)]
pub mod vanila;

#[cfg(test)]
fn pre_test() {
    unsafe { env::set_var("LSP_SKIP_POST_CLONE", "true") };
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
//#[traced_test]
async fn run_server_tests() {
    pre_test();
    let implemented_servers = ["go", "python", "ruby", "rust", "typescript", "java"];
    for server in implemented_servers.iter() {
        tracing::info!("Running server tests for {}", server);
        let repo = Some(server.to_string());
        let language = Lang::from_language(Language::from_str(server).unwrap());

        let tester = test_backend::BackendTester::<ArrayGraph>::from_repo(language, repo)
            .await
            .unwrap();
        tester.test_backend().unwrap();
    }
}

// #[test(tokio::test)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn run_client_tests() {
    pre_test();
    let implemented_clients = ["react", "kotlin", "swift"];

    for server in implemented_clients.iter() {
        let repo = Some(server.to_string());
        let language = Lang::from_language(Language::from_str(server).unwrap());

        let tester = test_frontend::FrontendTester::<ArrayGraph>::from_repo(language, repo)
            .await
            .unwrap();
        tester.test_frontend().unwrap();
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_react() {
    #[cfg(not(feature = "neo4j"))]
    {
        annotations::run_fixture_test::<ArrayGraph>("src/testing/react", "tsx", Language::Typescript).await.unwrap();
        annotations::run_fixture_test::<BTreeMapGraph>("src/testing/react", "tsx", Language::Typescript).await.unwrap();
    }
    #[cfg(feature = "neo4j")]
    {
        use crate::lang::graphs::Neo4jGraph;
        let graph = Neo4jGraph::default();
        graph.clear().await.unwrap();
        annotations::run_fixture_test::<Neo4jGraph>("src/testing/react", "tsx", Language::Typescript).await.unwrap();
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_typescript() {
    #[cfg(not(feature = "neo4j"))]
    {
        annotations::run_fixture_test::<ArrayGraph>("src/testing/typescript", "ts", Language::Typescript).await.unwrap();
        annotations::run_fixture_test::<BTreeMapGraph>("src/testing/typescript", "ts", Language::Typescript).await.unwrap();
    }
    #[cfg(feature = "neo4j")]
    {
        use crate::lang::graphs::Neo4jGraph;
        let graph = Neo4jGraph::default();
        graph.clear().await.unwrap();
        annotations::run_fixture_test::<Neo4jGraph>("src/testing/typescript", "ts", Language::Typescript).await.unwrap();
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_nextjs() {
    #[cfg(not(feature = "neo4j"))]
    {
        annotations::run_fixture_test::<ArrayGraph>("src/testing/nextjs", "tsx", Language::Typescript).await.unwrap();
        annotations::run_fixture_test::<BTreeMapGraph>("src/testing/nextjs", "tsx", Language::Typescript).await.unwrap();
    }
    #[cfg(feature = "neo4j")]
    {
        use crate::lang::graphs::Neo4jGraph;
        let graph = Neo4jGraph::default();
        graph.clear().await.unwrap();
        annotations::run_fixture_test::<Neo4jGraph>("src/testing/nextjs", "tsx", Language::Typescript).await.unwrap();
    }
}

pub fn _print_nodes(nodes: Vec<NodeData>) {
    println!(
        "{:#?}",
        nodes
            .iter()
            .map(|n| (n.name.clone(), n.file.clone()))
            .collect::<Vec<(String, String)>>()
    );
}
