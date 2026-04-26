use crate::lang::{ArrayGraph, BTreeMapGraph, Lang};
use lsp::Language;
use std::env;
use std::str::FromStr;
// use tracing_test::traced_test;

pub mod angular;
pub mod annotations;
pub mod bash_toml;

#[cfg(test)]
pub mod coverage;
pub mod csharp;
pub mod graphs;
pub mod kotlin;
#[cfg(test)]
pub mod monorepo;
pub mod php;
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

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_go() {
    #[cfg(not(feature = "neo4j"))]
    {
        annotations::run_fixture_test::<ArrayGraph>("src/testing/go", "go", Language::Go).await.unwrap();
        annotations::run_fixture_test::<BTreeMapGraph>("src/testing/go", "go", Language::Go).await.unwrap();
    }
    #[cfg(feature = "neo4j")]
    {
        use crate::lang::graphs::Neo4jGraph;
        let graph = Neo4jGraph::default();
        graph.clear().await.unwrap();
        annotations::run_fixture_test::<Neo4jGraph>("src/testing/go", "go", Language::Go).await.unwrap();
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_go_non_web() {
    #[cfg(not(feature = "neo4j"))]
    {
        annotations::run_fixture_test::<ArrayGraph>("src/testing/go_non_web", "go", Language::Go).await.unwrap();
        annotations::run_fixture_test::<BTreeMapGraph>("src/testing/go_non_web", "go", Language::Go).await.unwrap();
    }
    #[cfg(feature = "neo4j")]
    {
        use crate::lang::graphs::Neo4jGraph;
        let graph = Neo4jGraph::default();
        graph.clear().await.unwrap();
        annotations::run_fixture_test::<Neo4jGraph>("src/testing/go_non_web", "go", Language::Go).await.unwrap();
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_rust() {
    #[cfg(not(feature = "neo4j"))]
    {
        annotations::run_fixture_test::<ArrayGraph>("src/testing/rust", "rust", Language::Rust).await.unwrap();
        annotations::run_fixture_test::<BTreeMapGraph>("src/testing/rust", "rust", Language::Rust).await.unwrap();
    }
    #[cfg(feature = "neo4j")]
    {
        use crate::lang::graphs::Neo4jGraph;
        let graph = Neo4jGraph::default();
        graph.clear().await.unwrap();
        annotations::run_fixture_test::<Neo4jGraph>("src/testing/rust", "rust", Language::Rust).await.unwrap();
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_python_web() {
    #[cfg(not(feature = "neo4j"))]
    {
        annotations::run_fixture_test::<ArrayGraph>("src/testing/python/web", "python", Language::Python).await.unwrap();
        annotations::run_fixture_test::<BTreeMapGraph>("src/testing/python/web", "python", Language::Python).await.unwrap();
    }
    #[cfg(feature = "neo4j")]
    {
        use crate::lang::graphs::Neo4jGraph;
        let graph = Neo4jGraph::default();
        graph.clear().await.unwrap();
        annotations::run_fixture_test::<Neo4jGraph>("src/testing/python/web", "python", Language::Python).await.unwrap();
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_python_data_science() {
    #[cfg(not(feature = "neo4j"))]
    {
        annotations::run_fixture_test::<ArrayGraph>("src/testing/python/data_science", "python", Language::Python).await.unwrap();
        annotations::run_fixture_test::<BTreeMapGraph>("src/testing/python/data_science", "python", Language::Python).await.unwrap();
    }
    #[cfg(feature = "neo4j")]
    {
        use crate::lang::graphs::Neo4jGraph;
        let graph = Neo4jGraph::default();
        graph.clear().await.unwrap();
        annotations::run_fixture_test::<Neo4jGraph>("src/testing/python/data_science", "python", Language::Python).await.unwrap();
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_python_cli() {
    #[cfg(not(feature = "neo4j"))]
    {
        annotations::run_fixture_test::<ArrayGraph>("src/testing/python/cli", "python", Language::Python).await.unwrap();
        annotations::run_fixture_test::<BTreeMapGraph>("src/testing/python/cli", "python", Language::Python).await.unwrap();
    }
    #[cfg(feature = "neo4j")]
    {
        use crate::lang::graphs::Neo4jGraph;
        let graph = Neo4jGraph::default();
        graph.clear().await.unwrap();
        annotations::run_fixture_test::<Neo4jGraph>("src/testing/python/cli", "python", Language::Python).await.unwrap();
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_ruby() {
    #[cfg(not(feature = "neo4j"))]
    {
        annotations::run_fixture_test::<ArrayGraph>("src/testing/ruby", "ruby", Language::Ruby).await.unwrap();
        annotations::run_fixture_test::<BTreeMapGraph>("src/testing/ruby", "ruby", Language::Ruby).await.unwrap();
    }
    #[cfg(feature = "neo4j")]
    {
        use crate::lang::graphs::Neo4jGraph;
        let graph = Neo4jGraph::default();
        graph.clear().await.unwrap();
        annotations::run_fixture_test::<Neo4jGraph>("src/testing/ruby", "ruby", Language::Ruby).await.unwrap();
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_c() {
    #[cfg(not(feature = "neo4j"))]
    {
        annotations::run_fixture_test::<ArrayGraph>("src/testing/c", "c", Language::C).await.unwrap();
        annotations::run_fixture_test::<BTreeMapGraph>("src/testing/c", "c", Language::C).await.unwrap();
    }
    #[cfg(feature = "neo4j")]
    {
        use crate::lang::graphs::Neo4jGraph;
        let graph = Neo4jGraph::default();
        graph.clear().await.unwrap();
        annotations::run_fixture_test::<Neo4jGraph>("src/testing/c", "c", Language::C).await.unwrap();
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_cpp_web_api() {
    #[cfg(not(feature = "neo4j"))]
    {
        annotations::run_fixture_test::<ArrayGraph>("src/testing/cpp/web_api", "cpp", Language::Cpp).await.unwrap();
        annotations::run_fixture_test::<BTreeMapGraph>("src/testing/cpp/web_api", "cpp", Language::Cpp).await.unwrap();
    }
    #[cfg(feature = "neo4j")]
    {
        use crate::lang::graphs::Neo4jGraph;
        let graph = Neo4jGraph::default();
        graph.clear().await.unwrap();
        annotations::run_fixture_test::<Neo4jGraph>("src/testing/cpp/web_api", "cpp", Language::Cpp).await.unwrap();
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_cpp_cuda() {
    #[cfg(not(feature = "neo4j"))]
    {
        annotations::run_fixture_test::<ArrayGraph>("src/testing/cpp/cuda", "cpp", Language::Cpp).await.unwrap();
        annotations::run_fixture_test::<BTreeMapGraph>("src/testing/cpp/cuda", "cpp", Language::Cpp).await.unwrap();
    }
    #[cfg(feature = "neo4j")]
    {
        use crate::lang::graphs::Neo4jGraph;
        let graph = Neo4jGraph::default();
        graph.clear().await.unwrap();
        annotations::run_fixture_test::<Neo4jGraph>("src/testing/cpp/cuda", "cpp", Language::Cpp).await.unwrap();
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_java() {
    #[cfg(not(feature = "neo4j"))]
    {
        annotations::run_fixture_test::<ArrayGraph>("src/testing/java", "java", Language::Java).await.unwrap();
        annotations::run_fixture_test::<BTreeMapGraph>("src/testing/java", "java", Language::Java).await.unwrap();
    }
    #[cfg(feature = "neo4j")]
    {
        use crate::lang::graphs::Neo4jGraph;
        let graph = Neo4jGraph::default();
        graph.clear().await.unwrap();
        annotations::run_fixture_test::<Neo4jGraph>("src/testing/java", "java", Language::Java).await.unwrap();
    }
}
