use crate::lang::{BTreeMapGraph,ArrayGraph, Lang};
use lsp::Language;
use std::env;
use std::str::FromStr;

pub mod annotations;
pub mod bash_toml;

use annotations::{run_fixture_test, run_fixture_test_with_lsp};

#[cfg(test)]
pub mod builder;

#[cfg(test)]
pub mod coverage;

pub mod graphs;

#[cfg(test)]
pub mod monorepo;
pub mod test_backend;
pub mod test_frontend;
#[cfg(test)]
pub mod vanila;

#[cfg(test)]
fn pre_test() {
    unsafe { env::set_var("LSP_SKIP_POST_CLONE", "true") };
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
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
       run_fixture_test::<ArrayGraph>("src/testing/react", "tsx", Language::Typescript).await.unwrap();
        run_fixture_test::<BTreeMapGraph>("src/testing/react", "tsx", Language::Typescript).await.unwrap();
    }
    #[cfg(feature = "neo4j")]
    {
        use crate::{lang::graphs::Neo4jGraph, testing::annotations::run_fixture_test};
        let graph = Neo4jGraph::default();
        graph.clear().await.unwrap();
        run_fixture_test::<Neo4jGraph>("src/testing/react", "tsx", Language::Typescript).await.unwrap();
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_typescript() {
    #[cfg(not(feature = "neo4j"))]
    {
        run_fixture_test::<ArrayGraph>("src/testing/typescript", "ts", Language::Typescript).await.unwrap();
        run_fixture_test::<BTreeMapGraph>("src/testing/typescript", "ts", Language::Typescript).await.unwrap();
    }
    #[cfg(feature = "neo4j")]
    {
        use crate::{lang::graphs::Neo4jGraph, testing::annotations::run_fixture_test};
        let graph = Neo4jGraph::default();
        graph.clear().await.unwrap();
        run_fixture_test::<Neo4jGraph>("src/testing/typescript", "ts", Language::Typescript).await.unwrap();
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_nextjs() {
    #[cfg(not(feature = "neo4j"))]
    {
        run_fixture_test::<ArrayGraph>("src/testing/nextjs", "tsx", Language::Typescript).await.unwrap();
        run_fixture_test::<BTreeMapGraph>("src/testing/nextjs", "tsx", Language::Typescript).await.unwrap();
    }
    #[cfg(feature = "neo4j")]
    {
        use crate::{lang::graphs::Neo4jGraph, testing::annotations::run_fixture_test};
        let graph = Neo4jGraph::default();
        graph.clear().await.unwrap();
        run_fixture_test::<Neo4jGraph>("src/testing/nextjs", "tsx", Language::Typescript).await.unwrap();
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_go() {
    #[cfg(not(feature = "neo4j"))]
    {
        run_fixture_test::<ArrayGraph>("src/testing/go", "go", Language::Go).await.unwrap();
        run_fixture_test::<BTreeMapGraph>("src/testing/go", "go", Language::Go).await.unwrap();
    }
    #[cfg(feature = "neo4j")]
    {
        use crate::{lang::graphs::Neo4jGraph, testing::annotations::run_fixture_test};
        let graph = Neo4jGraph::default();
        graph.clear().await.unwrap();
        run_fixture_test::<Neo4jGraph>("src/testing/go", "go", Language::Go).await.unwrap();
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_go_non_web() {
    #[cfg(not(feature = "neo4j"))]
    {
        run_fixture_test::<ArrayGraph>("src/testing/go_non_web", "go", Language::Go).await.unwrap();
        run_fixture_test::<BTreeMapGraph>("src/testing/go_non_web", "go", Language::Go).await.unwrap();
    }
    #[cfg(feature = "neo4j")]
    {
        use crate::{lang::graphs::Neo4jGraph, testing::annotations::run_fixture_test};
        let graph = Neo4jGraph::default();
        graph.clear().await.unwrap();
        run_fixture_test::<Neo4jGraph>("src/testing/go_non_web", "go", Language::Go).await.unwrap();
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_rust() {
    #[cfg(not(feature = "neo4j"))]
    {
        run_fixture_test::<ArrayGraph>("src/testing/rust", "rust", Language::Rust).await.unwrap();
        run_fixture_test::<BTreeMapGraph>("src/testing/rust", "rust", Language::Rust).await.unwrap();
    }
    #[cfg(feature = "neo4j")]
    {
        use crate::{lang::graphs::Neo4jGraph, testing::annotations::run_fixture_test};
        let graph = Neo4jGraph::default();
        graph.clear().await.unwrap();
        run_fixture_test::<Neo4jGraph>("src/testing/rust", "rust", Language::Rust).await.unwrap();
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_python_web() {
    #[cfg(not(feature = "neo4j"))]
    {
        run_fixture_test::<ArrayGraph>("src/testing/python/web", "python", Language::Python).await.unwrap();
        run_fixture_test::<BTreeMapGraph>("src/testing/python/web", "python", Language::Python).await.unwrap();
    }
    #[cfg(feature = "neo4j")]
    {
        use crate::{lang::graphs::Neo4jGraph, testing::annotations::run_fixture_test};
        let graph = Neo4jGraph::default();
        graph.clear().await.unwrap();
        run_fixture_test::<Neo4jGraph>("src/testing/python/web", "python", Language::Python).await.unwrap();
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_python_data_science() {
    #[cfg(not(feature = "neo4j"))]
    {
        run_fixture_test::<ArrayGraph>("src/testing/python/data_science", "python", Language::Python).await.unwrap();
        run_fixture_test::<BTreeMapGraph>("src/testing/python/data_science", "python", Language::Python).await.unwrap();
    }
    #[cfg(feature = "neo4j")]
    {
        use crate::{lang::graphs::Neo4jGraph, testing::annotations::run_fixture_test};
        let graph = Neo4jGraph::default();
        graph.clear().await.unwrap();
        run_fixture_test::<Neo4jGraph>("src/testing/python/data_science", "python", Language::Python).await.unwrap();
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_python_cli() {
    #[cfg(not(feature = "neo4j"))]
    {
        run_fixture_test::<ArrayGraph>("src/testing/python/cli", "python", Language::Python).await.unwrap();
        run_fixture_test::<BTreeMapGraph>("src/testing/python/cli", "python", Language::Python).await.unwrap();
    }
    #[cfg(feature = "neo4j")]
    {
        use crate::{lang::graphs::Neo4jGraph, testing::annotations::run_fixture_test};
        let graph = Neo4jGraph::default();
        graph.clear().await.unwrap();
        run_fixture_test::<Neo4jGraph>("src/testing/python/cli", "python", Language::Python).await.unwrap();
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_ruby() {
    #[cfg(not(feature = "neo4j"))]
    {
        run_fixture_test::<ArrayGraph>("src/testing/ruby", "ruby", Language::Ruby).await.unwrap();
        run_fixture_test::<BTreeMapGraph>("src/testing/ruby", "ruby", Language::Ruby).await.unwrap();
    }
    #[cfg(feature = "neo4j")]
    {
        use crate::{lang::graphs::Neo4jGraph, testing::annotations::run_fixture_test};
        let graph = Neo4jGraph::default();
        graph.clear().await.unwrap();
        run_fixture_test::<Neo4jGraph>("src/testing/ruby", "ruby", Language::Ruby).await.unwrap();
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_kotlin() {
    let use_lsp = Language::Kotlin.default_do_lsp();

    #[cfg(not(feature = "neo4j"))]
    {
        run_fixture_test_with_lsp::<ArrayGraph>(
            "src/testing/kotlin",
            "kotlin",
            Language::Kotlin,
            use_lsp,
        ).await.unwrap();
        run_fixture_test_with_lsp::<BTreeMapGraph>(
            "src/testing/kotlin",
            "kotlin",
            Language::Kotlin,
            use_lsp,
        ).await.unwrap();
    }
    #[cfg(feature = "neo4j")]
    {
        use crate::{lang::graphs::Neo4jGraph, testing::annotations::run_fixture_test_with_lsp};
        let graph = Neo4jGraph::default();
        graph.clear().await.unwrap();
        run_fixture_test_with_lsp::<Neo4jGraph>(
            "src/testing/kotlin",
            "kotlin",
            Language::Kotlin,
            use_lsp,
        ).await.unwrap();
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_c() {
    #[cfg(not(feature = "neo4j"))]
    {
        run_fixture_test::<ArrayGraph>("src/testing/c", "c", Language::C).await.unwrap();
        run_fixture_test::<BTreeMapGraph>("src/testing/c", "c", Language::C).await.unwrap();
    }
    #[cfg(feature = "neo4j")]
    {
        use crate::{lang::graphs::Neo4jGraph, testing::annotations::run_fixture_test};
        let graph = Neo4jGraph::default();
        graph.clear().await.unwrap();
        run_fixture_test::<Neo4jGraph>("src/testing/c", "c", Language::C).await.unwrap();
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_cpp_web_api() {
    #[cfg(not(feature = "neo4j"))]
    {
        run_fixture_test::<ArrayGraph>("src/testing/cpp/web_api", "cpp", Language::Cpp).await.unwrap();
        run_fixture_test::<BTreeMapGraph>("src/testing/cpp/web_api", "cpp", Language::Cpp).await.unwrap();
    }
    #[cfg(feature = "neo4j")]
    {
        use crate::{lang::graphs::Neo4jGraph, testing::annotations::run_fixture_test};
        let graph = Neo4jGraph::default();
        graph.clear().await.unwrap();
        run_fixture_test::<Neo4jGraph>("src/testing/cpp/web_api", "cpp", Language::Cpp).await.unwrap();
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_cpp_cuda() {
    #[cfg(not(feature = "neo4j"))]
    {
        run_fixture_test::<ArrayGraph>("src/testing/cpp/cuda", "cpp", Language::Cpp).await.unwrap();
        run_fixture_test::<BTreeMapGraph>("src/testing/cpp/cuda", "cpp", Language::Cpp).await.unwrap();
    }
    #[cfg(feature = "neo4j")]
    {
        use crate::{lang::graphs::Neo4jGraph, testing::annotations::run_fixture_test};
        let graph = Neo4jGraph::default();
        graph.clear().await.unwrap();
        run_fixture_test::<Neo4jGraph>("src/testing/cpp/cuda", "cpp", Language::Cpp).await.unwrap();
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_java() {
    #[cfg(not(feature = "neo4j"))]
    {
        run_fixture_test::<ArrayGraph>("src/testing/java", "java", Language::Java).await.unwrap();
        run_fixture_test::<BTreeMapGraph>("src/testing/java", "java", Language::Java).await.unwrap();
    }
    #[cfg(feature = "neo4j")]
    {
        use crate::{lang::graphs::Neo4jGraph, testing::annotations::run_fixture_test};
        let graph = Neo4jGraph::default();
        graph.clear().await.unwrap();
        run_fixture_test::<Neo4jGraph>("src/testing/java", "java", Language::Java).await.unwrap();
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_csharp() {
    #[cfg(not(feature = "neo4j"))]
    {
        run_fixture_test::<ArrayGraph>("src/testing/csharp", "csharp", Language::CSharp).await.unwrap();
        run_fixture_test::<BTreeMapGraph>("src/testing/csharp", "csharp", Language::CSharp).await.unwrap();
    }
    #[cfg(feature = "neo4j")]
    {
       use crate::{lang::graphs::Neo4jGraph, testing::annotations::run_fixture_test};
        let graph = Neo4jGraph::default();
        graph.clear().await.unwrap();
        run_fixture_test::<Neo4jGraph>("src/testing/csharp", "csharp", Language::CSharp).await.unwrap();
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_swift() {
    #[cfg(not(feature = "neo4j"))]
    {
        run_fixture_test::<ArrayGraph>("src/testing/swift/LegacyApp", "swift", Language::Swift).await.unwrap();
        run_fixture_test::<BTreeMapGraph>("src/testing/swift/LegacyApp", "swift", Language::Swift).await.unwrap();
        run_fixture_test::<ArrayGraph>("src/testing/swift/ModernApp", "swift", Language::Swift).await.unwrap();
        run_fixture_test::<BTreeMapGraph>("src/testing/swift/ModernApp", "swift", Language::Swift).await.unwrap();
    }
    #[cfg(feature = "neo4j")]
    {
        use crate::{lang::graphs::Neo4jGraph, testing::annotations::run_fixture_test};
        let graph = Neo4jGraph::default();
        graph.clear().await.unwrap();
        run_fixture_test::<Neo4jGraph>("src/testing/swift/LegacyApp", "swift", Language::Swift).await.unwrap();
        graph.clear().await.unwrap();
        run_fixture_test::<Neo4jGraph>("src/testing/swift/ModernApp", "swift", Language::Swift).await.unwrap();
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_angular() {
    #[cfg(not(feature = "neo4j"))]
    {
        run_fixture_test::<ArrayGraph>("src/testing/angular", "angular", Language::Angular).await.unwrap();
        run_fixture_test::<BTreeMapGraph>("src/testing/angular", "angular", Language::Angular).await.unwrap();
    }
    #[cfg(feature = "neo4j")]
    {
        use crate::{lang::graphs::Neo4jGraph, testing::annotations::run_fixture_test};
        let graph = Neo4jGraph::default();
        graph.clear().await.unwrap();
        run_fixture_test::<Neo4jGraph>("src/testing/angular", "angular", Language::Angular).await.unwrap();
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_svelte() {
    #[cfg(not(feature = "neo4j"))]
    {
        run_fixture_test::<ArrayGraph>("src/testing/svelte", "svelte", Language::Svelte).await.unwrap();
        run_fixture_test::<BTreeMapGraph>("src/testing/svelte", "svelte", Language::Svelte).await.unwrap();
    }
    #[cfg(feature = "neo4j")]
    {
        use crate::{lang::graphs::Neo4jGraph, testing::annotations::run_fixture_test};
        let graph = Neo4jGraph::default();
        graph.clear().await.unwrap();
        run_fixture_test::<Neo4jGraph>("src/testing/svelte", "svelte", Language::Svelte).await.unwrap();
    }
}
