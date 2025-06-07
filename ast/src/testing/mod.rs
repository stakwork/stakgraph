use crate::lang::{ArrayGraph, Lang};
use lsp::Language;
use std::env;
use std::str::FromStr;
// use tracing_test::traced_test;

pub mod angular;
pub mod cpp;
pub mod go;
pub mod graphs;
pub mod java;
pub mod kotlin;
pub mod python;
pub mod react;
pub mod ruby;
pub mod rust_test;
pub mod svelte;
pub mod swift;
pub mod test_backend;
pub mod test_frontend;
pub mod typescript;
pub mod utils;

#[cfg(test)]
fn pre_test() {
    env::set_var("LSP_SKIP_POST_CLONE", "true");
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
