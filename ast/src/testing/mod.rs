use crate::lang::Lang;
use lsp::Language;
use std::env;
use std::str::FromStr;
use test_log::test;

pub mod go;
pub mod kotlin;
pub mod python;
pub mod react;
pub mod swift;
pub mod svelte;
pub mod test_backend;
pub mod test_frontend;
pub mod utils;

#[cfg(test)]
fn pre_test() {
    env::set_var("LSP_SKIP_POST_CLONE", "true");
}

#[test(tokio::test)]
async fn run_server_tests() {
    pre_test();
    let implemented_servers = ["go", "python", "ruby", "rust", "typescript"];
    for server in implemented_servers.iter() {
        let repo = Some(server.to_string());
        let language = Lang::from_language(Language::from_str(server).unwrap());

        let tester = test_backend::BackendTester::new(language, repo)
            .await
            .unwrap();
        tester.test_backend().unwrap();
    }
}

#[test(tokio::test)]
async fn run_client_tests() {
    pre_test();
    let implemented_clients = ["react", "kotlin", "swift"];

    for server in implemented_clients.iter() {
        let repo = Some(server.to_string());
        let language = Lang::from_language(Language::from_str(server).unwrap());

        let tester = test_frontend::FrontendTester::new(language, repo)
            .await
            .unwrap();
        tester.test_frontend().unwrap();
    }
}
