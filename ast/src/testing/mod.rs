use crate::lang::Lang;
use lsp::Language;
use std::str::FromStr;
use test_log::test;

pub mod go;
pub mod kotlin;
pub mod python;
pub mod react_ts;
pub mod test_backend;
pub mod test_frontend;
pub mod utils;

#[test(tokio::test)]
async fn run_server_tests() {
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
    let implemented_clients = ["react"];
    for server in implemented_clients.iter() {
        let repo = Some(server.to_string() + "_ts");
        let language = Lang::from_language(Language::from_str(server).unwrap());

        let tester = test_frontend::FrontendTester::new(language, repo)
            .await
            .unwrap();
        tester.test_frontend().unwrap();
    }
}
