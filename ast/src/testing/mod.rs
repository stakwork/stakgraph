use crate::lang::Lang;
use lsp::Language;
use std::str::FromStr;
use test_log::test;

pub mod go;
pub mod python;
pub mod react_ts;
pub mod test_backend;
pub mod utils;
pub mod kotlin;
pub mod swift;

#[test(tokio::test)]
async fn run_server_tests() {
    let implemented_servers = ["go", "python", "ruby", "rust"];

    for server in implemented_servers.iter() {
        let repo = Some(server.to_string());
        let language = Lang::from_language(Language::from_str(server).unwrap());

        let tester = test_backend::BackendTester::new(language, repo)
            .await
            .unwrap();
        tester.test_backend().unwrap();
    }
}
