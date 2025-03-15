use crate::lang::Lang;
use crate::utils::logger;
use lsp::Language;
use std::str::FromStr;

pub mod go;
pub mod python;
pub mod react_ts;
pub mod test_backend;
pub mod utils;

#[tokio::test]
async fn run_server_tests() {
    logger();
    let implemented_servers = ["go", "python", "ruby"];

    for server in implemented_servers.iter() {
        let repo = Some(server.to_string());
        let language = Lang::from_language(Language::from_str(server).unwrap());

        let tester = test_backend::BackendTester::new(language, repo)
            .await
            .unwrap();
        tester.test_backend().unwrap();
    }
}
