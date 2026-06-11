use crate::builder::utils::process_files;
use crate::lang::graphs::{BTreeMapGraph, NodeType};
use crate::lang::{Graph, Lang};
use crate::repo::Repo;
use std::str::FromStr;

#[test]
fn one_bad_file_does_not_abort_stage() {
    let filez = vec![
        ("good.rs".to_string(), "fn a() {}".to_string()),
        ("bad.rs".to_string(), "corrupt".to_string()),
    ];
    let (oks, failed) = process_files(
        &filez,
        true,
        "test_stage",
        |_| true,
        |(name, code)| {
            if name == "bad.rs" {
                Err(shared::error::Error::Custom("parse error".into()))
            } else {
                Ok(code.len())
            }
        },
    );
    assert_eq!(oks.len(), 1);
    assert_eq!(failed, vec!["bad.rs".to_string()]);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn bad_files_do_not_abort_build() {
    super::pre_test();
    let dir = std::env::temp_dir().join(format!("stakgraph_bad_files_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("good.rs"), "pub fn alpha() -> i32 {\n    42\n}\n").unwrap();
    std::fs::write(dir.join("binary.rs"), [0xFFu8, 0xFE, 0x00, 0x9F, 0x12, 0x80]).unwrap();

    let repo = Repo::new(
        dir.to_str().unwrap(),
        Lang::from_str("rust").unwrap(),
        false,
        Vec::new(),
        Vec::new(),
    )
    .unwrap();
    let graph = repo.build_graph_inner::<BTreeMapGraph>().await.unwrap();

    let funcs = graph.find_nodes_by_name(NodeType::Function, "alpha");
    assert_eq!(funcs.len(), 1, "function from good.rs should be parsed");

    let files = graph.find_nodes_by_name(NodeType::File, "binary.rs");
    assert_eq!(files.len(), 1, "binary.rs should still get a File node");
    assert_eq!(
        files[0].meta.get("skipped").map(|s| s.as_str()),
        Some("unreadable")
    );

    std::fs::remove_dir_all(&dir).ok();
}
