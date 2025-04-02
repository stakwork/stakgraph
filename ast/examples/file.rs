use anyhow::Result;
use ast::lang::ArrayGraph;
use ast::utils::logger;
use ast::Lang;
use ast::{self, repo::Repo};
use lsp::language;
/*
FILE=routes.rb cargo run --example file
FILE=db/schema.rb cargo run --example file
FILE=controllers/profiles_controller.rb cargo run --example file
*/

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<()> {
    logger();

    let filename = std::env::var("FILE").expect("FILE is not set");
    let p = std::path::Path::new(&filename);
    let stem = p.file_stem().expect("file_stem").to_str().expect("to_str");
    let ext = p.extension().expect("extension").to_str().expect("to_str");
    let mut lang = None;
    for l in language::PROGRAMMING_LANGUAGES {
        if l.exts().contains(&ext) {
            lang = Some(l);
            break;
        }
    }
    let lang = lang.expect("no lang");
    let files_filter = vec![filename.to_string()];
    let repo = Repo::new(
        "ast/examples/files",
        Lang::from_language(lang),
        false,
        files_filter,
        Vec::new(),
    )?;

    let graph = repo.build_graph::<ArrayGraph>().await?;
    println!(
        "Final Graph => {} nodes and {} edges",
        graph.nodes.len(),
        graph.edges.len()
    );
    let pretty = serde_json::to_string_pretty(&graph)?;

    let final_path = format!("ast/examples/files/{stem}.json");
    std::fs::write(final_path, pretty)?;
    Ok(())
}
