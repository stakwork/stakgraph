use ast::repo::Repo;
use ast::utils::{logger, print_json};
use ast::Lang;
use shared::{ Error, Result};
use std::env;

#[tokio::main]
async fn main() -> Result<()> {
    logger();
    let mut args = std::env::args().skip(1);
    let file_path = args.next().ok_or_else(|| Error::Custom("No file path provided".into()))?;
    if !std::path::Path::new(&file_path).exists() {
        return Err(Error::Custom("File does not exist".into()));
    }

    let language = lsp::Language::from_path(&file_path);
   
    let lang = match language {
        Some(lang) => Lang::from_language(lang),
        None => {
            return Err(Error::Custom(
                "Could not determine language from file extension".into(),
            ))
        }
    };
    let use_lsp = env::var("USE_LSP").ok().map(|v| v == "true");
    let repo = Repo::from_single_file(&file_path, lang, use_lsp)?;
    let graph = repo.build_graph().await?;
    print_json(&graph, "single_file")?;
    Ok(())
}
