pub mod utils;
pub mod coverage;

use chrono::Utc;
use serde::Deserialize;
use shared::Result;
use std::fs;
use std::path::Path;
use utils::sanitize_repo;
use coverage::TestCoverage;
use crate::types::Report;

#[derive(Deserialize)]
pub struct CodecovBody {
    pub repo_url: String,
    pub username: Option<String>,
    pub pat: Option<String>,
    pub commit: Option<String>,
}


pub async fn run(body: CodecovBody) -> Result<Report> {
    let repo_path = ast::repo::Repo::get_path_from_url(&body.repo_url)?;
    ast::repo::clone_repo(&body.repo_url, &repo_path, body.username.clone(), body.pat.clone(), body.commit.as_deref()).await?;
    let commit = match lsp::git::get_commit_hash(&repo_path).await { Ok(h) => h, Err(_) => body.commit.clone().unwrap_or_default() };
    let mut errors = Vec::new();
    let mut languages = Vec::new();
    let mut artifact_sources = Vec::new();
    for p in providers() {
        match p.run(Path::new(&repo_path)) {
            Ok(Some(r)) => { artifact_sources.push(p); languages.push(r); },
            Ok(None) => {},
            Err(e) => errors.push(format!("{}: {}", p.name(), e)),
        }
    }
    let report = Report { repo_url: body.repo_url, commit, generated_at: Utc::now().to_rfc3339(), languages, errors };
    let out_dir = std::env::var("CODECOV_OUTPUT_ROOT").unwrap_or_else(|_| "standalone/coverage_reports".into());
    let repo_key = sanitize_repo(&report.repo_url);
    let dir = Path::new(&out_dir).join(&repo_key);
    fs::create_dir_all(&dir)?;
    let short_commit: String = report.commit.chars().take(12).collect();
    let file = dir.join(format!("{}.json", short_commit));
    let f = fs::File::create(&file)?;
    serde_json::to_writer_pretty(f, &report)?;
    for provider in artifact_sources.into_iter() {
        for ap in provider.artifact_paths(Path::new(&repo_path)) { 
            if let Some(name) = ap.file_name().and_then(|s| s.to_str()) {
                 let _ = fs::copy(&ap, dir.join(format!("{}-{}", short_commit, name)));
                 } 
            }
    }
    Ok(report)
}
fn providers() -> Vec<Box<dyn TestCoverage>> { 
    vec![
        Box::new(coverage::typescript::TypeScriptCoverage),
        Box::new(coverage::python::PythonCoverage),
    ] 
}

