pub mod coverage;
pub mod utils; // full implementation centralized

use crate::Result;
use chrono::Utc;
use coverage::TestCoverage;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use utils::{augment_and_copy_summary, sanitize_repo};

#[derive(Serialize, Clone, Debug, Deserialize)]
pub struct Metric {
    pub total: u64,
    pub covered: u64,
    pub pct: f64,
}

#[derive(Serialize, Clone, Debug, Deserialize)]
pub struct LanguageReport {
    pub language: String,
    pub lines: Option<Metric>,
    pub branches: Option<Metric>,
    pub functions: Option<Metric>,
    pub statements: Option<Metric>,
}

#[derive(Serialize, Clone, Debug, Deserialize)]
pub struct Report {
    pub repo_url: String,
    pub commit: String,
    pub generated_at: String,
    pub languages: Vec<LanguageReport>,
    pub errors: Vec<String>,
}

pub use utils::parse_summary_or_final;

pub fn run(repo_path: &str, repo_url: &str, commit: &str) -> Result<Report> {
    let mut errors = Vec::new();
    let mut languages = Vec::new();
    let mut artifact_sources: Vec<Box<dyn TestCoverage>> = Vec::new();

    for provider in providers() {
        match provider.run(Path::new(repo_path)) {
            Ok(Some(lang_report)) => {
                languages.push(lang_report);
                artifact_sources.push(provider);
            }
            Ok(None) => {}
            Err(e) => errors.push(format!("{}: {}", provider.name(), e)),
        }
    }

    let report = Report {
        repo_url: repo_url.to_string(),
        commit: commit.to_string(),
        generated_at: Utc::now().to_rfc3339(),
        languages,
        errors,
    };

    write_report_and_artifacts(repo_path, &report, artifact_sources)?;
    Ok(report)
}

fn write_report_and_artifacts(
    repo_path: &str,
    report: &Report,
    artifact_sources: Vec<Box<dyn TestCoverage>>,
) -> Result<()> {
    let out_dir = std::env::var("CODECOV_OUTPUT_ROOT")
        .unwrap_or_else(|_| "standalone/coverage_reports".into());
    let repo_key = sanitize_repo(&report.repo_url);
    let dir = Path::new(&out_dir).join(&repo_key);
    fs::create_dir_all(&dir)?;

    let short_commit: String = report.commit.chars().take(12).collect();
    let file = dir.join(format!("{}.json", short_commit));
    let f = fs::File::create(&file)?;
    serde_json::to_writer_pretty(f, &report)?;

    let repo_path_buf = PathBuf::from(repo_path);
    for provider in artifact_sources.into_iter() {
        for ap in provider.artifact_paths(&repo_path_buf) {
            if let Some(name) = ap.file_name().and_then(|s| s.to_str()) {
                let dest = dir.join(format!("{}-{}", short_commit, name));
                if name == "coverage-summary.json" {
                    let _ = augment_and_copy_summary(&repo_path_buf, &ap, &dest);
                } else {
                    let _ = fs::copy(&ap, &dest);
                }
            }
        }
    }
    Ok(())
}

fn providers() -> Vec<Box<dyn TestCoverage>> {
    vec![Box::new(coverage::typescript::TypeScriptCoverage)]
}
