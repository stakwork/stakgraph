use crate::types::LanguageReport;
use shared::Result;
use std::path::{Path, PathBuf};

pub mod typescript;
pub mod python;
pub mod package_managers;
pub mod test_runners;
pub mod coverage_tools;
pub mod execution;

pub trait TestCoverage: Send + Sync {
    fn name(&self) -> &'static str;
    fn detect(&self, repo_path: &Path) -> bool; 


    fn needs_install(&self, _repo_path: &Path) -> bool { false }
    fn install(&self, _repo_path: &Path) -> Result<()> { Ok(()) }
    fn prepare(&self, _repo_path: &Path) -> Result<()> { Ok(()) }
    fn has_existing_coverage(&self, _repo_path: &Path) -> bool { false }
    fn execute(&self, _repo_path: &Path) -> Result<()> { Ok(()) }
    fn parse(&self, _repo_path: &Path) -> Result<Option<LanguageReport>> { Ok(None) }
    fn artifact_paths(&self, _repo_path: &Path) -> Vec<PathBuf> { Vec::new() }

    fn run(&self, repo_path: &Path) -> Result<Option<LanguageReport>> {
        if !self.detect(repo_path) { return Ok(None); }
        if self.needs_install(repo_path) { self.install(repo_path)?; }
        self.prepare(repo_path)?;
        if !self.has_existing_coverage(repo_path) { self.execute(repo_path)?; }
        self.parse(repo_path)
    }
}
