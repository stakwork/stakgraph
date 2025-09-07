pub mod typescript;
pub mod rust;
pub mod python;
pub mod java;
pub mod go;

use crate::coverage::types::{CoverageRunner, CoverageStrategy, Language, ProjectConfig};
use crate::{Result, Error};
use std::path::Path;
use std::process::Command;

pub struct DefaultCoverageRunner;

impl CoverageRunner for DefaultCoverageRunner {
    fn run_coverage(&self, _repo_path: &Path, strategy: &CoverageStrategy) -> Result<()> {
        let mut cmd = Command::new(&strategy.command);
        cmd.args(&strategy.args)
           .current_dir(&strategy.working_dir);

        let output = cmd.output()?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(Error::Custom(format!(
                "Coverage command failed: {}", stderr
            )));
        }

        Ok(())
    }
}

impl DefaultCoverageRunner {
    pub fn new() -> Self {
        Self
    }

    /// Build a coverage strategy based on project config
    pub fn build_strategy(&self, repo_path: &Path, config: &ProjectConfig) -> Result<CoverageStrategy> {
        match config.language {
            Language::TypeScript => {
                typescript::TypeScriptRunner::new().build_strategy(repo_path, config)
            }
            Language::Rust => unimplemented!(),
            Language::Python => unimplemented!(),
            Language::Java => unimplemented!(),
            Language::Go => unimplemented!(),
        }
    }
}