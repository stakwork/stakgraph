pub mod typescript;
pub mod rust;
pub mod python;
pub mod java;
pub mod go;

use crate::coverage::types::{Language, ProjectAnalyzer, ProjectConfig};
use crate::Result;
use std::path::Path;

pub struct DefaultProjectAnalyzer;


impl ProjectAnalyzer for DefaultProjectAnalyzer {
    fn analyze(&self, repo_path: &Path, language: &Language) -> Result<ProjectConfig> {
        match language {
            Language::TypeScript => typescript::TypeScriptAnalyzer::new().analyze(repo_path),
            Language::Rust => unimplemented!(),
            Language::Python => unimplemented!(),
            Language::Java => unimplemented!(),
            Language::Go => unimplemented!(),
        }
    }
}

impl DefaultProjectAnalyzer {
    pub fn new() -> Self {
        Self
    }
}