pub mod typescript;
pub mod rust;
pub mod python;
pub mod java;
pub mod go;

use crate::coverage::types::{Language, ProjectConfig, ReportParser, CoverageReport};
use crate::Result;
use std::path::Path;

pub struct DefaultReportParser;

impl ReportParser for DefaultReportParser {
    fn parse_report(&self, repo_path: &Path, config: &ProjectConfig) -> Result<CoverageReport> {
        match config.language {
            Language::TypeScript => {
                typescript::TypeScriptParser::new().parse_report(repo_path, config)
            }
            Language::Rust => unimplemented!(),
            Language::Python => unimplemented!(),
            Language::Java => unimplemented!(),
            Language::Go => unimplemented!(),
        }
    }
}


impl DefaultReportParser {
    pub fn new() -> Self {
        Self
    }
}