use crate::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Language {
    TypeScript,
    Python,
    Rust,
    Java,
    Go,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TestFramework{
    // JavaScript/TypeScript
    Jest,
    Vitest,
    Mocha,
    Jasmine,
    Playwright,
    Cypress,

    // Rust
    Cargo,

    // Python
    Pytest,
    Unittest,

    // Java
    JUnit,
    TestNG,

    // Go
    GoTest,
}


#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PackageManager {
    Npm,
    Yarn,
    Pnpm,
    Cargo,
    Pip,
    Poetry,
    Maven,
    Gradle,
    GoMod,
}

#[derive(Debug, Clone)]
pub struct ProjectConfig {
    pub language: Language,
    pub package_manager: Option<PackageManager>,
    pub test_frameworks: Vec<TestFramework>,
    pub config_files: Vec<PathBuf>,
}

#[derive(Debug, Clone)]
pub struct CoverageStrategy {
    pub framework: TestFramework,
    pub command: String,
    pub args: Vec<String>,
    pub working_dir: PathBuf,
    pub output_dir: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoverageReport {
    pub language: Language,
    pub framework: TestFramework,
    pub summary: CoverageSummary,
    pub files: HashMap<String, FileCoverage>,
    pub timestamp: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoverageSummary {
    pub lines: CoverageMetric,
    pub functions: CoverageMetric,
    pub statements: CoverageMetric,
    pub branches: CoverageMetric,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoverageMetric {
    pub covered: u32,
    pub total: u32,
    pub percentage: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileCoverage {
    pub path: String,
    pub summary: CoverageSummary,
    pub lines: HashMap<u32, u32>, // (line_number, hit_count)
}

pub trait LanguageDetector {
    fn detect(&self, repo_path: &Path) -> Vec<Language>;
}


pub trait ProjectAnalyzer {
    fn analyze(&self, repo_path: &Path, language: &Language) -> Result<ProjectConfig>;
}

pub trait CoverageRunner {
    fn run_coverage(&self, repo_path: &Path, strategy: &CoverageStrategy) -> Result<()>;
}

pub trait ReportParser {
    fn parse_report(&self, repo_path: &Path, config: &ProjectConfig) -> Result<CoverageReport>;
}

pub trait LanguageProvider: Send + Sync {
    fn language(&self) -> Language;
    fn detector(&self) -> Box<dyn LanguageDetector>;
    fn analyzer(&self) -> Box<dyn ProjectAnalyzer>;
    fn runner(&self) -> Box<dyn CoverageRunner>;
    fn parser(&self) -> Box<dyn ReportParser>;
}