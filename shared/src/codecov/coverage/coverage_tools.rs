use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CoverageTool {
    C8,
    Istanbul,
    Vitest,
    CoveragePy,
    Tarpaulin,
    JaCoCo,
}

impl CoverageTool {
    pub fn for_language(language: &str) -> Vec<CoverageTool> {
        match language {
            "typescript" | "javascript" => {
                vec![
                    CoverageTool::Vitest,
                    CoverageTool::C8,
                    CoverageTool::Istanbul,
                ]
            }
            "python" => vec![CoverageTool::CoveragePy],
            "rust" => vec![CoverageTool::Tarpaulin],
            "java" => vec![CoverageTool::JaCoCo],
            _ => vec![],
        }
    }

    pub fn check_availability(&self, repo_path: &Path) -> bool {
        match self {
            CoverageTool::C8 => std::process::Command::new("npx")
                .args(["c8", "--version"])
                .current_dir(repo_path)
                .output()
                .map(|o| o.status.success())
                .unwrap_or(false),
            CoverageTool::Vitest => std::process::Command::new("npx")
                .args(["vitest", "--version"])
                .current_dir(repo_path)
                .output()
                .map(|o| o.status.success())
                .unwrap_or(false),
            _ => false,
        }
    }

    pub fn install_command(&self) -> Option<(&'static str, Vec<&'static str>)> {
        match self {
            CoverageTool::C8 => Some(("npm", vec!["install", "--save-dev", "c8"])),
            CoverageTool::Vitest => {
                Some(("npm", vec!["install", "--save-dev", "@vitest/coverage-v8"]))
            }
            CoverageTool::CoveragePy => Some(("pip", vec!["install", "coverage"])),
            CoverageTool::Tarpaulin => Some(("cargo", vec!["install", "cargo-tarpaulin"])),
            _ => None,
        }
    }

    pub fn coverage_command<'a>(&self, target: &'a str) -> Vec<&'a str> {
        match self {
            CoverageTool::C8 => vec![
                "c8",
                "--reporter=json-summary",
                "--reporter=json",
                "--reports-dir=./coverage",
                target,
            ],
            CoverageTool::Vitest => vec!["vitest", "--coverage"],
            CoverageTool::CoveragePy => vec!["coverage", "run", target],
            CoverageTool::Tarpaulin => vec!["cargo", "tarpaulin", "--out", "Json"],
            _ => vec![],
        }
    }
}
