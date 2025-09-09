use crate::coverage::types::{
    CoverageReport, CoverageSummary, CoverageMetric, FileCoverage, 
    Language, TestFramework, ProjectConfig
};
use crate::{Result, Error};
use serde_json::Value;
use std::collections::HashMap;
use std::path::Path;

pub struct TypeScriptParser;

impl TypeScriptParser {
    pub fn new() -> Self {
        Self
    }

    pub fn parse_report(&self, repo_path: &Path, config: &ProjectConfig) -> Result<CoverageReport> {
        let coverage_dir = repo_path.join("coverage");

        let coverage_final = coverage_dir.join("coverage-final.json");
        let coverage_summary = coverage_dir.join("coverage-summary.json");

        if !coverage_final.exists() {
            return Err(Error::Custom(
                "No coverage-final.json found".to_string()
            ));
        }

        let final_content = std::fs::read_to_string(&coverage_final)?;
        let final_json: Value = serde_json::from_str(&final_content)?;

        let summary = if coverage_summary.exists() {
            let summary_content = std::fs::read_to_string(&coverage_summary)?;
            let summary_json: Value = serde_json::from_str(&summary_content)?;
            self.parse_summary_from_summary_json(&summary_json)?
        } else {
            self.calculate_summary_from_final_json(&final_json)?
        };

  
        let files = self.parse_files_from_final_json(&final_json, repo_path)?;

        let framework = config.test_frameworks.first()
            .unwrap_or(&TestFramework::Jest)
            .clone();

        Ok(CoverageReport {
            language: Language::TypeScript,
            framework,
            summary,
            files,
            timestamp: chrono::Utc::now().to_rfc3339(),
        })

    }

    fn parse_summary_from_summary_json(&self, json: &Value) -> Result<CoverageSummary> {
        let total = json.get("total")
            .ok_or_else(|| Error::Custom("No 'total' in summary".to_string()))?;

        Ok(CoverageSummary {
            lines: self.parse_metric(total, "lines")?,
            functions: self.parse_metric(total, "functions")?,
            statements: self.parse_metric(total, "statements")?,
            branches: self.parse_metric(total, "branches")?,
        })
    }

    fn calculate_summary_from_final_json(&self, json: &Value) -> Result<CoverageSummary> {
        let mut total_lines = CoverageMetric { covered: 0, total: 0, percentage: 0.0 };
        let mut total_functions = CoverageMetric { covered: 0, total: 0, percentage: 0.0 };
        let mut total_statements = CoverageMetric { covered: 0, total: 0, percentage: 0.0 };
        let mut total_branches = CoverageMetric { covered: 0, total: 0, percentage: 0.0 };


        if let Some(obj) = json.as_object() {
            for (_, file_data) in obj {
                if let Ok(lines) = self.parse_metric(file_data, "lines") {
                    total_lines.covered += lines.covered;
                    total_lines.total += lines.total;
                }
                if let Ok(functions) = self.parse_metric(file_data, "functions") {
                    total_functions.covered += functions.covered;
                    total_functions.total += functions.total;
                }
                if let Ok(statements) = self.parse_metric(file_data, "statements") {
                    total_statements.covered += statements.covered;
                    total_statements.total += statements.total;
                }
                if let Ok(branches) = self.parse_metric(file_data, "branches") {
                    total_branches.covered += branches.covered;
                    total_branches.total += branches.total;
                }
            }
        }
        total_lines.percentage = self.calculate_percentage(total_lines.covered, total_lines.total);
        total_functions.percentage = self.calculate_percentage(total_functions.covered, total_functions.total);
        total_statements.percentage = self.calculate_percentage(total_statements.covered, total_statements.total);
        total_branches.percentage = self.calculate_percentage(total_branches.covered, total_branches.total);

        Ok(CoverageSummary {
            lines: total_lines,
            functions: total_functions,
            statements: total_statements,
            branches: total_branches,
        })
    }

    fn parse_files_from_final_json(&self, json: &Value, repo_path: &Path) -> Result<HashMap<String, FileCoverage>> {
        let mut files = HashMap::new();

        if let Some(obj) = json.as_object() {
            for (file_path, file_data) in obj {
                let normalized_path = self.normalize_file_path(file_path, repo_path);
                
                let file_coverage = FileCoverage {
                    path: normalized_path.clone(),
                    summary: CoverageSummary {
                        lines: self.parse_metric(file_data, "lines").unwrap_or_default(),
                        functions: self.parse_metric(file_data, "functions").unwrap_or_default(),
                        statements: self.parse_metric(file_data, "statements").unwrap_or_default(),
                        branches: self.parse_metric(file_data, "branches").unwrap_or_default(),
                    },
                    lines: self.parse_line_coverage(file_data)?,
                };

                files.insert(normalized_path, file_coverage);
            }
        }

        Ok(files)
    }

    fn parse_metric(&self, data: &Value, metric_name: &str) -> Result<CoverageMetric> {
        let metric = data.get(metric_name)
            .ok_or_else(|| Error::Custom(format!("No '{}' metric found", metric_name)))?;

        let covered = metric.get("covered")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as u32;

        let total = metric.get("total")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as u32;

        let percentage = self.calculate_percentage(covered, total);

        Ok(CoverageMetric { covered, total, percentage })
    }

        fn parse_line_coverage(&self, file_data: &Value) -> Result<HashMap<u32, u32>> {
        let mut line_coverage = HashMap::new();

        if let Some(line_data) = file_data.get("s").and_then(|v| v.as_object()) {
            for (line_str, hit_count) in line_data {
                if let (Ok(line_num), Some(count)) = (line_str.parse::<u32>(), hit_count.as_u64()) {
                    line_coverage.insert(line_num, count as u32);
                }
            }
        }

        Ok(line_coverage)
    }

    fn normalize_file_path(&self, file_path: &str, repo_path: &Path) -> String {
        if let Ok(repo_canonical) = repo_path.canonicalize() {
            if let Ok(file_canonical) = Path::new(file_path).canonicalize() {
                if let Ok(relative) = file_canonical.strip_prefix(&repo_canonical) {
                    return relative.to_string_lossy().to_string();
                }
            }
        }
    
        file_path.to_string()
    }

    fn calculate_percentage(&self, covered: u32, total: u32) -> f64 {
        if total == 0 {
            0.0
        } else {
            (covered as f64 / total as f64) * 100.0
        }
    }

}


impl Default for CoverageMetric {
    fn default() -> Self {
        Self { covered: 0, total: 0, percentage: 0.0 }
    }
}

/*
This parser:
- Reads standard JSON coverage files (Jest/Vitest/c8 format)
- Handles both summary and final coverage files
- Normalizes file paths relative to repo root
- Calculates percentages consistently
- Extracts line-by-line coverage data
- Handles missing data gracefully
*/