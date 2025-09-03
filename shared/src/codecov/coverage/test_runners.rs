use std::collections::HashMap;


#[derive(Debug, Clone)]
pub struct RunnerSpec {
    pub id: &'static str,                 // e.g. "jest"
    pub detect_tokens: &'static [&'static str], // script substrings to detect runner
    pub coverage_dependency: Option<&'static str>, // optional extra dep for coverage
    pub coverage_style: CoverageStyle,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CoverageStyle {
    // Native flag patterns
    Vitest,   // uses --coverage --coverage.reporter
    Jest,     // uses --coverage --coverageReporters
    Jasmine,  // needs c8 wrapper unless script already has coverage
    Mocha,    // can be wrapped with c8
    Generic,  // just append --coverage if supported
}

#[derive(Debug, Clone)]
pub struct TestScript {
    pub name: String,
    pub runner_id: String,
    pub has_coverage: bool,
}

// Static registry of known runners. Extend here to support more.
fn registry() -> &'static [RunnerSpec] {
    &[
        RunnerSpec {
            id: "vitest",
            detect_tokens: &["vitest"],
            coverage_dependency: Some("@vitest/coverage-v8"),
            coverage_style: CoverageStyle::Vitest,
        },
        RunnerSpec {
            id: "jest",
            detect_tokens: &["jest"],
            coverage_dependency: None,
            coverage_style: CoverageStyle::Jest,
        },
        RunnerSpec {
            id: "jasmine",
            detect_tokens: &["jasmine"],
            coverage_dependency: None,
            coverage_style: CoverageStyle::Jasmine,
        },
        RunnerSpec {
            id: "mocha",
            detect_tokens: &["mocha"],
            coverage_dependency: None,
            coverage_style: CoverageStyle::Mocha,
        },
    ]
}

pub fn get_runner(id: &str) -> Option<&'static RunnerSpec> {
    registry().iter().find(|r| r.id == id)
}

pub fn detect_from_package_json(scripts: &HashMap<String, String>) -> Vec<TestScript> {
    let patterns = [
        "test:coverage",
        "test",
        "test:unit",
        "coverage",
        "test:integration",
    ];
    let mut out = Vec::new();
    for script_name in patterns {
        if let Some(script_content) = scripts.get(script_name) {
            let script_lower = script_content.to_lowercase();
            if let Some(spec) = registry().iter().find(|spec| spec
                .detect_tokens
                .iter()
                .any(|tok| script_lower.contains(&tok.to_lowercase())))
            {
                let has_cov = script_lower.contains("coverage") || script_name.contains("coverage");
                out.push(TestScript {
                    name: script_name.to_string(),
                    runner_id: spec.id.to_string(),
                    has_coverage: has_cov,
                });
                continue;
            }
            out.push(TestScript {
                name: script_name.to_string(),
                runner_id: "jest".to_string(), // sane default
                has_coverage: script_lower.contains("coverage") || script_name.contains("coverage"),
            });
        }
    }
    out
}

pub fn coverage_dependency(runner_id: &str) -> Option<&'static str> {
    get_runner(runner_id).and_then(|r| r.coverage_dependency)
}

pub fn coverage_reporter_args(runner_id: &str, reports_dir: &str) -> Option<Vec<String>> {
    let spec = get_runner(runner_id)?;
    let args = match spec.coverage_style {
        CoverageStyle::Vitest => vec![
            "--coverage".into(),
            "--coverage.reporter=json-summary".into(),
            "--coverage.reporter=json".into(),
            format!("--coverage.reportsDirectory={}", reports_dir),
        ],
        CoverageStyle::Jest => vec![
            "--coverage".into(),
            "--coverageReporters=json-summary".into(),
            "--coverageReporters=json".into(),
            format!("--coverageDirectory={}", reports_dir),
        ],
        CoverageStyle::Jasmine | CoverageStyle::Mocha => {
            // These typically rely on c8 wrapper; return None so caller can wrap.
            return None;
        }
        CoverageStyle::Generic => vec!["--coverage".into()],
    };
    Some(args)
}
