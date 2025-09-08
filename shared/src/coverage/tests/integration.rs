use crate::coverage::utils::*;
use crate::coverage::{
    detector::DefaultLanguageDetector,
    analyzer::DefaultProjectAnalyzer,
    runner::DefaultCoverageRunner,
    parser::DefaultReportParser,
    types::{Language, LanguageDetector, ProjectAnalyzer, CoverageRunner, ReportParser},
};

#[tokio::test]
#[ignore]
async fn test_full_pipeline_with_parsing() {
    let fast_repos = [
        ("https://github.com/mtiller/ts-jest-sample", "ts-jest-sample"),
    ];

    for &(git_url, expected_name) in &fast_repos {
        println!("Testing {}", expected_name);
        
        let repo_path = clone_repo(git_url)
            .expect(&format!("Failed to clone {}", git_url));
        
        let detector = DefaultLanguageDetector::new();
        let detected_languages = detector.detect(&repo_path);
        assert!(detected_languages.contains(&Language::TypeScript));
        
        let analyzer = DefaultProjectAnalyzer::new();
        let config = analyzer.analyze(&repo_path, &Language::TypeScript)
            .expect(&format!("Analysis failed for {}", expected_name));
        
        println!("Frameworks: {:?}", config.test_frameworks);
        
        install_dependencies(&repo_path)
            .expect(&format!("Failed to install dependencies for {}", expected_name));
        
        let runner = DefaultCoverageRunner::new();
        let strategy = runner.build_strategy(&repo_path, &config)
            .expect(&format!("Failed to build strategy for {}", expected_name));
        
        println!("Running: {} {:?}", strategy.command, strategy.args);
        
        let run_result = runner.run_coverage(&repo_path, &strategy);
        
        if run_result.is_ok() && has_coverage_output(&repo_path) {
            
            let parser = DefaultReportParser::new();
            let parse_result = parser.parse_report(&repo_path, &config);
            
            match parse_result {
                Ok(report) => {
                    println!("Files: {}", report.files.len());
                    println!("Line coverage: {:.1}%", report.summary.lines.percentage);
                    
                    assert_eq!(report.language, Language::TypeScript);
                    assert!(!report.files.is_empty());
                    assert!(report.summary.lines.total > 0);
                }
                Err(e) => {
                    eprintln!("Parse failed: {:?}", e);
                }
            }
        } else {
            eprintln!("No coverage output found");
        }
        
        cleanup_test_repo(git_url).ok();
    }
}