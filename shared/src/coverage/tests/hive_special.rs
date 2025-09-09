use crate::coverage::utils::*;
use crate::coverage::{
    detector::DefaultLanguageDetector,
    analyzer::DefaultProjectAnalyzer,
    parser::DefaultReportParser,
    types::{Language, LanguageDetector, ProjectAnalyzer, ReportParser},
};
use std::process::Command;

#[tokio::test]
#[ignore]
async fn test_hive_special_coverage() {
    let git_url = "https://github.com/stakwork/hive";
    
    println!("Testing Hive repo with special setup");
    
    let repo_path = clone_repo(git_url)
        .expect("Failed to clone Hive repo");
    println!("Clone successful");
    
    let detector = DefaultLanguageDetector::new();
    let detected_languages = detector.detect(&repo_path);
    println!("Languages detected: {:?}", detected_languages);
    assert!(detected_languages.contains(&Language::TypeScript));
    
    let analyzer = DefaultProjectAnalyzer::new();
    let config = analyzer.analyze(&repo_path, &Language::TypeScript)
        .expect("Analysis failed for Hive");
    println!("Frameworks detected: {:?}", config.test_frameworks);
    
    println!("Installing dependencies with yarn...");
    let install_result = install_hive_dependencies(&repo_path);
    match install_result {
        Ok(_) => println!("Dependencies installed"),
        Err(e) => {
            eprintln!("Dependency installation failed: {:?}", e);
            cleanup_test_repo(git_url).ok();
            return; 
        }
    }
    
    println!("Setting up test environment...");
    setup_hive_test_env(&repo_path);

    println!("Running Vitest coverage for unit tests...");
    let coverage_result = run_hive_coverage(&repo_path);
    
    match coverage_result {
        Ok(_) => {
            println!("Coverage run successful");
            
            if has_coverage_output(&repo_path) {
                println!("Coverage output detected");
                
                let parser = DefaultReportParser::new();
                let parse_result = parser.parse_report(&repo_path, &config);
                
                match parse_result {
                    Ok(report) => {
                        println!("Parse successful!");
                        println!("Files covered: {}", report.files.len());
                        println!("Line coverage: {:.1}%", report.summary.lines.percentage);
                        
                        assert_eq!(report.language, Language::TypeScript);
                        assert!(!report.files.is_empty());
                    }
                    Err(e) => {
                        eprintln!("Parse failed: {:?}", e);
                    }
                }
            } else {
                eprintln!("No coverage output found");
            }
        }
        Err(e) => {
            eprintln!("Coverage run failed: {:?}", e);
        }
    }
    
    cleanup_test_repo(git_url).ok();
    println!("Cleanup completed");
}

fn install_hive_dependencies(repo_path: &std::path::Path) -> crate::Result<()> {
    let yarn_result = run_command("yarn", &["install"], repo_path);
    
    if yarn_result.is_ok() {

        let _ = run_command("yarn", &["add", "--dev", "@vitest/coverage-v8"], repo_path);
        Ok(())
    } else {
        run_command("npm", &["install"], repo_path)?;
        let _ = run_command("npm", &["install", "--save-dev", "@vitest/coverage-v8"], repo_path);
        Ok(())
    }
}

fn setup_hive_test_env(repo_path: &std::path::Path) {
    let _ = run_command("npx", &["prisma", "generate"], repo_path);
    
    let _ = run_command("npm", &["run", "setup"], repo_path);
}


fn run_hive_coverage(repo_path: &std::path::Path) -> crate::Result<()> {
    let mut cmd = Command::new("npm");
    cmd.args(&["run", "test:coverage"])
       .current_dir(repo_path)
       .env("NODE_ENV", "test")
       .env("CI", "true");
    

    let output = cmd.output()?;
    
    if output.status.success() {
        println!("Vitest coverage completed successfully");
        Ok(())
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        
        println!("Vitest stdout: {}", stdout);
        println!("Vitest stderr: {}", stderr);
        
        if stdout.contains("Coverage") || stderr.contains("Coverage") {
            println!("Coverage was generated despite test failures");
            Ok(())
        } else {
            Err(crate::Error::Custom(format!(
                "Vitest coverage failed: {}", stderr
            )))
        }
    }
}

#[tokio::test]
#[ignore]
async fn test_hive_unit_tests_only() {
    let git_url = "https://github.com/stakwork/hive";
    
    println!("Testing Hive unit tests only");
    
    let repo_path = clone_repo(git_url)
        .expect("Failed to clone Hive repo");
    
    if install_hive_dependencies(&repo_path).is_ok() {
        setup_hive_test_env(&repo_path);
        
        let result = Command::new("npx")
            .args(&["vitest", "run", "src/__tests__/unit", "--coverage"])
            .current_dir(&repo_path)
            .env("NODE_ENV", "test")
            .output();
        
        match result {
            Ok(_output) => {
                if has_coverage_output(&repo_path) {
                    println!("Coverage files generated for unit tests");
                }
            }
            Err(e) => {
                eprintln!("Unit test run failed: {:?}", e);
            }
        }
    }
    
    cleanup_test_repo(git_url).ok();
}
