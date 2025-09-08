use crate::coverage::utils::*;
use crate::coverage::{
    detector::DefaultLanguageDetector,
    analyzer::DefaultProjectAnalyzer,
    runner::DefaultCoverageRunner,
    types::{Language, LanguageDetector, ProjectAnalyzer, CoverageRunner},
};

const TEST_REPOS: &[(&str, &str)] = &[
    ("https://github.com/mtiller/ts-jest-sample", "ts-jest-sample"),
    ("https://github.com/stakwork/hive", "hive"), 
    ("https://github.com/tomroper/jasmine-tests", "jasmine-tests"),
];

#[tokio::test]
#[ignore]
async fn test_clone_all_repos() {
    for &(git_url, _expected_name) in TEST_REPOS {
        let repo_path = clone_repo(git_url)
            .expect(&format!("Failed to clone {}", git_url));
        
        assert!(repo_path.exists());
        assert!(dir_exists_and_not_empty(&repo_path));
        
        let package_json = repo_path.join("package.json");
        assert!(package_json.exists());
        
        cleanup_test_repo(git_url).ok();
    }
}

#[tokio::test]
#[ignore]
async fn test_detection_and_analysis() {
    for &(git_url, expected_name) in TEST_REPOS {
        let repo_path = clone_repo(git_url)
            .expect(&format!("Failed to clone {}", git_url));
        
        let detector = DefaultLanguageDetector::new();
        let detected_languages = detector.detect(&repo_path);
        
        assert!(!detected_languages.is_empty());
        assert!(detected_languages.contains(&Language::TypeScript));
        
        let analyzer = DefaultProjectAnalyzer::new();
        let config = analyzer.analyze(&repo_path, &Language::TypeScript)
            .expect(&format!("Analysis should succeed for {}", expected_name));
        
        assert_eq!(config.language, Language::TypeScript);
        assert!(!config.test_frameworks.is_empty());
        assert!(!config.config_files.is_empty());
        
        cleanup_test_repo(git_url).ok();
    }
}

#[test]
fn test_get_repo_name_from_url() {
    assert_eq!(
        get_repo_name_from_url("https://github.com/mtiller/ts-jest-sample").unwrap(),
        "ts-jest-sample"
    );
    assert_eq!(
        get_repo_name_from_url("https://github.com/stakwork/hive.git").unwrap(),
        "hive"
    );
    assert_eq!(
        get_repo_name_from_url("https://github.com/tomroper/jasmine-tests").unwrap(),
        "jasmine-tests"
    );
}

#[tokio::test]
#[ignore]
async fn test_ts_jest_sample() {
    let git_url = "https://github.com/mtiller/ts-jest-sample";
    let expected_name = "ts-jest-sample";
    
    test_single_repo(git_url, expected_name).await;
}

#[tokio::test]
#[ignore]
async fn test_hive_repo() {
    let git_url = "https://github.com/stakwork/hive";
    let expected_name = "hive";
    
    test_single_repo(git_url, expected_name).await;
}

#[tokio::test]
#[ignore]
async fn test_jasmine_tests_repo() {
    let git_url = "https://github.com/tomroper/jasmine-tests";
    let expected_name = "jasmine-tests";
    
    test_single_repo(git_url, expected_name).await;
}

async fn test_single_repo(git_url: &str, expected_name: &str) {
    println!("Testing repo: {}", expected_name);
    
    let repo_path = clone_repo(git_url)
        .expect(&format!("Failed to clone {}", git_url));
    
    let detector = DefaultLanguageDetector::new();
    let detected_languages = detector.detect(&repo_path);
    assert!(detected_languages.contains(&Language::TypeScript));
    
    let analyzer = DefaultProjectAnalyzer::new();
    let config = analyzer.analyze(&repo_path, &Language::TypeScript)
        .expect(&format!("Analysis failed for {}", expected_name));
    println!("Frameworks detected: {:?}", config.test_frameworks);
    println!("Package manager: {:?}", config.package_manager);
    
    let install_result = install_dependencies(&repo_path);
    match install_result {
        Ok(_) => println!("Dependencies installed"),
        Err(_) => {
            cleanup_test_repo(git_url).ok();
            panic!("Cannot proceed without dependencies for {}", expected_name);
        }
    }
    
    let runner = DefaultCoverageRunner::new();
    let strategy = runner.build_strategy(&repo_path, &config)
        .expect(&format!("Failed to build strategy for {}", expected_name));
    
    println!("Running: {} {:?}", strategy.command, strategy.args);
    

    let run_result = runner.run_coverage(&repo_path, &strategy);
    
    match run_result {
        Ok(_) => {
            if has_coverage_output(&repo_path) {
                let coverage_files = list_coverage_files(&repo_path);
                println!("Coverage files: {:?}", coverage_files);
            } else {
                eprintln!("No coverage output found");
            }
        }
        Err(e) => {
            eprintln!("Coverage run failed: {:?}", e);
        }
    }
    
    cleanup_test_repo(git_url).ok();
}