use crate::coverage::utils::*;

const TEST_REPOS: &[(&str, &str)] = &[
    ("https://github.com/mtiller/ts-jest-sample", "ts-jest-sample"),
    ("https://github.com/stakwork/hive", "hive"), 
    ("https://github.com/ajithDav/javascript_p1", "javascript_p1"),
];

#[tokio::test]
#[ignore]
async fn test_clone_all_repos() {
    for &(git_url, expected_name) in TEST_REPOS {
        println!("Testing clone for: {}", git_url);
        
        let repo_path = clone_repo(git_url)
            .expect(&format!("Failed to clone {}", git_url));
        
        println!("Cloned to: {:?}", repo_path);

        assert!(repo_path.exists(), "Repo path should exist");
        assert!(dir_exists_and_not_empty(&repo_path), "Repo should not be empty");
        
 
        //let package_json = repo_path.join("package.json");
       //assert!(package_json.exists(), "package.json should exist");
        
        println!("Successfully cloned and verified {}", expected_name);
        
  
        cleanup_test_repo(git_url).ok();
        println!("Cleaned up {}", expected_name);
        println!("---");
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
        get_repo_name_from_url("https://github.com/ajithDav/javascript_p1/").unwrap(),
        "javascript_p1"
    );
}