
    use crate::coverage::utils::*;
    use std::path::Path;

    const TEST_REPOS: &[&str] = &[
        "https://github.com/mtiller/ts-jest-sample",
        "https://github.com/stakwork/hive", 
        "https://github.com/ajithDav/javascript_p1",
    ];

    #[tokio::test]
    #[ignore]
    async fn test_clone_all_repos() {
        for &git_url in TEST_REPOS {
            println!("\nTesting clone for: {}", git_url);
            
            let temp_dir = clone_repo(git_url)
                .expect(&format!("Failed to clone {}", git_url));
            
            let repo_path = get_repo_path(&temp_dir, git_url)
                .expect(&format!("Failed to get repo path for {}", git_url));
            
            assert!(repo_path.exists(), "Repo {} should exist", git_url);
            assert!(dir_exists_and_not_empty(&repo_path), "Repo {} should not be empty", git_url);
            
            let repo_name = get_repo_name_from_url(git_url).unwrap();
            println!("Successfully cloned {}", repo_name);
        }
    }

