use standalone::codecov::{run, CodecovBody};

#[tokio::test]
#[ignore]
async fn codecov_hive_repo() {
    if std::env::var("DATABASE_URL").is_err() {
        std::env::set_var("DATABASE_URL", "postgresql://hive_user:hive_password@localhost:5432/hive_db");
    }
    let body = CodecovBody { repo_url: "https://github.com/stakwork/hive".into(), username: None, pat: None, commit: None };
    let res = run(body).await;
    assert!(res.is_ok(), "coverage run failed: {:?}", res.err());
}

#[tokio::test]
#[ignore]
async fn codecov_openhtf_repo() {
    let body = CodecovBody { repo_url: "https://github.com/google/openhtf".into(), username: None, pat: None, commit: None };
    let res = run(body).await;
    assert!(res.is_ok(), "Python coverage run failed: {:?}", res.err());
}
