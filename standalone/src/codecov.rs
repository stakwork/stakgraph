use crate::types::{CodecovBody, Report};
use shared::Result;

pub async fn run(body: CodecovBody) -> Result<Report> {
    let repo_path = ast::repo::Repo::get_path_from_url(&body.repo_url)?;
    ast::repo::clone_repo(
        &body.repo_url,
        &repo_path,
        body.username.clone(),
        body.pat.clone(),
        body.commit.as_deref(),
        None,
    )
    .await?;

    let commit = match lsp::git::get_commit_hash(&repo_path).await {
        Ok(h) => h,
        Err(_) => body.commit.clone().unwrap_or_default(),
    };

    let mut report = match shared::codecov::run(&repo_path, &body.repo_url, &commit) {
        Ok(r) => r,
        Err(e) => Report {
            repo_url: body.repo_url.clone(),
            commit: commit.clone(),
            generated_at: chrono::Utc::now().to_rfc3339(),
            languages: vec![],
            errors: vec![e.to_string()],
        },
    };
    report.repo_url = body.repo_url;
    report.commit = commit;
    Ok(report)
}
