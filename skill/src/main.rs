use anyhow::{Context, Error, Result};
use ast::lang::ArrayGraph;
use ast::Repo;
use aws_config::BehaviorVersion;
use aws_sdk_s3::presigning::PresigningConfigBuilder;
use aws_sdk_s3::primitives::ByteStream;
use aws_sdk_s3::Client;
use lsp::Language;
use serde::Serialize;
use std::env;
use std::time::Duration;
use tracing_subscriber::filter::LevelFilter;
use tracing_subscriber::EnvFilter;

const DEFAULT_S3_BUCKET: &str = "stak-request-large-responses";
const DEFAULT_S3_REGION: &str = "us-east-1";

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<(), Error> {
    let filter = EnvFilter::builder()
        .with_default_directive(LevelFilter::INFO.into())
        .from_env_lossy();
    tracing_subscriber::fmt()
        .with_target(false)
        .with_env_filter(filter)
        .init();

    start().await
}

#[derive(Serialize)]
struct Response {
    presigned_url: String,
    note: String,
    custom_status_code: i32,
}

#[derive(Serialize)]
struct PostData {
    data: Response,
    status: String,
}

impl Response {
    fn new(presigned_url: String) -> Self {
        Self {
            presigned_url,
            note: "Response too large, stored in S3 and available via presigned URL".to_string(),
            custom_status_code: 999,
        }
    }
}

async fn start() -> std::result::Result<(), Error> {
    let webhook_url = env::var("WEBHOOK_URL").context("no WEBHOOK_URL")?;
    let langs = vec![
        Language::Rust,
        Language::Go,
        Language::Typescript,
        Language::Ruby,
    ];
    for l in langs {
        match lsp::get_lsp_version(l.clone()).await {
            Ok(ver) => println!("{:?} LSP version: {}", l, ver),
            Err(e) => println!("failed to get {:?} LSP version: {:?}", l, e),
        }
    }
    let res = match start_inner().await {
        Ok(data) => PostData {
            data,
            status: "success".to_string(),
        },
        Err(e) => {
            println!("Stackgraph Error: {:?}", e);
            PostData {
                data: Response {
                    presigned_url: "".to_string(),
                    note: format!("{:?}", e),
                    custom_status_code: 500,
                },
                status: "error".to_string(),
            }
        }
    };
    reqwest::Client::new()
        .post(&webhook_url)
        .json(&res)
        .send()
        .await?;
    Ok(())
}

async fn start_inner() -> Result<Response> {
    let repo_url = env::var("REPO_URL").context("no REPO_URL")?;
    let language = env_not_empty("LANGUAGE");
    let use_lsp = env::var("USE_LSP").ok().map(|v| v == "true");
    let username = env_not_empty("USERNAME");
    let pat = env_not_empty("PAT");
    let ff = env_not_empty("FILES_FILTER");
    let files_filter = ff
        .map(|fff| fff.split(',').map(|s| s.to_string()).collect())
        .unwrap_or_default();
    let rev = env_not_empty("REV");
    let revs = rev
        .map(|r| r.split(',').map(|s| s.to_string()).collect())
        .unwrap_or_default();

    tracing::info!("START!!!: {:?} {} {:?}", language, repo_url, use_lsp);
    tracing::info!("REVS: {:?}", revs);

    let graph = match language {
        Some(l) => {
            let lsp = use_lsp.unwrap_or(false);
            let repo =
                Repo::new_clone_to_tmp(&repo_url, Some(&l), lsp, username, pat, files_filter, revs)
                    .await?;
            repo.build_graph::<ArrayGraph>().await?
        }
        None => {
            let repos =
                Repo::new_clone_multi_detect(&repo_url, username, pat, files_filter, revs).await?;
            repos.build_graphs::<ArrayGraph>().await?
        }
    };

    let filename = repo_url.replace("/", "_");
    let presigned_url = upload_to_s3(&format!("{filename}.json"), graph).await?;
    let response = Response::new(presigned_url);

    Ok(response)
}

async fn upload_to_s3<T>(key: &str, json: T) -> Result<String>
where
    T: Serialize,
{
    let config = aws_config::defaults(BehaviorVersion::latest())
        .region(DEFAULT_S3_REGION)
        .load()
        .await;
    let client = Client::new(&config);
    let serialized = serde_json::to_vec(&json)?;
    let body = ByteStream::new(serialized.into());
    // upload file
    let s3_bucket = env::var("S3_BUCKET").unwrap_or_else(|_| DEFAULT_S3_BUCKET.to_string());
    client
        .put_object()
        .bucket(&s3_bucket)
        .key(key)
        .body(body)
        .send()
        .await?;
    // create presigned url
    let mut pre_config_builder = PresigningConfigBuilder::default();
    pre_config_builder.set_expires_in(Some(Duration::from_secs(300)));
    let pre_config = pre_config_builder.build()?;
    let presigned_req = client
        .get_object()
        .bucket(&s3_bucket)
        .key(key)
        .presigned(pre_config)
        .await?;
    Ok(presigned_req.uri().to_string())
}

fn env_not_empty(name: &str) -> Option<String> {
    // return None if it doesn't exist or is empty string
    env::var(name).ok().filter(|v| !v.is_empty())
}
