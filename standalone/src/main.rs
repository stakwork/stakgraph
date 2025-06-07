#[cfg(feature = "neo4j")]
mod handlers;
mod types;

use types::Result;

#[cfg(feature = "neo4j")]
#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<()> {
    use axum::{routing::post, Router};
    use tower_http::cors::CorsLayer;
    use tracing_subscriber::{filter::LevelFilter, EnvFilter};

    let filter = EnvFilter::builder()
        .with_default_directive(LevelFilter::INFO.into())
        .from_env_lossy();
    tracing_subscriber::fmt()
        .with_target(false)
        .with_env_filter(filter)
        .init();

    let cors_layer = CorsLayer::permissive();
    let app = Router::new()
        .route("/process", post(handlers::process))
        .route("/clear", post(handlers::clear_graph))
        .route("/ingest", post(handlers::ingest))
        .layer(cors_layer);

    let port = std::env::var("PORT").unwrap_or_else(|_| "7777".to_string());
    let bind = format!("0.0.0.0:{}", port);
    let listener = tokio::net::TcpListener::bind(bind).await.unwrap();
    println!("=> listening on http://{}", listener.local_addr().unwrap());
    axum::serve(listener, app).await.unwrap();
    Ok(())
}

#[cfg(not(feature = "neo4j"))]
fn main() -> Result<()> {
    println!(
        "The 'neo4j' feature must be enabled to build this binary. Use: cargo run --features neo4j"
    );
    Ok(())
}
