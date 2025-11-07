mod auth;
mod busy;
mod codecov;
#[cfg(feature = "neo4j")]
mod handlers;
mod types;
mod utils;
mod webhook;

use ast::repo::StatusUpdate;
use axum::extract::Request;
use axum::middleware::{self};
use axum::{routing::get, routing::post, Router};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use tokio::sync::{broadcast, Mutex};
use tower_http::cors::CorsLayer;
use tower_http::services::ServeFile;
use tower_http::trace::TraceLayer;
use tracing::{debug_span, Span};
use tracing_subscriber::{filter::LevelFilter, EnvFilter};
use types::{AsyncStatusMap, CodecovStatusMap, Result};

#[derive(Clone)]
struct AppState {
    tx: broadcast::Sender<StatusUpdate>,
    api_token: Option<String>,
    async_status: AsyncStatusMap,
    codecov_status: CodecovStatusMap,
    busy: Arc<AtomicBool>,
}

#[cfg(feature = "neo4j")]
#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<()> {
    let filter = EnvFilter::builder()
        .with_default_directive(LevelFilter::INFO.into())
        .from_env_lossy()
        .add_directive("tower_http=debug".parse().unwrap());
    tracing_subscriber::fmt()
        .with_target(false)
        .with_env_filter(filter)
        .init();

    let mut graph_ops = ast::lang::graphs::graph_ops::GraphOps::new();
    if let Err(e) = graph_ops.check_connection().await {
        panic!("Failed to connect to graph db: {:?}", e);
    }
    graph_ops.graph.create_indexes().await?;

    let (tx, _rx) = broadcast::channel(10000);

    let mut dummy_rx = tx.subscribe();
    tokio::spawn(async move {
        while let Ok(_) = dummy_rx.recv().await {
            // Just consume messages, don't do anything
            // this is required to keep the msgs fast. weird.
        }
    });

    // Get API token from environment variable - now optional
    let api_token = std::env::var("API_TOKEN").ok();

    if api_token.is_some() {
        tracing::info!("API_TOKEN provided - authentication enabled");
    } else {
        tracing::warn!("API_TOKEN not provided - authentication disabled");
    }

    let app_state = Arc::new(AppState {
        tx,
        api_token,
        async_status: Arc::new(Mutex::new(std::collections::HashMap::new())),
        codecov_status: Arc::new(Mutex::new(std::collections::HashMap::new())),
        busy: Arc::new(AtomicBool::new(false)),
    });

    tracing::debug!("starting server");
    let cors_layer = CorsLayer::permissive();

    let mut app = Router::new()
        .route("/events", get(handlers::sse_handler))
        .route("/busy", get(handlers::busy_handler));

    // Routes that use busy middleware (synchronous operations)
    let busy_routes = Router::new()
        .route("/process", post(handlers::process))
        .route("/sync", post(handlers::process))
        .route("/ingest", post(handlers::ingest))
        .route("/embed_code", post(handlers::embed_code_handler))
        .route_layer(middleware::from_fn_with_state(
            app_state.clone(),
            busy::busy_middleware,
        ));

    // Routes that manage busy flag internally (async operations with background tasks)
    let async_routes = Router::new()
        .route("/ingest_async", post(handlers::ingest_async))
        .route("/sync_async", post(handlers::sync_async));

    let mut protected_routes = Router::new()
        .route("/clear", post(handlers::clear_graph))
        .route("/status/:request_id", get(handlers::get_status))
        .route("/fetch-repo", post(handlers::fetch_repo))
        .route("/fetch-repos", get(handlers::fetch_repos))
        .route("/search", post(handlers::vector_search_handler))
        .route("/tests/coverage", get(handlers::coverage_handler))
        .route("/tests/nodes", get(handlers::nodes_handler))
        .route("/tests/has", get(handlers::has_handler))
        .route("/codecov", post(handlers::codecov_handler))
        .route(
            "/codecov/:request_id",
            get(handlers::codecov_status_handler),
        )
        .merge(busy_routes)
        .merge(async_routes);

    // Add bearer auth middleware only if API token is provided
    if app_state.api_token.is_some() {
        protected_routes = protected_routes.route_layer(middleware::from_fn_with_state(
            app_state.clone(),
            auth::bearer_auth,
        ));
    }
    app = app.merge(protected_routes);

    let mut static_router = Router::new()
        .route_service("/", static_file("index.html"))
        .route_service("/styles.css", static_file("styles.css"))
        .route_service("/app.js", static_file("app.js"))
        .route_service("/utils.js", static_file("utils.js"))
        .route("/token", get(auth::token_exchange));

    // Add basic auth middleware only if API token is provided
    if app_state.api_token.is_some() {
        static_router = static_router.route_layer(middleware::from_fn_with_state(
            app_state.clone(),
            auth::basic_auth,
        ));
    }

    let app = app
        .merge(static_router)
        .with_state(app_state)
        .layer(cors_layer)
        .layer(
            TraceLayer::new_for_http()
                .make_span_with(|request: &Request<_>| {
                    debug_span!(
                        "http_request",
                        method = ?request.method(),
                        uri = ?request.uri(),
                        version = ?request.version(),
                    )
                })
                .on_request(|request: &Request<_>, _span: &Span| {
                    tracing::info!("{} {}", request.method(), request.uri());
                })
                .on_response(
                    |_response: &axum::response::Response,
                     latency: std::time::Duration,
                     _span: &Span| {
                        tracing::debug!("finished processing request in {:?}", latency)
                    },
                ),
        );

    let port = std::env::var("PORT").unwrap_or_else(|_| "7799".to_string());
    let bind = format!("0.0.0.0:{}", port);
    let listener = tokio::net::TcpListener::bind(bind).await.unwrap();

    tokio::spawn(async {
        tokio::signal::ctrl_c()
            .await
            .expect("Failed to install Ctrl+C handler");
        // for docker container
        println!("\nReceived Ctrl+C, exiting immediately...");
        std::process::exit(0);
    });

    println!("=> listening on http://{}", listener.local_addr().unwrap());
    axum::serve(listener, app).await.unwrap();

    println!("Server shutdown complete.");
    Ok(())
}

fn static_file(path: &str) -> ServeFile {
    ServeFile::new(format!("standalone/static/{}", path))
}

#[cfg(not(feature = "neo4j"))]
fn main() -> Result<()> {
    println!(
        "The 'neo4j' feature must be enabled to build this binary. Use: cargo run --features neo4j"
    );
    Ok(())
}
