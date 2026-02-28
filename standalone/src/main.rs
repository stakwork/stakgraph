use standalone::types::Result;
#[cfg(feature = "neo4j")]
use standalone::{
    auth, busy,
    handlers::*,
    service::{graph_service::*, repo_service::*},
    types::AppState,
};

#[cfg(feature = "neo4j")]
use tower_http::services::ServeFile;

#[cfg(feature = "neo4j")]
#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<()> {
    use axum::extract::Request;
    use axum::middleware;
    use axum::{routing::get, routing::post, Router};
    use std::sync::atomic::AtomicBool;
    use std::sync::Arc;
    use tokio::sync::{broadcast, Mutex};
    use tower_http::cors::CorsLayer;
    use tower_http::trace::TraceLayer;
    use tracing::{debug_span, Span};
    use tracing_subscriber::{filter::LevelFilter, EnvFilter};

    let mut filter = EnvFilter::builder()
        .with_default_directive(LevelFilter::INFO.into())
        .from_env_lossy();
    if let Ok(directive) = "tower_http=debug".parse() {
        filter = filter.add_directive(directive);
    }
    tracing_subscriber::fmt()
        .with_target(false)
        .with_env_filter(filter)
        .init();

    let mut graph_ops = ast::lang::graphs::graph_ops::GraphOps::new();
    if let Err(e) = graph_ops.check_connection().await {
        eprintln!("Failed to connect to graph db: {:?}", e);
        return Err(standalone::types::WebError(shared::Error::Custom(format!(
            "Failed to connect to graph database: {:?}",
            e
        ))));
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
        busy: Arc::new(AtomicBool::new(false)),
    });

    tracing::debug!("starting server");
    let cors_layer = CorsLayer::permissive();

    let mut app = Router::new()
        .route("/events", get(sse_handler))
        .route("/busy", get(busy_handler));

    // Routes that use busy middleware (synchronous operations)
    let busy_routes = Router::new()
        .route("/process", post(sync))
        .route("/sync", post(sync))
        .route("/ingest", post(ingest))
        .route("/embed_code", post(embed_code_handler))
        .route_layer(middleware::from_fn_with_state(
            app_state.clone(),
            busy::busy_middleware,
        ));

    // Routes that manage busy flag internally (async operations with background tasks)
    let async_routes = Router::new()
        .route("/ingest_async", post(ingest_async))
        .route("/sync_async", post(sync_async));

    let mut protected_routes = Router::new()
        .route("/clear", post(clear_graph))
        .route("/status/:request_id", get(get_status))
        .route("/fetch-repo", post(fetch_repo))
        .route("/fetch-repos", get(fetch_repos))
        .route("/search", post(vector_search_handler))
        .route("/tests/coverage", get(coverage_handler))
        .route("/tests/nodes", get(nodes_handler))
        .route("/tests/has", get(has_handler))
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
    let listener = tokio::net::TcpListener::bind(&bind).await.map_err(|e| {
        eprintln!("Failed to bind to {}: {}", bind, e);
        standalone::types::WebError(shared::Error::Custom(format!(
            "Failed to bind to {}: {}",
            bind, e
        )))
    })?;

    tokio::spawn(async {
        if let Err(err) = tokio::signal::ctrl_c().await {
            eprintln!("failed waiting for Ctrl+C signal: {}", err);
            return;
        }
        // for docker container
        println!("\nReceived Ctrl+C, exiting immediately...");
        std::process::exit(0);
    });

    let local_addr = listener.local_addr().map_err(|e| {
        eprintln!("Failed to get listener address: {}", e);
        standalone::types::WebError(shared::Error::Custom(format!(
            "Failed to get listener address: {}",
            e
        )))
    })?;
    println!("=> listening on http://{}", local_addr);
    axum::serve(listener, app).await.map_err(|e| {
        eprintln!("Server error: {}", e);
        standalone::types::WebError(shared::Error::Custom(format!("Server error: {}", e)))
    })?;

    println!("Server shutdown complete.");
    Ok(())
}

#[cfg(feature = "neo4j")]
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
