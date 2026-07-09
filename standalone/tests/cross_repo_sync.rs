#![cfg(feature = "neo4j")]

use ast::lang::graphs::graph_ops::GraphOps;
use ast::lang::graphs::EdgeType;
use ast::lang::linker::normalize_frontend_path;
use ast::lang::{Graph, NodeType};
use ast::repo::Repo;
use axum::{extract::State, Json};
use standalone::{ingest, sync, AppState, ProcessBody};
use std::collections::{BTreeSet, HashMap};
use std::sync::{atomic::AtomicBool, Arc};
use tokio::sync::{broadcast, Mutex};

const FE: &str = "https://github.com/fayekelmith/graph-update-frontend";
const BE: &str = "https://github.com/fayekelmith/graph-update-backend";

fn test_state() -> Arc<AppState> {
    let (tx, mut rx) = broadcast::channel(10000);
    tokio::spawn(async move { while rx.recv().await.is_ok() {} });
    Arc::new(AppState {
        tx,
        api_token: None,
        async_status: Arc::new(Mutex::new(HashMap::new())),
        busy: Arc::new(AtomicBool::new(false)),
    })
}

fn body(repo_url: &str, branch: Option<&str>) -> ProcessBody {
    ProcessBody {
        repo_url: Some(repo_url.to_string()),
        repo_path: None,
        username: None,
        pat: None,
        use_lsp: Some(false),
        commit: None,
        branch: branch.map(str::to_string),
        callback_url: None,
        realtime: None,
        docs: None,
        mocks: None,
        embeddings: None,
        embeddings_limit: None,
    }
}

fn reset_clones() {
    for u in [FE, BE] {
        if let Ok(path) = Repo::get_path_from_url(u) {
            let _ = std::fs::remove_dir_all(path);
        }
    }
}

async fn ingest_both(state: &Arc<AppState>) {
    reset_clones();
    let url = format!("{},{}", FE, BE);
    let _ = ingest(State(state.clone()), Json(body(&url, Some("before"))))
        .await
        .expect("combined ingest failed");
}

async fn sync_repo(state: &Arc<AppState>, repo: &str, branch: &str) {
    let _ = sync(State(state.clone()), Json(body(repo, Some(branch))))
        .await
        .unwrap_or_else(|_| panic!("sync {repo} @ {branch} failed"));
}

fn endpoint_verbs(g: &mut GraphOps) -> HashMap<(String, String, usize), String> {
    g.graph
        .find_nodes_by_type(NodeType::Endpoint)
        .into_iter()
        .map(|n| {
            let verb = n.meta.get("verb").cloned().unwrap_or_default();
            ((n.name, n.file, n.start), verb)
        })
        .collect()
}

fn calls_edges(g: &mut GraphOps) -> BTreeSet<String> {
    let verbs = endpoint_verbs(g);
    g.graph
        .find_nodes_with_edge_type(NodeType::Request, NodeType::Endpoint, EdgeType::Calls)
        .iter()
        .map(|(_req, ep)| {
            let verb = verbs
                .get(&(ep.name.clone(), ep.file.clone(), ep.start))
                .cloned()
                .unwrap_or_default();
            format!("{} {}", verb, ep.name)
        })
        .collect()
}

fn linked_requests(g: &mut GraphOps) -> BTreeSet<String> {
    g.graph
        .find_nodes_with_edge_type(NodeType::Request, NodeType::Endpoint, EdgeType::Calls)
        .iter()
        .map(|(req, _ep)| normalize_frontend_path(&req.name).unwrap_or_else(|| req.name.clone()))
        .collect()
}

fn expected(items: &[&str]) -> BTreeSet<String> {
    items.iter().map(|s| s.to_string()).collect()
}

async fn fresh_graph() -> GraphOps {
    let mut g = GraphOps::new();
    g.connect().await.unwrap();
    g.clear().await.unwrap();
    g
}

fn baseline_edges() -> BTreeSet<String> {
    expected(&[
        "GET /bounties",
        "POST /bounties",
        "GET /bounties/{id}",
        "PUT /bounties/{id}",
        "DELETE /bounties/{id}",
        "GET /people",
        "GET /people/{id}",
        "POST /auth/login",
    ])
}

fn evolved_edges() -> BTreeSet<String> {
    expected(&[
        "GET /bounties",
        "POST /bounties",
        "GET /bounties/{id}",
        "PUT /bounties/{id}",
        "POST /bounties/{id}/assign",
        "GET /users",
        "GET /users/{id}",
        "GET /workspaces/{id}/bounties",
        "POST /auth/login",
    ])
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_cross_repo_sync_lifecycle() {
    let state = test_state();
    let mut g = fresh_graph().await;

    ingest_both(&state).await;
    assert_eq!(calls_edges(&mut g), baseline_edges(), "state 0 baseline edges");
    assert!(
        !calls_edges(&mut g).contains("GET /workspaces/{id}"),
        "dangling endpoint must have no caller"
    );
    assert!(
        !linked_requests(&mut g).contains("/notifications"),
        "dangling request must stay unlinked"
    );

    sync_repo(&state, BE, "after").await;
    assert_eq!(
        calls_edges(&mut g),
        expected(&[
            "GET /bounties",
            "POST /bounties",
            "GET /bounties/{id}",
            "PUT /bounties/{id}",
            "POST /auth/login",
        ]),
        "state 1 backend-only sync"
    );

    sync_repo(&state, FE, "after").await;
    assert_eq!(calls_edges(&mut g), evolved_edges(), "state 2 fully evolved");

    assert!(!calls_edges(&mut g).contains("GET /workspaces/{id}"));
    assert!(!linked_requests(&mut g).contains("/notifications"));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_cross_repo_resync_idempotent() {
    let state = test_state();
    let mut g = fresh_graph().await;

    ingest_both(&state).await;
    sync_repo(&state, BE, "after").await;
    sync_repo(&state, FE, "after").await;
    let before = calls_edges(&mut g);
    assert_eq!(before, evolved_edges());

    sync_repo(&state, BE, "after").await;
    sync_repo(&state, FE, "after").await;
    assert_eq!(calls_edges(&mut g), before, "idempotent re-sync changed the graph");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_backward_sync_is_silent_noop_characterization() {
    let state = test_state();
    let mut g = fresh_graph().await;

    ingest_both(&state).await;
    sync_repo(&state, BE, "after").await;
    sync_repo(&state, FE, "after").await;
    let evolved = calls_edges(&mut g);
    assert_eq!(evolved, evolved_edges());

    sync_repo(&state, BE, "before").await;
    sync_repo(&state, FE, "before").await;
    assert_eq!(
        calls_edges(&mut g),
        evolved,
        "backward sync changed the graph — the limitation may now be fixed; update this test"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_cross_repo_muted_preservation() {
    let state = test_state();
    let g = fresh_graph().await;

    ingest_both(&state).await;

    let file = "fayekelmith/graph-update-backend/handlers/bounties.go";
    let muted = g
        .set_node_muted(&NodeType::Function, "ListBounties", file, true)
        .await
        .unwrap_or(0);
    assert!(muted > 0, "expected to mute the ListBounties function node");
    assert!(g
        .is_node_muted(&NodeType::Function, "ListBounties", file)
        .await
        .unwrap_or(false));

    sync_repo(&state, BE, "after").await;
    assert!(
        g.is_node_muted(&NodeType::Function, "ListBounties", file)
            .await
            .unwrap_or(false),
        "mute lost across cross-repo sync"
    );
}
