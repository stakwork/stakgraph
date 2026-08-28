//! Live-Neo4j integration tests for `POST /api/hive/query`.
//!
//! Proves that the read-mode bolt transaction (`GraphOps::execute_raw_cypher` →
//! `neo4rs::Graph::execute_read`, which sends `mode: "r"` autocommit metadata)
//! makes the **database itself** refuse writes — including write procedures
//! invoked via `CALL`, which the keyword denylist does not generally catch —
//! and that those rejections surface as HTTP 403 with the same body the
//! denylist uses, leaving no trace in the graph.
//!
//! The denylist is bypassed via the test-only entry point
//! `hive_query_handler_denylist_bypassed` (never routed), so every rejection
//! observed here comes from Neo4j, not from string matching.
//!
//! Every probe query is terminated with a `RETURN` clause: the handler appends
//! a server-controlled `LIMIT`, and a bare `CREATE .../ CALL ... LIMIT` is a
//! Cypher syntax error (which would fail for the wrong reason).
//!
//! Skips gracefully when Neo4j is unreachable (e.g. plain `cargo test` without
//! a database); run with `cargo test --features neo4j` against a live instance.

#![cfg(feature = "neo4j")]

use axum::Json;
use neo4rs::query;
use standalone::handlers::hive_query::{
    hive_query_handler, hive_query_handler_denylist_bypassed, HiveQueryBody,
};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Label used for probe nodes; deleted before and after each run.
const PROBE_LABEL: &str = "_ReadOnlyProbe";

/// Env-configured direct (write-mode) bolt connection, mirroring
/// `Neo4jConfig::default()` (NEO4J_URI / NEO4J_USERNAME / NEO4J_PASSWORD /
/// NEO4J_DATABASE).
///
/// Returns `None` (after printing a skip notice) when the server cannot be
/// reached, so the test skips gracefully instead of failing.
fn direct_connection() -> Option<neo4rs::Graph> {
    let uri = std::env::var("NEO4J_URI").unwrap_or_else(|_| "bolt://localhost:7687".to_string());
    let user = std::env::var("NEO4J_USERNAME").unwrap_or_else(|_| "neo4j".to_string());
    let pass = std::env::var("NEO4J_PASSWORD").unwrap_or_else(|_| "testtest".to_string());
    let db = std::env::var("NEO4J_DATABASE").unwrap_or_else(|_| "neo4j".to_string());
    let config = match neo4rs::ConfigBuilder::default()
        .uri(uri)
        .user(user)
        .password(pass)
        .db(db)
        .build()
    {
        Ok(config) => config,
        Err(e) => {
            eprintln!("skipping hive read-only test: invalid Neo4j config ({e})");
            return None;
        }
    };
    // Building the pool is synchronous; the first query below is what actually
    // dials the server, so reachability is checked separately.
    Some(neo4rs::Graph::connect(config).expect("build neo4j pool"))
}

/// Cheap reachability probe: `RETURN 1` under a timeout. The bolt pool is
/// built synchronously, but the first query is what dials the server.
async fn reachable(conn: &neo4rs::Graph) -> bool {
    match tokio::time::timeout(Duration::from_secs(10), run_direct(conn, "RETURN 1")).await {
        Ok(Ok(())) => true,
        Ok(Err(e)) => {
            eprintln!("skipping hive read-only test: Neo4j not reachable ({e})");
            false
        }
        Err(_) => {
            eprintln!("skipping hive read-only test: Neo4j connection timed out");
            false
        }
    }
}

/// Run a write-mode statement directly against Neo4j, draining the stream so
/// the query actually executes (neo4rs executes lazily on stream consumption).
async fn run_direct(conn: &neo4rs::Graph, cypher: &str) -> neo4rs::Result<()> {
    let mut stream = conn.execute(query(cypher)).await?;
    while stream.next().await?.is_some() {}
    Ok(())
}

/// Count probe nodes with the given id, via the direct write-mode connection.
async fn count_probe(conn: &neo4rs::Graph, id: &str) -> i64 {
    let mut stream = conn
        .execute(query(&format!(
            "MATCH (n:{label} {{id: $id}}) RETURN count(n) AS c",
            label = PROBE_LABEL
        ))
        .param("id", id))
        .await
        .expect("probe count query");
    match stream.next().await.expect("probe count row") {
        Some(row) => row.get::<i64>("c").expect("count column"),
        None => 0,
    }
}

fn body(query: impl Into<String>) -> Json<HiveQueryBody> {
    Json(HiveQueryBody {
        language: Some("cypher".to_string()),
        query: query.into(),
        limit: None,
    })
}

/// (status, parsed JSON body) of a handler response.
async fn status_and_json(
    resp: axum::response::Response,
) -> (axum::http::StatusCode, serde_json::Value) {
    let status = resp.status();
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .expect("read response body");
    (status, serde_json::from_slice(&bytes).expect("json body"))
}

/// True when `apoc.create.node` is registered on the connected server.
async fn apoc_create_node_installed(conn: &neo4rs::Graph) -> bool {
    let mut stream = match conn
        .execute(query(
            "SHOW PROCEDURES YIELD name WHERE name = 'apoc.create.node' RETURN count(*) AS c",
        ))
        .await
    {
        Ok(s) => s,
        Err(_) => return false,
    };
    match stream.next().await {
        Ok(Some(row)) => row.get::<i64>("c").unwrap_or(0) > 0,
        _ => false,
    }
}

/// Unique probe id per test run.
fn probe_id() -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    format!("ro-probe-{}", nanos)
}

/// End-to-end: with the denylist bypassed, both a plain `CREATE` and a write
/// procedure invoked via `CALL` are refused by the database inside the
/// read-mode bolt transaction (403, denylist-identical body), and leave no
/// node behind. A direct write-mode connection first proves the database is
/// writable at all, so the absence assertions are not vacuous.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_hive_write_rejected_at_database_level() {
    let conn = match direct_connection() {
        Some(conn) => conn,
        None => return,
    };
    if !reachable(&conn).await {
        return;
    }

    let id = probe_id();
    let cleanup = format!("MATCH (n:{}) DETACH DELETE n", PROBE_LABEL);
    run_direct(&conn, &cleanup).await.expect("cleanup before");

    // Sanity: the database accepts writes through a normal (write-mode)
    // connection. If it did not (e.g. a read-only instance), every 403 below
    // would be trivially true and prove nothing.
    run_direct(
        &conn,
        &format!(
            "CREATE (n:{} {{id: '{}'}})",
            PROBE_LABEL, id
        ),
    )
    .await
    .expect("direct write-mode CREATE");
    assert_eq!(count_probe(&conn, &id).await, 1, "sanity write landed");
    run_direct(&conn, &cleanup).await.expect("cleanup sanity node");

    // 1. Plain CREATE through the handler, denylist bypassed — the database
    //    must refuse it. (RETURN keeps the appended LIMIT valid.)
    let resp = hive_query_handler_denylist_bypassed(body(format!(
        "CREATE (n:{label} {{id: '{id}'}}) RETURN n.id",
        label = PROBE_LABEL,
        id = id
    )))
    .await;
    let (status, json) = status_and_json(resp).await;
    assert_eq!(status, axum::http::StatusCode::FORBIDDEN, "json: {json}");
    assert_eq!(
        json,
        serde_json::json!({"error": "write operations not permitted"})
    );
    assert_eq!(
        count_probe(&conn, &id).await,
        0,
        "database must not have executed the CREATE"
    );

    // 2. Write procedure via CALL — the case the denylist's CALL allowance
    //    would normally leave to the transaction layer — also refused.
    let resp = hive_query_handler_denylist_bypassed(body(format!(
        "CALL apoc.create.node(['{label}'], {{id: '{id}'}}) YIELD node RETURN node.id",
        label = PROBE_LABEL,
        id = id
    )))
    .await;
    let (status, json) = status_and_json(resp).await;
    if status == axum::http::StatusCode::INTERNAL_SERVER_ERROR {
        // Distinguish "apoc not installed here" from a real failure.
        assert!(
            !apoc_create_node_installed(&conn).await,
            "apoc.create.node is installed but the write-procedure call did not return 403: {json}"
        );
        eprintln!("skipping apoc assertion: apoc.create.node not installed on this server");
    } else {
        assert_eq!(status, axum::http::StatusCode::FORBIDDEN, "json: {json}");
        assert_eq!(
            json,
            serde_json::json!({"error": "write operations not permitted"})
        );
    }
    assert_eq!(
        count_probe(&conn, &id).await,
        0,
        "write procedure must leave no trace in the graph"
    );

    run_direct(&conn, &cleanup).await.expect("cleanup after");
}

/// End-to-end: legitimate reads — plain `MATCH` and a read-only procedure
/// (`CALL db.labels()`) — still work through the full handler on the read-mode
/// transaction, and the production entry point still enforces the denylist.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_hive_reads_still_work() {
    let conn = match direct_connection() {
        Some(conn) => conn,
        None => return,
    };
    if !reachable(&conn).await {
        return;
    }

    // Plain MATCH returns 200 with columns/rows.
    let (status, json) =
        status_and_json(hive_query_handler(body("MATCH (n) RETURN n LIMIT 5")).await).await;
    assert_eq!(status, axum::http::StatusCode::OK, "json: {json}");
    assert!(json.get("columns").is_some(), "expected columns: {json}");

    // Read-only procedure returns 200 — CALL usage must not be collateral damage
    // of the read-mode transaction. (The query uses the `YIELD ... RETURN` form
    // because the handler appends a server-controlled LIMIT, and bare
    // `CALL db.labels() LIMIT n` is a Cypher syntax error — pre-existing
    // forced-LIMIT behavior, independent of read mode.)
    let (status, json) = status_and_json(
        hive_query_handler(body("CALL db.labels() YIELD label RETURN label")).await,
    )
    .await;
    assert_eq!(status, axum::http::StatusCode::OK, "json: {json}");
    assert!(json.get("rows").is_some(), "expected rows: {json}");

    // Production entry point still enforces the denylist (CREATE → 403 from the
    // keyword check, before the query ever reaches Neo4j).
    let (status, json) =
        status_and_json(hive_query_handler(body("CREATE (n:Denied)")).await).await;
    assert_eq!(status, axum::http::StatusCode::FORBIDDEN, "json: {json}");
    assert_eq!(
        json,
        serde_json::json!({"error": "write operations not permitted"})
    );
}
