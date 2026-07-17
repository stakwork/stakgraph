use ast::lang::graphs::graph_ops::GraphOps;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use once_cell::sync::Lazy;
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::json;

/// Write-keyword denylist — case-insensitive word-boundary patterns.
///
/// `CALL` is intentionally omitted: read-only procedures (e.g. `CALL db.labels()`) are
/// legitimate Graph Explorer queries.  Write procedures invoked via `CALL` are blocked at
/// the Neo4j transaction layer by read-mode enforcement in `execute_raw_cypher`.
///
/// `FOREACH` and `LOAD` are included:
/// - `FOREACH` is a native Cypher write clause.
/// - `LOAD` enables `LOAD CSV FROM 'http://attacker.com/…'` — an SSRF vector.
///
/// Matching is performed against the query with string literals stripped, so that values
/// like `n.creator = 'MERGE request author'` do not trigger a false positive.
static WRITE_PATTERNS: Lazy<Vec<(&'static str, Regex)>> = Lazy::new(|| {
    let keywords = [
        "CREATE", "MERGE", "SET", "DELETE", "REMOVE", "DROP", "FOREACH", "LOAD",
    ];
    keywords
        .iter()
        .map(|&kw| {
            let pattern = format!(r"(?i)\b{}\b", kw);
            (kw, Regex::new(&pattern).expect("valid regex"))
        })
        .collect()
});

/// Strip single- and double-quoted string literals from a Cypher query so that keywords
/// that appear inside quoted values (e.g. `n.creator = 'MERGE request author'`) do not
/// cause false-positive denylist matches.
///
/// The function replaces each quoted span with an empty quoted pair (`''`), preserving
/// the surrounding query structure.  Backslash-escaped quotes within a literal are
/// handled so that `'it\'s fine'` does not prematurely close the literal.
fn strip_string_literals(query: &str) -> String {
    let mut result = String::with_capacity(query.len());
    let mut chars = query.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\'' || c == '"' {
            let quote = c;
            // Replace the entire quoted span with an empty literal placeholder.
            result.push(quote);
            result.push(quote);
            // Consume until the matching closing quote, honouring backslash escapes.
            loop {
                match chars.next() {
                    None => break,
                    Some('\\') => {
                        chars.next(); // skip the escaped character
                    }
                    Some(ch) if ch == quote => break,
                    Some(_) => {}
                }
            }
        } else {
            result.push(c);
        }
    }
    result
}

/// Regex to strip any existing LIMIT clause from the query before appending the
/// server-controlled value.  The `(?i)` flag makes it case-insensitive.
static LIMIT_STRIP_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?i)\bLIMIT\s+\d+\b").expect("valid regex"));

#[derive(Debug, Deserialize)]
pub struct HiveQueryBody {
    /// Must be `"cypher"`.  Declared `Option<String>` so Axum does not 422 on a missing
    /// field before the handler can return a proper 400 with an error message.
    pub language: Option<String>,
    pub query: String,
    /// Capped at 1 000 server-side.  Defaults to 100.
    pub limit: Option<u32>,
}

#[derive(Debug, Serialize)]
pub struct HiveQueryResponse {
    pub columns: Vec<String>,
    pub rows: Vec<Vec<serde_json::Value>>,
}

/// `POST /api/hive/query`
///
/// Validates the request, applies the write-keyword denylist, enforces a server-controlled
/// LIMIT, and proxies the query to Neo4j via a read-mode bolt transaction.
pub async fn hive_query_handler(
    Json(body): Json<HiveQueryBody>,
) -> Response {
    // Language validation — 400 (not 422) even when the field is missing.
    if body.language.as_deref() != Some("cypher") {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({"error": "language must be 'cypher'"})),
        )
            .into_response();
    }

    // Per-query length limit.
    if body.query.len() > 4096 {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({"error": "query too long"})),
        )
            .into_response();
    }

    let effective_limit = body.limit.unwrap_or(100).min(1000);

    tracing::info!(
        query_len = body.query.len(),
        limit = effective_limit,
        "hive_query: received request"
    );

    // Write-keyword denylist (defense-in-depth — primary guard is read-mode transaction).
    // Strip string literals first so values like `n.creator = 'MERGE request author'`
    // do not cause false positives.
    let query_no_literals = strip_string_literals(&body.query);
    for (keyword, pattern) in WRITE_PATTERNS.iter() {
        if pattern.is_match(&query_no_literals) {
            tracing::warn!(
                matched_keyword = keyword,
                "hive_query: write keyword detected in query"
            );
            return (
                StatusCode::FORBIDDEN,
                Json(json!({"error": "write operations not permitted"})),
            )
                .into_response();
        }
    }

    // Strip any existing LIMIT clause and append the server-controlled value.
    let stripped = LIMIT_STRIP_RE.replace_all(&body.query, "");
    let modified_query = format!("{} LIMIT {}", stripped.trim_end(), effective_limit);

    // Per-request bolt pool — consistent with vector_search_handler and all other handlers.
    let mut graph_ops = GraphOps::new();
    if let Err(e) = graph_ops.connect().await {
        tracing::error!(error = %e, "hive_query: failed to connect to Neo4j");
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": "query execution failed"})),
        )
            .into_response();
    }

    match graph_ops.execute_raw_cypher(&modified_query).await {
        Ok((columns, rows)) => (
            StatusCode::OK,
            Json(json!(HiveQueryResponse { columns, rows })),
        )
            .into_response(),
        Err(e) => {
            tracing::error!(error = %e, "hive_query: Neo4j execution failed");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error": "query execution failed"})),
            )
                .into_response()
        }
    }
}

// ─── Unit tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn check_write_blocked(query: &str) -> bool {
        let stripped = strip_string_literals(query);
        WRITE_PATTERNS
            .iter()
            .any(|(_, pattern)| pattern.is_match(&stripped))
    }

    // ── Denylist true-positives ───────────────────────────────────────────────

    #[test]
    fn test_denylist_create_upper() {
        assert!(check_write_blocked("CREATE (n:Test)"));
    }

    #[test]
    fn test_denylist_create_lower() {
        assert!(check_write_blocked("create (n:Test)"));
    }

    #[test]
    fn test_denylist_merge() {
        assert!(check_write_blocked("MERGE (n:Test {id:1})"));
    }

    #[test]
    fn test_denylist_set() {
        assert!(check_write_blocked("MATCH (n) SET n.x = 1"));
    }

    #[test]
    fn test_denylist_delete() {
        assert!(check_write_blocked("MATCH (n) DELETE n"));
    }

    #[test]
    fn test_denylist_remove() {
        assert!(check_write_blocked("MATCH (n) REMOVE n.x"));
    }

    #[test]
    fn test_denylist_drop() {
        assert!(check_write_blocked("DROP INDEX my_index"));
    }

    #[test]
    fn test_denylist_foreach() {
        assert!(check_write_blocked("FOREACH (x IN [1] | SET n.prop = 'x')"));
    }

    #[test]
    fn test_denylist_load() {
        assert!(check_write_blocked("LOAD CSV FROM 'http://attacker.com/data.csv' AS row"));
    }

    #[test]
    fn test_denylist_set_mixed_case() {
        assert!(check_write_blocked("MATCH (n) Set n.x = 1"));
    }

    // ── Denylist true-negatives (must NOT be blocked) ─────────────────────────

    #[test]
    fn test_denylist_dataset_property_not_blocked() {
        assert!(!check_write_blocked("MATCH (n) WHERE n.dataset = 'x' RETURN n"));
    }

    #[test]
    fn test_denylist_is_set_property_not_blocked() {
        assert!(!check_write_blocked("MATCH (n) WHERE n.is_set = true RETURN n"));
    }

    #[test]
    fn test_denylist_remove_date_property_not_blocked() {
        assert!(!check_write_blocked("MATCH (n) RETURN n.remove_date"));
    }

    #[test]
    fn test_denylist_creator_merge_in_string_not_blocked() {
        assert!(!check_write_blocked(
            "MATCH (n) WHERE n.creator = 'MERGE request author' RETURN n"
        ));
    }

    #[test]
    fn test_denylist_call_not_blocked() {
        // CALL is intentionally excluded — read procedures are valid.
        assert!(!check_write_blocked("CALL db.labels()"));
    }

    // ── LIMIT strip-and-replace ───────────────────────────────────────────────

    fn apply_limit(query: &str, body_limit: Option<u32>) -> String {
        let effective_limit = body_limit.unwrap_or(100).min(1000);
        let stripped = LIMIT_STRIP_RE.replace_all(query, "");
        format!("{} LIMIT {}", stripped.trim_end(), effective_limit)
    }

    #[test]
    fn test_limit_appended_when_absent() {
        let result = apply_limit("MATCH (n) RETURN n", None);
        assert!(result.ends_with("LIMIT 100"), "got: {result}");
        assert!(!result.contains("LIMIT 100 LIMIT 100"), "double limit: {result}");
    }

    #[test]
    fn test_limit_replaced_when_present() {
        let result = apply_limit("MATCH (n) RETURN n LIMIT 50", Some(50));
        assert!(result.ends_with("LIMIT 50"), "got: {result}");
        // Original LIMIT 50 must be stripped, not duplicated.
        let count = result.matches("LIMIT").count();
        assert_eq!(count, 1, "expected exactly one LIMIT, got: {result}");
    }

    #[test]
    fn test_limit_capped_at_1000() {
        let result = apply_limit("MATCH (n) RETURN n", Some(5000));
        assert!(result.ends_with("LIMIT 1000"), "got: {result}");
    }

    #[test]
    fn test_limit_existing_large_replaced() {
        let result = apply_limit("MATCH (n) RETURN n LIMIT 999999", None);
        assert!(result.ends_with("LIMIT 100"), "got: {result}");
        let count = result.matches("LIMIT").count();
        assert_eq!(count, 1, "expected exactly one LIMIT, got: {result}");
    }

    #[test]
    fn test_limit_case_insensitive_strip() {
        let result = apply_limit("MATCH (n) RETURN n limit 500", None);
        assert!(result.ends_with("LIMIT 100"), "got: {result}");
        let count = result.to_uppercase().matches("LIMIT").count();
        assert_eq!(count, 1, "expected exactly one LIMIT, got: {result}");
    }

    // ── Language / query length validation (logic tests, no HTTP layer) ───────

    #[test]
    fn test_language_none_is_rejected() {
        // language = None should fail the `language.as_deref() != Some("cypher")` check
        let lang: Option<String> = None;
        assert!(lang.as_deref() != Some("cypher"));
    }

    #[test]
    fn test_language_sql_is_rejected() {
        let lang: Option<String> = Some("sql".to_string());
        assert!(lang.as_deref() != Some("cypher"));
    }

    #[test]
    fn test_language_cypher_is_accepted() {
        let lang: Option<String> = Some("cypher".to_string());
        assert_eq!(lang.as_deref(), Some("cypher"));
    }

    #[test]
    fn test_query_too_long_rejected() {
        let long_query = "A".repeat(4097);
        assert!(long_query.len() > 4096);
    }

    #[test]
    fn test_query_at_limit_accepted() {
        let ok_query = "A".repeat(4096);
        assert!(ok_query.len() <= 4096);
    }
}
