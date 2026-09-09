//! Canonical timestamp helper for the Neo4j writer.
//!
//! All `date_added_to_graph` writes in this module must route through
//! [`now_epoch_ms`] — no call site may compute a timestamp inline.

use std::time::{SystemTime, UNIX_EPOCH};

/// Returns the current wall-clock time as `BoltType::Integer` of epoch
/// milliseconds.
///
/// Canonical wire format for `date_added_to_graph`:
/// **epoch milliseconds, Neo4j Integer, set once on create — do not use for
/// ON MATCH.** Write the value only under `ON CREATE SET` (or an equivalent
/// first-creation branch such as `COALESCE`); re-merging an existing node
/// must never re-stamp its creation time.
///
/// This replaces the legacy inline
/// `BoltType::String(format!("{:.7}", epoch_seconds))` — the old 7-decimal
/// string is no longer a valid encoding for this field. Any new Cypher
/// parameter needing a timestamp in this module must route through here
/// instead of computing one inline.
pub fn now_epoch_ms() -> neo4rs::BoltType {
    let ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|dur| dur.as_millis() as i64)
        // A pre-epoch system clock saturates at 0 rather than leaving the
        // `$now` parameter unbound — an unbound `$now` would fail the whole
        // query with "Expected parameter(s): now".
        .unwrap_or(0);
    neo4rs::BoltType::Integer(ms.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn now_epoch_ms_returns_integer_of_epoch_millis() {
        let before = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock after 1970")
            .as_millis() as i64;
        let ts = now_epoch_ms();
        let after = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock after 1970")
            .as_millis() as i64;

        // Generous delta: test machines can pause between clock reads.
        let delta = 5_000;
        let lo = before.saturating_sub(delta);
        let hi = after.saturating_add(delta);

        match ts {
            neo4rs::BoltType::Integer(i) => {
                // The bounds also pin the unit: a seconds value (~1.7e9) or a
                // 7-decimal formatted string would fall outside any
                // ms-magnitude window (~1.7e12+), and the match arm itself
                // rejects a String/Float variant.
                assert!(
                    i.value >= lo && i.value <= hi,
                    "expected epoch milliseconds in [{lo}, {hi}], got {}",
                    i.value
                );
            }
            other => panic!("expected BoltType::Integer, got {:?}", other),
        }
    }
}
