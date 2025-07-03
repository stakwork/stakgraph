use lazy_static::lazy_static;
use std::collections::BTreeSet;
use std::sync::Mutex;
use tracing::warn;

lazy_static! {
    static ref CURRENT_ANALYSIS: Mutex<BTreeSet<String>> = Mutex::new(BTreeSet::new());
}

/// Clears the current analysis log before a graph runs its analysis.
pub fn clear_current_analysis() {
    CURRENT_ANALYSIS.lock().unwrap().clear();
}

/// Called by graph analysis methods to log a line of output.
pub fn log_analysis_line(line: String) {
    CURRENT_ANALYSIS.lock().unwrap().insert(line);
}

/// Parses a multi-line string into a BTreeSet to be used as a golden standard.
pub fn parse_golden_standard(golden_str: &str) -> BTreeSet<String> {
    golden_str
        .lines()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

pub fn assert_golden_standard(golden_standard: &BTreeSet<String>) {
    let current_log = CURRENT_ANALYSIS.lock().unwrap();

    let missing: Vec<_> = golden_standard.difference(&current_log).collect();
    let extra: Vec<_> = current_log.difference(golden_standard).collect();

    if !missing.is_empty() || !extra.is_empty() {
        warn!("Analysis does not match the golden standard.");
        if !missing.is_empty() {
            println!("\n--- Missing from Graph ---");
            missing.iter().for_each(|l| println!("- {}", l));
        }
        if !extra.is_empty() {
            println!("\n--- Extra in Graph ---");
            extra.iter().for_each(|l| println!("+ {}", l));
        }
        println!("\n");
        panic!("Graph analysis does not match the golden standard.");
    }
}
