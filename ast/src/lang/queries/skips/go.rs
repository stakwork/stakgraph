const STDLIB_PACKAGES: [&str; 20] = [
    "fmt",
    "log",
    "errors",
    "strings",
    "strconv",
    "time",
    "math",
    "os",
    "io",
    "bufio",
    "bytes",
    "sync",
    "json",
    "http",
    "net",
    "context",
    "regexp",
    "sort",
    "path",
    "filepath",
];

const BUILTIN_FUNCTIONS: [&str; 10] = [
    "len", "cap", "make", "append", "copy", "delete", "panic", "recover", "close", "new",
];

pub fn should_skip(called: &str, operand: &Option<String>) -> bool {
    if BUILTIN_FUNCTIONS.contains(&called) {
        return true;
    }

    if let Some(op) = operand {
        if STDLIB_PACKAGES.contains(&op.as_str()) {
            return true;
        }
    }

    false
}
