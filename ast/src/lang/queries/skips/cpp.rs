const CPP_STDLIB: [&str; 20] = [
    "push_back",
    "pop_back",
    "insert",
    "erase",
    "find",
    "clear",
    "size",
    "begin",
    "end",
    "get",
    "make_shared",
    "make_unique",
    "move",
    "swap",
    "emplace_back",
    "reserve",
    "resize",
    "front",
    "back",
    "at",
];

const SQLITE_METHODS: [&str; 15] = [
    "sqlite3_open",
    "sqlite3_close",
    "sqlite3_prepare",
    "sqlite3_prepare_v2",
    "sqlite3_step",
    "sqlite3_reset",
    "sqlite3_finalize",
    "sqlite3_bind_int",
    "sqlite3_bind_text",
    "sqlite3_bind_double",
    "sqlite3_column_int",
    "sqlite3_column_text",
    "sqlite3_column_double",
    "sqlite3_errmsg",
    "sqlite3_exec",
];

const CROW_FRAMEWORK: [&str; 8] = [
    "CROW_ROUTE",
    "CROW_WEBSOCKET_ROUTE",
    "CROW_BP_ROUTE",
    "methods",
    "enable_cors",
    "port",
    "multithreaded_service",
    "validate_json",
];

const HTTP_UTILS: [&str; 10] = [
    "status",
    "set_header",
    "body",
    "json",
    "response",
    "request",
    "query_string",
    "headers",
    "path",
    "method",
];

pub fn should_skip(called: &str, _operand: &Option<String>) -> bool {
    CPP_STDLIB.contains(&called)
        || SQLITE_METHODS.contains(&called)
        || CROW_FRAMEWORK.contains(&called)
        || HTTP_UTILS.contains(&called)
}
