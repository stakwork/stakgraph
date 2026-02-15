// Standard library functions to skip from function call graph
const STDLIB_FUNCTIONS: [&str; 61] = [
    // stdio.h
    "printf", "fprintf", "sprintf", "snprintf", "scanf", "fscanf", "sscanf", "fgets", "fputs",
    "fopen", "fclose", "fread", "fwrite", "fseek", "ftell", "rewind", "fflush", "putchar",
    "getchar", "puts", "gets", // stdlib.h
    "malloc", "free", "calloc", "realloc", "exit", "abort", "atexit", "atoi", "atof", "atol",
    "strtol", "strtod", "rand", "srand", "qsort", "bsearch", "abs", "labs",
    // string.h
    "strcpy", "strncpy", "strcat", "strncat", "strcmp", "strncmp", "strlen", "strchr", "strrchr",
    "strstr", "memcpy", "memmove", "memset", "memcmp", "memchr", // math.h
    "sqrt", "pow", "sin", "cos", "tan", "floor", "ceil",
];

const POSIX_FUNCTIONS: [&str; 30] = [
    // pthread.h
    "pthread_create",
    "pthread_join",
    "pthread_exit",
    "pthread_detach",
    "pthread_mutex_init",
    "pthread_mutex_lock",
    "pthread_mutex_unlock",
    "pthread_mutex_destroy",
    "pthread_cond_init",
    "pthread_cond_wait",
    "pthread_cond_signal",
    "pthread_cond_broadcast",
    // time.h
    "time",
    "clock",
    "difftime",
    "strftime",
    "localtime",
    "gmtime",
    // unistd.h
    "read",
    "write",
    "close",
    "open",
    "pipe",
    "fork",
    "execve",
    "sleep",
    "usleep",
    // signal.h
    "signal",
    "raise",
    "kill",
];

// Common C libraries (operand names to skip)
const COMMON_LIBRARIES: [&str; 15] = [
    "stdio", "stdlib", "string", "math", "time", "pthread", "unistd", "signal", "errno", "assert",
    "ctype", "limits", "float", "stddef", "stdint",
];

pub fn should_skip(called: &str, operand: &Option<String>) -> bool {
    // Skip standard library functions
    if STDLIB_FUNCTIONS.contains(&called) {
        return true;
    }

    // Skip POSIX functions
    if POSIX_FUNCTIONS.contains(&called) {
        return true;
    }

    // Skip if operand is a common library
    if let Some(op) = operand {
        if COMMON_LIBRARIES.contains(&op.as_str()) {
            return true;
        }
    }

    false
}
