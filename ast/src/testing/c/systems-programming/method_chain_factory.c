// Tests free-function call resolution via the C registry.
//
// create_pool() and use_pool() are free functions.  The resolver
// seeds dir_fns so that the call to create_pool() inside use_pool()
// resolves via directory-scoped function lookup.
//
// @ast node: Class "SimplePool"
// @ast node: Function "create_pool"
// @ast node: Function "use_pool"
// @ast edge: Calls -> Function "create_pool" "method_chain_factory.c"
#include <stdlib.h>

typedef struct {
    int size;
} SimplePool;

SimplePool* create_pool(int size) {
    SimplePool* p = (SimplePool*)malloc(sizeof(SimplePool));
    p->size = size;
    return p;
}

void use_pool() {
    SimplePool* pool = create_pool(64);
    pool->size = 128;
}
