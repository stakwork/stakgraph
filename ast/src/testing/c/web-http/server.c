#include <onion/onion.h>
#include <onion/log.h>
#include "routes.h"
#include <signal.h>

static onion *o = NULL;

void shutdown_server(int _) {
    if (o) onion_listen_stop(o);
}

int main(int argc, char **argv) {
    signal(SIGINT, shutdown_server);
    signal(SIGTERM, shutdown_server);

    o = onion_new(O_POOL);
    onion_set_timeout(o, 5000);
    onion_set_hostname(o, "0.0.0.0");
    onion_set_port(o, "8080");

    onion_url *urls = onion_root_url(o);

    // Register routes
    onion_url_add(urls, "^users/([0-9]+)$", handler_get_user);
    onion_url_add(urls, "^users$", handler_post_user);
    onion_url_add(urls, "^products$", handler_list_products);

    // Anonymous handler for health check
    onion_url_add(urls, "^health$", (void*)0); // onion doesn't support lambdas directly like C++, but we can simulate capturing the pattern
                                              // In C, we often use a wrapper or macro. 
                                              // For this test, we'll use a pattern that matches our query expectation
                                              // to verify we can capture "anonymous" like constructs if they existed or 
                                              // if we use a specific macro.
    
    // Let's add a proper handler with data to test the third argument capture
    onion_url_add_with_data(urls, "^static/.*", handler_list_products, NULL, NULL);

    state_t s = onion_listen(o);
    onion_free(o);
    return s;
}
