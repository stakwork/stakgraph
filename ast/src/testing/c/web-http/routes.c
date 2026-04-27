#include "routes.h"
#include "models.h"
#include <stdio.h>

// @ast node: Function "handler_get_user"
// @ast edge: Calls -> Function "create_user" "models.c"
onion_connection_status handler_get_user(void *_, onion_request *req, onion_response *res) {
    const char *id_str = onion_request_get_query(req, "id");
    int id = id_str ? atoi(id_str) : 0;
    
    // Create a dummy user for response
    // @ast node: Instance "u"
    // @ast edge: Of -> Class "User" "models.h"
    User u = create_user(id, "John Doe", "john@example.com");
    
    char buffer[256];
    snprintf(buffer, sizeof(buffer), "{\"id\": %d, \"name\": \"%s\"}", u.id, u.name);
    
    onion_response_set_header(res, "Content-Type", "application/json");
    onion_response_write0(res, buffer);
    return OCS_PROCESSED;
}

// @ast node: Function "handler_post_user"
// @ast edge: Calls -> Function "create_user" "models.c"
onion_connection_status handler_post_user(void *_, onion_request *req, onion_response *res) {
    // Process POST data
    const char *name = onion_request_get_post(req, "name");
    const char *email = onion_request_get_post(req, "email");
    
    if (name && email) {
        // @ast node: Instance "u"
        // @ast edge: Of -> Class "User" "models.h"
        User u = create_user(1, name, email);
        onion_response_write0(res, "User created");
        return OCS_PROCESSED;
    }
    
    onion_response_set_code(res, 400);
    return OCS_PROCESSED;
}

// @ast node: Function "handler_list_products"
// @ast edge: Calls -> Function "create_product" "models.c"
onion_connection_status handler_list_products(void *_, onion_request *req, onion_response *res) {
    // @ast node: Instance "p"
    // @ast edge: Of -> Class "Product" "models.h"
    Product p = create_product(101, "Widget", 19.99);
    onion_response_write0(res, "[{\"id\": 101, \"name\": \"Widget\"}]");
    return OCS_PROCESSED;
}
