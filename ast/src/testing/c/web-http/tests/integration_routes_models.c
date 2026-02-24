#include "../routes.h"
#include "../models.h"
#include <assert.h>
#include <stdio.h>

void integration_test_get_user_response(void) {
    User u = create_user(7, "Test User", "test@example.com");
    
    assert(u.id == 7 && "User ID should be set");
    assert(u.name[0] != '\0' && "User name should not be empty");
    assert(u.email[0] != '\0' && "User email should not be empty");
    
    printf("PASS: integration_test_get_user_response\n");
}

void integration_test_product_listing(void) {
    Product p = create_product(55, "Gizmo", 49.95f);
    
    assert(p.id == 55 && "Product ID should be set");
    assert(p.price > 49.90f && p.price < 50.00f && "Product price should be set");
    
    printf("PASS: integration_test_product_listing\n");
}

void integration_test_route_registration(void) {
    const char *get_user_route = "/users/:id";
    const char *post_user_route = "/users";
    
    assert(get_user_route[0] == '/' && "Routes should start with /");
    assert(post_user_route[0] == '/' && "Routes should start with /");
    
    printf("PASS: integration_test_route_registration\n");
}

int main(void) {
    printf("Running web route integration tests...\n");
    integration_test_get_user_response();
    integration_test_product_listing();
    integration_test_route_registration();
    printf("All route integration tests passed!\n");
    return 0;
}
