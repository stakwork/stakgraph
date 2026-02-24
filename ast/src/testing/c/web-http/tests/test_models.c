#include "../models.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>

void test_create_user(void) {
    User u = create_user(42, "Alice Smith", "alice@example.com");
    
    assert(u.id == 42 && "User ID should be set correctly");
    assert(strcmp(u.name, "Alice Smith") == 0 && "User name should be set correctly");
    assert(strcmp(u.email, "alice@example.com") == 0 && "User email should be set correctly");
    
    printf("PASS: test_create_user\n");
}

void test_create_user_with_long_name(void) {
    char long_name[200];
    memset(long_name, 'A', 199);
    long_name[199] = '\0';
    
    User u = create_user(1, long_name, "test@example.com");
    
    assert(u.id == 1 && "User ID should be set");
    assert(strlen(u.name) < 100 && "User name should be truncated to fit buffer");
    
    printf("PASS: test_create_user_with_long_name\n");
}

void test_create_product(void) {
    Product p = create_product(101, "Laptop", 999.99);
    
    assert(p.id == 101 && "Product ID should be set correctly");
    assert(strcmp(p.name, "Laptop") == 0 && "Product name should be set correctly");
    assert(p.price == 999.99f && "Product price should be set correctly");
    assert(strcmp(p.category, "General") == 0 && "Product category should default to General");
    
    printf("PASS: test_create_product\n");
}

void test_create_product_with_decimal_price(void) {
    Product p = create_product(202, "Coffee Mug", 12.50);
    
    assert(p.id == 202 && "Product ID should be set");
    assert(p.price > 12.49 && p.price < 12.51 && "Product price should handle decimals");
    
    printf("PASS: test_create_product_with_decimal_price\n");
}

void test_multiple_users(void) {
    User u1 = create_user(1, "User One", "one@test.com");
    User u2 = create_user(2, "User Two", "two@test.com");
    
    assert(u1.id != u2.id && "Users should have different IDs");
    assert(strcmp(u1.name, u2.name) != 0 && "Users should have different names");
    
    printf("PASS: test_multiple_users\n");
}

int main(void) {
    printf("Running models unit tests...\n");
    test_create_user();
    test_create_user_with_long_name();
    test_create_product();
    test_create_product_with_decimal_price();
    test_multiple_users();
    printf("All models unit tests passed!\n");
    return 0;
}
