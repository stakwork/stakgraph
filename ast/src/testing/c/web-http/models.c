#include "models.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

/**
 * @brief Creates a new user with the given details.
 * 
 * @param id The unique user ID.
 * @param name The user's full name.
 * @param email The user's email address.
 * @return User The initialized user struct.
 */
User create_user(int id, const char* name, const char* email) {
    User u;
    u.id = id;
    strncpy(u.name, name, sizeof(u.name) - 1);
    strncpy(u.email, email, sizeof(u.email) - 1);
    return u;
}

/**
 * @brief Creates a new product with the given details.
 * 
 * @param id The unique product ID.
 * @param name The product name.
 * @param price The product price.
 * @return Product The initialized product struct.
 */
Product create_product(int id, const char* name, float price) {
    Product p;
    p.id = id;
    strncpy(p.name, name, sizeof(p.name) - 1);
    p.price = price;
    strcpy(p.category, "General");
    return p;
}

void print_user(const User* u) {
    if (u) {
        printf("User: %s (%s)\n", u->name, u->email);
    }
}
