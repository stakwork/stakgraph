#ifndef MODELS_H
#define MODELS_H

#include <stdint.h>

/**
 * @brief Represents a user in the system.
 */
typedef struct {
    int id;             /**< Unique user identifier */
    char name[100];     /**< User's full name */
    char email[100];    /**< User's email address */
} User;

/**
 * @brief Represents a product in the catalog.
 */
typedef struct {
    int id;             /**< Unique product identifier */
    char name[100];     /**< Product name */
    float price;        /**< Product price in USD */
    char category[50];  /**< Product category */
} Product;

/**
 * @brief Initialize a new user
 * @param id User ID
 * @param name User name
 * @param email User email
 * @return User struct
 */
User create_user(int id, const char* name, const char* email);

/**
 * @brief Create a new product
 */
Product create_product(int id, const char* name, float price);

#endif // MODELS_H
