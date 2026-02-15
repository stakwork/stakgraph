#ifndef ROUTES_H
#define ROUTES_H

#include <onion/onion.h>
#include <onion/request.h>
#include <onion/response.h>

/**
 * @brief Get user by ID handler
 * @route GET /users/:id
 */
onion_connection_status handler_get_user(void *_, onion_request *req, onion_response *res);

/**
 * @brief Create new user handler
 * @route POST /users
 */
onion_connection_status handler_post_user(void *_, onion_request *req, onion_response *res);

/**
 * @brief List products handler
 * @route GET /products
 */
onion_connection_status handler_list_products(void *_, onion_request *req, onion_response *res);

#endif // ROUTES_H
