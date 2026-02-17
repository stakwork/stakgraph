#ifndef UART_H
#define UART_H

#include "../types.h"
#include <stddef.h>

/**
 * @brief Initialize UART peripheral
 */
status_t uart_init(uint8_t uart_id, uint32_t baudrate);

/**
 * @brief Send character
 */
void uart_putc(uint8_t uart_id, char c);

/**
 * @brief Send string
 */
void uart_puts(uint8_t uart_id, const char *str);

#endif // UART_H
