#include "uart.h"
#include <stdio.h>

// @ast node: Function "uart_init"
status_t uart_init(uint8_t uart_id, uint32_t baudrate) {
    printf("Initializing UART %d at %d baud\n", uart_id, baudrate);
    return STATUS_OK;
}

// @ast node: Function "uart_putc"
void uart_putc(uint8_t uart_id, char c) {
    // Direct hardware write simulation
    // volatile uint32_t *UART_DR = (uint32_t*)(0x40004000 + (uart_id * 0x1000));
    // *UART_DR = c;
    printf("[UART%d] %c", uart_id, c);
}

// @ast node: Function "uart_puts"
// @ast edge: Calls -> Function "uart_putc" "uart.c"
void uart_puts(uint8_t uart_id, const char *str) {
    while (*str) {
        uart_putc(uart_id, *str++);
    }
    uart_putc(uart_id, '\n');
}
