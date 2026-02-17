#include "spi.h"
#include <stdio.h>

status_t spi_init(uint8_t spi_id) {
    printf("Initializing SPI %d\n", spi_id);
    return STATUS_OK;
}

status_t spi_transfer(uint8_t spi_id, const uint8_t *tx_data, uint8_t *rx_data, size_t len) {
    if (!tx_data || len == 0) return STATUS_ERROR;
    
    printf("SPI Transfer [ID %d]: %zu bytes\n", spi_id, len);
    
    // Simulate loopback or dummy response
    if (rx_data) {
        for (size_t i = 0; i < len; i++) {
            rx_data[i] = tx_data[i] ^ 0xFF; // Invert bits as dummy logic
        }
    }
    
    return STATUS_OK;
}
