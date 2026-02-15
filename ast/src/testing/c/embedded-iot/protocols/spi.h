#ifndef SPI_H
#define SPI_H

#include "../types.h"
#include <stddef.h>

/**
 * @brief Initialize SPI peripheral
 */
status_t spi_init(uint8_t spi_id);

/**
 * @brief Transfer data over SPI (Full Duplex)
 */
status_t spi_transfer(uint8_t spi_id, const uint8_t *tx_data, uint8_t *rx_data, size_t len);

#endif // SPI_H
