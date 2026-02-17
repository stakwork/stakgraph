#ifndef I2C_H
#define I2C_H

#include "../types.h"
#include <stddef.h>

/**
 * @brief Initialize I2C peripheral
 */
status_t i2c_init(uint8_t bus_id);

/**
 * @brief Write data to I2C device
 */
status_t i2c_write(uint8_t bus_id, uint8_t addr, const uint8_t *data, size_t len);

/**
 * @brief Read data from I2C device
 */
status_t i2c_read(uint8_t bus_id, uint8_t addr, uint8_t *data, size_t len);

#endif // I2C_H
