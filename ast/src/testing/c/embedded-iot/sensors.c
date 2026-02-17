#include "sensors.h"
#include "protocols/i2c.h"
#include <stdio.h>

#define TEMP_SENSOR_ADDR 0x48

/**
 * @brief Internal function to initialize the temperature sensor.
 * @return status_t STATUS_OK if successful.
 */
static status_t temp_init(void) {
    return i2c_init(1);
}

/**
 * @brief Internal function to read raw temperature data.
 * @param val Pointer to store the floating point temperature.
 * @return status_t STATUS_OK on success, STATUS_ERROR on I2C failure.
 */
static status_t temp_read(float *val) {
    uint8_t raw[2];
    if (i2c_read(1, TEMP_SENSOR_ADDR, raw, 2) != STATUS_OK) {
        return STATUS_ERROR;
    }
    
    // Combine bytes
    int16_t t = (raw[0] << 8) | raw[1];
    *val = t * 0.0625f;
    return STATUS_OK;
}

/**
 * @brief Resets the temperature sensor to default settings.
 * @return status_t STATUS_OK.
 */
static status_t temp_reset(void) {
    uint8_t cmd = 0x06;
    return i2c_write(1, TEMP_SENSOR_ADDR, &cmd, 1);
}

// Vtable definition
SensorOps temp_ops = {
    .init = temp_init,
    .read = temp_read,
    .write = NULL, // Read-only sensor
    .reset = temp_reset,
};

status_t temp_sensor_init(void) {
    return temp_ops.init();
}

status_t temp_sensor_read(float *temp) {
    return temp_ops.read(temp);
}
