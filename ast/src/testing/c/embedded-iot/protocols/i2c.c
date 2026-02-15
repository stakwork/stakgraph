#include "i2c.h"
#include <stdio.h>

status_t i2c_init(uint8_t bus_id) {
    printf("Initializing I2C bus %d\n", bus_id);
    return STATUS_OK;
}

status_t i2c_write(uint8_t bus_id, uint8_t addr, const uint8_t *data, size_t len) {
    if (!data || len == 0) return STATUS_ERROR;
    
    printf("I2C Write [Bus %d] Addr 0x%02X: %zu bytes\n", bus_id, addr, len);
    // Simulate low-level register access
    // volatile uint32_t *I2C_DR = (uint32_t*)(0x40005400 + (bus_id * 0x400));
    // *I2C_DR = data[0];
    
    return STATUS_OK;
}

status_t i2c_read(uint8_t bus_id, uint8_t addr, uint8_t *data, size_t len) {
    if (!data || len == 0) return STATUS_ERROR;
    
    printf("I2C Read [Bus %d] Addr 0x%02X: %zu bytes\n", bus_id, addr, len);
    // Determine simulated data based on address
    for (size_t i = 0; i < len; i++) {
        data[i] = (uint8_t)(i + addr); // Dummy data
    }
    
    return STATUS_OK;
}
