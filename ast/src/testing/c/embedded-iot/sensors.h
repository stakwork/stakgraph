#ifndef SENSORS_H
#define SENSORS_H

#include "types.h"

/**
 * @brief Virtual table for sensor operations.
 * 
 * This struct defines the interface that all driver implementations must support.
 */
typedef struct {
    status_t (*init)(void);         /**< Initialize the sensor hardware */
    status_t (*read)(float *value); /**< Read a value from the sensor */
    status_t (*write)(float value); /**< Write a value to the sensor (if supported) */
    status_t (*reset)(void);        /**< Reset the sensor */
} SensorOps;

/**
 * @brief Represents a physical sensor instance.
 */
typedef struct {
    uint8_t id;         /**< Unique sensor ID */
    char name[32];      /**< Human-readable name */
    SensorOps *ops;     /**< Pointer to operation vtable */
    void *priv_data;    /**< Driver-specific private data */
} Sensor;

/**
 * @brief Initialize temperature sensor
 */
status_t temp_sensor_init(void);

/**
 * @brief Read temperature
 */
status_t temp_sensor_read(float *temp);

extern SensorOps temp_ops;

#endif // SENSORS_H
