#ifndef TYPES_H
#define TYPES_H

#include <stdint.h>
#include <stdbool.h>

typedef enum {
    STATUS_OK = 0,
    STATUS_ERROR = 1,
    STATUS_BUSY = 2,
    STATUS_TIMEOUT = 3
// @ast node: Class "status_t"
} status_t;

typedef struct {
    uint32_t timestamp;
    uint32_t processing_time;
// @ast node: Class "telemetery_t"
} telemetery_t;

#endif // TYPES_H
