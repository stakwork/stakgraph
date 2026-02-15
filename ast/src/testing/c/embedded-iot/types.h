#ifndef TYPES_H
#define TYPES_H

#include <stdint.h>
#include <stdbool.h>

typedef enum {
    STATUS_OK = 0,
    STATUS_ERROR = 1,
    STATUS_BUSY = 2,
    STATUS_TIMEOUT = 3
} status_t;

typedef struct {
    uint32_t timestamp;
    uint32_t processing_time;
} telemetery_t;

#endif // TYPES_H
