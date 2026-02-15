#ifndef GPIO_H
#define GPIO_H

#include "types.h"

// Simulated register map
typedef struct {
    volatile uint32_t MODER;    // Mode register
    volatile uint32_t OTYPER;   // Output type register
    volatile uint32_t OSPEEDR;  // Output speed register
    volatile uint32_t PUPDR;    // Pull-up/pull-down register
    volatile uint32_t IDR;      // Input data register
    volatile uint32_t ODR;      // Output data register
} GPIO_TypeDef;

typedef enum {
    GPIO_PIN_0 = 0,
    GPIO_PIN_1 = 1,
    GPIO_PIN_2 = 2,
    // ...
    GPIO_PIN_13 = 13,
} gpio_pin_t;

typedef enum {
    GPIO_MODE_INPUT = 0,
    GPIO_MODE_OUTPUT = 1,
    GPIO_MODE_AF = 2,
    GPIO_MODE_ANALOG = 3,
} gpio_mode_t;

void gpio_init(GPIO_TypeDef *port, gpio_pin_t pin, gpio_mode_t mode);
void gpio_write(GPIO_TypeDef *port, gpio_pin_t pin, bool state);
bool gpio_read(GPIO_TypeDef *port, gpio_pin_t pin);
void gpio_toggle(GPIO_TypeDef *port, gpio_pin_t pin);

#endif // GPIO_H
