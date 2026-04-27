#include "gpio.h"
#include <stdio.h>

// @ast node: Function "gpio_init"
void gpio_init(GPIO_TypeDef *port, gpio_pin_t pin, gpio_mode_t mode) {
    if (!port) return;
    
    // Clear mode bits
    port->MODER &= ~(0x3 << (pin * 2));
    // Set new mode
    port->MODER |= (mode << (pin * 2));
    
    printf("GPIO Init: Pin %d Mode %d\n", pin, mode);
}

// @ast node: Function "gpio_write"
void gpio_write(GPIO_TypeDef *port, gpio_pin_t pin, bool state) {
    if (!port) return;
    
    if (state) {
        port->ODR |= (1 << pin);
    } else {
        port->ODR &= ~(1 << pin);
    }
}

// @ast node: Function "gpio_read"
bool gpio_read(GPIO_TypeDef *port, gpio_pin_t pin) {
    if (!port) return false;
    return (port->IDR & (1 << pin)) != 0;
}

// @ast node: Function "gpio_toggle"
void gpio_toggle(GPIO_TypeDef *port, gpio_pin_t pin) {
    if (!port) return;
    port->ODR ^= (1 << pin);
}
