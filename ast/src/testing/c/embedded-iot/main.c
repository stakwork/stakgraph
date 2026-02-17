#include <stdio.h>
#include <unistd.h>
#include "types.h"
#include "gpio.h"
#include "protocols/uart.h"
#include "protocols/spi.h"
#include "sensors.h"

// Defined in linker script normally
extern uint32_t _estack;

// Simulated main function for embedded system
int main(void) {
    // 1. Initialize Peripherals
    uart_init(1, 115200);
    spi_init(1);
    
    // 2. Initialize GPIO
    // Simulate finding the port address
    GPIO_TypeDef *GPIOA = (GPIO_TypeDef *)0x40020000;
    gpio_init(GPIOA, GPIO_PIN_5, GPIO_MODE_OUTPUT); // LED
    
    // 3. Initialize Sensors
    if (temp_sensor_init() != STATUS_OK) {
        uart_puts(1, "Sensor init failed!\n");
        return -1;
    }
    
    uart_puts(1, "System Initialized\n");
    
    // 4. Main Loop
    while (1) {
        float temp;
        if (temp_sensor_read(&temp) == STATUS_OK) {
            char buf[32];
            snprintf(buf, sizeof(buf), "Temp: %.2f C\n", temp);
            uart_puts(1, buf);
        }
        
        // Blink LED
        gpio_toggle(GPIOA, GPIO_PIN_5);
        
        // Simulate delay
        usleep(100000); // 100ms
        
        // Break loop for test purposes so we don't hang
        break;
    }
    
    return 0;
}
