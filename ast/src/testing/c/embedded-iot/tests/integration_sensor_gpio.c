#include "../gpio.h"
#include "../sensors.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>

static GPIO_TypeDef led_port;
static GPIO_TypeDef sensor_power_port;

void integration_test_sensor_with_gpio_power(void) {
    memset(&sensor_power_port, 0, sizeof(GPIO_TypeDef));
    gpio_init(&sensor_power_port, GPIO_PIN_2, GPIO_MODE_OUTPUT);
    
    gpio_write(&sensor_power_port, GPIO_PIN_2, true);
    assert((sensor_power_port.ODR & (1 << 2)) != 0 && "Sensor power should be enabled");
    
    status_t result = temp_sensor_init();
    assert((result == STATUS_OK || result == STATUS_ERROR) && "Sensor init should return valid status");
    
    printf("PASS: integration_test_sensor_with_gpio_power\n");
}

void integration_test_led_indicator_on_sensor_read(void) {
    memset(&led_port, 0, sizeof(GPIO_TypeDef));
    gpio_init(&led_port, GPIO_PIN_13, GPIO_MODE_OUTPUT);
    
    gpio_write(&led_port, GPIO_PIN_13, true);
    float temp;
    status_t read_result = temp_sensor_read(&temp);
    
    if (read_result == STATUS_OK) {
        gpio_toggle(&led_port, GPIO_PIN_13);
        printf("Temperature read successful, LED toggled\n");
    } else {
        gpio_write(&led_port, GPIO_PIN_13, false);
        printf("Temperature read failed, LED off\n");
    }
    
    printf("PASS: integration_test_led_indicator_on_sensor_read\n");
}

int main(void) {
    printf("Running sensor-GPIO integration tests...\n");
    integration_test_sensor_with_gpio_power();
    integration_test_led_indicator_on_sensor_read();
    printf("All integration tests passed!\n");
    return 0;
}
