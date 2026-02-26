#include "../gpio.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>

static GPIO_TypeDef test_port;

void test_gpio_init(void) {
    memset(&test_port, 0, sizeof(GPIO_TypeDef));
    gpio_init(&test_port, GPIO_PIN_5, GPIO_MODE_OUTPUT);
    
    uint32_t expected_moder = (GPIO_MODE_OUTPUT << (5 * 2));
    assert(test_port.MODER == expected_moder && "GPIO init should set mode register");
    printf("PASS: test_gpio_init\n");
}

void test_gpio_write(void) {
    memset(&test_port, 0, sizeof(GPIO_TypeDef));
    gpio_init(&test_port, GPIO_PIN_3, GPIO_MODE_OUTPUT);
    
    gpio_write(&test_port, GPIO_PIN_3, true);
    assert((test_port.ODR & (1 << 3)) != 0 && "GPIO write high should set ODR bit");
    
    gpio_write(&test_port, GPIO_PIN_3, false);
    assert((test_port.ODR & (1 << 3)) == 0 && "GPIO write low should clear ODR bit");
    
    printf("PASS: test_gpio_write\n");
}

void test_gpio_read(void) {
    memset(&test_port, 0, sizeof(GPIO_TypeDef));
    test_port.IDR = (1 << 7);
    
    bool state = gpio_read(&test_port, GPIO_PIN_7);
    assert(state == true && "GPIO read should detect high state");
    
    state = gpio_read(&test_port, GPIO_PIN_2);
    assert(state == false && "GPIO read should detect low state");
    
    printf("PASS: test_gpio_read\n");
}

void test_gpio_toggle(void) {
    memset(&test_port, 0, sizeof(GPIO_TypeDef));
    gpio_init(&test_port, GPIO_PIN_1, GPIO_MODE_OUTPUT);
    
    gpio_toggle(&test_port, GPIO_PIN_1);
    assert((test_port.ODR & (1 << 1)) != 0 && "First toggle should set bit");
    
    gpio_toggle(&test_port, GPIO_PIN_1);
    assert((test_port.ODR & (1 << 1)) == 0 && "Second toggle should clear bit");
    
    printf("PASS: test_gpio_toggle\n");
}

int main(void) {
    printf("Running GPIO unit tests...\n");
    test_gpio_init();
    test_gpio_write();
    test_gpio_read();
    test_gpio_toggle();
    printf("All GPIO unit tests passed!\n");
    return 0;
}
