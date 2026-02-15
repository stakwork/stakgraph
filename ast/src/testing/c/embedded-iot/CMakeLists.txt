cmake_minimum_required(VERSION 3.10)
project(embedded_iot C)

add_executable(firmware
    main.c
    gpio.c
    sensors.c
    protocols/i2c.c
    protocols/spi.c
    protocols/uart.c
)

target_include_directories(firmware PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
