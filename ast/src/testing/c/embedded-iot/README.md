# Embedded/IoT Test Server

This simulates a bare-metal embedded C project to test:

- Direct hardware access (pointers to registers)
- Protocol implementations (I2C, SPI, UART)
- HAL (Hardware Abstraction Layer) patterns
- Vtables / Function pointers in structs (`SensorOps`)
- Typedefs for register maps and enums

## Structure

- `main.c`: System startup and main loop
- `protocols/`: Communication protocols
- `gpio.*`: General Purpose I/O driver
- `sensors.*`: Sensor abstraction layer
