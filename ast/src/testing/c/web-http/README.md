# Web HTTP Test Server

This is a simulated C web server using the `libonion` framework style. It is used to test stakgraph's ability to parse C code, specifically:

- Function definitions and calls
- Structs and typedefs
- HTTP route registration patterns
- Library includes and dependencies

## Structure

- `server.c`: Main entry point and route setup
- `routes.c`: HTTP request handlers
- `models.c`: Data structures and logic
