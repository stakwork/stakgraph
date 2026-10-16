---
description: "RabbitMQ beside the app: one container, the AMQP URL at localhost."
parent: Pod Setup - Local Services
---
RabbitMQ beside the app: one container, the AMQP URL at localhost.

- A `rabbitmq` compose service (`image: rabbitmq:3-management`),
  `ports: ["5672:5672", "15672:15672"]`, with `RABBITMQ_DEFAULT_USER` and
  `RABBITMQ_DEFAULT_PASS` fixed.
- The app's AMQP URL is `amqp://<user>:<pass>@localhost:5672`.
