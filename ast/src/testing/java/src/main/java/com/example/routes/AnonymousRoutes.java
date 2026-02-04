package com.example.routes;

import org.springframework.context.annotation.Bean;
import org.springframework.web.reactive.function.server.RouterFunction;
import org.springframework.web.reactive.function.server.ServerResponse;
import org.springframework.web.reactive.function.server.ServerRequest;
import static org.springframework.web.reactive.function.server.RouterFunctions.route;

public class AnonymousRoutes {

    @Bean
    public RouterFunction<ServerResponse> anonymousRouter() {
        return route()
            // Simple lambda
            .GET("/anon-get", request ->
                ServerResponse.ok().bodyValue("Anonymous GET"))

            // Lambda with block
            .POST("/anon-post", request -> {
                return ServerResponse.ok()
                    .bodyValue("Anonymous POST");
            })
            .build();
    }
}
