// @ast node: Class "AnonymousRoutes"
// @ast node: Function "anonymousRouter"
// @ast node: Function "GET_anon_get_lambda_L24"
// @ast edge: NestedIn -> Function "anonymousRouter" "AnonymousRoutes.java"
// @ast node: Function "POST_anon_post_lambda_L28"
// @ast edge: NestedIn -> Function "anonymousRouter" "AnonymousRoutes.java"
// @ast node: Endpoint "/anon-get" [verb=GET]
// @ast edge: Handler -> Function "GET_anon_get_lambda_L24" "AnonymousRoutes.java"
// @ast node: Endpoint "/anon-post" [verb=POST]
// @ast edge: Handler -> Function "POST_anon_post_lambda_L28" "AnonymousRoutes.java"
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
