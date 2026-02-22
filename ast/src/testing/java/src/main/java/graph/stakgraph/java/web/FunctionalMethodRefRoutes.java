package graph.stakgraph.java.web;

import org.springframework.context.annotation.Bean;
import org.springframework.web.reactive.function.server.RouterFunction;
import org.springframework.web.reactive.function.server.ServerRequest;
import org.springframework.web.reactive.function.server.ServerResponse;

import static org.springframework.web.reactive.function.server.RouterFunctions.route;

public class FunctionalMethodRefRoutes {

    @Bean
    public RouterFunction<ServerResponse> router() {
        return route()
            .GET("/fn-get", this::handleGet)
            .POST("/fn-post", this::handlePost)
            .PUT("/fn-put", request -> ServerResponse.ok().bodyValue("put"))
            .build();
    }

    public ServerResponse handleGet(ServerRequest request) {
        return ServerResponse.ok().bodyValue("method-ref-get");
    }

    public ServerResponse handlePost(ServerRequest request) {
        return ServerResponse.ok().bodyValue("method-ref-post");
    }
}
