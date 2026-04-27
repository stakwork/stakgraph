// @ast node: Class "FunctionalMethodRefRoutes"
// @ast node: Function "router"
// @ast node: Function "handleGet"
// @ast node: Function "handlePost"
// @ast node: Function "PUT_fn_put_lambda_L28"
// @ast edge: NestedIn -> Function "router" "FunctionalMethodRefRoutes.java"
// @ast node: Endpoint "/fn-get" [verb=GET]
// @ast edge: Handler -> Function "handleGet" "FunctionalMethodRefRoutes.java"
// @ast node: Endpoint "/fn-post" [verb=POST]
// @ast edge: Handler -> Function "handlePost" "FunctionalMethodRefRoutes.java"
// @ast node: Endpoint "/fn-put" [verb=PUT]
// @ast edge: Handler -> Function "PUT_fn_put_lambda_L28" "FunctionalMethodRefRoutes.java"
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
