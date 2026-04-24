package main

// @ast node: Endpoint "/anon-get" [verb=GET]
// @ast edge: Handler -> Function "GET_anon-get_func_L16" "anonymous_functions.go"
// @ast node: Endpoint "/anon-post" [verb=POST]
// @ast edge: Handler -> Function "POST_anon-post_func_L21" "anonymous_functions.go"
// @ast node: Function "mainAnon"
// @ast node: Function "GET_anon-get_func_L16"
// @ast node: Function "POST_anon-post_func_L21"

import "github.com/gin-gonic/gin"

func mainAnon() {
	r := gin.Default()

    // Test 1: GET with anonymous func
	r.GET("/anon-get", func(c *gin.Context) {
		c.JSON(200, gin.H{"message": "anon get"})
	})

    // Test 2: POST with anonymous func
	r.POST("/anon-post", func(c *gin.Context) {
		c.JSON(200, gin.H{"message": "anon post"})
	})
}
