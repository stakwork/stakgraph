package main

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
