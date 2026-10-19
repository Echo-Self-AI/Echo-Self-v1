package main

import (
	"net/http"

	"github.com/gin-gonic/gin"
)

func main() {
	router := gin.Default()

	router.GET("/health", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{
			"status": "ok",
			"service": "echoself-api",
		})
	})

	if err := router.Run(":8080"); err != nil {
		panic(err)
	}
}
