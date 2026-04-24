package main

// @ast node: Function "main"
// @ast edge: Calls -> Function "Execute" "cli.go"
// @ast edge: Calls -> Function "StartPipeline" "pipeline.go"
// @ast edge: Calls -> Function "RegisterRoutes" "http_stdlib.go"

func main() {
	Execute()
	StartPipeline()
	RegisterRoutes()
}
