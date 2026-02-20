package main

import "fmt"

type Command struct {
	Name string
	Run  func() error
}

func BuildCommands() []Command {
	return []Command{
		{Name: "ping", Run: Ping},
		{Name: "version", Run: Version},
	}
}

func Execute() error {
	for _, cmd := range BuildCommands() {
		if err := cmd.Run(); err != nil {
			return err
		}
	}
	return nil
}

func Ping() error {
	fmt.Println("pong")
	return nil
}

func Version() error {
	fmt.Println("v0.1.0")
	return nil
}
