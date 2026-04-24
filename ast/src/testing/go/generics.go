package main

// @ast node: Class "Stack"
// @ast edge: Operand -> Function "Push" "generics.go"
// @ast edge: Operand -> Function "Pop" "generics.go"
// @ast node: DataModel "Stack"
// @ast node: Function "Push"
// @ast node: Function "Pop"
// @ast node: Function "Map"

// Stack is a generic stack implementation
type Stack[T any] struct {
	items []T
}

// Push adds an item to the stack
func (s *Stack[T]) Push(item T) {
	s.items = append(s.items, item)
}

// Pop removes and returns the top item from the stack
func (s *Stack[T]) Pop() T {
	var zero T
	if len(s.items) == 0 {
		return zero
	}
	item := s.items[len(s.items)-1]
	s.items = s.items[:len(s.items)-1]
	return item
}

// Map applies a function to a slice and returns a new slice
func Map[T any, U any](input []T, f func(T) U) []U {
	result := make([]U, len(input))
	for i, v := range input {
		result[i] = f(v)
	}
	return result
}
