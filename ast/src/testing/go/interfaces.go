package main

import "math"

// Shape is a geometric shape interface
type Shape interface {
	Area() float64
	Perimeter() float64
}

// Rectangle represents a rectangle
type Rectangle struct {
	Width, Height float64
}

// Area calculates the area of the rectangle
func (r Rectangle) Area() float64 {
	return r.Width * r.Height
}

// Perimeter calculates the perimeter of the rectangle
func (r Rectangle) Perimeter() float64 {
	return 2*r.Width + 2*r.Height
}

// Circle represents a circle
type Circle struct {
	Radius float64
}

// Area calculates the area of the circle
func (c Circle) Area() float64 {
	return math.Pi * c.Radius * c.Radius
}

// Perimeter calculates the perimeter of the circle
func (c Circle) Perimeter() float64 {
	return 2 * math.Pi * c.Radius
}

// ColoredShape demonstrates struct embedding
type ColoredShape struct {
	Shape
	Color string
}

// Describe prints the description (Needed to keep ColoredShape from being pruned)
func (cs ColoredShape) Describe() string {
	return "I am a " + cs.Color + " shape"
}
