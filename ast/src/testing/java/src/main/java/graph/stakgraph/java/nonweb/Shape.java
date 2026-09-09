// @ast node: Class "Shape"
// @ast edge: Imports -> Class "Circle" "Shape.java"
// @ast node: Class "Circle"
// @ast node: Var "radius"
package graph.stakgraph.java.nonweb;

public sealed class Shape permits Circle {
}

final class Circle extends Shape {
    double radius;
}
