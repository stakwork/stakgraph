package graph.stakgraph.java.nonweb;

// Enum types must be captured as BOTH a Class and a DataModel (mirroring how
// records and Rust enums are handled). Covers package-private and public enums.

// package-private enum
// @ast node: Class "Suit"
// @ast node: DataModel "Suit"
enum Suit {
    HEARTS,
    DIAMONDS,
    CLUBS,
    SPADES
}

// public enum
// @ast node: Class "Priority"
// @ast node: DataModel "Priority"
public enum Priority {
    LOW,
    MEDIUM,
    HIGH
}
