package graph.stakgraph.java.nonweb;

public record InvoiceRecord(String id, long amount) {
}

enum BillingStatus {
    PENDING,
    PAID,
    FAILED
}
