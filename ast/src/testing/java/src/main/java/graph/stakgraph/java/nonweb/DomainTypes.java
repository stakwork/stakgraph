// @ast node: DataModel "InvoiceRecord"
package graph.stakgraph.java.nonweb;

public record InvoiceRecord(String id, long amount) {
}

// @ast node: Class "BillingStatus"
// @ast node: DataModel "BillingStatus"
enum BillingStatus {
    PENDING,
    PAID,
    FAILED
}
