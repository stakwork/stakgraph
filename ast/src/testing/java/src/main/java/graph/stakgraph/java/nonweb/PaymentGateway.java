// @ast node: Trait "PaymentGateway"
// @ast node: Function "charge"
package graph.stakgraph.java.nonweb;

public interface PaymentGateway {
    boolean charge(String accountId, long cents);
}
