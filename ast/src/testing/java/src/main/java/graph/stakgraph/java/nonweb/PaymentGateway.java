package graph.stakgraph.java.nonweb;

public interface PaymentGateway {
    boolean charge(String accountId, long cents);
}
