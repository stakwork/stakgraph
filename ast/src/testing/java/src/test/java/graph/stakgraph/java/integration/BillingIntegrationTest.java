package graph.stakgraph.java.integration;

import graph.stakgraph.java.nonweb.BillingService;
import graph.stakgraph.java.nonweb.PaymentGateway;
import graph.stakgraph.java.nonweb.StripePaymentGateway;
import org.junit.jupiter.api.Test;

public class BillingIntegrationTest {

    @Test
    void integration_charge_flow() {
        PaymentGateway gateway = new StripePaymentGateway();
        BillingService service = new BillingService(gateway);
        service.findCustomer("integration@example.com");
    }
}
