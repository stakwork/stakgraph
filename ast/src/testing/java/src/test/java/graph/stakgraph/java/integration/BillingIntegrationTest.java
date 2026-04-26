// @ast node: Class "BillingIntegrationTest"
// @ast node: IntegrationTest "integration_charge_flow"
// @ast edge: Calls -> Function "StripePaymentGateway" "StripePaymentGateway.java"
// @ast edge: Calls -> Function "BillingService" "BillingService.java"
// @ast edge: Calls -> Function "findCustomer" "BillingService.java"
// @ast node: Instance "service"
// @ast edge: Of -> Class "BillingService" "BillingService.java"
// @ast node: Var "gateway"
// @ast node: Var "service"
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
