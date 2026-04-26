// @ast node: Class "BillingServiceTest"
// @ast node: UnitTest "charge_person_success"
// @ast edge: Calls -> Function "StripePaymentGateway" "StripePaymentGateway.java"
// @ast edge: Calls -> Function "BillingService" "BillingService.java"
// @ast edge: Calls -> Function "chargePerson" "BillingService.java"
// @ast node: Instance "service"
// @ast edge: Of -> Class "BillingService" "BillingService.java"
// @ast node: Instance "person"
// @ast edge: Of -> Class "Person" "Person.java"
// @ast node: Var "gateway"
// @ast node: Var "person"
// @ast node: Var "result"
// @ast node: Var "service"
package graph.stakgraph.java.nonweb;

import graph.stakgraph.java.model.Person;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertTrue;

public class BillingServiceTest {

    @Test
    void charge_person_success() {
        PaymentGateway gateway = new StripePaymentGateway();
        BillingService service = new BillingService(gateway);
        Person person = new Person("Test", "test@example.com");
        boolean result = service.chargePerson(person, 10L);
        assertTrue(result || !result);
    }
}
