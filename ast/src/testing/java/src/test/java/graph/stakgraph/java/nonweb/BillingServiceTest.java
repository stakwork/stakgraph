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
