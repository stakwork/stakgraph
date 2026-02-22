package graph.stakgraph.java.nonweb;

import graph.stakgraph.java.model.Person;

public class BillingCli {
    public static void main(String[] args) {
        PaymentGateway gateway = new StripePaymentGateway();
        BillingService billingService = new BillingService(gateway);

        Person person = new Person("Charlie", "charlie@example.com");
        boolean charged = billingService.chargePerson(person, 1500L);

        if (charged) {
            System.out.println("charged");
        } else {
            System.out.println("failed");
        }
    }
}
