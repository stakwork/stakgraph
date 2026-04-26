// @ast node: Class "BillingCli"
// @ast node: Function "main"
// @ast edge: Calls -> Function "StripePaymentGateway" "StripePaymentGateway.java"
// @ast edge: Calls -> Function "BillingService" "BillingService.java"
// @ast edge: Calls -> Function "chargePerson" "BillingService.java"
// @ast node: Instance "billingService"
// @ast edge: Of -> Class "BillingService" "BillingService.java"
// @ast node: Instance "person"
// @ast edge: Of -> Class "Person" "Person.java"
// @ast node: Var "billingService"
// @ast node: Var "charged"
// @ast node: Var "gateway"
// @ast node: Var "person"
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
