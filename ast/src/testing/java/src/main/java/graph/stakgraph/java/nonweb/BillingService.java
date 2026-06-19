// @ast node: Class "BillingService"
// @ast node: Function "BillingService"
// @ast node: Function "chargePerson"
// @ast edge: Calls -> Function "getEmail" "Person.java"
// @ast edge: Calls -> Function "findCustomer" "BillingService.java"
// @ast edge: Calls -> Function "charge" "StripePaymentGateway.java"
// @ast node: Function "findCustomer"
// @ast edge: Calls -> Function "getEmail" "Person.java"
// @ast node: Var "customers"
// @ast node: Var "found"
// @ast node: Var "gateway"
package graph.stakgraph.java.nonweb;

import graph.stakgraph.java.model.Person;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

public class BillingService {
    private final PaymentGateway gateway;
    private final List<Person> customers = new ArrayList<>();

    public BillingService(PaymentGateway gateway) {
        this.gateway = gateway;
    }

    public boolean chargePerson(Person person, long cents) {
        Optional<Person> found = findCustomer(person.getEmail());
        if (found.isEmpty()) {
            customers.add(person);
        }
        return gateway.charge(person.getEmail(), cents);
    }

    public Optional<Person> findCustomer(String email) {
        return customers.stream().filter(c -> c.getEmail().equals(email)).findFirst();
    }
}
