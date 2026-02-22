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
