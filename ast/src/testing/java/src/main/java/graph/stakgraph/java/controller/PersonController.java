// @ast node: Class "PersonController"
// @ast node: Function "PersonController"
// @ast node: Function "getPerson"
// @ast edge: Calls -> Function "getPersonById" "PersonController.java"
// @ast node: Function "createPerson"
// @ast edge: Calls -> Function "newPerson" "PersonController.java"
// @ast node: Function "getPersonById"
// @ast node: Function "newPerson"
// @ast node: Endpoint "/person/{id}" [verb=GET]
// @ast edge: Handler -> Function "getPerson" "PersonController.java"
// @ast node: Endpoint "/person" [verb=POST]
// @ast edge: Handler -> Function "createPerson" "PersonController.java"
// @ast node: Instance "examplePerson"
// @ast edge: Of -> Class "Person" "Person.java"
// @ast node: Var "appName"
// @ast node: Var "examplePerson"
// @ast node: Var "person"
// @ast node: Var "repository"
package graph.stakgraph.java.controller;

import graph.stakgraph.java.model.Person;
import graph.stakgraph.java.repository.PersonRepository;
import org.springframework.web.bind.annotation.*;
import org.springframework.http.ResponseEntity;
import java.util.Optional;

protected static String appName = "stakgraph";

@RestController
public class PersonController {
    private final PersonRepository repository;

    private static final Person examplePerson = new Person("Alice", "alice@example.com");

    public PersonController(PersonRepository repository) {
        this.repository = repository;
    }

    /**
     * Get a person by ID.
     */
    @GetMapping("/person/{id}")
    public ResponseEntity<Person> getPerson(@PathVariable Long id) {
        Optional<Person> person = getPersonById(id);
        return person.map(ResponseEntity::ok)
                     .orElse(ResponseEntity.notFound().build());
    }

    @PostMapping("/person")
    public Person createPerson(@RequestBody Person person) {
        return newPerson(person);
    }

    private Optional<Person> getPersonById(Long id) {
        return repository.findById(id);
    }

    private Person newPerson(Person person) {
        return repository.save(person);
    }
}