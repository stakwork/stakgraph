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