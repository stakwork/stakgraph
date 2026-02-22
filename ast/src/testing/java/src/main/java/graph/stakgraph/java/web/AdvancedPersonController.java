package graph.stakgraph.java.web;

import graph.stakgraph.java.model.Person;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.RestController;

import java.util.ArrayList;
import java.util.List;

@RestController
@RequestMapping(path = "/api/v1/people")
public class AdvancedPersonController {
    private final List<Person> localPeople = new ArrayList<>();

    public AdvancedPersonController() {
    }

    @GetMapping(path = "/{id}")
    public ResponseEntity<Person> getById(@PathVariable Long id) {
        return ResponseEntity.notFound().build();
    }

    @PostMapping(value = "/bulk")
    public ResponseEntity<List<Person>> createBulk(@RequestBody List<Person> people) {
        localPeople.addAll(people);
        return ResponseEntity.ok(localPeople);
    }

    @RequestMapping(path = "/search", method = RequestMethod.GET)
    public ResponseEntity<List<Person>> search() {
        return ResponseEntity.ok(localPeople);
    }

    @DeleteMapping(path = "/{id}")
    public ResponseEntity<Void> delete(@PathVariable Long id) {
        return ResponseEntity.noContent().build();
    }
}
