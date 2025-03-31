package graph.stakgraph.java.repository;

import graph.stakgraph.java.model.Person;
import org.springframework.data.jpa.repository.JpaRepository;

public interface PersonRepository extends JpaRepository<Person, Long> {}