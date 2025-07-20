package graph.stakgraph.java;

import graph.stakgraph.java.model.Person;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
public class JavaApplication {

	public static void main(String[] args) {
		 Person testPerson = new Person("Bob", "bob@example.com");
		SpringApplication.run(JavaApplication.class, args);
	}

}
