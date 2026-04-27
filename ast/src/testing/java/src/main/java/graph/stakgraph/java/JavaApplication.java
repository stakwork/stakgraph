// @ast node: Class "JavaApplication"
// @ast node: Function "main"
// @ast node: Instance "testPerson"
// @ast edge: Of -> Class "Person" "Person.java"
// @ast node: Var "testPerson"
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
