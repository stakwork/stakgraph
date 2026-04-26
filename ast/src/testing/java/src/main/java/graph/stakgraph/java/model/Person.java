// @ast node: Class "Person"
// @ast node: Class "Animal"
// @ast edge: ParentOf -> Class "Dog" "Person.java"
// @ast node: Class "Dog"
// @ast node: DataModel "Person"
// @ast node: Function "Person"
// @ast node: Function "Person"
// @ast node: Function "getId"
// @ast node: Function "getName"
// @ast node: Function "getEmail"
// @ast node: Function "setId"
// @ast node: Function "setName"
// @ast node: Function "setEmail"
// @ast node: Function "Animal"
// @ast node: Function "speak"
// @ast node: Function "Dog"
// @ast node: Function "speak"
// @ast node: Var "breed"
// @ast node: Var "email"
// @ast node: Var "id"
// @ast node: Var "name"
// @ast node: Var "name"
package graph.stakgraph.java.model;

import jakarta.persistence.Entity;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;

@Entity
public class Person {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;
    private String name;
    private String email;
    
    
    public Person() {}
    public Person(String name, String email) {
        this.name = name;
        this.email = email;
    }
    
    public Long getId() { return id; }
    public String getName() { return name; }
    public String getEmail() { return email; }
    public void setId(Long id) { this.id = id; }
    public void setName(String name) { this.name = name; }
    public void setEmail(String email) { this.email = email; }
}

class Animal {
    String name;

    public Animal(String name) {
        this.name = name;
    }

    public void speak() {
        System.out.println("I am " + name);
    }
}


class Dog extends Animal {
    String breed;

    public Dog(String name, String breed) {
        super(name);
        this.breed = breed;
    }

    @Override 
    public void speak() {
        System.out.println("Woof! I'm a " + breed + " named " + name);
    }
}