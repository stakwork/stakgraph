// @ast node: Class "PersonBuilder"
// @ast node: Function "PersonBuilder"
// @ast node: Function "withName"
// @ast node: Function "withEmail"
// @ast node: Function "build"
// @ast node: Var "name"
// @ast node: Var "email"
// @ast node: Class "BuilderClient"
// @ast node: Function "BuilderClient"
// @ast node: Function "createAlice"
// @ast edge: Calls -> Function "PersonBuilder" "FluentBuilder.java"
// @ast edge: Calls -> Function "withName" "FluentBuilder.java"
// @ast edge: Calls -> Function "withEmail" "FluentBuilder.java"
// @ast edge: Calls -> Function "build" "FluentBuilder.java"
package graph.stakgraph.java.nonweb;

import graph.stakgraph.java.model.Person;

class PersonBuilder {
    private String name;
    private String email;

    public PersonBuilder() {}

    public PersonBuilder withName(String name) {
        this.name = name;
        return this;
    }

    public PersonBuilder withEmail(String email) {
        this.email = email;
        return this;
    }

    public Person build() {
        return new Person(name, email);
    }
}

class BuilderClient {
    public BuilderClient() {}

    public Person createAlice() {
        return new PersonBuilder()
            .withName("Alice")
            .withEmail("alice@example.com")
            .build();
    }
}
