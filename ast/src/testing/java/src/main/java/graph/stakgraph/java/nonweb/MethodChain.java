// Tests for the Java hybrid LSP resolver.
//
// Engine is defined in Engine.java.  Vehicle and Fleet live here.
// Because "run" and "count" exist in both Engine.java and MethodChain.java,
// the global-unique heuristic fails — the hybrid resolver must track receiver
// types to route the calls correctly.
//
// Vehicle::run  calls this.engine.run()   → Engine::run  in Engine.java
//   (eval `this` → "Vehicle", look up field `engine` → "Engine", find run in Engine.java)
// Vehicle::count calls this.engine.count() → Engine::count in Engine.java
// Fleet::drive creates local Engine/Vehicle vars and calls v.run() → Vehicle::run here
// Fleet::status creates local Engine var and calls e.count() → Engine::count in Engine.java
//
// @ast node: Class "Vehicle"
// @ast node: Instance "engine"
// @ast edge: Of -> Class "Engine" "Engine.java"
// @ast node: Var "engine"
// @ast node: Function "Vehicle"
// @ast node: Function "run"
// @ast edge: Calls -> Function "run" "Engine.java"
// @ast node: Function "count"
// @ast edge: Calls -> Function "count" "Engine.java"
// @ast node: Class "Fleet"
// @ast node: Function "Fleet"
// @ast node: Function "drive"
// @ast edge: Calls -> Function "Engine" "Engine.java"
// @ast edge: Calls -> Function "Vehicle" "MethodChain.java"
// @ast edge: Calls -> Function "run" "MethodChain.java"
// @ast node: Instance "e"
// @ast edge: Of -> Class "Engine" "Engine.java"
// @ast node: Var "e"
// @ast node: Instance "v"
// @ast edge: Of -> Class "Vehicle" "MethodChain.java"
// @ast node: Var "v"
// @ast node: Function "status"
// @ast edge: Calls -> Function "Engine" "Engine.java"
// @ast edge: Calls -> Function "count" "Engine.java"
// @ast node: Instance "e"
// @ast edge: Of -> Class "Engine" "Engine.java"
// @ast node: Var "e"
package graph.stakgraph.java.nonweb;

/**
 * Vehicle wraps an Engine field.
 * run() and count() delegate to the engine field — same method names exist in
 * both classes, so the global-unique heuristic cannot resolve them correctly.
 */
class Vehicle {
    private Engine engine;

    public Vehicle(Engine engine) {
        this.engine = engine;
    }

    public void run() {
        this.engine.run();
    }

    public int count() {
        return this.engine.count();
    }
}

/**
 * Fleet creates Engine and Vehicle instances via local variables.
 * drive() calls v.run() — resolver must track that v: Vehicle.
 * status() calls e.count() — resolver must track that e: Engine.
 */
class Fleet {
    public Fleet() {}

    public void drive() {
        Engine e = new Engine();
        Vehicle v = new Vehicle(e);
        v.run();
    }

    public int status() {
        Engine e = new Engine();
        return e.count();
    }
}
