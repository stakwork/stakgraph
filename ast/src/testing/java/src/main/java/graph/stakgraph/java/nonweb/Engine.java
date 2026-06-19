// @ast node: Class "Engine"
// @ast node: Function "Engine"
// @ast node: Function "run"
// @ast node: Function "count"
package graph.stakgraph.java.nonweb;

/** Leaf class — run/count defined here; other classes call these via receiver-type resolution. */
public class Engine {
    public Engine() {}

    public void run() {
        System.out.println("engine running");
    }

    public int count() {
        return 1;
    }
}
