// Tests static-factory call resolution.
//
// WidgetFactory.create() is a static method returning Widget.  The resolver
// must recognise "WidgetFactory" as a class-name receiver (not a local var)
// so it can look up (WidgetFactory, create) in method_returns → Widget.
// Then w.render() resolves Widget::render via class_fields dispatch.
//
// @ast node: Class "Widget"
// @ast node: Function "render"
// @ast node: Class "WidgetFactory"
// @ast node: Function "create"
// @ast node: Class "WidgetClient"
// @ast node: Function "WidgetClient"
// @ast node: Function "run"
// @ast edge: Calls -> Function "create" "MethodChainFactory.java"
// @ast edge: Calls -> Function "render" "MethodChainFactory.java"
// @ast node: Instance "w"
// @ast edge: Of -> Class "Widget" "MethodChainFactory.java"
// @ast node: Var "w"
package graph.stakgraph.java.nonweb;

class Widget {
    public void render() {}
}

class WidgetFactory {
    public static Widget create() {
        return new Widget();
    }
}

class WidgetClient {
    public WidgetClient() {}

    public void run() {
        Widget w = WidgetFactory.create();
        w.render();
    }
}
