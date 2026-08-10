// Tests static-factory call resolution via the uppercase-identifier fallback.
//
// WidgetFactory.Create() is a static method returning Widget.  The resolver
// must recognise "WidgetFactory" as a class-name receiver (not a local var)
// so it can look up the method in the graph.  Then w.Render() resolves
// Widget::Render via find_method_in_class.
//
// @ast node: Class "Widget"
// @ast node: Class "WidgetClient"
// @ast node: Class "WidgetFactory"
// @ast node: Function "Create"
// @ast node: Function "Render"
// @ast node: Function "WidgetClient"
// @ast node: Function "Run"
// @ast edge: Calls -> Function "Create" "MethodChainFactory.cs"
// @ast edge: Calls -> Function "Render" "MethodChainFactory.cs"
namespace MethodChain;

class Widget
{
    public void Render() { }
}

class WidgetFactory
{
    public static Widget Create() { return new Widget(); }
}

class WidgetClient
{
    public WidgetClient() { }

    public void Run()
    {
        Widget w = WidgetFactory.Create();
        w.Render();
    }
}
