// Tests method chain resolution via the C++ registry.
//
// makeWidget() is a free function returning Widget.  The resolver
// seeds fn_returns so that `Widget w = makeWidget()` binds w → Widget,
// enabling w.render() to resolve to Widget::render via
// find_method_in_class.
//
// @ast node: Class "Widget"
// @ast node: Function "render"
// @ast node: Function "makeWidget"
// @ast node: Function "factoryPipeline"
// @ast edge: Calls -> Function "makeWidget" "MethodChainFactory.cpp"
// @ast edge: Calls -> Function "render" "MethodChainFactory.cpp"

class Widget {
public:
    void render() {}
};

Widget makeWidget() {
    return Widget();
}

void factoryPipeline() {
    Widget w = makeWidget();
    w.render();
}
