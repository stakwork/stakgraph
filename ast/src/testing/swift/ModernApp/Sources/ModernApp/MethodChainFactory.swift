// Tests free-function return-type seeding (fn_returns) for Swift.
// makeWidget() is a top-level function returning Widget.  The resolver
// seeds fn_returns["makeWidget"] = "Widget" so that
// `let w = makeWidget()` binds w → Widget, enabling w.render() to
// resolve to Widget::render via find_method_in_class.
//
// @ast node: Class "Widget"
// @ast node: Function "render"
// @ast node: Function "makeWidget"
// @ast node: Function "factoryPipeline"
// @ast edge: Calls -> Function "makeWidget" "MethodChainFactory.swift"
// @ast edge: Calls -> Function "render" "MethodChainFactory.swift"

class Widget {
    func render() {}
}

func makeWidget() -> Widget {
    Widget()
}

func factoryPipeline() {
    let w = makeWidget()
    w.render()
}
