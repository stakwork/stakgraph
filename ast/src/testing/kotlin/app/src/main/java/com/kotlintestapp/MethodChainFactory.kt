package com.kotlintestapp
// Tests free-function return-type seeding (fn_returns) for Kotlin.
// makeWidget() returns Widget; fn_returns seeds the return type so that
// `val w = makeWidget()` binds w → Widget, enabling w.render() to
// resolve to Widget::render via find_method_in_class.
//
// @ast node: Class "Widget"
// @ast node: Function "render"
// @ast node: Function "makeWidget"
// @ast node: Function "factoryPipeline"
// @ast edge: Calls -> Function "makeWidget" "MethodChainFactory.kt"
// @ast edge: Calls -> Function "render" "MethodChainFactory.kt"
// @ast node: Import "import-imports-srctestingkotlinappsrcmainjavacomkotlintestappmethodchainfactorykt-0"

class Widget {
    fun render() {}
}

fun makeWidget(): Widget = Widget()

fun factoryPipeline() {
    val w = makeWidget()
    w.render()
}
