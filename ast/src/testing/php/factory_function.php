<?php
// @ast node: Class "Widget"
// @ast edge: Operand -> Function "render" "factory_function.php"
// @ast node: Function "render"
// @ast node: Function "createWidget"
// @ast node: Function "useWidget"
// @ast edge: Calls -> Function "createWidget" "factory_function.php"
// @ast edge: Calls -> Function "render" "factory_function.php"
// @ast node: Var "$w"

namespace App;

class Widget
{
    public function render(): string
    {
        return 'rendered';
    }
}

function createWidget(): Widget
{
    return new Widget();
}

function useWidget(): string
{
    $w = createWidget();
    return $w->render();
}
