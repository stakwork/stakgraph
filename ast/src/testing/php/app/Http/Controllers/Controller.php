<?php
// @ast node: Class "Controller"
// @ast node: Import "import-imports-srctestingphpapphttpcontrollerscontrollerphp-6"

namespace App\Http\Controllers;

use Illuminate\Foundation\Auth\Access\AuthorizesRequests;
use Illuminate\Foundation\Validation\ValidatesRequests;
use Illuminate\Routing\Controller as BaseController;

abstract class Controller extends BaseController
{
    use AuthorizesRequests, ValidatesRequests;
}
